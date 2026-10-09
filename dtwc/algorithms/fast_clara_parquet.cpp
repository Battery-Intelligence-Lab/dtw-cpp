/**
 * @file fast_clara_parquet.cpp
 * @brief FastCLARA over series streamed from Parquet: dtwc_io, in a build with Parquet.
 *
 * @details The RAM-limited route of dtwc_cl --ram-limit. The main Problem holds settings only; each
 *   sample, the medoids and the assignment's row groups are read from the file as they are needed.
 *   It moved out of fast_clara.cpp so that the core, which the Python module and the MEX link, holds
 *   no Parquet; the rest of FastCLARA is fast_clara.cpp's, which this file calls.
 * @author Volkan Kumtepeli
 */

#ifdef DTWC_HAS_PARQUET

#include "fast_clara.hpp"
#include "detail/fast_clara_assign.hpp"
#include "detail/fast_clara_plan.hpp"
#include "fast_pam.hpp"
#include "../Problem.hpp"
#include "../base/error.hpp"
#include "../base/missing_utils.hpp"
#include "../core/medoid_assignment_policy.hpp"
#include "../io/parquet_chunk_reader.hpp"
#include "../warping.hpp" // detail::require_finite

#include <algorithm>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace dtwc::algorithms {

namespace {

  /// Where each of `rows` sits in `sorted`, which holds every one of them.
  std::vector<index_t> positions_in(const std::vector<int64_t> &sorted, const std::vector<index_t> &rows)
  {
    std::vector<index_t> positions;
    positions.reserve(rows.size());
    for (const index_t row : rows)
      positions.push_back(std::lower_bound(sorted.begin(), sorted.end(), row) - sorted.begin());
    return positions;
  }

  /**
   * The in-RAM route's checks of the series it clusters (Problem::validate_fill_request), on rows read from the file:
   * rows() refuses an empty series, a value no distance is defined for (+-inf, and NaN unless the missing strategy
   * takes it) and under Interpolate a series entirely NaN, each an InvalidInput naming the row; band() refuses a band
   * narrower than the length difference of the shortest and the longest rows rows() has seen, the first of each.
   */
  class RowCheck
  {
  public:
    explicit RowCheck(const core::DistanceConfig &distance) : distance_(distance) {}

    /// The series of `data`, the i-th the file's row row_of(i).
    template <typename RowOf>
    void rows(const Data &data, const RowOf &row_of)
    {
      const bool nan_is_missing = distance_.missing != core::MissingStrategy::Error;
      std::string name;
      for (std::size_t i = 0; i < static_cast<std::size_t>(data.size()); ++i) {
        const auto row = static_cast<int64_t>(row_of(i));
        const auto length = data.series_flat_size(i);
        if (length == 0)
          throw InvalidInput("fast_clara: series " + std::to_string(row) + " ('" + std::string(data.name(i))
                             + "') is empty; every series needs at least one value.");
        name.assign("series '").append(data.name(i)).append("' (index ").append(std::to_string(row)).append(")");
        const auto check = [&](const auto values) {
          dtwc::detail::require_finite(values, name, "fast_clara", nan_is_missing);
          if (distance_.missing == core::MissingStrategy::Interpolate && all_missing(values))
            throw InvalidInput("fast_clara: " + name + " is entirely NaN, so MissingStrategy::Interpolate has nothing "
                               "to interpolate from. Use ZeroCost or AROW, or drop the series.");
        };
        if (data.is_f32())
          check(data.series_f32(i));
        else
          check(data.series(i));
        if (length < shortest_.length) shortest_ = { length, row, std::string(data.name(i)) };
        if (length > longest_.length) longest_ = { length, row, std::string(data.name(i)) };
      }
    }

    void band() const
    {
      const auto variant = distance_.variant.variant;
      if (distance_.band < 0 || longest_.length <= shortest_.length || variant == core::DTWVariant::SoftDTW
          || variant == core::DTWVariant::MSM || variant == core::DTWVariant::TWE)
        return;
      const std::size_t gap = longest_.length - shortest_.length;
      if (gap <= static_cast<std::size_t>(distance_.band)) return;
      const auto describe = [](const Extreme &e) {
        return "series '" + e.name + "' (index " + std::to_string(e.row) + ", length " + std::to_string(e.length) + ")";
      };
      throw InvalidInput("fast_clara: band = " + std::to_string(distance_.band) + " is narrower than the length "
                         "difference between " + describe(shortest_) + " and " + describe(longest_)
                         + ", so no warping path fits that pair. The smallest feasible band is " + std::to_string(gap)
                         + "; pass band >= " + std::to_string(gap) + ", or band = -1 for full DTW.");
    }

  private:
    struct Extreme
    {
      std::size_t length;
      int64_t row;
      std::string name;
    };
    core::DistanceConfig distance_;
    Extreme shortest_{ std::numeric_limits<std::size_t>::max(), 0, {} }, longest_{ 0, 0, {} };
  };

  /**
   * @brief FastCLARA streamed from Parquet, its series read as T (float with F32, else double).
   *
   * All samples are drawn first; the rows they hold, their union, are read once and each sample's PAM runs on a copy
   * of its series. The medoids' series are copied out and the union released. Then one pass over the file: a batch
   * of row groups at a time, every point against the medoids of every sample, while the next batch decodes on another
   * thread when two batches fit the limit (else batches decode and are assigned in turn). Within ram_limit_bytes, as
   * the reader prices them: the union beside one decode (read_rows' check), then the medoids, every sample's labels,
   * one block of nearest distances, one decode and the batches. The union, and each batch as it decodes, go through
   * the in-RAM route's checks (RowCheck): the stream refuses what the in-RAM route refuses, naming the row.
   */
  template <bool F32>
  core::ClusteringResult fast_clara_chunked(
    Problem &prob, const CLARAOptions &opts, const io::ParquetChunkReader &reader, const detail::ClaraPlan &plan)
  {
    detail::validate_streaming_clara_plan(plan, "fast_clara");
    using T = std::conditional_t<F32, float, data_t>;
    const auto series_at = [](const Data &d, index_t index) {
      if constexpr (F32)
        return d.series_f32(static_cast<size_t>(index));
      else
        return d.series(static_cast<size_t>(index));
    };
    // The settings' check, before any read: the series come later, row by row.
    const auto &pair = [&]() -> const auto & {
      if constexpr (F32)
        return prob.dtw_function_f32();
      else
        return prob.dtw_function();
    }();
    const auto n_points = static_cast<std::size_t>(plan.n_points);
    const auto samples = detail::draw_clara_samples(plan.n_points, plan.sample_size, opts);

    std::vector<int64_t> rows; // the samples' union, ascending
    for (const auto &sample : samples) rows.insert(rows.end(), sample.begin(), sample.end());
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

    std::vector<index_t> medoids; // sample s's medoid m is slot s * k + m
    std::vector<core::ClusteringResult> pams;
    Data medoid_data;
    {
      const Data pool = reader.read_rows<T>(rows, opts.ram_limit_bytes);
      RowCheck sampled(prob.distance());
      sampled.rows(pool, [&](std::size_t i) { return rows[i]; });
      sampled.band();
      std::vector<std::vector<index_t>> positions;
      for (const auto &sample : samples) positions.push_back(positions_in(rows, sample));
      // The PAMs run one at a time beside the union: a sample's copy and its packed distance matrix.
      std::size_t union_values = 0, sample_values = 0;
      for (std::size_t i = 0; i < rows.size(); ++i) union_values += pool.series_flat_size(i);
      for (const auto &at : positions) {
        std::size_t values = 0;
        for (const index_t i : at) values += pool.series_flat_size(static_cast<std::size_t>(i));
        sample_values = std::max(sample_values, values);
      }
      const auto s = static_cast<std::size_t>(plan.sample_size);
      const size_t union_bytes = io::ParquetChunkReader::series_bytes(union_values, rows.size(), F32);
      const size_t pam_bytes =
        io::ParquetChunkReader::series_bytes(sample_values, s, F32) + core::packed_size(s) * sizeof(double);
      if (union_bytes + pam_bytes > opts.ram_limit_bytes)
        throw InvalidInput("fast_clara: ram_limit_bytes (--ram-limit) = " + std::to_string(opts.ram_limit_bytes)
                           + " cannot hold a sample's PAM beside the samples' series, which needs "
                           + std::to_string(union_bytes + pam_bytes) + " bytes: " + std::to_string(union_bytes)
                           + " for the " + std::to_string(rows.size()) + " sampled series, "
                           + std::to_string(pam_bytes) + " for a sample's copy and its distance matrix (sample_size = "
                           + std::to_string(s) + "); raise the limit, or lower sample_size.");
      pams = detail::clara_pams(prob, opts, pool, samples, positions, true, medoids);
      medoid_data = detail::clara_sample(pool, positions_in(rows, medoids), true);
    }

    const size_t limit = opts.ram_limit_bytes;
    std::size_t medoid_values = 0;
    for (std::size_t j = 0; j < medoids.size(); ++j) medoid_values += medoid_data.series_flat_size(j);
    const size_t medoid_bytes = io::ParquetChunkReader::series_bytes(medoid_values, medoids.size(), F32);
    const size_t fixed = medoid_bytes + detail::clara_pass_bytes(plan, samples.size()) + reader.decode_bytes();
    const size_t group = reader.largest_row_group_bytes(F32);
    // After the pass the Problem copies the winner's labels (set_result) beside every sample's.
    const size_t finish = medoid_bytes + samples.size() * n_points * sizeof(index_t) + n_points * sizeof(index_t)
                        + n_points / 8;
    if (std::max(fixed + group, finish) > limit)
      throw InvalidInput(
        "fast_clara: ram_limit_bytes (--ram-limit) = " + std::to_string(limit) + " cannot hold the pass over the "
        "file, which needs " + std::to_string(std::max(fixed + group, finish)) + " bytes: " + std::to_string(fixed)
        + " for the medoids' series, every sample's labels and nearest distances (n_samples = "
        + std::to_string(samples.size()) + ", k = " + std::to_string(opts.n_clusters) + ", "
        + std::to_string(n_points) + " rows) and decoding, " + std::to_string(group)
        + " for one row group's series, " + std::to_string(finish)
        + " at its end with the result's labels; raise the limit, or rewrite the file with smaller row groups.");
    const bool overlap = fixed + 2 * group <= limit; // a batch decoding beside the one assigned
    const int per_batch = reader.row_groups_per_batch(overlap ? (limit - fixed) / 2 : limit - fixed, F32);

    std::vector<std::vector<index_t>> labels(samples.size(), std::vector<index_t>(n_points));
    std::vector<core::detail::OrderedMedoidObjective> costs(samples.size());
#ifdef DTWC_HAS_CUDA
    bool on_gpu = false;
    if constexpr (!F32) // Float32 series never reach a GPU: the sample fill refuses them first
      on_gpu = prob.device().first == Device::GPU;
#else
    detail::say_cpu_assignment(prob, n_points, samples.size(), static_cast<std::size_t>(opts.n_clusters));
#endif

    // The reader and `check` are used by one thread at a time: the decoding one while a batch decodes and is checked
    // (its first row known, as the batches before it are), the team only after the last batch.
    RowCheck check(prob.distance());
    const auto decode = [&reader, &check](int rg, int count, index_t first_row) {
      Data batch = reader.read_row_groups<T>(rg, count);
      check.rows(batch, [first_row](std::size_t i) { return first_row + static_cast<index_t>(i); });
      return batch;
    };
    const auto policy = overlap ? std::launch::async : std::launch::deferred;
    const int total_rg = reader.num_row_groups();
    std::future<Data> next = std::async(policy, decode, 0, std::min(per_batch, total_rg), index_t{ 0 });
    index_t first = 0;
    for (int rg = 0; rg < total_rg; rg += per_batch) {
      const Data batch = next.get();
      if (rg + per_batch < total_rg)
        next = std::async(policy, decode, rg + per_batch, std::min(per_batch, total_rg - rg - per_batch),
                          first + batch.size());
      // The batch is loaded: the assignment reads it, never the reader.
#ifdef DTWC_HAS_CUDA
      if (on_gpu)
        detail::assign_on_gpu(prob, batch.p_vec, first, medoid_data.p_vec, medoids, labels, costs);
      else
#endif
        detail::assign_in_blocks(first, static_cast<std::size_t>(batch.size()), costs,
                                 [&](index_t block_first, std::size_t n, std::span<double> best) {
                                   detail::assign_points_cpu<T>(
                                     block_first, n, best, medoids, labels, prob.distance(), pair,
                                     [&](index_t p) { return series_at(batch, p - first); },
                                     [&](index_t j) { return series_at(medoid_data, j); });
                                 });
      first += batch.size();
    }
    check.band();

    auto result = detail::clara_winner(pams, medoids, labels, costs);
    prob.set_result(result);
    return result;
  }

} // anonymous namespace

core::ClusteringResult fast_clara_parquet(Problem &prob, const CLARAOptions &opts)
{
  detail::validate_clara_request(prob, opts);
  // Chunked mode: stream from Parquet when ram_limit is set
  if (opts.ram_limit_bytes > 0 && !opts.parquet_path.empty()) {
    io::ParquetChunkReader reader(opts.parquet_path, opts.parquet_column, opts.parquet_skip_cols,
                                  opts.parquet_skip_rows);
    const auto resident_bytes =
      reader.estimated_resident_bytes(opts.use_float32);

    // If data fits in RAM, skip chunked mode
    if (opts.force_parquet_streaming
        || resident_bytes > opts.ram_limit_bytes) {
      if (prob.size() != 0)
        throw InvalidInput(
          "fast_clara: Parquet streaming was selected, but Problem still "
          "contains resident series; use a settings-only Problem to avoid "
          "resident-plus-chunk memory.");
      if (!reader.rows_are_series()) {
        throw InvalidInput(
          "fast_clara: RAM-limited Parquet streaming requires one series per "
          "row (a list column, or several Float32/Float64 columns); a single "
          "scalar column is one time series and cannot be streamed as "
          "independent clustering points.");
      }
      const auto stream_plan = detail::resolve_clara_plan(
        reader.logical_series_count(), opts, "fast_clara");
      if (prob.verbose())
        std::cout << "FastCLARA: streaming from Parquet ("
                  << reader.logical_series_count()
                  << " rows, " << reader.num_row_groups() << " row groups, ~"
                  << resident_bytes / (1ULL << 20) << " MB resident estimate)\n";
      return opts.use_float32 ? fast_clara_chunked<true>(prob, opts, reader, stream_plan)
                              : fast_clara_chunked<false>(prob, opts, reader, stream_plan);
    }
  }
  // The series fit the limit, or no stream was asked for: the Problem's own series.
  CLARAOptions in_ram = opts;
  in_ram.parquet_path.clear();
  return fast_clara(prob, in_ram);
}

} // namespace dtwc::algorithms

#endif // DTWC_HAS_PARQUET
