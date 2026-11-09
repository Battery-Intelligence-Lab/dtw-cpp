/**
 * @file fast_clara_stream.hpp
 * @brief FastCLARA over series read piece by piece from a file, on any reader of its pieces: Parquet's row groups
 *        (io::ParquetChunkReader, fast_clara_parquet.cpp in dtwc_io) and Arrow IPC's record batches (io::IpcFile, the
 *        core's), each instantiation's assignment the same code.
 * @author Volkan Kumtepeli
 */

#pragma once

#include "fast_clara_assign.hpp"
#include "fast_clara_plan.hpp"
#include "../fast_clara.hpp"
#include "../../Data.hpp"
#include "../../Problem.hpp"
#include "../../base/error.hpp"
#include "../../base/missing_utils.hpp"
#include "../../core/medoid_assignment_policy.hpp"
#include "../../warping.hpp" // detail::all_usable, refuse_unusable

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dtwc::algorithms::detail {

/// Where each of `rows` sits in `sorted`, which holds every one of them.
inline std::vector<index_t> positions_in(const std::vector<index_t> &sorted, const std::vector<index_t> &rows)
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
 * narrower than the length difference of the shortest and the longest rows rows() has seen (in time steps, as the
 * in-RAM check counts them), the first of each. A team of readers checks each batch in a RowCheck of its own and
 * merge()s them in batch order: the first of each again.
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
    for (std::size_t i = 0; i < static_cast<std::size_t>(data.size()); ++i) {
      const auto row = static_cast<int64_t>(row_of(i));
      const auto flat = data.series_flat_size(i);
      if (flat == 0)
        throw InvalidInput("fast_clara: series " + std::to_string(row) + " ('" + std::string(data.name(i))
                           + "') is empty; every series needs at least one value.");
      const auto length = flat / data.ndim; // whole time steps: the reader's Data checked them
      // The in-RAM check's predicate (Problem's check); the name only for a refusal.
      const auto name = [&] {
        return "series '" + std::string(data.name(i)) + "' (index " + std::to_string(row) + ")";
      };
      const auto check = [&](const auto values) {
        if (!dtwc::detail::all_usable(values, nan_is_missing))
          dtwc::detail::refuse_unusable(values, name(), "fast_clara", nan_is_missing);
        if (distance_.missing == core::MissingStrategy::Interpolate && all_missing(values))
          throw InvalidInput("fast_clara: " + name() + " is entirely NaN, so MissingStrategy::Interpolate has "
                             "nothing to interpolate from. Use ZeroCost or AROW, or drop the series.");
      };
      if (data.is_f32())
        check(data.series_f32(i));
      else
        check(data.series(i));
      if (length < shortest_.length) shortest_ = { length, row, std::string(data.name(i)) };
      if (length > longest_.length) longest_ = { length, row, std::string(data.name(i)) };
    }
  }

  /// The extremes of `later`, whose rows follow these, as if rows() had read them after: a tie keeps the earlier.
  void merge(const RowCheck &later)
  {
    if (later.shortest_.length < shortest_.length) shortest_ = later.shortest_;
    if (later.longest_.length > longest_.length) longest_ = later.longest_;
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
 * @brief FastCLARA streamed from `reader`, its series read as T (float with F32, else data_t).
 *
 * All samples are drawn first; the rows they hold, their union, are read once and each sample's PAM runs on a copy
 * of its series. The medoids' series are copied out and the union released. Then one pass over the file: a batch
 * of the reader's pieces at a time, every point against the medoids of every sample. Within ram_limit_bytes, as the
 * reader prices them: the union beside its decode (read_rows' check), then a sample's PAM beside the union and what
 * the open file holds, then the medoids, every sample's labels and one block of nearest distances beside the decode
 * and the batches, and at the end the labels beside the Problem's copy and the open file. The union, and each batch as
 * it is read, go through the in-RAM route's checks (RowCheck): the stream refuses what the in-RAM route refuses,
 * naming the row.
 *
 * The reader: read_rows<T>(rows, limit), num_row_groups(), row_groups_per_batch(budget, f32),
 * largest_row_group_bytes(f32), decode_bytes(), open_bytes() (what it holds while open, beside any read: IpcFile's
 * walk), series_bytes(values, rows, f32), and the traits piece and pieces (what its refusals call a batch's parts) and
 * decodes_with_team. Without a team (Parquet) read_row_groups<T>(first, count)
 * decodes on one thread, which checks the batch and, when two batches fit, overlaps the assignment of the one before.
 * With a team (Arrow IPC) read_row_groups<T>(first, count, check) decodes a batch's pieces in parallel and hands
 * check(j, piece, first_row) each one from the thread that decoded it, while it is in that thread's cache: each piece
 * has a RowCheck of its own, their extremes merged in piece order, and the reader rethrows the lowest failing piece's
 * refusal (run_openmp), so the verdict is the serial check's; the batch is assigned after, and freed (its memory
 * returned to the system, core::release_free_memory) before the next is read; so is the union's, before the pass.
 *
 * `prob` holds no series; its distance takes the series' features a time step from its data (an empty Data of the
 * file's ndim, for a file of more than one). Static, so each including file's instantiation is local to it: GCC
 * inlines the assignment into its caller only then (2026-11-01).
 */
template <bool F32, typename Reader>
static core::ClusteringResult fast_clara_chunked(Problem &prob, const CLARAOptions &opts, const Reader &reader,
                                                 const ClaraPlan &plan)
{
  validate_streaming_clara_plan(plan, "fast_clara");
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
  const auto samples = draw_clara_samples(plan.n_points, plan.sample_size, opts);

  const std::vector<int64_t> rows = sample_union(samples);
  const size_t open = reader.open_bytes(); // held beside every phase

  std::vector<index_t> medoids; // sample s's medoid m is slot s * k + m
  std::vector<core::ClusteringResult> pams;
  Data medoid_data;
  {
    const Data pool = reader.template read_rows<T>(rows, opts.ram_limit_bytes);
    if (pool.ndim != prob.distance().ndim)
      throw std::logic_error("fast_clara: the file's series have " + std::to_string(pool.ndim) + " features a time step "
                             "and the Problem's distance " + std::to_string(prob.distance().ndim)
                             + ": a stream's Problem holds an empty Data of the series' ndim");
    RowCheck sampled(prob.distance());
    sampled.rows(pool, [&](std::size_t i) { return rows[i]; });
    sampled.band();
    std::vector<std::vector<index_t>> positions;
    // A handful of samples; no reserve, since this function's size decides whether GCC inlines the assignment (above).
    // NOLINTNEXTLINE(performance-inefficient-vector-operation)
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
    const size_t union_bytes = Reader::series_bytes(union_values, rows.size(), F32);
    const size_t pam_bytes = Reader::series_bytes(sample_values, s, F32)
                             + core::packed_size(s) * core::value_bytes(prob.matrix_dtype());
    if (open + union_bytes + pam_bytes > opts.ram_limit_bytes)
      throw InvalidInput("fast_clara: ram_limit_bytes (--ram-limit) = " + std::to_string(opts.ram_limit_bytes)
                         + " cannot hold a sample's PAM beside the samples' series, which needs "
                         + std::to_string(open + union_bytes + pam_bytes) + " bytes: "
                         + (open != 0 ? std::to_string(open) + " the open file holds, " : std::string())
                         + std::to_string(union_bytes) + " for the " + std::to_string(rows.size())
                         + " sampled series, " + std::to_string(pam_bytes)
                         + " for a sample's copy and its distance matrix (sample_size = " + std::to_string(s)
                         + "); raise the limit, or lower sample_size.");
    pams = clara_pams(prob, opts, pool, samples, positions, true, medoids);
    medoid_data = clara_sample(pool, positions_in(rows, medoids), true);
  }
  core::release_free_memory(); // the union's and its read's memory, before the labels

  const size_t limit = opts.ram_limit_bytes;
  // CUDA assigns a batch from its staging buffer, the medoids' series and the batch's padded to the longest (pinned
  // where it can be; launch_medoid_kernel), and uploads owned series, a batch that is a view (Arrow IPC's) as an owned
  // copy, freed after it: on a GPU the medoids are held twice, and a batch beside its staging and its copy, each about
  // the batch's bytes (the staging more where its series' lengths differ). Metal assigns on the CPU.
#ifdef DTWC_HAS_CUDA
  bool on_gpu = false;
  if constexpr (!F32) // Float32 series never reach a GPU: the sample fill refuses them first
    on_gpu = prob.device().first == Device::GPU;
#else
  constexpr bool on_gpu = false;
#endif
  const size_t times = on_gpu ? (Reader::decodes_with_team ? 3 : 2) : 1; // a batch, its copy and its staging
  std::size_t medoid_values = 0;
  for (std::size_t j = 0; j < medoids.size(); ++j) medoid_values += medoid_data.series_flat_size(j);
  const size_t medoid_bytes = Reader::series_bytes(medoid_values, medoids.size(), F32);
  const size_t fixed = (on_gpu ? 2 : 1) * medoid_bytes + clara_pass_bytes(plan, samples.size()) + reader.decode_bytes();
  const size_t one = reader.largest_row_group_bytes(F32), group = times * one;
  // After the pass Problem::set_result copies the winner's labels into the Problem (n_points labels) and builds its
  // is_medoid bitmap (n_points / 8 bytes), beside every sample's labels, the file still open.
  const size_t finish = open + medoid_bytes + samples.size() * n_points * sizeof(index_t) + n_points * sizeof(index_t)
                      + n_points / 8;
  if (std::max(fixed + group, finish) > limit)
    throw InvalidInput(
      "fast_clara: ram_limit_bytes (--ram-limit) = " + std::to_string(limit) + " cannot hold the pass over the "
      "file, which needs " + std::to_string(std::max(fixed + group, finish)) + " bytes: " + std::to_string(fixed)
      + " for the medoids' series" + (on_gpu ? " and the GPU's staging of them" : "")
      + ", every sample's labels and nearest distances (n_samples = " + std::to_string(samples.size()) + ", k = "
      + std::to_string(opts.n_clusters) + ", " + std::to_string(n_points) + " rows) and decoding, "
      + std::to_string(group) + " for one " + Reader::piece + "'s series"
      + (!on_gpu                       ? ""
         : Reader::decodes_with_team ? ", the owned copy the GPU uploads of them and its staging of them"
                                     : " and the GPU's staging of them")
      + ", " + std::to_string(finish) + " at its end with the result's labels"
      + (open != 0 ? " and the open file" : "") + "; raise the limit, or rewrite the file with smaller "
      + Reader::pieces + ".");
  // A batch decoding beside the one assigned; a team decodes in turn with the assignment.
  const bool overlap = !Reader::decodes_with_team && fixed + group + one <= limit;
  const int per_batch = reader.row_groups_per_batch((limit - fixed) / (overlap ? times + 1 : times), F32);

  std::vector<std::vector<index_t>> labels(samples.size()); // each sized in place: a prototype would add n_points
  for (auto &sample_labels : labels) sample_labels.resize(n_points);
  std::vector<core::detail::OrderedMedoidObjective> costs(samples.size());
#ifndef DTWC_HAS_CUDA
  say_cpu_assignment(prob, n_points, samples.size(), static_cast<std::size_t>(opts.n_clusters));
#endif

  // The reader and `check` are used by one thread at a time: the decoding one while a batch decodes and is checked
  // (its first row known, as the batches before it are), the team only after the last batch.
  RowCheck check(prob.distance());
  const auto decode = [&reader, &check, &prob](int rg, int count, index_t first_row) {
    if constexpr (Reader::decodes_with_team) {
      std::vector<RowCheck> pieces(static_cast<std::size_t>(count), RowCheck(prob.distance()));
      const auto piece_check = [&pieces](std::size_t j, const Data &piece, index_t first) {
        pieces[j].rows(piece, [first](std::size_t i) { return first + static_cast<index_t>(i); });
      };
      Data batch = reader.template read_row_groups<T>(rg, count, piece_check);
      for (const RowCheck &piece : pieces) check.merge(piece);
      return batch;
    } else {
      (void)prob;
      Data batch = reader.template read_row_groups<T>(rg, count);
      check.rows(batch, [first_row](std::size_t i) { return first_row + static_cast<index_t>(i); });
      return batch;
    }
  };
  const auto policy = overlap ? std::launch::async : std::launch::deferred;
  const int total_rg = reader.num_row_groups();
  std::future<Data> next = std::async(policy, decode, 0, std::min(per_batch, total_rg), index_t{ 0 });
  index_t first = 0;
  for (int rg = 0; rg < total_rg; rg += per_batch) {
    {
      const Data batch = next.get();
      if (rg + per_batch < total_rg)
        next = std::async(policy, decode, rg + per_batch, std::min(per_batch, total_rg - rg - per_batch),
                          first + batch.size());
      // The batch is loaded: the assignment reads it, never the reader.
#ifdef DTWC_HAS_CUDA
      if (on_gpu) {
        std::optional<Data> copy; // a view's owned copy, which the GPU uploads
        if (batch.is_view()) copy.emplace(dtwc::detail::owned_copy(batch, core::Precision::Float64));
        assign_on_gpu(prob, copy ? copy->p_vec : batch.p_vec, first, medoid_data.p_vec, medoids, labels, costs);
      } else
#endif
        assign_in_blocks(first, static_cast<std::size_t>(batch.size()), costs,
                         [&](index_t block_first, std::size_t n, std::span<double> best) {
                           assign_points_cpu<T>(
                             block_first, n, best, medoids, labels, prob.distance(), pair,
                             [&](index_t p) { return series_at(batch, p - first); },
                             [&](index_t j) { return series_at(medoid_data, j); });
                         });
      first += batch.size();
    }
    // A team's threads freed the batch's pieces (into their own arenas on glibc; Windows' heap keeps them committed):
    // returned to the system before the next.
    if constexpr (Reader::decodes_with_team) core::release_free_memory();
  }
  // The end phase is priced as if the batches went back to the system; glibc kept 320-680 MB (1 GiB, 2026-10-09).
  if constexpr (!Reader::decodes_with_team) core::release_free_memory();
  check.band();

  auto result = clara_winner(pams, medoids, labels, costs);
  prob.set_result(result);
  return result;
}

} // namespace dtwc::algorithms::detail
