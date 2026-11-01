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
#include "../fast_pam.hpp" // fast_pam_thread_bytes
#include "../../Data.hpp"
#include "../../Problem.hpp"
#include "../../base/error.hpp"
#include "../../base/missing_utils.hpp"
#include "../../base/parallelisation.hpp" // TeamCap
#include "../../core/dtw_dispatch.hpp"    // thread_scratch_bytes
#include "../../core/thread_scratch.hpp"  // KernelTeam
#include "../../core/medoid_assignment_policy.hpp"
#include "../../warping.hpp" // detail::all_usable, refuse_unusable

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
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
 * file's ndim, for a file of more than one). Static, so each including file's instantiation is local to it, as it was
 * in fast_clara_parquet.cpp: GCC inlines the assignment into its caller only then (2026-11-01).
 *
 * The threads' scratch (`team`, within an explicit limit): each thread's DTW kernels' for the longest series it meets,
 * FasterPAM's and the assignment's tables, kept to the end of the call. The PAMs' team is the most that fit beside the
 * PAMs, the pass beside one piece (the medoids at most `slots` of the union's longest series) and its end; a block of
 * longer series than any before fits the assignment's team again, before it is assigned, beside the blocks in flight.
 */
template <bool F32, typename Reader>
static core::ClusteringResult fast_clara_chunked(Problem &prob, const CLARAOptions &opts, const Reader &reader,
                                                 const ClaraPlan &plan, core::KernelTeam &team)
{
  validate_streaming_clara_plan(plan, "fast_clara");
  if constexpr (Reader::decodes_with_team)
    if (prob.device().first == Device::GPU)
      throw DeviceError("fast_clara: a GPU assigns owned series in RAM, but a file streamed a record batch at a time "
                        "(--ram-limit) is read as views of its batches; no backend call or CPU fallback was "
                        "attempted. Use device cpu.");
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

  // What each thread holds of scratch for series of up to `length` time steps: the DTW kernels' where the CPU runs
  // them (a GPU fills the samples' matrices; CUDA's assignment is its own), FasterPAM's swap table, the assignment's.
  const auto precision = F32 ? core::Precision::Float32 : core::Precision::Float64;
  const std::size_t slots = samples.size() * static_cast<std::size_t>(opts.n_clusters);
#ifdef DTWC_HAS_CUDA
  const bool cpu_assignment = F32 || prob.device().first != Device::GPU;
#else
  const bool cpu_assignment = true; // Metal has no kernel for the assignment
#endif
  const bool cpu_kernels = cpu_assignment || prob.device().first != Device::GPU;
  const auto per_thread = [&](std::size_t length) {
    return (cpu_kernels ? core::thread_scratch_bytes(prob.distance(), length, precision) : 0)
           + dtwc::detail::fast_pam_thread_bytes(opts.n_clusters)
           + (cpu_assignment ? clara_assign_thread_bytes(slots, precision) : 0);
  };
  std::size_t longest = 0; // time steps: of the union's series, then of every block's so far

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
    // The PAMs' threads keep their scratch for the union's longest series to the end: their team fits beside the
    // PAMs, the pass's least (one piece, the medoids at most `slots` of those series) and its end.
    for (std::size_t i = 0; i < rows.size(); ++i) longest = std::max(longest, pool.series_flat_size(i) / pool.ndim);
    const size_t medoids_most = Reader::series_bytes(slots * longest * pool.ndim, slots, F32);
    const size_t pass_least = medoids_most + clara_pass_bytes(plan, samples.size()) + reader.decode_bytes()
                              + reader.largest_row_group_bytes(F32);
    const size_t end_least = open + medoids_most + samples.size() * n_points * sizeof(index_t)
                             + n_points * sizeof(index_t) + n_points / 8;
    const size_t rest = std::max(std::max(open + union_bytes + pam_bytes, pass_least), end_least);
    const TeamCap pam_team(team.fit("FastCLARA's PAMs", rest, per_thread(longest), longest));
    pams = clara_pams(prob, opts, pool, samples, positions, true, medoids);
    medoid_data = clara_sample(pool, positions_in(rows, medoids), true);
  }
  core::release_free_memory(); // the union's and its read's memory, before the labels

  const size_t limit = opts.ram_limit_bytes;
  std::size_t medoid_values = 0;
  for (std::size_t j = 0; j < medoids.size(); ++j) medoid_values += medoid_data.series_flat_size(j);
  const size_t medoid_bytes = Reader::series_bytes(medoid_values, medoids.size(), F32);
  const size_t fixed_rest = medoid_bytes + clara_pass_bytes(plan, samples.size()) + reader.decode_bytes();
  const size_t group = reader.largest_row_group_bytes(F32);
  // After the pass the Problem copies the winner's labels (set_result) beside every sample's, the file still open.
  const size_t finish_rest = open + medoid_bytes + samples.size() * n_points * sizeof(index_t)
                             + n_points * sizeof(index_t) + n_points / 8;
  // The threads' scratch, held to the end (team; 0 under the automatic budget).
  const size_t fixed = fixed_rest + team.priced(), finish = finish_rest + team.priced();
  if (std::max(fixed + group, finish) > limit)
    throw InvalidInput(
      "fast_clara: ram_limit_bytes (--ram-limit) = " + std::to_string(limit) + " cannot hold the pass over the "
      "file, which needs " + std::to_string(std::max(fixed + group, finish)) + " bytes: " + std::to_string(fixed)
      + " for the medoids' series, every sample's labels and nearest distances (n_samples = "
      + std::to_string(samples.size()) + ", k = " + std::to_string(opts.n_clusters) + ", "
      + std::to_string(n_points) + " rows), decoding and the threads' DTW scratch, " + std::to_string(group)
      + " for one " + Reader::piece + "'s series, " + std::to_string(finish) + " at its end with the result's labels"
      + (open != 0 ? " and the open file" : "") + "; raise the limit, or rewrite the file with smaller "
      + Reader::pieces + ".");
  // A batch decoding beside the one assigned; a team decodes in turn with the assignment.
  const bool overlap = !Reader::decodes_with_team && fixed + 2 * group <= limit;
  const int per_batch = reader.row_groups_per_batch(overlap ? (limit - fixed) / 2 : limit - fixed, F32);

  std::vector<std::vector<index_t>> labels(samples.size()); // each sized in place: a prototype would add n_points
  for (auto &sample_labels : labels) sample_labels.resize(n_points);
  std::vector<core::detail::OrderedMedoidObjective> costs(samples.size());
#ifdef DTWC_HAS_CUDA
  bool on_gpu = false;
  if constexpr (!F32) // Float32 series never reach a GPU: the sample fill refuses them first
    on_gpu = prob.device().first == Device::GPU;
#else
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
      // A block of longer series than any before grows the assignment's threads' scratch: their team fitted again,
      // beside the blocks in flight and the pass's end, before it is assigned.
      std::size_t block_longest = 0;
      for (std::size_t i = 0; i < static_cast<std::size_t>(batch.size()); ++i)
        block_longest = std::max(block_longest, batch.series_flat_size(i) / batch.ndim);
      if (block_longest > longest) {
        longest = block_longest;
        const size_t blocks = (overlap ? 2 : 1) * static_cast<size_t>(per_batch) * group;
        team.fit("FastCLARA's assignment", std::max(fixed_rest + blocks, finish_rest), per_thread(longest), longest);
      }
      const TeamCap assignment_team(team.threads());
      // The batch is loaded: the assignment reads it, never the reader.
#ifdef DTWC_HAS_CUDA
      if (on_gpu)
        assign_on_gpu(prob, batch.p_vec, first, medoid_data.p_vec, medoids, labels, costs);
      else
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

/// fast_clara_chunked, its threads' scratch within the Problem's explicit limit (Problem::ram_limit, which --ram-limit
/// sets: ram_limit_bytes then); under the automatic budget (0) no team is fitted, as nothing is refused for it.
template <bool F32, typename Reader>
static core::ClusteringResult fast_clara_chunked(Problem &prob, const CLARAOptions &opts, const Reader &reader,
                                                 const ClaraPlan &plan)
{
  core::KernelTeam team(prob.ram_limit() != 0 ? opts.ram_limit_bytes : 0, prob.verbose());
  return fast_clara_chunked<F32>(prob, opts, reader, plan, team);
}

} // namespace dtwc::algorithms::detail
