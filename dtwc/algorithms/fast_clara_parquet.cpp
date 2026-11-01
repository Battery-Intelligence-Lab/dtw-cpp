/**
 * @file fast_clara_parquet.cpp
 * @brief FastCLARA over series streamed from Parquet: dtwc_io, in a build with Parquet.
 *
 * @details The RAM-limited route of dtwc_cl --ram-limit. The main Problem holds settings only; each
 *   sample, the medoids and the assignment's row groups are read from the file as they are needed, by
 *   detail::fast_clara_chunked (detail/fast_clara_stream.hpp, which also streams Arrow IPC). It moved out
 *   of fast_clara.cpp so that the core, which the Python module and the MEX link, holds no Parquet.
 * @author Volkan Kumtepeli
 */

#ifdef DTWC_HAS_PARQUET

#include "fast_clara.hpp"
#include "detail/fast_clara_plan.hpp"
#include "detail/fast_clara_stream.hpp"
#include "../Problem.hpp"
#include "../base/error.hpp"
#include "../core/thread_scratch.hpp" // KernelTeam
#include "../io/parquet_chunk_reader.hpp"

#include <iostream>

namespace dtwc::algorithms {

core::ClusteringResult detail::fast_clara_parquet(Problem &prob, const CLARAOptions &opts, core::KernelTeam &team)
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
      return opts.use_float32 ? detail::fast_clara_chunked<true>(prob, opts, reader, stream_plan, team)
                              : detail::fast_clara_chunked<false>(prob, opts, reader, stream_plan, team);
    }
  }
  // The series fit the limit, or no stream was asked for: the Problem's own series.
  CLARAOptions in_ram = opts;
  in_ram.parquet_path.clear();
  return fast_clara(prob, in_ram);
}

core::ClusteringResult fast_clara_parquet(Problem &prob, const CLARAOptions &opts)
{
  core::KernelTeam team(opts.ram_limit_bytes, prob.verbose()); // the stream's limit, its threads' scratch within it
  return detail::fast_clara_parquet(prob, opts, team);
}

} // namespace dtwc::algorithms

#endif // DTWC_HAS_PARQUET
