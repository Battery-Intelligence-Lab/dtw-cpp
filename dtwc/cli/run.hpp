/**
 * @file run.hpp
 * @brief dtwc::run — one clustering run of a dtwc::Config, the pipeline behind
 *        dtwc_cl and Tier-1 cluster().
 *
 * @details run() checks the whole configuration before it reads a series,
 * resolves the method and the device together, loads the input (CSV / TSV,
 * a folder, Parquet or Arrow IPC) into RAM, computes, and writes the outputs
 * into `output` (nothing when it is empty).
 *
 * | method        | cpu                           | gpu                               |
 * |---------------|-------------------------------|-----------------------------------|
 * | auto          | pam for N <= 5000, else clara | pam                               |
 * | onebatch      | runs                          | DeviceError                       |
 * | tadpole       | runs                          | DeviceError                       |
 * | clara         | runs                          | runs when its sample covers every |
 * |               |                               | series, else DeviceError          |
 * | pam, kmedoids, mip, lrcore, hierarchical | run | run; the GPU fills the matrix     |
 *
 * `--device hpc` is refused while the configuration is parsed (DeviceError): a
 * run computes where it starts, and SLURM submission belongs to Python's
 * dtwcpp.device("hpc") and `slurm_remote.sh`. `gpu` on a build without a GPU backend
 * raises the api-contract-2.0.md §6.1 DeviceError; on a GPU a variant,
 * missing-data strategy, Float32 dtype, index or precision the backend does not
 * implement raises validate_gpu_request()'s DeviceError, all before any I/O.
 *
 * @date 24 Sep 2026
 */

#pragma once

#include "../api.hpp"
#include "config.hpp"

#include <cstddef>
#include <iosfwd>

namespace dtwc {

/// Cluster the series `config.input` names. Config's strings are UTF-8.
/// @throws InvalidInput for a setting no run can honour (an empty input, k,
///         max-iter or n-init below 1, a format option the input cannot take,
///         k above the number of series), DeviceError as above, IOError for an
///         input this build cannot read or a file that cannot be read or written,
///         SolverError for `gurobi` on a build without Gurobi.
Result run(const Config &config);

/// Cluster `data`, already in memory: `input` and the options only a file
/// reader applies (`column`, `skip-rows`, `skip-cols`, `delimiter`,
/// `ram-limit`) must be at their defaults.
Result run(const Config &config, Data data);

namespace detail {

/// How a Parquet input stores its series, as its metadata shows it.
enum class ParquetLayout { ListColumn, ScalarColumn, Directory };

/// What run() does with a Parquet input, decided from metadata alone.
struct ParquetPlan
{
  Method method;            ///< `auto` resolved for the device and the series count
  bool stream_payload;      ///< the series exceed `ram_limit`, so FastCLARA streams them
};

/// Decide from Parquet metadata whether reading the payload is legal: only
/// FastCLARA on a single list-per-row file streams under `ram_limit` (0 = no cap).
/// @throws InvalidInput for no series, or a route that cannot honour the cap.
ParquetPlan plan_parquet_load(Method method, Device device, std::size_t series_count,
                              std::size_t estimated_resident_bytes, std::size_t ram_limit,
                              ParquetLayout layout);

/**
 * @brief Write a clustered Problem's result files into `directory`, which is created if missing.
 *
 * `<name>_labels.csv` and `<name>_medoids.csv`, then, when the distance matrix is filled,
 * `<name>_distance_matrix.csv` and, for k > 1, `<name>_silhouettes.csv`. An undefined silhouette is a
 * warning on stderr; any other failure propagates. A Problem without series (a RAM-limited Parquet run)
 * names its series `series_<i>` and has no matrix or silhouettes to write.
 *
 * @param complete  true (Result::save): fill the matrix first, so all four files are written, and throw
 *                  InvalidInput after the labels and medoids when there is no matrix to fill. false (the
 *                  CLI): write the matrix and silhouettes only if the matrix is already filled, so a
 *                  matrix-free run does not fill O(N^2) for them.
 * @param progress  when set, one line per file written.
 * @throws IOError for a file that cannot be written in full.
 */
void write_result_files(Problem &prob, const std::filesystem::path &directory, bool complete,
                        std::ostream *progress = nullptr);

} // namespace detail
} // namespace dtwc
