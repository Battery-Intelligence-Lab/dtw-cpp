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
 * | auto          | pam for N <= 5000, else clara;| pam; clara where pam's matrix     |
 * |               | clara where pam's matrix can  | can neither stay in RAM nor be    |
 * |               | neither stay in RAM nor be    | mapped                            |
 * |               | mapped (the memory plan)      |                                   |
 * | onebatch      | runs                          | DeviceError                       |
 * | tadpole       | runs                          | DeviceError                       |
 * | clara         | runs                          | runs; the GPU fills the samples   |
 * |               |                               | (and, on CUDA, assigns the series)|
 * | pam, kmedoids, mip, lrcore, hierarchical | run | run; the GPU fills the matrix     |
 *
 * `--device hpc` is refused while the configuration is parsed (DeviceError): a
 * run computes where it starts, and SLURM submission belongs to Python's
 * dtwcpp.device("hpc") and `slurm_remote.sh`. `gpu` on a build without a GPU backend
 * raises a DeviceError; on a GPU a variant,
 * missing-data strategy, Float32 dtype or precision the backend does not
 * implement raises validate_gpu_request()'s DeviceError, and a GPU index Metal
 * cannot honour Problem::set_device's, all before any I/O.
 *
 * @author Volkan Kumtepeli
 * @date 24 Sep 2026
 */

#pragma once

#include "../api.hpp"
#include "config.hpp"

#include <cstddef>
#include <iosfwd>

namespace dtwc {

/// Cluster the series `config.input` names. Config's strings are UTF-8.
///
/// The run plans its memory once, where its input's facts are known (an Arrow IPC file's walk, a Parquet file's
/// metadata, else the series read): within the budget (core::memory_budget), a reserve for its threads' working set,
/// 1.1 x threads x a thread's bytes (the DTW kernels' scratch for the longest series it expects and the method's
/// tables), and the series, the method's own terms (TADPole's pair flags and cache, the hierarchical clustering's
/// dense copy, OneBatchPAM's batch table) and the distance matrix in what is left, the data cap. The series go into
/// RAM on every thread where they fit in it beside the matrix, which is mapped where it does not (llfio and an output
/// directory to map it in) and else held in RAM; else FastCLARA streams them within the data cap where it can (an
/// Arrow IPC or Parquet file, a sample smaller than N), on every thread; else they load on the most threads whose
/// working set fits, the threads fitted once. `-v` prints the plan. It is a plan, not a cap: a run may use about a
/// tenth more, more with series longer than planned; MIP's and LR-core's solver memory is not planned.
/// @throws InvalidInput for a setting no run can honour (an empty input, k,
///         max-iter or n-init below 1, a format option the input cannot take,
///         k above the number of series, an explicit --ram-limit that cannot
///         hold the run, before any work: in RAM naming the least limit that
///         runs ("at least N bytes"); on a stream naming what the plan holds,
///         the stream's own floor coming on top: the stream checks it within
///         the data cap, which its refusal names as --ram-limit, and fewer
///         threads, OMP_NUM_THREADS, lower the reserve; and in place of a
///         std::bad_alloc, the system refusing memory to the run, a reader or a
///         library (nanoarrow's ENOMEM, Arrow's out-of-memory status, a map
///         llfio is refused memory for), once the run has given its memory
///         back: "<stage> stage: out of memory; plan: <the plan>; the plan
///         holds H bytes in RAM; the process peaked at P bytes resident, mapped
///         files' pages included; ways out: ...", the stages check, plan, read,
///         storage, cluster and write, the plan "none yet" until it is made
///         (a text file's series are planned once read), each way out named
///         where it holds less in this run), DeviceError as above,
///         IOError for an
///         input this build cannot read or a file that cannot be read or written,
///         SolverError for `gurobi` on a build without Gurobi. Where the system
///         refuses a small allocation in a parallel region that cannot carry an
///         exception (TADPole's two, with an explicit --dc), on a thread's stack
///         or in the process's start-up, the process ends without a report
///         (Windows 0xC0000409, or 0xC00000FD on a stack; Linux an abort).
Result run(const Config &config);

/// Cluster `data`, already in memory: `input` and the options only a file
/// reader applies (`column`, `skip-rows`, `skip-cols`, `delimiter`) must be at
/// their defaults.
Result run(const Config &config, Data data);

/// The method `auto` stands for within run()'s memory plan: resolve_method(method, device, n_series) (Problem.hpp),
/// and clara where pam's distance matrix can be held neither in RAM beside the series and one thread's working set
/// nor mapped (`pam_matrix_holds` false). Any other method is itself. Tier-2 Problem::cluster resolves without it.
/// The method follows the limit monotonically, clara below the switch and pam above it; the threads the plan fits
/// and the clustering are each resolved method's, so pam just above the switch may run on fewer threads than clara
/// just below it, with pam's clustering.
Method resolve_method(Method method, Device device, std::size_t n_series, bool pam_matrix_holds);

namespace core {
/// The memory a run plans for (--ram-limit): `ram_limit`, else 90 % of what the process may use (available_memory(),
/// a Linux cgroup's limit among it); 0 where neither is known (a build without llfio): no plan.
std::size_t memory_budget(std::size_t ram_limit);
} // namespace core

namespace detail {

/// How a Parquet input stores its series, as its metadata shows it: one file of
/// one series per row (a list column, or the row layout of several Float32/Float64
/// columns: ListColumn), one file whose one scalar column is one series, or a folder.
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

} // namespace detail
} // namespace dtwc
