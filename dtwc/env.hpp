/**
 * @file env.hpp
 * @brief Process-wide compute environment: device / thread policy (Task 1.3).
 *
 * @details `dtwc::Env` owns the selected compute device (`cpu` / `gpu` / `hpc`)
 * and the thread policy; the language-level `device()` setters (Python/MATLAB/CLI)
 * delegate to the single `dtwc::env()` instance so every front-end shares one
 * source of truth (api-contract-2.0.md §6).
 *
 * **No silent fallback (global constraint).** A device request that cannot be
 * honoured raises `dtwc::DeviceError` (dtwc/error.hpp) with an actionable message
 * — it never quietly degrades to CPU:
 *   - an unknown device name lists the valid names,
 *   - `gpu` on a build with no GPU backend names the CMake flag to rebuild with,
 *   - `hpc` reads SLURM credentials from a `.env` file and reports the exact
 *     failure (missing file / missing key / auth failure) — never a stack trace.
 *
 * The three `.env` failure messages are authored verbatim in api-contract-2.0.md
 * §6.2 and are asserted byte-for-byte in tests/unit/test_env_device.cpp.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 07 Jul 2026
 */

#pragma once

#include "error.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace dtwc {

/// @brief Compute device selected via dtwc::Env (api-contract-2.0.md §6.1).
enum class Device {
  CPU, ///< Local CPU execution (the default).
  GPU, ///< Local GPU execution (CUDA on NVIDIA, Metal on macOS).
  HPC  ///< Remote SLURM cluster; credentials read from `.env`.
};

/// @brief Canonical lower-case name of a Device ("cpu" / "gpu" / "hpc").
std::string to_string(Device d);

namespace detail {

/// @brief Why DTWC++ would run single-threaded — drives the Env sequential
///        loudness warning (Task 3.2). `None` means parallelism is available.
enum class SeqCause {
  None,               ///< Parallel execution available — no warning.
  RuntimeSingleThread,///< OpenMP present but only 1 thread usable (e.g. OMP_NUM_THREADS=1) on a multicore host.
  SequentialBuild     ///< Compiled without OpenMP via -DDTWC_ALLOW_SEQUENTIAL=ON (DTWC_SEQUENTIAL_BUILD).
};

/// @brief Decide whether (and why) to emit the sequential warning. PURE — depends
///        only on its arguments, so the message/predicate is unit-testable without
///        touching global OpenMP state or the build configuration.
/// @param effective_max_threads  omp_get_max_threads() (or 1 without OpenMP).
/// @param hw_concurrency         std::thread::hardware_concurrency() (0 = unknown).
/// @param sequential_build       true iff compiled with DTWC_SEQUENTIAL_BUILD.
/// @return SequentialBuild if @p sequential_build; else RuntimeSingleThread when
///         only 1 thread is usable on a host that reports >1 hardware thread; else
///         None (a genuine single-core host or unknown concurrency never warns —
///         no crying wolf).
SeqCause sequential_cause(int effective_max_threads, unsigned hw_concurrency,
                          bool sequential_build) noexcept;

/// @brief The exact stderr warning string for a given cause (empty for None).
/// @details Single source of truth for the two loudness messages emitted by the
///          dtwc::Env constructor; the strings are asserted byte-for-byte in
///          tests/unit/test_runtime_loudness.cpp. Each ends with a trailing '\n'.
std::string sequential_warning_text(SeqCause cause);

} // namespace detail

/**
 * @brief Owns the process-wide compute-device and thread policy.
 *
 * Reachable as the singleton dtwc::env(); also default-constructible as a local
 * instance (used by the unit tests to point the `.env` search at a scratch dir).
 * All device selection funnels through set_device(), so the no-silent-fallback
 * rules live in exactly one place.
 */
class Env
{
public:
  /// @brief `hpc` authentication probe: returns true iff (host,user) authenticate.
  /// @details Injectable so device="hpc" credential validation stays deterministic
  ///          and offline in tests; the default probe attempts a batch-mode ssh.
  using AuthProbe = std::function<bool(const std::string &host, const std::string &user)>;

  Env();

  /**
   * @brief Select the compute device by name (case-insensitive).
   * @param name One of: "cpu", "gpu", "gpu:N", "cuda", "cuda:N", "hpc"
   *             (`gpu` ≡ `cuda`; N is a non-negative GPU ordinal).
   * @throws dtwc::DeviceError on an unknown name, on `gpu` when the build has no
   *         GPU backend compiled in, or on any of the three device="hpc" `.env`
   *         failures (§6.2). On failure the previously selected device is left
   *         unchanged — the request never silently falls back to CPU.
   */
  void set_device(std::string_view name);

  /// @brief Currently selected device (Device::CPU until set otherwise).
  Device device() const { return device_; }

  /// @brief GPU ordinal from the last "gpu:N"/"cuda:N" selection (0 otherwise).
  int device_index() const { return device_index_; }

  /// @brief Resolved thread count for parallel regions (OpenMP max threads, else 1).
  int threads() const;

  // ---- configuration seams (no runtime dependence on repo-relative paths) ----

  /// @brief Directory searched for the device="hpc" `.env` file.
  /// @details Defaults to `$DTWC_REPO_ROOT` when set, else the current working
  ///          directory. Overridable so callers (and tests) never depend on a
  ///          repo-relative path baked in at build time.
  void set_env_file_dir(std::filesystem::path dir) { env_file_dir_ = std::move(dir); }

  /// @brief Directory currently searched for the `.env` file.
  const std::filesystem::path &env_file_dir() const { return env_file_dir_; }

  /// @brief Override the device="hpc" authentication probe (default: batch-mode ssh).
  void set_hpc_auth_probe(AuthProbe probe) { auth_probe_ = std::move(probe); }

private:
  /// Validate `.env` (file, keys) then authenticate; sets device_ only on success.
  void select_hpc();

  /// Emit ONE loud stderr warning (process-wide) when DTWC++ is running
  /// single-threaded — either forced to 1 thread at runtime or compiled without
  /// OpenMP (DTWC_SEQUENTIAL_BUILD). Called from the constructor (Task 3.2).
  void warn_if_sequential() const;

  Device device_{ Device::CPU };
  int device_index_{ 0 };
  std::filesystem::path env_file_dir_;
  AuthProbe auth_probe_;
};

/// @brief The process-wide Env singleton (api-contract-2.0.md §6).
Env &env();

/// @brief Emit ONE process-wide stderr warning when DTWC++ is running
///        single-threaded — the RuntimeSingleThread case (OpenMP present but only
///        1 usable thread on a multicore host, e.g. OMP_NUM_THREADS=1) or a
///        DTWC_SEQUENTIAL_BUILD (Task 3.6, review finding H1).
///
/// @details Task 3.2 wired this warning into the dtwc::Env constructor, but the
/// compute hot paths (Problem::fill_distance_matrix, the Python distance-matrix
/// bindings, direct C++ Problem use) never construct dtwc::env(), so on such a
/// build they ran SILENTLY single-threaded. Every compute path funnels through
/// dtwc::get_max_threads() (parallelisation.hpp), which now calls this; the Python
/// free function calls it directly too. Shares the single process-once guard with
/// the Env constructor, so a front-end that both builds env() and computes (the
/// CLI) still warns at most once. Thread-safe; O(1) atomic load after the first
/// call. The exact strings are the SSOT in detail::sequential_warning_text.
void warn_if_single_threaded();

} // namespace dtwc
