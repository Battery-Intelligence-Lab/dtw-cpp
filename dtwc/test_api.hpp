/**
 * @file test_api.hpp
 * @brief Header-only `dtwc::test` self-introspection API (Task 3.3).
 *
 * @details Two capability probes with an IDENTICAL result schema in C++, Python
 * (`dtwcpp.test.*`) and MATLAB (`dtwc_mex('test_parallelisation'|'test_gpu')`):
 *
 *   - `parallelisation()` runs a REAL OpenMP parallel region and counts the
 *     DISTINCT thread ids that actually executed — proof-of-engagement, not a
 *     compile-flag read. Fields: `available, max_threads, threads_engaged, pass,
 *     reason`. In a `DTWC_SEQUENTIAL_BUILD` (or any TU compiled without OpenMP)
 *     it returns `available=false` with a non-empty `reason` and NEVER throws.
 *
 *   - `gpu()` executes a tiny REAL GPU kernel (a small DTW distance matrix) and
 *     validates it against a CPU oracle (`dtwc::dtwFull_L`) to within
 *     `detail::kGpuOracleTol`. Fields: `available, backend, device_name,
 *     validated, pass, reason`. When no GPU backend is compiled in — or the
 *     backend is compiled but no device is present — it returns
 *     `available=false` with a `reason` naming EXACTLY what is missing, and
 *     NEVER throws, NEVER silently degrades (2.0 no-silent-fallback rule).
 *
 * Header-only by design: it adds NO new source file to any CMake target list.
 * The SAME source compiles and works in both GPU-ON and GPU-OFF builds — the
 * branch is at compile time on `DTWC_HAS_CUDA` / `DTWC_HAS_METAL` and at RUNTIME
 * on `cuda_available()` / `metal_available()`, so Task 3.5's CUDA build exercises
 * the `available -> validated==true` path through this very header while the
 * baseline CUDA-OFF build exercises the `available==false, reason non-empty` path.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#pragma once

#include "dtwc.hpp" // CPU oracle (dtwc::dtwFull_L / core::MetricType) + conditional cuda/metal decls

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwc::test {

/// @brief Result of the OpenMP parallelisation self-probe (same field names in
///        C++/Python/MATLAB).
struct ParallelReport
{
  bool available{ false };     ///< OpenMP compiled into THIS build and usable.
  int max_threads{ 1 };        ///< omp_get_max_threads() (1 without OpenMP).
  int threads_engaged{ 1 };    ///< DISTINCT thread ids observed in a real region.
  bool pass{ false };          ///< Parallelism genuinely engaged (see pass rule).
  std::string reason;          ///< Non-empty explanation when !available (empty otherwise).
};

/// @brief Result of the GPU-backend self-probe (same field names in
///        C++/Python/MATLAB).
struct GpuReport
{
  bool available{ false };     ///< A GPU backend is compiled in AND a device is present.
  std::string backend;         ///< "cuda" / "metal" / "" (none).
  std::string device_name;     ///< Human-readable device string (empty when unavailable).
  bool validated{ false };     ///< GPU kernel matched the CPU oracle within tolerance.
  bool pass{ false };          ///< available && validated.
  std::string reason;          ///< Non-empty explanation when !available or !validated.
};

namespace detail {

/// @brief Oracle tolerance for the CUDA FP64 path (the contract figure, §Task 3.3).
///        CUDA is forced to FP64 below so this ≤1e-12 bound is meetable and is the
///        band runtime-verified on the local RTX 4000 Ada in Task 3.5.
inline constexpr double kGpuOracleTol = 1e-12;

/// @brief Oracle tolerance for the Metal path. Metal has no FP64, so the kernel
///        accumulates in FP32; a 1e-12 bound would be a guaranteed false-negative
///        on legitimate Apple hardware. This FP32-appropriate absolute bound is
///        used ONLY for the Metal branch (compiled-but-verified-on-macOS-CI).
inline constexpr double kMetalOracleTol = 1e-4;

/// @brief Deterministic, non-degenerate probe series for the GPU-vs-CPU check.
///        Length 8, varied shape (not constant/symmetric) so a wrong kernel
///        cannot pass by a gauge coincidence.
inline std::vector<std::vector<double>> gpu_probe_series()
{
  return {
    { 0.0, 1.0, 2.0, 1.0, 0.0, -1.0, 0.5, 1.5 },
    { 1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 2.0, 0.0 },
    { 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0 },
  };
}

/// @brief Max abs deviation of a GPU N*N (row-major) matrix from the CPU oracle.
///        CPU oracle = full (band=-1) L1 DTW via the LIVE public `dtwc::dtwFull_L`.
inline double gpu_matrix_max_abs_error(const std::vector<std::vector<double>> &series,
                                       const std::vector<double> &gpu_matrix)
{
  const std::size_t N = series.size();
  double max_abs = 0.0;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j) {
      const double oracle =
        dtwc::dtwFull_L<double>(series[i], series[j], -1.0, dtwc::core::MetricType::L1);
      const double got = gpu_matrix[i * N + j];
      max_abs = std::max(max_abs, std::abs(got - oracle));
    }
  return max_abs;
}

} // namespace detail

/**
 * @brief Probe OpenMP parallelisation by ENGAGING it, not by reading a flag.
 * @return A ParallelReport. Never throws.
 *
 * `pass` is true iff parallelism is available AND genuinely engaged: on a host
 * that offers ≥2 threads we require ≥2 DISTINCT thread ids to have executed; on a
 * genuine single-thread host (`max_threads == 1`) engaging the one available
 * thread is the honest ceiling and also passes.
 */
inline ParallelReport parallelisation()
{
  ParallelReport r{};

#if defined(DTWC_SEQUENTIAL_BUILD)
  r.reason = "DTWC++ was compiled WITHOUT OpenMP (configured with "
             "-DDTWC_ALLOW_SEQUENTIAL=ON): this is a sequential build, so "
             "parallelisation is unavailable. Rebuild without that flag (with an "
             "OpenMP-capable toolchain) for multi-threaded execution.";
  return r;
#elif !defined(_OPENMP)
  r.reason = "This build was compiled without OpenMP (_OPENMP is undefined), so "
             "parallelisation is unavailable. Rebuild with an OpenMP-capable "
             "toolchain (and without -DDTWC_ALLOW_SEQUENTIAL=ON).";
  return r;
#else
  r.available = true;
  const int mt = omp_get_max_threads();
  r.max_threads = mt > 0 ? mt : 1;

  // Proof-of-engagement: run a REAL parallel region and record, per slot, which
  // OpenMP thread ids actually executed. Each thread writes ONLY its own slot
  // (index == its unique omp_get_thread_num()), so there is no data race.
  const std::size_t cap = static_cast<std::size_t>(r.max_threads);
  std::vector<unsigned char> seen(cap, 0u);
  unsigned char *const seen_ptr = seen.data();

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    // A little real work so the runtime actually spins up the team and the
    // region cannot be optimised away.
    volatile double spin = 0.0;
    for (int i = 0; i < 1000; ++i)
      spin += static_cast<double>((tid + 1) * i);
    (void)spin;
    if (tid >= 0 && static_cast<std::size_t>(tid) < cap)
      seen_ptr[static_cast<std::size_t>(tid)] = 1u;
  }

  int engaged = 0;
  for (const unsigned char v : seen)
    engaged += (v != 0u) ? 1 : 0;
  r.threads_engaged = engaged > 0 ? engaged : 1;

  const int needed = (r.max_threads >= 2) ? 2 : 1;
  r.pass = r.threads_engaged >= needed;
  return r;
#endif
}

/**
 * @brief Probe a GPU backend by EXECUTING a tiny kernel and validating it.
 * @return A GpuReport. Never throws, never silently degrades.
 *
 * Compiles and behaves correctly in BOTH GPU-ON and GPU-OFF builds:
 *   - GPU backend + device present -> run a small DTW distance matrix on the GPU
 *     and require it to match the CPU oracle within tolerance (`validated`/`pass`).
 *   - GPU backend compiled but NO device -> `available=false`, `reason` says so.
 *   - No GPU backend compiled -> `available=false`, `reason` names the OFF flags.
 */
inline GpuReport gpu()
{
  GpuReport r{};

#if defined(DTWC_HAS_CUDA)
  r.backend = "cuda";
  if (!dtwc::cuda::cuda_available()) {
    r.reason = "DTWC++ was built with CUDA (DTWC_ENABLE_CUDA=ON) but no CUDA-capable "
               "GPU was detected (cuda_available()==false). Check nvidia-smi and the "
               "CUDA driver installation.";
    return r;
  }
  r.available = true;
  r.device_name = dtwc::cuda::cuda_device_info(0);

  const auto series = detail::gpu_probe_series();
  const std::size_t N = series.size();

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = -1;                                       // full DTW
  opts.use_squared_l2 = false;                          // L1 (matches CPU oracle)
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;     // FP64 => bit-close to CPU (≤1e-12)
  opts.device_id = 0;
  opts.verbose = false;

  dtwc::cuda::CUDADistMatResult res;
  try {
    res = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  } catch (const std::exception &e) {
    r.reason = std::string("CUDA kernel execution failed: ") + e.what();
    return r;
  }
  if (res.n != N || res.matrix.size() != N * N) {
    r.reason = "CUDA kernel returned an unexpected matrix shape; validation could not run.";
    return r;
  }
  const double max_abs = detail::gpu_matrix_max_abs_error(series, res.matrix);
  r.validated = (max_abs <= detail::kGpuOracleTol);
  r.pass = r.validated;
  if (!r.validated)
    r.reason = "CUDA GPU result disagreed with the CPU oracle beyond tolerance (max |Δ| = "
             + std::to_string(max_abs) + " > 1e-12).";
  return r;

#elif defined(DTWC_HAS_METAL)
  r.backend = "metal";
  if (!dtwc::metal::metal_available()) {
    r.reason = "DTWC++ was built with Metal (DTWC_ENABLE_METAL=ON) but no Metal-capable "
               "GPU was detected (metal_available()==false).";
    return r;
  }
  r.available = true;
  r.device_name = dtwc::metal::metal_device_info();

  const auto series = detail::gpu_probe_series();
  const std::size_t N = series.size();

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = -1;               // full DTW
  opts.use_squared_l2 = false;  // L1 (matches CPU oracle)
  opts.verbose = false;

  dtwc::metal::MetalDistMatResult res;
  try {
    res = dtwc::metal::compute_distance_matrix_metal(series, opts);
  } catch (const std::exception &e) {
    r.reason = std::string("Metal kernel execution failed: ") + e.what();
    return r;
  }
  if (res.n != N || res.matrix.size() != N * N) {
    r.reason = "Metal kernel returned an unexpected matrix shape; validation could not run.";
    return r;
  }
  // Metal accumulates in FP32 (no FP64), so the oracle bound is FP32-appropriate.
  const double max_abs = detail::gpu_matrix_max_abs_error(series, res.matrix);
  r.validated = (max_abs <= detail::kMetalOracleTol);
  r.pass = r.validated;
  if (!r.validated)
    r.reason = "Metal GPU result disagreed with the CPU oracle beyond tolerance (max |Δ| = "
             + std::to_string(max_abs) + ").";
  return r;

#else
  r.reason = "DTWC++ was built without a GPU backend (DTWC_ENABLE_CUDA=OFF and "
             "DTWC_ENABLE_METAL=OFF); rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, "
             "on macOS, -DDTWC_ENABLE_METAL=ON.";
  return r;
#endif
}

} // namespace dtwc::test
