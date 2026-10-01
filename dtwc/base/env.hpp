/**
 * @file env.hpp
 * @brief The device vocabulary and the single-thread warning.
 *
 * @details `Device` names where a run computes (`cpu` / `gpu`), `GpuPrecision`
 * what a GPU computes in. The process-wide default device is read and set
 * through `dtwc::device()` / `dtwc::device(name)` (api.hpp); a `Problem` never
 * reads it. `hpc` is not a C++ device: it submits a
 * whole run to a SLURM cluster, which Python's `dtwcpp.device("hpc")` and
 * `slurm_remote.sh` do, so the grammar here refuses it with a `DeviceError`
 * naming them.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 07 Jul 2026
 */

#pragma once

#include "error.hpp"
#include "names.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace dtwc {

/// @brief Where a run computes (api-contract-2.0.md §6.1).
enum class Device {
  CPU, ///< Local CPU execution (the default).
  GPU  ///< Local GPU execution (CUDA on NVIDIA, Metal on macOS).
};

/// GPU compute precision, on every GPU backend. `Auto` is FP32 on consumer CUDA
/// GPUs and FP64 on HPC ones; Metal computes in FP32 and rejects FP64.
enum class GpuPrecision { Auto, FP32, FP64 };

/// The spellings of `--gpu-precision`.
inline constexpr Name<GpuPrecision> gpu_precision_names[]{
  { "auto", GpuPrecision::Auto },
  { "fp32", GpuPrecision::FP32 }, { "float32", GpuPrecision::FP32 }, { "f32", GpuPrecision::FP32 },
  { "float", GpuPrecision::FP32 },
  { "fp64", GpuPrecision::FP64 }, { "float64", GpuPrecision::FP64 }, { "f64", GpuPrecision::FP64 },
  { "double", GpuPrecision::FP64 },
};

/// @brief Canonical lower-case name of a Device ("cpu" / "gpu").
std::string to_string(Device d);

namespace detail {

/// @brief Parse a device name: the one grammar behind dtwc::device(name), the
///        CLI's `--device` and the bindings' Problem device setters.
/// @details "cpu", "gpu", "gpu:N", "cuda", "cuda:N"; case-insensitive,
///          surrounding whitespace ignored, `cuda` ≡ `gpu`, N a non-negative
///          `int` GPU ordinal (0 when absent). Grammar only: the build is checked
///          by the caller.
/// @throws DeviceError listing the valid names; for "hpc" / "hpc:…", naming
///         Python's dtwcpp.device("hpc") and slurm_remote.sh.
std::pair<Device, int> parse_device(std::string_view name);

/// @brief The frozen §6.1 message for `gpu` on a build with no GPU backend.
std::string gpu_not_built_message();

} // namespace detail

/// @brief Print, once per process, why DTWC++ is running single-threaded and
///        how to fix it: OpenMP present but only 1 usable thread on a multicore
///        host (e.g. OMP_NUM_THREADS=1), or a build without OpenMP
///        (-DDTWC_ALLOW_SEQUENTIAL=ON). A single-core host is not warned.
/// @details Every compute path reaches it through dtwc::get_max_threads()
///          (parallelisation.hpp). Thread-safe; an atomic load after the first call.
void warn_if_single_threaded();

} // namespace dtwc
