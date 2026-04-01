/**
 * @file gpu_config.cuh
 * @brief Runtime GPU capability detection and configuration.
 * @date 02 Apr 2026
 */
#pragma once

#ifdef DTWC_HAS_CUDA

#include <cuda_runtime.h>
#include <string>
#include <mutex>

namespace dtwc::cuda {

/// FP64 throughput classification
enum class FP64Rate {
  Full,  ///< FP64:FP32 = 1:2 (HPC: P100, V100, A30, A100, H100)
  Slow   ///< FP64:FP32 = 1:32 or 1:64 (Consumer: RTX, GTX, L40S)
};

/// Cached GPU configuration queried once per device
struct GPUConfig {
  int device_id = 0;
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  size_t shared_mem_per_sm = 0;        ///< bytes
  size_t max_shared_per_block = 0;     ///< bytes (opt-in maximum)
  size_t total_global_mem = 0;         ///< bytes
  int max_threads_per_block = 1024;
  int max_threads_per_sm = 2048;
  int warp_size = 32;
  FP64Rate fp64_rate = FP64Rate::Slow;
  std::string device_name;

  /// Compute capability as a single integer (e.g., 86 for CC 8.6)
  int cc() const { return compute_major * 10 + compute_minor; }
};

/// Query GPU config for a device. Result is cached per device_id.
inline GPUConfig query_gpu_config(int device_id = 0) {
  // Thread-safe lazy initialization per device
  static std::mutex mtx;
  static GPUConfig configs[16];  // support up to 16 GPUs
  static bool initialized[16] = {};

  if (device_id < 0 || device_id >= 16) device_id = 0;

  std::lock_guard<std::mutex> lock(mtx);
  if (initialized[device_id]) return configs[device_id];

  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) return configs[device_id];

  GPUConfig &cfg = configs[device_id];
  cfg.device_id = device_id;
  cfg.compute_major = prop.major;
  cfg.compute_minor = prop.minor;
  cfg.sm_count = prop.multiProcessorCount;
  cfg.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
  cfg.total_global_mem = prop.totalGlobalMem;
  cfg.max_threads_per_block = prop.maxThreadsPerBlock;
  cfg.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  cfg.warp_size = prop.warpSize;
  cfg.device_name = prop.name;

  // Query opt-in max shared memory per block (CC >= 7.0)
  int max_shared_optin = 0;
  cudaDeviceGetAttribute(&max_shared_optin,
      cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
  cfg.max_shared_per_block = (max_shared_optin > 0)
      ? static_cast<size_t>(max_shared_optin)
      : prop.sharedMemPerBlock;

  // Classify FP64 throughput
  // HPC GPUs: CC 6.0, 7.0, 8.0, 9.0 have full-rate FP64 (1:2)
  // Consumer: CC 6.1, 6.2, 7.5, 8.6, 8.9 have slow FP64 (1:32 or 1:64)
  if (prop.major >= 7) {
    cfg.fp64_rate = (prop.minor == 0) ? FP64Rate::Full : FP64Rate::Slow;
  } else if (prop.major == 6) {
    cfg.fp64_rate = (prop.minor == 0) ? FP64Rate::Full : FP64Rate::Slow;
  } else {
    // CC 5.x and below — assume slow
    cfg.fp64_rate = FP64Rate::Slow;
  }

  initialized[device_id] = true;
  return cfg;
}

} // namespace dtwc::cuda

#endif // DTWC_HAS_CUDA
