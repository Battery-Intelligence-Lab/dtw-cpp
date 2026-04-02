/**
 * @file cuda_memory.cuh
 * @brief RAII wrappers for CUDA device memory.
 */

#pragma once

#ifdef DTWC_HAS_CUDA

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#define CUDA_CHECK_ALLOC(call)                                               \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +   \
                               ":" + std::to_string(__LINE__) + ": " +      \
                               cudaGetErrorString(err));                     \
    }                                                                        \
  } while (0)

namespace dtwc::cuda {

/// Deleter that calls cudaFree for use with std::unique_ptr.
struct CudaDeleter {
  void operator()(void *p) const {
    if (p) cudaFree(p);
  }
};

/// RAII smart pointer for CUDA device memory.
template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

/// Allocate device memory and return an owning CudaPtr.
template <typename T>
CudaPtr<T> cuda_alloc(size_t count)
{
  T *ptr = nullptr;
  CUDA_CHECK_ALLOC(cudaMalloc(&ptr, count * sizeof(T)));
  return CudaPtr<T>(ptr);
}

// -------------------------------------------------------------------------
// Pinned (page-locked) host memory
// -------------------------------------------------------------------------

/// Deleter that calls cudaFreeHost for use with std::unique_ptr.
struct PinnedDeleter {
  void operator()(void *p) const {
    if (p) cudaFreeHost(p);
  }
};

/// RAII smart pointer for CUDA pinned host memory.
template <typename T>
using PinnedPtr = std::unique_ptr<T, PinnedDeleter>;

/// Allocate pinned host memory and return an owning PinnedPtr.
template <typename T>
PinnedPtr<T> pinned_alloc(size_t count)
{
  T *ptr = nullptr;
  CUDA_CHECK_ALLOC(cudaMallocHost(&ptr, count * sizeof(T)));
  return PinnedPtr<T>(ptr);
}

/// Try to allocate pinned host memory; returns nullptr on failure (no throw).
template <typename T>
PinnedPtr<T> pinned_alloc_nothrow(size_t count)
{
  T *ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, count * sizeof(T));
  if (err != cudaSuccess) {
    // Clear the error so it doesn't poison later CUDA calls
    cudaGetLastError();
    return PinnedPtr<T>(nullptr);
  }
  return PinnedPtr<T>(ptr);
}

// -------------------------------------------------------------------------
// CUDA stream (RAII)
// -------------------------------------------------------------------------

/// Deleter that calls cudaStreamDestroy for use with std::unique_ptr.
struct StreamDeleter {
  void operator()(CUstream_st *s) const {
    if (s) cudaStreamDestroy(s);
  }
};

/// RAII wrapper for a CUDA stream.
using CudaStream = std::unique_ptr<CUstream_st, StreamDeleter>;

/// Create a CUDA stream and return an owning CudaStream.
inline CudaStream make_cuda_stream()
{
  cudaStream_t s = nullptr;
  CUDA_CHECK_ALLOC(cudaStreamCreate(&s));
  return CudaStream(s);
}

// -------------------------------------------------------------------------
// CUDA event (RAII)
// -------------------------------------------------------------------------

/// Deleter that calls cudaEventDestroy for use with std::unique_ptr.
struct EventDeleter {
  void operator()(CUevent_st *e) const {
    if (e) cudaEventDestroy(e);
  }
};

/// RAII wrapper for a CUDA event.
using CudaEvent = std::unique_ptr<CUevent_st, EventDeleter>;

/// Create a CUDA event and return an owning CudaEvent.
inline CudaEvent make_cuda_event()
{
  cudaEvent_t e = nullptr;
  CUDA_CHECK_ALLOC(cudaEventCreate(&e));
  return CudaEvent(e);
}

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
