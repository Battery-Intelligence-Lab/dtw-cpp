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

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
