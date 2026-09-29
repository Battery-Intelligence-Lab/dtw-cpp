// W4a: does the library's public entry fail where dynamic shared memory is <= 48 KiB
// but dynamic + the kernel's 16 B static shared memory is > 48 KiB?
// FP32, 3 buffers (L > 2048): 12 L + 16 > 49152 for L in {4095, 4096}; L = 4094 is below.
#include "cuda/cuda_dtw.cuh"
#include "support/deterministic_series.hpp"

#include <cstdio>
#include <exception>

int main()
{
  for (int L : { 4094, 4095, 4096, 4097 }) {
    const auto series = dtwc::test_support::benchmark_series_set(4, std::size_t(L), 200u);
    dtwc::cuda::CUDADistMatOptions opts;
    opts.precision = dtwc::cuda::CUDAPrecision::FP32;
    try {
      const auto r = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
      std::printf("L=%d ok kernel=%s d01=%.6g\n", L, r.kernel_used.c_str(), r.matrix[1]);
    } catch (const std::exception &e) {
      std::printf("L=%d THROWS %s\n", L, e.what());
    }
  }
}
