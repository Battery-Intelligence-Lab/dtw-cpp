// C4 probe (scratch, not part of the library): the route the library's own CUDA fill takes (`kernel_used`),
// FP32, 3 series of each L. Usage: route <L>...
#include <dtwc.hpp>
#include <cuda/cuda_dtw.cuh>

#include "support/deterministic_series.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
  for (int a = 1; a < argc; ++a) {
    const std::size_t L = std::strtoull(argv[a], nullptr, 10);
    const auto series = dtwc::test_support::benchmark_series_set(3, L, 200u);
    dtwc::cuda::CUDADistMatOptions opts;
    opts.precision = dtwc::cuda::CUDAPrecision::FP32;
    dtwc::core::DistanceMatrix packed;
    const auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts, packed);
    std::printf("FP32 L %zu: kernel_used %s\n", L, result.kernel_used.c_str());
  }
  return 0;
}
