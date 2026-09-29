// W4a host oracle (measurement only; lives outside the repository).
// The library's own CPU DTW, dtwFull_L<T> (L1, full band), in the GPU's compute
// precision T, over every pair. Compiled with cl and the CUDA tests' flags.
#include "warping.hpp"

#include <cstddef>

template <typename T>
void w4a_oracle(const T *flat, int N, int L, T *out)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      const T d = dtwc::dtwFull_L<T>(flat + std::size_t(i) * L, std::size_t(L),
                                     flat + std::size_t(j) * L, std::size_t(L));
      out[std::size_t(i) * N + j] = d;
      out[std::size_t(j) * N + i] = d;
    }
}

template void w4a_oracle<float>(const float *, int, int, float *);
template void w4a_oracle<double>(const double *, int, int, double *);
