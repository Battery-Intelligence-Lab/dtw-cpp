// The f64 L1 lanes kernel, shipped and fmin variant, compiled for x86-64-v3 (cross, -S only) to see what
// the min becomes there.
#include "core/dtw_kernel.hpp"
#include "v_fmin.hpp"
#include <cmath>
struct L1 { double operator()(double a, double b) const { return std::abs(a - b); } };
void ship(const double *x, const double *const *ys, std::size_t n, int band, double *out) {
  auto d = dtwc::core::dtw_kernel_lanes<double>(x, ys, n, band, L1{}, dtwc::core::StandardCell{});
  for (int w = 0; w < 8; ++w) out[w] = d[w];
}
void fmin_(const double *x, const double *const *ys, std::size_t n, int band, double *out) {
  auto d = v_fmin::dtw_kernel_lanes<double>(x, ys, n, band, L1{}, v_fmin::StandardCell{});
  for (int w = 0; w < 8; ++w) out[w] = d[w];
}
