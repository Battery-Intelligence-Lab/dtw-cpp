// The per-pair kernels instantiated as warping.hpp's dtwBanded does (run_dtw, StandardCell, L1),
// with a runtime early-abandon threshold: is the column minimum out of the loop without one?
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace dc = dtwc::core;
extern "C" __attribute__((noinline)) double pair_f64_l1(const double *x, std::size_t nx, const double *y,
                                                        std::size_t ny, int band, double ea)
{
  return dc::run_dtw<dc::SpanL1Cost>(x, nx, y, ny, band, dc::StandardCell{}, ea);
}
int main(int argc, char **argv)
{
  const std::size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 100;
  std::vector<double> x(n, 1.0), y(n + 1, 2.0);
  std::printf("%g\n", pair_f64_l1(x.data(), n, y.data(), n + 1, argc > 2 ? std::atoi(argv[2]) : -1,
                                  argc > 3 ? std::atof(argv[3]) : -1.0));
}
