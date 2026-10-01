// Codegen probe for the DTW hot paths (ledger X-04).
//
// Not part of any target. scripts/codegen_report.py compiles this file with the
// project's real flags -- taken from compile_commands.json, so the same -O level,
// floating-point model and architecture tuning the library ships with -- and reads
// clang's loop-vectorize remarks off it, or (--no-calls, test_codegen_no_calls)
// fails when an innermost loop of these kernels makes a call.
//
// It has to exist because the kernels are function templates in dtwc/warping.hpp
// and dtwc/core/. A template that nobody instantiates generates no code, so no
// library translation unit reports on them. The exported wrappers below are what
// force the loops into existence where a compiler can be asked about them.
//
// Add an instantiation here when a kernel joins the hot path.

// Included as "warping.hpp", not "dtwc/warping.hpp": the build puts dtwc/ on the
// include path and deliberately never the repo root, because on a
// case-insensitive filesystem the root VERSION file shadows <version>.
#include "warping.hpp"
#include "core/msm.hpp"
#include "core/twe.hpp"

#include <cstddef>
#include <span>

namespace {

// A plain absolute difference for the kernels called directly, so nothing here
// is pessimised by an indirect call the real code would not make.
struct AbsDiff
{
  template <typename data_t>
  data_t operator()(data_t a, data_t b) const noexcept
  {
    return a < b ? b - a : a - b;
  }
};

} // namespace

// Exported wrappers, not explicit instantiations.
//
// An explicit instantiation whose template argument has internal linkage (AbsDiff
// lives in an anonymous namespace) is itself internal, and nothing in this TU
// calls it — so clang discards it as dead before the vectoriser runs, and the
// object comes out with no kernel symbols and no remarks at all. Wrappers with
// external linkage cannot be eliminated: the compiler has to emit the loops,
// which is the entire point of the probe.
//
// Both precisions are covered: f32 and f64 vectorise to different widths, and
// X-13/X-14 are about collapsing those twins, so the report should show both.
extern "C" {

double dtwc_probe_dtwFull_f64(const double *x, std::size_t nx, const double *y, std::size_t ny)
{
  return dtwc::dtwFull<double>(x, nx, y, ny);
}

float dtwc_probe_dtwFull_f32(const float *x, std::size_t nx, const float *y, std::size_t ny)
{
  return dtwc::dtwFull<float>(x, nx, y, ny);
}

double dtwc_probe_dtwFull_L_f64(
  const double *x, std::size_t nx, const double *y, std::size_t ny, double early_abandon)
{
  return dtwc::dtwFull_L<double>(x, nx, y, ny, early_abandon);
}

float dtwc_probe_dtwFull_L_f32(
  const float *x, std::size_t nx, const float *y, std::size_t ny, float early_abandon)
{
  return dtwc::dtwFull_L<float>(x, nx, y, ny, early_abandon);
}

// The remaining cells and kernels: ADTW, AROW, banded, MSM, TWE.
double dtwc_probe_kernels_f64(const double *x, std::size_t nx, const double *y, std::size_t ny)
{
  namespace core = dtwc::core;
  const auto cost = [x, y](std::size_t i, std::size_t j) noexcept { return AbsDiff{}(x[i], y[j]); };
  return core::dtw_kernel_linear<double>(nx, ny, cost, core::ADTWCell<double>{0.5})
         + core::dtw_kernel_banded<double>(nx, ny, 8, cost, core::AROWCell{})
         + dtwc::dtwBanded<double>(x, nx, y, ny, 8, -1.0)
         + core::msm_distance<double>(x, nx, y, ny) + core::twe_distance<double>(x, nx, y, ny);
}

// The fill's lane kernel: W pairs per call, f64 under L1 and f32 under squared L2.
double dtwc_probe_lanes_f64(const double *x, const double *const *ys, std::size_t n, int band)
{
  double sum = 0;
  for (const double d : dtwc::core::dtw_kernel_lanes<double>(x, ys, n, band, AbsDiff{},
                                                             dtwc::core::StandardCell{}))
    sum += d;
  return sum;
}

float dtwc_probe_lanes_f32(const float *x, const float *const *ys, std::size_t n, int band)
{
  float sum = 0;
  const auto squared = [](float a, float b) noexcept {
    const float d = a - b;
    return d * d;
  };
  for (const float d : dtwc::core::dtw_kernel_lanes<float>(x, ys, n, band, squared,
                                                           dtwc::core::StandardCell{}))
    sum += d;
  return sum;
}

} // extern "C"
