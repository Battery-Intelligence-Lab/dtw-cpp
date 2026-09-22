// Codegen probe for the DTW hot paths (ledger X-04).
//
// Not a test and not part of any target. scripts/codegen_report.py compiles this
// file with the project's real flags -- taken from compile_commands.json, so the
// same -O level, floating-point model and architecture tuning the library ships
// with -- and reads clang's loop-vectorize remarks off it.
//
// It has to exist because the kernels are function templates in dtwc/warping.hpp.
// A template that nobody instantiates generates no code, so no library
// translation unit reports on them: dtwc/core/dtw.cpp contains zero loops. The
// explicit instantiations below are what force the loops into existence where a
// compiler can be asked about them.
//
// Add an instantiation here when a kernel joins the hot path, then re-record the
// expectation table.

// Included as "warping.hpp", not "dtwc/warping.hpp": the build puts dtwc/ on the
// include path and deliberately never the repo root, because on a
// case-insensitive filesystem the root VERSION file shadows <version>.
#include "warping.hpp"

#include <cstddef>
#include <span>

namespace {

// The distance functor the library uses in its own hot paths: a plain absolute
// difference, so nothing here is pessimised by an indirect call the real code
// would not make.
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
  return dtwc::detail::dtwFull_impl<double>(x, nx, y, ny, AbsDiff{});
}

float dtwc_probe_dtwFull_f32(const float *x, std::size_t nx, const float *y, std::size_t ny)
{
  return dtwc::detail::dtwFull_impl<float>(x, nx, y, ny, AbsDiff{});
}

double dtwc_probe_dtwFull_L_f64(
  const double *x, std::size_t nx, const double *y, std::size_t ny, double early_abandon)
{
  return dtwc::detail::dtwFull_L_impl<double>(x, nx, y, ny, early_abandon, AbsDiff{});
}

float dtwc_probe_dtwFull_L_f32(
  const float *x, std::size_t nx, const float *y, std::size_t ny, float early_abandon)
{
  return dtwc::detail::dtwFull_L_impl<float>(x, nx, y, ny, early_abandon, AbsDiff{});
}

} // extern "C"
