/**
 * @file codegen_probe.cpp
 * @author Volkan Kumtepeli
 */
// Codegen probe for the DTW hot paths (ledger X-04).
//
// Not part of any target. scripts/codegen_report.py compiles this file with the
// project's real flags -- taken from compile_commands.json, so the same -O level,
// floating-point model and architecture tuning the library ships with -- and reads
// clang's loop-vectorize remarks off it, or (--no-calls, test_codegen_no_calls)
// fails when a call sits inside any loop of these kernels, however deep.
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
#include "soft_dtw.hpp"
#include "warping.hpp"
#include "warping_adtw.hpp"
#include "warping_missing.hpp"
#include "warping_missing_arow.hpp"
#include "warping_wdtw.hpp"
#include "core/distance_matrix.hpp"
#include "core/msm.hpp"
#include "core/twe.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace {

// The library's absolute difference (dtw_cost.hpp: std::abs(a - b), one fabd or
// andpd), called directly, so nothing here is pessimised by an indirect call or
// a compare-and-select the real code would not make.
struct AbsDiff
{
  template <typename data_t>
  data_t operator()(data_t a, data_t b) const noexcept
  {
    return std::abs(a - b);
  }
};

// The multivariate kernels dtw_dispatch.cpp binds for ndim > 1, through the wrappers it calls: the dependent DTW under
// the runtime metric (L1, L2, squared L2), ZeroCost's NaN-aware costs, AROW's, WDTW's weighted L1 and ADTW; for the
// independent channels (dtw_independent_mv, whose channel loop calls one kernel a channel) the univariate kernel each
// channel runs, under the metric. The band is the caller's, so each cost has its banded kernel and its linear one (band
// -1, or a pair the band covers whole), each with and without early abandon (dtw_kernel_banded picks at run time).
template <typename T>
T probe_mv(const T *x, std::size_t nx, const T *y, std::size_t ny, std::size_t ndim, int band, int metric,
           const T *weights, std::size_t n_weights)
{
  namespace core = dtwc::core;
  const auto m = static_cast<core::MetricType>(metric);
  return dtwc::dtwBanded_mv<T>(x, nx, y, ny, ndim, band, T(-1), m)
         + dtwc::dtwMissing_banded_mv<T>(x, nx, y, ny, ndim, band, T(-1), m)
         + core::run_dtw<core::SpanMVAROWSquaredL2Cost>(x, nx, y, ny, band, core::AROWCell{}, T(-1), ndim)
         + core::run_dtw<core::SpanMVAROWL1Cost>(x, nx, y, ny, band, core::AROWCell{}, T(-1), ndim)
         + dtwc::wdtwBanded_mv<T>(x, nx, y, ny, ndim, { weights, n_weights }, band)
         + dtwc::adtwBanded_mv<T>(x, nx, y, ny, ndim, band, T(0.5))
         + dtwc::dtwBanded<T>(x, nx, y, ny, band, T(-1), m);
}

// The univariate kernels dtw_dispatch.cpp binds beside the lanes and dtwBanded: AROW's for missing data (its cost by the
// runtime metric; banded, and linear for band -1 or a pair the band covers whole) and Soft-DTW's (unbanded).
template <typename T>
T probe_univariate(const T *x, std::size_t nx, const T *y, std::size_t ny, int band, int metric, T gamma)
{
  return dtwc::dtwAROW_banded<T>(x, nx, y, ny, band, static_cast<dtwc::core::MetricType>(metric))
         + dtwc::soft_dtw<T>({ x, nx }, { y, ny }, gamma);
}

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

// The remaining cells and kernels: ADTW, AROW, banded, MSM, TWE. The dtwFull_* wrappers above
// pass band -1, so no banded loop reaches the listing from them: each precision needs its own.
double dtwc_probe_kernels_f64(const double *x, std::size_t nx, const double *y, std::size_t ny)
{
  namespace core = dtwc::core;
  const auto cost = [x, y](std::size_t i, std::size_t j) noexcept { return AbsDiff{}(x[i], y[j]); };
  return core::dtw_kernel_linear<double>(nx, ny, cost, core::ADTWCell<double>{0.5})
         + core::dtw_kernel_banded<double>(nx, ny, 8, cost, core::AROWCell{})
         + dtwc::dtwBanded<double>(x, nx, y, ny, 8, -1.0)
         + core::msm_distance<double>(x, nx, y, ny) + core::twe_distance<double>(x, nx, y, ny);
}

float dtwc_probe_kernels_f32(const float *x, std::size_t nx, const float *y, std::size_t ny)
{
  namespace core = dtwc::core;
  const auto cost = [x, y](std::size_t i, std::size_t j) noexcept { return AbsDiff{}(x[i], y[j]); };
  return core::dtw_kernel_linear<float>(nx, ny, cost, core::ADTWCell<float>{0.5f})
         + core::dtw_kernel_banded<float>(nx, ny, 8, cost, core::AROWCell{})
         + dtwc::dtwBanded<float>(x, nx, y, ny, 8, -1.0f)
         + core::msm_distance<float>(x, nx, y, ny) + core::twe_distance<float>(x, nx, y, ny);
}

// The fill's lane kernel and cell: W pairs per call, f64 under L1 and f32 under squared L2.
double dtwc_probe_lanes_f64(const double *x, std::size_t nx, const std::span<const double> *ys, int band)
{
  double sum = 0;
  for (const double d : dtwc::core::dtw_kernel_lanes<double>({ x, nx }, ys, band, AbsDiff{},
                                                             dtwc::core::LanesCell{}))
    sum += d;
  return sum;
}

float dtwc_probe_lanes_f32(const float *x, std::size_t nx, const std::span<const float> *ys, int band)
{
  float sum = 0;
  const auto squared = [](float a, float b) noexcept {
    const float d = a - b;
    return d * d;
  };
  for (const float d : dtwc::core::dtw_kernel_lanes<float>({ x, nx }, ys, band, squared,
                                                           dtwc::core::LanesCell{}))
    sum += d;
  return sum;
}

// The multivariate kernels, the dispatcher's (probe_mv above), at both precisions.
double dtwc_probe_mv_f64(const double *x, std::size_t nx, const double *y, std::size_t ny, std::size_t ndim,
                         int band, int metric, const double *weights, std::size_t n_weights)
{
  return probe_mv<double>(x, nx, y, ny, ndim, band, metric, weights, n_weights);
}

float dtwc_probe_mv_f32(const float *x, std::size_t nx, const float *y, std::size_t ny, std::size_t ndim, int band,
                        int metric, const float *weights, std::size_t n_weights)
{
  return probe_mv<float>(x, nx, y, ny, ndim, band, metric, weights, n_weights);
}

// AROW's and Soft-DTW's univariate kernels (probe_univariate above), at both precisions.
double dtwc_probe_univariate_f64(const double *x, std::size_t nx, const double *y, std::size_t ny, int band, int metric,
                                 double gamma)
{
  return probe_univariate<double>(x, nx, y, ny, band, metric, gamma);
}

float dtwc_probe_univariate_f32(const float *x, std::size_t nx, const float *y, std::size_t ny, int band, int metric,
                                float gamma)
{
  return probe_univariate<float>(x, nx, y, ny, band, metric, gamma);
}

// The packed matrix read eight rows a point (core::for_each_point: FasterPAM's k = 1, silhouette), at both widths:
// a float32 matrix's values are widened as they are read.
double dtwc_probe_rows_f64(const double *packed, std::int64_t n, std::int64_t a, double *sums)
{
  dtwc::core::for_each_point(packed, n, a, [sums](std::int64_t, const double *v) {
    for (int g = 0; g < 8; ++g) sums[g] += v[g];
  });
  return sums[0];
}

double dtwc_probe_rows_f32(const float *packed, std::int64_t n, std::int64_t a, double *sums)
{
  dtwc::core::for_each_point(packed, n, a, [sums](std::int64_t, const double *v) {
    for (int g = 0; g < 8; ++g) sums[g] += v[g];
  });
  return sums[0];
}

// The series check's predicate (Problem's check, the Parquet stream's), each value of every series once.
bool dtwc_probe_all_usable_f64(const double *x, std::size_t n, bool nan_is_missing)
{
  return dtwc::detail::all_usable<double>({ x, n }, nan_is_missing);
}

bool dtwc_probe_all_usable_f32(const float *x, std::size_t n, bool nan_is_missing)
{
  return dtwc::detail::all_usable<float>({ x, n }, nan_is_missing);
}

} // extern "C"
