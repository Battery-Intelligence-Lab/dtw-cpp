/**
 * @file dtw.cpp
 * @brief Implementation of the runtime-dispatched DTW entry point.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include "dtw.hpp"
#include "../distance.hpp"
#include "../warping_adtw.hpp"
#include "../warping_ddtw.hpp"
#include "../warping_wdtw.hpp"
#include "../soft_dtw.hpp"
#include "msm.hpp"
#include "twe.hpp"
#include "variant_validation.hpp"
#include "selector_validation.hpp"
#include "distance_semantics.hpp"

#include <span>

namespace dtwc::core {

double dtw_runtime(const double* x, std::size_t nx,
                   const double* y, std::size_t ny,
                   const DTWOptions& opts)
{
  validate_variant_params(opts.variant_params);
  validate_variant_missing_semantics(
    opts.variant_params, opts.missing_strategy);
  validate_metric_type(opts.metric);
  validate_constraint_type(opts.constraint);
  const int band = opts.band;
  const bool banded = (opts.constraint == ConstraintType::SakoeChibaBand) && (band >= 0);

  // Missing-data recurrences live in the shared public facade. Delegating the
  // accepted Standard cross-product here prevents runtime dispatch from
  // ignoring the requested policy; the facade also rejects non-Standard mixes.
  if (opts.missing_strategy != MissingStrategy::Error) {
    return dtwc::distance::dtw<double>(
      std::span<const double>{x, nx}, std::span<const double>{y, ny},
      opts.variant_params, banded ? band : -1, opts.metric,
      opts.missing_strategy);
  }

  // Dispatch on variant. Historically this function always used Standard DTW,
  // silently dropping `opts.variant_params.variant` — that bug is fixed here.
  switch (opts.variant_params.variant) {

    case DTWVariant::ADTW: {
      const double penalty = opts.variant_params.adtw_penalty;
      return banded
        ? dtwc::adtwBanded<double>(x, nx, y, ny, band, penalty, -1.0)
        : dtwc::adtwFull_L<double>(x, nx, y, ny, penalty, -1.0);
    }

    case DTWVariant::WDTW: {
      const double g = opts.variant_params.wdtw_g;
      return banded
        ? dtwc::wdtwBanded<double>(x, nx, y, ny, band, g)
        : dtwc::wdtwFull<double>(x, nx, y, ny, g);
    }

    case DTWVariant::DDTW: {
      // DDTW preprocesses via derivative_transform then runs Standard DTW.
      // Use the span-based entry which handles the preprocessing itself.
      std::span<const double> xs{x, nx};
      std::span<const double> ys{y, ny};
      return banded
        ? dtwc::ddtwBanded<double>(xs, ys, band, opts.metric)
        : dtwc::ddtwFull_L<double>(xs, ys, opts.metric);
    }

    case DTWVariant::SoftDTW: {
      // BUGFIX (Task 0.6): this case previously `[[fallthrough]]`-ed to Standard,
      // so a SoftDTW request SILENTLY returned a Standard-L1 distance (a wrong
      // number, not a differentiable Soft-DTW value) and the gamma>0 precondition
      // was skipped (NaN poison for gamma<=0). Route to the dedicated soft_dtw()
      // kernel, which computes the real Soft-DTW and throws std::invalid_argument
      // for gamma<=0. soft_dtw() uses an L1 pointwise cost and is unbanded, so
      // opts.band / opts.metric do not apply on this path (SoftDTW support is
      // L1/full-matrix only) — consistent with ADTW/WDTW above ignoring metric.
      const double gamma = opts.variant_params.sdtw_gamma;
      return dtwc::soft_dtw<double>(std::span<const double>{x, nx},
                                    std::span<const double>{y, ny}, gamma);
    }

    case DTWVariant::MSM:
      // MSM/TWE are univariate + unbanded (v1); opts.band/metric do not apply
      // (same as ADTW/WDTW/SoftDTW above ignoring some axes).
      return dtwc::core::msm_distance<double>(x, nx, y, ny,
                                              opts.variant_params.msm_c);

    case DTWVariant::TWE:
      return dtwc::core::twe_distance<double>(x, nx, y, ny,
                                              opts.variant_params.twe_nu,
                                              opts.variant_params.twe_lambda);

    case DTWVariant::Standard:
      return banded
        ? dtwBanded<double>(x, nx, y, ny, band, -1.0, opts.metric)
        : dtwFull_L<double>(x, nx, y, ny, -1.0, opts.metric);
    default:
      validate_dtw_variant(opts.variant_params.variant);
      throw std::logic_error("dtw_runtime: unreachable DTWVariant");
  }
}

} // namespace dtwc::core
