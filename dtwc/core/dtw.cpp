/**
 * @file dtw.cpp
 * @brief Implementation of the runtime-dispatched DTW entry point.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include "dtw.hpp"
#include "../warping_adtw.hpp"
#include "../warping_ddtw.hpp"
#include "../warping_wdtw.hpp"

namespace dtwc::core {

double dtw_runtime(const double* x, std::size_t nx,
                   const double* y, std::size_t ny,
                   const DTWOptions& opts)
{
  const int band = opts.band;
  const bool banded = (opts.constraint == ConstraintType::SakoeChibaBand) && (band >= 0);

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

    case DTWVariant::SoftDTW:
      // SoftDTW needs log-sum-exp recurrence and a different scratch layout.
      // Phase 1 does not migrate SoftDTW into the unified kernel; users must
      // call dtwc::soft_dtw() directly. Fall through to Standard for now so
      // the call still returns a finite distance rather than throwing.
      [[fallthrough]];

    case DTWVariant::Standard:
    default:
      return banded
        ? dtwBanded<double>(x, nx, y, ny, band, -1.0, opts.metric)
        : dtwFull_L<double>(x, nx, y, ny, -1.0, opts.metric);
  }
}

} // namespace dtwc::core
