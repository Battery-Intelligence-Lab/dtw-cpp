/**
 * @file dtw.cpp
 * @brief Implementation of the runtime-dispatched DTW entry point.
 *
 * @date 28 Mar 2026
 */

#include "dtw.hpp"

namespace dtwc::core {

double dtw_runtime(const double* x, std::size_t nx,
                   const double* y, std::size_t ny,
                   const DTWOptions& opts)
{
  const std::vector<double> vx(x, x + nx);
  const std::vector<double> vy(y, y + ny);

  switch (opts.constraint) {
  case ConstraintType::SakoeChibaBand:
    return dtwBanded<double>(vx, vy, opts.band_width, -1.0, opts.metric);
  case ConstraintType::None:
  default:
    return dtwFull_L<double>(vx, vy, -1.0, opts.metric);
  }
}

} // namespace dtwc::core
