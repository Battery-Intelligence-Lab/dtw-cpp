/**
 * @file dtw.cpp
 * @brief Implementation of the runtime-dispatched DTW entry point.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include "dtw.hpp"

namespace dtwc::core {

double dtw_runtime(const double* x, std::size_t nx,
                   const double* y, std::size_t ny,
                   const DTWOptions& opts)
{
  switch (opts.constraint) {
  case ConstraintType::SakoeChibaBand:
    return dtwBanded<double>(x, nx, y, ny, opts.band, -1.0, opts.metric);
  case ConstraintType::None:
  default:
    return dtwFull_L<double>(x, nx, y, ny, -1.0, opts.metric);
  }
}

} // namespace dtwc::core
