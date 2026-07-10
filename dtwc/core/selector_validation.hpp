/**
 * @file selector_validation.hpp
 * @brief Cycle-safe membership guards for core distance selectors.
 */

#pragma once

#include "dtw_options.hpp"
#include "../error.hpp"

namespace dtwc::core {

inline void validate_dtw_variant(DTWVariant value)
{
  switch (value) {
  case DTWVariant::Standard:
  case DTWVariant::DDTW:
  case DTWVariant::WDTW:
  case DTWVariant::ADTW:
  case DTWVariant::SoftDTW:
  case DTWVariant::MSM:
  case DTWVariant::TWE:
    return;
  }
  throw InvalidInput("Invalid DTWVariant value.");
}

inline void validate_missing_strategy(MissingStrategy value)
{
  switch (value) {
  case MissingStrategy::Error:
  case MissingStrategy::ZeroCost:
  case MissingStrategy::AROW:
  case MissingStrategy::Interpolate:
    return;
  }
  throw InvalidInput("Invalid MissingStrategy value.");
}

inline void validate_metric_type(MetricType value)
{
  switch (value) {
  case MetricType::L1:
  case MetricType::L2:
  case MetricType::SquaredL2:
    return;
  }
  throw InvalidInput("Invalid MetricType value.");
}

inline void validate_mv_mode(MVMode value)
{
  switch (value) {
  case MVMode::Dependent:
  case MVMode::Independent:
    return;
  }
  throw InvalidInput("Invalid MVMode value.");
}

inline void validate_constraint_type(ConstraintType value)
{
  switch (value) {
  case ConstraintType::None:
  case ConstraintType::SakoeChibaBand:
    return;
  }
  throw InvalidInput("Invalid ConstraintType value.");
}

} // namespace dtwc::core
