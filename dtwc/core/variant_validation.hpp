/**
 * @file variant_validation.hpp
 * @brief The per-pair kernels' own parameter checks (core::validate checks a
 *        configuration once, where it is set or bound).
 */

#pragma once

#include "dtw_options.hpp"
#include "../base/error.hpp"

#include <cmath>

namespace dtwc::core {

/** WDTW's zero-steepness limit is a valid constant half-weight function. */
template <typename T>
inline void validate_wdtw_g(T value)
{
  if (!std::isfinite(value) || value < T(0))
    throw InvalidInput("WDTW g must be finite and non-negative.");
}

/** ADTW penalty zero is the exact Standard-DTW recurrence. */
template <typename T>
inline void validate_adtw_penalty(T value)
{
  if (!std::isfinite(value) || value < T(0))
    throw InvalidInput("ADTW penalty must be finite and non-negative.");
}

template <typename T>
inline void validate_sdtw_gamma(T value)
{
  if (!std::isfinite(value) || value <= T(0))
    throw InvalidInput("Soft-DTW gamma must be finite and positive.");
}

template <typename T>
inline void validate_msm_c(T value)
{
  if (!std::isfinite(value) || value <= T(0))
    throw InvalidInput("MSM c must be finite and positive.");
}

template <typename T>
inline void validate_twe_nu(T value)
{
  if (!std::isfinite(value) || value <= T(0))
    throw InvalidInput("TWE nu must be finite and positive.");
}

template <typename T>
inline void validate_twe_lambda(T value)
{
  if (!std::isfinite(value) || value <= T(0))
    throw InvalidInput("TWE lambda must be finite and positive.");
}

} // namespace dtwc::core
