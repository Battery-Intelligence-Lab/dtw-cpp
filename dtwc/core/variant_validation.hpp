/**
 * @file variant_validation.hpp
 * @brief One source of truth for public DTW-variant parameter domains.
 */

#pragma once

#include "dtw_options.hpp"
#include "../error.hpp"

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

/**
 * Validate the whole value object, not only its active selector.
 *
 * DTWVariantParams is public and all fields participate in persistent-cache
 * identity.  Rejecting an invalid inactive field prevents a later selector
 * change from activating poisoned state and keeps every aggregate entry point
 * deterministic about validation order.
 */
inline void validate_variant_params(const DTWVariantParams &params)
{
  validate_wdtw_g(params.wdtw_g);
  validate_adtw_penalty(params.adtw_penalty);
  validate_sdtw_gamma(params.sdtw_gamma);
  validate_msm_c(params.msm_c);
  validate_twe_nu(params.twe_nu);
  validate_twe_lambda(params.twe_lambda);
}

} // namespace dtwc::core
