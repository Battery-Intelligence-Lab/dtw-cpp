/**
 * @file variant_validation.hpp
 * @brief One source of truth for public DTW-variant parameter domains.
 */

#pragma once

#include "selector_validation.hpp"

#include <cmath>
#include <limits>

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
  validate_dtw_variant(params.variant);
  validate_mv_mode(params.mv_mode);
  validate_wdtw_g(params.wdtw_g);
  validate_adtw_penalty(params.adtw_penalty);
  validate_sdtw_gamma(params.sdtw_gamma);
  validate_msm_c(params.msm_c);
  validate_twe_nu(params.twe_nu);
  validate_twe_lambda(params.twe_lambda);
}

/**
 * Whether a finite double parameter retains the domain-defining information
 * needed by a float32 recurrence.
 *
 * General rounding is intentional, but a non-zero value becoming zero changes
 * a positive parameter into an invalid limit, and a finite value becoming an
 * infinity changes the recurrence entirely. Exact zero remains representable
 * for the WDTW and ADTW parameters whose double-domain contracts allow it.
 */
inline bool variant_parameter_representable_f32(double value) noexcept
{
  constexpr double f32_max = static_cast<double>(
    std::numeric_limits<float>::max());
  if (!std::isfinite(value) || value > f32_max || value < -f32_max)
    return false;
  const float narrowed = static_cast<float>(value);
  return value == 0.0 || narrowed != 0.0f;
}

/** Return the active field's float32 narrowing diagnostic, if any. */
inline const char *active_variant_params_f32_error(
  const DTWVariantParams &params) noexcept
{
  switch (params.variant) {
  case DTWVariant::WDTW:
    if (!variant_parameter_representable_f32(params.wdtw_g))
      return "WDTW g cannot be represented in float32 without becoming zero or non-finite.";
    break;
  case DTWVariant::ADTW:
    if (!variant_parameter_representable_f32(params.adtw_penalty))
      return "ADTW penalty cannot be represented in float32 without becoming zero or non-finite.";
    break;
  case DTWVariant::SoftDTW:
    if (!variant_parameter_representable_f32(params.sdtw_gamma))
      return "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite.";
    break;
  case DTWVariant::MSM:
    if (!variant_parameter_representable_f32(params.msm_c))
      return "MSM c cannot be represented in float32 without becoming zero or non-finite.";
    break;
  case DTWVariant::TWE:
    if (!variant_parameter_representable_f32(params.twe_nu))
      return "TWE nu cannot be represented in float32 without becoming zero or non-finite.";
    if (!variant_parameter_representable_f32(params.twe_lambda))
      return "TWE lambda cannot be represented in float32 without becoming zero or non-finite.";
    break;
  case DTWVariant::Standard:
  case DTWVariant::DDTW:
    break;
  default:
    return "Invalid DTWVariant value.";
  }
  return nullptr;
}

inline bool active_variant_params_representable_f32(
  const DTWVariantParams &params) noexcept
{
  return active_variant_params_f32_error(params) == nullptr;
}

inline void validate_active_variant_params_f32(const DTWVariantParams &params)
{
  if (const char *error = active_variant_params_f32_error(params))
    throw InvalidInput(error);
}

} // namespace dtwc::core
