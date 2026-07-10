/**
 * @file distance_semantics.hpp
 * @brief Shared validation for runtime distance tokens and semantic axes.
 */

#pragma once

#include "dtw_options.hpp"
#include "selector_validation.hpp"
#include "variant_validation.hpp"
#include "../error.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace dtwc::core {

inline constexpr std::string_view metric_token_help =
  "Expected one of: l1, squared_euclidean, sqeuclidean.";

inline MetricType parse_metric_token(std::string_view token)
{
  if (token == "l1") return MetricType::L1;
  if (token == "squared_euclidean" || token == "sqeuclidean")
    return MetricType::SquaredL2;
  throw InvalidInput(
    "Unknown metric '" + std::string(token) + "'. "
    + std::string(metric_token_help));
}

inline void validate_variant_missing_semantics(
  DTWVariant variant, MissingStrategy missing_strategy)
{
  validate_dtw_variant(variant);
  validate_missing_strategy(missing_strategy);
  if (variant != DTWVariant::Standard
      && missing_strategy != MissingStrategy::Error) {
    throw InvalidInput(
      "Non-Standard DTW variants require MissingStrategy::Error.");
  }
}

inline void validate_variant_missing_semantics(
  const DTWVariantParams &params, MissingStrategy missing_strategy)
{
  validate_variant_missing_semantics(params.variant, missing_strategy);
}

/**
 * Validate every Problem-level distance axis whose legality is known before
 * dispatch, allocation, or cache mutation.
 *
 * Ordering is part of the public diagnostic contract and mirrors the legacy
 * resolver: whole-object parameter domains, variant/missing compatibility,
 * optional active-float32 narrowing, multivariate-mode compatibility, then
 * variant dimensionality. Selector membership is validated before any
 * cross-product comparison so invalid values cannot resemble a valid branch.
 */
inline void validate_problem_distance_semantics(
  const DTWVariantParams &params,
  MissingStrategy missing_strategy,
  std::size_t ndim,
  bool validate_float32)
{
  validate_variant_params(params);
  validate_variant_missing_semantics(params, missing_strategy);
  if (validate_float32)
    validate_active_variant_params_f32(params);

  if (params.mv_mode == MVMode::Independent && ndim > 1) {
    if (params.variant != DTWVariant::Standard) {
      throw InvalidInput(
        "Independent multivariate mode is implemented for the Standard "
        "DTW variant only in this release");
    }
    if (missing_strategy != MissingStrategy::Error) {
      throw InvalidInput(
        "Independent multivariate mode does not support a missing-data "
        "strategy in this release (set missing_strategy = Error)");
    }
  }

  if (ndim > 1 && params.variant == DTWVariant::MSM) {
    throw InvalidInput(
      "MSM distance is univariate in this release (ndim must be 1)");
  }
  if (ndim > 1 && params.variant == DTWVariant::TWE) {
    throw InvalidInput(
      "TWE distance is univariate in this release (ndim must be 1)");
  }
}

} // namespace dtwc::core
