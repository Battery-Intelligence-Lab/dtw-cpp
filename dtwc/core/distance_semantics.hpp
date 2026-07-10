/**
 * @file distance_semantics.hpp
 * @brief Shared validation for runtime distance tokens and semantic axes.
 */

#pragma once

#include "dtw_options.hpp"
#include "../error.hpp"

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

} // namespace dtwc::core
