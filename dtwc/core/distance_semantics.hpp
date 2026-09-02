/**
 * @file distance_semantics.hpp
 * @brief Shared validation for runtime distance tokens and semantic axes.
 */

#pragma once

#include "dtw_options.hpp"
#include "selector_validation.hpp"
#include "variant_validation.hpp"
#include "../error.hpp"
#include "../missing_utils.hpp"

#include <cstddef>
#include <span>
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
 * Enforce the MissingStrategy::Error contract ("throw if NaN") on a pairwise
 * call. Without it the recurrence returns NaN — also DenseDistanceMatrix's
 * "uncomputed" sentinel — so a computed result would be indistinguishable from
 * an unfilled entry and all_computed() could never become true. The per-pair
 * counterpart of Problem::fill_distance_matrix's scan: O(n + m) against an
 * O(n·m) recurrence.
 */
template <typename T>
inline void reject_missing_under_error_strategy(
  std::span<const T> x, std::span<const T> y, std::string_view where)
{
  if (has_missing(x) || has_missing(y))
    throw InvalidInput(
      std::string(where)
      + ": NaN detected in input under MissingStrategy::Error. Set the missing "
        "strategy to ZeroCost, AROW, or Interpolate to handle missing data.");
}

/// Reject a multivariate call on a feature that is univariate in this release.
/// Shared with the bind-time guards in dtw_dispatch.cpp so the Problem-level and
/// per-call boundaries cannot state the same rule differently.
inline void require_univariate(std::size_t ndim, const char *feature)
{
  if (ndim > 1)
    throw InvalidInput(std::string(feature)
                       + " is univariate in this release (ndim must be 1)");
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

  if (params.variant == DTWVariant::MSM) require_univariate(ndim, "MSM distance");
  if (params.variant == DTWVariant::TWE) require_univariate(ndim, "TWE distance");
  if (params.variant == DTWVariant::SoftDTW) require_univariate(ndim, "Soft-DTW");

  // Interpolation is defined on a single channel stream: interpolate_linear()
  // would fill a gap from the neighbouring *channel's* values on an
  // interleaved multivariate buffer, and `band` would count flat elements
  // rather than timesteps. Reject rather than return a meaningless number.
  if (missing_strategy == MissingStrategy::Interpolate)
    require_univariate(ndim, "MissingStrategy::Interpolate");
}

} // namespace dtwc::core
