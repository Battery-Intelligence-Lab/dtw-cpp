/**
 * @file settings.hpp
 * @brief This file contains settings and configurations for DTWC++ library.
 *
 * @details It includes settings for data types, the std::filesystem alias,
 * debugging options, and default algorithmic settings.
 *
 * @date 21 Jan 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include "../enums/enums.hpp"

#include <cstdint>
// Retained for downstream TUs that rely on transitive <string> visibility.
#include <string>
#include <filesystem>

namespace dtwc {
// Data type settings:

namespace settings {
  /// @brief Default scalar type for public templated APIs.
  /// @details This controls default template arguments such as
  ///          `template <typename T = settings::default_data_t>`.
  using default_data_t = double;
} // namespace settings

/// @brief Core storage / internal accumulation scalar type. The same type as `settings::default_data_t`, the default
///        template argument of the public distance helpers; the two names stay separate by role.
using data_t = double;

/// Counts and indices of series, clusters and rows: 64-bit, so no count needs a
/// range check.
using index_t = std::int64_t;
} // namespace dtwc


namespace dtwc::settings {
/// @brief Namespace alias for std::filesystem.
namespace fs = std::filesystem;

/// @brief Flag for debug mode for developers.
constexpr bool isDebug = false;

/// Invocation-local default seed for deterministic sampling algorithms (distinct from legacy `dtwc::randGenerator`).
inline constexpr std::uint64_t DEFAULT_RANDOM_SEED = 42;

/// @brief Default Sakoe-Chiba band half-width (-1 = unconstrained).
constexpr int DEFAULT_BAND = -1;

/// @brief Default mixed-integer programming solver.
constexpr dtwc::Solver DEFAULT_MIP_SOLVER = dtwc::Solver::HiGHS;

/// @brief Default method for clustering.
constexpr dtwc::Method DEFAULT_CLUSTERING_METHOD = dtwc::Method::Kmedoids;

/// @brief Default maximum number of iterations.
constexpr int DEFAULT_MAX_ITER = 100;
} // namespace dtwc::settings
