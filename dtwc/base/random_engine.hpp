/**
 * @file random_engine.hpp
 * @brief Legacy process-global Mersenne Twister engine for v1 one-argument `init::random` / `init::Kmeanspp`,
 *        kept out of settings.hpp so that settings.hpp need not include `<random>`.
 * @author Volkan Kumtepeli
 */

#pragma once

#include <random>

namespace dtwc {

/// @brief Legacy mutable Mersenne Twister engine: the seed source of the v1
///        one-argument init::random and init::Kmeanspp.
/// @details Its initial seed remains 29 for compatibility. Every other entry
///          point takes its seed as an argument (default
///          `settings::DEFAULT_RANDOM_SEED`) and never consumes this state.
inline std::mt19937 randGenerator(29); // NOLINT(cert-msc51-cpp): fixed seed is the documented reproducibility contract.

} // namespace dtwc
