/**
 * @file random_engine.hpp
 * @brief The legacy process-global Mersenne Twister engine.
 *
 * @details Split out of settings.hpp so that header no longer has
 * to include `<random>`. `settings.hpp` reaches roughly forty translation units,
 * and it was pulling a standard-library header in for one object that almost
 * none of them use.
 *
 * The engine itself is unchanged — `dtwc::randGenerator`, seeded 29 — and
 * `dtwc/dtwc.hpp` includes this header, so anything including the umbrella header
 * still sees it. Only code that included `settings.hpp` *directly* and relied on
 * it transitively needs to include this file instead.
 *
 * The engine's future is undecided: it is shared mutable state that only the
 * v1 one-argument init::random and init::Kmeanspp read (one draw each, as their
 * seed), and Tier-1 deliberately never touches it.
 * This move is about the include graph only and changes no behaviour.
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
