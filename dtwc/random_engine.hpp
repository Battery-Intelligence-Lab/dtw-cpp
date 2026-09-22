/**
 * @file random_engine.hpp
 * @brief The legacy process-global Mersenne Twister engine.
 *
 * @details Split out of settings.hpp (ledger X-12) so that header no longer has
 * to include `<random>`. `settings.hpp` reaches roughly forty translation units,
 * and it was pulling a standard-library header in for one object that almost
 * none of them use.
 *
 * The engine itself is unchanged and stays where the 2.0 API contract says it is
 * — `dtwc::randGenerator`, seeded 29 — and `dtwc/dtwc.hpp` includes this header,
 * so anything including the umbrella header still sees it. Only code that
 * included `settings.hpp` *directly* and relied on it transitively needs to
 * include this file instead.
 *
 * A21/S-11 will decide the engine's future: it is shared mutable state consumed
 * by the unseeded Tier-2 entry points, and Tier-1 deliberately never touches it.
 * This move is about the include graph only and changes no behaviour.
 */

#pragma once

#include <random>

namespace dtwc {

/// @brief Legacy mutable Mersenne Twister engine for unseeded Tier-2 calls.
/// @details Its initial seed remains 29 for compatibility. Deterministic Tier-1
///          entry points instead construct invocation-local engines from
///          `settings::DEFAULT_RANDOM_SEED`; they never consume this state.
inline std::mt19937 randGenerator(29); // NOLINT(cert-msc51-cpp): fixed seed is the documented reproducibility contract.

} // namespace dtwc
