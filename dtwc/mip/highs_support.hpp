/**
 * @file highs_support.hpp
 * @brief HiGHS option guards shared by every HiGHS-backed MIP entry point.
 *
 * @details `Highs::setOptionValue` returns a status: an unknown option name, or
 * a value outside the option's domain, yields `kError`. Discarding that status
 * leaves the solve running on HiGHS's DEFAULTS — a mistyped `solver` string
 * quietly runs dual simplex while the result is still reported as a PDLP value.
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#pragma once

#include "../error.hpp"

#include <cstdio>
#include <string>
#include <string_view>

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

namespace dtwc::mip {

#ifdef DTWC_ENABLE_HIGHS
/// Set a HiGHS option, or throw — never let a rejected option select a default.
template <typename Value>
inline void set_highs_option(
  Highs &highs, const char *name, const Value &value, std::string_view backend)
{
  if (highs.setOptionValue(name, value) == HighsStatus::kError)
    throw SolverError(std::string(backend) + ": HiGHS rejected option '"
      + std::string(name)
      + "'. The option is unknown to this HiGHS build or the value is outside "
        "its domain; continuing would silently run a different solver "
        "configuration.");
}

/// Set a TUNING-only option, tolerating a HiGHS build that does not know it
/// (`kkt_tolerance` postdates HiGHS 1.7). Falling back to the build's default
/// changes neither the solver identity — the `solver` string stays hard-failing
/// above — nor the reported objective, whereas failing would mean no bound at all.
template <typename Value>
inline void set_highs_option_best_effort(
  Highs &highs, const char *name, const Value &value, std::string_view backend)
{
  if (highs.setOptionValue(name, value) == HighsStatus::kError)
    std::fprintf(stderr,
      "%.*s: this HiGHS build rejected the optional tuning option '%s'; "
      "continuing with its built-in default.\n",
      static_cast<int>(backend.size()), backend.data(), name);
}
#endif

} // namespace dtwc::mip
