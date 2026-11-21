/**
 * @file highs_support.hpp
 * @brief HiGHS option guard shared by HiGHS-backed MIP entry points.
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#pragma once

#include "../base/error.hpp"

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
#endif

} // namespace dtwc::mip
