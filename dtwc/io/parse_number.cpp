/**
 * @file parse_number.cpp
 * @brief fast_float behind dtwc::io::parse_number (see parse_number.hpp).
 *
 * @date 23 Sep 2026
 */

#include "parse_number.hpp"

#include "fast_float/fast_float.h" // vendored v8.3.0; PRIVATE include of dtwc++

#include <system_error>

namespace dtwc::io {

namespace {

template <typename T>
std::from_chars_result parse(const char *first, const char *last, T &value) noexcept
{
  T parsed{};
  const auto result = fast_float::from_chars(first, last, parsed,
                                             fast_float::chars_format::general);
  if (result.ec == std::errc{}) value = parsed;
  return { result.ptr, result.ec };
}

} // namespace

std::from_chars_result parse_number(const char *first, const char *last,
                                    double &value) noexcept
{
  return parse(first, last, value);
}

std::from_chars_result parse_number(const char *first, const char *last,
                                    float &value) noexcept
{
  return parse(first, last, value);
}

} // namespace dtwc::io
