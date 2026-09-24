/**
 * @file parse_number.hpp
 * @brief Locale-free, correctly rounded text-to-floating-point parsing.
 *
 * @details Apple's libc++ ships the floating-point overloads of
 * `std::from_chars` only from macOS 26 (they compile only at
 * `-mmacosx-version-min=26.0`), so a binary that called them ran on nothing
 * older. `parse_number()` has the contract of
 * `std::from_chars(first, last, value, std::chars_format::general)`: no leading
 * whitespace and no leading '+', "nan" / "inf" accepted, `result_out_of_range`
 * on overflow and on underflow to zero, and `value` assigned only on success.
 * It is implemented in parse_number.cpp with the vendored fast_float
 * (dtwc/extern/fast_float, Apache-2.0 / MIT / BSL-1.0), so no installed header
 * includes fast_float.
 *
 * @date 23 Sep 2026
 */

#pragma once

#include <charconv> // std::from_chars_result

namespace dtwc::io {

std::from_chars_result parse_number(const char *first, const char *last,
                                    double &value) noexcept;
std::from_chars_result parse_number(const char *first, const char *last,
                                    float &value) noexcept;

} // namespace dtwc::io
