/**
 * @file names.hpp
 * @brief String <-> enum tables: one per enum, kept beside the enum it names.
 *
 * @details A table lists every accepted spelling of every value. The first entry
 * for a value is its canonical name: name_of() returns it, a config file is
 * written with it, and the "Valid:" list of an error names only it. Later entries
 * are aliases. parse_name() matches ASCII case-insensitively and nothing else (no
 * trimming, no '-' / '_' folding): an alias exists because a table lists it.
 *
 * @date 24 Sep 2026
 */

#pragma once

#include "error.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

namespace dtwc {

/// One spelling of one enum value.
template <class E>
struct Name
{
  std::string_view text;
  E value;
};

namespace detail {

constexpr bool equals_ignoring_ascii_case(std::string_view a, std::string_view b) noexcept
{
  if (a.size() != b.size()) return false;
  constexpr auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; };
  for (std::size_t i = 0; i < a.size(); ++i)
    if (lower(a[i]) != lower(b[i])) return false;
  return true;
}

} // namespace detail

/// The value spelled `text` in `table`, ignoring ASCII case.
/// @throws InvalidInput "unknown <what> '<text>'. Valid: <canonical names>."
template <class E, std::size_t N>
E parse_name(const Name<E> (&table)[N], std::string_view text, std::string_view what)
{
  for (const auto &entry : table)
    if (detail::equals_ignoring_ascii_case(entry.text, text)) return entry.value;

  std::string valid;
  for (std::size_t i = 0; i < N; ++i) {
    bool canonical = true;
    for (std::size_t j = 0; j < i && canonical; ++j) canonical = table[j].value != table[i].value;
    if (!canonical) continue;
    if (!valid.empty()) valid += ", ";
    valid += table[i].text;
  }
  throw InvalidInput("unknown " + std::string(what) + " '" + std::string(text) + "'. Valid: " + valid + ".");
}

/// The canonical name of `value`: its first entry in `table`.
/// @throws InvalidInput when no entry names `value` (e.g. MetricType::L2, which no
///         front end spells).
template <class E, std::size_t N>
std::string_view name_of(const Name<E> (&table)[N], E value)
{
  for (const auto &entry : table)
    if (entry.value == value) return entry.text;
  long long number = 0;
  if constexpr (std::is_enum_v<E>)
    number = static_cast<long long>(static_cast<std::underlying_type_t<E>>(value));
  else
    number = static_cast<long long>(value);
  throw InvalidInput("value " + std::to_string(number) + " has no name, so it cannot be written as text.");
}

} // namespace dtwc
