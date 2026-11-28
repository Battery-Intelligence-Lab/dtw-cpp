/**
 * @file series.hpp
 * @brief What the per-pair distance functions take as a series: a contiguous range of float or double.
 *
 * @details A std::vector, a std::span, a std::array or a C array of float or double is passed as it is, and the element
 *          type, which is the precision of the distance, is read off it. A pointer and a length are passed as
 *          `std::span<const T>(pointer, length)`.
 *
 * @author Volkan Kumtepeli
 * @date 24 Nov 2026
 */

#pragma once

#include <concepts>
#include <ranges>

namespace dtwc::core {

/// The element type of a series: the `T` of a std::vector<T>, a std::span<T>, a std::array<T, N> or a T[N].
template <typename Series>
using series_value_t = std::ranges::range_value_t<Series>;

/// Two series of one element type, float or double: contiguous ranges, so a pointer and a length reach the kernel.
/// The entry points take them by const reference, so the const ranges are what must be contiguous and sized.
template <typename X, typename Y>
concept SeriesPair = std::ranges::contiguous_range<const X> && std::ranges::sized_range<const X>
                     && std::ranges::contiguous_range<const Y> && std::ranges::sized_range<const Y>
                     && (std::same_as<series_value_t<X>, float> || std::same_as<series_value_t<X>, double>)
                     && std::same_as<series_value_t<X>, series_value_t<Y>>;

} // namespace dtwc::core
