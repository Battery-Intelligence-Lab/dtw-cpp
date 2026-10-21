/// @file storage.hpp — Series precision enum.
///
/// @author Volkan Kumtepeli
/// @author Claude 4.6
/// @date 08 Apr 2026
#pragma once

#include "../base/names.hpp"

namespace dtwc::core {

/// Controls the precision of stored time series data (`dtype`) and of the distance matrix's values (`matrix_dtype`),
/// each on its own: neither sets the other.
/// DTW functions are templated — both float and double codepaths are always compiled.
enum class Precision {
  Float32, ///< Store series as float (4 bytes). Opt-in — 2x memory saving.
  Float64  ///< Store series as double (8 bytes). Default — full precision.
};

/// The spellings of `--dtype` and `--matrix-dtype`.
inline constexpr Name<Precision> precision_names[]{
  { "float32", Precision::Float32 },
  { "f32", Precision::Float32 },
  { "fp32", Precision::Float32 },
  { "float", Precision::Float32 },
  { "float64", Precision::Float64 },
  { "f64", Precision::Float64 },
  { "fp64", Precision::Float64 },
  { "double", Precision::Float64 },
};

} // namespace dtwc::core
