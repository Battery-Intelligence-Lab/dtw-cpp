/// @file storage.hpp — Storage policy and precision enums.
#pragma once

namespace dtwc::core {

/// Controls how Problem stores time series data.
enum class StoragePolicy {
  Auto, ///< Choose based on dataset size (heap for small, mmap for large).
  Heap, ///< In-memory vector-of-vectors (default for small datasets).
  Mmap  ///< Memory-mapped file via MmapDataStore.
};

/// Controls the precision of stored time series data.
/// DTW functions are templated — both float and double codepaths are always compiled.
/// Distance matrix always uses double regardless of this setting.
enum class Precision {
  Float32, ///< Store series as float (4 bytes). Default — 2x memory saving.
  Float64  ///< Store series as double (8 bytes). Full precision.
};

} // namespace dtwc::core
