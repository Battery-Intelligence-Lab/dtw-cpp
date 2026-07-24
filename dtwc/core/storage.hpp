/// @file storage.hpp — Storage policy and precision enums.
///
/// @author Volkan Kumtepeli
/// @author Claude 4.6
/// @date 08 Apr 2026
#pragma once

#include "../error.hpp"

namespace dtwc::core {

/// Controls how Problem stores time series data.
enum class StoragePolicy {
  Auto, ///< Choose at the next owning load/set-data boundary: heap when the
        ///< estimated series footprint fits, else the mmap-backed store. A
        ///< DataLoader::ram_limit() overrides its threshold; Problem::set_data()
        ///< uses the platform default. This is unrelated to CLI `--ram-limit`.
  Heap, ///< In-memory vector-of-vectors (default for small datasets).
  Mmap  ///< Memory-mapped file via MmapDataStore.
};

/// Controls the precision of stored time series data.
/// DTW functions are templated — both float and double codepaths are always compiled.
/// Distance matrix always uses double regardless of this setting.
enum class Precision {
  Float32, ///< Store series as float (4 bytes). Opt-in — 2x memory saving.
  Float64  ///< Store series as double (8 bytes). Default — full precision.
};

inline void validate_storage_policy(StoragePolicy value)
{
  switch (value) {
  case StoragePolicy::Auto:
  case StoragePolicy::Heap:
  case StoragePolicy::Mmap:
    return;
  }
  throw InvalidInput("Invalid StoragePolicy value.");
}

inline void validate_precision(Precision value)
{
  switch (value) {
  case Precision::Float32:
  case Precision::Float64:
    return;
  }
  throw InvalidInput("Invalid Precision value.");
}

} // namespace dtwc::core
