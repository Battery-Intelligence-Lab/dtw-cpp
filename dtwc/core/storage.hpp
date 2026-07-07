/// @file storage.hpp — Storage policy and precision enums.
///
/// @author Volkan Kumtepeli
/// @author Claude 4.6
/// @date 08 Apr 2026
#pragma once

namespace dtwc::core {

/// Controls how Problem stores time series data.
enum class StoragePolicy {
  Auto, ///< Choose at load time: heap when the estimated footprint (rows x lengths x
        ///< sizeof(data_t)) fits, else the mmap-backed store. Threshold defaults to
        ///< 50% of free RAM, overridable via DataLoader::ram_limit() / `--ram-limit`.
        ///< Routing lives in DataLoader::load_stored() (Task 1.4).
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

} // namespace dtwc::core
