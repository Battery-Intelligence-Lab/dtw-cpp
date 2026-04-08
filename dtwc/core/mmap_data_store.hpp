/**
 * @file mmap_data_store.hpp
 * @brief Memory-mapped contiguous cache for time series data.
 *
 * @details Writes loaded series data to a contiguous mmap file for fast
 * restart. This is an internal cache, not a user-facing format.
 *
 * Binary layout (64-byte header + offset table + data):
 *   bytes 0-3:    magic "DTWS"
 *   bytes 4-5:    version uint16 = 1
 *   bytes 6-9:    endian marker uint32 = 0x01020304
 *   byte  10:     elem_size uint8 = 8 (sizeof(double))
 *   byte  11:     reserved = 0
 *   bytes 12-19:  N (uint64_t) -- series count
 *   bytes 20-27:  ndim (uint64_t) -- features per timestep
 *   bytes 28-31:  header CRC32 (of bytes 0-27)
 *   bytes 32-63:  reserved (zero)
 *
 * Offset table ((N+1) * uint64_t):
 *   offsets[i] = byte offset from start of data section to series i
 *   offsets[N] = total data size in bytes (sentinel)
 *
 * Data section:
 *   Contiguous doubles for all series. No padding between series.
 *
 * @author Claude 4.6
 * @date 08 Apr 2026
 */

#pragma once

#include "crc32.hpp"
#include "../Data.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

#include <llfio/v2.0/llfio.hpp>

namespace dtwc::core {

namespace llfio = LLFIO_V2_NAMESPACE;

class MmapDataStore {
  static constexpr size_t HEADER_SIZE = 64;
  static constexpr char MAGIC[4] = { 'D', 'T', 'W', 'S' };
  static constexpr uint16_t VERSION = 1;
  static constexpr uint32_t ENDIAN_MARKER = 0x01020304u;
  static constexpr uint8_t ELEM_SIZE = 8;

  llfio::mapped_file_handle mfh_;
  const double *data_base_{ nullptr };   // start of data section
  const uint64_t *offsets_{ nullptr };   // offset table (N+1 entries)
  size_t n_{ 0 };
  size_t ndim_{ 1 };

  MmapDataStore() = default;

  /// Compute total file size for N series with total_values doubles.
  static size_t compute_file_size(size_t n, size_t total_values)
  {
    return HEADER_SIZE + (n + 1) * sizeof(uint64_t) + total_values * sizeof(double);
  }

  /// Write the 64-byte header at base.
  static void write_header(uint8_t *base, size_t n, size_t ndim)
  {
    std::memset(base, 0, HEADER_SIZE); // zero entire header first
    std::memcpy(base, MAGIC, 4);
    const uint16_t ver = VERSION;
    std::memcpy(base + 4, &ver, 2);
    const uint32_t em = ENDIAN_MARKER;
    std::memcpy(base + 6, &em, 4);
    base[10] = ELEM_SIZE;
    base[11] = 0; // reserved
    const uint64_t n64 = static_cast<uint64_t>(n);
    std::memcpy(base + 12, &n64, 8);
    const uint64_t ndim64 = static_cast<uint64_t>(ndim);
    std::memcpy(base + 20, &ndim64, 8);
    // CRC32 of bytes 0-27
    const uint32_t crc = detail::crc32_naive(base, 28);
    std::memcpy(base + 28, &crc, 4);
    // bytes 32-63 already zeroed above
  }

  /// Validate the header. Throws on failure.
  static void validate_header(const uint8_t *base, size_t file_len)
  {
    if (file_len < HEADER_SIZE)
      throw std::runtime_error("MmapDataStore: file too small for header");

    if (std::memcmp(base, MAGIC, 4) != 0)
      throw std::runtime_error("MmapDataStore: bad magic bytes");

    uint16_t ver{};
    std::memcpy(&ver, base + 4, 2);
    if (ver != VERSION)
      throw std::runtime_error("MmapDataStore: unsupported version " + std::to_string(ver));

    uint32_t em{};
    std::memcpy(&em, base + 6, 4);
    if (em != ENDIAN_MARKER)
      throw std::runtime_error("MmapDataStore: endian mismatch");

    uint8_t es = base[10];
    if (es != ELEM_SIZE)
      throw std::runtime_error("MmapDataStore: unexpected elem_size " + std::to_string(es));

    uint32_t stored_crc{};
    std::memcpy(&stored_crc, base + 28, 4);
    const uint32_t computed_crc = detail::crc32_naive(base, 28);
    if (stored_crc != computed_crc)
      throw std::runtime_error("MmapDataStore: header CRC mismatch");
  }

  /// Set up internal pointers from the mapped base address.
  void setup_pointers(const uint8_t *base)
  {
    offsets_ = reinterpret_cast<const uint64_t *>(base + HEADER_SIZE);
    data_base_ = reinterpret_cast<const double *>(
      base + HEADER_SIZE + (n_ + 1) * sizeof(uint64_t));
  }

public:
  MmapDataStore(MmapDataStore &&) = default;
  MmapDataStore &operator=(MmapDataStore &&) = default;
  MmapDataStore(const MmapDataStore &) = delete;
  MmapDataStore &operator=(const MmapDataStore &) = delete;

  /// Create a new mmap cache from loaded Data.
  static MmapDataStore create(const std::filesystem::path &path, const Data &data)
  {
    MmapDataStore store;
    store.n_ = data.size();
    store.ndim_ = data.ndim;

    // Calculate total values across all series
    size_t total_values = 0;
    for (size_t i = 0; i < store.n_; ++i)
      total_values += data.series_flat_size(i);

    const size_t file_sz = compute_file_size(store.n_, total_values);

    auto result = llfio::mapped_file_handle::mapped_file(
      file_sz, {}, path,
      llfio::file_handle::mode::write,
      llfio::file_handle::creation::if_needed,
      llfio::file_handle::caching::all,
      llfio::file_handle::flag::none);

    if (!result)
      throw std::runtime_error(std::string("MmapDataStore::create: ") +
                               result.error().message().c_str());

    store.mfh_ = std::move(result.value());
    store.mfh_.truncate(file_sz).value();
    store.mfh_.update_map().value();

    auto *base = reinterpret_cast<uint8_t *>(store.mfh_.address());
    if (!base)
      throw std::runtime_error("MmapDataStore::create: null address after mapping");

    // Write header
    write_header(base, store.n_, store.ndim_);

    // Write offset table
    auto *offsets = reinterpret_cast<uint64_t *>(base + HEADER_SIZE);
    uint64_t byte_offset = 0;
    for (size_t i = 0; i < store.n_; ++i) {
      offsets[i] = byte_offset;
      byte_offset += static_cast<uint64_t>(data.series_flat_size(i)) * sizeof(double);
    }
    offsets[store.n_] = byte_offset; // sentinel

    // Write data section
    auto *data_start = base + HEADER_SIZE + (store.n_ + 1) * sizeof(uint64_t);
    size_t write_pos = 0;
    for (size_t i = 0; i < store.n_; ++i) {
      const auto sp = data.series(i);
      const size_t nbytes = sp.size() * sizeof(double);
      std::memcpy(data_start + write_pos, sp.data(), nbytes);
      write_pos += nbytes;
    }

    store.setup_pointers(base);
    return store;
  }

  /// Open an existing mmap cache file (warm-start).
  static MmapDataStore open(const std::filesystem::path &path)
  {
    MmapDataStore store;

    auto result = llfio::mapped_file_handle::mapped_file(
      0, {}, path,
      llfio::file_handle::mode::write,
      llfio::file_handle::creation::open_existing,
      llfio::file_handle::caching::all,
      llfio::file_handle::flag::none);

    if (!result)
      throw std::runtime_error(std::string("MmapDataStore::open: ") +
                               result.error().message().c_str());

    store.mfh_ = std::move(result.value());
    store.mfh_.update_map().value();

    auto *base = reinterpret_cast<const uint8_t *>(store.mfh_.address());
    if (!base)
      throw std::runtime_error("MmapDataStore::open: null address after mapping");

    const auto file_len = static_cast<size_t>(store.mfh_.maximum_extent().value());
    validate_header(base, file_len);

    // Read N and ndim from header
    uint64_t n64{}, ndim64{};
    std::memcpy(&n64, base + 12, 8);
    std::memcpy(&ndim64, base + 20, 8);
    store.n_ = static_cast<size_t>(n64);
    store.ndim_ = static_cast<size_t>(ndim64);

    // Validate file size
    const size_t offset_table_end = HEADER_SIZE + (store.n_ + 1) * sizeof(uint64_t);
    if (file_len < offset_table_end)
      throw std::runtime_error("MmapDataStore::open: file truncated (offset table)");

    store.setup_pointers(base);

    // Validate total data size against file
    if (store.n_ > 0) {
      const size_t total_data_bytes = static_cast<size_t>(store.offsets_[store.n_]);
      const size_t expected = offset_table_end + total_data_bytes;
      if (file_len < expected)
        throw std::runtime_error("MmapDataStore::open: file truncated (data section)");
    }

    return store;
  }

  /// Number of series.
  size_t size() const { return n_; }

  /// Number of features (dimensions) per timestep.
  size_t ndim() const { return ndim_; }

  /// Pointer to series i's flat data (timesteps * ndim contiguous doubles).
  const double *series_data(size_t i) const
  {
    assert(i < n_);
    return data_base_ + offsets_[i] / sizeof(double);
  }

  /// Number of scalar values in series i (timesteps * ndim).
  size_t series_flat_size(size_t i) const
  {
    assert(i < n_);
    return static_cast<size_t>(offsets_[i + 1] - offsets_[i]) / sizeof(double);
  }

  /// Number of timesteps in series i (flat_size / ndim).
  size_t series_length(size_t i) const
  {
    return series_flat_size(i) / ndim_;
  }

  /// Span view of series i's flat data (zero-copy).
  std::span<const double> series(size_t i) const
  {
    return { series_data(i), series_flat_size(i) };
  }

  /// Flush mapped memory to disk.
  void sync()
  {
    mfh_.barrier({}, llfio::mapped_file_handle::barrier_kind::nowait_data_only).value();
  }
};

} // namespace dtwc::core
