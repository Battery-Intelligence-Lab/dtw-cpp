/**
 * @file mmap_distance_matrix.hpp
 * @brief Memory-mapped symmetric distance matrix with packed triangular storage.
 *
 * @details Uses llfio's mapped_file_handle to memory-map a file-backed packed
 * lower-triangular array. Supports warm-start: create once, destroy, reopen
 * later with all computed distances intact. Uncomputed entries use a NaN
 * sentinel (same convention as DenseDistanceMatrix).
 *
 * Binary layout (32-byte header + packed doubles):
 *   bytes 0-3:    magic "DTWM"
 *   bytes 4-5:    version uint16 = 1
 *   bytes 6-9:    endian marker uint32 = 0x01020304
 *   byte  10:     elem_size uint8 = 8 (sizeof(double))
 *   byte  11:     reserved = 0
 *   bytes 12-19:  N (uint64_t) — matrix dimension
 *   bytes 20-23:  header CRC32 (of bytes 0-19)
 *   bytes 24-31:  reserved (zero, alignment padding)
 *   bytes 32+:    double[N*(N+1)/2], NaN = uncomputed
 *
 * Thread-safety contract: same as DenseDistanceMatrix — no locking.
 * Parallel fills partition pairs so each (i,j) written by exactly one thread.
 *
 * @author Claude 4.6
 * @date 08 Apr 2026
 */

#pragma once

#include "crc32.hpp"           // detail::crc32_naive
#include "distance_matrix.hpp" // tri_index, packed_size

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>

// llfio is an OPTIONAL dependency (DTWC_HAS_MMAP). When it is absent the class
// below is still a COMPLETE type — it keeps its packed-double storage and every
// pure-arithmetic accessor — so `Problem::distMat_t`
// (std::variant<DenseDistanceMatrix, MmapDistanceMatrix>) and all std::visit /
// std::get sites compile unchanged. Only the four members that actually touch a
// memory-mapped file (the mapped-file handle, the file-creating constructor,
// open(), and sync()) are compiled out; their no-llfio replacements throw, so a
// build without llfio can never silently pretend to have mmap support.
#ifdef DTWC_HAS_MMAP
#include <llfio/v2.0/llfio.hpp>
#endif

namespace dtwc::core {

#ifdef DTWC_HAS_MMAP
namespace llfio = LLFIO_V2_NAMESPACE;
#endif

class MmapDistanceMatrix {
public:
  static constexpr size_t header_size = 32;
  static constexpr char magic[4] = { 'D', 'T', 'W', 'M' };
  static constexpr uint16_t version = 1;
  static constexpr uint32_t endian_marker = 0x01020304u;
  static constexpr uint8_t elem_size = 8;

private:
#ifdef DTWC_HAS_MMAP
  llfio::mapped_file_handle mfh_;
#endif
  double *data_{ nullptr };
  size_t n_{ 0 };

  /// Compute total file size: header + packed doubles.
  /// Throws if n*(n+1)/2 or the byte total overflows size_t. Without this guard a
  /// crafted N (e.g. 2^62, where packed_size(n)*8 == 2^64 == 0 mod 2^64) wraps the
  /// size to header_size, defeating the truncation check and enabling OOB reads
  /// (audit CRITICAL #4).
  static size_t file_size(size_t n)
  {
    constexpr size_t max_sz = std::numeric_limits<size_t>::max();
    // Guard n+1 (n == SIZE_MAX would wrap to 0 and hide the overflow below).
    if (n == max_sz)
      throw std::runtime_error("MmapDistanceMatrix: N too large (overflow)");
    const size_t np1 = n + 1;
    // Guard n*(n+1); one of the two factors is even, so packed = n*(n+1)/2 is exact.
    if (n != 0 && np1 > max_sz / n)
      throw std::runtime_error("MmapDistanceMatrix: N too large (packed size overflows size_t)");
    const size_t packed = n * np1 / 2; // == packed_size(n)
    // Guard packed * sizeof(double) + header_size.
    if (packed > (max_sz - header_size) / sizeof(double))
      throw std::runtime_error("MmapDistanceMatrix: file size overflows size_t");
    return header_size + packed * sizeof(double);
  }

  /// Write the 32-byte header at base.
  static void write_header(uint8_t *base, size_t n)
  {
    std::memcpy(base + 0, magic, 4);
    const uint16_t ver = version;
    std::memcpy(base + 4, &ver, 2);
    const uint32_t em = endian_marker;
    std::memcpy(base + 6, &em, 4);
    base[10] = elem_size;
    base[11] = 0; // reserved
    const uint64_t n64 = static_cast<uint64_t>(n);
    std::memcpy(base + 12, &n64, 8);
    // CRC32 of bytes 0-19
    const uint32_t crc = detail::crc32_naive(base, 20);
    std::memcpy(base + 20, &crc, 4);
    // bytes 24-31: reserved (zero)
    std::memset(base + 24, 0, 8);
  }

  /// Validate the header at base. Returns N on success, throws on failure.
  static size_t validate_header(const uint8_t *base, size_t file_len)
  {
    if (file_len < header_size)
      throw std::runtime_error("MmapDistanceMatrix: file too small for header");

    if (std::memcmp(base, magic, 4) != 0)
      throw std::runtime_error("MmapDistanceMatrix: bad magic bytes");

    uint16_t ver{};
    std::memcpy(&ver, base + 4, 2);
    if (ver != version)
      throw std::runtime_error("MmapDistanceMatrix: unsupported version " + std::to_string(ver));

    uint32_t em{};
    std::memcpy(&em, base + 6, 4);
    if (em != endian_marker)
      throw std::runtime_error("MmapDistanceMatrix: endian mismatch");

    uint8_t es = base[10];
    if (es != elem_size)
      throw std::runtime_error("MmapDistanceMatrix: unexpected elem_size " + std::to_string(es));

    uint32_t stored_crc{};
    std::memcpy(&stored_crc, base + 20, 4);
    const uint32_t computed_crc = detail::crc32_naive(base, 20);
    if (stored_crc != computed_crc)
      throw std::runtime_error("MmapDistanceMatrix: header CRC mismatch");

    uint64_t n64{};
    std::memcpy(&n64, base + 12, 8);
    const auto n = static_cast<size_t>(n64);

    const size_t expected = file_size(n); // throws if n*(n+1)/2 or the byte total overflows size_t
    if (file_len < expected)
      throw std::runtime_error("MmapDistanceMatrix: file truncated (expected " +
                               std::to_string(expected) + " bytes, got " + std::to_string(file_len) + ")");

    return n;
  }

#ifdef DTWC_HAS_MMAP
  /// Private constructor used by both create and open paths.
  MmapDistanceMatrix(llfio::mapped_file_handle mfh, double *data, size_t n)
    : mfh_(std::move(mfh)), data_(data), n_(n) {}
#endif

public:
  MmapDistanceMatrix() = default;
  MmapDistanceMatrix(MmapDistanceMatrix &&) = default;
  MmapDistanceMatrix &operator=(MmapDistanceMatrix &&) = default;
  MmapDistanceMatrix(const MmapDistanceMatrix &) = delete;
  MmapDistanceMatrix &operator=(const MmapDistanceMatrix &) = delete;

#ifndef DTWC_HAS_MMAP
  /// No-llfio build: memory-mapped storage is unavailable. Constructing or
  /// opening a file-backed matrix throws rather than silently degrading.
  [[noreturn]] explicit MmapDistanceMatrix(const std::filesystem::path &, size_t)
  {
    throw std::runtime_error(
      "MmapDistanceMatrix: this build has no memory-mapped support "
      "(rebuild with -DDTWC_ENABLE_LLFIO=ON / llfio available).");
  }

  [[noreturn]] static MmapDistanceMatrix open(const std::filesystem::path &)
  {
    throw std::runtime_error(
      "MmapDistanceMatrix::open: this build has no memory-mapped support "
      "(rebuild with -DDTWC_ENABLE_LLFIO=ON / llfio available).");
  }

  [[noreturn]] void sync()
  {
    throw std::runtime_error("MmapDistanceMatrix::sync: no memory-mapped support in this build.");
  }
#else
  /// Create a new memory-mapped distance matrix at cache_path.
  explicit MmapDistanceMatrix(const std::filesystem::path &cache_path, size_t n)
  {
    const size_t total = file_size(n);

    auto result = llfio::mapped_file_handle::mapped_file(
      total, {}, cache_path,
      llfio::file_handle::mode::write,
      llfio::file_handle::creation::if_needed,
      llfio::file_handle::caching::all,
      llfio::file_handle::flag::none);

    if (!result)
      throw std::runtime_error(std::string("MmapDistanceMatrix: failed to create file: ") +
                               result.error().message());

    mfh_ = std::move(result.value());
    mfh_.truncate(total).value();
    mfh_.update_map().value();

    auto *base = reinterpret_cast<uint8_t *>(mfh_.address());
    if (!base)
      throw std::runtime_error("MmapDistanceMatrix: null address after mapping");

    write_header(base, n);
    data_ = reinterpret_cast<double *>(base + header_size);
    n_ = n;

    // Fill data region with NaN (uncomputed sentinel)
    const size_t count = packed_size(n);
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < count; ++i)
      data_[i] = nan_val;
  }

  /// Open an existing memory-mapped distance matrix (warm-start).
  static MmapDistanceMatrix open(const std::filesystem::path &cache_path)
  {
    auto result = llfio::mapped_file_handle::mapped_file(
      0, {}, cache_path,
      llfio::file_handle::mode::write,
      llfio::file_handle::creation::open_existing,
      llfio::file_handle::caching::all,
      llfio::file_handle::flag::none);

    if (!result)
      throw std::runtime_error(std::string("MmapDistanceMatrix::open: failed to open file: ") +
                               result.error().message());

    auto mfh = std::move(result.value());
    mfh.update_map().value();

    auto *base = reinterpret_cast<uint8_t *>(mfh.address());
    if (!base)
      throw std::runtime_error("MmapDistanceMatrix::open: null address after mapping");

    const auto file_len = static_cast<size_t>(mfh.maximum_extent().value());
    const size_t n = validate_header(base, file_len);

    auto *data = reinterpret_cast<double *>(base + header_size);
    return MmapDistanceMatrix(std::move(mfh), data, n);
  }
#endif // DTWC_HAS_MMAP

  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return data_[tri_index(i, j)];
  }

  /// Set distance. Parallel fills must use disjoint (i,j) pairs — no locking needed.
  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_ && !std::isnan(v));
    data_[tri_index(i, j)] = v;
  }

  bool is_computed(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return !std::isnan(data_[tri_index(i, j)]);
  }

  size_t size() const { return n_; }

  double max() const
  {
    const size_t count = packed_size(n_);
    double result = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < count; ++i) {
      const double d = data_[i];
      if (!std::isnan(d) && d > result)
        result = d;
    }
    return std::isfinite(result) ? result : 0.0;
  }

  size_t count_computed() const
  {
    const size_t count = packed_size(n_);
    size_t computed = 0;
    for (size_t i = 0; i < count; ++i)
      if (!std::isnan(data_[i]))
        ++computed;
    return computed;
  }

  bool all_computed() const
  {
    const size_t count = packed_size(n_);
    for (size_t i = 0; i < count; ++i)
      if (std::isnan(data_[i]))
        return false;
    return true;
  }

  double *raw() { return data_; }
  const double *raw() const { return data_; }
  size_t packed_count() const { return packed_size(n_); }

#ifdef DTWC_HAS_MMAP
  /// Flush mapped memory to disk.
  void sync()
  {
    mfh_.barrier({}, llfio::mapped_file_handle::barrier_kind::nowait_data_only).value();
  }
#endif
};

} // namespace dtwc::core
