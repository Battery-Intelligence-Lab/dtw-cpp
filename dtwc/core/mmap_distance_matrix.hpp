/**
 * @file mmap_distance_matrix.hpp
 * @brief Memory-mapped symmetric distance matrix with packed triangular storage.
 *
 * @details Uses llfio's mapped_file_handle to memory-map a file-backed packed
 * lower-triangular array. Supports warm-start: create once, destroy, reopen
 * later with all computed distances intact. Uncomputed entries use a NaN
 * sentinel (same convention as DenseDistanceMatrix).
 *
 * Binary layout (64-byte v3 header + packed doubles + row digests):
 *   bytes 0-3:    magic "DTWM"
 *   bytes 4-5:    version uint16 = 3
 *   bytes 6-9:    endian marker uint32 = 0x01020304
 *   byte  10:     elem_size uint8 = 8 (sizeof(double))
 *   byte  11:     fingerprint algorithm = 1 (SHA-256)
 *   bytes 12-19:  N (uint64_t) — matrix dimension
 *   bytes 20-51:  SHA-256 distance-semantics fingerprint
 *   byte  52:     publication state (0 = initializing, 1 = ready)
 *   byte  53:     payload-integrity algorithm = 1
 *   byte  54:     payload digest lanes = 2
 *   byte  55:     payload digest word size = 8
 *   bytes 56-59:  reserved (zero)
 *   bytes 60-63:  header CRC32 (of bytes 0-59)
 *   bytes 64+:    double[N*(N+1)/2], NaN = uncomputed
 *   footer:       N rows x 2 uint64_t digest lanes
 *
 * Session/thread-safety contract: one live MmapDistanceMatrix owns a nonblocking
 * exclusive file lease for the cache path, so reopen validation cannot race a
 * writer in another object or process. Within that session, parallel fills may
 * write disjoint (i,j) cells; lock-free atomic row-digest deltas make disjoint
 * writes to one row safe. Concurrent writes to the same cell are unsupported.
 *
 * @author Claude 4.6
 * @date 08 Apr 2026
 */

#pragma once

#include "crc32.hpp"           // detail::crc32_naive
#include "distance_matrix.hpp" // tri_index, packed_size

#include <algorithm>
#include <array>
#include <cassert>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

// llfio is an OPTIONAL dependency (DTWC_HAS_MMAP). When it is absent the class
// below is still a COMPLETE type — it keeps its packed-double storage and every
// pure-arithmetic accessor — so `Problem::distMat_t`
// (std::variant<DenseDistanceMatrix, MmapDistanceMatrix>) and all std::visit /
// std::get sites compile unchanged. Only the four members that actually touch a
// memory-mapped file (the mapped-file handle, the file-creating constructor,
// open(), and sync()) are compiled out; their no-llfio replacements throw, so a
// build without llfio can never silently pretend to have mmap support.
#ifdef DTWC_HAS_MMAP
#include "llfio_include.hpp"
#endif

namespace dtwc::core {

#ifdef DTWC_HAS_MMAP
namespace llfio = LLFIO_V2_NAMESPACE;
#endif

class MmapDistanceMatrix {
public:
  using fingerprint_type = std::array<std::uint8_t, 32>;

  static constexpr size_t header_size = 64;
  static constexpr char magic[4] = { 'D', 'T', 'W', 'M' };
  static constexpr uint16_t version = 3;
  static constexpr uint32_t endian_marker = 0x01020304u;
  static constexpr uint8_t elem_size = 8;
  static constexpr uint8_t fingerprint_algorithm = 1; // SHA-256
  static constexpr uint8_t payload_integrity_algorithm = 1;
  static constexpr uint8_t payload_digest_lanes = 2;
  static constexpr uint8_t payload_digest_word_size = sizeof(std::uint64_t);
  static constexpr size_t publication_state_offset = 52;
  static constexpr size_t payload_integrity_algorithm_offset = 53;
  static constexpr size_t payload_digest_lanes_offset = 54;
  static constexpr size_t payload_digest_word_size_offset = 55;
  static constexpr uint8_t publication_state_initializing = 0;
  static constexpr uint8_t publication_state_ready = 1;
  static constexpr fingerprint_type unbound_fingerprint{};

private:
#ifdef DTWC_HAS_MMAP
  llfio::mapped_file_handle mfh_;
  bool owns_session_lease_{ false };
#endif
  double *data_{ nullptr };
  std::uint64_t *row_digests_{ nullptr };
  size_t n_{ 0 };
  fingerprint_type fingerprint_{};

  static_assert(sizeof(double) == elem_size,
                "MmapDistanceMatrix requires 8-byte IEEE double storage");
  static_assert(sizeof(std::uint64_t) == payload_digest_word_size,
                "MmapDistanceMatrix requires 8-byte digest words");
  static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free,
                "MmapDistanceMatrix v3 requires lock-free 64-bit atomic_ref");
  static_assert(std::atomic_ref<std::uint64_t>::required_alignment
                  <= sizeof(std::uint64_t),
                "MmapDistanceMatrix v3 requires at most 8-byte atomic alignment");

  struct Layout {
    size_t packed_count;
    size_t data_bytes;
    size_t digest_offset;
    size_t digest_word_count;
    size_t digest_bytes;
    size_t total_bytes;
  };

  struct HeaderMetadata {
    size_t n;
    fingerprint_type fingerprint;
    Layout layout;
  };

  /// Compute every v3 region offset/count/length with checked size_t arithmetic.
  /// Throws if n*(n+1)/2 or any byte total overflows size_t. Without this guard a
  /// crafted N (e.g. 2^62, where packed_size(n)*8 == 2^64 == 0 mod 2^64) wraps the
  /// size to header_size, defeating the truncation check and enabling OOB reads
  /// (audit CRITICAL #4).
  static Layout checked_layout(size_t n)
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
    if (packed > max_sz / sizeof(double))
      throw std::runtime_error("MmapDistanceMatrix: file size overflows size_t");
    const size_t data_bytes = packed * sizeof(double);
    if (data_bytes > max_sz - header_size)
      throw std::runtime_error("MmapDistanceMatrix: data offset overflows size_t");
    const size_t digest_offset = header_size + data_bytes;

    if (n > max_sz / payload_digest_lanes)
      throw std::runtime_error(
        "MmapDistanceMatrix: digest word count overflows size_t");
    const size_t digest_word_count = n * payload_digest_lanes;
    if (digest_word_count > max_sz / payload_digest_word_size)
      throw std::runtime_error(
        "MmapDistanceMatrix: digest byte count overflows size_t");
    const size_t digest_bytes = digest_word_count * payload_digest_word_size;
    if (digest_bytes > max_sz - digest_offset)
      throw std::runtime_error("MmapDistanceMatrix: total file size overflows size_t");
    const size_t total_bytes = digest_offset + digest_bytes;

    if (header_size % alignof(double) != 0
        || digest_offset % std::atomic_ref<std::uint64_t>::required_alignment != 0) {
      throw std::runtime_error(
        "MmapDistanceMatrix: v3 payload layout does not meet 8-byte alignment");
    }
    return {packed, data_bytes, digest_offset, digest_word_count,
            digest_bytes, total_bytes};
  }

  static_assert(sizeof(size_t) <= sizeof(std::uint64_t),
                "MmapDistanceMatrix v3 cannot encode size_t wider than uint64_t");

  static std::uint64_t avalanche64(std::uint64_t value) noexcept
  {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
  }

  /// Two domain-separated 64-bit contributions form each row's commutative
  /// digest. Including both row and packed index prevents the same IEEE payload
  /// at different cells from cancelling by position alone. This is strong
  /// accidental-corruption detection, not a keyed cryptographic authenticator:
  /// a stable corruption must collide in both lanes, while a crash that persists
  /// only one of the two lane deltas is guarded by at least the other 64-bit lane.
  static std::uint64_t payload_contribution(
    std::size_t row, std::size_t packed_index, std::uint64_t value_bits,
    std::size_t lane) noexcept
  {
    static constexpr std::array<std::uint64_t, payload_digest_lanes> domains{
      0x4454574d2d76332aull, 0x7061796c6f61642bull
    };
    static constexpr std::array<std::uint64_t, payload_digest_lanes> row_keys{
      0x9e3779b97f4a7c15ull, 0xd1b54a32d192ed03ull
    };
    static constexpr std::array<std::uint64_t, payload_digest_lanes> index_keys{
      0x94d049bb133111ebull, 0x369dea0f31a53f85ull
    };
    const std::uint64_t row_part = avalanche64(
      static_cast<std::uint64_t>(row) ^ row_keys[lane]);
    const std::uint64_t index_part = avalanche64(
      static_cast<std::uint64_t>(packed_index) ^ index_keys[lane]);
    const std::uint64_t bits_part = avalanche64(value_bits ^ domains[lane]);
    return avalanche64(domains[lane] ^ row_part
                       ^ std::rotl(index_part, lane == 0 ? 17 : 41)
                       ^ std::rotl(bits_part, lane == 0 ? 43 : 23));
  }

  static void validate_mapping_alignment(
    const std::uint8_t *base, const Layout &layout)
  {
    const auto data_address = reinterpret_cast<std::uintptr_t>(base + header_size);
    const auto digest_address = reinterpret_cast<std::uintptr_t>(
      base + layout.digest_offset);
    if (data_address % alignof(double) != 0
        || digest_address
             % std::atomic_ref<std::uint64_t>::required_alignment != 0) {
      throw std::runtime_error(
        "MmapDistanceMatrix: mapped v3 payload is not 8-byte aligned");
    }
  }

  static std::array<std::uint64_t, payload_digest_lanes> compute_row_digest(
    const double *data, std::size_t row) noexcept
  {
    std::array<std::uint64_t, payload_digest_lanes> digest{};
    const std::size_t row_start = row * (row + 1) / 2;
    for (std::size_t column = 0; column <= row; ++column) {
      const std::size_t index = row_start + column;
      const std::uint64_t bits = std::bit_cast<std::uint64_t>(data[index]);
      for (std::size_t lane = 0; lane < payload_digest_lanes; ++lane)
        digest[lane] ^= payload_contribution(row, index, bits, lane);
    }
    return digest;
  }

  static void initialize_payload(double *data, std::uint64_t *row_digests,
                                 std::size_t n)
  {
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t row = 0; row < n; ++row) {
      const std::size_t row_start = row * (row + 1) / 2;
      for (std::size_t column = 0; column <= row; ++column)
        data[row_start + column] = nan_value;
      const auto digest = compute_row_digest(data, row);
      for (std::size_t lane = 0; lane < payload_digest_lanes; ++lane)
        row_digests[row * payload_digest_lanes + lane] = digest[lane];
    }
  }

  /// Recompute into locals and expose no object until every row agrees. This
  /// function is deliberately read-only: corrupt files are never repaired or
  /// rewritten as a side effect of validation.
  static void validate_payload(
    const double *data, std::uint64_t *row_digests, std::size_t n)
  {
    for (std::size_t row = 0; row < n; ++row) {
      const auto expected = compute_row_digest(data, row);
      for (std::size_t lane = 0; lane < payload_digest_lanes; ++lane) {
        std::atomic_ref<std::uint64_t> stored(
          row_digests[row * payload_digest_lanes + lane]);
        if (!stored.is_lock_free()) {
          throw std::runtime_error(
            "MmapDistanceMatrix: platform lacks lock-free 64-bit payload digest atomics");
        }
        if (stored.load(std::memory_order_relaxed) != expected[lane]) {
          throw std::runtime_error(
            "MmapDistanceMatrix: payload integrity mismatch at row "
            + std::to_string(row)
            + "; the cache may be corrupt or incompletely persisted. "
              "The file was left unchanged; delete or rename it and recompute.");
        }
      }
    }
  }

  void xor_digest_delta(std::size_t row, std::size_t packed_index,
                        std::uint64_t old_bits, std::uint64_t new_bits,
                        std::size_t lane) noexcept
  {
    const std::uint64_t delta =
      payload_contribution(row, packed_index, old_bits, lane)
      ^ payload_contribution(row, packed_index, new_bits, lane);
    std::atomic_ref<std::uint64_t> digest(
      row_digests_[row * payload_digest_lanes + lane]);
    assert(digest.is_lock_free());
    digest.fetch_xor(delta, std::memory_order_relaxed);
  }

  /// Write the 64-byte v3 header at base.
  static void write_header(uint8_t *base, size_t n,
                           const fingerprint_type &fingerprint,
                           uint8_t publication_state)
  {
    std::memset(base, 0, header_size);
    std::memcpy(base + 0, magic, 4);
    const uint16_t ver = version;
    std::memcpy(base + 4, &ver, 2);
    const uint32_t em = endian_marker;
    std::memcpy(base + 6, &em, 4);
    base[10] = elem_size;
    base[11] = fingerprint_algorithm;
    const uint64_t n64 = static_cast<uint64_t>(n);
    std::memcpy(base + 12, &n64, 8);
    std::memcpy(base + 20, fingerprint.data(), fingerprint.size());
    base[publication_state_offset] = publication_state;
    base[payload_integrity_algorithm_offset] = payload_integrity_algorithm;
    base[payload_digest_lanes_offset] = payload_digest_lanes;
    base[payload_digest_word_size_offset] = payload_digest_word_size;
    // The CRC covers every metadata byte, including the fingerprint and
    // publication state/reserved area. A corrupted identity or torn state
    // transition can therefore never degrade into a valid ready cache.
    const uint32_t crc = detail::crc32_naive(base, 60);
    std::memcpy(base + 60, &crc, 4);
  }

  /// Publish a fully initialized cache by changing only the state and its CRC.
  /// The caller must durably flush the NaN-initialized data region before this
  /// transition and durably flush the updated header before returning.
  static void publish_ready_header(uint8_t *base)
  {
    base[publication_state_offset] = publication_state_ready;
    const uint32_t crc = detail::crc32_naive(base, 60);
    std::memcpy(base + 60, &crc, 4);
  }

  /// Validate the v3 header at base. Version is inspected before requiring the
  /// full v3 length so legacy caches get an actionable migration
  /// error rather than a generic "too small" error.
  static HeaderMetadata validate_header(const uint8_t *base, size_t file_len)
  {
    if (file_len < 4)
      throw std::runtime_error("MmapDistanceMatrix: file too small for magic bytes");

    if (std::memcmp(base, magic, 4) != 0)
      throw std::runtime_error("MmapDistanceMatrix: bad magic bytes");

    if (file_len < 6)
      throw std::runtime_error("MmapDistanceMatrix: file too small for version field");

    uint16_t ver{};
    std::memcpy(&ver, base + 4, 2);
    if (ver != version) {
      std::string message = "MmapDistanceMatrix: unsupported version " + std::to_string(ver);
      if (ver == 1) {
        message += " (legacy caches have no data/config fingerprint and cannot be "
                   "resumed safely; delete or rename the cache and recompute it)";
      } else if (ver == 2) {
        message += " (v2 authenticates its header and data/config identity but not "
                   "mutable packed distances; delete or rename the cache and "
                   "recompute it as payload-authenticated v3)";
      }
      throw std::runtime_error(message);
    }

    if (file_len < header_size)
      throw std::runtime_error("MmapDistanceMatrix: file too small for v3 header");

    uint32_t em{};
    std::memcpy(&em, base + 6, 4);
    if (em != endian_marker)
      throw std::runtime_error("MmapDistanceMatrix: endian mismatch");

    uint8_t es = base[10];
    if (es != elem_size)
      throw std::runtime_error("MmapDistanceMatrix: unexpected elem_size " + std::to_string(es));

    const uint8_t algorithm = base[11];
    if (algorithm != fingerprint_algorithm)
      throw std::runtime_error("MmapDistanceMatrix: unsupported fingerprint algorithm "
                               + std::to_string(algorithm));

    uint32_t stored_crc{};
    std::memcpy(&stored_crc, base + 60, 4);
    const uint32_t computed_crc = detail::crc32_naive(base, 60);
    if (stored_crc != computed_crc)
      throw std::runtime_error("MmapDistanceMatrix: header CRC mismatch");

    const uint8_t publication_state = base[publication_state_offset];
    if (publication_state == publication_state_initializing) {
      throw std::runtime_error(
        "MmapDistanceMatrix: cache initialization incomplete; its ready header "
        "was never durably published. Delete or rename the cache and recompute it.");
    }
    if (publication_state != publication_state_ready) {
      throw std::runtime_error(
        "MmapDistanceMatrix: unsupported publication state "
        + std::to_string(publication_state));
    }

    const uint8_t payload_algorithm = base[payload_integrity_algorithm_offset];
    if (payload_algorithm != payload_integrity_algorithm) {
      throw std::runtime_error(
        "MmapDistanceMatrix: unsupported payload integrity algorithm "
        + std::to_string(payload_algorithm));
    }
    const uint8_t digest_lanes = base[payload_digest_lanes_offset];
    if (digest_lanes != payload_digest_lanes) {
      throw std::runtime_error(
        "MmapDistanceMatrix: unexpected payload digest lane count "
        + std::to_string(digest_lanes));
    }
    const uint8_t digest_word_size = base[payload_digest_word_size_offset];
    if (digest_word_size != payload_digest_word_size) {
      throw std::runtime_error(
        "MmapDistanceMatrix: unexpected payload digest word size "
        + std::to_string(digest_word_size));
    }
    if (std::any_of(base + payload_digest_word_size_offset + 1, base + 60,
                    [](std::uint8_t byte) { return byte != 0; }))
      throw std::runtime_error("MmapDistanceMatrix: reserved header bytes are nonzero");

    uint64_t n64{};
    std::memcpy(&n64, base + 12, 8);
    if (n64 > static_cast<std::uint64_t>(std::numeric_limits<size_t>::max()))
      throw std::runtime_error("MmapDistanceMatrix: N cannot be represented by size_t");
    const auto n = static_cast<size_t>(n64);

    const Layout layout = checked_layout(n);
    if (file_len != layout.total_bytes)
      throw std::runtime_error("MmapDistanceMatrix: file length mismatch (expected " +
                               std::to_string(layout.total_bytes) + " bytes, got "
                               + std::to_string(file_len) + ")");

    fingerprint_type fingerprint{};
    std::memcpy(fingerprint.data(), base + 20, fingerprint.size());
    return {n, fingerprint, layout};
  }

#ifdef DTWC_HAS_MMAP
  static void acquire_session_lease(llfio::mapped_file_handle &mfh,
                                    const char *operation)
  {
    // Whole-file locks use flock on POSIX and a dedicated whole-file lock byte
    // on Windows in LLFIO. The zero-wait try is intentional: cache contention
    // is a typed configuration/runtime failure, never an unbounded wait.
    if (!mfh.try_lock_file()) {
      throw std::runtime_error(
        std::string("MmapDistanceMatrix::") + operation
        + ": exclusive session lease unavailable; another live matrix or "
          "process may already own this cache, or the filesystem may not "
          "support LLFIO whole-file locking. Close the owner or use a "
          "different cache path.");
    }
  }

  /// Private constructor used by both create and open paths.
  MmapDistanceMatrix(llfio::mapped_file_handle mfh, double *data,
                     std::uint64_t *row_digests, size_t n,
                     fingerprint_type fingerprint)
    : mfh_(std::move(mfh)), owns_session_lease_(true), data_(data),
      row_digests_(row_digests), n_(n), fingerprint_(std::move(fingerprint)) {}

  /// Durably flush one mapped range. The second publication barrier only
  /// needs the 64-byte header; flushing the complete O(N^2) packed region a
  /// second time would make safe creation unnecessarily twice as expensive.
  static void persist_range(
    llfio::mapped_file_handle &mfh, const uint8_t *base,
    size_t offset, size_t length,
    llfio::mapped_file_handle::barrier_kind kind)
  {
    llfio::mapped_file_handle::const_buffer_type buffer(
      reinterpret_cast<const llfio::byte *>(base + offset), length);
    const llfio::mapped_file_handle::const_buffers_type buffers(&buffer, 1);
    const llfio::mapped_file_handle::io_request<
      llfio::mapped_file_handle::const_buffers_type> request(buffers, offset);
    mfh.barrier(request, kind).value();
  }

  static MmapDistanceMatrix open_impl(
    const std::filesystem::path &cache_path,
    const fingerprint_type &expected_fingerprint)
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
    acquire_session_lease(mfh, "open");
    mfh.update_map().value();

    auto *base = reinterpret_cast<uint8_t *>(mfh.address());
    if (!base)
      throw std::runtime_error("MmapDistanceMatrix::open: null address after mapping");

    const auto extent = mfh.maximum_extent().value();
    if (extent > std::numeric_limits<size_t>::max())
      throw std::runtime_error(
        "MmapDistanceMatrix: mapped file length cannot be represented by size_t");
    const auto file_len = static_cast<size_t>(extent);
    const HeaderMetadata metadata = validate_header(base, file_len);

    if (metadata.fingerprint != expected_fingerprint) {
      throw std::runtime_error(
        "MmapDistanceMatrix: distance-cache fingerprint mismatch; this cache was "
        "created for different data or DTW configuration. Delete or rename the "
        "cache to recompute it, or use the original data, band, variant parameters, "
        "missing-data strategy, metric, precision, and compute backend.");
    }

    auto *data = reinterpret_cast<double *>(base + header_size);
    auto *row_digests = reinterpret_cast<std::uint64_t *>(
      base + metadata.layout.digest_offset);
    validate_mapping_alignment(base, metadata.layout);
    validate_payload(data, row_digests, metadata.n);
    return MmapDistanceMatrix(
      std::move(mfh), data, row_digests, metadata.n, metadata.fingerprint);
  }
#endif

public:
  MmapDistanceMatrix() = default;
#ifdef DTWC_HAS_MMAP
  MmapDistanceMatrix(MmapDistanceMatrix &&other) noexcept
    : mfh_(std::move(other.mfh_)),
      owns_session_lease_(std::exchange(other.owns_session_lease_, false)),
      data_(std::exchange(other.data_, nullptr)),
      row_digests_(std::exchange(other.row_digests_, nullptr)),
      n_(std::exchange(other.n_, 0)), fingerprint_(other.fingerprint_)
  {
    other.fingerprint_ = {};
  }

  MmapDistanceMatrix &operator=(MmapDistanceMatrix &&other) noexcept
  {
    if (this == &other) return *this;
    // The lock belongs to the native handle. mapped_file_handle assignment
    // unmaps and closes this object's old handle while its lease is still held,
    // then transfers the source native handle and lease. No RAII guard stores
    // an address that could point at a moved-from member.
    mfh_ = std::move(other.mfh_);
    owns_session_lease_ = std::exchange(other.owns_session_lease_, false);
    data_ = std::exchange(other.data_, nullptr);
    row_digests_ = std::exchange(other.row_digests_, nullptr);
    n_ = std::exchange(other.n_, 0);
    fingerprint_ = other.fingerprint_;
    other.fingerprint_ = {};
    return *this;
  }

  // mapped_file_handle closes its map before its native file handle. Keeping
  // the lease tied to that handle prevents a competing validator from entering
  // during teardown; closing the handle then releases the OS lock.
  ~MmapDistanceMatrix() = default;
#else
  MmapDistanceMatrix(MmapDistanceMatrix &&) = default;
  MmapDistanceMatrix &operator=(MmapDistanceMatrix &&) = default;
#endif
  MmapDistanceMatrix(const MmapDistanceMatrix &) = delete;
  MmapDistanceMatrix &operator=(const MmapDistanceMatrix &) = delete;

#ifndef DTWC_HAS_MMAP
  /// No-llfio build: memory-mapped storage is unavailable. Constructing or
  /// opening a file-backed matrix throws rather than silently degrading.
  [[noreturn]] explicit MmapDistanceMatrix(const std::filesystem::path &, size_t,
                                           const fingerprint_type & = {})
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

  [[noreturn]] static MmapDistanceMatrix open(const std::filesystem::path &,
                                              const fingerprint_type &)
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
  explicit MmapDistanceMatrix(const std::filesystem::path &cache_path, size_t n,
                              const fingerprint_type &fingerprint = {})
  {
    const Layout layout = checked_layout(n);
    const size_t total = layout.total_bytes;

    auto result = llfio::mapped_file_handle::mapped_file(
      total, {}, cache_path,
      llfio::file_handle::mode::write,
      llfio::file_handle::creation::only_if_not_exist,
      llfio::file_handle::caching::all,
      llfio::file_handle::flag::none);

    if (!result)
      throw std::runtime_error(
        std::string("MmapDistanceMatrix: failed to create cache exclusively; "
                    "the path must not already exist and another creator may "
                    "have won the race: ")
        + result.error().message());

    mfh_ = std::move(result.value());
    acquire_session_lease(mfh_, "create");
    owns_session_lease_ = true;
    mfh_.truncate(total).value();
    mfh_.update_map().value();

    auto *base = reinterpret_cast<uint8_t *>(mfh_.address());
    if (!base)
      throw std::runtime_error("MmapDistanceMatrix: null address after mapping");

    validate_mapping_alignment(base, layout);
    write_header(base, n, fingerprint, publication_state_initializing);
    data_ = reinterpret_cast<double *>(base + header_size);
    row_digests_ = reinterpret_cast<std::uint64_t *>(base + layout.digest_offset);
    n_ = n;
    fingerprint_ = fingerprint;

    if (layout.digest_word_count != 0) {
      std::atomic_ref<std::uint64_t> first_digest(row_digests_[0]);
      if (!first_digest.is_lock_free()) {
        throw std::runtime_error(
          "MmapDistanceMatrix: platform lacks lock-free 64-bit payload digest atomics");
      }
    }
    initialize_payload(data_, row_digests_, n);

    // Crash-consistent two-phase publication. The first blocking barrier makes
    // file metadata, every canonical NaN sentinel, and every initial row digest
    // durable while the CRC-valid header still says "initializing". Only then
    // may a ready header be published. A crash before/during the second barrier
    // yields either initializing, a CRC mismatch, or a payload mismatch.
    mfh_.barrier({}, llfio::mapped_file_handle::barrier_kind::wait_all).value();
    publish_ready_header(base);
    persist_range(mfh_, base, 0, header_size,
                  llfio::mapped_file_handle::barrier_kind::wait_all);
  }

  /// Open an existing memory-mapped distance matrix (warm-start).
  /// This overload is for matrices constructed without a semantic identity;
  /// it still verifies the explicit all-zero unbound fingerprint. Problem
  /// caches always use the expected-fingerprint overload below.
  static MmapDistanceMatrix open(const std::filesystem::path &cache_path)
  {
    return open_impl(cache_path, unbound_fingerprint);
  }

  /// Open a warm-start cache and require its semantic identity to match before
  /// returning any access to the computed-bit region.
  static MmapDistanceMatrix open(const std::filesystem::path &cache_path,
                                 const fingerprint_type &expected_fingerprint)
  {
    return open_impl(cache_path, expected_fingerprint);
  }
#endif // DTWC_HAS_MMAP

  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return data_[tri_index(i, j)];
  }

  /// Set distance. Parallel fills may use disjoint (i,j) pairs; per-row digest
  /// words use lock-free atomic XOR. Same-cell concurrent writes are unsupported.
  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_ && !std::isnan(v));
    const size_t row = std::max(i, j);
    const size_t index = tri_index(i, j);
    const std::uint64_t old_bits = std::bit_cast<std::uint64_t>(data_[index]);
    const std::uint64_t new_bits = std::bit_cast<std::uint64_t>(v);
    data_[index] = v;
    if (old_bits == new_bits) return;

    // Same-cell concurrency is outside the class contract. Disjoint cells may
    // share a row digest, so each 64-bit XOR lane is updated atomically. A crash
    // may persist the cell and lanes in any order. During a one-lane persistence
    // window, accidental corruption detection is at least 64-bit; once both
    // lanes are durable, stable corruption must collide in both domain-separated
    // lanes. Reopen recomputes read-only and rejects disagreement before exposure.
    std::atomic_thread_fence(std::memory_order_release);
    xor_digest_delta(row, index, old_bits, new_bits, 0);
    xor_digest_delta(row, index, old_bits, new_bits, 1);
  }

  bool is_computed(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return !std::isnan(data_[tri_index(i, j)]);
  }

  size_t size() const { return n_; }
  const fingerprint_type &fingerprint() const { return fingerprint_; }

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

  /// Mutable packed storage access bypasses set() and therefore does not update
  /// row digests. It exists for API compatibility and inspection; writing
  /// through this pointer intentionally makes the next reopen reject the cache
  /// for payload-integrity mismatch. Use set() for supported mutations.
  double *raw() { return data_; }
  const double *raw() const { return data_; }
  size_t packed_count() const { return packed_size(n_); }

#ifdef DTWC_HAS_MMAP
  /// Flush mapped memory to disk.
  void sync()
  {
    mfh_.barrier({}, llfio::mapped_file_handle::barrier_kind::wait_data_only).value();
  }
#endif
};

} // namespace dtwc::core
