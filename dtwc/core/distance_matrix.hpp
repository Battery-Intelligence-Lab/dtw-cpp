/**
 * @file distance_matrix.hpp
 * @brief The pairwise distance matrix: a packed lower triangle on the heap or in
 *        a memory-mapped `.dtwm` file.
 *
 * @details One layout in memory and on disk: N(N+1)/2 doubles, row i holding
 * columns 0..i, NaN = not computed (Soft-DTW can return a negative distance, so
 * no finite sentinel works). get, set and is_computed index a raw `double *`
 * whichever storage holds the doubles; the storage is chosen once, when the
 * matrix is made.
 *
 * A `.dtwm` file is a 48-byte header and the packed doubles, in the host's byte
 * order (every supported platform is little-endian):
 *
 *   bytes  0-3   magic "DTWM"
 *   bytes  4-7   version (uint32) = 4; versions 1-3 were earlier cache layouts
 *   bytes  8-15  N (uint64)
 *   bytes 16-47  SHA-256 fingerprint of the data and the distance settings
 *   bytes 48-    double[N(N+1)/2]
 *
 * distance_matrix.cpp, the one translation unit that includes llfio, owns the
 * format: map() (memory-mapped, needs llfio) and read() / write() (plain
 * streams, every build).
 *
 * No locks and no atomics: parallel fills write disjoint (i, j) cells, and a
 * read never writes.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "../base/names.hpp"
#include "../base/settings.hpp" // index_t
#include "storage.hpp"          // Precision

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace dtwc::core {

/// How a distance-matrix file marks a pair it never computed. NaN: an empty field
/// or a NaN spelling, as this library writes it. Negative: also every value below
/// zero, as v1.0.0 wrote -1 (-0 is not below zero: it stays the distance 0).
enum class NotComputed
{
  NaN,
  Negative
};

inline constexpr Name<NotComputed> not_computed_names[]{
  { "nan", NotComputed::NaN },
  { "negative", NotComputed::Negative },
};

/// Map (i,j) to a packed lower-triangular index (symmetric: tri_index(i,j)==tri_index(j,i)).
inline size_t tri_index(size_t i, size_t j)
{
  if (i < j) std::swap(i, j);
  return i * (i + 1) / 2 + j;
}

/// Number of elements in a packed lower-triangular matrix of dimension n.
inline size_t packed_size(size_t n) { return n * (n + 1) / 2; }

/// Bytes a distance takes at `width`: 8 for float64, 4 for float32.
inline constexpr size_t value_bytes(Precision width) { return width == Precision::Float32 ? sizeof(float) : sizeof(double); }

/// Rows for_each_point reads together by default: one 64-byte line of a packed row of doubles.
inline constexpr size_t group_rows = 8;

/// The symmetric rows a, a + 1, ..., a + W - 1 of a packed matrix of n series, point by point: f(o, v) for o = 0, 1,
/// ..., n - 1 in order, with v[g] = D(a + g, o). A row a + g past n - 1 repeats row n - 1; its values are the caller's
/// to drop. For o < a the values come from W packed rows, for o >= a + W from W neighbours in packed row o (eight: one
/// 64-byte line), where a scan of one row reads a line per point.
/// Each value is read as a lookup of one pair reads it (Problem::dist_by_ind: index_t indices converted at tri_index):
/// a compiler that reassociates (-fassociative-math) orders a sum's operands by how each was computed, so f's sums
/// group as a one-pair loop's do. GCC 13 grouped the swap's sums differently when the values came through a running
/// pointer, and when the point index was a size_t, on which it split the loop at the first row's tri_index.
/// A float32 matrix's values are widened as they are read: f sees doubles, and its sums stay in double.
template <size_t W = group_rows, typename T, typename F>
inline void for_each_point(const T *packed, index_t n, index_t a, F &&f)
{
  index_t rows[W];
  for (size_t g = 0; g < W; ++g) rows[g] = std::min(a + static_cast<index_t>(g), n - 1);
  for (index_t o = 0; o < n; ++o) {
    double v[W];
    for (size_t g = 0; g < W; ++g)
      v[g] = packed[tri_index(static_cast<size_t>(rows[g]), static_cast<size_t>(o))];
    f(o, static_cast<const double *>(v));
  }
}

/// The memory this process may use now, in bytes: the physical memory available,
/// or a Linux cgroup v2 limit when lower (what a SLURM job's --mem sets); 0 when
/// unknown, as in a build without llfio.
size_t available_memory();

/// The refusal of a distance a float32 matrix cannot hold: InvalidInput, after `where`, naming the pair, `distance`
/// (computed in double) and the fix, matrix_dtype float64; a distance that is itself ±inf is no number at any width.
[[noreturn]] void refuse_float32_range(std::string_view where, size_t i, size_t j, double distance);

/// The values are float64 or float32 (Problem's `matrix_dtype`, chosen when the matrix is made; a resize keeps it).
/// get() widens what it reads; set() rounds what it stores to the nearest float, where a distance of 2^128 - 2^103
/// (about 3.4e38) or more becomes inf; visit() hands a loop the values at their width.
///
/// The `.dtwm` layout of version 5, which every write produces: bytes 0-47 as the file header above, with version 5,
/// then bytes 48-51 the bytes of a value (uint32: 8 binary64, 4 binary32), 52-55 zero, 56-63 the rows written R
/// (uint64), and the values from byte 64. Rows from R on were never written and hold no distance: a new mapped file
/// has R = 0 and no NaN written, and a fill records R once its rows are on the device. A version-4 file (48 bytes of
/// header, binary64, every row written) is read and mapped as float64, and a mapped one stays version 4.
class DistanceMatrix
{
public:
  /// SHA-256 of what the distances were computed from (Problem::distance_checkpoint_identity).
  using fingerprint_type = std::array<std::uint8_t, 32>;

  DistanceMatrix() noexcept; ///< empty, float64
  /// n x n on the heap, every entry NaN, its values `width` wide.
  explicit DistanceMatrix(size_t n, Precision width = Precision::Float64);
  ~DistanceMatrix();
  DistanceMatrix(DistanceMatrix &&other) noexcept;
  DistanceMatrix &operator=(DistanceMatrix &&other) noexcept;
  DistanceMatrix(const DistanceMatrix &) = delete;
  DistanceMatrix &operator=(const DistanceMatrix &) = delete;

  /// Map `path` as the matrix of n series computed under `fingerprint`, its values `width` wide: an existing file
  /// is opened, an absent one is created with no row written. Writes reach the file through the page cache.
  /// @throws InvalidInput if the file holds another N, fingerprint or width; IOError if it is not a whole `.dtwm`
  ///         file, and on a build without llfio.
  static DistanceMatrix map(const std::filesystem::path &path, size_t n, const fingerprint_type &fingerprint,
                            Precision width = Precision::Float64);
  /// Read a `.dtwm` file into a heap matrix, with map()'s checks and errors; its rows never written read as NaN.
  static DistanceMatrix read(const std::filesystem::path &path, size_t n, const fingerprint_type &fingerprint,
                             Precision width = Precision::Float64);
  /// Write this matrix as a `.dtwm` file: to `path` + ".tmp", flushed to the
  /// device, then renamed over `path`. A matrix mapped to `path` is flushed in
  /// place instead. @throws IOError
  void write(const std::filesystem::path &path, const fingerprint_type &fingerprint) const;
  /// Write packed rows [r0, r1), the rows changed since the last write to `path`:
  /// when `path` holds this matrix's header (N, fingerprint, width and rows written), only their bytes,
  /// in place, flushed to the device; a matrix mapped to `path` flushes them.
  /// Otherwise the whole matrix, as write(path, fingerprint). Every entry of the
  /// file stays NaN or its distance, whenever a write stops. @throws IOError
  void write(const std::filesystem::path &path, const fingerprint_type &fingerprint, size_t r0, size_t r1);
  /// Start reading packed rows [r0, r1) of a mapped matrix into memory, those written; a hint,
  /// so a heap matrix, and a failure, change nothing.
  void prefetch_rows(size_t r0, size_t r1) const;
  /// Write packed rows [r0, r1) of a mapped matrix back to its file: started,
  /// or with `wait` on the device when this returns. A heap matrix has nothing
  /// to write. A wait that leaves every row before r1 on the device records them written, in the file's
  /// header too. @throws IOError
  void flush_rows(size_t r0, size_t r1, bool wait);
  /// NaN in packed rows [r0, r1), written by the OpenMP threads (first touch where they run).
  void clear_rows(size_t r0, size_t r1);
  /// The first entry of packed rows [r0, r1) that is ±inf, as (i, j) with j <= i; (n, n) when none is.
  std::pair<size_t, size_t> first_infinite(size_t r0, size_t r1) const;

  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    const size_t k = tri_index(i, j);
    return f64_ ? f64_[k] : f32_[k];
  }

  /// Parallel fills write disjoint (i,j) pairs; no locking is needed.
  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_ && !std::isnan(v));
    const size_t k = tri_index(i, j);
    if (f64_)
      f64_[k] = v;
    else
      f32_[k] = static_cast<float>(v);
  }

  bool is_computed(size_t i, size_t j) const { return !std::isnan(get(i, j)); }

  /// f(values): the packed values at their width, `const double *` or `const float *`, for a loop that reads many.
  template <typename F>
  decltype(auto) visit(F &&f) const
  {
    if (f64_) return f(static_cast<const double *>(f64_));
    return f(static_cast<const float *>(f32_));
  }
  /// f(values) with `double *` or `float *`, for a loop that writes them.
  template <typename F>
  decltype(auto) visit(F &&f)
  {
    if (f64_) return f(f64_);
    return f(f32_);
  }

  size_t size() const noexcept { return n_; }
  size_t packed_count() const noexcept { return packed_size(n_); }
  Precision width() const noexcept { return width_; }
  /// Rows [0, rows_written()) hold NaN or a distance: n, but for a mapped matrix whose file's rows from there on
  /// were never written (a new file, a fill that stopped), which every reader here takes as not computed. A get()
  /// or visit() of such a row reads what the file holds there, 0 in a new file.
  size_t rows_written() const noexcept { return rows_; }
  bool is_mapped() const noexcept { return mapping_ != nullptr; }

  /// Become an n x n heap matrix of the same width, every entry NaN. A mapped matrix lets go of
  /// its file and leaves it as it is.
  void resize(size_t n);
  /// Flush a mapped matrix's values to its file; a heap matrix has nothing to flush.
  void sync() const;

  double max() const; ///< Largest computed distance; 0 when none is computed.
  size_t count_computed() const;
  /// Whether every pair is computed (none is NaN, every row written), in one pass that also refuses
  /// a distance that is ±inf, and one below zero unless `negative_is_distance`
  /// (Soft-DTW's can be): InvalidInput, after `where`, names the first such pair,
  /// and for a negative one ends with `negative_way_out`, what the caller's route
  /// offers. Every matrix that enters a Problem from outside its fill passes this
  /// scan, so the clustering loops read the matrix unchecked.
  bool all_computed(std::string_view where, bool negative_is_distance, std::string_view negative_way_out = {}) const;

private:
  struct Mapping; // the llfio file handle and its map, in distance_matrix.cpp

  void point_at(std::byte *values) noexcept; ///< f64_ or f32_, by width_
  size_t row_offset(size_t r) const noexcept; ///< the byte offset of packed row r in this matrix's file

  std::unique_ptr<std::byte[]> heap_; ///< allocated uninitialised, then NaN-filled in parallel
  std::unique_ptr<Mapping> mapping_;
  double *f64_{ nullptr }; ///< the values of a float64 matrix: heap_ or the map
  float *f32_{ nullptr };  ///< the values of a float32 matrix
  size_t n_{ 0 };
  size_t rows_{ 0 }; ///< rows written
  Precision width_{ Precision::Float64 };
};

} // namespace dtwc::core
