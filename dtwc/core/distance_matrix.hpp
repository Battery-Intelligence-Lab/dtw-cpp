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

/// Map (i,j) to a packed lower-triangular index (symmetric: tri_index(i,j)==tri_index(j,i)).
inline size_t tri_index(size_t i, size_t j)
{
  if (i < j) std::swap(i, j);
  return i * (i + 1) / 2 + j;
}

/// Number of elements in a packed lower-triangular matrix of dimension n.
inline size_t packed_size(size_t n) { return n * (n + 1) / 2; }

class DistanceMatrix
{
public:
  /// SHA-256 of what the distances were computed from (Problem::distance_checkpoint_identity).
  using fingerprint_type = std::array<std::uint8_t, 32>;

  DistanceMatrix() noexcept;
  explicit DistanceMatrix(size_t n); ///< n x n on the heap, every entry NaN.
  ~DistanceMatrix();
  DistanceMatrix(DistanceMatrix &&other) noexcept;
  DistanceMatrix &operator=(DistanceMatrix &&other) noexcept;
  DistanceMatrix(const DistanceMatrix &) = delete;
  DistanceMatrix &operator=(const DistanceMatrix &) = delete;

  /// Map `path` as the matrix of n series computed under `fingerprint`: an
  /// existing file is opened, an absent one is created with every entry NaN.
  /// Writes reach the file through the page cache.
  /// @throws InvalidInput if the file holds another N or fingerprint; IOError if
  ///         it is not a whole `.dtwm` file, and on a build without llfio.
  static DistanceMatrix map(const std::filesystem::path &path, size_t n,
                            const fingerprint_type &fingerprint);
  /// Read a `.dtwm` file into a heap matrix, with map()'s checks and errors.
  static DistanceMatrix read(const std::filesystem::path &path, size_t n,
                             const fingerprint_type &fingerprint);
  /// Write this matrix as a `.dtwm` file: to `path` + ".tmp", flushed to the
  /// device, then renamed over `path`. A matrix mapped to `path` is flushed in
  /// place instead. @throws IOError
  void write(const std::filesystem::path &path, const fingerprint_type &fingerprint) const;

  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return data_[tri_index(i, j)];
  }

  /// Parallel fills write disjoint (i,j) pairs; no locking is needed.
  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_ && !std::isnan(v));
    data_[tri_index(i, j)] = v;
  }

  bool is_computed(size_t i, size_t j) const { return !std::isnan(get(i, j)); }

  size_t size() const noexcept { return n_; }
  size_t packed_count() const noexcept { return packed_size(n_); }
  double *raw() noexcept { return data_; }
  const double *raw() const noexcept { return data_; }
  bool is_mapped() const noexcept { return mapping_ != nullptr; }

  /// Become an n x n heap matrix, every entry NaN. A mapped matrix lets go of
  /// its file and leaves it as it is.
  void resize(size_t n);
  /// Flush a mapped matrix's doubles to its file; a heap matrix has nothing to flush.
  void sync() const;

  double max() const; ///< Largest computed distance; 0 when none is computed.
  size_t count_computed() const;
  /// Whether every pair is computed (none is NaN), in one pass that also refuses
  /// a distance that is ±inf: InvalidInput, after `where`, names the first such
  /// pair. Every matrix that enters a Problem from outside its fill passes this
  /// scan, so the clustering loops read the matrix unchecked.
  bool all_computed(std::string_view where) const;

private:
  struct Mapping; // the llfio file handle and its map, in distance_matrix.cpp

  std::vector<double> heap_;
  std::unique_ptr<Mapping> mapping_;
  double *data_{ nullptr }; ///< heap_.data() or the mapped doubles
  size_t n_{ 0 };
};

} // namespace dtwc::core
