/**
 * @file checkpoint.hpp
 * @brief Save/resume checkpointing for distance matrix computation.
 *
 * @details For large datasets, fill_distance_matrix() can take hours. These
 * functions save a (possibly partial) distance matrix to disk and restore it
 * later, so pairs already computed are not recomputed.
 *
 * Saving is either explicit (call save_checkpoint()) or automatic. Automatic
 * saving is driven by Problem::checkpoint (a CheckpointOptions): with
 * `enabled`, fill_distance_matrix() saves a new generation after every
 * `save_interval` completed matrix rows, on the calling thread, after the
 * parallel row block has joined. Never call save_checkpoint() yourself from
 * inside a parallel region.
 *
 * If an automatic save throws, the exception propagates out of
 * fill_distance_matrix(): the distances computed so far stay in memory and the
 * previously published generation on disk remains valid and loadable.
 *
 * Dense checkpoint v2 publishes immutable generations. A directory holds
 * exactly one generation after a successful save: the old generation is removed
 * only once CURRENT points at the new one, so a reader never observes a
 * directory without a valid payload.
 *   - CURRENT -- one lowercase 64-hex generation identifier
 *   - generations/<id>/distances.csv -- exact full NxN matrix; an empty field
 *     is the only uncomputed representation
 *   - generations/<id>/metadata.txt -- strict version, dimension, pair count,
 *     UTC timestamp, full Problem identity, and payload SHA-256
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#pragma once

#include "core/clustering_result.hpp"
#include "core/dtw_options.hpp"
#include "error.hpp"

#include <string>
#include <filesystem>

namespace dtwc {

// Forward declaration
class Problem;

/// Options controlling automatic checkpoint behavior.
///
/// Consumed by Problem::fill_distance_matrix() through Problem::checkpoint.
/// With `enabled` the fill runs the BruteForce row schedule in consecutive
/// blocks of `save_interval` rows and publishes one generation after each
/// block, the last block included, so a completed fill leaves a complete
/// checkpoint. `enabled` requires dense distance storage and
/// `save_interval >= 1`; either violation is an InvalidInput raised before any
/// distance is computed.
struct CheckpointOptions {
  std::string directory = "./checkpoints";  ///< Directory to save checkpoint files.
  /// Completed matrix rows between automatic saves (>= 1). Every save writes the
  /// whole N-by-N CSV, so it costs O(N^2) bytes and time and a fill costs
  /// O(N^3 / save_interval) in total. Choose `save_interval` so a save is a small
  /// fraction of a block: a block costs about save_interval * N DTWs, a save
  /// about N^2 number formats.
  int save_interval = 100;
  bool enabled = false;                     ///< Whether automatic mid-fill checkpointing is enabled.
};

/// Save the current distance matrix state to a checkpoint directory.
///
/// Validates the complete source before filesystem effects, streams a new
/// immutable generation, and atomically replaces CURRENT. Existing active
/// generations are never overwritten. After CURRENT names the new generation,
/// every other generation directory is removed (best effort), so a directory
/// holds exactly one generation after a successful save. Unverifiable legacy
/// direct-file directories are upgraded only by a successful save.
///
/// @param prob   The Problem whose distance matrix to save.
/// @param path   Directory path for checkpoint files.
/// @param metric Pointwise metric the stored distances were computed with. It is
///        part of the identity fingerprint: an L1 and a SquaredL2 matrix over
///        the same data are different numbers.
/// @throws std::runtime_error if files cannot be written.
void save_checkpoint(const Problem &prob, const std::string &path,
                     core::MetricType metric = core::MetricType::L1);

/// Load a checkpoint and restore the distance matrix into the Problem.
///
/// Validates CURRENT, the exact seven-key v2 manifest, the full Problem
/// data/configuration identity, payload digest, CSV shape, finite full-token
/// numbers, bit-identical symmetry, and pair count. Parsing occurs into a local
/// candidate and publishes with one non-throwing move only after every check.
/// Legacy direct-file directories are rejected because their data identity is
/// unverifiable.
///
/// @param prob   The Problem to restore the distance matrix into.
/// @param path   Directory path containing checkpoint files.
/// @param metric Pointwise metric this run computes with; a checkpoint written
///        under a different metric no longer matches the identity fingerprint.
/// @return true if checkpoint was loaded successfully; false without changing
///         Problem state if it is absent, incompatible, or malformed.
bool load_checkpoint(Problem &prob, const std::string &path,
                     core::MetricType metric = core::MetricType::L1);

// ---- Binary checkpoint for ClusteringResult --------------------------------

/// Save clustering result to a compact binary file.
///
/// Binary format (strict little-endian):
///   bytes 0-3:   magic "DCKP"
///   bytes 4-5:   version uint16 = 1
///   bytes 6-7:   reserved (0)
///   bytes 8-11:  k (int32) -- number of medoids
///   bytes 12-15: N (int32) -- number of data points
///   bytes 16-19: iterations (int32)
///   byte  20:    converged (uint8, 0 or 1)
///   bytes 21-23: padding (0)
///   bytes 24-31: total_cost (double)
///   bytes 32+:   medoid_indices (k * int32)
///   then:        labels (N * int32)
///
/// @param result  The clustering result to save.
/// @param path    File path for the binary checkpoint.
/// @throws InvalidInput if a count or integer field is not representable by
///         the version-1 int32 wire format.
/// @throws IOError if the file or its parent directories cannot be written.
void save_binary_checkpoint(const core::ClusteringResult &result,
                            const std::filesystem::path &path);

/// Load clustering result from a binary checkpoint file.
///
/// Validates the complete header, canonical structural bytes, exact file
/// length, and payload before publishing a local candidate. Structural
/// validation deliberately does not impose clustering-semantic or provenance
/// policy. Returns false without changing @p result if the file is missing,
/// inaccessible, malformed, or cannot be decoded.
///
/// @param result  The ClusteringResult to populate.
/// @param path    File path of the binary checkpoint.
/// @return true if loaded successfully, false otherwise.
bool load_binary_checkpoint(core::ClusteringResult &result,
                            const std::filesystem::path &path);

} // namespace dtwc
