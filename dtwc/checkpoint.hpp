/**
 * @file checkpoint.hpp
 * @brief Save/resume checkpointing for distance matrix computation.
 *
 * @details For large datasets, fillDistanceMatrix() can take hours.
 * These functions allow saving a (possibly partial) distance matrix
 * to disk and resuming later, avoiding re-computation of already
 * computed pairs.
 *
 * Dense checkpoint v2 publishes immutable generations:
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

#include <string>
#include <filesystem>

namespace dtwc {

// Forward declaration
class Problem;

/// Options controlling automatic checkpoint behavior.
struct CheckpointOptions {
  std::string directory = "./checkpoints";  ///< Directory to save checkpoint files.
  int save_interval = 100;                  ///< Save every N pairs computed (reserved for future use).
  bool enabled = false;                     ///< Whether checkpointing is enabled.
};

/// Save the current distance matrix state to a checkpoint directory.
///
/// Validates the complete source before filesystem effects, streams a new
/// immutable generation, and atomically replaces CURRENT. Existing active
/// generations are never overwritten. Unverifiable legacy direct-file
/// directories are upgraded only by a successful save.
///
/// @param prob  The Problem whose distance matrix to save.
/// @param path  Directory path for checkpoint files.
/// @throws std::runtime_error if files cannot be written.
void save_checkpoint(const Problem &prob, const std::string &path);

/// Load a checkpoint and restore the distance matrix into the Problem.
///
/// Validates CURRENT, the exact seven-key v2 manifest, the full Problem
/// data/configuration identity, payload digest, CSV shape, finite full-token
/// numbers, bit-identical symmetry, and pair count. Parsing occurs into a local
/// candidate and publishes with one non-throwing move only after every check.
/// Legacy direct-file directories are rejected because their data identity is
/// unverifiable.
///
/// @param prob  The Problem to restore the distance matrix into.
/// @param path  Directory path containing checkpoint files.
/// @return true if checkpoint was loaded successfully; false without changing
///         Problem state if it is absent, incompatible, or malformed.
bool load_checkpoint(Problem &prob, const std::string &path);

// ---- Binary checkpoint for ClusteringResult --------------------------------

/// Save clustering result to a compact binary file.
///
/// Binary format (little-endian):
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
/// @throws std::runtime_error if the file cannot be written.
void save_binary_checkpoint(const core::ClusteringResult &result,
                            const std::filesystem::path &path);

/// Load clustering result from a binary checkpoint file.
///
/// Validates the magic bytes and version. Returns false if the file
/// does not exist or has an invalid header.
///
/// @param result  The ClusteringResult to populate.
/// @param path    File path of the binary checkpoint.
/// @return true if loaded successfully, false otherwise.
bool load_binary_checkpoint(core::ClusteringResult &result,
                            const std::filesystem::path &path);

} // namespace dtwc
