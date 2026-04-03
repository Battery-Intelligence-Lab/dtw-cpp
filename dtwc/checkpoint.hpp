/**
 * @file checkpoint.hpp
 * @brief Save/resume checkpointing for distance matrix computation.
 *
 * @details For large datasets, fillDistanceMatrix() can take hours.
 * These functions allow saving a (possibly partial) distance matrix
 * to disk and resuming later, avoiding re-computation of already
 * computed pairs.
 *
 * Checkpoint format (CSV + metadata text file, no extra dependencies):
 *   - distances.csv  -- the NxN distance matrix (NaN for uncomputed pairs)
 *   - metadata.txt   -- key=value pairs: n, band, variant, pairs_computed, timestamp
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#pragma once

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
/// Creates the directory if it does not exist. Writes:
///   - distances.csv: the full NxN matrix (NaN for uncomputed entries)
///   - metadata.txt: key=value metadata (n, band, variant, pairs_computed, timestamp)
///
/// @param prob  The Problem whose distance matrix to save.
/// @param path  Directory path for checkpoint files.
/// @throws std::runtime_error if files cannot be written.
void save_checkpoint(const Problem &prob, const std::string &path);

/// Load a checkpoint and restore the distance matrix into the Problem.
///
/// Reads distances.csv and metadata.txt from the given directory.
/// Validates that the matrix dimension matches the Problem's data size.
/// Sets is_distMat_filled to true only if all pairs are computed.
///
/// @param prob  The Problem to restore the distance matrix into.
/// @param path  Directory path containing checkpoint files.
/// @return true if checkpoint was loaded successfully, false if not found or invalid.
bool load_checkpoint(Problem &prob, const std::string &path);

} // namespace dtwc
