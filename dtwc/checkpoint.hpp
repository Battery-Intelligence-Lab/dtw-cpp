/**
 * @file checkpoint.hpp
 * @brief Save and resume a Problem's distance matrix as one `.dtwm` file.
 *
 * @details A checkpoint is the file checkpoint_path() names: the layout of a
 * mapped matrix (core/distance_matrix.hpp), so a checkpoint can be mapped with
 * Problem::use_mmap_distance_matrix and a mapped matrix loaded as a checkpoint.
 * A Problem whose matrix is mapped to that file is its own checkpoint: saving it
 * flushes the mapping in place.
 *
 * Saving is explicit (save_checkpoint) or automatic through Problem::checkpoint
 * (CheckpointOptions): with `enabled`, fill_distance_matrix() saves after every
 * `save_interval` completed matrix rows, on the calling thread, after the row
 * block has joined. A save that throws propagates out of the fill; the distances
 * computed so far stay in the Problem and the previous file stays whole.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#pragma once

#include "core/dtw_options.hpp"

#include <filesystem>
#include <string>

namespace dtwc {

class Problem;

/// Options controlling automatic checkpointing, read by
/// Problem::fill_distance_matrix() through Problem::checkpoint. With `enabled`
/// the fill runs the BruteForce row schedule in blocks of `save_interval` rows
/// and saves after each block, the last included, so a completed fill leaves a
/// complete checkpoint. `enabled` with `save_interval < 1` or an empty
/// `directory` is InvalidInput before any distance is computed.
struct CheckpointOptions {
  std::string directory = "./checkpoints"; ///< Directory of the checkpoint file.
  /// Completed matrix rows between automatic saves (>= 1). A save of a matrix in
  /// RAM writes all N(N+1)/2 doubles, so a fill writes O(N^3 / save_interval)
  /// bytes in total; a mapped matrix is flushed in place instead.
  int save_interval = 100;
  bool enabled = false; ///< Whether automatic checkpointing is enabled.
};

/// The checkpoint file of `prob` in `directory`: `<directory>/<name>.dtwm`,
/// "distances" standing in for an empty name. Both strings are UTF-8.
std::filesystem::path checkpoint_path(const Problem &prob, const std::string &directory);

/// Save the Problem's distance matrix to checkpoint_path(prob, path): written to
/// a ".tmp" file, flushed to the device, then renamed over any previous
/// checkpoint. The directory is created if missing. The identity in the file
/// includes prob.metric(): an L1 and a SquaredL2 matrix of the same data differ.
/// @throws IOError if the directory or the file cannot be written; InvalidInput
///         if the Problem holds no series, or a matrix of another size.
void save_checkpoint(const Problem &prob, const std::string &path);

/// As above, tagged with `metric`, the pointwise metric the stored distances
/// were computed with. It may differ from prob.metric() only for a matrix a
/// producer outside this Problem filled; prefer the two-argument form.
void save_checkpoint(const Problem &prob, const std::string &path, core::MetricType metric);

/// Read checkpoint_path(prob, path) into the Problem's distance matrix, in RAM
/// (a mapped matrix is let go of; its file is left as it is).
/// @return false, with the Problem unchanged, only when there is no such file.
/// @throws InvalidInput if the file holds distances of other series or other
///         distance settings, prob.metric() included, or a distance that is
///         ±inf; IOError if it is not a whole `.dtwm` file (short, foreign,
///         another version, wrong length).
[[nodiscard]] bool load_checkpoint(Problem &prob, const std::string &path);

/// As above, but expecting distances computed with `metric`, which may differ
/// from prob.metric() only for a matrix a producer outside this Problem fills;
/// prefer the two-argument form.
[[nodiscard]] bool load_checkpoint(Problem &prob, const std::string &path,
                                   core::MetricType metric);

} // namespace dtwc
