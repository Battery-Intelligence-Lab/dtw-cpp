/**
 * @file checkpoint.cpp
 * @brief Checkpoint save and load: a Problem's matrix as one `.dtwm` file.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include "checkpoint.hpp"
#include "Problem.hpp"
#include "base/error.hpp"
#include "fileOperations.hpp" // utf8_to_path

#include <string>
#include <system_error>
#include <utility>

namespace dtwc {

namespace fs = std::filesystem;

fs::path checkpoint_path(const Problem &prob, const std::string &directory)
{
  const std::string &name = prob.name();
  return utf8_to_path(directory) / utf8_to_path((name.empty() ? std::string("distances") : name) + ".dtwm");
}

void save_checkpoint(const Problem &prob, const std::string &path)
{
  save_checkpoint(prob, path, prob.metric());
}

void save_checkpoint(const Problem &prob, const std::string &path, core::MetricType metric)
{
  // Every check before the filesystem is touched.
  const std::size_t n = prob.size();
  if (n == 0) throw InvalidInput("save_checkpoint: the Problem holds no series.");
  const auto identity = prob.distance_checkpoint_identity(metric);
  const core::DistanceMatrix &matrix = prob.distance_matrix();
  if (matrix.size() != 0 && matrix.size() != n)
    throw InvalidInput("save_checkpoint: the distance matrix has " + std::to_string(matrix.size())
                       + " rows, but the Problem holds " + std::to_string(n) + " series.");

  const fs::path file = checkpoint_path(prob, path);
  std::error_code ec;
  fs::create_directories(file.parent_path(), ec); // an existing directory is no error
  if (ec) throw IOError("Cannot create the checkpoint directory '" + path + "': " + ec.message());
  // A matrix not yet allocated has no pair computed: the fill allocates it.
  if (matrix.size() == n)
    matrix.write(file, identity);
  else
    core::DistanceMatrix(n).write(file, identity);
}

bool load_checkpoint(Problem &prob, const std::string &path)
{
  return load_checkpoint(prob, path, prob.metric());
}

bool load_checkpoint(Problem &prob, const std::string &path, core::MetricType metric)
{
  // A raw edit to a distance setting since the last bind would discard the
  // loaded matrix at the next lookup, a silent recompute: refuse it here.
  prob.validate_mmap_cache_identity();
  prob.validate_dense_cache_configuration();
  const auto identity = prob.distance_checkpoint_identity(metric);
  const fs::path file = checkpoint_path(prob, path);
  std::error_code ec;
  const bool exists = fs::exists(file, ec);
  if (ec) throw IOError("Cannot inspect the checkpoint in '" + path + "': " + ec.message());
  if (!exists) return false;

  prob.distMat = core::DistanceMatrix::read(file, prob.size(), identity);
  prob.filled_ = prob.distMat.size() > 0 && prob.distMat.all_computed();
  prob.clear_mmap_cache_identity();
  prob.fill_request_validated_ = false; // other pairs known: re-check lazily
  return true;
}

} // namespace dtwc
