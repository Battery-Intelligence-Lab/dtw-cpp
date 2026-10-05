/**
 * @file config.hpp
 * @brief dtwc::Config — every setting of one clustering run, keyed by the CLI long names —
 *        and apply(), which hands its clustering settings to a Problem.
 *
 * @details The command line, a TOML or YAML config file, Python keywords and
 * MATLAB name-value pairs are four spellings of one Config. Its member
 * initialisers are dtwc_cl's defaults. The settings that describe the
 * clustering (distance, method, solver, device) reach a Problem through
 * apply(); the file options (input, output, checkpoint, ...) are run()'s, in
 * cli/run.hpp, and the text forms (cli::bind, parse_config, to_config_text)
 * are the CLI's, in cli/config.hpp.
 *
 * @date 24 Sep 2026
 */

#pragma once

#include "Problem.hpp" // MIPSettings
#include "algorithms/hierarchical.hpp"
#include "base/env.hpp"
#include "base/settings.hpp"
#include "core/dtw_options.hpp"
#include "core/storage.hpp"
#include "enums/Method.hpp"
#include "enums/Solver.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace dtwc {

struct Config
{
  // Input and storage
  std::string input;                     ///< `--input`: CSV, Parquet, Arrow IPC, or a folder.
  std::string column;                    ///< `--column`: Parquet column holding the series.
  index_t skip_rows = 0;                 ///< `--skip-rows`
  index_t skip_cols = 0;                 ///< `--skip-cols`
  char delimiter = '\0';                 ///< `--delimiter`; '\0' infers it from the extension.
  core::Precision dtype = core::Precision::Float64; ///< `--dtype`
  std::size_t ram_limit = 0;             ///< `--ram-limit` in bytes; 0 = no limit.
  std::size_t mmap_threshold = 50000;    ///< `--mmap-threshold`
  std::string dist_matrix;               ///< `--dist-matrix`: precomputed distance-matrix CSV.
  // Method
  Method method = Method::Auto;          ///< `--method`
  index_t k = 0;                         ///< `--n-clusters`, required: 0 = not given
  int max_iter = 100;                    ///< `--max-iter`
  int n_init = 1;                        ///< `--n-init`
  std::uint64_t seed = settings::DEFAULT_RANDOM_SEED; ///< `--seed`
  index_t sample_size = -1;              ///< `--sample-size` (CLARA; -1 = auto)
  int n_samples = 5;                     ///< `--n-samples` (CLARA)
  index_t batch_size = -1;               ///< `--batch-size` (OneBatchPAM; -1 = auto)
  algorithms::Linkage linkage = algorithms::Linkage::Average; ///< `--linkage`
  double tadpole_dc = -1.0;              ///< `--dc` (TADPole; -1 = auto)
  // Distance
  int band = -1;                         ///< `--band` (-1 = full DTW)
  core::MetricType metric = core::MetricType::L1; ///< `--metric`
  core::DTWVariantParams variant;        ///< `--variant`, `--mv-mode` and the variant parameters
  core::MissingStrategy missing = core::MissingStrategy::Error; ///< `--missing-strategy`
  // Device
  Device device = Device::CPU;           ///< `--device`
  int device_index = 0;                  ///< N of `--device gpu:N`
  GpuPrecision gpu_precision = GpuPrecision::Auto; ///< `--gpu-precision`
  // Solver
  Solver solver = Solver::HiGHS;         ///< `--solver`
  MIPSettings mip;                       ///< `--mip-gap`, `--time-limit`, `--lr-max-nodes`, ...
  // Checkpoint
  std::string checkpoint;                ///< `--checkpoint` directory
  int checkpoint_interval = 0;           ///< `--checkpoint-interval` (0 = at the end only)
  // Output
  std::string output = "./results";      ///< `--output` ("" = write nothing)
  std::string name;                      ///< `--name` ("" = the input's file or folder name)
  bool verbose = false;                  ///< `--verbose`
};

/// `device` as to_config_text() writes it and detail::parse_device() reads it
/// back: cpu, gpu, gpu:N (N the GPU index when it is not 0).
inline std::string device_text(const Config &config)
{
  std::string text = to_string(config.device);
  if (config.device == Device::GPU && config.device_index != 0) text += ':' + std::to_string(config.device_index);
  return text;
}

/// Hand `prob` the settings of `config` that describe the clustering: the
/// distance, the method and its controls, the MIP solver and settings, the
/// device and verbose. The Problem checks each as it takes it; `name` and the
/// file options (input, output, checkpoint, dtype, ...) are not read.
/// @throws InvalidInput for a value outside its domain or a distance no kernel
///         implements; SolverError for gurobi on a build without Gurobi;
///         DeviceError for gpu on a build without a GPU backend, and for
///         onebatch or tadpole on a GPU (they compute on the CPU as they go).
void apply(const Config &config, Problem &prob);

} // namespace dtwc
