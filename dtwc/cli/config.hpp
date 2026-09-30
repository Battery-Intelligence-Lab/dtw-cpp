/**
 * @file config.hpp
 * @brief dtwc::Config — every setting of one clustering run, keyed by the CLI long names.
 *
 * @details The command line, a TOML or YAML config file, Python keywords and
 * MATLAB name-value pairs are four spellings of one Config. Its member
 * initialisers are dtwc_cl's defaults. cli::bind() is the one key table: each key
 * is the CLI long name without "--", nested fields keep flat keys (`wdtw-g` sets
 * `variant.wdtw_g`), and enums are read through the Name tables beside them.
 * to_config_text() and parse_config() are built on bind(), so no second field
 * list exists. Settings are only read here; run-time checks belong to the run.
 *
 * @date 24 Sep 2026
 */

#pragma once

#include "../Problem.hpp" // CUDASettings, MIPSettings
#include "../algorithms/hierarchical.hpp"
#include "../algorithms/one_batch_pam.hpp"
#include "../base/env.hpp"
#include "../base/names.hpp"
#include "../base/settings.hpp"
#include "../core/dtw_options.hpp"
#include "../core/storage.hpp"
#include "../enums/Solver.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace CLI {
class App;
}

namespace dtwc {

/// The algorithm `--method` selects. Method keeps the four values Problem::cluster()
/// dispatches; this enum spells what a run can be asked for.
enum class ClusterMethod { Auto, PAM, OneBatch, CLARA, Kmedoids, MIP, LRCore, TADPole, Hierarchical };

inline constexpr Name<ClusterMethod> cluster_method_names[]{
  { "auto", ClusterMethod::Auto },
  { "pam", ClusterMethod::PAM },
  { "onebatch", ClusterMethod::OneBatch },
  { "obp", ClusterMethod::OneBatch },
  { "clara", ClusterMethod::CLARA },
  { "kmedoids", ClusterMethod::Kmedoids },
  { "mip", ClusterMethod::MIP },
  { "lrcore", ClusterMethod::LRCore },
  { "lr", ClusterMethod::LRCore },
  { "tadpole", ClusterMethod::TADPole },
  { "hierarchical", ClusterMethod::Hierarchical },
  { "hclust", ClusterMethod::Hierarchical },
};

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
  ClusterMethod method = ClusterMethod::Auto; ///< `--method`
  index_t k = 3;                         ///< `--n-clusters`
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
  CUDASettings gpu;                      ///< device_id from `--device gpu:N`; `--gpu-precision`
  // Solver
  Solver solver = Solver::HiGHS;         ///< `--solver`
  MIPSettings mip;                       ///< `--mip-gap`, `--time-limit`, `--lr-max-nodes`, ...
  // Checkpoint
  std::string checkpoint;                ///< `--checkpoint` directory
  int checkpoint_interval = 0;           ///< `--checkpoint-interval` (0 = at the end only)
  // Output
  std::string output = "./results";      ///< `--output` ("" = write nothing)
  std::string name = "dtwc";             ///< `--name`
  bool verbose = false;                  ///< `--verbose`
};

namespace cli {

/// Adds every Config key to `app`, bound to `config`, plus `--config <file>`
/// (TOML or YAML, the same keys; flags beat the file; an unknown key is an error).
/// `--help` shows `config`'s values as the defaults. `--clusters` stays a hidden
/// spelling that warns on stderr and yields to `--n-clusters`.
/// A value no spelling reads raises during the parse: CLI11's error for a bad
/// choice or number, InvalidInput for `--ram-limit` / `--delimiter`, DeviceError
/// for `--device`.
void bind(CLI::App &app, Config &config);

} // namespace cli

/// Every key of `config`, one `key = value` line each in bind() order: enums by
/// canonical name, doubles in shortest round-trip form. parse_config() and
/// `--config` read it back to an equal Config.
/// @throws InvalidInput when a field holds a value no name spells (MetricType::L2).
std::string to_config_text(const Config &config);

/// `device` as to_config_text() writes it and detail::parse_device() reads it
/// back: cpu, gpu, gpu:N (N the GPU index when it is not 0).
std::string device_text(const Config &config);

/// A Config from (key, value) pairs, as Python keywords and MATLAB name-value
/// pairs give them: `_` in a key reads as `-` (`max_iter` is `max-iter`), a
/// one-letter key is the short flag (`k` is `n-clusters`); the pairs are read as
/// a config file, so the keys, spellings and precedence are bind()'s.
/// @throws InvalidInput for an unknown key or a value CLI11 rejects.
Config parse_config(const std::vector<std::pair<std::string, std::string>> &pairs);

} // namespace dtwc
