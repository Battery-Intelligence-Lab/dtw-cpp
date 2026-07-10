/**
 * @file dtwc_cl.cpp
 * @brief Command line interface for DTWC++ with TOML/YAML configuration support.
 *
 * @details Full CLI tool using CLI11 with TOML config file support and optional
 * YAML support via yaml-cpp. Supports PAM, CLARA, MIP, and hierarchical clustering
 * methods, all DTW variants, checkpointing, and flexible output.
 *
 * Usage:
 *   dtwc_cl --input data.csv -k 5 --method pam -v
 *   dtwc_cl --config config.toml
 *   dtwc_cl --config config.yaml   (requires -DDTWC_ENABLE_YAML=ON)
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 * @authors Volkan Kumtepeli
 * @authors Becky Perriment
 */

#include "dtwc.hpp"
#include "env.hpp"
#ifdef DTWC_HAS_MMAP
#include "core/mmap_data_store.hpp"
#endif

#ifdef DTWC_HAS_ARROW
#include "io/arrow_ipc_reader.hpp"
#endif
#ifdef DTWC_HAS_PARQUET
#include "io/parquet_reader.hpp"
#endif

// CLI11 is only used inside main(). Guard it (and main) behind DTWC_CL_NO_MAIN
// so the pure argument-parsing helpers below can be #included and unit-tested
// without linking CLI11 (see tests/unit/unit_test_cli_args.cpp, Task 0.9).
#ifndef DTWC_CL_NO_MAIN
#include <CLI/CLI.hpp>

#ifdef DTWC_HAS_YAML
#include <yaml-cpp/yaml.h>
#endif
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

/// Parse a human-readable size string like "2G", "500M", "128G" to bytes.
/// Returns 0 if parsing fails or string is empty.
static size_t parse_ram_limit(const std::string &s)
{
  if (s.empty()) return 0;
  char *end = nullptr;
  double val = std::strtod(s.c_str(), &end);
  if (end == s.c_str()) return 0;
  char suffix = (*end) ? static_cast<char>(std::toupper(static_cast<unsigned char>(*end))) : 'B';
  switch (suffix) {
  case 'T': return static_cast<size_t>(val * (1ULL << 40));
  case 'G': return static_cast<size_t>(val * (1ULL << 30));
  case 'M': return static_cast<size_t>(val * (1ULL << 20));
  case 'K': return static_cast<size_t>(val * (1ULL << 10));
  default:  return static_cast<size_t>(val);
  }
}

/// Parsed and validated `--device` specification.
struct DeviceSpec
{
  bool valid = true;      ///< false → parse/validate error (message in `error`).
  bool is_cuda = false;   ///< true if a CUDA device was requested.
  int cuda_id = 0;        ///< CUDA device ordinal (0 for a bare "cuda").
  std::string error;      ///< Actionable message; populated only when !valid.
};

/// Parse and validate a `--device` string (case-insensitive).
/// Accepts exactly: "cpu", "cuda", "cuda:N" (N a non-negative integer).
///
/// Task 0.9 / audit cli-ux fixes (all three were silent or fatal before):
///  - case-insensitive: "CUDA:0" no longer misses the case-sensitive
///    `rfind("cuda")` and silently fall back to CPU;
///  - an unknown device name returns valid=false (hard error) instead of a
///    silent CPU fallback (no-silent-fallback global constraint);
///  - "cuda:abc" / "cuda:" return valid=false instead of letting
///    `std::stoi(device.substr(5))` throw std::invalid_argument uncaught,
///    which propagated out of main() and called std::terminate.
static DeviceSpec parse_device(std::string device)
{
  for (auto &c : device)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  DeviceSpec spec;
  if (device == "cpu")
    return spec;
  if (device == "cuda") {
    spec.is_cuda = true;
    return spec;
  }
  if (device.rfind("cuda:", 0) == 0) {
    const std::string id = device.substr(5);
    if (id.empty() || id.find_first_not_of("0123456789") != std::string::npos) {
      spec.valid = false;
      spec.error = "Invalid CUDA device id in --device '" + device
                 + "': expected a non-negative integer after 'cuda:' (e.g. cuda:0).";
      return spec;
    }
    try {
      spec.cuda_id = std::stoi(id);
    } catch (const std::out_of_range &) {
      spec.valid = false;
      spec.error = "CUDA device id out of range in --device '" + device + "'.";
      return spec;
    }
    spec.is_cuda = true;
    return spec;
  }
  spec.valid = false;
  spec.error = "Unknown --device '" + device
             + "'. Valid devices: cpu, cuda, cuda:N (N a non-negative integer).";
  return spec;
}

/// Validate that the requested pointwise metric can be honoured on the chosen
/// device. Returns an empty string when OK, otherwise an actionable message.
///
/// Task 0.9 / audit cli-ux fix: `--metric` was consumed ONLY by the CUDA path
/// (use_squared_l2). On the CPU path resolve_dtw_fn() always binds
/// MetricType::L1, so a non-L1 metric was silently ignored (it computed L1).
/// Per the no-silent-fallback rule we reject it rather than quietly degrade.
static std::string validate_metric_for_device(const std::string &metric, bool is_cuda)
{
  if (!is_cuda && metric != "l1")
    return "metric '" + metric + "' is unsupported on the cpu path "
           "(only 'l1' is implemented on CPU; use --device cuda for '" + metric + "').";
  return "";
}

/// Run the CLI PAM route with the user-visible `--seed` on an invocation-local
/// engine. Keeping this seam outside main() lets the CLI regression exercise the
/// production dispatch without depending on CLI11 in the unit-test executable.
static dtwc::core::ClusteringResult run_cli_pam(
  dtwc::Problem &prob, int n_clusters, int max_iter, std::uint64_t random_seed)
{
  return dtwc::fast_pam_seeded(prob, n_clusters, random_seed, max_iter);
}

/// Apply the CLI's distance-matrix storage policy after data and `auto` method
/// resolution. Returns the selected mmap cache path, or nullopt when the method
/// stays on its own storage / below the threshold.
///
/// OneBatchPAM is the sole exemption: it owns a fixed O(Nm) table and never
/// calls Problem::dist_by_ind(). TADPole is deliberately NOT exempt. Although
/// its pruning schedule is matrix-free, every exact/fallback distance goes
/// through dist_by_ind(), whose dense cache lazily allocates packed O(N^2)
/// doubles. MmapDistanceMatrix implements that same lazy get/set interface, so
/// threshold routing is both compatible and necessary for large TADPole jobs.
static std::optional<fs::path> configure_cli_distance_storage(
  dtwc::Problem &prob,
  std::string_view method,
  size_t mmap_threshold,
  const fs::path &cache_path,
  dtwc::core::MetricType cache_metric = dtwc::core::MetricType::L1,
  bool legacy_checkpoint_requested = false,
  bool legacy_distance_matrix_requested = false)
{
  if (method == "onebatch") return std::nullopt;
  if (mmap_threshold != 0 && prob.size() < mmap_threshold) return std::nullopt;
  if (legacy_checkpoint_requested) {
    throw std::runtime_error(
      "--checkpoint uses the legacy dense CSV checkpoint format and cannot be "
      "combined with memory-mapped distance storage. The mmap cache already "
      "resumes automatically; omit --checkpoint, or raise --mmap-threshold if "
      "the dense matrix and CSV checkpoint fit in RAM.");
  }
  if (legacy_distance_matrix_requested) {
    throw std::runtime_error(
      "--dist-matrix uses a legacy dense CSV matrix and cannot be combined "
      "with memory-mapped distance storage. Omit --dist-matrix to resume the "
      "fingerprinted mmap cache, or raise --mmap-threshold if importing the "
      "dense CSV matrix fits in RAM.");
  }

#ifdef DTWC_HAS_MMAP
  prob.use_mmap_distance_matrix(cache_path, cache_metric);
  return cache_path;
#else
  throw std::runtime_error(
    "method='" + std::string(method) + "' at N=" + std::to_string(prob.size())
    + " requires memory-mapped distance storage because --mmap-threshold="
    + std::to_string(mmap_threshold)
    + " was reached, but this binary was built without mmap support. Rebuild "
      "with -DDTWC_ENABLE_LLFIO=ON, raise --mmap-threshold only if the packed "
      "heap matrix fits in RAM, or use --method onebatch.");
#endif
}

/// Convert float64 Data to float32 in-place.
static dtwc::Data convert_to_f32(dtwc::Data &&data_f64)
{
  const size_t n = data_f64.size();
  std::vector<std::vector<float>> vecs_f32(n);
  for (size_t i = 0; i < n; ++i) {
    const auto &src = data_f64.p_vec[i];
    vecs_f32[i].resize(src.size());
    for (size_t j = 0; j < src.size(); ++j)
      vecs_f32[i][j] = static_cast<float>(src[j]);
  }
  return dtwc::Data(std::move(vecs_f32), std::move(data_f64.p_names), data_f64.ndim);
}

/// Write cluster labels to CSV: one line per point with "name,cluster_id".
static void write_labels_csv(const fs::path &path,
                             const dtwc::Problem &prob,
                             const dtwc::core::ClusteringResult &result)
{
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot open output file: " + path.string());

  out << "name,cluster\n";
  for (size_t i = 0; i < result.labels.size(); ++i) {
    out << prob.get_name(i) << "," << result.labels[i] << "\n";
  }
}

/// Write medoid information to CSV.
static void write_medoids_csv(const fs::path &path,
                              const dtwc::Problem &prob,
                              const dtwc::core::ClusteringResult &result)
{
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot open output file: " + path.string());

  out << "cluster,medoid_index,medoid_name\n";
  for (int c = 0; c < result.n_clusters(); ++c) {
    int idx = result.medoid_indices[c];
    out << c << "," << idx << "," << prob.get_name(idx) << "\n";
  }
}

/// Write silhouette scores to CSV.
static void write_silhouettes_csv(const fs::path &path,
                                  const std::vector<double> &sil,
                                  const dtwc::Problem &prob,
                                  const dtwc::core::ClusteringResult &result)
{
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot open output file: " + path.string());

  out << "name,cluster,silhouette\n";
  for (size_t i = 0; i < sil.size(); ++i) {
    out << prob.get_name(i) << "," << result.labels[i] << ","
        << std::setprecision(8) << sil[i] << "\n";
  }
}

// ---------------------------------------------------------------------------
// CLI / TOML flag deprecation registry (api-contract-2.0.md §4, §7 item 3)
// ---------------------------------------------------------------------------
// The CLI flag set and the TOML/YAML config keys are a de-facto API (they are
// composed by `scripts/slurm/jobs/cluster_generic.slurm` and
// `python/dtwcpp/_hpc.py::build_dtwc_command`). When a flag/key is renamed to the
// 2.0 contract vocabulary the OLD spelling stays ACCEPTED but emits a one-line
// deprecation warning to stderr pointing at the new spelling (one warning per
// use). TOML/YAML keys are the long-flag names without the leading "--", so a
// single table below drives the CLI, the `--config` TOML path (CLI11 maps config
// keys onto the same options) and the `--yaml-config` path.
//
//   old CLI flag / TOML key   ->  new canonical CLI flag / TOML key    contract ref
//   --clusters  / clusters    ->  --n-clusters / n-clusters            §1.5, §2.1 (n_clusters)
//   --restart   / restart     ->  --resume / resume                    §2.7
//
// These helpers are intentionally CLI11-free so tests/unit/unit_test_cli_args.cpp
// (compiled with DTWC_CL_NO_MAIN, which excludes CLI11) pins them against the
// live production code (Task 2.3).
struct CliRename
{
  std::string old_flag; ///< deprecated spelling incl. leading "--" (e.g. "--clusters")
  std::string new_flag; ///< canonical spelling incl. leading "--" (e.g. "--n-clusters")
};

/// Single source of truth for every 2.0 CLI/TOML rename; drives both the runtime
/// deprecation warnings (main) and the regression tests.
inline const std::vector<CliRename> &cli_renames()
{
  static const std::vector<CliRename> table{
      { "--clusters", "--n-clusters" }, // concept n_clusters (api-contract-2.0.md §1.5/§2.1)
      { "--restart", "--resume" },      // api-contract-2.0.md §2.7
  };
  return table;
}

/// Exact one-line deprecation warning written to stderr when an old spelling is
/// used. Format is fixed and shared by every rename (CLI flag AND TOML/YAML key).
inline std::string format_deprecation_warning(std::string_view old_flag,
                                              std::string_view new_flag)
{
  std::string msg = "[dtwc] warning: '";
  msg.append(old_flag);
  msg += "' is deprecated, use '";
  msg.append(new_flag);
  msg += "' instead";
  return msg;
}

/// Canonical spelling for a deprecated flag/key, or "" if `spelling` is not a
/// known deprecated name. Accepts either the "--flag" form or the bare
/// TOML/YAML "key" form (no leading dashes).
inline std::string canonical_flag_for(std::string_view spelling)
{
  for (const auto &r : cli_renames()) {
    std::string_view bare_old{ r.old_flag };
    if (bare_old.size() >= 2) bare_old.remove_prefix(2); // drop "--"
    if (spelling == r.old_flag || spelling == bare_old)
      return r.new_flag;
  }
  return {};
}

#ifndef DTWC_CL_NO_MAIN
int main(int argc, char *argv[])
{
  CLI::App app{"DTWC++ -- Dynamic Time Warping Clustering"};
  app.set_version_flag("--version", DTWC_VERSION_STRING,
                       "Print the DTWC++ version and exit");

  // TOML config file support (CLI11 built-in, processes before parsing).
  // TOML keys are the canonical long-flag names without "--" (e.g. `n-clusters`,
  // `max-iter`). Deprecated keys are accepted with a warning per cli_renames():
  //   clusters -> n-clusters,  restart -> resume.
  app.set_config("--config", "", "Read TOML configuration file");

  // YAML config file (optional, processed after CLI parsing)
  std::string yaml_config_path;
  app.add_option("--yaml-config", yaml_config_path, "Read YAML configuration file (requires -DDTWC_ENABLE_YAML=ON)");

  // Input/output
  std::string input_file;
  std::string output_dir = "./results";
  std::string prob_name = "dtwc";
  std::string parquet_column;
  app.add_option("-i,--input", input_file, "Input file (CSV, Parquet, Arrow IPC, .dtws) or folder");
  app.add_option("-o,--output", output_dir, "Output directory");
  app.add_option("--name", prob_name, "Problem name (used in output filenames)");
  app.add_option("--column", parquet_column, "Column name to use as time series (Parquet only)");

  std::string dtype_str = "float64";
  app.add_option("--dtype,--data-precision,--data-type", dtype_str,
      "Series data type: float64 (default, full precision) or float32 (2x memory saving) (aliases: f32, f64, float, double)")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"float32", "float32"}, {"f32", "float32"}, {"fp32", "float32"},
              {"float64", "float64"}, {"f64", "float64"}, {"fp64", "float64"},
              {"double", "float64"}, {"float", "float32"}},
          CLI::ignore_case));

  std::string ram_limit_str;
  app.add_option("--ram-limit", ram_limit_str,
      "Max RAM for series data (e.g. 2G, 500M, 128G). Default: no limit.");

  // Clustering parameters
  int n_clusters = 3;
  std::string method = "auto";
  int band = -1;
  std::string metric = "l1";
  std::string variant = "standard";
  int max_iter = 100;
  int n_init = 1;

  auto *nclusters_opt = app.add_option("-k,--n-clusters", n_clusters, "Number of clusters")
                            ->check(CLI::PositiveNumber);
  // Deprecated spelling: --clusters -> --n-clusters (api-contract-2.0.md §1.5).
  // Bound to its own variable so the canonical flag always wins; hidden from
  // --help but accepted (from CLI and from a --config TOML `clusters` key) with a
  // one-line stderr warning emitted in the post-parse block below.
  int n_clusters_deprecated = -1;
  auto *clusters_dep_opt = app.add_option("--clusters", n_clusters_deprecated,
                                          "DEPRECATED alias of --n-clusters")
                               ->group("");
  app.add_option("-m,--method", method, "Clustering method: auto, pam, onebatch, clara, kmedoids, mip, lrcore, hierarchical, tadpole")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"auto", "auto"}, {"pam", "pam"}, {"onebatch", "onebatch"},
              {"obp", "onebatch"}, {"clara", "clara"},
              {"kmedoids", "kmedoids"}, {"mip", "mip"},
              {"lrcore", "lrcore"}, {"lr", "lrcore"},
              {"hierarchical", "hierarchical"}, {"hclust", "hierarchical"},
              {"tadpole", "tadpole"}},
          CLI::ignore_case));
  app.add_option("-b,--band", band, "Sakoe-Chiba band width (-1 = full DTW)");
  app.add_option("--metric", metric, "Distance metric: l1, squared_euclidean")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"l1", "l1"}, {"squared_euclidean", "squared_euclidean"},
              {"sqeuclidean", "squared_euclidean"}, {"l2sq", "squared_euclidean"}},
          CLI::ignore_case));
  app.add_option("--variant", variant, "DTW variant: standard, ddtw, wdtw, adtw, softdtw, msm, twe")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"standard", "standard"}, {"ddtw", "ddtw"}, {"wdtw", "wdtw"},
              {"adtw", "adtw"}, {"softdtw", "softdtw"}, {"soft-dtw", "softdtw"},
              {"msm", "msm"}, {"twe", "twe"}},
          CLI::ignore_case));
  app.add_option("--max-iter", max_iter, "Maximum iterations");
  app.add_option("--n-init", n_init, "Number of random restarts (PAM/kMedoids)");
  double tadpole_dc = -1.0;
  app.add_option("--dc", tadpole_dc, "TADPole density cutoff distance (default: auto-select)");

  // DTW variant parameters
  double wdtw_g = 0.05;
  double adtw_penalty = 1.0;
  double sdtw_gamma = 1.0;
  double msm_c = 1.0;
  double twe_nu = 0.001;
  double twe_lambda = 1.0;
  app.add_option("--wdtw-g", wdtw_g, "WDTW logistic weight steepness");
  app.add_option("--adtw-penalty", adtw_penalty, "ADTW non-diagonal step penalty");
  app.add_option("--sdtw-gamma", sdtw_gamma, "Soft-DTW smoothing parameter");
  app.add_option("--msm-c", msm_c, "MSM split/merge cost (default 1.0)");
  app.add_option("--twe-nu", twe_nu, "TWE stiffness nu (default 0.001)");
  app.add_option("--twe-lambda", twe_lambda, "TWE edit penalty lambda (default 1.0)");
  std::string mv_mode = "dependent";
  app.add_option("--mv-mode", mv_mode, "Multivariate mode (ndim>1): dependent, independent")
    ->check(CLI::IsMember({ "dependent", "independent" }));

  // Sampling-based clustering. One seed spelling and default cover PAM,
  // OneBatchPAM, and CLARA.
  int sample_size = -1;
  int n_samples = 5;
  unsigned clara_seed = dtwc::settings::DEFAULT_RANDOM_SEED;
  app.add_option("--sample-size", sample_size, "CLARA subsample size (-1 = auto)");
  app.add_option("--n-samples", n_samples, "CLARA number of subsamples");
  app.add_option("--seed", clara_seed,
                 "Random seed for PAM, OneBatchPAM, and CLARA")
      ->check(CLI::Range(0u, std::numeric_limits<unsigned>::max()));

  // OneBatchPAM-specific. The seed is shared with CLARA so reproducibility has
  // one CLI spelling across sampling-based methods.
  int onebatch_size = -1;
  std::string onebatch_weighting = "nniw";
  app.add_option("--batch-size", onebatch_size,
                 "OneBatchPAM fixed objective batch size (-1 = logarithmic auto)");
  app.add_option("--batch-weighting", onebatch_weighting,
                 "OneBatchPAM weighting: uniform, debiased, nniw")
    ->transform(CLI::CheckedTransformer(
        std::map<std::string, std::string>{{"uniform", "uniform"},
                                           {"debiased", "debiased"},
                                           {"debias", "debiased"},
                                           {"nniw", "nniw"}},
        CLI::ignore_case));

  // Hierarchical-specific
  std::string linkage_str = "average";
  app.add_option("--linkage", linkage_str, "Hierarchical linkage: single, complete, average")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"single", "single"}, {"complete", "complete"}, {"average", "average"}},
          CLI::ignore_case));

  // CSV parsing
  int skip_rows = 0;
  int skip_cols = 0;
  app.add_option("--skip-rows", skip_rows, "Number of header rows to skip");
  app.add_option("--skip-cols", skip_cols, "Number of leading columns to skip");

  // Distance matrix I/O
  std::string dist_mat_path;
  app.add_option("--dist-matrix", dist_mat_path, "Path to precomputed distance matrix CSV");

  // Checkpointing
  std::string checkpoint_dir;
  app.add_option("--checkpoint", checkpoint_dir, "Checkpoint directory for save/resume");

  // Binary checkpoint resume & mmap threshold
  bool resume = false;
  size_t mmap_threshold = 50000;
  app.add_flag("--resume", resume, "Resume from checkpoint (distance matrix cache + clustering state)");
  // Deprecated spelling: --restart -> --resume (api-contract-2.0.md §2.7). Hidden
  // from --help; accepted with a one-line stderr warning (post-parse block below).
  bool restart_deprecated = false;
  auto *restart_dep_opt = app.add_flag("--restart", restart_deprecated,
                                       "DEPRECATED alias of --resume")
                              ->group("");
  app.add_option("--mmap-threshold", mmap_threshold, "N above which to use memory-mapped distance matrix (0=always)")
      ->check(CLI::NonNegativeNumber);

  // MIP solver (for method=mip)
  std::string solver = "highs";
  app.add_option("--solver", solver, "MIP solver: highs, gurobi")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"highs", "highs"}, {"gurobi", "gurobi"}},
          CLI::ignore_case));

  // MIP solver settings (kebab-case: matches CLI flags, TOML keys, and YAML keys)
  double mip_gap = 1e-5;
  int time_limit = -1;
  bool no_warm_start = false;
  int numeric_focus = 1;
  int mip_focus = 2;
  bool verbose_solver = false;

  app.add_option("--mip-gap", mip_gap, "MIP optimality gap tolerance (default: 1e-5)");
  app.add_option("--time-limit", time_limit, "MIP solver time limit in seconds (-1 = unlimited)");
  app.add_flag("--no-warm-start", no_warm_start, "Disable FastPAM warm start for MIP");
  app.add_option("--numeric-focus", numeric_focus, "Gurobi NumericFocus (0-3, default: 1)");
  app.add_option("--mip-focus", mip_focus, "Gurobi MIPFocus (0-3, default: 2)");
  app.add_flag("--verbose-solver", verbose_solver, "Show MIP solver log output");

  std::string benders_mode = "auto";
  app.add_option("--benders", benders_mode, "Benders decomposition: auto (N>200), on, off");

  // Compute device
  std::string device = "cpu";
  std::string gpu_precision = "auto";
  app.add_option("-d,--device", device, "Compute device: cpu, cuda, cuda:N");
  app.add_option("--gpu-precision,--gpu-dtype", gpu_precision,
      "GPU kernel precision: auto (default), float32/f32/fp32, float64/f64/fp64/double")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"auto", "auto"},
              {"float32", "fp32"}, {"f32", "fp32"}, {"fp32", "fp32"}, {"float", "fp32"},
              {"float64", "fp64"}, {"f64", "fp64"}, {"fp64", "fp64"}, {"double", "fp64"}},
          CLI::ignore_case));

  // Verbosity
  bool verbose = false;
  app.add_flag("-v,--verbose", verbose, "Verbose output");

  // Show help if no arguments provided (before CLI11 parses, so --input
  // is not required yet — YAML may provide it).
  if (argc == 1) {
    std::cout << app.help() << '\n';
    return EXIT_SUCCESS;
  }

  CLI11_PARSE(app, argc, argv);

  // ---- CLI/TOML flag deprecations (api-contract-2.0.md §4, §7 item 3) ----
  // Old spellings are accepted from the command line AND from a --config TOML file
  // (CLI11 maps config keys onto these same options), but each emits one stderr
  // warning per use and yields precedence to the canonical spelling. See
  // cli_renames() for the SSOT table.
  if (clusters_dep_opt->count() > 0) {
    std::cerr << format_deprecation_warning("--clusters", "--n-clusters") << "\n";
    if (nclusters_opt->count() == 0) n_clusters = n_clusters_deprecated;
  }
  if (restart_dep_opt->count() > 0) {
    std::cerr << format_deprecation_warning("--restart", "--resume") << "\n";
    resume = resume || restart_deprecated;
  }

  // ---- YAML config loading (post-parse, CLI flags take precedence) ----
  if (!yaml_config_path.empty()) {
#ifdef DTWC_HAS_YAML
    try {
      YAML::Node config = YAML::LoadFile(yaml_config_path);

      // TODO: This always overrides CLI values if the YAML key exists.
      // Should use app["--flag"]->count() to check if CLI was explicitly set.
      // Pre-existing issue — fixing requires refactoring all set_if_unset calls.
      auto set_if_unset = [&](const std::string &key, auto &var) {
        if (config[key]) {
          using T = std::decay_t<decltype(var)>;
          var = config[key].as<T>();
        }
      };

      set_if_unset("input", input_file);
      set_if_unset("output", output_dir);
      set_if_unset("name", prob_name);
      set_if_unset("n-clusters", n_clusters);
      set_if_unset("method", method);
      set_if_unset("band", band);
      set_if_unset("metric", metric);
      set_if_unset("variant", variant);
      set_if_unset("max-iter", max_iter);
      set_if_unset("n-init", n_init);
      set_if_unset("solver", solver);
      set_if_unset("device", device);
      set_if_unset("dtype", dtype_str);
      set_if_unset("gpu-precision", gpu_precision);
      set_if_unset("resume", resume);
      set_if_unset("verbose", verbose);

      // MIP solver settings
      set_if_unset("mip-gap", mip_gap);
      set_if_unset("time-limit", time_limit);
      set_if_unset("no-warm-start", no_warm_start);
      set_if_unset("numeric-focus", numeric_focus);
      set_if_unset("mip-focus", mip_focus);
      set_if_unset("verbose-solver", verbose_solver);

      // DTW variant parameters
      set_if_unset("wdtw-g", wdtw_g);
      set_if_unset("adtw-penalty", adtw_penalty);
      set_if_unset("sdtw-gamma", sdtw_gamma);
      set_if_unset("msm-c", msm_c);
      set_if_unset("twe-nu", twe_nu);
      set_if_unset("twe-lambda", twe_lambda);
      set_if_unset("mv-mode", mv_mode);

      // CLARA parameters
      set_if_unset("sample-size", sample_size);
      set_if_unset("n-samples", n_samples);
      set_if_unset("seed", clara_seed);

      // Hierarchical
      set_if_unset("linkage", linkage_str);

      // Deprecated YAML keys (accepted with a warning; canonical key wins).
      if (config["clusters"]) {
        std::cerr << format_deprecation_warning("--clusters", "--n-clusters") << "\n";
        if (!config["n-clusters"]) n_clusters = config["clusters"].as<int>();
      }
      if (config["restart"]) {
        std::cerr << format_deprecation_warning("--restart", "--resume") << "\n";
        if (!config["resume"]) resume = config["restart"].as<bool>();
      }

      if (verbose)
        std::cout << "Loaded YAML config: " << yaml_config_path << "\n";
    } catch (const YAML::Exception &e) {
      std::cerr << "Error loading YAML config: " << e.what() << "\n";
      return EXIT_FAILURE;
    }
#else
    std::cerr << "Error: YAML config requires building with -DDTWC_ENABLE_YAML=ON\n";
    return EXIT_FAILURE;
#endif
  }

  // ---- Post-parse validation (catches both CLI and YAML values) ----
  if (input_file.empty()) {
    std::cerr << "Error: --input is required via CLI or config file "
                 "(TOML; YAML if built with DTWC_ENABLE_YAML)\n";
    return EXIT_FAILURE;
  }
  if (n_clusters < 1) {
    std::cerr << "Error: --clusters must be a positive integer\n";
    return EXIT_FAILURE;
  }

  // Normalize YAML string values that bypass CLI11's CheckedTransformer.
  // Must replicate both case-folding AND alias mappings from the CLI definitions.
  auto to_lower = [](std::string &s) {
    for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  };
  to_lower(method);
  to_lower(metric);
  to_lower(variant);
  to_lower(linkage_str);

  // Alias mappings (mirror the CheckedTransformer maps above)
  if (method == "hclust") method = "hierarchical";
  if (metric == "sqeuclidean" || metric == "l2sq") metric = "squared_euclidean";
  if (variant == "soft-dtw") variant = "softdtw";

  // Normalize dtype/gpu-precision aliases from YAML (bypass CheckedTransformer)
  to_lower(dtype_str);
  if (dtype_str == "f32" || dtype_str == "fp32" || dtype_str == "float") dtype_str = "float32";
  if (dtype_str == "f64" || dtype_str == "fp64" || dtype_str == "double") dtype_str = "float64";
  to_lower(gpu_precision);
  if (gpu_precision == "float32" || gpu_precision == "f32" || gpu_precision == "float") gpu_precision = "fp32";
  if (gpu_precision == "float64" || gpu_precision == "f64" || gpu_precision == "double") gpu_precision = "fp64";

  // ---- Device + metric validation (Task 0.9) ----
  // Parse the device ONCE, up front. An invalid device is a hard error (never a
  // silent CPU fallback); this also folds "cuda:N" id parsing into one checked
  // place so a bad id can no longer std::terminate.
  const DeviceSpec dev = parse_device(device);
  if (!dev.valid) {
    std::cerr << "Error: " << dev.error << "\n";
    return EXIT_FAILURE;
  }
  if (const std::string merr = validate_metric_for_device(metric, dev.is_cuda);
      !merr.empty()) {
    std::cerr << "Error: " << merr << "\n";
    return EXIT_FAILURE;
  }

  // Forward --device to the process-wide dtwc::Env (Task 1.3) so device selection
  // has ONE source of truth and the no-silent-fallback rules apply — e.g. a GPU
  // request on a build with no GPU backend becomes a hard DeviceError here rather
  // than a quiet CPU fallback. cpu/cuda/cuda:N (validated above) all pass on a GPU
  // build; on a CPU-only build a cuda request stops here with the rebuild hint.
  try {
    dtwc::env().set_device(device);
  } catch (const dtwc::DeviceError &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  // ---- Setup ----
  dtwc::Clock clk;

  if (verbose) {
    std::cout << "DTWC++ Clustering\n"
              << "  Input:    " << input_file << "\n"
              << "  Output:   " << output_dir << "\n"
              << "  Name:     " << prob_name << "\n"
              << "  Clusters: " << n_clusters << "\n"
              << "  Method:   " << method << "\n"
              << "  Band:     " << (band < 0 ? "full" : std::to_string(band)) << "\n"
              << "  Metric:   " << metric << "\n"
              << "  Variant:  " << variant << "\n"
              << "  MaxIter:  " << max_iter << "\n"
              << "  N-init:   " << n_init << "\n"
              << "  Device:   " << device << "\n"
              << "  Dtype:    " << dtype_str << "\n"
              << "  GPU Prec: " << gpu_precision << "\n";
    if (method == "clara") {
      std::cout << "  CLARA sample_size: "
                << (sample_size < 0 ? "auto" : std::to_string(sample_size)) << "\n"
                << "  CLARA n_samples:   " << n_samples << "\n"
                << "  CLARA seed:        " << clara_seed << "\n";
    }
    if (method == "hierarchical") {
      std::cout << "  Linkage:   " << linkage_str << "\n";
    }

    // ---- System diagnostics (useful for SLURM .out logs) ----
    std::cout << "\n=== System Diagnostics ===\n";
#ifdef _OPENMP
    std::cout << "  OpenMP threads:  " << omp_get_max_threads() << "\n";
#else
    std::cout << "  OpenMP:          not available\n";
#endif
    auto print_env = [](const char *name) {
      const char *val = std::getenv(name);
      if (val) std::cout << "  " << name << ": " << val << "\n";
    };
    print_env("OMP_NUM_THREADS");
    print_env("SLURM_JOB_ID");
    print_env("SLURM_CPUS_PER_TASK");
    print_env("SLURM_NODELIST");
    print_env("SLURM_JOB_PARTITION");
    print_env("SLURM_GPUS");
#if defined(__linux__)
    {
      std::ifstream cpuinfo("/proc/cpuinfo");
      std::string line;
      while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
          auto pos = line.find(':');
          if (pos != std::string::npos)
            std::cout << "  CPU:             " << line.substr(pos + 2) << "\n";
          break;
        }
      }
    }
    {
      std::ifstream status("/proc/self/status");
      std::string line;
      while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmPeak:", 0) == 0)
          std::cout << "  " << line << "\n";
      }
    }
#endif
    std::cout << "==========================\n\n" << std::flush;
  }

  // Create output directory
  fs::create_directories(output_dir);

  // ---- Load data ----
  dtwc::Problem prob{prob_name};

  const bool is_dir = fs::is_directory(input_file);
  const auto input_ext = is_dir ? "" : fs::path(input_file).extension().string();

#ifdef DTWC_HAS_PARQUET
  // Check if directory contains .parquet files
  if (is_dir) {
    bool has_parquet = false;
    for (const auto &e : fs::directory_iterator(input_file)) {
      auto ext = e.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".parquet" || ext == ".pq") { has_parquet = true; break; }
    }
    if (has_parquet) {
      prob.set_data(dtwc::io::load_parquet_directory(input_file, parquet_column));
      if (verbose)
        std::cout << "Data loaded from Parquet directory: " << prob.size() << " series [" << clk << "]\n";
      goto data_loaded;
    }
  }
#endif

  if (input_ext == ".dtws") {
#ifndef DTWC_HAS_MMAP
    throw std::runtime_error(
      ".dtws memory-mapped input requires a build with llfio "
      "(-DDTWC_ENABLE_LLFIO=ON). This binary was built without mmap support.");
#else
    // Memory-mapped binary cache — zero-copy load
    auto store = dtwc::core::MmapDataStore::open(input_file);
    const size_t n = store.size();
    const size_t ndim = store.ndim();

    // Copy into Problem's Data (MmapDataStore integration into Problem is a future step)
    std::vector<std::vector<dtwc::data_t>> vecs(n);
    for (size_t i = 0; i < n; ++i) {
      auto sp = store.series(i);
      vecs[i].assign(sp.begin(), sp.end());
    }

    // Load names from sidecar file if it exists
    std::vector<std::string> names(n);
    auto names_path = fs::path(input_file).string() + ".names";
    if (fs::exists(names_path)) {
      std::ifstream nf(names_path);
      for (size_t i = 0; i < n && std::getline(nf, names[i]); ++i) {}
    } else {
      for (size_t i = 0; i < n; ++i) names[i] = "series_" + std::to_string(i);
    }

    prob.set_data(dtwc::Data(std::move(vecs), std::move(names), ndim));
    if (verbose)
      std::cout << "Data loaded from .dtws cache: " << prob.size() << " series [" << clk << "]\n";
#endif // DTWC_HAS_MMAP
  }
#ifdef DTWC_HAS_ARROW
  else if (input_ext == ".arrow" || input_ext == ".ipc" || input_ext == ".feather") {
    // Arrow IPC — zero-copy memory-mapped load
    auto src = dtwc::io::ArrowIPCDataSource::open(input_file);
    const size_t n = src.size();
    const size_t ndim = src.ndim();

    // Copy into Problem's Data (direct ArrowIPCDataSource integration is a future step)
    std::vector<std::vector<dtwc::data_t>> vecs(n);
    for (size_t i = 0; i < n; ++i) {
      auto sp = src.series(i);
      vecs[i].assign(sp.begin(), sp.end());
    }

    auto names = src.all_names();
    prob.set_data(dtwc::Data(std::move(vecs), std::move(names), ndim));
    if (verbose)
      std::cout << "Data loaded from Arrow IPC: " << prob.size() << " series [" << clk << "]\n";
  }
#endif
#ifdef DTWC_HAS_PARQUET
  else if (input_ext == ".parquet" || input_ext == ".pq") {
    // Parquet: direct reading via Arrow Parquet reader
    if (fs::is_directory(input_file)) {
      prob.set_data(dtwc::io::load_parquet_directory(input_file, parquet_column));
    } else {
      prob.set_data(dtwc::io::load_parquet_file(input_file, parquet_column));
    }
    if (verbose)
      std::cout << "Data loaded from Parquet: " << prob.size() << " series [" << clk << "]\n";
  }
#endif
  else {
    // Default: CSV/TSV via DataLoader
    dtwc::DataLoader dl{input_file};
    dl.startColumn(skip_cols).startRow(skip_rows);
    prob.set_data(dl.load());
    if (verbose)
      std::cout << "Data loaded: " << prob.size() << " series [" << clk << "]\n";
  }

  data_loaded:
  if (verbose && prob.size() > 0) {
    size_t total_elements = 0;
    for (const auto &v : prob.data.p_vec) total_elements += v.size();
    size_t data_bytes = total_elements * sizeof(dtwc::data_t);
    std::cout << "  Data memory: ~" << (data_bytes / (1ULL << 20)) << " MB ("
              << prob.size() << " series, "
              << (total_elements / prob.size()) << " avg length, "
              << dtype_str << ")\n";
  }

  // ---- Apply precision conversion ----
  if (dtype_str == "float32" && !prob.data.is_f32() && !prob.data.is_view()) {
    prob.set_data(convert_to_f32(std::move(prob.data)));
    if (verbose)
      std::cout << "Converted to float32 (2x memory saving)\n";
  }

  // Parse and store ram limit for chunked CLARA processing
  const size_t ram_limit = parse_ram_limit(ram_limit_str);
  if (ram_limit > 0 && verbose)
    std::cout << "RAM limit: " << (ram_limit / (1ULL << 30)) << " GB\n";

  // ---- Auto method selection ----
  if (method == "auto") {
    const size_t N = prob.size();
    method = (N <= 5000) ? "pam" : "clara";
    if (verbose)
      std::cout << "Auto-selected method: " << method << " (N=" << N << ")\n";
  }

  const bool matrix_free_method = (method == "onebatch" || method == "tadpole");

  if (resume) {
    auto ckpt_path = fs::path(output_dir) / (prob_name + "_checkpoint.bin");
    dtwc::core::ClusteringResult ckpt_result;
    if (dtwc::load_binary_checkpoint(ckpt_result, ckpt_path)) {
      if (verbose)
        std::cout << "Loaded checkpoint: " << ckpt_result.iterations
                  << " iterations, cost=" << ckpt_result.total_cost << "\n";
    }
  }

  // ---- Configure DTW ----
  prob.set_band(band);
  prob.maxIter = max_iter;
  prob.N_repetition = n_init;
  prob.output_folder = output_dir;
  prob.verbose = verbose;

  // Wire MIP solver settings
  prob.mip_settings.mip_gap = mip_gap;
  prob.mip_settings.time_limit_sec = time_limit;
  prob.mip_settings.warm_start = !no_warm_start;
  prob.mip_settings.numeric_focus = numeric_focus;
  prob.mip_settings.mip_focus = mip_focus;
  prob.mip_settings.verbose_solver = verbose_solver;
  prob.mip_settings.benders = benders_mode;

  // Wire GPU settings from --device and --gpu-precision
  if (dev.is_cuda) {
    prob.distance_strategy = dtwc::DistanceMatrixStrategy::CUDA;
    prob.cuda_settings.device_id = dev.cuda_id;
    if (gpu_precision == "fp32") prob.cuda_settings.precision = 1;
    else if (gpu_precision == "fp64") prob.cuda_settings.precision = 2;
  }

  // Set DTW variant
  dtwc::core::DTWVariantParams vparams;
  if (variant == "standard")
    vparams.variant = dtwc::core::DTWVariant::Standard;
  else if (variant == "ddtw")
    vparams.variant = dtwc::core::DTWVariant::DDTW;
  else if (variant == "wdtw") {
    vparams.variant = dtwc::core::DTWVariant::WDTW;
    vparams.wdtw_g = wdtw_g;
  } else if (variant == "adtw") {
    vparams.variant = dtwc::core::DTWVariant::ADTW;
    vparams.adtw_penalty = adtw_penalty;
  } else if (variant == "softdtw") {
    vparams.variant = dtwc::core::DTWVariant::SoftDTW;
    vparams.sdtw_gamma = sdtw_gamma;
  } else if (variant == "msm") {
    vparams.variant = dtwc::core::DTWVariant::MSM;
    vparams.msm_c = msm_c;
  } else if (variant == "twe") {
    vparams.variant = dtwc::core::DTWVariant::TWE;
    vparams.twe_nu = twe_nu;
    vparams.twe_lambda = twe_lambda;
  }
  to_lower(mv_mode);
  vparams.mv_mode = (mv_mode == "independent") ? dtwc::core::MVMode::Independent
                                               : dtwc::core::MVMode::Dependent;
  prob.set_variant(vparams);

  // Bind persistent distance storage only after every distance-affecting CLI
  // option has reached Problem. Opening earlier made the cache identity observe
  // the default band/variant/backend rather than the user's configuration.
  const auto cache_metric = metric == "squared_euclidean"
    ? dtwc::core::MetricType::SquaredL2
    : dtwc::core::MetricType::L1;
  try {
    const auto mmap_cache = configure_cli_distance_storage(
      prob, method, mmap_threshold,
      fs::path(output_dir) / (prob_name + "_distmat.cache"), cache_metric,
      !checkpoint_dir.empty(), !dist_mat_path.empty());
    if (mmap_cache && verbose)
      std::cout << "Using memory-mapped distance matrix: " << *mmap_cache << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  // Set MIP solver (relevant for method=mip)
  if (solver == "highs")
    prob.set_solver(dtwc::Solver::HiGHS);
  else if (solver == "gurobi")
    prob.set_solver(dtwc::Solver::Gurobi);

  // ---- Load precomputed distance matrix if provided ----
  if (!dist_mat_path.empty()) {
    try {
      prob.readDistanceMatrix(dist_mat_path);
      if (verbose)
        std::cout << "Loaded distance matrix from " << dist_mat_path << "\n";
    } catch (const std::exception &e) {
      std::cerr << "Warning: Could not load distance matrix: " << e.what()
                << "\nContinuing without precomputed matrix.\n";
    }
  }

  // ---- Load checkpoint if available ----
  if (!checkpoint_dir.empty()) {
    if (dtwc::load_checkpoint(prob, checkpoint_dir)) {
      if (verbose)
        std::cout << "Resumed from checkpoint: " << checkpoint_dir << "\n";
    } else if (verbose) {
      std::cout << "No valid checkpoint found at " << checkpoint_dir << ", starting fresh.\n";
    }
  }

  // ---- GPU distance matrix (if --device cuda) ----
  if (dev.is_cuda && matrix_free_method) {
    std::cerr << "Error: --method " << method
              << " uses a matrix-free CPU distance schedule; CUDA execution is not "
                 "implemented for that schedule. Use --device cpu.\n";
    return EXIT_FAILURE;
  }
  if (dev.is_cuda && !prob.isDistanceMatrixFilled()) {
#ifdef DTWC_HAS_CUDA
    if (!dtwc::cuda::cuda_available()) {
      std::cerr << "Error: --device cuda requested but no CUDA GPU detected.\n";
      return EXIT_FAILURE;
    }

    // Device id already parsed & validated by parse_device (Task 0.9).
    const int cuda_device_id = dev.cuda_id;

    if (variant != "standard") {
      std::cerr << "Error: --device cuda only supports --variant standard "
                << "(got '" << variant << "'). Use CPU for other variants.\n";
      return EXIT_FAILURE;
    }

    dtwc::cuda::CUDADistMatOptions cuda_opts;
    cuda_opts.band = band;
    cuda_opts.use_squared_l2 = (metric == "squared_euclidean");
    cuda_opts.device_id = cuda_device_id;
    cuda_opts.verbose = verbose;

    if (gpu_precision == "fp32")
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP32;
    else if (gpu_precision == "fp64")
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP64;
    // else Auto (default)

    if (verbose)
      std::cout << "Computing distance matrix on GPU ("
                << dtwc::cuda::cuda_device_info(cuda_device_id) << ") ...\n";

    auto cuda_result = dtwc::cuda::compute_distance_matrix_cuda(prob.data.p_vec, cuda_opts);

    // Inject GPU results into whichever storage policy was selected. The mmap
    // path is essential when the CLI threshold was crossed; its fingerprint
    // already binds the cache to the CUDA metric/precision configuration.
    std::visit([&](auto &dm) {
      using Matrix = std::decay_t<decltype(dm)>;
      if constexpr (std::is_same_v<Matrix, dtwc::core::DenseDistanceMatrix>) {
        dm.resize(cuda_result.n);
      } else if (dm.size() != cuda_result.n) {
        throw std::runtime_error("CUDA result size does not match mmap distance cache");
      }
      for (size_t i = 0; i < cuda_result.n; ++i)
        for (size_t j = i; j < cuda_result.n; ++j)
          dm.set(i, j, cuda_result.matrix[i * cuda_result.n + j]);
    }, prob.distance_matrix());
    if (verbose)
      std::cout << "GPU distance matrix: " << cuda_result.pairs_computed
                << " pairs in " << std::setprecision(3)
                << cuda_result.gpu_time_sec * 1000 << " ms [" << clk << "]\n";
#else
    std::cerr << "Error: --device cuda requested but DTWC++ was built without CUDA.\n"
              << "Rebuild with: cmake -DDTWC_ENABLE_CUDA=ON ...\n";
    return EXIT_FAILURE;
#endif
  }

  // ---- Run clustering ----
  dtwc::core::ClusteringResult result;

  if (method == "pam") {
    // FastPAM
    if (verbose)
      std::cout << "Running FastPAM (k=" << n_clusters << ") ...\n";

    result = run_cli_pam(prob, n_clusters, max_iter, clara_seed);

    if (verbose) {
      std::cout << "FastPAM "
                << (result.converged ? "converged" : "did not converge")
                << " in " << result.iterations << " iterations"
                << ", cost=" << std::setprecision(6) << result.total_cost
                << " [" << clk << "]\n";
    }
  } else if (method == "onebatch") {
    dtwc::algorithms::OneBatchPAMOptions onebatch_options;
    onebatch_options.n_clusters = n_clusters;
    onebatch_options.batch_size = onebatch_size;
    onebatch_options.max_iter = max_iter;
    onebatch_options.random_seed = clara_seed;
    if (onebatch_weighting == "uniform")
      onebatch_options.weighting = dtwc::algorithms::OneBatchWeighting::Uniform;
    else if (onebatch_weighting == "debiased")
      onebatch_options.weighting = dtwc::algorithms::OneBatchWeighting::Debiased;
    else
      onebatch_options.weighting = dtwc::algorithms::OneBatchWeighting::NearestNeighbor;

    dtwc::algorithms::OneBatchPAMStats onebatch_stats;
    result = dtwc::algorithms::one_batch_pam(prob, onebatch_options, &onebatch_stats);
    if (verbose) {
      std::cout << "OneBatchPAM finished, cost=" << std::setprecision(6)
                << result.total_cost << ", batch=" << onebatch_stats.batch_size
                << ", distance-matrix fraction=" << std::setprecision(3)
                << onebatch_stats.full_matrix_fraction << " [" << clk << "]\n";
    }
  } else if (method == "clara") {
    // FastCLARA
    if (verbose)
      std::cout << "Running FastCLARA (k=" << n_clusters << ") ...\n";

    // Auto-scale sample size for large N
    if (sample_size < 0 && prob.size() > 50000)
      sample_size = std::max(40 + 2 * n_clusters,
        static_cast<int>(std::sqrt(static_cast<double>(prob.size())) * n_clusters));

    dtwc::algorithms::CLARAOptions clara_opts;
    clara_opts.n_clusters = n_clusters;
    clara_opts.sample_size = sample_size;
    clara_opts.n_samples = n_samples;
    clara_opts.max_iter = max_iter;
    clara_opts.random_seed = clara_seed;

    // Wire RAM-limit chunked processing for Parquet input
    if (ram_limit > 0) {
      bool is_parquet_input = (input_ext == ".parquet" || input_ext == ".pq");
      if (is_parquet_input) {
        clara_opts.ram_limit_bytes = ram_limit;
        clara_opts.parquet_path = input_file;
        clara_opts.parquet_column = parquet_column;
        clara_opts.use_float32 = (dtype_str == "float32");
      } else if (verbose) {
        std::cerr << "Warning: --ram-limit only effective with Parquet input for streaming CLARA\n";
      }
    }

    result = dtwc::algorithms::fast_clara(prob, clara_opts);

    if (verbose) {
      std::cout << "FastCLARA finished"
                << ", cost=" << std::setprecision(6) << result.total_cost
                << " [" << clk << "]\n";
    }
  } else if (method == "kmedoids") {
    // Legacy kMedoids Lloyd
    prob.set_numberOfClusters(n_clusters);
    prob.method = dtwc::Method::Kmedoids;
    prob.cluster();

    // Build result from prob state
    result.labels = prob.clusters_ind;
    result.medoid_indices = prob.centroids_ind;
    result.total_cost = prob.findTotalCost();
    result.converged = (prob.last_iterations < max_iter);
    result.iterations = prob.last_iterations;

    if (verbose)
      std::cout << "kMedoids Lloyd finished, cost=" << result.total_cost
                << " [" << clk << "]\n";
  } else if (method == "mip") {
    // MIP method
    prob.set_numberOfClusters(n_clusters);
    prob.method = dtwc::Method::MIP;
    prob.cluster();

    result.labels = prob.clusters_ind;
    result.medoid_indices = prob.centroids_ind;
    result.total_cost = prob.findTotalCost();
    result.converged = true;

    if (verbose)
      std::cout << "MIP clustering finished, cost=" << result.total_cost
                << " [" << clk << "]\n";
  } else if (method == "lrcore" || method == "lr") {
    // LR-core exact (Lagrangian bound + reduced-cost fixing + y-branching).
    prob.set_numberOfClusters(n_clusters);
    prob.method = dtwc::Method::LRCore;
    prob.cluster();

    result.labels = prob.clusters_ind;
    result.medoid_indices = prob.centroids_ind;
    result.total_cost = prob.findTotalCost();
    result.converged = true;

    if (verbose)
      std::cout << "LR-core clustering finished, cost=" << result.total_cost
                << " [" << clk << "]\n";
  } else if (method == "tadpole") {
    // TADPole density-peaks with admissible LB/UB DTW pruning.
    prob.set_numberOfClusters(n_clusters);
    prob.tadpole_dc = tadpole_dc; // <0 ⇒ auto-select from a DTW subsample
    prob.method = dtwc::Method::TADPole;
    prob.cluster();

    result.labels = prob.clusters_ind;
    result.medoid_indices = prob.centroids_ind;
    result.total_cost = prob.findTotalCost();
    result.converged = true;

    if (verbose)
      std::cout << "TADPole clustering finished, cost=" << result.total_cost
                << " [" << clk << "]\n";
  } else if (method == "hierarchical") {
    // Agglomerative hierarchical clustering
    if (verbose)
      std::cout << "Running hierarchical clustering (k=" << n_clusters
                << ", linkage=" << linkage_str << ") ...\n";

    dtwc::algorithms::HierarchicalOptions hier_opts;
    if (linkage_str == "single")
      hier_opts.linkage = dtwc::algorithms::Linkage::Single;
    else if (linkage_str == "complete")
      hier_opts.linkage = dtwc::algorithms::Linkage::Complete;
    else
      hier_opts.linkage = dtwc::algorithms::Linkage::Average;

    prob.fillDistanceMatrix(); // hierarchical requires full pairwise distances
    auto dend = dtwc::algorithms::build_dendrogram(prob, hier_opts);
    result = dtwc::algorithms::cut_dendrogram(dend, prob, n_clusters);

    if (verbose) {
      std::cout << "Hierarchical clustering finished"
                << ", cost=" << std::setprecision(6) << result.total_cost
                << " [" << clk << "]\n";
    }
  }

  // ---- Save binary checkpoint of clustering result ----
  {
    auto ckpt_path = fs::path(output_dir) / (prob_name + "_checkpoint.bin");
    dtwc::save_binary_checkpoint(result, ckpt_path);
  }

  // ---- Apply result to prob for scoring ----
  if (method == "pam" || method == "clara" || method == "hierarchical") {
    prob.set_numberOfClusters(n_clusters);
    prob.clusters_ind = result.labels;
    prob.centroids_ind = result.medoid_indices;
  }

  // ---- Save checkpoint ----
  if (!checkpoint_dir.empty()) {
    try {
      dtwc::save_checkpoint(prob, checkpoint_dir);
      if (verbose)
        std::cout << "Checkpoint saved to " << checkpoint_dir << "\n";
    } catch (const std::exception &e) {
      std::cerr << "Warning: Could not save checkpoint: " << e.what() << "\n";
    }
  }

  // ---- Write results ----
  const fs::path out_dir{output_dir};

  // Cluster labels
  const auto labels_path = out_dir / (prob_name + "_labels.csv");
  write_labels_csv(labels_path, prob, result);
  if (verbose)
    std::cout << "Labels written to " << labels_path << "\n";

  // Medoids
  const auto medoids_path = out_dir / (prob_name + "_medoids.csv");
  write_medoids_csv(medoids_path, prob, result);
  if (verbose)
    std::cout << "Medoids written to " << medoids_path << "\n";

  // Distance matrix (if computed)
  if (prob.isDistanceMatrixFilled()) {
    const auto dm_path = out_dir / (prob_name + "_distance_matrix.csv");
    prob.writeDistanceMatrix(prob_name + "_distance_matrix.csv");
    if (verbose)
      std::cout << "Distance matrix written to " << dm_path << "\n";
  }

  // Silhouette scores (requires filled distance matrix)
  if (prob.isDistanceMatrixFilled() && n_clusters > 1) {
    try {
      auto sil = dtwc::scores::silhouette(prob);
      const auto sil_path = out_dir / (prob_name + "_silhouettes.csv");
      write_silhouettes_csv(sil_path, sil, prob, result);

      double mean_sil = 0.0;
      if (!sil.empty()) {
        mean_sil = std::accumulate(sil.begin(), sil.end(), 0.0) / static_cast<double>(sil.size());
      }

      if (verbose)
        std::cout << "Silhouette scores written, mean=" << std::setprecision(4) << mean_sil << "\n";
    } catch (const std::exception &e) {
      std::cerr << "Warning: Could not compute silhouette scores: " << e.what() << "\n";
    }
  }

  // Summary
  std::cout << "\n=== Results ===\n"
            << "  Method:     " << method << "\n"
            << "  Clusters:   " << n_clusters << "\n"
            << "  Total cost: " << std::setprecision(6) << result.total_cost << "\n"
            << "  Converged:  " << (result.converged ? "yes" : "no") << "\n"
            << "  Iterations: " << result.iterations << "\n"
            << "  Output:     " << output_dir << "/\n"
            << "  Time:       " << clk << "\n";

  return EXIT_SUCCESS;
}
#endif // DTWC_CL_NO_MAIN
