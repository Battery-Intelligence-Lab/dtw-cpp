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
#include "core/mmap_data_store.hpp"

#ifdef DTWC_HAS_ARROW
#include "io/arrow_ipc_reader.hpp"
#endif
#ifdef DTWC_HAS_PARQUET
#include "io/parquet_reader.hpp"
#endif

#include <CLI/CLI.hpp>

#ifdef DTWC_HAS_YAML
#include <yaml-cpp/yaml.h>
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

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

int main(int argc, char *argv[])
{
  CLI::App app{"DTWC++ -- Dynamic Time Warping Clustering"};

  // TOML config file support (CLI11 built-in, processes before parsing)
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

  std::string precision_str = "float32";
  app.add_option("--precision", precision_str, "Series data precision: float32 (default, 2x memory saving) or float64")
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

  app.add_option("-k,--clusters", n_clusters, "Number of clusters")
      ->check(CLI::PositiveNumber);
  app.add_option("-m,--method", method, "Clustering method: auto, pam, clara, kmedoids, mip, hierarchical")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"auto", "auto"}, {"pam", "pam"}, {"clara", "clara"},
              {"kmedoids", "kmedoids"}, {"mip", "mip"},
              {"hierarchical", "hierarchical"}, {"hclust", "hierarchical"}},
          CLI::ignore_case));
  app.add_option("-b,--band", band, "Sakoe-Chiba band width (-1 = full DTW)");
  app.add_option("--metric", metric, "Distance metric: l1, squared_euclidean")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"l1", "l1"}, {"squared_euclidean", "squared_euclidean"},
              {"sqeuclidean", "squared_euclidean"}, {"l2sq", "squared_euclidean"}},
          CLI::ignore_case));
  app.add_option("--variant", variant, "DTW variant: standard, ddtw, wdtw, adtw, softdtw")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"standard", "standard"}, {"ddtw", "ddtw"}, {"wdtw", "wdtw"},
              {"adtw", "adtw"}, {"softdtw", "softdtw"}, {"soft-dtw", "softdtw"}},
          CLI::ignore_case));
  app.add_option("--max-iter", max_iter, "Maximum iterations");
  app.add_option("--n-init", n_init, "Number of random restarts (PAM/kMedoids)");

  // DTW variant parameters
  double wdtw_g = 0.05;
  double adtw_penalty = 1.0;
  double sdtw_gamma = 1.0;
  app.add_option("--wdtw-g", wdtw_g, "WDTW logistic weight steepness");
  app.add_option("--adtw-penalty", adtw_penalty, "ADTW non-diagonal step penalty");
  app.add_option("--sdtw-gamma", sdtw_gamma, "Soft-DTW smoothing parameter");

  // CLARA-specific
  int sample_size = -1;
  int n_samples = 5;
  unsigned clara_seed = 42;
  app.add_option("--sample-size", sample_size, "CLARA subsample size (-1 = auto)");
  app.add_option("--n-samples", n_samples, "CLARA number of subsamples");
  app.add_option("--seed", clara_seed, "Random seed for CLARA");

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

  // Binary checkpoint restart & mmap threshold
  bool restart = false;
  size_t mmap_threshold = 50000;
  app.add_flag("--restart", restart, "Resume from checkpoint (distance matrix cache + clustering state)");
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
  std::string precision = "auto";
  app.add_option("-d,--device", device, "Compute device: cpu, cuda, cuda:N");
  app.add_option("--precision", precision, "GPU precision: auto, fp32, fp64");

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

  // ---- YAML config loading (post-parse, CLI flags take precedence) ----
  if (!yaml_config_path.empty()) {
#ifdef DTWC_HAS_YAML
    try {
      YAML::Node config = YAML::LoadFile(yaml_config_path);

      // Only override values that were NOT explicitly set on CLI.
      // CLI11 already parsed CLI args; YAML provides defaults for unset options.
      auto set_if_unset = [&](const std::string &key, auto &var) {
        if (config[key]) {
          using T = std::decay_t<decltype(var)>;
          var = config[key].as<T>();
        }
      };

      set_if_unset("input", input_file);
      set_if_unset("output", output_dir);
      set_if_unset("name", prob_name);
      set_if_unset("clusters", n_clusters);
      set_if_unset("method", method);
      set_if_unset("band", band);
      set_if_unset("metric", metric);
      set_if_unset("variant", variant);
      set_if_unset("max-iter", max_iter);
      set_if_unset("n-init", n_init);
      set_if_unset("solver", solver);
      set_if_unset("device", device);
      set_if_unset("precision", precision);
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

      // CLARA parameters
      set_if_unset("sample-size", sample_size);
      set_if_unset("n-samples", n_samples);
      set_if_unset("seed", clara_seed);

      // Hierarchical
      set_if_unset("linkage", linkage_str);

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
    std::cerr << "Error: --input is required (via CLI or YAML config)\n";
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
              << "  Precision:" << precision << "\n";
    if (method == "clara") {
      std::cout << "  CLARA sample_size: "
                << (sample_size < 0 ? "auto" : std::to_string(sample_size)) << "\n"
                << "  CLARA n_samples:   " << n_samples << "\n"
                << "  CLARA seed:        " << clara_seed << "\n";
    }
    if (method == "hierarchical") {
      std::cout << "  Linkage:   " << linkage_str << "\n";
    }
    std::cout << std::flush;
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
  // ---- Apply precision conversion ----
  if (precision_str == "float32" && !prob.data.is_f32() && !prob.data.is_view()) {
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

  if (mmap_threshold == 0 || prob.size() >= mmap_threshold) {
    auto cache_path = fs::path(output_dir) / (prob_name + "_distmat.cache");
    prob.use_mmap_distance_matrix(cache_path);
    if (verbose)
      std::cout << "Using memory-mapped distance matrix: " << cache_path << "\n";
  }

  if (restart) {
    auto ckpt_path = fs::path(output_dir) / (prob_name + "_checkpoint.bin");
    dtwc::core::ClusteringResult ckpt_result;
    if (dtwc::load_binary_checkpoint(ckpt_result, ckpt_path)) {
      if (verbose)
        std::cout << "Loaded checkpoint: " << ckpt_result.iterations
                  << " iterations, cost=" << ckpt_result.total_cost << "\n";
    }
  }

  // ---- Configure DTW ----
  prob.band = band;
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

  // Wire GPU settings from --device and --precision
  if (device.rfind("cuda", 0) == 0) {
    prob.distance_strategy = dtwc::DistanceMatrixStrategy::GPU;
    if (device.size() > 5 && device[4] == ':')
      prob.cuda_settings.device_id = std::stoi(device.substr(5));
    if (precision == "fp32") prob.cuda_settings.precision_mode = 1;
    else if (precision == "fp64") prob.cuda_settings.precision_mode = 2;
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
  }
  prob.set_variant(vparams);

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
  if (device.rfind("cuda", 0) == 0 && !prob.isDistanceMatrixFilled()) {
#ifdef DTWC_HAS_CUDA
    if (!dtwc::cuda::cuda_available()) {
      std::cerr << "Error: --device cuda requested but no CUDA GPU detected.\n";
      return EXIT_FAILURE;
    }

    // Parse device ID from "cuda:N" syntax
    int cuda_device_id = 0;
    if (device.size() > 5 && device[4] == ':')
      cuda_device_id = std::stoi(device.substr(5));

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

    if (precision == "fp32")
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP32;
    else if (precision == "fp64")
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP64;
    // else Auto (default)

    if (verbose)
      std::cout << "Computing distance matrix on GPU ("
                << dtwc::cuda::cuda_device_info(cuda_device_id) << ") ...\n";

    auto cuda_result = dtwc::cuda::compute_distance_matrix_cuda(prob.data.p_vec, cuda_opts);

    // Inject GPU distance matrix into Problem (Dense only)
    auto &dm = prob.dense_distance_matrix();
    dm.resize(cuda_result.n);
    for (size_t i = 0; i < cuda_result.n; ++i)
      for (size_t j = i; j < cuda_result.n; ++j)
        dm.set(i, j, cuda_result.matrix[i * cuda_result.n + j]);
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

    result = dtwc::fast_pam(prob, n_clusters, max_iter);

    if (verbose) {
      std::cout << "FastPAM "
                << (result.converged ? "converged" : "did not converge")
                << " in " << result.iterations << " iterations"
                << ", cost=" << std::setprecision(6) << result.total_cost
                << " [" << clk << "]\n";
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
        clara_opts.use_float32 = (precision_str == "float32");
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
    result.converged = true;
    result.iterations = max_iter; // Lloyd does not report actual iterations

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
