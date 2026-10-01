/**
 * @file dtwc_cl.cpp
 * @brief Command line interface for DTWC++: cli::bind() reads the command line
 *        and --config files into a dtwc::Config, and dtwc::run() runs it.
 *
 * @details
 *   dtwc_cl --input data.csv -k 5 --method pam -v
 *   dtwc_cl --config config.toml          (or config.yaml: the same keys)
 *   dtwc_cl --config config.toml --print-config > job.toml
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 * @authors Volkan Kumtepeli
 * @authors Becky Perriment
 */

#include "base/timing.hpp"
#include "cli/config.hpp"
#include "cli/run.hpp"

#include <CLI/CLI.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <array>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {

/// The settings in use and the machine, which --verbose prints first (a SLURM
/// .out log's header).
void print_settings(const dtwc::Config &config)
{
  using dtwc::name_of;
  std::cout << "DTWC++ Clustering\n"
            << "  Input:    " << config.input << "\n"
            << "  Output:   " << config.output << "\n"
            << "  Name:     "
            << (config.name.empty() ? dtwc::detail::default_name(dtwc::utf8_to_path(config.input)) : config.name) << "\n"
            << "  Clusters: " << config.k << "\n"
            << "  Method:   " << name_of(dtwc::method_names, config.method) << "\n"
            << "  Band:     " << (config.band < 0 ? "full" : std::to_string(config.band)) << "\n"
            << "  Metric:   " << name_of(dtwc::core::metric_names, config.metric) << "\n"
            << "  Variant:  " << name_of(dtwc::core::variant_names, config.variant.variant) << "\n"
            << "  Missing:  " << name_of(dtwc::core::missing_strategy_names, config.missing) << "\n"
            << "  MaxIter:  " << config.max_iter << "\n"
            << "  N-init:   " << config.n_init << "\n"
            << "  Device:   " << dtwc::device_text(config) << "\n"
            << "  Dtype:    " << name_of(dtwc::core::precision_names, config.dtype) << "\n"
            << "  GPU Prec: " << name_of(dtwc::gpu_precision_names, config.gpu_precision) << "\n";
  if (config.method == dtwc::Method::CLARA)
    std::cout << "  CLARA sample_size: " << (config.sample_size < 0 ? "auto" : std::to_string(config.sample_size))
              << "\n"
              << "  CLARA n_samples:   " << config.n_samples << "\n"
              << "  CLARA seed:        " << config.seed << "\n";
  if (config.method == dtwc::Method::Hierarchical)
    std::cout << "  Linkage:   " << name_of(dtwc::algorithms::linkage_names, config.linkage) << "\n";

  std::cout << "\n=== System Diagnostics ===\n";
#ifdef _OPENMP
  std::cout << "  OpenMP threads:  " << omp_get_max_threads() << "\n";
#else
  std::cout << "  OpenMP:          not available\n";
#endif
  for (const char *name : { "OMP_NUM_THREADS", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_NODELIST",
                            "SLURM_JOB_PARTITION", "SLURM_GPUS" })
    if (const char *value = std::getenv(name)) std::cout << "  " << name << ": " << value << "\n";
#if defined(__linux__)
  std::ifstream cpuinfo("/proc/cpuinfo");
  for (std::string line; std::getline(cpuinfo, line);) {
    if (line.rfind("model name", 0) != 0) continue;
    if (const auto colon = line.find(':'); colon != std::string::npos)
      std::cout << "  CPU:             " << line.substr(colon + 2) << "\n";
    break;
  }
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);)
    if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmPeak:", 0) == 0) std::cout << "  " << line << "\n";
#endif
  std::cout << "==========================\n\n" << std::flush;
}

} // namespace

int main(int argc, char *argv[])
{
  try {
    CLI::App app{ "DTWC++ -- Dynamic Time Warping Clustering" };
    app.set_version_flag("--version", DTWC_VERSION_STRING, "Print the DTWC++ version and exit");
    dtwc::Config config;
    dtwc::cli::bind(app, config);
    bool print_config = false;
    app.add_flag("--print-config", print_config, "Print the settings as a TOML config file and exit")
      ->configurable(false);

    // No arguments: the help, before a config file could supply --input.
    if (argc == 1) {
      std::cout << app.help() << '\n';
      return EXIT_SUCCESS;
    }
    CLI11_PARSE(app, argc, argv);
    if (print_config) {
      std::cout << dtwc::to_config_text(config);
      return EXIT_SUCCESS;
    }

    const dtwc::Clock clock;
    if (config.verbose) print_settings(config);
    const dtwc::Result result = dtwc::run(config);
    // Shortest round-trip form: the printed cost is the exact double, so two runs
    // can be compared bit for bit from their output.
    std::array<char, 32> cost{};
    const char *cost_end = std::to_chars(cost.data(), cost.data() + cost.size(), result.cost()).ptr;

    std::cout << "\n=== Results ===\n"
              << "  Method:     " << dtwc::name_of(dtwc::method_names, result.method()) << "\n";
    std::cout << "  Clusters:   " << config.k << "\n"
              << "  Total cost: " << std::string_view(cost.data(), cost_end - cost.data()) << "\n"
              << "  Converged:  " << (result.converged() ? "yes" : "no") << "\n"
              << "  Iterations: " << result.iterations() << "\n"
              << "  Output:     " << config.output << "/\n"
              << "  Time:       " << clock << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "Error: unknown non-standard exception\n";
  }
  return EXIT_FAILURE;
}
