/**
 * @file config.cpp
 * @brief cli::bind(), the one key table of dtwc::Config, and the text forms built on it.
 *
 * @details The only translation unit of dtwc++ that includes CLI11 (linked
 * PRIVATE). Config files, TOML or YAML, are read by dtwc::cli::ConfigFile, the
 * reader dtwc_cl uses, so both formats take the same keys and CLI11 keeps
 * precedence (a flag beats the file) and the unknown-key error.
 *
 * @date 24 Sep 2026
 */

#include "config.hpp"

#include "config_file.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

namespace dtwc {
namespace {

/// Parse a human-readable binary size such as 2G, 500M, or 1.5GiB.
/// Empty/zero means no limit; every malformed, fractional-byte, negative, or
/// unrepresentable value fails closed rather than silently disabling the cap.
/// (CLI11's AsSizeValue would truncate the documented `1.5G`.)
size_t parse_ram_limit(const std::string &s)
{
  if (s.empty()) return 0;

  size_t integer_end = 0;
  while (integer_end < s.size()
         && std::isdigit(static_cast<unsigned char>(s[integer_end])))
    ++integer_end;
  const bool has_integer_digits = integer_end != 0;
  size_t fraction_begin = integer_end;
  size_t fraction_end = integer_end;
  if (fraction_begin < s.size() && s[fraction_begin] == '.') {
    ++fraction_begin;
    fraction_end = fraction_begin;
    while (fraction_end < s.size()
           && std::isdigit(static_cast<unsigned char>(s[fraction_end])))
      ++fraction_end;
    if (fraction_end == fraction_begin)
      throw dtwc::InvalidInput(
        "Invalid --ram-limit '" + s +
        "': expected digits after the decimal point.");
  }
  if (!has_integer_digits && fraction_end == fraction_begin)
    throw dtwc::InvalidInput(
      "Invalid --ram-limit '" + s +
      "': expected a non-negative size such as 2G, 500M, or 1.5GiB.");

  const size_t suffix_begin = fraction_end == integer_end
    ? integer_end : fraction_end;
  std::string suffix = s.substr(suffix_begin);
  for (auto &c : suffix)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

  std::uint64_t multiplier = 1;
  if (suffix.empty() || suffix == "B") multiplier = 1;
  else if (suffix == "K" || suffix == "KB" || suffix == "KIB") multiplier = 1ULL << 10;
  else if (suffix == "M" || suffix == "MB" || suffix == "MIB") multiplier = 1ULL << 20;
  else if (suffix == "G" || suffix == "GB" || suffix == "GIB") multiplier = 1ULL << 30;
  else if (suffix == "T" || suffix == "TB" || suffix == "TIB") multiplier = 1ULL << 40;
  else {
    throw dtwc::InvalidInput(
      "Invalid --ram-limit '" + s +
      "': unit must be B, K/KiB, M/MiB, G/GiB, or T/TiB.");
  }

  const auto platform_max = std::numeric_limits<size_t>::max();
  const auto integer_limit = static_cast<std::uint64_t>(platform_max) / multiplier;
  std::uint64_t integer_part = 0;
  for (size_t i = 0; i < integer_end; ++i) {
    const auto digit = static_cast<unsigned>(s[i] - '0');
    if (integer_part > integer_limit / 10
        || (integer_part == integer_limit / 10
            && digit > integer_limit % 10))
      throw dtwc::InvalidInput(
        "Invalid --ram-limit '" + s +
        "': value exceeds this platform's size limit.");
    integer_part = integer_part * 10 + digit;
  }
  size_t result = static_cast<size_t>(integer_part * multiplier);

  if (fraction_end != fraction_begin) {
    while (fraction_end > fraction_begin && s[fraction_end - 1] == '0')
      --fraction_end;
    if (fraction_end > fraction_begin) {
      std::uint64_t numerator = 0;
      std::uint64_t denominator = 1;
      for (size_t i = fraction_begin; i < fraction_end; ++i) {
        const auto digit = static_cast<unsigned>(s[i] - '0');
        if (numerator > (std::numeric_limits<std::uint64_t>::max() - digit) / 10
            || denominator > std::numeric_limits<std::uint64_t>::max() / 10)
          throw dtwc::InvalidInput(
            "Invalid --ram-limit '" + s +
            "': decimal precision is too large to resolve exactly.");
        numerator = numerator * 10 + digit;
        denominator *= 10;
      }

      std::uint64_t reduced_multiplier = multiplier;
      auto divisor = std::gcd(reduced_multiplier, denominator);
      reduced_multiplier /= divisor;
      denominator /= divisor;
      divisor = std::gcd(numerator, denominator);
      numerator /= divisor;
      denominator /= divisor;
      if (denominator != 1)
        throw dtwc::InvalidInput(
          "Invalid --ram-limit '" + s +
          "': value must resolve to a whole positive byte count.");
      if (numerator != 0
          && reduced_multiplier >
               (static_cast<std::uint64_t>(platform_max) - result) / numerator)
        throw dtwc::InvalidInput(
          "Invalid --ram-limit '" + s +
          "': value exceeds this platform's size limit.");
      result += static_cast<size_t>(numerator * reduced_multiplier);
    }
  }
  return result;
}

/// A bound field as config text: the default `--help` shows and the value
/// to_config_text() writes.
template <class T>
std::string text_of(const T &value)
{
  if constexpr (std::is_same_v<T, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_same_v<T, char>) {
    return value == '\0' ? std::string{} : std::string(1, value);
  } else if constexpr (std::is_same_v<T, double>) {
    std::array<char, 32> buffer{}; // the shortest round-trip form of a double is at most 24 characters
    return std::string(buffer.data(), std::to_chars(buffer.data(), buffer.data() + buffer.size(), value).ptr);
  } else if constexpr (std::is_integral_v<T>) {
    return std::to_string(value);
  } else {
    return value;
  }
}

/// Binds `field` under `flags`; its text is rendered from the field.
template <class T>
CLI::Option *key(CLI::App &app, const std::string &flags, T &field, const std::string &help)
{
  CLI::Option *option = nullptr;
  if constexpr (std::is_same_v<T, bool>)
    option = app.add_flag(flags, field, help);
  else
    option = app.add_option(flags, field, help);
  return option->default_function([&field] { return text_of(field); })->capture_default_str();
}

/// CLI11's view of a Name table: every spelling onto its canonical name.
template <class E, std::size_t N>
std::map<std::string, std::string> choices(const Name<E> (&table)[N])
{
  std::map<std::string, std::string> spellings;
  for (const auto &entry : table) spellings.emplace(entry.text, name_of(table, entry.value));
  return spellings;
}

/// Binds an enum field: CLI11 maps each spelling, ignoring case, to its
/// canonical name, which parse_name() turns into the value.
template <class E, std::size_t N>
CLI::Option *key(CLI::App &app, const std::string &flags, E &field, const Name<E> (&table)[N],
                 std::string_view what, const std::string &help)
{
  return app
    .add_option_function<std::string>(
      flags, [&field, &table, what](const std::string &text) { field = parse_name(table, text, what); }, help)
    ->transform(CLI::CheckedTransformer(choices(table), CLI::ignore_case))
    ->default_function([&field, &table] { return std::string(name_of(table, field)); })
    ->capture_default_str();
}

/// A value as a config-file token that CLI11's reader reads back unchanged.
/// CLI11's writer does that, but writes any byte isprint() rejects, UTF-8 and a
/// tab included, as 'B"(donn\xc3\xa9es.csv)"'; such text becomes a quoted string
/// with UTF-8 kept and control characters escaped.
std::string config_value(const std::string &text)
{
  if (std::all_of(text.begin(), text.end(), [](char c) { return std::isprint(static_cast<unsigned char>(c)) != 0; }))
    return CLI::detail::convert_arg_for_ini(text);
  std::string quoted = "\"";
  for (const char c : CLI::detail::add_escaped_characters(text)) { // \b \t \n \f \r " and backslash
    const auto byte = static_cast<unsigned char>(c);
    if (byte < 0x20 || byte == 0x7f) {
      std::array<char, 8> code{};
      std::snprintf(code.data(), code.size(), "\\u%04x", static_cast<unsigned>(byte));
      quoted += code.data();
    } else {
      quoted += c;
    }
  }
  return quoted + '"';
}

void warn_deprecated(std::string_view old_flag, std::string_view new_flag)
{
  std::cerr << "[dtwc] warning: '" << old_flag << "' is deprecated, use '" << new_flag << "' instead\n";
}

} // namespace

std::string device_text(const Config &config)
{
  std::string text = to_string(config.device);
  if (config.device == Device::GPU && config.gpu.device_id != 0) text += ':' + std::to_string(config.gpu.device_id);
  return text;
}

namespace cli {

void bind(CLI::App &app, Config &config)
{
  app.set_config("--config", "", "Read TOML or YAML configuration file");
  app.config_formatter(std::make_shared<ConfigFile>(&app));
  // A key CLI11 cannot map to an option is a typo, not a comment.
  app.allow_config_extras(CLI::config_extras_mode::error);

  // Input and output
  key(app, "-i,--input", config.input, "Input file (CSV, Parquet, Arrow IPC, .dtws) or folder");
  key(app, "-o,--output", config.output, "Output directory");
  key(app, "--name", config.name, "Problem name (used in output filenames)");
  key(app, "--column", config.column, "Column name to use as time series (Parquet only)");
  key(app, "--dtype,--data-precision,--data-type", config.dtype, core::precision_names, "dtype",
      "Series data type: float64 (default, full precision) or float32 (2x memory saving) (aliases: f32, f64, "
      "float, double)");
  app.add_option_function<std::string>(
       "--ram-limit", [&config](const std::string &text) { config.ram_limit = parse_ram_limit(text); },
       "Max RAM for series data (e.g. 2G, 500M, 128G). Default: no limit.")
    ->default_function([&config] { return text_of(config.ram_limit); })
    ->capture_default_str();

  // Clustering
  CLI::Option *n_clusters = key(app, "-k,--n-clusters", config.k, "Number of clusters")->check(CLI::PositiveNumber);
  app.add_option_function<int>(
       "--clusters",
       [&config, n_clusters](int k) {
         warn_deprecated("--clusters", "--n-clusters");
         if (n_clusters->count() == 0) config.k = k; // the canonical spelling wins
       },
       "DEPRECATED alias of --n-clusters")
    ->group("");
  key(app, "-m,--method", config.method, cluster_method_names, "method",
      "Clustering method: auto, pam, onebatch, clara, kmedoids, mip, lrcore, hierarchical, tadpole");
  key(app, "-b,--band", config.band, "Sakoe-Chiba band width (-1 = full DTW)");
  key(app, "--metric", config.metric, core::metric_names, "metric", "Distance metric: l1, squared_euclidean");
  key(app, "--variant", config.variant.variant, core::variant_names, "variant",
      "DTW variant: standard, ddtw, wdtw, adtw, softdtw, msm, twe");
  key(app, "--max-iter", config.max_iter, "Maximum iterations");
  key(app, "--n-init", config.n_init, "Number of random restarts (PAM/kMedoids)")->check(CLI::PositiveNumber);
  key(app, "--dc", config.tadpole_dc, "TADPole density cutoff distance (default: auto-select)");

  // DTW variant parameters
  key(app, "--wdtw-g", config.variant.wdtw_g, "WDTW logistic weight steepness");
  key(app, "--adtw-penalty", config.variant.adtw_penalty, "ADTW non-diagonal step penalty");
  key(app, "--sdtw-gamma", config.variant.sdtw_gamma, "Soft-DTW smoothing parameter");
  key(app, "--msm-c", config.variant.msm_c, "MSM split/merge cost (default 1.0)");
  key(app, "--twe-nu", config.variant.twe_nu, "TWE stiffness nu (default 0.001)");
  key(app, "--twe-lambda", config.variant.twe_lambda, "TWE edit penalty lambda (default 1.0)");
  key(app, "--mv-mode", config.variant.mv_mode, core::mv_mode_names, "mv mode",
      "Multivariate mode (ndim>1): dependent, independent");
  key(app, "--missing-strategy", config.missing, core::missing_strategy_names, "missing strategy",
      "Missing-data strategy: error, zero_cost, arow, interpolate");

  // CLARA and OneBatchPAM; one seed for every stochastic method and MIP warm starts
  key(app, "--sample-size", config.sample_size, "CLARA subsample size (-1 = auto)");
  key(app, "--n-samples", config.n_samples, "CLARA number of subsamples");
  key(app, "--seed", config.seed, "Random seed for stochastic clustering and MIP warm starts");
  key(app, "--batch-size", config.batch_size, "OneBatchPAM fixed objective batch size (-1 = logarithmic auto)");
  key(app, "--linkage", config.linkage, algorithms::linkage_names, "linkage",
      "Hierarchical linkage: single, complete, average");

  // CSV parsing
  key(app, "--skip-rows", config.skip_rows, "Number of header rows to skip");
  key(app, "--skip-cols", config.skip_cols, "Number of leading columns to skip");
  app.add_option_function<std::string>(
       "--delimiter",
       [&config](const std::string &text) {
         if (text.size() > 1)
           throw InvalidInput("--delimiter takes one character, or \"\" to infer it from the file extension; got '"
                              + text + "'.");
         config.delimiter = text.empty() ? '\0' : text.front();
       },
       "CSV field delimiter, one character (default: inferred from the file extension)")
    ->default_function([&config] { return text_of(config.delimiter); })
    ->capture_default_str();

  // Distance matrix and checkpoints
  key(app, "--dist-matrix", config.dist_matrix, "Path to precomputed distance matrix CSV");
  key(app, "--checkpoint", config.checkpoint, "Checkpoint directory for save/resume");
  key(app, "--checkpoint-interval", config.checkpoint_interval,
      "Save a checkpoint generation every N filled distance-matrix rows (requires --checkpoint)");
  key(app, "--resume", config.resume, "Replay the completed binary result at <output>/<name>_checkpoint.bin");
  app.add_flag_function(
       "--restart",
       [&config](std::int64_t count) {
         warn_deprecated("--restart", "--resume");
         if (count > 0) config.resume = true;
       },
       "DEPRECATED alias of --resume")
    ->group("");
  key(app, "--mmap-threshold", config.mmap_threshold, "N above which to use memory-mapped distance matrix (0=always)");

  // MIP solver
  key(app, "--solver", config.solver, solver_names, "solver", "MIP solver: highs, gurobi");
  key(app, "--mip-gap", config.mip.mip_gap, "MIP optimality gap tolerance (default: 1e-5)");
  key(app, "--time-limit", config.mip.time_limit_sec, "MIP solver time limit in seconds (-1 = unlimited)");
  app.add_flag_function(
       "--no-warm-start", [&config](std::int64_t count) { if (count > 0) config.mip.warm_start = false; },
       "Disable FastPAM warm start for MIP")
    ->default_function([&config] { return text_of(!config.mip.warm_start); })
    ->capture_default_str();
  key(app, "--numeric-focus", config.mip.numeric_focus, "Gurobi NumericFocus (0-3, default: 1)");
  key(app, "--mip-focus", config.mip.mip_focus, "Gurobi MIPFocus (0-3, default: 2)");
  key(app, "--verbose-solver", config.mip.verbose_solver, "Show MIP solver log output");
  key(app, "--lr-max-nodes", config.mip.lr_max_nodes, "Branch-and-bound node cap of --method lrcore");

  // Compute device
  app.add_option_function<std::string>(
       "-d,--device",
       [&config](const std::string &text) {
         const auto [device, index] = dtwc::detail::parse_device(text);
         config.device = device;
         config.gpu.device_id = index;
       },
       "Compute device: cpu, gpu, gpu:N (cuda and cuda:N are the same)")
    ->default_function([&config] { return device_text(config); })
    ->capture_default_str();
  key(app, "--gpu-precision,--gpu-dtype", config.gpu.precision, gpu_precision_names, "gpu precision",
      "GPU kernel precision: auto (default), float32/f32/fp32, float64/f64/fp64/double");

  key(app, "-v,--verbose", config.verbose, "Verbose output");
}

} // namespace cli

std::string to_config_text(const Config &config)
{
  Config bound = config; // bind() takes a mutable Config; rendering only reads it
  CLI::App app;
  cli::bind(app, bound);
  std::string text;
  for (const CLI::Option *option : app.get_options()) {
    // --help and --config are not keys; group "" holds the deprecated spellings.
    if (!option->get_configurable() || option->get_group().empty()) continue;
    text += option->get_single_name() + " = " + config_value(option->get_default_str()) + '\n';
  }
  return text;
}

Config parse_config(const std::vector<std::pair<std::string, std::string>> &pairs)
{
  std::string text;
  for (const auto &[name, value] : pairs) {
    std::string key_name = name;
    for (char &c : key_name) {
      if (c == '_') c = '-';
      // No option name has any other character, and one could start a value, a
      // comment or a second line: reject it as CLI11 rejects an unknown key.
      else if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '-')
        throw InvalidInput("INI was not able to parse " + name);
    }
    text += key_name + " = " + config_value(value) + '\n';
  }

  Config config;
  CLI::App app;
  cli::bind(app, config);
  std::istringstream stream(text);
  try {
    app.parse_from_stream(stream);
  } catch (const CLI::Error &error) {
    throw InvalidInput(error.what());
  }
  return config;
}

} // namespace dtwc
