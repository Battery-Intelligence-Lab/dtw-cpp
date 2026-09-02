/**
 * @file dtwc_cl.cpp
 * @brief Command line interface for DTWC++ with TOML/YAML configuration support.
 *
 * @details Full CLI tool using CLI11 with TOML or YAML config file support. Supports
 * PAM, CLARA, MIP, and hierarchical clustering methods, all DTW variants,
 * checkpointing, and flexible output.
 *
 * Usage:
 *   dtwc_cl --input data.csv -k 5 --method pam -v
 *   dtwc_cl --config config.toml
 *   dtwc_cl --config config.yaml
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 * @authors Volkan Kumtepeli
 * @authors Becky Perriment
 */

#include "dtwc.hpp"
#include "env.hpp"
#include "error.hpp"
#include "core/variant_validation.hpp"
#ifdef DTWC_HAS_MMAP
#include "core/mmap_data_store.hpp"
#endif

#ifdef DTWC_HAS_ARROW
#include "io/arrow_ipc_reader.hpp"
#endif
#ifdef DTWC_HAS_PARQUET
#include "io/parquet_chunk_reader.hpp"
#include "io/parquet_reader.hpp"
#endif
#include "algorithms/detail/fast_clara_plan.hpp"

// CLI11 is only used inside main(). Guard it (and main) behind DTWC_CL_NO_MAIN
// so the pure argument-parsing helpers below can be #included and unit-tested
// without linking CLI11 (see tests/unit/unit_test_cli_args.cpp, Task 0.9).
#ifndef DTWC_CL_NO_MAIN
#include <CLI/CLI.hpp>

#include "cli/config_file.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace fs = std::filesystem;

/// Parse a human-readable binary size such as 2G, 500M, or 1.5GiB.
/// Empty/zero means no limit; every malformed, fractional-byte, negative, or
/// unrepresentable value fails closed rather than silently disabling the cap.
static size_t parse_ram_limit(const std::string &s)
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

constexpr size_t CLI_AUTO_PAM_MAX_SERIES = 5000;

enum class ParquetCliLayout
{
  ListColumn,
  ScalarColumn,
  Directory
};

struct ParquetCliLoadPlan
{
  std::string method;
  size_t series_count = 0;
  size_t estimated_resident_bytes = 0;
  bool stream_payload = false;

  [[nodiscard]] bool materialize_payload() const noexcept
  {
    return !stream_payload;
  }
};

static std::string resolve_cli_auto_method(std::string method, size_t series_count)
{
  if (method == "auto")
    method = series_count <= CLI_AUTO_PAM_MAX_SERIES ? "pam" : "clara";
  return method;
}

[[maybe_unused]] static size_t checked_parquet_series_count(std::int64_t count)
{
  if (count < 0
      || static_cast<std::uint64_t>(count)
           > std::numeric_limits<size_t>::max())
    throw dtwc::InvalidInput(
      "Parquet logical series count exceeds this platform's size limit.");
  return static_cast<size_t>(count);
}

/// Reject a cap no reader can honour. `--ram-limit` governs Parquet series
/// decoding/materialisation; every other input format materialises its series
/// unconditionally, so accepting the flag there would report a limit that is
/// never applied — the very deceit this cap exists to remove.
[[maybe_unused]] static void require_ram_limit_is_applicable(
  size_t ram_limit, bool parquet_file_input, bool parquet_directory_input)
{
  if (ram_limit == 0 || parquet_file_input || parquet_directory_input)
    return;

  throw dtwc::InvalidInput(
    "--ram-limit caps Parquet series materialisation and cannot be honoured "
    "for this input; drop --ram-limit, or convert the series to a "
    "list-per-row Parquet file to stream them under the cap.");
}

/// Reject an input format whose reader this binary does not contain.
///
/// The rejection must exist in the build that LACKS the capability, so it sits
/// under `#ifndef`, like the `.dtws`/`DTWC_HAS_MMAP` branch. Inside
/// `#ifdef DTWC_HAS_PARQUET` it would be absent from the canonical
/// `DTWC_ENABLE_ARROW=OFF` gate build and `-i x.parquet` would fall through to
/// the CSV DataLoader, parsing Parquet bytes as text (LESSONS F9).
static void require_input_format_is_built(
  [[maybe_unused]] bool parquet_input, [[maybe_unused]] bool arrow_ipc_input)
{
#ifndef DTWC_HAS_PARQUET
  if (parquet_input)
    throw dtwc::InvalidInput(
      "Parquet input (.parquet/.pq) requires a build with Arrow/Parquet "
      "(-DDTWC_ENABLE_ARROW=ON). This binary was built without Parquet "
      "support; convert the input to CSV/TSV or use an Arrow-enabled build.");
#endif
#ifndef DTWC_HAS_ARROW
  if (arrow_ipc_input)
    throw dtwc::InvalidInput(
      "Arrow IPC input (.arrow/.ipc/.feather) requires a build with Arrow "
      "(-DDTWC_ENABLE_ARROW=ON). This binary was built without Arrow support; "
      "convert the input to CSV/TSV or use an Arrow-enabled build.");
#endif
}

/// Reject a parsing option the selected input format cannot honour.
///
/// `--column` is read only on Parquet routes; `--skip-rows`/`--skip-cols` only
/// in the DataLoader branch. Accepting and then ignoring them elsewhere is the
/// same silent deceit `require_ram_limit_is_applicable` exists to remove.
static void require_format_options_are_applicable(
  const std::string &parquet_column, int skip_rows, int skip_cols,
  bool parquet_input, bool text_input)
{
  if (!parquet_column.empty() && !parquet_input)
    throw dtwc::InvalidInput(
      "--column selects a Parquet column and cannot be honoured for this "
      "input; drop --column, or pass a .parquet/.pq file or directory.");
  if ((skip_rows != 0 || skip_cols != 0) && !text_input)
    throw dtwc::InvalidInput(
      "--skip-rows/--skip-cols are CSV/TSV parsing options and cannot be "
      "honoured for this input; drop them, or pass a text input.");
}

/// Decide from Parquet metadata alone whether reading the payload is legal.
/// The cap applies to resident series storage. Only list-per-row data in one
/// file has a valid row-group streaming implementation.
[[maybe_unused]] static ParquetCliLoadPlan resolve_parquet_cli_plan(
  std::string method,
  size_t series_count,
  size_t estimated_resident_bytes,
  size_t ram_limit,
  ParquetCliLayout layout)
{
  if (series_count == 0)
    throw dtwc::InvalidInput("Parquet input contains no time series.");

  method = resolve_cli_auto_method(std::move(method), series_count);
  ParquetCliLoadPlan plan{
    std::move(method), series_count, estimated_resident_bytes, false };
  if (ram_limit == 0 || estimated_resident_bytes <= ram_limit)
    return plan;

  if (plan.method != "clara") {
    throw dtwc::InvalidInput(
      "Parquet input needs approximately " +
      std::to_string(estimated_resident_bytes) +
      " bytes of resident series storage, exceeding --ram-limit=" +
      std::to_string(ram_limit) + "; method '" + plan.method +
      "' cannot stream it. Use --method clara with a single list-per-row "
      "Parquet file, or raise --ram-limit.");
  }
  if (layout == ParquetCliLayout::ScalarColumn) {
    throw dtwc::InvalidInput(
      "RAM-limited FastCLARA streaming requires list-per-row Parquet "
      "(one list cell per time series); a scalar column is one time series "
      "whose rows cannot be clustered as independent series.");
  }
  if (layout == ParquetCliLayout::Directory) {
    throw dtwc::InvalidInput(
      "RAM-limited FastCLARA streaming currently requires a single Parquet file; "
      "a Parquet directory exceeds --ram-limit. Convert it to one list-per-row "
      "Parquet file or raise --ram-limit.");
  }

  plan.stream_payload = true;
  return plan;
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

/// Validate normalized distance semantics before Env, data, or cache side
/// effects. It also rejects values that reach the kernels unvalidated.
static std::string validate_cli_distance_configuration(
  const std::string &variant, const std::string &metric,
  const std::string &mv_mode, const std::string &missing_strategy, bool is_cuda)
{
  const bool known_variant = variant == "standard" || variant == "ddtw"
    || variant == "wdtw" || variant == "adtw" || variant == "softdtw"
    || variant == "msm" || variant == "twe";
  if (!known_variant) return "unsupported --variant '" + variant + "'";
  if (metric != "l1" && metric != "squared_euclidean")
    return "unsupported --metric '" + metric + "'";
  if (mv_mode != "dependent" && mv_mode != "independent")
    return "unsupported --mv-mode '" + mv_mode + "'";
  if (missing_strategy != "error" && missing_strategy != "zero_cost"
      && missing_strategy != "arow" && missing_strategy != "interpolate")
    return "unsupported --missing-strategy '" + missing_strategy + "'";
  if (mv_mode == "independent"
      && (variant != "standard" || missing_strategy != "error"))
    return "--mv-mode independent requires --variant standard and "
           "--missing-strategy error";
  if (variant != "standard" && missing_strategy != "error")
    return "non-standard --variant cannot be combined with a non-error "
           "--missing-strategy";
  if (const auto metric_error = validate_metric_for_device(metric, is_cuda);
      !metric_error.empty())
    return metric_error;
  if (is_cuda && variant != "standard")
    return "--device cuda supports --variant standard only";
  if (is_cuda && missing_strategy != "error")
    return "--device cuda does not support --missing-strategy";
  if (is_cuda && mv_mode != "dependent")
    return "--device cuda does not support --mv-mode independent";
  return "";
}

/// Validate the three route selectors that choose an algorithm, a MIP solver,
/// and a linkage rule. Returns an empty string when OK.
///
/// `method`, `solver` and `linkage` reach their dispatch chains as raw strings,
/// a value that bypasses CLI11's CheckedTransformer used to select nothing at
/// all: an unknown `method` left `result`
/// default-constructed yet still wrote label files, an unknown `solver` stayed
/// on HiGHS and an unknown `linkage` became Average. Reject before any data,
/// cache, or filesystem side effect.
static std::string validate_cli_route_selectors(
  const std::string &method, const std::string &solver,
  const std::string &linkage)
{
  const bool known_method = method == "auto" || method == "pam"
    || method == "onebatch" || method == "clara" || method == "kmedoids"
    || method == "mip" || method == "lrcore" || method == "hierarchical"
    || method == "tadpole";
  if (!known_method)
    return "unsupported --method '" + method
      + "'. Valid: auto, pam, onebatch, clara, kmedoids, mip, lrcore, "
        "hierarchical, tadpole.";
  if (solver != "highs" && solver != "gurobi")
    return "unsupported --solver '" + solver + "'. Valid: highs, gurobi.";
  if (linkage != "single" && linkage != "complete" && linkage != "average")
    return "unsupported --linkage '" + linkage
      + "'. Valid: single, complete, average.";
  return "";
}

/// Run the CLI PAM route with invocation-local `--seed + restart` engines and
/// retain the strict best objective. Keeping this seam outside main() lets the
/// CLI regression exercise the production dispatch without depending on CLI11
/// in the unit-test executable.
static dtwc::core::ClusteringResult run_cli_pam(
  dtwc::Problem &prob, int n_clusters, int max_iter)
{
  const int n_init = prob.n_repetitions();
  const std::uint64_t random_seed = prob.random_seed();
  if (n_init < 1)
    throw dtwc::InvalidInput("run_cli_pam: n_init must be at least 1.");
  const auto restart_offset = static_cast<std::uint64_t>(n_init - 1);
  if (restart_offset > std::numeric_limits<std::uint64_t>::max() - random_seed)
    throw dtwc::InvalidInput(
      "run_cli_pam: random_seed + n_init - 1 overflows uint64.");

  auto best = dtwc::fast_pam_seeded(
    prob, n_clusters, random_seed, max_iter);
  for (int restart = 1; restart < n_init; ++restart) {
    auto candidate = dtwc::fast_pam_seeded(
      prob, n_clusters, random_seed + static_cast<std::uint64_t>(restart),
      max_iter);
    // Strict comparison deliberately retains the earlier seed on a tie.
    if (candidate.total_cost < best.total_cost) best = std::move(candidate);
  }
  return best;
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
  bool legacy_distance_matrix_requested = false,
  bool clara_uses_full_sample = false)
{
  (void)cache_path;
  (void)cache_metric;
  if (method == "clara" && !clara_uses_full_sample) {
    if (legacy_checkpoint_requested || legacy_distance_matrix_requested)
      throw std::runtime_error(
        "Non-full FastCLARA does not consume a parent distance matrix; "
        "--checkpoint and --dist-matrix would load or save unused O(N^2) "
        "state. Omit those options, or request a full sample deliberately.");
    return std::nullopt;
  }
  if (method == "onebatch")
    return std::nullopt;
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

/// Validate the structurally loadable binary result before the CLI applies it.
/// Binary v1 has no input/configuration identity, so this pins only the shape
/// and field invariants that can be proven without changing the frozen format.
static std::string validate_cli_resume_result(
  const dtwc::core::ClusteringResult &result,
  size_t expected_series,
  int expected_clusters)
{
  if (result.labels.size() != expected_series) {
    return "Binary result checkpoint has "
      + std::to_string(result.labels.size()) + " labels; current input has "
      + std::to_string(expected_series) + " series.";
  }
  if (expected_clusters <= 0
      || result.medoid_indices.size()
           != static_cast<size_t>(expected_clusters)) {
    return "Binary result checkpoint has "
      + std::to_string(result.medoid_indices.size())
      + " medoids; --n-clusters requests "
      + std::to_string(expected_clusters) + ".";
  }

  for (size_t i = 0; i < result.labels.size(); ++i) {
    const int label = result.labels[i];
    if (label < 0 || label >= expected_clusters) {
      return "Binary result checkpoint label[" + std::to_string(i) + "]="
        + std::to_string(label) + " is outside [0,"
        + std::to_string(expected_clusters) + ").";
    }
  }
  for (size_t i = 0; i < result.medoid_indices.size(); ++i) {
    const int medoid = result.medoid_indices[i];
    if (medoid < 0 || static_cast<size_t>(medoid) >= expected_series) {
      return "Binary result checkpoint medoid[" + std::to_string(i) + "]="
        + std::to_string(medoid) + " is outside [0,"
        + std::to_string(expected_series) + ").";
    }
    for (size_t previous = 0; previous < i; ++previous) {
      if (result.medoid_indices[previous] == medoid) {
        return "Binary result checkpoint medoid index "
          + std::to_string(medoid) + " is duplicated.";
      }
    }
  }
  if (result.iterations < 0) {
    return "Binary result checkpoint iteration count "
      + std::to_string(result.iterations) + " is negative.";
  }
  if (!std::isfinite(result.total_cost))
    return "Binary result checkpoint total cost is not finite.";
  return {};
}

/// Convert float64 Data to an owning float32 copy.
static dtwc::Data convert_to_f32(const dtwc::Data &data_f64)
{
  const size_t n = data_f64.size();
  std::vector<std::vector<float>> vecs_f32(n);
  for (size_t i = 0; i < n; ++i) {
    const auto &src = data_f64.p_vec[i];
    vecs_f32[i].resize(src.size());
    for (size_t j = 0; j < src.size(); ++j)
      vecs_f32[i][j] = static_cast<float>(src[j]);
  }
  auto names = data_f64.p_names;
  return dtwc::Data(std::move(vecs_f32), std::move(names), data_f64.ndim);
}

/// Write cluster labels to CSV: one line per point with "name,cluster_id".
static std::string output_series_name(
  const dtwc::Problem &prob,
  size_t index,
  std::optional<size_t> streamed_series_count)
{
  if (streamed_series_count) {
    if (index >= *streamed_series_count) {
      throw std::runtime_error(
        "Result index " + std::to_string(index) + " is outside the " +
        std::to_string(*streamed_series_count) + "-series input.");
    }
    // Matches parquet_reader.hpp and ParquetChunkReader exactly for list rows.
    return "series_" + std::to_string(index);
  }
  if (index >= prob.size()) {
    throw std::runtime_error(
      "Result index " + std::to_string(index) + " is outside the " +
      std::to_string(prob.size()) + "-series input.");
  }
  return std::string(prob.get_name(index));
}

static void write_labels_csv(const fs::path &path,
                             const dtwc::Problem &prob,
                             const dtwc::core::ClusteringResult &result,
                             std::optional<size_t> streamed_series_count = std::nullopt)
{
  const size_t expected = streamed_series_count.value_or(prob.size());
  if (result.labels.size() != expected) {
    throw std::runtime_error(
      "Clustering result has " + std::to_string(result.labels.size()) +
      " labels for a " + std::to_string(expected) + "-series input.");
  }
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot open output file: " + path.string());

  out << "name,cluster\n";
  for (size_t i = 0; i < result.labels.size(); ++i) {
    out << output_series_name(prob, i, streamed_series_count)
        << "," << result.labels[i] << "\n";
  }
}

/// Write medoid information to CSV.
static void write_medoids_csv(const fs::path &path,
                              const dtwc::Problem &prob,
                              const dtwc::core::ClusteringResult &result,
                              std::optional<size_t> streamed_series_count = std::nullopt)
{
  for (const int idx : result.medoid_indices) {
    if (idx < 0)
      throw std::runtime_error(
        "Result medoid index " + std::to_string(idx) + " is negative.");
    (void)output_series_name(
      prob, static_cast<size_t>(idx), streamed_series_count);
  }
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot open output file: " + path.string());

  out << "cluster,medoid_index,medoid_name\n";
  for (int c = 0; c < result.n_clusters(); ++c) {
    int idx = result.medoid_indices[c];
    out << c << "," << idx << ","
        << output_series_name(
             prob, static_cast<size_t>(idx), streamed_series_count)
        << "\n";
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
// The CLI flag set and the TOML config keys are a de-facto API (they are
// composed by `scripts/slurm/jobs/cluster_generic.slurm` and
// `python/dtwcpp/_hpc.py::build_dtwc_command`). When a flag/key is renamed to the
// 2.0 contract vocabulary the OLD spelling stays ACCEPTED but emits a one-line
// deprecation warning to stderr pointing at the new spelling (one warning per
// use). TOML keys are the long-flag names without the leading "--", so a
// single table below drives the CLI and the `--config` TOML path (CLI11 maps
// config keys onto the same options).
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
/// used. Format is fixed and shared by every rename (CLI flag AND TOML key).
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
/// TOML "key" form (no leading dashes).
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
static int run_cli_main(int argc, char *argv[])
{
  CLI::App app{"DTWC++ -- Dynamic Time Warping Clustering"};
  app.set_version_flag("--version", DTWC_VERSION_STRING,
                       "Print the DTWC++ version and exit");

  // Config file support (CLI11 built-in, processes before parsing). Keys are the
  // canonical long-flag names without "--" (e.g. `n-clusters`, `max-iter`), in
  // TOML or YAML; dtwc::cli::ConfigFile sniffs which. CLI11 owns precedence, so
  // a value given on the command line always beats the file. Deprecated keys are
  // accepted with a warning per cli_renames():
  //   clusters -> n-clusters,  restart -> resume.
  app.set_config("--config", "", "Read TOML or YAML configuration file");
  app.config_formatter(std::make_shared<dtwc::cli::ConfigFile>(&app));
  // A key CLI11 cannot map to an option is a typo, not a comment: fail loudly
  // instead of running with a silently ignored setting.
  app.allow_config_extras(CLI::config_extras_mode::error);

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
  app.add_option("--n-init", n_init, "Number of random restarts (PAM/kMedoids)")
      ->check(CLI::PositiveNumber);
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
  std::string missing_strategy = "error";
  app.add_option("--missing-strategy", missing_strategy,
                 "Missing-data strategy: error, zero_cost, arow, interpolate")
    ->transform(CLI::CheckedTransformer(
        std::map<std::string, std::string>{
            {"error", "error"}, {"zero_cost", "zero_cost"},
            {"zero-cost", "zero_cost"}, {"zerocost", "zero_cost"},
            {"arow", "arow"}, {"interpolate", "interpolate"}},
        CLI::ignore_case));

  // One invocation-local seed spelling covers PAM, OneBatchPAM, CLARA, Lloyd,
  // and MIP warm starts.
  int sample_size = -1;
  int n_samples = 5;
  unsigned clara_seed = dtwc::settings::DEFAULT_RANDOM_SEED;
  app.add_option("--sample-size", sample_size, "CLARA subsample size (-1 = auto)");
  app.add_option("--n-samples", n_samples, "CLARA number of subsamples");
  app.add_option("--seed", clara_seed,
                 "Random seed for stochastic clustering and MIP warm starts")
      ->check(CLI::Range(0u, std::numeric_limits<unsigned>::max()));

  // OneBatchPAM-specific. Reproducibility keeps the shared CLI seed above.
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
  int checkpoint_interval = 0;
  auto *checkpoint_interval_opt = app.add_option(
    "--checkpoint-interval", checkpoint_interval,
    "Save a checkpoint generation every N filled distance-matrix rows "
    "(requires --checkpoint)");

  // Binary checkpoint resume & mmap threshold
  bool resume = false;
  size_t mmap_threshold = 50000;
  app.add_flag(
    "--resume", resume,
    "Replay the completed binary result at <output>/<name>_checkpoint.bin");
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

  // MIP solver settings (kebab-case: matches CLI flags and TOML keys)
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

  // Without a transformer, `--benders ON`, `true` or the typo `of` all reach
  // Problem::cluster_by_mip(), whose test is `benders == "on"`, and silently
  // mean OFF on a large MIP job.
  std::string benders_mode = "auto";
  app.add_option("--benders", benders_mode, "Benders decomposition: auto (N>200), on, off")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, std::string>{
              {"auto", "auto"}, {"on", "on"}, {"off", "off"},
              {"true", "on"}, {"false", "off"},
              {"yes", "on"}, {"no", "off"},
              {"1", "on"}, {"0", "off"}},
          CLI::ignore_case));

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
  // is not required yet — a config file may provide it).
  if (argc == 1) {
    std::cout << app.help() << '\n';
    return EXIT_SUCCESS;
  }

  CLI11_PARSE(app, argc, argv);

  // ---- CLI/TOML flag deprecations (api-contract-2.0.md §4, §7 item 3) ----
  // Old spellings are accepted from the command line AND from a --config file
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

  // ---- Post-parse validation (catches both CLI and config-file values) ----
  if (input_file.empty()) {
    std::cerr << "Error: --input is required via CLI or config file (TOML or YAML)\n";
    return EXIT_FAILURE;
  }
  if (n_clusters < 1) {
    std::cerr << "Error: --clusters must be a positive integer\n";
    return EXIT_FAILURE;
  }
  if (n_init < 1) {
    std::cerr << "Error: --n-init must be a positive integer\n";
    return EXIT_FAILURE;
  }

  // Normalize selector spellings that can reach the dispatch chains without
  // passing through CLI11's CheckedTransformer (case-folding AND aliases).
  auto to_lower = [](std::string &s) {
    for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  };
  to_lower(method);
  to_lower(metric);
  to_lower(variant);
  to_lower(mv_mode);
  to_lower(missing_strategy);
  to_lower(linkage_str);

  // Alias mappings (mirror the CheckedTransformer maps above)
  if (method == "hclust") method = "hierarchical";
  if (method == "obp") method = "onebatch";
  if (method == "lr") method = "lrcore";
  if (metric == "sqeuclidean" || metric == "l2sq") metric = "squared_euclidean";
  if (variant == "soft-dtw") variant = "softdtw";
  if (missing_strategy == "zero-cost" || missing_strategy == "zerocost")
    missing_strategy = "zero_cost";

  // Normalize dtype/gpu-precision aliases (may bypass CheckedTransformer)
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
  if (const std::string route_error =
        validate_cli_route_selectors(method, solver, linkage_str);
      !route_error.empty()) {
    std::cerr << "Error: " << route_error << "\n";
    return EXIT_FAILURE;
  }
  if (const std::string config_error = validate_cli_distance_configuration(
        variant, metric, mv_mode, missing_strategy, dev.is_cuda);
      !config_error.empty()) {
    std::cerr << "Error: " << config_error << "\n";
    return EXIT_FAILURE;
  }

  // Materialize and validate the complete parameter value object before Env,
  // output-directory, input, cache, or distance effects.  Validate inactive
  // fields too: all CLI/config values are public and participate in cache identity.
  dtwc::core::DTWVariantParams vparams;
  if (variant == "ddtw")
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
  // Inactive options still belong to the aggregate public value object.
  vparams.wdtw_g = wdtw_g;
  vparams.adtw_penalty = adtw_penalty;
  vparams.sdtw_gamma = sdtw_gamma;
  vparams.msm_c = msm_c;
  vparams.twe_nu = twe_nu;
  vparams.twe_lambda = twe_lambda;
  vparams.mv_mode = (mv_mode == "independent") ? dtwc::core::MVMode::Independent
                                               : dtwc::core::MVMode::Dependent;
  try {
    dtwc::core::validate_variant_params(vparams);
  } catch (const dtwc::InvalidInput &error) {
    std::cerr << "Error: " << error.what() << "\n";
    return EXIT_FAILURE;
  }

  // Parse before Env, output-directory creation, or payload I/O. An invalid
  // cap must never degrade to the old unlimited behaviour.
  size_t ram_limit = 0;
  try {
    ram_limit = parse_ram_limit(ram_limit_str);
  } catch (const dtwc::InvalidInput &error) {
    std::cerr << "Error: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
  const bool auto_method_requested = method == "auto";

  dtwc::algorithms::CLARAOptions clara_opts;
  clara_opts.n_clusters = n_clusters;
  clara_opts.sample_size = sample_size;
  clara_opts.n_samples = n_samples;
  clara_opts.max_iter = max_iter;
  clara_opts.random_seed = clara_seed;
  if (method == "clara")
    dtwc::algorithms::detail::validate_clara_controls(clara_opts, "dtwc_cl");
  bool clara_plan_resolved = false;
  bool clara_uses_full_sample = false;

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
              << "  Missing:  " << missing_strategy << "\n"
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
  // The CLI has separate, already-frozen controls for distance-matrix mmap and
  // Parquet materialisation. It exposes no series-storage policy, so preserve
  // its historical resident-series behavior explicitly.
  prob.set_storage_policy(dtwc::core::StoragePolicy::Heap);

  const bool is_dir = fs::is_directory(input_file);
  auto input_ext = is_dir ? "" : fs::path(input_file).extension().string();
  std::transform(input_ext.begin(), input_ext.end(), input_ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  bool stream_parquet_payload = false;
  size_t input_series_count = 0;
  [[maybe_unused]] size_t parquet_resident_estimate = 0;
  auto resolve_clara_plan_for_input = [&]() {
    if (method != "clara" || clara_plan_resolved) return;
    if (input_series_count > static_cast<size_t>(
          std::numeric_limits<std::int64_t>::max()))
      throw dtwc::InvalidInput(
        "FastCLARA input count exceeds the int64 metadata limit.");
    const auto plan = dtwc::algorithms::detail::resolve_clara_plan(
      static_cast<std::int64_t>(input_series_count), clara_opts, "dtwc_cl");
    clara_uses_full_sample = plan.sample_size == plan.n_points;
    if (stream_parquet_payload)
      dtwc::algorithms::detail::validate_streaming_clara_plan(plan, "dtwc_cl");
    clara_plan_resolved = true;
  };

  // Classify the input by filesystem inspection alone. This must stay outside
  // DTWC_HAS_PARQUET: the cap has to be rejected on a build without Parquet too,
  // where nothing could ever apply it.
  [[maybe_unused]] std::vector<fs::path> parquet_directory_files;
  if (is_dir) {
    for (const auto &e : fs::directory_iterator(input_file)) {
      auto ext = e.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (ext == ".parquet" || ext == ".pq")
        parquet_directory_files.push_back(e.path());
    }
    std::sort(parquet_directory_files.begin(), parquet_directory_files.end());
  }

  [[maybe_unused]] const bool parquet_file_input = !is_dir
    && (input_ext == ".parquet" || input_ext == ".pq");
  [[maybe_unused]] const bool parquet_directory_input =
    !parquet_directory_files.empty();

  const bool arrow_ipc_input = !is_dir
    && (input_ext == ".arrow" || input_ext == ".ipc" || input_ext == ".feather");
  const bool dtws_input = !is_dir && input_ext == ".dtws";
  const bool parquet_input = parquet_file_input || parquet_directory_input;
  // Anything the typed readers do not claim routes to the CSV/TSV DataLoader.
  const bool text_input = !parquet_input && !arrow_ipc_input && !dtws_input;

  require_input_format_is_built(parquet_input, arrow_ipc_input);
  require_ram_limit_is_applicable(
    ram_limit, parquet_file_input, parquet_directory_input);
  require_format_options_are_applicable(
    parquet_column, skip_rows, skip_cols, parquet_input, text_input);

#ifdef DTWC_HAS_PARQUET
  // A non-zero cap changes the load decision, so inspect only Parquet metadata
  // before touching the selected column payload. MemoryMappedFile maps the
  // file and FileReader metadata; no ReadTable/ReadRowGroups call occurs here.
  const bool inspect_parquet_metadata = parquet_file_input || parquet_directory_input;
  if (inspect_parquet_metadata
      && (ram_limit > 0 || method == "auto" || method == "clara")) {
    auto saturating_add = [](size_t lhs, size_t rhs) {
      return rhs > std::numeric_limits<size_t>::max() - lhs
        ? std::numeric_limits<size_t>::max() : lhs + rhs;
    };

    ParquetCliLayout layout = ParquetCliLayout::Directory;
    if (parquet_file_input) {
      dtwc::io::ParquetChunkReader metadata(input_file, parquet_column);
      layout = metadata.is_list_layout()
        ? ParquetCliLayout::ListColumn : ParquetCliLayout::ScalarColumn;
      input_series_count = checked_parquet_series_count(
        metadata.logical_series_count());
      parquet_resident_estimate =
        metadata.estimated_materialization_peak_bytes(dtype_str == "float32");
    } else {
      for (const auto &path : parquet_directory_files) {
        dtwc::io::ParquetChunkReader metadata(path, parquet_column);
        input_series_count = saturating_add(
          input_series_count,
          checked_parquet_series_count(metadata.logical_series_count()));
        parquet_resident_estimate = saturating_add(
          parquet_resident_estimate,
          metadata.estimated_materialization_peak_bytes(dtype_str == "float32"));
      }
    }

    const auto plan = resolve_parquet_cli_plan(
      method, input_series_count, parquet_resident_estimate,
      ram_limit, layout);
    method = plan.method;
    stream_parquet_payload = plan.stream_payload;
    if (stream_parquet_payload) {
      clara_opts.ram_limit_bytes = ram_limit;
      clara_opts.parquet_path = input_file;
      clara_opts.parquet_column = parquet_column;
      clara_opts.use_float32 = (dtype_str == "float32");
      clara_opts.force_parquet_streaming = true;
    }
    resolve_clara_plan_for_input();

    if (method == "clara" && dev.is_cuda && !clara_uses_full_sample)
      throw dtwc::InvalidInput(
        "Non-full FastCLARA uses a matrix-free CPU distance schedule; "
        "--device cuda is supported only by its full-sample PAM fallback.");

    if (stream_parquet_payload && !checkpoint_dir.empty())
      throw dtwc::InvalidInput(
        "--checkpoint requires resident series data and cannot be combined "
        "with RAM-limited Parquet streaming; the binary clustering-result "
        "checkpoint is still written automatically.");
    if (stream_parquet_payload && !dist_mat_path.empty())
      throw dtwc::InvalidInput(
        "--dist-matrix requires resident series data and cannot be combined "
        "with RAM-limited Parquet streaming.");

    if (verbose && stream_parquet_payload) {
      std::cout << "Parquet metadata selected streaming: "
                << input_series_count << " series, ~"
                << (parquet_resident_estimate / (1ULL << 20))
                << " MB resident estimate exceeds the series-data cap ["
                << clk << "]\n";
    }
  }

  if (parquet_directory_input) {
    prob.set_data(dtwc::io::load_parquet_directory(input_file, parquet_column));
    if (verbose)
      std::cout << "Data loaded from Parquet directory: " << prob.size() << " series [" << clk << "]\n";
  }
  else
#endif
  if (dtws_input) {
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
  else if (arrow_ipc_input) {
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
  else if (parquet_file_input) {
    if (!stream_parquet_payload) {
      prob.set_data(dtwc::io::load_parquet_file(input_file, parquet_column));
      if (verbose)
        std::cout << "Data loaded from Parquet: " << prob.size() << " series [" << clk << "]\n";
    }
  }
#endif
  else {
    // Default: CSV/TSV via DataLoader
    dtwc::DataLoader dl{input_file};
    dl.start_column(skip_cols).start_row(skip_rows);
    prob.set_data(dl.load());
    if (verbose)
      std::cout << "Data loaded: " << prob.size() << " series [" << clk << "]\n";
  }

  if (!stream_parquet_payload)
    input_series_count = prob.size();

  if (verbose && prob.size() > 0) {
    size_t total_elements = 0;
    for (const auto &v : prob.data().p_vec) total_elements += v.size();
    size_t data_bytes = total_elements * sizeof(dtwc::data_t);
    std::cout << "  Data memory: ~" << (data_bytes / (1ULL << 20)) << " MB ("
              << prob.size() << " series, "
              << (total_elements / prob.size()) << " avg length, "
              << dtype_str << ")\n";
  }

  // ---- Apply precision conversion ----
  if (!stream_parquet_payload && dtype_str == "float32"
      && !prob.data().is_f32() && !prob.data().is_view()) {
    prob.set_data(convert_to_f32(prob.data()));
    if (verbose)
      std::cout << "Converted to float32 (2x memory saving)\n";
  }

  if (ram_limit > 0 && verbose)
    std::cout << "Series-data RAM limit: " << ram_limit << " bytes\n";

  // ---- Auto method selection ----
  if (method == "auto") {
    const size_t N = input_series_count;
    method = resolve_cli_auto_method(std::move(method), N);
  }
  if (auto_method_requested && verbose)
    std::cout << "Auto-selected method: " << method
              << " (N=" << input_series_count << ")\n";

  resolve_clara_plan_for_input();

  const bool matrix_free_method = method == "onebatch" || method == "tadpole"
    || (method == "clara" && !clara_uses_full_sample);

  const auto binary_checkpoint_path =
    fs::path(output_dir) / (prob_name + "_checkpoint.bin");
  std::optional<dtwc::core::ClusteringResult> resumed_result;
  if (resume) {
    dtwc::core::ClusteringResult candidate;
    if (!dtwc::load_binary_checkpoint(candidate, binary_checkpoint_path)) {
      throw dtwc::InvalidInput(
        "--resume requires a readable binary result checkpoint at '"
        + binary_checkpoint_path.string()
        + "'. Omit --resume to start a new clustering run.");
    }
    if (const auto error = validate_cli_resume_result(
          candidate, input_series_count, n_clusters);
        !error.empty()) {
      throw dtwc::InvalidInput(error);
    }
    resumed_result.emplace(std::move(candidate));
  }
  const bool replaying_result = resumed_result.has_value();

  // ---- Configure DTW ----
  prob.set_band(band);
  prob.set_max_iter(max_iter);
  prob.set_n_repetitions(n_init);
  prob.set_random_seed(static_cast<std::uint64_t>(clara_seed));
  prob.set_output_folder(output_dir);
  prob.set_verbose(verbose);

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

  if (missing_strategy == "zero_cost")
    prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  else if (missing_strategy == "arow")
    prob.missing_strategy = dtwc::core::MissingStrategy::AROW;
  else if (missing_strategy == "interpolate")
    prob.missing_strategy = dtwc::core::MissingStrategy::Interpolate;
  else
    prob.missing_strategy = dtwc::core::MissingStrategy::Error;

  // Set the already validated DTW variant value object.
  prob.set_variant(vparams);

  // Bind persistent distance storage only after every distance-affecting CLI
  // option has reached Problem. Opening earlier made the cache identity observe
  // the default band/variant/backend rather than the user's configuration.
  const auto cache_metric = metric == "squared_euclidean"
    ? dtwc::core::MetricType::SquaredL2
    : dtwc::core::MetricType::L1;
  const auto mmap_cache_path =
    fs::path(output_dir) / (prob_name + "_distmat.cache");
  // Result replay never creates unused O(N^2) state. An existing mmap cache
  // can still reopen independently for scoring; explicit dense imports and
  // directory checkpoints use the default dense destination below.
  const bool reopen_replay_mmap =
    replaying_result
    && checkpoint_dir.empty()
    && dist_mat_path.empty()
    && fs::is_regular_file(mmap_cache_path);
  if (!replaying_result || reopen_replay_mmap) {
    try {
      const auto mmap_cache = configure_cli_distance_storage(
        prob, method, mmap_threshold, mmap_cache_path, cache_metric,
        !checkpoint_dir.empty(), !dist_mat_path.empty(), clara_uses_full_sample);
      if (mmap_cache && verbose)
        std::cout << "Using memory-mapped distance matrix: " << *mmap_cache << "\n";
    } catch (const std::exception &e) {
      std::cerr << "Error: " << e.what() << "\n";
      return EXIT_FAILURE;
    }
  }
  // Set MIP solver (relevant for method=mip). The terminal else keeps an
  // unknown selector from silently leaving the default solver in place.
  if (solver == "highs")
    prob.set_solver(dtwc::Solver::HiGHS);
  else if (solver == "gurobi")
    prob.set_solver(dtwc::Solver::Gurobi);
  else
    throw dtwc::InvalidInput("unsupported --solver '" + solver + "'");

  // ---- Load precomputed distance matrix if provided ----
  if (!dist_mat_path.empty()) {
    try {
      prob.read_distance_matrix(dist_mat_path);
      if (verbose)
        std::cout << "Loaded distance matrix from " << dist_mat_path << "\n";
    } catch (const std::exception &e) {
      std::cerr << "Warning: Could not load distance matrix: " << e.what()
                << "\nContinuing without precomputed matrix.\n";
    }
  }

  // ---- Automatic mid-fill checkpointing ----
  // Opt-in: --checkpoint alone keeps the historical save-once-at-the-end
  // behaviour; the interval flag is what enables periodic saves.
  if (checkpoint_interval_opt->count() > 0) {
    if (checkpoint_dir.empty()) {
      std::cerr << "Error: --checkpoint-interval requires --checkpoint <dir>.\n";
      return EXIT_FAILURE;
    }
    prob.checkpoint.directory = checkpoint_dir;
    prob.checkpoint.save_interval = checkpoint_interval;
    prob.checkpoint.enabled = true;
  }

  // ---- Load checkpoint if available ----
  if (!checkpoint_dir.empty()) {
    if (dtwc::load_checkpoint(prob, checkpoint_dir, cache_metric)) {
      if (verbose)
        std::cout << "Resumed from checkpoint: " << checkpoint_dir << "\n";
    } else if (verbose) {
      std::cout << "No valid checkpoint found at " << checkpoint_dir << ", starting fresh.\n";
    }
  }

  // ---- GPU distance matrix (if --device cuda) ----
  if (!replaying_result && dev.is_cuda && matrix_free_method) {
    std::cerr << "Error: --method " << method
              << " uses a matrix-free CPU distance schedule; CUDA execution is not "
                 "implemented for that schedule. Use --device cpu.\n";
    return EXIT_FAILURE;
  }
  if (!replaying_result && dev.is_cuda && !prob.is_distance_matrix_filled()) {
#ifdef DTWC_HAS_CUDA
    if (!dtwc::cuda::cuda_available()) {
      std::cerr << "Error: --device cuda requested but no CUDA GPU detected.\n";
      return EXIT_FAILURE;
    }

    // Device id already parsed & validated by parse_device (Task 0.9).
    const int cuda_device_id = dev.cuda_id;

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

    auto cuda_result =
      dtwc::cuda::compute_distance_matrix_cuda(prob.data().p_vec, cuda_opts);

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

  if (replaying_result) {
    result = std::move(*resumed_result);
    std::cout
      << "Replaying completed result checkpoint: N=" << result.labels.size()
      << ", k=" << result.medoid_indices.size()
      << ", iterations=" << result.iterations
      << ", converged=" << (result.converged ? "yes" : "no")
      << " (requested method=" << method
      << "; binary v1 has no method provenance)\n";
  } else if (method == "pam") {
    // FastPAM
    if (verbose)
      std::cout << "Running FastPAM (k=" << n_clusters << ") ...\n";

    result = run_cli_pam(prob, n_clusters, max_iter);

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

    result = dtwc::algorithms::fast_clara(prob, clara_opts);

    if (verbose) {
      std::cout << "FastCLARA finished"
                << ", cost=" << std::setprecision(6) << result.total_cost
                << " [" << clk << "]\n";
    }
  } else if (method == "kmedoids") {
    // Legacy kMedoids Lloyd
    prob.set_n_clusters(n_clusters);
    prob.set_method(dtwc::Method::Kmedoids);
    prob.cluster();

    // Build result from prob state
    result.labels = prob.clusters_ind;
    result.medoid_indices = prob.centroids_ind;
    result.total_cost = prob.find_total_cost();
    result.converged = (prob.last_iterations() < max_iter);
    result.iterations = prob.last_iterations();

    if (verbose)
      std::cout << "kMedoids Lloyd finished, cost=" << result.total_cost
                << " [" << clk << "]\n";
  } else if (method == "mip") {
    // MIP method
    prob.set_n_clusters(n_clusters);
    prob.set_method(dtwc::Method::MIP);
    prob.cluster();

    result.labels = prob.clusters_ind;
    result.medoid_indices = prob.centroids_ind;
    result.total_cost = prob.find_total_cost();
    result.converged = true;

    if (verbose)
      std::cout << "MIP clustering finished, cost=" << result.total_cost
                << " [" << clk << "]\n";
  } else if (method == "lrcore" || method == "lr") {
    // LR-core exact (Lagrangian bound + reduced-cost fixing + y-branching).
    prob.set_n_clusters(n_clusters);
    prob.set_method(dtwc::Method::LRCore);
    prob.cluster();

    result.labels = prob.clusters_ind;
    result.medoid_indices = prob.centroids_ind;
    result.total_cost = prob.find_total_cost();
    result.converged = true;

    if (verbose)
      std::cout << "LR-core clustering finished, cost=" << result.total_cost
                << " [" << clk << "]\n";
  } else if (method == "tadpole") {
    // TADPole density-peaks with conditionally admissible LB/UB DTW pruning.
    prob.set_n_clusters(n_clusters);
    prob.set_tadpole_dc(tadpole_dc); // <0 ⇒ auto-select from a DTW subsample
    prob.set_method(dtwc::Method::TADPole);
    prob.cluster();

    result.labels = prob.clusters_ind;
    result.medoid_indices = prob.centroids_ind;
    result.total_cost = prob.find_total_cost();
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
    else if (linkage_str == "average")
      hier_opts.linkage = dtwc::algorithms::Linkage::Average;
    else
      throw dtwc::InvalidInput("unsupported --linkage '" + linkage_str + "'");

    prob.fill_distance_matrix(); // hierarchical requires full pairwise distances
    auto dend = dtwc::algorithms::build_dendrogram(prob, hier_opts);
    result = dtwc::algorithms::cut_dendrogram(dend, prob, n_clusters);

    if (verbose) {
      std::cout << "Hierarchical clustering finished"
                << ", cost=" << std::setprecision(6) << result.total_cost
                << " [" << clk << "]\n";
    }
  } else {
    // No branch ran, so `result` is still default-constructed; without this the
    // run would checkpoint and write an empty clustering as a success.
    throw dtwc::InvalidInput(
      "unsupported --method '" + method + "' reached clustering dispatch");
  }

  // ---- Save binary checkpoint of clustering result ----
  if (!replaying_result)
    dtwc::save_binary_checkpoint(result, binary_checkpoint_path);

  // ---- Apply result to prob for scoring ----
  if (replaying_result) {
    prob.set_n_clusters(n_clusters);
    prob.clusters_ind = result.labels;
    prob.centroids_ind = result.medoid_indices;
  } else if (method == "pam" || method == "clara" || method == "hierarchical") {
    prob.set_n_clusters(n_clusters);
    prob.clusters_ind = result.labels;
    prob.centroids_ind = result.medoid_indices;
  }

  // ---- Save checkpoint ----
  if (!checkpoint_dir.empty()) {
    try {
      dtwc::save_checkpoint(prob, checkpoint_dir, cache_metric);
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
  const std::optional<size_t> streamed_series_count = stream_parquet_payload
    ? std::optional<size_t>{input_series_count} : std::nullopt;
  write_labels_csv(labels_path, prob, result, streamed_series_count);
  if (verbose)
    std::cout << "Labels written to " << labels_path << "\n";

  // Medoids
  const auto medoids_path = out_dir / (prob_name + "_medoids.csv");
  write_medoids_csv(medoids_path, prob, result, streamed_series_count);
  if (verbose)
    std::cout << "Medoids written to " << medoids_path << "\n";

  // Distance matrix (if computed)
  if (prob.is_distance_matrix_filled()) {
    const auto dm_path = out_dir / (prob_name + "_distance_matrix.csv");
    prob.write_distance_matrix(prob_name + "_distance_matrix.csv");
    if (verbose)
      std::cout << "Distance matrix written to " << dm_path << "\n";
  }

  // Silhouette scores (requires filled distance matrix)
  if (prob.is_distance_matrix_filled() && n_clusters > 1) {
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
            << "  Method:     "
            << (replaying_result ? "checkpoint replay" : method) << "\n";
  if (replaying_result)
    std::cout << "  Requested:  " << method << "\n";
  std::cout << "  Clusters:   " << n_clusters << "\n"
            << "  Total cost: " << std::setprecision(6) << result.total_cost << "\n"
            << "  Converged:  " << (result.converged ? "yes" : "no") << "\n"
            << "  Iterations: " << result.iterations << "\n"
            << "  Output:     " << output_dir << "/\n"
            << "  Time:       " << clk << "\n";

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
  try {
    return run_cli_main(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "Error: unknown non-standard exception\n";
  }
  return EXIT_FAILURE;
}
#endif // DTWC_CL_NO_MAIN
