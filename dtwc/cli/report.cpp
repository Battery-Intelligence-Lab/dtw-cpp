/**
 * @file report.cpp
 * @brief RunReport: the clocks of a run's stages, and the report's text (render(), command_of()).
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 5.5 (RR1b)
 * @date 03 Nov 2026
 */

#include "report.hpp"

#include "../core/distance_matrix.hpp" // peak_memory
#include "config.hpp"                  // to_config_text

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <string_view>
#include <utility>

// The floating-point model and the arch level are CMake's (DTWC_FP_MODEL, DTWC_ARCH_LEVEL), defined for this file
// alone by cli/CMakeLists.txt.
#ifndef DTWC_FP_MODEL_NAME
#define DTWC_FP_MODEL_NAME "unknown"
#endif
#ifndef DTWC_ARCH_LEVEL_NAME
#define DTWC_ARCH_LEVEL_NAME "unknown"
#endif

namespace dtwc::detail {
namespace {

constexpr std::size_t width = 100;       // the report's columns at most
constexpr std::size_t value_column = 18; // three spaces and a label of up to 15

/// `head`, then `value` wrapped within the report's width at a space (a word longer than a line is cut where the line
/// ends), each further line indented 4 columns past `head`'s width; no line ends in a space.
void wrapped(std::string &out, std::string_view head, std::string_view value)
{
  out.append(head);
  std::size_t used = head.size();
  for (;;) {
    const std::size_t room = width - used;
    std::size_t cut = value.size();
    if (cut > room) {
      cut = value.rfind(' ', room);
      if (cut == std::string_view::npos || cut == 0) cut = room;
    }
    out.append(value.substr(0, cut));
    while (!out.empty() && out.back() == ' ') out.pop_back();
    out += '\n';
    value.remove_prefix(cut);
    while (!value.empty() && value.front() == ' ') value.remove_prefix(1);
    if (value.empty()) return;
    used = head.size() + 4;
    out.append(used, ' ');
  }
}

/// A section's line: `label` in a 15-column field, then `value`.
void row(std::string &out, std::string_view label, std::string_view value)
{
  char head[value_column + 1];
  std::snprintf(head, sizeof head, "   %-15.*s", static_cast<int>(label.size()), label.data());
  wrapped(out, head, value);
}

std::string number(double value, int decimals)
{
  char text[64];
  std::snprintf(text, sizeof text, "%.*f", decimals, value);
  return text;
}

/// A mean silhouette as -v prints it (four significant digits), so the two agree to the digit.
std::string four_digits(double value)
{
  char text[32];
  std::snprintf(text, sizeof text, "%.4g", value);
  return text;
}

/// Milliseconds as seconds with three decimals.
std::string seconds(std::int64_t ms)
{
  char text[32];
  std::snprintf(text, sizeof text, "%lld.%03lld", static_cast<long long>(ms / 1000), static_cast<long long>(ms % 1000));
  return text;
}

/// `t` seconds in whole milliseconds; a row's wall is the difference of its ends', so the rows sum to their span.
std::int64_t ms(double t) { return t > 0.0 ? static_cast<std::int64_t>(std::llround(t * 1000.0)) : 0; }

std::string bytes(std::size_t n)
{
  static const char *const units[] = { "bytes", "KiB", "MiB", "GiB", "TiB" };
  double value = static_cast<double>(n);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < std::size(units)) {
    value /= 1024.0;
    ++unit;
  }
  char text[48];
  if (unit == 0)
    std::snprintf(text, sizeof text, "%zu bytes", n);
  else
    std::snprintf(text, sizeof text, "%.1f %s", value, units[unit]);
  return text;
}

/// A byte count as bytes() gives it and, beyond a KiB, exactly, as -v's memory plan and a refusal print it.
std::string exact_bytes(std::size_t n)
{
  return n < 1024 ? bytes(n) : bytes(n) + " (" + std::to_string(n) + " bytes)";
}

/// `time` as a calendar date and time of day, UTC or local.
std::string calendar(std::time_t time, bool utc)
{
  std::tm parts{};
#ifdef _WIN32
  const bool known = (utc ? gmtime_s(&parts, &time) : localtime_s(&parts, &time)) == 0;
#else
  const bool known = (utc ? gmtime_r(&time, &parts) : localtime_r(&time, &parts)) != nullptr;
#endif
  char text[32];
  if (!known || std::strftime(text, sizeof text, "%Y-%m-%d %H:%M:%S", &parts) == 0) return "unknown";
  return text;
}

/// The shortest text that reads back as `value`, as dtwc_cl prints the cost.
std::string shortest(double value)
{
  std::array<char, 32> text{};
  return std::string(text.data(), std::to_chars(text.data(), text.data() + text.size(), value).ptr);
}

/// The first 20 of `values`, space-separated, and how many there are where there are more.
template <class T>
std::string first_of(const std::vector<T> &values)
{
  std::string out;
  for (std::size_t i = 0; i < values.size() && i < 20; ++i) out += (i ? " " : "") + std::to_string(values[i]);
  if (values.size() > 20) out += " ... (" + std::to_string(values.size()) + " in all)";
  return out;
}

/// The optional libraries this run's code was built with: dtwc_core's (machine_facts) and the command line's own.
std::string libraries(const core::MachineFacts &machine)
{
  std::string list = machine.libraries == "none" ? std::string() : machine.libraries;
#ifdef DTWC_HAS_ARROW
  list += list.empty() ? "Arrow" : ", Arrow";
#endif
#ifdef DTWC_HAS_PARQUET
  list += ", Parquet";
#endif
#ifdef DTWC_HAS_YAML
  list += list.empty() ? "YAML" : ", YAML";
#endif
  return list.empty() ? "no optional library" : list;
}

/// The criterion that stopped the clustering, as the method's result tells it.
std::string stopped(const RunReport::Clustering &clustering, const Config &config)
{
  switch (clustering.method) {
  case Method::PAM:
  case Method::CLARA:
    return clustering.converged ? "a sweep found no improving swap" : "max-iter reached";
  case Method::OneBatch:
    return clustering.converged ? "a sweep found no improving swap on its batch (k = 1 and k = N need none)"
                                : "max-iter reached";
  case Method::Kmedoids: // Problem::cluster's flag: Lloyd's iterations below max-iter
    return clustering.converged ? "the medoids stopped changing before max-iter"
                                : "max-iter reached (or the medoids stopped changing at the last iteration, which the "
                                  "converged flag does not tell apart)";
  case Method::MIP:
    return "the solver's optimum within --mip-gap (a stop short of it is a SolverError)";
  case Method::LRCore:
    return "optimality certified by branch and bound (a node-cap stop is a SolverError)";
  case Method::Hierarchical:
    return "no iterations: the dendrogram (" + std::string(name_of(algorithms::linkage_names, config.linkage))
           + " linkage) cut at k";
  case Method::TADPole:
    return "no iterations: one pass over the density peaks, dc "
           + (config.tadpole_dc > 0.0 ? shortest(config.tadpole_dc) : std::string("automatic"));
  case Method::Auto: break;
  }
  return "unknown";
}

} // namespace

RunReport::RunReport(const Config &config_)
  : config(config_), machine(core::machine_facts(config_.device, config_.device_index))
{
  method = config_.method; // until the run resolves it
  // The clocks start once the machine's facts are read: the run's time is its own.
  started = finished = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  start = std::chrono::steady_clock::now();
  cpu_start = cpu_end = core::cpu_seconds();
}

void RunReport::begin(const char *name, int threads)
{
  close(threads);
  Stage stage;
  stage.name = name;
  stage.begin = wall;
  stage.cpu = cpu_end; // the CPU seconds at its start, until it closes
  stages.push_back(std::move(stage));
  open = true;
}

void RunReport::close(int threads)
{
  wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  cpu_end = core::cpu_seconds();
  peak = core::peak_memory();
  finished = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  if (!open) return;
  Stage &stage = stages.back();
  stage.end = wall;
  stage.cpu = std::max(0.0, cpu_end - stage.cpu);
  stage.threads = threads;
  stage.peak = peak;
  open = false;
}

std::string RunReport::render() const
{
  const bool failed = !error_kind.empty();
  const int threads = plan ? plan->threads : 0;
  std::string out = "DTWC++ run report (format 1)\n" + std::string(width, '=') + "\n";

  out += "1. Run\n";
  row(out, "status", failed ? "FAILED in stage " + std::string(stages.empty() ? "check" : stages.back().name) + ": "
                                + error_kind
                            : std::string("completed"));
  row(out, "started", calendar(started, true) + " UTC (local time " + calendar(started, false) + ")");
  row(out, "finished", calendar(finished, true) + " UTC, " + seconds(ms(wall)) + " s after the start");
  row(out, "version", "DTWC++ " DTWC_VERSION_STRING "; " + machine.compiler + "; fp model " DTWC_FP_MODEL_NAME
                        "; arch level " DTWC_ARCH_LEVEL_NAME "; " + libraries(machine));
  std::string configuration = "method " + std::string(name_of(method_names, method));
  if (config.method == Method::Auto && method != Method::Auto) configuration += " (from auto)";
  configuration += ", k " + std::to_string(config.k) + ", band "
                   + (config.band < 0 ? std::string("full") : std::to_string(config.band)) + ", metric "
                   + std::string(name_of(core::metric_names, config.metric)) + ", variant "
                   + std::string(name_of(core::variant_names, config.variant.variant)) + ", missing-strategy "
                   + std::string(name_of(core::missing_strategy_names, config.missing)) + ", n-init "
                   + std::to_string(config.n_init) + ", max-iter " + std::to_string(config.max_iter) + ", seed "
                   + std::to_string(config.seed) + ", dtype "
                   + std::string(name_of(core::precision_names, config.dtype)) + ", matrix "
                   + std::string(name_of(core::precision_names, config.matrix_dtype)) + ", device "
                   + device_text(config) + ", threads "
                   + (plan ? std::to_string(plan->threads) + " of " + std::to_string(plan->requested)
                           : std::string("not planned yet"));
  row(out, "configuration", configuration);
  row(out, "command", command_of(config, method));

  out += "2. Machine\n";
  row(out, "host", machine.host);
  row(out, "os", machine.os);
  row(out, "cpu", machine.cpu + "; "
                    + (machine.physical_cores > 0 && machine.logical_cores > 0
                         ? std::to_string(machine.physical_cores) + " physical, "
                             + std::to_string(machine.logical_cores) + " logical cores"
                         : std::string("cores unknown")));
  row(out, "simd", "the CPU has " + machine.cpu_simd + "; the build may use " + machine.build_simd);
  row(out, "ram", machine.ram_total > 0 ? bytes(machine.ram_total) + " total, "
                                            + (machine.ram_available > 0 ? bytes(machine.ram_available)
                                                                         : std::string("unknown"))
                                            + " available at the start"
                                      : std::string("unknown"));
  if (machine.gpu) row(out, "gpu", *machine.gpu);

  out += "3. Memory plan\n";
  if (!plan) {
    row(out, "plan", "none yet: it is made once the input's facts are known");
  } else if (plan->budget == 0) {
    row(out, "budget", "none: no --ram-limit, and what the process may use is unknown; in RAM on "
                         + std::to_string(plan->requested) + " threads");
  } else {
    row(out, "budget", exact_bytes(plan->budget)
                         + (config.ram_limit == 0 ? ", automatic: 90% of the memory available at the plan"
                                                  : ", --ram-limit"));
    row(out, "reserve", exact_bytes(plan->reserve) + ": " + std::to_string(plan->threads) + " of "
                          + std::to_string(plan->requested) + " threads x " + bytes(plan->per_thread)
                          + " for series of up to " + std::to_string(plan->length) + " time steps, x1.1");
    row(out, "data cap", exact_bytes(plan->data_cap) + ": what the reserve leaves the series and the matrix");
    row(out, "route", plan->route);
    row(out, "plan holds", exact_bytes(plan->held) + " in RAM");
  }
  row(out, "peak", exact_bytes(peak) + " resident, mapped files included (the process's highest, at the end)");

  out += "4. Input\n";
  row(out, "input", config.input.empty() ? std::string("series passed in memory")
                                         : config.input + (input ? std::string(" (") + input->format + ")" : ""));
  if (!input) {
    row(out, "series", "not read yet");
  } else {
    std::string series = std::to_string(input->series) + " series of " + std::to_string(input->ndim)
                         + (input->ndim == 1 ? " feature" : " features") + "; ";
    const double mean = input->series > 0 && input->ndim > 0
                          ? static_cast<double>(input->values)
                              / static_cast<double>(input->series * input->ndim)
                          : 0.0;
    if (input->longest > 0)
      series += "time steps " + std::to_string(input->shortest) + " to " + std::to_string(input->longest)
                + ", mean " + number(mean, 1);
    else
      series += "time steps: mean " + number(mean, 1) + (input->streamed ? " (streamed: no lengths read)" : "");
    series += "; " + std::string(name_of(core::precision_names, config.dtype));
    if (input->bytes > 0) series += "; " + bytes(input->bytes) + " of values";
    row(out, "series", series);
  }

  out += "5. Stages\n";
  // The silhouettes, timed inside the write, print as a row of their own after it.
  std::vector<Stage> rows = stages;
  std::vector<std::int64_t> walls(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) walls[i] = ms(rows[i].end) - ms(rows[i].begin);
  if (silhouettes && !rows.empty() && std::string_view(rows.back().name) == "write") {
    const std::int64_t part = std::min(ms(silhouettes->wall), walls.back());
    walls.back() -= part;
    Stage row_of = rows.back();
    rows.back().cpu = std::max(0.0, rows.back().cpu - silhouettes->cpu);
    row_of.name = "silhouettes";
    row_of.cpu = silhouettes->cpu;
    row_of.note = silhouettes->mean ? "mean " + four_digits(*silhouettes->mean) : "not computed";
    rows.push_back(std::move(row_of));
    walls.push_back(part);
  }
  const std::int64_t total = ms(wall);
  const auto table_row = [&out, total](const char *name, std::int64_t wall_ms, double cpu, int on,
                                       std::size_t peak_bytes, const std::string &note) {
    const double wall_s = static_cast<double>(wall_ms) / 1000.0;
    const std::string share = total > 0 ? number(100.0 * static_cast<double>(wall_ms) / static_cast<double>(total), 1)
                                            + "%"
                                        : std::string("-");
    const std::string util = on > 0 && wall_s >= 0.05 ? number(100.0 * cpu / (wall_s * on), 0) + "%" : "-";
    char head[96];
    std::snprintf(head, sizeof head, "   %-11s %9s %7s %9s %9s  ", name, seconds(wall_ms).c_str(), share.c_str(),
                  util.c_str(), number(static_cast<double>(peak_bytes) / (1024.0 * 1024.0), 1).c_str());
    wrapped(out, head, note);
  };
  {
    char head[96];
    std::snprintf(head, sizeof head, "   %-11s %9s %7s %9s %9s  %s\n", "stage", "wall s", "share", "cpu util",
                  "peak MiB", "note");
    out += head;
  }
  std::int64_t covered = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    table_row(rows[i].name, walls[i], rows[i].cpu, rows[i].threads, rows[i].peak, rows[i].note);
    covered += walls[i];
  }
  table_row("total", total, cpu_end - cpu_start, threads, peak, "the rows cover " + seconds(covered) + " s");

  out += "6. Algorithm\n";
  if (!clustering) {
    row(out, "method", std::string(name_of(method_names, method)) + ": the run ended before it clustered");
  } else {
    row(out, "method", std::string(name_of(method_names, clustering->method)) + " (" + clustering->label + "); n-init "
                         + std::to_string(config.n_init) + ", seed " + std::to_string(config.seed));
    row(out, "iterations", std::to_string(clustering->iterations) + " (max-iter " + std::to_string(config.max_iter)
                             + "); converged " + (clustering->converged ? "yes" : "no"));
    row(out, "stopped", stopped(*clustering, config));
  }

  out += "7. Result\n";
  if (!clustering) {
    row(out, "result", "none");
  } else {
    row(out, "cost", shortest(clustering->cost));
    std::string silhouette = "not computed: no output directory (-o)";
    if (silhouettes) silhouette = silhouettes->mean ? "mean " + four_digits(*silhouettes->mean) : silhouettes->why_not;
    row(out, "silhouette", silhouette);
    row(out, "clusters", std::to_string(clustering->sizes.size()) + "; sizes " + first_of(clustering->sizes));
    row(out, "medoids", "series " + first_of(clustering->medoids) + " (indices in input order)");
  }

  out += "8. Notes\n";
  row(out, "cpu util", "the process's CPU seconds over a row's wall seconds x its planned threads; OpenMP threads that "
                       "wait may spin, which counts");
  if (plan && plan->budget > 0 && peak > plan->budget)
    row(out, "peak", "the process peaked above the budget: the budget is a plan, not a cap, and the peak counts the "
                     "pages of mapped files and libraries, which the plan does not hold");
  if (plan && input && input->longest > plan->length)
    row(out, "length", "a series of " + std::to_string(input->longest) + " time steps, longer than the "
                         + std::to_string(plan->length)
                         + " the plan priced: the threads' scratch can grow past the reserve");
  for (const auto &note : notes) row(out, "note", note);
  if (failed) row(out, "error", error_kind + ": " + error_text);
  return out;
}

std::string command_of(const Config &config, Method method)
{
  Config resolved = config;
  resolved.method = method;
  const auto lines = [](const std::string &text) {
    std::vector<std::string> out;
    for (std::size_t at = 0; at < text.size();) {
      const std::size_t end = std::min(text.find('\n', at), text.size());
      out.push_back(text.substr(at, end - at));
      at = end + 1;
    }
    return out;
  };
  const auto given = lines(to_config_text(resolved)), defaults = lines(to_config_text(Config{}));
  std::string command = "dtwc_cl";
  for (std::size_t i = 0; i < given.size() && i < defaults.size(); ++i) {
    if (given[i] == defaults[i]) continue;
    const std::size_t equals = given[i].find(" = ");
    if (equals == std::string::npos) continue;
    const std::string value = given[i].substr(equals + 3);
    command += " --" + given[i].substr(0, equals);
    if (value != "true") command += " " + value;
  }
  if (config.input.empty()) command += " (the series passed in memory)";
  return command;
}

} // namespace dtwc::detail
