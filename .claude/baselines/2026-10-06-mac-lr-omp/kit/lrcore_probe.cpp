// lrcore_probe: NOT part of the repository (a measurement tool kept in the lr-omp kit).
//
// Adapted from the 2026-10-06 lrcore record's probe to the b36fad43 API (fast_pam takes its seed). dtwc_cl prints none of what
// the LR routines return (nodes, iterations, root bound, certified). This program takes dtwc_cl's own command line
// (cli::bind -> Config), builds the Problem as dtwc::run() does (apply, read_data, set_data), repeats the preamble of
// dtwc::LR_core_clustering (fill the matrix, FastPAM seed -> UB, dense copy of D), then calls the routines of
// dtwc/mip/lagrangian_root.hpp on that D and UB and prints one JSON line per stage:
//   input            N, k, seed, threads, the UB, the timings of the fill and of FastPAM, the threshold knob
//   root_subgradient lagrangian_root(D, N, k, UB)            the root the HiGHS-OFF build uses
//   root_kelley      lagrangian_root_kelley(D, N, k, UB)     the root the HiGHS-ON build uses (SolverError when OFF)
//   exact            lagrangian_root_exact(D, N, k, UB, --lr-max-nodes): root + branch-and-bound, as LR_core_clustering
//   cli_path         Problem::cluster() i.e. LR_core_clustering itself: cost, medoids, labels, or the SolverError text
// LRPROBE_STAGES = comma list of sub,kel,exact,cli (default all four): the timing kit picks what it needs.
// LRPROBE_MIN_MS = a stage shorter than this is called again until the calls add up to it (at most 200 calls): `ms` is then the
// fastest call, `ms_med` the median, `reps` the count (default 0: one call, `ms` = that call). Every call's result is the same
// (deterministic), the first is the one printed.
// Doubles print with %.17g. mu_fnv is an FNV-1a hash of the multiplier bit patterns (equal hash = bit-identical mu).
#include "Problem.hpp"
#include "algorithms/fast_pam.hpp"
#include "base/error.hpp"
#include "base/settings.hpp"
#include "cli/config.hpp"
#include "config.hpp"
#include "fileOperations.hpp"
#include "io/read_data.hpp"
#include "mip/lagrangian_root.hpp"
#include "mip/mip.hpp"

#include <CLI/CLI.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// What the linked LR object uses for its threshold and for loop 2: the head probes' scratch copy of lagrangian_root.cpp defines these (its
// knobs, LR_OMP_MIN_N and LR_OMP_LOOP2, and the value they came to); the base probes link a stub that returns -1 (no knob, no threshold).
extern "C" long long lr_omp_probe_min_n();
extern "C" int lr_omp_probe_loop2();

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

std::uint64_t fnv_bytes(const void *data, std::size_t n, std::uint64_t h = 1469598103934665603ULL)
{
  const auto *p = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}
std::uint64_t fnv_doubles(const std::vector<double> &v) { return fnv_bytes(v.data(), v.size() * sizeof(double)); }
std::uint64_t fnv_index(const std::vector<dtwc::index_t> &v) { return fnv_bytes(v.data(), v.size() * sizeof(dtwc::index_t)); }

std::string list(const std::vector<dtwc::index_t> &v)
{
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s + "]";
}
std::string esc(const std::string &s)
{
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o;
}

struct Timed
{
  double ms = 0.0;       ///< the fastest call (warm)
  double ms_med = 0.0;   ///< the median call
  double ms_first = 0.0; ///< the first call (cold: the first fork of a process wakes the team)
  int reps = 1;
};

void print_result(const char *stage, const dtwc::mip::LagrangianResult &r, const Timed &t)
{
  std::printf("{\"stage\":\"%s\",\"lb\":%.17g,\"ub\":%.17g,\"gap\":%.17g,\"certified\":%s,\"iterations\":%d,"
              "\"n_core\":%lld,\"nodes\":%lld,\"ms\":%.3f,\"ms_med\":%.3f,\"ms_first\":%.3f,\"reps\":%d,\"mu_fnv\":\"%016llx\",\"medoids\":%s}\n",
              stage, r.lower_bound, r.upper_bound, r.gap, r.certified_optimal ? "true" : "false", r.iterations,
              static_cast<long long>(r.n_core), static_cast<long long>(r.nodes), t.ms, t.ms_med, t.ms_first, t.reps,
              static_cast<unsigned long long>(fnv_doubles(r.multipliers)), list(r.medoids).c_str());
  std::fflush(stdout);
}

bool stage_on(const char *name)
{
  const char *env = std::getenv("LRPROBE_STAGES");
  if (!env || !*env) return true;
  const std::string s = std::string(",") + env + ",";
  return s.find(std::string(",") + name + ",") != std::string::npos;
}

/// Calls @p f once for its result, then again while the stage is shorter than LRPROBE_MIN_MS.
template <typename F>
auto timed_stage(F f, Timed &t)
{
  const char *env = std::getenv("LRPROBE_MIN_MS");
  const double budget = env ? std::atof(env) : 0.0;
  std::vector<double> ms;
  auto t0 = Clock::now();
  auto result = f();
  ms.push_back(ms_since(t0));
  double total = ms.back();
  while (total < budget && ms.size() < 200) {
    t0 = Clock::now();
    (void)f();
    ms.push_back(ms_since(t0));
    total += ms.back();
  }
  t.ms_first = ms.front();
  std::sort(ms.begin(), ms.end());
  t.ms = ms.front();
  t.ms_med = ms[ms.size() / 2];
  t.reps = static_cast<int>(ms.size());
  return result;
}
} // namespace

int main(int argc, char **argv)
{
  try {
    CLI::App app{ "lrcore_probe: dtwc_cl's command line, then the LR routines' counters" };
    dtwc::Config config;
    dtwc::cli::bind(app, config);
    CLI11_PARSE(app, argc, argv);

    // ---- as dtwc::run(): the Problem, then the series ----
    dtwc::Problem prob(config.name.empty() ? dtwc::detail::default_name(dtwc::utf8_to_path(config.input)) : config.name);
    dtwc::apply(config, prob);
    dtwc::Data series = dtwc::read_data(dtwc::utf8_to_path(config.input), config.skip_cols, config.skip_rows,
                                        config.delimiter, config.column);
    prob.set_data(std::move(series));
    prob.set_method(config.method);

    const dtwc::index_t N = prob.size();
    const dtwc::index_t k = prob.n_clusters();
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *knob = std::getenv("LR_OMP_MIN_N"); // read by the kit's head object (see build_kit.sh); the product has a constant

    // ---- the preamble of dtwc::LR_core_clustering ----
    auto t0 = Clock::now();
    prob.fill_distance_matrix();
    const double fill_ms = ms_since(t0);
    t0 = Clock::now();
    const double ub = dtwc::fast_pam(prob, k, dtwc::settings::DEFAULT_MAX_ITER, prob.random_seed()).total_cost;
    const double fastpam_ms = ms_since(t0);
    t0 = Clock::now();
    std::vector<double> D(static_cast<std::size_t>(N) * static_cast<std::size_t>(N));
    for (dtwc::index_t i = 0; i < N; ++i)
      for (dtwc::index_t j = 0; j < N; ++j)
        D[static_cast<std::size_t>(i) * static_cast<std::size_t>(N) + static_cast<std::size_t>(j)] = prob.dist_by_ind(i, j);
    const double copy_ms = ms_since(t0);
    std::printf("{\"stage\":\"input\",\"N\":%lld,\"k\":%lld,\"seed\":%llu,\"threads\":%d,\"lr_max_nodes\":%lld,"
                "\"ub_seed\":%.17g,\"fill_ms\":%.3f,\"fastpam_ms\":%.3f,\"copy_ms\":%.3f,\"min_n_knob\":\"%s\","
                "\"min_n_object\":%lld,\"loop2_object\":%d}\n",
                static_cast<long long>(N), static_cast<long long>(k), static_cast<unsigned long long>(prob.random_seed()),
                threads, static_cast<long long>(config.mip.lr_max_nodes), ub, fill_ms, fastpam_ms, copy_ms,
                knob ? knob : "", lr_omp_probe_min_n(), lr_omp_probe_loop2());
    std::fflush(stdout);

    // ---- the two roots ----
    dtwc::mip::LagrangianResult sub;
    bool have_sub = false;
    if (stage_on("sub")) {
      Timed tm;
      sub = timed_stage([&] { return dtwc::mip::lagrangian_root(D.data(), N, k, ub); }, tm);
      have_sub = true;
      print_result("root_subgradient", sub, tm);
    }

    bool kelley_ok = true, have_kel = false;
    dtwc::mip::LagrangianResult kel;
    if (stage_on("kel")) {
      try {
        Timed tm;
        kel = timed_stage([&] { return dtwc::mip::lagrangian_root_kelley(D.data(), N, k, ub); }, tm);
        have_kel = true;
        print_result("root_kelley", kel, tm);
      } catch (const dtwc::SolverError &e) {
        kelley_ok = false;
        std::printf("{\"stage\":\"root_kelley\",\"unavailable\":true,\"error\":\"%s\"}\n", esc(e.what()).c_str());
        std::fflush(stdout);
      }
    }

    // ---- root + branch-and-bound, as LR_core_clustering calls it ----
    if (stage_on("exact")) {
      Timed tm;
      const auto ex = timed_stage([&] { return dtwc::mip::lagrangian_root_exact(D.data(), N, k, ub, config.mip.lr_max_nodes); }, tm);
      print_result("exact", ex, tm);
      // the root inside lagrangian_root_exact is Kelley when HiGHS is compiled in, else the subgradient: its multipliers
      // come back in the result, so equal hashes prove which root ran (needs that root's stage in LRPROBE_STAGES).
      const bool root_known = kelley_ok ? have_kel : have_sub;
      if (root_known) {
        const auto &used_root = kelley_ok ? kel : sub;
        const bool same_mu = fnv_doubles(ex.multipliers) == fnv_doubles(used_root.multipliers);
        std::printf("{\"stage\":\"exact_root_check\",\"root_used\":\"%s\",\"mu_identical_to_that_root\":%s,"
                    "\"iterations_equal\":%s,\"n_core_equal\":%s}\n",
                    kelley_ok ? "kelley" : "subgradient", same_mu ? "true" : "false",
                    ex.iterations == used_root.iterations ? "true" : "false",
                    ex.n_core == used_root.n_core ? "true" : "false");
        std::fflush(stdout);
      }
    }

    // ---- LR_core_clustering itself (what Problem::cluster() and so dtwc_cl runs) ----
    if (stage_on("cli")) {
      t0 = Clock::now();
      try {
        const auto result = prob.cluster();
        const double ms = ms_since(t0);
        std::printf("{\"stage\":\"cli_path\",\"ok\":true,\"cost\":%.17g,\"ms\":%.3f,\"medoids\":%s,\"labels_fnv\":\"%016llx\","
                    "\"labels\":%s}\n",
                    result.total_cost, ms, list(prob.medoids()).c_str(),
                    static_cast<unsigned long long>(fnv_index(prob.labels())), list(prob.labels()).c_str());
      } catch (const dtwc::SolverError &e) {
        std::printf("{\"stage\":\"cli_path\",\"ok\":false,\"ms\":%.3f,\"solver_error\":\"%s\"}\n", ms_since(t0),
                    esc(e.what()).c_str());
      }
      std::fflush(stdout);
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
  }
  return 1;
}
