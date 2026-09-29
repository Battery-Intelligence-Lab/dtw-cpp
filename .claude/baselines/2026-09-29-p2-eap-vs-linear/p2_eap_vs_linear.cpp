// P2 probe: EAPruned vs linear kernel for unbanded per-pair Standard DTW (L1, f64) on UCR data.
//   p2_eap_vs_linear.exe <ucr_root> <out_csv> <rounds> <dataset>[:z] ...
// A dataset suffixed ":z" is z-normalised per series with dtwc::core::z_normalize before use.
// Library entry points timed: dtwc::dtwFull_eap (what make_standard binds for band < 0) and dtwc::dtwFull_L
// (what dtwBanded forwards to for band < 0). dtwc::dtwFull is the third computation.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "warping.hpp"          // dtwc::dtwFull_eap, dtwc::dtwFull_L, dtwc::dtwFull, detail::L1Dist
#include "core/z_normalize.hpp" // dtwc::core::z_normalize

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace p2 {

// Verbatim copy of dtwc::core::dtw_kernel_eap (design-2.0 @ 59750c4, dtwc/core/dtw_kernel.hpp:334-425), comments
// dropped, with one change: `visited` counts the cells the kernel evaluates (each iteration of its two cell loops).
template <typename T, typename Cost>
T eap_counted(std::size_t n_short, std::size_t n_long, Cost cost, std::uint64_t &visited)
{
  constexpr T maxValue = std::numeric_limits<T>::max();
  if (n_short == 0 || n_long == 0) return maxValue;

  T ub = cost(0, 0);
  std::size_t sd = 1, ld = 1;
  while (sd < n_short && ld < n_long) { ub += cost(sd, ld); ++sd; ++ld; }
  const std::size_t s_last = n_short - 1;
  while (ld < n_long) { ub += cost(s_last, ld); ++ld; }

  const T thr = ub + std::abs(ub)
              * (static_cast<T>(n_long) * T(16) * std::numeric_limits<T>::epsilon());

  thread_local std::vector<T> prev_buf, curr_buf;
  prev_buf.assign(n_short, maxValue);
  curr_buf.assign(n_short, maxValue);
  T* prev = prev_buf.data();
  T* curr = curr_buf.data();

  std::size_t prev_lo = 0, prev_hi = 0;
  std::size_t comp_start = 0;

  {
    std::size_t last_live = 0;
    for (std::size_t s = 0; s < n_short; ++s) {
      ++visited;
      const T d = (s == 0) ? cost(0, 0)
                           : (curr[s - 1] == maxValue ? maxValue
                                                      : curr[s - 1] + cost(s, 0));
      if (d <= thr) { curr[s] = d; last_live = s; }
      else { curr[s] = maxValue; break; }
    }
    prev_lo = 0;
    prev_hi = last_live + 1;
    comp_start = 0;
  }

  for (std::size_t L = 1; L < n_long; ++L) {
    std::swap(prev, curr);

    std::size_t first_live = n_short;
    std::size_t last_live = comp_start;
    T left = maxValue;

    for (std::size_t s = comp_start; s < n_short; ++s) {
      ++visited;
      const T up   = (s >= prev_lo && s < prev_hi) ? prev[s] : maxValue;
      const T diag = (s >= 1 && (s - 1) >= prev_lo && (s - 1) < prev_hi)
                       ? prev[s - 1] : maxValue;

      T m = up;
      if (left < m) m = left;
      if (diag < m) m = diag;
      const T d = (m == maxValue) ? maxValue : m + cost(s, L);

      if (d <= thr) {
        left = d;
        last_live = s;
        if (first_live == n_short) first_live = s;
      } else {
        left = maxValue;
      }
      curr[s] = left;

      if (s < prev_hi) continue;
      if (d > thr) break;
    }

    comp_start = (first_live == n_short) ? last_live : first_live;
    prev_lo = (first_live == n_short) ? last_live : comp_start;
    prev_hi = last_live + 1;
  }

  return curr[n_short - 1];
}

// The wrapper dtwFull_eap_impl builds (equal lengths: no swap), with the library's L1 functor.
double eap_counted_pair(const double *x, const double *y, std::size_t n, std::uint64_t &visited)
{
  dtwc::detail::L1Dist distance;
  auto cost = [x, y, distance](std::size_t row, std::size_t col) noexcept { return distance(x[row], y[col]); };
  return eap_counted<double>(n, n, cost, visited);
}

// The metric is a runtime value, as p.metric() is in make_standard.
volatile int g_metric = static_cast<int>(dtwc::core::MetricType::L1);
dtwc::core::MetricType metric() { return static_cast<dtwc::core::MetricType>(g_metric); }

__attribute__((noinline)) double run_eap(const double *x, const double *y, std::size_t n)
{
  return dtwc::dtwFull_eap<double>(x, n, y, n, metric());
}
__attribute__((noinline)) double run_lin(const double *x, const double *y, std::size_t n)
{
  return dtwc::dtwFull_L<double>(x, n, y, n, -1.0, metric());
}
__attribute__((noinline)) double run_full(const double *x, const double *y, std::size_t n)
{
  return dtwc::dtwFull<double>(x, n, y, n, metric());
}

// ---------------------------------------------------------------------------------------------------------------
// Machine load over an interval (as in the PF5 probe): busy share of all logical CPUs, and the share not used by
// this process.
struct LoadMeter {
  ULONGLONG idle0 = 0, total0 = 0, own0 = 0;
  static ULONGLONG ft(FILETIME f) { return (ULONGLONG(f.dwHighDateTime) << 32) | f.dwLowDateTime; }
  static ULONGLONG own() { FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u); return ft(k) + ft(u); }
  void start() { FILETIME i, k, u; GetSystemTimes(&i, &k, &u); idle0 = ft(i); total0 = ft(k) + ft(u); own0 = own(); }
  void stop(double &busy_pct, double &other_pct) const {
    FILETIME i, k, u; GetSystemTimes(&i, &k, &u);
    const double total = double(ft(k) + ft(u) - total0), idle = double(ft(i) - idle0), mine = double(own() - own0);
    busy_pct = 100.0 * (total - idle) / total;
    other_pct = 100.0 * (total - idle - mine) / total;
  }
};

double g_ns_per_tick = 0;
double g_cycles_per_ns = 0;
inline std::int64_t ticks() { LARGE_INTEGER t; QueryPerformanceCounter(&t); return t.QuadPart; }
// Second clock: TSC cycles charged to this thread only (QueryThreadCycleTime), so time slices other processes
// take on the shared core are not counted.
inline double tcycles() { ULONG64 c; QueryThreadCycleTime(GetCurrentThread(), &c); return double(c); }

double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------------------------------------------------------
struct Dataset {
  std::string name;
  std::vector<std::string> labels;
  std::vector<std::vector<double>> series;
};

Dataset load_ucr(const std::string &root, const std::string &name, bool znorm)
{
  Dataset d;
  d.name = znorm ? name + ":z" : name;
  const std::string path = root + "/" + name + "/" + name + "_TEST.tsv";
  std::ifstream in(path);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(3); }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const char *p = line.c_str();
    const char *tab = std::strchr(p, '\t');
    if (!tab) { std::fprintf(stderr, "bad line in %s\n", path.c_str()); std::exit(3); }
    d.labels.emplace_back(p, tab);
    std::vector<double> v;
    p = tab + 1;
    while (*p) {
      char *end = nullptr;
      const double x = std::strtod(p, &end);
      if (end == p) { std::fprintf(stderr, "parse error in %s\n", path.c_str()); std::exit(3); }
      v.push_back(x);
      p = end;
      while (*p == '\t' || *p == '\r' || *p == ' ') ++p;
    }
    for (double x : v)
      if (!std::isfinite(x)) { std::fprintf(stderr, "non-finite value in %s\n", path.c_str()); std::exit(3); }
    if (znorm) dtwc::core::z_normalize(v.data(), v.size());
    d.series.push_back(std::move(v));
  }
  for (const auto &s : d.series)
    if (s.size() != d.series[0].size()) { std::fprintf(stderr, "unequal lengths in %s\n", path.c_str()); std::exit(3); }
  return d;
}

struct Pair { std::size_t i, j; bool within; };

// Uniform over the unordered pairs of the kind (same class / different class), without replacement; order within
// the pair as drawn. g() % n: the modulo bias is < 1e-15 here and the draw is portable.
std::vector<Pair> draw_pairs(const Dataset &d, bool within, std::size_t count, std::uint64_t seed)
{
  std::mt19937_64 g(seed);
  const std::size_t n = d.series.size();
  std::set<std::pair<std::size_t, std::size_t>> seen;
  std::vector<Pair> out;
  for (std::uint64_t attempt = 0; out.size() < count && attempt < 100000000ull; ++attempt) {
    const std::size_t i = g() % n, j = g() % n;
    if (i == j || (d.labels[i] == d.labels[j]) != within) continue;
    if (!seen.insert({std::min(i, j), std::max(i, j)}).second) continue;
    out.push_back({i, j, within});
  }
  return out;
}

struct PairResult {
  Pair pr;
  std::uint64_t visited = 0;
  double value = 0;
  bool eap_eq_lin = false, eap_eq_full = false, counted_eq_eap = false;
  std::vector<double> te, tl, ce, cl; // wall ns and thread cycles, per round
  double te_med = 0, tl_med = 0, ratio_med = 0, ce_med = 0, cl_med = 0, cratio_med = 0;
};

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

struct Summary {
  double eap_ns, lin_ns, ratio, sum_ratio, cratio, csum_ratio, frac_med, frac_lo, frac_hi, eap_ns_per_visited,
    lin_ns_per_cell;
  int n, mism;
};

Summary summarise(const std::vector<PairResult> &R, std::size_t L, int which /*0 within, 1 across, 2 pooled*/)
{
  std::vector<double> te, tl, ratio, cratio, frac, epv, lpc;
  double se = 0, sl = 0, sce = 0, scl = 0;
  int mism = 0, n = 0;
  for (const auto &r : R) {
    if (which == 0 && !r.pr.within) continue;
    if (which == 1 && r.pr.within) continue;
    ++n;
    te.push_back(r.te_med); tl.push_back(r.tl_med); ratio.push_back(r.ratio_med); cratio.push_back(r.cratio_med);
    const double f = double(r.visited) / (double(L) * double(L));
    frac.push_back(f);
    epv.push_back(r.te_med / double(r.visited));
    lpc.push_back(r.tl_med / (double(L) * double(L)));
    se += r.te_med; sl += r.tl_med; sce += r.ce_med; scl += r.cl_med;
    if (!r.eap_eq_lin || !r.eap_eq_full || !r.counted_eq_eap) ++mism;
  }
  Summary s{};
  s.n = n; s.mism = mism;
  s.eap_ns = median(te); s.lin_ns = median(tl); s.ratio = median(ratio); s.sum_ratio = se / sl;
  s.cratio = median(cratio); s.csum_ratio = sce / scl;
  s.frac_med = median(frac);
  s.frac_lo = *std::min_element(frac.begin(), frac.end());
  s.frac_hi = *std::max_element(frac.begin(), frac.end());
  s.eap_ns_per_visited = median(epv); s.lin_ns_per_cell = median(lpc);
  return s;
}

} // namespace p2

int main(int argc, char **argv)
{
  using namespace p2;
  if (argc < 5) {
    std::fprintf(stderr, "usage: p2_eap_vs_linear.exe <ucr_root> <out_csv> <rounds> <dataset>[:z] ...\n");
    return 2;
  }
  const std::string root = argv[1], csv_path = argv[2];
  const int rounds = std::atoi(argv[3]);
  LARGE_INTEGER f; QueryPerformanceFrequency(&f);
  g_ns_per_tick = 1e9 / double(f.QuadPart);
  DWORD_PTR pm = 0, sm = 0;
  GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm);
  std::printf("P2 probe: affinity mask 0x%llx, QPC %lld Hz, rounds %d\n", (unsigned long long)pm, (long long)f.QuadPart, rounds);
  {
    // Clock overheads (median of back-to-back reads) and the thread-cycle rate (best of 20 spins of 5 ms: a spin
    // that loses the core counts fewer cycles, so the maximum is the undisturbed rate).
    std::vector<double> oq, oc;
    for (int k = 0; k < 1001; ++k) {
      const std::int64_t a = ticks(), b = ticks(); oq.push_back(double(b - a) * g_ns_per_tick);
      const double c0 = tcycles(), c1 = tcycles(); oc.push_back(c1 - c0);
    }
    double best = 0;
    for (int k = 0; k < 20; ++k) {
      const std::int64_t t0 = ticks(); const double c0 = tcycles();
      while (double(ticks() - t0) * g_ns_per_tick < 5e6) {}
      best = std::max(best, (tcycles() - c0) / (double(ticks() - t0) * g_ns_per_tick));
    }
    g_cycles_per_ns = best;
    std::printf("clocks: QPC read %.0f ns, thread-cycle read %.0f cycles, thread cycles per ns %.4f\n", median(oq),
                median(oc), best);
  }
  std::fflush(stdout);

  FILE *csv = nullptr;
  if (fopen_s(&csv, csv_path.c_str(), "w") != 0) return 3;
  std::fprintf(csv, "dataset,kind,i,j,L,visited,computed_fraction,eap_ns,lin_ns,ratio_eap_over_lin,eap_cycles,lin_cycles,ratio_cycles,dtw,eap_eq_lin,eap_eq_full,counted_eq_eap\n");

  struct FR { double frac, ratio, cratio; };
  std::vector<FR> all_frac_ratio; // counted datasets (every argv dataset), for the crossover
  for (int a = 4; a < argc; ++a) {
    std::string arg = argv[a];
    const bool znorm = arg.size() > 2 && arg.substr(arg.size() - 2) == ":z";
    const std::string name = znorm ? arg.substr(0, arg.size() - 2) : arg;
    const Dataset d = load_ucr(root, name, znorm);
    const std::size_t L = d.series[0].size();
    // Seed from the dataset name (FNV-1a), not its argv position: Rock and Rock:z draw the same pairs.
    std::uint64_t seed = 1469598103934665603ull;
    for (unsigned char ch : name) seed = (seed ^ ch) * 1099511628211ull;
    seed ^= 20260929ull;
    auto within = draw_pairs(d, true, 200, seed);
    auto across = draw_pairs(d, false, 200, seed + 1);
    std::printf("\n== %s: N=%zu L=%zu within=%zu across=%zu seed=%llu\n", d.name.c_str(), d.series.size(), L,
                within.size(), across.size(), (unsigned long long)seed);

    // Interleave the kinds so drift over the timing phase hits both alike.
    std::vector<PairResult> R;
    for (std::size_t k = 0; k < std::max(within.size(), across.size()); ++k) {
      if (k < within.size()) R.push_back({within[k]});
      if (k < across.size()) R.push_back({across[k]});
    }

    // Verification pass (untimed; also the warm-up).
    int mism = 0;
    for (auto &r : R) {
      const double *x = d.series[r.pr.i].data(), *y = d.series[r.pr.j].data();
      const double e = run_eap(x, y, L), l = run_lin(x, y, L), fu = run_full(x, y, L);
      const double c = eap_counted_pair(x, y, L, r.visited);
      r.value = e;
      r.eap_eq_lin = same_bits(e, l); r.eap_eq_full = same_bits(e, fu); r.counted_eq_eap = same_bits(c, e);
      if (!r.eap_eq_lin || !r.eap_eq_full || !r.counted_eq_eap) {
        ++mism;
        std::printf("MISMATCH %s (%zu,%zu): eap=%.17g lin=%.17g full=%.17g counted=%.17g\n", d.name.c_str(), r.pr.i,
                    r.pr.j, e, l, fu, c);
      }
    }
    std::printf("verify: pairs=%zu mismatches=%d\n", R.size(), mism);
    std::fflush(stdout);

    // Timing: each round times every pair with both kernels back to back, order alternating.
    double sink = 0;
    LoadMeter lm; lm.start();
    for (int rd = 0; rd < rounds; ++rd) {
      LoadMeter lr; lr.start();
      double se = 0, sl = 0, sce = 0, scl = 0;
      // One kernel call between two reads of each clock; the same wrapper for both kernels.
      auto timed = [&](auto kernel, const double *x, const double *y, double &ns, double &cyc) {
        const std::int64_t t0 = ticks(); const double c0 = tcycles();
        sink += kernel(x, y, L);
        const double c1 = tcycles(); const std::int64_t t1 = ticks();
        ns = double(t1 - t0) * g_ns_per_tick; cyc = c1 - c0;
      };
      for (std::size_t p = 0; p < R.size(); ++p) {
        auto &r = R[p];
        const double *x = d.series[r.pr.i].data(), *y = d.series[r.pr.j].data();
        double te, tl, ce, cl;
        if (((rd + p) & 1) == 0) { timed(run_eap, x, y, te, ce); timed(run_lin, x, y, tl, cl); }
        else                     { timed(run_lin, x, y, tl, cl); timed(run_eap, x, y, te, ce); }
        r.te.push_back(te); r.tl.push_back(tl); r.ce.push_back(ce); r.cl.push_back(cl);
        se += te; sl += tl; sce += ce; scl += cl;
      }
      double busy, other; lr.stop(busy, other);
      std::printf("round %d: sum EAP %.1f ms, sum linear %.1f ms, ratio wall %.3f cycles %.3f, on-core share %.2f, "
                  "load busy=%.0f%% other=%.0f%%\n", rd, se * 1e-6, sl * 1e-6, se / sl, sce / scl,
                  (sce + scl) / g_cycles_per_ns / (se + sl), busy, other);
      std::fflush(stdout);
    }
    double busy, other; lm.stop(busy, other);

    for (auto &r : R) {
      std::vector<double> ratio, cratio;
      for (int rd = 0; rd < rounds; ++rd) { ratio.push_back(r.te[rd] / r.tl[rd]); cratio.push_back(r.ce[rd] / r.cl[rd]); }
      r.te_med = median(r.te); r.tl_med = median(r.tl); r.ratio_med = median(ratio);
      r.ce_med = median(r.ce); r.cl_med = median(r.cl); r.cratio_med = median(cratio);
      const double frac = double(r.visited) / (double(L) * double(L));
      std::fprintf(csv, "%s,%s,%zu,%zu,%zu,%llu,%.6f,%.1f,%.1f,%.4f,%.0f,%.0f,%.4f,%.17g,%d,%d,%d\n", d.name.c_str(),
                   r.pr.within ? "within" : "across", r.pr.i, r.pr.j, L, (unsigned long long)r.visited, frac, r.te_med,
                   r.tl_med, r.ratio_med, r.ce_med, r.cl_med, r.cratio_med, r.value, int(r.eap_eq_lin),
                   int(r.eap_eq_full), int(r.counted_eq_eap));
      all_frac_ratio.push_back({frac, r.ratio_med, r.cratio_med});
    }
    std::fflush(csv);

    const char *kinds[] = {"within", "across", "pooled"};
    for (int w = 0; w < 3; ++w) {
      const Summary s = summarise(R, L, w);
      std::printf("SUMMARY %-16s L=%-5zu %-6s n=%3d EAP %9.1f us  linear %9.1f us  ratio wall med %.3f sum %.3f  "
                  "cycles med %.3f sum %.3f  computed %.3f [%.3f-%.3f]  EAP ns/visited %.3f  linear ns/cell %.3f  "
                  "mismatches %d\n",
                  d.name.c_str(), L, kinds[w], s.n, s.eap_ns * 1e-3, s.lin_ns * 1e-3, s.ratio, s.sum_ratio, s.cratio,
                  s.csum_ratio, s.frac_med, s.frac_lo, s.frac_hi, s.eap_ns_per_visited, s.lin_ns_per_cell, s.mism);
    }
    std::printf("timing phase load: busy=%.0f%% other=%.0f%% (sink %.6g)\n", busy, other, sink);
    std::fflush(stdout);
  }
  std::fclose(csv);

  // Crossover: median per-pair ratio against the computed fraction, every pair of every dataset in this run.
  std::printf("\nCROSSOVER (all pairs of this run): computed-fraction bin, pairs, median ratio t_EAP/t_linear (wall, cycles)\n");
  const double edges[] = {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0001};
  for (std::size_t b = 0; b + 1 < std::size(edges); ++b) {
    std::vector<double> rs, cs;
    for (const auto &e : all_frac_ratio)
      if (e.frac >= edges[b] && e.frac < edges[b + 1]) { rs.push_back(e.ratio); cs.push_back(e.cratio); }
    if (rs.empty()) continue;
    std::printf("  [%.2f, %.2f): %4zu  %.3f  %.3f\n", edges[b], std::min(edges[b + 1], 1.0), rs.size(), median(rs),
                median(cs));
  }
  return 0;
}
