// rprobe: the kit probe of unit RL (ragged lanes: the CPU fill's lanes take series of different lengths without a
// band), base against the candidate B and the first candidate A, under clang++ and cl (Windows) and GCC 13.3 (WSL). From XL's xprobe.cpp
// (.claude/baselines/2026-10-08-xl-x86-lanes-128/kit/xprobe.cpp) and its ports by XG and MV (the route bound).
//
// Built per compiler and version (build_clang.sh, build_msvc.bat, build_gcc.sh) against that version's headers
// (src/<version>/dtwc) and its build tree's dtwc_core library, with dtwc_cl.cpp's flags, linked as dtwc_cl is. So the
// sweeps run the shipped block function (core::resolve_dtw_block_fn) and per-pair function (core::resolve_dtw_fn),
// and `fill` runs that version's Problem::fill_distance_matrix itself: its loop, order and schedule.
//
//   rprobe info                                   W per type, compiler, ISA macros
//   rprobe check [seeds]                          XL's sweep: equal-length blocks (bands -1, 0, 1, n/10, n) through the
//                                                 block function against the per-pair route; its hashes must equal base's
//   rprobe ragged [seeds]                         candidate only: blocks of lanes whose lengths are drawn in [1, n],
//                                                 unbanded, every lane against the per-pair route (a base block function
//                                                 reads past a shorter series)
//   rprobe fill <T> <D> <N> <lo> <hi> <band> <threads> <reps> [cpu]
//                                                 Problem::fill_distance_matrix on N random walks of lengths drawn in
//                                                 [lo, hi] (lo == hi: equal); threads 0 = the OpenMP default; threads 1
//                                                 pins the calling thread to logical CPU <cpu> (Windows; in WSL the
//                                                 driver runs taskset and this checks it). One untimed fill, then <reps>.
//   rprobe waste <N> <lo> <hi>                    counts only: the lanes' cells per useful cell in base's index
//                                                 order and in A's and B's groupings, at W 8, 16 and 32
//
// T: f64|f32, D: L1|Sq, band -1 = unbanded. The clock (clang and GCC builds; cl has no x64 inline assembly) is a
// dependent chain of register adds, one per cycle, on the calling thread.
#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sched.h>
#endif

#include "Data.hpp"
#include "Problem.hpp"
#include "core/dtw_dispatch.hpp" // resolve_dtw_fn, resolve_dtw_block_fn
#include "core/dtw_kernel.hpp"   // dtw_lanes, dtw_band_bounds
#include "core/dtw_options.hpp"  // DistanceConfig, MetricType
#if defined(RL_COUNT)
#include "count/count_lanes.hpp" // rl_count::lane_calls: the counting build only (build_count.*)
#endif

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace dc = dtwc::core;

template <class T>
using BlockFn = std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>;
template <class T>
using PairFn = std::function<double(std::span<const T>, std::span<const T>)>;

static dc::DistanceConfig config_for(bool squared, int band)
{
  dc::DistanceConfig c;
  c.metric = squared ? dc::MetricType::SquaredL2 : dc::MetricType::L1;
  c.band = band;
  return c;
}

template <class T> BlockFn<T> block_for(bool squared, int band)
{
  auto f = dc::resolve_dtw_block_fn<T>(config_for(squared, band));
  if (!f) {
    std::fprintf(stderr, "no lane function\n");
    std::exit(2);
  }
  return f;
}

template <class T> PairFn<T> pair_for(bool squared, int band)
{
  const dc::DistanceConfig c = config_for(squared, band);
  dc::validate(c, sizeof(T) == 4);
  return dc::resolve_dtw_fn<T>(c, dtwc::Data{});
}

// ---------------------------------------------------------------------------------------------
static std::uint64_t splitmix(std::uint64_t &s)
{
  std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double unit(std::uint64_t &s) { return double(splitmix(s) >> 11) * 0x1.0p-53; }

// 0 uniform [-1,1]; 1 random walk; 2 integers {0,1,2} (ties, zero costs); 3 huge (+-max/2..max: costs overflow to inf)
template <class T> void gen(std::vector<T> &v, std::size_t n, std::uint64_t seed, int kind)
{
  v.resize(n);
  std::uint64_t s = seed;
  double walk = 0;
  for (std::size_t i = 0; i < n; ++i) {
    switch (kind) {
    case 0: v[i] = T(2 * unit(s) - 1); break;
    case 1: walk += 2 * unit(s) - 1; v[i] = T(walk); break;
    case 2: v[i] = T(splitmix(s) % 3); break;
    default: {
      const double m = double(std::numeric_limits<T>::max());
      const double u = unit(s);
      v[i] = T(u < 0.25 ? 0.0 : (u < 0.6 ? m * (0.5 + unit(s) / 2) : -m * (0.5 + unit(s) / 2)));
    }
    }
  }
}

// The cells of the DP of one pair: nx * ny unbanded; under a band (equal lengths) dtw_band_bounds' window.
static double pair_cells(std::size_t nx, std::size_t ny, int band)
{
  if (band < 0 || nx != ny) return double(nx) * double(ny);
  double c = 0;
  for (std::size_t j = 0; j < nx; ++j) {
    const auto [lo, hi] = dc::dtw_band_bounds(band, j, nx);
    c += double(hi - lo);
  }
  return c;
}

static std::uint64_t fnv(std::uint64_t h, const void *p, std::size_t len)
{
  const auto *b = static_cast<const unsigned char *>(p);
  for (std::size_t i = 0; i < len; ++i) h = (h ^ b[i]) * 0x100000001b3ULL;
  return h;
}
constexpr std::uint64_t kFnv0 = 0xcbf29ce484222325ULL;

static const char *compiler()
{
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(_MSC_VER)
  static char buf[64];
  std::snprintf(buf, sizeof buf, "cl %d", _MSC_FULL_VER);
  return buf;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#else
  return "other";
#endif
}

// The route bound of tests/support/dtw_route_bound.hpp, dtw_routes_agree<T>: |a - b| <= 2 (nx + ny - 1) eps(T)
// max(|a|, |b|), an infinity agreeing only with itself. `dev` is |a - b| over (nx + ny - 1) eps(T) max(|a|, |b|), which
// the bound holds to 2.
template <class T> bool outside_bound(double a, double b, std::size_t nx, std::size_t ny, double &dev)
{
  dev = 0;
  if (std::isinf(a) || std::isinf(b)) return a != b;
  if (a == b) return false;
  dev = std::abs(a - b) / (double(nx + ny - 1) * double(std::numeric_limits<T>::epsilon()) * std::max(std::abs(a), std::abs(b)));
  return !(dev <= 2.0);
}

// ---------------------------------------------------------------------------------------------
// check (XL's sweep, unchanged): NY = 2 * 32 + 5 partners per x, so every W (8, 16, 32) meets full blocks and a tail
// block that repeats its last column in the lanes past it, as the fill does; every output against the per-pair
// route, bitwise, and against the route bound.
template <class T> std::uint64_t check_lanes(const char *tname, bool sq, int seeds)
{
  constexpr std::size_t W = dc::dtw_lanes<T>;
  const std::size_t NY = 2 * 32 + 5;
  std::uint64_t hash = kFnv0;
  long long outputs = 0, vs_pair = 0, nonfinite = 0, outside = 0;
  double worst = 0;
  for (int seed = 0; seed < seeds; ++seed)
    for (int kind = 0; kind < 4; ++kind) {
      std::vector<std::uint64_t> hrow(1001, kFnv0);
      std::vector<double> wrow(1001, 0.0); // per n: cl's OpenMP has no max reduction
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs, vs_pair, nonfinite, outside)
      for (int n = 1; n <= 1000; ++n) {
        std::vector<T> x;
        std::vector<std::vector<T>> y(NY);
        const std::uint64_t base = 0x5eed0000ULL * (kind + 1) + 7919ULL * n + 0x9e3779b9ULL * seed;
        gen(x, n, base, kind);
        for (std::size_t w = 0; w < NY; ++w) gen(y[w], n, base + 1 + w, kind);
        if (kind == 2)
          for (std::size_t w = 0; w < NY; w += 4) y[w] = x; // zero-cost diagonals
        for (int band : { -1, 0, 1, n / 10, n }) {
          const BlockFn<T> block = block_for<T>(sq, band);
          const PairFn<T> pair = pair_for<T>(sq, band);
          std::vector<double> res(NY);
          std::array<std::span<const T>, W> ys;
          std::array<double, W> d;
          for (std::size_t j = 0; j < NY; j += W) {
            const std::size_t count = std::min(W, NY - j);
            for (std::size_t w = 0; w < W; ++w) ys[w] = y[j + std::min(w, count - 1)];
            block(x, ys, d);
            for (std::size_t w = 0; w < count; ++w) res[j + w] = d[w];
          }
          hrow[n] = fnv(hrow[n], res.data(), NY * sizeof(double));
          outputs += NY;
          for (std::size_t w = 0; w < NY; ++w) {
            const double p = pair(x, y[w]);
            double dev;
            vs_pair += std::memcmp(&p, &res[w], sizeof(double)) != 0;
            nonfinite += !std::isfinite(res[w]);
            outside += outside_bound<T>(res[w], p, x.size(), y[w].size(), dev);
            wrow[n] = std::max(wrow[n], dev);
          }
        }
      }
      for (int n = 1; n <= 1000; ++n) {
        hash = fnv(hash, &hrow[n], 8);
        worst = std::max(worst, wrow[n]);
      }
    }
  std::printf("check lanes %s %-2s W=%-2zu seeds %d: outputs %lld  hash %016llx  lanes vs per-pair bitwise "
              "mismatches %lld  non-finite outputs %lld  outside the route bound %lld  worst dev %.4g\n",
              tname, sq ? "Sq" : "L1", W, seeds, outputs, (unsigned long long)hash, vs_pair, nonfinite, outside, worst);
  std::fflush(stdout);
  return hash;
}

// The per-pair route over unequal lengths (XL's): the unit leaves it alone, so its hashes must match too.
template <class T> std::uint64_t check_pairs(const char *tname, bool sq, int seeds)
{
  std::uint64_t hash = kFnv0;
  long long outputs = 0;
  for (int seed = 0; seed < seeds; ++seed)
    for (int kind = 0; kind < 4; ++kind) {
      std::vector<std::uint64_t> hrow(1001, kFnv0);
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs)
      for (int nx = 1; nx <= 1000; ++nx) {
        std::uint64_t s = 0xabcdefULL * (kind + 1) + 104729ULL * nx + 0x9e3779b9ULL * seed;
        const int nys[3] = { nx, 1 + int(splitmix(s) % 1000), std::max(1, nx + int(splitmix(s) % 21) - 10) };
        for (int ny : nys) {
          std::vector<T> x, y;
          gen(x, nx, s + 11, kind);
          gen(y, ny, s + 29, kind);
          const int L = std::max(nx, ny);
          for (int band : { -1, 0, 1, L / 10, L, std::abs(nx - ny) }) {
            const double r = pair_for<T>(sq, band)(x, y);
            hrow[nx] = fnv(hrow[nx], &r, sizeof r);
            ++outputs;
          }
        }
      }
      for (int n = 1; n <= 1000; ++n) hash = fnv(hash, &hrow[n], 8);
    }
  std::printf("check pair  %s %-2s       seeds %d: outputs %lld  hash %016llx\n", tname, sq ? "Sq" : "L1", seeds,
              outputs, (unsigned long long)hash);
  std::fflush(stdout);
  return hash;
}

static int check_mode(int seeds)
{
  std::printf("check %s W f64 %zu f32 %zu\n", compiler(), dc::dtw_lanes<double>, dc::dtw_lanes<float>);
  std::uint64_t all = kFnv0, h;
  for (bool sq : { false, true }) {
    h = check_lanes<double>("f64", sq, seeds);
    all = fnv(all, &h, 8);
    h = check_lanes<float>("f32", sq, seeds);
    all = fnv(all, &h, 8);
  }
  for (bool sq : { false, true }) {
    h = check_pairs<double>("f64", sq, seeds);
    all = fnv(all, &h, 8);
    h = check_pairs<float>("f32", sq, seeds);
    all = fnv(all, &h, 8);
  }
  std::printf("check all hashes combined %016llx\n", (unsigned long long)all);
  return 0;
}

// ragged (candidate only): for each n in 1..1000, x and NY partners of lengths drawn in [1, n], the four data kinds,
// blocks of W partners as the fill forms them (a tail block repeating its last column), unbanded; every lane output
// against the per-pair route, bitwise and against the route bound.
template <class T> std::uint64_t check_ragged(const char *tname, bool sq, int seeds)
{
  constexpr std::size_t W = dc::dtw_lanes<T>;
  const std::size_t NY = 2 * 32 + 5;
  const BlockFn<T> block = block_for<T>(sq, -1);
  const PairFn<T> pair = pair_for<T>(sq, -1);
  std::uint64_t hash = kFnv0;
  long long outputs = 0, vs_pair = 0, nonfinite = 0, outside = 0, longer = 0, shorter = 0;
  double worst = 0;
  for (int seed = 0; seed < seeds; ++seed)
    for (int kind = 0; kind < 4; ++kind) {
      std::vector<std::uint64_t> hrow(1001, kFnv0);
      std::vector<double> wrow(1001, 0.0);
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs, vs_pair, nonfinite, outside, longer, shorter)
      for (int n = 1; n <= 1000; ++n) {
        std::uint64_t s = 0x7a66edULL * (kind + 1) + 15485863ULL * n + 0x9e3779b9ULL * seed;
        std::vector<T> x;
        std::vector<std::vector<T>> y(NY);
        gen(x, 1 + splitmix(s) % n, s + 1, kind);
        for (std::size_t w = 0; w < NY; ++w) gen(y[w], 1 + splitmix(s) % n, s + 2 + w, kind);
        std::vector<double> res(NY);
        std::array<std::span<const T>, W> ys;
        std::array<double, W> d;
        for (std::size_t j = 0; j < NY; j += W) {
          const std::size_t count = std::min(W, NY - j);
          for (std::size_t w = 0; w < W; ++w) ys[w] = y[j + std::min(w, count - 1)];
          block(x, ys, d);
          for (std::size_t w = 0; w < count; ++w) res[j + w] = d[w];
        }
        hrow[n] = fnv(hrow[n], res.data(), NY * sizeof(double));
        outputs += NY;
        for (std::size_t w = 0; w < NY; ++w) {
          const double p = pair(x, y[w]);
          double dev;
          vs_pair += std::memcmp(&p, &res[w], sizeof(double)) != 0;
          nonfinite += !std::isfinite(res[w]);
          outside += outside_bound<T>(res[w], p, x.size(), y[w].size(), dev);
          wrow[n] = std::max(wrow[n], dev);
          longer += y[w].size() > x.size();
          shorter += y[w].size() < x.size();
        }
      }
      for (int n = 1; n <= 1000; ++n) {
        hash = fnv(hash, &hrow[n], 8);
        worst = std::max(worst, wrow[n]);
      }
    }
  std::printf("ragged lanes %s %-2s W=%-2zu seeds %d: outputs %lld (y longer than x %lld, shorter %lld)  hash %016llx  "
              "lanes vs per-pair bitwise mismatches %lld  non-finite outputs %lld  outside the route bound %lld  "
              "worst dev %.4g\n",
              tname, sq ? "Sq" : "L1", W, seeds, outputs, longer, shorter, (unsigned long long)hash, vs_pair, nonfinite,
              outside, worst);
  std::fflush(stdout);
  return hash;
}

static int ragged_mode(int seeds)
{
  std::printf("ragged %s W f64 %zu f32 %zu\n", compiler(), dc::dtw_lanes<double>, dc::dtw_lanes<float>);
  std::uint64_t all = kFnv0, h;
  for (bool sq : { false, true }) {
    h = check_ragged<double>("f64", sq, seeds);
    all = fnv(all, &h, 8);
    h = check_ragged<float>("f32", sq, seeds);
    all = fnv(all, &h, 8);
  }
  std::printf("ragged all hashes combined %016llx\n", (unsigned long long)all);
  return 0;
}

// ---------------------------------------------------------------------------------------------
static double now_ns()
{
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#if defined(__x86_64__)
// add reg, reg: this core folds a chain of add reg, imm at rename (XL's tools/clk), so the addend is a register.
__attribute__((noinline)) static void add_chain(std::uint64_t n, std::uint64_t b)
{
  std::uint64_t a = 0;
  asm volatile("1:\n\t.rept 100\n\taddq %[b], %[a]\n\t.endr\n\tdecq %[n]\n\tjnz 1b"
               : [n] "+r"(n), [a] "+r"(a)
               : [b] "r"(b)
               : "cc");
}
static volatile std::uint64_t g_one = 1;
static double ghz() // 30 M dependent adds, about 6 ms at 5 GHz
{
  const std::uint64_t it = 300000;
  const double t0 = now_ns();
  add_chain(it, g_one);
  return double(it) * 100.0 / (now_ns() - t0);
}
#else
static double ghz() { return 0; }
#endif

static double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

// Windows: the calling thread on logical CPU <cpu> alone, at the highest priority. WSL: the driver's taskset.
static bool pin(int cpu)
{
#if defined(_WIN32)
  return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu) != 0
         && SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) != 0;
#else
  cpu_set_t s;
  CPU_ZERO(&s);
  return sched_getaffinity(0, sizeof s, &s) == 0 && CPU_COUNT(&s) == 1 && CPU_ISSET(cpu, &s);
#endif
}

static std::vector<std::size_t> lengths_of(std::size_t N, std::size_t lo, std::size_t hi)
{
  std::vector<std::size_t> n(N);
  std::uint64_t s = 99;
  for (auto &v : n) v = lo + splitmix(s) % (hi - lo + 1);
  return n;
}

// How a fill groups a row's columns into blocks of W lanes. index: base's (row i against i + 1 .., in index order);
// A: the first candidate's (rows and columns longest first: row a is series order[a] against order[a + 1 ..]);
// B: the candidate's (row i against the series after it, taken longest first).
enum class Grouping { index, A, B };

// The lanes' cells of a fill of these lengths at W lanes, grouped as `g`, a row's last block repeating its last
// column. Returns {padding, duplicates, useful}: cells a lane computes past its own length, cells of the repeated tail
// lanes, cells of the pairs themselves.
static std::array<double, 3> lane_cells(const std::vector<std::size_t> &n, Grouping g, std::size_t W)
{
  const std::size_t N = n.size();
  std::vector<std::size_t> order(N), c(W);
  std::iota(order.begin(), order.end(), std::size_t{ 0 });
  if (g != Grouping::index)
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return n[a] > n[b]; });
  double pad = 0, dup = 0, useful = 0;
  for (std::size_t r = 0; r < N; ++r) {
    const std::size_t i = g == Grouping::A ? order[r] : r;
    const double nx = double(n[i]);
    for (std::size_t b = g == Grouping::B ? 0 : r + 1; b < N;) {
      std::size_t count = 0;
      for (; count < W && b < N; ++b)
        if (g != Grouping::B || order[b] > i) c[count++] = order[b];
      if (count == 0) break;
      std::size_t longest = 0;
      double sum = 0;
      for (std::size_t w = 0; w < count; ++w) {
        longest = std::max(longest, n[c[w]]);
        sum += double(n[c[w]]);
      }
      useful += nx * sum;
      pad += nx * (double(count) * double(longest) - sum);
      dup += nx * double(W - count) * double(longest);
    }
  }
  return { pad, dup, useful };
}

static int waste_mode(std::size_t N, std::size_t lo, std::size_t hi)
{
  const auto n = lengths_of(N, lo, hi);
  for (std::size_t W : { 8, 16, 32 })
    for (Grouping g : { Grouping::index, Grouping::A, Grouping::B }) {
      const auto [pad, dup, useful] = lane_cells(n, g, W);
      const double all = pad + dup + useful;
      std::printf("waste N %zu L %zu-%zu W %zu %s: padding %.4f  tail duplicates %.4f  of the lanes' cells; "
                  "computed/useful %.4f\n",
                  N, lo, hi, W, g == Grouping::index ? "index order" : g == Grouping::A ? "A          " : "B          ",
                  pad / all, dup / all, all / useful);
    }
  return 0;
}

// fill: Problem::fill_distance_matrix of the linked version on N random walks (kind 1) of lengths drawn in [lo, hi].
// Gcell/s counts each pair's own DP cells (pair_cells), the same for both versions; the matrix hash is FNV over the
// packed lower triangle read through get(i, j).
template <class T> int fill_mode(bool sq, std::size_t N, std::size_t lo, std::size_t hi, int band, int threads,
                                 int reps, int cpu)
{
  if (threads == 1 && !pin(cpu)) {
    std::fprintf(stderr, "could not pin to logical CPU %d alone (WSL: run under taskset -c %d)\n", cpu, cpu);
    return 3;
  }
  if (threads > 0) omp_set_num_threads(threads);
  const auto n = lengths_of(N, lo, hi);
  std::vector<std::vector<T>> rows(N);
  std::vector<std::string> names(N);
  double cells = 0;
  for (std::size_t i = 0; i < N; ++i) {
    gen(rows[i], n[i], 777 + i, 1);
    names[i] = "s" + std::to_string(i);
    for (std::size_t j = i + 1; j < N; ++j) cells += pair_cells(n[i], n[j], band);
  }
  dtwc::Problem prob("rprobe");
  prob.set_data(dtwc::Data(std::move(rows), std::move(names)));
  prob.set_band(band);
  prob.set_metric(sq ? dc::MetricType::SquaredL2 : dc::MetricType::L1);
  std::vector<double> times, clocks;
  for (int r = 0; r < reps + 1; ++r) { // the first fill warms the threads and the buffers, untimed
    prob.refresh_distance_matrix();
    const double c0 = ghz();
    const double t0 = now_ns();
    prob.fill_distance_matrix();
    const double ns = now_ns() - t0;
    const double c1 = ghz();
    if (r > 0) {
      times.push_back(ns);
      clocks.push_back((c0 + c1) / 2);
    }
  }
  std::uint64_t hash = kFnv0;
  const auto &m = prob.distance_matrix();
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      const double v = m.get(i, j);
      hash = fnv(hash, &v, sizeof v);
    }
  // The ragged lanes' cells per useful cell, padding and repeated tail lanes, in B's grouping and in A's.
  auto waste = [&](Grouping g) {
    if (band >= 0 || lo == hi) return std::array<double, 3>{ 1.0, 0.0, 0.0 };
    const auto [pad, dup, useful] = lane_cells(n, g, dc::dtw_lanes<T>);
    const double all = pad + dup + useful;
    return std::array<double, 3>{ all / useful, pad / all, dup / all };
  };
  const auto wb = waste(Grouping::B), wa = waste(Grouping::A);
  const double t = median(times);
  std::printf("fill %s %-2s threads %2d N %-5zu L %zu-%zu band %-4d W %-2zu ms %.2f (min %.2f max %.2f)  Gcell/s %.3f  "
              "clock %.3f GHz (min %.3f max %.3f)  lanes computed/useful B %.4f (padding %.4f, tail duplicates %.4f) "
              "A %.4f (padding %.4f, tail duplicates %.4f)  matrix hash %016llx\n",
              sizeof(T) == 8 ? "f64" : "f32", sq ? "Sq" : "L1", threads > 0 ? threads : omp_get_max_threads(), N, lo, hi,
              band, dc::dtw_lanes<T>, t / 1e6, *std::min_element(times.begin(), times.end()) / 1e6,
              *std::max_element(times.begin(), times.end()) / 1e6, cells / t, median(clocks),
              *std::min_element(clocks.begin(), clocks.end()), *std::max_element(clocks.begin(), clocks.end()), wb[0],
              wb[1], wb[2], wa[0], wa[1], wa[2], (unsigned long long)hash);
#if defined(RL_COUNT)
  std::printf("lane calls %lld in %d fills\n", rl_count::lane_calls.load(), reps + 1);
#endif
  return 0;
}

int main(int argc, char **argv)
{
  const std::string mode = argc > 1 ? argv[1] : "";
  auto arg = [&](int k, const char *def) { return argc > k ? argv[k] : def; };
  auto num = [&](int k, const char *def) { return std::strtoull(arg(k, def), nullptr, 10); };
  if (mode == "info") {
    std::printf("info %s W f64 %zu f32 %zu x86_64 %d AVX2 %d FMA %d omp threads %d\n", compiler(),
                dc::dtw_lanes<double>, dc::dtw_lanes<float>,
#if defined(__x86_64__) || defined(_M_X64)
                1,
#else
                0,
#endif
#if defined(__AVX2__)
                1,
#else
                0,
#endif
#if defined(__FMA__)
                1,
#else
                0,
#endif
                omp_get_max_threads());
    return 0;
  }
  if (mode == "check") return check_mode(std::atoi(arg(2, "3")));
  if (mode == "ragged") return ragged_mode(std::atoi(arg(2, "3")));
  if (mode == "waste") return waste_mode(num(2, "1000"), num(3, "90"), num(4, "110"));
  if (mode == "fill") {
    const bool f32 = std::string(arg(2, "f64")) == "f32", sq = std::string(arg(3, "L1")) == "Sq";
    const std::size_t N = num(4, "1000"), lo = num(5, "90"), hi = num(6, "110");
    const int band = std::atoi(arg(7, "-1")), threads = std::atoi(arg(8, "0")), reps = std::atoi(arg(9, "3"));
    const int cpu = std::atoi(arg(10, "12"));
    if (lo == 0 || hi < lo) return 2;
    return f32 ? fill_mode<float>(sq, N, lo, hi, band, threads, reps, cpu)
               : fill_mode<double>(sq, N, lo, hi, band, threads, reps, cpu);
  }
  std::fprintf(stderr, "usage: rprobe info | check [seeds] | ragged [seeds] | waste N lo hi | "
                       "fill T D N lo hi band threads reps [cpu]\n");
  return 2;
}
