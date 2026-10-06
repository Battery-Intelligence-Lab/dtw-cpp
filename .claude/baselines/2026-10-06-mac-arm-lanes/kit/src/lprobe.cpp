// lprobe: the fill's lane function as it ships, for the arm-lanes unit, base (254ecd3b) against head.
//
// Built twice (build.sh): against the base sources and against the head sources, each with its own
// dtwc/core/dtw_lanes.cpp compiled in (dtw_lanes.cpp's compile command, ThinLTO), so the lanes run
// through core::resolve_dtw_block_fn exactly as Problem's fill calls them, with that version's
// dtw_kernel.hpp, dtw_lanes and cell. The per-pair kernel is core::run_dtw (warping.hpp's dtwBanded).
//
//   lprobe check                                  bitwise sweep; prints hashes to compare across builds
//   lprobe time T D L band [reps] [cells]         single thread, the lanes alone: ns/cell, cycles/cell
//   lprobe fill T D N L band threads [reps] [ragged]  Problem's brute-force fill: Gcell/s, matrix hash
//
// T: f64|f32, D: L1|Sq. The clock is a dependent integer-add chain (one add per cycle on Apple cores).
#include "core/dtw_dispatch.hpp" // resolve_dtw_block_fn
#include "core/dtw_kernel.hpp"   // dtw_lanes, run_dtw, StandardCell, dtw_band_bounds
#include "core/dtw_cost.hpp"     // SpanL1Cost, SpanSquaredL2Cost
#include "core/public_distance.hpp"

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
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace dc = dtwc::core;

template <class T>
using BlockFn = std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>;

template <class T>
BlockFn<T> block_for(bool squared, int band)
{
  dc::DistanceConfig c;
  c.metric = squared ? dc::MetricType::SquaredL2 : dc::MetricType::L1;
  c.band = band;
  auto f = dc::resolve_dtw_block_fn<T>(c);
  if (!f) { std::fprintf(stderr, "no lane function\n"); std::exit(2); }
  return f;
}

// The per-pair distance the fill falls back to: run_dtw as dtwBanded calls it.
template <class T>
double pair_dtw(bool squared, const T *x, std::size_t nx, const T *y, std::size_t ny, int band)
{
  const T d = squared ? dc::run_dtw<dc::SpanSquaredL2Cost>(x, nx, y, ny, band, dc::StandardCell{}, T(-1))
                      : dc::run_dtw<dc::SpanL1Cost>(x, nx, y, ny, band, dc::StandardCell{}, T(-1));
  return dc::normalize_public_distance(d);
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

static std::size_t band_cells(std::size_t n, int band)
{
  std::size_t c = 0;
  for (std::size_t j = 0; j < n; ++j) {
    const auto [lo, hi] = dc::dtw_band_bounds(band, j, n);
    c += hi - lo;
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

// ---------------------------------------------------------------------------------------------
// check: every lane output over the sweep, hashed in a fixed order; each lane against the per-pair kernel
template <class T> std::uint64_t check_lanes(const char *tname, bool sq, int seeds)
{
  constexpr std::size_t W = dc::dtw_lanes<T>;
  const std::size_t NY = 64; // a multiple of every W
  std::uint64_t hash = kFnv0;
  long long outputs = 0, vs_pair = 0, nonfinite = 0;
  for (int seed = 0; seed < seeds; ++seed)
    for (int kind = 0; kind < 4; ++kind) {
      std::vector<std::uint64_t> hrow(1001, kFnv0);
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs, vs_pair, nonfinite)
      for (int n = 1; n <= 1000; ++n) {
        std::vector<T> x;
        std::vector<std::vector<T>> y(NY);
        const std::uint64_t base = 0x5eed0000ULL * (kind + 1) + 7919ULL * n + 0x9e3779b9ULL * seed;
        gen(x, n, base, kind);
        for (std::size_t w = 0; w < NY; ++w) gen(y[w], n, base + 1 + w, kind);
        if (kind == 2) for (std::size_t w = 0; w < NY; w += 4) y[w] = x; // zero-cost diagonals
        std::vector<std::span<const T>> ys(NY);
        for (std::size_t w = 0; w < NY; ++w) ys[w] = y[w];
        for (int band : { -1, 0, 1, n / 10, n }) {
          const BlockFn<T> block = block_for<T>(sq, band);
          std::vector<double> res(NY);
          for (std::size_t b = 0; b < NY; b += W)
            block(x, std::span<const std::span<const T>>(ys.data() + b, W), std::span<double>(res.data() + b, W));
          hrow[n] = fnv(hrow[n], res.data(), NY * sizeof(double));
          outputs += NY;
          for (std::size_t w = 0; w < NY; ++w) {
            const double p = pair_dtw<T>(sq, x.data(), n, y[w].data(), n, band);
            vs_pair += std::memcmp(&p, &res[w], sizeof(double)) != 0;
            nonfinite += !std::isfinite(res[w]);
          }
        }
      }
      for (int n = 1; n <= 1000; ++n) hash = fnv(hash, &hrow[n], 8);
    }
  std::printf("check lanes %s %-2s W=%-2zu seeds %d: outputs %lld  hash %016llx  lanes vs per-pair bitwise mismatches %lld  non-finite outputs %lld\n",
              tname, sq ? "Sq" : "L1", W, seeds, outputs, (unsigned long long)hash, vs_pair, nonfinite);
  std::fflush(stdout);
  return hash;
}

// the per-pair kernel over unequal lengths (unchanged by the unit: hashes must match too)
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
            const double r = pair_dtw<T>(sq, x.data(), nx, y.data(), ny, band);
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
  std::uint64_t all = kFnv0, h;
  for (bool sq : { false, true }) {
    h = check_lanes<double>("f64", sq, seeds); all = fnv(all, &h, 8);
    h = check_lanes<float>("f32", sq, seeds);  all = fnv(all, &h, 8);
  }
  for (bool sq : { false, true }) {
    h = check_pairs<double>("f64", sq, seeds); all = fnv(all, &h, 8);
    h = check_pairs<float>("f32", sq, seeds);  all = fnv(all, &h, 8);
  }
  std::printf("check all hashes combined %016llx\n", (unsigned long long)all);
  return 0;
}

// ---------------------------------------------------------------------------------------------
__attribute__((noinline)) static void add_chain(std::uint64_t n)
{
  asm volatile("1:\n"
               ".rept 100\n add x9, x9, #1\n .endr\n"
               "subs %[n], %[n], #1\n b.ne 1b\n"
               : [n] "+r"(n) : : "x9", "cc");
}
static double now_ns()
{
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double ns_per_cycle()
{
  const std::uint64_t it = 300000; // 30 M adds, ~6.5 ms at 4.6 GHz
  const double t0 = now_ns();
  add_chain(it);
  return (now_ns() - t0) / (double(it) * 100.0);
}
static double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class T> int time_mode(bool sq, std::size_t L, int band, int reps, double target_cells)
{
  constexpr std::size_t W = dc::dtw_lanes<T>;
  const std::size_t P = 128;
  std::vector<std::vector<T>> pool(P);
  for (std::size_t p = 0; p < P; ++p) gen(pool[p], L, 1000 + p, 1);
  const BlockFn<T> block = block_for<T>(sq, band);
  const std::size_t cells1 = band_cells(L, band);
  const std::size_t calls = std::max<std::size_t>(4, std::size_t(target_cells / double(W * cells1)));
  std::array<std::span<const T>, W> ys;
  std::array<double, W> out{};
  double checksum = 0;
  auto run = [&] {
    for (std::size_t c = 0; c < calls; ++c) {
      for (std::size_t w = 0; w < W; ++w) ys[w] = pool[(c * W + w + 1) % P];
      block(pool[c % P], ys, out);
      checksum += out[c % W];
    }
  };
  for (int i = 0; i < 20; ++i) ns_per_cycle(); // warm-up, ~130 ms
  run();
  std::vector<double> nspc, cyc_meas, ghz;
  for (int r = 0; r < reps; ++r) {
    const double c0 = ns_per_cycle();
    const double t0 = now_ns();
    run();
    const double ns = now_ns() - t0;
    const double c1 = ns_per_cycle();
    const double cells = double(calls) * double(W) * double(cells1);
    nspc.push_back(ns / cells);
    cyc_meas.push_back(ns / ((c0 + c1) / 2) / cells);
    ghz.push_back(2 / (c0 + c1));
  }
  const double m = median(nspc);
  std::printf("time %s %-2s L %-4zu band %-4d W %-2zu ns/cell %.5f (min %.5f)  cycles/cell@4.59GHz %.4f  "
              "cycles/cell@measured %.4f  clock %.3f GHz  Gcell/s %.3f  checksum %.17g\n",
              sizeof(T) == 8 ? "f64" : "f32", sq ? "Sq" : "L1", L, band, W, m,
              *std::min_element(nspc.begin(), nspc.end()), m * 4.59, median(cyc_meas), median(ghz), 1 / m, checksum);
  return 0;
}

// ---------------------------------------------------------------------------------------------
// fill: Problem::fillDistanceMatrix_BruteForce (Problem.cpp:772-813): each row takes its columns W at a
// time through the lane function where they are as long as the row's series, then pair by pair;
// run_openmp(fill_row, N, true, 8): schedule(dynamic, max(1, N / (threads * 8))).
template <class T> int fill_mode(bool sq, std::size_t N, std::size_t L, int band, int threads, int reps, bool ragged)
{
  constexpr std::size_t W = dc::dtw_lanes<T>;
  std::vector<std::vector<T>> S(N);
  std::uint64_t s = 99;
  for (std::size_t i = 0; i < N; ++i) gen(S[i], ragged ? L - 10 + splitmix(s) % 21 : L, 777 + i, 1);
  const BlockFn<T> block = block_for<T>(sq, band);
  const std::function<double(std::span<const T>, std::span<const T>)> pair =
    [sq, band](std::span<const T> x, std::span<const T> y) { return pair_dtw<T>(sq, x.data(), x.size(), y.data(), y.size(), band); };
  auto tri = [](std::size_t i, std::size_t j) { if (i < j) std::swap(i, j); return i * (i + 1) / 2 + j; };
  std::vector<double> M;
  const long chunk = std::max<long>(1, long(N) / (threads * 8));
  double cells = 0;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j)
      cells += S[i].size() == S[j].size() ? double(band_cells(S[i].size(), band)) : double(S[i].size()) * double(S[j].size());
  std::vector<double> times;
  std::uint64_t hash = 0;
  for (int r = 0; r < reps + 1; ++r) { // the first run warms the threads and the buffers, untimed
    M.assign(N * (N + 1) / 2, std::nan(""));
    const double t0 = now_ns();
#pragma omp parallel num_threads(threads)
    {
      std::array<std::span<const T>, W> ys;
      std::array<double, W> d;
#pragma omp for schedule(dynamic, chunk) nowait
      for (long il = 0; il < long(N); ++il) {
        const std::size_t i = std::size_t(il);
        M[tri(i, i)] = 0.0;
        const std::span<const T> x = S[i];
        for (std::size_t j = i + 1; j < N; j += W) {
          const std::size_t count = std::min(W, N - j);
          bool equal = true;
          for (std::size_t w = 0; w < W; ++w) {
            ys[w] = S[j + std::min(w, count - 1)];
            equal = equal && ys[w].size() == x.size();
          }
          if (!equal) continue;
          block(x, ys, d);
          for (std::size_t w = 0; w < count; ++w) M[tri(i, j + w)] = d[w];
        }
        for (std::size_t j = i + 1; j < N; ++j)
          if (std::isnan(M[tri(i, j)])) M[tri(i, j)] = pair(x, S[j]);
      }
    }
    const double ns = now_ns() - t0;
    if (r > 0) times.push_back(ns);
    hash = fnv(kFnv0, M.data(), M.size() * sizeof(double));
  }
  const double m = median(times);
  std::printf("fill %s %-2s threads %2d N %-5zu L %-4zu%s band %-4d W %-2zu ms %.2f (min %.2f)  Gcell/s %.2f  matrix hash %016llx\n",
              sizeof(T) == 8 ? "f64" : "f32", sq ? "Sq" : "L1", threads, N, L, ragged ? " ragged+-10" : "", band, W,
              m / 1e6, *std::min_element(times.begin(), times.end()) / 1e6, cells / m, (unsigned long long)hash);
  return 0;
}

int main(int argc, char **argv)
{
  const std::string mode = argc > 1 ? argv[1] : "";
  auto arg = [&](int k, const char *def) { return argc > k ? argv[k] : def; };
  if (mode == "check") return check_mode(std::atoi(arg(2, "3")));
  if (mode == "time" || mode == "fill") {
    const bool f32 = std::string(arg(2, "f64")) == "f32";
    const bool sq = std::string(arg(3, "L1")) == "Sq";
    if (mode == "time") {
      const std::size_t L = std::strtoul(arg(4, "100"), nullptr, 10);
      const int band = std::atoi(arg(5, "-1")), reps = std::atoi(arg(6, "5"));
      const double cells = std::atof(arg(7, "120e6"));
      return f32 ? time_mode<float>(sq, L, band, reps, cells) : time_mode<double>(sq, L, band, reps, cells);
    }
    const std::size_t N = std::strtoul(arg(4, "2000"), nullptr, 10), L = std::strtoul(arg(5, "100"), nullptr, 10);
    const int band = std::atoi(arg(6, "-1")), threads = std::atoi(arg(7, "18")), reps = std::atoi(arg(8, "3"));
    const bool ragged = std::atoi(arg(9, "0")) != 0;
    return f32 ? fill_mode<float>(sq, N, L, band, threads, reps, ragged)
               : fill_mode<double>(sq, N, L, band, threads, reps, ragged);
  }
  std::fprintf(stderr, "usage: lprobe check [seeds] | time T D L band [reps] [cells] | fill T D N L band threads [reps] [ragged]\n");
  return 2;
}
