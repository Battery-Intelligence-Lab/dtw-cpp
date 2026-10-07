// pprobe: the per-pair DTW kernels as they ship, for the pair-2col unit: base (e26d5680) against k1 (kernel 1
// two columns per pass) and head (kernels 1 and 2 two columns per pass).
//
// Built once per version and code placement (build.sh): this file and that version's dtwc/core/dtw_lanes.cpp,
// with the compile command of the library's sources (ThinLTO), against that version's headers. The per-pair
// distance is the one Problem's fill runs (dtw_dispatch.cpp, make_standard): a std::function over
// normalize_public_distance(dtwBanded<T>(x, y, band, T(-1), metric)); the lanes are resolve_dtw_block_fn's.
//
//   pprobe check [seeds]                              bitwise sweep through the public per-pair functions
//   pprobe time T D nx ny band [reps] [cells]         single thread, one pair at a time: ns/cell, cycles/cell
//   pprobe fill T D N L band threads [reps] [ragged]  Problem's brute-force fill: Gcell/s, matrix hash
//
// T: f64|f32, D: L1|Sq. The clock is a dependent integer-add chain (one add per cycle on Apple cores).
#include "core/dtw_dispatch.hpp"     // resolve_dtw_block_fn
#include "core/dtw_kernel.hpp"       // dtw_band_bounds, dtw_lanes
#include "core/public_distance.hpp"  // normalize_public_distance
#include "soft_dtw.hpp"              // soft_dtw
#include "warping.hpp"               // dtwBanded
#include "warping_adtw.hpp"          // adtwBanded
#include "warping_missing.hpp"       // dtwMissing_banded
#include "warping_missing_arow.hpp"  // dtwAROW_banded
#include "warping_wdtw.hpp"          // wdtwBanded

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
#include <vector>

namespace dc = dtwc::core;
using dc::MetricType;

template <class T>
using PairFn = std::function<double(std::span<const T>, std::span<const T>)>;
template <class T>
using BlockFn = std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>;

// What resolve_dtw_fn returns for Standard DTW, ndim 1 (dtw_dispatch.cpp, make_standard).
template <class T> PairFn<T> pair_for(bool squared, int band)
{
  const MetricType metric = squared ? MetricType::SquaredL2 : MetricType::L1;
  return [band, metric](std::span<const T> x, std::span<const T> y) -> double {
    return dc::normalize_public_distance(dtwc::dtwBanded<T>(x, y, band, T(-1), metric));
  };
}

template <class T> BlockFn<T> block_for(bool squared, int band)
{
  dc::DistanceConfig c;
  c.metric = squared ? MetricType::SquaredL2 : MetricType::L1;
  c.band = band;
  auto f = dc::resolve_dtw_block_fn<T>(c);
  if (!f) { std::fprintf(stderr, "no lane function\n"); std::exit(2); }
  return f;
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

// 0 uniform [-1,1]; 1 random walk; 2 integers {0,1,2} (ties, zero costs); 3 huge (+-max/2..max: costs overflow
// to inf); 4 random walk with about one sample in six NaN (missing)
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
    case 3: {
      const double m = double(std::numeric_limits<T>::max());
      const double u = unit(s);
      v[i] = T(u < 0.25 ? 0.0 : (u < 0.6 ? m * (0.5 + unit(s) / 2) : -m * (0.5 + unit(s) / 2)));
      break;
    }
    default: walk += 2 * unit(s) - 1; v[i] = unit(s) < 1.0 / 6 ? std::numeric_limits<T>::quiet_NaN() : T(walk);
    }
  }
}

// Cells the per-pair kernels compute for one pair (dtw_kernel_banded / dtw_kernel_linear), 0 if no path fits.
static double pair_cells(std::size_t nx, std::size_t ny, int band)
{
  const std::size_t ns = std::min(nx, ny), nl = std::max(nx, ny);
  if (ns == 0) return 0;
  if (band < 0 || std::size_t(band) >= nl - 1 || ns == 1) return double(ns) * double(nl);
  if (nl - ns > std::size_t(band)) return 0;
  double c = 0;
  for (std::size_t j = 0; j < ns; ++j) {
    const auto [lo, hi] = dc::dtw_band_bounds(band, j, nl);
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

// ---------------------------------------------------------------------------------------------
// check: every per-pair entry the kernels serve, over lengths 1..1000 (a partner of the same length, one of a
// random length, one within 10), bands -1, 0, 1, L/10, L, |nx-ny|, five data kinds and `seeds` seeds; each
// output hashed in a fixed order, so two builds agree bit for bit iff their hashes do.
enum Variant {
  kDtwL1, kDtwSq, kEaL1Zero, kEaL1Half, kEaL1ThreeQ, kEaL1Exact, kEaSqHalf, kAdtw, kAdtwEaHalf, kWdtw,
  kZeroL1, kZeroSq, kZeroEaHalf, kArowL1, kArowSq, kSoft, kVariants
};
static const char *kName[kVariants] = {
  "dtw L1", "dtw Sq", "dtw L1 abandon 0", "dtw L1 abandon d/2", "dtw L1 abandon 3d/4", "dtw L1 abandon d",
  "dtw Sq abandon d/2", "adtw L1 p0.5", "adtw abandon d/2", "wdtw g0.1", "zero_cost L1", "zero_cost Sq",
  "zero_cost abandon d/2", "arow L1", "arow Sq", "soft_dtw g1 (unbanded)"
};

template <class T> std::uint64_t check_type(const char *tname, int seeds)
{
  constexpr T maxValue = std::numeric_limits<T>::max();
  std::uint64_t hash[kVariants];
  long long outputs[kVariants] = {}, maxv[kVariants] = {}, nonfinite[kVariants] = {}, abandoned[kVariants] = {};
  for (auto &h : hash) h = kFnv0;
  for (int seed = 0; seed < seeds; ++seed)
    for (int kind = 0; kind < 5; ++kind) {
      std::vector<std::array<std::uint64_t, kVariants>> hrow(1001);
      for (auto &r : hrow) r.fill(kFnv0);
      std::vector<std::array<long long, kVariants>> o(1001), mx(1001), nf(1001), ab(1001);
#pragma omp parallel for schedule(dynamic, 1)
      for (int nx = 1; nx <= 1000; ++nx) {
        o[nx].fill(0); mx[nx].fill(0); nf[nx].fill(0); ab[nx].fill(0);
        std::uint64_t s = 0xabcdefULL * (kind + 1) + 104729ULL * nx + 0x9e3779b9ULL * seed;
        const int nys[3] = { nx, 1 + int(splitmix(s) % 1000), std::max(1, nx + int(splitmix(s) % 21) - 10) };
        for (int ny : nys) {
          std::vector<T> x, y;
          gen(x, nx, s + 11, kind);
          gen(y, ny, s + 29, kind);
          if (kind == 2 && ny == nx && nx % 3 == 0) y = x; // equal content, distinct storage
          auto put = [&](int v, T r, T plain) {
            hrow[nx][v] = fnv(hrow[nx][v], &r, sizeof r);
            ++o[nx][v];
            mx[nx][v] += r == maxValue;
            nf[nx][v] += !std::isfinite(r);
            ab[nx][v] += r == maxValue && plain != maxValue;
          };
          const T *xp = x.data(), *yp = y.data();
          const std::size_t ux = std::size_t(nx), uy = std::size_t(ny);
          const int L = std::max(nx, ny);
          for (int band : { -1, 0, 1, L / 10, L, std::abs(nx - ny) }) {
            const T d1 = dtwc::dtwBanded<T>(xp, ux, yp, uy, band, T(-1), MetricType::L1);
            const T d2 = dtwc::dtwBanded<T>(xp, ux, yp, uy, band, T(-1), MetricType::SquaredL2);
            put(kDtwL1, d1, d1);
            put(kDtwSq, d2, d2);
            put(kEaL1Zero, dtwc::dtwBanded<T>(xp, ux, yp, uy, band, T(0), MetricType::L1), d1);
            put(kEaL1Half, dtwc::dtwBanded<T>(xp, ux, yp, uy, band, d1 / 2, MetricType::L1), d1);
            put(kEaL1ThreeQ, dtwc::dtwBanded<T>(xp, ux, yp, uy, band, d1 * T(0.75), MetricType::L1), d1);
            put(kEaL1Exact, dtwc::dtwBanded<T>(xp, ux, yp, uy, band, d1, MetricType::L1), d1);
            put(kEaSqHalf, dtwc::dtwBanded<T>(xp, ux, yp, uy, band, d2 / 2, MetricType::SquaredL2), d2);
            const T a = dtwc::adtwBanded<T>(xp, ux, yp, uy, band, T(0.5), T(-1));
            put(kAdtw, a, a);
            put(kAdtwEaHalf, dtwc::adtwBanded<T>(xp, ux, yp, uy, band, T(0.5), a / 2), a);
            const T w = dtwc::wdtwBanded<T>(xp, ux, yp, uy, band, T(0.1));
            put(kWdtw, w, w);
            const T z1 = dtwc::dtwMissing_banded<T>(xp, ux, yp, uy, band, T(-1), MetricType::L1);
            const T z2 = dtwc::dtwMissing_banded<T>(xp, ux, yp, uy, band, T(-1), MetricType::SquaredL2);
            put(kZeroL1, z1, z1);
            put(kZeroSq, z2, z2);
            put(kZeroEaHalf, dtwc::dtwMissing_banded<T>(xp, ux, yp, uy, band, z1 / 2, MetricType::L1), z1);
            const T r1 = dtwc::dtwAROW_banded<T>(xp, ux, yp, uy, band, MetricType::L1);
            const T r2 = dtwc::dtwAROW_banded<T>(xp, ux, yp, uy, band, MetricType::SquaredL2);
            put(kArowL1, r1, r1);
            put(kArowSq, r2, r2);
          }
          if (seed == 0) { // Soft-DTW runs dtw_kernel_linear only, unbanded; exp and log per cell
            const T sd = dtwc::soft_dtw<T>(std::span<const T>(x), std::span<const T>(y), T(1));
            put(kSoft, sd, sd);
          }
        }
      }
      for (int n = 1; n <= 1000; ++n)
        for (int v = 0; v < kVariants; ++v) {
          hash[v] = fnv(hash[v], &hrow[n][v], 8);
          outputs[v] += o[n][v]; maxv[v] += mx[n][v]; nonfinite[v] += nf[n][v]; abandoned[v] += ab[n][v];
        }
    }
  std::uint64_t all = kFnv0;
  for (int v = 0; v < kVariants; ++v) {
    std::printf("check %s %-24s outputs %8lld  hash %016llx  max() %7lld  non-finite %7lld  abandoned %7lld\n",
                tname, kName[v], outputs[v], (unsigned long long)hash[v], maxv[v], nonfinite[v], abandoned[v]);
    all = fnv(all, &hash[v], 8);
  }
  std::fflush(stdout);
  return all;
}

// The multivariate entry points (ndim 2 and 3: costs that sum over a runtime ndim, which -fassociative-math lets
// the vectoriser regroup), seed 0, the same lengths, partners, bands and data kinds; lengths count time steps.
enum MvVariant { kMvL1, kMvSq, kMvL2, kMvEaHalf, kMvAdtw, kMvWdtw, kMvZeroL1, kMvZeroL2, kMvArowL1, kMvArowSq, kMvVariants };
static const char *kMvName[kMvVariants] = {
  "dtw_mv L1", "dtw_mv Sq", "dtw_mv L2", "dtw_mv L1 abandon d/2", "adtw_mv p0.5", "wdtw_mv g0.1",
  "zero_cost_mv L1", "zero_cost_mv L2", "arow_mv L1", "arow_mv Sq"
};

template <class T> std::uint64_t check_mv_type(const char *tname)
{
  std::uint64_t all = kFnv0;
  for (std::size_t ndim : { std::size_t{2}, std::size_t{3} }) {
    std::uint64_t hash[kMvVariants];
    long long outputs[kMvVariants] = {}, maxv[kMvVariants] = {}, nonfinite[kMvVariants] = {}, abandoned[kMvVariants] = {};
    for (auto &h : hash) h = kFnv0;
    for (int kind = 0; kind < 5; ++kind) {
      std::vector<std::array<std::uint64_t, kMvVariants>> hrow(1001);
      for (auto &r : hrow) r.fill(kFnv0);
      std::vector<std::array<long long, kMvVariants>> o(1001), mx(1001), nf(1001), ab(1001);
#pragma omp parallel for schedule(dynamic, 1)
      for (int nx = 1; nx <= 1000; ++nx) {
        o[nx].fill(0); mx[nx].fill(0); nf[nx].fill(0); ab[nx].fill(0);
        std::uint64_t s = 0x51ab1eULL * (kind + 1) + 104729ULL * nx + 0x9e3779b9ULL * ndim;
        const int nys[3] = { nx, 1 + int(splitmix(s) % 1000), std::max(1, nx + int(splitmix(s) % 21) - 10) };
        for (int ny : nys) {
          std::vector<T> x, y;
          gen(x, std::size_t(nx) * ndim, s + 11, kind);
          gen(y, std::size_t(ny) * ndim, s + 29, kind);
          auto put = [&](int v, T r, T plain) {
            hrow[nx][v] = fnv(hrow[nx][v], &r, sizeof r);
            ++o[nx][v];
            mx[nx][v] += r == std::numeric_limits<T>::max();
            nf[nx][v] += !std::isfinite(r);
            ab[nx][v] += r == std::numeric_limits<T>::max() && plain != std::numeric_limits<T>::max();
          };
          const T *xp = x.data(), *yp = y.data();
          const std::size_t ux = std::size_t(nx), uy = std::size_t(ny);
          const int L = std::max(nx, ny);
          for (int band : { -1, 0, 1, L / 10, L, std::abs(nx - ny) }) {
            const T d1 = dtwc::dtwBanded_mv<T>(xp, ux, yp, uy, ndim, band, T(-1), MetricType::L1);
            put(kMvL1, d1, d1);
            const T d2 = dtwc::dtwBanded_mv<T>(xp, ux, yp, uy, ndim, band, T(-1), MetricType::SquaredL2);
            put(kMvSq, d2, d2);
            const T d3 = dtwc::dtwBanded_mv<T>(xp, ux, yp, uy, ndim, band, T(-1), MetricType::L2);
            put(kMvL2, d3, d3);
            put(kMvEaHalf, dtwc::dtwBanded_mv<T>(xp, ux, yp, uy, ndim, band, d1 / 2, MetricType::L1), d1);
            const T a = dtwc::adtwBanded_mv<T>(xp, ux, yp, uy, ndim, band, T(0.5));
            put(kMvAdtw, a, a);
            const T w = dtwc::wdtwBanded_mv<T>(xp, ux, yp, uy, ndim, band, T(0.1));
            put(kMvWdtw, w, w);
            const T z1 = dtwc::dtwMissing_banded_mv<T>(xp, ux, yp, uy, ndim, band, T(-1), MetricType::L1);
            put(kMvZeroL1, z1, z1);
            const T z3 = dtwc::dtwMissing_banded_mv<T>(xp, ux, yp, uy, ndim, band, T(-1), MetricType::L2);
            put(kMvZeroL2, z3, z3);
            // dtw_dispatch.cpp's multivariate AROW (resolve_dtw_fn, ndim > 1)
            const T r1 = dc::run_dtw<dc::SpanMVAROWL1Cost>(xp, ux, yp, uy, band, dc::AROWCell{}, T(-1), ndim);
            put(kMvArowL1, r1, r1);
            const T r2 = dc::run_dtw<dc::SpanMVAROWSquaredL2Cost>(xp, ux, yp, uy, band, dc::AROWCell{}, T(-1), ndim);
            put(kMvArowSq, r2, r2);
          }
        }
      }
      for (int n = 1; n <= 1000; ++n)
        for (int v = 0; v < kMvVariants; ++v) {
          hash[v] = fnv(hash[v], &hrow[n][v], 8);
          outputs[v] += o[n][v]; maxv[v] += mx[n][v]; nonfinite[v] += nf[n][v]; abandoned[v] += ab[n][v];
        }
    }
    for (int v = 0; v < kMvVariants; ++v) {
      std::printf("check %s ndim %zu %-21s outputs %8lld  hash %016llx  max() %7lld  non-finite %7lld  abandoned %7lld\n",
                  tname, ndim, kMvName[v], outputs[v], (unsigned long long)hash[v], maxv[v], nonfinite[v], abandoned[v]);
      all = fnv(all, &hash[v], 8);
    }
  }
  std::fflush(stdout);
  return all;
}

static int check_mode(int seeds)
{
  std::uint64_t all = kFnv0;
  std::uint64_t h = check_type<double>("f64", seeds);
  all = fnv(all, &h, 8);
  h = check_type<float>("f32", seeds);
  all = fnv(all, &h, 8);
  std::printf("check all hashes combined %016llx\n", (unsigned long long)all);
  std::uint64_t mv = kFnv0;
  h = check_mv_type<double>("f64");
  mv = fnv(mv, &h, 8);
  h = check_mv_type<float>("f32");
  mv = fnv(mv, &h, 8);
  std::printf("check multivariate hashes combined %016llx\n", (unsigned long long)mv);
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

// time: one pair at a time through the fill's per-pair function, random walks, single thread
template <class T> int time_mode(bool sq, std::size_t nx, std::size_t ny, int band, int reps, double target_cells)
{
  const std::size_t P = 64;
  std::vector<std::vector<T>> px(P), py(P);
  for (std::size_t p = 0; p < P; ++p) {
    gen(px[p], nx, 5000 + p, 1);
    gen(py[p], ny, 9000 + p, 1);
  }
  const PairFn<T> pair = pair_for<T>(sq, band);
  const double cells1 = pair_cells(nx, ny, band);
  if (cells1 == 0) { std::fprintf(stderr, "no path fits nx %zu ny %zu band %d\n", nx, ny, band); return 2; }
  const std::size_t calls = std::max<std::size_t>(4, std::size_t(target_cells / cells1));
  double checksum = 0;
  auto run = [&] {
    for (std::size_t c = 0; c < calls; ++c) checksum += pair(px[c % P], py[(c + 1 + c / P) % P]);
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
    const double cells = double(calls) * cells1;
    nspc.push_back(ns / cells);
    cyc_meas.push_back(ns / ((c0 + c1) / 2) / cells);
    ghz.push_back(2 / (c0 + c1));
  }
  const double m = median(nspc);
  std::printf("time %s %-2s nx %-4zu ny %-4zu band %-4d ns/cell %.5f (min %.5f)  cycles/cell@4.59GHz %.4f  "
              "cycles/cell@measured %.4f  clock %.3f GHz  Gcell/s %.3f  checksum %.17g\n",
              sizeof(T) == 8 ? "f64" : "f32", sq ? "Sq" : "L1", nx, ny, band, m,
              *std::min_element(nspc.begin(), nspc.end()), m * 4.59, median(cyc_meas), median(ghz), 1 / m, checksum);
  return 0;
}

// ---------------------------------------------------------------------------------------------
// fill: Problem::fillDistanceMatrix_BruteForce (Problem.cpp:774-809): each row takes its columns W at a time
// through the lane function where they are as long as the row's series, then pair by pair through the per-pair
// function; run_openmp(fill_row, N, true, 8): schedule(dynamic, max(1, N / (threads * 8))). Cells counted are
// the ones the kernels compute (a pair no band path fits costs none).
template <class T> int fill_mode(bool sq, std::size_t N, std::size_t L, int band, int threads, int reps, bool ragged)
{
  constexpr std::size_t W = dc::dtw_lanes<T>;
  std::vector<std::vector<T>> S(N);
  std::uint64_t s = 99;
  for (std::size_t i = 0; i < N; ++i) gen(S[i], ragged ? L - 10 + splitmix(s) % 21 : L, 777 + i, 1);
  const BlockFn<T> block = block_for<T>(sq, band);
  const PairFn<T> pair = pair_for<T>(sq, band);
  auto tri = [](std::size_t i, std::size_t j) { if (i < j) std::swap(i, j); return i * (i + 1) / 2 + j; };
  std::vector<double> M;
  const long chunk = std::max<long>(1, long(N) / (threads * 8));
  double cells = 0;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j) cells += pair_cells(S[i].size(), S[j].size(), band);
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
      const std::size_t nx = std::strtoul(arg(4, "100"), nullptr, 10), ny = std::strtoul(arg(5, "100"), nullptr, 10);
      const int band = std::atoi(arg(6, "-1")), reps = std::atoi(arg(7, "5"));
      const double cells = std::atof(arg(8, "40e6"));
      return f32 ? time_mode<float>(sq, nx, ny, band, reps, cells) : time_mode<double>(sq, nx, ny, band, reps, cells);
    }
    const std::size_t N = std::strtoul(arg(4, "1000"), nullptr, 10), L = std::strtoul(arg(5, "100"), nullptr, 10);
    const int band = std::atoi(arg(6, "-1")), threads = std::atoi(arg(7, "18")), reps = std::atoi(arg(8, "3"));
    const bool ragged = std::atoi(arg(9, "0")) != 0;
    return f32 ? fill_mode<float>(sq, N, L, band, threads, reps, ragged)
               : fill_mode<double>(sq, N, L, band, threads, reps, ragged);
  }
  std::fprintf(stderr, "usage: pprobe check [seeds] | time T D nx ny band [reps] [cells] | fill T D N L band threads [reps] [ragged]\n");
  return 2;
}
