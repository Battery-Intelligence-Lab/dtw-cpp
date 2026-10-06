// kbench: the shipped DTW kernels (dtwc/core/dtw_kernel.hpp, included unmodified) against scratch
// variants of the same header (v_*.hpp, made by sed / make_skew.py), on this machine.
//
//   kbench time  [reps]          single-thread cycles per cell (clock: dependent add chain, interleaved)
//   kbench check                 bitwise sweep: every variant's outputs against the shipped kernel's
//   kbench fill  threads [reps]  the brute-force fill of Problem.cpp (lanes then per-pair), OpenMP
//
// Built with the compile command of dtwc/core/dtw_lanes.cpp from build/compile_commands.json.
#include "core/dtw_kernel.hpp" // the shipped kernel, unmodified
#include "core/dtw_cost.hpp"   // SpanL1Cost, SpanSquaredL2Cost
#include "v_fmin.hpp"
#include "v_w16.hpp"
#include "v_w32.hpp"
#include "v_fmin_w16.hpp"
#include "v_skew2.hpp"
#include "v_fmin_skew2.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <string>
#include <vector>

namespace dc = dtwc::core;

// ---------------------------------------------------------------------------------------------
// distances: the bodies of the lambdas in dtwc/core/dtw_lanes.cpp:51-56
template <class T> struct L1 { T operator()(T a, T b) const { return std::abs(a - b); } };
template <class T> struct Sq { T operator()(T a, T b) const { const T d = a - b; return d * d; } };

// lanes kernels: out[w] for w < W
#define LANES(NS, T, D)                                                                              \
  __attribute__((noinline)) void lanes_##NS##_##T##_##D(const T *x, const T *const *ys, std::size_t n, \
                                                        int band, T *out)                            \
  {                                                                                                   \
    const auto d = NS::dtw_kernel_lanes<T>(x, ys, n, band, D<T>{}, NS::StandardCell{});               \
    for (std::size_t w = 0; w < NS::dtw_lanes<T>; ++w) out[w] = d[w];                                \
  }
#define LANES_ALL(NS) LANES(NS, double, L1) LANES(NS, double, Sq) LANES(NS, float, L1) LANES(NS, float, Sq)
namespace shipped = dtwc::core;
LANES_ALL(shipped)
LANES_ALL(v_fmin)
LANES_ALL(v_w16)
LANES_ALL(v_w32)
LANES_ALL(v_fmin_w16)

// per-pair kernels: run_dtw, as warping.hpp's dtwBanded calls it (warping.hpp:125-127)
#define PAIR(NS, T, C, TAG)                                                                          \
  __attribute__((noinline)) T pair_##NS##_##T##_##TAG(const T *x, std::size_t nx, const T *y,        \
                                                      std::size_t ny, int band)                      \
  {                                                                                                   \
    return NS::run_dtw<dc::C>(x, nx, y, ny, band, NS::StandardCell{}, T(-1));                         \
  }
#define PAIR_ALL(NS) PAIR(NS, double, SpanL1Cost, L1) PAIR(NS, double, SpanSquaredL2Cost, Sq) \
                     PAIR(NS, float, SpanL1Cost, L1) PAIR(NS, float, SpanSquaredL2Cost, Sq)
PAIR_ALL(shipped)
PAIR_ALL(v_fmin)
PAIR_ALL(v_skew2)
PAIR_ALL(v_fmin_skew2)

template <class T> using LanesFn = void (*)(const T *, const T *const *, std::size_t, int, T *);
template <class T> using PairFn = T (*)(const T *, std::size_t, const T *, std::size_t, int);

template <class T> struct LanesK { const char *name; LanesFn<T> f; std::size_t W; };
template <class T> struct PairK { const char *name; PairFn<T> f; };

#define LK(NS, T, D) LanesK<T>{ #NS, lanes_##NS##_##T##_##D, NS::dtw_lanes<T> }
#define PK(NS, T, D) PairK<T>{ #NS, pair_##NS##_##T##_##D }

template <class T, int D> std::vector<LanesK<T>> lanes_kernels();
template <> std::vector<LanesK<double>> lanes_kernels<double, 0>() { return { LK(shipped, double, L1), LK(v_fmin, double, L1), LK(v_w16, double, L1), LK(v_w32, double, L1), LK(v_fmin_w16, double, L1) }; }
template <> std::vector<LanesK<double>> lanes_kernels<double, 1>() { return { LK(shipped, double, Sq), LK(v_fmin, double, Sq), LK(v_w16, double, Sq), LK(v_w32, double, Sq), LK(v_fmin_w16, double, Sq) }; }
template <> std::vector<LanesK<float>> lanes_kernels<float, 0>() { return { LK(shipped, float, L1), LK(v_fmin, float, L1), LK(v_w16, float, L1), LK(v_w32, float, L1), LK(v_fmin_w16, float, L1) }; }
template <> std::vector<LanesK<float>> lanes_kernels<float, 1>() { return { LK(shipped, float, Sq), LK(v_fmin, float, Sq), LK(v_w16, float, Sq), LK(v_w32, float, Sq), LK(v_fmin_w16, float, Sq) }; }
template <class T, int D> std::vector<PairK<T>> pair_kernels();
template <> std::vector<PairK<double>> pair_kernels<double, 0>() { return { PK(shipped, double, L1), PK(v_fmin, double, L1), PK(v_skew2, double, L1), PK(v_fmin_skew2, double, L1) }; }
template <> std::vector<PairK<double>> pair_kernels<double, 1>() { return { PK(shipped, double, Sq), PK(v_fmin, double, Sq), PK(v_skew2, double, Sq), PK(v_fmin_skew2, double, Sq) }; }
template <> std::vector<PairK<float>> pair_kernels<float, 0>() { return { PK(shipped, float, L1), PK(v_fmin, float, L1), PK(v_skew2, float, L1), PK(v_fmin_skew2, float, L1) }; }
template <> std::vector<PairK<float>> pair_kernels<float, 1>() { return { PK(shipped, float, Sq), PK(v_fmin, float, Sq), PK(v_skew2, float, Sq), PK(v_fmin_skew2, float, Sq) }; }

// ---------------------------------------------------------------------------------------------
// clock: a dependent integer-add chain is one cycle per add on every Apple core
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
  return (now_ns() - t0) / (it * 100.0);
}

// ---------------------------------------------------------------------------------------------
static std::uint64_t splitmix(std::uint64_t &s)
{
  std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double unit(std::uint64_t &s) { return (splitmix(s) >> 11) * 0x1.0p-53; }

// generators: 0 uniform [-1,1]; 1 random walk; 2 integers {0,1,2} (ties, zero costs); 3 huge (+-max/2..max: overflow to inf)
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

// ---------------------------------------------------------------------------------------------
// time mode
struct Stat { double med, min; };
static Stat stat(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  return { v[v.size() / 2], v.front() };
}

template <class T, int D> void time_lanes(const char *tname, int reps, std::size_t n, int band)
{
  auto ks = lanes_kernels<T, D>();
  const std::size_t P = 128;
  std::vector<std::vector<T>> pool(P);
  for (std::size_t p = 0; p < P; ++p) gen(pool[p], n, 1000 + p, 1);
  const std::size_t cells1 = band_cells(n, band);
  std::vector<std::vector<double>> cpc(ks.size()), gcs(ks.size());
  std::vector<T> out(64);
  for (int r = 0; r < reps; ++r) {
    for (std::size_t k = 0; k < ks.size(); ++k) {
      const std::size_t W = ks[k].W;
      // ~120 M cells per measurement
      const std::size_t calls = std::max<std::size_t>(4, 120000000 / (W * cells1));
      std::vector<const T *> ys(W);
      const double c0 = ns_per_cycle();
      const double t0 = now_ns();
      for (std::size_t c = 0; c < calls; ++c) {
        for (std::size_t w = 0; w < W; ++w) ys[w] = pool[(c * W + w + 1) % P].data();
        ks[k].f(pool[c % P].data(), ys.data(), n, band, out.data());
      }
      const double ns = now_ns() - t0;
      const double c1 = ns_per_cycle();
      const double cells = double(calls) * W * cells1;
      cpc[k].push_back(ns / ((c0 + c1) / 2) / cells);
      gcs[k].push_back(cells / ns);
    }
  }
  for (std::size_t k = 0; k < ks.size(); ++k) {
    const Stat s = stat(cpc[k]), g = stat(gcs[k]);
    std::printf("lanes %-6s %-2s W=%-3zu L=%-5zu band=%-4d %-12s cyc/cell med %.4f min %.4f  Gcell/s med %.3f  ratio_vs_shipped %.3f\n",
                tname, D ? "Sq" : "L1", ks[k].W, n, band, ks[k].name, s.med, s.min, g.med,
                stat(cpc[0]).med / s.med);
  }
  std::fflush(stdout);
}

template <class T, int D> void time_pairs(const char *tname, int reps, std::size_t n, int band)
{
  auto ks = pair_kernels<T, D>();
  const std::size_t P = 64;
  std::vector<std::vector<T>> pool(P);
  for (std::size_t p = 0; p < P; ++p) gen(pool[p], n, 5000 + p, 1);
  const std::size_t cells1 = band_cells(n, band);
  std::vector<std::vector<double>> cpc(ks.size()), gcs(ks.size());
  volatile double sink = 0;
  for (int r = 0; r < reps; ++r) {
    for (std::size_t k = 0; k < ks.size(); ++k) {
      const std::size_t calls = std::max<std::size_t>(4, 40000000 / cells1);
      const double c0 = ns_per_cycle();
      const double t0 = now_ns();
      double acc = 0;
      for (std::size_t c = 0; c < calls; ++c)
        acc += ks[k].f(pool[c % P].data(), n, pool[(c + 1 + c / P) % P].data(), n, band);
      const double ns = now_ns() - t0;
      const double c1 = ns_per_cycle();
      sink = sink + acc;
      const double cells = double(calls) * cells1;
      cpc[k].push_back(ns / ((c0 + c1) / 2) / cells);
      gcs[k].push_back(cells / ns);
    }
  }
  for (std::size_t k = 0; k < ks.size(); ++k) {
    const Stat s = stat(cpc[k]), g = stat(gcs[k]);
    std::printf("pair  %-6s %-2s       L=%-5zu band=%-4d %-12s cyc/cell med %.4f min %.4f  Gcell/s med %.3f  ratio_vs_shipped %.3f\n",
                tname, D ? "Sq" : "L1", n, band, ks[k].name, s.med, s.min, g.med, stat(cpc[0]).med / s.med);
  }
  std::fflush(stdout);
}

static void time_mode(int reps, const char *only)
{
  // warm-up: 300 ms
  for (int i = 0; i < 40; ++i) ns_per_cycle();
  std::printf("clock %.3f GHz\n", 1.0 / ns_per_cycle());
  const std::size_t Ls[] = { 100, 1000 };
  for (std::size_t L : Ls) {
    for (int band : { -1, int(L / 10) }) {
      if (!only || std::strstr(only, "lanes")) {
        time_lanes<double, 0>("f64", reps, L, band);
        time_lanes<double, 1>("f64", reps, L, band);
        time_lanes<float, 0>("f32", reps, L, band);
        time_lanes<float, 1>("f32", reps, L, band);
      }
      if (!only || std::strstr(only, "pair")) {
        time_pairs<double, 0>("f64", reps, L, band);
        time_pairs<double, 1>("f64", reps, L, band);
        time_pairs<float, 0>("f32", reps, L, band);
      }
    }
  }
  std::printf("clock %.3f GHz\n", 1.0 / ns_per_cycle());
}

// ---------------------------------------------------------------------------------------------
// check mode: every variant's outputs, bit for bit, against the shipped kernel's
template <class T, int D> void check_lanes(const char *tname)
{
  auto ks = lanes_kernels<T, D>();
  const std::size_t NY = 64; // a multiple of every W
  std::vector<std::uint64_t> hash(ks.size(), 0xcbf29ce484222325ULL);
  std::vector<long long> mism(ks.size(), 0), nonfinite(ks.size(), 0);
  long long outputs = 0, vs_pair_mism = 0;
  for (int kind = 0; kind < 4; ++kind) {
    std::vector<std::vector<std::uint64_t>> hrow(1001, std::vector<std::uint64_t>(ks.size(), 0xcbf29ce484222325ULL));
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs, vs_pair_mism)
    for (int n = 1; n <= 1000; ++n) {
      std::vector<T> x;
      std::vector<std::vector<T>> y(NY);
      const std::uint64_t seed = 0x5eed0000ULL * (kind + 1) + 7919ULL * n;
      gen(x, n, seed, kind);
      for (std::size_t w = 0; w < NY; ++w) gen(y[w], n, seed + 1 + w, kind);
      // half the lanes get series equal to x (zero-cost diagonal) under kind 2
      if (kind == 2) for (std::size_t w = 0; w < NY; w += 4) y[w] = x;
      std::vector<const T *> yp(NY);
      for (std::size_t w = 0; w < NY; ++w) yp[w] = y[w].data();
      for (int band : { -1, 0, 1, n / 10, n }) {
        std::vector<std::vector<T>> res(ks.size(), std::vector<T>(NY));
        for (std::size_t k = 0; k < ks.size(); ++k) {
          const std::size_t W = ks[k].W;
          for (std::size_t b = 0; b < NY; b += W) ks[k].f(x.data(), yp.data() + b, n, band, res[k].data() + b);
        }
        outputs += NY;
        for (std::size_t w = 0; w < NY; w += 16) { // lanes vs the shipped per-pair kernel, a sample
          const T p = D ? dc::run_dtw<dc::SpanSquaredL2Cost>(x.data(), n, y[w].data(), n, band, dc::StandardCell{}, T(-1))
                        : dc::run_dtw<dc::SpanL1Cost>(x.data(), n, y[w].data(), n, band, dc::StandardCell{}, T(-1));
          if (std::memcmp(&p, &res[0][w], sizeof(T)) != 0 && !(x.data() == y[w].data())) ++vs_pair_mism;
        }
        for (std::size_t k = 0; k < ks.size(); ++k) {
          hrow[n][k] = fnv(hrow[n][k], res[k].data(), NY * sizeof(T));
          if (std::memcmp(res[k].data(), res[0].data(), NY * sizeof(T)) != 0) {
#pragma omp atomic
            ++mism[k];
          }
          for (std::size_t w = 0; w < NY; ++w)
            if (!std::isfinite(res[k][w])) {
#pragma omp atomic
              ++nonfinite[k];
            }
        }
      }
    }
    for (int n = 1; n <= 1000; ++n)
      for (std::size_t k = 0; k < ks.size(); ++k) hash[k] = fnv(hash[k], &hrow[n][k], 8);
  }
  for (std::size_t k = 0; k < ks.size(); ++k)
    std::printf("check lanes %-3s %-2s %-12s outputs %lld  (n,band,kind) blocks differing from shipped: %lld  non-finite outputs %lld  hash %016llx%s\n",
                tname, D ? "Sq" : "L1", ks[k].name, outputs, mism[k], nonfinite[k], (unsigned long long)hash[k],
                k && hash[k] == hash[0] ? "  == shipped" : (k ? "  DIFFERS" : ""));
  std::printf("check lanes %-3s %-2s shipped lanes vs shipped per-pair (sampled): %lld mismatches\n", tname, D ? "Sq" : "L1", vs_pair_mism);
  std::fflush(stdout);
}

template <class T, int D> void check_pairs(const char *tname)
{
  auto ks = pair_kernels<T, D>();
  std::vector<std::uint64_t> hash(ks.size(), 0xcbf29ce484222325ULL);
  std::vector<long long> mism(ks.size(), 0);
  long long outputs = 0;
  for (int kind = 0; kind < 4; ++kind) {
    std::vector<std::vector<std::uint64_t>> hrow(1001, std::vector<std::uint64_t>(ks.size(), 0xcbf29ce484222325ULL));
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs)
    for (int nx = 1; nx <= 1000; ++nx) {
      std::uint64_t s = 0xabcdefULL * (kind + 1) + 104729ULL * nx;
      // three partners: equal length, a random length, and a near length
      const int nys[3] = { nx, 1 + int(splitmix(s) % 1000), std::max(1, nx + int(splitmix(s) % 21) - 10) };
      for (int ny : nys) {
        std::vector<T> x, y;
        gen(x, nx, s + 11, kind);
        gen(y, ny, s + 29, kind);
        if (kind == 2 && ny == nx && (nx % 3 == 0)) y = x;
        const int L = std::max(nx, ny);
        for (int band : { -1, 0, 1, L / 10, L, std::abs(nx - ny) }) {
          T r[8];
          for (std::size_t k = 0; k < ks.size(); ++k) r[k] = ks[k].f(x.data(), nx, y.data(), ny, band);
          ++outputs;
          for (std::size_t k = 0; k < ks.size(); ++k) {
            hrow[nx][k] = fnv(hrow[nx][k], &r[k], sizeof(T));
            if (std::memcmp(&r[k], &r[0], sizeof(T)) != 0) {
#pragma omp atomic
              ++mism[k];
            }
          }
        }
      }
    }
    for (int n = 1; n <= 1000; ++n)
      for (std::size_t k = 0; k < ks.size(); ++k) hash[k] = fnv(hash[k], &hrow[n][k], 8);
  }
  for (std::size_t k = 0; k < ks.size(); ++k)
    std::printf("check pair  %-3s %-2s %-12s outputs %lld  differing from shipped: %lld  hash %016llx%s\n", tname,
                D ? "Sq" : "L1", ks[k].name, outputs, mism[k], (unsigned long long)hash[k],
                k && hash[k] == hash[0] ? "  == shipped" : (k ? "  DIFFERS" : ""));
  std::fflush(stdout);
}

static void check_mode()
{
  check_lanes<double, 0>("f64");
  check_lanes<double, 1>("f64");
  check_lanes<float, 0>("f32");
  check_lanes<float, 1>("f32");
  check_pairs<double, 0>("f64");
  check_pairs<double, 1>("f64");
  check_pairs<float, 0>("f32");
  check_pairs<float, 1>("f32");
}

// ---------------------------------------------------------------------------------------------
// fill mode: Problem::fillDistanceMatrix_BruteForce (Problem.cpp:748-813), lanes then per-pair,
// run_openmp's schedule(dynamic, max(1, N / (threads * 8))) (parallelisation.hpp:59-63, 104-116)
template <class T>
double fill(const std::vector<std::vector<T>> &S, int band, LanesFn<T> lanes, std::size_t W, PairFn<T> pair,
            std::vector<double> &M, int threads)
{
  const std::size_t N = S.size();
  M.assign(N * (N + 1) / 2, std::nan(""));
  auto tri = [](std::size_t i, std::size_t j) { if (i < j) std::swap(i, j); return i * (i + 1) / 2 + j; };
  const long chunk = std::max<long>(1, long(N) / (threads * 8));
  const double t0 = now_ns();
#pragma omp parallel num_threads(threads)
  {
    std::vector<const T *> ys(W);
    std::vector<T> d(W);
#pragma omp for schedule(dynamic, chunk) nowait
    for (long il = 0; il < long(N); ++il) {
      const std::size_t i = std::size_t(il);
      M[tri(i, i)] = 0.0;
      const std::vector<T> &x = S[i];
      for (std::size_t j = i + 1; j < N; j += W) {
        const std::size_t count = std::min(W, N - j);
        bool equal = true;
        for (std::size_t w = 0; w < W; ++w) {
          const std::size_t c = j + std::min(w, count - 1);
          ys[w] = S[c].data();
          equal = equal && S[c].size() == x.size();
        }
        if (!equal) continue;
        lanes(x.data(), ys.data(), x.size(), band, d.data());
        for (std::size_t w = 0; w < count; ++w) M[tri(i, j + w)] = double(d[w]);
      }
      for (std::size_t j = i + 1; j < N; ++j)
        if (std::isnan(M[tri(i, j)])) M[tri(i, j)] = double(pair(x.data(), x.size(), S[j].data(), S[j].size(), band));
    }
  }
  return now_ns() - t0;
}

template <class T> void fill_shapes(int threads, int reps, const char *only)
{
  struct Shape { std::size_t N, L; int band; bool ragged; };
  const Shape shapes[] = { { 2000, 100, -1, false }, { 2000, 100, 10, false }, { 500, 1000, -1, false },
                           { 500, 1000, 100, false }, { 1000, 100, -1, true }, { 1000, 100, 10, true } };
  int shape_index = -1;
  for (const Shape &sh : shapes) {
    ++shape_index;
    if (only && std::strstr(only, sh.ragged ? "equal" : "ragged")) continue;
    if (const char *e = std::getenv("KB_SHAPE"); e && std::atoi(e) != shape_index) continue;
    std::vector<std::vector<T>> S(sh.N);
    std::uint64_t s = 99;
    for (std::size_t i = 0; i < sh.N; ++i)
      gen(S[i], sh.ragged ? sh.L - 10 + splitmix(s) % 21 : sh.L, 777 + i, 1);
    struct V { const char *name; LanesFn<T> lanes; std::size_t W; PairFn<T> pair; };
    std::vector<V> vs;
    if constexpr (std::is_same_v<T, double>) {
      vs = { { "shipped", lanes_shipped_double_L1, shipped::dtw_lanes<double>, pair_shipped_double_L1 },
             { "v_fmin", lanes_v_fmin_double_L1, v_fmin::dtw_lanes<double>, pair_v_fmin_double_L1 },
             { "v_w16", lanes_v_w16_double_L1, v_w16::dtw_lanes<double>, pair_shipped_double_L1 },
             { "v_fmin_w16", lanes_v_fmin_w16_double_L1, v_fmin_w16::dtw_lanes<double>, pair_v_fmin_double_L1 },
             { "v_skew2", lanes_shipped_double_L1, shipped::dtw_lanes<double>, pair_v_skew2_double_L1 },
             { "v_fmin_skew2", lanes_v_fmin_double_L1, v_fmin::dtw_lanes<double>, pair_v_fmin_skew2_double_L1 } };
    } else {
      vs = { { "shipped", lanes_shipped_float_L1, shipped::dtw_lanes<float>, pair_shipped_float_L1 },
             { "v_fmin", lanes_v_fmin_float_L1, v_fmin::dtw_lanes<float>, pair_v_fmin_float_L1 },
             { "v_fmin_w16", lanes_v_fmin_w16_float_L1, v_fmin_w16::dtw_lanes<float>, pair_v_fmin_float_L1 } };
    }
    if (std::getenv("KB_SHIPPED_ONLY")) vs.resize(1);
    std::vector<std::vector<double>> t(vs.size());
    std::vector<double> ref, M;
    std::vector<int> same(vs.size(), 1);
    for (int r = 0; r < reps; ++r)
      for (std::size_t k = 0; k < vs.size(); ++k) {
        t[k].push_back(fill<T>(S, sh.band, vs[k].lanes, vs[k].W, vs[k].pair, M, threads));
        if (k == 0 && r == 0) ref = M;
        else if (std::memcmp(M.data(), ref.data(), M.size() * sizeof(double)) != 0) same[k] = 0;
      }
    double cells = 0;
    for (std::size_t i = 0; i < sh.N; ++i)
      for (std::size_t j = i + 1; j < sh.N; ++j) {
        if (S[i].size() == S[j].size()) cells += double(band_cells(S[i].size(), sh.band));
        else cells += double(S[i].size()) * double(S[j].size()); // approximate for a band
      }
    for (std::size_t k = 0; k < vs.size(); ++k) {
      const Stat st = stat(t[k]);
      std::printf("fill %-3s threads %2d N %-5zu L %-5zu%s band %-4d %-12s ms med %9.2f min %9.2f  Gcell/s %.2f  speedup_vs_shipped %.3f  matrix %s\n",
                  sizeof(T) == 8 ? "f64" : "f32", threads, sh.N, sh.L, sh.ragged ? "+-10" : "    ", sh.band, vs[k].name,
                  st.med / 1e6, st.min / 1e6, cells / st.med, stat(t[0]).med / st.med, same[k] ? "identical" : "DIFFERS");
    }
    std::fflush(stdout);
  }
}

int main(int argc, char **argv)
{
  const std::string mode = argc > 1 ? argv[1] : "time";
  if (mode == "time") time_mode(argc > 2 ? std::atoi(argv[2]) : 7, argc > 3 ? argv[3] : nullptr);
  else if (mode == "check") check_mode();
  else if (mode == "fill") {
    const int threads = argc > 2 ? std::atoi(argv[2]) : 18;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const char *only = argc > 4 ? argv[4] : nullptr;
    fill_shapes<double>(threads, reps, only);
    if (!only || !std::strstr(only, "f64only")) fill_shapes<float>(threads, reps, only);
  }
  return 0;
}
