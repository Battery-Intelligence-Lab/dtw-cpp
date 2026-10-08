// xprobe: the fill's lane function as it ships, for unit XG (GCC's x86-64 lanes at 128 bytes), base against
// candidate, in WSL. XL's probe (.claude/baselines/2026-10-08-xl-x86-lanes-128/kit/xprobe.cpp) ported to Linux:
// the pinning is taskset's (checked here), the clock chain is compiled by GCC too.
//
// Built twice (build_gcc.sh): GCC 13.3 x {base, cand}, each against that version's own sources (src/<version>/dtwc)
// with the flags CMake gives them in WSL at DTWC_ARCH_LEVEL=v3: dtw_lanes.cpp and dtw_dispatch.cpp as dtwc_core's
// objects, this file as dtwc_cl.cpp, linked with LTO as dtwc_cl is. So the lanes run through
// core::resolve_dtw_block_fn and the per-pair route through core::resolve_dtw_fn, the two functions Problem binds,
// with that version's dtw_kernel.hpp.
//
//   xprobe info                                          W per type, compiler, ISA macros
//   xprobe check [seeds]                                 bitwise sweep: every lane against the per-pair route
//   xprobe kinds                                         the sweep's f64 per-pair outputs hashed per data kind and band
//   xprobe time <cpu> <T> <D> <L> <band> [reps] [ms]     one thread, run under `taskset -c <cpu>`: the lane function
//   xprobe fill <T> <D> <N> <L> <band> [threads] [reps] [ragged]   Problem's fill loop and schedule
//
// T: f64|f32, D: L1|Sq, band -1 = unbanded, threads 0 = the OpenMP default. The clock is a dependent chain of
// register adds, one per cycle, on whichever host core runs the vCPU at that moment.
#include <sched.h>

#include "Data.hpp"
#include "core/dtw_dispatch.hpp" // resolve_dtw_fn, resolve_dtw_block_fn
#include "core/dtw_kernel.hpp"   // dtw_lanes, dtw_band_bounds
#include "core/dtw_options.hpp"  // DistanceConfig, MetricType

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

// Cells the lanes compute per pair of length n (dtw_band_bounds is the kernels' band).
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

static const char *compiler()
{
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#else
  return "other";
#endif
}

// ---------------------------------------------------------------------------------------------
// check: NY = 2 * 32 + 5 partners per x, so every W (8, 16, 32) meets full blocks and a tail block that repeats
// its last column in the lanes past it, as the fill does; every output against the per-pair route, bitwise, and
// against the route bound the tests use (tests/support/dtw_route_bound.hpp, dtw_routes_agree<T>: |a - b| <=
// 2 * (2n - 1) * eps(T) * max(|a|, |b|), an infinity agreeing only with itself), with the worst |a - b| over
// (2n - 1) * eps(T) * max(|a|, |b|), which the bound holds to 2.
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
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs, vs_pair, nonfinite, outside) reduction(max : worst)
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
            const double p = pair(x, y[w]), a = res[w];
            vs_pair += std::memcmp(&p, &a, sizeof(double)) != 0;
            nonfinite += !std::isfinite(a);
            if (std::isinf(p) || std::isinf(a)) {
              outside += p != a;
              continue;
            }
            const double scale = double(2 * n - 1) * double(std::numeric_limits<T>::epsilon())
                                 * std::max(std::abs(p), std::abs(a));
            const double r = p == a ? 0.0 : std::abs(p - a) / scale;
            worst = std::max(worst, r);
            outside += r > 2.0;
          }
        }
      }
      for (int n = 1; n <= 1000; ++n) hash = fnv(hash, &hrow[n], 8);
    }
  std::printf("check lanes %s %-2s W=%-2zu seeds %d: outputs %lld  hash %016llx  lanes vs per-pair bitwise "
              "mismatches %lld  non-finite outputs %lld  outside the route bound %lld  worst |a-b|/((2n-1) eps max) "
              "%.4g\n",
              tname, sq ? "Sq" : "L1", W, seeds, outputs, (unsigned long long)hash, vs_pair, nonfinite, outside,
              worst);
  std::fflush(stdout);
  return hash;
}

// The per-pair route over unequal lengths: the unit leaves it alone, so its hashes must match too.
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

// kinds: check_pairs' f64 per-pair outputs hashed per data kind and band, to say where two builds' sweeps part.
static int kinds_mode()
{
  for (bool sq : { false, true })
    for (int kind = 0; kind < 4; ++kind) {
      std::uint64_t h[6];
      std::fill(h, h + 6, kFnv0);
      long long nonfinite = 0;
      for (int seed = 0; seed < 3; ++seed)
        for (int nx = 1; nx <= 1000; ++nx) {
          std::uint64_t s = 0xabcdefULL * (kind + 1) + 104729ULL * nx + 0x9e3779b9ULL * seed;
          const int nys[3] = { nx, 1 + int(splitmix(s) % 1000), std::max(1, nx + int(splitmix(s) % 21) - 10) };
          for (int ny : nys) {
            std::vector<double> x, y;
            gen(x, nx, s + 11, kind);
            gen(y, ny, s + 29, kind);
            const int L = std::max(nx, ny);
            const int bands[6] = { -1, 0, 1, L / 10, L, std::abs(nx - ny) };
            for (int b = 0; b < 6; ++b) {
              const double r = pair_for<double>(sq, bands[b])(x, y);
              h[b] = fnv(h[b], &r, sizeof r);
              nonfinite += !std::isfinite(r);
            }
          }
        }
      std::printf("kinds f64 %s kind %d: band -1 %016llx  0 %016llx  1 %016llx  L/10 %016llx  L %016llx  |nx-ny| %016llx"
                  "  non-finite %lld\n", sq ? "Sq" : "L1", kind, (unsigned long long)h[0], (unsigned long long)h[1],
                  (unsigned long long)h[2], (unsigned long long)h[3], (unsigned long long)h[4], (unsigned long long)h[5],
                  nonfinite);
    }
  return 0;
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

// ---------------------------------------------------------------------------------------------
static double now_ns()
{
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#if defined(__x86_64__)
// add reg, reg: this core folds a chain of add reg, imm at rename (about 5.7 per cycle measured, XL's tools/clk), so
// the addend is a register; an imul chain (3 cycles each) gives the same clock.
__attribute__((noinline)) static void add_chain(std::uint64_t n, std::uint64_t b)
{
  std::uint64_t a = 0;
  asm volatile("1:\n\t.rept 100\n\taddq %[b], %[a]\n\t.endr\n\tdecq %[n]\n\tjnz 1b"
               : [n] "+r"(n), [a] "+r"(a)
               : [b] "r"(b)
               : "cc");
}
static volatile std::uint64_t g_one = 1;
static double ns_per_cycle() // 30 M dependent adds, about 6 ms at 5 GHz
{
  const std::uint64_t it = 300000;
  const double t0 = now_ns();
  add_chain(it, g_one);
  return (now_ns() - t0) / (double(it) * 100.0);
}
#else
static double ns_per_cycle() { return 0; }
#endif

static double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

// The driver runs `taskset -c <cpu>`: the process may run on that vCPU and no other. (WSL's vCPUs are threads of
// the host, which may run one on a P- or an E-core: hence the clock with every measurement.)
static bool pin(int cpu)
{
  cpu_set_t s;
  CPU_ZERO(&s);
  return sched_getaffinity(0, sizeof s, &s) == 0 && CPU_COUNT(&s) == 1 && CPU_ISSET(cpu, &s);
}

template <class T> int time_mode(int cpu, bool sq, std::size_t L, int band, int reps, double ms)
{
  if (!pin(cpu)) {
    std::fprintf(stderr, "not pinned to vCPU %d alone: run under taskset -c %d\n", cpu, cpu);
    return 3;
  }
  constexpr std::size_t W = dc::dtw_lanes<T>;
  const std::size_t P = 128;
  std::vector<std::vector<T>> pool(P);
  for (std::size_t p = 0; p < P; ++p) gen(pool[p], L, 1000 + p, 1);
  const BlockFn<T> block = block_for<T>(sq, band);
  const std::size_t cells1 = band_cells(L, band);
  std::array<std::span<const T>, W> ys;
  std::array<double, W> out{};
  double checksum = 0;
  std::size_t calls = std::max<std::size_t>(4, std::size_t(1e6 / double(W * cells1)));
  auto run = [&](std::size_t k) {
    for (std::size_t c = 0; c < k; ++c) {
      for (std::size_t w = 0; w < W; ++w) ys[w] = pool[(c * W + w + 1) % P];
      block(pool[c % P], ys, out);
      checksum += out[c % W];
    }
  };
  // Warm-up and calibration: at least 300 ms on the core, then calls sized so one measurement takes about `ms`.
  const double start = now_ns();
  double t = 0;
  for (;;) {
    const double t0 = now_ns();
    run(calls);
    t = now_ns() - t0;
    if (t >= 20e6 && now_ns() - start >= 300e6) break;
    if (t < 20e6) calls *= 2;
  }
  calls = std::max<std::size_t>(4, std::size_t(double(calls) * ms * 1e6 / t));
  const double cells = double(calls) * double(W) * double(cells1);
  std::vector<double> nspc, ghz;
  for (int r = 0; r < reps; ++r) {
    const double c0 = ns_per_cycle();
    const double t0 = now_ns();
    run(calls);
    const double ns = now_ns() - t0;
    const double c1 = ns_per_cycle();
    nspc.push_back(ns / cells);
    ghz.push_back(c0 > 0 ? 2 / (c0 + c1) : 0);
  }
  const double m = median(nspc), g = median(ghz);
  std::printf("time %s %-2s L %-4zu band %-4d W %-2zu cpu %d calls %zu cells %.4g ns/cell %.5f (min %.5f max %.5f)  "
              "clock %.3f GHz (min %.3f max %.3f)  cycles/row-step %.3f  checksum %.17g\n",
              sizeof(T) == 8 ? "f64" : "f32", sq ? "Sq" : "L1", L, band, W, cpu, calls, cells, m,
              *std::min_element(nspc.begin(), nspc.end()), *std::max_element(nspc.begin(), nspc.end()), g,
              *std::min_element(ghz.begin(), ghz.end()), *std::max_element(ghz.begin(), ghz.end()),
              m * g * double(W), checksum);
  return 0;
}

// ---------------------------------------------------------------------------------------------
// fill: Problem::fillDistanceMatrix_BruteForce (dtwc/Problem.cpp) on a packed lower triangle with NaN for
// "not computed", as core::DistanceMatrix: the diagonal set serially, then run_openmp(fill_row, N, true, 8), i.e.
// schedule(dynamic, max(1, N / (threads * 8))). Row i takes its columns W at a time through the lane function
// where the block's series are as long as series i and not all its pairs are known (the block at the row's end
// repeats its last column), then every pair still missing through the per-pair function.
template <class T> int fill_mode(bool sq, std::size_t N, std::size_t L, int band, int threads, int reps, bool ragged)
{
  constexpr std::size_t W = dc::dtw_lanes<T>;
  std::vector<std::vector<T>> S(N);
  std::uint64_t s = 99;
  for (std::size_t i = 0; i < N; ++i) gen(S[i], ragged ? L - 10 + splitmix(s) % 21 : L, 777 + i, 1);
  const BlockFn<T> block = block_for<T>(sq, band);
  const PairFn<T> pair = pair_for<T>(sq, band);
  const int nthreads = threads > 0 ? threads : omp_get_max_threads();
  const long long chunk = std::max<long long>(1, (long long)N / ((long long)nthreads * 8));
  auto tri = [](std::size_t i, std::size_t j) { return i * (i + 1) / 2 + j; }; // i >= j
  std::vector<double> cells_of(L + 11, -1); // the cells of an equal-length pair, by length
  double cells = 0;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j) {
      const std::size_t n = S[i].size();
      if (n != S[j].size()) {
        cells += double(n) * double(S[j].size()); // the ragged fill is unbanded
        continue;
      }
      if (cells_of[n] < 0) cells_of[n] = double(band_cells(n, band));
      cells += cells_of[n];
    }
  std::vector<double> M, times;
  std::uint64_t hash = 0;
  for (int r = 0; r < reps + 1; ++r) { // the first run warms the threads and the buffers, untimed
    M.assign(N * (N + 1) / 2, std::numeric_limits<double>::quiet_NaN());
    const double t0 = now_ns();
    for (std::size_t i = 0; i < N; ++i) M[tri(i, i)] = 0.0;
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp for schedule(dynamic, chunk) nowait
      for (long long il = 0; il < (long long)N; ++il) {
        const std::size_t i = std::size_t(il);
        const std::span<const T> x = S[i];
        std::array<std::span<const T>, W> ys;
        std::array<double, W> d;
        for (std::size_t j = i + 1; j < N; j += W) {
          const std::size_t count = std::min(W, N - j);
          bool equal = true, known = true;
          for (std::size_t w = 0; w < W; ++w) {
            const std::size_t c = j + std::min(w, count - 1);
            ys[w] = S[c];
            equal = equal && ys[w].size() == x.size();
            known = known && !std::isnan(M[tri(c, i)]);
          }
          if (!equal || known) continue;
          block(x, ys, d);
          for (std::size_t w = 0; w < count; ++w)
            if (std::isnan(M[tri(j + w, i)])) M[tri(j + w, i)] = d[w];
        }
        for (std::size_t j = i + 1; j < N; ++j)
          if (std::isnan(M[tri(j, i)])) M[tri(j, i)] = pair(x, S[j]);
      }
    }
    const double ns = now_ns() - t0;
    if (r > 0) times.push_back(ns);
    hash = fnv(kFnv0, M.data(), M.size() * sizeof(double));
  }
  const double m = median(times);
  std::printf("fill %s %-2s threads %2d N %-5zu L %-4zu%s band %-4d W %-2zu ms %.2f (min %.2f)  Gcell/s %.3f  "
              "matrix hash %016llx\n",
              sizeof(T) == 8 ? "f64" : "f32", sq ? "Sq" : "L1", nthreads, N, L, ragged ? " ragged+-10" : "", band, W,
              m / 1e6, *std::min_element(times.begin(), times.end()) / 1e6, cells / m, (unsigned long long)hash);
  return 0;
}

int main(int argc, char **argv)
{
  const std::string mode = argc > 1 ? argv[1] : "";
  auto arg = [&](int k, const char *def) { return argc > k ? argv[k] : def; };
  if (mode == "info") {
    std::printf("info %s W f64 %zu f32 %zu x86_64 %d FMA %d AVX2 %d omp threads %d\n", compiler(),
                dc::dtw_lanes<double>, dc::dtw_lanes<float>,
#if defined(__x86_64__)
                1,
#else
                0,
#endif
#if defined(__FMA__)
                1,
#else
                0,
#endif
#if defined(__AVX2__)
                1,
#else
                0,
#endif
                omp_get_max_threads());
    return 0;
  }
  if (mode == "check") return check_mode(std::atoi(arg(2, "3")));
  if (mode == "kinds") return kinds_mode();
  if (mode == "time") {
    const int cpu = std::atoi(arg(2, "12"));
    const bool f32 = std::string(arg(3, "f64")) == "f32", sq = std::string(arg(4, "L1")) == "Sq";
    const std::size_t L = std::strtoul(arg(5, "100"), nullptr, 10);
    const int band = std::atoi(arg(6, "-1")), reps = std::atoi(arg(7, "5"));
    const double ms = std::atof(arg(8, "100"));
    return f32 ? time_mode<float>(cpu, sq, L, band, reps, ms) : time_mode<double>(cpu, sq, L, band, reps, ms);
  }
  if (mode == "fill") {
    const bool f32 = std::string(arg(2, "f64")) == "f32", sq = std::string(arg(3, "L1")) == "Sq";
    const std::size_t N = std::strtoul(arg(4, "2000"), nullptr, 10), L = std::strtoul(arg(5, "100"), nullptr, 10);
    const int band = std::atoi(arg(6, "-1")), threads = std::atoi(arg(7, "0")), reps = std::atoi(arg(8, "3"));
    const bool ragged = std::atoi(arg(9, "0")) != 0;
    return f32 ? fill_mode<float>(sq, N, L, band, threads, reps, ragged)
               : fill_mode<double>(sq, N, L, band, threads, reps, ragged);
  }
  std::fprintf(stderr, "usage: xprobe info | check [seeds] | kinds | time cpu T D L band [reps] [ms] | "
                       "fill T D N L band [threads] [reps] [ragged]\n");
  return 2;
}
