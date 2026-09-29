// PF-5 probe: W independent equal-length DTW pairs in SIMD lanes, in plain C++ (no intrinsics, no SIMD
// library), against the fixed scalar kernel (K1 form: nested two-argument min, dp[i-1, j] carried in a
// register, row pointer hoisted).
//   pf5_lanes.exe verify   bit-identity: lanes vs scalar reference vs library kernel, every (T, W, L, band)
//   pf5_lanes.exe single   single-thread interleaved A/B timing (run it pinned to one P-core)
//   pf5_lanes.exe fill     OpenMP packed-matrix fill, N = 256, L = 1000, f64, all cores (advisory)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "core/dtw_kernel.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace dtwc::core;

// ---------------------------------------------------------------------------------------------------
// Scalar reference: linear_carry and NestedCell copied verbatim from
// .claude/baselines/2026-09-29-windows-kernel/min_carry_probe.cpp.
struct NestedCell {
  template <typename T> T combine(T d, T u, T l, T c, std::size_t, std::size_t) const noexcept { return std::min(std::min(d, u), l) + c; }
  template <typename T> T seed(T c, std::size_t, std::size_t) const noexcept { return c; }
};
// Same recurrence as dtw_kernel_linear (no early abandon), with the row pointer hoisted and dp[i-1, j] carried.
template <typename T, typename Cost, typename Cell>
T linear_carry(std::size_t n_short, std::size_t n_long, Cost cost, Cell cell) {
  constexpr T maxValue = std::numeric_limits<T>::max();
  thread_local static std::vector<T> buf;
  buf.resize(n_short);
  T *s = buf.data();
  s[0] = cell.seed(cost(0, 0), 0, 0);
  for (std::size_t i = 1; i < n_short; ++i) s[i] = cell.combine(maxValue, s[i - 1], maxValue, cost(i, 0), i, 0);
  for (std::size_t j = 1; j < n_long; ++j) {
    T diag = s[0];
    T left = cell.combine(maxValue, maxValue, s[0], cost(0, j), 0, j);
    s[0] = left;
    for (std::size_t i = 1; i < n_short; ++i) {
      const T up = s[i];
      left = cell.combine(diag, up, left, cost(i, j), i, j);
      diag = up;
      s[i] = left;
    }
  }
  return s[n_short - 1];
}

// Banded version of the same reference: Sakoe-Chiba |i - j| <= band over an n x n grid, the rows of
// column j given by dtw_band_bounds (equal lengths only). Cells outside the band hold maxValue.
template <typename T, typename Cost, typename Cell>
T banded_carry(std::size_t n, int band, Cost cost, Cell cell) {
  constexpr T maxValue = std::numeric_limits<T>::max();
  thread_local static std::vector<T> buf;
  buf.assign(n, maxValue);
  T *s = buf.data();
  s[0] = cell.seed(cost(0, 0), 0, 0);
  const std::size_t hi0 = dtw_band_bounds(band, 0, n).second;
  for (std::size_t i = 1; i < hi0; ++i) s[i] = cell.combine(maxValue, s[i - 1], maxValue, cost(i, 0), i, 0);
  for (std::size_t j = 1; j < n; ++j) {
    const auto [lo, hi] = dtw_band_bounds(band, j, n);
    T diag, left;
    std::size_t i = lo;
    if (lo == 0) {
      diag = s[0];
      left = cell.combine(maxValue, maxValue, s[0], cost(0, j), 0, j);
      s[0] = left;
      i = 1;
    } else {
      diag = s[lo - 1]; // dp[lo-1, j-1], inside the previous column's band
      left = maxValue;  // dp[lo-1, j] is outside the band
    }
    for (; i < hi; ++i) {
      const T up = s[i];
      left = cell.combine(diag, up, left, cost(i, j), i, j);
      diag = up;
      s[i] = left;
    }
  }
  return s[n - 1];
}

// ---------------------------------------------------------------------------------------------------
// Lanes kernel: W equal-length pairs (x, y_w) sharing x. Y holds the W series interleaved [n][W]; s is
// scratch [n][W]. Each lane performs banded_carry's arithmetic in banded_carry's order (band < 0: full,
// which is linear_carry's), so every lane is bit-identical to the scalar reference.
template <typename T, std::size_t W>
void lanes_dtw(const T *x, const T *Y, std::size_t n, int band, T *s, T *out) {
  constexpr T maxValue = std::numeric_limits<T>::max();
  std::fill(s, s + n * W, maxValue);
  const std::size_t hi0 = dtw_band_bounds(band, 0, n).second;
  for (std::size_t w = 0; w < W; ++w) s[w] = std::abs(x[0] - Y[w]);
  for (std::size_t i = 1; i < hi0; ++i)
    for (std::size_t w = 0; w < W; ++w)
      s[i * W + w] = std::min(std::min(maxValue, s[(i - 1) * W + w]), maxValue) + std::abs(x[i] - Y[w]);
  for (std::size_t j = 1; j < n; ++j) {
    const auto [lo, hi] = dtw_band_bounds(band, j, n);
    T y[W], diag[W], left[W]; // one or two registers each; y copied so no store can alias it
    for (std::size_t w = 0; w < W; ++w) y[w] = Y[j * W + w];
    std::size_t i = lo;
    if (lo == 0) {
      for (std::size_t w = 0; w < W; ++w) {
        diag[w] = s[w];
        left[w] = std::min(std::min(maxValue, maxValue), s[w]) + std::abs(x[0] - y[w]);
        s[w] = left[w];
      }
      i = 1;
    } else {
      for (std::size_t w = 0; w < W; ++w) { diag[w] = s[(lo - 1) * W + w]; left[w] = maxValue; }
    }
    for (; i < hi; ++i) {
      const T xi = x[i];
      T *si = s + i * W;
      for (std::size_t w = 0; w < W; ++w) { // the lane loop: contiguous, must become ymm ops
        const T up = si[w];
        left[w] = std::min(std::min(diag[w], up), left[w]) + std::abs(xi - y[w]);
        diag[w] = up;
        si[w] = left[w];
      }
    }
  }
  for (std::size_t w = 0; w < W; ++w) out[w] = s[(n - 1) * W + w];
}

// Exported instantiations: named symbols for the assembly listing.
extern "C" {
void pf5_lanes_f64_w4(const double *x, const double *Y, std::size_t n, int b, double *s, double *o) { lanes_dtw<double, 4>(x, Y, n, b, s, o); }
void pf5_lanes_f64_w8(const double *x, const double *Y, std::size_t n, int b, double *s, double *o) { lanes_dtw<double, 8>(x, Y, n, b, s, o); }
void pf5_lanes_f32_w8(const float *x, const float *Y, std::size_t n, int b, float *s, float *o) { lanes_dtw<float, 8>(x, Y, n, b, s, o); }
void pf5_lanes_f32_w16(const float *x, const float *Y, std::size_t n, int b, float *s, float *o) { lanes_dtw<float, 16>(x, Y, n, b, s, o); }
double pf5_scalar_f64(const double *x, const double *y, std::size_t n) {
  auto cost = [x, y](std::size_t i, std::size_t j) { return std::abs(x[i] - y[j]); };
  return linear_carry<double>(n, n, cost, NestedCell{});
}
}

template <typename T, std::size_t W>
void lanes_call(const T *x, const T *Y, std::size_t n, int band, T *s, T *out) {
  if constexpr (std::is_same_v<T, double> && W == 4) pf5_lanes_f64_w4(x, Y, n, band, s, out);
  else if constexpr (std::is_same_v<T, double> && W == 8) pf5_lanes_f64_w8(x, Y, n, band, s, out);
  else if constexpr (std::is_same_v<T, float> && W == 8) pf5_lanes_f32_w8(x, Y, n, band, s, out);
  else pf5_lanes_f32_w16(x, Y, n, band, s, out);
}

template <typename T> T *align64(T *p) {
  return reinterpret_cast<T *>((reinterpret_cast<std::uintptr_t>(p) + 63) & ~std::uintptr_t(63));
}

// What a fill would call: pack W column series into [n][W] (O(n*W)), run the lanes. thread_local
// scratch grows and never shrinks, as in the library kernels.
template <typename T, std::size_t W>
void lanes_block(const T *x, const T *const *ys, std::size_t n, int band, T *out) {
  thread_local std::vector<T> ybuf, sbuf;
  const std::size_t need = n * W + 64 / sizeof(T);
  if (ybuf.size() < need) { ybuf.resize(need); sbuf.resize(need); }
  T *Y = align64(ybuf.data()), *s = align64(sbuf.data());
  for (std::size_t t = 0; t < n; ++t)
    for (std::size_t w = 0; w < W; ++w) Y[t * W + w] = ys[w][t];
  lanes_call<T, W>(x, Y, n, band, s, out);
}

template <typename T> T scalar_pair(const T *x, const T *y, std::size_t n, int band) {
  auto cost = [x, y](std::size_t i, std::size_t j) { return std::abs(x[i] - y[j]); };
  return band < 0 ? linear_carry<T>(n, n, cost, NestedCell{}) : banded_carry<T>(n, band, cost, NestedCell{});
}

// Third computation: the library's own kernels (StandardCell, init-list min).
template <typename T> T library_pair(const T *x, const T *y, std::size_t n, int band) {
  auto cost = [x, y](std::size_t i, std::size_t j) { return std::abs(x[i] - y[j]); };
  return band < 0 ? dtw_kernel_linear<T>(n, n, cost, StandardCell{}) : dtw_kernel_banded<T>(n, n, band, cost, StandardCell{});
}

double band_cells(std::size_t n, int band) {
  double c = 0;
  for (std::size_t j = 0; j < n; ++j) { const auto [lo, hi] = dtw_band_bounds(band, j, n); c += double(hi - lo); }
  return c;
}

// ---------------------------------------------------------------------------------------------------
// Machine load over an interval: busy share of all logical CPUs, and the share not used by this process.
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

using clk = std::chrono::steady_clock;
double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }
struct Stat { double med, lo, hi; };
Stat stat(std::vector<double> v) { std::sort(v.begin(), v.end()); return {v[v.size() / 2], v.front(), v.back()}; }

template <typename T> std::vector<T> make_series(std::mt19937_64 &g, std::size_t n, bool walk) {
  std::normal_distribution<double> nd;
  std::vector<T> v(n);
  double acc = 0;
  for (auto &e : v) { const double z = nd(g); acc += z; e = T(walk ? acc : z); }
  return v;
}

// ---------------------------------------------------------------------------------------------------
template <typename T, std::size_t W>
int verify_config(std::size_t n, int band, bool walk, bool with_library) {
  std::mt19937_64 g(1000003ull * n + 7919ull * W + 31ull * sizeof(T) + (walk ? 1 : 0) + std::uint64_t(band + 2));
  const auto x = make_series<T>(g, n, walk);
  std::vector<std::vector<T>> y;
  const T *ys[W];
  for (std::size_t w = 0; w < W; ++w) y.push_back(make_series<T>(g, n, walk));
  for (std::size_t w = 0; w < W; ++w) ys[w] = y[w].data();
  T out[W];
  lanes_block<T, W>(x.data(), ys, n, band, out);
  double max_lane = 0, max_lib = 0;
  int bad = 0;
  for (std::size_t w = 0; w < W; ++w) {
    const T ref = scalar_pair<T>(x.data(), ys[w], n, band);
    if (std::memcmp(&ref, &out[w], sizeof(T)) != 0) ++bad;
    max_lane = std::max(max_lane, std::abs(double(ref) - double(out[w])));
    if (with_library) {
      const T lib = library_pair<T>(x.data(), ys[w], n, band);
      if (std::memcmp(&ref, &lib, sizeof(T)) != 0) ++bad;
      max_lib = std::max(max_lib, std::abs(double(ref) - double(lib)));
    }
  }
  std::printf("verify %s W=%-2zu L=%-4zu band=%-4d data=%-6s lanes-vs-scalar max|diff|=%g  scalar-vs-library max|diff|=%s  mismatches=%d\n",
              sizeof(T) == 8 ? "f64" : "f32", W, n, band, walk ? "walk" : "normal", max_lane,
              with_library ? std::to_string(max_lib).c_str() : "n/a", bad);
  return bad;
}

template <typename T, std::size_t W> int verify_all() {
  int bad = 0;
  for (std::size_t n : {1, 2, 3, 7, 33}) // edge cases: empty band interior, band 0 and 1
    for (int band : {-1, 0, 1, 2})
      for (bool walk : {false, true}) bad += verify_config<T, W>(n, band, walk, true);
  for (std::size_t n : {100, 500, 1000, 4000})
    for (int band : {-1, int(n / 10)})
      for (bool walk : {false, true}) bad += verify_config<T, W>(n, band, walk, true);
  return bad;
}

// ---------------------------------------------------------------------------------------------------
template <typename T, std::size_t W>
void time_config(std::size_t n, int band, int rounds) {
  std::mt19937_64 g(42 + n + W);
  const auto x = make_series<T>(g, n, false);
  std::vector<std::vector<T>> y;
  const T *ys[W];
  for (std::size_t w = 0; w < W; ++w) y.push_back(make_series<T>(g, n, false));
  for (std::size_t w = 0; w < W; ++w) ys[w] = y[w].data();
  const double cells = double(W) * band_cells(n, band); // one block: W pairs
  const int reps = std::max(1, int(std::lround(2e7 / cells)));
  double sink_s = 0, sink_l = 0;
  T out[W];
  auto run_scalar = [&] {
    const auto t0 = clk::now();
    for (int r = 0; r < reps; ++r)
      for (std::size_t w = 0; w < W; ++w) sink_s += double(scalar_pair<T>(x.data(), ys[w], n, band));
    return secs(t0, clk::now());
  };
  auto run_lanes = [&] {
    const auto t0 = clk::now();
    for (int r = 0; r < reps; ++r) {
      lanes_block<T, W>(x.data(), ys, n, band, out);
      for (std::size_t w = 0; w < W; ++w) sink_l += double(out[w]);
    }
    return secs(t0, clk::now());
  };
  run_scalar(); run_lanes(); // warm-up: scratch allocation, clock ramp
  sink_s = sink_l = 0;
  std::vector<double> ts, tl, ratio;
  LoadMeter lm; lm.start();
  for (int r = 0; r < rounds; ++r) {
    double a, b;
    if (r % 2 == 0) { a = run_scalar(); b = run_lanes(); } else { b = run_lanes(); a = run_scalar(); }
    ts.push_back(1e9 * a / (reps * cells));
    tl.push_back(1e9 * b / (reps * cells));
    ratio.push_back(a / b);
  }
  double busy, other; lm.stop(busy, other);
  const Stat S = stat(ts), La = stat(tl), R = stat(ratio);
  std::printf("time %s W=%-2zu L=%-4zu band=%-4d reps=%-4d scalar %.3f [%.3f-%.3f] ns/cell  lanes %.3f [%.3f-%.3f] ns/cell  "
              "ratio %.2f [%.2f-%.2f]  sums_equal=%d  load busy=%.0f%% other=%.0f%%\n",
              sizeof(T) == 8 ? "f64" : "f32", W, n, band, reps, S.med, S.lo, S.hi, La.med, La.lo, La.hi,
              R.med, R.lo, R.hi, int(sink_s == sink_l), busy, other);
  std::fflush(stdout);
}

template <typename T, std::size_t W> void time_all(int rounds) {
  for (std::size_t n : {100, 500, 1000, 4000})
    for (int band : {-1, int(n / 10)}) time_config<T, W>(n, band, rounds);
}

// ---------------------------------------------------------------------------------------------------
// Packed upper-triangle fill, OpenMP over rows (row i against columns j > i).
std::size_t tri(std::size_t N, std::size_t i, std::size_t j) { return i * N - i * (i + 1) / 2 + (j - i - 1); }

void fill_scalar(const std::vector<std::vector<double>> &S, double *D) {
  const std::size_t N = S.size(), n = S[0].size();
#pragma omp parallel for schedule(dynamic, 1)
  for (long long i = 0; i < (long long)N - 1; ++i)
    for (std::size_t j = std::size_t(i) + 1; j < N; ++j) D[tri(N, i, j)] = scalar_pair<double>(S[i].data(), S[j].data(), n, -1);
}

template <std::size_t W> void fill_lanes(const std::vector<std::vector<double>> &S, double *D) {
  const std::size_t N = S.size(), n = S[0].size();
#pragma omp parallel for schedule(dynamic, 1)
  for (long long i = 0; i < (long long)N - 1; ++i)
    for (std::size_t j0 = std::size_t(i) + 1; j0 < N; j0 += W) {
      const std::size_t cnt = std::min(W, N - j0);
      const double *ys[W];
      for (std::size_t w = 0; w < W; ++w) ys[w] = S[j0 + std::min(w, cnt - 1)].data(); // pad the tail
      double out[W];
      lanes_block<double, W>(S[i].data(), ys, n, -1, out);
      for (std::size_t w = 0; w < cnt; ++w) D[tri(N, i, j0 + w)] = out[w];
    }
}

void fill_mode() {
  const std::size_t N = 256, n = 1000, P = N * (N - 1) / 2;
  std::mt19937_64 g(2026);
  std::vector<std::vector<double>> S;
  for (std::size_t k = 0; k < N; ++k) S.push_back(make_series<double>(g, n, true));
  std::vector<double> Ds(P), D4(P), D8(P);
  const double cells = double(P) * n * n;
  std::printf("fill N=%zu L=%zu pairs=%zu threads=%d\n", N, n, P, omp_get_max_threads());
  std::vector<double> ts, t4, t8;
  for (int r = 0; r < 3; ++r) {
    LoadMeter lm; lm.start();
    auto a = clk::now(); fill_scalar(S, Ds.data());
    auto b = clk::now(); fill_lanes<4>(S, D4.data());
    auto c = clk::now(); fill_lanes<8>(S, D8.data());
    auto d = clk::now();
    double busy, other; lm.stop(busy, other);
    ts.push_back(secs(a, b)); t4.push_back(secs(b, c)); t8.push_back(secs(c, d));
    std::printf("fill round %d: scalar %.3f s (%.2f Gcell/s)  lanes W=4 %.3f s (%.2f)  lanes W=8 %.3f s (%.2f)  "
                "x%.2f x%.2f  load busy=%.0f%% other=%.0f%%\n",
                r, ts.back(), cells / ts.back() * 1e-9, t4.back(), cells / t4.back() * 1e-9, t8.back(),
                cells / t8.back() * 1e-9, ts.back() / t4.back(), ts.back() / t8.back(), busy, other);
    std::fflush(stdout);
  }
  const bool same4 = std::memcmp(Ds.data(), D4.data(), P * sizeof(double)) == 0;
  const bool same8 = std::memcmp(Ds.data(), D8.data(), P * sizeof(double)) == 0;
  const Stat S0 = stat(ts), S4 = stat(t4), S8 = stat(t8);
  std::printf("fill median: scalar %.3f s  W=4 %.3f s (x%.2f)  W=8 %.3f s (x%.2f)  identical W4=%d W8=%d\n",
              S0.med, S4.med, S0.med / S4.med, S8.med, S0.med / S8.med, int(same4), int(same8));
}

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "verify";
  if (mode == "verify") {
    int bad = verify_all<double, 4>() + verify_all<double, 8>() + verify_all<float, 8>() + verify_all<float, 16>();
    std::printf("VERIFY %s mismatches=%d\n", bad == 0 ? "PASS" : "FAIL", bad);
    return bad == 0 ? 0 : 1;
  }
  if (mode == "single") {
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 15;
    DWORD_PTR pm = 0, sm = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm);
    std::printf("single: affinity mask 0x%llx, rounds %d\n", (unsigned long long)pm, rounds);
    time_all<double, 4>(rounds);
    time_all<double, 8>(rounds);
    time_all<float, 8>(rounds);
    time_all<float, 16>(rounds);
    return 0;
  }
  if (mode == "fill") { fill_mode(); return 0; }
  std::fprintf(stderr, "usage: pf5_lanes.exe verify|single [rounds]|fill\n");
  return 2;
}
