// Per-pair linear kernel, cycles per cell against the column length n_short (inner loop trip count),
// at a fixed cell count: is the shipped kernel's speed below its chain latency an overlap of
// consecutive columns in the out-of-order window? dtw_kernel_linear(n_short, n_long, ...) directly.
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"
#include "v_fmin.hpp"
#include "v_skew2.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
namespace dc = dtwc::core;
__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\n .rept 100\n add x9, x9, #1\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "x9", "cc"); }
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }
__attribute__((noinline)) double k_ship(const double *x, const double *y, size_t ns, size_t nl) { return dc::dtw_kernel_linear<double>(ns, nl, dc::SpanL1Cost<double>{ x, y }, dc::StandardCell{}); }
__attribute__((noinline)) double k_fmin(const double *x, const double *y, size_t ns, size_t nl) { return v_fmin::dtw_kernel_linear<double>(ns, nl, dc::SpanL1Cost<double>{ x, y }, v_fmin::StandardCell{}); }
__attribute__((noinline)) double k_skew(const double *x, const double *y, size_t ns, size_t nl) { return v_skew2::dtw_kernel_linear<double>(ns, nl, dc::SpanL1Cost<double>{ x, y }, v_skew2::StandardCell{}); }
int main() {
  const size_t total = 1 << 24; // cells per call
  std::vector<double> x(1 << 21), y(1 << 21);
  double w = 0; uint64_t s = 1;
  for (auto &v : x) { s = s * 6364136223846793005ULL + 1; w += double(s >> 11) * 0x1.0p-53 - 0.5; v = w; }
  for (auto &v : y) { s = s * 6364136223846793005ULL + 1; w += double(s >> 11) * 0x1.0p-53 - 0.5; v = w; }
  for (int i = 0; i < 40; ++i) ns_per_cycle();
  struct K { const char *name; double (*f)(const double *, const double *, size_t, size_t); };
  const K ks[] = { { "shipped", k_ship }, { "v_fmin", k_fmin }, { "v_skew2", k_skew } };
  volatile double sink = 0;
  for (size_t ns : { 8, 16, 32, 64, 100, 128, 256, 512, 1000, 2048, 8192, 65536, 1 << 20 }) {
    const size_t nl = total / ns;
    double med[3];
    for (int k = 0; k < 3; ++k) {
      std::vector<double> v;
      for (int r = 0; r < 5; ++r) {
        const double c0 = ns_per_cycle(); const double t0 = now_ns(); sink = sink + ks[k].f(x.data(), y.data(), ns, nl);
        const double t = now_ns() - t0; const double c1 = ns_per_cycle();
        v.push_back(t / ((c0 + c1) / 2) / double(ns * nl));
      }
      std::sort(v.begin(), v.end()); med[k] = v[2];
    }
    std::printf("n_short %8zu n_long %8zu  cycles/cell  shipped %.3f  v_fmin %.3f  v_skew2 %.3f\n", ns, nl, med[0], med[1], med[2]);
  }
}
