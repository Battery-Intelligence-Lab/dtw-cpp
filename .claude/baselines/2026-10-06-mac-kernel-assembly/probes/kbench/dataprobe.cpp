// Per-pair linear kernel at one long column shape (n_short 4096): does the speed depend on the data?
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"
#include "v_fmin.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
namespace dc = dtwc::core;
__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\n .rept 100\n add x9, x9, #1\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "x9", "cc"); }
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }
__attribute__((noinline)) double k_ship(const double *x, const double *y, size_t ns, size_t nl) { return dc::dtw_kernel_linear<double>(ns, nl, dc::SpanL1Cost<double>{ x, y }, dc::StandardCell{}); }
__attribute__((noinline)) double k_fmin(const double *x, const double *y, size_t ns, size_t nl) { return v_fmin::dtw_kernel_linear<double>(ns, nl, dc::SpanL1Cost<double>{ x, y }, v_fmin::StandardCell{}); }
int main() {
  const size_t ns = 4096, nl = 4096;
  std::vector<double> x(ns), y(nl);
  for (int kind = 0; kind < 6; ++kind) {
    uint64_t s = 7; double w = 0;
    auto rnd = [&] { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return double(s >> 11) * 0x1.0p-53; };
    for (size_t i = 0; i < ns; ++i) {
      switch (kind) {
      case 0: w += rnd() - 0.5; x[i] = w; break;          // random walk
      case 1: x[i] = rnd(); break;                        // uniform [0,1)
      case 2: x[i] = 0; break;                            // zeros
      case 3: x[i] = std::sin(i * 0.01); break;           // smooth
      case 4: x[i] = double(int(rnd() * 3)); break;       // integers 0..2
      case 5: x[i] = rnd() * 1e-300; break;               // tiny (subnormal costs possible)
      }
    }
    w = 0;
    for (size_t i = 0; i < nl; ++i) {
      switch (kind) {
      case 0: w += rnd() - 0.5; y[i] = w; break;
      case 1: y[i] = rnd(); break;
      case 2: y[i] = 0; break;
      case 3: y[i] = std::sin(i * 0.011 + 0.3); break;
      case 4: y[i] = double(int(rnd() * 3)); break;
      case 5: y[i] = rnd() * 1e-300; break;
      }
    }
    if (kind == 0) for (int i = 0; i < 40; ++i) ns_per_cycle();
    double med[2];
    for (int k = 0; k < 2; ++k) {
      std::vector<double> v;
      for (int r = 0; r < 5; ++r) {
        const double c0 = ns_per_cycle(); const double t0 = now_ns();
        volatile double d = k ? k_fmin(x.data(), y.data(), ns, nl) : k_ship(x.data(), y.data(), ns, nl);
        (void)d;
        const double t = now_ns() - t0; const double c1 = ns_per_cycle();
        v.push_back(t / ((c0 + c1) / 2) / double(ns * nl));
      }
      std::sort(v.begin(), v.end()); med[k] = v[2];
    }
    const char *names[] = { "random walk", "uniform", "zeros", "sine", "ints 0..2", "tiny 1e-300" };
    std::printf("%-12s n 4096x4096  cycles/cell  shipped %.3f  v_fmin %.3f\n", names[kind], med[0], med[1]);
  }
}
