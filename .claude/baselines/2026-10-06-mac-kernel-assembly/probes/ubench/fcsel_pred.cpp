// Is the scalar min chain fcmp -> fcsel -> fadd a true data dependency on this core, whatever the
// select picks? Per "cell": fcmp d0, d3 ; fcsel d0, d0, d3, mi ; fadd d0, d0, d2  (d0 = chain).
//  A: always picks d0 (the chain)      d0 = 1, d3 = 2, d2 = 0
//  B: always picks d3 (not the chain)  d0 = 3, d3 = 2, d2 = 0  -> d0 stays 2: fcmp 2 < 2 false, picks d3
//  C: d3 loaded from a random array in [0, 4), d2 = 1: the pick is data-dependent and unpredictable
//  D: like C with fminnm d0, d0, d3 instead of fcmp+fcsel
//  E: like C, the pick computed branch-free in the vector unit (fcmgt+bif on d registers via .8b)
// cycles per cell against a dependent integer-add chain, interleaved.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\n .rept 100\n add x9, x9, #1\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "x9", "cc"); }
static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }
alignas(64) static double g_rand[1 << 16];
__attribute__((noinline)) static void cellA(uint64_t n) { asm volatile("fmov d0, #1.0\n fmov d3, #2.0\n movi d2, #0\n 1:\n .rept 64\n fcmp d0, d3\n fcsel d0, d0, d3, mi\n fadd d0, d0, d2\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "v0", "v2", "v3", "cc"); }
__attribute__((noinline)) static void cellB(uint64_t n) { asm volatile("fmov d0, #3.0\n fmov d3, #2.0\n movi d2, #0\n 1:\n .rept 64\n fcmp d0, d3\n fcsel d0, d0, d3, mi\n fadd d0, d0, d2\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "v0", "v2", "v3", "cc"); }
// C/D/E walk g_rand (64 Ki doubles, 512 KiB: L2) with a wrap every 1024 cells
__attribute__((noinline)) static void cellC(uint64_t n) { asm volatile("fmov d0, #1.0\n fmov d2, #1.0\n 1:\n mov x10, %[r]\n mov x11, #16\n 2:\n .rept 64\n ldr d3, [x10], #8\n fcmp d0, d3\n fcsel d0, d0, d3, mi\n fadd d0, d0, d2\n .endr\n subs x11, x11, #1\n b.ne 2b\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : [r] "r"(g_rand) : "v0", "v2", "v3", "x10", "x11", "cc", "memory"); }
__attribute__((noinline)) static void cellD(uint64_t n) { asm volatile("fmov d0, #1.0\n fmov d2, #1.0\n 1:\n mov x10, %[r]\n mov x11, #16\n 2:\n .rept 64\n ldr d3, [x10], #8\n fminnm d0, d0, d3\n fadd d0, d0, d2\n .endr\n subs x11, x11, #1\n b.ne 2b\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : [r] "r"(g_rand) : "v0", "v2", "v3", "x10", "x11", "cc", "memory"); }
__attribute__((noinline)) static void cellE(uint64_t n) { asm volatile("fmov d0, #1.0\n fmov d2, #1.0\n 1:\n mov x10, %[r]\n mov x11, #16\n 2:\n .rept 64\n ldr d3, [x10], #8\n fcmgt d4, d3, d0\n bif v0.8b, v3.8b, v4.8b\n fadd d0, d0, d2\n .endr\n subs x11, x11, #1\n b.ne 2b\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : [r] "r"(g_rand) : "v0", "v2", "v3", "v4", "x10", "x11", "cc", "memory"); }
// F: C with the same data but every value 3.5 (always picks the chain... d0 in [1, ...) grows: picks d3 once d0 > 3.5)
int main() {
  uint64_t s = 12345;
  for (auto &v : g_rand) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; v = double(s >> 11) * 0x1.0p-53 * 4.0; }
  for (int i = 0; i < 40; ++i) ns_per_cycle();
  struct T { const char *name; void (*f)(uint64_t); double cells_per_iter; };
  const T tests[] = { { "A fcsel picks chain", cellA, 64 }, { "B fcsel picks other", cellB, 64 },
                      { "C fcsel random pick", cellC, 1024 }, { "D fminnm random", cellD, 1024 },
                      { "E fcmgt+bif random", cellE, 1024 } };
  for (const T &t : tests) {
    std::vector<double> v;
    const uint64_t it = uint64_t(4e7 / t.cells_per_iter);
    for (int r = 0; r < 7; ++r) {
      const double c0 = ns_per_cycle(); const double t0 = now_ns(); t.f(it); const double ns = now_ns() - t0; const double c1 = ns_per_cycle();
      v.push_back(ns / ((c0 + c1) / 2) / (it * t.cells_per_iter));
    }
    std::sort(v.begin(), v.end());
    std::printf("%-24s cycles/cell median %.3f min %.3f max %.3f\n", t.name, v[3], v[0], v[6]);
  }
  // how often does C pick the chain?
  double d0 = 1; long chain = 0, n = 0;
  for (int rep = 0; rep < 4; ++rep) for (double r : g_rand) { if (d0 < r) ++chain; else d0 = r; d0 += 1; ++n; }
  std::printf("C/D/E data: the min picks the chain value in %.1f %% of cells\n", 100.0 * chain / n);
}
