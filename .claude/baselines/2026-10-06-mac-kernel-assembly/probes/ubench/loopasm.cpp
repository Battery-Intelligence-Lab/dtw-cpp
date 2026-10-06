// The per-pair inner loops, as linked (shipped: fcmp/fcsel) and as the fmin variant compiles them,
// replayed in asm over an L1-resident column (n = 1024 cells, 8 KiB), one column per call and the
// column re-entered from the top, so consecutive columns never overlap beyond one loop exit.
// Variants isolate what costs the fmin loop its extra cycle per cell.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\n .rept 100\n add x9, x9, #1\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "x9", "cc"); }
static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }
alignas(128) static double g_col[1024], g_x[1024], g_y[4];
// x13 = column (short_side), x14 = x, x21 = y, x12 = 0; d2/d3 = left/diag
#define LOOP(name, body)                                                                                  \
  __attribute__((noinline)) static void name(uint64_t reps)                                              \
  {                                                                                                       \
    asm volatile("1:\n mov x13, %[c]\n mov x14, %[x]\n mov x21, %[y]\n mov x12, #0\n mov x15, #1023\n"   \
                 " ldr d2, [x13], #8\n ldr d3, [x14]\n .p2align 6\n 2:\n" body                             \
                 " subs x15, x15, #1\n b.ne 2b\n subs %[n], %[n], #1\n b.ne 1b\n"                       \
                 : [n] "+r"(reps) : [c] "r"(g_col), [x] "r"(g_x), [y] "r"(g_y)                            \
                 : "x12", "x13", "x14", "x15", "x21", "v2", "v3", "v4", "v5", "v6", "cc", "memory");     \
  }
// shipped (dtwc_cl 1000ad708): 13 instructions
LOOP(l_shipped, " ldr d4, [x13]\n ldr d5, [x14], #8\n ldr d6, [x21, x12, lsl #3]\n fabd d5, d5, d6\n fcmp d4, d3\n fcsel d3, d4, d3, mi\n fcmp d2, d3\n fcsel d2, d2, d3, mi\n fadd d2, d2, d5\n str d2, [x13], #8\n mov v3.16b, v4.16b\n")
// fmin variant (kbench v_fmin 1000233c4): 11 instructions; left in d3, diag in d2
LOOP(l_fmin, " ldr d4, [x13]\n ldr d5, [x14], #8\n ldr d6, [x21, x12, lsl #3]\n fabd d5, d5, d6\n fminnm d2, d2, d4\n fminnm d2, d2, d3\n fadd d3, d5, d2\n str d3, [x13], #8\n mov v2.16b, v4.16b\n")
// fmin, the off-chain min written to a fresh register (no reuse of the diag register)
LOOP(l_fmin_fresh, " ldr d4, [x13]\n ldr d5, [x14], #8\n ldr d6, [x21, x12, lsl #3]\n fabd d5, d5, d6\n fminnm d6, d2, d4\n fminnm d6, d6, d3\n fadd d3, d5, d6\n str d3, [x13], #8\n mov v2.16b, v4.16b\n")
// fmin without the store (is it the store of the chain value?)
LOOP(l_fmin_nostore, " ldr d4, [x13], #8\n ldr d5, [x14], #8\n ldr d6, [x21, x12, lsl #3]\n fabd d5, d5, d6\n fminnm d2, d2, d4\n fminnm d2, d2, d3\n fadd d3, d5, d2\n mov v2.16b, v4.16b\n")
// shipped without the store
LOOP(l_shipped_nostore, " ldr d4, [x13], #8\n ldr d5, [x14], #8\n ldr d6, [x21, x12, lsl #3]\n fabd d5, d5, d6\n fcmp d4, d3\n fcsel d3, d4, d3, mi\n fcmp d2, d3\n fcsel d2, d2, d3, mi\n fadd d2, d2, d5\n mov v3.16b, v4.16b\n")
// fmin chain only: m1 from loads, no mov
LOOP(l_fmin_noload_m1, " ldr d4, [x13]\n ldr d5, [x14], #8\n fminnm d6, d4, d5\n fminnm d6, d6, d3\n fadd d3, d5, d6\n str d3, [x13], #8\n")
int main() {
  for (int i = 0; i < 1024; ++i) { g_col[i] = 1.0 + (i % 7) * 0.25; g_x[i] = (i % 5) * 0.5; }
  g_y[0] = 0.75;
  for (int i = 0; i < 40; ++i) ns_per_cycle();
  struct T { const char *name; void (*f)(uint64_t); };
  const T ts[] = { { "shipped fcmp/fcsel (13)", l_shipped }, { "v_fmin as compiled (11)", l_fmin }, { "v_fmin fresh temp", l_fmin_fresh },
                   { "v_fmin no store", l_fmin_nostore }, { "shipped no store", l_shipped_nostore }, { "fmin chain, m1 from loads", l_fmin_noload_m1 } };
  for (const T &t : ts) {
    std::vector<double> v;
    const uint64_t reps = 20000;
    for (int r = 0; r < 7; ++r) {
      const double c0 = ns_per_cycle(); const double t0 = now_ns(); t.f(reps); const double ns = now_ns() - t0; const double c1 = ns_per_cycle();
      v.push_back(ns / ((c0 + c1) / 2) / (reps * 1023.0));
    }
    std::sort(v.begin(), v.end());
    std::printf("%-28s cycles/cell median %.3f min %.3f\n", t.name, v[3], v[0]);
  }
}
