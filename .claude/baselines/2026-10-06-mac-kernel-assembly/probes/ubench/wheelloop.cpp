// The wheel's f64 per-pair Standard L1 inner loop (_dtwcpp_core...so @1ec44, 16 instructions: the cell plus
// the early-abandon row minimum, fcmp/fccmp/fcsel, not unswitched) against the CLI's 13-instruction loop
// (dtwc_cl @1000ad554), replayed over a long L1-resident column (4096 cells), single thread.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\n .rept 100\n add x9, x9, #1\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "x9", "cc"); }
static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }
alignas(128) static double g_col[4097], g_x[4097], g_y[4];
#define LOOP(name, body)                                                                                         \
  __attribute__((noinline)) static void name(uint64_t reps)                                                     \
  {                                                                                                              \
    asm volatile("fmov d8, #-1.0\n movi d1, #0\n 1:\n mov x14, %[c]\n mov x13, %[x]\n mov x20, %[y]\n mov x11, #0\n" \
                 " mov x12, #4095\n ldr d4, [x14], #8\n ldr d5, [x13]\n fmov d3, #1.0\n .p2align 6\n 2:\n" body      \
                 " subs x12, x12, #1\n b.ne 2b\n subs %[n], %[n], #1\n b.ne 1b\n"                                   \
                 : [n] "+r"(reps) : [c] "r"(g_col), [x] "r"(g_x), [y] "r"(g_y)                                      \
                 : "x11", "x12", "x13", "x14", "x20", "v1", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "cc", "memory"); \
  }
LOOP(wheel16, " ldr d6, [x14]\n ldr d7, [x20, x11, lsl #3]\n ldr d16, [x13], #8\n fabd d7, d7, d16\n fcmp d6, d5\n fcsel d5, d6, d5, mi\n"
              " fcmp d4, d5\n fcsel d4, d4, d5, mi\n fadd d4, d4, d7\n fcmp d4, d3\n fccmp d8, d1, #0x8, mi\n str d4, [x14], #8\n"
              " fcsel d3, d4, d3, ge\n mov v5.16b, v6.16b\n")
LOOP(cli13, " ldr d6, [x14]\n ldr d7, [x20, x11, lsl #3]\n ldr d16, [x13], #8\n fabd d7, d7, d16\n fcmp d6, d5\n fcsel d5, d6, d5, mi\n"
            " fcmp d4, d5\n fcsel d4, d4, d5, mi\n fadd d4, d4, d7\n str d4, [x14], #8\n mov v5.16b, v6.16b\n")
// the row-minimum chain alone: fcmp + fccmp + fcsel per cell
LOOP(rowmin_only, " ldr d4, [x14], #8\n fcmp d4, d3\n fccmp d8, d1, #0x8, mi\n fcsel d3, d4, d3, ge\n")
int main() {
  for (int i = 0; i < 4097; ++i) { g_col[i] = 1.0 + (i % 11) * 0.25; g_x[i] = (i % 5) * 0.5; }
  g_y[0] = 0.75;
  for (int i = 0; i < 40; ++i) ns_per_cycle();
  struct T { const char *name; void (*f)(uint64_t); };
  const T ts[] = { { "wheel 16-insn loop", wheel16 }, { "CLI 13-insn loop", cli13 }, { "row-min chain only", rowmin_only } };
  for (const T &t : ts) {
    std::vector<double> v;
    const uint64_t reps = 2000;
    for (int r = 0; r < 7; ++r) {
      const double c0 = ns_per_cycle(); const double t0 = now_ns(); t.f(reps); const double ns = now_ns() - t0; const double c1 = ns_per_cycle();
      v.push_back(ns / ((c0 + c1) / 2) / (reps * 4095.0));
    }
    std::sort(v.begin(), v.end());
    std::printf("%-22s cycles/cell median %.3f min %.3f\n", t.name, v[3], v[0]);
  }
}
