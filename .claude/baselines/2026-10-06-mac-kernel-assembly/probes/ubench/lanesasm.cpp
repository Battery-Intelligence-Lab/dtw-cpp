// The W=8 f64 lanes inner loop replayed in asm over an L1-resident column of 1024 rows x 64 B
// (x2 = rows, x1 = x values): shipped (fcmgt/bif) and the fmin variant as compiled, then the fmin
// loop with parts removed, to find what holds it above its 5-cycle chain. Cycles per row (8 cells).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\n .rept 100\n add x9, x9, #1\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "x9", "cc"); }
static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }
alignas(128) static double g_s[1025 * 8], g_x[1024];
#define CLOB "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "cc", "memory"
#define LOOP(name, body)                                                                                         \
  __attribute__((noinline)) static void name(uint64_t reps)                                                     \
  {                                                                                                              \
    asm volatile("fmov d1, #1.0\n dup v1.2d, v1.d[0]\n fmov d2, #0.5\n dup v2.2d, v2.d[0]\n fmov d3, #0.25\n"   \
                 " dup v3.2d, v3.d[0]\n fmov d4, #2.0\n dup v4.2d, v4.d[0]\n"                                    \
                 "1:\n mov x2, %[s]\n add x2, x2, #0x60\n mov x1, %[x]\n mov x0, #1023\n"                        \
                 " ldp q6, q16, [x2, #-0x60]\n ldp q17, q19, [x2, #-0x40]\n mov v5.16b, v6.16b\n mov v7.16b, v16.16b\n" \
                 " mov v18.16b, v17.16b\n mov v20.16b, v19.16b\n .p2align 6\n 2:\n" body                           \
                 " subs x0, x0, #1\n b.ne 2b\n subs %[n], %[n], #1\n b.ne 1b\n"                                  \
                 : [n] "+r"(reps) : [s] "r"(g_s), [x] "r"(g_x) : CLOB);                                          \
  }
// fmin variant as compiled (kbench lanes_v_fmin_double_L1, 28 instructions)
#define FMIN_BODY \
  " ldur q21, [x2, #-0x20]\n ld1r {v22.2d}, [x1], #8\n fabd v23.2d, v22.2d, v1.2d\n fminnm v6.2d, v6.2d, v21.2d\n fminnm v5.2d, v6.2d, v5.2d\n fadd v5.2d, v5.2d, v23.2d\n" \
  " ldp q23, q24, [x2, #-0x10]\n fminnm v6.2d, v16.2d, v23.2d\n fminnm v6.2d, v6.2d, v7.2d\n fabd v7.2d, v22.2d, v2.2d\n fadd v7.2d, v6.2d, v7.2d\n stp q5, q7, [x2, #-0x20]\n" \
  " fabd v6.2d, v22.2d, v3.2d\n fminnm v16.2d, v17.2d, v24.2d\n fminnm v16.2d, v16.2d, v18.2d\n fadd v18.2d, v16.2d, v6.2d\n ldr q25, [x2, #0x10]\n fabd v6.2d, v22.2d, v4.2d\n" \
  " fminnm v16.2d, v19.2d, v25.2d\n fminnm v16.2d, v16.2d, v20.2d\n fadd v20.2d, v16.2d, v6.2d\n stp q18, q20, [x2], #0x40\n mov v6.16b, v21.16b\n mov v16.16b, v23.16b\n mov v17.16b, v24.16b\n mov v19.16b, v25.16b\n"
LOOP(r_fmin, FMIN_BODY)
// shipped (dtwc_cl 1000ecd8c, 36 instructions; same registers)
LOOP(r_shipped,
  " ldur q21, [x2, #-0x20]\n ld1r {v22.2d}, [x1], #8\n fabd v23.2d, v22.2d, v1.2d\n fcmgt v24.2d, v6.2d, v21.2d\n bit v6.16b, v21.16b, v24.16b\n fcmgt v24.2d, v6.2d, v5.2d\n bif v5.16b, v6.16b, v24.16b\n fadd v5.2d, v5.2d, v23.2d\n"
  " ldp q23, q24, [x2, #-0x10]\n fcmgt v6.2d, v16.2d, v23.2d\n bsl v6.16b, v23.16b, v16.16b\n fcmgt v16.2d, v6.2d, v7.2d\n bif v7.16b, v6.16b, v16.16b\n fabd v6.2d, v22.2d, v2.2d\n fadd v7.2d, v7.2d, v6.2d\n stp q5, q7, [x2, #-0x20]\n"
  " fabd v6.2d, v22.2d, v3.2d\n fcmgt v16.2d, v17.2d, v24.2d\n bsl v16.16b, v24.16b, v17.16b\n fcmgt v17.2d, v16.2d, v18.2d\n bit v16.16b, v18.16b, v17.16b\n fadd v18.2d, v16.2d, v6.2d\n ldr q25, [x2, #0x10]\n"
  " fabd v6.2d, v22.2d, v4.2d\n fcmgt v16.2d, v19.2d, v25.2d\n bsl v16.16b, v25.16b, v19.16b\n fcmgt v17.2d, v16.2d, v20.2d\n bit v16.16b, v20.16b, v17.16b\n fadd v20.2d, v16.2d, v6.2d\n stp q18, q20, [x2], #0x40\n"
  " mov v6.16b, v21.16b\n mov v16.16b, v23.16b\n mov v17.16b, v24.16b\n mov v19.16b, v25.16b\n")
// fmin, m1 = up (no off-chain min): is the min(diag, up) what costs?
LOOP(r_fmin_no_m1,
  " ldur q21, [x2, #-0x20]\n ld1r {v22.2d}, [x1], #8\n fabd v23.2d, v22.2d, v1.2d\n fminnm v5.2d, v21.2d, v5.2d\n fadd v5.2d, v5.2d, v23.2d\n"
  " ldp q23, q24, [x2, #-0x10]\n fminnm v6.2d, v23.2d, v7.2d\n fabd v7.2d, v22.2d, v2.2d\n fadd v7.2d, v6.2d, v7.2d\n stp q5, q7, [x2, #-0x20]\n"
  " fabd v6.2d, v22.2d, v3.2d\n fminnm v16.2d, v24.2d, v18.2d\n fadd v18.2d, v16.2d, v6.2d\n ldr q25, [x2, #0x10]\n fabd v6.2d, v22.2d, v4.2d\n"
  " fminnm v16.2d, v25.2d, v20.2d\n fadd v20.2d, v16.2d, v6.2d\n stp q18, q20, [x2], #0x40\n")
// fmin, no stores (the column is only read)
LOOP(r_fmin_no_store,
  " ldur q21, [x2, #-0x20]\n ld1r {v22.2d}, [x1], #8\n fabd v23.2d, v22.2d, v1.2d\n fminnm v6.2d, v6.2d, v21.2d\n fminnm v5.2d, v6.2d, v5.2d\n fadd v5.2d, v5.2d, v23.2d\n"
  " ldp q23, q24, [x2, #-0x10]\n fminnm v6.2d, v16.2d, v23.2d\n fminnm v6.2d, v6.2d, v7.2d\n fabd v7.2d, v22.2d, v2.2d\n fadd v7.2d, v6.2d, v7.2d\n"
  " fabd v6.2d, v22.2d, v3.2d\n fminnm v16.2d, v17.2d, v24.2d\n fminnm v16.2d, v16.2d, v18.2d\n fadd v18.2d, v16.2d, v6.2d\n ldr q25, [x2, #0x10]\n fabd v6.2d, v22.2d, v4.2d\n"
  " fminnm v16.2d, v19.2d, v25.2d\n fminnm v16.2d, v16.2d, v20.2d\n fadd v20.2d, v16.2d, v6.2d\n add x2, x2, #0x40\n mov v6.16b, v21.16b\n mov v16.16b, v23.16b\n mov v17.16b, v24.16b\n mov v19.16b, v25.16b\n")
// fmin, the chain only: 4 x (fminnm with a loaded value, fadd), loads kept, no stores, no movs
LOOP(r_fmin_chain_only,
  " ldur q21, [x2, #-0x20]\n ldp q23, q24, [x2, #-0x10]\n ldr q25, [x2, #0x10]\n add x2, x2, #0x40\n"
  " fminnm v5.2d, v21.2d, v5.2d\n fadd v5.2d, v5.2d, v1.2d\n fminnm v7.2d, v23.2d, v7.2d\n fadd v7.2d, v7.2d, v1.2d\n"
  " fminnm v18.2d, v24.2d, v18.2d\n fadd v18.2d, v18.2d, v1.2d\n fminnm v20.2d, v25.2d, v20.2d\n fadd v20.2d, v20.2d, v1.2d\n")
// independent fcmp + fcsel mix: do they share two pipes? (throughput test, 16 of each per row)
__attribute__((noinline)) static void fcmp_fcsel_mix(uint64_t n) {
  asm volatile("fmov d1, #1.0\n fmov d2, #2.0\n 1:\n .rept 8\n fcmp d1, d2\n fcsel d3, d1, d2, mi\n fcmp d2, d1\n fcsel d4, d1, d2, mi\n"
               " fcmp d1, d1\n fcsel d5, d1, d2, mi\n fcmp d2, d2\n fcsel d6, d1, d2, mi\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n"
               : [n] "+r"(n) : : "v1", "v2", "v3", "v4", "v5", "v6", "cc");
}
int main() {
  for (int i = 0; i < 1025 * 8; ++i) g_s[i] = 1.0 + (i % 13) * 0.125;
  for (int i = 0; i < 1024; ++i) g_x[i] = (i % 7) * 0.375;
  for (int i = 0; i < 40; ++i) ns_per_cycle();
  struct T { const char *name; void (*f)(uint64_t); };
  const T ts[] = { { "shipped fcmgt/bif (36)", r_shipped }, { "fmin as compiled (28)", r_fmin }, { "fmin, m1 = up", r_fmin_no_m1 },
                   { "fmin, no stores", r_fmin_no_store }, { "fmin chain only", r_fmin_chain_only } };
  for (const T &t : ts) {
    std::vector<double> v;
    const uint64_t reps = 8000;
    for (int r = 0; r < 7; ++r) {
      const double c0 = ns_per_cycle(); const double t0 = now_ns(); t.f(reps); const double ns = now_ns() - t0; const double c1 = ns_per_cycle();
      v.push_back(ns / ((c0 + c1) / 2) / (reps * 1023.0));
    }
    std::sort(v.begin(), v.end());
    std::printf("%-26s cycles/row (8 cells) median %.3f min %.3f\n", t.name, v[3], v[0]);
  }
  std::vector<double> v;
  for (int r = 0; r < 7; ++r) {
    const uint64_t it = 1000000;
    const double c0 = ns_per_cycle(); const double t0 = now_ns(); fcmp_fcsel_mix(it); const double ns = now_ns() - t0; const double c1 = ns_per_cycle();
    v.push_back((it * 64.0) / (ns / ((c0 + c1) / 2)));
  }
  std::sort(v.begin(), v.end());
  std::printf("fcmp+fcsel independent mix: %.3f instructions/cycle (median), %.3f max\n", v[3], v[6]);
}
