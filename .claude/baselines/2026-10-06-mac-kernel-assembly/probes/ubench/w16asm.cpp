// The fmin W=16 f64 lanes inner loop (kbench lanes_v_fmin_w16_double_L1 @10001b18c, 56 instructions)
// replayed over an L1-resident column (1024 rows x 128 B); the stack reload goes to a static slot.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\n .rept 100\n add x9, x9, #1\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "x9", "cc"); }
static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }
alignas(128) static double g_s[1026 * 16], g_x[1024], g_slot[2];
__attribute__((noinline)) static void replay(uint64_t reps) {
  asm volatile("mov x3, %[slot]\n 1:\n mov x2, %[s]\n add x2, x2, #0x80\n mov x1, %[x]\n mov x0, #1023\n .p2align 6\n 2:\nldur q9, [x2, #-0x40]\nld1r {v12.2d}, [x1], #8\nfabd v10.2d, v12.2d, v30.2d\nfminnm v19.2d, v19.2d, v9.2d\nfminnm v17.2d, v19.2d, v17.2d\nfadd v17.2d, v17.2d, v10.2d\nldp q10, q11, [x2, #-0x30]\nfminnm v19.2d, v21.2d, v10.2d\nfminnm v18.2d, v19.2d, v18.2d\nfabd v19.2d, v12.2d, v13.2d\nfadd v18.2d, v18.2d, v19.2d\nstp q17, q18, [x2, #-0x40]\nfabd v19.2d, v12.2d, v14.2d\nfminnm v21.2d, v23.2d, v11.2d\nfminnm v20.2d, v21.2d, v20.2d\nfadd v20.2d, v20.2d, v19.2d\nldp q13, q14, [x2, #-0x10]\nfminnm v19.2d, v24.2d, v13.2d\nfminnm v19.2d, v19.2d, v22.2d\nfabd v21.2d, v12.2d, v4.2d\nfadd v22.2d, v19.2d, v21.2d\nstp q20, q22, [x2, #-0x20]\nfabd v19.2d, v12.2d, v5.2d\nfminnm v21.2d, v26.2d, v14.2d\nfminnm v21.2d, v21.2d, v25.2d\nfadd v25.2d, v21.2d, v19.2d\nldp q15, q30, [x2, #0x10]\nfminnm v19.2d, v28.2d, v15.2d\nfminnm v19.2d, v19.2d, v27.2d\nfabd v21.2d, v12.2d, v6.2d\nfadd v27.2d, v19.2d, v21.2d\nstp q25, q27, [x2]\nfabd v19.2d, v12.2d, v7.2d\nfminnm v0.2d, v0.2d, v30.2d\nfminnm v0.2d, v0.2d, v29.2d\nfadd v29.2d, v0.2d, v19.2d\nldr q8, [x2, #0x30]\nfabd v0.2d, v12.2d, v16.2d\nfminnm v1.2d, v1.2d, v8.2d\nfminnm v1.2d, v1.2d, v31.2d\nfadd v31.2d, v1.2d, v0.2d\nstp q29, q31, [x2, #0x20]\nadd x2, x2, #0x80\nmov v19.16b, v9.16b\nmov v21.16b, v10.16b\nmov v23.16b, v11.16b\nmov v24.16b, v13.16b\nmov v13.16b, v3.16b\nmov v26.16b, v14.16b\nmov v14.16b, v2.16b\nmov v28.16b, v15.16b\nmov v0.16b, v30.16b\nldr q30, [x3]\nmov v1.16b, v8.16b\n subs x0, x0, #1\n b.ne 2b\n subs %[n], %[n], #1\n b.ne 1b\n"
    : [n] "+r"(reps) : [s] "r"(g_s), [x] "r"(g_x), [slot] "r"(g_slot)
    : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory");
}
int main() {
  for (int i = 0; i < 1026 * 16; ++i) g_s[i] = 1.0 + (i % 13) * 0.125;
  for (int i = 0; i < 1024; ++i) g_x[i] = (i % 7) * 0.375;
  g_slot[0] = g_slot[1] = 0.5;
  for (int i = 0; i < 40; ++i) ns_per_cycle();
  std::vector<double> v;
  const uint64_t reps = 4000;
  for (int r = 0; r < 7; ++r) {
    const double c0 = ns_per_cycle(); const double t0 = now_ns(); replay(reps); const double ns = now_ns() - t0; const double c1 = ns_per_cycle();
    v.push_back(ns / ((c0 + c1) / 2) / (reps * 1023.0));
  }
  std::sort(v.begin(), v.end());
  std::printf("fmin W16 f64 lanes loop replay: cycles/row (16 cells) median %.3f min %.3f -> cycles/cell %.4f\n", v[3], v[0], v[3] / 16);
}
