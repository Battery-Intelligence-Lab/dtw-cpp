// Does the chain operand's position change the latency of fminnm / fadd / fcmgt+bif on this core?
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\n .rept 100\n add x9, x9, #1\n .endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "x9", "cc"); }
static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }
#define CELL(name, setup, body) \
  __attribute__((noinline)) static void name(uint64_t n) { asm volatile(setup "1:\n .rept 64\n" body ".endr\n subs %[n], %[n], #1\n b.ne 1b\n" : [n] "+r"(n) : : "v0", "v1", "v2", "v3", "v4", "v5", "cc"); }
#define S "fmov d0, #1.0\n fmov d3, #2.0\n movi d2, #0\n fmov d5, #1.0\n"
#define SV "fmov d0, #1.0\n dup v0.2d, v0.d[0]\n fmov d3, #2.0\n dup v3.2d, v3.d[0]\n movi v2.2d, #0\n"
CELL(c_mn_first_add_first, S, " fminnm d0, d0, d3\n fadd d0, d0, d2\n")
CELL(c_mn_second_add_first, S, " fminnm d0, d3, d0\n fadd d0, d0, d2\n")
CELL(c_mn_first_add_second, S, " fminnm d0, d0, d3\n fadd d0, d2, d0\n")
CELL(c_mn_second_add_second, S, " fminnm d0, d3, d0\n fadd d0, d2, d0\n")
CELL(c_mn2d_second_add_second, SV, " fminnm v0.2d, v3.2d, v0.2d\n fadd v0.2d, v2.2d, v0.2d\n")
CELL(c_mn2d_first_add_first, SV, " fminnm v0.2d, v0.2d, v3.2d\n fadd v0.2d, v0.2d, v2.2d\n")
CELL(c_fadd_only_first, S, " fadd d0, d0, d2\n")
CELL(c_fadd_only_second, S, " fadd d0, d2, d0\n")
CELL(c_fadd_nonzero, S, " fadd d0, d0, d5\n")
CELL(c_fmul_fadd, S, " fmul d0, d0, d5\n fadd d0, d0, d2\n")
CELL(c_fminnm_only_second, S, " fminnm d0, d3, d0\n")
CELL(c_fminnm_fmul, S, " fminnm d0, d0, d3\n fmul d0, d0, d5\n")
CELL(c_fcmgt_bif_fadd_2d_kernelform, SV, " fcmgt v4.2d, v3.2d, v0.2d\n bif v0.16b, v3.16b, v4.16b\n fadd v0.2d, v0.2d, v2.2d\n")
CELL(c_fcmp_fcsel_fadd_kernelform, S, " fcmp d0, d3\n fcsel d0, d0, d3, mi\n fadd d0, d0, d2\n")
int main() {
  for (int i = 0; i < 40; ++i) ns_per_cycle();
  struct T { const char *name; void (*f)(uint64_t); };
  const T ts[] = { { "fminnm(chain,c)+fadd(chain,c)", c_mn_first_add_first }, { "fminnm(c,chain)+fadd(chain,c)", c_mn_second_add_first },
                   { "fminnm(chain,c)+fadd(c,chain)", c_mn_first_add_second }, { "fminnm(c,chain)+fadd(c,chain)", c_mn_second_add_second },
                   { "fminnm.2d(c,chain)+fadd.2d(c,chain)", c_mn2d_second_add_second }, { "fminnm.2d(chain,c)+fadd.2d(chain,c)", c_mn2d_first_add_first },
                   { "fadd(chain,0)", c_fadd_only_first }, { "fadd(0,chain)", c_fadd_only_second }, { "fadd(chain,1.0)", c_fadd_nonzero },
                   { "fmul+fadd", c_fmul_fadd }, { "fminnm(c,chain) only", c_fminnm_only_second }, { "fminnm+fmul", c_fminnm_fmul },
                   { "fcmgt+bif+fadd .2d", c_fcmgt_bif_fadd_2d_kernelform }, { "fcmp+fcsel+fadd", c_fcmp_fcsel_fadd_kernelform } };
  for (const T &t : ts) {
    std::vector<double> v;
    const uint64_t it = 400000;
    for (int r = 0; r < 7; ++r) {
      const double c0 = ns_per_cycle(); const double t0 = now_ns(); t.f(it); const double ns = now_ns() - t0; const double c1 = ns_per_cycle();
      v.push_back(ns / ((c0 + c1) / 2) / (it * 64.0));
    }
    std::sort(v.begin(), v.end());
    std::printf("%-38s cycles per chain step median %.3f min %.3f\n", t.name, v[3], v[0]);
  }
}
