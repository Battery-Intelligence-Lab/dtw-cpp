"""Generates ubench.cpp: latency and throughput microbenchmarks in inline asm for AArch64 (Apple).
Each test is one function f(uint64_t iters) whose loop body holds `per_iter` instructions of the class
under test; the harness times it against a dependent integer-add chain (1 cycle/add) run in the same
process, interleaved, and prints cycles per instruction (latency tests) or instructions per cycle
(throughput tests)."""
import textwrap

tests = []  # (name, kind, per_iter_count, setup_asm, body_asm_unit, clobbers)

def add(name, kind, n, setup, body, clob):
    tests.append((name, kind, n, setup, body, clob))

VREGS = [f'v{i}' for i in range(32)]
DCLOB = ','.join(f'"v{i}"' for i in range(32))

# --- latency (dependent chain through the first register) ---
add('int_add_lat', 'lat', 100, '', '.rept 100\n add x9, x9, #1\n .endr', '"x9"')
setup_fp = 'fmov d0, #1.0\n fmov d1, #1.0\n movi d2, #0\n fmov d3, #2.0\n dup v0.2d, v0.d[0]\n dup v1.2d, v1.d[0]\n dup v3.2d, v3.d[0]\n'
setup_fp_s = 'fmov s0, #1.0\n fmov s1, #1.0\n movi d2, #0\n fmov s3, #2.0\n dup v0.4s, v0.s[0]\n dup v1.4s, v1.s[0]\n dup v3.4s, v3.s[0]\n'
for op, rhs in [('fadd', 'd2'), ('fmul', 'd1'), ('fabd', 'd2'), ('fmin', 'd3'), ('fminnm', 'd3'), ('fmax', 'd2')]:
    add(f'{op}_d_lat', 'lat', 100, setup_fp, f'.rept 100\n {op} d0, d0, {rhs}\n .endr', DCLOB)
for op, rhs in [('fadd', 'v2'), ('fmul', 'v1'), ('fabd', 'v2'), ('fmin', 'v3'), ('fminnm', 'v3')]:
    add(f'{op}_2d_lat', 'lat', 100, setup_fp, f'.rept 100\n {op} v0.2d, v0.2d, {rhs}.2d\n .endr', DCLOB)
    add(f'{op}_4s_lat', 'lat', 100, setup_fp_s, f'.rept 100\n {op} v0.4s, v0.4s, {rhs}.4s\n .endr', DCLOB)
# fcmgt alone: v0 = fcmgt(v0, v3) -> mask bits; values become NaN patterns but compare cost is data-independent
add('fcmgt_2d_lat', 'lat', 100, setup_fp, '.rept 100\n fcmgt v0.2d, v0.2d, v3.2d\n .endr', DCLOB)
add('bif_16b_lat', 'lat', 100, setup_fp, '.rept 100\n bif v0.16b, v1.16b, v2.16b\n .endr', DCLOB)
add('bsl_16b_lat', 'lat', 100, setup_fp, '.rept 100\n bsl v0.16b, v1.16b, v3.16b\n .endr', DCLOB)
add('mov16b_lat', 'lat', 100, setup_fp, '.rept 50\n mov v1.16b, v0.16b\n mov v0.16b, v1.16b\n .endr', DCLOB)
# scalar min chain: fcmp + fcsel  (the per-pair kernel's min: left < m ? left : m)
add('fcmp_fcsel_d_lat', 'lat_pair', 50, setup_fp, '.rept 50\n fcmp d0, d3\n fcsel d0, d0, d3, mi\n .endr', DCLOB + ',"cc"')
# fcsel alone with flags fixed (the fcmp is on unrelated regs)
add('fcsel_d_lat', 'lat', 100, setup_fp + 'fcmp d1, d3\n', '.rept 100\n fcsel d0, d0, d3, mi\n .endr', DCLOB + ',"cc"')
# per-pair DP chain: fcmp + fcsel + fadd per cell (left -> fcmp -> fcsel -> fadd -> left)
add('cell_fcmp_fcsel_fadd_d_lat', 'lat_cell', 50, setup_fp, '.rept 50\n fcmp d0, d3\n fcsel d0, d0, d3, mi\n fadd d0, d0, d2\n .endr', DCLOB + ',"cc"')
# per-pair ADTW chain: fadd(pen) + fcmp + fcsel + fadd
add('cell_adtw_d_lat', 'lat_cell', 50, setup_fp, '.rept 50\n fadd d0, d0, d2\n fcmp d0, d3\n fcsel d0, d0, d3, mi\n fadd d0, d0, d2\n .endr', DCLOB + ',"cc"')
# alternative per-pair chain: fminnm + fadd
add('cell_fminnm_fadd_d_lat', 'lat_cell', 50, setup_fp, '.rept 50\n fminnm d0, d0, d3\n fadd d0, d0, d2\n .endr', DCLOB)
add('cell_fmin_fadd_d_lat', 'lat_cell', 50, setup_fp, '.rept 50\n fmin d0, d0, d3\n fadd d0, d0, d2\n .endr', DCLOB)
# vector min chain: fcmgt + bif (lanes kernel: v24 = fcmgt(min1, left); bif left, min1, v24)
add('fcmgt_bif_2d_lat', 'lat_pair', 50, setup_fp, '.rept 50\n fcmgt v4.2d, v3.2d, v0.2d\n bif v0.16b, v3.16b, v4.16b\n .endr', DCLOB)
# vector DP chain: fcmgt + bif + fadd (exact lanes-kernel form)
add('cell_fcmgt_bif_fadd_2d_lat', 'lat_cell', 50, setup_fp, '.rept 50\n fcmgt v4.2d, v3.2d, v0.2d\n bif v0.16b, v3.16b, v4.16b\n fadd v0.2d, v0.2d, v2.2d\n .endr', DCLOB)
add('cell_fcmgt_bif_fadd_4s_lat', 'lat_cell', 50, setup_fp_s, '.rept 50\n fcmgt v4.4s, v3.4s, v0.4s\n bif v0.16b, v3.16b, v4.16b\n fadd v0.4s, v0.4s, v2.4s\n .endr', DCLOB)
add('cell_fminnm_fadd_2d_lat', 'lat_cell', 50, setup_fp, '.rept 50\n fminnm v0.2d, v0.2d, v3.2d\n fadd v0.2d, v0.2d, v2.2d\n .endr', DCLOB)
add('cell_fminnm_fadd_4s_lat', 'lat_cell', 50, setup_fp_s, '.rept 50\n fminnm v0.4s, v0.4s, v3.4s\n fadd v0.4s, v0.4s, v2.4s\n .endr', DCLOB)
# load-to-use latency (pointer chase, integer) and FP load latency through address dependence
add('ldr_x_chase_lat', 'lat', 100, 'mov x10, %[buf]\n str x10, [x10]\n', '.rept 100\n ldr x10, [x10]\n .endr', '"x10","memory"')

# --- throughput (independent chains) ---
def irp(regs, line):
    return '\n'.join(' ' + line.replace('\\r', str(r)).replace('(' + str(r) + '*', '(' + str(r) + '*') for r in regs)
R24 = list(range(0, 24))
add('int_add_tp', 'tp', 10 * 12, '', '.rept 10\n' + irp([9,10,11,12,13,14,15,16,17,19,20,21], 'add x\\r, x\\r, #1') + '\n .endr', ','.join(f'"x{i}"' for i in [9,10,11,12,13,14,15,16,17,19,20,21]))
setup_many = 'fmov d31, #1.0\n dup v31.2d, v31.d[0]\n movi v30.2d, #0\n fmov d29, #2.0\n dup v29.2d, v29.d[0]\n' + ''.join(f'fmov d{i}, #1.0\n dup v{i}.2d, v{i}.d[0]\n' for i in R24)
setup_many_s = 'fmov s31, #1.0\n dup v31.4s, v31.s[0]\n movi v30.2d, #0\n fmov s29, #2.0\n dup v29.4s, v29.s[0]\n' + ''.join(f'fmov s{i}, #1.0\n dup v{i}.4s, v{i}.s[0]\n' for i in R24)
for op, rhs in [('fadd', '30'), ('fmul', '31'), ('fabd', '30'), ('fmin', '29'), ('fminnm', '29')]:
    add(f'{op}_d_tp', 'tp', 4 * 24, setup_many, '.rept 4\n' + irp(R24, f'{op} d\\r, d\\r, d{rhs}') + '\n .endr', DCLOB)
    add(f'{op}_2d_tp', 'tp', 4 * 24, setup_many, '.rept 4\n' + irp(R24, f'{op} v\\r.2d, v\\r.2d, v{rhs}.2d') + '\n .endr', DCLOB)
    add(f'{op}_4s_tp', 'tp', 4 * 24, setup_many_s, '.rept 4\n' + irp(R24, f'{op} v\\r.4s, v\\r.4s, v{rhs}.4s') + '\n .endr', DCLOB)
add('fcmgt_2d_tp', 'tp', 4 * 24, setup_many, '.rept 4\n' + irp(R24, 'fcmgt v\\r.2d, v29.2d, v31.2d') + '\n .endr', DCLOB)
add('bif_16b_tp', 'tp', 4 * 24, setup_many, '.rept 4\n' + irp(R24, 'bif v\\r.16b, v31.16b, v30.16b') + '\n .endr', DCLOB)
add('mov16b_tp', 'tp', 4 * 24, setup_many, '.rept 4\n' + irp(R24, 'mov v\\r.16b, v31.16b') + '\n .endr', DCLOB)
add('fcmp_d_tp', 'tp', 4 * 24, setup_many, '.rept 4\n' + irp(R24, 'fcmp d\\r, d29') + '\n .endr', DCLOB + ',"cc"')
add('fcsel_d_tp', 'tp', 4 * 24, setup_many + 'fcmp d31, d29\n', '.rept 4\n' + irp(R24, 'fcsel d\\r, d\\r, d29, mi') + '\n .endr', DCLOB + ',"cc"')
# mixes: the FP-op mix of the lanes loop (per 2-lane vector: fabd, 2 fcmgt, 2 bit/bif, fadd), all independent
mixregs = list(range(0, 24, 3))  # 8 groups of (a, b, m)
mix = ''.join(f' fabd v{a}.2d, v31.2d, v29.2d\n fcmgt v{m}.2d, v{a}.2d, v29.2d\n bif v{b}.16b, v31.16b, v{m}.16b\n fadd v{a}.2d, v29.2d, v30.2d\n' for a, b, m in [(r, r + 1, r + 2) for r in mixregs])
add('mix_fabd_fcmgt_bif_fadd_tp', 'tp', 4 * 8 * 4, setup_many, '.rept 4\n' + mix + ' .endr', DCLOB)
# loads / stores (L1-resident buffer)
add('ldr_q_tp', 'tp', 4 * 16, 'mov x10, %[buf]\n', '.rept 4\n' + irp(list(range(16)), 'ldr q\\r, [x10, #(\\r*16)]') + '\n .endr', DCLOB + ',"x10","memory"')
add('ldp_q_tp', 'tp', 4 * 8, 'mov x10, %[buf]\n', '.rept 4\n' + irp(list(range(0, 16, 2)), 'ldp q\\r, q31, [x10, #(\\r*32)]') + '\n .endr', DCLOB + ',"x10","memory"')
add('ld1r_2d_tp', 'tp', 4 * 16, 'mov x10, %[buf]\n', '.rept 4\n' + irp(list(range(16)), 'ld1r {v\\r.2d}, [x10]') + '\n .endr', DCLOB + ',"x10","memory"')
add('stp_q_tp', 'tp', 4 * 8, 'mov x10, %[buf]\n', '.rept 4\n' + irp(list(range(8)), 'stp q0, q1, [x10, #(\\r*32)]') + '\n .endr', DCLOB + ',"x10","memory"')
add('str_q_tp', 'tp', 4 * 16, 'mov x10, %[buf]\n', '.rept 4\n' + irp(list(range(16)), 'str q0, [x10, #(\\r*16)]') + '\n .endr', DCLOB + ',"x10","memory"')

out = []
out.append(textwrap.dedent('''\
    // Generated by gen_ubench.py. Build: clang++ -O2 -std=c++20 ubench.cpp -o ubench
    #include <algorithm>
    #include <chrono>
    #include <cstdint>
    #include <cstdio>
    #include <cstring>
    #include <vector>
    alignas(128) static unsigned char g_buf[4096];
    '''))
for name, kind, n, setup, body, clob in tests:
    uses_buf = '%[buf]' in setup
    inputs = ': [buf] "r"(g_buf)' if uses_buf else ': '
    asm = (setup + '1:\n' + body + '\nsubs %[n], %[n], #1\nb.ne 1b\n')
    asm_lines = '\n'.join('      "' + l.replace('\\', '\\\\') + '\\n"' for l in asm.split('\n') if l.strip())
    out.append(f'__attribute__((noinline)) static void t_{name}(uint64_t n) {{\n  asm volatile(\n{asm_lines}\n      : [n] "+r"(n) {inputs} : {clob}, "cc");\n}}\n')
out.append('struct Test { const char *name; const char *kind; int per_iter; void (*f)(uint64_t); };\n')
out.append('static const Test kTests[] = {\n' + ''.join(f'  {{"{name}", "{kind}", {n}, t_{name}}},\n' for name, kind, n, *_ in tests) + '};\n')
out.append(textwrap.dedent('''\
    static double now_ns() {
      return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static double time_ns(void (*f)(uint64_t), uint64_t iters) {
      const double t0 = now_ns();
      f(iters);
      return now_ns() - t0;
    }
    // ns per dependent add, from t_int_add_lat (100 adds per iteration).
    static double ns_per_cycle(uint64_t iters) { return time_ns(t_int_add_lat, iters) / (iters * 100.0); }
    int main(int argc, char **argv) {
      const int reps = argc > 1 ? atoi(argv[1]) : 7;
      const char *only = argc > 2 ? argv[2] : nullptr;
      std::memset(g_buf, 0, sizeof g_buf);
      // warm up: 200 ms of adds so the core ramps to its top clock
      for (int i = 0; i < 20; ++i) ns_per_cycle(1000000);
      std::printf("%-30s %-8s %9s %9s %9s %8s\\n", "test", "kind", "median", "min", "max", "GHz");
      for (const Test &t : kTests) {
        if (only && !std::strstr(t.name, only)) continue;
        // pick iterations so a run is ~40M cycles at the measured rate
        const double c = ns_per_cycle(1000000);
        double probe = time_ns(t.f, 20000) / c;            // cycles for 20000 iterations
        uint64_t iters = std::max<uint64_t>(1000, (uint64_t)(20000 * 4e7 / std::max(probe, 1.0)));
        std::vector<double> v, ghz;
        for (int r = 0; r < reps; ++r) {
          const double c0 = ns_per_cycle(400000);
          const double ns = time_ns(t.f, iters);
          const double c1 = ns_per_cycle(400000);
          const double cyc = ns / ((c0 + c1) / 2);
          const double per = cyc / (double(iters) * t.per_iter);   // cycles per instruction (or per pair/cell unit)
          v.push_back(std::strncmp(t.kind, "tp", 2) == 0 ? 1.0 / per : per);
          ghz.push_back(2.0 / (c0 + c1));
        }
        std::sort(v.begin(), v.end());
        std::sort(ghz.begin(), ghz.end());
        std::printf("%-30s %-8s %9.3f %9.3f %9.3f %8.3f\\n", t.name, t.kind, v[v.size() / 2], v.front(), v.back(), ghz[ghz.size() / 2]);
      }
    }
    '''))
open('ubench.cpp', 'w').write(''.join(out))
print(len(tests), 'tests')
