# fptp2.cpp: FP/SIMD throughput with 24 accumulators and the second source rotating over 8 registers
# (v24-v31), so no one register is read by every instruction; single ops and the lanes-loop mixes.
ops = {
 'fadd.2d': 'fadd v{r}.2d, v{r}.2d, v{c}.2d', 'fminnm.2d': 'fminnm v{r}.2d, v{r}.2d, v{c}.2d',
 'fabd.2d': 'fabd v{r}.2d, v{r}.2d, v{c}.2d', 'fmul.2d': 'fmul v{r}.2d, v{r}.2d, v{c}.2d',
 'fcmgt.2d': 'fcmgt v{r}.2d, v{c}.2d, v{c2}.2d', 'bif.16b': 'bif v{r}.16b, v{c}.16b, v{c2}.16b',
 'fadd.4s': 'fadd v{r}.4s, v{r}.4s, v{c}.4s', 'fminnm.d': 'fminnm d{r}, d{r}, d{c}', 'fadd.d': 'fadd d{r}, d{r}, d{c}',
 'fcmp.d': 'fcmp d{r}, d{c}', 'fcsel.d': 'fcsel d{r}, d{r}, d{c}, mi',
}
mixes = [(n, [n]) for n in ops] + [
 ('mix fabd+2fminnm+fadd', ['fabd.2d', 'fminnm.2d', 'fminnm.2d', 'fadd.2d']),
 ('mix fabd+2fcmgt+2bif+fadd', ['fabd.2d', 'fcmgt.2d', 'fcmgt.2d', 'bif.16b', 'bif.16b', 'fadd.2d']),
 ('mix fcmp+fcsel', ['fcmp.d', 'fcsel.d']),
]
R, N = 24, 96
out = ['#include <algorithm>\n#include <chrono>\n#include <cstdint>\n#include <cstdio>\n#include <vector>\n',
       'static double now_ns() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }\n',
       '__attribute__((noinline)) static void add_chain(uint64_t n) { asm volatile("1:\\n .rept 100\\n add x9, x9, #1\\n .endr\\n subs %[n], %[n], #1\\n b.ne 1b\\n" : [n] "+r"(n) : : "x9", "cc"); }\n',
       'static double ns_per_cycle() { const uint64_t it = 300000; double t = now_ns(); add_chain(it); return (now_ns() - t) / (it * 100.0); }\n']
clob = ','.join(f'"v{i}"' for i in range(32)) + ',"cc"'
setup = ''.join(f'"fmov d{i}, #1.0\\n dup v{i}.2d, v{i}.d[0]\\n"' for i in range(24)) + ''.join(f'"fmov d{i}, #{v}\\n dup v{i}.2d, v{i}.d[0]\\n"' for i, v in zip(range(24, 32), ['0.5', '0.25', '2.0', '1.5', '0.75', '1.25', '3.0', '0.125']))
for t, (name, seq) in enumerate(mixes):
    lines = []
    for k in range(N):
        lines.append(ops[seq[k % len(seq)]].format(r=k % R, c=24 + (k % 8), c2=24 + ((k + 3) % 8)))
    b = ''.join(f'"{l}\\n"' for l in lines)
    out.append(f'__attribute__((noinline)) static void t{t}(uint64_t n) {{ asm volatile({setup} "1:\\n" {b} "subs %[n], %[n], #1\\n b.ne 1b\\n" : [n] "+r"(n) : : {clob}); }}\n')
out.append('int main() { for (int i = 0; i < 40; ++i) ns_per_cycle();\n struct T { const char *n; void (*f)(uint64_t); } ts[] = {' + ','.join(f'{{"{n}", t{i}}}' for i, (n, s) in enumerate(mixes)) + '};\n')
out.append(f''' for (auto &t : ts) {{ std::vector<double> v; const uint64_t it = 250000;
   for (int r = 0; r < 7; ++r) {{ double c0 = ns_per_cycle(); double t0 = now_ns(); t.f(it); double ns = now_ns() - t0; double c1 = ns_per_cycle(); v.push_back(it * {N}.0 / (ns / ((c0 + c1) / 2))); }}
   std::sort(v.begin(), v.end()); std::printf("%-28s instructions/cycle median %.3f max %.3f\\n", t.n, v[3], v[6]); }} }}
''')
open('fptp2.cpp', 'w').write(''.join(out))
