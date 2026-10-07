"""Maps the loops of an objdump -d listing to their enclosing function (demangled, shortened).
usage: funcs.py <listing> [--min N]   prints: function, then its loops with instruction counts"""
import re
import subprocess
import sys

path = sys.argv[1]
minlen = int(sys.argv[sys.argv.index('--min') + 1]) if '--min' in sys.argv else 10
fn_rx = re.compile(r'^([0-9a-f]+) <(.+)>:$')
ins_rx = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')
funcs, ins = [], []
for line in open(path):
    if m := fn_rx.match(line.strip()):
        funcs.append((int(m.group(1), 16), m.group(2)))
    elif m := ins_rx.match(line):
        ins.append((int(m.group(1), 16), m.group(2), re.sub(r'\s*<[^>]*>', '', m.group(3).split(';')[0]).strip()))
idx = {a: k for k, (a, _, _) in enumerate(ins)}
loops = set()
for k, (a, mn, op) in enumerate(ins):
    if mn in ('b', 'cbz', 'cbnz', 'tbz', 'tbnz') or mn.startswith('b.'):
        t = re.findall(r'0x([0-9a-f]+)', op)
        if t and int(t[-1], 16) <= a and int(t[-1], 16) in idx:
            loops.add((idx[int(t[-1], 16)], k))
inner = [l for l in loops if not any(o != l and o[0] >= l[0] and o[1] <= l[1] for o in loops)]
def owner(addr):
    best = None
    for a, n in funcs:
        if a <= addr:
            best = n
    return best
names = {}
for s, e in sorted(inner):
    if e - s + 1 < minlen:
        continue
    n = owner(ins[s][0])
    d = names.setdefault(n, subprocess.run(['c++filt', '-_', n], capture_output=True, text=True).stdout.strip() or n)
    d = re.sub(r'dtwc::core::', '', d)
    d = re.sub(r'std::__1::', '', d)
    body = ins[s:e + 1]
    mins = sum(1 for b in body if b[1] in ('fcmp', 'fccmp'))
    print(f'{ins[s][0]:6x}-{ins[e][0]:6x} {e - s + 1:3d} insns  fcmp/fccmp {mins:2d}  in {d[:150]}')
