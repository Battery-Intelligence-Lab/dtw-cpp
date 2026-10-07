"""Where each probe's per-pair DP loops landed: for every Standard-cell per-pair loop (a scalar fcmp+fcsel+fadd
loop in a dtw_kernel_* / detail::dtw_* function, L1 = fabd, Sq = fsub+fmul), its start address, the start mod 64,
the number of 64-byte boundaries inside it, and a digest of its instructions (registers and targets kept) so the
same code can be recognised across placements.
usage: placement.py <binary> [<binary> ...]       (runs objdump -d itself)"""
import bisect
import collections
import hashlib
import re
import subprocess
import sys

fn_rx = re.compile(r'^([0-9a-f]+) <(.+)>:$')
ins_rx = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')


def loops_of(binary):
    dis = subprocess.run(['objdump', '-d', '--no-show-raw-insn', binary], capture_output=True, text=True).stdout
    funcs, ins = [], []
    for line in dis.splitlines():
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
    inner = sorted(l for l in loops if not any(o != l and o[0] >= l[0] and o[1] <= l[1] for o in loops))
    starts = [a for a, _ in funcs]
    demangled = {}
    out = []
    for s, e in inner:
        body = ins[s:e + 1]
        c = collections.Counter(b[1] for b in body)
        if not (c['fcsel'] and c['fadd'] and (c['fabd'] or c['fmul'])):
            continue
        k = bisect.bisect_right(starts, body[0][0]) - 1
        name = funcs[k][1]
        if name not in demangled:
            demangled[name] = subprocess.run(['c++filt', '-_', name], capture_output=True, text=True).stdout.strip()
        d = demangled[name]
        m = re.search(r'(dtw_kernel_linear|dtw_kernel_banded|detail::dtw_linear<(?:true|false)|detail::dtw_banded<(?:true|false))'
                      r'.*?(double|float), dtwc::core::Span(L1|SquaredL2)Cost<(?:double|float)>, dtwc::core::StandardCell>', d)
        if not m:
            continue
        kern = m.group(1).replace('detail::', '').replace('dtw_kernel_', 'dtw_')
        start, end = body[0][0], body[-1][0] + 4
        text = [f'{b[1]} {re.sub(r"0x[0-9a-f]+", "T", b[2])}' for b in body]
        digest = hashlib.sha1('\n'.join(text).encode()).hexdigest()[:10]
        regs = {}
        def ren(mo):  # registers renamed by first appearance, per class (x/w, d/s/q/v)
            cls = 'g' if mo.group(1) in 'xw' else 'f'
            key = (cls, mo.group(2))
            regs.setdefault(key, f'{cls}{sum(1 for k in regs if k[0] == cls)}')
            return mo.group(1) + regs[key]
        norm = hashlib.sha1('\n'.join(re.sub(r'\b([xwdsqv])(\d+)\b', ren, t) for t in text).encode()).hexdigest()[:10]
        out.append((f'{kern} {m.group(2)} {"L1" if m.group(3) == "L1" else "Sq"}', start, len(body), start % 64,
                    (end - 1) // 64 - start // 64, digest, norm))
    return out


for b in sys.argv[1:]:
    print(f'== {b}')
    for kern, start, n, mod, crossings, digest, norm in loops_of(b):
        print(f'  {kern:28s} loop {start:x} {n:3d} insns  start mod 64 = {mod:2d}  64-byte boundaries inside {crossings}  '
              f'code {digest}  registers renamed {norm}')
