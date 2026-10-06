# Usage: loops.py <objdump -d --no-show-raw-insn listing> [--all] [--min N]
# Lists backward-branch loops; prints the body of each innermost loop.
import sys, re
path = sys.argv[1]
show_all = '--all' in sys.argv
minlen = int(sys.argv[sys.argv.index('--min') + 1]) if '--min' in sys.argv else 1
ins = []
rx = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')
for line in open(path):
    m = rx.match(line)
    if m:
        ins.append((int(m.group(1), 16), m.group(2), re.sub(r"\s*<[^>]*>", "", m.group(3).split(";")[0]).strip()))
idx = {a: k for k, (a, _, _) in enumerate(ins)}
loops = []
for k, (a, mn, op) in enumerate(ins):
    if mn in ('b', 'cbz', 'cbnz', 'tbz', 'tbnz') or mn.startswith('b.'):
        t = re.findall(r'0x([0-9a-f]+)', op)
        if not t:
            continue
        tgt = int(t[-1], 16)
        if tgt <= a and tgt in idx:
            loops.append((idx[tgt], k))
loops = sorted(set(loops))
def inner(l):
    return not any((o != l and o[0] >= l[0] and o[1] <= l[1]) for o in loops)
for s, e in loops:
    body = ins[s:e + 1]
    n = len(body)
    if n < minlen:
        continue
    mns = [b[1] for b in body]
    vec = sum(1 for b in body if re.search(r'\.(2d|4s|16b|8h|2s|8b|4h)\b', b[1]) or re.search(r'v\d+\.(2d|4s|16b|8h)', b[2]) or re.search(r'\bq\d+\b', b[2]))
    calls = [f'{b[0]:x} {b[1]} {b[2]}' for b in body if b[1] in ('bl', 'blr', 'brk')]
    tag = 'INNER' if inner((s, e)) else 'outer'
    print(f'{tag} loop {ins[s][0]:x}-{ins[e][0]:x}: {n} insns, vector-operand insns {vec}, calls/traps {calls}')
    if tag == 'INNER' or show_all:
        for b in body:
            print(f'    {b[0]:x}: {b[1]:8s} {b[2]}')
