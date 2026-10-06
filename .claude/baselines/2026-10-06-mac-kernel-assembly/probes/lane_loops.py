# Usage: lane_loops.py <full objdump listing> <regex>  -> innermost loops whose body matches regex, bodies printed
import sys, re
rx = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')
ins = []
for line in open(sys.argv[1]):
    m = rx.match(line)
    if m:
        ins.append((int(m.group(1), 16), m.group(2), re.sub(r'\s*<[^>]*>', '', m.group(3).split(';')[0]).strip()))
idx = {a: k for k, (a, _, _) in enumerate(ins)}
loops = []
for k, (a, mn, op) in enumerate(ins):
    if mn in ('b', 'cbz', 'cbnz', 'tbz', 'tbnz') or mn.startswith('b.'):
        t = re.findall(r'0x([0-9a-f]+)', op)
        if t:
            tgt = int(t[-1], 16)
            if tgt <= a and tgt in idx and k - idx[tgt] < 400:
                loops.append((idx[tgt], k))
loops = sorted(set(loops))
pat = re.compile(sys.argv[2])
for s, e in loops:
    if any(o != (s, e) and o[0] >= s and o[1] <= e for o in loops):
        continue
    body = ins[s:e + 1]
    if not any(pat.search(b[1] + ' ' + b[2]) for b in body):
        continue
    calls = [f'{b[0]:x} {b[1]}' for b in body if b[1] in ('bl', 'blr', 'brk', 'br')]
    print(f'LOOP {ins[s][0]:x}-{ins[e][0]:x}: {len(body)} insns, calls/traps {calls}')
    for b in body:
        print(f'    {b[1]:9s} {b[2]}')
