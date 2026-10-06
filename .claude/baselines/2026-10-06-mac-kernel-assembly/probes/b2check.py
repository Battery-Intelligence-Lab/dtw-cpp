# B2: every innermost loop whose body holds a DP cell (a min - fcsel, bif/bit/bsl or fminnm - feeding an
# fadd) is listed with any bl/blr/br/brk/b-to-outside inside it. Usage: b2check.py <full listing>
import re, sys
rx = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')
ins = []
for l in open(sys.argv[1]):
    m = rx.match(l)
    if m:
        ins.append((int(m.group(1), 16), m.group(2), re.sub(r'\s*<[^>]*>', '', m.group(3).split(';')[0]).strip()))
idx = {a: k for k, (a, _, _) in enumerate(ins)}
loops = set()
for k, (a, mn, op) in enumerate(ins):
    if mn in ('b', 'cbz', 'cbnz', 'tbz', 'tbnz') or mn.startswith('b.'):
        t = re.findall(r'0x([0-9a-f]+)', op)
        if t and int(t[-1], 16) <= a and int(t[-1], 16) in idx and k - idx[int(t[-1], 16)] < 400:
            loops.add((idx[int(t[-1], 16)], k))
loops = sorted(loops)
n_dp = n_bad = 0
for s, e in loops:
    if any(o != (s, e) and o[0] >= s and o[1] <= e for o in loops):
        continue
    body = ins[s:e + 1]
    mns = [b[1] for b in body]
    has_min = any(m.startswith(('fcsel', 'bif', 'bit.', 'bsl', 'fminnm')) for m in mns)
    has_add = any(m.startswith('fadd') for m in mns)
    if not (has_min and has_add):
        continue
    n_dp += 1
    lo, hi = body[0][0], body[-1][0]
    bad = []
    for a, m, op in body:
        if m in ('bl', 'blr', 'br', 'brk', 'blraa', 'braa'):
            bad.append(f'{a:x} {m} {op}')
        elif m == 'b' or m.startswith('b.') or m in ('cbz', 'cbnz', 'tbz', 'tbnz'):
            t = re.findall(r'0x([0-9a-f]+)', op)
            if t and not (lo <= int(t[-1], 16) <= hi + 4):
                bad.append(f'{a:x} {m} {op} (exits loop)')
    if bad:
        n_bad += 1
        print(f'LOOP {lo:x}-{hi:x} ({len(body)} insns): ' + '; '.join(bad))
print(f'{sys.argv[1]}: {n_dp} innermost DP-cell loops; {n_bad} with a call, trap or exit branch (listed above)')
