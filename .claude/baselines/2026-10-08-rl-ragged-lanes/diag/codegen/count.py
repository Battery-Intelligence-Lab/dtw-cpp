"""Innermost loops (a backward branch to a label with no other loop inside) per function of a clang -S listing, for
the functions whose name holds `probe`. usage: count.py <listing.s>"""
import re
import sys

funcs, cur = [], None
for ln in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.match(r'^("[^"]+"|[A-Za-z_?$@][^\s:]*):', ln)
    if m and not m.group(1).startswith(("$", "__real", "__xmm", "__ymm")):
        cur = (m.group(1).strip('"'), [])
        funcs.append(cur)
        continue
    m = re.match(r"^(\.LBB\d+_\d+):", ln)
    if m and cur:
        cur[1].append((m.group(1), None))
    elif cur and ln.startswith("\t") and not ln.strip().startswith((".", "#")):
        cur[1].append((None, ln.strip()))
for name, body in funcs:
    if "probe" not in name:
        continue
    pos = {lab: k for k, (lab, _) in enumerate(body) if lab}
    loops = []
    for k, (_, ins) in enumerate(body):
        m = re.match(r"^j\w+\s+(\.LBB\d+_\d+)", ins or "")
        if m and m.group(1) in pos and pos[m.group(1)] < k:
            loops.append((pos[m.group(1)], k))
    inner = [l for l in loops if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
    print(f"{name[:40]:40s} innermost loops {len(inner)}")
