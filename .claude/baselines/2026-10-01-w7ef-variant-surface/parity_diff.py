"""Compare two parity outputs line by line: lines that differ, by kind, and the max
relative change of every hex-float field. Usage: parity_diff.py <base.txt> <head.txt>"""
import sys

base = open(sys.argv[1]).read().splitlines()
head = open(sys.argv[2]).read().splitlines()
assert len(base) == len(head), (len(base), len(head))
worst = {}
changed = {}
for a, b in zip(base, head):
    if a == b:
        continue
    ta, tb = a.split(), b.split()
    kind = " ".join(t for t in ta if not t.startswith(("0x", "-0x", "n=", "m=", "gamma=", "band=")))
    changed[kind] = changed.get(kind, 0) + 1
    for x, y in zip(ta, tb):
        if x == y or not x.lstrip("-").startswith("0x"):
            continue
        fx, fy = float.fromhex(x), float.fromhex(y)
        rel = abs(fx - fy) / max(abs(fx), abs(fy))
        worst[kind] = max(worst.get(kind, 0.0), rel)
print("lines:", len(base), "changed:", sum(changed.values()))
for kind in sorted(changed):
    print(f"  {kind}: {changed[kind]} lines, max relative change {worst.get(kind, 0.0):.3e}")
