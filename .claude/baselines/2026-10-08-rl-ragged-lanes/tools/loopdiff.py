"""Compare the min loops of two loops.py --all listings, function by function (by kind and lambda), with register
names and labels normalised: prints, per loop, whether the instruction sequences are the same up to register
renaming. usage: loopdiff.py <loops a> <loops b>"""
import re
import sys


def loops(path):
    out, key, cur = {}, None, None
    for ln in open(path).read().splitlines():
        if ln.startswith("== "):
            kind = ln.split()[1]
            lam = re.search(r"lambda_(\d)", ln)
            key = (kind, lam.group(1) if lam else "?")
            continue
        if ln.startswith("  loop"):
            cur = []
            out.setdefault(key, []).append((ln.strip(), cur))
            continue
        if ln.startswith("      ") and cur is not None:
            cur.append(ln.strip())
    return out


def norm(ins):
    ins = re.sub(r"\b[xyz]mm\d+\b", "V", ins)
    ins = re.sub(r"\b(r\d+[dwb]?|e?[re]?[abcd]x|[re]?[sd]il?|[re]?[sb]pl?|[abcd]l)\b", "R", ins)
    ins = re.sub(r"\.L\w+|\$LN\d+@\w+", "L", ins)
    ins = re.sub(r"\btv\d+", "tv", ins)
    ins = re.sub(r"-?\d+\[", "[", ins)
    return ins


a, b = loops(sys.argv[1]), loops(sys.argv[2])
for k in sorted(set(a) | set(b)):
    la, lb = a.get(k, []), b.get(k, [])
    for i in range(max(len(la), len(lb))):
        sa = la[i] if i < len(la) else ("-", [])
        sb = lb[i] if i < len(lb) else ("-", [])
        same = [norm(x) for x in sa[1]] == [norm(x) for x in sb[1]]
        ms = sorted(map(norm, sa[1])) == sorted(map(norm, sb[1]))
        print(f"{k[0]} lambda_{k[1]} loop {i}: {sa[0][:60]:60s} | {sb[0][:60]:60s} | "
              f"{'same sequence' if same else ('same multiset' if ms else 'DIFFERENT')}")
