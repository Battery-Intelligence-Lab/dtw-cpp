"""The innermost DP loops of each kernel, base against head.

A loop is a backward branch and the instructions from its target to it; it is
innermost when no other loop lies inside it, and a DP loop when it holds a DTW
min (FMNMX in FP32, DSETP.MIN in FP64) and no global store (the backward jumps
of the warp-divergence fix-up blocks at a kernel's end re-enter a loop from
behind the store, and are not loops of the program). Instructions compare as
shapes: register names replaced by their class, .reuse (an operand-cache hint)
and branch displacements dropped. Per DP loop, in program order: length in both
dumps, whether the shapes are the same sequence, and the multiset difference.
"""
import collections
import difflib
import re
import sys

sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from sass_shape_diff import parse, relative_targets, shape  # noqa: E402

BRANCH = re.compile(r"\bBRA\b[^<]*<(-\d+)>")


def loop_shape(s):
    return re.sub(r"<[+-]\d+>", "<t>", shape(s)).replace(".reuse", "")


def dp_loops(insns):
    text = relative_targets(insns)
    loops = []
    for i, s in enumerate(text):
        m = BRANCH.search(s)
        if m:
            loops.append((i + int(m.group(1)), i))
    inner = [(s, e) for (s, e) in loops
             if not any((s2, e2) != (s, e) and s <= s2 and e2 <= e for (s2, e2) in loops)]
    out = []
    for s, e in inner:
        body = [loop_shape(x) for x in text[s:e + 1]]
        if any(("FMNMX" in x) or ("DSETP.MIN" in x) for x in body) and not any(x.startswith(("STG", "@P STG", "@!P STG")) for x in body):
            out.append(body)
    return out


def main(a_path, b_path):
    a, b = parse(a_path), parse(b_path)
    totals = [0, 0]
    for key in sorted(a):
        la, lb = dp_loops(a[key]), dp_loops(b[key])
        print(f"== {key[0]}<{key[1]}>: {len(la)} DP loops vs {len(lb)}")
        for ba, bb in zip(la, lb):
            ca, cb = collections.Counter(ba), collections.Counter(bb)
            totals[0] += len(ba)
            totals[1] += len(bb)
            line = f"   {len(ba)} vs {len(bb)} instructions; same sequence {ba == bb}"
            if ca != cb:
                line += f"; head adds {dict(cb - ca)}; head drops {dict(ca - cb)}"
            elif ba != bb:
                line += "; same instructions, reordered"
            print(line)
    print(f"total DP-loop instructions: {totals[0]} vs {totals[1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
