"""GC: is each fill (triangle) kernel's SASS byte-identical to base's, and what
do the rectangle kernels add? Kernels are keyed by name and template arguments;
head's Pairs argument (E0 = Triangle, E1 = Rectangle) is read off and dropped,
and so is the mangling's substitution index, which the extra argument shifts.
Usage: uv run --no-project python sass_triangle_identical.py base.txt head.txt"""
import re
import sys

FUNC = re.compile(r"^\s*Function : (\S+)\s*$", re.M)


def key(mangled):
    pairs = "Triangle"
    m = re.search(r"LNS0_5PairsE(\d)E", mangled)
    if m:
        pairs = ["Triangle", "Rectangle"][int(m.group(1))]
        mangled = mangled.replace(m.group(0), "")
    mangled = re.sub(r"PS\d_$", "PS_", mangled)
    return pairs, mangled


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    parts = FUNC.split(text)
    out = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        lines = [l.strip() for l in body.splitlines() if l.strip().startswith("/*")]
        out[key(name)] = lines
    return out


base, head = parse(sys.argv[1]), parse(sys.argv[2])
same = 0
for (pairs, name), lines in sorted(head.items()):
    if pairs == "Triangle":
        ok = base.get(("Triangle", name)) == lines
        same += ok
        print(f"triangle {name}: {len(lines)} lines; byte-identical to base {ok}")
    else:
        tri = head.get(("Triangle", name), [])
        print(f"rectangle {name}: {len(lines)} lines (triangle {len(tri)})")
print(f"triangle kernels byte-identical: {same}/{sum(1 for k in base)} (base has {len(base)} kernels)")
