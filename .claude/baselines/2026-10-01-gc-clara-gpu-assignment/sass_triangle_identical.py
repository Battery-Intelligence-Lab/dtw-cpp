"""GC: is each fill (triangle) kernel's SASS byte-identical to base's, and what
do the rectangle kernels add? Kernels are keyed by architecture (cuobjdump's
"arch = sm_NN" header of each ELF, so a fat binary keeps every architecture's
code apart) and by name and template arguments; head's Pairs argument (E0 =
Triangle, E1 = Rectangle) is read off and dropped, and so is the mangling's
substitution index, which the extra argument shifts.
Usage: uv run --no-project python sass_triangle_identical.py base.txt head.txt"""
import re
import sys

ARCH = re.compile(r"^\s*arch = (sm_\d+)\s*$")
FUNC = re.compile(r"^\s*Function : (\S+)\s*$")


def key(mangled):
    pairs = "Triangle"
    m = re.search(r"LNS0_5PairsE(\d)E", mangled)
    if m:
        pairs = ["Triangle", "Rectangle"][int(m.group(1))]
        mangled = mangled.replace(m.group(0), "")
    mangled = re.sub(r"PS\d_$", "PS_", mangled)
    return pairs, mangled


def parse(path):
    """{(arch, pairs, kernel): [instruction lines]}"""
    out, arch, current = {}, "?", None
    for line in open(path, encoding="utf-8", errors="replace"):
        if m := ARCH.match(line):
            arch, current = m.group(1), None
        elif m := FUNC.match(line):
            pairs, name = key(m.group(1))
            current = out.setdefault((arch, pairs, name), [])
        elif current is not None and line.strip().startswith("/*"):
            current.append(line.strip())
    return out


base, head = parse(sys.argv[1]), parse(sys.argv[2])
same = 0
for (arch, pairs, name), lines in sorted(head.items()):
    if pairs == "Triangle":
        ok = base.get((arch, "Triangle", name)) == lines
        same += ok
        print(f"{arch} triangle {name}: {len(lines)} lines; byte-identical to base {ok}")
    else:
        tri = head.get((arch, "Triangle", name), [])
        print(f"{arch} rectangle {name}: {len(lines)} lines (triangle {len(tri)})")
print(f"triangle kernels byte-identical: {same}/{len(base)} (base: {len(base)} kernels in "
      f"{len({a for a, _, _ in base})} architecture(s))")
