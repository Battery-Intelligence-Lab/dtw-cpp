"""Per kernel, are two cuobjdump -sass dumps byte-identical (instructions and
encodings, addresses included)? Kernels are matched by kernel name and the
leading template argument (T, TILE_W); a WavefrontBuffers argument of the head
is matched to the base kernel without it when it is Shared (E0).
Usage: sass_identical.py base.txt head.txt"""
import re
import sys

FUNC = re.compile(r"^\s*Function : (\S+)\s*$", re.M)


def key(mangled):
    m = re.match(r"_ZN4dtwc4cuda\d+(\w+?)I([fd])(Li\d+E|LNS0_16WavefrontBuffersE(\d)E)?E", mangled)
    if not m:
        return mangled
    name, t, extra, buffers = m.groups()
    if buffers is not None:
        return f"{name}<{t}>" if buffers == "0" else f"{name}<{t},Global>"
    return f"{name}<{t}{',' + extra if extra else ''}>"


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    parts = FUNC.split(text)
    out = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        lines = [l.strip() for l in body.splitlines() if l.strip().startswith("/*")]
        out[key(name)] = lines
    return out


a, b = parse(sys.argv[1]), parse(sys.argv[2])
for k in sorted(set(a) | set(b)):
    if k not in a or k not in b:
        print(f"{k}: only in {'base' if k in a else 'head'} ({len(a.get(k, b.get(k)))} lines)")
        continue
    same = a[k] == b[k]
    diff = sum(1 for x, y in zip(a[k], b[k]) if x != y) + abs(len(a[k]) - len(b[k]))
    print(f"{k}: {len(a[k])} vs {len(b[k])} lines; byte-identical {same}; differing lines {diff}")
