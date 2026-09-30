"""Per kernel, are two cuobjdump -sass dumps byte-identical (instructions and
encodings, addresses included)? Kernels are matched by kernel name and template
arguments; the wavefront's second template argument is read as a mode, whether it
is spelled WavefrontBuffers (Shared = 0, Global = 1) or Wavefront (Preload = 0,
Shared = 1, Global = 2), and a wavefront without it (a7d3a8f) is the Shared one.
Usage: sass_identical.py base.txt head.txt"""
import re
import sys

FUNC = re.compile(r"^\s*Function : (\S+)\s*$", re.M)
MODES = {"16WavefrontBuffers": ["Shared", "Global"], "9Wavefront": ["Preload", "Shared", "Global"]}


def key(mangled):
    m = re.match(r"_ZN4dtwc4cuda\d+(\w+?)I([fd])(Li\d+E|LNS0_(\d+\w+?)E(\d)E)?E", mangled)
    if not m:
        return mangled
    name, t, extra, enum, value = m.groups()
    if enum is not None:
        return f"{name}<{t},{MODES[enum][int(value)]}>"
    if name == "dtw_wavefront_kernel":
        return f"{name}<{t},Shared>"
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
