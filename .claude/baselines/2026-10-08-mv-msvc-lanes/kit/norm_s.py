"""norm_s.py <a.s> <b.s>: compare two clang/GCC -S listings of dtw_lanes.cpp instruction for instruction, after
normalising what names alone change: the anonymous-namespace hash clang derives from the file path (?A0x...),
the Cell's name in mangled symbols (StandardCell -> LanesCell when the lanes' Cell becomes its own struct), and
comments, .file and source-path lines. Prints 'identical' or the first differing lines."""
import re, sys, difflib
def norm(path):
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.split("#", 1)[0].rstrip() if not ln.lstrip().startswith('"') else ln.rstrip()
        if not ln or ln.lstrip().startswith((".file", ".ident", "#")):
            continue
        ln = re.sub(r"\?A0x[0-9A-Fa-f]{8}", "?A0xHASH", ln)
        ln = ln.replace("UStandardCell@", "ULanesCell@").replace("StandardCell", "LanesCell")
        ln = re.sub(r"N4dtwc4core12StandardCellE", "N4dtwc4core9LanesCellE", ln)
        out.append(ln)
    return out
a, b = norm(sys.argv[1]), norm(sys.argv[2])
if a == b:
    print(f"identical ({len(a)} normalised lines)")
else:
    d = list(difflib.unified_diff(a, b, lineterm="", n=2))
    print(f"DIFFER: {sum(1 for x in d if x.startswith(('+', '-')) and not x.startswith(('+++', '---')))} changed lines")
    print("\n".join(x[:160] for x in d[:80]))
