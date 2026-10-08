"""warnings.py <build log> [<build log>]: the distinct compiler warnings of a build log (clang 'warning:' lines, cl
'warning C....' / 'warning STL....' lines), paths made relative to the worktree; with two logs, those of the second
not in the first."""
import re, sys
def ws(path):
    out = set()
    for ln in open(path, encoding="utf-8", errors="replace"):
        if re.search(r"warning( [A-Z]+\d+)?:|warning [A-Z]+\d+", ln):
            s = ln.strip().replace("\\", "/")
            s = re.sub(r"(?i)^.*?/(dtwc|tests|benchmarks|examples|_deps|build[^/]*)/", r"\1/", s)
            out.add(s)
    return out
a = ws(sys.argv[1])
if len(sys.argv) == 2:
    print(f"{len(a)} distinct warnings"); [print(" ", w[:200]) for w in sorted(a)]
else:
    b = ws(sys.argv[2])
    new = sorted(b - a)
    print(f"{len(a)} distinct in base, {len(b)} in candidate, {len(new)} new")
    for w in new: print("  NEW:", w[:220])
