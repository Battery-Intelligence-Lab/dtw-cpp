"""Unique clang warnings from a build log: 'file:line:col: warning: msg [-Wflag]'.

usage: uv run --no-project python warnings.py <build.log> <out.txt>
Writes one line per unique (file, line, col, flag, msg), sorted; prints counts per flag.
Header warnings repeat per TU, so the set is deduplicated.
"""
import collections
import re
import sys

pat = re.compile(r"^(?P<file>[A-Za-z]:[^:]+|[^:\s][^:]*):(?P<line>\d+):(?P<col>\d+): warning: (?P<msg>.*?) \[(?P<flag>-W[^\]]+)\]\s*$")
seen = set()
for raw in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = pat.match(raw.strip())
    if not m:
        continue
    f = m["file"].replace("\\", "/")
    f = re.sub(r"^.*?/(dtwc/|python/|bindings/|tests/)", r"\1", f)
    seen.add((f, int(m["line"]), int(m["col"]), m["flag"], m["msg"]))
rows = sorted(seen)
with open(sys.argv[2], "w", encoding="utf-8") as out:
    for f, line, col, flag, msg in rows:
        out.write(f"{f}:{line}:{col} {flag} {msg}\n")
by_flag = collections.Counter(r[3] for r in rows)
print(f"unique warnings: {len(rows)}")
for flag, n in by_flag.most_common():
    print(f"  {n:5d} {flag}")
