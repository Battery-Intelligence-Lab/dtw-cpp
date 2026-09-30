"""Innermost loops of a clang -S -masm=intel listing, per function.

For each function whose simplified name matches a filter, and each innermost
loop (clang's '=>This Inner Loop Header' annotation), report: instruction
count, calls, spill/reload annotations, movsxd, dword/qword memory loads.
usage: uv run --no-project python loops.py <file.s> [name-substring ...]
"""
import re
import sys

path = sys.argv[1]
filters = sys.argv[2:]
lines = open(path, encoding="utf-8", errors="replace").read().split("\n")


def simplify(sym):
    s = sym.strip('"')
    parts = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:<lambda_\d+>)?", s)
    keep = [p for p in parts if p not in {"dtwc", "std", "core", "algorithms", "detail", "Problem",
                                          "vector", "allocator", "V", "AEAV", "AEBV", "YA", "YAX", "Z",
                                          "QEAA", "QEBA", "AUClusteringResult", "H", "N", "_J", "_K",
                                          "omp_outlined", "Zh", "HA", "A0xF0EE7021"}]
    base = "/".join(dict.fromkeys(keep))[:120]
    if "omp_outlined" in s:
        base += " [omp]"
    return base


funcs = []  # (name, start, end)
cur = None
for i, l in enumerate(lines):
    m = re.match(r'^("?[?$@A-Za-z_][^:]*"?):\s*(#.*)?$', l)
    if m and not l.startswith(".") and not m.group(1).startswith('".L') and "@" in m.group(1) \
            and not m.group(1).startswith('"?dtor$') and not m.group(1).startswith('"?catch$'):
        if cur:
            funcs.append((cur[0], cur[1], i))
        cur = (m.group(1), i)
if cur:
    funcs.append((cur[0], cur[1], len(lines)))

label_re = re.compile(r"^(\.LBB\d+_\d+):\s*(#.*)?$")
for fname, a, b in funcs:
    name = simplify(fname)
    if filters and not any(f in fname for f in filters):
        continue
    # blocks: label -> (annotation text incl. following comment lines, instruction lines)
    blocks = []
    cur_label, cur_ann, cur_ins = None, "", []
    for l in lines[a:b]:
        m = label_re.match(l)
        mb = re.match(r"^# %bb\.\d+:\s*(#.*)?$", l)
        if m or mb:
            if cur_label is not None:
                blocks.append((cur_label, cur_ann, cur_ins))
            cur_label = m.group(1) if m else l.split(":")[0]
            cur_ann = (m.group(2) if m else mb.group(1)) or ""
            cur_ins = []
            continue
        if cur_label is not None:
            if l.strip().startswith("#"):
                cur_ann += " " + l.strip()
            elif l.startswith("\t") and not l.strip().startswith("."):
                cur_ins.append(l.strip())
    if cur_label is not None:
        blocks.append((cur_label, cur_ann, cur_ins))
    headers = [lab for lab, ann, _ in blocks if "This Inner Loop Header" in ann]
    if not headers:
        continue
    print(f"== {name}")
    for h in headers:
        hid = h.replace(".L", "")
        body = []
        for lab, ann, ins in blocks:
            if lab == h or re.search(rf"in Loop: Header={re.escape(hid)} Depth", ann):
                body += ins
        calls = [re.sub(r"^call\s+", "", x) for x in body if x.startswith("call")]
        spills = sum(1 for x in body if "Spill" in x)
        reloads = sum(1 for x in body if "Reload" in x)
        movsxd = sum(1 for x in body if x.startswith("movsxd"))
        dword = sum(1 for x in body if "dword ptr [" in x)
        qword = sum(1 for x in body if "qword ptr [" in x)
        callnames = ",".join(simplify(c)[:40] for c in calls)
        print(f"   {h:12s} ins={len(body):3d} calls={len(calls)} spill={spills} reload={reloads} "
              f"movsxd={movsxd} dword={dword} qword={qword} {callnames}")
