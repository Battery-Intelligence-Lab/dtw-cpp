"""PF-5: loop-vectorize remarks for hot loops, from the real compile commands (no -flto, objects to PF5).

Replays the compile_commands.json entry of each named TU (or, for the probe, the entry of dtwc/core/dtw.cpp
with the probe as source), drops -flto* and -o, adds the remark flags, and prints every loop-vectorize
remark that points into dtwc/ or the probe, one line each with its analysis reason.
"""
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
OUT = Path(__file__).resolve().parent
REMARK = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+remark:\s+(?P<msg>.*?)\s*\[-R(?P<kind>pass|pass-missed|pass-analysis)=(?P<pass>[\w-]+)\]\s*$")
# One -Rpass regex: a second -Rpass= flag replaces the first instead of adding to it.
FLAGS = ["-Rpass=loop-vectorize|slp-vectorizer", "-Rpass-missed=loop-vectorize", "-Rpass-analysis=loop-vectorize",
         "-fno-color-diagnostics"]


def entry_for(db, suffix):
    for e in db:
        if e["file"].replace(chr(92), "/").endswith(suffix):
            return e
    raise SystemExit(f"no compile command for {suffix}")


def run(entry, source, tag, extra=()):
    argv = shlex.split(entry["command"], posix=False) if "command" in entry else list(entry["arguments"])
    cleaned, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "-o":
            skip = True
            continue
        if a.startswith("-flto") or a == "-c" or a.replace(chr(92), "/").endswith(Path(entry["file"]).name):
            continue
        cleaned.append(a.strip('"') if a.startswith('"') and a.endswith('"') and " " not in a else a)
    out = OUT / (f"{tag}.s" if extra else f"{tag}.obj")
    cmd = cleaned + FLAGS + list(extra) + ["-c", str(source), "-o", str(out)]
    p = subprocess.run(cmd, cwd=entry["directory"], capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        print(p.stderr[-3000:])
        raise SystemExit(f"compile failed: {tag}")
    (OUT / f"{tag}.remarks.txt").write_text(p.stderr, encoding="utf-8")
    return p.stderr


def summarise(stderr, tag):
    loops = {}
    for line in stderr.splitlines():
        m = REMARK.match(line.strip())
        if not m:
            continue
        f = m.group("file").replace(chr(92), "/")
        if "/dtwc/" not in f and "vec_probe" not in f and "pf5" not in f:
            continue
        f = re.sub(r".*/dtwc/(\./)?", "dtwc/", f)
        key = (f, int(m.group("line")), int(m.group("col")))
        rec = loops.setdefault(key, {"state": set(), "why": []})
        if m.group("kind") == "pass":
            rec["state"].add("VECTORIZED" if m.group("pass") == "loop-vectorize" else "SLP")
            rec["why"].append(m.group("msg"))
        elif m.group("kind") == "pass-missed":
            rec["state"].add("missed")
        elif m.group("msg") not in rec["why"]:
            rec["why"].append(m.group("msg"))
    print(f"== {tag}: {len(loops)} remark sites")
    for (f, l, c), rec in sorted(loops.items()):
        print(f"{f}:{l}:{c}  {'/'.join(sorted(rec['state'])) or 'analysis'}  | {' | '.join(rec['why'])[:220]}")


def main():
    db = json.loads((ROOT / "build" / "compile_commands.json").read_text(encoding="utf-8"))
    donor = entry_for(db, "dtwc/core/dtw.cpp")
    summarise(run(donor, OUT / "vec_probe.cpp", "vec_probe"), "vec_probe.cpp (flags of dtwc/core/dtw.cpp)")
    run(donor, OUT / "vec_probe.cpp", "vec_probe", extra=("-S", "-masm=intel"))
    for tu in sys.argv[1:]:
        e = entry_for(db, tu)
        summarise(run(e, Path(e["file"]), Path(tu).stem), tu)


if __name__ == "__main__":
    main()
