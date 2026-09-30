"""Emit Intel-syntax assembly for the W11a hot TUs with the build's own flags, minus LTO.

usage: uv run --no-project python emit_asm.py <build_dir> <out_dir> [tu-suffix ...]

The compile_commands.json string is what Ninja hands to CreateProcess, so it is
run as a string (no re-quoting); only -flto=* and the -o/-c pair change.
"""
import json
import pathlib
import re
import subprocess
import sys

HOT = [
    "dtwc/algorithms/fast_pam.cpp",
    "dtwc/algorithms/fast_clara.cpp",
    "dtwc/algorithms/one_batch_pam.cpp",
    "dtwc/algorithms/tadpole.cpp",
    "dtwc/algorithms/hierarchical.cpp",
    "dtwc/Problem.cpp",
    "dtwc/scores.cpp",
]

build = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2]).resolve()
wanted = sys.argv[3:] or HOT
out.mkdir(parents=True, exist_ok=True)
entries = json.loads((build / "compile_commands.json").read_text())
for suffix in wanted:
    hits = [e for e in entries
            if e["file"].replace("\\", "/").endswith(suffix) and "dtwc++.dir" in e.get("output", "")]
    if not hits:
        print(f"MISSING {suffix}")
        continue
    e = hits[0]
    target = out / (pathlib.Path(suffix).stem + ".s")
    cmd = re.sub(r"\s-flto(=\S+)?", "", e["command"])
    cmd, n = re.subn(r"\s-o\s+\S+\s+-c\s+(\S+)$",
                     lambda m: f" -S -masm=intel -o {target} -c {m.group(1)}", cmd)
    assert n == 1, cmd[-300:]
    r = subprocess.run(cmd, cwd=e["directory"], capture_output=True, text=True)
    print(f"{suffix}: exit {r.returncode}")
    if r.returncode:
        print(r.stderr[-2000:])
