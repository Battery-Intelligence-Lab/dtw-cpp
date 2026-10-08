"""The loops test_codegen_no_calls sees, listed. The probe compiled exactly as scripts/codegen_report.py --no-calls
compiles it (the donor dtw_dispatch.cpp's command from the RL clang tree's compile_commands.json, LTO off, -S, its own
rebuild_argv), but against one version's probe and headers; then the gate's own count (an innermost loop is a block
clang marks "This Inner Loop Header", in a function whose name holds "dtwc") and, for each such loop of the lanes
probes, its instructions: min and FP ops with their register width, loads, stores, calls, and its role.

usage: gate_loops.py <dtwc dir> <codegen_probe.cpp> <out.s>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "C:/D/git/wt/RL/scripts")
import codegen_report as cr  # noqa: E402  the gate's own regexes and argv builder

TREE = Path("C:/D/git/wt/RL/build")
dtwc, probe, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
entry = next(e for e in json.loads((TREE / "compile_commands.json").read_text())
             if re.search(r"dtwc/core/dtw_dispatch\.cpp", e["file"].replace("\\", "/")))
argv = cr.rebuild_argv(entry, out, probe, ("-S", "-fno-color-diagnostics"))
argv = [a.replace("C:/D/git/wt/RL/dtwc", dtwc.as_posix()) for a in argv]
proc = subprocess.run(argv, cwd=entry.get("directory"), capture_output=True, text=True, errors="replace")
if proc.returncode:
    sys.exit(proc.stderr[-3000:])
asm = out.read_text(errors="replace")
count, calls = cr.loop_calls(asm)

fn, block, cur, loops = "", "", None, {}  # (fn, header) -> instructions
HDR = re.compile(r"in Loop: Header=BB(\w+)")
for n, line in enumerate(asm.splitlines()):
    if m := cr.FUNCTION.match(line):
        fn = m.group(1)
    if m := cr.BLOCK.match(line):  # a block starts; its loop comments may run on to the next lines
        block, cur = m.group(1) or f"line{n}", None
    if "This Inner Loop Header" in line:
        cur = (fn, block)
        loops.setdefault(cur, [])
    elif m := HDR.search(line):
        cur = (fn, m.group(1))
    if cur in loops and line.startswith("	") and not line.strip().startswith((".", "#")):
        loops[cur].append(line.strip().split("#")[0].strip())

MIN = re.compile(r"^v?min(pd|ps|sd|ss)\b")
FPOP = re.compile(r"^v?(add|sub|mul|and|fmadd\w*)(pd|ps|sd|ss)\b")
print(f"gate count (the gate's own loop_calls): inner_loops={count} calls={len(calls)}")
for (f, h), ins in loops.items():
    if "dtwc" not in f or "lanes" not in f:
        continue
    mins = [i for i in ins if MIN.match(i)]
    widths = sorted({w for i in mins + [i for i in ins if FPOP.match(i)] for w in re.findall(r"\b([xyz])mm\d+", i)})
    mem = [i for i in ins if "(%" in i]
    stack = [i for i in mem if "(%rsp)" in i]
    callsin = [i for i in ins if i.startswith("call")]
    packed = any(MIN.match(i) and "ymm" in i for i in ins)
    role = ("DP row step (packed)" if packed and len(mins) >= 4 else
            "DP cells, scalar" if mins else
            "copy / fill (no min)" if mem else "other")
    print(f"{f[:22]:22s} BB{h:8s} {len(ins):4d} insns  mins {len(mins):3d} {','.join(w + 'mm' for w in widths) or '-':10s}"
          f" fp {sum(1 for i in ins if FPOP.match(i)):3d} memory {len(mem):3d} stack {len(stack):3d} calls {len(callsin)}  {role}")
