"""The packed DP loops of the lanes kernel in a linked executable (no symbols needed): llvm-objdump -d of the binary,
every backward branch whose loop holds no other backward branch and at least 8 ymm vminpd / vminps, with its address,
its start mod 64, its byte size, the 64-byte lines it spans, its instruction count and its kind (L1: vandp*, squared:
vmulp*/vfmadd*). usage: exe_loops.py <exe> [--list]"""
import re
import subprocess
import sys

exe = sys.argv[1]
show = "--list" in sys.argv
out = subprocess.run(["C:/Program Files/LLVM/bin/llvm-objdump.exe", "-d", "--no-show-raw-insn", "-M", "intel", exe],
                     capture_output=True, text=True, errors="replace").stdout
ins = []  # (address, text)
for ln in out.splitlines():
    m = re.match(r"^\s*([0-9a-f]+):\s+(.*)$", ln)
    if m:
        ins.append((int(m.group(1), 16), m.group(2).strip()))
index = {a: k for k, (a, _) in enumerate(ins)}
loops = []
for k, (a, t) in enumerate(ins):
    m = re.match(r"^j\w+\s+(?:0x)?([0-9a-f]+)", t)
    if not m:
        continue
    target = int(m.group(1), 16)
    if target in index and target < a and a - target < 4096:
        loops.append((index[target], k))
inner = [l for l in loops if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
for s, e in inner:
    body = ins[s:e + 1]
    mins = [t for _, t in body if re.match(r"^vmin(pd|ps)\s+ymm", t)]
    if len(mins) < 8:
        continue
    start, last = body[0][0], body[-1][0]
    nxt = ins[e + 1][0] if e + 1 < len(ins) else last + 2
    size = nxt - start
    kind = ("f64" if "pd" in mins[0].split()[0] else "f32") + (
        " L1" if any(t.startswith("vandp") for _, t in body) else " squared")
    lines = (nxt - 1) // 64 - start // 64 + 1
    print(f"{kind:12s} loop at {start:#x} (mod 64: {start % 64:2d}), {size} bytes, {lines} 64-byte lines, "
          f"{len(body)} instructions, {len(mins)} vmin")
    if show:
        for a, t in body:
            print(f"    {a:#x} ({a % 64:2d})  {t}")
