"""llvm-mca on the clang lanes main loops (the innermost loop with packed ymm mins) of a listing: a model's
prediction of cycles per row step and per cell, not a measurement. usage: mca.py <clang .s> <mcpu>"""
import re
import subprocess
import sys

src, mcpu = sys.argv[1], sys.argv[2]
out = subprocess.run(["uv", "run", "--no-project", "python", "tools/loops.py", src, "--all"], capture_output=True,
                     text=True).stdout
blocks, cur, name = [], None, None
for ln in out.splitlines():
    if ln.startswith("== "):
        name = ln.split()[1] + (" Sq" if "lambda_1" in ln else " L1")
    elif ln.startswith("  loop"):
        cur = {"name": name, "head": ln, "ins": []}
        blocks.append(cur)
    elif ln.startswith("      ") and cur is not None:
        cur["ins"].append(ln.strip().split("   ;")[0])
for b in blocks:
    if not re.search(r"min\w* on ymm", b["head"]):
        continue
    W = {"f64": 16, "f32": 32}[b["name"][:3]] if "cand" in src else {"f64": 8, "f32": 16}[b["name"][:3]]
    vec = W // (4 if b["name"].startswith("f64") else 8)
    mins = sum(1 for i in b["ins"] if re.match(r"vmin(pd|ps)", i))
    steps = mins // (2 * vec)
    if steps == 0:
        continue  # not a row-step loop (column 0 runs one min per lane)
    body = ".intel_syntax noprefix\n" + "\n".join(i for i in b["ins"] if not i.startswith("j")) + "\n"
    r = subprocess.run(["C:/Program Files/LLVM/bin/llvm-mca.exe", f"-mcpu={mcpu}", "-iterations=2000"], input=body,
                       capture_output=True, text=True)
    cyc = float(re.search(r"Total Cycles:\s+(\d+)", r.stdout).group(1)) / 2000
    rthr = re.search(r"Block RThroughput:\s+([\d.]+)", r.stdout).group(1)
    print(f"{b['name']:7s} W {W:2d}: {len(b['ins'])} insns, {steps} row step(s)/iteration; mca {mcpu}: {cyc:.2f} cycles/"
          f"iteration = {cyc / steps:.2f} per row step = {cyc / steps / W:.3f} per cell (block RThroughput {rthr})")
