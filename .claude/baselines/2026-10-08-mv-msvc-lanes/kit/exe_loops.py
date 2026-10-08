"""exe_loops.py <exe> [pattern...]: innermost loops of a linked exe (llvm-objdump, no symbols needed) whose body
matches every regex given (default: a min); prints each loop's address, size and instructions."""
import re, subprocess, sys
exe, pats = sys.argv[1], sys.argv[2:] or [r"vmin"]
dis = subprocess.run(["C:/Program Files/LLVM/bin/llvm-objdump.exe", "-d", "--no-show-raw-insn", "-M", "intel", exe],
                     capture_output=True, text=True).stdout
seq = []
for ln in dis.splitlines():
    m = re.match(r"^\s*([0-9a-f]+):\s+(.*)$", ln)
    if m:
        seq.append((m.group(1).lstrip("0"), m.group(2).strip()))
pos = {a: k for k, (a, i) in enumerate(seq)}
loops = []
for k, (a, ins) in enumerate(seq):
    m = re.match(r"^j\w+\s+(?:0x)?([0-9a-f]+)", ins)
    if m:
        t = m.group(1).lstrip("0")
        if t in pos and pos[t] < k:
            loops.append((pos[t], k))
inner = [l for l in loops if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
for a, b in inner:
    body = [i for (_, i) in seq[a:b + 1]]
    if all(any(re.search(p, i) for i in body) for p in pats):
        mem = sum(1 for i in body if "[" in i)
        stack = sum(1 for i in body if re.search(r"\[rsp", i))
        print(f"--- {len(body)} insns at {seq[a][0]}  mem {mem}  rsp {stack}  calls {sum(1 for i in body if i.startswith('call'))}")
        if "--all" in sys.argv[0:1] or len(body) <= 140:
            for i in body:
                print("   ", i)
