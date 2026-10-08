"""The L1 lanes main loops (one DP row step of the kernel's inner loop, f64 and f32) out of loops.py --all
outputs, chosen by the instruction counts the record names; every copy GCC made of the loop (it keeps two, one per
way into the row loop) is printed. usage: main_loops.py <loops.txt> <f64 insns> <f32 insns>"""
import re
import sys

text = open(sys.argv[1]).read().splitlines()
want = {"f64": int(sys.argv[2]), "f32": int(sys.argv[3])}
kind, keep, copies = None, False, {}
for ln in text:
    if ln.startswith("== "):
        kind = ln.split()[1] if "lambda_2" in ln else None  # lambda_2 is the L1 lambda for every compiler
        keep = False
        continue
    m = re.match(r"^  loop\s+(\d+) insns", ln)
    if m:
        keep = kind is not None and int(m.group(1)) == want[kind]
        if keep:
            copies[kind] = copies.get(kind, 0) + 1
            print(f"== {kind} L1 main loop, copy {copies[kind]}: {ln.strip()}")
        continue
    if keep and ln.startswith("    "):
        if not ln.startswith("    stack"):
            print("  " + ln.strip())
