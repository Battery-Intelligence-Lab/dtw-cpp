"""Diagnostic only: in a clang -S -masm=intel listing, swap two pairs of callee-saved GPRs (every width) inside one
function, the f64 L1 lanes kernel (lambda_2, double): a consistent renaming, so the code computes what it did, with
other encodings. usage: rename_regs.py <in.s> <out.s> <regA:regB> [<regC:regD>]  e.g. r12:rbp rsi:rbx"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
pairs = [p.split(":") for p in sys.argv[3:]]
WIDTHS = {  # 64, 32, 16, 8
    "rax": ["rax", "eax", "ax", "al"], "rbx": ["rbx", "ebx", "bx", "bl"], "rcx": ["rcx", "ecx", "cx", "cl"],
    "rdx": ["rdx", "edx", "dx", "dl"], "rsi": ["rsi", "esi", "si", "sil"], "rdi": ["rdi", "edi", "di", "dil"],
    "rbp": ["rbp", "ebp", "bp", "bpl"],
    **{f"r{n}": [f"r{n}", f"r{n}d", f"r{n}w", f"r{n}b"] for n in range(8, 16)},
}
table = {}
for a, b in pairs:
    for wa, wb in zip(WIDTHS[a], WIDTHS[b]):
        table[wa], table[wb] = wb, wa
pattern = re.compile(r"\b(" + "|".join(sorted(table, key=len, reverse=True)) + r")\b")
lines = open(src, encoding="utf-8").read().split("\n")
out, inside, done = [], False, False
for ln in lines:
    if not inside and not done and ln.startswith('"??$dtw_kernel_lanes@NV<lambda_2>') and ln.rstrip().endswith('"'):
        inside = True  # the function's label line (the .seh_proc line follows)
    if inside and not ln.lstrip().startswith(("\"", "#", ".seh_proc", ".def", ".scl", ".type", ".endef")):
        code, sep, comment = ln.partition("#")
        ln = pattern.sub(lambda m: table[m.group(1)], code) + sep + comment
    if inside and ln.strip().startswith(".seh_endproc"):
        inside, done = False, True
    out.append(ln)
open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out))
print("renamed" if done else "FUNCTION NOT FOUND")
