"""loop_iso.py <a> <b> <kind> <insns>: the innermost lanes loops of two listings (loops.py --all on a .s or .dis) of
the given kind ('f32 L1', 'f32 Sq', 'f64 L1', 'f64 Sq') and instruction count: are they the same up to register
names (registers renamed in order of first appearance per class), or, weaker, the same instructions and memory
operands in the same order with every register ignored? Branch targets are dropped; a cmp of two registers is read
without operand order (it feeds jne/je)."""
import re
import subprocess
import sys


def loops(path, kind, n):
    out = subprocess.run(["uv", "run", "--no-project", "python", "tools/loops.py", path, "--all"], capture_output=True,
                         text=True).stdout
    want_t, want_d = kind.split()
    res, cur, take = [], None, False
    for ln in out.splitlines():
        if ln.startswith("== "):
            take = ln.split()[1] == want_t and ("L1" if "lambda_2" in ln else "Sq") == want_d
        elif ln.startswith("  loop"):
            cur = [] if take and int(ln.split()[1]) == n else None
            if cur is not None:
                res.append(cur)
        elif ln.startswith("      ") and cur is not None:
            cur.append(ln.strip().split("   ;")[0])
    return res


def canon(body, exact):
    names = {}

    def reg(m):
        r = m.group(0)
        if not exact:
            return "R" + (r[0] if r[0] in "xyz" else "")
        cls = "v" if r[0] in "xyz" else "g"
        key = r[1:] if r[0] in "xyz" else r
        if (cls, key) not in names:
            names[(cls, key)] = f"{cls}{sum(1 for c, _ in names if c == cls)}"
        return names[(cls, key)] + ("" if cls == "g" else r[0])

    out = []
    for i in body:
        i = re.sub(r"^(j\w+)\s+.*", r"\1", i)
        i = re.sub(r"\s+", " ", re.sub(r"\b([xyz]mm\d+|r[a-z0-9]+|e[a-z]{2})\b", reg, i))
        m = re.match(r"^cmp (\w+), ?(\w+)$", i)
        if m:
            i = "cmp " + ",".join(sorted(m.groups()))
        out.append(i)
    return out


a, b = loops(sys.argv[1], sys.argv[3], int(sys.argv[4])), loops(sys.argv[2], sys.argv[3], int(sys.argv[4]))
if not a or not b:
    sys.exit(f"no {sys.argv[3]} loop of {sys.argv[4]} instructions in {'a' if not a else 'b'}")
ea, eb = {tuple(canon(x, True)) for x in a}, {tuple(canon(x, True)) for x in b}
ra, rb = {tuple(canon(x, False)) for x in a}, {tuple(canon(x, False)) for x in b}
print(f"{sys.argv[3]} {sys.argv[4]}-insn loops: a {len(a)}, b {len(b)}; identical up to register names: "
      f"{'yes' if ea & eb else 'no'}; same instructions and memory operands in the same order (registers aside): "
      f"{'yes' if ra & rb else 'NO'}")
