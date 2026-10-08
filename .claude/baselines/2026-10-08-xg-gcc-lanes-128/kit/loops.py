"""Innermost loops of the lanes kernels in an assembly listing: clang -S -masm=intel (.s), cl /FAs (.asm), or an
llvm-objdump --x86-asm-syntax=intel disassembly of a linked binary (.dis).

usage: loops.py <listing> [--all]

For every function whose name holds dtw_kernel_lanes (both compilers keep the kernel out of line), every innermost
loop that takes a min (a backward branch to a label with no other loop inside): its instruction count, the min
instructions and their register width, FP ops, memory operands, calls, and stack traffic. A stack access is an
instruction clang's listing marks "Spill" or "Reload" (clang and cl may use rbp as an ordinary register, so an
rbp-relative operand is not one), or, in a cl listing or a disassembly, an rsp-relative operand. --all prints
each loop's instructions too.
"""
import re
import sys

path = sys.argv[1]
show = "--all" in sys.argv
lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
masm = path.endswith(".asm")
objdump = path.endswith(".dis")

funcs = []  # (name, [(label, None) | (None, instruction)])
cur = None
for ln in lines:
    if masm:
        m = re.match(r"^(\S+)\s+PROC\b", ln)
        if m:
            cur = (m.group(1), [])
            funcs.append(cur)
            continue
        if re.match(r"^\S+\s+ENDP\b", ln):
            cur = None
            continue
        if cur is None:
            continue
        m = re.match(r"^(\$\w+@\w+):", ln)
        if m:
            cur[1].append((m.group(1), None))
            ln = ln[len(m.group(0)):]
        s = ln.split(";")[0].strip()
        if s and not re.match(r"^(npad|ALIGN|EXTRN|PUBLIC|COMDAT|DB|DD|DQ|DW|ORG)\b", s):
            cur[1].append((None, s))
    elif objdump:
        m = re.match(r"^([0-9a-f]+) <(.*)>:$", ln)
        if m:
            cur = (m.group(2), [])
            funcs.append(cur)
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+(.*)$", ln)
        if m and cur is not None:
            cur[1].append((m.group(1).lstrip("0"), None))
            cur[1].append((None, m.group(2).strip()))
    else:
        m = re.match(r'^("[^"]+"|[A-Za-z_?$@][^\s:]*):', ln)
        if m and not m.group(1).startswith(("$", "__real", "__xmm", "__ymm")):
            cur = (m.group(1).strip('"'), [])
            funcs.append(cur)
            continue
        m = re.match(r"^(\.LBB\d+_\d+|\.L\d+):", ln)  # clang's and GCC's block labels
        if m and cur is not None:
            cur[1].append((m.group(1), None))
            continue
        if cur is not None and ln.startswith("\t") and not ln.strip().startswith((".", "#")):
            ins, _, comment = ln.strip().partition("#")
            tag = "   ; " + comment.strip() if re.search(r"Spill|Reload", comment) else ""
            cur[1].append((None, ins.strip() + tag))


def branch_target(ins):
    m = re.match(r"^j\w+\s+(?:SHORT\s+)?(\S+)", ins)
    if not m:
        return None
    t = m.group(1)
    if objdump:
        m2 = re.match(r"^(?:0x)?([0-9a-f]+)", t)
        return m2.group(1).lstrip("0") if m2 else None
    return t


FP = re.compile(r"^v?(min|max|add|sub|mul|and|andn|xor|fmadd\w*|fnmadd\w*|fmsub\w*)(pd|ps|sd|ss)\b")
MIN = re.compile(r"^v?min(pd|ps|sd|ss)\b")
for name, body in funcs:
    # cl's LTCG and GCC inline the kernel into the block function's std::function call operator
    inlined = ("_Do_call" in name or "_M_invoke" in name or "clE" in name) and "block_fn" in name
    if ("dtw_kernel_lanes" not in name and not inlined) or re.search(r"kUnreachable|_buf|__J|_Resize|_Xlength|vector@URow|ED2Ev", name):
        continue
    if inlined and "block_fn@" in name:
        name = name.replace("block_fn@", "dtw_kernel_lanes@", 1)
    elif inlined:  # GCC: block_fnId... / block_fnIf..., the second lambda (E0_) is L1
        t = "@N" if "block_fnId" in name else "@M"
        name = f"dtw_kernel_lanes{t}V<lambda_{'2' if re.search(r'Ul(dd|ff)E0_', name) else '1'}> (GCC) " + name[:60]
    pos = {lab: k for k, (lab, ins) in enumerate(body) if lab is not None}
    loops = []
    for k, (lab, ins) in enumerate(body):
        t = branch_target(ins) if ins is not None else None
        if t is not None and t in pos and pos[t] < k:
            loops.append((pos[t], k))
    inner = [l for l in loops if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
    tail = name.split("dtw_kernel_lanes", 1)[1]
    kind = "f64" if tail.startswith(("@N", "<double")) else "f32" if tail.startswith(("@M", "<float")) else "?"
    printed = False
    for a, b in inner:
        ins = [i for (lab, i) in body[a:b + 1] if i is not None]
        mins = [i for i in ins if MIN.match(i)]
        if not mins:
            continue
        if not printed:
            print(f"== {kind} {name[:100]}")
            printed = True
        width = sorted({w + "mm" for i in mins for w in re.findall(r"\b([xyz])mm\d+", i)})
        ops = sorted({MIN.match(i).group(0).lstrip("v") for i in mins})
        fp = [i for i in ins if FP.match(i)]
        mem = [i for i in ins if re.search(r"\[[^\]]+\]", i)]
        if masm or objdump or "/gcc_" in path.replace("\\", "/"):  # GCC marks no spills: rsp-relative operands
            stack = [i for i in mem if re.search(r"\[rsp\b", i.lower())]
        else:
            stack = [i for i in ins if re.search(r"Spill|Reload", i)]
        calls = [i for i in ins if i.startswith("call")]
        print(f"  loop {len(ins):3d} insns  min {len(mins):2d} ({'/'.join(ops)} on {','.join(width)})  fp {len(fp):3d}"
              f"  mem {len(mem):3d}  stack {len(stack):2d}  calls {len(calls)}")
        for s in stack:
            print(f"    stack: {s}")
        if show:
            for i in ins:
                print(f"      {i}")
