"""Do cl's /FAs lanes loops (compiled without /GL, which leaves no code to list) appear in an LTCG-linked exe?
Compares innermost loops that take a scalar min by their instruction sequence, memory operands normalised to [M]
(the listing names stack slots, a disassembly gives offsets) and branch targets dropped.
usage: msvc_loops_match.py <listing .asm> <exe> [<exe>...]"""
import re
import subprocess
import sys


def norm(ins):
    ins = ins.split(";")[0].split("#")[0].strip().lower()
    ins = re.sub(r"\s+", " ", ins)
    ins = re.sub(r"(byte|word|dword|qword|xmmword|ymmword) ptr ", "", ins)
    ins = re.sub(r"\S*\[[^\]]*\]", "[M]", ins)
    ins = re.sub(r"^(j\w+) .*", r"\1", ins)
    ins = re.sub(r"\b0x([0-9a-f]+)\b", lambda m: str(int(m.group(1), 16)), ins)  # objdump's hex immediates
    ins = re.sub(r"\b([0-9][0-9a-f]*)h\b", lambda m: str(int(m.group(1), 16)), ins)  # MASM's 0ffH
    return ins.replace(", ", ",")


def innermost(seq, key):
    """seq: [(label, None) | (None, instruction)]; the innermost backward-branch loops that take a scalar min,
    each as a tuple of normalised instructions. key maps a branch operand to a label."""
    pos = {lab: k for k, (lab, ins) in enumerate(seq) if lab is not None}
    loops = []
    for k, (lab, ins) in enumerate(seq):
        if ins is None:
            continue
        m = re.match(r"^j\w+\s+(?:short\s+)?(\S+)", ins.strip(), re.I)
        if m:
            t = key(m.group(1))
            if t in pos and pos[t] < k:
                loops.append((pos[t], k))
    inner = [l for l in loops if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
    out = []
    for a, b in inner:
        body = tuple(norm(i) for (lab, i) in seq[a:b + 1] if i is not None)
        if any(re.match(r"vmins[sd]", i) for i in body):
            out.append(body)
    return out


# The listing, one function at a time (cl reuses its $LL labels in every function); or loops.py --all output
# of a named disassembly (.txt): each "  loop" block's instructions.
listing, seq, inside = [], [], False
if sys.argv[1].endswith(".txt"):
    for ln in open(sys.argv[1], encoding="utf-8", errors="replace"):
        if ln.startswith("  loop"):
            listing.append([])
        elif ln.startswith("      ") and listing:
            listing[-1].append(norm(ln.strip()))
    listing = [tuple(l) for l in listing]
for ln in open(sys.argv[1], encoding="utf-8", errors="replace") if sys.argv[1].endswith(".asm") else []:
    if re.match(r"^\S+\s+PROC\b", ln):
        inside = "dtw_kernel_lanes" in ln and not re.search(r"_buf|kUnreachable", ln)
        seq = []
        continue
    if re.match(r"^\S+\s+ENDP\b", ln):
        if inside:
            listing += innermost(seq, lambda t: t)
        inside = False
        continue
    if not inside:
        continue
    m = re.match(r"^(\$\w+@\w+):", ln)
    if m:
        seq.append((m.group(1), None))
        ln = ln[len(m.group(0)):]
    s = ln.split(";")[0].strip()
    if s and not re.match(r"^(npad|ALIGN)\b", s):
        seq.append((None, s))
print(f"{sys.argv[1]}: {len(listing)} loops with a scalar min ({', '.join(str(len(l)) for l in listing)} insns)")

for exe in sys.argv[2:]:
    dis = subprocess.run(["C:/Program Files/LLVM/bin/llvm-objdump.exe", "-d", "--no-show-raw-insn", "-M", "intel", exe],
                         capture_output=True, text=True).stdout
    seq = []
    for ln in dis.splitlines():
        m = re.match(r"^\s*([0-9a-f]+):\s+(.*)$", ln)
        if m:
            seq.append((m.group(1).lstrip("0"), None))
            seq.append((None, m.group(2)))
    loops = set(innermost(seq, lambda t: t.lower().replace("0x", "").lstrip("0")))
    found = [l for l in listing if l in loops]
    print(f"  {exe}: {len(loops)} loops with a scalar min; the listing's found instruction for instruction: "
          f"{len(found)} of {len(listing)}")
