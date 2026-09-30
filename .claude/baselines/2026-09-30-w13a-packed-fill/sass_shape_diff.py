"""Kernel-by-kernel SASS comparison modulo register allocation.

Each instruction becomes a shape: register names (R, UR, P, UP, B) are
replaced by their class, branch targets by their distance in instructions;
opcodes, modifiers, immediates and constant-bank operands stay. The two
shape sequences are aligned (difflib). An aligned equal run is the same code
up to register names; the renaming is then checked to be one consistent
bijection over the run. Prints per kernel: instructions, instructions in equal
runs, the largest equal run and whether its renaming is consistent, and every
differing hunk as instructions.
"""
import difflib
import re
import sys

FUNC = re.compile(r"^\s*Function : (\S+)\s*$", re.M)
INSN = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s*(.*?)\s*;?\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$")
REG = re.compile(r"\b(UR|R|UP|P|B)(\d+)\b")


def kernel_key(mangled):
    m = re.match(r"_ZN4dtwc4cuda\d+(\w+?)I(.*?)EEv", mangled)
    return (m.group(1), m.group(2)) if m else (mangled, "")


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    parts = FUNC.split(text)
    out = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        insns = []
        for line in body.splitlines():
            m = INSN.match(line)
            if m:
                insns.append((int(m.group(1), 16), m.group(2).strip()))
        out[kernel_key(name)] = insns
    return out


def relative_targets(insns):
    addr_to_idx = {a: i for i, (a, _) in enumerate(insns)}
    res = []
    for i, (a, s) in enumerate(insns):
        if re.search(r"\b(BRA|BSSY|CALL\.REL(\.NOINC)?|BREAK|BRX|JMP)\b", s) or re.match(r"^MOV R\d+, 0x[0-9a-f]+$", s):
            def rel(m):
                t = int(m.group(0), 16)
                return f"<{addr_to_idx[t] - i:+d}>" if t in addr_to_idx else m.group(0)
            s = re.sub(r"0x[0-9a-f]+", rel, s)
        res.append(s)
    return res


def shape(s):
    return REG.sub(lambda m: m.group(1), s)


def renaming_consistent(a_run, b_run):
    fwd, back = {}, {}
    for x, y in zip(a_run, b_run):
        ra, rb = REG.findall(x), REG.findall(y)
        if len(ra) != len(rb):
            return False
        for u, v in zip(ra, rb):
            u, v = "".join(u), "".join(v)
            if fwd.setdefault(u, v) != v or back.setdefault(v, u) != u:
                return False
    return True


def main(a_path, b_path, verbose):
    a, b = parse(a_path), parse(b_path)
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            print(f"== {key}: only in {'first' if key in a else 'second'}")
            continue
        ta, tb = relative_targets(a[key]), relative_targets(b[key])
        sa, sb = [shape(s) for s in ta], [shape(s) for s in tb]
        sm = difflib.SequenceMatcher(a=sa, b=sb, autojunk=False)
        blocks = [bl for bl in sm.get_matching_blocks() if bl.size]
        same = sum(bl.size for bl in blocks)
        big = max(blocks, key=lambda bl: bl.size)
        consistent = renaming_consistent(ta[big.a:big.a + big.size], tb[big.b:big.b + big.size])
        print(f"== {key[0]}<{key[1]}>: {len(sa)} vs {len(sb)} instructions; {same} in equal runs; "
              f"largest run {big.size} (first[{big.a}:{big.a + big.size}]), renaming consistent: {consistent}")
        if verbose:
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                print(f"   {tag} first[{i1}:{i2}] second[{j1}:{j2}]")
                for s in ta[i1:i2]:
                    print(f"     - {s}")
                for s in tb[j1:j2]:
                    print(f"     + {s}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3)
