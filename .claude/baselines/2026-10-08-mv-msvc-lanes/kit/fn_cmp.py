"""fn_cmp.py <a.s> <b.s>: compare two GCC/clang -S listings function by function: each function's instructions, with
its local labels (.L123, .LBB1_2, .LFB7, .LC4 ...) renumbered in order of first appearance, the Cell's name and the
anonymous-namespace hash normalised as in norm_s.py; the constant pool's .LC entries compared by content. Prints
'identical' (N functions) or the functions that differ."""
import re, sys
def parse(path):
    funcs, cur, consts, cname = {}, None, {}, None
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.rstrip("\n")
        ln = re.sub(r"\?A0x[0-9A-Fa-f]{8}", "?A0xHASH", ln)
        ln = ln.replace("UStandardCell@", "ULanesCell@").replace("12StandardCell", "9LanesCell")
        m = re.match(r'^("[^"]+"|[A-Za-z_?$@][^\s:]*):', ln)
        if m and not m.group(1).startswith((".L", "$")):
            cur = m.group(1)
            funcs[cur] = []
            cname = None
            continue
        m = re.match(r"^(\.LC\d+|\.LCPI\d+_\d+|__real@\w+|__ymm@\w+|__xmm@\w+):", ln)
        if m:
            cname = m.group(1)
            consts[cname] = []
            continue
        s = ln.split("#", 1)[0].strip() if not ln.lstrip().startswith('"') else ln.strip()
        if cname is not None and re.match(r"^\.(long|quad|byte|short|value|zero|string|ascii)", s):
            consts[cname].append(s)
            continue
        if cur is None or not s or s.startswith((".cfi", ".p2align", ".align", ".size", ".type", ".file", ".ident",
                                                  ".section", ".weak", ".globl", ".set", ".def", ".scl", ".endef",
                                                  ".seh", ".text", ".linkonce")):
            continue
        funcs[cur].append(s)
    return funcs, consts
def canon(body, consts):
    ids, out = {}, []
    for s in body:
        def sub(m):
            t = m.group(0)
            if t in consts:
                return "CONST[" + "|".join(consts[t]) + "]"
            return ids.setdefault(t, f".L#{len(ids)}")
        out.append(re.sub(r"\.L[A-Za-z]*\d+(?:_\d+)?", sub, s))
    return out
fa, ca = parse(sys.argv[1])
fb, cb = parse(sys.argv[2])
na = {k: canon(v, ca) for k, v in fa.items()}
nb = {k: canon(v, cb) for k, v in fb.items()}
only_a, only_b = sorted(set(na) - set(nb)), sorted(set(nb) - set(na))
diff = [k for k in na if k in nb and na[k] != nb[k]]
if not only_a and not only_b and not diff:
    print(f"identical ({len(na)} functions, {sum(len(v) for v in na.values())} instructions)")
else:
    print(f"DIFFER: {len(diff)} functions differ, {len(only_a)} only in a, {len(only_b)} only in b")
    for k in diff:
        print("  differs:", k[:150], len(na[k]), "->", len(nb[k]), "instructions")
    for k in only_a[:10]:
        print("  only a:", k[:150])
    for k in only_b[:10]:
        print("  only b:", k[:150])
