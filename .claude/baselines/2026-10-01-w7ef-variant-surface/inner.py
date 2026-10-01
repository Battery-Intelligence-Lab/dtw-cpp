"""Print the innermost loops (clang's "Inner Loop Header" blocks and the blocks
"in Loop: Header=" them) of one function in an Intel-syntax .s file, with the
instruction and call counts. Usage: inner.py <file.s> <function-substring>"""
import re
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(errors="replace").splitlines()
want = sys.argv[2]
FUNCTION = re.compile(r'^(?![.Ll])("[^"]+"|[^\s#;]+):')
BLOCK = re.compile(r"^\.?LBB(\d+_\d+):|^\s*# %bb\.(\d+):")
in_fn = False
blocks, order, cur = {}, [], None
for line in lines:
    m = FUNCTION.match(line)
    if m:
        in_fn = want in m.group(1)
        if in_fn:
            print("function:", m.group(1)[:120])
        continue
    if not in_fn:
        continue
    if b := BLOCK.match(line):
        cur = b.group(1) or b.group(2)
        blocks[cur] = {"text": [], "note": line}
        order.append(cur)
        continue
    if cur is None:
        continue
    if line.strip().startswith("#"):
        blocks[cur]["note"] += line
        continue
    code = line.split("#")[0].rstrip()
    if code.strip() and not code.strip().startswith("."):
        blocks[cur]["text"].append(code.strip())
headers = [k for k in order if "Inner Loop Header" in blocks[k]["note"]]
for h in headers:
    members = [k for k in order if k == h or f"Header=BB{h}" in blocks[k]["note"]]
    body = [t for k in members for t in blocks[k]["text"]]
    calls = [t for t in body if t.startswith("call")]
    print(f"inner loop BB{h}: {len(members)} blocks, {len(body)} instructions, calls: {calls}")
    for k in members:
        print(f"  BB{k}:")
        for t in blocks[k]["text"]:
            print("    " + t)
