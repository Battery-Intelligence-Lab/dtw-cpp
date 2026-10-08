"""Disassembly of an MSVC-linked probe with the function names of its /MAP file (llvm-objdump; .text is section
0001 at RVA 0x1000, image base 0x140000000). usage: msvc_dis.py <exe> <map> <out .dis>"""
import re
import subprocess
import sys

exe, mapf, out = sys.argv[1:4]
names = {}
for ln in open(mapf, encoding="utf-8", errors="replace"):
    m = re.match(r"^\s*0001:([0-9a-f]{8})\s+(\S+)", ln)
    if m:
        addr = 0x140001000 + int(m.group(1), 16)
        names.setdefault(addr, m.group(2))
dis = subprocess.run(["C:/Program Files/LLVM/bin/llvm-objdump.exe", "-d", "--no-show-raw-insn", "-M", "intel", exe],
                     capture_output=True, text=True).stdout
with open(out, "w") as f:
    for ln in dis.splitlines():
        m = re.match(r"^\s*([0-9a-f]+):", ln)
        if m and int(m.group(1), 16) in names:
            f.write(f"\n{int(m.group(1), 16):016x} <{names[int(m.group(1), 16)]}>:\n")
        f.write(ln + "\n")
