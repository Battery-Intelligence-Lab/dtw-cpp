# Renames general-purpose registers (x/w) by first appearance, branch targets to T; prints the body.
import sys, re
m = {}
def ren(mo):
    r = mo.group(2)
    if r not in m: m[r] = f'R{len(m)}'
    return mo.group(1) + m[r]
for line in open(sys.argv[1]).read().splitlines()[1:]:
    line = re.sub(r'0x[0-9a-f]+$', 'T', line.strip())
    print(re.sub(r'\b([xw])(\d+)\b', ren, line))
