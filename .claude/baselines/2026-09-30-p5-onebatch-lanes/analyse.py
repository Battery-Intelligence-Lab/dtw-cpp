import re, sys, statistics as st
log, rounds = sys.argv[1], int(sys.argv[2])
rows = {"base": [], "head": []}
cur = {}
for line in open(log):
    m = re.match(r"(base|head) OBP fill_s=([\d.]+) table_hash=(\w+)", line)
    if m: cur[m.group(1)] = (float(m.group(2)), m.group(3)); continue
    m = re.match(r"(base|head) call_s=([\d.]+) hash=(\w+) cost=(\S+) evals=(\d+)", line)
    if m:
        rows[m.group(1)].append((cur[m.group(1)][0], float(m.group(2)), cur[m.group(1)][1], m.group(3), m.group(4), m.group(5)))
n = len(rows["base"])
print("samples base/head:", n, len(rows["head"]))
for k in ("base", "head"):
    f = [r[0] for r in rows[k]]; c = [r[1] for r in rows[k]]
    print(k, "fill median %.4f [%.4f-%.4f]  call median %.4f [%.4f-%.4f]" % (st.median(f), min(f), max(f), st.median(c), min(c), max(c)))
print("identity: table hashes", {r[2] for r in rows["base"] + rows["head"]}, "call hashes", {r[3] for r in rows["base"] + rows["head"]}, "cost", {r[4] for r in rows["base"] + rows["head"]}, "evals", {r[5] for r in rows["base"] + rows["head"]})
# paired per process pair (rounds samples each, same position in the log)
pf, pc = [], []
for p in range(n // rounds):
    b = rows["base"][p*rounds:(p+1)*rounds]; h = rows["head"][p*rounds:(p+1)*rounds]
    pf.append(st.median(r[0] for r in b) / st.median(r[0] for r in h))
    pc.append(st.median(r[1] for r in b) / st.median(r[1] for r in h))
print("paired fill ratio median %.2f [%.2f-%.2f]; whole call %.2f [%.2f-%.2f]" % (st.median(pf), min(pf), max(pf), st.median(pc), min(pc), max(pc)))
print("pairs fill", [round(x, 2) for x in pf]); print("pairs call", [round(x, 2) for x in pc])
