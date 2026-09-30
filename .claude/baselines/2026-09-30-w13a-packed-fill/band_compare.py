"""Median real time per W13a band case, base vs head, from Google Benchmark JSON."""
import json
import sys


def stats(path):
    rows = json.load(open(path))["benchmarks"]
    out = {}
    for r in rows:
        if r.get("run_type") != "aggregate":
            continue
        name = r["run_name"]
        out.setdefault(name, {})[r["aggregate_name"]] = r
    return out


base, head = stats(sys.argv[1]), stats(sys.argv[2])
print(f"{'case':44} {'base ms':>9} {'head ms':>9} {'head/base':>9}  {'cv base,head':>13}  Gcell/s base,head  pass")
for name in base:
    b, h = base[name], head[name]
    bm, hm = b["median"]["real_time"], h["median"]["real_time"]
    ratio = hm / bm
    print(f"{name.replace('BM_cuda_fill/', ''):44} {bm:9.1f} {hm:9.1f} {ratio:9.4f}  "
          f"{b['cv']['real_time']*100:5.2f}%,{h['cv']['real_time']*100:5.2f}%  "
          f"{b['median']['Gcell/s']:7.2f},{h['median']['Gcell/s']:7.2f}  {'yes' if ratio <= 1.05 else 'NO'}")
