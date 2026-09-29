"""W4a: markdown tables and band verdicts from the driver's output (stdlib only).

Usage: summarize.py [results.txt]   (reads the "path," and "lib," rows)
"""
import sys

rows, libs = [], []
for line in open(sys.argv[1] if len(sys.argv) > 1 else "results.txt"):
    p = line.strip().split(",")
    if p[0] == "path":
        rows.append(dict(prec=p[1], N=int(p[2]), L=int(p[3]), path=p[4], auto=p[5] == "1",
                         med=float(p[6]), mn=float(p[7]), mx=float(p[8]), gcell=float(p[9]),
                         bad=int(p[11]), nbufs=int(p[15]), persistent=p[16] == "1",
                         bps=int(p[17]), regs=int(p[18])))
    elif p[0] == "lib":
        libs.append(dict(prec=p[1], N=int(p[2]), L=int(p[3]), override=p[4], kernel=p[5],
                         bad=int(p[8])))


def get(prec, L, path):
    return next((r for r in rows if r["prec"] == prec and r["L"] == L and r["path"] == path), None)


def auto_of(prec, L):
    return next(r for r in rows if r["prec"] == prec and r["L"] == L and r["auto"])


def cell(prec, L, path):
    r = get(prec, L, path)
    if r is None:
        return "–", "–", "–", "–", "–"
    a = auto_of(prec, L)
    spread = (r["mx"] - r["mn"]) / r["med"] * 100
    return (f"{r['med']:.3f}", f"{spread:.0f} %", f"{r['gcell']:.0f}", f"{r['med'] / a['med']:.3f}",
            "y" if r["bad"] == 0 else f"NO ({r['bad']})")


def table(Ls):
    print("| L | path | FP32 median ms | spread | Gcell/s | r | FP64 N | FP64 median ms | spread | Gcell/s | r | correct |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for L in Ls:
        paths = [r["path"] for r in rows if r["prec"] == "f32" and r["L"] == L]
        for path in paths:
            f32, f64 = cell("f32", L, path), cell("f64", L, path)
            n64 = get("f64", L, path)["N"]
            name = f"**{path}**" if get("f32", L, path)["auto"] else path
            ok = "y" if f32[4] == "y" and f64[4] == "y" else f"{f32[4]} / {f64[4]}"
            print(f"| {L} | {name} | {f32[0]} | {f32[1]} | {f32[2]} | {f32[3]} | {n64} | "
                  f"{f64[0]} | {f64[1]} | {f64[2]} | {f64[3]} | {ok} |")


def mode(r):
    if not r["path"].startswith("wavefront"):
        return r["path"]
    return ("preload, 5 buffers" if r["nbufs"] == 5 else f"{r['nbufs']} buffers") + \
        (", persistent" if r["persistent"] else "")


if __name__ == "__main__":
    print("Auto's pick (N = 200):\n")
    print("| L | Auto path | FP32 regs, blocks/SM | FP32 Gcell/s | FP64 Gcell/s |")
    print("|---|---|---|---|---|")
    for L in (128, 256, 257, 384, 500, 1024, 2048, 2049):
        a, b = auto_of("f32", L), auto_of("f64", L)
        print(f"| {L} | {mode(a)} | {a['regs']}, {a['bps']} | {a['gcell']:.0f} | {b['gcell']:.0f} |")
    print("\nwarp vs regtile (N = 1000):\n")
    table([16, 32])
    print("\nboundary set (N = 200; FP64 N as stated):\n")
    table([128, 256, 257, 384, 500, 1024, 2048, 2049])
    print("\ndouble- vs 3-buffer (N = 200; FP64 N as stated):\n")
    table([1100, 1500, 2000, 2048])
    print(f"\npublic-entry rows: {len(libs)}, with any mismatch: {sum(1 for l in libs if l['bad'])}")
    print(f"path rows: {len(rows)}, with any mismatch: {sum(1 for r in rows if r['bad'])}")
