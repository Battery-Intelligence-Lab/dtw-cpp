"""Per-cell hardware counters of the RL probes from a profile_rl_etw.cmd result folder (rl_etw, rl_etw2).

Each case <build>_<shape> has <case>.pmc.txt (etw_pmc.py's sums for the probe process) and <case>.probe.txt (the timed
fill's line). A probe run is two fills of the same shape (one untimed, one timed), so its cells are 2 x the timed
fill's (Gcell/s x ms). Usage:  uv run --no-project python etw_table.py <folder>
Prints a markdown table: instructions, cycles and branch mispredictions per cell, IPC, LLC references and misses per
million cells, LLC miss/ref, and the timed fill's ms.
"""
import pathlib
import re
import sys

COUNT = re.compile(r"(\w+)=([\d,]+)")
FILL = re.compile(r"^fill .* ms ([\d.]+) .* Gcell/s ([\d.]+) ")
SHAPES = {"s1": "f64 L1 N 1000 L 90-110, 1 thread", "s2": "f64 L1 N 150 L 100-1000, 1 thread",
          "s5": "f64 L1 N 24 L 8000, 1 thread", "s6": "f64 L1 N 8 L 8000, 1 thread",
          "s3": "f64 L1 N 8000 L 40-60, 24 threads"}


def main(folder):
    d = pathlib.Path(folder)
    rows = []
    for pmc in sorted(d.glob("*.pmc.txt")):
        case = pmc.name[: -len(".pmc.txt")]
        build, shape = case.rsplit("_", 1)
        line = next(ln for ln in pmc.read_text().splitlines() if "rprobe.exe" in ln)
        c = {k: int(v.replace(",", "")) for k, v in COUNT.findall(line)}
        fill = next(m for ln in (d / f"{case}.probe.txt").read_text().splitlines() if (m := FILL.match(ln)))
        ms, gcells = float(fill[1]), float(fill[2])
        cells = 2 * gcells * 1e9 * ms / 1e3
        rows.append((list(SHAPES).index(shape) if shape in SHAPES else 99, shape, build, ms, cells, c))
    print("| shape | build | timed fill ms | instructions / cell | cycles / cell | IPC | LLC references / Mcell | "
          "LLC misses / Mcell | LLC miss / ref | branch mispredictions / Mcell |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, shape, build, ms, cells, c in sorted(rows, key=lambda r: (r[0], r[2])):
        ins, cyc = c["InstructionRetired"], c["UnhaltedCoreCycles"]
        ref, miss = c["LLCReference"], c["LLCMisses"]
        print(f"| {shape} {SHAPES.get(shape, '')} | {build} | {ms:.1f} | {ins / cells:.3f} | {cyc / cells:.3f} | "
              f"{ins / cyc:.2f} | {1e6 * ref / cells:,.0f} | {1e6 * miss / cells:,.0f} | {miss / max(ref, 1):.3f} | "
              f"{1e6 * c['BranchMispredictions'] / cells:,.0f} |")


if __name__ == "__main__":
    main(sys.argv[1])
