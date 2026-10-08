"""CA driver: interleaved rounds of the release archive's dtwc_cl.exe (clang-cl) and dtwc_cl_msvc.exe (MSVC) on the
CLI's distance-matrix fill, then medians, ranges, ratios and the registered band (stdlib only).

uv run --no-project python driver.py run      # 3 rounds -> raw.txt (every measurement line verbatim)
uv run --no-project python driver.py smoke    # 1 tiny round -> smoke.txt (proves the kit; numbers in no table)
uv run --no-project python driver.py report [smoke]   # medians, ratios, ranges, the band -> tables.md

A round runs every shape at 24 threads, then pinned on one thread; at 24 threads each pair of runs is repeated
REPS times in the round (the fills take well under a second there). Rounds alternate which CLI goes first.
"""
import pathlib
import re
import statistics
import subprocess
import sys

KIT = pathlib.Path(__file__).resolve().parent
EXES = ("dtwc_cl_msvc", "dtwc_cl")
SHAPES = ("a", "b", "c")
THREADS = (24, 1)
REPS = {24: 5, 1: 1}
ROUNDS = 3
LABEL = {"a": "(a) N 2000, L 100, unbanded", "b": "(b) N 500, L 1000, band 100",
         "c": "(c) N 1000, L 90-110, unbanded (ragged)"}


def run(rounds, out, extra):
    with open(out, "a") as raw:
        for r in range(rounds):
            order = EXES if r % 2 == 0 else EXES[::-1]
            for shape in SHAPES:
                for threads in THREADS:
                    for _ in range(REPS[threads]):
                        for exe in order:
                            args = [sys.executable, str(KIT / "measure.py"), exe, shape, str(threads), *extra]
                            p = subprocess.run(args, capture_output=True, text=True)
                            raw.write(f"round={r} {p.stdout.strip()}\n")
                            raw.flush()
                            if p.returncode != 0:
                                sys.exit(f"measurement failed: {args} -> {p.returncode}\n{p.stdout}{p.stderr}")
            print(f"round {r} done", flush=True)


def outcome(ratio):
    """The registered band (orchestrator, 2026-10-08, fixed before any run)."""
    if all(ratio[(s, 24)] >= 1.2 for s in ("a", "b")) and min(ratio.values()) >= 0.95:
        return ("PASS: the archive ships as built (dtwc_cl.exe fills each equal-length shape >= 1.2x as fast as "
                "dtwc_cl_msvc.exe at 24 threads, no shape below 0.95x)")
    return ("FAIL: the result goes to Volkan before the merge (an equal-length shape below 1.2x at 24 threads, or a "
            "shape below 0.95x)")


def report(path, rounds):
    runs = {}
    for line in path.read_text().splitlines():
        f = dict(re.findall(r"(\S+?)=(\S+)", line))
        runs.setdefault((f["shape"], int(f["threads"]), f["exe"]), []).append(f)
    problems, ratio, rows = [], {}, []
    for shape in SHAPES:
        for threads in THREADS:
            med, per_round = {}, {}
            for exe in EXES:
                fs = runs.get((shape, threads, exe), [])
                if len(fs) != rounds * REPS[threads]:
                    problems.append(f"{shape} t{threads} {exe}: {len(fs)} of {rounds * REPS[threads]} runs")
                if not fs:
                    continue
                if len({f["cost"] for f in fs}) != 1 or len({f["sha"] for f in fs}) != 1:
                    problems.append(f"{shape} t{threads} {exe}: the cost or the executable differs between runs")
                if any(f["exit"] != "0" or int(f["omp"]) != threads for f in fs):
                    problems.append(f"{shape} t{threads} {exe}: an exit code or the thread count is off")
                s = [float(f["seconds"]) for f in fs]
                med[exe] = (statistics.median(s), min(s), max(s), float(fs[0]["cells"]), fs[0]["cost"])
                per_round[exe] = [statistics.median(float(f["seconds"]) for f in fs if f["round"] == str(r))
                                  for r in range(rounds) if any(f["round"] == str(r) for f in fs)]
            if len(med) == 2:
                ratio[(shape, threads)] = med["dtwc_cl_msvc"][0] / med["dtwc_cl"][0]
                rr = [m / c for m, c in zip(per_round["dtwc_cl_msvc"], per_round["dtwc_cl"])]
                a, b = med["dtwc_cl_msvc"][4], med["dtwc_cl"][4]
                same = "same" if a == b else f"differs ({abs(float(a) - float(b)) / abs(float(a)):.1e} rel.)"
                cell = {e: f"{med[e][0]:.3f} ({med[e][1]:.3f}-{med[e][2]:.3f}), {med[e][3] / med[e][0] / 1e9:.2f}"
                        for e in EXES}
                rows.append(f"| {LABEL[shape]} | {threads} | {cell['dtwc_cl_msvc']} | {cell['dtwc_cl']} | "
                            f"**{ratio[(shape, threads)]:.2f}** ({min(rr):.2f}-{max(rr):.2f}) | {same} |")
    smoke = path.name.startswith("smoke")
    out = [f"{'SMOKE RUN: proves the kit only; these numbers go in no table. ' if smoke else ''}"
           f"Fill seconds as the CLI stamps them (from 'Data loaded' to the FastPAM k=1 result): median (range) over "
           f"{rounds} round(s) x {REPS[24]} runs at 24 threads, x {REPS[1]} pinned to CPU 12 at 1 thread; then "
           "Gcell/s of the median. Ratio = dtwc_cl_msvc.exe / dtwc_cl.exe seconds of the medians (above 1: dtwc_cl.exe, "
           "clang-cl, faster), and the range of the per-round ratios.", "",
           "| shape | threads | dtwc_cl_msvc.exe s (range), Gcell/s | dtwc_cl.exe s (range), Gcell/s | "
           "speed-up (per-round range) | cost |",
           "| --- | --- | --- | --- | --- | --- |", *rows, "",
           "band: " + (outcome(ratio) if len(ratio) == len(SHAPES) * len(THREADS) else "incomplete table"),
           "problems: " + ("none" if not problems else "; ".join(problems))]
    if not smoke:
        (KIT / "tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "run":
        run(ROUNDS, KIT / "raw.txt", [])
    elif command == "smoke":
        (KIT / "smoke.txt").unlink(missing_ok=True)
        run(1, KIT / "smoke.txt", ["--smoke"])
    elif command == "report":
        smoke = sys.argv[2:] == ["smoke"]
        report(KIT / ("smoke.txt" if smoke else "raw.txt"), 1 if smoke else ROUNDS)
    else:
        sys.exit(__doc__)
