"""CW driver (WM's kit, pointed at CW's wheels): interleaved rounds of the MSVC wheel (design-2.0, as CI builds it
today) and the clang-cl wheel (pb/CW), then medians, ranges, ratios and the registered band (stdlib only).

uv run --no-project python driver.py run      # 3 rounds -> raw.txt (every measurement line verbatim)
uv run --no-project python driver.py smoke    # 1 tiny round -> smoke.txt (proves the kit; numbers in no table)
uv run --no-project python driver.py report [smoke]   # medians, clang-cl/MSVC ratios, the rule -> tables.md
"""
import os
import pathlib
import re
import statistics
import subprocess
import sys

KIT = pathlib.Path(__file__).resolve().parent
PYTHON = {"msvc": "C:/D/git/wt/venv/cw-msvc/Scripts/python.exe",
          "clang": "C:/D/git/wt/venv/cw-clang/Scripts/python.exe"}
SHAPES = ["a", "b", "c", "d_full", "d_band"]
LABEL = {"a": "(a) N 2000, L 100, unbanded: lanes fill", "b": "(b) N 500, L 1000, band 100: lanes fill",
         "c": "(c) N 1000, L 90-110, unbanded: per-pair fill", "d_full": "(d) dtw, one pair, L 1000, unbanded",
         "d_band": "(d) dtw, one pair, L 1000, band 100"}
THREADS = [1, 24]
ROUNDS = 3


def threads_for(shape):
    """(d) is one pair per call, single-threaded whatever OpenMP is set to: timed pinned only (orchestrator, before any
    run: unpinned it can land on an E-core and swing the ratio by itself)."""
    return [1] if shape.startswith("d") else THREADS


def env_for(threads):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "KMP_"))}
    if threads == 1:
        env["OMP_NUM_THREADS"] = "1"
    return env


def run(rounds, out, extra):
    with open(out, "a") as raw:
        for r in range(rounds):
            order = ("msvc", "clang") if r % 2 == 0 else ("clang", "msvc")
            for shape in SHAPES:
                for threads in threads_for(shape):
                    for wheel in order:
                        args = [PYTHON[wheel], str(KIT / "measure.py"), shape, str(threads), *extra]
                        p = subprocess.run(args, capture_output=True, text=True, env=env_for(threads))
                        raw.write(f"round={r} exit={p.returncode} {p.stdout.strip()}\n")
                        raw.flush()
                        if p.returncode != 0:
                            sys.exit(f"measurement failed: {args} -> {p.returncode}\n{p.stdout}{p.stderr}")
            print(f"round {r} done", flush=True)


def outcome(ratio):
    """The registered band (orchestrator, 2026-10-08, fixed before any run)."""
    if all(ratio[(s, 24)] >= 1.2 for s in ("a", "b")) and min(ratio.values()) >= 0.95:
        return "PASS: the clang-cl wheel ships as built (both lanes fills >= 1.2x at 24 threads, no shape below 0.95x)"
    return "FAIL: the result goes to Volkan before the merge (a lanes fill below 1.2x at 24 threads, or a shape below 0.95x)"


def report(path, rounds):
    runs = {}
    for line in path.read_text().splitlines():
        f = dict(re.findall(r"(\S+?)=(\S+)", line))
        runs.setdefault((f["shape"], int(f["threads"]), f["wheel"].removeprefix("cw-")), []).append(f)
    problems, ratio, rows = [], {}, []
    for shape in SHAPES:
        for threads in threads_for(shape):
            med = {}
            for wheel in PYTHON:
                fs = runs.get((shape, threads, wheel), [])
                if len(fs) != rounds:
                    problems.append(f"{shape} t{threads} {wheel}: {len(fs)} of {rounds} rounds")
                if not fs:
                    continue
                if len({f["result"] for f in fs}) != 1 or len({f["pyd"] for f in fs}) != 1:
                    problems.append(f"{shape} t{threads} {wheel}: result or module differs between rounds")
                if any(f["exit"] != "0" or int(f["engaged"]) != threads for f in fs):
                    problems.append(f"{shape} t{threads} {wheel}: an exit code or the thread count is off")
                g = [float(f["gcells"]) for f in fs]
                med[wheel] = (statistics.median(g), min(float(f["seconds"]) for f in fs), fs[0]["result"],
                              f"{min(g):.3f}-{max(g):.3f}")
            if len(med) == 2:
                ratio[(shape, threads)] = med["clang"][0] / med["msvc"][0]
                same = "same" if med["clang"][2] == med["msvc"][2] else "differs"
                rows.append(f"| {LABEL[shape]} | {threads} | {med['msvc'][0]:.3f} ({med['msvc'][3]}) | "
                            f"{med['clang'][0]:.3f} ({med['clang'][3]}) | **{ratio[(shape, threads)]:.2f}** | "
                            f"{min(med['msvc'][1], med['clang'][1]):.2f} | {same} |")
    smoke = path.name.startswith("smoke")
    out = [f"{'SMOKE RUN: proves the kit only; these numbers go in no table. ' if smoke else ''}"
           f"Medians (range) over {rounds} round(s), Gcell/s; ratio = clang-cl / MSVC of the medians "
           "(above 1: clang-cl faster).", "",
           "| shape | threads | MSVC Gcell/s | clang-cl Gcell/s | clang-cl/MSVC | shortest timed s | result |",
           "| --- | --- | --- | --- | --- | --- | --- |", *rows, "",
           "band: " + (outcome(ratio) if len(ratio) == sum(len(threads_for(s)) for s in SHAPES) else "incomplete table"),
           "problems: " + ("none" if not problems else "; ".join(problems))]
    if not smoke:
        (KIT / "tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    command = sys.argv[1]
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
