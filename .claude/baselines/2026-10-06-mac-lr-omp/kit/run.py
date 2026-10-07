"""lr-omp timing kit: base b36fad43 against head, interleaved (stdlib only). Use run.sh.

For every (flavor, regime, N) the base probe and the head probe alternate (base head / head base / base head ...), REPEATS times, and
so do the base and the head dtwc_cl. One probe run gives three timings of the LR phase:
  root     the build's root: Kelley with HiGHS (flavor on), the subgradient root without (flavor off, what the wheel runs)
  LR phase lagrangian_root_exact: the root and the branch-and-bound tree (what LR_core_clustering spends in LR)
  cluster  Problem::cluster(): FastPAM seed + the dense copy of D + the LR phase + the labels (the matrix is filled before)
and one dtwc_cl run gives the end-to-end wall time (read the CSV, fill the matrix, cluster, write the outputs).
`root`, `LR phase`: a stage shorter than MIN_MS is called again inside the probe until the calls add up to it, and its time is the
fastest call (WARM). `cold x` is the ratio of the first call of the root alone, `cluster()` and `dtwc_cl` are single cold calls.
Environment (run.sh passes it through):
  REPEATS=3  SIZES="100 200 400 800 1600 3200"  REGIMES="noise line"  FLAVORS="off on"  THREADS=18  MIN_MS=300  E2E=1 (0: skip dtwc_cl)
  HEAD_MIN_N     unset: the head probe uses the threshold its object was built with (the product's constant at build time; the header and
                 every transcript say which). Set: the probe's LR_OMP_MIN_N knob, evaluate_dual forks from this N up. The sweep that finds the
                 crossover sets it to 1 (every N forks). The head dtwc_cl has the product's compiled-in constant whatever this says.
  HEAD_LOOP2     unset: both loops as the product. 0: the second loop (g[j], N*k work) stays serial in the head probe: the experiment the
                 one constant cannot make.
Prints, per cell, base and head times and their ratio (base / head: > 1 = head faster), then the registered bands:
  B1  flavor off, the root at N >= 800, 18 threads: >= 2x
  B2  every cell (flavor, regime, N), the root: >= 0.95x
  B3  flavor on (Kelley), the root: >= 0.95x                      (a subset of B2, shown apart)
  B4  every base / head pair bit-identical: the input, root, exact and cluster() fields (bounds, iterations, nodes, multiplier hash,
      medoids, labels, the FastPAM bound) and the same dtwc_cl total cost
A missing stage (a crash, a timeout) is NaN and fails B1-B3; it also fails B4. Completed cells are appended to cells.jsonl as they finish, and
the report is written from whatever finished, also after Ctrl-C.
"""
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

from build_kit import BIN, KIT, TREES

REPEATS = int(os.environ.get("REPEATS", "3"))
SIZES = [int(x) for x in os.environ.get("SIZES", "100 200 400 800 1600 3200").split()]
REGIMES = os.environ.get("REGIMES", "noise line").split()
FLAVORS = os.environ.get("FLAVORS", "off on").split()
THREADS = os.environ.get("THREADS", "18")
HEAD_MIN_N = os.environ.get("HEAD_MIN_N") or None
HEAD_LOOP2 = os.environ.get("HEAD_LOOP2") or None
MIN_MS = os.environ.get("MIN_MS", "300")
E2E = os.environ.get("E2E", "1") == "1"
AA = os.environ.get("AA") == "1"  # a null experiment: the "head" slot runs the base probe again (base against itself), to see what the harness alone does to a ratio
K = {"noise": 4, "line": 3}
# Base and head HiGHS-ON trees each built their own libhighs.1.dylib from the same sources and flags, and the two differ by 1-3 % in speed on the Kelley
# root (diag_lib.py: the same probe on one or the other dylib), which would read as a 3 % loss of the head below the threshold. Every HiGHS-ON run therefore
# loads the one dylib below (DYLD_LIBRARY_PATH), the base tree's, whichever binary it is. HIGHS_LIB=none turns this off.
HIGHS_LIB = None if os.environ.get("HIGHS_LIB") == "none" else (os.environ.get("HIGHS_LIB") or str(TREES[("base", "on")][1] / "lib"))
ROOT_STAGE = {"off": ("sub", "root_subgradient"), "on": ("kel", "root_kelley")}
# what differs between runs and builds without meaning a different result: timings, the thread count, and what the probe says about its knobs
HARNESS = {"ms", "ms_med", "ms_first", "reps", "fill_ms", "fastpam_ms", "copy_ms", "threads", "min_n_knob", "min_n_object", "loop2_object"}
CAP_S = 3600
NAN = float("nan")


def cli_path(version, flavor):
    return TREES[(version, flavor)][1] / "bin" / "dtwc_cl"


def args_for(regime, n):
    return ["-i", str(KIT / "data" / f"{regime}_N{n}.csv"), "--skip-rows", "1", "--skip-cols", "1", "-k", str(K[regime])]


def conditions():
    """power source, battery, thermal state and the busiest processes, one line each (logged at the start and at the end of a sweep)"""
    out = []
    for cmd, label in ((["pmset", "-g", "batt"], "power"), (["pmset", "-g", "therm"], "thermal")):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            out.append(f"{label}: " + " | ".join(l.strip() for l in r.stdout.strip().splitlines()[:3]))
        except (subprocess.TimeoutExpired, OSError):
            out.append(f"{label}: unreadable")
    try:
        r = subprocess.run(["ps", "-Ao", "pcpu,comm", "-r"], capture_output=True, text=True, timeout=20)
        top = [l.strip() for l in r.stdout.splitlines()[1:5]]
        out.append("busiest processes (%cpu command): " + " | ".join(top))
    except (subprocess.TimeoutExpired, OSError):
        out.append("busiest processes: unreadable")
    return out


def idle_pct():
    """system-wide idle CPU over the last second, by `top` (a second sample: the first is since boot); None if it cannot be read"""
    try:
        r = subprocess.run(["top", "-l", "2", "-s", "1", "-n", "0"], capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None
    for line in reversed(r.stdout.splitlines()):
        m = re.search(r"CPU usage:.*?([\d.]+)% idle", line)
        if m:
            return float(m.group(1))
    return None


def lib_env(flavor):
    return {"DYLD_LIBRARY_PATH": HIGHS_LIB} if (flavor == "on" and HIGHS_LIB) else {}


def probe_env(version, stages, flavor="off"):
    env = {**os.environ, "OMP_NUM_THREADS": THREADS, "LRPROBE_MIN_MS": MIN_MS, "LRPROBE_STAGES": stages, **lib_env(flavor)}
    env.pop("LR_OMP_MIN_N", None)
    env.pop("LR_OMP_LOOP2", None)
    if version == "head":
        if HEAD_MIN_N is not None:
            env["LR_OMP_MIN_N"] = HEAD_MIN_N
        if HEAD_LOOP2 is not None:
            env["LR_OMP_LOOP2"] = HEAD_LOOP2
    return env


def parse_stages(text):
    stages = {}
    for line in text.splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        stages[j["stage"]] = j
    return stages


def run_probe(version, flavor, regime, n, stages=None):
    stages = stages if stages is not None else f"{ROOT_STAGE[flavor][0]},exact,cli"
    load = os.getloadavg()[0]
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(BIN / f"probe_{'base' if AA else version}_{flavor}"), *args_for(regime, n), "-m", "lrcore"],
                           capture_output=True, text=True, env=probe_env(version, stages, flavor), timeout=CAP_S)
    except subprocess.TimeoutExpired:
        return {"rc": "timeout", "wall_s": CAP_S, "load": load, "stages": {}, "stderr": ""}
    return {"rc": r.returncode, "wall_s": time.perf_counter() - t0, "load": load, "stages": parse_stages(r.stdout), "stderr": r.stderr}


def run_cli(version, flavor, regime, n, scratch):
    out = scratch / f"cli_{version}_{flavor}"
    shutil.rmtree(out, ignore_errors=True)
    env = {**os.environ, "OMP_NUM_THREADS": THREADS, **lib_env(flavor)}
    load = os.getloadavg()[0]
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(cli_path(version, flavor)), *args_for(regime, n), "-m", "lrcore", "-o", str(out)],
                           capture_output=True, text=True, env=env, timeout=CAP_S)
    except subprocess.TimeoutExpired:
        shutil.rmtree(out, ignore_errors=True)
        return {"rc": "timeout", "wall_s": CAP_S, "load": load, "cost": None}
    wall = time.perf_counter() - t0
    shutil.rmtree(out, ignore_errors=True)
    cost = next((l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if l.strip().startswith("Total cost")), None)
    return {"rc": r.returncode, "wall_s": wall, "load": load, "cost": cost}


def canon(stages):
    return {s: {k: v for k, v in j.items() if k not in HARNESS} for s, j in stages.items()}


def field(stages, name, key="ms"):
    j = stages.get(name)
    return j[key] if j and key in j else NAN


def metrics(res, flavor):
    st = res["stages"]
    root = ROOT_STAGE[flavor][1]
    return {"root": field(st, root), "root_cold": field(st, root, "ms_first"), "lr": field(st, "exact"), "cluster": field(st, "cli_path")}


def med(xs):
    return statistics.median(xs) if xs else NAN


def ratio(b, h):
    return b / h if h and h == h and b == b else NAN


def fmt_t(v):
    return f"{v:9.2f}" if v < 1e5 else f"{v / 1000:8.1f}s"


def nan(x):
    return x != x


def make_report(header, cells, mismatches, idles, loads):
    lines = list(header)
    lines.append("")
    lines.append(f"{'flavor':7}{'regime':7}{'N':>5} | {'root ms (warm)':^35} | {'LR phase ms':^28} | {'cluster() ms':^28} | {'dtwc_cl s':^24} | cert iters nodes | bits")
    lines.append(f"{'':19} | {'base':>9}{'head':>9}{'x':>8}{'cold x':>9} | {'base':>9}{'head':>9}{'x':>8}   | {'base':>9}{'head':>9}{'x':>8}   | {'base':>7}{'head':>7}{'x':>8} |")
    for c in cells:
        parts = [f"{c['flavor']:7}{c['regime']:7}{c['N']:>5}"]
        d = c["root"]
        parts.append(f"{fmt_t(med(d['base']))}{fmt_t(med(d['head']))}{d['ratio']:8.2f}{c['root_cold']['ratio']:9.2f}")
        for key in ("lr", "cluster"):
            d = c[key]
            parts.append(f"{fmt_t(med(d['base']))}{fmt_t(med(d['head']))}{d['ratio']:8.2f}  ")
        if E2E:
            d = c["e2e"]
            parts.append(f"{med(d['base']):7.2f}{med(d['head']):7.2f}{d['ratio']:8.2f}")
        else:
            parts.append(f"{'-':>22}")
        i = c["info"]
        parts.append(f"{str(i.get('cert'))[:1]:>4} {i.get('iters')!s:>5} {i.get('nodes')!s:>5}")
        parts.append("same" if c["identical"] else "DIFFER")
        lines.append(" | ".join(parts))
    lines.append("")
    lines.append("times are medians over the repeats; x = base / head (> 1: head faster); a pair of repeats ran base, head, then head, base, ...")

    def worst(cs):
        bad = [c for c in cs if nan(c["root"]["ratio"])]
        if bad:
            c = bad[0]
            return NAN, f"{c['flavor']} {c['regime']} N={c['N']}: a stage is missing"
        c = min(cs, key=lambda c: c["root"]["ratio"])
        return c["root"]["ratio"], f"{c['flavor']} {c['regime']} N={c['N']}"

    def verdict(r, floor):
        return "PASS" if (not nan(r) and r >= floor) else "FAIL"

    big_off = [c for c in cells if c["flavor"] == "off" and c["N"] >= 800]
    lines.append("")
    if big_off:
        r, where = worst(big_off)
        lines.append(f"B1 HiGHS-OFF subgradient root >= 2x at N >= 800 ({THREADS} threads): lowest x{r:.2f} ({where})  -> {verdict(r, 2.0)}")
    else:
        lines.append("B1 not evaluated (no flavor off at N >= 800 in this run)")
    if cells:
        r, where = worst(cells)
        lines.append(f"B2 no cell below 0.95x on the root: lowest x{r:.2f} ({where})  -> {verdict(r, 0.95)}")
    on = [c for c in cells if c["flavor"] == "on"]
    if on:
        r, where = worst(on)
        lines.append(f"B3 HiGHS-ON Kelley root not below 0.95x: lowest x{r:.2f} ({where})  -> {verdict(r, 0.95)}")
    # the crossover: with every N forking (threshold 1) the ratio at N is the fork's gain at N alone. Only the HiGHS-OFF cells pick the value: the
    # Kelley root is dominated by HiGHS's LP solves, its ratio sits near 1 within noise at every N, and B3 is what judges it.
    firsts = []
    for flavor, regime in sorted({(c["flavor"], c["regime"]) for c in cells}):
        cs = sorted((c for c in cells if c["flavor"] == flavor and c["regime"] == regime), key=lambda c: c["N"])
        first = next((c["N"] for i, c in enumerate(cs) if all(d["root"]["ratio"] >= 1.0 for d in cs[i:])), None)
        lines.append(f"crossover {flavor:3} {regime:5}: the root ratio is >= 1.0 at every swept N from N = {first if first else 'none (not in this sweep)'}"
                     f"{'' if flavor == 'off' else '   (not used: HiGHS-dominated)'}")
        if flavor == "off":
            firsts.append(first)
    forks_everywhere = bool(cells) and all(c.get("head_min_n") is not None and c["head_min_n"] <= 1 for c in cells)
    if forks_everywhere and firsts and all(firsts):
        lines.append(f"  every N forked in this run: the smallest swept N from which every HiGHS-OFF cell gains = {max(firsts)}, the threshold's upper candidate "
                     f"(the true crossover lies between it and the swept N below it)")
    elif not forks_everywhere:
        lines.append("  (the head probe's threshold is above 1: below it the head ran serial, so these lines describe this run, not the fork alone)")
    lines.append(f"B4 bit-identical base / head pairs: {sum(c['identical'] for c in cells)} of {len(cells)} cells identical"
                 f"{'' if not mismatches else '; MISMATCHES: ' + str(mismatches)}  -> {'PASS' if not mismatches else 'FAIL'}")
    if idles:
        lines.append(f"hygiene: system idle CPU sampled before each cell (nothing of the kit running): min {min(idles):.0f}%, median {statistics.median(idles):.0f}%"
                     f"{'  -> NOT QUIET: the bands are not evidence' if min(idles) < 90 else ''}")
    if loads:
        lines.append(f"         1-min load average before each run (includes the kit's own threads): min {min(loads):.1f}, median {statistics.median(loads):.1f}, max {max(loads):.1f}")
    return "\n".join(lines)


def main():
    stamp = (os.environ.get("RUN_TAG", "") + "-" if os.environ.get("RUN_TAG") else "") + time.strftime("%Y%m%d-%H%M%S")
    run_dir = KIT / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    for version in ("base", "head"):
        for flavor in ("on", "off"):
            if not (BIN / f"probe_{version}_{flavor}").exists():
                sys.exit("a probe is missing: run `uv run --no-project python build_kit.py`")
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    ncpu = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip()
    # what threshold does the head probe's object use under this environment? (the input stage only: no LR stage runs)
    resolved = run_probe("head", FLAVORS[0], REGIMES[0], min(SIZES), stages="none")["stages"].get("input", {})
    head_min_n, head_loop2 = resolved.get("min_n_object"), resolved.get("loop2_object")
    header = [f"lr-omp timing kit {stamp}: {cpu}, {ncpu} cores, {platform.platform()}, OMP_NUM_THREADS={THREADS}{'  *** AA: base against itself ***' if AA else ''}",
              f"REPEATS={REPEATS} SIZES={SIZES} REGIMES={REGIMES} FLAVORS={FLAVORS} MIN_MS={MIN_MS} E2E={int(E2E)}",
              f"head probe object: kParallelMinN = {head_min_n} (HEAD_MIN_N {'= ' + HEAD_MIN_N if HEAD_MIN_N else 'unset: the value it was built with'}), "
              f"loop 2 {'forked as loop 1' if head_loop2 == 1 else 'SERIAL' if head_loop2 == 0 else '?'} (HEAD_LOOP2 {'= ' + HEAD_LOOP2 if HEAD_LOOP2 else 'unset'}); "
              f"the head dtwc_cl has the product's compiled-in constant",
              f"HiGHS-ON runs load one libhighs for base and head: {HIGHS_LIB}",
              f"load average at start: {os.getloadavg()}, system idle {idle_pct()}%", *conditions()]
    print("\n".join(header), flush=True)

    cells, loads, idles, mismatches = [], [], [], []
    try:
        for regime in REGIMES:
            for n in SIZES:
                for flavor in FLAVORS:
                    idle = idle_pct()
                    if idle is not None:
                        idles.append(idle)
                    base_m, head_m, base_e, head_e, pair_ok, info = [], [], [], [], [], {}
                    for rep in range(REPEATS):
                        order = ("base", "head") if rep % 2 == 0 else ("head", "base")
                        probes = {v: run_probe(v, flavor, regime, n) for v in order}
                        for v in order:
                            loads.append(probes[v]["load"])
                        b, h = probes["base"], probes["head"]
                        ok = b["rc"] == 0 and h["rc"] == 0 and bool(b["stages"]) and canon(b["stages"]) == canon(h["stages"])
                        pair_ok.append(ok)
                        if not ok:
                            mismatches.append((flavor, regime, n, rep, b["rc"], h["rc"]))
                        base_m.append(metrics(b, flavor))
                        head_m.append(metrics(h, flavor))
                        rj = h["stages"].get(ROOT_STAGE[flavor][1], {})
                        ej = h["stages"].get("exact", {})
                        info = {"cert": rj.get("certified"), "iters": rj.get("iterations"), "nodes": ej.get("nodes")}
                        if E2E:
                            clis = {v: run_cli(v, flavor, regime, n, run_dir) for v in order}
                            for v in order:
                                loads.append(clis[v]["load"])
                            base_e.append(clis["base"]["wall_s"] if clis["base"]["rc"] == 0 else NAN)
                            head_e.append(clis["head"]["wall_s"] if clis["head"]["rc"] == 0 else NAN)
                            if clis["base"]["cost"] != clis["head"]["cost"] or clis["base"]["rc"] != clis["head"]["rc"] or clis["base"]["rc"] != 0:
                                pair_ok[-1] = False
                                mismatches.append((flavor, regime, n, rep, "cli", clis["base"]["rc"], clis["head"]["rc"], clis["base"]["cost"], clis["head"]["cost"]))
                    cell = {"flavor": flavor, "regime": regime, "N": n, "info": info, "identical": all(pair_ok), "repeats": REPEATS,
                            "head_min_n": head_min_n, "idle_before": idle}
                    for key in ("root", "root_cold", "lr", "cluster"):
                        bs, hs = [m[key] for m in base_m], [m[key] for m in head_m]
                        cell[key] = {"base": bs, "head": hs, "ratio": ratio(med(bs), med(hs)),
                                     "pair_ratios": [ratio(b_, h_) for b_, h_ in zip(bs, hs)]}
                    if E2E:
                        cell["e2e"] = {"base": base_e, "head": head_e, "ratio": ratio(med(base_e), med(head_e)),
                                       "pair_ratios": [ratio(b_, h_) for b_, h_ in zip(base_e, head_e)]}
                    cells.append(cell)
                    with open(run_dir / "cells.jsonl", "a") as f:
                        f.write(json.dumps(cell) + "\n")
                    print(f"  done {flavor} {regime} N={n}: root x{cell['root']['ratio']:.2f}, idle before {idle}%", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted: the report covers the cells that finished", flush=True)
    text = make_report(header, cells, mismatches, idles, loads) + "\nconditions at the end:\n  " + "\n  ".join(conditions())
    (run_dir / "report.txt").write_text(text + "\n")
    (run_dir / "results.json").write_text(json.dumps(cells, indent=1) + "\n")
    print("\n" + text)
    print(f"\nraw: {run_dir}/results.json (cells.jsonl as they finished), report: {run_dir}/report.txt")


if __name__ == "__main__":
    main()
