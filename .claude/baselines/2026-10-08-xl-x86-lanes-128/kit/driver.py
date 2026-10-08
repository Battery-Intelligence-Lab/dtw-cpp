"""XL timing kit driver: x86-64 lanes at 128 bytes (cand) against 64 (base), clang++ and cl (stdlib only).

  uv run --no-project python driver.py run [--rounds 3] [--compilers clang,msvc]   the measurement (quiet machine)
  uv run --no-project python driver.py smoke                                     one tiny round: the kit works, no numbers
  uv run --no-project python driver.py report [raw file]                         medians, ratios, matrices, band verdicts

`run` appends every probe line verbatim to raw.txt (`smoke` to raw_smoke.txt), with load records before and after
(load.sh -> load.txt). Each round alternates which build goes first and runs both compilers: single thread pinned
to logical CPU 12 (a P-core), the lane function alone, f64/f32 x L1/squared x L 100 and 1000 x unbanded and band
L/10 (ns per cell); then three fills through Problem's fill loop and schedule at the OpenMP default (24 threads):
f64 L1 N 2000 L 100 unbanded, N 500 L 1000 band 100, ragged N 1000 L 90-110 (the per-pair route: a control),
Gcell/s with a matrix hash. `report` writes report.md too.
"""
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import time

KIT = pathlib.Path(__file__).resolve().parent
CPU = 12  # a P-core (efficiency class 1); P-cores are logical CPUs 0, 1, 10-13, 22, 23
BUILDS = {"clang": ("clang_base", "clang_cand"), "msvc": ("msvc_base", "msvc_cand")}
SHAPES = [(t, d, L, b) for t in ("f64", "f32") for d in ("L1", "Sq") for L in (100, 1000) for b in (-1, L // 10)]
FILLS = [("f64", "L1", 2000, 100, -1, 0), ("f64", "L1", 500, 1000, 100, 0), ("f64", "L1", 1000, 100, -1, 1)]
SMOKE_SHAPES = [(t, d, 100, b) for t in ("f64", "f32") for d in ("L1", "Sq") for b in (-1, 10)]
SMOKE_FILLS = [("f64", "L1", 100, 100, -1, 0), ("f64", "L1", 40, 1000, 100, 0), ("f64", "L1", 60, 100, -1, 1)]
# The registered band (the orchestrator's, fixed before any run), per compiler.
BAND_SINGLE, BAND_FILL, BAND_RAGGED = 1.10, 1.10, (0.97, 1.03)
# cl's candidate failed the pre-timing assembly check (a spill or reload added on a loop-carried value: at 32 floats
# every lane's `left` goes to the stack, 5 of 16 lanes at base); its rounds are timed for information only.
ASM_DISQUALIFIED = {"msvc": "assembly: cl's f32 lanes keep every lane's left on the stack at W 32 (5 of 16 at W 16)"}


def probe(build, *args):
    exe = KIT / "bin" / build / "xprobe.exe"
    out = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    return out.returncode, (out.stdout.strip() or out.stderr.strip())


def load(label):
    sh = shutil.which("sh") or "C:/Users/engs2321/AppData/Local/Programs/Git/usr/bin/sh.exe"
    try:
        subprocess.run([sh, str(KIT / "load.sh"), label], check=False)
    except OSError as e:
        print(f"no load record ({label}): {e}; run load.sh by hand", flush=True)


def run(smoke=False, rounds=3, compilers=("clang", "msvc")):
    raw = KIT / ("raw_smoke.txt" if smoke else "raw.txt")
    shapes, fills = (SMOKE_SHAPES, SMOKE_FILLS) if smoke else (SHAPES, FILLS)
    reps, ms, fill_reps = (2, 10, 1) if smoke else (5, 100, 3)
    if smoke:
        rounds = 1
    tag = "smoke" if smoke else "run"
    start = time.time()
    with open(raw, "a") as f:
        f.write(f"# {tag} {time.strftime('%Y-%m-%d %H:%M:%S')} rounds={rounds} compilers={','.join(compilers)} "
                f"cpu={CPU}{' SMOKE: not a measurement' if smoke else ''}\n")
        for c in compilers:
            for b in BUILDS[c]:
                f.write(f"# info {b}: {probe(b, 'info')[1]}\n")
        f.flush()
        load(f"{tag} before")
        for r in range(rounds):
            for c in compilers:
                order = BUILDS[c] if r % 2 == 0 else BUILDS[c][::-1]
                for t, d, L, band in shapes:
                    for b in order:
                        code, out = probe(b, "time", CPU, t, d, L, band, reps, ms)
                        f.write(f"round={r} compiler={c} build={b} exit={code} {out}\n")
                        f.flush()
                        if code:
                            sys.exit(f"probe failed: {b} time {t} {d} {L} {band} -> {code}: {out}")
                for t, d, N, L, band, ragged in fills:
                    for b in order:
                        code, out = probe(b, "fill", t, d, N, L, band, 0, fill_reps, ragged)
                        f.write(f"round={r} compiler={c} build={b} exit={code} {out}\n")
                        f.flush()
                        if code:
                            sys.exit(f"probe failed: {b} fill {t} {d} {N} {L} {band} {ragged} -> {code}: {out}")
            print(f"round {r} done, {time.time() - start:.0f} s", flush=True)
        load(f"{tag} after")
        f.write(f"# {tag} done in {time.time() - start:.0f} s\n")
    report(raw)


TIME = re.compile(r"^round=(\d+) compiler=(\w+) build=(\w+) exit=0 time (f64|f32) (L1|Sq)\s+L (\d+)\s+band (-?\d+)\s+"
                  r"W (\d+)\s+cpu \d+ calls \d+ cells \S+ ns/cell ([\d.]+) \(min ([\d.]+) max ([\d.]+)\)\s+"
                  r"clock ([\d.]+) GHz\s+cycles/row-step ([\d.]+)")
FILL = re.compile(r"^round=(\d+) compiler=(\w+) build=(\w+) exit=0 fill (f64|f32) (L1|Sq)\s+threads\s+(\d+) N (\d+)\s+"
                  r"L (\d+)\s*( ragged\+-10)? band (-?\d+)\s+W (\d+)\s+ms ([\d.]+) \(min ([\d.]+)\)\s+Gcell/s ([\d.]+)\s+"
                  r"matrix hash ([0-9a-f]+)")
CHECK = re.compile(r"^check lanes (f64|f32) (L1|Sq)\s+W=(\d+)\s+seeds (\d+): outputs (\d+)\s+hash ([0-9a-f]+)\s+lanes vs "
                   r"per-pair bitwise mismatches (\d+)\s+non-finite outputs (\d+)")
PAIRS = re.compile(r"^check pair\s+(f64|f32) (L1|Sq)\s+seeds (\d+): outputs (\d+)\s+hash ([0-9a-f]+)")


def sweep(build):
    """The bitwise sweep's lines for a build (check/<build>.txt), or None."""
    p = KIT / "check" / f"{build}.txt"
    if not p.exists():
        return None
    lanes, pairs = {}, {}
    for ln in p.read_text().splitlines():
        if m := CHECK.match(ln):
            lanes[(m[1], m[2])] = (int(m[5]), m[6], int(m[7]), int(m[8]), int(m[3]))
        elif m := PAIRS.match(ln):
            pairs[(m[1], m[2])] = (int(m[4]), m[5])
    return lanes, pairs


def report(raw=None):
    raw = pathlib.Path(raw) if raw else KIT / "raw.txt"
    text = raw.read_text().splitlines()
    smoke = any("SMOKE" in ln for ln in text if ln.startswith("#"))
    times, fills = {}, {}
    for ln in text:
        if m := TIME.match(ln):
            r, c, b, t, d, L, band, W, ns, lo, hi, ghz, cyc = m.groups()
            times.setdefault((c, t, d, int(L), int(band)), {}).setdefault(b, {})[int(r)] = (
                float(ns), int(W), float(ghz), float(cyc))
        elif m := FILL.match(ln):
            r, c, b, t, d, th, N, L, rag, band, W, ms, mn, g, h = m.groups()
            fills.setdefault((c, int(N), int(L), bool(rag), int(band), int(th)), {}).setdefault(b, {})[int(r)] = (
                float(g), h, int(W))
    med = statistics.median
    # cl has no x64 inline assembly, so its probes measure no clock: their cycles use the round's clang clock.
    clocks = {}
    for (c, *_), builds in times.items():
        for b, rounds in builds.items():
            for r, v in rounds.items():
                if v[2] > 0:
                    clocks.setdefault(r, []).append(v[2])
    clock = {r: med(v) for r, v in clocks.items()}
    out = [f"# XL report from {raw.name}" + (" -- SMOKE RUN: tiny sizes, one round; no number here is a measurement"
                                              if smoke else ""), ""]
    if clock:
        out += ["clock of CPU {} (the clang probes' dependent register-add chain, median per round): {}".format(
            CPU, ", ".join(f"round {r} {g:.3f} GHz" for r, g in sorted(clock.items()))), ""]
    for c, (base, cand) in BUILDS.items():
        keys = [k for k in times if k[0] == c]
        fkeys = [k for k in fills if k[0] == c]
        if not keys and not fkeys:
            continue
        out += [f"## {c}: candidate (cand) against base, median over rounds; ratio = base ns / cand ns (single "
                "thread) or cand Gcell/s / base Gcell/s (fill), per round", ""]
        out += ["| shape | W base/cand | base ns/cell | cand ns/cell | ratio (median) | range | cycles/row step base/cand |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        single, notes = [], []
        for k in sorted(keys, key=lambda k: (k[1], k[2], k[3], k[4])):
            b, n = times[k].get(base, {}), times[k].get(cand, {})
            reps = sorted(set(b) & set(n))
            if not reps:
                continue
            ratios = [b[r][0] / n[r][0] for r in reps]
            single.append((k, med(ratios)))
            if all(b[r][2] > 0 and n[r][2] > 0 for r in reps):
                cyc = f"{med(b[r][3] for r in reps):.2f} / {med(n[r][3] for r in reps):.2f}"
            elif all(r in clock for r in reps):
                cyc = (f"{med(b[r][0] * b[r][1] * clock[r] for r in reps):.2f} / "
                       f"{med(n[r][0] * n[r][1] * clock[r] for r in reps):.2f} (clang's clock)")
            else:
                cyc = "n/a (no clock)"
            name = f"{k[1]} {k[2]} L {k[3]} {'unbanded' if k[4] < 0 else f'band {k[4]}'}"
            out.append(f"| {name} | {b[reps[0]][1]}/{n[reps[0]][1]} | {med(b[r][0] for r in reps):.5f} | "
                       f"{med(n[r][0] for r in reps):.5f} | {med(ratios):.3f} | {min(ratios):.3f}-{max(ratios):.3f} | "
                       f"{cyc} |")
        out += ["", "| fill (f64 L1, OpenMP default threads) | W base/cand | base Gcell/s | cand Gcell/s | ratio (median) "
                "| range | matrices |", "| --- | --- | --- | --- | --- | --- | --- |"]
        fill_eq, fill_rag, same_all = [], [], True
        for k in sorted(fkeys, key=lambda k: (k[3], -k[1])):
            b, n = fills[k].get(base, {}), fills[k].get(cand, {})
            reps = sorted(set(b) & set(n))
            if not reps:
                continue
            ratios = [n[r][0] / b[r][0] for r in reps]
            same = all(b[r][1] == n[r][1] for r in reps) and len({b[r][1] for r in reps}) == 1
            same_all = same_all and same
            (fill_rag if k[3] else fill_eq).append(med(ratios))
            name = (f"N {k[1]} L {k[2]}{' +-10 ragged' if k[3] else ''} "
                    f"{'unbanded' if k[4] < 0 else f'band {k[4]}'}, {k[5]} threads")
            out.append(f"| {name} | {b[reps[0]][2]}/{n[reps[0]][2]} | {med(b[r][0] for r in reps):.2f} | "
                       f"{med(n[r][0] for r in reps):.2f} | {med(ratios):.3f} | {min(ratios):.3f}-{max(ratios):.3f} | "
                       f"{'identical' if same else 'DIFFER'} |")
        # Bitwise: the sweeps (check/<build>.txt) and the fill matrices.
        sb, sc = sweep(base), sweep(cand)
        if sb and sc:
            equal = sb[0].keys() == sc[0].keys() and all(sb[0][k][:2] == sc[0][k][:2] for k in sb[0]) and \
                    sb[1] == sc[1]
            mism = sum(v[2] for v in sb[0].values()) + sum(v[2] for v in sc[0].values())
            outputs = sum(v[0] for v in sc[0].values())
            bitwise = equal and mism == 0 and same_all
            notes.append(f"sweep: {outputs:,} lane outputs per build, hashes base == cand: {'yes' if equal else 'NO'}, "
                         f"lanes vs per-pair mismatches (base + cand): {mism}; fill matrices equal: "
                         f"{'yes' if same_all else 'NO'}")
        else:
            bitwise = same_all
            notes.append(f"sweep files missing for {base}/{cand}; fill matrices equal: {'yes' if same_all else 'NO'}")
        out += [""] + [f"- {x}" for x in notes]
        f64l1 = [v for k, v in single if k[1] == "f64" and k[2] == "L1" and k[4] < 0]
        if f64l1:
            out.append(f"- f64 L1 unbanded (V3's prediction about 2x; no band): "
                       f"{', '.join(f'{v:.3f}' for v in f64l1)}")
        if single and fill_eq and fill_rag:
            s_ok = min(v for _, v in single) >= BAND_SINGLE and len(single) == len(SHAPES)
            f_ok = min(fill_eq) >= BAND_FILL and len(fill_eq) == 2
            r_ok = all(BAND_RAGGED[0] <= v <= BAND_RAGGED[1] for v in fill_rag)
            verdict = "PASS" if s_ok and f_ok and r_ok and bitwise else "FALSIFIED"
            out += [f"- band: every single-thread shape >= {BAND_SINGLE}x: min {min(v for _, v in single):.3f} "
                    f"({len(single)} shapes) -> {'pass' if s_ok else 'fail'}; both equal-length fills >= {BAND_FILL}x: "
                    f"{', '.join(f'{v:.3f}' for v in fill_eq)} -> {'pass' if f_ok else 'fail'}; ragged fill in "
                    f"{BAND_RAGGED[0]}-{BAND_RAGGED[1]}x: {', '.join(f'{v:.3f}' for v in fill_rag)} -> "
                    f"{'pass' if r_ok else 'fail'}; bitwise -> {'pass' if bitwise else 'fail'}",
                    f"- **verdict {c}: {verdict}**" + (" (smoke run: not a verdict)" if smoke else "")
                    + (f" -- but {c} was disqualified before timing by the {ASM_DISQUALIFIED[c]}; these rounds are "
                       "information" if c in ASM_DISQUALIFIED else "")]
        out.append("")
    (raw.parent / ("report_smoke.md" if smoke else "report.md")).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        args = sys.argv[2:]
        rounds = int(args[args.index("--rounds") + 1]) if "--rounds" in args else 3
        comps = tuple(args[args.index("--compilers") + 1].split(",")) if "--compilers" in args else ("clang", "msvc")
        run(False, rounds, comps)
    elif cmd == "smoke":
        run(True)
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
