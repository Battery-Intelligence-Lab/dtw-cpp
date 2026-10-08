"""MV timing kit driver: cl's lanes packed (cand: a value-returning min, 128-byte lanes under cl, contraction off in
dtw_lanes.cpp) against cl's scalar lanes (base: design-2.0 b3b3a57e, 64-byte lanes). Stdlib only.

  uv run --no-project python driver.py run [--rounds 3]     the measurement (quiet machine only)
  uv run --no-project python driver.py smoke                 one tiny round: the kit works, no numbers
  uv run --no-project python driver.py report [raw file]     medians, ratios, matrices, the band's verdict

`run` appends every probe line verbatim to raw.txt (`smoke` to raw_smoke.txt), with load records before and after
(load.sh -> load.txt). Each round first reads the clock of CPU 12 (bin/clk.exe, a clang build of tools/clk.cpp: a
dependent register-add chain; cl has no x64 inline assembly), then alternates which build goes first: single thread
pinned to logical CPU 12 (a P-core), the lane function alone, f64/f32 x L1/squared x L 100 and 1000 x unbanded and
band L/10 (ns per cell); then three fills through Problem's fill loop and schedule at the OpenMP default (24
threads): f64 L1 N 2000 L 100 unbanded, N 500 L 1000 band 100, ragged N 1000 L 90-110 (the per-pair route: a
control), Gcell/s with a matrix hash. `report` writes report.md too.

clang is not timed: its dtw_lanes.cpp assembly is identical to base (asm/clang_check.txt), which the band accepts.
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
BUILDS = {"msvc": ("msvc_base", "msvc_cand")}
SHAPES = [(t, d, L, b) for t in ("f64", "f32") for d in ("L1", "Sq") for L in (100, 1000) for b in (-1, L // 10)]
FILLS = [("f64", "L1", 2000, 100, -1, 0), ("f64", "L1", 500, 1000, 100, 0), ("f64", "L1", 1000, 100, -1, 1)]
SMOKE_SHAPES = [(t, d, 100, b) for t in ("f64", "f32") for d in ("L1", "Sq") for b in (-1, 10)]
SMOKE_FILLS = [("f64", "L1", 100, 100, -1, 0), ("f64", "L1", 40, 1000, 100, 0), ("f64", "L1", 60, 100, -1, 1)]
# The registered band (the orchestrator's, fixed before any run): under cl every single-thread shape >= 2.0x base,
# both 24-thread equal-length fills >= 1.5x, the ragged fill within 0.97-1.03x, outputs bitwise equal; clang
# unchanged (identical assembly, or timed within 0.97-1.03x on every shape).
BAND_SINGLE, BAND_FILL, BAND_RAGGED = 2.0, 1.5, (0.97, 1.03)
CLANG_CHECK = KIT / "asm" / "clang_check.txt"


def probe(build, *args):
    exe = KIT / "bin" / build / "xprobe.exe"
    out = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    return out.returncode, (out.stdout.strip() or out.stderr.strip())


def clock():
    """Median add-reg clock of CPU 12 over clk.exe's five readings, in GHz, or 0 without the probe."""
    exe = KIT / "bin" / "clk.exe"
    if not exe.exists():
        return 0.0
    out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
    vals = [float(v) for v in re.findall(r"add-reg ([\d.]+) GHz", out)]
    return statistics.median(vals) if vals else 0.0


def load(label):
    sh = shutil.which("sh") or "C:/Users/engs2321/AppData/Local/Programs/Git/usr/bin/sh.exe"
    try:
        subprocess.run([sh, str(KIT / "load.sh"), label], check=False)
    except OSError as e:
        print(f"no load record ({label}): {e}; run load.sh by hand", flush=True)


def run(smoke=False, rounds=3):
    raw = KIT / ("raw_smoke.txt" if smoke else "raw.txt")
    shapes, fills = (SMOKE_SHAPES, SMOKE_FILLS) if smoke else (SHAPES, FILLS)
    reps, ms, fill_reps = (2, 10, 1) if smoke else (5, 100, 3)
    if smoke:
        rounds = 1
    tag = "smoke" if smoke else "run"
    start = time.time()
    with open(raw, "a") as f:
        f.write(f"# {tag} {time.strftime('%Y-%m-%d %H:%M:%S')} rounds={rounds} cpu={CPU}"
                f"{' SMOKE: not a measurement' if smoke else ''}\n")
        for c, builds in BUILDS.items():
            for b in builds:
                f.write(f"# info {b}: {probe(b, 'info')[1]}\n")
        f.flush()
        load(f"{tag} before")
        for r in range(rounds):
            f.write(f"round={r} clock {clock():.3f} GHz\n")
            for c, builds in BUILDS.items():
                order = builds if r % 2 == 0 else builds[::-1]
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
                  r"W (\d+)\s+cpu \d+ calls \d+ cells \S+ ns/cell ([\d.]+) \(min ([\d.]+) max ([\d.]+)\)")
FILL = re.compile(r"^round=(\d+) compiler=(\w+) build=(\w+) exit=0 fill (f64|f32) (L1|Sq)\s+threads\s+(\d+) N (\d+)\s+"
                  r"L (\d+)\s*( ragged\+-10)? band (-?\d+)\s+W (\d+)\s+ms ([\d.]+) \(min ([\d.]+)\)\s+Gcell/s ([\d.]+)\s+"
                  r"matrix hash ([0-9a-f]+)")
CLOCK = re.compile(r"^round=(\d+) clock ([\d.]+) GHz")
CHECK = re.compile(r"^check lanes (f64|f32) (L1|Sq)\s+W=(\d+)\s+seeds (\d+): outputs (\d+)\s+hash ([0-9a-f]+)\s+lanes vs "
                   r"per-pair bitwise mismatches (\d+)\s+non-finite outputs (\d+)"
                   r"(?:\s+outside the route bound (\d+)\s+max dev ([\d.]+))?")
PAIRS = re.compile(r"^check pair\s+(f64|f32) (L1|Sq)\s+seeds (\d+): outputs (\d+)\s+hash ([0-9a-f]+)")


def sweep(build):
    """The bitwise sweep's lines for a build (check/<build>.txt), or None."""
    p = KIT / "check" / f"{build}.txt"
    if not p.exists():
        return None
    lanes, pairs = {}, {}
    for ln in p.read_text().splitlines():
        if m := CHECK.match(ln):
            lanes[(m[1], m[2])] = (int(m[5]), m[6], int(m[7]), int(m[8]), int(m[3]),
                                   int(m[9]) if m[9] else None, float(m[10]) if m[10] else None)
        elif m := PAIRS.match(ln):
            pairs[(m[1], m[2])] = (int(m[4]), m[5])
    return lanes, pairs


def report(raw=None):
    raw = pathlib.Path(raw) if raw else KIT / "raw.txt"
    text = raw.read_text().splitlines()
    smoke = any("SMOKE" in ln for ln in text if ln.startswith("#"))
    times, fills, clock_of = {}, {}, {}
    for ln in text:
        if m := TIME.match(ln):
            r, c, b, t, d, L, band, W, ns, lo, hi = m.groups()
            times.setdefault((c, t, d, int(L), int(band)), {}).setdefault(b, {})[int(r)] = (float(ns), int(W))
        elif m := FILL.match(ln):
            r, c, b, t, d, th, N, L, rag, band, W, ms, mn, g, h = m.groups()
            fills.setdefault((c, int(N), int(L), bool(rag), int(band), int(th)), {}).setdefault(b, {})[int(r)] = (
                float(g), h, int(W))
        elif m := CLOCK.match(ln):
            if float(m[2]) > 0:
                clock_of[int(m[1])] = float(m[2])
    med = statistics.median
    out = [f"# MV report from {raw.name}" + (" -- SMOKE RUN: tiny sizes, one round; no number here is a measurement"
                                              if smoke else ""), ""]
    if clock_of:
        out += ["clock of CPU {} (bin/clk.exe, a dependent register-add chain, median of five per round): {}".format(
            CPU, ", ".join(f"round {r} {g:.3f} GHz" for r, g in sorted(clock_of.items()))), ""]
    verdicts = {}
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
            if all(r in clock_of for r in reps):
                cyc = (f"{med(b[r][0] * b[r][1] * clock_of[r] for r in reps):.2f} / "
                       f"{med(n[r][0] * n[r][1] * clock_of[r] for r in reps):.2f}")
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
            for k in sorted(sc[0]):
                v, b0 = sc[0][k], sb[0].get(k)
                bound = "" if v[5] is None else f", outside the route bound {v[5]}, max dev {v[6]:.3f} (bound 2)"
                notes.append(f"  cand {k[0]} {k[1]} W {v[4]}: hash {'= base' if b0 and b0[1] == v[1] else 'differs from base'}"
                             f", mismatches against per-pair {v[2]:,} of {v[0]:,}{bound}")
        else:
            bitwise = False
            notes.append(f"sweep files missing for {base}/{cand} (check/<build>.txt): bitwise not shown; fill matrices "
                         f"equal: {'yes' if same_all else 'NO'}")
        out += [""] + [f"- {x}" for x in notes]
        if single and fill_eq and fill_rag:
            s_ok = min(v for _, v in single) >= BAND_SINGLE and len(single) == len(SHAPES)
            f_ok = min(fill_eq) >= BAND_FILL and len(fill_eq) == 2
            r_ok = all(BAND_RAGGED[0] <= v <= BAND_RAGGED[1] for v in fill_rag)
            verdicts[c] = s_ok and f_ok and r_ok and bitwise
            out += [f"- band: every single-thread shape >= {BAND_SINGLE}x: min {min(v for _, v in single):.3f} "
                    f"({len(single)} shapes) -> {'pass' if s_ok else 'fail'}; both equal-length fills >= {BAND_FILL}x: "
                    f"{', '.join(f'{v:.3f}' for v in fill_eq)} -> {'pass' if f_ok else 'fail'}; ragged fill in "
                    f"{BAND_RAGGED[0]}-{BAND_RAGGED[1]}x: {', '.join(f'{v:.3f}' for v in fill_rag)} -> "
                    f"{'pass' if r_ok else 'fail'}; bitwise -> {'pass' if bitwise else 'fail'}"]
            if sb and sc:  # facts for a reading of `bitwise` the band does not register: L1 bitwise, squared bounded
                l1 = all(sc[0][k][2] == 0 and sb[0][k][:2] == sc[0][k][:2] for k in sc[0] if k[1] == "L1")
                sq = [sc[0][k][5] for k in sc[0] if k[1] == "Sq"]
                out.append(f"- not the band, for the record: speed clauses -> {'pass' if s_ok and f_ok and r_ok else 'fail'}; "
                           f"L1 lanes bitwise (= per-pair, hashes = base) -> {'yes' if l1 else 'NO'}; squared lanes "
                           f"outside the route bound -> {'n/a' if None in sq else sum(sq)}")
        out.append("")
    clang_ok = CLANG_CHECK.exists() and CLANG_CHECK.read_text().startswith("identical")
    out.append(f"- clang: dtw_lanes.cpp assembly identical to base: {'yes' if clang_ok else 'NOT SHOWN'} "
               f"({CLANG_CHECK.relative_to(KIT).as_posix()}; identical assembly needs no timing under the band)")
    if "msvc" in verdicts:
        verdict = "PASS" if verdicts["msvc"] and clang_ok else "FALSIFIED"
        out.append(f"- **verdict: {verdict}**" + (" (smoke run: not a verdict)" if smoke else ""))
    (raw.parent / ("report_smoke.md" if smoke else "report.md")).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        args = sys.argv[2:]
        run(False, int(args[args.index("--rounds") + 1]) if "--rounds" in args else 3)
    elif cmd == "smoke":
        run(True)
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
