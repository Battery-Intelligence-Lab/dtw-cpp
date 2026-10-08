"""XG timing kit driver: GCC 13.3's x86-64 lanes at 128 bytes (cand) against 64 (base), in WSL (stdlib only).

Run from Git Bash on Windows; every probe runs in WSL (`wsl -d Ubuntu-24.04 -e ...`), as build_gcc.sh built it into
~/xg/bin/gcc_{base,cand}/xprobe.

  uv run --no-project python driver.py run [--rounds 3]   the measurement (quiet machine): about 4 minutes
  uv run --no-project python driver.py smoke              one tiny round: the kit works, no numbers
  uv run --no-project python driver.py check              the bitwise sweep, no timing: check/<build>.txt
  uv run --no-project python driver.py report [raw file]  medians, ratios, clocks, matrices, band verdict

`run` boots WSL and waits until systemd has started and runs no job (a timer due at boot, apt-daily-upgrade say, runs
as one), then appends every probe line verbatim to raw.txt (`smoke` to raw_smoke.txt), with load records before and
after (load.sh -> load.txt: Windows, which sees WSL as vmmemWSL, then WSL). Each round alternates which build goes
first: one thread under `taskset -c 2`, the lane function alone, f64/f32 x L1/squared x L 100 and 1000 x unbanded and
band L/10 (ns per cell; the host may run vCPU 2 on a P- or an E-core, so every measurement carries its clock); then
three fills through Problem's fill loop and schedule at the OpenMP default (all 24 vCPUs): f64 L1 N 2000 L 100
unbanded, N 500 L 1000 band 100, ragged N 1000 L 90-110 (the per-pair route: a control), Gcell/s with a matrix hash.
`report` writes report.md too.
"""
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import time

KIT = pathlib.Path(__file__).resolve().parent
DISTRO = "Ubuntu-24.04"
VCPU = 2  # any vCPU will do (each is a host thread, not a core); vCPU 0 takes more of the guest's interrupts
BUILDS = ("gcc_base", "gcc_cand")
SHAPES = [(t, d, L, b) for t in ("f64", "f32") for d in ("L1", "Sq") for L in (100, 1000) for b in (-1, L // 10)]
FILLS = [("f64", "L1", 2000, 100, -1, 0), ("f64", "L1", 500, 1000, 100, 0), ("f64", "L1", 1000, 100, -1, 1)]
SMOKE_SHAPES = [(t, d, 100, b) for t in ("f64", "f32") for d in ("L1", "Sq") for b in (-1, 10)]
SMOKE_FILLS = [("f64", "L1", 100, 100, -1, 0), ("f64", "L1", 40, 1000, 100, 0), ("f64", "L1", 60, 100, -1, 1)]
# The registered band (XL's, per compiler, fixed before any run).
BAND_SINGLE, BAND_FILL, BAND_RAGGED = 1.10, 1.10, (0.97, 1.03)
# A round whose clock differs from the rounds' median by more than this is flagged; so is a probe whose own clock
# moves by more across its measurements.
CLOCK_TOL = 0.03


def wsl(*args):
    out = subprocess.run(["wsl", "-d", DISTRO, "-e", *map(str, args)], capture_output=True, text=True)
    return out.returncode, (out.stdout.strip() or out.stderr.strip())


_home = []


def exe(build):
    if not _home:
        _home.append(wsl("sh", "-c", 'printf %s "$HOME"')[1])
    return f"{_home[0]}/xg/bin/{build}/xprobe"


def probe(build, *args, pinned=False):
    return wsl(*(("taskset", "-c", VCPU) if pinned else ()), exe(build), *args)


def jobs():
    """The systemd jobs WSL runs now ('' if none)."""
    return wsl("systemctl", "list-jobs", "--no-legend")[1]


def settle():
    """Boot WSL, wait until systemd has started and runs no job (at most 10 minutes); say what was waited for."""
    script = ('systemctl is-system-running --wait >/dev/null; t=0; '
              'while [ -n "$(systemctl list-jobs --no-legend)" ] && [ $t -lt 600 ]; do sleep 5; t=$((t + 5)); done; '
              'echo "systemd $(systemctl is-system-running), waited $t s for its jobs; jobs now: '
              '$(systemctl list-jobs --no-legend | wc -l); up $(cut -d" " -f1 /proc/uptime) s"')
    return wsl("sh", "-c", script)[1]


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
        f.write(f"# {tag} {time.strftime('%Y-%m-%d %H:%M:%S')} rounds={rounds} vcpu={VCPU}"
                f"{' SMOKE: not a measurement' if smoke else ''}\n")
        f.write(f"# wsl: {settle()}\n")
        for b in BUILDS:
            f.write(f"# info {b}: {probe(b, 'info')[1]}\n")
        f.flush()
        load(f"{tag} before")
        for r in range(rounds):
            if j := jobs():
                f.write(f"# round={r} wsl jobs: {' | '.join(j.splitlines())}\n")
            order = BUILDS if r % 2 == 0 else BUILDS[::-1]
            for t, d, L, band in shapes:
                for b in order:
                    code, out = probe(b, "time", VCPU, t, d, L, band, reps, ms, pinned=True)
                    f.write(f"round={r} build={b} exit={code} {out}\n")
                    f.flush()
                    if code:
                        sys.exit(f"probe failed: {b} time {t} {d} {L} {band} -> {code}: {out}")
            for t, d, N, L, band, ragged in fills:
                for b in order:
                    code, out = probe(b, "fill", t, d, N, L, band, 0, fill_reps, ragged)
                    f.write(f"round={r} build={b} exit={code} {out}\n")
                    f.flush()
                    if code:
                        sys.exit(f"probe failed: {b} fill {t} {d} {N} {L} {band} {ragged} -> {code}: {out}")
            print(f"round {r} done, {time.time() - start:.0f} s", flush=True)
        if j := jobs():
            f.write(f"# after the rounds, wsl jobs: {' | '.join(j.splitlines())}\n")
        f.flush()
        load(f"{tag} after")
        f.write(f"# {tag} done in {time.time() - start:.0f} s\n")
    report(raw)


def check():
    """The bitwise sweep (xprobe check 3, OpenMP 8 threads) of each build into check/<build>.txt."""
    (KIT / "check").mkdir(exist_ok=True)
    for b in BUILDS:
        t0 = time.time()
        code, out = wsl("env", "OMP_NUM_THREADS=8", exe(b), "check", 3)
        (KIT / "check" / f"{b}.txt").write_text(out + "\n")
        print(f"{b}: exit {code}, {time.time() - t0:.0f} s\n{out}", flush=True)
        if code:
            sys.exit(f"check failed: {b} -> {code}")


TIME = re.compile(r"^round=(\d+) build=(\w+) exit=0 time (f64|f32) (L1|Sq)\s+L (\d+)\s+band (-?\d+)\s+W (\d+)\s+"
                  r"cpu \d+ calls \d+ cells \S+ ns/cell ([\d.]+) \(min [\d.]+ max [\d.]+\)\s+"
                  r"clock ([\d.]+) GHz \(min ([\d.]+) max ([\d.]+)\)\s+cycles/row-step ([\d.]+)")
FILL = re.compile(r"^round=(\d+) build=(\w+) exit=0 fill (f64|f32) (L1|Sq)\s+threads\s+(\d+) N (\d+)\s+L (\d+)\s*"
                  r"( ragged\+-10)? band (-?\d+)\s+W (\d+)\s+ms [\d.]+ \(min [\d.]+\)\s+Gcell/s ([\d.]+)\s+"
                  r"matrix hash ([0-9a-f]+)")
CHECK = re.compile(r"^check lanes (f64|f32) (L1|Sq)\s+W=(\d+)\s+seeds (\d+): outputs (\d+)\s+hash ([0-9a-f]+)\s+lanes vs "
                   r"per-pair bitwise mismatches (\d+)\s+non-finite outputs (\d+)\s+outside the route bound (\d+)\s+"
                   r"worst \|a-b\|/\(\(2n-1\) eps max\) (\S+)")
PAIRS = re.compile(r"^check pair\s+(f64|f32) (L1|Sq)\s+seeds (\d+): outputs (\d+)\s+hash ([0-9a-f]+)")


def sweep(build):
    """The bitwise sweep's lines for a build (check/<build>.txt): lanes {(T, D): (outputs, hash, mismatches, W,
    outside the route bound, worst ratio to (2n - 1) eps max)}, pairs {(T, D): hash}; or None."""
    p = KIT / "check" / f"{build}.txt"
    if not p.exists():
        return None
    lanes, pairs = {}, {}
    for ln in p.read_text().splitlines():
        if m := CHECK.match(ln):
            lanes[(m[1], m[2])] = (int(m[5]), m[6], int(m[7]), int(m[3]), int(m[9]), m[10])
        elif m := PAIRS.match(ln):
            pairs[(m[1], m[2])] = m[5]
    return lanes, pairs


def clustering():
    """The clustering check (clust/: compare.txt from the CLI runs of both builds, ctests.out from both WSL trees):
    (identical, what it rests on)."""
    cmp_, cts = KIT / "clust" / "compare.txt", KIT / "clust" / "ctests.out"
    if not (cmp_.exists() and cts.exists()):
        return False, "clust/compare.txt or clust/ctests.out missing: run the clust/ scripts"
    cli = cmp_.read_text().rstrip().splitlines()
    t = cts.read_text()
    conf = len(re.findall(r"Test +#\d+: cpp_conformance \.+ +Passed", t))
    regen = t.count("regenerated reference vs b3b3a57e's: 0 differing lines")
    runs = next((ln for ln in cli if ln.startswith("runs compared")), "no CLI summary")
    return (cli[-1] == "clustering identical: yes" and conf == 2 and regen == 2,
            f"CLI {runs}; cpp_conformance passed in {conf}/2 builds, its regenerated reference equal to the tracked "
            f"one in {regen}/2")


def report(raw=None):
    raw = pathlib.Path(raw) if raw else KIT / "raw.txt"
    text = raw.read_text().splitlines()
    starts = [i for i, ln in enumerate(text) if ln.startswith(("# run ", "# smoke ")) and " rounds=" in ln]
    text = text[starts[-1]:] if starts else text  # a raw file a run appends to: its last run
    smoke = any("SMOKE" in ln for ln in text if ln.startswith("#"))
    base, cand = BUILDS
    med = statistics.median
    times, fills, spread = {}, {}, []
    for ln in text:
        if m := TIME.match(ln):
            r, b, t, d, L, band, W, ns, ghz, gmin, gmax, cyc = m.groups()
            times.setdefault((t, d, int(L), int(band)), {}).setdefault(b, {})[int(r)] = (
                float(ns), int(W), float(ghz), float(cyc))
            if float(gmax) > float(gmin) * (1 + CLOCK_TOL):
                spread.append(f"round {r} {b} {t} {d} L {L} band {band}: {gmin}-{gmax} GHz")
        elif m := FILL.match(ln):
            r, b, t, d, th, N, L, rag, band, W, g, h = m.groups()
            fills.setdefault((int(N), int(L), bool(rag), int(band), int(th)), {}).setdefault(b, {})[int(r)] = (
                float(g), h, int(W))
    out = [f"# XG report from {raw.name}" + (" -- SMOKE RUN: tiny sizes, one round; no number here is a measurement"
                                              if smoke else ""), ""]
    out += [ln for ln in text if ln.startswith(("# info", "# wsl", "# round=", "# after"))] + [""]

    # Clocks: vCPU 2's clock (the probes' dependent register-add chain), median per round over every probe.
    per_round = {}
    for builds in times.values():
        for rounds in builds.values():
            for r, v in rounds.items():
                per_round.setdefault(r, []).append(v[2])
    clock = {r: med(v) for r, v in sorted(per_round.items())}
    ref = med(clock.values()) if clock else 0  # the rounds' median: one round off does not move it
    flagged = [r for r, g in clock.items() if len(clock) > 1 and abs(g / ref - 1) > CLOCK_TOL]
    out.append(f"clock of vCPU {VCPU} (dependent register-add chain, median over the round's probes): "
               + ", ".join(f"round {r} {g:.3f} GHz" + (" FLAGGED" if r in flagged else "") for r, g in clock.items()))
    out.append(f"- rounds whose clock differs from the rounds' median by more than {CLOCK_TOL:.0%}: "
               + (", ".join(map(str, flagged)) if flagged else "none"))
    out.append(f"- probes whose own clock moved by more than {CLOCK_TOL:.0%} over their measurements: {len(spread)}"
               + "".join(f"\n  - {s}" for s in spread))
    out.append("")

    out += ["## gcc: candidate (cand) against base, median over rounds; ratio = base ns / cand ns (single thread) "
            "or cand Gcell/s / base Gcell/s (fill), per round", ""]
    out += ["| shape | W base/cand | base ns/cell | cand ns/cell | ratio (median) | range | cycles/row step base/cand "
            "| clock GHz base/cand |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    single = []
    for k in sorted(times, key=lambda k: (k[0], k[1], k[2], k[3])):
        b, n = times[k].get(base, {}), times[k].get(cand, {})
        reps = sorted(set(b) & set(n))
        if not reps:
            continue
        ratios = [b[r][0] / n[r][0] for r in reps]
        single.append(med(ratios))
        name = f"{k[0]} {k[1]} L {k[2]} {'unbanded' if k[3] < 0 else f'band {k[3]}'}"
        out.append(f"| {name} | {b[reps[0]][1]}/{n[reps[0]][1]} | {med(b[r][0] for r in reps):.5f} | "
                   f"{med(n[r][0] for r in reps):.5f} | {med(ratios):.3f} | {min(ratios):.3f}-{max(ratios):.3f} | "
                   f"{med(b[r][3] for r in reps):.2f} / {med(n[r][3] for r in reps):.2f} | "
                   f"{med(b[r][2] for r in reps):.3f} / {med(n[r][2] for r in reps):.3f} |")
    out += ["", "| fill (f64 L1, OpenMP default threads) | W base/cand | base Gcell/s | cand Gcell/s | ratio (median) "
            "| range | matrices |", "| --- | --- | --- | --- | --- | --- | --- |"]
    fill_eq, fill_rag, same_all = [], [], True
    for k in sorted(fills, key=lambda k: (k[2], -k[0])):
        b, n = fills[k].get(base, {}), fills[k].get(cand, {})
        reps = sorted(set(b) & set(n))
        if not reps:
            continue
        ratios = [n[r][0] / b[r][0] for r in reps]
        same = all(b[r][1] == n[r][1] for r in reps) and len({b[r][1] for r in reps}) == 1
        same_all = same_all and same
        (fill_rag if k[2] else fill_eq).append(med(ratios))
        name = (f"N {k[0]} L {k[1]}{' +-10 ragged' if k[2] else ''} "
                f"{'unbanded' if k[3] < 0 else f'band {k[3]}'}, {k[4]} threads")
        out.append(f"| {name} | {b[reps[0]][2]}/{n[reps[0]][2]} | {med(b[r][0] for r in reps):.2f} | "
                   f"{med(n[r][0] for r in reps):.2f} | {med(ratios):.3f} | {min(ratios):.3f}-{max(ratios):.3f} | "
                   f"{'identical' if same else 'DIFFER'} |")

    # The outputs, read two ways. Literal ("as bitwise at 128 as at 64"): every lane output equal to base's, the
    # per-pair route unchanged, the fill matrices equal. The ruling (Volkan 10-01: contraction stays on, last-bit
    # differences between routes are fine, the clustering must not change): the L1 lanes bit for bit (hashes equal,
    # none differing from the per-pair route) with the per-pair route and the fill matrices unchanged, the squared
    # lanes within the tests' route bound at both widths, and the clustering identical (clustering()).
    out += ["", "| sweep (check/*.txt) | W base/cand | lane outputs | hash base | hash cand | lanes vs per-pair "
            "mismatches base / cand | outside the route bound base / cand | worst / (2n-1) eps max, base / cand "
            "(bound 2) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    sb, sc = sweep(base), sweep(cand)
    if sb and sc and sb[0].keys() == sc[0].keys() and len(sb[0]) == 4:
        for k in sorted(sb[0]):
            vb, vc = sb[0][k], sc[0][k]
            out.append(f"| {k[0]} {k[1]} | {vb[3]}/{vc[3]} | {vc[0]:,} | `{vb[1]}` | `{vc[1]}`"
                       f"{'' if vb[1] == vc[1] else ' DIFFER'} | {vb[2]:,} / {vc[2]:,} | {vb[4]:,} / {vc[4]:,} | "
                       f"{vb[5]} / {vc[5]} |")
        lanes_eq = all(sb[0][k][1] == sc[0][k][1] for k in sb[0])
        pairs_eq = sb[1] == sc[1] and len(sb[1]) == 4
        l1_same = all(sb[0][k][1] == sc[0][k][1] and sb[0][k][2] == sc[0][k][2] == 0 for k in sb[0] if k[1] == "L1")
        sq_bound = all(sb[0][k][4] == 0 and sc[0][k][4] == 0 for k in sb[0] if k[1] == "Sq")
        out += ["", f"- lane outputs equal at both widths: {'yes' if lanes_eq else 'NO'}; per-pair route unchanged "
                f"(4 hashes): {'yes' if pairs_eq else 'NO'}; fill matrices equal: {'yes' if same_all else 'NO'}"]
    else:
        lanes_eq = pairs_eq = l1_same = sq_bound = False
        out += ["", f"- sweep files missing or incomplete for {base}/{cand}: run `driver.py check`; fill matrices "
                f"equal: {'yes' if same_all else 'NO'}"]
    literal = lanes_eq and pairs_eq and same_all
    l1_ok = l1_same and pairs_eq and same_all
    clustered, clustered_note = clustering()
    out.append(f"- the ruling's reading: L1 lanes bit for bit at both widths (with the per-pair route and the fill "
               f"matrices) -> {'yes' if l1_ok else 'NO'}; squared lanes within the route bound at both widths -> "
               f"{'yes' if sq_bound else 'NO'}; clustering identical -> {'yes' if clustered else 'NO'} "
               f"({clustered_note})")
    if single and fill_eq and fill_rag:
        s_ok = min(single) >= BAND_SINGLE and len(single) == len(SHAPES)
        f_ok = min(fill_eq) >= BAND_FILL and len(fill_eq) == 2
        r_ok = all(BAND_RAGGED[0] <= v <= BAND_RAGGED[1] for v in fill_rag)
        timing = s_ok and f_ok and r_ok
        ruling = "PASS" if timing and l1_ok and sq_bound and clustered else "FALSIFIED"
        literal_verdict = "PASS" if timing and literal else "FALSIFIED"
        out += [f"- timing band: every single-thread shape >= {BAND_SINGLE}x: min {min(single):.3f} ({len(single)} "
                f"shapes) -> {'pass' if s_ok else 'fail'}; both equal-length fills >= {BAND_FILL}x: "
                f"{', '.join(f'{v:.3f}' for v in fill_eq)} -> {'pass' if f_ok else 'fail'}; ragged fill in "
                f"{BAND_RAGGED[0]}-{BAND_RAGGED[1]}x: {', '.join(f'{v:.3f}' for v in fill_rag)} -> "
                f"{'pass' if r_ok else 'fail'}",
                f"- **verdict gcc under the ruling (timing band + L1 identical + squared within the route bound + "
                f"clustering identical): {ruling}**" + (" (smoke run: not a verdict)" if smoke else ""),
                f"- beside it, the literal reading (timing band + every lane output as at 64): {literal_verdict}"
                + (" (smoke run: not a verdict)" if smoke else "")]
    out.append("")
    (raw.parent / ("report_smoke.md" if smoke else "report.md")).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        args = sys.argv[2:]
        run(False, int(args[args.index("--rounds") + 1]) if "--rounds" in args else 3)
    elif cmd == "smoke":
        run(True)
    elif cmd == "check":
        check()
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
