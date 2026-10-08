"""RL timing kit driver: the CPU fill's lanes take ragged blocks. The candidate B3 (pb/RL at f8c3d23a: B's fill, rows
in index order, each taking its columns longest first; cl's one-pair route for blocks of unequal length, 0b5cfe42; the
kernel's column loop without a capture test) against design-2.0 e691e48c (base), and on the large-matrix shape the
first candidate A too (rows and columns longest first; for the record), under clang++ and cl (Windows) and GCC 13.3
(WSL). B and B2 (0b5cfe42) stay in bin/ for diagnosis. Stdlib only; run from Git Bash on Windows.

  uv run --no-project python driver.py run [--rounds 3] [--compilers clang,msvc,gcc]   the measurement (quiet machine)
  uv run --no-project python driver.py smoke                                         one tiny round: the kit works
  uv run --no-project python driver.py check                                         the bitwise sweeps, no timing
  uv run --no-project python driver.py report [raw file]                             medians, ratios, hashes, verdicts
  uv run --no-project python driver.py run --cl-retime                               cl only: base against B3 on
      CL_FILLS, into raw_cl.txt and report_cl.md, with Windows-only load records; `smoke --cl-retime` is its one tiny
      round

Every timed shape is a fill through the linked version's Problem::fill_distance_matrix (its loop, order and schedule):
one untimed fill, then 3 timed; Gcell/s counts each pair's own cells, the same for every version. Single-thread
shapes are pinned: logical CPU 12 (a P-core) on Windows, `taskset -c 2` in WSL, each probe with the calling thread's
clock (a dependent register-add chain; cl builds measure none). The rest run at the OpenMP default (24 threads).
Each round runs every compiler and changes which build goes first (base and B alternate; base, B and A rotate). `run`
appends every probe line verbatim to raw.txt (`smoke` to raw_smoke.txt), with load records before and after (load.sh
-> load.txt: Windows, then WSL), and boots WSL first, waiting until systemd has started and runs no job (XG's
driver). `report` writes report.md.
"""
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import time

KIT = pathlib.Path(__file__).resolve().parent
CPU, VCPU = 12, 2  # Windows: a P-core (P-cores are logical 0, 1, 10-13, 22, 23); WSL: any vCPU (a host thread)
DISTRO = "Ubuntu-24.04"
COMPILERS = ("clang", "msvc", "gcc")
# (T, D, N, shortest, longest, band, threads; threads 1 = pinned), the registered band's floor and ceiling for B, and
# whether A runs too. Under cl every shape's floor is CL_FLOOR instead.
FILLS = [
    ("f64", "L1", 1000, 90, 110, -1, 1, 2.0, None, False),
    ("f64", "L1", 1000, 90, 110, -1, 0, 2.0, None, False),
    ("f64", "L1", 150, 100, 1000, -1, 1, 1.5, None, False),
    ("f64", "L1", 500, 100, 1000, -1, 0, 1.5, None, False),
    ("f32", "L1", 1000, 90, 110, -1, 0, 1.5, None, False),
    ("f64", "Sq", 1000, 90, 110, -1, 0, 1.5, None, False),
    ("f64", "L1", 2000, 100, 100, -1, 0, 0.97, 1.03, False),
    ("f64", "L1", 500, 1000, 1000, 100, 0, 0.97, 1.03, False),
    ("f64", "L1", 1000, 90, 110, 30, 0, 0.97, 1.03, False),
    ("f64", "L1", 8000, 40, 60, -1, 0, 1.5, None, True),  # a 256 MB matrix: write locality
    # long equal-length series on one thread: B ran 4-5 % slower here under clang (its column loop's capture test changed
    # the row loop's registers); B3 is to match base. N 8 (7 blocks of one row each) keeps cl's scalar lanes near 2.6 s
    # a fill; clang showed the same gap at N 8 as at N 24 (diag/ab.txt, 23:03)
    ("f64", "L1", 8, 8000, 8000, -1, 1, 0.97, 1.03, False),
]
SMOKE_FILLS = [("f64", "L1", 60, 90, 110, -1, 1, 2.0, None, False), ("f64", "L1", 60, 90, 110, -1, 0, 2.0, None, False),
               ("f64", "L1", 30, 100, 1000, -1, 1, 1.5, None, False),
               ("f32", "L1", 60, 90, 110, -1, 0, 1.5, None, False), ("f64", "Sq", 60, 90, 110, -1, 0, 1.5, None, False),
               ("f64", "L1", 100, 100, 100, -1, 0, 0.97, 1.03, False),
               ("f64", "L1", 30, 1000, 1000, 100, 0, 0.97, 1.03, False),
               ("f64", "L1", 60, 90, 110, 30, 0, 0.97, 1.03, False), ("f64", "L1", 400, 40, 60, -1, 0, 1.5, None, True),
               ("f64", "L1", 4, 1000, 1000, -1, 1, 0.97, 1.03, False)]
CAND = "B3"
# The cl re-time after Volkan's ruling (2026-10-08, "Adopt, cl one-pair"): cl's four f64 L1 ragged shapes, N 8000,
# the controls and the long equal-length shape, base against CAND, under the registered cl clause (no shape below
# CL_FLOOR).
CL_FILLS = [f for f in FILLS if f[0] == "f64" and f[1] == "L1"]
SMOKE_CL_FILLS = [f for f in SMOKE_FILLS if f[0] == "f64" and f[1] == "L1"]
CL_FLOOR = 0.97  # under cl no timed shape of B below this
CLOCK_TOL = 0.03  # a pinned probe whose clock moves by more over its fills is flagged


def build(c, v):
    return f"{c}_{v}"


def wsl(*args):
    out = subprocess.run(["wsl", "-d", DISTRO, "-e", *map(str, args)], capture_output=True, text=True)
    return out.returncode, (out.stdout.strip() or out.stderr.strip())


_home = []


def probe(b, *args, pinned=False, threads=None):
    """Run a probe: in WSL for GCC (under taskset when pinned), else the Windows executable; OMP_NUM_THREADS=threads
    if given. Returns (exit code, stdout or else stderr)."""
    if b.startswith("gcc"):
        if not _home:
            _home.append(wsl("sh", "-c", 'printf %s "$HOME"')[1])
        env = ("env", f"OMP_NUM_THREADS={threads}") if threads else ()
        return wsl(*(("taskset", "-c", VCPU) if pinned else ()), *env, f"{_home[0]}/rl/bin/{b}/rprobe", *args)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)} if threads else None
    out = subprocess.run([str(KIT / "bin" / b / "rprobe.exe"), *map(str, args)], capture_output=True, text=True,
                         env=env)
    return out.returncode, (out.stdout.strip() or out.stderr.strip())


def settle():
    """Boot WSL, wait until systemd has started and runs no job (at most 10 minutes); say what was waited for."""
    script = ('systemctl is-system-running --wait >/dev/null; t=0; '
              'while [ -n "$(systemctl list-jobs --no-legend)" ] && [ $t -lt 600 ]; do sleep 5; t=$((t + 5)); done; '
              'echo "systemd $(systemctl is-system-running), waited $t s for its jobs; jobs now: '
              '$(systemctl list-jobs --no-legend | wc -l); up $(cut -d" " -f1 /proc/uptime) s"')
    return wsl("sh", "-c", script)[1]


def load(label, windows_only=False):
    sh = shutil.which("sh") or "C:/Users/engs2321/AppData/Local/Programs/Git/usr/bin/sh.exe"
    try:
        subprocess.run([sh, str(KIT / "load.sh"), label, *(("windows",) if windows_only else ())], check=False)
    except OSError as e:
        print(f"no load record ({label}): {e}; run load.sh by hand", flush=True)


def run(smoke=False, rounds=3, compilers=COMPILERS, cl_retime=False):
    cand = CAND
    if cl_retime:
        compilers = ("msvc",)
        raw = KIT / ("raw_cl_smoke.txt" if smoke else "raw_cl.txt")
        fills, reps = (SMOKE_CL_FILLS, 1) if smoke else (CL_FILLS, 3)
    else:
        raw = KIT / ("raw_smoke.txt" if smoke else "raw.txt")
        fills, reps = (SMOKE_FILLS, 1) if smoke else (FILLS, 3)
    if smoke:
        rounds = 1
    tag = "smoke" if smoke else "run"
    start = time.time()
    with open(raw, "a") as f:
        f.write(f"# {tag} {time.strftime('%Y-%m-%d %H:%M:%S')} rounds={rounds} compilers={','.join(compilers)} "
                f"cpu={CPU} vcpu={VCPU} cand={cand}{' mode=cl-retime' if cl_retime else ''}"
                f"{' SMOKE: not a measurement' if smoke else ''}\n")
        if "gcc" in compilers:
            f.write(f"# wsl: {settle()}\n")
        for c in compilers:
            for v in ("base", cand) if cl_retime else ("base", cand, "A"):
                f.write(f"# info {build(c, v)}: {probe(build(c, v), 'info')[1]}\n")
        f.flush()
        load(f"{tag} before", windows_only="gcc" not in compilers)
        for r in range(rounds):
            for c in compilers:
                if c == "gcc" and (jobs := wsl("systemctl", "list-jobs", "--no-legend")[1]):
                    f.write(f"# round={r} wsl jobs: {' | '.join(jobs.splitlines())}\n")
                for t, d, n, lo, hi, band, threads, _, _, with_a in fills:
                    versions = ["base", cand, "A"] if with_a and not cl_retime else ["base", cand]
                    k = r % len(versions)
                    for v in versions[k:] + versions[:k]:
                        b = build(c, v)
                        code, out = probe(b, "fill", t, d, n, lo, hi, band, threads, reps, VCPU if c == "gcc" else CPU,
                                          pinned=threads == 1)
                        f.write(f"round={r} compiler={c} build={b} exit={code} {out}\n")
                        f.flush()
                        if code:
                            sys.exit(f"probe failed: {b} fill {t} {d} {n} {lo}-{hi} {band} {threads} -> {code}: {out}")
            print(f"round {r} done, {time.time() - start:.0f} s", flush=True)
        load(f"{tag} after", windows_only="gcc" not in compilers)
        f.write(f"# {tag} done in {time.time() - start:.0f} s\n")
    report(raw)


def check():
    """The bitwise sweeps (8 threads): XL's equal-length sweep of base and CAND into check/<build>.txt, the ragged sweep
    of CAND into check/<build>_ragged.txt. (A's, B's and cl's B2, from before, stay in check/.)"""
    (KIT / "check").mkdir(exist_ok=True)
    for c in COMPILERS:
        for v, modes in (("base", ("check",)), (CAND, ("check", "ragged"))):
            b = build(c, v)
            for mode in modes:
                t0 = time.time()
                code, out = probe(b, mode, 3, threads=8)
                name = f"{b}.txt" if mode == "check" else f"{b}_ragged.txt"
                (KIT / "check" / name).write_text(out + "\n")
                print(f"{b} {mode}: exit {code}, {time.time() - t0:.0f} s", flush=True)
                if code:
                    sys.exit(f"check failed: {b} {mode} -> {code}")


FILL = re.compile(r"^round=(\d+) compiler=(\w+) build=(\w+) exit=0 fill (f64|f32) (L1|Sq)\s+threads\s+(\d+) N (\d+)\s+"
                  r"L (\d+)-(\d+) band (-?\d+)\s+W (\d+)\s+ms ([\d.]+) \(min ([\d.]+) max ([\d.]+)\)\s+Gcell/s ([\d.]+)\s+"
                  r"clock ([\d.]+) GHz \(min ([\d.]+) max ([\d.]+)\)\s+lanes computed/useful B ([\d.]+) \(padding "
                  r"([\d.]+), tail duplicates ([\d.]+)\) A ([\d.]+) \(padding ([\d.]+), tail duplicates ([\d.]+)\)\s+"
                  r"matrix hash ([0-9a-f]+)")
SWEEP = re.compile(r"^(check|ragged) lanes (f64|f32) (L1|Sq)\s+W=(\d+)\s+seeds (\d+): outputs (\d+).*?hash ([0-9a-f]+)\s+"
                   r"lanes vs per-pair bitwise mismatches (\d+)\s+non-finite outputs (\d+)\s+outside the route bound (\d+)"
                   r"\s+worst dev (\S+)")
PAIRS = re.compile(r"^check pair\s+(f64|f32) (L1|Sq)\s+seeds (\d+): outputs (\d+)\s+hash ([0-9a-f]+)")


def sweep(name):
    """A sweep file's lanes {(T, D): (outputs, hash, mismatches, W, outside, worst)} and pairs {(T, D): hash}."""
    p = KIT / "check" / name
    if not p.exists():
        return None
    lanes, pairs = {}, {}
    for ln in p.read_text().splitlines():
        if m := SWEEP.match(ln):
            lanes[(m[2], m[3])] = (int(m[6]), m[7], int(m[8]), int(m[4]), int(m[10]), m[11])
        elif m := PAIRS.match(ln):
            pairs[(m[1], m[2])] = m[5]
    return lanes, pairs


def outputs_note(c, v="B"):
    """The PROOF items for the candidate v under compiler c, from check/ (and cli/, clust/): (ok, lines)."""
    base, cand = build(c, "base"), build(c, v)
    sb, sc, sr = sweep(f"{base}.txt"), sweep(f"{cand}.txt"), sweep(f"{cand}_ragged.txt")
    lines, ok = [], True
    if sb and sc and len(sb[0]) == 4 and sb[0].keys() == sc[0].keys():
        # As registered: the per-pair route's hashes and the lanes' L1 hashes as at base under every compiler, the
        # squared ones under clang; GCC's and cl's squared lanes within the route bound (the multiply-add caveat).
        same = {k: sb[0][k][1] == sc[0][k][1] for k in sb[0]}
        pairs_same = sb[1] == sc[1] and len(sb[1]) == 4
        outside = sum(v[4] for v in sb[0].values()) + sum(v[4] for v in sc[0].values())
        mism = sum(v[2] for v in sb[0].values()) + sum(v[2] for v in sc[0].values())
        good = (pairs_same and outside == 0 and all(v for k, v in same.items() if k[1] == "L1")
                and (c != "clang" or all(same.values())))
        ok = ok and good
        lines.append(f"equal-length sweep: {sum(v[0] for v in sc[0].values()):,} lane outputs per build; lane hashes "
                     f"base == {v}: " + ", ".join(f"{k[0]} {k[1]} {'yes' if v else 'no'}" for k, v in sorted(same.items()))
                     + f"; per-pair hashes base == {v}: {'yes' if pairs_same else 'NO'}; lanes vs per-pair bitwise "
                     f"mismatches base + {v}: {mism:,}; outside the route bound: {outside:,} -> "
                     f"{'pass' if good else 'FAIL'}")
    else:
        ok = False
        lines.append(f"equal-length sweep missing for {base}/{cand}: run `driver.py check`")
    if sr and len(sr[0]) == 4:
        l1 = sum(v[2] for k, v in sr[0].items() if k[1] == "L1")
        sq = sum(v[2] for k, v in sr[0].items() if k[1] == "Sq")
        out_ = sum(v[4] for v in sr[0].values())
        want_sq_bitwise = c == "clang"  # elsewhere the squared lanes need only the route bound (the FMA caveat)
        good = l1 == 0 and out_ == 0 and (sq == 0 or not want_sq_bitwise)
        ok = ok and good
        lines.append(f"ragged sweep ({v}): {sum(v[0] for v in sr[0].values()):,} lane outputs; vs per-pair bitwise "
                     f"mismatches L1 {l1:,}, squared {sq:,}; outside the route bound {out_:,}; worst dev "
                     f"{', '.join(f'{k[0]} {k[1]} {v[5]}' for k, v in sorted(sr[0].items()))} -> "
                     f"{'pass' if good else 'FAIL'}")
    else:
        ok = False
        lines.append(f"ragged sweep missing for {cand}: run `driver.py check`")
    if c == "clang":
        p = KIT / "cli" / "compare.txt"
        last = p.read_text().strip().splitlines()[-1] if p.exists() else "cli/compare.txt missing"
        ok = ok and last.startswith("byte-identical: yes")
        lines.append(f"CLI (w8b_cli_runs.sh + w8b_cli_extra.sh, data/dummy): {last}")
    if c == "gcc":
        p = KIT / "clust" / "compare.txt"
        last = p.read_text().strip().splitlines()[-1] if p.exists() else "clust/compare.txt missing"
        ok = ok and last == "clustering identical: yes"
        lines.append(f"clustering (clust/, data/dummy uncut): {last}")
    return ok, lines


def report(raw=None):
    raw = pathlib.Path(raw) if raw else KIT / "raw.txt"
    text = raw.read_text().splitlines()
    starts = [i for i, ln in enumerate(text) if ln.startswith(("# run ", "# smoke ")) and " rounds=" in ln]
    text = text[starts[-1]:] if starts else text  # a raw file a run appends to: its last run
    smoke = any("SMOKE" in ln for ln in text if ln.startswith("#"))
    cand = next((m[1] for ln in text if (m := re.search(r" cand=(\w+)", ln))), "B")
    cl_retime = any(" mode=cl-retime" in ln or " cand=B2" in ln for ln in text if ln.startswith("#"))
    med = statistics.median
    fills = {}
    for ln in text:
        if m := FILL.match(ln):
            (r, c, b, t, d, th, n, lo, hi, band, w, ms, mn, mx, g, clk, cmin, cmax,
             bcu, bpad, bdup, acu, apad, adup, h) = m.groups()
            # the shape as FILLS names it: threads 1 (pinned) or 0 (the OpenMP default, which the probe prints)
            key = (c, t, d, int(n), int(lo), int(hi), int(band), 1 if int(th) == 1 else 0)
            fills.setdefault(key, {}).setdefault(b.split("_", 1)[1], {})[int(r)] = dict(
                ms=float(ms), g=float(g), h=h, w=int(w), clk=float(clk), cmin=float(cmin), cmax=float(cmax),
                waste={"B": (float(bcu), float(bpad), float(bdup)), "A": (float(acu), float(apad), float(adup))},
                th=int(th))
    out = [f"# RL report from {raw.name}" + (" -- SMOKE RUN: tiny sizes, one round; no number here is a measurement"
                                              if smoke else ""), ""]
    out += [ln for ln in text if ln.startswith(("# info", "# wsl", "# round=", "# run ", "# smoke "))] + [""]
    shapes = (SMOKE_CL_FILLS if smoke else CL_FILLS) if cl_retime else (SMOKE_FILLS if smoke else FILLS)
    spec = {tuple(f[:7]): f[7:] for f in shapes}
    for c in COMPILERS:
        keys = [k for k in fills if k[0] == c]
        if not keys:
            continue
        out += [f"## {c}: {cand} against base, median over rounds; ratio = base ms / {cand} ms per round", "",
                f"| fill | W | base ms | {cand} ms | base Gcell/s | {cand} Gcell/s | ratio (median) | range | band | "
                f"matrix hash base / {cand} | B's lanes: computed/useful (padding, duplicates) | clock GHz base / {cand} |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        verdicts, flags, hashes_ok, a_rows = [], [], True, []
        for k in sorted(keys, key=lambda k: list(spec).index(k[1:]) if k[1:] in spec else len(spec)):
            runs = fills[k]
            bb, nn = runs.get("base", {}), runs.get(cand, {})
            reps = sorted(set(bb) & set(nn))
            if not reps:
                continue
            ratios = [bb[r]["ms"] / nn[r]["ms"] for r in reps]
            t, d, n, lo, hi, band, th = k[1:]
            floor, ceil, with_a = spec.get(k[1:], (None, None, False))
            r_med = med(ratios)
            if c == "msvc":
                ok, band_txt = r_med >= CL_FLOOR, f">= {CL_FLOOR}"
            elif ceil is not None:
                ok, band_txt = floor <= r_med <= ceil, f"{floor}-{ceil}"
            else:
                ok, band_txt = r_med >= floor, f">= {floor}"
            verdicts.append(ok)
            hb, hc = {bb[r]["h"] for r in reps}, {nn[r]["h"] for r in reps}
            same = len(hb) == 1 and hb == hc
            fma_ok = c != "clang" and d == "Sq"  # squared: as at base under clang; elsewhere the multiply-add caveat
            hashes_ok = hashes_ok and (same or fma_ok)
            hash_txt = (f"`{next(iter(hb))}` = {cand}" if same else
                        f"`{'/'.join(sorted(hb))}` / `{'/'.join(sorted(hc))}`" + (" (FMA caveat)" if fma_ok else " DIFFER"))
            name = (f"{t} {d} N {n} L {lo}" + (f"-{hi}" if hi != lo else "") + (" unbanded" if band < 0 else f" band {band}")
                    + (", 1 thread pinned" if th == 1 else f", {nn[reps[0]]['th']} threads"))
            x = nn[reps[0]]
            clk = (f"{med(bb[r]['clk'] for r in reps):.3f} / {med(nn[r]['clk'] for r in reps):.3f}"
                   if th == 1 and x["clk"] > 0 else "n/a")
            if th == 1:
                for v, rr in (("base", bb), (cand, nn)):
                    for r in reps:
                        if rr[r]["cmax"] > rr[r]["cmin"] * (1 + CLOCK_TOL):
                            flags.append(f"round {r} {v} {name}: clock {rr[r]['cmin']}-{rr[r]['cmax']} GHz")
            wb = x["waste"]["B"]
            out.append(f"| {name} | {x['w']} | {med(bb[r]['ms'] for r in reps):.2f} | {med(nn[r]['ms'] for r in reps):.2f} "
                       f"| {med(bb[r]['g'] for r in reps):.3f} | {med(nn[r]['g'] for r in reps):.3f} | {r_med:.3f} | "
                       f"{min(ratios):.3f}-{max(ratios):.3f} | {band_txt}: {'pass' if ok else 'FAIL'} | {hash_txt} | "
                       + ("- (no ragged block in the lanes under cl)" if cl_retime else
                          f"{wb[0]:.4f} ({wb[1]:.4f}, {wb[2]:.4f})" if band < 0 and lo != hi else "-")
                       + f" | {clk} |")
            if with_a and (aa := runs.get("A", {})):
                ra = sorted(set(bb) & set(aa))
                if ra:
                    ar = [bb[r]["ms"] / aa[r]["ms"] for r in ra]
                    wa = aa[ra[0]]["waste"]["A"]
                    ha = {aa[r]["h"] for r in ra}
                    a_rows.append(f"| {name} | {med(bb[r]['g'] for r in ra):.3f} | {med(aa[r]['g'] for r in ra):.3f} | "
                                  f"{med(ar):.3f} | {min(ar):.3f}-{max(ar):.3f} | {med(nn[r]['g'] for r in reps):.3f} | "
                                  f"{r_med:.3f} | {'= base' if ha == hb else 'DIFFER'} | {wa[0]:.4f} ({wa[1]:.4f}, "
                                  f"{wa[2]:.4f}) |")
        if a_rows:
            out += ["", "A (rows and columns longest first; for the record, no band) on the same shape:", "",
                    f"| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | {cand} Gcell/s | {cand} ratio "
                    "(median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |", *a_rows]
        outputs_ok, notes = outputs_note(c, cand)
        out += [""] + [f"- {x}" for x in notes]
        out.append(f"- pinned probes whose clock moved by more than {CLOCK_TOL:.0%} over their fills: {len(flags)}"
                   + "".join(f"\n  - {s}" for s in flags))
        timing_ok = bool(verdicts) and all(verdicts) and len(verdicts) == len(spec)
        verdict = "PASS" if timing_ok and hashes_ok and outputs_ok else "FALSIFIED"
        out.append(f"- band: {sum(verdicts)}/{len(spec)} shapes in band; matrix hashes base == {cand} (L1 everywhere, "
                   f"squared under clang): {'yes' if hashes_ok else 'NO'}; outputs as registered: "
                   f"{'yes' if outputs_ok else 'NO'}")
        out.append(f"- **verdict {c} ({cand}): {verdict}**" + (" (smoke run: not a verdict)" if smoke else ""))
        out.append("")
    name = ("report_cl" if cl_retime else "report") + ("_smoke" if smoke else "") + ".md"
    (raw.parent / name).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    args = sys.argv[2:]
    comps = tuple(args[args.index("--compilers") + 1].split(",")) if "--compilers" in args else COMPILERS
    cl_retime = "--cl-retime" in args
    if cmd == "run":
        run(False, int(args[args.index("--rounds") + 1]) if "--rounds" in args else 3, comps, cl_retime)
    elif cmd == "smoke":
        run(True, 1, comps, cl_retime)
    elif cmd == "check":
        check()
    elif cmd == "report":
        report(args[0] if args and not args[0].startswith("--") else None)
    else:
        sys.exit(__doc__)
