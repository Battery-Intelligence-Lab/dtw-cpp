"""Loop 2 forked or serial, head against head (stdlib only):  uv run --no-project python loop2_ab.py
The one experiment the product's single constant cannot make: evaluate_dual's second loop (g[j], N*k work) forked as the first, or kept serial,
the first loop forked at every N (LR_OMP_MIN_N=1) in both. HiGHS-OFF probe, the subgradient root only (stage `sub`), 18 threads, interleaved
A B / B A, REPEATS=3. x = t(loop 2 forked) / t(loop 2 serial): > 1 means keeping loop 2 serial is faster. The multiplier hashes must agree."""
import json
import os
import statistics
import subprocess
import time

from build_kit import BIN, KIT

REPEATS = int(os.environ.get("REPEATS", "3"))
SIZES = [int(x) for x in os.environ.get("SIZES", "200 400 800 1600 3200").split()]
K = {"noise": int(os.environ.get("K_NOISE", "4")), "line": int(os.environ.get("K_LINE", "3"))}
REGIMES = os.environ.get("REGIMES", "noise line").split()


def once(regime, n, loop2):
    env = {**os.environ, "OMP_NUM_THREADS": "18", "LRPROBE_MIN_MS": "300", "LRPROBE_STAGES": "sub", "LR_OMP_MIN_N": "1"}
    env.pop("LR_OMP_LOOP2", None)
    if not loop2:
        env["LR_OMP_LOOP2"] = "0"
    r = subprocess.run([str(BIN / "probe_head_off"), "-i", str(KIT / "data" / f"{regime}_N{n}.csv"), "--skip-rows", "1", "--skip-cols", "1",
                        "-k", str(K[regime]), "-m", "lrcore"], capture_output=True, text=True, env=env, timeout=3600)
    for line in r.stdout.splitlines():
        j = json.loads(line)
        if j["stage"] == "root_subgradient":
            return j["ms"], j["mu_fnv"]
    return float("nan"), None


print(f"loop 2 A/B, head HiGHS-OFF probe, root only, REPEATS={REPEATS}, k noise/line = {K['noise']}/{K['line']}, started {time.strftime('%H:%M:%S')}")
print(f"{'regime':7}{'N':>5} | {'forked ms':>10}{'serial ms':>10}{'x':>7} | same bits")
for regime in REGIMES:
    for n in SIZES:
        t = {True: [], False: []}
        hashes = set()
        for rep in range(REPEATS):
            for loop2 in ((True, False) if rep % 2 == 0 else (False, True)):
                ms, h = once(regime, n, loop2)
                t[loop2].append(ms)
                hashes.add(h)
        a, b = statistics.median(t[True]), statistics.median(t[False])
        print(f"{regime:7}{n:>5} | {a:10.2f}{b:10.2f}{a / b:7.2f} | {'yes' if len(hashes) == 1 else 'NO'}", flush=True)
