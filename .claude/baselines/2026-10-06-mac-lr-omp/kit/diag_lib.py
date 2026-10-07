"""Does the 3 % follow the HiGHS dylib? (stdlib only):  uv run --no-project python diag_lib.py
Two probes (base; noomp = the head's source compiled without OpenMP, linked with the head tree's libraries) each run against both trees' libhighs.1.dylib
(DYLD_LIBRARY_PATH makes dyld look in that folder for the leaf name first). HiGHS-ON root, noise N=200, k=4, REPEATS=9, interleaved."""
import json
import os
import statistics
import subprocess

from build_kit import BIN, KIT, SCR, WT

LIBS = {"baselib": SCR / "base-build" / "lib", "headlib": WT / "build" / "lib"}
PROBES = {"base": BIN / "probe_base_on", "noomp": BIN / "diag_noomp"}
combos = [(p, l) for p in PROBES for l in LIBS]
t = {c: [] for c in combos}
hashes = set()
for r in range(9):
    for c in (combos if r % 2 == 0 else list(reversed(combos))):
        env = {**os.environ, "OMP_NUM_THREADS": "18", "LRPROBE_MIN_MS": "300", "LRPROBE_STAGES": "kel", "DYLD_LIBRARY_PATH": str(LIBS[c[1]])}
        env.pop("LR_OMP_MIN_N", None)
        out = subprocess.run([str(PROBES[c[0]]), "-i", str(KIT / "data" / "noise_N200.csv"), "--skip-rows", "1", "--skip-cols", "1", "-k", "4", "-m", "lrcore"],
                             capture_output=True, text=True, env=env, timeout=3600).stdout
        for line in out.splitlines():
            j = json.loads(line)
            if j["stage"] == "root_kelley":
                t[c].append(j["ms"])
                hashes.add(j["mu_fnv"])
ref = statistics.median(t[("base", "baselib")])
print(f"HiGHS-ON noise N=200 root ms, median of 9 (all multiplier hashes equal: {len(hashes) == 1})")
for c in combos:
    m = statistics.median(t[c])
    print(f"   probe {c[0]:6} on {c[1]:8} {m:9.2f}   x{ref / m:5.3f}")
