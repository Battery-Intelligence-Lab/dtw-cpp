"""One CA measurement: one shape, one thread setting, one of the release archive's two CLIs.

uv run --no-project python measure.py <exe> <shape> <threads> [--smoke]
  exe      dtwc_cl (clang-cl) | dtwc_cl_msvc (MSVC): bin/<exe>.exe of the archive unpacked in archive/
  shape    a | b | c: data/<shape>.csv (a and c unbanded, b --band 100); --smoke reads data/<shape>_smoke.csv
  threads  1: OMP_NUM_THREADS=1, and this process pins itself to logical CPU 12 (a P-core) before it starts the
           CLI, which inherits the affinity (read back from the CLI's own process). 24: no OMP_*/KMP_* variable,
           no pinning (the OpenMP default).
The CLI runs `-i <csv> -k 1 -m pam --max-iter 1 -o "" -v`: it reads the series, fills the distance matrix, runs
FastPAM for one cluster (BUILD and one sweep, O(N^2) lookups of the filled matrix against the fill's O(N^2 L^2)
DTW cells) and writes no file. The time is the CLI's own: from its "Data loaded" stamp to the stamp on its
"FastPAM ... cost=" line, so process start and CSV parsing are outside it.
Prints one line of key=value fields; exits non-zero if the CLI failed or ran with the wrong threads or affinity.
"""
import ctypes
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

KIT = pathlib.Path(__file__).resolve().parent
BIN = KIT / "archive" / "dtwc-2.0.0rc1-Windows-AMD64" / "bin"
CPU = 12
BAND = {"a": -1, "b": 100, "c": -1}
STAMP = re.compile(r"\[(\d+):([0-9.eE+-]+) min:sec\]")

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.POINTER(ctypes.c_size_t)]


def affinity(handle):
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(ctypes.c_void_p(handle), ctypes.byref(mask), ctypes.byref(system)):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask")
    return mask.value


def band_cells(n, band):
    """DP cells of an n x n matrix inside |i - j| <= band (all n * n when band < 0)."""
    if band < 0 or band >= n - 1:
        return n * n
    return n * (2 * band + 1) - band * (band + 1)


def fill_cells(lengths, band):
    if band < 0:
        total = sum(lengths)
        return (total * total - sum(n * n for n in lengths)) // 2
    if len(set(lengths)) != 1:
        raise SystemExit("a banded shape here has equal lengths")
    return len(lengths) * (len(lengths) - 1) // 2 * band_cells(lengths[0], band)


def seconds(stamp):
    minutes, secs = STAMP.search(stamp).groups()
    return 60 * int(minutes) + float(secs)


def main():
    exe, shape, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
    smoke = "--smoke" in sys.argv[4:]
    if exe not in ("dtwc_cl", "dtwc_cl_msvc") or shape not in BAND or threads not in (1, 24):
        raise SystemExit(__doc__)
    csv_name = f"{shape}{'_smoke' if smoke else ''}.csv"
    entry = json.loads((KIT / "data" / "manifest.json").read_text())["files"][csv_name]
    csv = KIT / "data" / csv_name
    if hashlib.sha256(csv.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"{csv} does not match manifest.json")
    binary = BIN / f"{exe}.exe"

    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "KMP_"))}
    if threads == 1:
        env["OMP_NUM_THREADS"] = "1"
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), 1 << CPU):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
    args = [str(binary), "-i", str(csv), "-k", "1", "-m", "pam", "--max-iter", "1", "-o", "", "-v"]
    if BAND[shape] >= 0:
        args += ["--band", str(BAND[shape])]
    child = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=KIT)
    child_mask = affinity(int(child._handle))  # read while it runs: the mask it inherited
    out, err = child.communicate()

    lines = out.splitlines()
    loaded = next(line for line in lines if line.startswith("Data loaded"))
    finished = next(line for line in lines if line.startswith("FastPAM") and "cost=" in line)
    omp = int(next(line for line in lines if "OpenMP threads:" in line).split(":")[1])
    cost = next(line for line in lines if "Total cost:" in line).split(":")[1].strip()
    fill = seconds(finished) - seconds(loaded)
    cells = fill_cells(entry["lengths"], BAND[shape])
    expected_mask = (1 << CPU) if threads == 1 else affinity(k32.GetCurrentProcess())
    print(f"exe={exe} sha={hashlib.sha256(binary.read_bytes()).hexdigest()[:12]} shape={shape} threads={threads} "
          f"affinity={child_mask:#x} omp={omp} smoke={int(smoke)} exit={child.returncode} seconds={fill:.6f} "
          f"cells={cells} gcells={cells / fill / 1e9:.4f} cost={cost}")
    if child.returncode != 0 or omp != threads or child_mask != expected_mask:
        sys.exit(f"bad run: exit {child.returncode}, OpenMP threads {omp}, affinity {child_mask:#x}\n{err}")


if __name__ == "__main__":
    main()
