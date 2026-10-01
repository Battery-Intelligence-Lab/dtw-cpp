"""W7ef measurement data, deterministic (random walks, Python's Mersenne Twister, seed 7).

softdtw_8x8000/   8 series x 8000 samples, one CSV file per series (header row + index
                  column, data/dummy's layout): the Soft-DTW peak-memory run.
softdtw_32x1000/  32 series x 1000 samples, same layout: the Soft-DTW fill timing.
interp_1000x64.csv  1000 series x 64 samples, one series per row, no header: every
                  second series has NaN gaps (a leading, an interior and a trailing
                  run), so half the pairs interpolate one series and a quarter both;
                  the Interpolate fill timing.
Usage: uv run --no-project python make_data.py <out_dir>
"""
import random
import sys
from pathlib import Path

out = Path(sys.argv[1])
rng = random.Random(7)


def walk(n):
    v, s = 0.0, []
    for _ in range(n):
        v += rng.gauss(0.0, 1.0)
        s.append(v)
    return s


def folder(name, count, length):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for k in range(count):
        lines = [",0"] + [f"{i},{v!r}" for i, v in enumerate(walk(length))]
        (d / f"s{k:03d}.csv").write_text("\n".join(lines) + "\n")


folder("softdtw_8x8000", 8, 8000)
folder("softdtw_32x1000", 32, 1000)

rows = []
for k in range(1000):
    s = [repr(v) for v in walk(64)]
    if k % 2 == 1:
        for i in list(range(0, 3)) + list(range(20, 27)) + list(range(60, 64)):
            s[i] = "nan"
    rows.append(",".join(s))
(out / "interp_1000x64.csv").write_text("\n".join(rows) + "\n")

# plain_1000x64.csv: 1000 series x 64 samples without NaN (the WDTW fill timing),
# drawn after the files above so they stay as they were.
rows = [",".join(repr(v) for v in walk(64)) for _ in range(1000)]
(out / "plain_1000x64.csv").write_text("\n".join(rows) + "\n")
print("wrote", out)
