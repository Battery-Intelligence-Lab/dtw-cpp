"""W7d band data: 100 variable-length windows of data/dummy, lengths 2000..2048.

Window w (0..3) of file data_k.csv (k = 1..25) starts at sample 1000*w and has
2000 + 2*((4*(k-1) + w) % 25) samples, so every pair differs by at most 48
samples and both band 50 and band 200 are feasible for the Problem fill (which
refuses a band narrower than a pair's length difference). The files keep
data/dummy's layout: a header row and an index column.
Usage: uv run --no-project python w7d_make_varlen.py <data/dummy> <out_dir>
"""
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for k in range(1, 26):
    rows = (src / f"data_{k}.csv").read_text().splitlines()[1:]
    values = [r.split(",")[1] for r in rows]
    for w in range(4):
        n = 2000 + 2 * ((4 * (k - 1) + w) % 25)
        window = values[1000 * w : 1000 * w + n]
        assert len(window) == n, (k, w, len(values))
        lines = [",0"] + [f"{i},{v}" for i, v in enumerate(window)]
        (out / f"s{k:02d}_{w}.csv").write_text("\n".join(lines) + "\n")
print("wrote", len(list(out.glob("*.csv"))), "series to", out)
