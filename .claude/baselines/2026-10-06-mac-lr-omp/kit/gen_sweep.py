"""Sweep inputs of the lr-omp timing kit (stdlib only):  uv run --no-project python gen_sweep.py <outdir> [N ...]

The 2026-10-06 lrcore record's generators (gen_data.py, gen_line.py), at the sizes N = 100, 200, 400, 800, 1600, 3200, in its two regimes.
Series have length 40; CSV layout as the record's: header `name,t0..t39`, then `s<row>_g<group>,v0..v39` (read with
--skip-rows 1 --skip-cols 1, values with 6 decimals, so the CSV text is the data).

  noise_N<N>.csv  k = 4  four templates (sin 1 period, sin 2 periods, ramp, bump) + N(0, 0.5) noise on every sample, N/4 series per
                  group, shuffled: the record's C / X1 / Z2. Its subgradient root certifies in tens of iterations (28-66).
                  random.Random(20261100 + i)
  line_N<N>.csv   k = 3  constant series c_i = 100 (i mod 3) + U(-1, 1) + 0.001 i, in index order: the record's Y1..Y6, a line metric
                  (full-band L1 DTW = 40 |c_i - c_j|). Its subgradient root runs the 4000-iteration cap; Kelley certifies in ~10 majors.
                  random.Random(20261200 + i)
i = the index of N in (100, 200, 400, 800, 1600, 3200); any other N (a finer grid, a multiple of 4) takes seed 20261300 + N (noise) and 20261500 + N (line).
"""
import math
import random
import sys
from pathlib import Path

L = 40
SIZES = (100, 200, 400, 800, 1600, 3200)
K = {"noise": 4, "line": 3}


def templates():
    xs = [i / (L - 1) for i in range(L)]
    return [
        [math.sin(2 * math.pi * x) for x in xs],
        [math.sin(4 * math.pi * x) for x in xs],
        [2 * x - 1 for x in xs],
        [2 * math.exp(-(((x - 0.5) / 0.18) ** 2)) - 1 for x in xs],
    ]


def noise_rows(n, seed, sigma=0.5):
    rng = random.Random(seed)
    tpl, per = templates(), n // 4
    z = [[rng.gauss(0.0, 1.0) for _ in range(L)] for _ in range(n)]
    order = list(range(n))
    rng.shuffle(order)
    return [(f"s{r}_g{src // per}", [tpl[src // per][t] + sigma * z[src][t] for t in range(L)]) for r, src in enumerate(order)]


def line_rows(n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        c = 100.0 * (i % 3) + rng.uniform(-1.0, 1.0) + 0.001 * i
        rows.append((f"s{i}_g{i % 3}", [c] * L))
    return rows


def write(path, rows):
    text = "name," + ",".join(f"t{t}" for t in range(L)) + "\n"
    text += "\n".join(name + "," + ",".join(f"{v:.6f}" for v in vals) for name, vals in rows) + "\n"
    Path(path).write_text(text)


def main(outdir, sizes):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        if n % 4:
            raise SystemExit(f"gen_sweep: N = {n} is not a multiple of 4 (four template groups)")
        # the six sweep sizes keep the seeds they were first generated with; any other N (a finer grid) gets its own
        i = SIZES.index(n) if n in SIZES else None
        noise_seed = 20261100 + i if i is not None else 20261300 + n
        line_seed = 20261200 + i if i is not None else 20261500 + n
        if not (out / f"noise_N{n}.csv").exists():
            write(out / f"noise_N{n}.csv", noise_rows(n, noise_seed))
        if not (out / f"line_N{n}.csv").exists():
            write(out / f"line_N{n}.csv", line_rows(n, line_seed))


if __name__ == "__main__":
    main(sys.argv[1], [int(a) for a in sys.argv[2:]] or SIZES)
