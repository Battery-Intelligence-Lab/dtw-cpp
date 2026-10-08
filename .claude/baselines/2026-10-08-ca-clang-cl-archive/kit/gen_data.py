"""Write CA's timing data once: fixed-seed random walks, one series per CSV row, in data/ with a SHA-256 manifest.

uv run --no-project python gen_data.py
  a.csv  2000 series of length 100            equal length, timed unbanded
  b.csv  500 series of length 1000            equal length, timed with --band 100
  c.csv  1000 series of lengths uniform in 90..110, ragged, timed unbanded
  <shape>_smoke.csv  the first 160, 40 and 120 series of each, for the kit's smoke round only
Each series is its own walk, the cumulative sum of N(0, 1) steps (random.Random(SEED).gauss); repr writes each
double so that it reads back exactly.
"""
import hashlib
import json
import pathlib
import random

DATA = pathlib.Path(__file__).resolve().parent / "data"
SEED = 20261008
SMOKE_N = {"a": 160, "b": 40, "c": 120}


def walk(rng, length):
    value, series = 0.0, []
    for _ in range(length):
        value += rng.gauss(0.0, 1.0)
        series.append(value)
    return series


def main():
    if (DATA / "manifest.json").exists():
        raise SystemExit(f"{DATA} already holds the data; it is generated once")
    DATA.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    shapes = {"a": [walk(rng, 100) for _ in range(2000)], "b": [walk(rng, 1000) for _ in range(500)]}
    shapes["c"] = [walk(rng, length) for length in [rng.randint(90, 110) for _ in range(1000)]]
    manifest = {"seed": SEED, "files": {}}
    for name, series in shapes.items():
        for suffix, rows in (("", series), ("_smoke", series[:SMOKE_N[name]])):
            path = DATA / f"{name}{suffix}.csv"
            path.write_bytes("".join(",".join(map(repr, s)) + "\n" for s in rows).encode("ascii"))
            manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                            "lengths": [len(s) for s in rows]}
    (DATA / "manifest.json").write_text(json.dumps(manifest) + "\n")
    for name, entry in manifest["files"].items():
        print(name, len(entry["lengths"]), "series,", min(entry["lengths"]), "-", max(entry["lengths"]), "samples,",
              entry["sha256"][:16])


if __name__ == "__main__":
    main()
