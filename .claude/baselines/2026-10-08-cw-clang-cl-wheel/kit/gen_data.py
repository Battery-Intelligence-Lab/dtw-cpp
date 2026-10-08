"""Write the WM timing data once: fixed-seed random walks, float64, in data/ with a SHA-256 manifest.

Run once, with either venv's python (numpy): C:/D/git/wt/venv/wm-msvc/Scripts/python.exe gen_data.py
(a) 2000 x 100, (b) 500 x 1000, (c) 1000 series of lengths uniform in 90..110 (values concatenated in
c_values.npy, lengths in c_lengths.npy), (d) 2 x 1000. Each series is its own walk: cumsum of N(0, 1) steps.
"""
import hashlib
import json
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).resolve().parent / "data"
SEED = 20261007


def main():
    if (DATA / "manifest.json").exists():
        raise SystemExit(f"{DATA} already holds the data; it is generated once")
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    lengths = rng.integers(90, 111, size=1000)  # 90..110 inclusive
    arrays = {
        "a": np.cumsum(rng.standard_normal((2000, 100)), axis=1),
        "b": np.cumsum(rng.standard_normal((500, 1000)), axis=1),
        "c_values": np.concatenate([np.cumsum(rng.standard_normal(n)) for n in lengths]),
        "c_lengths": lengths.astype(np.int64),
        "d": np.cumsum(rng.standard_normal((2, 1000)), axis=1),
    }
    manifest = {"seed": SEED, "numpy": np.__version__, "files": {}}
    for name, values in arrays.items():
        path = DATA / f"{name}.npy"
        np.save(path, values)
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
