"""
@file 09_device_clustering.py
@brief Unified device clustering demo — cpu / gpu / hpc with a single switch.
@details
    Set the device once, then cluster. The library handles everything else
    (device resolution, local-vs-remote execution, plotting):

        dtwc.device("hpc")            # cpu | gpu | hpc
        data = dtwc.load(source)      # lazy handle (no local read on hpc)
        res  = dtwc.cluster(data, k=3)
        res.plot()

      cpu : local CPU.
      gpu : local GPU (CUDA); auto-falls back to CPU if unavailable.
      hpc : offload the whole job to a SLURM cluster (data never read locally).

    Works on Windows and macOS. Requires numpy + matplotlib; hpc also needs a
    configured .env at the repo root and ssh + rsync (Git Bash on Windows).

    Usage:
        python 09_device_clustering.py cpu
        python 09_device_clustering.py gpu
        python 09_device_clustering.py hpc                       # synthetic data, offloaded
        python 09_device_clustering.py cpu path/to/UCR_TRAIN.tsv # real data (col 0 = label)
@author Volkan Kumtepeli
"""
import os
import sys
from pathlib import Path

import numpy as np
import dtwcpp as dtwc


def make_demo_data(seed=0):
    """3 well-separated shape groups (sine / square / ramp), 20 series each."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, 50)
    shapes = (np.sin(t), np.sign(np.sin(t)), t / t.max())
    return np.array([np.roll(sh, rng.integers(-3, 4)) + 0.15 * rng.standard_normal(t.size)
                     for sh in shapes for _ in range(20)])


def main(device="gpu", k=3, source=None):
    dtwc.device(device)                                  # cpu | gpu | hpc  (set once)
    if device == "hpc":                                  # tell the offloader where the repo is
        os.environ.setdefault("DTWC_REPO_ROOT", str(Path(__file__).resolve().parents[2]))

    data = dtwc.load(source if source is not None else make_demo_data())
    res = dtwc.cluster(data, k=k)                         # local for cpu/gpu, offloaded for hpc
    print(res.summary())
    res.plot()


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "gpu"    # cpu | gpu | hpc
    # Optional 2nd arg: a UCR .tsv path (column 0 = class label -> skip_cols=1).
    src = dtwc.load(sys.argv[2], skip_cols=1) if len(sys.argv) > 2 else None
    main(device=dev, source=src)
