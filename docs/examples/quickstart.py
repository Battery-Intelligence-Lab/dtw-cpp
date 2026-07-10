"""Executable Python version of the documentation quickstart."""

from pathlib import Path
import sys

import dtwcpp as dtwc

csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "tests/conformance/data/conformance_series.csv"
)

dtwc.device("cpu")
data = dtwc.load(csv, delimiter=",", name="quickstart")
result = dtwc.cluster(data, k=3, method="pam", band=3, device="cpu", max_iter=100)

medoids = sorted(int(m) for m in result.medoids)
rank = {m: i for i, m in enumerate(medoids)}
labels = [rank[int(result.medoids[int(label)])] for label in result.labels]
expected = [0] * 9 + [1] * 9 + [2] * 9
assert labels == expected and medoids == [4, 13, 22]

print("labels: 0x9 1x9 2x9")
print("medoids: 4 13 22")
print(f"mean silhouette: {result.score('silhouette'):.6f}")
