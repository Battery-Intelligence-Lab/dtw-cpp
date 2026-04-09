#!/usr/bin/env python3
"""Verify DTWC++ clustering output against UCR ground-truth class labels.

Computes Adjusted Rand Index (ARI) between true class labels from a UCR TSV
file and predicted cluster assignments from a DTWC++ *_labels.csv file.

Pure Python implementation -- no sklearn/scipy dependency.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from math import comb
from pathlib import Path


# -- ARI implementation -------------------------------------------------------


def _contingency(true_labels: list[int], pred_labels: list[int]) -> dict:
    """Build contingency table and marginals.

    Returns dict with keys 'nij', 'a', 'b', 'n' where:
      nij: Counter of (true, pred) pairs
      a:   Counter of true-label sizes
      b:   Counter of pred-label sizes
      n:   total number of items
    """
    assert len(true_labels) == len(pred_labels)
    n = len(true_labels)
    nij: Counter[tuple[int, int]] = Counter()
    a: Counter[int] = Counter()
    b: Counter[int] = Counter()
    for t, p in zip(true_labels, pred_labels):
        nij[(t, p)] += 1
        a[t] += 1
        b[p] += 1
    return {"nij": nij, "a": a, "b": b, "n": n}


def adjusted_rand_index(true_labels: list[int], pred_labels: list[int]) -> float:
    """Compute the Adjusted Rand Index (ARI).

    ARI = (sum C(n_ij,2) - expected) / (mean_marginal - expected)

    where expected = [sum C(a_i,2) * sum C(b_j,2)] / C(n,2)
    and   mean_marginal = 0.5 * [sum C(a_i,2) + sum C(b_j,2)]

    Returns 0.0 when the denominator is zero (all items in one cluster).
    """
    ct = _contingency(true_labels, pred_labels)
    n = ct["n"]
    if n < 2:
        return 0.0

    sum_comb_nij = sum(comb(v, 2) for v in ct["nij"].values())
    sum_comb_a = sum(comb(v, 2) for v in ct["a"].values())
    sum_comb_b = sum(comb(v, 2) for v in ct["b"].values())
    comb_n = comb(n, 2)

    expected = (sum_comb_a * sum_comb_b) / comb_n
    mean_marginal = 0.5 * (sum_comb_a + sum_comb_b)
    denom = mean_marginal - expected

    if denom == 0.0:
        return 0.0

    return (sum_comb_nij - expected) / denom


# -- I/O helpers ---------------------------------------------------------------


def load_ucr_labels(tsv_path: Path) -> dict[str, int]:
    """Read UCR TSV and return {row_name: class_label}.

    UCR format: first column is integer class label, rest are time-series
    values, tab-separated, no header.

    Row names are generated as 0-indexed strings to match DTWC++ naming when
    loading a single batch file (e.g. "0", "1", ...).
    """
    labels: dict[str, int] = {}
    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if not row:
                continue
            # First column is the class label (may be float like "1.0")
            label = int(float(row[0]))
            labels[str(idx)] = label
    return labels


def load_predicted_labels(csv_path: Path) -> dict[str, int]:
    """Read DTWC++ labels CSV (header: name,cluster)."""
    labels: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["name"]] = int(row["cluster"])
    return labels


# -- Verification --------------------------------------------------------------

# Default ARI thresholds per dataset and k.
DEFAULT_THRESHOLDS: dict[str, dict[int, float]] = {
    "Coffee": {2: 0.7},
    "Beef": {5: 0.3},
}


def verify_single(
    true_path: Path,
    pred_path: Path,
    k: int | None = None,
    name: str | None = None,
    threshold: float | None = None,
) -> tuple[bool, float]:
    """Compare predicted labels against ground truth.

    Returns (passed: bool, ari: float).
    """
    true_labels_map = load_ucr_labels(true_path)
    pred_labels_map = load_predicted_labels(pred_path)

    # Align on common names
    common_names = sorted(set(true_labels_map) & set(pred_labels_map))
    if not common_names:
        print(f"ERROR: no common names between {true_path} and {pred_path}")
        return False, 0.0

    true_vec = [true_labels_map[n] for n in common_names]
    pred_vec = [pred_labels_map[n] for n in common_names]

    ari = adjusted_rand_index(true_vec, pred_vec)

    # Determine threshold
    if threshold is None and name is not None and k is not None:
        threshold = DEFAULT_THRESHOLDS.get(name, {}).get(k)
    if threshold is None:
        threshold = 0.0  # any non-negative ARI is a pass if no threshold set

    passed = ari >= threshold
    status = "PASS" if passed else "FAIL"
    label = name or pred_path.stem
    detail = f"k={k}" if k is not None else ""

    print(f"[{status}] {label} {detail}  ARI={ari:.4f}  (threshold={threshold:.2f}, n={len(common_names)})")
    return passed, ari


def scan_results_dir(
    results_dir: Path,
    true_path: Path,
    k: int | None = None,
    name: str | None = None,
    threshold: float | None = None,
) -> bool:
    """Scan a directory tree for *_labels.csv files and verify each."""
    label_files = sorted(results_dir.rglob("*_labels.csv"))
    if not label_files:
        print(f"No *_labels.csv files found under {results_dir}")
        return False

    all_pass = True
    for lf in label_files:
        passed, _ = verify_single(true_path, lf, k=k, name=name, threshold=threshold)
        if not passed:
            all_pass = False

    return all_pass


# -- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify DTWC++ clustering against UCR ground-truth labels."
    )
    parser.add_argument(
        "--true", required=True, type=Path,
        help="Path to UCR TSV file (first column = class label).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--predicted", type=Path,
        help="Path to a single DTWC++ *_labels.csv file.",
    )
    group.add_argument(
        "--results-dir", type=Path,
        help="Directory to scan for *_labels.csv files.",
    )
    parser.add_argument("--k", type=int, default=None, help="Number of clusters.")
    parser.add_argument("--name", type=str, default=None, help="Dataset name (e.g. Coffee).")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum ARI to pass.")

    args = parser.parse_args()

    if not args.true.exists():
        print(f"ERROR: true-labels file not found: {args.true}")
        sys.exit(1)

    if args.predicted:
        if not args.predicted.exists():
            print(f"ERROR: predicted-labels file not found: {args.predicted}")
            sys.exit(1)
        passed, _ = verify_single(
            args.true, args.predicted, k=args.k, name=args.name, threshold=args.threshold
        )
    else:
        if not args.results_dir.exists():
            print(f"ERROR: results directory not found: {args.results_dir}")
            sys.exit(1)
        passed = scan_results_dir(
            args.results_dir, args.true, k=args.k, name=args.name, threshold=args.threshold
        )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
