"""Python + CLI routes of the cross-language conformance fixture (Task 2.4).

The permanent parity gate for docs/api-contract-2.0.md §9. Two live routes assert
against the SAME reference recorded by the C++ route
(``conformance_reference.txt``):

* ``test_python_route_matches_reference`` drives the LIVE ``dtwcpp`` Tier-2
  surface (``Problem.set_band`` -> ``fill_distance_matrix`` -> ``fast_pam`` ->
  ``silhouette``/``davies_bouldin``/``dunn``) — exactly what a Python user calls.
* ``test_cli_route_matches_reference`` invokes the built ``dtwc_cl`` binary with
  the TOML fixture (``conformance.toml``) and asserts its emitted
  ``*_labels.csv`` / ``*_medoids.csv`` / ``*_silhouettes.csv``.

Fixed pipeline (kept in lockstep with cpp_conformance.cpp, conformance.toml and
test_conformance.m): load conformance_series.csv -> banded DTW (band=3) ->
FastPAM k=3 -> scores. Labels/medoids are canonicalised (sorted medoid SET; each
point labelled by the rank of its assigned medoid) so RNG-dependent medoid array
order / cluster-id numbering cannot make identical clusterings compare unequal.
"""
import math
import os
import subprocess
from pathlib import Path

import pytest

import dtwcpp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_CSV = HERE / "data" / "conformance_series.csv"
REFERENCE = HERE / "conformance_reference.txt"
TOML = HERE / "conformance.toml"

N_CLUSTERS = 3
BAND = 3
MAX_ITER = 100
SCORE_REL_TOL = 1e-12
# The CLI emits per-point silhouettes at setprecision(8) (dtwc_cl.cpp), so its
# recomputed mean is compared at 1e-6 rel; labels/medoids stay digit-identical.
CLI_SCORE_REL_TOL = 1e-6


# ---------------------------------------------------------------------------
# Shared helpers (mirror cpp_conformance.cpp exactly)
# ---------------------------------------------------------------------------
def load_series(path: Path):
    """Parse the recorded CSV: one comma-separated series per row (no header/id)."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([float(x) for x in line.split(",")])
    return rows


def canonicalise(raw_labels, raw_medoids):
    """Sorted medoid SET + each point labelled by the rank of its assigned medoid.

    ``raw_labels[i]`` indexes into ``raw_medoids`` (FastPAM contract);
    ``raw_medoids[m]`` is a series index.
    """
    sorted_medoids = sorted(int(m) for m in raw_medoids)
    rank = {m: i for i, m in enumerate(sorted_medoids)}
    labels = [rank[int(raw_medoids[int(lab)])] for lab in raw_labels]
    return labels, sorted_medoids


def read_reference(path: Path):
    ref = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        key = parts[0]
        if key in ("labels", "medoids"):
            ref[key] = [int(x) for x in parts[1:]]
        else:
            ref[key] = float(parts[1])
    return ref


def find_cli_binary():
    env = os.environ.get("DTWC_CL_BIN")
    if env and Path(env).exists():
        return Path(env)
    for base in (REPO / "build" / "bin", REPO / "bin", REPO / "build"):
        for name in ("dtwc_cl.exe", "dtwc_cl"):
            cand = base / name
            if cand.exists():
                return cand
    return None


@pytest.fixture(scope="module")
def reference():
    assert REFERENCE.exists(), (
        f"missing {REFERENCE}; regenerate via the C++ route "
        f"(DTWC_CONFORMANCE_REGEN=1 ./bin/cpp_conformance)"
    )
    return read_reference(REFERENCE)


# ---------------------------------------------------------------------------
# Python route — drives the LIVE dtwcpp Tier-2 pipeline
# ---------------------------------------------------------------------------
def test_python_route_matches_reference(reference):
    series = load_series(DATA_CSV)

    prob = dtwcpp.Problem("conformance")
    # names are cosmetic (labels/medoids/scores are name-independent); pass the
    # CLI's 1-based row names so all routes ingest identical Data.
    prob.set_data(series, [str(i + 1) for i in range(len(series))])
    prob.set_band(BAND)
    prob.fill_distance_matrix()
    res = dtwcpp.fast_pam(prob, N_CLUSTERS, MAX_ITER)  # writes labels/medoids back

    labels, medoids = canonicalise(res.labels, res.medoid_indices)

    sil = dtwcpp.silhouette(prob)
    silhouette = sum(sil) / len(sil)
    davies_bouldin = dtwcpp.davies_bouldin(prob)
    dunn = dtwcpp.dunn(prob)

    # Labels + medoids: DIGIT-IDENTICAL.
    assert labels == reference["labels"]
    assert medoids == reference["medoids"]
    assert len(medoids) == N_CLUSTERS

    # Scores: equal to 1e-12 relative.
    assert math.isclose(silhouette, reference["silhouette"], rel_tol=SCORE_REL_TOL)
    assert math.isclose(davies_bouldin, reference["davies_bouldin"], rel_tol=SCORE_REL_TOL)
    assert math.isclose(dunn, reference["dunn"], rel_tol=SCORE_REL_TOL)


# ---------------------------------------------------------------------------
# CLI route — invokes the built dtwc_cl binary with the TOML fixture
# ---------------------------------------------------------------------------
def test_cli_route_matches_reference(reference, tmp_path):
    binary = find_cli_binary()
    if binary is None:
        pytest.skip(
            "=" * 74 + "\nSKIPPING CLI conformance route: dtwc_cl binary not found.\n"
            "Build with: cmake --build build --target dtwc_cl, or set DTWC_CL_BIN.\n" + "=" * 74
        )

    proc = subprocess.run(
        [str(binary), "--config", str(TOML), "-i", str(DATA_CSV), "-o", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"dtwc_cl failed:\n{proc.stdout}\n{proc.stderr}"

    # Parse <name>_labels.csv ("name,cluster") in series-index order.
    labels_rows = (tmp_path / "conformance_labels.csv").read_text().splitlines()[1:]
    raw_labels = [int(r.split(",")[1]) for r in labels_rows if r.strip()]

    # Parse <name>_medoids.csv ("cluster,medoid_index,medoid_name"): row c gives
    # the series index of medoid-array position c (raw_medoids[c]).
    med_rows = (tmp_path / "conformance_medoids.csv").read_text().splitlines()[1:]
    raw_medoids = [0] * N_CLUSTERS
    for r in med_rows:
        if r.strip():
            c, idx = int(r.split(",")[0]), int(r.split(",")[1])
            raw_medoids[c] = idx

    labels, medoids = canonicalise(raw_labels, raw_medoids)

    # Labels + medoids: DIGIT-IDENTICAL to the recorded reference.
    assert labels == reference["labels"]
    assert medoids == reference["medoids"]
    assert len(medoids) == N_CLUSTERS

    # The CLI's own per-point silhouettes -> mean, compared to the reference at the
    # CLI's emitted precision (8 sig figs). Labels/medoids being digit-identical
    # already guarantees every score matches (scores are pure functions of the
    # identical D + partition); this pins the CLI's silhouette output too.
    sil_rows = (tmp_path / "conformance_silhouettes.csv").read_text().splitlines()[1:]
    sil = [float(r.split(",")[2]) for r in sil_rows if r.strip()]
    silhouette = sum(sil) / len(sil)
    assert math.isclose(silhouette, reference["silhouette"], rel_tol=CLI_SCORE_REL_TOL)
