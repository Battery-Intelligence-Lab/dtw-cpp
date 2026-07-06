# Handoff — PyTorch-style device API + HPC auto-dispatch (2026-06-30)

## What was asked
A ~10-line clustering demo with a PyTorch-style device switch (`cpu` / `gpu` / `hpc`),
timing, and a 2D cluster plot; ARC/SLURM commands to check jobs. User then opted in to
(1) full native `device="hpc"` auto-dispatch and (2) promoting `gpu`/`hpc` + a global
`dtwcpp.device()` into the core library.

## State found (verified, not assumed)
- Python pkg `dtwcpp` v2.0.0 (nanobind). Device API existed only as `cpu`/`cuda`/`cuda:N`
  per-call kwarg; `"cuda"` already auto-falls-back to CPU. No global setter; `gpu`/`hpc`
  raised ValueError; no HPC auto-dispatch. Crop not bundled (24 classes anyway).
- **The active install is a non-editable wheel** (`Z:/dist/dtwcpp-2.0.0-...whl`) →
  `import dtwcpp` resolves to site-packages, so repo `python/dtwcpp/*.py` edits are NOT
  live until reinstall. Tested via a scratch overlay (installed pkg copy + repo `.py` on
  top, `PYTHONPATH`).

## What was built (all changes in repo source)
**Piece 1 — native device names + global setter** ([python/dtwcpp/__init__.py](../../python/dtwcpp/__init__.py))
- `_parse_device` accepts `gpu` (→cuda, CPU fallback) and `hpc` (execution location).
- `dtwcpp.device(name)` setter + `dtwcpp.get_device()` getter (PyTorch-style global default).
- `compute_distance_matrix(device=None)` → resolves None to global; rejects `hpc`.

**Piece 2 — `device="hpc"` auto-dispatch**
- [python/dtwcpp/_hpc.py](../../python/dtwcpp/_hpc.py) — `write_series_tsv`, `parse_labels_csv`
  (maps dtwc_cl 1-based `name,cluster` back to input order; robust to lexical row sort),
  `build_dtwc_command`, `find_dtwc_binary`, `SlurmRemoteRunner`, `cluster_on_hpc`.
- [python/dtwcpp/_clustering.py](../../python/dtwcpp/_clustering.py) — `DTWClustering(device=None)`
  default; `device="hpc"` dispatches to `cluster_on_hpc` (sets `labels_` only).
- [scripts/slurm/jobs/cluster_generic.slurm](../../scripts/slurm/jobs/cluster_generic.slurm) —
  generic env-parametrized job (first non-hardcoded-dataset job).
- [scripts/slurm/slurm_remote.sh](../../scripts/slurm/slurm_remote.sh) — new `submit-cluster` subcommand.
- [examples/python/09_device_clustering.py](../../examples/python/09_device_clustering.py) — the demo.

**Piece 3 — unified interface** (user asked: no internals leaked; set device then "just cluster";
hpc must not load data locally). [python/dtwcpp/_api.py](../../python/dtwcpp/_api.py):
- `dtwc.load(source, skip_cols=…)` → lazy `Dataset` (path or array; a path is NOT read on hpc).
- `dtwc.cluster(data, k, device=None)` → reads global device; local FastPAM for cpu/gpu, offload
  for hpc; returns `ClusterResult` (`labels`, timing, `cost`/`medoid_indices`, `summary()`, `plot()`).
- The MDS 2D plot moved out of the example INTO `ClusterResult.plot()`. `device="hpc"` with a path
  source passes it through to the cluster (no local read; `_hpc.cluster_on_hpc` + `submit-cluster`
  gained an `upload=0` remote-path mode; `parse_labels_csv` infers N when unknown).
- Example rewritten to ~6 lines: `device()` → `load()` → `cluster()` → `summary()`/`plot()`.
- 12 tests in [tests/python/test_api.py](../../tests/python/test_api.py) incl. "hpc does not read a path locally".

## Verification
- Full Python suite via overlay: **201 passed, 10 skipped** (baseline 178 → +23: test_device.py 11,
  test_hpc.py 12; updated 1 existing default-device test). No regressions.
- Demo cpu + gpu(→CPU fallback) run end-to-end; identical clustering (cost 354.21); MDS plot saved.
- **Real local round-trip** (`tests/python/test_hpc.py::TestLocalRoundTrip`): serialize → run
  `build/bin/dtwc_cl.exe` → parse labels → 2 groups recovered. This is the cluster job minus ssh/rsync.
- Shell scripts pass `bash -n`; `submit-cluster` dispatches.

## NOT verified — user must do
1. **Reinstall** for changes to go live: `pip install -e .` (rebuilds C++) or rebuild the wheel.
   Current `.venv` still runs the OLD wheel.
2. **Remote submission on ARC** cannot be tested from a laptop. One-time: configure `.env`
   (already present at repo root), `bash scripts/slurm/slurm_remote.sh build htc-cpu`, then
   `DTWClustering(device="hpc").fit(X)` or `examples/python/09_device_clustering.py hpc`.

## Most-likely-wrong claim
The remote `submit-cluster` → poll → download → `NAME_labels.csv` retrieval chain is unverified
end-to-end on a real cluster; the path handling (repo-relative input, GRES flags, output landing
under `results/`) is my best reading of the existing wrapper but needs one real ARC run.

## Open
- Could also download `NAME_medoids.csv` to populate `medoid_indices_`/enable `predict` after hpc fit.
- Could expose C++ `cluster_by_kMedoidsLloyd` to Python if exact Lloyd (not FastPAM) is wanted.
