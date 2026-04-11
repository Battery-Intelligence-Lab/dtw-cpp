# Session Handoff: SLURM HPC Integration (2026-04-09)

Generalized SLURM scripts, CLI fixes (`--dtype`, `--gpu-precision`, `--resume`), verbose diagnostics, and end-to-end pipeline tested on Oxford ARC (CPU + GPU + checkpoint/resume).

## What Was Done

### 1. CLI Fixes (dtwc/dtwc_cl.cpp)

- **`--precision` crash bug fixed**: CLI11 threw `OptionAlreadyAdded` because `--precision` was defined twice (data + GPU). Split into:
  - `--dtype` (aliases: `--data-precision`, `--data-type`, f32/fp32/float/f64/fp64/double) — default `float32`
  - `--gpu-precision` (alias: `--gpu-dtype`, same alias set + `auto`) — default `auto`
- **`--restart` renamed to `--resume`** (`--restart` kept as hidden deprecated alias)
- **YAML config bug fixed**: data precision was never loaded from YAML (only GPU precision was bound). Added `set_if_unset("dtype", dtype_str)`, `set_if_unset("gpu-precision", gpu_precision)`, `set_if_unset("resume", resume)`
- **YAML normalization**: added alias resolution for dtype/gpu-precision from YAML (bypasses CLI11's CheckedTransformer)
- **TOML/YAML example configs updated**: `precision` key split into `dtype` + `gpu-precision`
- **Pre-existing YAML bug documented**: `set_if_unset` always overrides CLI values if YAML key exists (TODO comment at line 299)

### 2. Verbose Diagnostics (dtwc/dtwc_cl.cpp)

Added to `--verbose` output:
- OpenMP thread count + `OMP_NUM_THREADS` env
- SLURM env vars: `SLURM_JOB_ID`, `SLURM_CPUS_PER_TASK`, `SLURM_NODELIST`, `SLURM_JOB_PARTITION`, `SLURM_GPUS`
- CPU model name (Linux `/proc/cpuinfo`)
- Process memory (`VmPeak`, `VmRSS` from `/proc/self/status`)
- Data memory estimate after loading: `~N MB (X series, Y avg length, dtype)`

### 3. SLURM Script Infrastructure (scripts/slurm/)

**Pure bash** — zero Python dependency. Uses `ssh` + `scp` (with `rsync` auto-detected as preferred when available).

| File | Purpose |
|------|---------|
| `slurm_remote.sh` | Main CLI: test, upload, build, submit-*, status, download, ssh, interactive |
| `env.example` | Config template with Oxford ARC example (anonymized) |
| `build-arc.sh` | 6 build profiles (now with env-var overrides: `DTWC_BUILD_TESTING`, `DTWC_ENABLE_ARROW`) |
| `jobs/cpu_test.slurm` | Coffee k=2 + Beef k=5 |
| `jobs/gpu_test.slurm` | CUDA fp32 + fp64 |
| `jobs/checkpoint_test.slurm` | Resume verification (diff labels) |
| `jobs/parquet_test.slurm` | Parquet + .dtws I/O (not yet tested — needs Arrow build fix) |
| `README.md` | Index + ARC quick-reference |

**Key design decisions:**
- `.env` at project root (in `.gitignore`) stores credentials + paths
- Upload uses explicit allowlist (dtwc/, cmake/, scripts/, data/) — never uploads `.env`, `.git/`, `build*/`
- `scp` fallback when `rsync` not available (Windows Git Bash doesn't ship rsync)
- SLURM jobs use portable scratch: `WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"`
- Build submits batch job to interactive partition (not login node)
- `DTWC_BUILD_TESTING=OFF` and `DTWC_ENABLE_ARROW=OFF` for remote builds (tests/ not uploaded, Arrow CPM has include path issue)
- Data path resolution: jobs check `../data/` (slurm_remote.sh layout) then `data/benchmark/UCRArchive_2018/` (local layout)

### 4. Benchmark Scripts (benchmarks/)

| File | Purpose |
|------|---------|
| `verify_results.py` | Pure Python ARI comparison against UCR ground truth |
| `convert_ucr.py` | UCR TSV → Parquet converter (uses pyarrow) |

### 5. Documentation

- `docs/content/getting-started/cli.md` — updated flag names, added `--resume`, `--mmap-threshold`
- `docs/content/getting-started/slurm.md` — new page: prerequisites, config, workflow, ARC reference
- `docs/content/getting-started/configuration.md` — `precision` → `gpu-precision`
- `scripts/slurm/README.md` — file index, quick-start, ARC quick-reference
- `CHANGELOG.md` — new SLURM section, CLI changes, bug fixes

### 6. Cleanup

- `arc/` folder deleted (15 files with hardcoded credentials)
- All personal identifiers removed from committed files (env.example uses anonymized placeholders)

### 7. Adversarial Review (4 agents)

2 during planning + 2 post-implementation (security + correctness). Key findings fixed:
- Wrong binary name in preflight check (`dtwc_test` → `bin/dtwc_cl`)
- Exit code always 0 (`|| true` pattern → `|| DTWC_EXIT=$?`)
- Stale `--precision` in configuration.md
- Missing `--name` flag in SLURM templates
- PII in env.example (anonymized)
- `$SCRATCH` portability (fallback chain added)
- `-k` not `--k` in SLURM templates (CLI11 single-char flag)

## ARC Test Results

All tested on Oxford ARC HTC cluster, 2026-04-09.

### Build Performance

| Profile | Time | Node | Notes |
|---------|------|------|-------|
| htc-cpu (no Arrow) | ~5 min | htc-g048 | First build (CPM downloads deps). Binary: 963 KB |
| htc-gpu (no Arrow) | ~8 min | htc-g048 | CUDA archs 60-90. Binary: 5.1 MB |

### CPU Test (Job 7457674)

| Dataset | N | Length | k | Iterations | Cost | Silhouette | DTW Time | Total |
|---------|---|--------|---|------------|------|------------|----------|-------|
| Coffee | 28 | 286 | 2 | 1 | 294.778 | 0.2264 | 49ms | 0.22s |
| Beef | 30 | 470 | 5 | 6 | 646.668 | 0.5961 | 52ms | 0.06s |

- **Node**: htc-c045 (Intel Xeon Platinum 8268 @ 2.90GHz)
- **OpenMP**: 8 threads
- **Exit**: 0 (both tests)

### GPU Test (Job 7457696)

| Dataset | Precision | DTW Pairs | DTW Time | Cost | Total |
|---------|-----------|-----------|----------|------|-------|
| Coffee k=2 | FP32 | 378 | CUDA kernel | 294.778 | ~0.35s |
| Coffee k=2 | FP64 | 378 | 1.95ms | 294.778 | 0.35s |

- **Node**: htc-g077 (Intel Xeon Gold 6548N)
- **GPU**: NVIDIA L40S (compute 8.9, 46 GB VRAM, CUDA 13.0)
- **Result**: FP32 and FP64 produce identical clustering (cost=294.778)
- **Exit**: 0 (both tests)

### Checkpoint/Resume Test (Job 7457679)

| Run | Checkpoint | DTW Time | Total | Cost |
|-----|------------|----------|-------|------|
| Fresh | Save 406 entries | 63ms | 0.21s | 294.778 |
| Resume | Load 406 entries | 18ms (skip DTW) | 0.03s | 294.778 |

- **Speedup from resume**: 6.5x (distance matrix loaded from cache)
- **Label comparison**: PASS (identical)

## Current State

- **Branch**: Claude (uncommitted changes — user asked not to commit)
- **Tests**: 67/67 pass locally (2 CUDA skipped, no NVIDIA GPU on dev machine)
- **ARC builds**: htc-cpu and htc-gpu binaries present at `/data/engs-unibatt-gp/engs2321/dtw-cpp/src/build-*/`

## What To Do Next

### Immediate: Full UCR Benchmark on ARC
1. Upload ALL UCR datasets (128 datasets in `data/benchmark/UCRArchive_2018/`)
2. Create a benchmark job script that iterates over all datasets, runs each with appropriate k (from `UCR_DataSummary.csv` class count column)
3. Run on CPU (htc-c045 class nodes, 8+ cores)
4. Run on different GPU types: L40S (done), request V100, A100, H100 via `--gres=gpu:v100:1` etc.
5. Collect timing JSON files, aggregate into benchmark report
6. Use `benchmarks/verify_results.py` to compute ARI for each dataset
7. Measure: DTW time, clustering time, total time, memory usage, queue wait time (via sacct)

### Arrow Build Fix
- Arrow CPM build succeeds (libparquet.a at 97%) but include path not propagated to DTWC++ sources
- `fast_clara.cpp` includes `parquet_chunk_reader.hpp` which needs `<arrow/api.h>`
- Likely needs `target_include_directories` fix in CMakeLists.txt for the CPM-built Arrow
- Once fixed: re-enable `DTWC_ENABLE_ARROW=ON` in slurm_remote.sh and test Parquet I/O on ARC

### Benchmark Report for Website
- Aggregate timing JSON into a summary (JSON or CSV)
- Create plots: DTW time vs dataset size, GPU vs CPU speedup, scaling across GPU types
- Integrate into Hugo docs site

### Remaining Polish
- Commit all changes (user will review first)
- Consider keeping both `slurm_remote.sh` (bash) and `slurm_remote.py` (Python) — user expressed preference for both options
- Python version needs its own `pyproject.toml` inside `scripts/slurm/` (was created but deleted when switching to bash)

## Key Files

| File | Role |
|------|------|
| `dtwc/dtwc_cl.cpp` | CLI: `--dtype`, `--gpu-precision`, `--resume`, diagnostics |
| `scripts/slurm/slurm_remote.sh` | Pure bash remote helper (13 commands) |
| `scripts/slurm/env.example` | Config template |
| `scripts/slurm/build-arc.sh` | Build profiles (env-var overrides added) |
| `scripts/slurm/jobs/*.slurm` | 4 test job templates |
| `benchmarks/verify_results.py` | ARI ground truth comparison |
| `benchmarks/convert_ucr.py` | TSV → Parquet converter |
| `docs/content/getting-started/slurm.md` | Hugo docs page |
| `.env` | User credentials (NEVER commit) |

## Known Issues

- **Arrow CPM include path**: Arrow builds from source on ARC but headers not found during DTWC++ compilation. Workaround: `DTWC_ENABLE_ARROW=OFF`
- **YAML `set_if_unset` override bug**: pre-existing — YAML values always override CLI flags. Needs refactor to use `app["--flag"]->count()`. Documented as TODO.
- **rsync not in Git Bash**: Windows Git Bash doesn't ship rsync. Scripts fall back to scp. Can install rsync via `pacman -S rsync` in MSYS2/Git Bash.
- **L40S FP64**: L40S has 1:64 FP64 rate (consumer-class). For FP64-intensive workloads, use A100/H100 (`--gres=gpu:a100:1`).