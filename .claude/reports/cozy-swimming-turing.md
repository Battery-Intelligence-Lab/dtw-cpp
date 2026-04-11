# Plan: Generalized SLURM HPC Scripts for DTWC++

## Context

The project has a reference `arc/` folder from another project (GP-ECM battery pipeline) with hardcoded Oxford credentials, paths, and project-specific logic. We need to generalize these into portable SLURM scripts under `scripts/slurm/` that work on **any** SLURM cluster, using `.env` for configuration. We also fix CLI bugs, add debug diagnostics, test on ARC with small UCR datasets, verify results, and document everything.

The existing `scripts/slurm/build-arc.sh` is already well-structured with 6 build profiles -- we keep it as-is.

---

## Phase 1: CLI Fixes

### 1a. Rename and unify precision flags

**File:** `dtwc/dtwc_cl.cpp`

**Bug:** `--precision` defined twice (line 148 for data, line 270 for GPU).

**New naming convention:**

| Flag | Primary | Aliases | Default | Variable |
|------|---------|---------|---------|----------|
| Data type | `--dtype` | `--data-precision`, `--data-type` | `float32` | `dtype_str` |
| GPU precision | `--gpu-precision` | `--gpu-dtype` | `auto` | `gpu_precision_str` |

Both accept: `float32`, `f32`, `fp32`, `float` -> "float32"; `float64`, `f64`, `fp64`, `double` -> "float64". GPU precision also accepts `auto` (default -- lets GPU select based on capability). **Keep `auto` as GPU default** to avoid silently downgrading FP64-capable GPUs.

**NOTE:** The duplicate `--precision` is a **crash bug** -- CLI11 throws `OptionAlreadyAdded` at startup. This must be fixed immediately.

**Changes in `dtwc/dtwc_cl.cpp`:**

- Line 148-155: rename `precision_str` -> `dtype_str`, flag `--precision` -> `--dtype,--data-precision,--data-type`
- Line 268-270: rename `precision` -> `gpu_precision_str`, flag `--precision` -> `--gpu-precision,--gpu-dtype`, keep default `"auto"`, add alias map (f32/fp32/float->float32, f64/fp64/double->float64, plus auto)
- Update all downstream references (~6 occurrences for GPU variable at lines 312, 389, 556-557, 631-633)
- Line 389 verbose output: show both `Dtype` and `GPU Precision`
- **YAML fix (BUG):** Line 312 `set_if_unset("precision", precision)` binds to GPU var only; data precision was NEVER loaded from YAML. Fix: split into `set_if_unset("dtype", dtype_str)` AND `set_if_unset("gpu-precision", gpu_precision_str)`
- **TOML fix:** TOML config key `precision` must also split into `dtype` and `gpu-precision`. Update any example TOML/YAML configs.
- **Add YAML for resume:** Add `set_if_unset("resume", resume)` (currently `--restart` has no YAML binding)

### 1b. Rename `--restart` to `--resume`

**File:** `dtwc/dtwc_cl.cpp:236`

```cpp
// Before:
app.add_flag("--restart", restart, "Resume from checkpoint ...");
// After:
app.add_flag("--resume", resume, "Resume from checkpoint (distance matrix cache + clustering state)");
```

Rename the variable `restart` -> `resume` throughout the file. Keep `--restart` as a **hidden deprecated alias** for one release cycle (print deprecation warning when used). This avoids breaking existing user scripts silently.

### 1c. Update docs

**File:** `docs/content/getting-started/cli.md`

- Precision table: `--precision` -> `--dtype` with aliases listed
- GPU Options table (line 124): already says `--gpu-precision` -- verify it matches
- Line 188 example: `--precision fp32` -> `--gpu-precision fp32`
- Checkpoint table: `--restart` -> `--resume`
- Checkpoint example (line 198-204): use `--resume`

---

## Phase 2: Debug Diagnostics

**File:** `dtwc/dtwc_cl.cpp` -- insert after CLI parse in the `if (verbose)` block (~line 399)

### 2a. System diagnostics block (printed to stdout, captured in SLURM `.out` log)

```
=== DTWC++ System Diagnostics ===
  OpenMP threads:    8
  OMP_NUM_THREADS:   8
  SLURM_CPUS_PER_TASK: 8  (if set)
  SLURM_JOB_ID:     12345 (if set)
  SLURM_NODELIST:    htc-c001 (if set)
  CPU:               [/proc/cpuinfo model name if Linux]
  CUDA device:       NVIDIA H100 80GB HBM3 (if --device cuda)
  GPU SMs:           132
  GPU memory:        80 GB
  Compute cap:       9.0
  FP64 rate:         full (1:2)
================================
```

Key: print SLURM env vars so `slurm.out` logs are self-documenting. SLURM sets `SLURM_CPUS_PER_TASK`, `SLURM_JOB_ID`, `SLURM_NODELIST`, `SLURM_GPUS` etc. -- just read from `std::getenv()`.

### 2b. Data memory estimate (after loading)

```
  Data memory: ~12 MB (28 series, 286 avg length, float64)
```

### 2c. Phase timing (already exists via `dtwc::Clock`, no changes needed)

---

## Phase 3: SLURM Script Infrastructure

### 3a. File structure

```
scripts/slurm/
  build-arc.sh              (KEEP -- already exists, 6 profiles)
  slurm_remote.py           (NEW -- generalized from arc/arc_remote.py)
  env.example               (NEW -- .env template, committed to git)
  README.md                 (NEW -- index + ARC quick-reference)
  jobs/
    cpu_test.slurm           (NEW)
    gpu_test.slurm           (NEW)
    checkpoint_test.slurm    (NEW)
    parquet_test.slurm       (NEW)
  verify_results.py          (NEW -- ground-truth comparison)
  convert_ucr.py             (NEW -- TSV -> Parquet + .dtws conversion)
```

### 3b. Configuration: `scripts/slurm/env.example`

```bash
# === SLURM Remote Configuration ===
# Copy this file to `.env` at the PROJECT ROOT directory and edit all values.
# .env is in .gitignore and will NEVER be committed.

# --- Connection (required) ---
SLURM_USER=your_username
SLURM_HOST=login-node.your-cluster.edu

# Two-hop gateway (leave empty for direct SSH)
SLURM_GATEWAY=

# --- Authentication ---
# SSH key is RECOMMENDED (most secure). Password is fallback.
# SLURM_SSH_KEY=~/.ssh/id_ed25519
SLURM_PASSWORD=

# --- Remote paths ---
# Where your cluster home and data directories are.
SLURM_HOME_FOLDER=/home/${SLURM_USER}
SLURM_DATA_FOLDER=/data/your-project/${SLURM_USER}
SLURM_REMOTE_BASE=${SLURM_DATA_FOLDER}/dtw-cpp

# --- Local paths ---
# Relative to project root. Which test datasets to upload.
SLURM_LOCAL_DATA=data/benchmark/UCRArchive_2018

# --- Cluster defaults ---
SLURM_PARTITION=short
SLURM_CLUSTER=                    # e.g. "htc" (leave empty if single cluster)
SLURM_GPU_GRES=gpu:1              # GPU resource request, e.g. gpu:v100:1

# --- Notifications (optional) ---
SLURM_EMAIL=

# === Oxford ARC Example ===
# SLURM_USER=engs2321
# SLURM_HOST=htc-login.arc.ox.ac.uk
# SLURM_GATEWAY=                  # direct SSH from university network/VPN
# SLURM_HOME_FOLDER=/home/engs2321
# SLURM_DATA_FOLDER=/data/engs-unibatt-gp/engs2321
# SLURM_REMOTE_BASE=/data/engs-unibatt-gp/engs2321/dtw-cpp
# SLURM_PARTITION=short
# SLURM_CLUSTER=htc
# SLURM_GPU_GRES=gpu:1
```

### 3c. Security approach for `slurm_remote.py`

Based on independent security review:

1. **SSH key auth is recommended default** -- check `~/.ssh/id_ed25519` or `SLURM_SSH_KEY` first
2. **Password as fallback** -- read from `os.environ.get("SLURM_PASSWORD")` first, then `.env` file, then `getpass.getpass()` interactive prompt
3. **Clear password from memory** after `paramiko.connect()`: `password = None` (note: CPython strings are immutable so this is best-effort, not cryptographic -- document this limitation)
4. **Replace `AutoAddPolicy()`** with `RejectPolicy()` + `load_system_host_keys()` as default. Provide `--trust-unknown-hosts` flag for first-time setup. `WarningPolicy()` is insufficient -- it warns but still connects, enabling MITM.
5. **Never log or print credentials** -- the script must never echo the password. Diagnostics must use a hardcoded allowlist of safe SLURM env vars (SLURM_JOB_ID, SLURM_CPUS_PER_TASK, SLURM_NODELIST, etc.) -- never iterate or print `*PASSWORD*`/`*SECRET*`/`*KEY*` vars
6. **SFTP upload must exclude `.env`** -- use explicit allowlist of directories to upload (dtwc/, cmake/, scripts/, data/) rather than recursive glob from project root. Hardcode exclusion of `.env`, `.git/`, `build*/`, `*.pdb`

The Config dataclass:

```python
@dataclasses.dataclass
class Config:
    user: str
    host: str
    gateway: str = ""
    password: str = ""       # cleared after use
    ssh_key: str = ""
    home_folder: str = ""
    data_folder: str = ""
    remote_base: str = ""
    local_data: str = ""
    partition: str = "short"
    cluster: str = ""
    gpu_gres: str = "gpu:1"
    email: str = ""
```

### 3d. `slurm_remote.py` commands

| Command | Purpose |
|---------|---------|
| `test` | Test SSH connection, print hostname + cluster info |
| `setup-keys` | Copy SSH public key to authorized_keys |
| `upload` | Upload source + test data to `$SLURM_REMOTE_BASE` |
| `build [profile]` | Submit short batch build job (non-interactive), poll for completion |
| `submit-cpu` | Submit `jobs/cpu_test.slurm` |
| `submit-gpu` | Submit `jobs/gpu_test.slurm` |
| `submit-checkpoint` | Submit `jobs/checkpoint_test.slurm` |
| `submit-parquet` | Submit `jobs/parquet_test.slurm` |
| `status` | `squeue -u $USER` + completion summary |
| `stats [--job ID]` | `sacct` timing statistics |
| `download` | Rsync results + logs back to local |
| `ssh "cmd"` | Run arbitrary command |
| `interactive` | Print instructions for starting interactive debug session |

**Build command** uses batch job submission (not interactive srun via paramiko):

```python
def cmd_build(config, profile="htc-gpu"):
    """Submit a short batch job to build on a compute node."""
    build_script = f"""#!/bin/bash
#SBATCH --partition=interactive
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=dtwc-build
module load CMake/3.27.6 GCC/13.2.0 CUDA/12.4.0 2>/dev/null || true
cd {config.remote_base}/src
source scripts/slurm/build-arc.sh {profile}
"""
    # Upload build script, sbatch, poll squeue until done
```

For interactive debugging, print:
```
ssh {user}@{host}
srun -p interactive --pty /bin/bash
cd {remote_base}/src
module load CMake/3.27.6 GCC/13.2.0 CUDA/12.4.0
source scripts/slurm/build-arc.sh {profile}
```

### 3e. SLURM Job Templates

All jobs use a portable scratch directory for I/O. `$SCRATCH` is ARC-specific; standard SLURM only provides `$TMPDIR`. Each job starts with:

```bash
WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "$WORKDIR"
cd "$WORKDIR" || exit 1
```

Copy data in, run, copy results out. All variable expansions are **quoted** to handle spaces. Results include a `.json` timing file for benchmarking.

#### `jobs/cpu_test.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=dtwc-cpu-test
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/cpu_test_%j.out
#SBATCH --error=logs/cpu_test_%j.err

echo "=== DTWC++ CPU Test ==="
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "  Partition:  ${SLURM_JOB_PARTITION}"
echo "  Submit:     ${SLURM_JOB_START_TIME:-$(date)}"
echo "  Queue wait: computed after job completes via sacct"
echo ""

# Copy to $SCRATCH
WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "$WORKDIR"
cd "$WORKDIR" || exit 1
rsync -a "${SLURM_SUBMIT_DIR}/data/" ./data/
cp "${SLURM_SUBMIT_DIR}/build-htc-cpu/bin/dtwc_cl" ./

module purge 2>/dev/null || true
module load GCC/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Test 1: Coffee k=2
echo "--- Coffee k=2 ---"
./dtwc_cl --input data/Coffee/Coffee_TRAIN.tsv --skip-cols 1 \
    -k 2 --method pam --dtype float64 \
    --output results/coffee_k2 --name coffee --verbose

# Test 2: Beef k=5
echo "--- Beef k=5 ---"
./dtwc_cl --input data/Beef/Beef_TRAIN.tsv --skip-cols 1 \
    -k 5 --method pam --dtype float64 \
    --output results/beef_k5 --name beef --verbose

# Copy results back
rsync -a results/ "${SLURM_SUBMIT_DIR}/results/cpu_test_${SLURM_JOB_ID}/"
echo "=== Done: $(date) ==="
```

#### `jobs/gpu_test.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=dtwc-gpu-test
#SBATCH --partition=short
#SBATCH --clusters=htc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err

echo "=== DTWC++ GPU Test ==="
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    $(hostname)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
nvidia-smi 2>/dev/null || echo "  nvidia-smi not available"
echo ""

WORKDIR="${SCRATCH:-${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "$WORKDIR"
cd "$WORKDIR" || exit 1
rsync -a "${SLURM_SUBMIT_DIR}/data/" ./data/
cp "${SLURM_SUBMIT_DIR}/build-htc-gpu/bin/dtwc_cl" ./

module purge 2>/dev/null || true
module load GCC/13.2.0 CUDA/12.4.0 2>/dev/null || module load gcc cuda 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# fp32 test
echo "--- GPU fp32 ---"
./dtwc_cl --input data/Coffee/Coffee_TRAIN.tsv --skip-cols 1 \
    -k 2 --method pam --device cuda --gpu-precision f32 \
    --output results/gpu_fp32 --name coffee_fp32 --verbose

# fp64 test
echo "--- GPU fp64 ---"
./dtwc_cl --input data/Coffee/Coffee_TRAIN.tsv --skip-cols 1 \
    -k 2 --method pam --device cuda --gpu-precision f64 \
    --output results/gpu_fp64 --name coffee_fp64 --verbose

rsync -a results/ "${SLURM_SUBMIT_DIR}/results/gpu_test_${SLURM_JOB_ID}/"
echo "=== Done: $(date) ==="
```

#### `jobs/checkpoint_test.slurm`

- CPU-only, uses `--checkpoint $SCRATCH/ckpt`
- Run 1: full run
- Run 2: `--resume` from checkpoint
- `diff` labels -> PASS/FAIL
- Copy both results back to `$DATA`

#### `jobs/parquet_test.slurm`

- CPU-only, Arrow-enabled build
- Tests Parquet input (converted locally before upload)
- Tests `.dtws` round-trip
- Arrow module load with fallback

### 3f. `scripts/slurm/convert_ucr.py`

Converts UCR TSV files to Parquet and `.dtws` formats **locally** before upload:

```python
# Usage: uv run scripts/slurm/convert_ucr.py data/benchmark/UCRArchive_2018/Coffee
# Output: data/benchmark/UCRArchive_2018/Coffee/Coffee_TRAIN.parquet
#         data/benchmark/UCRArchive_2018/Coffee/Coffee_TRAIN.dtws
```

Uses pyarrow to write Parquet, and the project's `.dtws` binary format for the cache file.

### 3g. Benchmark results `.json`

Each SLURM job writes a `results_<job_id>.json` with:

```json
{
  "dataset": "Coffee_TRAIN",
  "n_series": 28,
  "series_length": 286,
  "k": 2,
  "method": "pam",
  "dtype": "float64",
  "device": "cpu",
  "gpu_name": null,
  "slurm_job_id": "12345",
  "node": "htc-c001",
  "cpus": 8,
  "omp_threads": 8,
  "wall_time_sec": 1.23,
  "queue_wait_sec": null,
  "dtw_time_sec": 0.45,
  "cluster_time_sec": 0.78,
  "ari": 0.85,
  "silhouette_avg": 0.72
}
```

Queue wait time captured via `sacct --format=Submit,Start -j $SLURM_JOB_ID` in a post-job step. These JSON files can be aggregated for the website to show performance across hardware.

### 3h. `scripts/slurm/verify_results.py`

Pure Python ARI implementation. Reports per-dataset PASS/FAIL with ARI.

### 3i. `scripts/slurm/README.md`

Comprehensive index covering:

- File index (all scripts + purpose)
- Quick-start workflow
- `.env` configuration with Oxford ARC example
- Build profiles table
- ARC quick-reference:
  - Partitions: short (≤12hr), medium (≤48hr), long (unlimited), devel (10min), interactive (24hr)
  - GPU gres: `--gres=gpu:v100:1`, `--constraint='gpu_sku:H100'`
  - Available GPUs: P100, V100, RTX, RTX8000, A100, H100 (co-invest), L40S (co-invest)
  - Storage: `$HOME` (15 GiB), `$DATA` (5 TiB shared), `$SCRATCH`/`$TMPDIR` (per-job, deleted on exit)
  - **Must build on interactive nodes**, not login (1hr CPU limit on login)
  - Link: https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html
- Common SLURM commands
- Troubleshooting (Windows `dos2unix`, modules, two-hop SSH, `#SBATCH --signal=B:SIGINT@60` for graceful timeout)

---

## Phase 4: Documentation

### `docs/content/getting-started/slurm.md` (weight: 10)

- Prerequisites
- Configuration (`.env` at project root from `env.example`)
- Quick-start workflow
- Build profiles
- Oxford ARC reference section (link to ARC user guide)
- GPU types table, partition limits
- Troubleshooting

---

## Phase 5: Cleanup

1. Delete entire `arc/` folder (15 files)
2. Update CHANGELOG.md (Unreleased section)
3. Update `.claude/LESSONS.md` if new gotchas discovered

---

## Phase 6: Adversarial Review

2 Opus + 1 Codex:

- **Security:** Grep for `engs2321`, `engs-unibatt`, `arc.ox.ac.uk`, passwords. Verify `.env` never committed. Check `WarningPolicy` used (not `AutoAddPolicy`). Verify password cleared after use.
- **Correctness:** SLURM best practices ($SCRATCH, modules, resource requests). CLI changes compile. ARI computation correct. `--resume` works.
- **Codex:** Independent code review.

---

## Implementation Order

| Step | What | Files | Depends on |
|------|------|-------|------------|
| 1 | Rename precision flags (`--dtype`, `--gpu-precision`) | `dtwc/dtwc_cl.cpp` | -- |
| 2 | Rename `--restart` to `--resume` | `dtwc/dtwc_cl.cpp` | -- |
| 3 | Add verbose diagnostics (SLURM env, OMP, CUDA, memory) | `dtwc/dtwc_cl.cpp` | -- |
| 4 | Create `env.example` | `scripts/slurm/env.example` | -- |
| 5 | Create `slurm_remote.py` | `scripts/slurm/slurm_remote.py` | Step 4 |
| 6 | Create SLURM job templates | `scripts/slurm/jobs/*.slurm` | Steps 1-3 |
| 7 | Create `verify_results.py` | `scripts/slurm/verify_results.py` | -- |
| 8 | Create `convert_ucr.py` | `scripts/slurm/convert_ucr.py` | -- |
| 9 | Create `scripts/slurm/README.md` | `scripts/slurm/README.md` | Steps 4-8 |
| 10 | Update CLI docs | `docs/content/getting-started/cli.md` | Steps 1-2 |
| 11 | Create SLURM docs page | `docs/content/getting-started/slurm.md` | Steps 4-9 |
| 12 | Delete `arc/` folder | `arc/*` | Steps 5-6 |
| 13 | Update CHANGELOG | `CHANGELOG.md` | Steps 1-12 |
| 14 | Adversarial review | -- | Steps 1-13 |

Steps 1-4, 7, 8 are independent and can be parallelized.

---

## Verification Plan

1. **Local:** Build, run `dtwc_cl -i data/dummy -k 3 -v` -- verify diagnostics, `--dtype`, `--gpu-precision`
2. **Local:** Test `--resume` flag parses correctly
3. **Local:** Run `convert_ucr.py` on Coffee -- verify Parquet + .dtws output
4. **Remote:** `slurm_remote.py test` -- SSH connection
5. **Remote:** `upload` + `build --profile htc-cpu` (batch job)
6. **Remote:** `submit-cpu` -- check all 8 cores used in `slurm.out`
7. **Remote:** `build --profile htc-gpu` + `submit-gpu` -- check nvidia-smi in output
8. **Remote:** `submit-checkpoint` -- labels match between runs
9. **Remote:** `submit-parquet` -- Parquet and .dtws input works
10. **Remote:** `download` + `verify_results.py` -- ARI > thresholds
11. **Grep:** No `engs2321`, `engs-unibatt`, hardcoded passwords in any committed file

---

## Critical Files

| File | Role |
|------|------|
| `dtwc/dtwc_cl.cpp` | CLI: `--dtype`, `--gpu-precision`, `--resume`, diagnostics |
| `scripts/slurm/slurm_remote.py` | Generalized SSH/SFTP remote helper (from arc_remote.py) |
| `scripts/slurm/env.example` | Config template with Oxford ARC example |
| `scripts/slurm/jobs/cpu_test.slurm` | CPU test with $SCRATCH + diagnostics |
| `scripts/slurm/jobs/gpu_test.slurm` | GPU test with nvidia-smi |
| `scripts/slurm/jobs/checkpoint_test.slurm` | Resume verification |
| `scripts/slurm/jobs/parquet_test.slurm` | Parquet + .dtws test |
| `scripts/slurm/verify_results.py` | ARI ground truth comparison |
| `scripts/slurm/convert_ucr.py` | TSV -> Parquet + .dtws converter |
| `scripts/slurm/README.md` | Index + ARC quick-reference |
| `docs/content/getting-started/slurm.md` | Hugo docs page |
| `docs/content/getting-started/cli.md` | CLI docs updates |

## Adversarial Review Findings (2 agents, 2026-04-09)

### CRITICAL (fixed in plan)

| Issue | Fix |
|-------|-----|
| Duplicate `--precision` crashes CLI11 (`OptionAlreadyAdded`) | Rename to `--dtype` + `--gpu-precision` |
| `.env` could be uploaded via SFTP glob | Allowlist upload dirs, exclude `.env` |

### HIGH (fixed in plan)

| Issue | Fix |
|-------|-----|
| YAML only loads GPU precision, not data precision | Add `set_if_unset("dtype", dtype_str)` |
| `$SCRATCH` is ARC-specific, not standard SLURM | Fallback chain: `${SCRATCH:-${TMPDIR:-/tmp/...}}` |
| TOML config `precision` key collision | Split into `dtype` and `gpu-precision` keys |
| SSH `WarningPolicy` still accepts MITM | Use `RejectPolicy` + `load_system_host_keys()` |

### MEDIUM (fixed in plan)

| Issue | Fix |
|-------|-----|
| GPU default "auto"->"float32" would silently downgrade | Keep `auto` default |
| `--restart` removal breaks scripts | Keep as hidden deprecated alias |
| `--gres=gpu:1` may get wrong GPU type | Document risk, recommend specific type |
| Hardcoded module names are ARC-specific | Fallback chain with generic names |
| ARI zero-denominator edge case | Return 0.0 when denominator is 0 |
| Race: submit before build completes | Preflight check for binary existence |

### LOW (noted)

| Issue | Fix |
|-------|-----|
| Unquoted variable expansions in bash | Quote all `${}` expansions |
| Missing `.env` has no graceful error | Startup check pointing to `env.example` |
| UCR first column is integer (not float) | `int(float(x))` still works |
| Password remains in CPython heap after `None` | Document limitation; acceptable for research tool |

## Resolved Questions

- **Build method:** Batch jobs (non-interactive). Print `interactive` instructions for debugging.
- **Parquet test data:** Convert TSV -> Parquet + .dtws locally via `convert_ucr.py`, upload both.
- **Password security:** SSH key recommended, `RejectPolicy` default, password as fallback, clear after use.
- **SLURM paths:** `SLURM_HOME_FOLDER` + `SLURM_DATA_FOLDER` instead of `SLURM_PROJECT`.
- **GPU precision default:** Keep `auto` (don't change to float32).
- **`--restart` deprecation:** Keep as hidden alias for one release cycle.
- **`$SCRATCH` portability:** Fallback chain with `$TMPDIR` and `/tmp`.
- **SFTP upload safety:** Allowlist of directories, never upload `.env`.

## ASSUMED

- H100 gres: `--gres=gpu:1 --constraint='gpu_sku:H100'` (follows documented pattern). Will verify on first test.
- UCR first column is integer class label. `verify_results.py` handles via `int(float(label))`.
