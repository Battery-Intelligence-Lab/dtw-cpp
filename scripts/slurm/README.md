# SLURM HPC Scripts

General-purpose scripts for building and testing DTWC++ on SLURM-managed HPC clusters. Oxford [ARC](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html) is the reference deployment.

## Files

| File | Purpose |
|------|---------|
| `env.example` | Configuration template -- copy to `.env` at project root |
| `slurm_remote.py` | SSH/SFTP remote helper: upload, build, submit, download |
| `build-arc.sh` | Multi-profile CMake build script (6 hardware targets) |
| `../../benchmarks/verify_results.py` | Compare clustering output against UCR ground truth (in benchmarks/) |
| `../../benchmarks/convert_ucr.py` | Convert UCR TSV files to Parquet format (in benchmarks/) |
| `jobs/cpu_test.slurm` | CPU-only test: Coffee k=2, Beef k=5 |
| `jobs/gpu_test.slurm` | GPU test: Coffee with fp32 and fp64 precision |
| `jobs/checkpoint_test.slurm` | Checkpoint/resume verification |
| `jobs/parquet_test.slurm` | Parquet and .dtws format I/O test |

## Quick Start

```bash
# 1. Configure
cp scripts/slurm/env.example .env
# Edit .env with your cluster details

# 2. Test connection
uv run scripts/slurm/slurm_remote.py test

# 3. Upload source + test data
uv run scripts/slurm/slurm_remote.py upload

# 4. Build on cluster
uv run scripts/slurm/slurm_remote.py build --profile htc-cpu

# 5. Submit tests
uv run scripts/slurm/slurm_remote.py submit-cpu

# 6. Monitor
uv run scripts/slurm/slurm_remote.py status

# 7. Download results
uv run scripts/slurm/slurm_remote.py download

# 8. Verify
uv run benchmarks/verify_results.py \
    --true data/benchmark/UCRArchive_2018/Coffee/Coffee_TRAIN.tsv \
    --predicted results/cpu_test_JOBID/coffee_k2/coffee_labels.csv -k 2
```

## Build Profiles

| Profile | CPU | GPU | Target Hardware |
|---------|-----|-----|-----------------|
| `arc` | AVX-512 | -- | ARC cluster (Cascade Lake + Turin) |
| `htc-cpu` | AVX2 | -- | HTC all CPU nodes (portable) |
| `htc-gpu` | AVX2 | All archs | HTC any GPU (P100--H100) |
| `htc-v4` | AVX-512 | -- | HTC AVX-512 nodes only |
| `h100` | AVX-512 | sm_90 | H100 nodes (fastest compile) |
| `grace` | AArch64 | -- | Grace Hopper (ARM, CPU only) |

## .env Configuration

See `env.example` for all variables and their descriptions. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `SLURM_USER` | Yes | Cluster username |
| `SLURM_HOST` | Yes | Login node hostname |
| `SLURM_DATA_FOLDER` | Yes | Shared data directory |
| `SLURM_REMOTE_BASE` | Yes | Working directory for dtw-cpp |
| `SLURM_SSH_KEY` | Recommended | Path to SSH private key |
| `SLURM_PASSWORD` | Fallback | SSH password (less secure) |

## Oxford ARC Quick Reference

### Partitions

| Name | Max Time | Priority |
|------|----------|----------|
| short | 12 hours | Highest |
| medium | 48 hours | Medium |
| long | Unlimited | Lowest |
| devel | 10 min | -- (batch testing) |
| interactive | 24 hours | -- (builds only) |

### GPU Access (HTC cluster only)

```bash
#SBATCH --gres=gpu:1                     # Any GPU
#SBATCH --gres=gpu:v100:1               # Specific type
#SBATCH --gres=gpu:1 --constraint='gpu_sku:H100'  # Via constraint
```

Available: P100, V100, RTX, RTX8000, A100, H100 (co-invest), L40S (co-invest).

### Storage

- `$HOME`: 15 GiB persistent
- `$DATA`: 5 TiB shared persistent
- `$SCRATCH` / `$TMPDIR`: Per-job, auto-deleted

### Rules

- **Do not compute on login nodes** (1-hour CPU limit)
- Build software on **interactive** nodes: `srun -p interactive --pty /bin/bash`
- Co-investment GPU nodes limited to **short** partition (12h max)
- Windows scripts: run `dos2unix` before submitting

### Common Commands

```bash
sbatch script.slurm          # Submit job
squeue -u $USER              # Check jobs
scancel JOB_ID               # Cancel job
sacct -j JOB_ID --format=JobID,Elapsed,MaxRSS  # Job stats
sinfo -p short               # Partition info
module avail                 # Available modules
```
