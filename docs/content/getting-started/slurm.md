---
title: SLURM HPC Clusters
weight: 10
---

# Running on SLURM HPC Clusters

DTWC++ includes scripts for building and testing on SLURM-managed HPC clusters. These scripts are general-purpose and work on any SLURM cluster, with [Oxford ARC](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html) as the reference deployment.

## Prerequisites

- SSH access to a SLURM cluster (`ssh` and `rsync` — both ship with Git Bash on Windows)
- Cluster modules: GCC >= 13, CMake >= 3.26, and optionally CUDA >= 12.0, Arrow >= 15.0

## Configuration

1. Copy the template to your project root:

```bash
cp scripts/slurm/env.example .env
```

2. Edit `.env` with your cluster details (user, host, paths, partition).

3. The `.env` file is in `.gitignore` and will **never** be committed. It contains credentials.

### Required variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SLURM_USER` | Your cluster username | `jdoe` |
| `SLURM_HOST` | Login node hostname | `htc-login.arc.ox.ac.uk` |
| `SLURM_DATA_FOLDER` | Shared data directory | `/data/project/user` |
| `SLURM_REMOTE_BASE` | Working directory for dtw-cpp | `/data/project/user/dtw-cpp` |

### Optional variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SLURM_GATEWAY` | Two-hop SSH gateway (leave empty for direct) | (empty) |
| `SLURM_SSH_KEY` | Path to SSH private key (recommended) | `~/.ssh/id_ed25519` |
| `SLURM_PASSWORD` | SSH password (fallback) | (empty) |
| `SLURM_PARTITION` | Default SLURM partition | `short` |
| `SLURM_CLUSTER` | Target cluster name | (empty) |
| `SLURM_GPU_GRES` | GPU resource request | `gpu:1` |
| `SLURM_EMAIL` | Job notification email | (empty) |

## Quick Start

```bash
# Test SSH connection
bash scripts/slurm/slurm_remote.sh test

# Upload source code and test datasets
bash scripts/slurm/slurm_remote.sh upload

# Build on an interactive node (batch job)
bash scripts/slurm/slurm_remote.sh build --profile htc-cpu

# Submit CPU test job
bash scripts/slurm/slurm_remote.sh submit-cpu

# Check job status
bash scripts/slurm/slurm_remote.sh status

# Download results and logs
bash scripts/slurm/slurm_remote.sh download

# Verify results against known answers
uv run benchmarks/verify_results.py \
    --true data/benchmark/UCRArchive_2018/Coffee/Coffee_TRAIN.tsv \
    --predicted results/coffee_k2/coffee_labels.csv -k 2
```

## Build Profiles

The `scripts/slurm/build-arc.sh` script supports multiple hardware targets:

| Profile | CPU Arch | GPU | Use Case |
|---------|----------|-----|----------|
| `arc` | AVX-512 | No | ARC cluster (Cascade Lake + Turin) |
| `htc-cpu` | AVX2 | No | HTC CPU-only, portable across all nodes |
| `htc-gpu` | AVX2 | All CUDA archs | HTC with any GPU (P100 through H100) |
| `htc-v4` | AVX-512 | No | HTC nodes with AVX-512 (excludes Broadwell/Rome) |
| `h100` | AVX-512 | sm_90 only | H100 nodes, fastest compile |
| `grace` | AArch64 native | No | Grace Hopper (ARM), CPU only |

## Test Datasets

The scripts upload small UCR datasets for quick verification:

| Dataset | Samples | Length | Classes | Expected ARI |
|---------|---------|--------|---------|--------------|
| Coffee | 28 | 286 | 2 | > 0.7 |
| Beef | 30 | 470 | 5 | > 0.3 |

## Data Format Conversion

Convert UCR TSV files to Parquet for testing the Parquet I/O path:

```bash
uv run benchmarks/convert_ucr.py data/benchmark/UCRArchive_2018/Coffee
```

## Oxford ARC Reference

DTWC++ was developed and tested on Oxford's [Advanced Research Computing (ARC)](https://www.arc.ox.ac.uk/) clusters. Oxford users can use these ARC-specific settings.

### Partitions

| Partition | Default Time | Max Time | Notes |
|-----------|-------------|----------|-------|
| `short` | 1 hour | 12 hours | Default, highest priority |
| `medium` | 12 hours | 48 hours | |
| `long` | 24 hours | Unlimited | Lowest priority |
| `devel` | — | 10 minutes | Batch testing only |
| `interactive` | — | 24 hours | Software builds, pre/post-processing |

### GPU Resources (HTC cluster only)

GPUs are requested via the `--gres` SLURM directive:

```bash
#SBATCH --gres=gpu:1              # Any available GPU
#SBATCH --gres=gpu:v100:1         # Specific type
#SBATCH --gres=gpu:a100:1         # A100
```

Or via constraints:

```bash
#SBATCH --gres=gpu:1 --constraint='gpu_sku:V100'
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Ampere'
```

Available GPUs: P100, V100, RTX (Titan RTX), RTX8000, A100, H100 (co-investment), L40S (co-investment).

Co-investment GPU nodes are limited to the **short** partition (12-hour maximum).

### Storage

| Area | Path | Quota | Persistent |
|------|------|-------|------------|
| `$HOME` | `/home/username` | 15 GiB | Yes |
| `$DATA` | `/data/project/username` | 5 TiB (shared) | Yes |
| `$SCRATCH` | Per-job | Unlimited | No (deleted on job exit) |
| `$TMPDIR` | Per-job, node-local | Varies | No (deleted on job exit) |

SLURM jobs should use `$SCRATCH` (or `$TMPDIR`) for I/O and copy results back to `$DATA` before exit.

### Important Notes

- **Do not run computation on login nodes** (1-hour CPU limit). Use interactive nodes for builds.
- **Windows line endings**: If editing scripts on Windows, run `dos2unix script.sh` before submitting.
- **Fair-share scheduling**: Priority decreases with recent usage (14-day half-life).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `module load GCC/13.2.0` fails | Try `module load gcc` or check available modules with `module avail gcc` |
| Build fails on login node | Use `srun -p interactive --pty /bin/bash` first |
| SSH connection refused | Check VPN connection; ARC login nodes require university network |
| `$SCRATCH` not set | Your cluster may not set this; the job scripts fall back to `$TMPDIR` or `/tmp` |
| GPU not detected in job | Verify `--gres=gpu:1` is in your SBATCH directives |
| `dos2unix: command not found` | Use `sed -i 's/\r$//' script.sh` as alternative |
