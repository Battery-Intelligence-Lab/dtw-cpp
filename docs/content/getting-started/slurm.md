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
| `SLURM_USER` | ASCII cluster account/SSH alias token | `jdoe` |
| `SLURM_HOST` | ASCII login hostname or SSH-config alias | `htc-login.arc.ox.ac.uk` |
| `SLURM_REMOTE_BASE` | Absolute POSIX path without whitespace, `:`, or dot components | `/data/project/user/dtw-cpp` |

### Optional variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SLURM_PARTITION` | Single Slurm-name token | `short` |
| `SLURM_CLUSTER` | Single target-cluster token | (empty) |
| `SLURM_EMAIL` | Conventional ASCII notification address | (empty) |

The wrapper validates every consumed value before its first SSH or transfer
call. Configure proxy jumps, identity files, and authentication in your SSH
client; `SLURM_GATEWAY`, `SLURM_SSH_KEY`, and `SLURM_PASSWORD` are advisory
template entries and are not read by the wrapper.

## Quick Start

```bash
# Test SSH connection
bash scripts/slurm/slurm_remote.sh test

# Upload source code and test datasets
bash scripts/slurm/slurm_remote.sh upload

# Build on an interactive node (batch job)
bash scripts/slurm/slurm_remote.sh build --profile htc-cpu

# Submit the CPU smoke test (modes: cpu, gpu, checkpoint, parquet)
bash scripts/slurm/slurm_remote.sh submit-smoke cpu

# Check job status
bash scripts/slurm/slurm_remote.sh status

# Download results and logs
bash scripts/slurm/slurm_remote.sh download

# Verify results against known answers
uv run benchmarks/verify_results.py \
    --true data/benchmark/UCRArchive_2018/Coffee/Coffee_TRAIN.tsv \
    --predicted results/slurm/smoke_cpu_JOBID/coffee_k2/coffee_labels.csv -k 2
```

## From Python: `device="hpc"`

`DTWClustering(device="hpc")` and `dtwcpp.cluster(..., device="hpc")` send the whole
clustering job through the same wrapper, which ships inside the `dtwcpp` package, so an
installed wheel needs no source checkout. Put `.env` in the directory you run Python from,
or set `DTWC_REPO_ROOT` to the directory that holds it; the labels are downloaded to
`results/slurm/cluster_<job id>/` there. `DTWC_REPO_ROOT` steers only this Python route:
`bash scripts/slurm/slurm_remote.sh` always reads the `.env` of, and uploads, its own
checkout.

The run travels as one file, `job.toml`: the input, `n-clusters`, `name`, `device`, and every
key you passed (`DTWClustering` passes all its parameters), in the format `dtwc_cl --config`
reads; a key you did not pass is not written, so the cluster's `dtwc_cl` applies its own default. Series in memory travel beside
it as `input.tsv`; a path names a file on the cluster, read there and never on your machine,
so give its absolute path. `slurm_remote.sh submit-job <dir>` uploads the directory and the
job runs `dtwc_cl --config job.toml --output results/cluster_<job id>`. The cluster needs a
`dtwc_cl` build, made once from a source checkout with `upload` and `build` as above: CPU
runs use `build-htc-cpu`, `device="hpc:gpu"` runs `build-htc-gpu`. A key that build does not
know, say one a newer `dtwcpp` writes, fails the job and names the key in
`logs/cluster_<job id>.err` under `SLURM_REMOTE_BASE/src`.

`device="hpc:gpu"` asks SLURM for any GPU of compute capability 8.0 or newer.
`gpu_device=` names one, and the run uses the build made for it:

```python
res = dtwcpp.cluster("/data/project/me/series.tsv", k=8, device="hpc:gpu", gpu_device="a100")
```

```bash
bash scripts/slurm/slurm_remote.sh build htc-gpu --gpu-device a100   # once: build-a100
```

The types are those of `python/dtwcpp/_slurm/gpu_devices.txt` (below); a type below compute
capability 8.0, or one the table does not name, raises `DeviceError` before anything is sent,
and `gpu_device` with a device other than `"hpc:gpu"` raises `InvalidInput`.

## Build Profiles

The `scripts/slurm/build-arc.sh` script supports multiple hardware targets:

| Profile | CPU Arch | GPU | Use Case |
|---------|----------|-----|----------|
| `arc` | AVX-512 | No | ARC cluster (Cascade Lake + Turin) |
| `htc-cpu` | AVX2 | No | HTC CPU-only, portable across all nodes |
| `htc-gpu` | AVX2 | sm_80, sm_86, sm_89 | HTC A100, RTX A6000 and L40S nodes (for H100 use `h100`) |
| `htc-v4` | AVX-512 | No | HTC nodes with AVX-512 (excludes Broadwell/Rome) |
| `h100` | AVX-512 | sm_90 only | H100 nodes, fastest compile |
| `grace` | AArch64 native | No | Grace Hopper (ARM), CPU only |

### Building on the target node

Run `htc-gpu` or `h100` on a GPU node, where `nvidia-smi` lists a GPU of compute capability 8.0 or newer,
and the script builds for that node: `CMAKE_CUDA_ARCHITECTURES=native` and `-march=native`, the most
specialised binary. Such a build runs only on that node type, so build again for each GPU type you use.
Anywhere else (a node without a GPU, or one whose GPU is below 8.0) the profile's portable lists above
apply. `slurm_remote.sh build htc-gpu --gpu-device <type>` asks for that GPU as its jobs do, on the
`.env` partition, so the build runs on such a node and goes into `build-<type>`, the build a
`gpu_device="<type>"` run uses: that GPU's CUDA architecture, with the profile's portable CPU code,
since one GPU type sits on nodes of different CPUs (ARC's A100 nodes are Cascade Lake and AMD Rome, its
H100 nodes Ice Lake and Sapphire Rapids) and a job may land on any of them. Without that flag, `build`
submits to the `interactive` partition without a GPU request and takes the portable route.

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

GPUs are requested with an `#SBATCH --gres` directive. ARC's [job scheduling guide](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html#gpu-resources) documents these forms:

```bash
#SBATCH --gres=gpu:1                                 # Any GPU
#SBATCH --gres=gpu:a100:1                            # A100
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Ampere'   # By generation
```

It documents the types P100, V100, RTX (Titan RTX), RTX8000 and A100 and the constraints `gpu_sku:`, `gpu_gen:`,
`gpu_cc:`, `gpu_mem:` and `nvlink:`, and lists no type, SKU or generation for the RTX A6000, H100 and L40S nodes.
DTWC++ keeps the requests in one table, `python/dtwcpp/_slurm/gpu_devices.txt`, which `gpu_device=` and
`slurm_remote.sh` (`submit-job`, `build --gpu-device`, `submit-benchmark-gpu`, `submit-smoke gpu`) read:

| `gpu_device` | Compute capability | sbatch request |
|---|---|---|
| (none) | 8.0 or newer | `sbatch --gres=gpu:1 --constraint='gpu_cc:8.0\|gpu_cc:8.6\|gpu_cc:8.9\|gpu_cc:9.0'` |
| `a100` | 8.0 | `sbatch --gres=gpu:a100:1` |
| `a6000` | 8.6 | `sbatch --gres=gpu:1 --constraint=gpu_cc:8.6` |
| `l40s` | 8.9 | `sbatch --gres=gpu:1 --constraint=gpu_cc:8.9` |
| `h100` | 9.0 | `sbatch --gres=gpu:1 --constraint=gpu_cc:9.0` |

The compute capabilities are the [systems page](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html#gpu-resources)'s.
That ARC tags those nodes with exactly these `gpu_cc:` values is inferred from its guide: check with
`scontrol show node <node>` (`AvailableFeatures`) and correct the table if they differ.

GPUs on the htc cluster, from the systems page: P100, V100, RTX8000, Titan RTX, A100, RTX A6000, H100 and L40S, plus one
MI210 node and one GH200 (Grace Hopper) node. Co-investment GPU nodes are limited to the **short** partition (12-hour maximum).

#### GPUs DTWC++ can use

DTWC++ needs CUDA compute capability 8.0 (Ampere) or newer. On ARC that is the A100 (8.0), RTX A6000 (8.6),
L40S (8.9) and H100 (9.0). The P100 (6.0), V100 (7.0), RTX8000 and Titan RTX (7.5) are refused with a
`DeviceError` when the GPU is selected; nothing falls back to the CPU. `gpu_device=` refuses them before
submitting, and the requests of the table above never land on them; a job script of your own that asks for
`#SBATCH --gres=gpu:1` alone may be given one. The MI210 is not a CUDA device, and the GH200 node is AArch64 (the
`grace` profile builds without CUDA).

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
| GPU not detected in job | Verify the job script has `#SBATCH --gres=gpu:1` |
| `DeviceError` naming a compute capability below 8.0 | The job was given a P100, V100, RTX8000 or Titan RTX; request an A100, RTX A6000, L40S or H100 |
| `dos2unix: command not found` | Use `sed -i 's/\r$//' script.sh` as alternative |
