# SLURM HPC Scripts

General-purpose scripts for building and testing DTWC++ on SLURM-managed HPC clusters. Oxford [ARC](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html) is the reference deployment.

## Files

| File | Purpose |
|------|---------|
| `env.example` | Configuration template -- copy to `.env` at project root |
| `slurm_remote.sh` | SSH/rsync remote helper: upload, build, submit, download. Forwards to `python/dtwcpp/_slurm/slurm_remote.sh` with this checkout as the project |
| `../../python/dtwcpp/_slurm/` | The wrapper and `cluster_generic.slurm`, shipped in the wheel so `device="hpc"` needs no checkout |
| `build-arc.sh` | Multi-profile CMake build script (6 hardware targets) |
| `../../benchmarks/verify_results.py` | Compare clustering output against UCR ground truth (in benchmarks/) |
| `../../benchmarks/convert_ucr.py` | Convert UCR TSV files to Parquet format (in benchmarks/) |
| `jobs/cpu_test.slurm` | CPU-only test: Coffee k=2, Beef k=5 |
| `jobs/gpu_test.slurm` | GPU test: Coffee with fp32 and fp64 precision |
| `jobs/checkpoint_test.slurm` | Distance-checkpoint save verification |
| `jobs/parquet_test.slurm` | Parquet format I/O test |

## Quick Start

```bash
# 1. Configure
cp scripts/slurm/env.example .env
# Edit .env with your cluster details

# 2. Test connection
bash scripts/slurm/slurm_remote.sh test

# 3. Upload source + test data
bash scripts/slurm/slurm_remote.sh upload

# 4. Build on cluster
bash scripts/slurm/slurm_remote.sh build --profile htc-cpu

# 5. Submit tests
bash scripts/slurm/slurm_remote.sh submit-cpu

# 6. Monitor
bash scripts/slurm/slurm_remote.sh status

# 7. Download results
bash scripts/slurm/slurm_remote.sh download

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
| `htc-gpu` | AVX2 | sm_80;86;89 | HTC A100, RTX A6000, L40S (H100: use `h100`) |
| `htc-v4` | AVX-512 | -- | HTC AVX-512 nodes only |
| `h100` | AVX-512 | sm_90 | H100 nodes (fastest compile) |
| `grace` | AArch64 | -- | Grace Hopper (ARM, CPU only) |

`htc-gpu` and `h100` run on a GPU node (`nvidia-smi` lists a GPU of compute capability 8.0 or newer) build for
that node: `CMAKE_CUDA_ARCHITECTURES=native` and `-march=native`. That binary runs only on that node type.
Elsewhere the profile's portable lists above apply. `slurm_remote.sh build` submits to the `interactive`
partition without a GPU request, so it always builds the portable binary.

## .env Configuration

See `env.example` for all variables and their descriptions. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `SLURM_USER` | Yes | ASCII cluster account/SSH alias token |
| `SLURM_HOST` | Yes | ASCII login hostname or SSH-config alias |
| `SLURM_REMOTE_BASE` | Yes | Absolute POSIX working path; no whitespace, `:`, or dot components |
| `SLURM_PARTITION` | No | Single Slurm-name token (default `short`) |
| `SLURM_CLUSTER` | No | Single Slurm-name token |
| `SLURM_GPU_GRES` | No | `gpu:<count>` or `gpu:<type>:<count>` |
| `SLURM_EMAIL` | No | Conventional ASCII notification address |

All consumed values are validated before any SSH or transfer command. Configure
SSH keys, passwords, and proxy jumps in your SSH client; the similarly named
advisory `.env` entries are not consumed by this wrapper.

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
#SBATCH --gres=gpu:1                                 # Any GPU
#SBATCH --gres=gpu:a100:1                            # A100
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Ampere'   # By generation
```

ARC's [job scheduling guide](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html#gpu-resources)
documents type names for P100, V100, RTX (Titan RTX), RTX8000 and A100, and the constraints `gpu_sku:`, `gpu_gen:`,
`gpu_cc:`, `gpu_mem:`, `nvlink:`. It names nothing for the RTX A6000, H100 and L40S nodes, and the
[systems page](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html#gpu-resources) that lists the hardware
does not say how to request a node type. `slurm_remote.sh submit-benchmark-gpu` passes `gpu:l40s:1` and `gpu:h100:1`;
ask ARC support for the rest, or read `Gres` and `AvailableFeatures` from `scontrol show node <node>`.

GPUs on htc (systems page): P100, V100, RTX8000, Titan RTX, A100, RTX A6000, H100, L40S, one MI210 node and one
GH200 node. DTWC++ needs CUDA compute capability 8.0 (Ampere) or newer: A100, RTX A6000, L40S and H100 qualify.
P100, V100, RTX8000 and Titan RTX are refused with a `DeviceError` (no CPU fallback), and a request for any GPU
(`gpu:1`) may be given one of them. The MI210 is not CUDA; the GH200 node is AArch64 (`grace` profile, no CUDA).

Co-investment GPU nodes are limited to the **short** partition (12 h).

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
