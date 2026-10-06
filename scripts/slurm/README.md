# SLURM HPC Scripts

General-purpose scripts for building and testing DTWC++ on SLURM-managed HPC clusters. Oxford [ARC](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html) is the reference deployment.

## Files

| File | Purpose |
|------|---------|
| `env.example` | Configuration template -- copy to `.env` at project root |
| `slurm_remote.sh` | SSH/rsync remote helper: upload, build, submit, download. Forwards to `python/dtwcpp/_slurm/slurm_remote.sh` with this checkout as the project |
| `../../python/dtwcpp/_slurm/` | The wrapper, `cluster_generic.slurm` (runs a `job.toml`) and `gpu_devices.txt` (the GPU requests), shipped in the wheel so `device="hpc"` needs no checkout |
| `build-arc.sh` | Multi-profile CMake build script (6 hardware targets) |
| `../../benchmarks/verify_results.py` | Compare clustering output against UCR ground truth (in benchmarks/) |
| `../../benchmarks/convert_ucr.py` | Convert UCR TSV files to Parquet format (in benchmarks/) |
| `jobs/smoke.slurm` | Smoke test, `MODE=cpu` (Coffee k=2, Beef k=5), `gpu` (Coffee at fp32 and fp64), `checkpoint` (save, then resume) or `parquet` (Parquet gives the TSV's labels; needs Arrow) |
| `jobs/ucr_benchmark_{cpu,gpu}.slurm` | Full UCR benchmark |

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

# 5. Submit a smoke test (cpu, gpu, checkpoint or parquet)
bash scripts/slurm/slurm_remote.sh submit-smoke cpu

# 6. Monitor
bash scripts/slurm/slurm_remote.sh status

# 7. Download results
bash scripts/slurm/slurm_remote.sh download

# 8. Verify
uv run benchmarks/verify_results.py \
    --true data/benchmark/UCRArchive_2018/Coffee/Coffee_TRAIN.tsv \
    --predicted results/slurm/smoke_cpu_JOBID/coffee_k2/coffee_labels.csv -k 2
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
Elsewhere the profile's portable lists above apply.

| Command | Partition | GPU request | Builds into | Run by |
|---|---|---|---|---|
| `build htc-gpu` | `interactive` | none | `build-htc-gpu`, portable | `device="hpc:gpu"` |
| `build htc-gpu --gpu-device <type>` | `.env`'s (`short`) | that type's (`gpu_devices.txt`) | `build-<type>`, native to that node | `gpu_device="<type>"` |
| `build htc-cpu` | `interactive` | none | `build-htc-cpu` | `device="hpc"` |

## .env Configuration

See `env.example` for all variables and their descriptions. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `SLURM_USER` | Yes | ASCII cluster account/SSH alias token |
| `SLURM_HOST` | Yes | ASCII login hostname or SSH-config alias |
| `SLURM_REMOTE_BASE` | Yes | Absolute POSIX working path; no whitespace, `:`, or dot components |
| `SLURM_PARTITION` | No | Single Slurm-name token (default `short`) |
| `SLURM_CLUSTER` | No | Single Slurm-name token |
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
`gpu_cc:`, `gpu_mem:`, `nvlink:`; it names no type, SKU or generation for the RTX A6000, H100 and L40S nodes.
`python/dtwcpp/_slurm/gpu_devices.txt` is DTWC++'s one table of requests, read by Python's `gpu_device=` and by
`submit-job`, `build --gpu-device`, `submit-benchmark-gpu` and `submit-smoke gpu`: the A100 by its gres type, the
others by `gpu_cc:` and the compute capability the
[systems page](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html#gpu-resources) gives, and no type
by every capability at or above 8.0. Check the `gpu_cc:` values against `AvailableFeatures` in
`scontrol show node <node>`.

GPUs on htc (systems page): P100, V100, RTX8000, Titan RTX, A100, RTX A6000, H100, L40S, one MI210 node and one
GH200 node. DTWC++ needs CUDA compute capability 8.0 (Ampere) or newer: A100, RTX A6000, L40S and H100 qualify.
P100, V100, RTX8000 and Titan RTX are refused with a `DeviceError` (no CPU fallback); the table's requests
never ask for them, but a bare `--gres=gpu:1` may be given one. The MI210 is not CUDA; the GH200 node is AArch64
(`grace` profile, no CUDA).

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
