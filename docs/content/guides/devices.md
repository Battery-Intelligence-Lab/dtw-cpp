---
title: "Devices, HPC, and parallelism"
weight: 10
---

# Devices, HPC, and parallelism

Select a device once, or override it for one clustering call:

```python
import dtwcpp as dtwc

dtwc.device("cpu")                 # cpu | gpu | gpu:N | cuda | cuda:N | hpc
result = dtwc.cluster(data, 3)
result = dtwc.cluster(data, 3, device="gpu:1")
```

`gpu` resolves to CUDA where compiled and to Metal on macOS. A request that
cannot be honoured raises a device error; DTWC++ does not silently run the same
request on CPU. Matrix-free schedules (`onebatch`, `clara`, and `tadpole`) are
currently CPU-only and reject a GPU request.

GPU distance kernels cannot read mmap-backed series. `StoragePolicy::Auto`
spills a dataset above half the free physical RAM into the mapped store, so a
large GPU run through the Tier-2 `Problem` API must select
`StoragePolicy::Heap` (or raise `ram_limit`) before `set_data`; otherwise
`fill_distance_matrix` raises a device error naming `mmap-backed series data`
rather than falling back to the CPU. The Tier-1 `cluster(...)` entry point
already pins `Heap` when the selected device is a GPU, and the native CLI never
routes series storage at all. See
[Data formats and conversion](../data-formats/) for the free-RAM
quantities each platform reports.

## HPC setup (beta)

Copy `scripts/slurm/env.example` to `.env` at the repository root:

```dotenv
SLURM_HOST=arc-login.arc.ox.ac.uk
SLURM_USER=abcd1234
SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs
```

Passwordless `ssh $SLURM_USER@$SLURM_HOST` must succeed. Python owns the beta
end-to-end submission route; C++ validates `hpc` selection but directs users to
the Python or shell transport until a real Oxford ARC validation closes the beta.

The following messages are copied verbatim from `dtwc/env.cpp`.

Missing `.env`:

```text
[dtwc] device='hpc' requires a .env file at the repository root, but none was found.
Copy scripts/slurm/env.example to .env and set SLURM_HOST, SLURM_USER, and SLURM_REMOTE_BASE.
Example .env:
  SLURM_HOST=arc-login.arc.ox.ac.uk
  SLURM_USER=abcd1234
  SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs
```

Missing key (the live key name replaces `<key>`):

```text
[dtwc] device='hpc': the .env file is missing required key '<key>'.
Set it in .env at the repository root. Example .env:
  SLURM_HOST=arc-login.arc.ox.ac.uk
  SLURM_USER=abcd1234
  SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs
```

Authentication failure (the configured values replace `<host>` and `<user>`):

```text
[dtwc] device='hpc': could not authenticate to SLURM host '<host>' as user '<user>'.
Check that your SSH key is authorized on that host (ssh <user>@<host> must succeed without a password prompt) and that SLURM_HOST and SLURM_USER in .env are correct.
```

## Prove what this process can use

These are execution probes, not compile-flag reads. The parallel probe enters a
real OpenMP region; the GPU probe executes a tiny kernel and compares it with a
CPU oracle.

```python
parallel = dtwc.test.parallelisation()
gpu = dtwc.test.gpu()
assert parallel["pass"]
print(parallel)  # available, max_threads, threads_engaged, pass, reason
print(gpu)       # available, backend, device_name, validated, pass, reason
```

```cpp
const auto parallel = dtwc::test::parallelisation();
const auto gpu = dtwc::test::gpu();
```

```matlab
parallel = dtwc.test.parallelisation();
gpu = dtwc.test.gpu();
assert(parallel.pass);
```

## Why OpenMP configuration can fail

Parallel execution is part of the shipped performance contract. CMake therefore
stops when OpenMP is unavailable. For constrained targets, explicitly acknowledge
the tradeoff with:

```sh
cmake -S . -B build -DDTWC_ALLOW_SEQUENTIAL=ON
```

That build emits a process-once `SINGLE-THREADED` warning. An OpenMP build whose
runtime is capped to one thread (for example `OMP_NUM_THREADS=1`) emits a distinct
warning and recommends raising or unsetting the cap. These warnings mean the
answer remains correct, but an all-pairs distance matrix may be extremely slow.
