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

## A Problem's device

A Tier-2 `Problem` takes the same names, set once on the object. It does not
follow the process-wide `device(...)`: a `Problem` computes on the CPU until
told otherwise.

```python
prob = dtwc.Problem("run", device="gpu")   # or prob.set_device("gpu:1")
```

```cpp
dtwc::Problem prob("run");
prob.set_device(dtwc::Device::GPU);        // index: prob.set_device(dtwc::Device::GPU, 1)
```

```matlab
prob = dtwc.Problem('run', 'Device', 'gpu');   % or prob.set_device('gpu')
```

`cpu` keeps a CPU `distance_strategy` you chose (`BruteForce`) and
moves a GPU one back to `Auto`; `gpu` selects this build's GPU backend (CUDA,
else Metal) and records the index. `gpu` on a build without a GPU backend
raises a device error at the call; `hpc` raises an invalid-argument error,
because it submits a whole run (`dtwc.cluster(..., device="hpc")` in Python)
rather than computing a `Problem` locally.

Before a `Problem` fills its distance matrix, or computes its first pair on
demand (`dist_by_ind`), one check decides whether the request can be honoured.
The GPU kernels compute Standard DTW on univariate Float64 series held in RAM,
in L1 or squared L2, with no missing-data strategy; anything else on a GPU raises
a device error that names the setting and its value (`variant = WDTW`,
`missing_strategy = ZeroCost`, `ndim = 3`, `precision = Float32`, mmap-backed or
view-mode series). Metal also rejects precision FP64 (its kernels are FP32) and
a GPU index other than 0 (it runs on the system default GPU). On every device, a
Sakoe-Chiba band narrower than the length difference between the shortest and
longest series is an invalid-argument error naming both series and the smallest
feasible band: such a pair has no warping path, and its distance would otherwise
be the finite `1.8e308` sentinel.

Not yet covered by that check: OneBatchPAM and FastCLARA's assignment step,
which compute through `Problem::dtw_function()`. On-demand distances and the
matrix-free schedules (`onebatch`, `clara`, `tadpole`, `clarans`) compute on the
CPU even when a `Problem`'s device is a GPU; `dtwc_cl` and Tier-1 `cluster(...)`,
which share `dtwc::run`, reject that combination instead (a `clara` sample that
covers every series is PAM on the whole set, whose matrix the GPU fills).

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
