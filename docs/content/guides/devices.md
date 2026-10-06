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
request on CPU. `onebatch` and `tadpole` compute on the CPU as they go and
reject a GPU request. `clara` runs on a GPU (below) from `dtwc_cl`, C++ and
Python alike, on a build with a GPU backend.

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

`gpu` selects this build's GPU backend (CUDA, else Metal) and records the
index; Metal has GPU 0 only, so `gpu:1` there is a device error, never GPU 0.
`set_gpu_precision` chooses what the GPU computes in (`auto`, `fp32`, `fp64`).
`gpu` on a build without a GPU backend raises a device error at the call; `hpc`
raises an invalid-argument error,
because it submits a whole run (`dtwc.cluster(..., device="hpc")` in Python)
rather than computing a `Problem` locally.

Before a `Problem` fills its distance matrix, or computes its first pair on
demand (`dist_by_ind`), one check decides whether the request can be honoured.
The GPU kernels compute Standard DTW on univariate Float64 series held in RAM,
in L1 or squared L2, with no missing-data strategy; anything else on a GPU raises
a device error that names the setting and its value (`variant = WDTW`,
`missing_strategy = ZeroCost`, `ndim = 3`, `precision = Float32`, mmap-backed or
view-mode series). Metal also rejects precision FP64 (its kernels are FP32);
`set_device` already refused a GPU index other than 0 there (Metal runs on the
system default GPU). On every device, a
Sakoe-Chiba band narrower than the length difference between the shortest and
longest series is an invalid-argument error naming both series and the smallest
feasible band: such a pair has no warping path, and its distance would otherwise
be the finite `1.8e308` sentinel.

On-demand distances and the matrix-free schedules `onebatch` and `tadpole`
compute through `Problem::dtw_function()`, on the CPU even when a `Problem`'s
device is a GPU; `dtwc_cl`, C++ and Python `cluster(...)` and `DTWClustering`,
which hand their settings to a `Problem` through `dtwc::apply`, reject
`onebatch` and `tadpole` on a GPU instead. FastCLARA on a GPU fills its
sample matrices there (each sample is a copy of its series, since the GPU
uploads owned series) and, with CUDA, assigns every series to the medoids there
too, a block of series at a time, so the GPU's and the host's memory stay bounded
whatever the number of series. Metal has no kernel for the assignment, which then
runs on the CPU; a verbose run says so.

## HPC setup (beta)

Copy `scripts/slurm/env.example` to `.env` in the directory you run Python from (or
set `DTWC_REPO_ROOT` to the directory that holds it):

```dotenv
SLURM_HOST=arc-login.arc.ox.ac.uk
SLURM_USER=abcd1234
SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs
```

Passwordless `ssh $SLURM_USER@$SLURM_HOST` must succeed. Python owns the beta
end-to-end submission route; C++ validates `hpc` selection but directs users to
the Python or shell transport until a real Oxford ARC validation closes the beta.

`dtwc.cluster(data, k, device="hpc", **keys)` checks the keys as a local run
does, writes the run as one `job.toml`, which the cluster's `dtwc_cl --config`
reads, and returns the labels. `device="hpc:gpu"` runs it on a GPU of compute
capability 8.0 or newer; `gpu_device="a100"` (or `"a6000"`, `"l40s"`, `"h100"`)
names one ([SLURM](../getting-started/slurm.md)). A request that cannot be
honoured fails before anything is sent: no `.env` (or no `bash`) is a
`DeviceError` naming the directory searched, a GPU type below compute capability
8.0 a `DeviceError` naming it, and `gpu_device` with a device other than
`"hpc:gpu"` an `InvalidInput`. A key missing from `.env`, or a value unsafe to
pass to `ssh`, stops the wrapper before it connects, naming the key.

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
