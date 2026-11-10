---
title: "Devices, HPC, and parallelism"
weight: 10
---

# Devices, HPC, and parallelism

Select a device once, or override it for one clustering call:

```python
import dtwcpp as dtwc

dtwc.device("cpu")                 # cpu | gpu | gpu:N | cuda | cuda:N | cpu+gpu | cpu+gpu:N | hpc | hpc:gpu
result = dtwc.cluster(data, 3)
result = dtwc.cluster(data, 3, device="gpu:1")
```

`gpu` resolves to CUDA where compiled and to Metal on macOS. `cpu+gpu` runs the
CPU's threads and a CUDA GPU together ([below](#cpu-and-gpu-together-cpugpu)).
A request that cannot be honoured raises a device error; DTWC++ does not silently
run the same request on CPU. `onebatch` and `tadpole` compute on the CPU as they go
and reject a GPU request, `cpu+gpu` included. `clara` runs on a GPU (below) from
`dtwc_cl`, C++, Python and MATLAB alike, on a build with a GPU backend.

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
`set_gpu_precision` chooses what the GPU computes in (`auto`, `fp32`, `fp64`), and
`set_gpu_share` how `cpu+gpu` splits its work (below).
`gpu` on a build without a GPU backend, and `cpu+gpu` on one without CUDA, raise a
device error at the call; `hpc`
raises a device error,
because it submits a whole run (`dtwc.cluster(..., device="hpc")` in Python)
rather than computing a `Problem` locally.

Before a `Problem` fills its distance matrix, or hands out the function that
computes pairs on demand (`dtw_function()`), one check decides whether the
request can be honoured.
The GPU kernels compute Standard DTW on univariate Float64 series held in RAM,
in L1 or squared L2, with no missing-data strategy; anything else on a GPU raises
a device error that names the setting and its value (`variant = WDTW`,
`missing_strategy = ZeroCost`, `ndim = 3`, `precision = Float32`, owned or a
view). Float64 series that a `Data` only views (an Arrow IPC file read in place,
a decoded record batch, `set_view_data`) are checked where they lie. CUDA reads them
there: the fill, FastCLARA's samples and its assignment pad them from the view into
the GPU's staging, and make no copy of them. Metal's fill uploads an owned copy of a view's
series, freed when the fill returns. `dtwc_cl`'s memory plan (`--ram-limit`) counts
the GPU's staging buffer, and on Metal that copy. Metal also
rejects precision FP64 (its kernels are FP32);
`set_device` already refused a GPU index other than 0 there (Metal runs on the
system default GPU). On every device, a
Sakoe-Chiba band narrower than the length difference between the shortest and
longest series is an invalid-argument error naming both series and the smallest
feasible band: such a pair has no warping path, and its distance would otherwise
be the finite `1.8e308` sentinel. So is an empty series, which `Data::p_vec` can leave after `set_data`.

On-demand distances and the schedules `onebatch` and `tadpole`, which compute
the pairs they need as they go, compute through `Problem::dtw_function()`, on the
CPU even when a `Problem`'s device is a GPU; `dtwc_cl`, C++, Python and MATLAB `cluster(...)` and `DTWClustering`,
which hand their settings to a `Problem` through `dtwc::apply`, reject
`onebatch` and `tadpole` on a GPU instead. On `cpu+gpu`, and with a forced `gpu_share`
on any device, the two are refused on a `Problem` too (`one_batch_pam(prob)`,
`tadpole(prob)`, `Problem::cluster`): they have no fill and no pass to split. FastCLARA on a GPU fills its
sample matrices there (each sample is a view of its series on CUDA, a copy on
Metal, whose fill uploads owned series) and, with CUDA, assigns every series to
the medoids there too, a block of series at a time through a two-stream pipeline
(below), so the GPU's and the host's memory stay bounded whatever the number of
series; streamed (`--ram-limit`), each record batch is assigned where it lies, an
Arrow IPC file's too. Metal has no kernel for the assignment, which then runs on
the CPU; a verbose run says so.

On CUDA, the assignment's blocks pass through a pipeline of two streams: two host
and two device slots of series padded to the longest, so that the caller's threads pad
a block and a copy stream uploads it while the block before it computes, and three
pinned slots for the blocks' distances, which the host reads in point order, so the
objective is summed in the same order whatever the overlap. Where the runtime refuses
pinned host memory a buffer falls back to pageable memory, which serialises its
copies (slower, the same results), and one line on stderr says so, naming the bytes
asked for, once for each buffer. A failed CUDA call raises a device error naming its
file, line and call (`CUDA error at <file>:<line> (<call>): <CUDA's message>`).
A verbose run prints `CUDA DTW: <pairs> pairs [FP32|FP64] in <ms>ms on <device>` after
a fill, and after the assignment `CUDA DTW: <pairs> pairs (<n> series x <m> medoids)
[FP64] in <ms>ms on <device>, in blocks of <b> series`; the time is the CUDA events'
from the first upload (the fill's series, the assignment's medoids) to the last
download.

## CPU and GPU together: `cpu+gpu`

`cpu+gpu` (`cpu+gpu:N` for GPU N) fills the distance matrix on the CPU's threads and
a CUDA GPU at once, and splits FastCLARA's assignment pass between them. It needs a
CUDA build with a device: a build without CUDA, or with Metal only (Metal has no kernel
for that pass), raises a device error where the device is selected, and a request the
GPU cannot honour (Float32 series, a variant other than Standard, a missing-data
strategy) is refused as on `gpu`. `onebatch` and `tadpole` compute on the CPU as they go,
and a streamed run (a `--ram-limit` below the series) assigns each batch on one device,
so each is a device error too. `auto` is `pam` at any N, as on `gpu`.

| Work | Where it runs on `cpu+gpu` |
|---|---|
| the distance matrix of `pam`, `kmedoids`, `mip`, `lrcore`, `hierarchical`, and of `clara` when its sample is every series | both: whole blocks of rows, split by the share |
| FastCLARA's samples (a `clara` sample smaller than N) | the GPU, as on `gpu` |
| FastCLARA's assignment of every series | both: the GPU the leading points, the CPU's threads the rest |

### How the work is split

One number, the GPU's *share*, sets every split, and `--gpu-share` (below) forces it.
It means two things, one for each use.

**The matrix fill.** The share is a rate share: the GPU's rate over the sum of the
two devices' rates. The fill goes in slabs, each one parallel loop over whole
tile-rows, blocks of consecutive rows sized so that their series fit 512 KiB: the whole
matrix when it is in RAM, the rows between two checkpoint saves, or for a mapped
matrix an eighth of what the series leave of the memory budget (at most 1 GiB). Within
a slab the GPU takes the leading tile-rows and the CPU's threads the others; no
tile-row is split, and each entry has one writer. The GPU's end bound is the one at
which the slab's predicted time, the larger of its entries over `s` and the CPU's
entries over `1 - s`, is least. The share is measured once, in the first slab with a
tile-row to compute whose tile-rows hold both calibration units: the GPU's leading
tile-rows up to 2^19 pairs and, after them, the team's first columns of the slab's
last tile-row up to as many (both count as work done; the CPU's rate is scaled to the
threads less the one that drives the GPU). Where no slab holds both units (a small fill),
or in the slabs before the one that does, a recorded ratio stands: the design's rates
for an RTX 4000 Ada beside 24 threads (inferred, not measured on your machine), 28.5
against 55 Gcell/s, 0.341 under FP64, and 0.871 under FP32 (13 times the FP64 rate).
A resumed checkpointed fill gives the GPU only tile-rows with nothing computed yet.

**FastCLARA's pass.** The share is the fraction of the series the GPU takes: the first
`round(share * N)` points, the rest the CPU's threads, in one parallel region. Measured,
the GPU assigns the first 65,536 points and the CPU's team the last 4,096 (both count
as work done), and the share is clamped between them; fewer than 69,632 series use the
recorded 0.321 (26 against 55 Gcell/s). The printed share is the points taken over N.

### Forcing the share

`--gpu-share s` (config key `gpu-share`; C++ `Problem::set_gpu_share`; Python
`gpu_share=` and `Problem.set_gpu_share`; MATLAB `'GpuShare'` and
`Problem.set_gpu_share`) is that one key. `-1`, the default, measures; a value in
[0, 1] forces it; any other value is an invalid-argument error, and so is a forced
value on a device other than `cpu+gpu`. A measured share is a timing, so a run differs
a little from the next, and under FP32 the calibrated share can differ from run to run.
`-v` and the run report print the share a run used with 17 digits and how it was set
(`recorded`, `calibrated` or `forced`); giving the printed share back as `--gpu-share`
repeats the split of every slab and of the pass.

A forced share does not reach every slab of a fill:

- The slabs before the calibrating one take the recorded ratio in a forced run too (so
  that a measured run's printed share repeats it). `--gpu-share 0` is not "CPU only"
  there and `1` not "GPU only"; `--device cpu` and `--device gpu` are the ways to run
  one device.
- Where no slab holds both calibration units, the forced share applies to every slab.
- In the calibrating slab any share above 0 keeps the GPU's calibration tile-rows (a
  forced 0.001 still gives the GPU those).
- A slab of one tile-row goes whole to one device, the one the share favours (the GPU
  above a half, the CPU at or below it): a checkpoint interval at or below the tile-row's
  height, a mapped fill's last slabs and a small N give the GPU everything or nothing
  there. At the recorded ratio that is the CPU under FP64 and the GPU under FP32.
  `-v` reports how many entries the GPU filled.

### What the split changes

Under FP64 and L1 the GPU's distances equal the CPU's bit for bit in the tests that compare
them (series of 17 to 256 steps: the warp and register-tile kernels), so the matrix equals the CPU's
and the GPU's whatever the split, and labels, medoids and cost equal both single-device
runs'. Beyond that (FP32, squared L2, series of more than 256 steps) a distance of
one device is within the route bound of the other's, not equal to it: the matrix mixes
the two (under FP32 the GPU's rows are FP32's and the CPU's FP64's, which `-v` says),
and a label or medoid can differ from a single device's run only at a near-tie whose
two distances lie within that bound.

`-v` prints the fill's split, the pass's split, and the run report's `fill` and
`cluster` rows carry the same text:

```text
Distance matrix cpu+gpu: the GPU filled 0 of 378 entries, the triangle with its diagonal (rate share 0.3413173652694611, recorded), the CPU's team the rest
FastCLARA cpu+gpu: the GPU assigned the first 128 of 256 series (share 0.5 of the series, forced), the CPU's team the rest
```

"Entries" are the packed matrix's: a triangle of N(N+1)/2 with its diagonal. Where the
share was measured its parenthesis also gives the two rates, in pairs a second (series
a second for the pass).

## HPC setup (beta)

Copy `scripts/slurm/env.example` to `.env` in the directory you run Python from (or
set `DTWC_REPO_ROOT` to the directory that holds it):

```dotenv
SLURM_HOST=arc-login.arc.ox.ac.uk
SLURM_USER=abcd1234
SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs
```

Passwordless `ssh $SLURM_USER@$SLURM_HOST` must succeed. Python owns the beta
end-to-end submission route; C++, MATLAB and `dtwc_cl` refuse `hpc` with a
`DeviceError` that names the Python and shell transports, until a real Oxford ARC
validation closes the beta.

`dtwc.cluster(data, k, device="hpc", **keys)` checks the keys and their values as a
local run does, writes the run as one `job.toml`, which the cluster's
`dtwc_cl --config` reads, and returns the labels. `device="hpc:gpu"` runs it on a
GPU of compute capability 8.0 or newer; `gpu_device="a100"` (or `"a6000"`,
`"l40s"`, `"h100"`) names one ([SLURM](../../getting-started/slurm/)). A request
that cannot be honoured fails before anything is sent: a value outside its domain
is an `InvalidInput`, as on the CPU; no `.env` is a `DeviceError`
naming the directory searched, no `bash` a `DeviceError` too, a GPU type below compute
capability 8.0 a `DeviceError` naming it, and `gpu_device` with a device other than
`"hpc:gpu"` an `InvalidInput`. A key missing from `.env`, or a value unsafe to pass to `ssh`,
stops the wrapper before it connects, naming the key. What only the cluster's GPU
can judge, a setting its kernels do not compute (a variant other than Standard, a
missing-data strategy), fails the job with that `DeviceError` in its log
(`logs/cluster_<job id>.err` under `SLURM_REMOTE_BASE/src`).

## Prove what this process can use

These are execution probes, not compile-flag reads. The parallel probe enters a
real OpenMP region; the GPU probe executes a tiny kernel and compares it with a
CPU oracle.

```python
parallel = dtwc.test.parallelisation()
gpu = dtwc.test.gpu()
assert parallel["pass"]
print(parallel)  # available, max_threads, threads_engaged, pass, reason
print(gpu)       # available, backend, device_name, validated, pass, reason, hybrid, hybrid_reason
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

`hybrid` says whether device `cpu+gpu` can run here (a CUDA build with a device), and
`hybrid_reason` why not: that the build has no CUDA, that Metal has no kernel for that
pass, or the missing device, as `available` and `reason` say it of `gpu`.

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
