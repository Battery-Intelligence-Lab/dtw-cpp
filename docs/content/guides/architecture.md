---
title: "Architecture"
weight: 40
---

# Architecture

The language surfaces converge on one C++ core. Device and method validation
happen before expensive work; algorithms write their result back into `Problem`,
so Tier-1 results and Tier-2 scores observe the same state. `dtwc_cl`, C++
`dtwc::cluster`, Python `dtwcpp.cluster` and MATLAB `dtwc.cluster` are one C++
function, `dtwc::run`: the same checks, memory plan, loading decision, dispatch
and run report, whichever language calls it.

```mermaid
flowchart TB
    CPP[C++ Tier 1 / Tier 2]
    PY[Python nanobind + Python orchestration]
    MAT[MATLAB classes + MEX gateway]
    CLI[dtwc_cl CLI]
    ENV[device: cpu / gpu / cpu+gpu, fallback]
    LOAD[Lazy Dataset / DataLoader / Arrow IPC & Parquet]
    PROB[Problem: data, DistanceConfig, storage, result state]
    MATRIX[Distance fills & cdist: CPU SIMD lanes / CUDA warptile / Metal SeriesAt / .dtwm mmap]
    FREE[No parent matrix: OneBatchPAM / non-full CLARA / TADPole sparse cache / barycentres]
    ALG[Matrix consumers: FastPAM / Lloyd / Day-Edelsbrunner hierarchy]
    SOLVE[Exact routes: HiGHS 1.16 / Gurobi / LR-core PackedMatrixView]
    OUT[Result, matrix-free & matrix scores, checkpoints, reports]

    CPP --> ENV
    PY --> ENV
    MAT --> ENV
    CLI --> ENV
    CPP --> LOAD
    PY --> LOAD
    MAT --> LOAD
    CLI --> LOAD
    ENV --> PROB
    LOAD --> PROB
    PROB --> MATRIX
    PROB --> FREE
    MATRIX --> ALG
    MATRIX --> SOLVE
    FREE --> OUT
    ALG --> OUT
    SOLVE --> OUT
```

The important scaling decision is the branch after `Problem`: changing file
format (such as memory-mapped Arrow IPC) reduces ingestion cost, while choosing an algorithm that holds no full $N \times N$ distance matrix (OneBatchPAM, non-full FastCLARA, or `barycenter_kmeans`, plus matrix-free `davies_bouldin` / `inertia` scoring) avoids quadratic distance storage altogether.
