---
title: "Architecture"
weight: 40
---

# Architecture

The language surfaces converge on one C++ core. Device and method validation
happen before expensive work; algorithms write their result back into `Problem`,
so Tier-1 results and Tier-2 scores observe the same state.

```mermaid
flowchart TB
    CPP[C++ Tier 1 / Tier 2]
    PY[Python nanobind + Python orchestration]
    MAT[MATLAB classes + MEX gateway]
    CLI[dtwc_cl CLI]
    ENV[Env: device, OpenMP policy, loud errors]
    LOAD[Lazy Dataset / DataLoader / Arrow ingest]
    PROB[Problem: data, DTW policy, distance storage, result state]
    MATRIX[Distance schedules: CPU OpenMP / CUDA / Metal / mmap]
    FREE[Matrix-free schedules: OneBatchPAM / CLARA / TADPole]
    ALG[Matrix consumers: FastPAM / Lloyd / hierarchy]
    SOLVE[Exact routes: HiGHS / Gurobi / LR-core]
    OUT[Result, scores, checkpoints, CSV outputs]

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
format reduces ingestion cost, while choosing a matrix-free algorithm avoids the
quadratic distance matrix itself.
