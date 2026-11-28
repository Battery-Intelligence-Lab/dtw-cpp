---
title: Method
weight: 2
---


# Method

DTW-C++ computes pairwise distances with [dynamic time warping](dtw.md) and six [elastic variants](dtw-variants.md) (DDTW, WDTW, ADTW, Soft-DTW, MSM, and TWE), supporting three point metrics (`L1`, `L2`, Squared `L2`) with `LB_Keogh` and two-endpoint `LB_Kim` lower bounds ([metrics](metrics.md)), [missing-data handling](missing-data.md), [multivariate series](multivariate.md), and [CUDA and Metal backends](gpu-backends.md). Clustering uses [LR-core](algorithms.md#lr-core), [exact MIP](algorithms.md#mixed-integer-programming-mip), TADPole, FastPAM, OneBatchPAM, FastCLARA, Hierarchical clustering, and Lloyd iteration, alongside DTW barycentres and barycentre k-means, evaluated with seven [clustering scores](scores.md). Additionally, users can read our [journal article](../publications/joss-paper.md) for background and references.
