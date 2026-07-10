# DTWC++ Citations

References used during development. Verify each citation independently before publishing.

---

## DTW and Time Series

- Sakoe, H. & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.
- Marteau, P.-F. (2009). Time warp edit distances with stiffness adjustment for time series matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 31(2), 306-318. Task 5.5 `dtwc::core::twe_distance` matches aeon 1.5.0 exactly (front zero-pad both series; d = |a−b| for univariate; defaults ν=0.001, λ=1.0, window=None).
- Jain, B. J. (2018). Semi-Metrification of the Dynamic Time Warping Distance. arXiv:1808.09964.
- Yurtman, A., Soenen, J., Meert, W., & Blockeel, H. (2023). Estimating DTW Distance Between Time Series with Missing Data. *ECML-PKDD 2023*, LNCS 14173.

## Lower Bounds and Fast DTW

- Keogh, E. & Ratanamahatana, C. A. (2005). Exact Indexing of Dynamic Time Warping. *Knowledge and Information Systems*, 7(3), 358-386.
- Kim, S.-W., Park, S., & Chu, W. W. (2001). An Index-Based Approach for Similarity Search Supporting Time Warping in Large Sequence Databases. *ICDE 2001*, 607-614.
- Rakthanmanon, T. et al. (2012). Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping. *ACM SIGKDD*, 262-270.
- Lemire, D. (2009). Faster retrieval with a two-pass dynamic-time-warping lower bound. *Pattern Recognition*, 42(9), 2169-2180.

## DTW Variants

- Keogh, E. & Pazzani, M. (2001). Derivative Dynamic Time Warping. *SIAM SDM 2001*.
- Jeong, Y.-S., Jeong, M. K., & Omitaomu, O. A. (2011). Weighted dynamic time warping for time series classification. *Pattern Recognition*, 44(9), 2231-2240.
- Cuturi, M. & Blondel, M. (2017). Soft-DTW: a Differentiable Loss Function for Time-Series. In *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 894–903. https://proceedings.mlr.press/v70/cuturi17a.html — Defines the differentiable soft-DTW value and gradient used for soft-DTW barycenters.
- Itakura, F. (1975). Minimum Prediction Residual Principle Applied to Speech Recognition. *IEEE TASSP*, 23(1), 67-72.

## Clustering and k-Medoids

- Kaufman, L. & Rousseeuw, P. J. (1987). Clustering by Means of Medoids. In *Statistical Data Analysis Based on the L1-Norm*, North-Holland, 405-416. — Original PAM paper.
- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley. — PAM (Ch. 2), CLARA (Ch. 3).
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20, 53-65.
- Schubert, E. & Rousseeuw, P. J. (2021). Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. *JMLR*, 22(1), 4653-4688. — FastPAM.
- Ng, R. T. & Han, J. (2002). CLARANS: A Method for Clustering Objects for Spatial Data Mining. *IEEE TKDE*, 14(5), 1003-1016.
- Charikar, M., Guha, S., Tardos, E., & Shmoys, D. B. (2002). A constant-factor approximation algorithm for the k-median problem. *Journal of Computer and System Sciences*, 65(1), 129-149.
- Li, S. & Svensson, O. (2013). Approximating k-median via pseudo-approximation. *STOC 2013*.
- Duran-Mateluna, C., Ales, Z., & Elloumi, S. (2023). An efficient Benders decomposition for the p-median problem. *European Journal of Operational Research*.

## Cross-Language Binding Design

- Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., & Diehl, M. (2019). CasADi: a software framework for nonlinear optimization and optimal control. *Mathematical Programming Computation*, 11(1), 1-36. — Design philosophy for cross-language API consistency (same class/method names across C++/Python/MATLAB).

## pybind11

- Jakob, W., Rhinelander, J., & Moldovan, D. (2017). pybind11 — Seamless operability between C++11 and Python. https://github.com/pybind/pybind11
- pybind11 documentation on GIL management: https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
- pybind11 documentation on return value policies: https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies

## MATLAB MEX

- MathWorks. "C MEX File Applications." MATLAB Documentation. https://www.mathworks.com/help/matlab/matlab_external/c-mex-file-applications.html
- MathWorks. "mexErrMsgIdAndTxt." — Note: This function calls `longjmp`, which skips C++ stack unwinding / destructors.
- MathWorks. "Interleaved Complex API" (R2018a+). https://www.mathworks.com/help/matlab/matlab_external/matlab-support-for-interleaved-complex.html

## Armadillo

- Sanderson, C. & Curtin, R. (2016). Armadillo: a template-based C++ library for linear algebra. *Journal of Open Source Software*, 1(2), 26.
- Column-major storage matches MATLAB (zero-copy possible); differs from NumPy row-major default.

## CUDA Architecture & Precision

- NVIDIA. "CUDA C++ Programming Guide: Compute Capabilities." https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities — FP64:FP32 throughput ratios per compute capability.
- NVIDIA. "CUDA C++ Programming Guide: Shared Memory." https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory — Opt-in extended shared memory via cudaFuncSetAttribute.
- NVIDIA. "cudaDeviceProp Reference." https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html — Runtime GPU property queries.

## GPU DTW and GPU Dynamic Programming

- Schmidt, B., & Hundt, C. (2020). *cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-Enabled GPUs*. In *Euro-Par 2020: Parallel Processing*, LNCS 12247, 597-612. Springer. https://doi.org/10.1007/978-3-030-57675-2_37 - Informed warp-intrinsic/register-tiled DTW and the goal of reducing memory traffic enough to become compute-bound.
- asbschmidt/cuDTW. GitHub repository. https://github.com/asbschmidt/cuDTW - Informed concrete CUDA implementation details for cuDTW++-style kernel structure and tiling.
- Schmidt, B., Kallenborn, F., Chacon, A., et al. (2024). *CUDASW++4.0: ultra-fast GPU-based Smith-Waterman protein sequence database search*. *BMC Bioinformatics*, 25, 342. https://doi.org/10.1186/s12859-024-05965-6 - Informed length binning, batch partitioning, warp-shuffle communication, mixed-precision ideas, and Hopper follow-up concepts.
- Latta-Lin, D., & Padilla Munoz, S. I. (2024). *Optimizing sDTW for AMD GPUs*. arXiv:2403.06931. https://doi.org/10.48550/arXiv.2403.06931 - Informed tuning of values-per-thread/reference-width ownership and architecture-aware wavefront design.
- NVIDIA. *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html - Informed warp shuffle intrinsics, shared-memory carveout, and dynamic shared-memory opt-in behavior.
- NVIDIA. *Hopper Tuning Guide*. https://docs.nvidia.com/cuda/archive/12.1.0/hopper-tuning-guide/index.html - Informed DPX, TMA, distributed shared memory, and Hopper-specific performance ceilings.

## Literature survey 2026-07-06 (see .claude/reports/literature-2026-07-06.md)

### Correction

- **CORRECTION** to the entry above listing Schubert & Rousseeuw (2021) as *JMLR 22(1), 4653-4688*: the FasterPAM paper is Schubert, E. & Rousseeuw, P. J. (2021). Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. *Information Systems*, 101, 101804. https://doi.org/10.1016/j.is.2021.101804 (arXiv:2008.05171). Conference precursor: Schubert & Rousseeuw, SISAP 2019 (arXiv:1810.05691). Verified against arXiv abstract 2026-07-06.

### k-medoids and scalable clustering

- Schubert, E. & Lenssen, L. (2022). Fast k-medoids Clustering in Rust and Python. *Journal of Open Source Software*, 7(75), 4183. https://joss.theoj.org/papers/10.21105/joss.04183 — Reference FasterPAM/LAB implementation; parallel rayon variant.
- Tiwari, M., Zhang, M. J., Mayclin, J., Thrun, S., Piech, C., & Shomorony, I. (2020). BanditPAM: Almost Linear Time k-Medoids Clustering via Multi-Armed Bandits. *NeurIPS 2020*. arXiv:2006.06856 — O(n log n) per iteration under distributional assumptions; wins only when distances are computed on demand.
- Tiwari, M., et al. (2023). BanditPAM++: Faster k-medoids Clustering. *NeurIPS 2023*. arXiv:2310.18844 — "O(k) faster than BanditPAM"; 10x on CIFAR10.
- de Mathelin, A., Cecchi, N. E., Deheeger, F., Mougeot, M., & Vayatis, N. (2025). OneBatchPAM: A Fast and Frugal K-Medoids Algorithm. *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(15), 16172–16180. https://doi.org/10.1609/aaai.v39i15.33776; arXiv:2501.19285 (https://arxiv.org/abs/2501.19285) — Uses one batch of size m ≪ n, reducing dissimilarity work and memory to O(mn). Its m = O(log n) sufficient-size statement holds with the theorem's D/Δ, iteration-count, and failure-probability quantities fixed. The paper specifies a literal +∞ sampled diagonal and presents Debias and NNIW separately; it only requires NNIW weights proportional to Voronoi counts.
- de Mathelin et al., paper-linked experiment code, `obpam@ee823101bd43dbb4095b103ef248221c682da88e`, [`onebatch.py` lines 31–42](https://github.com/antoinedemathelin/obpam/blob/ee823101bd43dbb4095b103ef248221c682da88e/onebatch.py#L31-L42) — Normalizes by the actual finite table maximum, writes the sampled diagonal as normalized 1, and uses count/mean NNIW; this is the exact hybrid estimator implemented by DTWC++.
- de Mathelin et al., maintained `onebatch` v0.1.0, commit `298ea91af821d9ecb1d1fe17b6cddafb303708df`, [`onebatchpam.py` lines 253–280](https://github.com/antoinedemathelin/onebatch/blob/298ea91af821d9ecb1d1fe17b6cddafb303708df/onebatch/onebatchpam.py#L253-L280) — Retains guarded finite-maximum normalization and count/mean NNIW but omits sampled-diagonal replacement; this differs from both the paper literal and the original experiment hybrid.
- Loog, M. (2012). Nearest neighbor-based importance weighting. *2012 IEEE International Workshop on Machine Learning for Signal Processing*, 1–6. https://doi.org/10.1109/MLSP.2012.6349714; author preprint: https://arxiv.org/abs/2102.02291 — NNeW weights each sampled point by the number of targets in its Voronoi cell. Dividing all counts by their mean preserves proportionality while setting mean weight to one.
- Lenssen, L. & Schubert, E. (2024). Medoid Silhouette clustering with automatic cluster number selection (FasterMSC, DynMSC). *Information Systems*, 120. arXiv:2209.12553.
- Begum, N., Ulanova, L., Wang, J., & Keogh, E. (2015). Accelerating Dynamic Time Warping Clustering with a Novel Admissible Pruning Strategy (TADPole). *ACM SIGKDD 2015*. Extended: arXiv:1612.00637 — envelope UB/LB pruning of the pairwise matrix, ~order-of-magnitude speedup, results identical to brute force. Density kernel is the hard CUTOFF count ρ(i)=|{j: d(i,j)<dc}| (Table 1; the only kernel a bound-based binary test can prune); cases A–D (Table 5): UB<dc⇒neighbour, LB>dc⇒not, else exact. δ(highest-density)=max of the others' δ (Table 2, differs from Rodriguez–Laio's max_j d). Theorem 1: identical labels to brute-force DP_DTW. Used by Task 5.3 (`Method::TADPole`).
- Rodriguez, A. & Laio, A. (2014). Clustering by fast search and find of density peaks. *Science*, 344(6191), 1492–1496. doi:10.1126/science.1242072 — density-peaks core underlying TADPole: local density ρ, separation δ = min distance to a higher-density point, centers = high ρ·δ, single-pass assignment to the nearest higher-density neighbour.
- Paparrizos, J. & Gravano, L. (2015). k-Shape: Efficient and Accurate Clustering of Time Series. *SIGMOD 2015* (journal: *TODS* 2017). https://www.paparrizos.org/papers/PaparrizosSIGMOD15.pdf — FFT-based SBD distance, O(L log L).
- Holder, C., Middlehurst, M., & Bagnall, A. (2024). A Review and Evaluation of Elastic Distance Functions for Time Series Clustering. *Knowledge and Information Systems*, 66, 765-809. arXiv:2205.15181 — MSM best, TWE second; DTW+k-medoids not significantly better than Euclidean; k-medoids > k-means for all nine elastic distances.
- Holder, C. & Bagnall, A. (2024). KASBA: MSM-based accelerated k-means for time series. arXiv:2411.17838.
- Van Craenendonck, T., Meert, W., Dumančić, S., & Blockeel, H. (2018). COBRAS-TS: A new approach to Semi-Supervised Clustering of Time Series. *Discovery Science 2018*. arXiv:1805.00779.

### Barycenters / DTW averaging

- Petitjean, F., Ketterlin, A., & Gançarski, P. (2011). A global averaging method for dynamic time warping, with applications to clustering (DBA). *Pattern Recognition*, 44(3), 678-693.
- Schultz, D. & Jain, B. J. (2018). Nonsmooth analysis and subgradient methods for averaging in dynamic time warping spaces. *Pattern Recognition*, 74, 340–358. https://doi.org/10.1016/j.patcog.2017.08.012; arXiv:1701.06393 (https://arxiv.org/abs/1701.06393) — Analyzes DBA as majorize-minimize and proposes stochastic subgradient (SSG) averaging; the implementation-to-paper multiplicity check remains a separate Phase 8 review item.

### Lower bounds, pruning, fast exact DTW (additions)

- Wu, R. & Keogh, E. (2022). FastDTW is approximate and Generally Slower than the Algorithm it Approximates. *IEEE TKDE*, 34(8), 3779-3785 (also ICDE 2021). arXiv:2003.11246 — verbatim: "In any realistic data mining application, the approximate FastDTW is much slower than the exact DTW." Do not add FastDTW.
- Herrmann, M. & Webb, G. I. (2021). Early abandoning and pruning for elastic distances including dynamic time warping (EAPruned). *Data Mining and Knowledge Discovery* 35(6). arXiv:2102.05221 — covers DTW, CDTW, WDTW, ERP, MSM, TWE; reference C++ impl: https://github.com/MonashTS/tempo (GPL — NOT ported; Task 5.4 is clean-room from the paper). DTW-specific precursor: Herrmann & Webb (2020), Early Abandoning PrunedDTW, arXiv:2010.05371. **Exact-mode use (Task 5.4 `dtw_kernel_eap`):** seed the prune with UB = the cost of ONE concrete warping path (the diagonal/L-path); every optimal-path cell has partial-cost ≤ DTW ≤ UB, so pruning cells above UB is provably exact. The headline ~2.88× is NN-search with a tightening cutoff; exact all-pairs (diagonal UB, no NN cutoff) prunes only near-diagonal pairs — data-cohesion-dependent, not a flat multiplier. Composes with a Sakoe-Chiba band but a tight band already excises the prunable region (so DTWC routes only UNBANDED Standard DTW through EAP).
- Silva, D. F. & Batista, G. E. A. P. A. (2016). Speeding Up All-Pairwise Dynamic Time Warping Matrix Calculation (PrunedDTW). *SIAM SDM 2016*.
- Tan, C. W., Petitjean, F., & Webb, G. I. (2019). Elastic bands across the path: A new framework and method to lower bound DTW (LB_Enhanced). *SIAM SDM 2019*. arXiv:1808.09617.
- Webb, G. I. & Petitjean, F. (2021). Tight lower bounds for Dynamic Time Warping (LB_Petitjean, LB_Webb). *Pattern Recognition*, 115, 107895. arXiv:2102.07076 — LB_Webb always tighter than LB_Keogh, cheaper than LB_Improved.
- Ratanamahatana, C. A. & Keogh, E. (2004). Everything you know about Dynamic Time Warping is Wrong. *KDD Workshop on Mining Temporal and Sequential Data* — narrow bands (~10%) typically match or beat full DTW.
- Tan, C. W., Herrmann, M., & Webb, G. I. (2021). Ultra fast warping window optimization for Dynamic Time Warping (UltraFastWWSearch). *IEEE ICDM 2021*.
- Mueen, A., Chavoshi, N., Abu-El-Rub, N., et al. (2016). AWarp: Fast Warping Distance for Sparse Time Series. *IEEE ICDM 2016* — exact on binary series, orders of magnitude faster on sparse data.
- Froese, V., et al. (2022). Fast Exact Dynamic Time Warping on Run-Length Encoded Time Series. *Algorithmica*.
- (2026). A New Lower Bounding Paradigm and Tighter Lower Bounds for Elastic Similarity Measures (BGLB/DBGLB). arXiv:2603.14899 — general LB for DTW/ERP/MSM/TWED/EDR/LCSS.
- Shen, D., et al. (2021). TC-DTW: Accelerating Multivariate Dynamic Time Warping Through Triangle Inequality and Point Clustering. arXiv:2101.07731 — multivariate LB tightening, "speedups up to 25x (7.5x average)".
- Tang, Y., et al. (2015). Cache-Oblivious Wavefront: Improving Parallelism of Recursive Dynamic Programming Algorithms without Losing Cache-Efficiency. *PPoPP 2015*.

### Distance variants (additions)

- Stefan, A., Athitsos, V., & Das, G. (2013). The Move-Split-Merge Metric for Time Series (MSM). *IEEE TKDE*, 25(6), 1425-1438. Task 5.5 `dtwc::core::msm_distance` matches aeon 1.5.0 exactly (split/merge cost `C(new,a,b)=c` if a≤new≤b else c+min(|new−a|,|new−b|); default c=1.0, window=None).
- Zhao, J. & Itti, L. (2018). shapeDTW: Shape Dynamic Time Warping. *Pattern Recognition*, 74, 171-184. arXiv:1606.01601.
- Shokoohi-Yekta, M., et al. (2017). Generalizing DTW to the multi-dimensional case requires an adaptive approach (independent vs dependent multivariate DTW). *Data Mining and Knowledge Discovery*, 31, 1-31.
- (2026). Memory-efficient differentiable soft-DTW on GPU. arXiv:2602.17206 — tiled anti-diagonal kernel, "up to 98% memory reduction" via fused distance computation.

### Ecosystem

- Apache Arrow. The Arrow C Data Interface. https://arrow.apache.org/docs/format/CDataInterface.html — zero-copy interchange with no Arrow build dependency; nanoarrow helpers: https://arrow.apache.org/nanoarrow/
- scikit-learn. Developing scikit-learn estimators (incl. `__sklearn_tags__`, sklearn >= 1.6). https://scikit-learn.org/stable/developers/develop.html
- conda-forge. Contributing packages (staged-recipes). https://conda-forge.org/docs/maintainer/adding_pkgs/
- cibuildwheel. Supported platforms (incl. Pyodide/WASM target). https://cibuildwheel.pypa.io/en/stable/platforms/

## First-Order LP / PDLP (Phase 4 · Task 4.5 — HiGHS PDLP arbiter)

- Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., & Schudy, W. (2021). *Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient*. NeurIPS 2021. arXiv:2106.04756 — the PDLP method (PDHG on the LP saddle point; core op is matrix-vector, hence GPU-friendly).
- Lu, H., et al. (2023). *cuPDLP-C: A Strengthened Implementation of cuPDLP for Linear Programming by C language*. arXiv:2312.14832 — the C/CUDA implementation HiGHS vendors as `solver="pdlp"`.
- HiGHS. ERGO-Code/HiGHS v1.15.1, `highs/pdlp/` (cuPDLP-C port + `hipdlp/` PDHG); GPU behind CMake `CUPDLP_GPU`. https://github.com/ERGO-Code/HiGHS — the maintained library used as the LP arbiter of the LR-core Lagrangian bound (supersedes the killed 2023 custom OSLP).

## Multivariate DTW (Phase 5 · Task 5.6 — independent DTW)

- Shokoohi-Yekta, M., Hu, B., Jin, H., Wang, J., & Keogh, E. (2017). *Generalizing DTW to the multi-dimensional case requires an adaptive approach*. Data Mining and Knowledge Discovery, 31(1), 1–31. — DTW_I (independent, per-channel sum) vs DTW_D (dependent, shared path); the finding that neither dominates and both are needed. Basis for `MVMode{Dependent,Independent}` + `dtw_independent_mv`.
- Shen, Y., & Chen, Y. (2021). *TC-DTW: Accelerating Multivariate DTW Through Triangle Inequality and Point Clustering*. arXiv:2101.07731. — multivariate LB tightening (triangle inequality + point clustering). NOTED, DEFERRED (bound-tightening for the pruning/NN path, orthogonal to the DTW_I distance deliverable).

## Arrow ingest (Phase 5 · Task 5.7)

- Apache Arrow. *The Arrow C Data Interface* and *The Arrow PyCapsule Interface* (`__arrow_c_array__` / `__arrow_c_stream__`). https://arrow.apache.org/docs/format/CDataInterface.html + .../format/CDataInterface/PyCapsuleInterface.html — the stable C ABI (ArrowSchema/ArrowArray/ArrowArrayStream) and Python capsule protocol used to ingest zero-copy from polars/DuckDB/pyarrow/pandas without a pyarrow dependency.
- Apache Arrow nanoarrow 0.8.0 (Apache-2.0). https://github.com/apache/arrow-nanoarrow — dependency-free C reader/builder for the Arrow C Data interface; vendored as the namespaced amalgamation `dtwc/extern/nanoarrow/nanoarrow.{h,c}` (`NANOARROW_NAMESPACE=DtwcNanoarrow`). Used by `dtwc::io::data_from_arrow`.
