# Literature Survey — Missing Abilities & Speed Techniques for DTWC++

Date: 2026-07-06. Method: 4 parallel research subagents (competing libraries / CPU speed techniques / clustering algorithms / ecosystem), consolidated by the coordinating agent, with independent primary-source verification of the load-bearing claims (FasterPAM, FastDTW, EAPruned) via direct arXiv abstract fetches.

Confidence tags: **[C]** = confirmed from a fetched primary source; **[S]** = from a search snippet of a named source (abstract-level, not independently opened); **[inferred]** = design recommendation, no cited measurement; **UNVERIFIED** = could not confirm.

Baseline (what DTWC++ already has, not re-surveyed): full/banded/pruned DTW, early-abandon, Lemire O(n) LB_Keogh envelope, ADTW, soft-DTW (forward+gradient), DDTW-via-preprocessing, missing-data AROW, multivariate dependent DTW, k-medoids (Lloyd-style), CLARA + streaming CLARA, MIP exact (Gurobi/HiGHS/Benders), CUDA wavefront, Metal, MPI, mmap distance matrix, Parquet/Arrow IPC/CSV/HDF5, fp32/fp64 runtime switch.

---

## Verified load-bearing claims (independently fetched by coordinator)

1. **FasterPAM** — Schubert & Rousseeuw, *Information Systems* 101 (2021) 101804, arXiv:2008.05171. **[C]**
   - Abstract claims an **"O(k)-fold speedup in the second ('SWAP') phase"**. Mechanism (from search-result summary of the paper): for each candidate point, the loss change of removing each of the k medoids is computed in one shared pass — evaluating all k swaps per candidate costs O(n) instead of O(kn); one SWAP iteration drops from O(k(n−k)²) to O((n−k)²). So the task-prompt phrasing "exact swap in O(n) per iteration" is imprecise: it is **O(n) per candidate point for all k swaps jointly; O(n²) per full SWAP iteration**.
   - Verbatim speedups: **"With k=100,200, we observed a 458x respectively 1191x speedup compared to the original PAM SWAP algorithm."** [C]
   - FastPAM1 (shared accumulator + removal-loss precomputation, no eager swap) finds **"the same results as the original PAM algorithm"** (exact); FasterPAM adds eager swap execution with **"comparable quality"** (relaxed, not bit-identical to PAM). [C]
   - The same paper gives **FasterCLARA** and **FasterCLARANS** — a free upgrade path for our existing CLARA. [C, abstract title covers "PAM, CLARA, and CLARANS"]
   - **Citation correction**: `.claude/CITATIONS.md` currently lists this paper as *JMLR 22(1), 4653-4688*. Verified venue is **Information Systems 101 (2021) 101804**, DOI 10.1016/j.is.2021.101804. The earlier conference version is Schubert & Rousseeuw, SISAP 2019 (arXiv:1810.05691). Corrected entry appended to CITATIONS.md.
2. **FastDTW is a trap** — Wu & Keogh, *IEEE TKDE* 34(8):3779-3785 (also ICDE 2021), arXiv:2003.11246. Verbatim: **"In any realistic data mining application, the approximate FastDTW is much slower than the exact DTW."** Justifies never adding FastDTW; note competing crates (dtw_rs, pyts "fast" mode) still ship it. **[C]**
3. **EAPruned** — Herrmann & Webb, "Early Abandoning and Pruning for Elastic Distances including Dynamic Time Warping", arXiv:2102.05221 (journal version in *Data Mining and Knowledge Discovery* 2021). Abstract confirms the strategy "tightly integrates pruning with early abandoning" and covers **DTW, CDTW, WDTW, ERP, MSM, TWE** with "substantial speedup". **[C]** The specific numbers "around 7.62 times faster than a simple implementation, and around 2.88 times faster than … the usual computation abandoning scheme" come from a search snippet of the paper body — **[S]**. Reference implementation: `MonashTS/tempo` C++ library.

---

## A. Competing libraries — what they have that we lack

| Library | Notable capabilities beyond ours | Source |
|---|---|---|
| tslearn v0.9 | DBA-k-means, soft-DTW-k-means (`softdtw_barycenter`), KShape, GAK kernel k-means, subsequence DTW, limited-path-length DTW; joblib/numba, no GPU | https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html [C] |
| aeon v1.4 | Distance zoo: **MSM, TWE, ERP, EDR, LCSS, WDTW, WDDTW, shapeDTW, SBD**; clusterers: CLARANS, KShape, KASBA, ElasticSOM, kernel k-means; barycenters: DBA/subgradient/shift-invariant/KASBA; **independent AND dependent multivariate modes**; numba `n_jobs`, no GPU | https://www.aeon-toolkit.org/en/latest/api_reference/distances.html [C] |
| dtaidistance v2.3 | psi (cyclic) relaxation, `penalty`, `max_step`; k-means+DBA; hierarchical linkage; subsequence KNN, local concurrences/motifs; C core + OpenMP "30–300x" vs pure Python (C-vs-Python, not SIMD) | https://github.com/wannesm/dtaidistance [C] |
| pyts | Classification-focused; multiscale/fast DTW (avoid); no clustering primitives | JMLR 21(46) [C] |
| dtw-python (Giorgino) | Breadth of **step patterns** (Rabiner-Juang/Myers families), **open-begin/open-end** alignment, slope constraints; no clustering/parallelism | https://dynamictimewarping.github.io/py-api/html/api/dtw.StepPattern.html [C] |
| rust-kmedoids (kno10) | **FasterPAM, FastPAM1, parallel rayon FasterPAM** ("typically faster when you have more than 5000 instances"), **FasterMSC/DynMSC (auto-k via medoid silhouette), PAMSIL**; LAB init default | https://github.com/kno10/rust-kmedoids ; JOSS 7(75):4183 [C] |
| scikit-learn-extra | KMedoids/CLARA — effectively dead (last release 2023-03, NumPy 2.0 ABI broken) — an ecosystem gap we can fill | https://pypi.org/project/scikit-learn-extra/ [C] |
| GPU 2024–2026 | No new pairwise hard-DTW kernel found beating cuDTW++ (Schmidt & Hundt, Euro-Par 2020: ">90% of theoretical peak … Volta", ">1 order of magnitude" over cudaDTW — abstract snippet [S]). New work is soft-DTW-loss focused: arXiv:2602.17206 (2026) tiled anti-diagonal soft-DTW, "up to 98% memory reduction" by fusing distance computation [C abstract]; Maghoumi pytorch-softdtw-cuda. UNVERIFIED whether any 2024-25 hard-DTW GPU record exists. |

Our differentiators nobody surveyed has: exact MIP clustering, MPI, mmap distance matrix, streaming CLARA, Metal. TADPole is implemented in none of the surveyed Python libraries (only R dtwclust).

## B. Speed techniques we lack (CPU, clustering workloads)

1. **EAPruned kernel restructure** — integrate pruning with early abandoning in every elastic-distance kernel; ~7.62× vs naive, ~2.88× vs plain early-abandon [S]; exact; also accelerates future MSM/TWE. Herrmann & Webb 2021, arXiv:2102.05221; `MonashTS/tempo`. Effort **M**.
2. **Lower-bound cascade upgrade** — current single LB_Keogh → cascade **LB_KimFL (O(1)) → LB_Keogh → LB_Webb or LB_Enhanced**. Webb & Petitjean, *Pattern Recognition* 115 (2021), arXiv:2102.07076: "LB_WEBB … is always tighter than LB_KEOGH" and cheaper than LB_Improved [S]. Tan, Petitjean & Webb, SDM 2019, arXiv:1808.09617: LB_Enhanced "tighter than the popular Keogh lower bound, while requiring similar computation time", one parameter V trades speed/tightness, stays tight at wide bands [C]. UCR Suite (Rakthanmanon KDD 2012) supplies cascade ordering + reordered early abandon. Effort **S**.
3. **TADPole-style matrix-build pruning** — reuse each series' cached envelope + Euclidean upper bound across ALL pairs; density-peaks variant "provably identical to the brute force algorithm, but is at least an order of magnitude faster", ~94% pruning on showcase data [S]. Begum, Ulanova, Wang & Keogh, KDD 2015 (extended arXiv:1612.00637). The pruning idea transfers to our medoid-assignment step independent of density-peaks. O(n²) bound storage — pairs with our mmap matrix. Effort **M**.
4. **PrunedDTW on all-pairs** (Silva & Batista, SDM 2016): "two to ten times" on all-pairwise workloads [S] — we have PrunedDTW; the gap is feeding it a good upper bound (Euclidean or best-so-far from assignment loop) rather than none. Effort **S** (plumbing).
5. **Default narrow band + auto-tune** — Ratanamahatana & Keogh 2004: bands ≈10% typically match/beat full DTW [S]; UltraFastWWSearch (Tan, Herrmann & Webb, ICDM 2021) finds the best window "up to one order of magnitude" faster than FastWWSearch [S]. No clustering-specific band selection paper found (UNVERIFIED); pragmatic recipe: subsample, sweep band, pick smallest width where silhouette stabilises [inferred]. Effort **S**.
6. **Inter-pair SIMD batching (Highway)** — vectorize ACROSS series pairs (one pair per lane) instead of intra-matrix anti-diagonals, sidestepping the recurrence dependency; clustering has unlimited independent pairs. **No peer-reviewed CPU SIMD-DTW speedup number found — UNVERIFIED; prototype-and-measure first.** Consistent with our memory note: DTW is memory-bound (0.125 FLOP/byte); lane-batching multiplies bandwidth demand, so tile within L2 per lane group [inferred]. Effort **M–L**.
7. **Cache tiling** — two-row O(n) DP (assumed present) + diamond/parallelogram tiling of the banded wavefront; Tang et al., PPoPP 2015 "Cache-Oblivious Wavefront" [S]. Effort **M**.
8. **Sparse/RLE inputs** — AWarp (Mueen et al., ICDM 2016): "exact for binary-valued time series", "several orders of magnitude faster … on sparse time series" [S]; exact RLE-DTW: Froese et al., *Algorithmica* 2022 [S]. Only if user data is sparse. Effort **M**.
9. **Hopper DPX instructions** for the CUDA kernel — `__viaddmin`/`__vimax3` fused min-add ops; confirmed gains only for Smith-Waterman (7.8× vs A100) / Floyd-Warshall, **UNVERIFIED for DTW** [S, NVIDIA blog]. Effort **S** (kernel-local, guarded by arch).
10. **2026 general LB paradigm** — BGLB/DBGLB, arXiv:2603.14899: bipartite-graph lower bounds for elastic measures; 1-NN improvement vs prior general bound: DTW 84.9%, MSM 37.4%, TWED 48.2%; DBSCAN "up to 29%" faster; caveat: specialized DTW bounds can be slightly tighter [C]. Watch; adopt if we add MSM/TWE. Effort **M**.

**Avoid**: FastDTW (verified above). **Caution**: Elkan/Hamerly triangle-inequality pruning is invalid on raw DTW — DTW is not a metric; bound-based pruning must use LB/UB pairs (TADPole-style), not the triangle inequality [S].

## C. Algorithm additions

1. **FasterPAM + LAB init + FasterCLARA** — see verified section. LAB (Linear Approximative BUILD): subsampled BUILD, 10+⌈√n⌉ candidates per medoid; with fast SWAP, cheap init suffices [S/JOSS]. Fits our cached/streamed matrix exactly ("requires a dissimilarity matrix as input" — JOSS 4183). Effort **S–M**. **Replaces our Lloyd-style loop with a true, faster PAM.**
2. **MSM + TWE distances** — Holder, Middlehurst & Bagnall, *KAIS* 66:765-809 (2024), arXiv:2205.15181: "The move-split-merge (MSM) distance is the best performing algorithm", TWE close second; **"DTW … is not significantly better than Euclidean distance with k-medoids"**; "Using k-medoids … rather than k-means improved the clusterings for all nine elastic distance measures" [C]. **This contradicts a DTW-only strategy** — the highest-quality lever is a new distance, not a new clusterer. Both are O(n²) DPs, EAPruned-compatible, slot into our matrix builder. Effort **M**.
3. **DBA + soft-DTW barycenters (k-means mode)** — DBA: Petitjean, Ketterlin & Gançarski, *Pattern Recognition* 44(3):678-693 (2011); soft-DTW barycenter: Cuturi & Blondel, ICML 2017 (we already have the gradient); SSG improvement: Schultz & Jain, *Pattern Recognition* 74:340-358 (2018): "more stable and finds better solutions in shorter time than DBA on average" [C]. Feature-parity with tslearn/aeon/dtaidistance; but the KAIS 2024 evidence says medoids ≥ barycentric k-means on quality — position as a completeness feature, not a quality win. Needs raw-series access (not matrix) — new code path. Effort **M**.
4. **OneBatchPAM** — arXiv:2501.19285 (AAAI 2025): "reduces pairwise dissimilarity computations to O(mn) instead of O(n²)", m = O(log n) batch, "similar performances as … FasterPAM and BanditPAM++ with a drastically reduced running time" [C]. Rectangular n×m matrix — the only PAM-family algorithm architecturally viable at 100M series besides CLARA. Effort **M**.
5. **BanditPAM / BanditPAM++** — Tiwari et al., NeurIPS 2020, arXiv:2006.06856: "reduces the complexity of each PAM iteration from O(n²) to O(n log n)", "up to 200x fewer distance computations" [C]; BanditPAM++, NeurIPS 2023, arXiv:2310.18844: "O(k) faster than BanditPAM", "over 10× faster" on CIFAR10 [C]. **Caveat: its entire win is avoiding distance evaluations; with a cached/mmap matrix each evaluation is O(1) and FasterPAM wins [inferred — no published head-to-head on precomputed matrices found].** Only worth it for the on-demand-distance 100M regime, where OneBatchPAM/CLARA are simpler. Effort **L**. Deprioritise.
6. **DynMSC / FasterMSC auto-k** — Lenssen & Schubert, *Information Systems* 120 (2024), arXiv:2209.12553: direct medoid-silhouette optimization with automatic k selection, same O(k)-removal trick [C]. Natural companion to FasterPAM; answers "what k?" which MIP users ask. Effort **M**.
7. **k-shape / SBD** — Paparrizos & Gravano, SIGMOD 2015: "k-Shape outperforms all scalable approaches in terms of accuracy … [the one non-scalable match, k-medoids+cDTW] is two orders of magnitude slower than k-Shape" [C, fetched PDF]. FFT-based O(L log L) distance; SBD alone could join our distance zoo cheaply. Effort **M** (centroid extraction needs eigendecomposition).
8. **TADPole (as a clusterer)** — see B.3; anytime density-peaks with admissible pruning. Effort **M**; niche vs. using its pruning inside k-medoids.
9. **Hierarchical clustering with DTW linkage** — trivial consumer of our existing matrix (SciPy-style linkage); useful for dendrogram-based k selection; O(n²) memory-capped. No scalable DTW-linkage literature found (UNVERIFIED absence). Effort **S**.
10. **Multivariate independent DTW** — sum of per-channel DTWs alongside our dependent mode; aeon/tslearn support both; Shokoohi-Yekta et al. (DMKD 2017) established neither dominates. TC-DTW (arXiv:2101.07731) claims "speedups up to 25× (7.5× average)" for multivariate LB_Keogh tightening [S]. Effort **S**.
11. **shapeDTW / WDTW** — Zhao & Itti, *Pattern Recognition* 74:171-184 (2018): "beats DTW on 64 out of 84 UCR datasets" (classification; clustering gains UNVERIFIED) [C]; WDTW: Jeong et al. 2011 (already cited). Note: neither beat MSM/TWE in the KAIS 2024 clustering evaluation. Effort **S** each (distance kernels only).
12. **COBRAS-TS semi-supervised** — Van Craenendonck et al., Discovery Science 2018, arXiv:1805.00779: active must-link/cannot-link constraints, "outperforms unsupervised and semi-supervised competitors by a large margin" [C]. Interactive querying is orthogonal to our batch/HPC use; **L**, deprioritise.
13. **KASBA** — Holder & Bagnall 2024, arXiv:2411.17838: MSM-based k-means, "orders of magnitude improvement in run time over the most performant k-means alternatives" [C]. Only after MSM lands. Effort **M**.

## D. Ecosystem

1. **Arrow C Data Interface + PyCapsule protocol** — consume `__arrow_c_array__`/`__arrow_c_stream__` in the pybind11 layer via nanoarrow (two files, zero deps): zero-copy ingest from polars (≥1.3), pyarrow, DuckDB, pandas with **no pyarrow dependency**. https://arrow.apache.org/docs/format/CDataInterface.html [C]. Effort **S**. Skip ADBC (database drivers) and DataFusion UDFs (Rust-centric).
2. **conda-forge (CPU-only first)** — staged-recipes PR → bot-maintained feedstock; dtaidistance and tslearn feedstocks are precedents; autotick bot handles migrations; CUDA variants raise cost, defer. https://conda-forge.org/docs/maintainer/adding_pkgs/ [C]. Effort **S–M**. Also makes dtwcpp pixi-installable for free.
3. **scikit-learn estimator wrapper** — `BaseEstimator` + `ClusterMixin`, `__sklearn_tags__()` (sklearn ≥1.6 public tags API), pass `check_estimator`; mirror aeon `TimeSeriesKMedoids` API; support `metric="precomputed"` so any sklearn clusterer can consume our matrix. scikit-learn-extra's death leaves a KMedoids vacuum. https://scikit-learn.org/stable/developers/develop.html [C]. Effort **M** (variable-length input policy is the hard part).
4. **cibuildwheel consolidation** — one config for Linux (incl. aarch64), macOS, Windows ARM64, optional Pyodide/WASM (`--platform pyodide`, CPU-only, OpenMP off); uv as build frontend. https://cibuildwheel.pypa.io/en/stable/platforms/ [C]. Effort **S–M**.
5. **ONNX — skip** [C mechanics, verdict inferred]: DTW has no ONNX operator; onnxruntime custom ops (`RegisterCustomOps` shared lib) force every consumer to ship our binary anyway, destroying portability rationale. Benefit ≈ 0, effort L.
6. **R / Julia bindings — defer/skip**: CRAN `dtw` saturates R-side distance computation (active, June 2026 release); Julia has native DynamicAxisWarping.jl. Only differentiator would be clustering-at-scale via Rcpp — on demand only. Effort **L**.

---

## Consolidated ranking (impact for large-N clustering ÷ effort)

| # | Addition | Impact | Effort | Key evidence |
|---|---|---|---|---|
| 1 | FasterPAM + LAB + FasterCLARA | Very high (458–1191× SWAP; true PAM quality) | S–M | Schubert & Rousseeuw IS 2021 [C] |
| 2 | LB cascade: LB_KimFL→LB_Keogh→LB_Webb/LB_Enhanced | High (exact pruning, wide-band robust) | S | Webb & Petitjean PR 2021 [S]; Tan SDM 2019 [C] |
| 3 | EAPruned kernels | High (~2.9× over current EA; covers MSM/TWE) | M | Herrmann & Webb DMKD 2021 [C/S] |
| 4 | MSM + TWE distances | High (best clustering quality; DTW≈Euclidean under k-medoids) | M | Holder KAIS 2024 [C] |
| 5 | TADPole-style envelope-cached matrix pruning | High (~94% pruning, ≥10× clustering) | M | Begum KDD 2015 [S] |
| 6 | Multivariate independent DTW (+TC-DTW LB) | Med-high (multi-dim ask; parity) | S | Shokoohi-Yekta DMKD 2017; TC-DTW [S] |
| 7 | Arrow PyCapsule zero-copy ingest | Medium (polars/DuckDB era) | S | Arrow C Data docs [C] |
| 8 | OneBatchPAM (100M tier) | Medium-high at extreme N | M | arXiv:2501.19285 AAAI 2025 [C] |
| 9 | DBA + soft-DTW barycenter k-means | Medium (parity; quality caveat) | M | Petitjean 2011; Cuturi & Blondel 2017 [C] |
| 10 | conda-forge + sklearn wrapper (KMedoids vacuum) | Medium | S–M | conda-forge docs; sklearn dev docs [C] |

Below the line: hierarchical linkage (S, easy win but O(n²)-capped), DynMSC auto-k, band auto-tune default, k-shape/SBD, shapeDTW/WDTW kernels, inter-pair SIMD (prototype first — UNVERIFIED literature), DPX CUDA ops (UNVERIFIED for DTW), AWarp (sparse only), BanditPAM++ (loses to FasterPAM on cached matrices), KASBA (after MSM), COBRAS-TS, ONNX/R/Julia (skip/defer).

## Contradictions with our current approach

1. **DTW-centric quality assumption**: Holder/Middlehurst/Bagnall (KAIS 2024) — "DTW … is not significantly better than Euclidean distance with k-medoids"; MSM/TWE are the best clustering distances. Our entire distance investment is DTW-family; the literature says the next quality gain is a *different elastic distance*. [C]
2. **Lloyd-style k-medoids**: the same evaluation confirms k-medoids > k-means for all elastic distances, but our Lloyd-style alternation is dominated by FasterPAM in both quality (true SWAP) and speed. [C]
3. **SIMD next step**: no verified published CPU SIMD-DTW speedup exists; combined with DTW's memory-bound nature (0.125 FLOP/byte), SIMD before memory-layout work is likely wasted — matches our existing MEMORY.md note. [inferred/UNVERIFIED]
4. **Citation hygiene**: CITATIONS.md line 33 mis-attributes Schubert & Rousseeuw 2021 to JMLR; correct venue Information Systems 101:101804. Correction appended. [C]
