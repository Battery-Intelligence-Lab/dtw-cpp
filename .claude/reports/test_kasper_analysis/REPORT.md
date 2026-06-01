# Kasper Parquet Sample — DTWC++ Analysis Report

> **PRIVATE — do not push to GitHub.** Internal evaluation report only.
> Date: 2026-05-17.  Author: Volkan + Claude Opus 4.7 (1M ctx).
> Library: DTWC++ wheel v1.0.0 (installed in `.venv`).
> Sample file: `data/test_kasper/sequence_example.parquet` (1.44 MB).
> All artefacts in `.claude/reports/test_kasper_analysis/`.

---

## 1. Executive summary

- **The sample is one Polars-written Parquet file with 137 ride traces stored as a `large_list<double>` column.** Sequences are variable length (3681–6170 samples, mean 4834), values lie in [−734, +1046], all rides start near 0 (≈ 5 W quiescent) and many end at 0. Strongly consistent with **e-bike / light-EV traction-power [W]** logged at ~1 Hz with a ~3-sample zero-order-hold (ZOH) on the CAN bus.
- **DTWC++ can load this file**, but only via the C++ `load_parquet_file(path, "sequence")` path. **Auto-column-detection fails** (`find_column` only matches scalar float columns, not `large_list<double>`), and the Python `dtwcpp.load_dataset_parquet` will not load it at all (it assumes a rectangular columnar layout — `np.column_stack`). The blessed user workflow is `python -m dtwcpp.convert in.parquet -o out.arrow` then load the Arrow IPC.
- **DTW-based clustering finds only weak structure on this data**, regardless of band, k, or preprocessing. Best mean silhouette across the whole sweep is **+0.095** (k = 2, full DTW, z-normalised) — well below the conventional 0.25 "meaningful structure" threshold. The k = 2 split from raw L1 DTW (band = 50) is, however, **internally consistent** with simple energy/intensity features: it separates high-energy rides (mean 296 W, energy ≈ 1.5 M) from low-energy rides (mean 194 W, energy ≈ 1.0 M) at Cohen's *d* ≈ 1.5 on `mean_power`. Feature-space silhouette of that DTW split is +0.11, so the split is real, just weak.
- **A simple feature-based kMeans (7 summary features, k = 2) achieves silhouette ≈ 0.24** — more than 2× the best DTW result. **For this dataset, DTW is the wrong tool for clustering; the signal is in summary statistics, not in time-warped shape.**
- **The preprocessing chain we built (strip-idle + ZOH-decimate + SG + derivative + z-norm) made clustering *worse***. Post-preprocessing, k ≥ 4 produces singleton clusters; the k = 2 split becomes a weak ride-length proxy (Cohen's *d* ≈ 0.8 on `ride_length`, dropping below 0.3 on every magnitude feature). Stripping magnitude information removed the only meaningful signal the data carried. We still ship the pipeline as `dtwcpp.preprocess` because it is the right tool **when the user actually wants shape-only clustering** — but the docstring is explicit about when not to apply it.
- **The library is not at fault.** Independent library review confirms FastPAM is ragged-input-correct, the parquet loader correctly handles `large_list<double>` (when given an explicit column name), and OpenMP parallelism is wired through in v1.0.0. The misleading auto-detect docstring and the rectangular-only Python loader are minor fix-ups, listed in §7. **One real regression surfaced**: the fresh v2.0.0 rebuild through `pip wheel` silently lost OpenMP (`find_package(OpenMP)` returns not-found under scikit-build-core + MSVC 18), making the new wheel ~3× slower than v1.0.0 on PAM. Logged as §4.6 / §7.1.
- **No method (PAM, hierarchical, CLARA, feature kMeans) achieves an honest silhouette above ~0.24, and pairwise ARI between the three families is within noise of zero (§4.5).** The data does not contain a robust cluster structure under any of these notions. Hierarchical's apparent k=2 silhouette of +0.25 is a degenerate [136, 1] split — one outlier ride peeled off — not a clustering.

---

## 2. Dataset characterisation

### 2.1 Schema and on-disk layout

```text
File:        data/test_kasper/sequence_example.parquet
Size:        1,437,934 bytes (1.44 MB)
Producer:    Polars
Format:      Parquet v1.0, 1 row group, 137 rows, 5.30 MB uncompressed
Columns:
  ride_number : uint32
  sequence    : large_list<element: double>
  ride_length : uint32   (equals len(sequence) for every row — verified)
```

### 2.2 Series-level statistics (full dataset)

| Quantity                | min      | mean   | median | max     |
|-------------------------|---------:|-------:|-------:|--------:|
| Length (samples)        |     3681 |   4834 |   4819 |    6170 |
| Per-series mean         |    49.95 | 222.35 | —      |  376.96 |
| Per-series std-dev      |   127.02 | 231.05 | —      |  366.53 |
| Per-series min          |  −734.03 |    —   | —      |  −64.17 |
| Per-series max          |   625.54 |    —   | —      | 1046.21 |
| Per-series range (max−min) | 726.5  | 1219.9 | —      | 1697.8  |

All 137 series have a negative minimum (regen-braking sign), all start within ≈[0, 6.3] (quiescent plateau), and 36 of 137 end at exactly 0 (vehicle shut-down). No NaN/Inf — `np.isfinite(all).all() == True`.

### 2.3 Temporal structure — ZOH artefact (critical)

| Quantity                                | min  | mean | median | max  |
|-----------------------------------------|-----:|-----:|-------:|-----:|
| Fraction of `s[i] == s[i-1]`            | 0.66 | 0.70 |  0.69  | 0.91 |
| Median run-length (constant-value runs) |  —   |  —   |  3.0   |  —   |
| Idle fraction (`|v| < 10`)              | 0.03 | 0.20 |  0.18  | 0.73 |
| Regen fraction (`v < 0`)                | 0.02 | 0.15 |  —     | 0.39 |

≈ 70 % of the samples are exact repeats of the previous sample, with a median run length of 3 — a textbook **ZOH at one-third the underlying logger rate**. Effective unique-event counts per ride are roughly L/3 ≈ 1200–2000, which at 1 Hz corresponds to ~20–34 minute rides. Only one ride exceeds 50 % idle (`ride_84`, 73 %); the rest are genuinely active traces.

### 2.4 Physical interpretation (domain reviewer)

Most plausible quantity is **battery/traction power in watts** of an e-bike or light electric two-wheeler:
- Peak discharge ≈ 1.05 kW, peak regen ≈ −0.73 kW — fits a 1 kW-class brushless motor.
- Quiescent ~5 W is consistent with BMS + lights idle draw.
- Velocity is ruled out (−734 m/s); pack current at 48 V would imply ~50 kW peak (implausible at this voltage class).

---

## 3. How DTWC++ handles this data

### 3.1 Reader path that works

- **C++**: `dtwc::io::load_parquet_file(path, col_name="sequence")` in [parquet_reader.hpp:70-147](../../../dtwc/io/parquet_reader.hpp#L70-L147). The `LARGE_LIST` branch (lines 117-127) correctly casts to `arrow::LargeListArray`, walks offsets with `int64_t`, and copies each list element into an owned `std::vector<data_t>` — safe even though the file is mmap-backed (line 74).
- **C++ chunked**: `dtwc::io::ParquetChunkReader` in [parquet_chunk_reader.hpp:222-446](../../../dtwc/io/parquet_chunk_reader.hpp#L222-L446) supports row-group streaming and float32 path, useful for larger Kasper-style data sets.
- **Python (blessed)**: `python -m dtwcpp.convert sequence_example.parquet -o out.arrow` (`python/dtwcpp/convert.py`). The conversion script auto-detects both columnar and list-column Polars output and writes a zero-copy Arrow IPC the rest of the toolchain can mmap.

### 3.2 Reader paths that silently fail

- **Auto-column detection**: `find_column` ([parquet_reader.hpp:43-59](../../../dtwc/io/parquet_reader.hpp#L43-L59)) only matches `arrow::Type::DOUBLE` / `arrow::Type::FLOAT` scalar columns. The Kasper schema is `{uint32, large_list<double>, uint32}`; calling `load_parquet_file(path)` with no name throws `"No numeric column found in Parquet schema. Use --column to specify."` The docstring at line 5 ("a single file with a `List<Float64>` column are both supported") is **misleading**.
- **`dtwcpp.load_dataset_parquet`** ([python/dtwcpp/io.py:236-262](../../../python/dtwcpp/io.py#L236-L262)) calls `np.column_stack([table.column(c).to_numpy() ...])`. For a single `large_list` column with ragged rows this raises `ValueError` (or worse, silently truncates). **Do not use this on Polars list-column Parquet.**

### 3.3 CLI binary state

`bin/dtwc_cl.exe` is **stale** — it predates the `--column`, `--method pam`, `--band` and `--dtype` flags that are present in `dtwc/dtwc_cl.cpp:139-282`. Its help still says "Method (kMedoids or MIP)". The wheel `dist/dtwcpp-1.0.0-cp313-cp313-win_amd64.whl` is also v1.0.0 — older than the C++ source — and is missing `Linkage`, `HierarchicalOptions`, `calinski_harabasz_index`, `dunn_index`, `clarans`, and `compute_distance_matrix(device=...)`.  Rebuilding the wheel fails on this machine with `error MSB3491 — path exceeds OS max path limit (260)` inside `llfio/quickcpplib/outcome` populate. **A clean rebuild from a shorter working directory or with `subst Z: .` would resolve it; until then, all results below come from the installed v1.0.0 wheel.**

### 3.4 Correctness of FastPAM on ragged input

`fast_pam` (`dtwc/algorithms/fast_pam.cpp:92-251`) operates exclusively on the precomputed distance matrix via `prob.distByInd(p, x)` — it never touches `data.p_vec`. `Data::p_vec` is `vector<vector<data_t>>` ([Data.hpp:34](../../../dtwc/Data.hpp#L34)) and `Data::series_length(i)` returns the per-series length. **Variable-length input is handled correctly through the precompute-once, lookup-many pattern.** Independent review confirms this; no structural fix needed in the algorithm.

---

## 4. Clustering results

### 4.1 Raw L1 DTW, FastPAM (rescored after wiring `clusters_ind`/`centroids_ind` back into Problem)

| band | k | total_cost     | mean silhouette | DB index | sizes                       | t_PAM (s) |
|-----:|--:|---------------:|----------------:|---------:|-----------------------------|----------:|
|   50 | 2 | 100 929 071.68 |       **+0.091**|    2.61  | [38, 99]                    |     0.93  |
|   50 | 3 |  98 231 549.98 |          +0.028 |    3.03  | [57, 36, 44]                |     0.93  |
|   50 | 5 |  94 312 570.87 |          +0.019 |    2.32  | [12, 31, 54, 18, 22]        |     0.94  |
|  100 | 2 |  84 001 573.72 |          +0.049 |    2.08  | [48, 89]                    |     1.87  |
|  100 | 3 |  81 787 053.24 |          +0.043 |    2.74  | [71, 36, 30]                |     1.85  |
|  100 | 5 |  78 367 144.88 |          +0.011 |    2.38  | [28, 44, 20, 24, 21]        |     1.90  |
|  200 | 2 |  65 074 813.05 |          +0.067 |    2.91  | [60, 77]                    |     3.82  |
|  200 | 3 |  62 709 304.92 |          +0.035 |    2.75  | [42, 50, 45]                |     3.78  |
|  200 | 5 |  60 269 406.84 |          +0.018 |    2.61  | [28, 30, 31, 37, 11]        |     3.85  |

Full table in `results.json` (6 k values × 3 bands).

**Cross-check with FastCLARA (band=100):**

| k | total_cost   | mean silhouette | DB index | sizes                | t (s) |
|--:|-------------:|----------------:|---------:|----------------------|------:|
| 3 | 82 835 491.2 | +0.040          | 1.97     | [43, 7, 87]          |  6.25 |
| 5 | 80 003 343.2 | +0.032          | 1.78     | [26, 31, **1**, 74, **5**] | 10.69 |

CLARA at k=5 produces a singleton cluster — a red flag.

### 4.2 Z-normalised L1 DTW, FastPAM (per-series `z_normalize` applied before clustering)

| band | k | total_cost  | mean silhouette | DB index | sizes                    | t_PAM (s) |
|-----:|--:|------------:|----------------:|---------:|--------------------------|----------:|
|  100 | 2 | 363 089.66  |       +0.027    |    2.39  | [67, 70]                 |     1.89  |
|  100 | 3 | 351 529.19  |       +0.019    |    2.25  | [34, 52, 51]             |     1.85  |
|  100 | 5 | 337 789.06  |       +0.008    |    2.27  | [24, 27, 29, 40, 17]     |     1.87  |
|   -1 | 2 | 192 614.45  |   **+0.095**    |    2.14  | [61, 76]                 |    48.75  |
|   -1 | 3 | 186 261.51  |       +0.071    |    2.20  | [44, 35, 58]             |    48.14  |
|   -1 | 5 | 179 952.28  |       +0.045    |    2.23  | [46, 14, 15, 23, 39]     |    55.80  |
|   -1 | 6 | 177 718.07  |       +0.045    |    1.95  | [**1**, 46, 23, 14, 14, 39]   |    48.84  |
|   -1 | 8 | 173 531.49  |       +0.022    |    1.84  | [**1**, 31, 14, **1**, 40, 14, 19, 17] |    48.24  |

Full table in `znorm_results.json`.

### 4.3 Preprocessed (strip-idle + ZOH-decimate + SG + derivative + z-norm), FastPAM full DTW

Pipeline applied per-ride via the prototype `preprocess.power_signal` (now shipped — see §6.5). Post-pipeline length distribution: min = 378, mean = 1355, max = 1975. Length ratio rises to 5.22 (from 1.68 raw) because the strip + decimate step removes more from short / plateau-heavy rides than long ones.

| k  | total_cost  | mean silhouette | DB index | sizes                              | t (s) | flag                                  |
|---:|------------:|----------------:|---------:|------------------------------------|------:|---------------------------------------|
|  2 | 63 678.81   |       +0.031    |    3.60  | [54, 83]                           |  3.8  | —                                     |
|  3 | 62 828.25   |       +0.013    |    3.38  | [53, 31, 53]                       |  3.8  | —                                     |
|  4 | 62 013.00   |       +0.015    |    2.70  | [1, 30, 53, 53]                    |  3.9  | **SINGLETON**                         |
|  5 | 61 248.50   |       +0.016    |    2.30  | [1, 29, 53, 53, 1]                 |  3.8  | **SINGLETON**                         |
|  6 | 60 516.59   |       +0.017    |    2.04  | [1, 1, 28, 53, 53, 1]              |  5.9  | **SINGLETON**                         |
|  8 | 59 081.71   |       +0.020    |    1.71  | [53, 53, 1, 1, 1, 26, 1, 1]        |  5.1  | **SINGLETON**                         |
| 10 | 57 689.81   |       +0.019    |    1.51  | [1, 1, 1, 1, 26, 1, 51, 1, 53, 1]  |  3.9  | **SINGLETON**                         |

Singleton members (k = 8, full PAM): rides {7, 31, 119, 107, 21} — all *active* rides (idle fraction 8–24 %, std 175–279 W), not degenerate plateau cases. They are unusual *shapes* that PAM picks as their own medoid because no swap improves total cost; the underlying data clusters as two groups of ~53 plus a residual ~31 that the solver fails to split usefully.

### 4.4 Hierarchical (average linkage) + new scoring — extended sweep on wheel 2.0.0

Re-ran on the freshly rebuilt v2.0.0 wheel (after `subst Z: C:\D\git\dtw-cpp` cleared the OS-260-char path block) so we get `build_dendrogram` / `cut_dendrogram` and the new CH / Dunn / ARI scorers.

Raw band=100:

| method               | k | sil      | DB    | CH    | Dunn   | sizes              | t (s) |
|----------------------|--:|---------:|------:|------:|-------:|--------------------|------:|
| fast_pam             | 2 |  +0.0492 | 2.08  | 42.9  | 0.231  | [48, 89]           | 6.0   |
| fast_pam             | 3 |  +0.0430 | 2.74  | 20.9  | 0.260  | [71, 36, 30]       | 6.0   |
| fast_pam             | 5 |  +0.0174 | 2.54  | 15.1  | 0.247  | [22, 23, 40, 29, 23]| 6.2  |
| hierarchical_avg     | 2 |  **+0.2497** | 2.10 | **87.8** | 0.419 | **[136, 1]**       | < 1   |
| hierarchical_avg     | 3 |  +0.1464 | 2.31  | 43.0  | 0.419  | [135, 1, 1]        | < 1   |
| hierarchical_avg     | 4 |  +0.1026 | 2.85  | 28.3  | 0.401  | [134, 1, 1, 1]     | < 1   |

**The hierarchical k=2 silhouette of +0.25 is a degenerate result**, not a clustering: the linkage simply peels off **one outlier ride** and lumps the remaining 136 into a single cluster. Each higher k just peels off one more outlier (chain-of-singletons), confirming the dataset has no robust hierarchical structure either. CH index inflates correspondingly because the single-vs-many partition trivially maximises between-cluster variance. Treating any of these splits as a real partition would be a mistake — the only honest reading is "there is no DTW-space hierarchy that splits the bulk into multiple meaningful groups."

### 4.5 ARI cross-method agreement at k = 2..4

Three independent methods on the same data — do they agree on which rides go together?

| k | raw-DTW vs feature-kMeans | raw-DTW vs preprocessed-DTW | preprocessed-DTW vs feature-kMeans |
|--:|--------------------------:|----------------------------:|-----------------------------------:|
| 2 |                +0.017     |                  −0.007     |                          −0.004    |
| 3 |                +0.004     |                  +0.020     |                          +0.031    |
| 4 |                +0.039     |                  +0.019     |                          +0.016    |

Every cross-method ARI is within noise of zero (ARI = 0 is "random labelling"). **No two of the three methods agree on the partition.** This is the strongest possible argument that the data has no robust cluster structure — different reasonable methods see different (and uncorrelated) groupings. Even the modestly-positive raw-DTW k=2 split (energy intensity) does not match the kMeans k=2 split on similar features, because PAM and KMeans optimise different cost functions and the data is dense enough in feature space that small cost differences move the boundary substantially.

### 4.6 Build regression discovered during the v2.0.0 rebuild

`pip wheel --no-deps .` from the shortened `Z:\` path **succeeds but produces an OpenMP-less binary**: `dtwcpp.OPENMP_AVAILABLE == False`, `openmp_max_threads() == 1`. Single-pair raw-band=100 PAM took **6.0 s** on v2.0.0 vs **1.87 s** on v1.0.0 (≈ 3× slowdown) on identical inputs, identical machine. The `dtwc/CMakeLists.txt:108-122` OpenMP detection block has not changed, so `find_package(OpenMP COMPONENTS CXX)` is silently returning NOT-FOUND under scikit-build-core's MSVC environment when run through `pip wheel` (it found OpenMP for the v1.0 build that lives in the released wheel). Likely cause: the MSVC 18 / VS 2026 preview toolchain bundled with the dev environment is not on the FindOpenMP module's known-version list, or the experimental `OpenMP_RUNTIME_MSVC=experimental` shim only fires when the source path matches a different pattern. Not investigated this session — flagged in §7.

### 4.7 Feature-based kMeans baseline (sanity check)

7 simple features per ride (mean, std, peak, max-regen, idle fraction, length, energy sum), standardised, then sklearn `KMeans`:

| k | feature-space silhouette | sizes                |
|--:|-------------------------:|----------------------|
| 2 |              **+0.239** | [45, 92]             |
| 3 |              +0.189     | [28, 63, 46]         |
| 4 |              +0.206     | [30, 32, 49, 26]     |
| 5 |              +0.214     | [30, 48, 18, 30, 11] |

This is **2.4× the best DTW-space silhouette** on the same data. Feature-space silhouette of the *DTW* k = 2 raw split is +0.11; of the preprocessed k = 2 split is only +0.04. **The signal in this dataset is in summary statistics, not in time-warped shape.**

### 4.5 What the numbers actually say

- **Best mean silhouette overall is 0.095** (z-norm, full DTW, k=2). Conventional rule of thumb: < 0.25 = no structure / weak structure; > 0.50 = strong structure. Every configuration tested is well inside the "no useful structure" zone.
- **DB index never goes below 1.78.** Lower is better; values > 1 already indicate substantial cluster overlap.
- **Z-normalisation alone did not help.** Full DTW + z-norm just barely matched raw DTW with band=50, and at smaller bands z-norm was *worse* (silhouettes 0.007–0.027). This is consistent with the domain reviewer's diagnosis: the limiting factor is not magnitude but the ZOH staircase plus idle plateaus, neither of which z-normalisation addresses.
- **Singleton clusters appear at k ≥ 6** with full DTW — overfitting noise / capturing a single odd ride.
- **The cost function decreases monotonically with k and with band, as expected** — confirms the optimiser is working; the absence of structure is in the data + metric, not the solver.

### 4.4 Performance numbers (Windows, MSVC build, OpenMP on, this laptop)

| configuration                       | DTW cells/pair | total pairs | wall time / k |
|-------------------------------------|----------------|-------------|---------------|
| band = 50,  N=137                   |  ≈ 4.9 × 10⁵   |   9 316     |   ~0.93 s     |
| band = 100, N=137                   |  ≈ 9.7 × 10⁵   |   9 316     |   ~1.88 s     |
| band = 200, N=137                   |  ≈ 1.9 × 10⁶   |   9 316     |   ~3.80 s     |
| band = full DTW, N=137 (mean L²≈23M) | ≈ 2.3 × 10⁷   |   9 316     |  ~48.5 s      |
| FastCLARA (band=100, sample=46…50)  | small subsets  |   varies    |   ~6–11 s     |

`compute_distance_matrix` in [`_dtwcpp_core.cpp:511-556`](../../../python/src/_dtwcpp_core.cpp#L511-L556) does `#pragma omp parallel for schedule(dynamic, 16)` and releases the GIL — parallelism is genuine. The 48 s/run for full DTW on 16 logical threads matches the back-of-envelope estimate (≈ 2 × 10¹¹ cell ops at ~2 ns/cell × 8-way scaling).

---

## 5. Adversarial review — consolidated

### 5.1 Library reviewer (Opus, independent read of source)

1. **Loader is correct for `large_list<double>` content** but auto-detect cannot pick the `sequence` column — must pass `col_name="sequence"` explicitly.
2. **Python `load_dataset_parquet` is rectangular-only** and will not load this file at all.
3. **Band = 50 is too tight** for length ratio 1.68; the band kernel returns a feasible-path (not optimal) distance and never warns.
4. **No automatic z-normalisation** in the load path; raw L1 on [−734, +1046] clusters by amplitude.
5. **FastPAM correct for ragged input** (operates only on precomputed distance matrix).
6. **OpenMP parallelism is properly wired** in both `compute_distance_matrix` and `Problem::fillDistanceMatrix_BruteForce`.

### 5.2 Domain reviewer (Sonnet, independent read of data)

1. **Quantity = battery/traction power [W], 1 kW class e-bike**, 1 Hz log with ~3-sample ZOH.
2. **ZOH staircase ≈ 70 %** of samples — DTW trivially aligns flat plateaus, distance dominated by plateau length not drive shape.
3. **Refuse raw DTW** on this signal; recommend (in order): strip idle, decimate ZOH, Savitzky-Golay smooth, take first derivative (or DDTW), z-normalise the derivative, then PAM with band = 10 % of decimated length.
4. **Sanity checks** to pass before publishing:
   - No medoid has idle fraction > 0.5.
   - Within-cluster ride_length variance ≥ 50 % of between-cluster ride_length variance (otherwise we're just clustering by duration).
   - No cluster has fewer than ~5 members for k ≤ 8.

### 5.3 Workflow reviewer (Explore agent, repo scan)

1. **Blessed path** for this exact file shape: `python -m dtwcpp.convert sequence_example.parquet -o out.arrow` → load Arrow IPC → `Problem.set_data(...)` → `fast_pam(...)` (see `examples/python/08_parquet_io.py`).
2. Existing parquet tests in `tests/python/test_io.py` only cover the **rectangular** Parquet layout — no test exists for Polars `large_list<double>`. **Gap.**
3. `benchmarks/bench_parquet_access.py` benchmarks the I/O path and confirms DTW dominates I/O by ~100× on real battery data — consistent with the timings above.

---

## 6. Recommended workflow for Kasper-class data

This is the path that we should run before treating any cluster assignment as a result the partner can act on.

```text
1.  python -m dtwcpp.convert sequence_example.parquet -o sequence_example.arrow
        # converts Polars large_list to zero-copy Arrow IPC.

2.  preprocess each ride (in Python):
        a. trim leading + trailing samples where |v| < 10 W
        b. run-length-decode (drop repeated consecutive samples)
        c. Savitzky-Golay smooth(window=5, poly=2)
        d. first derivative  (np.diff)
        e. dtwcpp.z_normalize per ride

3.  load as Problem (set_data with the processed lists), band ≈ 10 % of
    median decimated length.

4.  sweep k = 2..12 with fast_pam; record cost, silhouette, DB.

5.  for each k that survives the silhouette / DB curve:
        - check no medoid has idle fraction > 0.5
        - check size of smallest cluster >= 5
        - check within-vs-between ride_length variance ratio >= 0.5

6.  for k larger than ~200 series (future Kasper-class files), switch to
    fast_clara with sample_size >= 40 + 2*k.
```

The library exposes everything needed (`z_normalize`, `fast_pam`, `fast_clara`, list-column Parquet loader, OpenMP). What it does **not** do is run any of these checks automatically — the user must.

### 6.5 What we built and shipped

- **`dtwcpp.preprocess` module** ([python/dtwcpp/preprocess.py](../../../python/dtwcpp/preprocess.py)) — the strip-idle + ZOH-decimate + SG-smooth + derivative + z-norm chain, exposed as `strip_idle`, `decimate_zoh`, `sg_smooth`, `derivative`, `z_normalize`, plus the `power_signal` chained entry point. `sg_smooth` lazy-imports scipy (optional dep). The docstring is explicit that the chain is for **shape** clustering — when the user actually wants energy/intensity clustering, raw DTW or feature-based methods are the right tool (this dataset is exactly that case).
- **24 unit tests** ([tests/python/test_preprocess.py](../../../tests/python/test_preprocess.py)) covering each helper, the chained entry point, scipy-missing-error path, and module export. All pass under the installed v1.0.0 wheel after the preprocess.py file is dropped into the venv (no C++ rebuild needed since it is pure Python). Other test suites unaffected.
- **CHANGELOG.md Unreleased** updated with an `### Added` entry describing the module and its intended-vs-misapplied use.

What we explicitly did **not** ship:

- **No "auto-preprocess" mode in the loader.** The pipeline is the wrong choice for Kasper-shape data; baking it in would silently destroy the only meaningful signal.
- **No feature-extraction helper.** That would be the right thing for Kasper, but it is a separate (and broader) scope decision; flagged as follow-up in §7.

---

## 7. Risks, gaps and follow-ups

1. ~~**CLI binary and Python wheel are stale.**~~ **PARTIALLY DONE this session.** `subst Z: C:\D\git\dtw-cpp` + `pip wheel --no-deps .` produced `dtwcpp-2.0.0` (hierarchical, CH index, ARI, NMI, `compute_distance_matrix(device=...)`, `clarans`, `preprocess` all present). **NEW ACTION**: investigate why the rebuild **dropped OpenMP** — see §4.6. PAM is ~3× slower until that is fixed. CLI binary `bin/dtwc_cl.exe` was not rebuilt this session (Python wheel was sufficient for the analysis).
2. **No regression test for Polars `large_list<double>` Parquet.** `tests/python/test_io.py` should add a fixture that writes a ragged-list Parquet via pyarrow (or Polars if available) and round-trips through both the C++ `load_parquet_file` path and `dtwc-convert`.
3. **`find_column` auto-detect is misleading.** Either advertise the actual support (scalar float only) in the docstring or extend the detector to pick the first `LIST` / `LARGE_LIST<float|double>` column.
4. **`load_dataset_parquet` (Python) will crash on Kasper-class files.** Either widen it to handle list columns or raise a clear "use `dtwc-convert` for ragged Parquet" error.
5. ~~**Recommended preprocessing should be exposed as a helper**~~ **DONE** — `dtwcpp.preprocess.power_signal(...)` shipped in this session (see §6.5). Caveat: this dataset is the one case where you should **not** apply it; ship-along docstring spells that out.
6. **Feature-extraction helper.** Kasper proves that for some telemetry the right tool is summary features + kMeans, not DTW. A `dtwcpp.features.summarise(series, ...)` returning a (N, K) matrix of mean / std / peak / energy / regen-fraction / idle-fraction / length would let users run the §4.4 baseline in two lines. Recommended scope: keep narrow (numeric features only, no domain-specific names), explicitly mark it as a sanity-check baseline, not a replacement for DTW.
7. **No baseline / sanity checks shipped with the clustering API.** A `dtwcpp.diagnose_clusters(prob, result)` returning the three sanity checks above (medoid-idle, singleton, length-correlation) would catch silent failures. Not built this session — flagged as follow-up.
8. **Headline cluster labels saved**: `.claude/reports/test_kasper_analysis/headline_pam_band100_k3_labels.csv` (FastPAM, band=100, k=3, raw signal — a reproducibility anchor, not a recommended assignment) and `preprocessed_pam_k2_labels.csv` (preprocessed k=2 — also not recommended; see §4.4 for the better feature-based baseline).

---

## 8. Artefacts produced

| File                                                  | Purpose                                  |
|-------------------------------------------------------|------------------------------------------|
| `run_kasper.py`                                       | Raw + CLARA sweep driver                 |
| `rescore_kasper.py`                                   | Re-runs PAM and wires labels back for silhouette/DBI |
| `rerun_znorm.py`                                      | Z-norm rerun (band=100 + full DTW)       |
| `run_kasper.log`, `rescore_kasper.log`, `rerun_znorm.log` | stdout from each script               |
| `results.json`                                        | All raw-signal PAM + CLARA runs, scored  |
| `znorm_results.json`                                  | All z-normalised PAM runs, scored        |
| `headline_pam_band100_k3_labels.csv`                  | (ride_number, cluster) for raw k=3 head  |
| `preprocess.py`                                       | Prototype preprocessing pipeline (later promoted to `python/dtwcpp/preprocess.py`) |
| `run_preprocessed.py`                                 | Sweep driver using the prototype + sanity checks |
| `run_preprocessed.log`, `preprocessed_results.json`   | stdout + scored results for the preprocessed sweep |
| `preprocessed_pam_k2_labels.csv`                      | Headline labels for preprocessed best (k=2); see §4.3 |
| `run_extended.py`                                     | Extended sweep driver (hierarchical, CH, Dunn, ARI) for wheel 2.0.0 |
| `run_extended.log`, `extended_results.json`           | Output of the extended sweep used in §4.4–§4.7 |
| `REPORT.md`                                           | This document                            |

**Repo-side changes shipped this session:**

- [python/dtwcpp/preprocess.py](../../../python/dtwcpp/preprocess.py) — new public module.
- `python/dtwcpp/__init__.py` — imports `preprocess`, adds it to `__all__`.
- [tests/python/test_preprocess.py](../../../tests/python/test_preprocess.py) — 24 unit tests + 1 skip for the no-scipy path.
- [CHANGELOG.md](../../../CHANGELOG.md) — Unreleased `### Added` entry.
