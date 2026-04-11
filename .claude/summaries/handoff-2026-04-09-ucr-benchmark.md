# Session Handoff: UCR Benchmark Infrastructure & Results (2026-04-09)

Full UCR Archive benchmark infrastructure created, all 128 datasets run on 4 architectures, results verified, timing audited by adversarial agents.

## What Was Done

### 1. Benchmark SLURM Scripts

Created two new job scripts that iterate all 128 UCR datasets with Lloyd k-medoids:

- **`scripts/slurm/jobs/ucr_benchmark_cpu.slurm`**: CPU benchmark (configurable cores/partition)
- **`scripts/slurm/jobs/ucr_benchmark_gpu.slurm`**: GPU benchmark (configurable GPU type via `--gres`, precision via `GPU_PRECISION`)

Both scripts:
- Parse `UCR_DataSummary.csv` for dataset names and k (class count)
- Strip UTF-8 BOM from CSV, validate numeric fields
- Per-dataset timeout (default 30 min, configurable via `DTWC_TIMEOUT`)
- Write per-dataset timing JSON with hardware info, cost, iterations, silhouette
- Write `benchmark_meta.json` with run summary
- Copy results back to submit directory
- `json_escape()` helper for safe hardware string embedding
- Leading-zero fix for `bc` output (`.053` → `0.053`)
- Silhouette regex handles negative values correctly

### 2. Benchmark Results (All 128 datasets, 0 failures)

| Platform | Processor | Total Time | Speedup |
|----------|-----------|-----------|---------|
| CPU (baseline) | Intel Xeon Platinum 8268, 16 cores | 43.5 min | 1.0x |
| CPU (Genoa) | AMD EPYC 9645, 168 cores | 3.7 min | 11.8x |
| GPU (L40S) | NVIDIA L40S (Ada Lovelace) | 3.9 min | 11.1x |
| GPU (H100) | NVIDIA H100 NVL (Hopper) | 3.1 min | 14.2x |

SLURM Job IDs: 7457733 (Xeon 16c), 7457745 (EPYC 168c), 7457735 (L40S), 7457734 (H100)

Key dataset timings (Xeon → H100):
- HandOutlines (N=1000, L=2709): 770s → 15.8s = **48.7x**
- ElectricDevices (N=8926, L=96): 129s → 31.0s = **4.2x** (short series, CPU wins on per-core)
- FordA (N=3601, L=500): 344s → 17.7s = **19.4x**

### 3. Verification

- **Cross-platform consistency**: ARI=1.0 between all 4 platforms (identical clusterings)
- **Against UCR ground truth**: Mean ARI=0.23, Median=0.16 (expected for unsupervised vs supervised labels). 5 datasets ARI>0.8, 34 FAIR, 89 POOR.
- **Distance matrices saved** for all 4 runs in `results/slurm/`

### 4. Adversarial Timing Audit (2 agents: Opus + Codex)

Both confirmed **timings are genuine**:
- No mmap caching (threshold=50000, all datasets N<9000)
- No pruning (band=-1 disables pruned strategy)
- Back-of-envelope: ElectricDevices predicted 28-40s DTW compute on 16 cores; actual 68.7s DTW + 60s CSV I/O = 129s. Consistent.
- GPU log confirms 39,832,275 pairs computed for ElectricDevices (= N*(N-1)/2 exactly)

Found and fixed 2 reporting bugs:
- `gpu_dtw_ms` regex missed scientific notation (`4.37e+03` → extracted `4.37`)
- `iterations` reported `max_iter` (100) instead of actual count

### 5. OpenMP Dynamic Chunk Sizing

Replaced all hardcoded `schedule(dynamic, 16)` with runtime-adaptive `omp_chunk_size()`:

```cpp
inline int omp_chunk_size(int n_iterations, int chunks_per_thread = 4) {
    return std::max(1, n_iterations / (get_max_threads() * chunks_per_thread));
}
```

Files changed:
- `dtwc/parallelisation.hpp` — added `omp_chunk_size()`, updated `run_openmp()`
- `dtwc/core/pruned_distance_matrix.cpp` — pair-based and row-based loops
- `dtwc/algorithms/fast_pam.cpp` — SWAP loop (moved chunk computation outside `omp parallel`)
- `dtwc/Problem.cpp` — Lloyd distance matrix fill
- `dtwc/mpi/mpi_distance_matrix.cpp` — MPI per-rank loop

### 6. Lloyd Iteration Count Bug Fix

`Problem::cluster_by_kMedoidsLloyd_single()` now returns actual iteration count via `std::tuple<int, double, int>`. New `Problem::last_iterations` member stores it. CLI reads it instead of hardcoded `max_iter`.

### 7. Output Files Created

| File | Purpose |
|------|---------|
| `benchmarks/ucr_benchmark_results.json` | Website-ready JSON, 126 datasets × 4 architectures |
| `benchmarks/ucr_benchmark_results.md` | Markdown tables (JOSS-style) |
| `benchmarks/ucr_benchmark_results.tex` | LaTeX longtable with bold headers, green-highlighted heavy datasets |
| `benchmarks/ucr_benchmark_results.pdf` | 4-page compiled PDF |
| `benchmarks/gen_latex.py` | LaTeX generator script |
| `benchmarks/aggregate_results.py` | Aggregates per-dataset timing JSONs into single JSON |
| `results/slurm/ucr_benchmark_*/` | Full results with distance matrices (4 runs) |

### 8. slurm_remote.sh Updates

- `submit-benchmark-cpu` — submits full UCR CPU benchmark
- `submit-benchmark-gpu [type]` — submits GPU benchmark with optional `--gres` override (e.g., `a100`, `h100`)
- `_submit_job()` now accepts 4th arg for extra sbatch flags
- Usage comments and help text updated

### 9. Other Fixes

- `cmake/Coverage.cmake`: user reverted `TIMEOUT 120` addition (back to original)
- `CHANGELOG.md`: added benchmark entries
- Pre-existing flaky test `test_fast_pam_adversarial`: MSVC OpenMP runtime `abort()` on thread pool cleanup. All 166 assertions pass individually; crash only when running all 19 test cases together. Not related to any changes in this session.

## Current State

- **Branch**: Claude (uncommitted changes)
- **Tests**: 66/66 pass (2 CUDA skipped, 1 pre-existing flaky excluded)
- **ARC builds**: htc-cpu and htc-gpu binaries at `/data/engs-unibatt-gp/engs2321/dtw-cpp/src/build-*/` (old chunk sizes — need rebuild to get dynamic scheduling)
- **Results downloaded**: all 4 runs in `results/slurm/` locally

## What To Do Next

### Immediate
1. **Rebuild on ARC** with dynamic OpenMP chunk sizes and iteration count fix, then re-run benchmarks
2. **Investigate `test_fast_pam_adversarial` flaky crash** — MSVC OpenMP runtime issue, may need `omp_set_nested(0)` or thread pool size limiting
3. **Add pruned strategy for band=-1** — Opus agent noted LB_Kim is O(1) and valid without banding. Could skip 20-40% of DTW pairs.

### Documentation
4. Integrate benchmark results into Hugo docs website
5. Create benchmark plots (DTW time vs dataset size, GPU vs CPU speedup scaling)

### Benchmark Improvements
6. Re-run with `--verbose` to capture per-dataset DTW time separately from total (for more granular analysis)
7. Add A100 GPU run for completeness
8. Consider running on TEST splits too (not just TRAIN) for larger N datasets

## Key Files Modified

| File | Change |
|------|--------|
| `dtwc/parallelisation.hpp` | `omp_chunk_size()` + dynamic scheduling |
| `dtwc/core/pruned_distance_matrix.cpp` | Dynamic chunk + `parallelisation.hpp` include |
| `dtwc/algorithms/fast_pam.cpp` | Dynamic chunk (outside parallel block) |
| `dtwc/Problem.cpp` | Dynamic chunk + `last_iterations` tracking |
| `dtwc/Problem.hpp` | `last_iterations` member, tuple return type |
| `dtwc/mpi/mpi_distance_matrix.cpp` | Dynamic chunk + include |
| `dtwc/dtwc_cl.cpp` | Read `last_iterations` instead of `max_iter` |
| `scripts/slurm/jobs/ucr_benchmark_cpu.slurm` | NEW — full UCR CPU benchmark |
| `scripts/slurm/jobs/ucr_benchmark_gpu.slurm` | NEW — full UCR GPU benchmark |
| `scripts/slurm/slurm_remote.sh` | `submit-benchmark-cpu/gpu`, `_submit_job` 4th arg |
| `benchmarks/aggregate_results.py` | NEW — JSON aggregator |
| `benchmarks/gen_latex.py` | NEW — LaTeX table generator |
| `benchmarks/ucr_benchmark_results.json` | NEW — website JSON |
| `benchmarks/ucr_benchmark_results.md` | NEW — markdown tables |
| `benchmarks/ucr_benchmark_results.tex` | NEW — LaTeX source |
| `benchmarks/ucr_benchmark_results.pdf` | NEW — compiled PDF |
| `cmake/Coverage.cmake` | User reverted TIMEOUT change |
| `CHANGELOG.md` | Benchmark entries added |

## Known Issues

- **`test_fast_pam_adversarial` flaky**: MSVC OpenMP runtime crash when running all 19 test cases together. Pre-existing, not caused by session changes.
- **ARC Arrow CPM build**: include path still broken (`DTWC_ENABLE_ARROW=OFF` workaround)
- **Distance matrix CSV I/O dominates**: For large datasets, writing the CSV takes longer than computing DTW. Consider binary-only output option.
- **GPU slower than 168-core CPU on short series**: Crop (L=46), ElectricDevices (L=96) — CUDA kernel launch overhead dominates. Expected behavior.