# DTWC++ Session Summary -- 2026-04-02

## Branch: Claude

## Machine

- Windows 11, Intel i7-12800H (10C/20T), RTX 3070 Laptop (8GB, CC 8.6)
- MSVC 19.40, CUDA 12.2 (driver supports 13.1, VS MSBuild targets have 13.2)
- MS-MPI SDK installed, OpenMP via MSVC `/openmp:experimental`

## What Was Done

### GPU Kernel Optimization (baseline 0.9 -> 75 Gcells/sec, 85x)

1. Anti-diagonal wavefront kernel (replaced 1-thread-per-block)
2. FP32/FP64 auto-precision by GPU architecture (GPUConfig detection)
3. Register-tiled kernel for L<=256 (cuDTW++-inspired, `__shfl_sync`)
4. Warp-level kernel for L<=32 (8 pairs/block)
5. Persistent kernel with atomic work queue (large N)
6. Double-buffer for L>1024 (saves 1 shared-mem buffer for occupancy)
7. Series preloading into shared memory (L<=256)
8. On-device pair index computation (eliminate pair arrays + H2D transfers)
9. GPU-side NxN matrix write (eliminate host fill loop)
10. Integer band boundary precomputation (avoid FP64 in inner loop)
11. CUDA streams + pinned memory + event-based timing

### New GPU APIs

- `compute_distance_matrix_cuda(series, opts)` -- full NxN pairwise matrix
- `compute_dtw_one_vs_all(series, query_index, opts)` -- 1-vs-N for clustering
- `compute_dtw_k_vs_all(series, query_indices, opts)` -- K-vs-N batched
- `compute_lb_keogh_cuda(series, band)` -- GPU LB_Keogh lower bounds
- `CUDADistMatOptions`: band, precision (Auto/FP32/FP64), use_lb_pruning, skip_threshold

### CPU Improvements

- Parallel pruned distance matrix with OpenMP + lock-free atomic NN tracking
- `DistanceMatrixStrategy` enum: Auto, BruteForce, Pruned, GPU
- Auto selects Pruned for standard DTW (LB_Kim + LB_Keogh early abandon)

### Python API

- `compute_distance_matrix(series, device='cpu'|'cuda'|'cuda:N')`
- `DTWClustering(n_clusters=3, device='cuda')`
- `CUDA_AVAILABLE`, `cuda_available()`, `cuda_device_info()`
- Graceful fallback with RuntimeWarning when CUDA unavailable

### CLI

- `dtwc_cl --device cuda --precision auto --band 10`
- Config TOML: `device = "cuda"`, `precision = "auto"`

### Build System

- macOS CUDA guard (clear "not supported" message)
- Linux nvcc auto-detection via CUDA_PATH and /usr/local/cuda
- Windows multi-CUDA-version scanning + Directory.Build.props
- CUDA compiler warnings with `-Xcompiler=` prefix
- `CMAKE_CUDA_STANDARD 17`
- CI workflow: `.github/workflows/cuda-mpi-detect.yml` (5 jobs)

### Tests

- 41/41 CTest pass
- 312/312 MPI tests pass
- CUDA tests: 41+ test cases with 6943+ assertions
- LB_Keogh tests: 6 cases with 583 assertions
- 1-vs-N/K-vs-N tests: 12 cases
- CPU pruned tests: 5 cases
- Python GPU tests: test_cuda.py (TestCUDAIntrospection, TestDeviceParsing, TestCUDADistanceMatrix, TestCUDAClustering)

## Build & Run Instructions

```bash
# Full build with everything
cmake -S . -B build \
  -DDTWC_BUILD_TESTING=ON \
  -DDTWC_BUILD_BENCHMARK=ON \
  -DDTWC_ENABLE_CUDA=ON \
  -DDTWC_ENABLE_MPI=ON

# On Windows: CUDA_PATH may point to v13.2 but driver supports 13.1
# The Directory.Build.props pins to CUDA_PATH for MSBuild compatibility
# Force v12.2: CUDA_PATH="C:\...\CUDA\v12.2" cmake ...

cmake --build build --config Release -j

# Tests
ctest --test-dir build --build-config Release -j8
mpiexec -n 4 ./build/bin/unit_test_mpi

# Benchmarks
./build/bin/bench_cuda_dtw          # GPU vs CPU
./build/bin/bench_dtw_baseline      # CPU microbenchmarks
mpiexec -n 4 ./build/bin/bench_mpi_dtw

# CLI
./build/bin/dtwc_cl --input data/dummy -k 3 --device cuda -v
```

## Known Issues

1. **CUDA driver/toolkit version mismatch**: CUDA_PATH points to v13.2 but driver (591.74) supports max 13.1. The VS MSBuild targets use v13.2. `Directory.Build.props` pins `CudaToolkitCustomDir` to the CUDA_PATH version. Benchmarks and tests work; CLI `--device cuda` may crash during data loading on this specific machine (WDDM driver issue).

2. **Register-tiled kernel (dtw_regtile_kernel)**: Has been verified correct for full DTW but some banded DTW boundary edge cases may have precision issues. Disabled for banded mode, falls through to wavefront kernel.

3. **`-ffast-math` in Release builds**: Breaks `std::isnan()` / NaN propagation. The adversarial review flagged this but it's a pre-existing issue, not introduced by this session. Consider `-ffast-math -fno-finite-math-only` as a safer alternative.

4. **Python 3.9 on this machine**: Cannot install torchdtw (requires 3.12+) or pytorch-softdtw-cuda for cross-library benchmarking.

## Key Files Modified/Created

### New files (6)
- `dtwc/cuda/cuda_memory.cuh` -- RAII wrappers (CudaPtr, PinnedPtr, CudaStream, CudaEvent)
- `dtwc/cuda/gpu_config.cuh` -- GPU capability detection + FP64 rate classification
- `.github/workflows/cuda-mpi-detect.yml` -- CI smoke test (5 jobs)
- `tests/unit/test_cuda_lb_keogh.cpp` -- LB_Keogh GPU tests
- `tests/python/test_cuda.py` -- Python GPU tests
- `benchmarks/bench_cuda_python.py` -- Python GPU benchmark

### Heavily modified
- `dtwc/cuda/cuda_dtw.cu` -- 5 kernel variants + host dispatch (~1500 lines)
- `dtwc/cuda/cuda_dtw.cuh` -- Full GPU API surface
- `dtwc/Problem.cpp` + `dtwc/Problem.hpp` -- Strategy enum, pruned path wiring
- `dtwc/core/pruned_distance_matrix.cpp` -- Parallel OpenMP + atomic NN
- `python/dtwcpp/__init__.py` -- device= API
- `python/dtwcpp/_clustering.py` -- DTWClustering device=
- `python/src/_dtwcpp_core.cpp` -- CUDA bindings
- `dtwc/dtwc_cl.cpp` -- CLI --device --precision flags

## Performance Summary

| Backend | Config | Time | Technique |
|---|---|---|---|
| CPU brute-force | 100x500 | 320ms | OpenMP parallel, no pruning |
| CPU pruned | 100x500 | ~220ms* | OpenMP + LB_Keogh early-abandon |
| GPU (FP32 auto) | 100x500 | **20ms** | Register-tiled + persistent |
| GPU K-vs-N (K=5) | 5x200x500 | **~5ms** | Dedicated 1-vs-N kernel |

*Estimated, depends on data distribution and pruning effectiveness.
