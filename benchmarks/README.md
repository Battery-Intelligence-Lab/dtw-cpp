# DTWC++ Benchmarks

Performance measurements on a laptop with:

- **CPU**: Intel Core i7-12800H (10 cores / 20 threads, 2.8 GHz base)
- **GPU**: NVIDIA GeForce RTX 3070 Laptop (5120 CUDA cores, 8 GB VRAM, CC 8.6)
- **OS**: Windows 11, MSVC 19.40, CUDA 12.2

## Final GPU Performance (FP32 Auto-Precision)

| N | L | GPU (ms) | CPU (ms) | **Speedup** | Kernel Used |
|---|---|---|---|---|---|
| 100 | 100 | **0.80** | 12.3 | **15x** | Register-tiled (TILE_W=4) |
| 50 | 500 | **5.68** | 89.7 | **16x** | Wavefront (persistent) |
| 100 | 500 | **20.3** | 320 | **16x** | Wavefront (persistent) |
| 200 | 500 | **80.0** | 1297 | **16x** | Wavefront (persistent) |
| 50 | 1000 | **20.8** | 356 | **17x** | Wavefront (persistent) |
| 100 | 1000 | **80.6** | 1330 | **17x** | Wavefront (persistent) |

## GPU Kernel Throughput by Series Length

| L | Gcells/sec | Kernel | Key Technique |
|---|---|---|---|
| 100 | ~70 | `dtw_regtile_kernel<4>` | Register tiling, `__shfl_sync` |
| 250 | ~75 | `dtw_regtile_kernel<8>` | Register tiling, `__shfl_sync` |
| 500 | ~60 | `dtw_wavefront_kernel` | 3-buffer, persistent, preload |
| 1000 | ~63 | `dtw_wavefront_kernel` | 3-buffer, persistent |
| 2000 | ~68 | `dtw_wavefront_kernel` | Double-buffer, persistent |
| 4000 | ~74 | `dtw_wavefront_kernel` | Double-buffer, persistent |

Peak: **75 Gcells/sec** at L=250. Sustained >60 Gcells/sec across all lengths.

## Five Kernel Variants

| Series Length | Kernel | Threads | Technique |
|---|---|---|---|
| L <= 32 | `dtw_warp_kernel` | 8 warps/block | `__shfl_sync`, 8 pairs/block |
| 32 < L <= 128 | `dtw_regtile_kernel<4>` | 8 warps/block | cuDTW++-style register tiling |
| 128 < L <= 256 | `dtw_regtile_kernel<8>` | 8 warps/block | TILE_W=8 register tiling |
| 256 < L <= 1024 | `dtw_wavefront_kernel` | Block (128-256) | Shared-mem 3-buffer, persistent |
| L > 1024 | `dtw_wavefront_kernel` | Block (256) | Double-buffer for occupancy |

## MPI Distance Matrix (4 ranks, each with OpenMP)

| N | L | Band | MPI time (ms) | Serial time (ms) | Speedup |
|---|---|---|---|---|---|
| 50 | 500 | full | 90 | 1357 | 15x |
| 100 | 500 | full | 310 | 5678 | 18x |
| 200 | 500 | full | 1235 | 19618 | 16x |
| 50 | 500 | 50 | 19 | 984 | 52x |

## Running Benchmarks

```bash
# Build with CUDA + benchmarks
cmake -S . -B build \
  -DDTWC_BUILD_BENCHMARK=ON \
  -DDTWC_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# GPU vs CPU distance matrix
./build/bin/bench_cuda_dtw

# CPU DTW microbenchmarks
./build/bin/bench_dtw_baseline

# MPI benchmarks
cmake -S . -B build -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_MPI=ON
cmake --build build --config Release -j
mpiexec -n 4 ./build/bin/bench_mpi_dtw

# Python GPU benchmark
uv run python benchmarks/bench_cuda_python.py

# JSON output for plotting
./build/bin/bench_cuda_dtw --benchmark_format=json --benchmark_out=results/cuda.json
```

## Optimization History

| Date | Change | Throughput | vs CPU |
|---|---|---|---|
| Apr 01 | Baseline (1 thread/block) | 0.9 Gcells/s | 4x slower |
| Apr 01 | Anti-diagonal wavefront | 22 Gcells/s | 5-7x faster |
| Apr 02 | FP32 auto-precision | 50 Gcells/s | 15x faster |
| Apr 02 | Register-tiled kernel (L<=256) | 75 Gcells/s | 16x faster |
| Apr 02 | Persistent kernel + double-buffer | 74 Gcells/s (L=4000) | 17x faster |
| Apr 02 | On-device pairs + GPU matrix write | 70 Gcells/s (+36% small L) | 15-17x faster |

**85x kernel speedup** from baseline to final (0.9 -> 75 Gcells/s).
