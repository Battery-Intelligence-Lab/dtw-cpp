# DTWC++ Benchmarks

Performance measurements on a laptop with:
- **CPU**: Intel Core i7-12800H (10 cores / 20 threads, 2.8 GHz base)
- **GPU**: NVIDIA GeForce RTX 3070 Laptop (5120 CUDA cores, 8 GB VRAM)
- **OS**: Windows 11, MSVC 19.40, CUDA 12.2

## CPU Distance Matrix (OpenMP, 10 cores)

| N | L | Band | Time (ms) | Pairs/sec | Gcells/sec |
|---|---|---|---|---|---|
| 20 | 100 | full | 1.1 | 179K | 1.8 |
| 50 | 100 | full | 3.4 | 390K | 3.9 |
| 100 | 100 | full | 12.5 | 403K | 4.0 |
| 50 | 500 | full | 88 | 14.8K | 3.7 |
| 100 | 500 | full | 318 | 16.7K | 4.2 |
| 200 | 500 | full | 1251 | 17.0K | 4.2 |
| 50 | 1000 | full | 360 | 3.7K | 3.7 |
| 100 | 1000 | full | 1324 | 4.2K | 4.2 |

## GPU Distance Matrix (CUDA wavefront kernel)

| N | L | Band | Time (ms) | Pairs/sec | Gcells/sec | vs CPU |
|---|---|---|---|---|---|---|
| 20 | 100 | full | 0.63 | 340K | 3.4 | 1.7x |
| 50 | 100 | full | 1.0 | 1.36M | 13.6 | 3.3x |
| 100 | 100 | full | 2.8 | 1.92M | 19.2 | 4.5x |
| 50 | 500 | full | 13.3 | 97.6K | 24.4 | 6.6x |
| 100 | 500 | full | 52.2 | 96.8K | 24.2 | 6.1x |
| 200 | 500 | full | 200 | 98.0K | 24.5 | 6.3x |
| 50 | 1000 | full | 51 | 24.5K | 24.5 | 7.1x |
| 100 | 1000 | full | 196 | 25.0K | 25.0 | 6.8x |

### GPU Scaling

**By N (fixed L=500):**

| N | Pairs | Time (ms) | Pairs/sec |
|---|---|---|---|
| 10 | 45 | 1.3 | 37K |
| 20 | 190 | 2.9 | 66K |
| 50 | 1225 | 14.2 | 88K |
| 100 | 4950 | 52.3 | 97K |
| 200 | 19900 | 208 | 96K |
| 500 | 124750 | 1201 | 105K |

Throughput saturates at ~100K pairs/sec for N >= 100 (enough blocks to fill all SMs).

**By L (fixed N=50):**

| L | Time (ms) | Gcells/sec |
|---|---|---|
| 100 | 1.1 | 13.0 |
| 250 | 4.0 | 21.7 |
| 500 | 13.5 | 24.5 |
| 1000 | 55.3 | 22.1 |
| 2000 | 206 | 24.1 |
| 4000 | 916 | 21.3 |

Cell throughput is stable at ~22-24 Gcells/sec for L >= 250.

## MPI Distance Matrix (4 ranks, each with OpenMP)

| N | L | Band | MPI time (ms) | Serial time (ms) | Speedup | Efficiency |
|---|---|---|---|---|---|---|
| 20 | 100 | full | 1.1 | 7.9 | 7.2x | 180% |
| 50 | 500 | full | 90 | 1357 | 15.0x | 376% |
| 100 | 500 | full | 310 | 5678 | 18.3x | 458% |
| 200 | 500 | full | 1235 | 19618 | 15.9x | 397% |
| 100 | 1000 | full | 1297 | 19926 | 15.4x | 384% |
| 50 | 500 | 50 | 19 | 984 | 52.2x | 1305% |

Note: "Serial time" is single-threaded. MPI with 4 ranks uses OpenMP (10
cores each), so the >100% efficiency reflects MPI + OpenMP combined
parallelism vs the single-thread serial reference.

## Running Benchmarks

```bash
# CPU + GPU benchmarks (Google Benchmark)
cmake -S . -B build -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_CUDA=ON
cmake --build build --config Release -j
./build/bin/bench_dtw_baseline   # CPU DTW microbenchmarks
./build/bin/bench_cuda_dtw       # GPU vs CPU distance matrix

# MPI benchmarks (standalone)
cmake -S . -B build -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_MPI=ON
cmake --build build --config Release -j
mpiexec -n 4 ./build/bin/bench_mpi_dtw       # distributed benchmark
mpiexec -n 1 ./build/bin/bench_mpi_dtw       # single-rank baseline

# JSON output for plotting
./build/bin/bench_cuda_dtw --benchmark_format=json --benchmark_out=results/cuda.json
```

## Optimization History

| Date | Change | Impact |
|---|---|---|
| 2026-04-01 | Baseline: 1-thread-per-block CUDA kernel | 910M cells/sec (4x slower than CPU) |
| 2026-04-01 | Anti-diagonal wavefront kernel (multi-threaded blocks) | 22 Gcells/sec (24x kernel speedup, 5-7x faster than CPU) |
