# DTWC++ Development TODO

**Last Updated:** 2026-04-02 (GPU optimization session)

## Completed (This Session)

- [x] CUDA anti-diagonal wavefront kernel (85x kernel speedup)
- [x] FP32/FP64 auto-precision by GPU architecture
- [x] Register-tiled kernel (L<=256, cuDTW++-inspired)
- [x] Warp-level kernel (L<=32, 8 pairs/block)
- [x] Persistent kernel with atomic work queue
- [x] Double-buffer for long series (L>1024)
- [x] Series preloading into shared memory (L<=256)
- [x] On-device pair index computation (eliminate pair arrays)
- [x] GPU-side NxN matrix write (eliminate host fill loop)
- [x] Integer band boundary precomputation
- [x] CUDA streams + pinned memory + event-based timing
- [x] GPU LB_Keogh lower bound kernels
- [x] GPU pruned distance matrix with skip_threshold
- [x] 1-vs-N and K-vs-N GPU kernels for clustering
- [x] Parallel CPU pruned distance matrix (OpenMP + atomic NN)
- [x] DistanceMatrixStrategy enum (Auto/BruteForce/Pruned/GPU)
- [x] Python `device='cpu'|'cuda'` API
- [x] CLI `--device cuda --precision auto`
- [x] Cross-platform CUDA/MPI/OpenMP detection
- [x] CI workflow for CUDA/MPI smoke testing
- [x] 3 adversarial reviews (GPU kernel, Python API, build system)
- [x] 41/41 tests, 312/312 MPI tests

## Remaining Work

### High Priority
- [ ] Wire K-vs-N kernel into fast_pam clustering loop (C++ side)
- [ ] Wire GPU LB_Keogh into clustering (prune before DTW in iterations)
- [ ] Fix register-tiled kernel for banded DTW edge cases (some precision issues at boundaries)
- [ ] CUDA streams: multi-stream pipelining for very large N (currently single stream)
- [ ] Profile register pressure with `--ptxas-options=-v`, add `__launch_bounds__`

### Medium Priority
- [ ] Hilbert-curve pair ordering for L2 cache locality (helps when N*L > L2)
- [ ] GPU early-abandon within DTW kernels (periodic threshold check)
- [ ] Template kernels on `use_squared_l2` (compile-time metric dispatch)
- [ ] HIPify for AMD GPU support (~1-2 days for 300 LOC)
- [ ] MATLAB MEX bindings with GPU support

### Low Priority / Research
- [ ] Multi-GPU support (data sharding across devices)
- [ ] TMA (Tensor Memory Accelerator) for Hopper GPUs
- [ ] `cp.async` for compute-transfer overlap on Ampere+
- [ ] GPU DTW variants (DDTW, WDTW, ADTW on GPU)
- [ ] Soft-DTW GPU kernel

### Technical Debt
- [ ] Clean up wavefront kernel dead code (preload branch for L<=256 is now unreachable)
- [ ] Unify kernel dispatch logic (currently scattered across launch_dtw_kernel)
- [ ] Add `DeviceContext` abstraction (carries device_id, stream, workspace pool)
