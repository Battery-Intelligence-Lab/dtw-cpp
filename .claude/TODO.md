# DTWC++ Development TODO

**Last Updated:** 2026-04-02

## Completed

### Waves 1A/1B/2A/2B (2026-04-02)
- [x] Wave 1A: Metrics + Missing Data (missing_utils, MissingStrategy, DTW-AROW, 5 scoring metrics)
- [x] Wave 1B: Multivariate Foundation (ndim, MVL1/MVSquaredL2, MV DTW, MV DDTW)
- [x] Wave 2A: Clustering Algorithms (deferred alloc, medoid utils, hierarchical, CLARANS, FastCLARA fixes)
- [x] Wave 2B: MV Variants + Lower Bounds (MV WDTW/ADTW/DDTW, per-channel LB_Keogh, MV missing DTW)

### GPU Optimization (2026-04-02)
- [x] CUDA anti-diagonal wavefront kernel (85x kernel speedup)
- [x] FP32/FP64 auto-precision by GPU architecture
- [x] Register-tiled kernel (L<=256), warp-level kernel (L<=32)
- [x] Persistent kernel with atomic work queue
- [x] GPU LB_Keogh, 1-vs-N and K-vs-N kernels
- [x] Parallel CPU pruned distance matrix (OpenMP + atomic NN)
- [x] DistanceMatrixStrategy enum (Auto/BruteForce/Pruned/GPU)
- [x] Python `device='cpu'|'cuda'` API, CLI `--device cuda`
- [x] Cross-platform CUDA/MPI/OpenMP detection + CI workflow

## Remaining Work

### MIP Solver Improvements (from UNIMODULAR.md analysis)
- [ ] MIP start from PAM (warm start for both HiGHS and Gurobi, ~10 lines each)
- [ ] Gurobi: reduce NumericFocus 3->1, add MIPFocus=2, branching priority on A[i,i] diagonals
- [ ] Benders decomposition for N > 200 (master: N binary vars, subproblem: O(Nk) assignment)
- [ ] Odd-cycle cutting planes ({0,1/2}-CG cuts) as lazy constraints

### CUDA Next Phase (see .claude/superpowers/plans/2026-04-02-cuda-next-phase.md)
- [ ] Device-side pruning: stop launching DTW for LB-pruned pairs
- [ ] Architecture-aware dispatch (DispatchProfile by compute capability)
- [ ] Wire K-vs-N kernel into fast_pam clustering loop
- [ ] Wire GPU LB_Keogh into clustering iterations
- [ ] Benchmark expansion: standalone LB, pruned matrix, 1-vs-N, K-vs-N

### CUDA Medium Priority
- [ ] Fix register-tiled kernel for banded DTW edge cases
- [ ] Multi-stream pipelining for very large N
- [ ] Profile register pressure, add `__launch_bounds__`
- [ ] Hilbert-curve pair ordering for L2 cache locality
- [ ] GPU early-abandon within DTW kernels
- [ ] Template kernels on `use_squared_l2`

### Algorithms & Scale
- [ ] Condensed distance matrix (half memory for symmetric storage)
- [ ] Two-phase clustering for pre-categorized data (within-group + cross-group)
- [ ] Lazy loading (FileBackedDataSource, CachedDataSource)
- [ ] Binary distance matrix storage (HDF5 + mmap'd flat binary)
- [ ] Algorithm auto-selection based on cost = N^2 * min(L, band) * ndim

### Bindings
- [ ] MATLAB MEX bindings (skill drafted in .claude/skills/matlab-wrapper-skill.md)
- [ ] Python binding updates (skill drafted in .claude/skills/python-wrapper-skill.md)
- [ ] HIPify for AMD GPU support

### Technical Debt
- [ ] Clean up wavefront kernel dead code (unreachable preload branch for L<=256)
- [ ] Unify kernel dispatch logic
- [ ] Add `DeviceContext` abstraction (device_id, stream, workspace pool)

### Low Priority / Research
- [ ] Multi-GPU support (data sharding across devices)
- [ ] TMA for Hopper GPUs
- [ ] GPU DTW variants (DDTW, WDTW, ADTW on GPU)
- [ ] Soft-DTW GPU kernel
- [ ] SIMD via Google Highway (LB_Keogh, z_normalize, multi-pair DTW)
