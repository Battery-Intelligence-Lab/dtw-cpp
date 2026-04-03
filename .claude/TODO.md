# DTWC++ Development TODO

**Last Updated:** 2026-04-03

## Remaining Work

### MIP Solver
- [ ] Benders decomposition: verify on machine with HiGHS enabled (code exists, tests skip without HiGHS)
- [ ] Odd-cycle cutting planes ({0,1/2}-CG cuts) as lazy constraints

### CUDA Next Phase
- [ ] Device-side pruning: stop launching DTW for LB-pruned pairs
- [ ] Architecture-aware dispatch (DispatchProfile by compute capability)
- [ ] Wire K-vs-N kernel into CLARA clustering loop (for sample-based, not full matrix)
- [ ] Benchmark expansion: standalone LB, pruned matrix, 1-vs-N, K-vs-N

### CUDA Medium Priority
- [ ] Fix register-tiled kernel for banded DTW edge cases
- [ ] Multi-stream pipelining for very large N
- [ ] GPU early-abandon within DTW kernels

### Algorithms & Scale
- [ ] Condensed distance matrix (half memory for symmetric storage)
- [ ] Two-phase clustering for pre-categorized data (within-group + cross-group)
- [ ] Lazy loading (FileBackedDataSource, CachedDataSource)
- [ ] Algorithm auto-selection based on cost = N^2 * min(L, band) * ndim

### Bindings Phase 2
- [ ] MATLAB: Phase 2 parity (MIPSettings, CUDA, checkpointing, I/O)
- [ ] MATLAB: compile.m standalone build script (no CMake required)
- [ ] Python: PyPI first release (workflows ready, need trusted publisher setup)
- [ ] HIPify for AMD GPU support

### Technical Debt
- [ ] Clean up wavefront kernel dead code (unreachable preload branch)
- [ ] Unify kernel dispatch logic
