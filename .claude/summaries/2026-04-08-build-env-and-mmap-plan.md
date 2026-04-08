# Session: Build Environment Audit + Mmap/Auto-Select Plan (2026-04-08)

## What was done

### 1. Build environment audit (Windows 11, this machine)

Identified and fixed build issues:
- **Compiler**: Clang 21.1.8 (x86_64-pc-windows-msvc target)
- **OpenMP 5.1**: Present (LLVM libomp) but CMake can't auto-detect with Clang/Windows
- **CUDA**: nvcc 13.0 installed, nvidia-smi needs admin
- **Gurobi 13.01**: Installed at C:\gurobi1301\win64, finder was broken for Clang

### 2. CMake robustness fixes (committed: 4255c16)

All in one commit on branch `Claude`:

- **OpenMP auto-detection** for Clang/Windows: probes LLVM install dir for libomp (`dtwc/CMakeLists.txt`)
- **Gurobi finder fix** for Clang/Windows: scans C:/gurobi*/win64, uses import lib from lib/ not DLL from bin/ (`cmake/FindGUROBI.cmake`)
- **CMake WARNING** when OpenMP/Gurobi requested but not found (not silent single-core)
- **Runtime warning** on first parallel call if OpenMP absent (`dtwc/parallelisation.hpp`)
- **DTWC_HAS_OPENMP** compile definition
- **Configuration summary table** at end of cmake configure (`CMakeLists.txt`)
- **Test WORKING_DIRECTORY** fix: repo root not build/bin — fixes 5 path-dependent tests (`cmake/Coverage.cmake`)
- **SKIP_RETURN_CODE 4** so CUDA skip isn't reported as failure
- **CMakePresets.json** for cross-platform builds (clang-win, clang-win-debug, msvc, gcc-linux, clang-macos)

**Result**: 63/63 tests pass, 2 CUDA correctly skipped. Gurobi + HiGHS + OpenMP all detected.

### 3. Memory optimization benchmarks (reverted, not committed)

Benchmarked tri_index precomputed row offsets + schedule(guided):
- `random_get` 500K lookups: 2.61ms baseline → 2.75ms optimized (**5% slower**)
- `fillDistanceMatrix`: within noise (±10%)
- **Verdict**: Not worth it. Modern CPUs do `i*(i+1)/2` in 1-2 cycles; table lookup adds L1 load overhead.
- Lesson written to `.claude/LESSONS.md`

### 4. Mmap + Auto-Select plan (reviewed, not implemented)

Detailed 6-step plan at `.claude/reports/zippy-crafting-piglet.md`:

1. **Count-Before-Load**: `DataLoader::count()` — count files/lines without loading
2. **Auto Method Selection**: `--method auto` (default). N<=5K → pam, N>5K → clara
3. **Mmap Distance Matrix**: Separate `MmapDistanceMatrix` class, 32-byte header, NaN sentinel
4. **Integrate via std::variant**: Problem holds variant, resolve once at algorithm entry (not per-call std::visit)
5. **Checkpoint + Restart**: Save medoids + labels + cost alongside mmap cache. Resolves GitHub #22.
6. **Streaming CLARA**: Deferred — load only subsamples

Plan went through **4 adversarial reviews** (3 Opus + 1 Codex). Key findings incorporated:
- Separate class (not extend DenseDistanceMatrix) — avoids dangling pointer P0 bug
- 32-byte header (not 24) — alignment for ARM
- CRC header-only, not data
- Stale cache detection via input file hash
- Don't allocate mmap when auto-selecting CLARA (CLARA doesn't need full matrix)
- Resolve variant once at algorithm entry, not per distByInd call

## Uncommitted changes

| File | Status | Description |
|------|--------|-------------|
| `.claude/LESSONS.md` | Modified | Added: NaN sentinel, tri_index benchmark, llfio vs mio |
| `.gitignore` | Modified | User edit (opened in IDE) |
| `.claude/reports/zippy-crafting-piglet.md` | New | Full implementation plan |

**Action needed**: Commit these, then start implementation in next session.

## Open decisions

### llfio vs mio vs custom wrapper
- **llfio**: has file locking + pre-allocation, active. But heavy CMake, complex.
- **mio**: clean API, header-only. But dormant, no file locking.
- **custom**: ~80 lines, zero deps. But maintenance.
- **Recommendation**: Try llfio first. If CMake is painful, fall back to mio + manual locking.

### Checkpoint format
- Keep existing CSV checkpoint for small N / human-readable
- Binary checkpoint (mmap cache + medoid state) for large N / HPC
- `--restart` uses binary, old `--checkpoint` uses CSV

## What to do next

### Immediate (next session)
1. Start with Steps 1+2 (count-before-load + auto method selection) — no new dependencies
2. Then Step 3 (mmap distance matrix) — evaluate llfio CMake integration
3. Add tests for each step

### Implementation order
```
Step 1 (count)   ─── independent ──┐
Step 3 (mmap)    ─── independent ──┤
                                   ├── Step 2 (auto-select, needs Step 1)
                                   ├── Step 4 (integrate mmap, needs Step 3)
                                   ├── Step 5 (checkpoint, needs Step 3+4, resolves #22)
                                   └── Step 6 (streaming CLARA, needs all above)
```

### Open bugs (from previous sessions, still open)
1. **hierarchical + SoftDTW crashes** — build_dendrogram or distance fill with SoftDTW on large series (5000 pts)
2. **set_if_unset in YAML** — unconditionally overrides CLI values
3. **MV banded DTW silently ignores band** for Missing/WDTW/ADTW variants

## Key architectural decisions this session
- NaN is the ONLY safe sentinel (Soft-DTW returns negative distances, -1 was tried and broke)
- tri_index arithmetic is faster than lookup tables (benchmarked, lesson learned)
- llfio preferred over mio (active maintenance, has file locking + pre-allocation)
- std::variant for distance matrix type, but resolve once at algorithm entry (not per-call)
- CLARA auto-select should NOT trigger mmap allocation (CLARA skips full matrix)

## Build instructions (this machine)
```bash
cmake --preset clang-win    # or: cmake -B build -G Ninja ... (see CMakePresets.json)
cmake --build build --parallel 8
ctest --test-dir build -C Release -j8   # 63/63 pass, 2 CUDA skip
```
