---
description: "Diagnose and fix DTWC++ issues: build errors, runtime crashes, wrong results, performance. Read-only diagnosis."
allowed-tools:
  - Read
  - Bash
  - Glob
  - Grep
---

# DTWC++ Troubleshoot

Diagnose DTWC++ problems. `$ARGUMENTS` contains the user's error message or symptom description.

## Step 1: System diagnostic

Run if Python package is available:
```bash
python3 -c "import dtwcpp; print(f'Version: {dtwcpp.__version__}'); print(f'OpenMP: {dtwcpp.OPENMP_AVAILABLE} (threads={dtwcpp.openmp_max_threads()})'); print(f'GPU:    {dtwcpp.gpu_info()}')"
```

Check binary if CLI is used:
```bash
which dtwc_cl && dtwc_cl --version
```

## Step 2: Route to diagnostic

### Install / Import errors

**`ModuleNotFoundError: No module named 'dtwcpp'`**
- Install: `uv pip install .` from repo root
- Or wheel: `uv pip install dtwcpp`
- Check: `python3 -c "import sys; print(sys.path)"` — is the venv active?

**`ImportError: dlopen … libomp.dylib … not found` (macOS)**
- `brew install libomp && brew link --force libomp`
- Rebuild if needed

### Build errors

**`Undefined symbol: std::__1::__hash_memory`** (macOS)
- Homebrew LLVM clash with libc++. Use preset which forces Apple Clang:
  `cmake --preset clang-macos` (expects `/usr/bin/clang++`)

**Gurobi not found**
- Set env var: `export GUROBI_HOME=/Library/gurobi1301/macos_universal2` (macOS) or equivalent
- Or build without it: `-DDTWC_ENABLE_GUROBI=OFF` (the default; HiGHS solves the MIP)

**OpenMP not found**
- Mac: `brew install libomp && brew link --force libomp`
- Ubuntu: `apt install libomp-dev`
- Pass explicit hints if auto-detect fails (see README macOS section)

### Runtime errors

**`std::bad_alloc` / OOM**
- Dataset too large for in-memory distance matrix.
- Switch to CLARA: `--method clara` — uses subsamples
- Lower memory: `--dtype float32` (halves the series' RAM; the distance matrix stays float64)
- Memory-map: `--mmap-threshold 0` forces mmap distance matrix
- Parquet input: `--ram-limit 8G` caps the series in RAM, streaming one file of a series per row (a list column, or several float columns) through CLARA

**`series '…' (index i)[j] is NaN`**
- Check data: `python3 -c "import numpy as np, pandas as pd; d=pd.read_csv('data.csv').values; print('NaN?', np.isnan(d).any())"`
- If expected: use `--missing-strategy arow` (AROW-DTW) or `zero_cost`
- If unexpected: impute or drop NaN rows upstream

**`device='gpu' requested but …`** (`DeviceError`)
- `nvidia-smi` to confirm GPU
- Rebuild: `-DDTWC_ENABLE_CUDA=ON`
- macOS: no CUDA (Apple dropped NVIDIA drivers in Mojave); the GPU there is Metal, `-DDTWC_ENABLE_METAL=ON` (the default on Apple)

**`Clustering collapses all into one cluster`**
- Try different `k`; if silhouette < 0.25 on all k, the variant may be wrong for the data
- Try different variant (DDTW if shapes matter; WDTW if time offsets matter)
- z-normalize each series: `[dtwcpp.z_normalize(s) for s in series]`
- Different seed or more restarts: `--seed 7`, `--n-init 5`

### Performance issues

**Slow (not using all cores)**
- Check `openmp_max_threads()` output
- Set: `export OMP_NUM_THREADS=$(nproc)` (Linux) or `$(sysctl -n hw.ncpu)` (Mac)
- Verify build flag `DTWC_HAS_OPENMP` in compile output

**Slow distance matrix on large N**
- Use GPU: `--device cuda` (if available)
- Use bands: `--band 10` (reduces O(nm) to O(min(n,m)*band))
- Parquet + mmap for fast load

## Step 3: Dynamic investigation

Use when the error message doesn't match known patterns:

1. `Grep` the exact error string in source to find the origin:
   ```
   Grep pattern="exact error string here" path="."
   ```

2. Read nearby code to understand the context.

3. Check the CMake configure log: `build/CMakeFiles/CMakeConfigureLog.yaml`.

4. Reproduce with minimal input (2-3 series) to isolate.

## Step 4: Format the answer

Respond with:
1. **Root cause** (one sentence)
2. **Fix** (exact command or code change)
3. **Verify** (how to confirm the fix works)

## Related

- `/help` — general library reference
- `.claude/LESSONS.md` — accumulated gotchas from development
