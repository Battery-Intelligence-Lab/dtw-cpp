# Session: Mmap Distance Matrix + Auto-Select Implementation (2026-04-08)

## What was done

### Steps 1+2: Count-Before-Load + Auto Method Selection (commit 75c760d)
- `DataLoader::count()` — count series without loading data
- `--method auto` (new CLI default) — pam for N<=5000, clara for N>5000
- CLARA sample_size auto-scaling for N>50K
- 5 new DataLoader tests, all passing

### Step 3: Memory-Mapped Distance Matrix (commits 6457644, 9a6cfe0)
- llfio integrated via CPM (`DTWC_ENABLE_MMAP=ON` by default)
- `MmapDistanceMatrix` class in `dtwc/core/mmap_distance_matrix.hpp`
- Same API as `DenseDistanceMatrix` (get/set/is_computed/size/max/count_computed/all_computed/raw)
- 32-byte binary header (magic DTWM, version, endian, CRC32, N)
- `MmapDistanceMatrix::open()` for warmstart
- 18 test cases, all passing
- Extracted `tri_index`/`packed_size` as free functions in `dtwc::core` namespace

### Step 4: Variant Integration (commits d8dea97, 535bdf4)
- `Problem::distMat_t` = `std::variant<DenseDistanceMatrix, MmapDistanceMatrix>` when mmap enabled
- `visit_distmat()` helper dispatches; compiles to direct call when mmap disabled (zero overhead)
- `dense_distance_matrix()` accessor for Dense-specific paths (bindings, pruned fill, GPU inject)
- `use_mmap_distance_matrix(path)` to force mmap backend
- All external access sites updated (14 files modified)
- 3 integration tests (dense default, forced mmap, warmstart), all passing

### Step 5: Binary Checkpoint + CLI (commits 1134695, 60e0a3e, 55844ef)
- `save_binary_checkpoint` / `load_binary_checkpoint` for clustering state
- Binary format: 32-byte header (DCKP magic) + medoid_indices + labels
- `--restart` flag loads checkpoint
- `--mmap-threshold N` controls mmap activation (0=always, default=50000)
- Fixed threshold logic bug (0 > 0 was always false)
- 3 checkpoint tests, all passing

### Benchmarks
- `visit_distmat` overhead: ~0.000025% of total time (branch instruction vs seconds of DTW)
- Warmstart: first run 3.99s, second run 0.11s = **36x speedup** from mmap cache

## Test results
- 64/66 pass, 2 CUDA skipped, 2 pre-existing SEGFAULTs (benders, mip)
- New tests: 24 (18 mmap + 3 variant + 3 checkpoint)

## Commits (9 on branch Claude)
```
55844ef docs: update CHANGELOG, fix --mmap-threshold 0 logic
60e0a3e feat: add binary checkpoint and --restart/--mmap-threshold CLI flags
1134695 test(RED): add binary checkpoint tests
535bdf4 feat: integrate MmapDistanceMatrix into Problem via std::variant
d8dea97 test(RED): add variant distance matrix integration tests
9a6cfe0 feat: implement MmapDistanceMatrix with llfio backend
6457644 Add llfio as optional dependency for memory-mapped distance matrices
75c760d Add DataLoader::count() and --method auto (default) for CLI
c4fb30c Add lessons learned, mmap/auto-select implementation plan, session handoff
```

## What to do next

### Remaining from the original plan
1. **Step 6: Streaming CLARA** — deferred, needs Steps 1-5 proven (they are now)
2. **Stale cache detection** — store hash of input filenames in mmap header
3. **Checkpoint warm-start in algorithms** — `--restart` loads checkpoint but doesn't yet pass it as warm-start to FastPAM/CLARA

### Open bugs (still open)
1. `unit_test_benders` SEGFAULT — pre-existing
2. `unit_test_mip` SEGFAULT — pre-existing
3. hierarchical + SoftDTW crashes
4. set_if_unset in YAML overrides CLI values
5. MV banded DTW silently ignores band

### llfio note
- llfio emits a `#pragma message` warning about `ntkernel_category()` header-only form being unreliable. This is cosmetic — we don't use error category comparison. To suppress: switch to static library mode (`llfio_sl`) or add `NTKERNEL_ERROR_CATEGORY_INLINE=0`.

## Key decisions
- `std::visit` overhead is negligible (benchmarked) — keeping simpler per-call dispatch
- llfio over custom mmap wrapper (user preference: always use libraries)
- 32-byte header kept (version + CRC worth the 16 extra bytes for debuggability)
- Binary checkpoint is separate from CSV checkpoint (both coexist)
