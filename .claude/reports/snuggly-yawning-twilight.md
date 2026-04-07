# Plan: End-to-End Stress Test + Bug Fix

## Context

We fixed 14 bugs (NaN sentinel, ignoreBOM, ADTW early-abandon, AROW guard, scores div-by-zero, CLI YAML, distByInd race) and need to validate the full pipeline before planning the next phase. Testing discovered a 15th bug: hierarchical clustering in CLI doesn't fill the distance matrix before calling `build_dendrogram()`.

## Scope

1. **Fix the hierarchical CLI bug** (one-liner)
2. **Write a bash stress test script** that exercises every CLI code path
3. **Run it**, collect results, report pass/fail

---

## Step 1: Fix hierarchical CLI bug

**File:** [dtwc_cl.cpp:555](dtwc/dtwc_cl.cpp#L555)

Add `prob.fillDistanceMatrix();` before `build_dendrogram()`:

```cpp
// Line 554, after linkage selection:
prob.fillDistanceMatrix();  // hierarchical requires full pairwise distances
auto dend = dtwc::algorithms::build_dendrogram(prob, hier_opts);
```

---

## Step 2: Write stress test script

**File:** `tests/integration/stress_test_cli.sh`

### Phase 1 — Smoke tests (dummy data, 25 series, k=3)

Run every method x variant combination (4 methods x 5 variants = 20 runs), plus metric and band variations (+2). Total: 22 CLI invocations.

| Method | Variants | Extra |
|--------|----------|-------|
| pam | standard, ddtw, wdtw, adtw, softdtw | +sqeucl, +band=5 |
| clara | standard, ddtw, wdtw, adtw, softdtw | seed=42 |
| kmedoids | standard, ddtw, wdtw, adtw, softdtw | |
| hierarchical | standard, ddtw, wdtw, adtw, softdtw | linkage=average |

**Validation per run:**
- Exit code == 0
- `test_labels.csv`: 25 rows, all labels in {0,1,2}
- `test_medoids.csv`: 3 rows
- `test_silhouettes.csv`: all values in [-1, 1]
- Cost is finite (may be negative for SoftDTW)

### Phase 2 — Correctness (Coffee UCR, 28 series, k=2)

- Run PAM with each variant on `Coffee_TRAIN.tsv` (--skip-cols 1)
- Extract ground truth from column 0 of the TSV
- Compute Rand Index against predicted labels (Python one-liner)
- Threshold: RI > 0.5 (better than random) = PASS, else WARN

### Phase 3 — Stress tests

| Test | What | Validation |
|------|------|------------|
| 3a Checkpoint | Run → save checkpoint → reload → re-run | Labels identical, cost identical |
| 3b Dist-matrix reload | Save dist matrix → reload with --dist-matrix | Labels identical |
| 3c Banded DTW | band=10, 50, 200, 286 on Coffee | All finite, band>=286 == full DTW |
| 3d SoftDTW sentinel | SoftDTW on Coffee | No NaN/inf in distance matrix, negative values OK |
| 3e k=1 | Single cluster | All labels=0, no silhouette file |
| 3f k=N | k=28 on Coffee | Cost=0, each point its own medoid |
| 3g ADTW banded | ADTW penalty=2.0 band=10 | Finite result, exercises early-abandon fallback |
| 3h WDTW g sweep | g=0.01, 0.05, 0.5 | All finite, monotonic or stable |

Total: ~42 CLI invocations, estimated runtime ~2 minutes.

### Script structure

```
stress_test_cli.sh
  ├── Config (paths, binary, datasets)
  ├── run_test()          — run CLI, capture stdout/stderr/exit
  ├── validate_outputs()  — check labels/medoids/silhouettes/cost
  ├── Phase 1 loop        — method x variant x metric x band
  ├── Phase 2 loop        — variants on Coffee + ARI
  ├── Phase 3 tests       — checkpoint, reload, banded, edge cases
  └── Summary report      — PASS/FAIL/TOTAL counts
```

---

## Step 3: Run and collect results

Execute the script, capture the summary. Any FAIL triggers investigation.

---

## Key files to modify

| File | Change |
|------|--------|
| `dtwc/dtwc_cl.cpp` | Add `prob.fillDistanceMatrix()` before hierarchical clustering |
| `tests/integration/stress_test_cli.sh` | New file — stress test script |

## Verification

1. Build Release: `cmake --build build --config Release`
2. Run: `bash tests/integration/stress_test_cli.sh`
3. Expected: all tests PASS (0 FAIL)
4. Run existing test suite: `ctest --test-dir build -C Release` — still 60/63

## Important details

- **SoftDTW produces negative distances** — validators must not assert >= 0
- **Batch-file names are 1-indexed strings** — labels CSV sorts alphabetically ("1","10","11",...), ARI comparison must join on name, not row order
- **Distance matrix CSV has no header** — raw NxN grid, NaN sentinel written as empty field
- **Dummy data needs --skip-rows 1 --skip-cols 1** (header + index column)
- **Coffee TSV needs --skip-cols 1** (class label column), delimiter auto-detected from .tsv extension
