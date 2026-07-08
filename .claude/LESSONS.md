# DTWC++ Lessons Learned

Critical knowledge to avoid repeating mistakes.

---

## Mathematical

- **DTW is NOT a metric.** Violates triangle inequality. MIP gap bounds don't formally apply.
- **DTW-AROW ≠ zero-cost DTW.** AROW constrains missing values to diagonal alignment.
- **LB_Keogh valid only for L1 and squared L2.** Not cosine or Huber.
- **WDTW/ADTW/DDTW/Soft-DTW are distinct recurrence/cost policies, not metric swaps.** They share the unified kernel family, but each still changes DTW semantics in a real way.

## Apple Silicon / macOS (benchmarked 2026-04-12, M2 Max 8P+4E)

- **Don't worry about E-cores vs P-cores on Apple Silicon.** Measured on `fillDistanceMatrix/100/1000/-1`: 1→8 threads gives 6.88× (all P-cores), 8→12 threads gives another 1.33× (E-cores help). Full 12-thread speedup is 9.14× (76% efficiency). E-cores are a net win with dynamic scheduling — do NOT cap to P-core count.
- **`OMP_PROC_BIND` / `OMP_PLACES` are no-ops on Darwin** with Homebrew libomp. Pinning sweep at T=12: default / close / spread all within 0.3% of each other (1606 / 1610 / 1609 ms). macOS QoS scheduler handles placement. Don't document them as tuning knobs — they do nothing.
- **`sysctlbyname("hw.perflevel0.logicalcpu")` is not worth calling** for thread-count tuning. `omp_get_max_threads()` gives the right answer. Skip the P-core auto-cap code path unless a future benchmark contradicts this.

## Python packaging / bindings

- **The installed `dtwcpp` is a non-editable wheel** (built to `Z:/dist/...whl`). `import dtwcpp` resolves to `.venv/.../site-packages/dtwcpp`, NOT `python/dtwcpp/`. Editing the repo's pure-Python files does **nothing** until you reinstall. To TEST pure-Python changes without a full C++ rebuild: copy the installed package dir to scratch (keeps the compiled `_dtwcpp_core.pyd` + DLLs), overlay the repo's `*.py` on top, and run `PYTHONPATH=<overlay> pytest`. For real use, `pip install -e .` (rebuilds the extension) or rebuild the wheel.
- **`dtwc_cl` names batch-row series `1..N` (1-based row index), ignoring any id column.** With a TSV of one series per row + `--skip-cols 0`, the `NAME_labels.csv` (`name,cluster`) uses names `1..N`. Its rows are **not guaranteed in input order** (the binary may lexically sort: `1,10,11,2,...`). ALWAYS map labels by name (`labels[i] = clusters[str(i+1)]`), never by row position. Verified against `build/bin/dtwc_cl.exe` 2026-06-30.
- **`./bin/dtwc_cl.exe` can be stale.** A top-level `bin/` binary may predate current flags (it rejected `--skip-cols`/`-k`). Prefer `build-*/bin/dtwc_cl` (current builds) — `find_dtwc_binary` does this.

## C++ Performance

- **Generalising can beat specialising.** Phase 1 unified the Standard/ADTW/WDTW/DDTW/ZeroCost-missing banded paths behind one `dtw_kernel_banded<T, Cost, Cell>`. The per-variant loops had accumulated divergent bookkeeping — WDTW had a ~50-line `if (low == 0)` block that didn't exist in Standard, etc. Unification produced **1.54× (dtwBanded), 1.77× (wdtwBanded), 2.83× (wdtwBanded_g)** speedups on 1000-length series. Register pressure, icache behaviour, and constant-folding under `-O2` all favoured the uniform loop. Template Cost/Cell policies are pass-by-value 8-24-byte structs kept in registers — zero runtime indirection. Counter to the assumption that specialised loops should always win.
- **DTW is latency-bound** (10-cycle recurrence). Only 3% of L1 bandwidth used. SIMD gives max ~1.29x.
- **Branchless scalar matches explicit SIMD for DTW.** Highway was tried and removed. scatter/gather in recurrence kills perf.
- **Don't replace integer arithmetic with lookup tables.** `i*(i+1)/2` = 1-2 cycles (imul). Table lookup = 4 cycles + cache pollution. Benchmarked 5% slower.
- **`std::min({a,b,c})` is catastrophically slow.** Creates temporaries. Use `std::min(a, std::min(b,c))` for 2.5-3x speedup.
- **Lambda capture-by-value creates stale parameter bugs.** Capture `[this]` and read at invocation time, not `[b=band]`.
- **NaN is the ONLY safe sentinel** for distance matrix uncomputed entries. Soft-DTW returns negatives — any fixed sentinel collides.
- **Mmap is safe as default.** Only 5% slower random access, 78x faster open, 48x faster CLARA views vs copy. OS page cache handles everything.
- **llfio > mio.** Active, has file locking, pre-allocation, production-proven. mio dormant since 2020.

## Data Formats & I/O (benchmarked 2026-04-08)

- **DTW dominates I/O by 10-100x.** Don't over-optimize I/O when compute kernel dominates. Just read Parquet directly.
- **Battery voltage compresses 21x** with Parquet Zstd (199.6 MB → 9.7 MB).
- **Arrow IPC = same speed as .dtws** after open (~0.4 us/series, pointer+offset). Open overhead ~4ms (flatbuffers).
- **Arrow IPC inflation not justified** for high-compression data. 100GB Parquet → 2TB Arrow IPC wastes disk. Keep both paths.
- **Parquet is NOT zero-copy** (requires decode+decompress). Arrow IPC IS zero-copy (mmap + pointer cast).
- **HDF5 mmap is a fragile hack** via `H5Dget_offset()`. Don't use.
- **Always use LargeListArray** (int64 offsets) in Arrow IPC. ListArray int32 overflows silently at >2B elements.
- **Zstd decompression vs NVMe:** Single access: uncompressed wins. Full scan with 8 cores and >5x compression: compressed wins.

## Float32 (benchmarked 2026-04-08)

- **Float32 DTW speed = identical to float64.** DTW is latency-bound — narrower data width doesn't help.
- **Float32 benefit is purely memory:** 2x more series in RAM and cache. Fewer page faults for large N.
- **Float32 accuracy is excellent for clustering.** Max relative DTW error: 2.74e-05 (0.003%). Negligible for medoid selection.
- **Default to float32.** Battery voltage (6.615V, 3 decimal places) needs only ~4 significant digits. float32 gives 7.

## C++ Implementation

- **NaN for missing data.** Use `quiet_NaN()`, check via `std::isnan()`. Safe because `-ffinite-math-only` is NOT set.
- **HiGHS: row-major. Gurobi: column-major.** Both: diagonal at `i*(Nb+1)`.
- **View-mode Data: guard `p_vec(i)` and `get_name(i)`.** These access empty vectors in view mode. Assert `!is_view()`.

## Cross-Language Bindings

- **MEX longjmp skips destructors.** Catch exception → exit scope → then call `mexErrMsgIdAndTxt`.
- **mexLock() prevents shutdown crashes.** `mexAtExit` must drain HandleManager before DLL teardown.
- **MATLAB + MSVC OpenMP: exit segfault in `-batch` mode.** Functionality works; segfault after output. Check output, not exit code.
- **nanobind over pybind11.** Stable ABI, 5-10x smaller binaries, native CUDA ndarray. GIL release for >10ms calls.

## HiGHS MIP Solver (IMPORTANT — workaround in place)

- **HiGHS <=1.14.0 `assert(ub_consistent)` fires on warm-start MIP.** The assertion is in `updatePrimalDualIntegral()` — a performance metric tracker, NOT solution correctness. `prev_lb/prev_ub/prev_gap` are documented "Only for checking/debugging" (line 2802). The P-D integral is never used to accept/reject incumbents. Presolve restart rebases bounds with offset arithmetic that introduces roundoff exceeding the 1e-12 tolerance. **Current workaround:** `target_compile_definitions(highs PRIVATE NDEBUG)` in `cmake/Dependencies.cmake` — too blunt (suppresses ALL HiGHS assertions). **Proper fix needed:** patch HiGHS to skip `check_prev_data` after restart, or relax the tolerance in this specific block. File upstream issue at github.com/ERGO-Code/HiGHS.
- **Verified by Codex (GPT-5.4, xhigh reasoning):** Not a solution-correctness bug. The workaround is legitimate short-term.

## Build System

- **CUDA multi-version on Windows:** Generate `Directory.Build.props` with `<CudaToolkitCustomDir>`.
- **MSVC flags leak into nvcc:** Use `$<$<COMPILE_LANGUAGE:C,CXX>:...>` generator expressions.
- **`find_package` vars don't propagate from CMake functions.** Check `TARGET X` instead of `X_FOUND`.
- **PyArrow bundles Arrow C++ libs** but DLL loading on Windows is fragile. Use vcpkg/conda for proper install.

## ARC SLURM Hardware

- **htc GPU compute capabilities (corrected from docs).** The ARC docs list CUDA toolkit version, not compute capability. Actual values: P100=6.0, V100=7.0, RTX8000/TitanRTX=7.5, A100=8.0, RTXA6000=8.6, L40S=8.9, H100/GH200=9.0.
- **Rome (htc-g019) and Broadwell (htc-g045-049) lack AVX-512.** Use `DTWC_ARCH_LEVEL=v3` for portable htc builds. All arc nodes support v4.
- **Grace Hopper (htc-g057) is AArch64.** Needs separate ARM build. CUDA kernel not yet ported.

## Arrow/Parquet

- **Never `static_pointer_cast<DoubleArray>` without checking value type.** Parquet list columns can store Float (32-bit) values. Casting to DoubleArray reinterprets float bits as double — silent data corruption. Always check `values->type_id()` first.
- **Parquet row-group metadata is free.** `num_rows`, `total_uncompressed_size` per row group available without reading data. Use for RAM budgeting.

## Refactoring Process

- **Cross-validation is the gate for policy migrations.** Before swapping a dispatch from impl A to impl B (e.g. `dtwAROW_banded` → `dtw_kernel_banded<T, SpanAROWL1Cost<T>, AROWCell>`), write a test that runs BOTH on representative inputs (no-NaN, interior NaN, leading/trailing NaN, all-NaN × bands {1..4}) and asserts bit-for-bit agreement within 1e-10. If the test passes, migrate; if it fails, diagnose BEFORE touching production dispatch. Applied in Phase 3.2 (AROW) and 3.3 (Soft-DTW) — both landed with zero regression.
- **Silent-dispatch bugs hide in asymmetries.** `Problem::dtw_function_f32()` was hardwired to Standard DTW regardless of `variant_params.variant` / `missing_strategy` — nobody noticed because every existing test exercised only the f64 path. Found when unifying dispatch via a templated resolver. **Pattern: dual-type APIs (f32/f64) need parity tests, not just one-side tests.** Same failure mode showed up twice in this codebase (also `dtw_runtime()` silently ignoring variant pre-Phase 1).
- **Cell/Cost policy contracts are forward-extensible.** Adding `seed(cost, i, j)` to the Cell policy to support AROW's `C(0,0) = 0 on NaN` semantics did NOT require touching existing cells — `StandardCell::seed` defaulted to `return cost`, which is identical to the pre-refactor `col[0] = cost(0, 0)` assignment. Pattern: new contract methods with sensible defaults are additive, not breaking.

## Audit / Testing

- **Always rerun ctest failures serially after the first parallel pass.** A clean handoff on another platform is not evidence of a green local tree. On Windows Release (2026-04-13), `ctest -j 4 -C Release` failed with `0xc0000409`; serial rerun showed `test_fast_pam_adversarial` was a deterministic crash while `unit_test_clustering_algorithms` was a parallel-only failure mode. Audit skills must distinguish "real blocker" from "flake".
- **A test name must match the algorithm path it actually exercises.** `tests/unit/adversarial/test_fast_pam_adversarial.cpp` sounds like FastPAM coverage, but its helper sets `prob.method = Method::Kmedoids` and calls the legacy Lloyd path. That creates false confidence. For algorithm migrations, mislabeled tests are worse than missing tests because they silently certify the wrong implementation.
- **Tests must pin the LIVE code path — name the public entry point in a comment.** Phase 0 task 0.6 "fixed" multivariate L2 in `core::dispatch_mv_metric`, a dead duplicate with zero call sites; the live `detail::dispatch_mv_metric` (warping.hpp) kept aliasing L2→L1, the new test validated the dead function, and the CHANGELOG claim was false. Caught only by adversarial review tracing the real dispatch chain (`dtwBanded_mv` → warping.hpp). Rules: (1) before fixing a dispatcher, grep call sites and delete dead duplicates — two dispatchers for one concept is itself the bug; (2) every regression test states in a comment which public entry point it exercises. (Fixed in Phase 0 remediation R1, commit ffb7a8d.)

## LR-core Solver (Phase 4)

- **Killed 2023 ideas — keep them killed.** Every 2023 attempt (removed in `f7064b3`) solved the p-median LP in x-space explicitly: dense/sparse tableau simplex, Gomory cuts, OSQP/ADMM on the N²-variable relaxation. All failed on scale. The right structure is to DUALIZE the assignment equalities and bound matrix-free: the Cardinality+Linking substructure is TU for all N (Ghouila-Houri), so the Lagrangian dual equals the LP bound (Geoffrion) WITHOUT forming the N²-column LP. Do not re-open x-space LP solving; if tempted, re-read UNIMODULAR.md §8.
- **Falsified polytope claims (registered bands, scipy/HiGHS vertex LPs + brute-force IP oracle).** p-median constraint matrix is TU only for N≤2 (6×6 3-cycle has det −2). Half-integrality of fractional vertices is FALSE (values 1/4, 1/3, 3/4 occur). "LP is 80–90% integral" is NOT a polytope property — it is data-regime-dependent (clustered non-metric D: 250/250 integral; uniform D at N=20: 26% fractional, gap ≤13.7%). The user's "almost unimodular" observation = his data lives in the integral regime, not a theorem.
- **Two different gaps need two different tools — never conflate.** The LR *primal* (repair) is exactly optimal at every tested N; only the *dual certificate* is the problem. (a) The DUAL/LP gap is continuous — the subgradient stalls at the non-smooth optimum (λ=2.0 oscillates on ~25% of separated instances; λ=1.0 damped + CFM deflection converges; even 12000 iters barely move a stalled N=800). The right tool is the Kelley cutting-plane (finite convergence, ~15 major iters flat in N). (b) The INTEGRALITY gap is discrete — no dual method closes it; only branching does. `lagrangian_root_exact` uses Kelley for (a) and y-branching B&B for (b). On clustered (real-world) data there is no integrality gap ⇒ the root certifies ⇒ 0 B&B nodes.
- **Prune-if-bound-exceeds-incumbent tests must clear the incumbent by a tolerance, never `>` exactly.** Reduced-cost fixing on a CERTIFIED instance (gap≈0, LB≈UB) wrongly eliminated an optimal medoid whose score merely TIED ρ_(k): independently-summed LB/UB/ρ round to a few-ULP difference that crosses an exact `>`. A legit alternative optimum got fixed out. Fix: only fix when the conditional bound clears UB by `tol=1e-9·(1+max(|LB|,|UB|))` — safe (real eliminations have O(scale) margin ≫ tol), caught only because the N≤14 correctness test checks every fixed facility against the brute-force optimum. General rule: any "kill if bound > best" comparison near a zero gap needs a magnitude-scaled slack.
- **PDLP (HiGHS first-order LP) is an ARBITER, not a re-opening of the killed x-space LP (Task 4.5).** The 2023 kill was hand-rolled x-space solvers (custom OSQP/ADMM/OSLP tableau) that failed on scale. HiGHS PDLP is different on two axes: (1) it's a maintained library, not custom code (honours the prefer-libraries rule); (2) it's used strictly to CROSS-CHECK the Lagrangian bound, never as the production bound engine — the matrix-free Lagrangian still dominates on the TU-structured p-median. The arbiter has real teeth: PDLP (explicit LP, first-order primal-dual) and Kelley (Lagrangian dual, matrix-free) reach the same LP optimum by different mathematics — measured agreement **7.14e-09** across 24 instances (registered band 1e-4), validating the clever bound where the brute-force IP oracle (N≤14) can't reach. Do NOT promote PDLP to a production `Method`: it's LP-only (no integer certificate) and dominated by LR-core for the bound.
- **HiGHS PDLP GPU is a build flag, not a runtime switch.** v1.15.1 vendors real cuPDLP CUDA kernels (`pdlp/cupdlp/cuda/*.cu`, `pdlp/hipdlp/pdhg.cu`) but gates them behind the HiGHS CMake option `CUPDLP_GPU` (default OFF) — a stock build gives CPU PDLP with the identical bound. To warn honestly on `use_gpu=true` without a runtime GPU query, drive the warning off OUR OWN compile flag (`DTWC_HIGHS_GPU`, set only when we forward `CUPDLP_GPU=ON`): requested-GPU-on-CPU-build → stderr warning + `gpu_used=false`, never a silent GPU claim. GPU verified live on the RTX 4000 Ada — HiGHS forces itself SHARED on Windows for CUPDLP_GPU (`highs.dll` + `cudalin.dll` land beside the exe; add the CUDA `bin` to PATH at run time for cudart/cublas/cusparse).
- **A PUBLIC compile-def on an OBJECT lib does NOT reach test TUs — use a runtime capability query.** `target_compile_definitions(mip-solvers PUBLIC DTWC_HIGHS_GPU)` reaches code compiled INTO mip-solvers (so `pdlp_lp.cpp` saw it, set `gpu_used=true`) but NOT `test_pdlp_lp.cpp` (the define stops at the dtwc++ link boundary) — an `#ifdef DTWC_HIGHS_GPU` in the test compiled the wrong branch and failed while the feature worked. This is exactly why the mip tests use runtime try/catch, not `#ifdef DTWC_ENABLE_HIGHS`. Fix: the library exposes `pdlp_gpu_available()` (compiled where the define lives) and the test branches on that. General rule: a test cannot see a dependency's private/object-scoped defines; expose capability at runtime.
- **HiGHS `CUPDLP_GPU` is a COMPILE-TIME device switch — a per-call `use_gpu` flag cannot toggle it, and reporting off it lies.** The bench first ran two columns (`use_gpu=false` vs `true`) expecting a CPU-vs-GPU comparison inside one build. On the GPU build both columns were **digit-identical in iteration count** (880/880 … 6840/6840) and time: `solver="pdlp"` always runs on the GPU once HiGHS is built with `CUPDLP_GPU=ON`, there is no per-solve CPU path. The old code set `gpu_used=true` only when the caller requested the GPU, so a `use_gpu=false` solve on a GPU build ran on the GPU yet reported `gpu_used=false` — a false report (CLAUDE.md §1). Fix: `gpu_used = pdlp_gpu_available() && variant=="pdlp"` (build + variant, not the request); `use_gpu` only drives the CPU-build warning. Cross-device comparison must therefore be **cross-build** (same bench on a CPU-only and a `DTWC_HIGHS_GPU` build), not two calls in one process. Diagnostic tell that two "different" configs are secretly identical: iteration counts match to the digit.
- **Bench verdict — PDLP is a cross-validation ARBITER, never a production p-median solver.** On clustered p-median (`.claude/baselines/2026-07-08-pdlp-bench.md`): matrix-free Kelley stays near-flat (~15 major iters, streams D once each) while PDLP forms/solves the 3N² LP, so `pdlp/kelley` wall-time grows to **945× (CPU-PDLP) / 126× (GPU-PDLP)** at N=400. GPU only helps *PDLP-vs-PDLP*: a ~255 ms launch/transfer floor makes GPU-PDLP 67× slower than CPU-PDLP at N=20, crossing over ~N≈150 to 7.4× faster by N=400 — never within two orders of Kelley. Registered GPU prediction ("GPU-PDLP ≥ CPU-PDLP at all N≤400") was PARTIALLY FALSIFIED by the crossover; recorded as a deliverable. Structural takeaway: on a totally-unimodular problem a bespoke matrix-free Lagrangian beats a general first-order LP on any device — keep PDLP for the arbiter role only.
- **Two Windows-CUDA toolchain traps (both cost real time; both have one-line fixes).** (1) Git Bash / MSYS mangles a leading `/c` argument into `C:\`, so `cmd.exe /c "batch"` silently opens an INTERACTIVE cmd (banner + prompt) and exits doing nothing — no error, no output. Use `MSYS_NO_PATHCONV=1 cmd.exe /c …` or `cmd.exe //c …`. (2) CUDA 13.0's `nvcc` host_config REJECTS MSVC newer than VS 2022 ("unsupported Microsoft Visual Studio version! Only 2019–2022"), e.g. the VS-18 / MSVC 14.50 on this box — pass `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` (and the explicit `-DCMAKE_CUDA_COMPILER=<nvcc>` since vcvars does not put nvcc on PATH). Both are captured in the known-good cache at `build/cuda-verify/CMakeCache.txt` — read it before fighting a fresh CUDA build.

## Research Process

- **Always verify citations.** Author names, venues, volume numbers can be hallucinated.
