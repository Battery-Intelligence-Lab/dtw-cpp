# DTWC++ 2.0 — design review (2026-09-27, HEAD cd5d449; approved by Volkan 2026-09-28)

Answers to §7 (2026-09-28): Q1 approve, agents commit locally; Q2 open — "how is mio performance?" (measured in phase A);
Q3 "Are they equivalent methods? If they have trade-offs they both can stay, but first measure if they are equal."
(measured in phase A); Q4 2.0, the last performance item.

How it was made: 8 slice audits, each attacked by an adversarial verifier → a Fable-max panel (cut / keep /
interface) → a Fable-max chair → three Fable-max advisors (owner intent, execution, user walkthrough). The main
session opened the lines behind every headline claim (`2026-09-27-audit/main_session_checks.md`). Finding ids
(`kernels-01`, `data-io-05`, …) resolve in `2026-09-27-audit/digest.md`; step detail is in `chair.md` §4 and
`advisor_exec.md` §2. The audit folder is campaign evidence and is deleted when phase G closes.

## 1. Verdict

1. The library becomes its own skeleton: `device → load → cluster → Result` in C++, Python, MATLAB and the CLI over
   one flat `Config`; one `Problem` session; one distance resolver, validated once; one matrix fill per device; one
   binary matrix file; one names table; one `index_t`.
2. What goes is everything built beside that skeleton: second enums, dispatchers, readers and writers; diagnostic
   modes; research solvers; drift detection; count guards; unreachable backends; aliases nobody was given.
3. Size [inferred from the per-finding estimates; today measured with `git ls-files | wc -l`]: library + bindings
   45.6k → ~26k; tests 75.8k → ~38k; scripts + `.claude/` + docs + CHANGELOG 65.6k → ~16k. About −108k of 214k.
4. Users notice: a faster default CPU fill; Soft-DTW no longer holds ~0.5 GB per thread; Python reads Parquet; MSM
   and TWE in every language; a k-sweep fills the matrix once; `hpc` failures say what to fix; the v1 CLI flags work
   again. About 30 silent wrong answers are fixed on the way (§6).
5. Nothing that works today and that the charter asked for is lost. Three things the code never delivered are
   named as roadmap items instead of implied: GPU assignment for CLARA (Q4), a multi-node fill, checkpoint of a
   streaming run.

## 2. Decisions

| Decision | Now | Why |
|---|---|---|
| Integer type | `using index_t = std::int64_t;` in `base/settings.hpp`, no macro. Counts of series / clusters / rows, labels, medoids, `dist_by_ind` indices are `index_t`; tuning values (`band`, `max_iter`, `n_init`, `n_samples`) stay `int`; seed `uint64_t`; products are `size_t`/`int64_t` by type. **No count guard anywhere**: `fast_pam.cpp:60`, `fast_clara.cpp:71,201,486`, `one_batch_pam.cpp:56`, `solution_transaction.cpp:93`, `run_openmp`'s INT_MAX throw, Parquet saturating/INT_MAX guards, the Lloyd seed throw, CLARA's "int-indexed result limit" and CUDA's N ≤ 65,536 refusal (replaced by int64 chunking) all go | owner 2026-09-27; every target is 64-bit, numpy is int64 |
| 32-bit foreign APIs | `mip/index_guard.hpp` deleted; one inline `if (N·N > INT_MAX) throw SolverError` in `mip_Highs.cpp` and one in `mip_Gurobi.cpp` — the only two left | HiGHS and Gurobi take `int` counts; a wrapped count builds a silently wrong model, and 3N² nonzeros pass 2³¹ at N ≈ 27k, which a 2 TB node can hold. Two lines, no header |
| v1.0.0 fields `clusters_ind` / `centroids_ind` | `std::vector<index_t>`, plus a `[[deprecated]]` converting `set_clusters(std::vector<int>)` for one release | the one v1 C++ compile break (assignment from `vector<int>`); follows from the row above |
| Labels per language | C++ `int64_t`, Python `np.int64`, MATLAB **double**, 1-based | MATLAB's own `kmedoids` returns double; int64 arithmetic in MATLAB surprises users; exact to 2⁵³ |
| Compatibility | Frozen = what v1.0.0 shipped (C++ `Problem`/`DataLoader`/`Data`/`Method`/`Solver`/`init::*`/`scores::silhouette`/`dtwBanded,dtwFull,dtwFull_L`/`Range`/`Index`, four root headers, `Problem_IO` filenames, the CLI). Everything 2.0-born is pre-tag: no shims. No wire format is frozen before the tag. `docs/api-contract-2.0.md` retires; a break needs one dated `DECISIONS.md` line | 2.0 is unreleased |
| v1 CLI spellings | restored as hidden warn-once aliases from one table (~20 lines): `--Nc`, `--probName`, `--in/--out`, `--skipRows`, `--skipCols`, `--maxIter/--iter`, `--repeat/--Nrep…`, `--mip_solver`, `--bandwidth`, `--distMat`; `--Nc 3..5` → `InvalidInput` "one k per run" | all gone at HEAD (verified) while `migration.md:115` says they work; the CLI is how v1 was used |
| v1 Python camelCase names | no forwards; `migration.md` records the break | `dtwcpp` was never on PyPI (verified 404); v1 scripts differ in shape anyway |
| `hpc` | C++ `Device` = {CPU, GPU}. `hpc` / `hpc:gpu` live in Python and `slurm_remote.sh`; C++ and MATLAB `device("hpc")` throw `DeviceError` at the call, naming them. Transport = `job.toml` from `to_config_text`; the job runs `dtwc_cl --config job.toml` | one grammar; the 20-positional transport and its three validators go; D-10 kept |
| Method | `Method` gains `Auto, PAM, CLARA, OneBatch, Hierarchical` (nine values; v1 values keep their names); `ClusterMethod` and `problem_route` go; `Problem::cluster()` dispatches all nine; `run()` = apply Config, load, `cluster()`, write | two enums for one concept |
| Device vocabulary | `DistanceMatrixStrategy`, `CUDASettings`, `CUDAPrecision`, `MetalPrecision` → `set_device(Device, index)` + `set_gpu_precision(GpuPrecision{Auto, FP32, FP64})`; the fill resolves CUDA, else Metal, else `DeviceError`; the matrix fingerprint hashes the resolved backend and precision | the strategy enum restates the device |
| `auto` method | cpu: `pam` for N ≤ 5000, else `clara`. gpu: `pam`; a matrix that cannot be held is a typed error naming `clara` on `cpu` | GPU PAM at N = 5k–100k is the GPU's use; refusing it (the UI advisor's rule) would cost more than it saves |
| PAM swap | FasterPAM only: `PAMVariant`, `FastPAM1Naive`, the public `fast_pam_swap`, `medoid_utils` go; `pam` results change once (pre-registered) | baseline `2026-07-08-faster-pam-bench.md`: FastPAM1 stops unconverged at the cap for k = 200 (FasterPAM converges in one sweep, better objective) and is slower at k = 100 (52.5 vs 45.1 ms); a runtime switch between two swap algorithms is the machinery the owner rejects |
| Exact solvers | keep compact MIP (HiGHS, Gurobi, FasterPAM warm start) and LR-core with reduced-cost fixing; delete Benders, PDLP, CLARANS, `solution_transaction`/`warm_start` (→ one ~25-line `decode_assignment` + `Problem::set_result`); `Method::MIP` uses the selected solver at every N | LR-core is the large-N exact route (charter item 9); the rest is dominated or diagnostic |
| Pruned fill | deleted with `LowerBoundStrategy` and every bound except `compute_envelopes`, `lb_keogh`, `lb_keogh_symmetric` (TADPole); Auto = brute force | an exact matrix cannot skip a pair: the pruned fill re-runs every abandoned pair in full (`pruned_distance_matrix.cpp:290-294`), yet Auto picks it (`Problem.cpp:1189-1193`) |
| Persistence | one binary matrix file `.dtwm`: {magic, version, N, SHA-256 fingerprint} + packed doubles, NaN = not computed. The mmap cache **is** the checkpoint: `--checkpoint dir` maps `<dir>/<name>.dtwm`, the fill writes into it, one `sync()` per row block; reopening with the same fingerprint resumes; a mismatch is `InvalidInput`, a malformed file `IOError`, an absent file starts fresh. Gone: `.dtws` + `MmapDataStore` + CRC32, `StoragePolicy` auto-spill + `system_memory`, the binary result checkpoint and `--resume/--restart`, the CSV generation checkpoints, `checkpoint-interval`, the v3 digests and lease. CSV stays as interchange | today a mismatched checkpoint returns `false`, recomputes and overwrites (`checkpoint.cpp:626-667`); the auto-spill copies data already in RAM (`DataLoader.hpp:200-222`); owner: no custom binary data formats |
| mmap library | **Q2** — recommended: mio (MIT, one header) replaces llfio | after the cut the need is create/map/sync/open; every wheel, MEX and release archive builds with llfio OFF today (verified), so pip and MATLAB users have no mapped matrix and no checkpoint |
| Large data, honestly | N ≲ 50k: matrix in RAM; ~50k–300k: mapped matrix, GPU-filled in int64 chunks; series beyond RAM: list-per-row Parquet + `ram_limit`, streamed by CLARA / OneBatch on the CPU | charter item 3 |
| Backends | delete MPI (unreachable: only `bench_mpi_dtw.cpp` and `dtwc.hpp:60`), the CUDA/Metal 1-vs-N and K-vs-N kernels (no callers), GPU LB_Keogh, `KernelOverride` (after one A/B keeps the kernel variants within 5 %); one `fill()` translation unit; GPU writes the packed matrix; CUDA launches chunked by an int64 pair offset | one fill per device |
| Diagnostics | no new `system_check()`: `test.parallelisation()` and `test.gpu()` are the interface in C++, Python and MATLAB (charter item 6); `system_info`, `check_system`, `*_AVAILABLE` go | the owner's own spelling |
| Python I/O | keep the optional HDF5 pair (h5py), reached through `load()` by extension; delete `io.py`'s CSV/Parquet wrappers, `convert.py`, `dtwc-convert`, `preprocess`, `diagnose`, `features` | recorded preference "HDF5 + CSV"; the rest duplicates C++ |
| OneBatchPAM | **Q3** | no measurement against CLARA exists |
| Records | required reading ≈ 600 lines: runbook, one handoff, CHARTER, MAP (≤150, gains the target interface and invariants), PLAN (≤150, the phase list). `design.md` folds into MAP + DECISIONS (≤150); LESSONS ≤300. Deleted: `reports/` (incl. the PRIVATE Kasper folder), `specs/`, the PLAN archive, TODO, `openmp-crashcourse.md`, old handoffs, finished plans, uncited baselines | the reading floor was ~148 KB |
| Gates | `check_docs.py` (~100 lines: every flag named in docs, README and `.claude/commands` exists in the live `dtwc_cl --help`, plus the harness self-check); `check_pins.py` (~40 lines); gitleaks in CI. Deleted: `check_docs_contract.py` (1,854), `check_supply_chain_pins.py` and its two tests, `check_repo_hygiene.py`, `repo_map.py`, `check_ipo_inlining.py`, `generate_docs.py` once no page is generated. Kept: `codegen_report.py` (manual), `machine_facts.py`, `smoke_release_archive.py`, `build_libomp_macos.sh` | a gate checks behaviour, never a sentence |
| CHANGELOG | one `2.0.0 (unreleased)` section measured against v1.0.0; a user-visible change against v1.0.0 gets a line | replaces "every PR updates CHANGELOG" (runbook rule 2) |
| Standing | D-11 tag after the release gate, D-12 infeasible band is an error, D-13 Python ≥ 3.10, D-14 one estimator, D-15 MATLAB's thread setting governs the MEX, D-19 tolerance across platforms: kept. D-16 (Pruned as diagnostic) and D-22 (no widening): overturned. D-18 (llfio): Q2. D-E: moot (GPU LB deleted). PF-1 `PackedOracle`, IF-7 `dtw_path`, IF-8 zero-copy 2-D ingest, public `RunStats`: dropped | each adds code without deleting two copies |

## 3. Target interface

```cpp
dtwc::device("gpu");                                    // cpu | gpu | gpu:N  (cuda[:N] alias); "hpc" throws, naming Python/CLI
auto data = dtwc::load("cycles/", {.skip_cols = 1});    // lazy; CSV/TSV, folder, Parquet, Arrow by extension
auto res  = dtwc::cluster(data, 8, {.band = 100});      // any Config key; method auto, seed 42
res.labels(); res.medoids(); res.cost(); res.score("silhouette"); res.save("out/");
double d  = dtwc::distance::dtw(x, y, {.variant = DTWVariant::MSM, .msm_c = 0.5});
```
```python
import dtwcpp as dtwc
dtwc.device("hpc:gpu")                                   # cpu | gpu | gpu:N | hpc | hpc:gpu
res = dtwc.cluster(dtwc.load("series.parquet"), k=50, band=400)     # kwargs == Config keys
res.labels; res.medoids; res.score("silhouette"); res.save("out/"); res.plot()
for k in range(3, 9): dtwc.cluster(data, k=k, dist_matrix=res.distance_matrix)   # a k-sweep fills once
d = dtwc.distance.dtw(x, y, variant="twe", twe_nu=1e-3, twe_lambda=1)          # one function, every variant
est = dtwc.DTWClustering(n_clusters=3, n_init=3).fit(X)  # the one estimator; matrix filled once; score() never refits
```
```matlab
dtwc.device('gpu'); res = dtwc.cluster(dtwc.load('cycles/'), 8, 'band', 1500, 'variant', 'msm');
res.score('silhouette'); res.save('out'); D = dtwc.compute_distance_matrix(X, 'band', 10);
```
```sh
dtwc_cl -i cycles/ -k 8 --band 1500 --device gpu -o out        # or --config job.toml; flags beat the file
dtwc_cl -i series.csv -k 3 --print-config > job.toml
```

| device | series | matrix | precision | refused before any I/O, typed |
|---|---|---|---|---|
| `cpu` | heap f64/f32; beyond RAM only as streamed Parquet (CLARA/OneBatch) | packed in RAM below `mmap_threshold`, else the `.dtwm` file | `dtype` | band narrower than a length gap; `checkpoint` with streaming |
| `gpu[:N]` | heap f64 | GPU writes the packed matrix, int64-chunked; CUDA, else Metal, else `DeviceError` naming the build flag **and the artefact** (wheel / MEX / CLI) | `auto` = f32 when FP64 runs below half the FP32 rate; `Result.config` shows what ran | a variant, missing strategy, ndim or storage the kernels do not implement; matrix-free methods; never a zero matrix, never a CPU fallback |
| `hpc[:gpu]` | never read locally: a path that exists locally is uploaded, otherwise it is a cluster path, and the submit line says which | remote | remote | each a `DeviceError` whose text is the fix: no `.env` (three-line example inline), a missing key, `ssh -o BatchMode` fails, no `dtwc_cl` on the cluster, job not COMPLETED (state + tail of the `.err`), timeout / Ctrl-C (job id + the download command) |

- **Config** stays flat, keyed by the CLI long names; `cli::bind` is the one key table; TOML/YAML through CLI11;
  `DistanceConfig` is built once in `run()`. `k` required, `method = auto`, `name` = input stem. `--help` in four
  groups (Run, Variant, Advanced, Solver). One canonical spelling per enum value plus at most one synonym. Keys
  removed: `batch-weighting`, `benders`, `max-benders-iter`, `resume`, `checkpoint-interval`.
- **Result** (every language): `labels, medoids, cost, method, iterations, converged, device, config,
  score(name), save(dir), distance_matrix`; Python and MATLAB add `plot()`. An `hpc` result has labels and medoids.
- **Python exports** ≈ 48 names (105 today): `device load cluster run Dataset Result distance
  compute_distance_matrix DTWClustering test`, Tier-2 `Problem Data ClusteringResult Method Solver`, seven
  algorithms, seven scores, `z_normalize derivative_transform soft_dtw_gradient save_checkpoint load_checkpoint`,
  the error classes. Every other enum is a string. Extras: `plot`, `sklearn`, `hdf5`, `test`.
- **MATLAB**: the same names, snake_case keys, 1-based double labels, one MEX `run` command over `parse_config`.
- **Tier-2** (C++; Python and MATLAB identical): `Problem` with the v1 fields, distance semantics private behind
  `set_distance / set_band / set_metric / set_variant / set_missing_strategy / set_device / set_gpu_precision`
  (each invalidates the matrix), `fill_distance_matrix`, O(1) `dist_by_ind`, `is_distance_matrix_filled()` a bool,
  `use_mmap_distance_matrix`, `save_checkpoint / load_checkpoint`, `cluster()` over nine methods, `set_result`;
  algorithms `fast_pam, fast_clara, one_batch_pam, tadpole, build_dendrogram / cut_dendrogram, dtw_barycenter,
  barycenter_kmeans`; scores `silhouette, davies_bouldin, dunn, calinski_harabasz, inertia, adjusted_rand,
  normalized_mutual_info`; the unchecked `warping*.hpp` kernels stay public C++, documented as unchecked.

## 4. Plan — seven phases

Proof rule for every step: serial `ctest`; `cpp_conformance` digit-identical to the phase base unless the step
pre-registers a change; pytest from a fresh `uv` venv when bindings, readers or defaults change; `matlab_suite`
when the MEX changes; the CUDA build on the RTX for GPU steps (Metal: macOS CI); `git grep` proof for every
deleted name (and `git grep <name> v1.0.0` = 0). At most 4 agents, one per step, never two on the same files.

| Phase | Content | Chair waves | ≈ LOC |
|---|---|---|---|
| **A** baseline, gates, records | fix the Windows-red `test_config_spellings` (CRLF from text-mode stdout: `lines_of` keeps `\r`, `trimmed` strips only spaces — test-only, 3 lines); record the HEAD conformance digits and three benchmarks (CPU fill, GPU fill, PAM swap); prove the CUDA build dir (nvcc 13.0 rejects the installed MSVC 14.50: `-allow-unsupported-compiler`, else the v143 toolset); gates → behaviour checks; records floor; CHARTER entry | W0, W1 | −37k |
| **B** deletions | pruned fill + dead bounds; research solvers + FasterPAM-only + `set_result`; persistence to one `.dtwm` (+ mio if Q2) and `Env` → two free functions; dead surface, never-released aliases, enum validator tails (`-Werror=switch`), `index_t` alias + every count guard | W2, W3, W5, W6 | −35k |
| **C** GPU to one fill | kernel A/B first; delete MPI, 1-vs-N/K-vs-N, GPU LB, `KernelOverride`; one `fill()` TU, packed GPU output, int64-chunked CUDA; Metal scratch failure → `DeviceError` | W4 + W13 GPU half | −6.5k |
| **D** `index_t` in public counts | `Problem`, `Data`, `ClusteringResult`, `Config`, algorithm signatures and loops; bindings int64 / MATLAB double; converting shim | W11 | ±150 |
| **E** interface | one distance resolver validated once (O(1) `dist_by_ind`, `bool filled_`, Soft-DTW on the linear kernel, one `distance::dtw` per language) → then in parallel: one reader / writer / matrix entry, and one run description (Method ×9, Python and MATLAB on `run(Config)`, `job.toml`, one estimator, v1 CLI aliases) with one device vocabulary | W7 → W8 ‖ W9 + W10 | −12k |
| **F** tests to their oracles | dissolve `adversarial/`, wave/phase files, enum-probing files, repeated DTW property tests → one table-driven `core/test_dtw.cpp` against `tests/support/dtw_oracle.hpp`; each commit names the oracle it keeps | W12 | −17k |
| **G** docs and release prep | hand-written tier pages and a real v1.0.0 → 2.0 migration page; `api-contract-2.0.md` deleted; CHANGELOG collapsed; CMake `FATAL_ERROR` for an explicit `ON` it cannot honour; CI on the branch (a push — yours); MAP regenerated; audit folder deleted | W14 | −3k |

After G, each behind a registered benchmark band: `kmedoids_pp` as one function, the HiGHS model built row-wise,
barycenter workspace, SIMD across pairs (PF-5, kill criterion 1.5×), and GPU CLARA assignment if Q4 says 2.1.

Pre-registered result changes (everything else must be digit-identical): `pam` results (FasterPAM); LR-core seed
(same cost, medoids may differ on ties); Soft-DTW summation order (1e-12 relative); v1 one-argument
`init::Kmeanspp` sequence; `Method::MIP` above N = 200 now uses the selected solver.

## 5. Kept on purpose

Cost/Cell template kernels (linear, banded, EAP) and every variant (DDTW, WDTW, ADTW, Soft-DTW + gradient, MSM,
TWE), missing-data strategies, dependent/independent multivariate; thread_local scratch; lock-free packed matrix
with NaN = not computed; `decode_pair`; the sequential swap; FP-model flags and the `#error` under
finite-math-only; `require_finite` at the checked boundary; `validate_fill_request`; LR-core; TADPole;
hierarchical; barycenter (Tier-2); Lloyd (v1); the seven scores; Gurobi; Arrow/Parquet + nanoarrow + the
metadata-first chunk reader; fast_float (macOS `from_chars`); `portable_random` (same seed, same result on every
standard library); SHA-256 (identity); `cli::bind` + `--print-config`; the real-binary CLI tests; conformance in
three languages; the exhaustive LB_Keogh derivation test; GPU-vs-host oracles; the v1 compile probe;
`CHARTER`, `UNIMODULAR`, `CITATIONS`, skills, commands.

## 6. Silent wrong answers fixed on the way (confirmed unless marked)

| Where | Defect | Phase |
|---|---|---|
| `Problem.cpp:1189-1193`; `_dtwcpp_core.cpp:1202-1225,1247` | Auto → Pruned (≥ brute-force work); Python never reaches the EAP kernel | B, E |
| `checkpoint.cpp:626-667`; `run.cpp:657-661` | a checkpoint for other data → silent recompute, exit 0, then overwritten | B |
| `DataLoader.hpp:125-222`; `mmap_data_store.hpp:176` | Auto storage leaks a temp `.dtws` ≥ half of free RAM per run (llfio builds) | B |
| `Problem.cpp:1446-1451` | `Method::MIP` above N = 200 ignores the selected solver | B |
| `fast_pam.cpp:326` | `pam` at k > `max_iter` returns unconverged | B |
| `scores.cpp:111-113` (since v1) | `silhouette()` on an unclustered problem prints and returns N × −1 | B |
| `one_batch_pam.cpp:36-50` | an explicit `batch_size < k` is silently raised to k | B |
| `metal_dtw.mm:1436-1453` | Metal returns an all-zero matrix on scratch failure; Python passes it on | C |
| `gpu_config.cuh:94-95` | consumer Blackwell (sm_120) classed full-rate FP64 | C |
| `dtw_kernel.hpp:220-221`; `scratch_matrix.hpp:65-74` | Soft-DTW value holds a grow-only O(n·m) thread_local matrix (~0.5 GB/thread at 8k) | E |
| `Problem.cpp:873-876,775` | two full preflights per O(1) `dist_by_ind`; an O(N²) "is filled" scan per Lloyd iteration | E |
| `dtw_dispatch.cpp:66-69` | Interpolate allocates two vectors per pair inside the parallel fill | E |
| `run.cpp:748`; `api.cpp:216-225` | `Result::save` after a streamed Parquet run reads names of an empty `Data` (UB) | E |
| `_dtwcpp_core.cpp:259-264`; `_api.py:83-86` | `dtwcpp.load('x.parquet')` parsed as CSV | E |
| `arrow_ipc_reader.hpp:95-147`; `arrow_c_data.cpp:166-177` | IPC nulls become values; the stream path drops series names | E |
| `parquet_reader.hpp:127,146-150` | names restart at `series_0` per file (duplicates), locale `tolower`, dot-files read | E |
| `fileOperations.hpp:465,519` (since v1) | every text load deep-copies the dataset | E |
| `__init__.py:258-261` | `device='gpu:1'` on Metal silently runs on GPU 0 | E |
| `dtwc_mex.cpp` (32 sites) | raw `static_cast<int>` of MATLAB doubles: 2.7 → 2, NaN → UB | B |
| `DTWClustering.m:139-142`; `_clustering.py:376-397,477-483` | squared-Euclidean + GPU runs on the CPU; `n_init` refills N² per restart; `score(X)` refits | E |
| `_api.py:552-555`; `_hpc.py:513-545` | no path reaches a GPU partition; a failed job surfaces as "labels not found" | E |
| `v1 Problem.hpp:136` | `cluster_by_kMedoidsPAM()` lost its name with no shim | B |
| `migration.md:115`; `.claude/commands/*.md` | docs promise flags that do not exist | E, G |
| `nearest_medoid.hpp:44-51` and 3 more (traced, not run) | duplicate series as medoids → LR-core throws / an empty cluster; first action is the test `{a,a,b,c}`, k = 4 | B |

## 7. Questions for Volkan

1. **Commits.** May implementer agents commit each proven step on `design-2.0` (worktree branches merged locally)?
   Never a push, tag or history rewrite. Recommend yes — the phases cannot be merged or bisected otherwise.
2. **llfio → mio** (overturns D-18). The lease and digests go in phase B; every shipped artefact builds with llfio
   OFF today, so pip and MATLAB users have no mapped matrix or checkpoint. mio (MIT, one header) gives them both
   and deletes the quickcpplib patch. Recommend mio. If llfio stays, builds without it keep one Dense → file
   writer for the checkpoint.
3. **OneBatchPAM.** No measurement against CLARA exists. Recommend one A/B (N ≈ 20k, one dataset) and delete it
   unless it wins on time or cost — unless you want it kept as a literature method regardless.
4. **GPU assignment for CLARA** (a rectangular medoids × series pair source on the existing pairwise kernels). It
   is the only route that makes `gpu` / `hpc:gpu` useful beyond ~100k series (the charter's 100M target). Recommend
   2.0, as the last performance item after phase C; otherwise 2.1.

For you to do, not decide here: `.claude/reports/test_kasper_analysis/` is headed PRIVATE and is in the pushed
history of `origin/design-2.0` and `origin/Claude`. Phase A removes it from the tree; removing it from history is a
remote rewrite only you can do.
