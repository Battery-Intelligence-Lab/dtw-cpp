# Test corpus taxonomy — adversarial / integration / conformance / fixtures / support / registration

**Scope:** `tests/unit/adversarial/` (15 files), `tests/integration/` (13 files), `tests/conformance/` (7 files),
`tests/fixtures/` (3), `tests/support/` (1), `tests/test_util.hpp`, `tests/CMakeLists.txt`.
**Repo:** `C:\D\git\dtw-cpp`, branch `Claude`, HEAD `a31956e`.
**Status:** TAXONOMY with verdict CANDIDATES. **Nothing here authorises deletion.** Read-only pass; no tracked file modified.
**Timing source:** `build/msvc-debug/Testing/Temporary/CTestCostData.txt` (that build has `DTWC_ENABLE_ARROW=OFF`,
`DTWC_BUILD_MATLAB=OFF`, `DTWC_ENABLE_YAML=ON` — `build/msvc-debug/CMakeCache.txt`).

**Scope-count correction:** the task brief said "9 `.cmake` scripts, ~2.5k lines" in `tests/integration/`.
Observed at HEAD: **6** `.cmake` scripts (1,837 lines) plus 3 `.cc` fixture writers (267), 3 `.py` (678) and
1 `.sh` (374) = 13 files / 3,156 lines. All 13 are covered below.

---

## 1. Per-file ledger

### 1a. `tests/unit/adversarial/` (15 files, 9,605 lines, 334 TEST_CASEs, 63 SECTIONs, 1,034 assertion macros)

| path | lines | TEST_CASEs (SECTIONs) | assertions | subject | category | oracle | bug class guarded | overlap candidates | verdict candidate | evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| `tests/unit/adversarial/test_arow_adversarial.cpp` | 872 | 33 (0) | 91 | `dtwc/warping_missing_arow.hpp` — `dtwAROW_L` / `dtwAROW` / `dtwAROW_banded` | mixed(adversarial, trivial, vacuous) | self-consistency (linear vs full) + one real invariant (AROW ≥ ZeroCost); **no independent AROW value oracle anywhere in the file** | AROW collapsing into zero-cost DTW; NaN escaping the DP. `.claude/LESSONS.md:13` "DTW-AROW ≠ zero-cost DTW"; introduced `2f943f9`; NaN-seed mechanism `CHANGELOG.md:1346` | 18 near-identical `linear==full` cases; `test_banded_dtw_adversarial.cpp:319` covers `dtwAROW_banded` feasibility with exact values | **merge into one data-driven TEST_CASE + delete L619** | L141–526: 18 TEST_CASEs whose entire body is `linear=dtwAROW_L; full=dtwAROW; REQUIRE(is_valid_distance(..)); REQUIRE_THAT(linear, WithinAbs(full, EPS));` differing only in the literal input. L619–665 "Triangle inequality" has **zero assertions** — L664 states verbatim: `// No REQUIRE here -- this test always passes; it just documents behavior.` L46 `INF = numeric_limits<double>::max()` declared and never used (and misnamed). |
| `tests/unit/adversarial/test_banded_dtw_adversarial.cpp` | 603 | 13 (21) | 86 | `dtwc::dtwBanded`, `dtwBanded_mv`, `dtwAROW_banded`, `core::dtw_band_bounds` | contract/oracle | **independent formula/derivation ×2** — full-matrix DP oracle *and* exhaustive monotone-path enumeration, sharing no production helper | non-canonical (endpoint-scaled) band geometry returning a finite value where no path exists; signed overflow at `INT_MAX` band. Commit `9f78212` "fix: enforce canonical Sakoe-Chiba bands"; `CHANGELOG.md:655-662`; `.claude/LESSONS.md:485` | L121/L226/L367/L482 all assert "wide band == full DTW" four different ways | **keep** — highest-value file in the directory | L44–72 `canonical_banded_oracle` (independent DP), L82–113 `exhaustive_path_oracle` (third arbiter). L180–184 registered ledger `{band,paths,l1,sq} = {2,696,5,9},{3,1143,3,5},{7,1289,3,5}`. L198 `REQUIRE(dtwBanded(x,y,band) == max_value)` pins the no-path sentinel exactly. L307–317 `dtw_band_bounds` beyond `INT_MAX` as `size_t`. |
| `tests/unit/adversarial/test_dense_distance_matrix_adversarial.cpp` | 467 | 17 (24) | 71 | `dtwc/core/distance_matrix.hpp` `DenseDistanceMatrix`, `dtwc/core/matrix_io.hpp` | mixed(contract/oracle, implementation-detail) | recorded value + round-trip self-consistency | NaN "uncomputed" sentinel surviving into `max()`/`all_computed()`/CSV; packed-triangular index errors; `resize` is a wipe. Commits `b5a5c6a`, `751387e`; `.claude/LESSONS.md:1262` | L71 SECTION 2 ≡ L158 SECTION 2 (both assert all-uncomputed after resize) | **keep + fix two defects** | Defect 1: L327 and L359 both construct `TempFile tmp("small_roundtrip")` → the **same** path `temp_dir/dtwc_test_small_roundtrip.csv` (L260-262). Defect 2: L440–467 asserts read_csv's row-major **last-write-wins loop order** (L458-460 walks the loop) — an implementation-detail assertion, not a spec property. |
| `tests/unit/adversarial/test_dtw_mathematical_properties.cpp` | 488 | 34 (3) | 65 | `dtwc::dtwFull`, `dtwFull_L`, `dtwBanded` | mixed(contract/oracle ×1, trivial ×21) | independent formula (D1 registered witnesses) + hand-derived values; elsewhere self-consistency | (a) empty-input check placed after the pointer-identity shortcut → `dtw(empty,empty)` returned 0; (b) band geometry. Commit `751387e` ("empty-size check now runs before pointer-identity shortcut"; "Update adversarial tests … (was documenting bug)"), `9f78212` | 21 of 34 cases are the same 7 properties repeated once per variant (L50–132 / L211–291 / L295–374) | **rewrite as one TEMPLATE_TEST_CASE + delete L179** | L179–207 "triangle inequality violations are bounded": `REQUIRE(max_excess < 1e6)` — an undrived sanity bound that passes trivially when `max_excess` stays 0. It is superseded by L134–177 `[D1][oracle]` which pins exact witnesses (`REQUIRE(d_xz == 3.0); REQUIRE(d_xy == 1.0); REQUIRE(d_yz == 1.0); REQUIRE(d_xz > d_xy + d_yz);`). Coverage gap: L341 tests only `(empty,y)`/`(y,empty)`, missing the `(empty,empty)` case that L96 and L255 do assert. |
| `tests/unit/adversarial/test_eap_dtw.cpp` | 345 | 5 (8) | 12 | `dtwc/core/dtw_kernel.hpp` `dtw_kernel_eap` via `dtwc::dtwFull_eap` | mixed(contract/oracle, perf-fence) | independent oracle `dtwFull_L` for exactness; **`eap_probe` is a mirror, not an oracle** | prune threshold with no FP slack pruned the optimal final cell → `+inf` on ~1/200 random pairs. `.claude/LESSONS.md:95` and `:100`; commit `ea6615d`; `CHANGELOG.md:1145-1149` | none | **keep exactness cases; delete or de-fang L292** | L162–223 are the real value (800 checked pairs, `REQUIRE(checked == 800)` at L178). L89–153 `eap_probe` re-implements the production window **including the identical slack** `thr = ub + |ub|*(nl*16*eps)` (L113-114) — LESSONS.md:95 calls that factor "regression-tested, not a proved worst-case bound", so L243 "probe re-derives the exact distance" is largely self-referential. L292–345 bench is tagged `[.]` (never runs under CTest), prints `REGISTERED BAND-SPEEDUP floor: >=1.5x` (L342) and then asserts nothing but `SUCCEED()` (L344). |
| `tests/unit/adversarial/test_fast_pam_adversarial.cpp` | 501 | 19 (3) | 50 | **NOT FastPAM** — `Method::Kmedoids` → `cluster_by_kmedoids_lloyd` (legacy Lloyd) | mixed(contract/oracle ×1, implementation-detail/structure-only ×18) | one brute-force optimality oracle; elsewhere self-consistency only | structural k-medoids invariants. `.claude/LESSONS.md:482` names **this exact file** as the mislabelled-test example; `:483` records that structure-only assertions passed through a real wrong answer (FastPAM k=1 `second_dist=+inf`) | L97 + L150 ⊂ L384; L125/L224/L455 all re-check medoid uniqueness | **rename + merge structural cases; strengthen L270** | L57 `prob.set_method(Method::Kmedoids);` is the whole misnomer. L199–222 is the one genuine oracle (brute-force: no other point is a cheaper single medoid). L270–306 "Better than random": `REQUIRE(pam_cost <= worst_random_cost + 1e-10)` — beating the **worst** of 10 random draws is a near-vacuous bar. L113 constructs a `Problem` that is never used (Catch2 re-enters the case per SECTION, so it is built 4× for nothing). L340 title claims per-iteration monotonicity; the body (and its own comment, L342-345) compares `max_iter=1` vs `max_iter=100`. |
| `tests/unit/adversarial/test_lb_enhanced_webb.cpp` | 522 | 15 (0) | 40 | `dtwc/core/lower_bound_impl.hpp` `lb_enhanced`, `lb_webb`, `lb_webb_symmetric`, `compute_webb_envelope` | mixed(contract/oracle, perf-fence) | independent formula (`naive_minmax` for envelopes); randomised `LB ≤ DTW_w` against the live banded kernel | inadmissible bound (`LB > DTW`) silently pruning true nearest neighbours; `max(band,0)` treating band<0 as radius 0. `CHANGELOG.md:488-491` (counterexample A=[0,5,0,0], B=[0,0,5,0]: bound 10 vs DTW 0); `.claude/LESSONS.md:185`; commits `a09db32`, `6abff20` | envelope check at L403 ⊂ the exhaustive 2,004-case oracle in `test_lb_enhanced_webb_derivation.cpp:587-607` (small n only) | **keep; merge L119/137/155/173; delete L513** | L513–522 "LB tightness bench" is tagged `[.]` and contains **no assertion at all** — L508-510 prints the literal string `"< FALSIFIED"` when the ordering breaks and still exits 0. L119/L137/L155/L173 are four copies of the same loop differing only in (bound, metric). |
| `tests/unit/adversarial/test_lb_enhanced_webb_derivation.cpp` | 1113 | 1 (0) | 91 | `lb_enhanced`, `lb_webb`, `core::pruned_distance_matrix` routes | contract/oracle | **independent formula/derivation** — `direct_envelope`, `exact_path_costs`, `reference_enhanced`, `reference_webb`, `full_matrix_dtw` all written independently of production | bound-vs-implementation drift (an oracle-free bound whose proof is the implementation). Commits `244adf7`, `53506a9`; baseline `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`; `CHANGELOG.md:530-540`. `.claude/LESSONS.md:22-23` states the V≥2 ordering rests on *these* witnesses, not on SDM 2019 | none | **keep; fix the hardcoded marker and the stdout-format assertion** | Exhaustive inventories REQUIREd at L606 (2004), L630 (19), L781/782 (35982/7380), L806 (68787), L810 (35982), L818 (35982), L860 (140). Two weaknesses: (i) L1107–1112 the gate marker is a **fully hardcoded literal** — unlike D2 it interpolates no counter, so a deleted REQUIRE would still print the same "PASS" line; (ii) L1097–1103 pins the exact verbose stdout **including `"pruning ratio: 0.333333"`** — a cosmetic wording change fails the gate. |
| `tests/unit/adversarial/test_lb_keogh_derivation.cpp` | 527 | 1 (0) | 65 | `core::compute_envelopes`, `lb_keogh`, `lb_keogh_squared`, `*_mv`, `algorithms::tadpole`, `fill_distance_matrix_pruned` | contract/oracle | **independent formula/derivation** — `naive_envelope` + explicit monotone-path enumeration, sharing no DP recurrence with production | LB_Keogh admissibility breaking on unequal lengths and on a helper band not covering the real DTW window (`compute_envelopes(s, band<0)` → band-0 envelope → LB_Keogh = Euclidean ≥ DTW). `.claude/LESSONS.md:27` (cost TADPole a wrong-rho bug), `:14`, `:34`; commits `21ba41d`, `f4bdd55`, `39e9a92`; baseline `.claude/baselines/2026-07-30-d2-lb-keogh.md` | envelope inventory overlaps the 5 duplicated envelope cases in `test_lower_bounds_adversarial.cpp` (see next row) | **keep — this is the model the other gates should copy** | L299/321/345/520 REQUIRE the case counts, and L521–526 prints those **computed counters** (`envelope_cases`, `equal_cases`, `unequal_cases`, `call_sites`) into the marker, so the CTest `PASS_REGULAR_EXPRESSION` is non-vacuous. L410–445 explicitly pins the negative-radius gotcha: `REQUIRE(unsafe_negative_bound > dtwFull_L(x,y))`. |
| `tests/unit/adversarial/test_lb_webb_intmax.cpp` | 110 | 1 (0) | 24 | `compute_webb_envelope`, `lb_webb`, `lb_enhanced` radius saturation | contract/oracle | independent derivation (exact global-window values 4.0 / 8.0 / 2.0) | signed radius arithmetic overflowing at `INT_MAX`; doubled radius, free-run counters and shifted indices. `CHANGELOG.md:545-548`; commits `244adf7`, `13cd4f6` | none (deliberately isolated from D3) | **keep** | Keeps all three radius calls independent (L61-63) so a coincidentally shared result cannot hide a defect. Weakness: L107–109 marker is hardcoded (`l1=4/4 squared=8/8 …`) — same pattern as D3; the independent protection is the CTest assertion floor `(1[2-9]\|[2-9][0-9]\|…)` (`tests/CMakeLists.txt:227`). |
| `tests/unit/adversarial/test_lower_bounds_adversarial.cpp` | 893 | 29 (0) | 67 | `lb_kim`, `lb_keogh`, `compute_envelopes` (Lemire ring buffer) | mixed(contract/oracle, **self-oracle ×2**, duplicated) | independent formula (`naive_envelopes`) — but two cases have oracle = *none* (they test test-code) | LB > DTW; Lemire sliding-window ring-buffer aliasing/overflow, w=0 and w≥n fast paths, n=1/n=2. `.claude/LESSONS.md:14`, `:27`; envelope half added by commit `2dac179`; `CHANGELOG.md:492-498` | **5 internal duplicate pairs** + overlap with D2's exhaustive envelope oracle | **delete L506 + L527; merge 5 duplicate pairs; keep the rest** | **Self-oracles:** L81–105 define a file-local `SeriesSummary`/`compute_summary`/`lb_kim_from_summary` shadowing `dtwc::core::*`. L506–522 "compute_summary returns correct first, last, min, max" and L527–537 "…on empty series" call the **file-local** helper (unqualified; no `using namespace dtwc::core`; ADL cannot reach it for `std::vector<double>`) and compare it to `std::min_element` — they assert **nothing about the library**. By contrast L543–563 compares `dtwc::core::lb_kim` against the local reference and *is* a real oracle. **Duplicates:** L457 ≡ L768 (band-0 envelope is identity); L479 ≡ L789 (band ≥ n gives global min/max); L420 ⊂ L737 ∪ L819 (all three compare against a brute-force sliding window). |
| `tests/unit/adversarial/test_missing_utils_adversarial.cpp` | 770 | 66 (0) | 139 | `dtwc/missing_utils.hpp` (`is_missing`, `has_missing`, `missing_rate`, `interpolate_linear`), `dtwc/warping_missing.hpp` | mixed(contract/oracle, trivial, weak) | hand-derived values for interpolation and one SquaredL2 DTW; cross-implementation elsewhere | misclassifying non-NaN specials (−0.0, subnormals, ±inf, epsilon, DBL_MAX) as missing, or NaN escaping the DP — fatal because NaN is also `DenseDistanceMatrix`'s uncomputed sentinel. `.claude/LESSONS.md:256`, `:181-184`; commit `f47dbfa` | 4 mergeable groups (see below) | **merge into 4 table-driven cases; rewrite L663 and L749** | 66 TEST_CASEs, most single-assertion. Mergeable: `is_missing` L74–186 (13 cases), `dtwMissing_L`≡`dtwMissing` consistency L479–544 (7 cases, identical body), "gives zero" patterns L333–424 (9 cases), symmetry L715–743 (3 cases). **Weak assertions:** L670 `bool valid = std::isfinite(dist) \|\| (dist == max());` accepts *any* finite value while the title says "returns max" — compare `test_dtw_mathematical_properties.cpp:92` which asserts `== maxVal` exactly. Same pattern at L767. **Title/assertion mismatch:** L389 "…gives zero" asserts only `>= 0.0` and `isfinite`; L570 "interpolated >= missing" asserts neither (its own comment L574-575 says the relation is not guaranteed). **Dead code:** L256 `double expected = static_cast<double>(i);` computed and never used. L296 ⊂ L206/L224/L264. |
| `tests/unit/adversarial/test_multivariate_adversarial.cpp` | 772 | 25 (3) | 58 | `dtwFull_L_mv`, `dtwBanded_mv`, `derivative_transform_mv`, `Problem` ndim>1 | mixed(contract/oracle, perf-fence, vacuous) | cross-route (MV ndim=1 vs scalar; MV derivative vs per-channel scalar); hand-derived value for the single-timestep case | a dead-duplicate MV dispatcher — `core::dispatch_mv_metric` was fixed for L2 while the live `detail::dispatch_mv_metric` kept aliasing L2→L1; ndim>1 falling through to the univariate recurrence; channel bleed. `.claude/LESSONS.md:484` (names `dtwBanded_mv`→`warping.hpp` as the real chain, fix `ffb7a8d`), `:172`; commits `0f77fc8`, `9f78212` | L645–649 duplicates L71; complementary to `test_banded_dtw_adversarial.cpp:259` | **split L628 (keep correctness, drop the clock); delete L273** | **Perf-fence:** L670 `REQUIRE(ms_mv < ms_scalar * 3.0 + 50.0)` is a wall-clock assertion — the sole reason for `RUN_SERIAL TRUE` at `tests/CMakeLists.txt:668` and a latent CI flake. Its correctness half (L645–649) restates L71 with 500 pairs instead of 120. **Vacuous:** L273–309 "triangle inequality — document violations" runs 600 DTW calls and asserts only `REQUIRE(checks == TRIPLES)` (L308) i.e. that the loop ran; L307 says so verbatim. **Weakened:** L198/L221 `REQUIRE((d >= 0.0 \|\| d == max()))` admits the sentinel although inputs are never empty. **Title/assertion mismatch:** L608 comment "All off-diagonal distances should be positive" vs L609-611 `REQUIRE(… >= 0.0)`. |
| `tests/unit/adversarial/test_scores_adversarial.cpp` | 1219 | 59 (1) | 112 | `dtwc/scores.{hpp,cpp}` — silhouette, DBI, Dunn, CH, ARI, NMI, inertia | mixed(contract/oracle, trivial, **vacuous ×1**, duplicated) | independent formula (Rousseeuw 1987; Davies–Bouldin 1979 axiom 3; sklearn-style hand-computed ARI/NMI) | degenerate-partition sentinels producing a plausible *perfect* score: silhouette left `min` at `DBL_MAX` so `(MAX−a)/MAX ≈ +1`; DBI skipped `M_ij == 0` leaving `max_ratio` at its `0.0` initialiser; indices computed on the **declared** rather than the realised partition. `.claude/LESSONS.md:1084`, `:1092`, `:1098`; commit `0af4cbd`; `CHANGELOG.md:287-326` | 4 exact/near-exact duplicate pairs | **delete L228, L500, L514, L640; keep the A3–A7 block** | **Vacuous:** L228–263 wraps its only assertions in `if (!std::isnan(sil[i])) { … }` (L257) — a NaN result runs **no** assertion. It is fully superseded by L1123–1142 (same 4-identical-series/2-cluster configuration) which asserts `REQUIRE_FALSE(std::isnan(s))` **and** `WithinAbs(0.0, 1e-15)`. **Duplicates:** L500 ≡ L908 ("CH positive for well-separated"); L514 ≡ L923 ("CH k=1 throws"); L530 ≡ L640 (both `{0,0,1,1}` vs `{0,0,1,1}` → `WithinAbs(1.0,1e-12)`); L153 ⊃ L207. **Weak:** L618–638 asserts `-0.1 < ari < 0.1` for 10,000 random labels — an undrived statistical band (deterministic seed makes it a de-facto fingerprint). **Inconsistency:** L465 calls `fill_distance_matrix()` before scoring; L816/L833/L860/L880/L900 do not. |
| `tests/unit/adversarial/test_scratch_matrix_adversarial.cpp` | 403 | 16 (0) | 63 | `dtwc/core/scratch_matrix.hpp` `ScratchMatrix<T>` | mixed(contract/oracle, trivial, language-behaviour) | independent derivation (`raw[j*R+i] == m(i,j)` on a non-square matrix) | a silent column-major → row-major layout flip when the backing store is swapped. `CHANGELOG.md:1831`; commit `b5a5c6a` (re-based on private Eigen inheritance); `.claude/LESSONS.md:688` "A symmetric distance matrix cannot validate row-major/column-major copy orientation" — this file is the deliberately non-symmetric (4×7) seam that lesson asks for | L50/L82/L97/L228/L380 all restate the layout fact proved exhaustively at L27 | **keep L27+L82; merge L97/L228/L380; delete L50; de-prioritise L250/L286** | L27–48 already proves column-major **exhaustively** for all (i,j) of a 4×7. L50–80 then asserts only `REQUIRE(mismatches > 0)` (L79) — it would pass if a single element of fifteen disagreed, and it adds nothing L27 has not already settled. L250 and L286 declare `thread_local ScratchMatrix<double> local_m;` inside the lambda (L259, L293): what they verify is the **C++ `thread_local` guarantee**, not any property of `ScratchMatrix` (which holds no static state to share). |

### 1b. `tests/integration/` (13 files, 3,156 lines)

| path | lines | cmake steps / cases | assertions | subject | category | oracle | overlap candidates | verdict candidate | evidence |
|---|---|---|---|---|---|---|---|---|---|
| `tests/integration/test_cli_resume_state.cmake` | 520 | 12 CLI runs + 2 fixture modes; 9 rejection cases | ~60 `message(FATAL_ERROR)` guards | real `dtwc_cl` binary, `--resume` | **integration(real binary)** | real-binary behaviour + recorded byte payloads | none | **keep; replace literal counters with accumulators** | Highest-value CLI test in the corpus: L61-72 pins the tracked input/config SHA-256 *before* the run and L503-514 re-checks them *after* (proves the CLI did not mutate tracked data); L367 `require_checkpoint_unchanged` compares SHA **and** timestamp; L413-418 proves replay labels **differ** from the fresh control (a real discriminator, not just "it ran"). **Weakness:** L516-520 emits `runs=12/12 … rejection_cases=9/9` as **hardcoded literals** — deleting `run_rejection(nonfinite_cost …)` (L499) would still print `rejection_cases=9/9` and still match `tests/CMakeLists.txt:754`. |
| `tests/integration/test_distance_matrix_csv_contract.cmake` | 404 | 2–3 runs (resident, native `Result::save`, +mmap) | ~40 guards | real `dtwc_cl` + native `Result::save` | **integration(real binary)** | real-binary behaviour + cross-route byte identity | none | **keep — reference implementation of a counted gate** | Uses **real accumulators**: `set_property(GLOBAL APPEND PROPERTY f14_runs …)` (L218, L243), read back and checked at L365-379/L389-395 before the marker is printed. Byte-level contract at L263-284: exactly 27 LF, 0 CR, final byte `0a`, no `0a0a`, no BOM. L57-63 refuses a resolved work-root escape before `file(REMOVE_RECURSE)`. |
| `tests/integration/test_fast_clara_assignment_contract.cmake` | 375 | 4 CLI runs + 2 poison-stream rejections | ~45 guards | real `dtwc_cl`, FastCLARA streaming vs resident | **integration(real binary)** | real-binary behaviour + IEEE-754 byte pins | shares the eager/stream marker logic with `test_fast_clara_parquet_parity.cmake` | **keep** | Six independent counters (L98-103) all verified at L357-368. L288-294 pins the exact stderr string `"Error: fast_clara: non-finite nearest-medoid distance at point 0, medoid slot 0 (index 65)."` — a real diagnostic contract. Not in the `msvc-debug` run: registered only under `DTWC_HAS_PARQUET` (`tests/CMakeLists.txt:823`). |
| `tests/integration/test_fast_clara_parquet_parity.cmake` | 329 | 6 CLI runs (3 configs × resident/stream) | ~40 guards | real `dtwc_cl`, Parquet eager vs streaming parity | **integration(real binary)** | real-binary behaviour + registered IEEE-754 encodings | as above | **keep; close two gaps** | Real counters at L64-67/L305-320. Gap 1: **no skip check anywhere in the script** (its sibling has them at L138/L259) and **no `FAIL_REGULAR_EXPRESSION` in its CTest registration** (`tests/CMakeLists.txt:844-849` sets only `LABELS`, `RUN_SERIAL`, `TIMEOUT`). Gap 2: L253-270 wraps the labels/medoids/checkpoint SHA pins in `if(WIN32)` — on Linux/macOS those three byte-level assertions silently do not run, yet the same marker prints. L284-294 documents the 2-ULP GCC Soft-DTW encoding honestly. |
| `tests/integration/test_cli_config_formats.cmake` | 154 | 7 or 9 CLI runs (per YAML flavour) | 23 (YAML on) / 7 (off), **counted** | real `dtwc_cl`, `--config` TOML+YAML through CLI11 | **integration(real binary)** | real-binary behaviour | none | **keep — the only script whose count is genuinely computed** | `set(checks 0)` (L21) incremented by every `expect`/`expect_zero`/`expect_nonzero` (L41, L48, L55) and by the manual case at L102; the CTest regex pins the exact total per flavour (`tests/CMakeLists.txt:813-814`, `_dtwc_cl_yaml_checks` 23/7 at L797-802). Verified by hand: YAML-on path sums to 23, YAML-off to 7. Minor: the final skip scan at L148 inspects only `${combined}` from the **last** run. |
| `tests/integration/test_cli_rejects_unknown_option.cmake` | 55 | 1 CLI run | 4 guards | real `dtwc_cl` rejecting `--yaml-config` | **integration(real binary)** | real-binary behaviour (exit code + flag named back) | none | **keep** | Deliberately asserts the **exit code** (L28) rather than relying on `PASS_REGULAR_EXPRESSION`, which ignores exit status — the rationale is stated at L3-7 and echoed at `tests/CMakeLists.txt:759-763`. L34 additionally rejects a non-numeric `result` (crash/timeout). |
| `tests/integration/f17_checkpoint_writer.cc` | 112 | n/a (fixture producer) | 3 internal checks | production `save_binary_checkpoint`/`load_binary_checkpoint` | capability-guard/fixture | production round-trip self-check | none | **keep** | L92-96 round-trips through the production loader and fails if any field changed — the fixture cannot be silently wrong. L99-103 back-dates `last_write_time` by 24 h so an accidental rewrite is observable **even if the bytes and SHA are identical**. |
| `tests/integration/f14_result_save_writer.cc` | 48 | n/a | 3 internal checks | public `dtwc::load`/`cluster`/`Result::save` | integration(real binary) | real-binary behaviour | none | **keep** | Drives the public Tier-1 surface (L28-29) and refuses to emit its marker unless `labels()==27 && medoids()==3 && device()=="cpu"` (L30-34). |
| `tests/integration/f13_poison_parquet_writer.cc` | 107 | n/a | Arrow status checks | Arrow/Parquet fixture generation (129 rows, 2 row groups) | capability-guard/fixture | n/a | none | **keep** | L27-39 writes `-DBL_MAX` at row 0 and `+DBL_MAX` elsewhere; L65 `WriteTable(…, 65)` yields 129 = 65+64 → exactly 2 row groups, matching the marker at L103-104. |
| `tests/integration/test_cross_language.py` | 497 | 8 classes, ~35 tests | ~60 asserts | `dtwcpp` Python bindings vs C++ core | integration(bindings) | cross-route parity + a few hand values | overlaps `tests/python/test_cross_validation.py` (stated in its own docstring L10-13) | **keep; move it into the CI-collected path** | **Outside CTest and outside CI.** `.github/workflows/python-tests.yml:36` runs `uv run pytest tests/python/ -v --tb=short` — `tests/integration/` is never collected, and no `add_test` references it. Its only automated invoker is a **skill**: `.claude/skills/check-code-quality.md:347-348` (`if [ -f tests/integration/test_cross_language.py ]; then uv run pytest … `), i.e. it runs only when an agent chooses to. Announced at `CHANGELOG.md:1771` and `:1393` as the C++≡Python≡MATLAB parity gate. Also carries a UTF-8 BOM at byte 0. |
| `tests/integration/test_cli_missing_data.py` | 82 | 4 CLI runs | 12 asserts | real `dtwc_cl`, `--missing-strategy` boundary | integration(real binary) | real-binary behaviour | none | **register it, or delete it** | Good content — L52-55 asserts the typed error *and* that neither `terminate` nor `abort` appears; L56-57 asserts no artifacts were produced. But it is a `main()`-style script requiring `--cli <path>` (L28) and **nothing invokes it**: no `add_test`, no workflow reference. |
| `tests/integration/test_cli_variant_domains.py` | 99 | 14 CLI runs | ~20 asserts | real `dtwc_cl`, variant parameter domains | integration(real binary) | real-binary behaviour | none | **register it, or delete it** | 8 invalid + 6 valid cases (L21-49); L79 asserts parameter validation wins **before** any filesystem effect. Same problem: unregistered. |
| `tests/integration/stress_test_cli.sh` | 374 | up to ~45 CLI runs in 3 phases | counted PASS/FAIL/SKIP | real `dtwc_cl` across method × variant × metric | integration(real binary), **skip-as-common-path** | real-binary behaviour + one Rand-Index sanity check | Phase 1 overlaps `test_cli_variant_domains.py` | **rewrite or delete** — see §4 | L11-13 hardcodes `$REPO/build/bin/dtwc_cl[.exe]` — a **repo-relative** path (violates project non-negotiable #1) that silently validates whatever stale binary sits there. L14 `COFFEE_TRAIN` points into `data/benchmark/UCRArchive_2018/Coffee/`, which is **untracked** (`git ls-files` returns nothing): on any fresh clone Phases 2 **and** 3 are skipped wholesale (L189-190, L343) and the script still prints `"All tests passed."` and `exit 0` (L371-373). L5 uses `set -uo pipefail` without `-e`. Unregistered in CTest and CI. |

### 1c. `tests/conformance/` (7 files, 656 lines)

| path | lines | cases | assertions | subject | category | oracle | overlap candidates | verdict candidate | evidence |
|---|---|---|---|---|---|---|---|---|---|
| `tests/conformance/cpp_conformance.cpp` | 233 | 1 | 6 | live C++ Tier-2 pipeline: `DataLoader` → `set_band(3)` → `fill_distance_matrix` → `fast_pam(k=3)` → scores | conformance | **cross-route parity** vs the recorded reference — *except* on the regen path, where it is a self-oracle | it is the reference **producer** for the other three routes | **keep; remove the self-healing regen path from the test binary** | L225-232 is the real gate (`REQUIRE(live.labels == ref.labels)`, `WithinRel(…, 1e-12)`). But L212-220: if `conformance_reference.txt` does **not** exist, the test **writes it from the live run** and then compares live against what it just wrote — a guaranteed pass, and a write into the **tracked source tree** (`reference_file()` resolves to `<repo>/tests/conformance/`, L67-73, L150). Registered only by the glob (`tests/CMakeLists.txt:1`, helper at `cmake/Coverage.cmake:31-32`) → **`SKIP_RETURN_CODE 4` with no `PASS_REGULAR_EXPRESSION` and no `FAIL_REGULAR_EXPRESSION`.** |
| `tests/conformance/conformance_reference.txt` | 15 | n/a | 4 recorded values | the pinned parity values | fingerprint | recorded value | consumed by all four routes | **keep** | `labels`, `medoids,4,13,22`, `silhouette,0.96894972764334841`, `davies_bouldin,0.038333333333333337`, `dunn,11.5`. **Drift marker:** line 3 says `fillDistanceMatrix` while the writer at `cpp_conformance.cpp:156` now emits `fill_distance_matrix` — evidence the tracked reference has not been regenerated since the API rename (values unaffected; header text only). |
| `tests/conformance/conformance.toml` | 21 | n/a | 7 keys | the CLI route's pipeline pin | fixture | n/a | its 7 keys are independently re-asserted at `test_distance_matrix_csv_contract.cmake:90-105` | **keep** | `n-clusters=3, method="pam", band=3, metric="l1", variant="standard", max-iter=100, name="conformance"` (L15-21). SHA pinned at `test_cli_resume_state.cmake:69-72`. |
| `tests/conformance/data/conformance_series.csv` | 27 | n/a | n/a | 27 series × 16 samples, all integers | fixture | n/a | none | **keep** | SHA-256 pinned at `test_cli_resume_state.cmake:65-68`; row count re-asserted at `test_distance_matrix_csv_contract.cmake:73-78`. |
| `tests/conformance/data/generate_conformance_data.py` | 63 | n/a | n/a | regenerates the CSV deterministically | fixture generator | n/a | none | **keep** | No RNG at all (L44-53): every value is an explicit integer. L20-31 documents *why* the FastPAM optimum is unique and init-independent (baselines 0/100/200; 9 members per cluster ⇒ odd ⇒ unique median; pulse centres 4 apart > band 3 so the band is load-bearing). |
| `tests/conformance/test_conformance.py` | 180 | 2 | 8 asserts | Python route + CLI route vs the same reference | conformance | cross-route parity | its CLI route duplicates what the `.cmake` gates now do against the real binary | **keep the Python route; the CLI route is redundant and fragile** | **Not registered anywhere** (CI runs only `pytest tests/python/`). `find_cli_binary()` (L83-92) probes `build/bin`, `bin`, `build` — **repo-relative** (non-negotiable #1) and stale-binary-prone; when it finds nothing the CLI route `pytest.skip`s (L142-145). The canonical gate build is `build/highs-1151`, whose binary is not on that list. |
| `tests/conformance/test_conformance.m` | 117 | 1 | 6 verifies | MATLAB route vs the same reference | conformance | cross-route parity | none | **keep, but note it is outside `matlab_suite`** | `matlab_suite` runs `runtests('${CMAKE_CURRENT_SOURCE_DIR}/matlab')` (`tests/CMakeLists.txt:936`) — i.e. `tests/matlab`, **not** `tests/conformance`. This file is executed only by `.github/workflows/matlab-mex.yml:55`. L41-43 `assumeTrue` makes a missing MEX an *Incomplete* (silent skip), which the `matlab_suite` allow-list at `tests/CMakeLists.txt:936` would reject — but the allow-list never sees this file. |

### 1d. `tests/fixtures/`, `tests/support/`, `tests/` root

| path | lines | cases | assertions | subject | category | oracle | verdict candidate | evidence |
|---|---|---|---|---|---|---|---|---|
| `tests/fixtures/f22_cpp_legacy_diagnostics.inc` | 164 | n/a (compiled by 2 object probes) | 33 named entities + 33 use sites | every retained `[[deprecated]]` C++ surface | fixture (compile-time) | compiler diagnostics | **keep** | Named 20 Problem overloads + 5 scores + 4 loader/path + 2 fields (L35-…, L120-163). Consumed by `f22_cpp_legacy_werror` / `f22_cpp_legacy_suppressed` (`tests/CMakeLists.txt:46-56`). |
| `tests/fixtures/f22_cpp_canonical_diagnostics.inc` | 136 | n/a | 31 overloads + 2 field pairs + 2 special members | the canonical (non-deprecated) counterparts | fixture (compile-time) | compiler diagnostics | **keep** | L122-134 additionally proves move-construct/move-assign stay diagnostic-free despite the two retained deprecated fields. |
| `tests/fixtures/fast_clara_streaming_8x4.parquet` | 1,451 B | n/a | n/a | 8 series × 4 samples, Parquet | fixture (binary) | n/a | **keep** | `PAR1` magic verified; size **and** SHA-256 pinned twice (`test_fast_clara_parquet_parity.cmake:31-43`, `test_fast_clara_assignment_contract.cmake:65-77`). |
| `tests/support/deterministic_series.hpp` | 122 | n/a | n/a | portable RNG + dense symmetric reference used by CUDA/Metal/MPI/F15 tests | support | n/a | **keep — exemplary** | L29-58 replaces the implementation-defined `uniform_real_distribution` with a `genrand_res53` schedule and hexadecimal power-of-two scales, with the FP argument spelled out (L40-42: "nothing for `-fassociative-math` to reassociate"; L48-56 covers x87 excess precision). Consumers: `tests/unit/unit_test_deterministic_series.cpp`, `unit_test_mpi.cpp`, `test_cuda_*.cpp`, `test_metal_*.cpp`. |
| `tests/test_util.hpp` | 102 | n/a | n/a | random data/name generation + CSV/TSV writers | support | n/a | **rewrite** — two real defects | **Defect 1 (silent wrong type):** `template <typename data_t> std::vector<std::vector<data_t>> get_random_data(...)` (L23-24) builds `std::vector<std::vector<double>>` internally (L26, L32) — the template parameter is **ignored**; a `get_random_data<float>` call would not compile-error, it would silently return doubles. **Defect 2 (repo-relative writes):** `write_data_to_folder` calls `fs::create_directory(folder_name)` with a **relative** name (L68) and every test runs with `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}` (`cmake/Coverage.cmake:20/27/31`) — so these tests create directories **inside the source tree**, against non-negotiable #1. Also `int i` vs `.size()` sign-compare at L77. Consumers: `unit_test_fileOperations.cpp:232/381`, plus 3 files that include it without using it (`unit_test_checkpoint.cpp:23`, `unit_test_clustering_algorithms.cpp:17`, `unit_test_distance_matrix_properties.cpp:17`). |
| `tests/CMakeLists.txt` | 945 | 21 registration sites | n/a | the whole registration layer | registration | n/a | **keep; extract a helper** | See §2. |

---

## 2. Registration layer — `tests/CMakeLists.txt`

### 2.1 How a test gets registered

`file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS … "*.cpp")` (L1) picks up **every** `.cpp` under `tests/`,
including `tests/conformance/cpp_conformance.cpp` and all 15 adversarial files. Each becomes an executable and a
CTest entry through `add_executable_with_coverage_and_test` (`cmake/Coverage.cmake:3-42`), which sets exactly one
property: **`set_tests_properties(${TARGET_NAME} PROPERTIES SKIP_RETURN_CODE 4)`** (`cmake/Coverage.cmake:32`) and
`WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}`. `test_problem_api_2_0` is the sole special case (L122-134): it is launched
through `scripts/test_f22_cpp_deprecations.py --launch`.

Measured on `build/arrow-pyarrow-23`: **125 CTest entries registered, 11 carry a `PASS_REGULAR_EXPRESSION`.**
The other ~114 are gated by exit code alone **and** carry `SKIP_RETURN_CODE 4` — a Catch2 `SKIP` or any exit-4 path
is scored green with no floor.

### 2.2 The 13 per-test gate blocks (confirmed — the IO/CLI map's count is correct)

| # | line | subject | skip regex | PASS_REGULAR_EXPRESSION (marker + floor) | extras |
|---|---|---|---|---|---|
| 1 | 146 | `unit_test_checkpoint_binary` (F51) | **B** | `F51_BINARY_CHECKPOINT corpus=85 rejected=85 throws=0 unchanged=85/85 size_preflight=1/1 valid_bytes=72/72 fields=5/5 resave=72/72 semantic_compat=7/7 skips=0 verdict=PASS` + ≥270 assertions in ≥2 cases | `ENVIRONMENT TMP/TEMP/TMPDIR`, `RUN_SERIAL`; FATAL_ERROR guards on the fixture root (L149-154) |
| 2 | 181 | `test_lb_keogh_derivation` (D2) | **A** | `D2_LB_KEOGH_GATE envelope_cases=2004 equal_cases=28602 unequal_cases=17712 call_sites=2/2 skips=0 verdict=PASS` + ≥65 assertions in 1 case | `OMP_NUM_THREADS=1`, `RUN_SERIAL` |
| 3 | 198 | `test_lb_enhanced_webb_derivation` (D3) | **B** | `D3_LB_ENHANCED_WEBB_GATE envelope_cases=2004 path_cases=35982 full_cover_cases=7380 enhanced_cases=68787 enhanced_v5=4/4 webb_cases=35982 webb_branches=4/4 webb_strict=2/2 tail_cases=35982 tail_strict=2/2 metric_cases=140 order_witnesses=2/2 cascade_routes=2/2 skips=0 verdict=PASS` + ≥40 assertions in 1 case | `OMP_NUM_THREADS=1`, `RUN_SERIAL`, `TIMEOUT 60` |
| 4 | 217 | `test_lb_webb_intmax` (F57) | **B** | `F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 admissible=2/2 skips=0 verdict=PASS` + ≥12 assertions in 1 case | `RUN_SERIAL`, `TIMEOUT 30` |
| 5 | 236 | `unit_test_DataLoader` (F21) | **A** | `F21_CPP_NAMES canonical=4/4 legacy=4/4 overloads=12/12 loader_state=22/22 path_state=16/16 cstring_copy=4/4 skips=0 verdict=PASS` + ≥79 assertions in ≥2 cases | — |
| 6 | 251 | `test_problem_api_2_0` (F22) | **A** | two markers (`F22_CPP_DIAGNOSTICS …` then `F22_CPP_COMPAT …`) + ≥65 assertions in ≥5 cases | `RUN_SERIAL`; FATAL_ERROR guards on the fixture root (L254-262) |
| 7 | 282 | `unit_test_nearest_medoid_assignment` (F13) | **C** (bare) | *floor only* — `All tests passed \((…) assertions in (…) test cases\)`, **no subject marker** | `OMP_NUM_THREADS=4`, `PROCESSORS 4` |
| 8 | 301 | `unit_test_distance_matrix_csv` (F14 focused) | **A** | `F14_CSV_CONTRACT dense=ran mmap=ran\|unavailable skips=0` + per-flavour floor | branches on `DTWC_HAS_MMAP` |
| 9 | 330 | `unit_test_problem_encapsulation` (F19) | **A** | `F19_PROBLEM_API getters=10/10 setters=9/9 lloyd=ran skips=0` + ≥30 assertions in ≥3 cases | — |
| 10 | 349 | `unit_test_problem_storage_policy` (F20) | **A** | `F20_PROBLEM_STORAGE_POLICY build=llfio-on\|off …` + per-flavour floor | `ENVIRONMENT TMP/TEMP/TMPDIR`, `RUN_SERIAL`; branches on `DTWC_HAS_MMAP` |
| 11 | 382 | `unit_test_deterministic_series` (F15) | **A** | `F15_TEST_SUPPORT generator=portable scalar=ran row_seeded=ran continuous=ran dense=ran source_audit=ran skips=0` + ≥177 assertions in ≥6 cases | — |
| 12 | 619 | `test_supply_chain_pinning` (F16) | **A** | `F16_CMAKE_PRESETS floor=3.26.0 compiler=clang++ host_conditions=ran metadata_guard=ran skips=0` + ≥35 assertions in ≥4 cases | preceded by ~200 lines of configure-time `CMakePresets.json` validation (L405-618) |
| 13 | 643 | `test_io_readers` (F6) | **B** | *floor only* — ≥300 assertions; **no subject marker**; registered **only** when `DTWC_HAS_ARROW` | rationale documented L632-641 |

Plus: `test_multivariate_adversarial` gets **`RUN_SERIAL TRUE` only** (L666-668) — no gate, no marker — solely
because of the wall-clock assertion at `test_multivariate_adversarial.cpp:670`.

### 2.3 The four skip regexes (verified by extraction)

| id | regex | uses | lines |
|---|---|---|---|
| **A** | `[Ss][Kk][Ii][Pp]([Pp]\|[ :])` | 11 | 189, 243, 273, 323, 341, 375, 395, 628, 715, 776, 811 |
| **B** | `(^\|[\r\n])[ \t]*[Ss][Kk][Ii][Pp]([Pp][Ee][Dd]\|[Pp][Ii][Nn][Gg])?([ :]\|$)` | 4 | 170, 206, 224, 650 |
| **C** | `[Ss][Kk][Ii][Pp]` (bare — matches "skipping", "skipped", "skips=0" too) | 1 | 290 |
| **D** | `(^\|[\r\n])[ \t]*[Ss][Kk][Ii][Pp]([Pp][Ee][Dd])?([ :]\|$)` (= **B** minus the `PING` alternative) | 1 | 750 |

The `.cmake` scripts carry a **fifth** and **sixth** copy of the same idea in-script:
variant **A** at `test_cli_resume_state.cmake:193`, `test_distance_matrix_csv_contract.cmake:130`,
`test_cli_config_formats.cmake:148`, `test_cli_rejects_unknown_option.cmake:48`; variant **C** at
`test_fast_clara_assignment_contract.cmake:138` and `:259`; and **none at all** in
`test_fast_clara_parquet_parity.cmake`.

### 2.4 What is inconsistent

1. **Four skip regexes for one concept.** A and B differ in strength (B is anchored and covers `SKIPPED`/`SKIPPING`);
   C matches its own marker's substring `skips=0`; D is B with one alternative dropped for no stated reason.
2. **Marker discipline splits three ways.** D2 (`test_lb_keogh_derivation.cpp:521-526`) and
   `test_distance_matrix_csv_contract.cmake` / `test_cli_config_formats.cmake` interpolate **computed counters**;
   D3 (`:1107-1112`), F57 (`:107-109`) and `test_cli_resume_state.cmake:516-520` print **hardcoded literals** that
   survive assertion deletion; blocks #7 and #13 have **no marker at all** (floor only).
3. **Two blocks are floor-only** (#7 `unit_test_nearest_medoid_assignment`, #13 `test_io_readers`) — they prove
   "many assertions ran", not "the subject ran".
4. **Fixture-root guarding is applied to 3 of 4 candidates.** F51 (L149-154) and F22 (L254-262) FATAL_ERROR on a
   non-absolute or drifted leaf; F20 (L351) and F19 (L334) set the same kind of root with **no** guard.
5. **`ENVIRONMENT TMP/TEMP/TMPDIR` redirection** is used by F51 (L168-169) and F20 (L373-374) only, although
   `test_dense_distance_matrix_adversarial.cpp:261` also writes into `temp_directory_path()`.
6. **`f13_poison_parquet_writer` links Arrow/Parquet but not `project_options`** (L829-833), unlike
   `f14_result_save_writer` (L679-684) and `f17_checkpoint_writer` (L727-732) — so it is built with different
   warning/FP settings from every other test artifact.
7. **`WORKING_DIRECTORY` is inconsistent**: `${CMAKE_SOURCE_DIR}` for the glob helper and for
   `test_distance_matrix_csv_contract` / `test_cli_resume_state` / `test_cli_rejects_yaml_config` /
   the F8+F13 pair, but `${CMAKE_CURRENT_BINARY_DIR}` for `test_cli_config_formats` (L807).
   Running tests with cwd = the source tree is what lets `tests/test_util.hpp:68` write into it.
8. **Labels are ad hoc**: `integration;f14`, `integration;f17`, `cli`, `integration;cli`, `integration;arrow;f8`,
   `integration;arrow;f13`, `matlab`. `test_cli_rejects_yaml_config` gets `cli` but **not** `integration`, although
   it drives the real binary. The 125 glob-registered tests get **no labels at all**, so `ctest -L unit` is impossible.

### 2.5 What a single registration helper would take

```
dtwc_register_gate(<target>
    MARKER   "<exact subject marker with computed counters>"
    ASSERT_FLOOR <n>  CASE_FLOOR <n>
    [ENV …] [SERIAL] [PROCESSORS n] [TIMEOUT s] [LABELS …]
    [FIXTURE_ROOT <abs path>]      # emits the absolute + leaf-drift FATAL_ERRORs
    [REQUIRES <COMPILE_DEF>])      # skip registration entirely, never a green stub
```
It would (a) clear `SKIP_RETURN_CODE`, (b) apply the **one** canonical skip regex (variant **B**), (c) compose
`PASS_REGULAR_EXPRESSION` from `MARKER` + the Catch2 floor, (d) apply the fixture-root guards, and
(e) redirect `TMP/TEMP/TMPDIR` whenever `FIXTURE_ROOT` is given. That collapses the 13 blocks (≈250 lines,
L146-397 + L619-653) to ~13 calls, and makes the four-regex divergence unrepresentable.

### 2.6 Registrations that can pass vacuously

| registration | why it can pass without testing the subject |
|---|---|
| **all ~114 ungated glob targets**, incl. `cpp_conformance` and 12 of 15 adversarial files | `SKIP_RETURN_CODE 4` from `cmake/Coverage.cmake:32` with **no** `PASS_REGULAR_EXPRESSION` and **no** `FAIL_REGULAR_EXPRESSION` — a Catch2 `SKIP` or any exit-4 path is green, with no assertion floor to notice |
| `cpp_conformance` specifically | additionally self-heals: `cpp_conformance.cpp:212-220` regenerates the reference from the live run when the file is absent, then compares against it → guaranteed pass |
| `test_fast_clara_parquet_parity` | no `FAIL_REGULAR_EXPRESSION` and no `PASS_REGULAR_EXPRESSION` (`tests/CMakeLists.txt:844-849`); the script itself has no skip check; and its 3 byte-level SHA pins are `if(WIN32)`-only (`:253-270`) |
| `test_cli_resume_state` | the pinned counters in its `PASS_REGULAR_EXPRESSION` (`:754`) are literals emitted unconditionally by `test_cli_resume_state.cmake:516-520`; deleting a `run_rejection` call still prints `rejection_cases=9/9` |
| `test_lb_enhanced_webb_derivation`, `test_lb_webb_intmax` | same literal-marker weakness (`:1107-1112`, `:107-109`); protected only by the independent Catch2 assertion floors |
| `unit_test_nearest_medoid_assignment`, `test_io_readers` | floor-only regex — proves assertions ran, not that the intended subject ran |
| `matlab_suite` | `DTWC_BUILD_MATLAB=OFF` (the default and the `msvc-debug` setting) emits **no test and no warning**; when ON but MATLAB is absent it emits only `message(WARNING)` (L942-943) |

---

## 3. Cross-file duplication of test scaffolding

| # | group | duplicated in | consolidation |
|---|---|---|---|
| S1 | **`random_series(rng, len, lo, hi)`** — byte-identical body | `test_arow_adversarial.cpp:50`, `test_banded_dtw_adversarial.cpp:32`, `test_eap_dtw.cpp:57`, `test_lb_enhanced_webb.cpp:57`, `test_lower_bounds_adversarial.cpp:43`, `test_dtw_mathematical_properties.cpp:35` (as `make_random_series`), `test_multivariate_adversarial.cpp:45` (as `random_mv_series`) | move into `tests/support/` beside `deterministic_series.hpp` — which already provides a **portable, FP-flag-proof** generator (`benchmark_series`, `accelerator_series_set`) that these seven all ignore |
| S2 | **`random_length(rng, min, max)`** | `test_lb_enhanced_webb.cpp:66`, `test_lower_bounds_adversarial.cpp:53` | same |
| S3 | **Naive sliding-window min/max envelope oracle** — 4 independent copies | `test_lb_enhanced_webb.cpp:389 naive_minmax`, `test_lower_bounds_adversarial.cpp:717 naive_envelopes` **and** the inline copy at `:435-443`, `test_lb_keogh_derivation.cpp:127 naive_envelope`, `test_lb_enhanced_webb_derivation.cpp:146 direct_envelope` | keep **one** independent implementation in a shared `tests/support/lb_oracles.hpp`; independence is preserved as long as it shares nothing with `dtwc/core/lower_bound_impl.hpp` |
| S4 | **Exhaustive monotone-path DTW enumerator** — 3 near-identical copies | `test_banded_dtw_adversarial.cpp:82 exhaustive_path_oracle`, `test_lb_keogh_derivation.cpp:150 enumerate_paths`, `test_lb_enhanced_webb_derivation.cpp:185 enumerate_paths` | same shared header; all three already return `{min_l1, min_squared, count}` |
| S5 | **Full-matrix DP DTW oracle** | `test_banded_dtw_adversarial.cpp:44 canonical_banded_oracle`, `test_lb_enhanced_webb_derivation.cpp:534 full_matrix_dtw` | same |
| S6 | **`Audit` accumulator + `words()` alphabet enumerator + `show()` + `SerialOpenMpScope`** | `test_lb_keogh_derivation.cpp:55/103/91/68` ≡ `test_lb_enhanced_webb_derivation.cpp:37/114/102/78` — essentially verbatim | one `tests/support/exhaustive_audit.hpp` |
| S7 | **`make_problem(series, band, name)` / `make_synthetic_problem`** | `test_lb_keogh_derivation.cpp:249`, `test_lb_enhanced_webb_derivation.cpp:561`, `test_scores_adversarial.cpp:54 make_problem` + `:76 assign_clusters`, `test_fast_pam_adversarial.cpp:34` | one `tests/support/problem_builders.hpp` |
| S8 | **Temp-dir helper** | `test_dense_distance_matrix_adversarial.cpp:258 TempFile` (RAII, `temp_directory_path()`, **fixed names — collides at `:327` and `:359`**) vs `tests/test_util.hpp:68` (relative `create_directory`, source-tree write) vs the CMake-injected `DTWC_F*_TEST_ROOT` pattern used by F14/F19/F20/F22/F51 | standardise on the CMake-injected absolute build-local root; make `TempFile` take it and append a unique suffix |
| S9 | **CLI-driving boilerplate in the `.cmake` scripts** — the *same five blocks* repeated in each of the 6 scripts: required-`-D` loop, `cmake_path(ABSOLUTE_PATH …)`, work-root escape/symlink check + `REMOVE_RECURSE`, `execute_process` + exit-code check, `require_occurrences` / `require_no_skip` | required-var loop: `test_cli_resume_state.cmake:3-8`, `test_distance_matrix_csv_contract.cmake:3-8`, `test_fast_clara_parquet_parity.cmake:3-7`, `test_fast_clara_assignment_contract.cmake:3-8`, `test_cli_config_formats.cmake:7-11`, `test_cli_rejects_unknown_option.cmake:8-12`. `require_occurrences` is **verbatim identical** at `test_cli_resume_state.cmake:174-189` and `test_distance_matrix_csv_contract.cmake:111-126`. `require_no_skip` verbatim at `:191-200` and `:128-137`. Work-root guard: strong form (REAL_PATH + symlink) at `test_cli_resume_state.cmake:37-59` and `test_distance_matrix_csv_contract.cmake:42-65`; **weak** `string(FIND …)` form at `test_fast_clara_parquet_parity.cmake:23-27` and `test_fast_clara_assignment_contract.cmake:31-35` | one `tests/integration/cli_gate_common.cmake` `include()`d by all six — this is the single largest mechanical reduction available (≈300 of 1,837 `.cmake` lines) and it would also fix the two weak work-root guards and the missing skip check in `test_fast_clara_parquet_parity.cmake` |
| S10 | **Hash-pin + fixture-preflight pattern** | `test_cli_resume_state.cmake:61-72` (input+config SHA), `test_distance_matrix_csv_contract.cmake:69-105` (SHA+size+rows+7 TOML lines), `test_fast_clara_parquet_parity.cmake:31-43` and `test_fast_clara_assignment_contract.cmake:65-77` (identical fixture size 1451 + identical SHA `2F259F…3CA8`) | one `dtwc_pin_fixture(<path> SIZE <n> SHA256 <hex>)` function; the two Parquet scripts currently duplicate the same constant |
| S11 | **Canonicalisation of FastPAM output (sorted medoid set + rank labels)** — implemented four times in four languages | `cpp_conformance.cpp:97-115`, `test_conformance.py:57-66`, `test_conformance.m:85-95`, and re-derived by hand in `test_conformance.py:157-166` for the CLI route | irreducible across languages (it *is* the parity contract), but the four should be pinned to one another by a shared documented spec block — currently only comments claim they "mirror exactly" |
| S12 | **Tolerant matchers** — `WithinAbs` with 8 different tolerances (1e-15, 1e-12, 1e-10, 1e-9, 1e-6, 1e-5, 1e-3, 1e-290) chosen ad hoc | throughout the adversarial corpus; e.g. `test_lb_enhanced_webb.cpp:149` (1e-6 "squared magnitudes: looser abs tol") vs `:131` (1e-9) vs `:377` (1e-3 for 1e12-magnitude inputs) | a small set of named tolerances with a stated derivation, per the tolerant-comparison precedent at `tests/unit/algorithms/unit_test_barycenter.cpp:396-424` |

---

## 4. Tests that test the wrong thing

**Self-oracles (assert nothing about the library):**
- `test_lower_bounds_adversarial.cpp:506-522` and `:527-537` — call the **file-local** `compute_summary` (L85) and
  compare it to `std::min_element`/`front()`/`back()`. `dtwc::core::compute_summary` is never invoked.
- `test_eap_dtw.cpp:243` — `probe.distance` is produced by a 65-line re-implementation of the production window
  (L89-153) that copies production's own `nl*16*eps` slack (L113-114). Agreement is close to tautological.
- `test_scratch_matrix_adversarial.cpp:250` and `:286` — verify the C++ `thread_local` storage guarantee
  (`thread_local ScratchMatrix<double> local_m;` at L259/L293), not anything about `ScratchMatrix`.

**Vacuous passes (a defect cannot make them fail):**
- `test_arow_adversarial.cpp:619-665` — zero assertions; L664 states it "always passes".
- `test_lb_enhanced_webb.cpp:513-522` — zero assertions; prints the literal `"< FALSIFIED"` on failure and exits 0.
- `test_eap_dtw.cpp:292-345` — `SUCCEED()` only, after printing a "REGISTERED floor" it never checks; `[.]`-hidden.
- `test_scores_adversarial.cpp:228-263` — `if (!std::isnan(sil[i])) { REQUIRE… }` (L257): a NaN runs no assertion.
- `test_multivariate_adversarial.cpp:273-309` — asserts only `checks == TRIPLES` (L308) after 600 DTW calls.
- `test_dtw_mathematical_properties.cpp:179-207` — `REQUIRE(max_excess < 1e6)` where `max_excess` is 0 unless a
  violation occurs, so the assertion cannot fail for the reason the test exists.
- `test_missing_utils_adversarial.cpp:663-672` and `:749-770` — `isfinite(d) || d == max()` accepts any finite value.
- `tests/conformance/cpp_conformance.cpp:212-220` — regenerates the reference from the live run when it is absent,
  then compares against it (and writes into the tracked source tree).

**Implementation-detail assertions:**
- `test_dense_distance_matrix_adversarial.cpp:440-467` — asserts `read_csv`'s row-major last-write-wins loop order
  (the comment at L458-460 walks the loop body).
- `test_lb_enhanced_webb_derivation.cpp:1097-1103` — pins the exact verbose stdout string including
  `"pruning ratio: 0.333333"`; a cosmetic message change fails the gate.
- `test_scratch_matrix_adversarial.cpp:50-80` — `REQUIRE(mismatches > 0)` is a "not row-major" statement already
  proved exhaustively at L27-48.
- Comment-level only (assertions are fine): `test_dtw_mathematical_properties.cpp:62` "pointer shortcut",
  `test_multivariate_adversarial.cpp:242` "should short-circuit", `test_scores_adversarial.cpp:157`
  "the code checks `mean_distances[i_c].first == 1`", `test_banded_dtw_adversarial.cpp:514` "thread-local buffer
  reuse" (the test is single-threaded).

**Titles that over-claim what is asserted:**
`test_missing_utils_adversarial.cpp:389` ("gives zero" → asserts `>= 0`), `:570` ("interpolated >= missing" →
asserts neither), `:663` ("returns max" → accepts any finite); `test_multivariate_adversarial.cpp:591`
(comment "should be positive" → `>= 0.0`); `test_fast_pam_adversarial.cpp:340` ("non-increasing across
iterations" → compares two separate runs); `test_fast_pam_adversarial.cpp` filename (says FastPAM, drives Lloyd
via `Method::Kmedoids` at L57 — `.claude/LESSONS.md:482`).

**Skip-as-common-path:**
- `tests/integration/stress_test_cli.sh` — `COFFEE_TRAIN` (L14) lives in **untracked** `data/benchmark/UCRArchive_2018/`
  (`git ls-files` returns nothing); on a fresh clone Phases 2 and 3 skip entirely (L189-190, L343) and the script
  still prints `"All tests passed."` and `exit 0` (L371-373).
- `tests/conformance/test_conformance.py:139-145` — the CLI parity route `pytest.skip`s whenever `find_cli_binary()`
  (L83-92, repo-relative `build/bin`/`bin`/`build`) misses; the canonical gate build `build/highs-1151` is not on
  that list. Moot in practice: the file is not collected by CI at all.
- `tests/conformance/test_conformance.m:41-43` — `assumeTrue` renders a missing MEX *Incomplete*, and this file is
  outside `matlab_suite` (which runs `tests/matlab` only, `tests/CMakeLists.txt:936`), so the allow-list guard at
  L936 never applies to it.

**Unregistered — in neither CTest nor CI** (5 files, ~1,332 lines):

| file | only automated invoker found | evidence |
|---|---|---|
| `tests/integration/test_cross_language.py` | a **skill**, `.claude/skills/check-code-quality.md:347-348` | announced as the parity gate at `CHANGELOG.md:1771` / `:1393` |
| `tests/integration/test_cli_missing_data.py` | **none** | referenced only by itself |
| `tests/integration/test_cli_variant_domains.py` | **none** | referenced only by itself |
| `tests/integration/stress_test_cli.sh` | **none** | referenced only by itself and `.claude/reports/2026-09-02-review-io-cli.md:7` (which cites `stress_test_cli.sh:275-286` as covering *only the happy path* of a live `read_distance_matrix` defect) |
| `tests/conformance/test_conformance.py` | **none** | 2 of the 4 conformance routes therefore never run automatically |

Verified by `grep -rn "add_test" --include=CMakeLists.txt` (only the 7 sites in `tests/CMakeLists.txt`),
by `.github/workflows/python-tests.yml:36` (`pytest tests/python/` only), and by a repo-wide reference sweep
over `*.md`, `*.yml`, `*.txt`, `*.cmake`, `*.py`, `*.sh`.

**Global-constraint violations found (non-negotiable #1, "no runtime dependence on repo-relative paths"):**
`tests/test_util.hpp:68` (`fs::create_directory` on a relative name, with cwd = `${CMAKE_SOURCE_DIR}`),
`tests/conformance/test_conformance.py:87`, `tests/integration/stress_test_cli.sh:11-14`,
`tests/conformance/cpp_conformance.cpp:150` (writes into the tracked source tree on the regen path).

---

## 5. Slow tests

**Nothing in this scope exceeds 10 s.** Full ranking from
`build/msvc-debug/Testing/Temporary/CTestCostData.txt` (MSVC Debug, `DTWC_ENABLE_ARROW=OFF`):

| test | seconds | what dominates |
|---|---|---|
| `test_multivariate_adversarial` | 2.79 | the 500-pair timing loop at `test_multivariate_adversarial.cpp:628-675` (1,000 full-DTW calls on 50–150-length series, run twice) plus `RUN_SERIAL TRUE` |
| `test_lb_enhanced_webb_derivation` | 2.62 | the 35,982-case quadruple loop at `:646-778`, each case running path enumeration + 4 Enhanced + 4 Webb + 4 capped-Webb evaluations; `OMP_NUM_THREADS=1` |
| `test_cli_resume_state` | 1.27 | 12 real `dtwc_cl` process launches + 11 fixture-writer launches |
| `test_lb_keogh_derivation` | 1.26 | 28,602 + 17,712 exhaustive path enumerations; `OMP_NUM_THREADS=1` |
| `test_cli_rejects_yaml_config` | 1.21 | one process launch (dominated by CLI11 startup, not work) |
| `test_dtw_mathematical_properties` | 1.05 | `:380-397` — 20 pairs of length up to 500 through both `dtwFull` and `dtwFull_L` |
| `test_lower_bounds_adversarial` | 0.87 | 50-pair loops at lengths 50–200 with full DTW |
| `test_eap_dtw` | 0.56 | 800 exactness checks; the `[.]`-hidden bench does **not** run |
| all others in scope | ≤ 0.55 | — |

Not measurable from this log (not registered in the `msvc-debug` configuration):
`test_fast_clara_parquet_parity` and `test_fast_clara_assignment_contract` (require `DTWC_HAS_PARQUET`,
`tests/CMakeLists.txt:823`); `matlab_suite` (`DTWC_BUILD_MATLAB=OFF`; `TIMEOUT 1800`). Their CTest timeouts are
120 s / 120 s / 1800 s respectively. **Not established:** their actual wall-clock at HEAD.

For reference, the four tests that *do* exceed 10 s repo-wide are all **outside this scope**:
`test_mip_backend_guards` 218 s, `unit_test_distance_matrix_properties` 116 s,
`unit_test_clustering_algorithms` 57.5 s, `unit_test_checkpoint` 52.0 s.

---

## 6. Totals

### Files and lines in scope

| area | files | lines |
|---|---|---|
| `tests/unit/adversarial/` | 15 | 9,605 |
| `tests/integration/` (6 `.cmake` / 3 `.cc` / 3 `.py` / 1 `.sh`) | 13 | 3,156 |
| `tests/conformance/` | 7 | 656 |
| `tests/fixtures/` | 3 (2 text + 1 binary) | 300 + 1,451 B |
| `tests/support/` | 1 | 122 |
| `tests/` root (`CMakeLists.txt`, `test_util.hpp`) | 2 | 1,047 |
| **total** | **41** | **14,886** |

### Adversarial corpus by category

| category | TEST_CASEs | assertion macros | files (primary) |
|---|---|---|---|
| contract/oracle (independent formula or derivation) | 21 | 373 | banded, lb_keogh_derivation, lb_enhanced_webb_derivation, lb_webb_intmax, eap (exactness), dense_matrix (partly) |
| adversarial (bug-class regression, self-consistency oracle) | 143 | 397 | arow, missing_utils, lower_bounds, multivariate, scores (A3–A7 block) |
| trivial / structure-only | 158 | 244 | dtw_mathematical_properties (×3 variants), fast_pam, scratch_matrix, scores (range/sign checks) |
| perf-fence | 3 | 2 | eap `[.]`, lb_enhanced_webb `[.]`, multivariate `:628` (the only one that runs under CTest) |
| vacuous / self-oracle | 9 | 6 | arow `:619`, scores `:228`, mv `:273`, dtw_math `:179`, lower_bounds `:506`/`:527`, eap `:292`, lb_enh_webb `:513`, scratch `:50` |
| **total** | **334** | **1,034** | 15 files, 63 SECTIONs |

### Rest of scope by category

| category | files | note |
|---|---|---|
| integration(real binary) | 9 (6 `.cmake` + 3 `.py`/`.sh`) | 4 of 9 are unregistered |
| capability-guard / fixture producer | 3 `.cc` | all three drive production serializers/writers |
| conformance | 4 (C++ / Python / MATLAB routes + reference) | only the C++ route is in CTest, and it is ungated |
| fixture (data) | 5 (2 `.inc`, 1 `.parquet`, 1 `.csv`, 1 `.toml`) | all pinned by size and/or SHA except the two `.inc` |
| support | 2 | `deterministic_series.hpp` exemplary; `test_util.hpp` defective |
| registration | 1 | 21 registration sites, 13 gate blocks, 4 skip regexes |

### Verdict candidates by class

| verdict | count | targets |
|---|---|---|
| **keep** | 19 | banded_dtw, lb_keogh_derivation, lb_enhanced_webb_derivation, lb_webb_intmax, all 6 `.cmake` scripts, all 3 `.cc` writers, all 5 conformance data/config/reference files, `deterministic_series.hpp`, both `.inc` fixtures, the `.parquet` fixture |
| **keep + fix a named defect** | 6 | dense_matrix (temp-name collision `:327`/`:359`), lb_enh_webb_derivation (literal marker `:1107`, stdout pin `:1097`), lb_webb_intmax (literal marker), `test_cli_resume_state.cmake` (literal counters `:516`), `test_fast_clara_parquet_parity.cmake` (no skip check; `if(WIN32)`-only SHA pins), `cpp_conformance.cpp` (self-healing regen `:212`) |
| **merge / rewrite as contract** | 8 | arow (18→1 data-driven), dtw_math (21→7 TEMPLATE_TEST_CASE), missing_utils (66→~30 via 4 tables), scores (delete 4 dupes), lower_bounds (merge 5 dupe pairs), scratch_matrix (merge 4 layout restatements), fast_pam (rename + merge structural), lb_enhanced_webb (merge 4 validity cases) |
| **delete (each covered elsewhere — named)** | 12 assertions/cases | see the ten strongest below, plus `test_scores_adversarial.cpp:500`/`:514` (→ `:908`/`:923`) |
| **register or delete** | 5 files | `test_cross_language.py`, `test_cli_missing_data.py`, `test_cli_variant_domains.py`, `stress_test_cli.sh`, `test_conformance.py` |
| **rewrite (defective helper)** | 1 | `tests/test_util.hpp` — ignored template parameter (L23-26) + source-tree writes (L68) |

### Ten strongest delete/merge candidates, each with the covering test named

| # | delete/merge | covered by |
|---|---|---|
| 1 | `test_arow_adversarial.cpp:619-665` (triangle inequality; **0 assertions**) | nothing needs to cover it — it asserts nothing. The scalar non-metric property is pinned exactly by `test_dtw_mathematical_properties.cpp:134-177` `[D1][oracle]`. |
| 2 | `test_lb_enhanced_webb.cpp:513-522` (tightness bench; **0 assertions**, `[.]`) | the ordering claims it prints are pinned as exact witnesses by `test_lb_enhanced_webb_derivation.cpp:924-955` (`order_witnesses == 2`) and `:958-974` (`webb_strict == 2`). |
| 3 | `test_scores_adversarial.cpp:228-263` (a=b=0 silhouette; assertions inside `if (!isnan)`) | `test_scores_adversarial.cpp:1123-1142` — same configuration, asserts `REQUIRE_FALSE(std::isnan(s))` and `WithinAbs(0.0, 1e-15)`. |
| 4 | `test_lower_bounds_adversarial.cpp:506-522` + `:527-537` (test the file's own `compute_summary`) | production `compute_summary` is exercised by `test_lb_enhanced_webb.cpp:486` and `test_lb_enhanced_webb_derivation.cpp:1043-1046`; the summary-vs-direct contract is kept by `test_lower_bounds_adversarial.cpp:543-563`. |
| 5 | `test_lower_bounds_adversarial.cpp:457-474` + `:479-501` (exact duplicates of `:768-784` and `:789-810`) | `test_lower_bounds_adversarial.cpp:768-784` (w=0 identity) and `:789-810` (w≥n global min/max) — plus exhaustively for n ≤ 5 by `test_lb_keogh_derivation.cpp:277-300`. |
| 6 | `test_dtw_mathematical_properties.cpp:179-207` (`max_excess < 1e6`) | `test_dtw_mathematical_properties.cpp:134-177` — exact registered witnesses `d_xz == 3.0`, `d_xy == 1.0`, `d_yz == 1.0`, `d_xz > d_xy + d_yz`. |
| 7 | `test_multivariate_adversarial.cpp:273-309` (`REQUIRE(checks == TRIPLES)` after 600 DTW calls) | nothing covers "MV DTW is non-metric" today. **Delete only if** the property is not wanted; otherwise **rewrite** as registered exact MV witnesses in the style of `test_dtw_mathematical_properties.cpp:134`. |
| 8 | `test_multivariate_adversarial.cpp:628-675` — split: delete the wall-clock `REQUIRE` at `:670`, merge the correctness loop `:645-649` | `test_multivariate_adversarial.cpp:71-94` already asserts MV(ndim=1) ≡ scalar over 120 random pairs. Deleting `:670` also removes the sole reason for `RUN_SERIAL TRUE` at `tests/CMakeLists.txt:668`. |
| 9 | `test_scratch_matrix_adversarial.cpp:50-80` (`REQUIRE(mismatches > 0)`) | `test_scratch_matrix_adversarial.cpp:27-48` proves `raw[j*R+i] == m(i,j)` for **every** (i,j) of a non-square 4×7 — strictly stronger. |
| 10 | `test_arow_adversarial.cpp` L141–526 — merge 18 TEST_CASEs into one table-driven case | no coverage is lost: every one has the identical body (`linear==full` + `is_valid_distance`); only the literal input differs. The random-input equivalent is `test_arow_adversarial.cpp:268-301`. |

*(Runners-up, same standard of evidence: `test_scores_adversarial.cpp:500`/`:514` → `:908`/`:923`;
`test_scores_adversarial.cpp:640-661` → `:530-537`; `test_dense_distance_matrix_adversarial.cpp:71` SECTION 2 →
`:158` SECTION 2; `test_dtw_mathematical_properties.cpp:120` and `:279` → the identity cases at `:50` and `:211`.)*

---

## Open questions / not established

1. Actual wall-clock of `test_fast_clara_parquet_parity`, `test_fast_clara_assignment_contract` and `matlab_suite`
   at HEAD — no run log for an Arrow-ON or MATLAB-ON configuration is present in `build/*/Testing/Temporary/`.
2. Whether the four unregistered `tests/integration/` scripts and `tests/conformance/test_conformance.py` are
   deliberately manual-only or are unintentional orphans. No `AGENTS.md`/`PLAN.md` statement was found either way;
   their own docstrings say "Run explicitly with the freshly built binary", which suggests deliberate.
3. Whether the `if(WIN32)` restriction on the F8 SHA pins (`test_fast_clara_parquet_parity.cmake:253-270`) is a
   recorded platform decision or an accident. Not established from the file, `CHANGELOG.md`, or `.claude/LESSONS.md`.
4. Whether the D3 stdout-format assertion (`test_lb_enhanced_webb_derivation.cpp:1097-1103`) is intentionally a
   user-visible-output contract or incidental. The surrounding comment (`:1088`) does not say.
