# Changelog {#changelog}

[TOC]

This changelog contains a non-exhaustive list of new features and notable bug-fixes (not all bug-fixes will be listed).


<br/><br/>
# Unreleased

- API contract 2.0 frozen: docs/api-contract-2.0.md

### Added (Phase 4 · Task 4.1 — LR-core solver: Lagrangian root bound)

- New `dtwc::mip::lagrangian_root` (dtwc/mip/lagrangian_root.{hpp,cpp}): a matrix-free lower bound for the p-median (k-medoids) problem — the bound engine of the LR-core exact solver (UNIMODULAR.md §8.3). Dualizes the assignment constraints, computes the per-facility score `ρ_i(μ)=Σ_j min(0,D_ij−μ_j)` streaming D once per iteration, maximizes `L(μ)=Σμ_j+Σ_{S_k}ρ_i` by damped Polyak subgradient ascent, repairs a feasible primal each iteration via a k-medoids local search, and reports `{lower_bound, upper_bound, gap, medoids, labels, multipliers, n_core}` plus Beasley reduced-cost fixing. Dense-`D` core + a `Problem` overload. No new dependencies; OpenMP over the ρ pass.
- Verified against a brute-force IP oracle (validated on a hand-computed non-degenerate instance) and the HiGHS/Gurobi compact MIP: on well-separated clustered data the root bound closes to the optimum within 0.1% (prediction P1) and the primal repair recovers the exact optimum, agreeing with the exact MIP to 1e-6; valid bounds `lower_bound ≤ opt ≤ upper_bound` hold on 80 mixed uniform/clustered instances. Tests: tests/unit/mip/test_lagrangian_root.cpp.
- Tuning: CFM subgradient deflection (damps zig-zag) + throttled primal polish (cheap O(Nk) assignment every iter, full O(N²) medoid polish every 16). Net win on the LR-core-vs-compact-MIP bench (identical optima, MIP scales ~N^2.6 while LR stays 1–2 orders cheaper): e.g. N=200 certified in 352 iters vs the earlier 4000-iter cap. The primal is exactly optimal at every tested N; the subgradient dual *certificate* stalls at ~5e-5 for N=800 (subgradient slow tail at the non-smooth dual optimum — raising iters 4000→12000 barely moves it).
- New `dtwc::mip::lagrangian_root_kelley`: the SAME Lagrangian bound solved by a stabilized (boxstep trust-region) **cutting-plane** method — the right tool for the concave piecewise-linear dual. Accumulates each `(L, subgradient)` as a supporting hyperplane and maximizes the polyhedral model by a tiny LP master over the N multipliers + one scalar θ (warm-started; NOT the N²-column compact LP). Converges FINITELY: on the bench it certifies to machine precision in ~15 major iterations *flat in N*, where the subgradient stalls — N=800 goes from 4000 iters / 5.8e-5 (subgradient) to 15 major / 6.6e-14 (Kelley), ~64× faster and ~1000× faster than the compact MIP. Unstabilized Kelley was verified to stall (LB never rises — μ thrown to box corners); the boxstep trust region around the incumbent fixes it. Requires HiGHS for the master LP (throws `SolverError` otherwise; the subgradient variant stays the solver-free default).

### Added (Phase 4 · Task 4.2 — reduced-cost fixing: the LR-core)

- New `dtwc::mip::reduced_cost_fixing` (dtwc/mip/reduced_cost_fixing.{hpp,cpp}): Beasley reduced-cost fixing driven by the Task 4.1 dual state `(ρ, k, LB, UB)`. Two exact conditional-bound tests decide each candidate facility: forcing `i` open drops the dual to `LB + (ρ_i − ρ_(k))` (`> UB ⇒ fix i CLOSED`); forcing `i` closed drops it to `LB + (ρ_(k+1) − ρ_i)` (`> UB ⇒ fix i OPEN`). Returns `{core, fixed_closed, fixed_open}`; a fixed facility is provably absent from / present in EVERY optimal solution, not a heuristic prune. `LagrangianResult` now carries the surviving `core` (`n_core = core.size()`); `lagrangian_root`'s `finalize` computes it via the new module (the earlier inline count is replaced — one code path).
- Robustness: the fix tests clear `UB` by a magnitude-scaled margin `tol = 1e-9·(1+max(|LB|,|UB|))`, never on rounding. Without it, a certified instance (gap≈0, `LB≈UB`) whose facility merely TIES `ρ_(k)` — a legitimate alternative optimal medoid — could be pushed a few ULP over an exact `>` and wrongly eliminated. Caught by the correctness test before the margin was added (`N=14,k=2`: an optimal medoid with `ρ_m == ρ_(k)` to 9 s.f. fixed out).
- Verified (tests/unit/mip/test_reduced_cost_fixing.cpp): hand-checked close/open partition; on 80 non-degenerate instances (N≤14, clustered + uniform) NO facility in `fixed_closed` is in the brute-force optimum and every `fixed_open` IS — fixing never loses the optimum.
- **P2 band — FALSIFIED as a universal floor, CONFIRMED in the mean.** Registered: "root gap ≤1% ⇒ ≥80% of candidates eliminated for ≥90% of instances." Measured (36 qualifying clustered instances): mean elimination **80.3%**, min **73.3%**, and **77.8%** of instances reach ≥80%. The ~80% figure holds in the mean; the per-instance ≥80% floor does not — within a cluster several near-optimal medoid candidates sit inside the ≈0-gap band, so fixing correctly declines to remove them. Recorded, not rescue-tuned.

### Added (Phase 4 · Task 4.3 — exact LR-core: y-branching branch-and-bound)

- New `dtwc::mip::lagrangian_root_exact` (dtwc/mip/lagrangian_root.{hpp,cpp}): the EXACT p-median solve — "core Benders with y-only branching." Solves the root Lagrangian dual once (Kelley cutting-plane when HiGHS is present, subgradient otherwise), applies reduced-cost fixing (Task 4.2) to get the candidate `core` and the facilities proven open in every optimum, then closes any residual **integrality** gap by branch-and-bound that branches on the open/close (y) decision of one candidate at a time. Because the LR dual already equals the LP/Benders master bound (Geoffrion), there is no N²-column master to re-solve per round; each node's bound is the fixed-root-dual value `Σμ*+Σ_S ρ*_i ≤ cost(S)`, matrix-free. The optimum is a subset of the core (reduced-cost fixing provably never removes an optimal medoid), so the tree is exact. `params.max_nodes` caps the tree; on overflow the best incumbent is returned with `certified_optimal=false` and a loud stderr note — never a silent wrong "optimal". `LagrangianResult` gains `nodes` (B&B nodes explored).
- **Correctness gate MET (the 4.3 primary):** matches the brute-force IP optimum to 1e-6 on ALL tested instances — 8 clustered (root certifies, 0 nodes) and 24 adversarial uniform-`D` (nonzero integrality gap). On the adversarial set the B&B genuinely engages (tree branches on several instances, thousands of nodes) and still certifies the exact optimum: Kelley closes the *dual/LP* gap, the B&B closes the *integrality* gap. Tests: tests/unit/mip/test_lagrangian_root.cpp `[exact]`.
- **Wall-time (ADVISORY, shared machine):** exact LR-core vs the compact HiGHS MIP on clustered data — N=100 2.3 ms vs 185 ms, N=400 12 ms vs 6.5 s, N=800 **56 ms vs 58 s (~1000×)**; the margin widens with N (the root certifies in 0 nodes, compact MIP scales ~N^2.6). The "beats wall-time at N≥2000" clause is expected to hold on well-separated (real-world) data and to be FALSIFIED on the adversarial large-N regime (integrality gap ⇒ tree growth) — either way the LR root ships as the bound/certificate tool. The legacy `benders.cpp` (re-solve-master-per-round) is untouched and remains available under `benders="on"`.

### Added (Phase 4 · Task 4.4 — API: `Method::LRCore`)

- New `dtwc::Method::LRCore` routes `Problem::cluster()` to `dtwc::LR_core_clustering` (dtwc/mip/): fills the distance matrix, seeds an upper bound from the k-medoids heuristic, runs `lagrangian_root_exact` on a dense copy of D, and writes the proven-optimal `centroids_ind`/`clusters_ind`. `Method::MIP` keeps its meaning (solver-backed exact via Gurobi/HiGHS/Benders). Regime of validity: dense D in RAM — N·N doubles is the memory budget; needs no external MIP solver (uses HiGHS only for a tighter Kelley root when available). Exposed across surfaces (CasADi-style consistency): C++ enum, Python `dtwcpp.Method.LRCore`, CLI `--method lrcore` (alias `lr`). Test: `[lrcore][api]` drives `Problem::cluster()` to the oracle optimum.

### Added (Phase 4 · Task 4.5 — PDLP first-order LP arbiter)

- Bumped the pinned HiGHS `v1.14.0 → v1.15.1` (cmake/Dependencies.cmake, SHA256 re-pinned). v1.15.1 ships the PDLP first-order LP solver (`solver="pdlp"`, the cuPDLP-C port; and `"hipdlp"`, HiGHS's own PDHG) with an optional GPU/cuPDLP backend behind the HiGHS CMake flag `CUPDLP_GPU` (default OFF → our build uses CPU PDLP). Full ctest re-run on 1.15.1: **100% passed, 0 failed / 88** — no regression from the bump.
- New `dtwc::mip::pdlp_lp_bound` (dtwc/mip/pdlp_lp.{hpp,cpp}): forms the explicit p-median **LP relaxation** from a dense D and solves it with HiGHS PDLP, returning the LP optimum in raw distance units. This is the modern, maintained successor to the 2023 custom OSLP (removed in `3a87ea3` for being 3× slower than OSQP and unable to certify) — using a library instead of hand-rolled ADMM, per the project's prefer-libraries rule. LP-only: it returns a lower bound on the integer p-median cost, NOT a clustering; it does not replace the LR-core (which reaches the same bound *exactly* and matrix-free) or the compact MIP.
- **Purpose = independent arbiter (CLAUDE.md §4).** The LR-core reaches the p-median LP-relaxation optimum by *maximizing the Lagrangian dual* (matrix-free); PDLP reaches it by *first-order primal-dual on the explicit LP*. Geoffrion's TU theorem makes the two values equal, so their agreement cross-validates the clever matrix-free bound at N far beyond the brute-force IP oracle's reach.
- **BAND-ARB — CONFIRMED with ~5 orders of margin.** Registered before the run: `|pdlp − kelley| / max(1,|kelley|) ≤ 1e-4` on every one of 24 instances (clustered + uniform). Measured **max rel = 7.14e-09**. Also verified: PDLP's LP bound is a valid lower bound on the integer optimum on every instance, and is tight (== integer optimum) on well-separated clustered data (the LP is integral there). Tests: tests/unit/mip/test_pdlp_lp.cpp.
- No-silent-fallback: requesting `use_gpu=true` on a HiGHS built without `CUPDLP_GPU` warns to stderr and runs on CPU (never a silent GPU claim). `dtwc::mip::pdlp_gpu_available()` is a runtime capability query (the library reports its own build, since the compile define does not reach consumer TUs).
- **GPU backend enabled and runtime-verified on the RTX 4000 Ada (sm_89).** New CMake option `DTWC_HIGHS_GPU` (default OFF) forwards `CUPDLP_GPU=ON` to HiGHS and defines `DTWC_HIGHS_GPU` on the mip target. Built with the same known-good CUDA recipe as the DTWC CUDA kernels (nvcc 13.0 + MSVC host + `-allow-unsupported-compiler`; HiGHS goes shared on Windows → `highs.dll` + `cudalin.dll`). The `[pdlp][gpu]` test confirms `pdlp_gpu_available()==true`, `gpu_used==true` (the cuPDLP CUDA backend actually ran), and the GPU LP bound matches the integer optimum. The full test suite passes on both the CPU-PDLP build (89/89) and the GPU build (test_pdlp_lp 90/90). Default builds are unaffected — the option is OFF.
- **`gpu_used` corrected to reflect the build, not the request flag.** The bench (below) revealed that HiGHS `CUPDLP_GPU` is a *compile-time* device switch: on a GPU build `solver="pdlp"` always runs on the GPU, so a `use_gpu=false` solve on a GPU build ran on the GPU yet reported `gpu_used=false` — a false report. Fixed: `gpu_used = pdlp_gpu_available() && variant=="pdlp"` (build + variant, not the caller's request); `use_gpu` now only drives the CPU-build warning. New `[pdlp][gpu]` assertion checks `gpu_used == pdlp_gpu_available()` for BOTH `use_gpu` values.
- **PDLP-vs-Kelley bench recorded (`[.][pdlp][bench]`, Task 4.5 solver-comparison deliverable).** Run on both builds → `.claude/baselines/2026-07-08-pdlp-bench.md`. Registered bands: agreement (HARD) CONFIRMED — `|pdlp − kelley|` rel ≤ 7.7e-09 across all N and 24 arbiter instances (band 1e-4). Scaling (ADVISORY) CONFIRMED — the matrix-free Lagrangian dominates PDLP on the TU-structured p-median on *either* device: `pdlp/kelley` wall-time grows to 945× (CPU) / 126× (GPU) at N=400. GPU prediction PARTIALLY FALSIFIED (deliverable): GPU-PDLP has a ~255 ms launch floor (67× slower than CPU-PDLP at N=20) but crosses over near N≈150 and is 7.4× faster than CPU-PDLP by N=400 — still 126× slower than Kelley. Verdict: PDLP is a cross-validation arbiter, not a production p-median solver.

### Changed (Phase 5 · Task 5.1 — FastPAM1 O(N)-decomposition swap + FasterPAM)

- **Replaced the k-medoids SWAP phase (`dtwc/algorithms/fast_pam.cpp`), which was the naive O(N²·k) per-iteration loop, with the paper's FastPAM1 O(N)-decomposition (Schubert & Rousseeuw 2021, arXiv:2008.05171).** Each candidate's ΔTD over all medoids now comes from ONE O(N) pass — `ΔTD(m,x_c) = acc + ploss[m]`, with `acc = Σ_o min(0, d(x_c,o)−d₁(o))` and `ploss[m]` seeded from the removal loss `ρ(m)=Σ_{m₁(o)=m}(d₂−d₁)` — so an iteration is **O(N²) and parallel over candidates**. `fast_pam` now runs this by default (K-means++ BUILD unchanged). `dtwc::algorithms::PAMVariant { FastPAM1Naive, FastPAM1, FasterPAM }` + `fast_pam_swap(prob, initial_medoids, max_iter, variant)` expose all three (the naive is kept only as the bench baseline and digit-oracle).
- **New `FasterPAM` variant (Alg. 4, eager):** performs swaps as found and shares removal loss across medoids, converging in ≈1 sweep vs the best-swap variants' O(k) iterations. O(N²)/sweep but sequential (eager dependency). Never worse in objective; strictly better when the best-swap variant exhausts `max_iter` (measured at k=200).
- **Correctness (HARD, all CONFIRMED):** every variant converges to a **brute-force-verified local optimum** (independent full-reassignment ΔTD arbiter — tie-independent); the decomposition FastPAM1 reaches the **same objective** as the naive to 1e-9 (exact medoid identity is not required — the two group the ΔTD sum in different order, so an exact symmetric tie can pick a different but equal-cost medoid); FasterPAM ≤ FastPAM1 from an identical BUILD. Tests: `tests/unit/algorithms/unit_test_faster_pam.cpp` (240 assertions), plus the existing `unit_test_fast_pam`/`unit_test_fast_clara` green.
- **Fixed a k=1 NaN the change introduced:** with one medoid `second_dist = +inf`, so the removal-loss decomposition is undefined (`ρ=inf`, corrections=`−inf` ⇒ `NaN` ⇒ no swap accepted ⇒ the BUILD medoid returned instead of the optimum). k=1 is now special-cased to the direct `argmin_x Σ_o d(x,o)`. Caught by the CLARA exact-median oracle (`unit_test_fast_clara:619`), not the weak k=1 unit test — a k=1 regression test now covers all variants.
- **Bench (`[.][faster_pam][bench]`, ADVISORY) → `.claude/baselines/2026-07-08-faster-pam-bench.md`.** decomposition/naive speedup **2.3×–8.1×, growing with k** (parallel, objective-identical). The "≥10× faster" band is **NOT met at k≤50 and is explained, not rescued: DTW is memory-bound** (0.125 FLOP/byte) — both variants do N² distance-matrix lookups per iteration, and the decomposition only removes the naive's cheap O(k) *arithmetic*, so the speedup is the arithmetic-vs-memory ratio (small at low k, ~8× by k=200). FasterPAM converges in 1 sweep at every k and finishes where the best-swap variants hit `max_iter`; the parallel FastPAM1 wins wall-time at large N (85 ms vs 150 ms at N=2000, identical objective) → it is the right default. **LAB init intentionally skipped** (the paper says eager swapping nullifies it and recommends the K-means++ init already used); FasterCLARA inherited (CLARA calls `fast_pam`), carry-over refinement deferred.

### Added (Phase 5 · Task 5.2 — LB cascade upgrade: LB_Enhanced + LB_Webb)

- New tighter DTW lower bounds in `dtwc/core/lower_bound_impl.hpp`, templated on the pointwise metric (`L1Metric` + `SquaredL2Metric`): **`lb_enhanced`** (Tan, Petitjean & Webb, *SDM 2019*, elastic bands + LB_Keogh middle) and **`lb_webb`** / `lb_webb_symmetric` (Webb & Petitjean, *Pattern Recognition* 2021 — always ≥ LB_Keogh). `lb_webb` is **clean-room from Algorithm 2** (the authors' Java is GPL-3.0 and is NOT reproduced); it omits the paper's MinLRPaths corner DP and blanket-caps the tail free-flag — both LOOSEN, never break, the bound (validity is the hard gate). New `WebbEnvelope` (U, L, plus secondary `LU=L(U)`, `UL=U(L)`) built by reusing `compute_envelopes`. Wired into `LowerBoundStrategy { …, Enhanced, Webb }` and the Problem pruned cascade as selectable primitives.
- **HONEST REFRAME — the plan's "≥25% fewer full DTW calls on matrix build" band was NOT chased; it is unachievable as written.** `[confirmed]` from source: `fill_distance_matrix_pruned` builds an EXACT matrix, and the kernel's early-abandon returns the `maxValue` sentinel (never the exact value), so every abandoned pair is recomputed fully (partial + full > full). Since `lb ≤ dtw`, `lb > threshold ⇒ dtw > threshold ⇒` abandon always fires ⇒ always recomputes; a tighter LB pushes MORE pairs into the worse bucket and drives `computed_full_dtw` DOWN while doing MORE work. A lower bound skips a DTW only where the exact value is not needed (NN-search); an exact full matrix has no such pairs. Exact-matrix DTW-work reduction is **Task 5.3 (TADPole)** / **5.4 (PrunedDTW cell-pruning)**; Task 5.2 delivers the primitives those consume. The pruned strategy is currently a pessimisation for exact matrices (documented — see LESSONS).
- **Validity [HARD] → CONFIRMED.** `LB_Enhanced ≤ DTW_w` and `LB_Webb ≤ DTW_w` (L1 and SquaredL2) over random + adversarial inputs (shared endpoints, query-just-outside-envelope, constant, extreme 1e12), bands {0,1,2,5,10,20, 10%-of-n}, lengths incl. edge {2..11} and n≈2V; non-negativity and `LB(x,x)=0`. Provable and asserted: `LB_Webb ≥ LB_Keogh` per instance (Webb = one-dir LB_Keogh + non-negative Thm-2 corrections; symmetric ⇒ ≥ symmetric Keogh). **No `LB_Enhanced ≥ LB_Keogh` claim** — SDM 2019 proves no such ordering; the cascade takes the max, so a looser Enhanced never regresses. Test `tests/unit/adversarial/test_lb_enhanced_webb.cpp`: **39273 assertions / 14 cases**. Pruned + {Enhanced, Webb, Keogh} give a **digit-identical** matrix vs BruteForce (`unit_test_pruned_distance_matrix.cpp [strategy]`).
- **Tightness bench (`[.][lb_tightness][bench]`, ADVISORY) → `.claude/baselines/2026-07-08-lb-cascade.md`.** mean(LB/DTW) over 2480 clustered pairs, n=128: at **band=10%** keogh 0.550 → enhanced 0.568 (+3.3%) → **webb 0.676 (+23.0%)**; at **band=40%** keogh 0.406 → enhanced 0.450 (+10.9%) → **webb 0.550 (+35.6%)**. LB_Webb tightens the envelope cascade meaningfully at both bands; **LB_Enhanced's gain grows with band width** (its designed regime — "optimal V increases with W"), a minor win at 10% and the tool for wide bands. LB_Kim ≈ 0.016 (near-useless standalone; correct as the O(1) pre-filter).

### Added (Phase 5 · Task 5.3 — TADPole density-peaks clustering with admissible pruning)

- New **`dtwc::Method::TADPole`** (`dtwc/algorithms/tadpole.{hpp,cpp}`): density-peaks clustering (Rodriguez & Laio, *Science* 2014) with the admissible LB/UB DTW pruning of Begum, Ulanova, Wang & Keogh (*KDD 2015*, extended arXiv:1612.00637). This is the **only** method that does not materialise the full N×N matrix: the cutoff-kernel density `ρ_i = |{j≠i : d(i,j) < dc}|` is a binary per-pair test, so `UB(i,j) < dc ⇒ neighbour`, `LB(i,j) ≥ dc ⇒ not` skip the exact DTW (Begum Table 5 cases A–D); the δ step prunes candidate q when `LB(i,q) ≥ best` (Table 7). UB = the no-warp diagonal `Σ_t|x_t−y_t|` (equal-length only, valid since the diagonal always satisfies the band); LB = symmetric LB_Keogh reusing each series' envelope across all pairs. Centers = top-k by γ = ρ·δ; single-pass assignment to the nearest higher-density neighbour. A strict total order (ρ desc, index asc) makes δ, parents, γ-ranking and labels deterministic under ρ-ties; δ(global densest) = max of the others' δ (Begum's convention). Non-Standard/multivariate/NaN variants and unequal-length pairs fall back to exact DTW (result unchanged, pruning degrades). Exposed CasADi-style: C++ enum, Python `dtwcpp.Method.TADPole`, CLI `--method tadpole` (+ `--dc`); `Problem::tadpole_dc` (<0 ⇒ auto-select from a deterministic DTW subsample).
- **SCOPE (user decision).** The plan's registered path — reuse TADPole's pruning *inside* k-medoids — is **not admissible**: PAM (`fast_pam_swap` fills the full matrix and the SWAP loop reads every candidate pair), MIP and LR-core all consume the whole matrix, so no lower/upper bound can skip a DTW there (the same wall as the Task 5.2 reframe). A bound only skips work for a consumer that never needs exact far-pair distances — an NN/density search. The registered "labels digit-identical, ≥50% pruned" band is therefore reachable only by *adding* such a consumer; Begum 2015 is density-peaks clustering. The plan file `pruned_matrix_build` became `tadpole` (the pruning is internal to the clusterer). Here the Task 5.2 tighter LBs would *help* (`LB ≥ dc ⇒ prune`), the opposite of their exact-matrix pessimisation.
- **Admissibility [HARD] → CONFIRMED.** `tadpole(prune=true)` and `tadpole(prune=false)` return digit-identical labels + medoids and total_cost within 1e-9·max(1,cost) (Begum Theorem 1), and both equal an INDEPENDENT brute-force density-peaks oracle re-implemented in the test. **Bounds [HARD] → CONFIRMED:** `LB_Keogh ≤ DTW ≤ Σ|x−y|` on every equal-length pair. Quality: on two well-separated clusters (interleaved by index) k=2 recovers the exact ground-truth grouping. Edge cases k=1 / k=N / N=1 / identical series / variable length all pass. Test `tests/unit/algorithms/unit_test_tadpole.cpp`: **6 cases, 2572 assertions**. Full gate **92/92** (baseline 91 → +1 suite), no regression.
- **Pruning bench (`[.][tadpole][bench]`, HARD floor 0.50) → CONFIRMED at 78.0%.** N=200, len=64, band=10%, dc=24.74 (auto): **dtw_calls = 4372 / 19900 pairs → 78.0% of brute-force DTW work avoided** (16000 pruned by LB, 247 by UB). Paper reports 80–88%. `.claude/baselines/2026-07-08-tadpole.md`.

### Added (Phase 5 · Task 5.4 — EAPruned exact DTW kernel)

- New **`dtwc::core::dtw_kernel_eap`** (dtwc/core/dtw_kernel.hpp) + public shim **`dtwc::dtwFull_eap`** (dtwc/warping.hpp): unbanded, EXACT standard DTW with cell pruning (Herrmann & Webb, *DMKD* 35(6) 2021, arXiv:2102.05221; DTW precursor arXiv:2010.05371). Seeds an upper bound = the cost of the diagonal "L-path" (diagonal to the corner, then straight along the last short-row, O(n_long)); every optimal-path cell has partial-cost ≤ DTW ≤ UB, so pruning every cell whose DP value exceeds UB is provably EXACT while skipping the cost-matrix corners the diagonal already beats. Rolling buffers O(n_short); a per-row live window `[comp_start, pp)` tracks the diagonal. Standard recurrence + L1/SquaredL2 only. Unlike LB-guided early-abandon (which recomputes the full DTW on abandon — the exact-matrix pessimisation noted in the 5.2/5.3 lessons), EAP returns the exact value in one pass AND does less work: it is the exact-matrix DTW-work lever the Task 5.3 run-log promised for the k-medoids / MIP consumers.
- **Wiring.** `make_standard` (dtw_dispatch.cpp) routes its scalar UNBANDED path (`band < 0`, the default since `settings::DEFAULT_BAND == -1`) through `dtwFull_eap`; banded builds keep `dtwBanded` (the Sakoe-Chiba band already excises the region EAP would prune). Multivariate + non-Standard variants (DDTW/WDTW/ADTW/Soft/AROW) untouched. No new enum / CLI / Python surface — a drop-in faster exact kernel for the default matrix build.
- **Fast-math fix (bug found + fixed before merge).** The build compiles with `-ffast-math`; reassociation of the DP vs UB sums could push the optimal FINAL cell a few ULP over `ub`, spuriously pruning it → `+inf`. Prune instead against a relaxed `thr = ub·(1 + n_long·16·ε)` — provably still exact (relaxing the prune only ADDS cells; it never drops an optimal-path cell nor changes the returned recurrence value).
- **BAND-EXACT [HARD] → CONFIRMED.** `dtwFull_eap` == `dtwFull_L` to rel ≤ 1e-12 / abs ≤ 1e-9 over random + warped + edge (identical, n=1, monotone, anti-correlated, constant, very-unequal) pairs, L1 + SquaredL2 — `tests/unit/adversarial/test_eap_dtw.cpp`, **2511 assertions / 4 cases**. Additionally the **entire 93-test gate** now builds all unbanded Standard distance matrices through EAP and passes digit-identically (end-to-end). **BAND-UB [HARD] → CONFIRMED:** diagonal L-path UB ≥ DTW on every pair. Full gate **93/93** (baseline 92 → +1 suite), no regression.
- **Speedup bench (`[.][eap][bench]`, ADVISORY floor 1.5×, warm/order-bias-free) → CONFIRMED.** Kernel wall-time `dtwFull_L / dtwFull_eap`, unbanded, lengths {128,256,512,1024}: **near-diagonal 6.26×→12.17×** (cells computed 21%→11% — the pruning mechanism), mild-warp 5.07×→9.89×, unrelated 1.46×–1.69× (≈98% cells — the leaner inner loop, NOT pruning). Real matrix-build speedup scales with data cohesion (within-cluster pairs prune hard, cross-cluster don't); never slower than plain DP in any measured case. `.claude/baselines/2026-07-08-eap.md`.

### Added (Phase 5 · Task 5.5 — MSM + TWE elastic distances)

- New metric elastic distances as DTW variants: **`dtwc::core::msm_distance`** (dtwc/core/msm.hpp; Move-Split-Merge, Stefan/Athitsos/Das IEEE TKDE 2013) and **`dtwc::core::twe_distance`** (dtwc/core/twe.hpp; Time Warp Edit, Marteau IEEE TPAMI 2009). The 2024 KAIS clustering evaluation (Holder, Middlehurst & Bagnall) ranks MSM the single best k-medoids clustering distance while DTW is barely better than Euclidean — so these are a QUALITY lever. Standalone O(n·m) DP with a rolling buffer over the shorter axis (O(min(n,m)) scratch — a full matrix would be ~512 MB/thread at n≈8k); both are metrics hence symmetric. Recurrences match **aeon 1.5.0** exactly.
- **Wiring (CasADi-style parity):** `DTWVariant::{MSM,TWE}` + params `msm_c` (default 1.0), `twe_nu` (0.001), `twe_lambda` (1.0); dispatch (`make_msm`/`make_twe`), the `dtw_runtime` switch, the `dtwc::distance::{msm,twe}` facade, and checkpoint enum↔string; CLI `--variant msm|twe` with `--msm-c/--twe-nu/--twe-lambda` (+ TOML keys); Python `DTWVariant.MSM/TWE`, `DTWVariantParams.{msm_c,twe_nu,twe_lambda}`, and the `KMedoids` wrapper.
- **v1 scope (documented, not silent):** univariate + unbanded. A multivariate request is rejected at bind time with a clear `InvalidInput` (never silently collapsed); `Problem::band` is intentionally ignored (full elastic metric — the default build is unbanded).
- **BAND-ORACLE [HARD] → CONFIRMED:** MSM/TWE == aeon 1.5.0 to rel ≤ 1e-10 on 20 non-degenerate pairs each. **BAND-METRIC [HARD] → CONFIRMED:** d≥0, d(x,x)=0, symmetry, triangle inequality (200 triples ×2). **BAND-WIRING [HARD] → CONFIRMED:** Problem matrix == direct kernel; CLI runs end-to-end. Test `tests/unit/core/unit_test_msm_twe.cpp`: **5 cases, 1676 assertions**. Full gate **94/94** (baseline 93 → +1 suite), no regression.
- **BAND-SPEED [ADVISORY, ≤1.3× plain DTW DP]:** MSM 1.01–1.06× (comfortable), TWE 1.26–1.30× (at the boundary — TWE's cell does strictly more work: front padding + two |·| terms + the 2ν|i−j| penalty). `.claude/baselines/2026-07-08-msm-twe.md`.
- **Known limitation:** Python runtime parity for MSM/TWE is not yet verified — the wheel could not be rebuilt in this environment (a pre-existing scikit-build generator / llfio ExternalProject failure, independent of this change); the C++ core, CLI, and aeon oracle are fully gated.

### Changed (Phase 3 · wave A — no silent fallback)

- Build: OpenMP is now a hard configure requirement. A missing OpenMP aborts configuration with a FATAL_ERROR that names the `-DDTWC_ALLOW_SEQUENTIAL=ON` opt-out, ending silent single-threaded builds/wheels. The opt-out configures a loud-warning sequential build and defines `DTWC_SEQUENTIAL_BUILD` on the dtwc targets.
- Build: the OpenMP compile flag/link is now attached directly (PUBLIC) to the `dtwc++` target, not only via the `project_options` INTERFACE, so consumers linking `dtwc++` without `project_options` (e.g. the nanobind module on MSVC) no longer silently serialise their OpenMP loops.
- Parallelism: `dtwc::Env` now emits one loud stderr warning when DTWC++ would run single-threaded — OpenMP forced to 1 thread on a multicore host, or compiled without OpenMP via `-DDTWC_ALLOW_SEQUENTIAL=ON` (`DTWC_SEQUENTIAL_BUILD`). No silent serial execution (Task 3.2).
- GPU: GPU->CPU distance-matrix fallback messages are now always written to stderr (previously `if (verbose)`-gated on stdout), so a requested GPU path that degrades to CPU is never silent (Task 3.2).
- CI/wheels: add `CIBW_TEST_COMMAND` that imports every built wheel and asserts `dtwcpp.OPENMP_AVAILABLE` — a runtime belt-and-braces for the OpenMP no-silent-fallback guarantee, complementing the Task 3.1 build-time FATAL_ERROR. Uses the existing bound introspection symbol, not the future `dtwcpp.test.*` API.
- CI/wheels: drop macOS x86_64 (Intel) wheels (`CIBW_ARCHS_MACOS: arm64`). The arm64 `macos-latest` runner installs an arm64-only libomp, so cross-built x86_64 wheels would ship serial and cannot be import-tested by cibuildwheel; with Task 3.1's OpenMP FATAL_ERROR they would hard-fail the build outright. Intel-Mac users install from the sdist. Re-add x86_64 only behind a genuine x86_64 libomp on an Intel runner.

### Changed (Phase 3 · wave B — introspection & CUDA verification)

- Add `dtwc.test` self-introspection API with an identical result schema in C++ (`dtwc::test::parallelisation()`/`gpu()`, header-only `dtwc/test_api.hpp`), Python (`dtwcpp.test.parallelisation()`/`gpu()`) and MATLAB (`dtwc_mex('test_parallelisation'|'test_gpu')` + `dtwc.test.*`): `parallelisation()` proves engagement by running a real OpenMP region and counting distinct thread ids (fields available, max_threads, threads_engaged, pass, reason); `gpu()` executes a tiny real GPU kernel validated against a CPU oracle (fields available, backend, device_name, validated, pass, reason), naming exactly what is missing when no backend/device is present and never silently degrading.
- Report Metal availability in `dtwcpp.check_system()` (metal_available was bound but previously unreported), matching the MATLAB `dtwc.check_system` report.
- Build: CUDA GPU acceleration now builds and is runtime-verified on Windows (nvcc 13.0 + MSVC 14.50) via `-DDTWC_ENABLE_CUDA=ON`. The fat-binary arch list (incl. sm_89 Ada, sm_90 H100) pre-dates this change (2026-04, commits 3550bb9/367a9b4) — no new dispatch code was needed; the deliverable is the working Windows build recipe plus the verification below.
- Fix: add missing `#include <numeric>` in tests/unit/test_cuda_correctness.cpp (std::iota) so the CUDA correctness suite compiles under MSVC's STL (latent — the suite had never been compiled before CUDA was enabled).
- Test: first-ever runtime verification of the Phase 0 CUDA audit fixes (wavefront max_L>2048 3-buffer routing; int64 pair indexing) on NVIDIA RTX 4000 Ada — test_cuda_correctness (55 cases) and test_cuda_lb_keogh (8 cases) pass with 0 failures (verbatim run log: `.claude/baselines/2026-07-07-cuda-first-runtime-verification.md`).

### Fixed (Phase 3 · review H1 — no silent single-thread fallback)

- Parallelism: the single-thread warning now fires from the compute entry points, not only the `dtwc::Env` constructor. Every compute path funnels through `dtwc::get_max_threads()`, which now emits the warning; the high-level Python distance-matrix function warns directly too. Previously a build with OpenMP present but only 1 usable thread (`OMP_NUM_THREADS=1`, common on SLURM/containers) ran SILENTLY single-threaded whenever the caller never constructed `dtwc::env()` — i.e. `dtwcpp.compute_distance_matrix`/`cluster` without a `dtwcpp.device()` call, or direct C++ `Problem` use. The emitter is process-once and shares one guard with the `Env` constructor, so a front-end that does both (the CLI) still warns at most once. The duplicated hard-coded "OpenMP not available" string in `parallelisation.hpp` is removed — the warning text is now the single source of truth in `env.cpp`.

### Changed (Python · Phase 2 Task 2.1)

- Python (Task 2.1): bound the full 2.0 canonical Python surface in the nanobind module — error taxonomy `DtwcError`/`InvalidInput`/`SolverError`/`DeviceError`/`IOError` (each subclassing DtwcError plus the closest built-in ValueError/RuntimeError/OSError, with C++-exception translators, api-contract §5); `dtwc::Env`/`Device`/`env()`/`device_to_string` device registry (§6); `StoragePolicy`/`LowerBoundStrategy` enums and `CUDASettings`; `MIPSettings.benders`/`max_benders_iter`; and `Problem.{set_solver, set_variant_params, set_view_data, output_folder, storage_policy, lb_strategy, cuda_settings, use_mmap_distance_matrix, read_distance_matrix, print_distance_matrix, write_medoid_members, n_clusters, labels, medoids, series, series_name, centroid_of}`, plus multivariate `ndim` and float32 storage (`Data.from_float32`, `Problem.set_data(Data)`) in the load path.
- Python: unified distance-matrix access — `Problem.distance_matrix()`/`set_distance_matrix()` are canonical (old `distance_matrix_numpy`/`set_distance_matrix_from_numpy` kept one cycle as deprecated aliases, §4). All pairwise distance functions now take zero-copy float64 ndarrays uniformly — `ddtw`/`wdtw`/`adtw`/`soft_dtw`/`soft_dtw_gradient` were `std::vector` copies (§2.6).
- Python: canonical scores `davies_bouldin`/`dunn`/`calinski_harabasz`/`adjusted_rand`/`normalized_mutual_info` added; the `*_index`/`*_information` spellings kept one cycle as deprecated aliases (§2.4/§4).
- Python: deleted the wrapper-side result auto-wiring in the nanobind `fast_pam`/`fast_clara`/`clarans` bindings — the C++ core writes labels/medoids/k back into `Problem` since 1.6 (§2.5); behaviour preserved end-to-end.
- Python: Tier-1 `ClusterResult` renamed to `Result` (deprecated alias kept), gaining `medoids` (deprecated `medoid_indices` alias), `score(name)` (silhouette mean / davies_bouldin / dunn / calinski_harabasz / inertia; unknown -> InvalidInput), and `save(dir)` (4 result CSVs); `DTWClustering` gained the `metric='l1'` constructor param (parity with MATLAB, §1.5). `dtwcpp.device()` mirrors the selection into the shared `dtwc::Env` registry (§6).

### Changed (MATLAB · Phase 2 Task 2.2)

- MATLAB: add Tier-1 API (dtwc.device/load/cluster, dtwc.Dataset, dtwc.Result) delegating device selection to dtwc::Env with no silent fallback (api-contract-2.0.md §1, §6).
- MATLAB: extend dtwc.Problem with snake_case config setters (set_method/set_band/set_max_iter/set_n_repetitions/set_solver/set_lb_strategy/set_storage_policy/set_output_folder/set_mip_settings/set_cuda_settings/set_verbose), 2.0 methods (refresh_distance_matrix/read_distance_matrix/max_distance/distance_matrix/cluster) and read accessors (size/n_clusters/name/labels/medoids); PascalCase properties retained as aliases (§2.1-§2.2).
- MATLAB: set_data now accepts ragged cell arrays, series names, and ndim multivariate input, validated before any mx dereference (§2.1).
- MATLAB: add snake_case score names davies_bouldin/dunn/calinski_harabasz/adjusted_rand/normalized_mutual_info (§2.4) and the checkpoint surface CheckpointOptions/save_checkpoint/load_checkpoint/save_binary_checkpoint/load_binary_checkpoint (§2.7).
- MATLAB: dtwc.DTWClustering gains a Device parameter delegating to dtwc::Env (§1.5).
- MEX: map dtwc error taxonomy (InvalidInput/SolverError/DeviceError/IOError) to dtwc:invalidArgument/solverError/deviceError/ioError, keeping the std fallbacks so the 19 pinned input-validation cases still fire dtwc:invalidArgument (§5).
- MATLAB: add tests/matlab/test_contract_parity.m asserting every api-contract-2.0.md MATLAB-column symbol is callable.

### Changed (CLI · Phase 2 Task 2.3)

- **CLI/TOML flag conformance to api-contract-2.0.md.** Renamed `--clusters` -> `--n-clusters` (§1.5/§2.1 `n_clusters`; `-k` kept as the canonical short form) and `--restart` -> `--resume` (§2.7). The old spellings — as CLI flags AND as `--config` TOML / `--yaml-config` config keys (`clusters`, `restart`) — remain **accepted** but each emits one stderr deprecation warning per use: `[dtwc] warning: '<old>' is deprecated, use '<new>' instead`. The canonical spelling wins when both are supplied; deprecated flags are hidden from `--help`. The rename SSOT lives in `dtwc/dtwc_cl.cpp::cli_renames()`.
- The two known CLI callers (`scripts/slurm/jobs/cluster_generic.slurm`, `python/dtwcpp/_hpc.py::build_dtwc_command`) already compose only canonical flags (`-k`, `--skip-cols`, `--dtype`, `--method`, `--band`/`-b`, `--device`/`-d`, `--name`, `--output`/`-o`, `--input`/`-i`, `--verbose`/`-v`) and emit zero deprecation warnings (verified).

### Added (conformance · Phase 2 Task 2.4)

- Add cross-language conformance fixture (tests/conformance/): the permanent Phase 2 parity gate. One recorded dataset -> banded DTW (band=3) -> FastPAM k=3 -> silhouette/davies_bouldin/dunn, run from C++, Python, MATLAB and the CLI, asserting digit-identical canonical labels/medoids and scores within 1e-12 rel against a C++-recorded reference (docs/api-contract-2.0.md §9).

### Changed (Phase 1 · wave 2)

- Precision unification (Task 1.5): `dtwc::settings::default_data_t` default template scalar flipped `float` -> `double`, so `double` is now the default on every public `dtwc::distance::*`/`dtwBanded`/`soft_dtw` helper (api-contract-2.0.md §8, rename row 41). Explicit `float`/`Precision::Float32` remains a fully supported opt-in.
- CLI: `--dtype` default flipped `float32` -> `float64` (full precision by default). `float32`/`f32`/`fp32`/`float` remain accepted opt-ins via the unchanged CheckedTransformer map (api-contract-2.0.md §8, rename row 42).
- Docs: corrected `core/storage.hpp` `Precision` enum comment (Float64 is the default per `Data.hpp:38`, not Float32) and the stale "currently `float`" `default_data_t` notes in `core/dtw.hpp` and `soft_dtw.hpp`.
- Added `dtwc::Env` device registry (`dtwc/env.{hpp,cpp}`): `set_device()`/`device()`/`threads()` + process-wide singleton `dtwc::env()`. Selects `cpu`/`gpu`/`hpc` with no silent fallback — an unknown device name, `gpu` on a build with no GPU backend, and the three `device="hpc"` `.env` failures (missing file / missing key / auth failure) each raise `dtwc::DeviceError` with the actionable message from api-contract-2.0.md §6. CLI `--device` now forwards to `dtwc::env()`, so a GPU request on a CPU-only build errors up front instead of quietly running on CPU. (Task 1.3)
- Storage: `StoragePolicy::Auto` now routes at load time — `DataLoader::load_stored()` estimates the dataset footprint (rows x lengths x sizeof(data_t)) and spills to the mmap-backed `MmapDataStore` (returning a view into it) when it exceeds the threshold (default 50% of free RAM; override via `DataLoader::ram_limit()`/`mmap_cache_path()`), keeping small datasets on heap. View-mode spans (CLARA subsample path) unchanged. (Task 1.4)
- Storage: `device='hpc'` now performs a metadata-only local load — `DataLoader::load()`/`load_metadata()` read shapes/counts/names without materialising any payload (the bulk reader is never invoked), and `Data::series()`/`series_f32()` throw an actionable "data not resident locally" error for local access; bulk series stream to the SLURM cluster at submit. (Task 1.4)
- **Task 1.6 — `Problem`/`scores` 2.0 API cleanup (C++).** Canonical snake_case names with `[[deprecated("use X")]]` inline shims forwarding to them (old names still compile with a deprecation warning, identical behaviour): `Problem::{set_n_clusters, n_clusters, refresh_distance_matrix, dist_by_ind, max_distance, is_distance_matrix_filled, fill_distance_matrix, print_distance_matrix, find_total_cost, assign_clusters, calculate_medoids, cluster_by_mip, cluster_by_kmedoids_lloyd}` and `scores::{davies_bouldin, dunn, calinski_harabasz, adjusted_rand, normalized_mutual_info}`. Added canonical setters/accessors `set_method`, `set_band`, `set_max_iter`/`max_iter()`, `set_n_repetitions`/`n_repetitions()`, read accessors `labels()`/`medoids()`, and additive snake_case forwarders for the Problem_IO writers (`read_distance_matrix`, `write_distance_matrix`, `print_clusters`, `write_clusters`, `write_medoid_members`, `write_silhouettes`). `Problem::resize()` is now private.
- **Task 1.6 — result write-back into `Problem` (core).** `fast_pam`, `fast_clara` (in-RAM + Parquet-chunked), `clarans`, and `cut_dendrogram` now write `labels`/`medoids`/`k` back into the `Problem` in C++ (mirroring the Python/MATLAB wrapper auto-wire), so `scores::silhouette(prob)` and the other `scores::*` work in pure C++ immediately after clustering with no manual wiring. `set_variant(...)` remains the invariant-preserving path that rebinds `dtw_fn_`.

### Added (Phase 1 · error taxonomy)

- **New header [dtwc/error.hpp](dtwc/error.hpp)** — a small, header-only exception hierarchy rooted at `dtwc::Error : std::runtime_error`, with `dtwc::InvalidInput`, `dtwc::SolverError`, `dtwc::DeviceError`, and `dtwc::IOError` deriving from it. Message-preserving (constructors inherited from `std::runtime_error`); no error codes, no macros. Because every type derives from `std::runtime_error`, existing `catch (const std::runtime_error &)` / `catch (const std::exception &)` handlers — and tests that pin those types — keep working unchanged (task 1.2).
- **Migrated to the taxonomy** (task 1.2):
  - `soft_dtw_gradient` (dtwc/soft_dtw.hpp) now throws `dtwc::InvalidInput` on empty input. The former `assert(mx > 0 && my > 0)` was a no-op under `NDEBUG`, letting an empty span fall through to an out-of-bounds read of `x[0]`/`y[0]`.
  - The MIP solver status checks in [dtwc/mip/mip_Highs.cpp](dtwc/mip/mip_Highs.cpp) (model rejected / run failed / non-optimal status) and [dtwc/mip/mip_Gurobi.cpp](dtwc/mip/mip_Gurobi.cpp) (non-optimal status, `GRBException` wrap, unknown-exception wrap) now throw `dtwc::SolverError` instead of a bare `std::runtime_error`. Behaviour is unchanged for callers catching `std::runtime_error` (the Phase 0 task 0.5 regression test still passes).
- **New tests** in [tests/unit/test_error_taxonomy.cpp](tests/unit/test_error_taxonomy.cpp): constructibility + `what()` preservation, catchability up the hierarchy, sibling-type distinctness, and a live-path assertion that `soft_dtw_gradient` on an empty series throws `dtwc::InvalidInput` with the expected message.

### Fixed (Phase 0 correctness & security audit)

Hardening pass from the full-repo audit; each item ships with regression tests.

- **Pair-index decode SSOT** — replaced three divergent linear-upper-triangle decoders (CUDA int32, MPI `size_t`, Metal FP32-`sqrt`; the FP32 copy returned out-of-bounds pairs past N≈4096 and every int32 copy overflowed the `i*(2N-i-1)` intermediate at N≥46341) with a single `dtwc::detail::decode_pair` (FP64 seed + 64-bit integer correction loop). The C++ SSOT is compiled directly into the **CUDA and MPI** paths; Metal mirrors it as an integer-only MSL string (`kDecodePairMSL`) in the same header — bit-identical by construction but verified by **inspection only** (no local GPU CI) (tasks 0.2, 0.7).
- **CUDA wavefront long-series cap** — task 0.1 lifts the anti-diagonal cell cap (`MAX_SI`×blockDim = 2048) that silently truncated anti-diagonals for `max_L > 2048`, producing wrong DTW on long series. Inspection-verified locally; runtime confirmation needs H100 CI (task 0.1).
- **mmap stores** — `MmapDistanceMatrix`/`MmapDataStore` now reject sizes that overflow the packed layout and out-of-bounds interior offsets instead of mapping past the file (task 0.3).
- **MATLAB MEX inputs** — `dtwc_mex` validates argument types, rejecting int32/single/complex/logical/empty/struct/sparse inputs rather than misreading raw bytes (task 0.4).
- **MIP HiGHS** — a non-optimal (infeasible) solve now throws instead of silently returning an empty result (task 0.5).
- **DTW runtime dispatch** — Soft-DTW actually computes Soft-DTW (was Standard-L1) and throws on `gamma <= 0` (task 0.6); multivariate L2 is now a true Euclidean per-step cost, distinct from L1, on the **live** `dtwBanded_mv`/`dtwFull_L_mv` dispatch path. (Task 0.6's L2 fix had landed on a dead duplicate dispatcher — `core::dispatch_mv_metric`, zero call sites — so the live multivariate path still aliased L2→L1; the duplicate dispatcher was deleted and the live `dtwc::detail::dispatch_mv_metric` corrected in **R1**.)
- **Arrow/Parquet readers** — reject Float32 lists mislabelled as Float64, `ndim=0` metadata (div-by-zero), and out-of-bounds list offsets; scalar and list Float32 columns are now converted to double (task 0.8).
- **CLI device parsing** — `parse_device` handles `cpu`/`cuda`/`cuda:N` case-insensitively, rejects unknown devices (no silent CPU fallback) and non-L1 metrics on the CPU path (task 0.9).
- **TimeSeries views** — `view()` and explicit conversion preserve `ndim` for multivariate series (task 0.10).
- **FastCLARA sampling** — deterministic `mt19937_64` + `std::sample` seeding for reproducible in-RAM subsampling (task 0.11).
- **Supply-chain pinning** — `Dependencies.cmake` pins llfio/quickcpplib by SHA with URL hashes; CI no longer pipes a remote script into a shell (task 0.12).
- **Type utilities / build** — corrected `types_util.hpp`, README, and benchmark CMake wiring (task 0.13).
- **Python `cluster()` dispatch** — unknown methods raise before any HPC offload; local dispatch routes CLARA vs FastPAM correctly and normalizes the `hclust` alias (task 0.14).

### Added (unified `device()` → `load()` → `cluster()` → `result.plot()` interface)

- **One high-level flow** in new [python/dtwcpp/_api.py](python/dtwcpp/_api.py): set the device once, then cluster — the library handles device resolution, local-vs-remote execution, and plotting, so callers never touch internal plumbing:

  ```python
  import dtwcpp as dtwc
  dtwc.device("hpc")                  # cpu | gpu | hpc  (set once, sticky)
  data = dtwc.load("Crop_TRAIN.tsv")  # lazy handle — NOT read locally on hpc
  res  = dtwc.cluster(data, k=3)      # local for cpu/gpu; offloaded for hpc
  print(res.summary()); res.plot()
  ```

- **`dtwcpp.load(source, skip_cols=…)`** returns a lazy `Dataset` (a path or array); the file is only read when a *local* backend needs it. On `device="hpc"` a path is passed straight to the cluster and never read locally — so it scales past what fits on the calling machine.
- **`dtwcpp.cluster(data, k, …)`** reads the global device (or a per-call `device=`), runs FastPAM k-medoids locally for cpu/gpu (full distance matrix on the chosen device) or offloads the whole job for hpc, and returns a **`ClusterResult`** with `labels`, timing, `cost`/`medoid_indices` (local), and `summary()` / `plot()` (classical-MDS 2D scatter, moved out of the example into the library).
- [examples/python/09_device_clustering.py](examples/python/09_device_clustering.py) rewritten to this flow — ~6 lines of logic, no `resolve_device` or other internals leaked to the user.
- Covered by 12 tests in [tests/python/test_api.py](tests/python/test_api.py): lazy load (a missing path doesn't raise), local group recovery, result fields/summary, plot output, and that `device="hpc"` **does not read a path source locally**.

### Added (PyTorch-style device selection: `gpu`/`hpc` names + global `dtwcpp.device()`)

- **Friendly device names.** [python/dtwcpp/__init__.py](python/dtwcpp/__init__.py) `_parse_device` now accepts `"gpu"` (alias for `"cuda"`, so it auto-falls-back to CPU with a warning when no GPU is present) and `"hpc"` (an *execution location*, not a local compute backend). The previous `"cpu"`/`"cuda"`/`"cuda:N"` spellings are unchanged.
- **Global default device.** New `dtwcpp.device(name)` setter and `dtwcpp.get_device()` getter mirror PyTorch's `set_default_device`: `dtwcpp.device("gpu")` makes subsequent calls default to GPU. `compute_distance_matrix(..., device=None)` (new default) resolves `None` to the global default; an explicit `device=` argument always overrides it. Default remains `"cpu"`, so existing behavior is unchanged.
- **`hpc` is rejected by `compute_distance_matrix`** with a clear message — you cannot compute a matrix "on hpc" locally; it offloads the whole job. Use the high-level clustering path (`device="hpc"`) instead.
- Covered by 11 unit tests in [tests/python/test_device.py](tests/python/test_device.py) (gpu alias + CPU fallback, global get/set, explicit-overrides-global, hpc rejection).

### Added (`device="hpc"` — offload the whole clustering job to a SLURM cluster)

- **`DTWClustering(device="hpc").fit(X)`** serializes the series, submits a job to a SLURM cluster (e.g. Oxford ARC), waits, downloads the result, and returns `labels_` — all from Python. New module [python/dtwcpp/_hpc.py](python/dtwcpp/_hpc.py) owns the Python side: `write_series_tsv` (one series per row, `--skip-cols 0`), `parse_labels_csv` (maps dtwc_cl's 1-based `name,cluster` output back to **input order** — robust to the binary's lexical row-sorting), `build_dtwc_command`, `find_dtwc_binary`, and a `SlurmRemoteRunner` that drives the existing `scripts/slurm/slurm_remote.sh`. Only `labels_` is populated on the `hpc` path (the cluster computes remotely); `predict` needs a local fit.
- **New generic SLURM job** [scripts/slurm/jobs/cluster_generic.slurm](scripts/slurm/jobs/cluster_generic.slurm) — parametrized via `DTWC_INPUT`/`DTWC_K`/`DTWC_METHOD`/`DTWC_DEVICE`/`DTWC_BAND`/`DTWC_NAME` env vars (the previous job scripts hardcoded Coffee/Beef). It discovers a `build-*/bin/dtwc_cl` (preferring a GPU build when `device=cuda`) and writes output where `download` retrieves it.
- **New wrapper subcommand** `slurm_remote.sh submit-cluster <input> <k> [method] [device] [band] [name]` — uploads an arbitrary input file and submits the generic job (adding `--gres` for GPU runs). The Python side passes a **repo-relative** input path so Git Bash `rsync` doesn't misread a Windows `C:/…` path as a `host:path` target.
- New end-to-end example [examples/python/09_device_clustering.py](examples/python/09_device_clustering.py): one `device` switch over `cpu` / `gpu` / `hpc`, explicit full distance matrix, FastPAM, timing, and a classical-MDS 2D cluster plot.
- Covered by 12 tests in [tests/python/test_hpc.py](tests/python/test_hpc.py): serialization round-trip, label parsing (incl. lexical-order robustness + missing-series error), command construction, runner Job-ID parsing / poll loop, `device="hpc"` dispatch, and a **real round-trip against a local `dtwc_cl`** (the cluster job minus ssh/rsync). The remote submission itself must be verified on the cluster. Full Python suite: **201 passed, 10 skipped**.

### Added (Python preprocessing helpers for ZOH-corrupted telemetry)

- **New `dtwcpp.preprocess` module** in [python/dtwcpp/preprocess.py](python/dtwcpp/preprocess.py) with `strip_idle`, `decimate_zoh`, `sg_smooth`, `derivative`, `z_normalize`, and a chained `power_signal` entry point. Targets time-series that DTW handles poorly out-of-the-box: leading/trailing idle plateaus, zero-order-hold staircases from sub-sampled CAN-bus logging, and magnitude-dominated raw signals where shape is the variable of interest. `sg_smooth` lazy-imports `scipy.signal.savgol_filter` and raises a clear `ImportError` if scipy is missing; all other helpers are pure numpy + the existing C++ `z_normalize`. Docstrings call out that the chain is for **shape** clustering — if you want to cluster by magnitude/energy intensity, use raw DTW or feature-based methods instead. Covered by 24 unit tests in [tests/python/test_preprocess.py](tests/python/test_preprocess.py).

### Added (Python sanity-check and feature-baseline helpers)

- **New `dtwcpp.diagnose` module** in [python/dtwcpp/diagnose.py](python/dtwcpp/diagnose.py) with `diagnose_clusters(series, labels, medoid_indices)` and standalone `cluster_sizes`, `medoid_idle_fractions`, `within_between_length_ratio`. Returns a dict of measured quantities + a list of human-readable flag strings for the silent-failure modes: singleton clusters, idle-medoid (medoid is a near-constant signal carrying no information), length-dominated labels (clustering is just measuring ride duration, not shape), and degenerate one-vs-rest splits (e.g. hierarchical average-linkage's tendency to peel off a single outlier as its own cluster). Skips the length check when all input series have the same length. Covered by 13 unit tests in [tests/python/test_diagnose.py](tests/python/test_diagnose.py).
- **New `dtwcpp.features` module** in [python/dtwcpp/features.py](python/dtwcpp/features.py) with `summarise(series)` returning a `(N, K)` matrix of summary features (mean, std, max, min, idle-fraction, length, abs-sum by default; user can override with a custom `{name: callable}` dict). Standardised columns by default. Provides the feature-based clustering baseline that should be tried whenever DTW finds only weak structure — for some telemetry the signal lives in aggregate statistics rather than time-warped shape. Covered by 8 unit tests in [tests/python/test_features.py](tests/python/test_features.py).

### Fixed (WDTW univariate Problem path used wrong weights — off-by-one in cache key)

- WDTW direct (`dtwcpp.distance.wdtw(x, y, band, g)`) and Problem (`set_variant(WDTW)` then `dist_by_ind`) silently disagreed on univariate input: 4.126 vs 4.015 for the `short_pair` test fixture. Root cause: the univariate WDTW path in [dtwc/core/dtw_dispatch.cpp:166](dtwc/core/dtw_dispatch.cpp#L166) used `max_dev = max(len_x, len_y)` while the canonical `wdtwBanded(x, y, band, g)` (and Jeong et al. 2011) use `max_dev = max_len − 1`. The cache populated in `Problem::refresh_variant_caches` ([Problem.cpp:164](dtwc/Problem.cpp#L164)) had the same off-by-one — so cache lookups *did* hit, but with the wrong weight vector (one element too long). The multivariate path was already correct (`steps − 1`); the f32 path goes through the canonical `wdtwBanded(x, nx, y, ny, band, g)` overload that re-derives `max_dev` itself, so f32 was also unaffected. Fixed by aligning both the dispatch lambda and the cache populator to `len − 1`. Regression test `test_wdtw_matches_problem` now passes.

### Fixed (`DenseDistanceMatrix.to_numpy` had a misleading "zero-copy" contract)

- `to_numpy()` ([python/src/_dtwcpp_core.cpp:257](python/src/_dtwcpp_core.cpp#L257)) returned a numpy array with `OWNDATA=False` (capsule-owned) but the capsule wrapped a standalone heap buffer, not the C++ matrix's storage — writes through the array silently vanished. A true zero-copy view is structurally impossible because `DenseDistanceMatrix` uses packed triangular storage (`n*(n+1)/2` entries) and `to_numpy` expands to full `N*N`. Fixed the contract: docstring now states the result is an independent copy and that mutations must go through `set(i, j, v)`. Test renamed to `test_to_numpy_is_independent_copy` and asserts the truthful behavior. If we ever want a zero-copy view, that needs a separate `to_packed_numpy()` exposing the underlying storage layout — out of scope here.

### Fixed (Python wheel silently lost OpenMP — runs serially)

- [python/CMakeLists.txt](python/CMakeLists.txt) `_dtwcpp_core` (the nanobind extension) previously linked only `dtwc++ PRIVATE`, omitting `project_options`. The MSVC `/openmp:experimental` compile flag lives on `project_options` (set in [dtwc/CMakeLists.txt:113](dtwc/CMakeLists.txt#L113) when OpenMP is detected) so the python TU was compiled without it — `_OPENMP` was undefined and every `#pragma omp parallel for` inside `_dtwcpp_core.cpp` (e.g. `compute_distance_matrix`) compiled as a no-op. `dtwcpp.OPENMP_AVAILABLE` reported `False` even when CMake's `find_package(OpenMP)` had succeeded, and the wheel ran serially. The standalone `dtwc_main` CLI did link `project_options` ([CMakeLists.txt:217](CMakeLists.txt#L217)) so it was unaffected — the regression only hit users coming through the Python wheel. Fixed by linking `project_options` into `_dtwcpp_core` alongside `dtwc++`. Verified: `openmp_max_threads()` now returns the real thread count (24 on the dev box) and a 137-series × ~5000-sample distance matrix at band=100 goes from 124.7 s with `OMP_NUM_THREADS=1` to 5.8 s with 24 threads — 21.5× scaling.

### Fixed (`fast_clara` did not auto-wire result into Problem state)

- [python/src/_dtwcpp_core.cpp](python/src/_dtwcpp_core.cpp) `fast_clara` returned a `ClusteringResult` but, unlike `fast_pam` and `clarans`, did not store `result.labels` / `result.medoid_indices` back into the Problem. As a result, calling `silhouette(prob)` or `davies_bouldin_index(prob)` after `fast_clara` returned `-1.0` and "Cluster before calculating DBI" silently, unless the user manually copied the result fields into `prob.clusters_ind` / `prob.centroids_ind` / called `prob.set_number_of_clusters(k)`. Fix mirrors what `fast_pam` already does (auto-wire after the GIL is reacquired). Docstring updated to reflect the new contract.

### Fixed (Python `load_dataset_parquet` handles Polars `large_list<float>`)

- [python/dtwcpp/io.py](python/dtwcpp/io.py) `load_dataset_parquet` previously assumed a rectangular columnar layout (`np.column_stack` of the columns) and crashed on the Polars-written single-list-column format (`large_list<float>` or `list<float>`) used by some external data providers. Now auto-detects the layout: returns `(ndarray (N, L), col-names)` for the rectangular case as before, and `(list[ndarray], series-names)` for the ragged list-column case. New optional parameters: `column` (explicit list column name) and `name_column` (use a metadata column for series names instead of `series_0, series_1, ...`). Covers six new regression scenarios in [tests/python/test_io.py](tests/python/test_io.py) `TestParquetListColumn` — large_list auto-detect, plain `list<>` support, explicit column + name column, wrong-type + missing-column error paths, and variable-length round-trip.

### Changed (warping header family — unified DTW kernel, Phase 1 + Phase 2)

The Standard / ADTW / WDTW / DDTW + ZeroCost-missing paths now share **one** templated DTW kernel instead of per-variant copy-paste loops. Three axes of variation — pointwise cost, cell recurrence, and window shape — each become a policy; the banded/linear/full loop bodies live in exactly one place.

- **New `dtwc::core::dtw_kernel_{full,linear,banded}<T, Cost, Cell>`** in [dtwc/core/dtw_kernel.hpp](dtwc/core/dtw_kernel.hpp). All variants dispatch through these. `Cell` policies: `StandardCell` (min of 3 + cost), `ADTWCell<T>{penalty}` (penalty on horizontal/vertical steps).
- **New `dtwc::core::Span*Cost<T>`** cost functors in [dtwc/core/dtw_cost.hpp](dtwc/core/dtw_cost.hpp): `SpanL1Cost`, `SpanSquaredL2Cost`, `SpanWeightedL1Cost` (for WDTW), `SpanNanAwareL1Cost` / `SpanNanAwareSquaredL2Cost` (for ZeroCost missing), plus their multivariate counterparts. `dispatch_metric()` / `dispatch_mv_metric()` also live here now.
- [dtwc/warping.hpp](dtwc/warping.hpp), [warping_adtw.hpp](dtwc/warping_adtw.hpp), [warping_wdtw.hpp](dtwc/warping_wdtw.hpp), [warping_missing.hpp](dtwc/warping_missing.hpp) are now thin public-API wrappers that build a Cost + Cell and call the shared kernel. Touched-file LOC: 2,103 → 1,651 (−452 LOC); the banded loop previously duplicated in 5 files now exists once.
- [warping_ddtw.hpp](dtwc/warping_ddtw.hpp) unchanged — it already delegated to `dtwBanded` after derivative preprocessing.

### Fixed (silent dispatch / fallback bugs)

- **`dtwc::core::dtw_runtime()`** now honours `opts.variant_params.variant`. Previously [dtwc/core/dtw.cpp](dtwc/core/dtw.cpp) ignored the variant field and always ran Standard DTW, silently dropping ADTW / WDTW / DDTW requests from the simple binding entry point.
- **`adtwBanded_mv` / `wdtwBanded_mv` / `dtwMissing_banded_mv`** now use a real banded MV kernel instead of falling back to the unbanded MV path. Previous TODO comments acknowledged this was a "for now" — the unified kernel promotes banded MV to a first-class path across all variants. Regression tests in [unit_test_mv_variants.cpp](tests/unit/unit_test_mv_variants.cpp) and [unit_test_mv_missing.cpp](tests/unit/unit_test_mv_missing.cpp) confirm tight bands now produce strictly different (tighter) results than unbanded.

### Changed (dispatch unification — Phase 3 part 1)

`Problem::rebind_dtw_fn` previously held a ~130-line nested switch that dispatched on `{missing_strategy, variant, ndim}` with UV/MV forks duplicated across every variant case. Collapsed to two lines that call a new templated resolver:

- **New `dtwc::core::resolve_dtw_fn<T>(const Problem&)`** in [dtwc/core/dtw_dispatch.hpp](dtwc/core/dtw_dispatch.hpp) / [dtw_dispatch.cpp](dtwc/core/dtw_dispatch.cpp). Explicit instantiations for `T = data_t` (f64) and `T = float` (f32). Resolution happens once at rebind time; the returned `std::function` inner body is the existing templated kernel call — zero per-cell dispatch overhead. Benchmarked: 0% change on `BM_dtwBanded/1000/50`, `BM_wdtwBanded_g/1000/50` vs pre-refactor.
- Adding a new DTW variant now localises to `resolve_dtw_fn` — one `make_*<T>` helper + one switch arm, instead of two touch points (public wrapper + rebind case) with UV/MV duplication.
- New public `Problem::wdtw_weights_cache()` const accessor for the resolver to read the WDTW weights cache without friend machinery.

### Fixed (silent dispatch — f32 path)

- **`Problem::dtw_function_f32()`** now honours `variant_params.variant` and `missing_strategy`. Previously the float32 distance function was unconditionally bound to Standard DTW with `MissingStrategy::Error`, regardless of Problem configuration. The primary user of this path is `fast_clara`'s chunked-Parquet assignment (memory-saving f32 loader): anyone clustering with `variant = WDTW/ADTW/DDTW/SoftDTW` or with missing-data strategies and using the chunked path silently got Standard DTW results. Regression tests in [unit_test_mv_variants.cpp](tests/unit/unit_test_mv_variants.cpp) `[f32]` tag exercise ADTW, WDTW, and ZeroCost-missing via `dtw_function_f32()`.

### Changed (AROW via unified kernel — Phase 3 part 2)

The `Problem`-level AROW missing-data path now runs on `dtw_kernel_banded` with two new policies, completing the kernel-family unification for banded AROW:

- **`AROWCell`** in [dtwc/core/dtw_kernel.hpp](dtwc/core/dtw_kernel.hpp): interprets a NaN cost as the "missing pair" sentinel and carries the diagonal predecessor without adding cost. Boundary-aware fallback (uses `up` or `left` when `diag` is out-of-bounds).
- **`SpanAROWL1Cost<T>`** / **`SpanAROWSquaredL2Cost<T>`** in [dtwc/core/dtw_cost.hpp](dtwc/core/dtw_cost.hpp): return NaN when either operand is missing — signal to `AROWCell` without widening the Cell contract with an extra bool.
- **Cell contract extended with `seed(cost, i, j)`**: seed value at DP cell (0,0). Default (`StandardCell`/`ADTWCell`) returns `cost` unchanged. `AROWCell` returns 0 when cost is NaN, preventing NaN propagation through the DP when both (0,0) operands are missing.
- **Cross-validated** bit-for-bit against the legacy `dtwAROW_banded` on {no-NaN, interior-NaN, leading-NaN, trailing-NaN, all-NaN} × bands {1,2,3,4}. See [unit_test_arow_dtw.cpp](tests/unit/unit_test_arow_dtw.cpp) `[phase3]` tag.
- **Perf**: `BM_dtwBanded/1000/50` 146 μs → 145 μs (−0.7%); `seed()` method is inlined to a direct assignment for non-AROW cells — zero overhead on Standard/ADTW/WDTW/DDTW hot paths.
- `warping_missing_arow.hpp` was retained unchanged as the standalone public API surface; it has now been folded into the unified kernel (see Phase 4 entry below).

### Changed (Soft-DTW via unified kernel — Phase 3 part 3)

Problem-level SoftDTW dispatch now runs on `dtw_kernel_full<T, SpanL1Cost<T>, SoftCell<T>>` instead of the standalone `soft_dtw()` function.

- **`SoftCell<T>{gamma}`** in [dtwc/core/dtw_kernel.hpp](dtwc/core/dtw_kernel.hpp): log-sum-exp softmin with max-subtract stabilisation. Sentinel-aware — out-of-bounds predecessors (`maxValue`) are excluded from the LSE accumulator, so first-row/column cells where only one predecessor is valid reduce automatically to `predecessor + cost` (hard accumulation), matching the legacy `soft_dtw()` boundary treatment.
- **Cross-validated** bit-for-bit against `soft_dtw()` on equal-length, different-length, identical, and swap-symmetric inputs across gamma {0.1..10.0}. See [unit_test_soft_dtw.cpp](tests/unit/unit_test_soft_dtw.cpp) `[phase3]` tag. 4 test cases, 14 assertions, all passing within 1e-10 tolerance.
- `soft_dtw.hpp` originally retained unchanged as the standalone public API surface; its forward pass has now also been folded into the unified kernel (see Phase 4 entry below). `soft_dtw_gradient()` stays separate — it needs its own forward+backward matrices for the alignment matrix E.

### Added (Multivariate AROW — Phase 3 part 4)

Problem-level AROW dispatch now has a first-class MV branch; previously ndim > 1 was silently flattened to scalar AROW (documented TODO from pre-refactor).

- **`SpanMVAROWL1Cost<T>`** / **`SpanMVAROWSquaredL2Cost<T>`** in [dtwc/core/dtw_cost.hpp](dtwc/core/dtw_cost.hpp): per-channel skip for cost (summing `|x[d] - y[d]|` over comparable channels). Returns NaN only when a pair has *no* comparable channels — every channel has at least one NaN operand. Triggers `AROWCell`'s diagonal carry only when the whole pair is uninformative.
- **Design note**: the direct scalar→MV lift ("any channel missing → diagonal carry") would discard usable per-channel data. The per-channel-skip semantics here preserves information and is consistent with `SpanMVNanAwareL1Cost` (ZeroCost MV). Reduces exactly to scalar AROW when `ndim = 1`.
- **Tests** in [unit_test_arow_dtw.cpp](tests/unit/unit_test_arow_dtw.cpp) `[mv][phase3]` tag: ndim=1 parity with `dtwAROW_banded` across bands {1..4}, identical MV zero-distance, per-channel-skip correctness, fully-missing-step triggers diagonal carry.

### Phase 3 complete

All four deferred items from Phase 2 are now implemented:

| Part | Deliverable | Commits |
|------|-------------|---------|
| 3.1 | `rebind_dtw_fn` templated resolver + f32 silent-variant bug fix | `4d92881`, `cbdb942` |
| 3.2 | AROW fold via `AROWCell` + `SpanAROWL1Cost` + Cell `seed()` | `d595035` |
| 3.3 | Soft-DTW fold via `SoftCell` (log-sum-exp + stabilisation) | `ee0f798` |
| 3.4 | MV AROW first-class path via `SpanMVAROWL1Cost` | _(this)_ |

Kernel family now handles Standard / ADTW / WDTW / DDTW / Soft-DTW / AROW / ZeroCost-missing / Interpolate-missing with one templated core and orthogonal Cost + Cell policies. Adding a new variant = one Cost policy + one Cell policy + one switch arm in `resolve_dtw_fn`.

### Fixed (Python wheel build — llfio + quickcpplib ninja propagation)

`uv pip install -e .` / `pip install .` / any scikit-build-core-driven wheel build previously failed at llfio's nested `download_build_install()` configure step with `no such file or directory '.../ninja' --version`. Root cause: quickcpplib's `QuickCppLibUtils.cmake` spawns nested CMake processes at two levels without forwarding the build tool:

1. **Child** (line 261 — `download_build_install`): `execute_process(COMMAND "${CMAKE_COMMAND}" .)` — no `-G`, no `-DCMAKE_MAKE_PROGRAM`.
2. **Grandchild** (line 344 — `find_quickcpplib_library` builds a `cmakeargs` string template-substituted into an `ExternalProject_Add`): includes `-G` but not `-DCMAKE_MAKE_PROGRAM`.

In sandboxed wheel builds (scikit-build-core's `pip-build-env`, cibuildwheel) ninja lives at an ephemeral path like `/.../uv/builds-v0/.tmpXXX/bin/ninja` which is rewritten per reinstall — any cached `CMakeCache.txt` in the sub-build points at a dead path, and PATH-forwarding alone is insufficient because the child's generator autodetect runs once and freezes the stale path into its own cache. Only `-DCMAKE_MAKE_PROGRAM` passed on the command line reliably overrides the stale cache.

- **Fix in [cmake/Dependencies.cmake](cmake/Dependencies.cmake)**: pre-clone `quickcpplib` into the location llfio's bootstrap expects (`${CMAKE_BINARY_DIR}/quickcpplib/repo`) and apply two idempotent text patches to `QuickCppLibUtils.cmake` — one at the child-spawn site (adds `-G` + `-DCMAKE_MAKE_PROGRAM`), one at the grandchild `cmakeargs` assembly (appends `-DCMAKE_MAKE_PROGRAM`). Sentinel comments (`DTWC_NINJA_PROPAGATION_PATCH (child)` / `(grandchild)`) detect already-patched state per-patch so partially patched files self-heal. llfio's own bootstrap then finds the pre-existing repo and skips its own `git clone`, picking up our patched version via `include(QuickCppLibUtils)`.
- **No upstream dependency** — no `brew install ninja` workaround, no fork of llfio/quickcpplib. Upstream may eventually fix this (an issue is warranted); the patch self-retires once the sentinel pattern stops matching unpatched upstream.
- **No behavioural change for dev builds** where ninja is already on PATH and build dirs are fresh — the patch still applies but the added `-D` args match the autodetected value.
- **Verified** end-to-end: `env -i PATH=/minimal/path uv pip install --reinstall --no-cache -e .` completes on macOS arm64 in ~90s with no system ninja.
- Unblocks `tests/integration/test_cross_language.py` (C++ ≡ Python ≡ MATLAB numerical parity).

### Changed (standalone Soft-DTW forward API folded — Phase 4 cleanup)

The standalone [dtwc/soft_dtw.hpp](dtwc/soft_dtw.hpp) `soft_dtw(x, y, gamma)` forward pass now delegates to `core::dtw_kernel_full<T, SpanL1Cost<T>, SoftCell<T>>`. Its hand-rolled DP loop (~45 LOC including first-row/column prologue and interior softmin loop) is removed.

- `softmin_gamma()` helper is retained — still used by `soft_dtw_gradient()`.
- `soft_dtw_gradient()` is **unchanged** — the Cuturi–Blondel backward pass reads the full forward cost matrix C *and* writes the alignment matrix E, so it keeps its own forward+backward loops. Folding the gradient into the kernel would require emitting C as an out-parameter.
- **Behaviour**: identical on the existing test suite (`[soft_dtw]` tag — convergence to standard DTW as γ→0, symmetry, monotonicity, 3×3 known example, gradient finite-differences check, 22 test cases / 61 assertions passing).
- **No API break**: `soft_dtw` and `soft_dtw_gradient` signatures unchanged. Python (`dtwcpp._core.soft_dtw`) and MATLAB (`dtwc_mex`) bindings unaffected.

### Changed (standalone AROW API folded — Phase 4 cleanup)

The standalone [dtwc/warping_missing_arow.hpp](dtwc/warping_missing_arow.hpp) public API (`dtwAROW`, `dtwAROW_L`, `dtwAROW_banded`) now delegates to the unified DTW kernel. The hand-rolled `detail::dtwAROW_*_impl` helpers (full / linear / banded × span / pointer overloads, ~260 LOC) are removed — each wrapper builds a `SpanAROW{L1,SquaredL2}Cost` functor and calls `core::dtw_kernel_{full,linear,banded}<T, Cost, AROWCell>`.

- **Behaviour**: identical within the test suite (AROW tests, adversarial tests, MV tests, Problem-level AROW tests — 70/70 ctest passing). The legacy banded impl used a simpler `std::ceil/std::floor` band-bounds calculation and the unified kernel uses the `round-100` variant from [dtw_kernel.hpp:157-163](dtwc/core/dtw_kernel.hpp#L157-L163); the legacy's own comment (now removed) called out that the difference is harmless because out-of-band cells are sentinel-valued and never selected.
- **No API break**: all span / vector / pointer+size overloads retained with identical signatures.
- **No perf regression**: `dtw_kernel_banded` is the same kernel already exercised on the Problem-level AROW hot path via `resolve_dtw_fn` — the standalone wrapper now joins it. `BM_dtwBanded/1000/50` unchanged (145 μs).
- Wrapper header shrinks from 459 LOC to 214 LOC.

### Added (GPU parity + configuration)

- **`MetalDistMatResult::lb_time_sec`**: Metal now reports LB_Keogh pre-pass time separately from total GPU time, matching the CUDA field of the same name.
- **`dtwc::metal::compute_lb_keogh_metal(series, band)`**: standalone LB_Keogh lower bounds over all `N*(N-1)/2` pairs on the default Metal device. Mirrors `dtwc::cuda::compute_lb_keogh_cuda` (envelope + symmetric pairwise LB, no DTW dispatch). Returns `MetalLBResult{lb_values, n, gpu_time_sec}`.
- **`dtwc::LowerBoundStrategy` enum** (`Auto`/`None`/`Kim`/`Keogh`/`KimKeogh`) and `Problem::lb_strategy` field. Controls which lower bound(s) feed the Pruned CPU path. `Auto` keeps the historical Kim+Keogh cascade; `None` short-circuits to BruteForce.
- **`dtwc::KernelOverride` enum** (`Auto`/`Wavefront`/`WavefrontGlobal`/`BandedRow`/`RegTile`) and `max_length_hint` field on both `CUDADistMatOptions` and `MetalDistMatOptions`. Lets advanced users force a kernel path or hint the expected max series length for the selector. Unsupported overrides silently fall back to `Auto`. Metal wires the hint into its kernel-selection heuristics; CUDA stores the fields for API uniformity.

### Changed (GPU backend refactor — non-breaking)

- **Base structs `dtwc::gpu::DistMatOptionsBase` / `DistMatResultBase`** in [core/gpu_dtw_common.hpp](dtwc/core/gpu_dtw_common.hpp). `CUDADistMatOptions`, `MetalDistMatOptions`, `CUDADistMatResult`, `MetalDistMatResult` now inherit common fields (`band`, `verbose`, `use_lb_keogh`, `max_length_hint`, `kernel_override`, `matrix`, `n`, `gpu_time_sec`, `lb_time_sec`, `pairs_computed`, `pairs_pruned`, `kernel_used`). `lb_threshold` stays per-backend: CUDA default `-1.0` (sentinel), Metal default `0.0` (applied). Designated aggregate init preserved.
- **`dispatch_gpu_backend` lambda in `Problem::fillDistanceMatrix`**: collapses the parallel CUDA/Metal case blocks (result → matrix copy, verbose log, empty-result fallback) into one generic post-processor parameterised by backend result type.

### Changed (naming unification — breaking, pre-v2.0.0)

Unified option/field names across CPU, CUDA, and Metal backends. Hard renames (no compat aliases):

| Old | New | Scope |
|---|---|---|
| `DEFAULT_BAND_LENGTH` | `DEFAULT_BAND` | [settings.hpp](dtwc/settings.hpp) + 32 call sites |
| `DTWOptions::band_width` | `DTWOptions::band` | [dtw_options.hpp](dtwc/core/dtw_options.hpp) + 4 call sites |
| `CUDADistMatOptions::use_lb_pruning` | `use_lb_keogh` | CUDA — matches Metal, CPU naming |
| `CUDADistMatOptions::skip_threshold` | `lb_threshold` | CUDA |
| `MetalDistMatOptions::enable_lb_keogh` | `use_lb_keogh` | Metal — matches CUDA |
| `DistanceMatrixStrategy::GPU` | `::CUDA` | `::GPU` was ambiguous once `::Metal` landed. Enum / CLI `--device` / Python `DistanceMatrixStrategy.CUDA` / MATLAB `"cuda"` string |
| `CUDASettings::precision_mode` (int 0/1/2) | `precision` | Kept as int for header independence from `DTWC_HAS_CUDA` |

MATLAB strategy string `'gpu'` is no longer accepted — pass `'cuda'` instead.

Motivation: three different spellings for LB_Keogh (`use_lb_pruning`, `enable_lb_keogh`, implicit-via-`Pruned`-strategy) and two for the band (`band`, `band_width`). With two GPU backends now shipped, the drift would only compound.

Cited cuDTW++ (Schmidt & Hundt 2020) and LB_Keogh (Keogh & Ratanamahatana 2005) in [metal_dtw.hpp](dtwc/metal/metal_dtw.hpp) / [metal_dtw.mm](dtwc/metal/metal_dtw.mm) headers and in the Register-tile / LB_Keogh CHANGELOG entries below — these were missing when the kernels first landed.

### Added (Python bindings)

- `compute_distance_matrix_cuda()` now accepts `use_lb_keogh` and `lb_threshold` kwargs (previously CUDA pruning was exposed in C++ only). Default `use_lb_keogh=False` preserves existing behavior. Semantics match the C++ surface: pair pruned if `LB_Keogh > lb_threshold` and `lb_threshold > 0`.

### Added (Documentation)

- New **GPU Backends** page in the docs site ([docs/content/method/gpu-backends.md](docs/content/method/gpu-backends.md)): kernel dispatch tables for both CUDA (3 kernels) and Metal (5 kernels), Sakoe-Chiba envelope definition with ASCII diagram, `LB_Keogh` equation (one-directional + symmetric), full GPU LB_Keogh pipeline flowchart, measured speedups on Apple M2 Max, when-to-use-which decision tree, and citations to Schmidt & Hundt 2020, Keogh & Ratanamahatana 2005, Rakthanmanon 2012, Sakoe & Chiba 1978, and Lemire 2009 (future work).

### Added (Metal GPU backend — Apple Silicon)

- **New `dtwc::metal` backend**: anti-diagonal wavefront DTW kernel in MSL (Metal Shading Language), compiled at runtime via `newLibraryWithSource:`. Pairwise distance matrix on the Apple GPU, one threadgroup per pair, 3 rotating threadgroup-memory buffers for anti-diagonals.
- **CMake**: `DTWC_ENABLE_METAL` option (default ON on APPLE, silently disabled elsewhere). Adds `OBJCXX` language, links `Foundation` + `Metal` frameworks, defines `DTWC_HAS_METAL`.
- **`DistanceMatrixStrategy::Metal`** dispatch case in `Problem::fillDistanceMatrix`; graceful CPU fallback when Metal is unavailable or when `max_L` exceeds the threadgroup-memory cap (32 KB on M1/M2/M3, equivalent to ~2730 FP32 elements).
- **Python binding**: `DistanceMatrixStrategy.Metal`, `metal_available()`, `metal_device_info()`, `compute_distance_matrix_metal()`, `METAL_AVAILABLE` attribute. Integrated into `system_info()`. Measured Python wrapper overhead vs native C++ `compute_distance_matrix_metal`: **−0.5% to +0.9%** (well under 10% target).
- **MATLAB binding**: `"metal"` distance strategy accepted by `set_distance_strategy`; `system_check` struct now includes `metal` + `metal_info` fields; `dtwc.check_system()` reports Metal status. Measured MATLAB wrapper overhead vs Python (both using `Problem::fillDistanceMatrix`): **≤1%** across all 6 workloads. (Initial apples-to-oranges comparison showed 8–17%, but the gap turned out to be `Problem::fillDistanceMatrix` vs direct `compute_distance_matrix_metal` — same gap appears in Python too. Language-wrapper overhead itself is sub-1%.)
- **MATLAB R2024b linker fix**: `CMakeModules/FindMatlab.cmake` (shipped with CMake) unconditionally adds `cppMexFunction.map` to the exported-symbols list on macOS, which requires `_mexCreateMexFunction` / `_mexDestroyMexFunction` / `_mexFunctionAdapter` symbols we don't provide (we use the legacy C `mexFunction` API). [bindings/matlab/CMakeLists.txt](bindings/matlab/CMakeLists.txt) now clears `Matlab_HAS_CPP_API` before `matlab_add_mex()` so only `c_exportsmexfileversion.map` is linked.
- **Measured on Apple M2 Max (38-core GPU)** vs the 12-thread CPU path:

  | Workload | CPU (ms) | Metal (ms) | Speedup |
  |---|---|---|---|
  | 50 series × 500 length | 106 | 6.9 | 15.4× |
  | 100 × 500 | 404 | 26.2 | 15.4× |
  | 100 × 1000 | 1648 | 139 | 11.9× |
  | 200 × 500 | 1626 | 103 | 15.8× |
  | 50 × 2500 | n/a | 151 | — |
  | 10 × 10000 (long series, global-mem kernel) | 3058 | 108 | **28.3×** |
  | 30 × 10000 | 16061 | 929 | **17.3×** |

  Throughput: ~30–52 × 10⁹ DTW cells/sec; ~180–320 GFLOPS (≈1.3–2.4% of 13.6 TFLOPS FP32 peak). The low FLOP fraction is expected for DTW's memory-bandwidth-bound DP recurrence; further gains require register-tiling and warp-shuffle kernels (follow-on).
- **Three kernel variants share the same API:**
  - `dtw_wavefront` (threadgroup memory): 3 × max_L floats in 32 KB threadgroup memory → max_L ≤ 2730 on M1/M2/M3.
  - `dtw_wavefront_global` (device memory): scratch lives in unified GPU memory; any max_L supported.
  - `dtw_banded_row` (tight-band row-major): one thread per pair, no intra-threadgroup barriers, coalesced device-memory scratch with register-rotated prev/cur window. Fires when `band > 0 AND band * 20 < max_L AND band ≤ 512` — the regime where anti-diagonal barrier overhead dominates the wavefront kernel. On 75 × 10000 band=100: **1.01 s vs 1.40 s for wavefront (1.4×) and 1.76 s for CPU (1.7×)**. Dispatcher picks automatically; `MetalDistMatResult::kernel_used` reports which kernel ran.
- **macOS GPU-watchdog avoidance**: long dispatches (>~2 s per command buffer) fail with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`. The dispatcher now chunks pairs across multiple command buffers (budget ≈ 5 × 10⁹ cells per buffer) so arbitrary N × L workloads complete without triggering the watchdog.
- **LB_Keogh pruning path (Metal)**: opt-in lower-bound prune pipeline mirroring the CUDA envelope/LB/compaction kernels (`dtwc/cuda/cuda_dtw.cu:785-910`). Algorithm from Keogh & Ratanamahatana (2005) *"Exact Indexing of Dynamic Time Warping"* (KAIS 7(3), 358–386); symmetric `max(LB(j|env_i), LB(i|env_j))` form per Rakthanmanon et al. (2012). GPU orchestration pattern (envelope → LB → atomic compaction → DTW on survivors) follows the CUDA reference which itself derives from cuDTW++ (Schmidt & Hundt 2020). Three new MSL kernels:
  - `compute_envelopes` — sliding min/max window per series (one threadgroup per series).
  - `compute_lb_keogh` — symmetric `max(LB(j, env_i), LB(i, env_j))` per pair (one thread per pair).
  - `compact_active_pairs` — threshold filter: active pairs atomically appended to `active_pairs[]`; pruned pairs have `+∞` stamped into the result matrix.

  `dtw_wavefront` and `dtw_wavefront_global` gained an optional `pair_indices` buffer (+ `has_pair_indices` flag) so the DTW dispatch skips pruned pairs entirely. When pruning is disabled the wavefront path reads a 1-int dummy buffer (zero cost). Banded-row and regtile paths don't support pruning; requesting LB_Keogh on those paths silently disables it (with a `verbose` warning).

  New `MetalDistMatOptions` fields: `enable_lb_keogh` (default false), `lb_threshold` (default 0.0; pairs with `LB_Keogh > threshold` are pruned), `lb_envelope_band` (default −1 → use `band` or `max_L/10`). Result struct gained `pairs_pruned` counter.

  Measured on Apple M2 Max (random uniform series):

  | Workload | Baseline | Permissive LB (0% prune, pure overhead) | Strict LB (100% prune) |
  |---|---|---|---|
  | N=100, L=1000 | 158 ms | 159 ms (+0.6%) | **1.54 ms (102×)** |
  | N=200, L=1000 | 635 ms | 635 ms (~0%)  | **3.60 ms (176×)** |

  Strict numbers reflect 100% prune rate (random data + `lb_threshold=0`); real k-medoids cluster-assignment workloads land between depending on data + threshold tuning.

  Tests: `test_metal_lb_keogh.cpp` (4 cases, 136 assertions) covers: disabled path bit-identical to non-LB, permissive threshold keeps all pairs, strict threshold prunes + stamps `+∞`, and silent disable on banded_row. Exposed through Python (`compute_distance_matrix_metal(..., enable_lb_keogh=True, lb_threshold=..., lb_envelope_band=...)`). MATLAB exposure deferred (would require Metal-specific options on the backend-agnostic `Problem` class).

- **Register-tile Metal kernels** (`dtw_regtile_w4`, `dtw_regtile_w8`): short/medium unbanded workloads (max_L ≤ 256) now use a SIMD-group register-tile kernel. Algorithm from Schmidt & Hundt (2020) *"cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-Enabled GPUs"* (Euro-Par 2020, LNCS 12247, pp. 597–612); ported via the existing CUDA reference (`dtwc/cuda/cuda_dtw.cu:543-780`). Each of 32 lanes in a SIMD-group holds `TILE_W` (4 or 8) columns in registers; left-neighbor costs propagate via `simd_shuffle_up`, eliminating the per-diagonal `threadgroup_barrier` that dominated the wavefront kernel on short series. Eight pairs per threadgroup (`PAIRS_PER_TG = 8`). Dispatch: `TILE_W=4` for `max_L ≤ 128`, `TILE_W=8` for `128 < max_L ≤ 256`, band `== -1` only. Measured on Apple M2 Max (N=100, 4 950 pairs): L=64 → 0.37 ms; L=128 → 0.53 ms; L=192 → 0.87 ms; L=256 → 1.09 ms. Approximately **5–7× faster per cell than the extrapolated wavefront path** at these lengths. `MetalDistMatResult::kernel_used` reports `regtile_w4` / `regtile_w8`. Banded regtile is deferred (register pressure in the tile loop).
- **K-vs-N Metal kernels** (`dtw_kvn_wavefront`, `dtw_kvn_wavefront_global`): K queries × N targets parallel DTW on GPU, mirroring the CUDA `compute_dtw_k_vs_all` API. New host functions `compute_dtw_k_vs_all_metal` (two overloads: by-indices-into-series, and separate queries + targets) and `compute_dtw_one_vs_all_metal` (two overloads: by index, and external query). Result types `MetalKVsNResult` / `MetalOneVsNResult` match the CUDA shape. Enables GPU-accelerated k-medoids cluster-assignment loops on Apple Silicon — K*N pairs (instead of precomputing N² for small-K workloads).
- **Band-edge correctness fix** (affects all four wavefront kernels — `dtw_wavefront`, `dtw_wavefront_global`, `dtw_kvn_wavefront`, `dtw_kvn_wavefront_global`): the band-range iteration optimization relied on "out-of-band cells retain their INF initialization," which was wrong given the 3-buffer rotation — every 3 diagonals each buffer is reused as `cur`, so cells outside the current band can retain real DTW values from 3 diagonals ago. When the band range shifts by ±1 between diagonals, in-band reads at the new band edge hit those stale values and produce distances smaller than the true banded DTW. Pre-existing NxN wavefront banded paths silently returned wrong values (≤0.12% for the random-seed tests we had, up to ~7% on data where the stale values happen to lie on the DP path). Fix: after each diagonal's in-band writes, stamp INF into the two band-adjacent positions (`i_lo-1` and `i_hi+1`). Two extra writes per diagonal per pair; wavefront-banded benchmarks shift 0–9% (e.g. 75×2500 band=250: 212 → 232 ms; 75×10000 band=100 tight-band uses `dtw_banded_row` and is unaffected). Verified against CPU `dtwBanded` by new directed test "Metal wavefront NxN banded matches CPU dtwBanded".
- **Tests**: `test_metal_correctness.cpp` (18 cases, 2 044 assertions) validates GPU output against CPU reference at FP32 tolerance, covers length-2000, the 10 000-length global-memory fallback, three banded-row dispatch scenarios, an NxN wavefront-banded cross-check against `dtwBanded`, four K-vs-N cases (1-vs-N by index, 1-vs-N external query, K-vs-N by indices, K-vs-N banded), and four regtile cases (`w4` coverage, `w8` coverage, partial-tile edges L=127/255, variable-length asymmetric orientation). `test_metal_mmap.cpp` exercises the Metal strategy through `Problem::fillDistanceMatrix` into both `DenseDistanceMatrix` and the memory-mapped distance-matrix paths.
- **Benchmark**: `benchmarks/bench_metal_dtw.cpp` (Google Benchmark) compares Metal vs CPU at matching sizes; JSON lands in `benchmarks/results/mac_m2max/`.

### Added (macOS support)

- **`FindGUROBI.cmake`**: macOS search paths (`/Library/gurobi*/macos_universal2`, `/Library/gurobi*/mac64`) and `.dylib` library glob.
- **`CMakePresets.json`**: build and test presets for `clang-macos` (previously only `clang-win`). The preset now pins `/usr/bin/clang++` to avoid libc++ ABI conflicts when Homebrew LLVM is also installed.
- **`README.md`**: macOS-specific installation section (Homebrew libomp, Ninja, Gurobi location).
- **macOS CI workflow**: now installs Ninja, uses the `clang-macos` preset, enables HiGHS, tests Release config.

### Added (AI-Assisted Workflow)

- **Claude Code slash commands** in `.claude/commands/`: `/cluster`, `/distance`, `/evaluate`, `/convert`, `/visualize`, `/help`, `/troubleshoot`. Each command is a self-contained markdown file that guides Claude Code to drive the DTWC++ library on the user's behalf.
- **Docs page** `docs/content/getting-started/ai-commands.md` documenting the commands with examples and design principles.

### Fixed

- **`soft_dtw` / `soft_dtw_gradient`**: now throw `std::invalid_argument` if `gamma <= 0` instead of producing `inf`/`NaN`. `softmin_gamma` has a debug assert on the same condition.
- **`Problem.cpp`**: resolved TODO at line 808 — confirmed k-medoids objective uses raw DTW distances (not squared, unlike k-means).
- **`types/Index.hpp`**: added a debug assert for pointer underflow in `Index::operator-(difference_type)`.
- **`tests/unit/unit_test_clustering_algorithms.cpp`**: tests now write output CSVs to the system temp directory via `std::filesystem::temp_directory_path()`; previously tests polluted the project root with `test_clustering*.csv` files.
- **`CMakeLists.txt`**: Gurobi "not found" warning message now includes the macOS install path hint.

### Changed (Cleanup)

- Removed obsolete session artifacts: `.claude/reports/` (8 files), `.claude/summaries/` (3 files), `.claude/superpowers/` (1 file), `benchmarks/results/*.json` (16 machine-specific timing snapshots). These directories are now in `.gitignore`.
- No remaining PII in tracked files (paths/usernames). The `scripts/slurm/env.example` uses clear placeholder values.

### Added (I/O Formats)

- **Apache Arrow IPC reader** (`dtwc/io/arrow_ipc_reader.hpp`): zero-copy mmap loading via `ArrowIPCDataSource`. List + LargeList dispatch with int64 offsets for >2B elements.
- **Parquet reader** (`dtwc/io/parquet_reader.hpp`): scalar + list columns, directory loading, column auto-detection or `--column` override.
- **Parquet row-group streaming** (`dtwc/io/parquet_chunk_reader.hpp`): `ParquetChunkReader` class for reading individual row groups on demand — enables RAM-aware chunked CLARA.
- **`dtwc-convert`** Python CLI tool: converts Parquet/CSV/HDF5 → Arrow IPC or `.dtws` format.
- **CMake `DTWC_ENABLE_ARROW`**: optional Apache Arrow + Parquet support via `find_package` or CPM from source (static, ~20MB CLI binary).
- Auto-detection of input format from file extension: `.parquet`, `.arrow`, `.ipc`, `.feather`, `.dtws`, `.csv`.

### Added (Float32 Precision)

- **Runtime float32 precision**: `Precision` enum, `Data` float32 storage (`p_vec_f32`), `series_f32(i)` accessor, `dtw_fn_f32_t` dispatch.
- **Float32 view-mode**: `Data` supports non-owning `span<const float>` views for CLARA subsampling with float32 data.
- **CLI**: `--dtype float32|float64` (default float32, aliases: f32/fp32/float/f64/fp64/double). 2x memory saving, 0.003% max DTW error.

### Added (SLURM HPC Support)

- **Generalized SLURM scripts** (`scripts/slurm/`): portable remote helper, job templates (CPU, GPU, checkpoint, Parquet), `.env`-based configuration.
- **`slurm_remote.py`**: SSH/SFTP remote helper with SSH-key-first auth, two-hop gateway support, batch build/submit/download commands.
- **Job templates**: CPU test (Coffee k=2, Beef k=5), GPU test (fp32+fp64), checkpoint/resume verification, Parquet I/O test.
- **`verify_results.py`**: ARI-based ground truth comparison against UCR class labels.
- **`convert_ucr.py`**: UCR TSV to Parquet converter for testing Parquet I/O path.
- **`env.example`**: Configuration template with Oxford ARC example values.
- **Full UCR benchmark scripts** (`scripts/slurm/jobs/ucr_benchmark_cpu.slurm`, `ucr_benchmark_gpu.slurm`): run all 128 UCR datasets with Lloyd k-medoids, save distance matrices, per-dataset timing JSON, and aggregate summary.
- **`aggregate_results.py`**: merges per-dataset timing JSONs into a single benchmark results JSON for the docs website. Supports multiple runs (CPU, GPU, different architectures).
- **`slurm_remote.sh` benchmark commands**: `submit-benchmark-cpu`, `submit-benchmark-gpu [gpu_type]` for full UCR benchmarking on ARC.
- **SLURM documentation**: `docs/content/getting-started/slurm.md` with partition tables, GPU gres syntax, and troubleshooting.

### Changed (CLI)

- **`--precision` renamed to `--dtype`** (aliases: `--data-precision`, `--data-type`) to resolve CLI11 crash from duplicate `--precision` flag.
- **`--gpu-precision`** (alias: `--gpu-dtype`): GPU kernel precision with full alias set (f32/fp32/float32/f64/fp64/float64/double). Default: `auto`.
- **`--restart` renamed to `--resume`** (`--restart` kept as deprecated alias). Avoids confusion with "restart from scratch".
- **Verbose diagnostics**: `--verbose` now prints OpenMP threads, SLURM env vars (job ID, node, CPUs), CPU model, memory usage, and data memory estimate.
- **YAML/TOML config**: `precision` key split into `dtype` and `gpu-precision`. Added `resume` key.

### Fixed

- **CLI crash**: duplicate `--precision` flag caused CLI11 `OptionAlreadyAdded` exception at startup. Fixed by renaming to `--dtype` + `--gpu-precision`.
- **YAML config**: data precision was never loaded from YAML (only GPU precision was bound). Now both `dtype` and `gpu-precision` are loaded.

### Added (Data Access)

- **`Data::series(i)` → `span<const data_t>`**: uniform accessor for heap, mmap, and view modes.
- **CLARA zero-copy views** via `set_view_data()`: 48x subsample speedup by sharing parent memory.
- **`StoragePolicy` enum** (Auto/Heap/Mmap) in `dtwc/core/storage.hpp`.
- **Span overloads** for `compute_summary`, `compute_envelope`, `lb_keogh`, `lb_keogh_symmetric`.

### Added (RAM-Aware Chunked Processing)

- **`--ram-limit`** CLI flag: parsed with T/G/M/K suffixes (e.g., `--ram-limit 2G`).
- **Chunked CLARA**: when `--ram-limit` is set with Parquet input, CLARA streams row groups within the RAM budget. Subsamples and medoid series are loaded on demand; no full dataset in memory.
- **`ParquetChunkReader::read_rows()`**: sparse row access for loading subsamples by index.

### Changed (Build)

- **C++20 minimum:** All CMake targets upgraded from C++17 to C++20. CI matrix drops GCC 10, Clang 12, Clang 13 (no C++20 support).

### Changed (API)

- **`std::span` interfaces:** All public DTW distance functions (dtwFull, dtwFull_L, dtwBanded, ddtw*, adtw*, wdtw*, soft_dtw*, dtwMissing*, dtwAROW*) now accept `std::span<const data_t>` as primary overloads. `const std::vector<data_t>&` convenience overloads are retained for backward compatibility.
- **`dtw_fn_t` signature:** Changed from `std::function<data_t(const vector&, const vector&)>` to `std::function<data_t(std::span<const data_t>, std::span<const data_t>)>`.
- **`missing_utils.hpp`:** `has_missing()`, `missing_rate()`, `interpolate_linear()` now take `std::span<const T>` with vector convenience overloads.

### Added (SIMD)

- `lb_keogh()` dispatches to `dtwc::simd::lb_keogh_highway()` for `double` when `DTWC_ENABLE_SIMD=ON`, giving 2.7–3.3× speedup (measured AVX2, MSVC, i7).
- `DTWC_ENABLE_SIMD` now defaults to ON for standalone top-level builds (OFF for sub-projects and Python wheels). Google Highway provides runtime ISA dispatch — one binary runs optimally across SSE4/AVX2/AVX-512 nodes.

### Changed (SIMD performance)

- **Branchless scalar `lb_keogh`:** Replaced `std::max(T(0), std::max(eu, el))` with decomposed ternaries `max(0,eu) + max(0,el)` (valid for L≤U envelopes). Each ternary maps to a single `vmaxpd` instruction. Result: scalar lb_keogh is now **3.2–4.3× faster** and matches Highway performance — MSVC auto-vectorizer can now handle the loop. Added `#pragma omp simd reduction(+:sum)` to `lb_keogh_squared`, `lb_keogh_mv`, `lb_keogh_mv_squared`.
- **`dtw_multi_pair` uniform-length fast path:** When all 4 SIMD-lane pairs share the same dimensions (common case in DTW clustering), OOB masks are always all-false. New uniform path skips all `IfThenElse` and mask computation — 30% fewer ops per cell. Result: **4.6–5.6× faster** vs previous SIMD path; SIMD now **2.8× faster than sequential** (was 1.5× slower before).
- **Pre-hoisted row masks in `dtw_multi_pair` variable-length path:** `i_oob` masks (per-row OOB checks) are computed once before the j-loop into a `thread_local` buffer. Saves 4 scalar comparisons + stack write per inner-loop cell.
- **FMA in `z_normalize_highway`:** Normalize pass uses `MulAdd(val, inv_sd, bias)` (one FMA) instead of `Mul(Sub(val, mean), inv_sd)` (Sub + Mul). `bias = -mean * inv_sd` precomputed once.
- **`z_normalize_simd.cpp` header corrected:** Comment now accurately describes the two-pass König-Huygens algorithm (sum + sum-of-squares in one pass).

### Added (HPC build support)

- `DTWC_ARCH_LEVEL` CMake option (`""` / `"v3"` / `"v4"`): overrides `-march=native` with a portable x86-64 microarchitecture level. `v3` (AVX2+FMA) is safe for all modern HPC CPUs; `v4` targets AVX-512 nodes (Cascade Lake Xeon, Sapphire/Emerald Rapids, Genoa, Turin).
- CUDA builds now default to `CMAKE_CUDA_ARCHITECTURES=70;80;86;89;90` (V100 through H100) when not explicitly set. Override with `-DDTWC_CUDA_ARCH_LIST=...` or `-DCMAKE_CUDA_ARCHITECTURES=...`.
- GCC/Clang fast-math flags completed: added `-fno-rounding-math` and `-fno-signaling-nans` (safe for this codebase; complete the safe subset of `-ffast-math` excluding `-ffinite-math-only`).
- MSVC Release builds now include `/Gy` (function-level linking) for linker COMDAT elimination.

### Added (MIP Solver Improvements)

- MIP warm start: `--method mip` now runs FastPAM first and feeds the solution as a MIP start, dramatically reducing branch-and-bound solve time. Controlled by `--no-warm-start` flag.
- MIP solver settings exposed in CLI and TOML config: `--mip-gap`, `--time-limit`, `--no-warm-start`, `--numeric-focus`, `--mip-focus`, `--verbose-solver`.
- Gurobi branching priority on medoid selection variables A[i,i] — once medoids are fixed, assignment is a TU transportation problem (LP-integral).
- Optional YAML configuration file support (`--yaml-config config.yaml`) via yaml-cpp (`-DDTWC_ENABLE_YAML=ON`).
- `MIPSettings` struct on `Problem` for programmatic solver tuning.

### Added (MATLAB MEX Bindings)

- MATLAB MEX gateway (`bindings/matlab/dtwc_mex.cpp`) using C++ MEX API (R2018a+, RAII-safe).
- `+dtwc` MATLAB package: `dtw_distance`, `compute_distance_matrix`, `DTWClustering` class.
- `DTWClustering` handle class with `fit`, `fit_predict`, `predict` (mirrors Python API).
- CMake integration: `DTWC_BUILD_MATLAB=ON` automatically finds MATLAB and builds MEX.
- 1-based indexing conversion for all labels and medoid indices.
- FastPAM used for clustering (not legacy Lloyd).

### Changed (MIP Solver Improvements)

- Gurobi `NumericFocus` reduced from 3 to 1 (sufficient for 0/1/-1 constraint matrix, avoids 1.5-3x overhead).
- Gurobi now uses `MIPFocus=2` (optimality-focused) by default.
- MIP solver output suppressed by default (use `--verbose-solver` to see solver logs).

### Added (Wave 2A — Clustering Algorithms)

- Deferred dense distance-matrix allocation: `Problem::set_data()` no longer forces O(N^2) memory. Dense matrix allocated lazily on first `fillDistanceMatrix()` or `distByInd()` call.
- Shared medoid utilities (`algorithms/detail/medoid_utils.hpp`): `assign_to_nearest`, `compute_nearest_and_second`, `find_cluster_medoid`, `validate_medoids` — reusable, decoupled from Problem.
- Hierarchical agglomerative clustering (`algorithms/hierarchical.hpp`): Single, complete, and average linkage. `build_dendrogram()` + `cut_dendrogram()`. Small-N feature with hard `max_points=2000` guard. Ward's excluded (mathematically invalid for DTW).
- CLARANS experimental (`algorithms/clarans.hpp`): Bounded randomized k-medoids with `max_dtw_evals` and `max_neighbor` budget controls. Not exposed in CLI — requires benchmark evidence before promotion.

### Fixed (Wave 2A)

- FastCLARA now propagates `data.ndim`, `missing_strategy`, `distance_strategy`, and `verbose` to sub-problems. Previously, multivariate data was silently treated as univariate, and NaN data caused crashes.
- FastCLARA default sample size improved to `max(40+2k, min(N, 10k+100))` per Schubert & Rousseeuw 2021.
- `distByInd()` lazy allocation fix for checkpoint compatibility.

### Changed (Wave 2A)

- `Problem::set_data()` no longer calls `distMat.resize()`. The dense matrix is deferred to first actual use.

### Added (Wave 2B — Multivariate Variants + Lower Bounds)
- Multivariate WDTW: `wdtwFull_mv()`, `wdtwBanded_mv()` with position-dependent weights.
- Multivariate ADTW: `adtwFull_L_mv()`, `adtwBanded_mv()` with non-diagonal step penalty.
- Multivariate DDTW: via `derivative_transform_mv()` + standard multivariate DTW.
- Per-channel `compute_envelopes_mv()` and `lb_keogh_mv()`: valid lower bound on dependent multivariate DTW.
- `lb_keogh_squared()` and `lb_keogh_mv_squared()`: SquaredL2 metric LB_Keogh variants.
- Multivariate missing-data DTW: `dtwMissing_L_mv()`, `dtwMissing_banded_mv()` with per-channel NaN handling.

### Changed (Wave 2B)
- `Problem::rebind_dtw_fn()` dispatches to multivariate variants for WDTW, ADTW, DDTW, and ZeroCost missing when `data.ndim > 1`.

### Added (Wave 1B — Multivariate Foundation)
- Multivariate time series support via `Data.ndim` field (default 1, backward-compatible).
- `Data::series_length(i)` and `Data::validate_ndim()` for multivariate data management.
- `TimeSeriesView.ndim` with `at(i)` timestep access and `flat_size()`.
- Multivariate distance functors `MVL1Dist` and `MVSquaredL2Dist` in `warping.hpp`.
- `dtwFull_L_mv()` and `dtwBanded_mv()`: multivariate DTW with interleaved layout. `ndim=1` dispatches to existing scalar code (zero overhead).
- `derivative_transform_mv()`: stride-aware per-channel derivative transform for multivariate DDTW.

### Added (Wave 1C — Multivariate WDTW / ADTW / DDTW)

- `wdtwFull_mv()` and `wdtwBanded_mv()`: multivariate Weighted DTW with interleaved layout. `ndim=1` delegates to existing scalar code.
- `adtwFull_L_mv()` and `adtwBanded_mv()`: multivariate Amerced DTW with interleaved layout. `ndim=1` delegates to existing scalar code.
- `Problem::rebind_dtw_fn()` now dispatches WDTW, ADTW, and DDTW variants to their `_mv` counterparts when `data.ndim > 1`.

### Changed (Wave 1B)
- `Problem::rebind_dtw_fn()` dispatches to multivariate DTW when `data.ndim > 1`.
- `Problem::set_data()` calls `data.validate_ndim()` to catch malformed interleaved layouts early.

## Added

* **`missing_utils.hpp`**: Bitwise NaN check (`is_missing()`) safe under `-ffast-math`/`/fp:fast`, plus `has_missing()`, `missing_rate()`, `interpolate_linear()` with LOCF/NOCB edge handling.
* **`MissingStrategy` enum**: `Error` (default), `ZeroCost`, `AROW`, `Interpolate` for controlling missing-data handling in `Problem`.
* **DTW-AROW algorithm** (`warping_missing_arow.hpp`): One-to-one diagonal-only alignment for missing values (Yurtman et al., ECML-PKDD 2023). Linear-space, full-matrix, and banded variants.
* **5 new cluster quality metrics** in `scores.hpp`:
  * `dunnIndex()`: Min inter-cluster distance / max intra-cluster diameter.
  * `inertia()`: Total within-cluster sum of distances to medoids.
  * `calinskiHarabaszIndex()`: Medoid-adapted Calinski-Harabasz (uses overall medoid as global reference).
  * `adjustedRandIndex()`: Combinatorial agreement with ground-truth labels.
  * `normalizedMutualInformation()`: Information-theoretic agreement with ground-truth labels.

## Fixed

* **`warping_missing.hpp`**: Replaced `std::isnan()` with bitwise `is_missing()` check. The missing-data DTW feature was silently broken in Release builds due to `-ffast-math`/`/fp:fast` making `std::isnan()` unreliable.

## Changed

* **`Problem::fillDistanceMatrix()`** now pre-scans for NaN and throws with a helpful message under `MissingStrategy::Error`. Auto-disables LB pruning when missing data is detected under `ZeroCost`/`AROW` strategies.
* **`Problem::rebind_dtw_fn()`** dispatches based on `missing_strategy` member.

## Core

* **Parallel pruned distance matrix**: `fill_distance_matrix_pruned()` (Problem-based) is now parallelised with OpenMP. Uses lock-free atomic CAS on `nn_dist` for nearest-neighbor tracking across threads, with `schedule(dynamic, 16)` for load balancing. Precomputation of summaries and envelopes is also parallelised.
* **Distance matrix strategy selection**: new `DistanceMatrixStrategy` enum (`Auto`, `BruteForce`, `Pruned`, `GPU`) and `Problem::distance_strategy` member. `fillDistanceMatrix()` now dispatches based on the selected strategy. `Auto` (default) selects `Pruned` for Standard DTW variant, `BruteForce` for non-standard variants (DDTW, WDTW, ADTW, SoftDTW).

## CUDA

* **1-vs-N and K-vs-N GPU DTW kernels**: new `compute_dtw_one_vs_all()` and `compute_dtw_k_vs_all()` functions compute DTW distances from one or K query series against all N target series. Dedicated kernels (`dtw_one_vs_all_wavefront_kernel`, `dtw_one_vs_all_warp_kernel`, `dtw_one_vs_all_regtile_kernel`) use a 2D grid (targets x queries) with query loaded into shared memory per block. For k-medoids clustering with K=5 medoids and N=200 series, this avoids recomputing the full NxN matrix when medoids change (K*N = 1,000 pairs vs 19,900). Supports FP32/FP64 auto-precision, banded DTW, L1/squared-L2 metrics, and external query series. New result types: `CUDAOneVsNResult`, `CUDAKVsNResult`.
* **GPU-accelerated LB_Keogh pruning**: new `compute_lb_keogh_cuda()` function computes symmetric LB_Keogh lower bounds for all N*(N-1)/2 pairs on GPU. Two new CUDA kernels: `compute_envelopes_kernel` (sliding-window min/max per series) and `compute_lb_keogh_kernel` (embarrassingly parallel, one thread per pair). New `CUDADistMatOptions` fields: `use_lb_pruning` enables LB computation before DTW, `skip_threshold` prunes pairs with LB exceeding the threshold (set to INF without computing DTW). Standalone `CUDALBResult compute_lb_keogh_cuda(series, band)` API for use in clustering and nearest-neighbor search.
* **On-device pair index computation**: eliminated host-side pair index arrays (`h_pair_i`, `h_pair_j`) and their H2D transfers by computing `(i,j)` from the flat upper-triangle index directly on GPU via `decode_pair()`. For N=1000 this saves 4 MB of transfers and 2 device allocations.
* **GPU-side NxN result matrix**: kernels now write DTW distances directly into a symmetric NxN matrix on device, eliminating the per-pair distance array, its D2H transfer, and the host-side O(N^2) fill loop. A single contiguous `cudaMemcpyAsync` transfers the complete result.
* **Precomputed integer band boundaries**: banded DTW band checks (6 FP64 operations per cell) are now precomputed once per pair into shared memory (wavefront kernel) or registers (warp kernel), replacing per-cell FP64 math. On consumer GPUs with 1:64 FP64 rate, this eliminates ~192 FP32-equivalent cycles per cell in the banded path.
* **Stream-based async pipeline**: `launch_dtw_kernel` now uses a CUDA stream with `cudaMemcpyAsync` for H2D/D2H transfers and stream-ordered kernel launches. Pinned host memory (`cudaMallocHost`) is used for the two large buffers (flat series data and output distances) to enable true overlap of transfers with compute. Falls back gracefully to pageable memory if pinned allocation fails. GPU timing now uses `cudaEvent`-based measurement for accurate pipeline profiling. Added RAII wrappers (`PinnedPtr`, `CudaStream`, `CudaEvent`) to `cuda_memory.cuh`.
* **Persistent kernel mode**: the `dtw_wavefront_kernel` now supports persistent scheduling. When `num_pairs` significantly exceeds the GPU's resident block capacity (>4x), blocks loop over pairs using a global atomic work counter instead of the one-pair-per-block model. This eliminates block scheduling overhead for large-N workloads (e.g. N=1000: 499,500 pairs, but only ~80-160 blocks resident). Falls back to original behavior for small workloads. Fully transparent to the caller.
* Added optional CUDA GPU acceleration for batch DTW distance matrix computation (`DTWC_ENABLE_CUDA` CMake option, OFF by default).
* New `dtwc::cuda::compute_distance_matrix_cuda()` computes all N*(N-1)/2 DTW pairs on GPU.
* **Anti-diagonal wavefront kernel**: replaced single-threaded-per-block kernel with multi-threaded anti-diagonal wavefront parallelism. Achieved **24x kernel speedup** (910M -> 22 Gcells/sec), making GPU **5-7x faster than 10-core CPU** with OpenMP.
* **Warp-level DTW kernel for short series** (`dtw_warp_kernel`): for series with `max_L <= 32`, a new kernel packs 8 DTW pairs per block (one warp per pair) using register-based anti-diagonal propagation with `__shfl_sync()`. Eliminates shared-memory cost-matrix buffers and dramatically improves occupancy for short series workloads.
* **Register-tiled DTW kernel** (`dtw_regtile_kernel`): inspired by the cuDTW++ approach (Euro-Par 2020), a new kernel handles medium-length series (32 < max_L <= 256) using register tiling. Each thread processes a stripe of TILE_W columns entirely in registers, with inter-thread communication via `__shfl_sync`. TILE_W=4 covers up to 128 columns, TILE_W=8 covers up to 256. Eliminates shared-memory cost-matrix buffers for medium series, bridging the gap between the warp kernel (L<=32) and the shared-memory wavefront kernel (L>256).
* **`__ldg()` texture cache reads**: global memory reads for series data now use `__ldg()` intrinsic, forcing the read-only texture cache path for ~5-15% improvement on longer series.
* **Banded DTW on GPU**: the `CUDADistMatOptions::band` parameter is now honored by the kernel. Uses the same slope-adjusted Sakoe-Chiba window as the CPU `dtwBanded` implementation. When `band < 0` (default), full unconstrained DTW is computed with zero overhead.
* Python bindings expose `cuda_available()`, `cuda_device_info()`, `compute_distance_matrix_cuda()`, and `CUDA_AVAILABLE` flag.
* **GPU architecture detection** (`gpu_config.cuh`): runtime query and caching of GPU compute capability, SM count, shared memory limits, and FP64 throughput classification (Full vs Slow). Supports up to 16 devices.
* **FP32/FP64 templated kernel**: the DTW wavefront kernel is now templated on compute type (`float` or `double`). New `CUDAPrecision` enum (`Auto`, `FP32`, `FP64`) in `CUDADistMatOptions` controls precision. `Auto` selects FP32 on consumer GPUs (1:32 FP64 rate) and FP64 on HPC GPUs (1:2 FP64 rate), giving up to 32x throughput improvement on consumer hardware with ~1e-5 relative error.
* Requires CUDA Toolkit; all CUDA code is behind `#ifdef DTWC_HAS_CUDA` so the library builds without it.

## CI

* Added CUDA/MPI detection smoke test workflow (`.github/workflows/cuda-mpi-detect.yml`): Linux CUDA compile, Linux MPI build+test, macOS CUDA graceful rejection, macOS MPI build, and Windows MPI configure.

## Build system

* Fixed CUDA detection on Windows with multiple CUDA toolkit versions: auto-sets missing `CUDA_PATH_Vxx_y` env-vars and generates `Directory.Build.props` to persist `CudaToolkitCustomDir` for MSBuild at build time.
* Fixed MSVC flags (`/diagnostics:column`, `/fp:fast`, `/openmp:experimental`) leaking into nvcc by adding `$<$<COMPILE_LANGUAGE:C,CXX>:...>` generator expression guards.
* Fixed `find_package(OpenMP)` failure when CUDA language is enabled by requesting only the CXX component.
* Fixed MPI detection: `MPI_CXX_FOUND` variable didn't propagate from `dtwc_setup_dependencies()` function scope; now checks `TARGET MPI::MPI_CXX` instead.
* Improved MS-MPI SDK detection on Windows with fallback to default install path and actionable error messages.
* Added llfio dependency (optional, `DTWC_ENABLE_MMAP=ON`) for cross-platform memory-mapped I/O.

## Benchmarks

* Added GPU benchmark suite (`benchmarks/bench_cuda_dtw.cpp`): GPU vs CPU comparison, N-scaling, L-scaling with throughput counters (pairs/sec, cells/sec).
* Added MPI benchmark suite (`benchmarks/bench_mpi_dtw.cpp`): distributed distance matrix scaling across ranks with speedup/efficiency reporting.
* Added `benchmarks/README.md` with baseline performance numbers and optimization history.

## Performance / API

* Added `MmapDistanceMatrix` — memory-mapped distance matrix via llfio for large-N problems. Supports warmstart: reopen existing cache file to resume interrupted computation. Binary format with 32-byte header (magic, version, CRC32, N).
* Added `MmapDataStore` — memory-mapped contiguous cache for time series data. Supports variable-length and multivariate series. Binary format with 64-byte header (magic, version, CRC32, N, ndim) + offset table + contiguous data. Extracted shared `crc32.hpp` utility.
* Added `DataLoader::count()` — count series without loading data (directory iteration or line counting).
* Added pointer+length overloads for all core DTW functions (`dtwFull`, `dtwFull_L`, `dtwBanded`, `dtwMissing_L`, `dtwMissing_banded`) enabling zero-copy calls from bindings. The `detail::*_impl` functions now operate on raw pointers; vector overloads forward to them.
* Python `dtw_distance` and `dtw_distance_missing` now accept numpy arrays via `nb::ndarray` (zero-copy, no vector allocation).
* Eliminated vector copies in `dtwc::core::dtw_distance` pointer overload and `dtw_runtime`.

## CLI

* Rewrote `dtwc_cl` CLI with full TOML configuration file support via CLI11 `--config` flag.
* CLI now supports all clustering methods (FastPAM, FastCLARA, kMedoids Lloyd, MIP), all DTW variants (standard, DDTW, WDTW, ADTW, Soft-DTW), checkpointing, and flexible CSV output (labels, medoids, silhouette scores, distance matrix).
* Added `--method auto` (new default): auto-selects `pam` for N≤5000, `clara` for N>5000.
* Added CLARA sample size auto-scaling for N>50K: `max(40+2k, sqrt(N)*k)`.
* Added `--mmap-threshold` to control when memory-mapped distance matrix activates (default 50K).
* Added `--restart` to resume from binary checkpoint (distance matrix cache + clustering state).
* Added case-insensitive option validation for method, metric, variant, and solver flags.
* Added example TOML configuration file at `examples/config.toml`.

## Tests

* Added cross-language integration tests (`tests/integration/test_cross_language.py`): verifies C++ and Python interfaces produce identical results for DTW distances (L1/squared-euclidean, banded, missing-data), compute_distance_matrix, FastPAM/CLARA clustering, DTW variant consistency (DDTW/WDTW/ADTW/Soft-DTW), checkpoint save/load roundtrip, and end-to-end pipeline (data -> distance matrix -> clustering -> evaluation scores).

## Documentation

* Added MPI and CUDA installation guide (`docs/1_getting_started/3_mpi_cuda_setup.md`) covering all platforms (Windows, Linux, macOS), CMake detection, build flags, and troubleshooting.
* Updated installation guide: added Python installation section, GPU/MPI cross-references, macOS OpenMP note, and CUDA_PATH tip for Linux.

## Examples

* Added Python examples: `04_missing_data.py` (NaN-aware DTW), `05_fast_clara.py` (scalable clustering), `06_distance_matrix.py` (fast pairwise computation with timing), `07_checkpoint.py` (save/resume distance matrices).
* Added MATLAB quickstart example: `bindings/matlab/examples/example_quickstart.m` (DTW distance, distance matrix, clustering).
* Added C++ new-features example: `examples/example_new_features.cpp` (DTW variants, missing data, FastCLARA, checkpointing).

## Documentation
* Fixed DTW formula in `docs/2_method/2_dtw.md`: changed from squared L2 `(x_i - y_j)^2` to L1 `|x_i - y_j|` to match actual code implementation.
* Fixed pairwise comparison count: corrected from `1/2 * C(p,2)` to `C(p,2) = p(p-1)/2`.
* Fixed warping window description: band=1 allows a shift of 1 (not equivalent to Euclidean distance); band=0 forces diagonal alignment.
* Fixed spelling errors: "seies" -> "series", "wapring" -> "warping", "assertain" -> "ascertain".
* Added note clarifying that the default pointwise metric is L1 (absolute difference).
* Added z-normalization section with note that population stddev (N, not N-1) is used.
* Added `docs/2_method/5_algorithms.md`: clustering algorithm documentation with corrected FastPAM citation (JMLR 22(1), 4653-4688) and accurate complexity descriptions.
* Added `docs/2_method/6_metrics.md`: distance metrics documentation with LB_Keogh compatibility table and corrected Huber LB_Keogh reasoning.

## I/O

* Added **Python I/O utilities** (`dtwcpp.io`): `save_dataset_csv` / `load_dataset_csv` (always available), `save_dataset_hdf5` / `load_dataset_hdf5` (requires `h5py`), `save_dataset_parquet` / `load_dataset_parquet` (requires `pyarrow`).
* HDF5 files store series data, names, distance matrices, and metadata in a single compressed file.
* Added optional dependency groups in `pyproject.toml`: `hdf5`, `parquet`, `io` (both).

## New features

* Added **MATLAB MEX bindings** — new `bindings/matlab/` directory with C++ MEX API gateway and MATLAB `+dtwc` package: `dtwc.dtw_distance`, `dtwc.compute_distance_matrix`, `dtwc.DTWClustering`. Build with `cmake .. -DDTWC_BUILD_MATLAB=ON`.
* Added **checkpoint save/load** for distance matrix computation (`save_checkpoint`, `load_checkpoint`). Saves partial or complete distance matrices to disk as CSV + metadata text file, enabling resume after crashes.
* Added **binary checkpoint** (`save_binary_checkpoint`, `load_binary_checkpoint`) for clustering state (medoids, labels, cost, iteration). Used by `--restart` CLI flag.
* Added `count_computed()` and `all_computed()` methods to `DenseDistanceMatrix`.
* Added `distance_matrix()` accessors and `set_distance_matrix_filled()` to `Problem` class.
* Added Python bindings for `save_checkpoint`, `load_checkpoint`, and `CheckpointOptions`.
* Added **DTW with missing data** (`dtwMissing`, `dtwMissing_L`, `dtwMissing_banded`). NaN values contribute zero cost. Supports L1/SquaredL2, early abandon, banding. Reference: Yurtman et al. (2023), ECML-PKDD.
* Added `dtw_distance_missing` Python binding.
* Added **FastCLARA** scalable k-medoids clustering (`dtwc::algorithms::fast_clara`). Subsampling + FastPAM avoids O(N^2) memory. Reference: Kaufman & Rousseeuw (1990); Schubert & Rousseeuw (2021, JMLR).
* Added Python binding for `fast_clara()`.
* Added `MetricType` parameter (L1, SquaredL2) to all core DTW functions. Template lambda dispatch for zero inner-loop overhead.
* Refactored DTW into `detail::*_impl` helpers with distance callable template parameter.
* Added `metric` parameter to `dtw_distance` Python binding.
* Added `compute_distance_matrix` Python function with OpenMP parallelism.
* Added **LB-pruned distance matrix** (`compute_distance_matrix_pruned`). Precomputes envelopes and summaries once, then uses LB_Kim (O(1)) and LB_Keogh (O(n)) as early-abandon thresholds for each DTW computation. Reduces inner-loop work by 30-60% for correlated series. Enabled by default in the Python `compute_distance_matrix` binding via `use_pruning=True`.
* Added `distance_matrix_numpy()` method to `Problem` Python class.
* Added `-ffast-math` (GCC/Clang) and `/fp:fast` (MSVC) for Release builds.
* Added **Google Highway SIMD infrastructure** (`DTWC_ENABLE_SIMD` option, default OFF). Prototype kernels for future use.
* Added `#pragma omp simd` hints to LB_Keogh and z_normalize.
* Added LB_Keogh, z_normalize, envelope benchmarks.
* Added `DTWClustering` sklearn-compatible Python class.
* Added core type system (`dtwc::core` namespace).
* Added FastPAM1 k-medoids clustering algorithm.
* Added z-normalization, lower bounds, DenseDistanceMatrix, pruned distance matrix.
* Added Google Benchmark integration.

## Architecture

* Removed Armadillo dependency from all hot-path code.
* Replaced `arma::Mat` with `core::ScratchMatrix` and `core::DenseDistanceMatrix`.
* Unified `FastPAMResult` with `core::ClusteringResult`.

## Bug fixes

* Fixed `DenseDistanceMatrix` NaN sentinel broken under `-ffast-math` / `/fp:fast`. Replaced with boolean vector.
* Fixed `throw 1` bare integer throw — now throws `std::runtime_error`.
* Fixed `static std::mt19937` ODR violation — changed to `inline`.
* Fixed `.at()` in hot DTW loop — replaced with `operator[]`.
* Fixed `dtwBanded` 512MB/thread allocation — rolling buffer.
* Fixed `fillDistanceMatrix` integer overflow at N>46K.
* Fixed `dtwBanded` template default from `float` to `double`.
* Fixed `static` vectors data race in `calculateMedoids`.
* Fixed DBI formula.
* Fixed broken examples.
* Renamed `cluster_by_kMedoidsPAM` to `cluster_by_kMedoidsLloyd`.
* Fixed z_normalize tests.

<br/><br/>
# DTWC v1.0.0

## New features
* HiGHS solver is added for open-source alternative to Gurobi (which is now not necessary for compilation and can be enabled by necessary flags). 
* Command line interface is added. 
* Documentation is improved (Doxygen website).

## Notable Bug-fixes
* Sakoe-Chiba band implementation is now more accurate. 

## API changes
* Replaced `VecMatrix<data_t>` class with `arma::Mat<data_t>`. 

## Dependency updates:
* Required C++ standard is reduced from C++20 to C++17 as it was causing `call to consteval function 'std::chrono::hh_mm_ss::_S_fractional_width' is not a constant expression` error for clang versions older than clang-15.
* `OpenMP` for parallelisation is adopted as `Apple-clang` does not support `std::execution`. 

## Developer updates: 
* The software is now being tested via Catch2 library. 
* Dependabot is added. 
* `CURRENT_ROOT_FOLDER` and `DTWC_ROOT_FOLDER` are seperated as DTW-C++ library can be included by other libraries. 

<br/><br/>
# DTWC v0.3.0

## New features
* UCR_test_2018 data integration for benchmarking. 

## Notable Bug-fixes
* N/A

## API changes
* DataLoader class is added for data reading. 
* `settings::resultsPath` is changed with `out_folder` member variable to have more flexibility. 
* `get_name` function added to remove `settings::writeAsFileNames` repetition)
* `std::filesystem::path operator+` was unnecessary and removed. 

<br/><br/>
# DTWC v0.2.0

A user interface is created for other people's use. 

## New features / updates
- Scores file with silhouette score is added. 
- `dtwFull_L` (L = light) is added for reducing memory requirements substantially.  

## API changes
- Problem class for a better interface. 
- `mip.hpp` and `mip.cpp` files are created to contain MIP functions.

## Notable Bug-fixes
* Gurobi better path finding in macOS. 
* TBB could not be used in macOS so it is now option with alternative thread-based parallelisation. 
* Time was showing wrong on macOS with std::clock. Therefore, moved to chrono library.

## Formatting: 
- Include a clang-format file. 

## Dependency updates
  * Required C++ standard is upgraded from C++17 to C++20. 

<br/><br/>
# DTWC v0.1.0

This is the initial release of DTWC. 

## Features
- Iterative algortihms for K-means and K-medoids 
- Mixed-integer programming solution support via YALMIP/MATLAB. 
- Support for `*.csv` files generated by Pandas.  

## Dependencies
  * A compiler with C++17 support. 
  * We require at least CMake 3.16. 