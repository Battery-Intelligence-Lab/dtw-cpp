# DTWC++ 2.0 — PLAN

Branch `design-2.0`. Written 2026-09-21 against `MAP.md` (as-is) and `design.md` (target).
Row IDs (`C-`, `O-`, `A-`, `B-`, `T-`, `G-`) are rows of `specs/2026-09-07-diff-ledger.md`; `F-` and `D-`
numbers are the public finding / derivation numbers of the archived plan; `S-`, `X-`, `H-` are new here (§5).

## 0. How to use this plan

**Reading order for any session:** `CLAUDE.md` → `CHARTER.md` → `MAP.md` → `design.md` → *your wave's
card below* → the ledger rows it names → the deep-dive section a row cites. Nothing else. A wave card
lists its **context pack**: the only source files an implementer should open before starting.

**Who does what:** the orchestrating session plans, dispatches and reviews; implementers (Opus, ≤ 4 at a
time, never on the same files) take one card slice each: *failing or pinning test first → change → gate →
one commit*; a separate adversarial reviewer checks every wave's diff for design fit, new branches /
allocations / locks in hot paths, and a grep proof for every deletion. Decisions in §7 are Volkan's.

**Status legend:** ☐ todo · ◐ in progress · ☑ done · ✗ falsified (with evidence) · ⏸ blocked on a decision.
Row-level status lives in the ledger (add a `status` column when a wave opens); this file tracks waves.

## 1. Where we are (2026-09-21)

- Released: **v1.0.0**. `VERSION` = `2.0.0rc1`, untagged, 781 commits later. Users are on the 1.x shape.
- Spec and 129-row ledger **approved 2026-09-07**; every III.10 decision adopted (`DECISIONS.md` §3).
- **W0 ◐**: tasks 0, 1a, 1b, 1c done (`d21ffee`, `22c4c5f`): every test registers through
  `dtwc_add_test`, floors in `tests/floors.cmake`. Tasks 2–13 open.
- **Two regressions from that work, found and verified today** (`baselines/2026-09-21-macos-first-baseline.md`):
  floors were measured on Windows only → **5 of 131 tests fail on macOS on floors alone**; and the CI docs gate
  fails (`D2/D3 CTest drift`) because it still pins the old registration blocks. Neither was noticed because
  **pushes to `design-2.0` trigger no workflow**.
- New development machine is a Mac (Apple clang 21, libomp, Metal): the tree builds in 2.5 min and the serial
  suite runs in 129 s. Metal is now local; **CUDA is `[BLOCKED-ENV]` here**.
- `design.md` proposes eighteen amendments to the spec (§11 there); they need approval (§7, D-1). A1–A10 came
  from the mapping; **A11–A18 from a four-lens review on 2026-09-22** (Python API, C++ language, C++ software
  design, time-series science), each claim re-opened at its line before it entered the plan.

## 2. Dependencies

### 2.1 What a machine needs

| Need | macOS (this machine) | Linux | Windows | Required? |
| --- | --- | --- | --- | --- |
| C++20 compiler | Apple clang 21 (Command Line Tools) | GCC 11/12, Clang 14–17 | MSVC 19.3x, clang-cl | yes |
| CMake ≥ 3.26, Ninja | `brew install cmake ninja` | apt / pip | installer | yes |
| OpenMP | `brew install libomp`, configure with `-DOpenMP_ROOT=/opt/homebrew/opt/libomp` | bundled | `/openmp:experimental` | yes, unless `-DDTWC_ALLOW_SEQUENTIAL=ON` |
| Python via **`uv`** | present | — | — | tests (Python ≥ 3.9), scripts, wheels |
| Doxygen + Graphviz | `brew install doxygen graphviz` | apt | — | docs, `repo_map.py symbols` |
| Hugo extended 0.147.8, Pagefind 1.3.0 | not installed | CI | — | docs site only |
| CUDA toolkit | unavailable | CI smoke / RTX box / ARC | RTX box | optional — **W2b CUDA rows cannot be verified here** |
| MATLAB R2024b+, MPI, Gurobi, Arrow | not installed | CI | CI | optional |

Fetched at configure time (network needed until B-17 vendors CPM): Eigen 5.0.1, CLI11 2.6.2, fkYAML 0.4.4,
HiGHS 1.15.1, llfio + quickcpplib (commit pins), Catch2 3.13.0; on demand Arrow 19.0.1, Google Benchmark 1.9.5,
nanobind ≥ 2.0. Vendored: nanoarrow 0.8.0. Full table with gates: `MAP.md` §6.

```sh
cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp && cmake --build --preset clang-macos
ctest --test-dir build -C Release -j1 --output-on-failure      # serial is the evidence run
uv run --no-project python scripts/repo_map.py layers           # layer ratchet: 18 upward edges today
python3 scripts/check_record_hygiene.py && python3 scripts/check_repo_hygiene.py && python3 scripts/check_docs_contract.py
```

### 2.2 What blocks what

```mermaid
flowchart LR
  W0["W0 gates + tooling ◐"] --> W1["W1 base"]
  W1 --> W2["W2 core + seams"]
  W1 --> W5a["W5a io readers"]
  W2 --> W2b["W2b backends"]
  W2 --> W3["W3 algorithms + mip"]
  W2 --> W4["W4 session"]
  W3 --> W4
  W4 --> W5b["W5b io writers"]
  W5a --> W5b
  W4 --> W6["W6 cli + api + Config"]
  W5b --> W6
  W6 --> W7["W7 tests"]
  W2b --> W7
  W7 --> W8["W8 docs"]
  W8 --> R6["R6 release gate → tag 2.0.0 (Volkan)"]
  R2["R2 derivations D4–D19"] -. "each pins after its code's wave" .-> W7
  R2 -- "D7 lands with S-01" --> W2
  R2 --> R6
  W6 --> W9["W9 interface, all languages"]
  W4 --> W10["W10 scale-out"]
  W2b --> W10
  W9 --> W10
  R6 -.-> R7["R7 WASM playground (2.1)"]
```

**Critical path:** W0 → W1 → W2 → W3 → W4 → W5b → W6 → W7 → W8 → R6. **Parallel lanes:** W5a beside
W2–W4 (io needs only base + core — a gain from the corrected layer order; its entry gate is W1's exit); W2b
beside W3; R2 derivations beside everything **except D7, which W2 needs**. W9's additive parts and all of W10
may land after the tag (§6 marks what must be `pre-tag`).

Where a ledger row names two waves, the card of the first wave lists it and the second half rides with the
file's own wave (A-01, A-10, O-10, O-11, O-19, B-20); G-09 follows the spec's IV.2 placement (W2b), not the
ledger's W7; the interface-campaign halves of A-04 and A-05 (exposing LR / CLARANS knobs; keep-or-delete) are W9.

### 2.3 Row-level ordering that bites

| First | Then | Why |
| --- | --- | --- |
| W0-R1, W0-R2 (repairs) | anything | a red baseline proves nothing |
| W0 Task 2 (T-15 read-only, hash-pinned conformance) | every "digit-identical" claim (W1 onward) | until then conformance is not a gate |
| R5 five canonical baselines + W0 Task 11 benchmarks | W2, W4 | the no-regression proof and the O-01 / C-06 / C-07 evidence must predate the change |
| W0 Task 9 manifest **with `design.md` §4 ranks** | W1 | the spec's order (io above session) can never go strict |
| W1 `parallel::for_each_pair` | C-07 | the brute-force fill adopts it |
| T-11a: move `dtwFull` / `dtwMissing` / `dtwAROW` full-matrix oracles into `tests/support` | C-09 deletions | the oracles must outlive their library copies |
| C-05 (`Auto` = BruteForce) | C-04 (one pruned fill) → C-25 dies | decides what the pruned fill is for |
| C-02 (`visit_variant`) | C-14, C-23, C-21b, T-10 equivalence test | one table, then everything that leans on it |
| C-01 (`DistanceConfig` by value), W2 | O-02, C-22, O-24, G-07 | the closure must stop capturing `*this` first |
| `DistanceOracle` + `Problem::oracle()` | A-02 (values out) → A-03 (one algorithm per task) → A-19, A-12 (D14) | O-03 step 1 before step 2 |
| A-02 | O-03 step 2, O-16 (Benders `friend`), Tier-1 `Result` | Tier-1 silently depends on the write-back today |
| O-08 (correct DCL, bind at the gateway) | A-23 | three hand-rolled "prime serially" sites collapse into it |
| W4 X-01 / X-02 (execution target, resolver) | W6 `Config`, W10 | `device` must exist on `Problem` before a run description can name it |
| O-09 decision (D-2) | W5b | one writer module, with or without a legacy writer |
| T-01 (cross-file call-count assertion) | every other W7 edit | each edit trips it |
| D17 (error model) | W7 tolerance forms, any SIMD prototype | derive the band, do not guess |
| D7 (Soft-DTW negativity) | S-01, W2 exit | same subject: the derivation decides what the diagonal is; fix and derivation land together |

## 3. Waves

Every wave: entry gate = previous exit gate; exit = its gate green on **macOS + the CI matrix**, conformance
digit-identical for every "no-op" claim, upward-edge count not higher, CHANGELOG line per user-visible change,
run-log in `baselines/`, handoff in `summaries/`.

### W0 — gates and tooling ◐
**Goal:** the gates exist, and are true, before anything they protect changes.
**Repair first (new, verified 2026-09-21):**
- ☐ **W0-R1** floors portable: re-run `scripts/measure_test_floors.py` with `ctest -V` logs from Windows
  clang, MSVC Debug, macOS Apple clang and Linux GCC; it already takes the minimum. Case floors stay exact.
- ☐ **W0-R2** docs gate: `check_docs_contract.py:1195-1203` (D2) and its D3 twin near `:1843` pin the
  `if(TARGET test_lb_…_derivation) … endif()` blocks that `d21ffee` replaced; pin the `dtwc_add_test` registration.
- ☐ **W0-R3** floor holes: an explicit `ASSERT_FLOOR` silently beats the measured table
  (`cmake/DtwcTest.cmake:121-130`); `MAY_SKIP` carries no floor (`:157`); seven device tests sit at `1;1`;
  `unit_test_mpi` passes on its own skip line. Read each site first — some explicit floors are said to be
  deliberately low on a build-flavour branch.
- ☐ **W0-R4** CHANGELOG: `22c4c5f` has no entry and the `d21ffee` bullet claims "every CTest entry is now gated".
- ☐ **W0-R5** CI: add `design-2.0` to the push filters of the unit workflows (or keep a draft PR to `main` open).

**Then** tasks 2–13 of `plans/2026-09-07-w0-baseline-tooling.md`, with these amendments: *Task 2* — the file is
`tests/conformance/cpp_conformance.cpp`; the reference **is** tracked, so the holes are exactly two: regen rewrites
tracked data and compares with what it wrote, and an absent reference passes vacuously. *Task 9* — port the manifest
from `scripts/repo_map.py` (`design.md` §4 ranks), not the spec's. *Task 12* — extend the disassembly script into
the **codegen report** (X-04): probe TU, vectorisation remarks, expectation table seeded from today's probe.
*Task 13* — floors live in `tests/floors.cmake` and the run-log (`AGENTS.md` is gone); the exit matrix gains macOS.
Add the **five canonical baselines** of R5 to the exit gate (dense N = 1000 × 512; banded 10 %; PAM N = 2000;
OneBatchPAM 50k; barycenter k-means k = 3), counters beside every time.
**Rows:** T-15, T-16, B-12, B-13, B-14, B-15, B-17, C-08, C-24, A-06 (evidence), S-03, S-09, **X-15**
(`DTWC_FP_MODEL=strict|fast`: FP flags `PRIVATE` to `dtwc++`, the conformance reference pinned under
`strict`, one CI leg per model — lands with S-03 because it is the same CMake lines).
**Context pack:** the one task section being executed · `cmake/DtwcTest.cmake` · `tests/CMakeLists.txt` ·
`tests/floors.cmake` · `baselines/2026-09-07-design-2.0-W0.md` · `baselines/2026-09-21-macos-first-baseline.md`.
**Gate:** all matrices + MSVC Debug + macOS green with every test floored; three gate scripts green; layer
report = 18; benchmarks and baselines recorded.

### W1 — base
**Goal:** a foundation layer that includes only the standard library, and the two parallel helpers.
**Rows:** C-11 (move `error`, `settings`, `missing_utils`, … into `dtwc/base/` by script, **forwarding headers at
the old paths for one release**), C-12, A-10, B-11 (saturating arithmetic), C-21a (`settings.hpp` stops pulling
`<iostream>`; `<random>` goes only when the process-global `randGenerator` it defines at `:55` moves to its own
header — **X-12**, done here, the engine itself stays for A-21), `parallel::for_each_pair` + `reduce`, X-05
(declare `available_ram_bytes()` in a base header — removes the only upward edge that is not `→ Problem.hpp`),
one `narrow<int>` / `checked_cast` in base used at gateways (`dist_by_ind(int,int)`, `ClusteringResult::
n_clusters()`; `fast_pam.cpp:60` already does it right), **X-16** the typed-throw ratchet script (256 untyped
`throw std::…` vs 227 typed today; the count may only fall), and the `settings::paths::{data,results}`
mutable globals become `Config` values (with X-12).
**Context pack:** MAP §2 *base*, §5 · `dtwc/{error,settings,missing_utils,parallelisation,timing}.hpp` ·
`dtwc/types/` · `dtwc/system_memory.cpp` · `dtwc/CMakeLists.txt`.
**Gate:** floors unchanged; layer check strict for `base`; upward edges 18 → 17; bindings still compile.

### W2 — core and the seams
**Goal:** `core/` names neither `Problem` nor a loader; one dispatch table; one pruned fill; one unit of work.
**Rows:** C-01, C-02, C-03, C-04, C-05, C-06 (hoist the visit only), C-07, C-09 (under D-3), C-10, C-14, C-15,
C-19, C-20, C-21b, C-23, C-25; F47 (with C-09 and `check_docs_contract.py:1039`), F49 (= C-04, carry its gate);
**S-01** with **D7** (a hard dependency: derivation D7 must land in or before this wave), S-06, S-07, S-12.
**New:** X-06 — `DistanceConfig` (semantics only: variant, band, missing, ndim, metric, precision; it is
today's private `DistanceCacheConfiguration` made public), `PairRange`, and **X-13** `Data` templated on element
type *if* that collapses the f32/f64 twins (otherwise no `SeriesSource` concept); the concrete
`PackedOracle` and the oracle concept for lazy consumers (A12); **X-17** `distance::dtw_path`
(Standard/DDTW, banded, L1/SqL2 — the backtrack in `barycenter.cpp:184-224` becomes the one path kernel,
which also settles C-26's duplicate and A-20); C-15 as `concept`s for Cost/Cell; C-19 refined: `L2` wired
as the multivariate per-cell Euclidean only, plus the additive interop tokens `euclidean` (√ of the
squared-L2 total, applied outside the kernel) and `window_fraction`; `DistanceMatrixStrategy` moves from
`Problem.hpp` to `core` (its consumer is the pruned fill) with a `using` alias at the old name.
**Notes carried from the ledger:** C-17 — `.dtws` has no crash-consistency machinery; a feature decision,
recorded, not scheduled. **Registered here:** derivation **D19** (missing-data strategies: AROW vs Yurtman
et al. 2023 [verify], ZeroCost, Interpolate) — none of D4–D18 covers them.
**Context pack:** MAP §2 *core* · `core/dtw_dispatch.*`, `core/dtw.*`, `distance.hpp` · `warping.hpp:130-300`
(for C-03) · `core/pruned_distance_matrix.*` · `Problem.cpp:741-845` (the fills) · `core/mmap_data_store.hpp`.
**Gate:** conformance digit-identical; cells-computed band; fill-vs-facade equivalence on every variant ×
missing pair; UBSan on touched kernels; codegen report unchanged for *must-vectorise*; upward edges 17 → 15.

### W2b — backends
**Rows:** G-01, G-02 (Metal — **local now**), G-03 + F50, G-05, G-06, G-07 (MPI consumes the bound closure),
G-09; F27, F28, F29, F12 (Metal half), F31 (typed CUDA errors); add compute-sanitizer to the CUDA leg.
**X-14** one `Precision` enum (`CUDASettings::precision` is an `int`; the CUDA and Metal enums become
aliases of `core::Precision`). Fillers implement the core-owned Strategy and return a **packed slice**
for their `PairRange` (A13) — the N×N result buffer and its element-wise copy go.
**Context pack:** MAP §2 *backends* and §10 (line-anchor caveat) · the one backend being edited.
**Gate:** Metal suite real on this Mac; CUDA on CI / the RTX box (`[BLOCKED-ENV]` here, record the probe);
GPU vs CPU oracle within the recorded tolerance; `mpiexec -n 2` suite.

### W3 — algorithms and MIP
**Goal:** algorithms take an oracle and return a value; one Balinski model.
**Rows:** O-03 step 1, A-01, A-02, A-03, A-04 (composition only), A-05, A-07, A-08, A-09, A-11, A-12, A-13,
A-15, A-16, A-17, A-18, A-19, A-21 + S-11, A-22; F48.
**Shape:** `PackedOracle` + `Problem::oracle()` first (checks "filled" once — O-13 — instead of two
validations and a `std::visit` per lookup); then A-02; then A-03 **one algorithm per task**, FastPAM first (it
is the workhorse behind Tier-1 `pam`, CLARA and the MIP warm start). The frozen Tier-2 free functions
`fast_pam(Problem&, …)` stay as thin **adapters declared in a session header** (`dtwc/problem_algorithms.hpp`)
and implemented in one TU; a signature naming `Problem&` inside `algorithms/` would keep the layer bound
even at zero includes, so the old declarations survive one release as a manifest-listed exemption and the
layer check learns to flag foreign-layer types in signatures. Also here: `MIPSettings` moves to `mip/`
(alias at the old name); `init_fun`'s default becomes a seeded lambda over `random_seed()` through a new
`init::random_seeded(N, k, seed)` (the `std::function<void(Problem&)>` field is a session⇄algorithms
dependency by default member initialiser; A-21); **X-18** the UCR-Suite lower-bound cascade with early
abandon in the *nearest-medoid assignment* of CLARA, OneBatchPAM and barycenter k-means — an argmin over k
needs only the winner exactly, so this is admissible and not on the killed list (registered counter band:
DTW calls saved); **X-20** (**S-20**) scores take the oracle, with the O(N·k) medoid silhouette and inertia from the
assignment (with D18); the `fast_pam.hpp` / `fast_clara.hpp` headers cite "JMLR 22(1)" — the paper is
Information Systems 101 (2021), as `CITATIONS.md` has it.
**Context pack:** MAP §2 *algorithms*, *mip*, *session* (flows) · the one algorithm's `.hpp/.cpp` ·
`core/clustering_result.hpp`, `core/medoid_assignment_policy.hpp` · `mip/solution_transaction.*`.
**Gate:** per-algorithm fingerprints + conformance; byte-identical objects where a no-op is claimed; MIP
tests; **each matrix-based algorithm runs on a matrix-only `Problem`**; upward edges 15 → 6. The nine
`algorithms → Problem.hpp` edges close here; the six `mip → Problem.hpp` edges cannot until the Benders
`friend` goes (O-16, W4) and `ExactClusteringTransaction` — which mutates `Problem` and becomes *the* publish
path under A-02 — moves to the session layer (**X-11**, W4). W4's gate takes the count to 0.

### W4 — session
**Goal:** `Problem` is a façade over owned parts behind unchanged setters; the hot lookup is O(1).
**Rows:** O-01, O-02, O-03 step 2 (`DistanceCache`, `ClusterState`), O-04, O-05, O-06, O-08, O-13, O-14, O-16,
O-24, C-13, C-22, A-23, F25, **S-13**; **X-01** `ExecutionTarget` + `Problem::set_device` (additive), **X-02** one
execution resolver (absorbs G-04).
**Context pack:** MAP §2 *session* · `Problem.hpp` · `Problem.cpp:280-330, 440-460, 580-730, 856-1090` ·
`api.cpp:95-130` (`configure_device`) · `env.hpp` · W0 Task 11 / 12 results.
Also **X-11**: the publish transaction moves from `mip/` into the session layer (with O-16), and the
`mip-solvers` OBJECT library is folded into `dtwc++` (or loses its `PUBLIC .` include dir): today every mip TU
compiles `Problem.hpp` under its own `DTWC_HAS_MMAP` — an ODR hazard `mip/CMakeLists.txt:38-52` itself
documents, and a build-level cycle the include graph cannot see. **S-08** lands here, not W9: the llfio leak is
`Problem.hpp:24` (included for the `std::variant` member) and closes when `DistanceCache` holds the storage
behind a `unique_ptr` while the frozen `distMat_t` alias and `distance_matrix()` declaration stay.
**X-02 also enforces band feasibility**: band < |n − m| for any pair ⇒ typed error at the fill gateway
(today a finite `max()` reaches the objective — R1; per-pair widening only as an explicit token, D-12).
**X-09a** `RunStats` (pairs and cells computed, prunes, swaps, iterations, device used) is a C++ session
value returned by the gateways *here*; W9 only exposes it. `Data` gets accessors and its public storage
fields are deprecated (INTERNAL). Cache identity = `DistanceConfig` only (A11, **S-16**).
**Gate:** conformance; registered branch-count band on `dist_by_ind`; use-after-move test; GPU + unsupported
setting ⇒ typed error naming the axis; infeasible band ⇒ typed error; Python import smoke; **upward edges
6 → 0**, layer check strict everywhere; typed-throw count not higher.

### W5 — io
**W5a readers (parallel lane):** B-08, B-09, B-10, B-11, O-15, O-17, F44.
**W5b writers (after W4):** O-09 (under D-2), B-05, O-07, B-06, O-10.
**Context pack:** MAP §2 *io* · the one reader, or `Problem_IO.cpp` + `api.cpp:240-300` + `dtwc_cl.cpp:630-700`.
**Gate:** real-binary integration tests byte-identical; reader-hardening tests run in Release.

### W6 — CLI, Tier-1 and `Config`
**Rows:** B-01, B-02, B-03, B-04, B-07, O-11, O-12, O-18, O-19, O-21 → **X-03** `dtwc::Config` +
`run(Config) → Result` (the promoted `CliConfig`, composed as design.md §6 says: existing option structs,
member-initialiser defaults read by CLI11, snake_case keys + kebab alias table, `schema = 1`, `apply(Config,
Problem&)`), S-04 (a failed checkpoint save or `--dist-matrix` load is an error, not a warning with exit 0),
S-10, F37 (C++ side); **X-19** (**S-17**) Tier-1 `Result` owns `ClusteringResult` + `RunStats` by value (copies no longer
share a mutable `Problem`; the lazy matrix path sits behind a `std::once_flag`); **X-22** ground truth through
Tier-1: `load(..., label_col=)` (additive, contract addendum) and `Result::score("ari" | "nmi")`; the
`euclidean` and `window_fraction` tokens reach the CLI and `Config`; `[[nodiscard]]` on the bool-returning
`set_solver` / `load_checkpoint`.
**Context pack:** MAP §2 *surface* · `dtwc_cl.cpp` by the line ranges in the MAP card · `cli/config_file.hpp` ·
`api.*` · `enums/Method.hpp`.
**Gate:** the four real-CLI gates + the config-format test byte-identical; flags ≡ config keys ≡ `Config` fields.

### W7 — tests
**Rows, in slices:** (1) T-01 alone; (2) support library T-05, T-13, T-19 and the oracle moves of T-11
(if not already done in W2); (3) merges and deletions T-03, T-08, T-09, T-14, T-17, T-18, one commit each naming
the covering test; (4) rewrites T-06, T-10; (5) slow tests T-04, T-12; (6) dark and untracked tests T-02, T-20,
G-08, plus the real-binary resume test B-21 and T-21; (7) layout T-07 in the same commits as the merges;
(8) hidden floors A-24; the F22 apparatus B-19; **X-07** counters-prove-the-path tests (they assert the
`RunStats` that X-09a delivers in W4 — the earlier draft had X-07 before `RunStats` existed), **X-08**
quality-oracle tests (every heuristic within a registered gap of the exact optimum on small fixtures);
**X-21** the `env()` / `settings::paths` grep gate (no reads outside `api`, CLI, bindings) and a
`static_assert(std::is_nothrow_move_assignable_v<Problem>)` that settles S-13. T-15 and T-16 were W0.
**Context pack:** MAP §8 · the taxonomy report section for the files being merged · `tests/support/`.
**Gate:** contract coverage non-decreasing (each merge names its covering test); all matrices at the new inventory.

### W8 — docs
**Rows:** O-20, O-22, O-23, A-14, A-25, B-16, B-18, B-20, B-22, G-10, G-11; `design.md` §9 receives the 66
invariants; README truths (HDF5 is Python-only; `dtwc_main` takes no arguments; `develop` badge; `/convert`);
Doxygen as a `docs` target with `PROJECT_NUMBER` from `VERSION`; **the Pages upload has no `path:`** while the site
is built in `docs/public`; H-2, H-3, H-5. Science truths: an **interop conventions page** (L1 / integer-cell band /
no final sqrt vs dtaidistance, tslearn, aeon; the `euclidean` and `window_fraction` tokens; one same-definition
parity test per peer on the benchmarks page, which today times four different distances as if equal), ADTW credited
to Herrmann & **Webb** (Pattern Recognition 137, 2023; `dtw-variants.md:113` says Shifaz), the DTW_D/DTW_I note
(Shokoohi-Yekta et al. 2017 use a squared-Euclidean per-cell cost; ours is L1 — D9 holds for additive costs),
ARI per UCR dataset once X-22 lands, D15 citing Avella, Sassano & Vasil'ev (2007).
**Gate:** docs-contract + hygiene green; site builds; link check.

### W9 — one interface for every language
**First row (S-18) — Python Tier-1 becomes one call.** `_api.py` re-implements `api.cpp` (method resolution,
dispatch, `Result`) in ~600 lines while MATLAB makes one call into `dtwc::cluster` (`dtwc_mex.cpp:1488`);
`cluster(data, k, **options)` becomes "build `Config`, call `run`", `_hpc` submits the same `Config`, F37 closes
with it, and the Python-side device store `_HPC_SELECTED` (a second source of truth beside `Env`) is deleted
(F24 reuses Env's three messages).
**Pre-tag surface slice (D-11 splits W9 here):** the 13 Python alias shims and the `_F22` import guard, checked
one by one against `git show v1.0.0:python/py_main.cpp` (names that shipped there — `cluster_size`,
`refreshDistanceMatrix`… — keep their shims; names born in 2.0 — `ClusterResult`, `get_device`, the five
`*_index` duplicates, `distance_matrix_numpy`, `n_repetition` — go); `IOError` and `test` leave `__all__`
(`from dtwcpp import *` rebinds the builtin to an `OSError` subclass — the attributes stay, contract §10.5);
**one estimator** (`DTWClustering` per contract §1.5, with `random_state`, `n_jobs`, keyword-only `__init__`;
`DTWCKMedoids` folded, D-14); `.pyi` stubs via `nanobind_add_stub` (`py.typed` ships with none); `str | Enum`
setters from the C++ tables and the missing `MVMode` export; **S-14** `distance.msm` / `distance.twe` in Python's
dispatcher and MATLAB's `+distance` (reachable today only through Tier-2 `variant_params`); labels dtype pinned;
`set_data(series, names=None)`; keyword-only `distance.pairwise(...)` beside the boolean-trap
`compute_distance_matrix(..., use_pruning)`.
**Then:** X-03 in Python (`Problem.from_config`) and MATLAB; **X-09** exposure — `Result.stats`, a
`progress=callable(done, total)` polled per block on the calling thread with `PyErr_CheckSignals()` so Ctrl-C
works during a GIL-released fill, `verbose` routed through the same callback (C++ `std::cout` never reaches a
notebook), the cancel flag a relaxed `std::atomic<bool>` — the same after-block seam the checkpoint save uses;
zero-copy `set_data` / `Data` / `compute_distance_matrix` for 2-D float64/float32 arrays and lists of 1-D arrays
(F26 extended — every Python path copies through Python floats today); X-01 as `Problem(device=…)`;
`test.hpc()` beside `test.parallelisation()` / `test.gpu()` (`SlurmRemoteRunner.preflight()` exists) and
`gpu()` reporting every device; **S-02** an installable package (`find_package(dtwc)`; after S-08) and a
**C ABI** (opaque handle, `Config` text, arrays, status + `last_error`) as the only binary-stable surface —
never a C++ shared ABI; F52, F53, F56, **S-05**; the interface halves of A-04 / A-05.
**Gate:** conformance in three languages; a parity test generated from the C++ tables (it is also the stub
source); zero-copy checks; the Python floor question (D-13).

### W10 — scale-out (2.1 unless pulled forward)
**Rows:** G-07 in full (MPI as a strategy), **X-10** sharded fill (`--shard i/n`, `--merge`; with packed slices
the merge is a `memcpy` per range), D-10 (C++ Tier-1 `hpc`), SLURM job templates that consume `--config`;
TODO S03, G02, G05, A01, A02. **2.1 science additions with evidence** (TODO rows): SBD / k-Shape (Paparrizos &
Gravano 2015 — the one widely used shift-invariant, matrix-free competitor; naive O(n²) cross-correlation, no FFT
dependency), NN-chain hierarchical (Murtagh 1983; Müllner 2011 — O(N²) for single/complete/average, lets the
`max_points = 2000` guard go), AMI (Vinh, Epps & Bailey 2010).
**Gate:** shard + merge digit-identical to a single fill; `mpiexec -n 2` equivalence; BLK01 ARC run (Volkan).

## 4. Science and release tracks (carried from the archived plan)

| Track | Open | Rule |
| --- | --- | --- |
| **R2 derivations** (3 of 19 done: D1–D3) | D4 EAP slack constant 16 · D5 MSM · D6 TWE · D7 Soft-DTW negativity bound (recommendation: the Soft-DTW *divergence* of Blondel, Mensch & Vert 2021 — D(i,i)=0, D(i,j)>0, N extra self-distances, and D13's shift hack dissolves because D ≥ 0) · D8 WDTW / ADTW / DDTW limits · D9 DTW_I ≤ DTW_D · D10 FastPAM · D11 OneBatchPAM · D12 CLARA · D13 signed D-sampling · D14 TADPole · D15 LR-core · D16 barycenters · D17 error model + roofline (also: does the LB slack cover any summation order under `fast`?) · D18 scores (medoid-based definitions; the O(N·k) silhouette) · **D19 missing-data strategies** (new) | one file per topic in `docs/derivations/`, a verdict table, every DISCREPANCY becomes a numbered finding with a test. **D7 is needed by W2** (S-01). **Pin each code-conformance table after the wave that owns that code** (D4–D6, D8, D9, D19 after W2; D10–D16, D18 after W3; D17 after W0's baselines exist); D1–D3 tables need re-pinning after W1–W2 move files |
| **R3 findings** | evidence-only, close with the record: F11, F16, F17, F20, F22 (→ B-19), F32 (fixed), F39. Scheduled above: F12, F25, F27–F31, F37, F44, F47–F50. Small: F35, F36, F38. One MATLAB MEX run each: F41, F42, F43. Partly covered: F30 (G-04), F34 (B-17), F46 (A-10) | keep the numbering — it is public in the docs and pinned by the checker |
| **R3 lenses** | 21 review lenses | run once after W4 and once after W7, sized by consequence; the "two clean rounds" band is dropped |
| **R5 fences** | invariant suite (= spec II.6, A-24) · five canonical baselines (**moved into W0**) · hot-path profile pass (owns C-18: the banded kernel's setup passes, measured with `BM_dtwBanded` before any change) · PMU artefact for both regimes (O-22) | counters decide; SIMD stays killed unless D17 plus the profile show a correctness-complete route and a measured opportunity. D4 owns C-16 (the EAP slack constant) |
| **R6 release gate** | matrices + full pytest + MATLAB re-floored · capacity model within ± 20 % RSS at N ≥ 500k · rc2 CHANGELOG · llfio-ON wheel · one adversarial pass | tag, PyPI and ARC are Volkan's |
| **R7 WASM playground** | 29 tasks, its pre-made decisions in the archive (lines 1042–1051) | 2.1; never gates the tag; R7.2's `warping_path` is delivered early by X-17 (W2) — the backtrack already exists in `barycenter.cpp`, so "a design addition" (ledger A-20) was wrong |

## 5. New rows

| ID | Finding or element (verified at the cited line unless marked) | Wave |
| --- | --- | --- |
| S-01 | `dist_by_ind(i,i)` returns 0 and the fill zeroes the diagonal (`Problem.cpp:685, :802`), but Soft-DTW's self-distance is negative (the repo's own test says so, `unit_test_soft_dtw.cpp:192`): two duplicate series give `d(i,j) < d(i,i)`. Decide with D7 between the true diagonal and the Soft-DTW divergence | W2 |
| S-02 | `dtwc++` and its headers are never installed or exported; `cmake --install` ships only `dtwc_cl` | W9 |
| S-03 | Release FP flags are directory-scope `add_compile_options` and reach fetched HiGHS / llfio (`cmake/StandardProjectSettings.cmake:66-70`); GCC and MSVC Release are untested in CI | W0 |
| S-04 | checkpoint-save failure and `--dist-matrix` load failure are warnings, exit 0 (`dtwc_cl.cpp:1845, :1584, :1907`) | W6 |
| S-05 | MATLAB inputs through 32 unchecked `static_cast<int>` though `get_exact_int` exists (`dtwc_mex.cpp:273, :620, :1117`) | W9 |
| S-06 / S-07 | `.dtws` open accepts `ndim = 0` then divides by it (`mmap_data_store.hpp:234, :296`); a read-only input is opened `mode::write` (`:210`) | W2 |
| S-08 | the umbrella reaches `<llfio/…>` and llfio is a PUBLIC, never-installed dependency (bites once S-02 lands) | W9 |
| S-09 | tests hard-require Python 3.9 hinted from a repo `.venv` (`tests/CMakeLists.txt:5-18`) | W0 |
| S-10 | the `-k` error message names the deprecated `--clusters` (`dtwc_cl.cpp:1030`) | W6 |
| S-11 | `tests/test_util.hpp` draws fixtures from the global `randGenerator` | W3 |
| S-12 | `dtw_kernel_eap` can return a stale cell after `break` (`dtw_kernel.hpp:402, :412`) — mechanism real, no reachable trigger found; pin with a test or prove unreachable | W2 |
| S-13 | `Problem::operator=(Problem&&) noexcept = default` with an `unordered_map` member (`Problem.cpp:151`): a throwing move under `noexcept` is `std::terminate` on MSVC's STL — *unadjudicated* | W4 |
| X-01 / X-02 | `ExecutionTarget`, `Problem::set_device`; one execution resolver | W4 |
| X-03 | `dtwc::Config`, `run(Config)` | W6, W9 |
| X-04 | codegen report (probe TU + vectorisation remarks + expectation table) | W0 |
| X-05 | `available_ram_bytes()` declared in a base header | W1 |
| X-06 | `SeriesSource`, `DistanceConfig`, `PairRange` in `core` | W2 |
| X-07 / X-08 | counters-prove-the-path tests; quality-oracle tests | W7 |
| X-09 | `RunStats` on results; optional progress / cancel callback at gateways | W9 |
| X-10 | sharded fill and NaN-union merge | W10 |
| X-11 | `ExactClusteringTransaction` moves from `mip/` to the session layer (it mutates `Problem`; the last `mip → Problem.hpp` edges); `mip-solvers` folded into `dtwc++` | W4 |
| X-12 | the process-global `randGenerator` leaves `settings.hpp` for its own header (so `settings.hpp` stops pulling `<random>`); `settings::paths` globals become `Config` values | W1 |
| S-14 | `distance.msm` / `distance.twe` missing from Python's dispatcher (`distance.py:124-127`) and MATLAB's `+distance` (8 files); reachable only through Tier-2 `variant_params` | W9 |
| S-15 | infeasible band returns a finite `numeric_limits::max()` that passes the `isfinite` assignment guards (`warping.hpp:403`, `medoid_assignment_policy.hpp:43`) — banded clustering of variable-length data silently sums 1.8e308 (R1) | W4 (X-02) |
| S-16 | `distance_strategy` and `cuda_settings.device_id` are hashed into the distance-cache identity (`Problem.cpp:367-369, 397-399`): changing device discards the matrix | W4 (A11) |
| S-17 | Tier-1 `Result` is copyable but `score()` / `distance_matrix()` mutate the `Problem` all copies share (`api.hpp:78-101`, `api.cpp:213-247`) | W6 (X-19) |
| S-18 | Python `_api.py:461-500` re-implements `api.cpp`'s Tier-1 dispatch; `dtwc::cluster` is not bound | W9 |
| S-19 | `dtwcpp.__all__` exports `IOError` (rebinds the builtin under `import *`) and the `test` module | W9, pre-tag |
| S-20 | scores allocate the dense N×N through `dist_by_ind` and `silhouette` calls `fill_distance_matrix()` (`scores.cpp:122`), defeating matrix-free routes | W3 (X-20) |
| X-13 | `Data` templated on element type (if it collapses the f32/f64 twins) | W2 |
| X-14 | one `Precision` enum across core, CUDA, Metal | W2b |
| X-15 | `DTWC_FP_MODEL=strict\|fast`; conformance pinned under `strict`; FP flags `PRIVATE` | W0 |
| X-16 | typed-throw ratchet (`throw std::…` count in `dtwc/` may only fall; 256 today) | W1 |
| X-17 | `distance::dtw_path` (Standard/DDTW, banded, L1/SqL2) from the barycenter backtrack | W2 (C++), W9 (bindings) |
| X-18 | lower-bound cascade + early abandon in nearest-medoid assignment (CLARA, OneBatchPAM, barycenter k-means) with a counter band | W3 |
| X-19 | Tier-1 `Result` owns `ClusteringResult` + `RunStats`; lazy matrix behind `once_flag` | W6 |
| X-20 | scores take the oracle; O(N·k) medoid silhouette; typed error for the full silhouette on a matrix-free result | W3 |
| X-21 | grep gates: no `env()` / `settings::paths` outside `api`, CLI, bindings; `static_assert` nothrow-move on `Problem` | W7 |
| X-22 | `load(..., label_col=)` + `Result::score("ari" \| "nmi")`; ARI per UCR dataset on the benchmarks page | W6, W8 |

Housekeeping: **H-1** decouple CI gates from process records (move the pinned markers into `docs/derivations/`
and drop the legs that read a plan archive, a handoff and baselines). **H-2** compact `LESSONS.md` 119 KB → 30–40 KB
(10 topic sections + index; 50 gate-pinned markers stay verbatim or both gates change in the same commit;
replace `LESSONS.md:<line>` pins by quoted headlines). **H-3** cut the 127 KB "Development history absorbed
into 2.0.0rc1" body of `CHANGELOG.md` (only its heading is pinned). **H-4** move the 11 KB "Phase 4" section that
`generate_docs.py:201` reads into `docs/sources/`, then delete the 223 KB `PLAN-archive-2026-07-20-phases0-9.md`
(its path is quoted inside gate-pinned prose in `UNIMODULAR.md`). **H-5** repoint stale `PLAN.md` / `AGENTS.md`
mentions: `docs/derivations/README.md:6`, `docs/content/guides/migration.md`, eight `dtwc/` comments, three MIP
tests. **H-6** delete the as-is reports, the sweep report and the W0 plan when the campaign closes (~0.9 MB).

## 6. Break register

| Row | What a user would notice | Reason (`design.md` §2) | Mitigation | When |
| --- | --- | --- | --- | --- |
| C-05 | `Auto` fills by brute force instead of pruned | none needed: identical matrix, less work | — | W2 |
| C-09 | unwired or test-only symbols disappear (`core::dtw_distance`, `TimeSeries[View]`, four MV lower bounds, `sync()`, dead overloads) | R4 unreachable | **present in v1.0.0 ⇒ `[[deprecated]]` for one release; 2.0-only ⇒ delete** (D-3) | W2, pre-tag |
| C-11 | include paths of foundation headers move | internal | forwarding headers for one release | W1 |
| B-14 | `dtwc_*` CMake options become `DTWC_*` | consistency; adopted by Volkan 2026-09-07 (`DECISIONS.md` §3), which is what makes it allowed | old names accepted with a warning for one release | W0 |
| O-06 | `set_max_iter(0)` / `set_n_repetitions(0)` throw | R1: reports a "converged" cost after zero iterations | message names the fix | W4 |
| O-09 | 1.x artefact filenames (`_Nc_<k>.csv`, `medoids_rep_<r>.csv`, …) | duplication only — conflicts with the older rule "1.x shims stay" | **recommend a thin legacy writer instead of dropping** (D-2) | W5b |
| O-18 | `test_api.hpp` → `capabilities.hpp` | misnamed production header | forwarding header; `dtwc::test::` names kept | W6, pre-tag |
| G-04 / X-02 | GPU + a non-Standard variant, missing strategy, `ndim > 1` or an LB strategy now throws | R1: those settings are silently ignored today | error names the axis | W4 |
| A-07, S-04 | Benders on invalid input, a failed checkpoint save, a failed matrix import now fail | R1 silent success | typed error / non-zero exit | W3, W6 |
| S-01 | Soft-DTW diagonal changes | R1 | decided with D7 (recommended: the Soft-DTW divergence) | W2 |
| S-15 / X-02 | a band narrower than a pair's length difference now throws instead of silently contributing 1.8e308 | R1 | error names the pair and the minimum feasible band; per-pair widening as an explicit token (D-12) | W4 |
| S-19 and the pre-tag Python slice | `from dtwcpp import *` no longer rebinds `IOError`; 2.0-only aliases go; one estimator | PRE-TAG (never released) | attributes stay; contract §10.5 | W9, pre-tag |

Everything else in the ledger is additive or internal.

## 7. Decisions needed from Volkan

| # | Decision | Recommendation |
| --- | --- | --- |
| D-1 | Approve `design.md` §11 amendments A1–A10 | yes; A1 (layer order) is a correction, the rest follow the charter |
| D-2 | O-09: drop the 1.x artefact filenames (approved 09-07) or keep them through a legacy writer | keep — small cost, and "1.x shims stay" was binding |
| D-3 | C-09: delete vs deprecate | by release history: in v1.0.0 ⇒ deprecate one release; 2.0-only ⇒ delete before the tag |
| D-4 | `.claude/reports/test_kasper_analysis/REPORT.md` is headed "PRIVATE — do not push to GitHub" yet is tracked and present on `origin/Claude` and `origin/design-2.0` (not `main`) | yours alone: removing it from the tree does not remove it from pushed history. **If removed, edit `check_docs_contract.py:2070-2077` in the same commit** — the CI docs gate reads those six files |
| D-5 | S-01 Soft-DTW: true (negative) diagonal, or the Soft-DTW divergence for clustering | the divergence (Blondel, Mensch & Vert 2021): D(i,i) = 0, D(i,j) > 0, costs N self-distances, and it removes D13's shift hack; confirm inside D7 |
| D-6 | Medium-confidence deletions left in place: `benchmarks/{UCR_dtai,UCR_tslearn,random_centroid,bench_parquet_access}.py`, `benchmarks/results/_autorun/` (pinned by the hygiene script), `scripts/test_f22_*_mutations.py` (F22 still open), `scripts/check_record_hygiene.py` (goes with H-1), `bindings/matlab/test_mex.m`, `examples/cpp/example_new_features.cpp` (uncompiled: register or delete), `data/test/AllGestureWiimoteX_dist_50.csv` (orphan kept under a past data rule), `tests/matlab/f19_…oracle.m` and `f20_…oracle.m` (no user), `results/` | delete all but the examples file (register it) |
| D-7 | H-2 / H-3 / H-4 record compaction | yes, after W0 |
| D-8 | CI on `design-2.0`: push filter or draft PR | a draft PR to `main` runs the unit, docs, python and matlab workflows (`pull_request` triggers name only `main`; release and JOSS workflows stay tag / path-triggered). The **merge target stays `Claude`** as the spec says; the PR is for CI only |
| D-9 | S-02: is an installable CMake package part of 2.0 or 2.1 | 2.1 (additive) |
| D-10 | C++ Tier-1 `device="hpc"`: implement submit in C++ or declare it CLI / Python-only | declare; the `Config` + CLI route covers it |
| D-11 | Tag 2.0.0 after W8 (W9 additive parts and W10 become 2.1) or after W9 | after W8, once every `pre-tag` row in §6 **and W9's pre-tag surface slice** are in — the MATLAB and 2.0-only Python names are free to fix only until then |
| D-12 | Infeasible band (S-15): typed error at the gateway, or per-pair widening to \|n − m\| (dtaidistance's behaviour) | error by default; `band_mode = "widen"` as an explicit token — silent widening would change results without a trace |
| D-13 | Python floor: keep `>= 3.9` (EOL) or raise to 3.10 (numpy 2 needs it; abi3 wheels already target ≥ 3.12) | 3.10; a support change, not a break |
| D-14 | One estimator: keep `DTWClustering` (contract §1.5) and fold `DTWCKMedoids`, or the reverse | `DTWClustering`, gaining `random_state`, `n_jobs`, keyword-only `__init__` |
| D-15 | Not planned without clustering evidence: ERP, LCSS, EDR, ShapeDTW, Itakura (the 2023 review finds MSM/TWE + k-medoids best and DTW ≈ ED; ShapeDTW = descriptors + DTW_D by preprocessing; Itakura has no clustering evidence) | record in `DECISIONS.md` as *not planned*, distinct from *killed* |
| D-16 | `Pruned` and the `LowerBoundStrategy` selectors stay public as decided (C-05), or are demoted to diagnostics pre-tag now that lower bounds move to the admissible assignment site (X-18) | keep C-05; document them as diagnostics |

## 8. Records

| What | Where |
| --- | --- |
| Charter, map, design, plan, decisions | `.claude/{CHARTER,MAP,design,PLAN,DECISIONS}.md` |
| Row detail and verdicts | `specs/2026-09-07-diff-ledger.md` (add `status`) · spec · one plan per wave in `plans/` |
| Measurements, verbatim | `baselines/YYYY-MM-DD-<topic>.md` |
| Session state | `summaries/handoff-YYYY-MM-DD-<topic>.md` via the `session-handoff` skill; keep the last few |
| Lessons, citations | `LESSONS.md`, `CITATIONS.md` (both gate-pinned: edit with care) |
| Floors | `tests/floors.cmake` (generated) |
| Anything superseded | delete it; git keeps it. Do not start a new archive |
