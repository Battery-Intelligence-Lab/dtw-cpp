# DTWC++ — Design 2.0 (target architecture)

Status: **proposed 2026-09-21**, written against `MAP.md` (as-is) and the charter (`CHARTER.md`).
It keeps the approved spec `specs/2026-09-07-design-2.0-campaign.md` as the row-level record and
amends it where §11 says so. Signatures below are indicative: a wave plan fixes them against the
code. Anything marked *exists* was verified in the tree on 2026-09-21.

## 1. Premise and goals

The public design works and is in use: a `Problem` you configure with settings, a Tier-1
`device → load → cluster → Result` flow, a CLI with a config file, and Python/MATLAB mirrors. Design
2.0 does **not** replace that shape. It makes the seams *inside* it explicit, so that every language
reaches every feature through one contract, a new device / variant / algorithm plugs in at one place
instead of five, and correctness and speed are checked by the build instead of remembered.

Goals, in priority order:

1. **Correct and provable.** Each algorithm has a derivation and an independent oracle; a refactor
   is proven by digit-identical output, not by inspection.
2. **Fastest for large N.** Kernels stay compile-time specialised and lock-free; algorithms work on
   **indices and a distance oracle, never on the series**, so the data can stay in a view, an mmap
   store, another process, or nowhere at all (a precomputed matrix).
3. **One interface for every language, at no cost.** C++ is the contract; bindings mirror it and
   hand-write as little as possible. Selection happens once, outside hot paths.
4. **Runs where the user points it.** `device = cpu | gpu | hpc` is a setting, not a code path the
   user assembles. No silent fallback, ever.
5. **Easy to parallelise and to fit to a GPU.** One unit of work for every executor.
6. **A joy to develop.** Layers checked at configure time, small units, a handful of helpers,
   tests that each pin a named contract.
7. **Optional dependencies stay optional**; the core builds and tests with all of them OFF.

## 2. Compatibility policy — a break needs a solid reason

Facts: the last release is **v1.0.0**; `VERSION` is `2.0.0rc1` and untagged. Users are on the 1.x
shape. `docs/api-contract-2.0.md` (frozen 2026-07-07) fixes the 2.0 Tier-1 and Tier-2 surface and
maps every 1.x name to its 2.0 name through 33 `[[deprecated]]` shims.

| Class | What | Rule |
| --- | --- | --- |
| FROZEN | Contract Tier-1/Tier-2 C++ signatures and names, including `Problem::distance_matrix()` returning the `distMat_t` variant by reference and the public fields; CLI flags; artefact, checkpoint and `.dtws` wire formats; the 1.x shims (C++ names and the v1.0.0 pybind names such as `refreshDistanceMatrix`, `cluster_size`) | unchanged; a change is a contract addendum with a dated entry in `DECISIONS.md` |
| PRE-TAG | The MATLAB surface (no MATLAB file exists in v1.0.0) and every Python name that first appeared during 2.0 development — v1.0.0 shipped only a thin pybind wrapper of `Problem` with 1.x names, never on PyPI. Their freeze is the contract's own, not a user's | fix before the tag at no user cost; after the tag they become FROZEN. Check each name against `git show v1.0.0:python/py_main.cpp` |
| ADDITIVE | new setters, overloads, enum values, result fields, config keys | the default tool; needs a test and a CHANGELOG line |
| INTERNAL | everything behind the façade (members, private headers, file layout with forwarding headers left at the old path for one release) | free, if conformance output is digit-identical |
| BREAK | removing or changing observable behaviour | only with one of the reasons below, listed in the break register of `PLAN.md` with its migration |

Accepted reasons for a BREAK: **R1** it returns a silently wrong answer; **R2** it is unsound (UB,
data race, false `noexcept`); **R3** it blocks the cross-language contract and no additive route
exists; **R4** it has no in-tree production caller *and* was not part of v1.0.0's public surface
(the grep proof goes in the commit; a symbol that shipped in v1.0.0 is deprecated for one release
instead, because a grep cannot see users' code). "Cleaner" is not a reason. Because 2.0 is untagged, 2.0-only surface that has to change is cheapest
to change **before the tag**; PLAN marks those rows `pre-tag`.

## 3. The pipeline and its seams

```text
 source ──io readers──▶ Data ──┐                         ┌──▶ scores(oracle, result)
 (csv dir/file, parquet,       │ SeriesSource            │
  arrow ipc, .dtws, arrays,    ▼                         │
  views, or NONE)        DistanceConfig ─bind once─▶ DistanceFn
                               │                         │
                               ▼                         │
        executor(target) fills PairRanges ──▶ DistanceMatrix (dense | mmap)
        cpu brute/pruned · cuda · metal · mpi            │
                               ▼                         │
                        DistanceOracle  d(i,j), row(i), size()      ← indices only
                               │                         │
                               ▼                         │
        algorithms / mip :  (oracle, k, options, seed) ─▶ ClusteringResult ─▶ io writers
```

`Problem` is the session that owns and wires these; `api`, the CLI and the bindings are surfaces
over `Problem`. Six seams carry everything:

| Seam | Kind | Carries | Today |
| --- | --- | --- | --- |
| `SeriesSource` | `Data` itself, templated on the element type so the f32/f64 twins (`dtw_fn_`/`dtw_fn_f32_`, `series`/`series_f32`, the `is_f32()` forks) collapse; a separate concept is introduced **only if** it deletes those twins — one model deletes nothing | series as views, never copies | `Data` (four modes: heap f64, heap f32, view — also over an mmap store — and metadata-only) *exists*; `core/` names `Problem`/`Data` |
| `DistanceConfig` | value with a fingerprint: variant params, band, missing strategy, ndim, metric, precision — **semantics only, never the execution target** | *what distance means* | five public `Problem` fields read live through `*this` by the bound closure (C-01, C-22); `DistanceCacheConfiguration` is already this value but private; `distance_strategy` and the CUDA device index are hashed into the cache identity, so changing device discards an O(N²) matrix |
| `DistanceFn` | `std::function<double(span, span)>`, bound **once**; an overload with an early-abandon threshold for nearest-medoid searches | the chosen kernel, type-erased outside the hot loop | *exists* (`resolve_dtw_fn`); keep; the kernels already take `early_abandon` |
| `PairRange` | `{first, count}` over the linear pair index `[0, N(N−1)/2)`, decoded by the one `decode_pair`; a filler writes the **packed slice** for its range | the unit of work | pruned fill, CUDA, Metal and MPI already use it; the CPU brute-force fill goes by row (C-07); GPU results come back N×N and are copied element-wise into the packed store |
| `DistanceOracle` | one concrete **`PackedOracle{span<const double>, n}`** — dense and mmap both read `data_[tri_index(i,j)]`, so a 16-byte value with `noexcept d(i,j)` serves every filled-matrix algorithm with zero dispatch and no template in a header. The *concept* (`size()`, `d(i,j)`, `row(i, out)` gather — a packed triangle has no contiguous row) remains for the two lazy consumers (TADPole, CLARANS: compute on miss through the session) and the tests' reference oracle, with explicit instantiations, never `std::function` or a virtual per lookup | distances by index | does not exist: algorithms and MIP take `Problem&` and touch 33 of its members; every lookup pays three preflights and a `std::visit` |
| `ClusteringResult` | value: labels, medoids, cost (+ bound/gap, stats) | the single output of every algorithm | *exists*, but six algorithms also write `Problem` members as a side channel (A-02) |

Two rules follow. **Series never cross the oracle seam** — every matrix-based algorithm must run on
a `Problem` that holds a distance matrix and *no series* (this is a test, §8). And **one unit of
work**: a thread team, a GPU launch, an MPI rank, a SLURM array task and a checkpoint all talk about
the same thing, a `PairRange`. Because the dense matrix already encodes "not computed" as NaN,
merging partial fills is an element-wise union that errors on disagreement; sharding and resume
need no new format.

## 4. Layers, checkable

| Rank | Layer | Holds | May include |
| --- | --- | --- | --- |
| 0 | `base` | errors, settings/constants, types, enums, parallel primitives (`for_each_index`, `for_each_pair`, `reduce`), scratch memory, timing, portable RNG, hashing, checked/saturating arithmetic, `Env` | std |
| 1 | `core` | kernels + Cost/Cell policies, variant wrappers, lower bounds, `DistanceConfig` + the one dispatch table, **`Data`**, series stores, dense/mmap matrices, the CPU fills, the concepts, `ClusteringResult`, assignment policy | base |
| 2 | `io` | readers → `Data`; matrix IO; **one** artefact-writer module ← values | base, core |
| 2 | `backends` | CUDA, Metal, MPI fillers | base, core |
| 3 | `algorithms` | PAM family, CLARA, CLARANS, TADPole, hierarchical, barycenters, initialisation, scores | base, core, io (streaming sources only) |
| 4 | `mip` | one Balinski model, solver adapters, Benders, LR-core, PDLP arbiter | + algorithms (warm start) |
| 5 | `session` | `Problem` façade = `Data` + `DistanceCache{config, fn, matrix, identity}` + `ClusterState` + execution target; checkpoint | all below |
| 6 | `surface` | Tier-1 `api`, `Config`, CLI, `capabilities`, umbrella header | all below |
| — | bindings | Python, MATLAB (later: a C ABI, WASM) | surface + the frozen Tier-2 names |

Measured today against this table (`scripts/repo_map.py layers`): **18 upward edges, 17 of them
`→ Problem.hpp`** from `core/` (2), `algorithms/` (9) and `mip/` (6); the last is
`system_memory.cpp → DataLoader.hpp`, a declaration living in the wrong header. There are no include
cycles. So the layering problem *is* the missing oracle seam; nothing else needs restructuring.
`Problem.hpp` has fan-in 22, third after `error.hpp` and `settings.hpp`, and the bindings' probe
header `test_api.hpp` pulls in 69 project files.

Two corrections to the spec's layer table: **`io` sits below `session`**, because the frozen
constructor `Problem(name, DataLoader&)` and `Problem::write_*` both need it there, and every
`io → session` include today is `→ Data.hpp`; and **`Data` is a `core` value type**. Files stay where
they are unless a move deletes a problem: only the stdlib-only foundation headers move (to
`dtwc/base/`, C-11), with forwarding headers at the old paths for one release.

The layer table is a manifest checked at configure time (W0 Task 9), report-only until a layer is
clean, then strict for that layer. The upward-edge count is a ratchet: it may only go down.

## 5. Execution targets: `device = cpu | gpu | hpc`

Today "where does it run" is said three ways: `Device{CPU,GPU,HPC}` in the process-wide `Env`
(Tier-1 only), `DistanceMatrixStrategy{Auto,BruteForce,Pruned,CUDA,Metal}` on `Problem` (Tier-2),
and `detail::Tier1ExecutionTarget`. `Problem` never consults `Env`; `api.cpp::configure_device`
translates one vocabulary into the other. Precision is likewise said three times (`core::Precision`,
`CUDAPrecision`, `MetalPrecision`).

Target — additive, nothing removed:

- `ExecutionTarget{Device device; int index; int threads}` is a value in `base`.
  `Problem::set_device("gpu:1")` / `set_device(Device)` lifts the Tier-1 translation into the
  session, so `Problem(device=…)`-style settings work identically in C++, Python and MATLAB. A
  `Problem` that is never told a device behaves exactly as today; Tier-1 keeps passing the process
  default explicitly.
- `DistanceMatrixStrategy` stays. `Auto | BruteForce | Pruned` are CPU schedules; `CUDA` and `Metal`
  remain as spellings of `device = gpu`.
- **One resolver** `resolve_execution(config, target, data, capabilities)` returns a fill plan or
  throws a typed error naming the axis. It is the only place that knows, for example, that the GPU
  kernels implement Standard DTW on univariate data without a missing-data strategy (today those
  four settings are silently ignored on the GPU route, G-04), that GPU upload rejects an mmap
  series store, and that **a band narrower than a pair's length difference has no feasible path**
  — today `dtwBanded` returns a *finite* `numeric_limits::max()` for that case, the assignment
  guards test only `isfinite`, and banded clustering of variable-length data silently sums 1.8e308
  (`warping.hpp:403`, `medoid_assignment_policy.hpp:43`).
- **Fillers are a Strategy owned by `core`**: `fill(const Data&, PairRange, const DistanceConfig&,
  packed slice) → FillStats`, with one factory `switch` in an always-compiled TU under `backends/`
  (static-initialisation registries are dropped by a static-library linker). `capabilities` queries
  the same factory. One virtual call per *fill*, none per pair; `Problem.cpp` and `api.cpp` carry no
  backend `#ifdef`.
- `hpc` means two different things, both built from the same parts:
  1. *Submit*: ship a run description and a data path to a scheduler, run `dtwc_cl` there, fetch
     the artefacts. *Exists* in Python (`python/dtwcpp/_hpc.py`, `scripts/slurm/`); C++ Tier-1
     throws "beta". The design rule that makes this cheap: **every setting of a run is a `Config`
     field** (§6; in-memory data and callables such as `init_fun` are not, and a run that uses them
     is not submittable), so submission is "serialise the Config, run the CLI".
     Credentials stay in `.env` with the three frozen error messages (contract §6.2).
  2. *Scale out inside a job*: the MPI filler becomes a strategy (it is unreachable from `Problem`
     today), and a sharded fill — `--shard i/n` computes one `PairRange`, `--merge` unions the
     shards — turns a SLURM array job into the simplest executor of all.

## 6. One interface for every language

Principle: **C++ is the contract, bindings are mirrors, a run is data.**

- **Tiers stay.** Tier-1 functions (`device`, `load`, `cluster`, `Result`) and Tier-2 objects
  (`Problem`, algorithms, `distance.*`, `scores.*`, checkpoint) as frozen in the contract.
- **`Config` is the missing piece.** Today only the CLI can read a run description (TOML/YAML
  through CLI11). The CLI refactor already needs a parsed-options value (`CliConfig`, B-01);
  promote that value into the library as `dtwc::Config` with `run(Config) → Result`. CLI flags, a
  config file, Python keyword arguments and MATLAB name-value pairs become four spellings of one
  validated struct, and `hpc` submission gets its wire format for free. No second schema. Rules that
  keep it from becoming the next god object: it is a C++20 **aggregate composed of the existing
  per-concern structs** (`DistanceConfig`, fill/strategy options, a new `ClusterOptions{method, k,
  seed, max_iter, …}`, IO options, `ExecutionTarget`), each with one `validate()` (A-04); defaults
  live once, as member initialisers, and the CLI reads them from a default-constructed `Config`;
  keys are the snake_case C++ identifiers with the CLI's kebab spelling (`n-clusters`) as an alias
  table; a `schema = 1` key; applied through the existing setters (`apply(Config, Problem&)`).
  Suffix convention: `Options` for user knobs, `Params` for mathematics, no new `Settings`; one seed
  type (`uint64_t`) and one spelling of `max_iter`.
- **Python is the proof.** `_api.py` re-implements Tier-1 today (method resolution, dispatch,
  `Result`) in ~600 lines while MATLAB makes one call into `dtwc::cluster`; parity drifts by
  construction. With `Config`, Python's `cluster(data, k, **options)` builds the struct and calls
  `run`, and its `_hpc` path submits the same struct. `Result` exposes its `Config` for round-trips;
  no dataclass mirror.
- **One string↔enum table per enum, in C++**, used by the CLI, `Config`, Tier-1 and both bindings.
  Today `Method` names 4 of 10 algorithms, Tier-1 keeps its own nine strings, and the CLI re-parses
  twelve options by hand (A-01, O-11, B-02).
- **Zero-copy in, values out.** Series enter as views over caller memory (C++ `set_view_data`
  *exists*; the Python `set_data` / `Data` / `compute_distance_matrix` paths copy through Python
  floats today, and the Arrow docstring says "zero-copy" while the code copies); results leave as
  moved vectors; the distance matrix is exposed as a view of the packed store; an in-memory
  `cluster()` stops copying the dataset (O-12). Tier-1 `Result` holds `ClusteringResult` + `RunStats`
  **by value** — today it is copyable yet `score()` and `distance_matrix()` mutate the `Problem`
  shared by every copy — and keeps the session only for the lazy matrix path.
- **Strings wherever an enum is.** Setters accept `str | Enum` in Python and MATLAB from the one C++
  table per enum; the tables also generate the `.pyi` stubs (`py.typed` ships with none today) and
  the cross-language parity test.
- **What each binding hand-writes shrinks.** `_dtwcpp_core.cpp` and `dtwc_mex.cpp` are ~1 800 lines
  each today. Parsing, validation and option composition live in C++ once; a binding converts
  arrays and forwards.
- **Runs are observable.** Gateways return `RunStats` (pairs and cells computed, lower-bound
  prunes, swaps evaluated, iterations, the device actually used) following the existing
  `PruningStats` / `TADPoleStats` pattern, and may take a progress/cancel callback that is polled
  per pair block or per iteration — never inside a kernel.
- **C++ consumers get a package.** `dtwc++` and its headers are not installed or exported today
  (only `dtwc_cl` is); `find_package(dtwc)` is part of the interface.
- **More languages later.** A C ABI (opaque handle + `Config` text + array pointers) is a thin layer
  over `Config` and `Problem` and serves Julia/R/Rust alike. R/Julia bindings are a recorded killed
  idea, so nothing is scheduled; the seam simply must not preclude it.

## 7. Performance design, and how it is verified

Kept as is: the DO-NOT-BREAK mechanisms in §9. Changed: the CPU brute-force fill adopts balanced
pair blocks (C-07); `dist_by_ind` becomes an O(1) snapshot compare → packed lookup → compute on miss,
with the full validation moved to gateways (O-01, O-02); `filled` becomes a counter (O-13).

Every hot function is held by three instruments — the FFmpeg `checkasm` mindset:

1. **Reference vs optimised.** The optimised path is compared with a plain full-matrix reference
   that lives in `tests/support`, on non-degenerate inputs.
2. **Counters before clocks.** Cells computed, prune rate, swaps evaluated, allocations in steady
   state: machine-independent, with registered bands. Wall-clock is advisory.
3. **A codegen report.** A probe translation unit wraps each hot function in a `noinline` entry
   point, is compiled with the project's real Release flags, and the compiler's own vectorisation
   remarks (`-Rpass=loop-vectorize` / `-fopt-info-vec` / `/Qvec-report:2`) plus the disassembly are
   compared with an expectation table. First measurement (Apple clang 21, arm64, 2026-09-21):
   `lb_keogh` and the three `z_normalize` passes **vectorise**; `compute_envelopes` does not
   (early-exit deque loop); the DTW row recurrence **cannot** — it reads the cell it wrote one
   iteration earlier. The table therefore records three states: *must vectorise*, *known not to*,
   *structurally cannot*. A function leaving *must vectorise* fails the pinned-compiler leg. This is
   the same tool W0 Task 12 plans to create (`check_ipo_inlining.py`, a disassembly report — not yet
   written); build one script for both.

**The floating-point model must be named before any "digit-identical" claim.** Release builds
apply `-fassociative-math`, `-freciprocal-math` and `-march=native` at directory scope (MSVC:
`/fp:contract`); wheels drop `-march=native`. Every reduction — z-normalisation, `LB_Keogh` (which
vectorises *because* reassociation is allowed), scores, costs — is therefore vector-width and
FMA-dependent; only the pair recurrence (a `min` and one add) is safe. So: a `DTWC_FP_MODEL=strict|fast`
option (`strict` = `-ffp-contract=off`, no associative math), the conformance reference pinned under
`strict`, `fast` as the shipped default with D17 supplying the tolerance bands, and the flags
`PRIVATE` to `dtwc++` so fetched HiGHS/llfio are untouched. Cost and Cell policies get `concept`s;
their contract is prose today.

Hand-written SIMD stays killed (`DECISIONS.md`). What this adds is the measurement the kill-note
asks for. A SIMD win on the recurrence has to come from another axis — several equal-length pairs in
lockstep, or anti-diagonals — and enters only through R2-D17 (error model + roofline) as a registered
measure-first prototype. Note that "a batch of pairs" is exactly what the GPU executors consume, so
the `PairRange` seam serves both.

## 8. Verification — tests that earn their place

A test exists to pin a **named contract** against an **independent oracle**. Count is not a goal.

- **It ran.** Every test has a floor; a skip is opt-in and printed (W0, done). Case floors are
  portable; assertion floors are the minimum over every supported platform.
- **It took the path.** Tests assert `RunStats` counters, so "the GPU route really launched
  kernels", "the pruned fill really pruned" and "no silent fallback" are checked, not assumed.
- **It got the answer.** Oracles: the full-matrix reference DTW; brute-force p-median on small N;
  the hash-pinned cross-language conformance reference; and a **quality oracle** — on fixtures small
  enough for the exact solver, every heuristic must land within a registered gap of the proven
  optimum, and `Result` reports the bound and gap whenever one exists.
- **It needs no data.** Each matrix-based algorithm runs on a matrix-only `Problem` (§3).
- **Scores respect the route.** `Result::score` → `scores::*` → `dist_by_ind` lazily allocates the
  dense N×N today and `silhouette` calls `fill_distance_matrix()`, so one score after CLARA,
  OneBatchPAM or TADPole at large N defeats the matrix-free route. Scores take the oracle; the
  O(N·k) medoid silhouette (Van der Laan, Pollard & Bryan 2003; Lenssen & Schubert 2024) and
  inertia work from the assignment; the full silhouette on a matrix-free result is a typed error
  naming `distance_matrix()`.
- **Ratchets, not memory.** Three counts may only go down, checked by script: upward includes
  (18), untyped `throw std::…` in `dtwc/` (256 against 227 typed — `Data::validate_ndim` and
  `dist_by_ind` throw untyped for user input, so Python sees a bare `RuntimeError`), and `env()` /
  `settings::paths` reads outside `api`, the CLI and the bindings.
- **It survives the platform.** Floors, fingerprints and tolerances are derived (R2-D17 gives the
  f32/f64 band), not copied from one machine.

Whatever compares a wrapper with the function it calls, asserts only structure, or duplicates a
sibling is merged or deleted with the covering test named in the commit (ledger T-rows).

## 9. Invariants — deliberate, do not "clean up"

The spec (I.2) lists 66 with their evidence; W8 promotes that list into this section. The ones most
likely to be broken by a well-meaning refactor:

- Cost and Cell stay **template parameters**; never a `std::function` or a virtual per cell.
- Orient so `n_short ≤ n_long` before every kernel call; buffers are sized on that.
- `thread_local` scratch grows and never shrinks; a Cost must not re-enter its kernel.
- The dense matrix has **no locks or atomics**; fills partition pairs. `resize()` NaN-wipes, so it
  stays conditional (checkpoint resume depends on it).
- Bind the distance function **once**; every guard lives in the builder, outside the closure
  (throwing inside an OpenMP region is UB). Series travel as spans.
- The unnamed `omp critical` in the header and its CMake scanner; region-local `num_threads`, never
  `omp_set_num_threads`. Named criticals only in `.cpp`.
- `decode_pair` is the single pair decoder, on host and device, with its `while` corrections.
- FP model: `-fassociative-math` **without** `-ffinite-math-only`; NaN means missing/uncomputed; no
  `infinity()` sentinels. The EAP slack `ub·(1+n·16ε)` is regression-tested, not yet derived (D4).
- `find_best_swap` and the FasterPAM sweep are sequential on purpose (parallel measured ~10× slower).
- The seven nearest-medoid scans stay separate at *run time*: a runtime policy branch would enter
  the hottest loop. A compile-time collapse (a template helper with byte-identical object code,
  A-18) is allowed; the spec's "one `assign_to_nearest`" means that and nothing more.
- The kernels' `numeric_limits::max()` is the DP's *unreachable* value. It must never leave a
  kernel as a distance: the wrappers return it today for empty input, an infeasible band and early
  abandonment, and a finite poison passes every `isfinite` guard (worse than `inf`). NaN is the only
  "not a distance" sentinel outside kernels.
- Parquet planning is metadata-first; RAM arithmetic saturates; `.dtws` v1 and checkpoint v1 layouts
  are frozen; argv beats the config file and an unknown key is an error.

## 10. Carried over from the previous design (still true)

- **Variants.** One kernel family per topology (full, linear, EAP, banded); variants are Cost/Cell
  policies, not copied loops. WDTW and ADTW change cost/recurrence semantics, not the metric. DDTW
  is a preprocessing transform. Soft-DTW is a distinct recurrence through its own Cell. MSM and TWE
  have their own rolling DPs. Missing-data handling is part of dispatch.
- **Python.** nanobind + scikit-build-core, zero-copy `nb::ndarray`, sklearn-compatible estimators,
  GIL released around long C++ calls, one thread per `Problem` instance, `uv` only.
- **MATLAB.** C MEX API with handle classes and `mexLock`; `+dtwc` package; snake_case functions;
  indices shifted at the MEX boundary.
- **Cross-language.** Pairwise distances live under `distance.*` in all three languages, with a
  generic `dtw(..., variant=)` beside direct `soft_dtw`, `wdtw`, … — and `msm`, `twe`, which the
  C++ has but neither Python's dispatcher nor MATLAB's `+distance` exposes.
- **Conventions are the v1.0.0 / JOSS surface and stay:** local cost **L1**, band in **integer
  cells**, **no final square root**. Peers differ (dtaidistance: squared Euclidean with `sqrt` at
  the end; tslearn: √Σ‖·‖²; aeon: Σ(·)² and `window` as a fraction of length), which makes every
  cross-library number a trap. 2.0 adds tokens, never flips defaults: `euclidean` = √(squared-L2
  total) applied *outside* the kernel (the PAM objective and the silhouette are not
  square-root-invariant, so this is not cosmetic), `window_fraction`, and one same-definition parity
  test per peer library on the benchmarks page. `MetricType::L2` is wired only as the multivariate
  per-cell Euclidean (Giorgino 2009); for `ndim = 1` it is L1.
- **Alignment paths are public.** `distance::dtw_path` (Standard/DDTW, banded, L1/SqL2) — the
  barycenter code already backtracks one (`barycenter.cpp:184-224`), every peer exposes one, and it
  is the reference oracle of §7. 2.0, not a 2.1 WASM task.
- **MIP.** Balinski p-median; HiGHS or Gurobi; FastPAM warm start; Benders above N = 200; once
  medoids are fixed the assignment is totally unimodular; LR-core with matrix-free Kelley cuts is
  the large-N exact route and PDLP is a cross-check only.
- **Regimes.** One pair's recurrence is latency-bound (~10 cycles per cell, rolling buffers in L1);
  a large-N fill is memory-bound. Neither has a PMU artefact yet.

## 11. Amendments to the 2026-09-07 spec

| # | Amendment | Why |
| --- | --- | --- |
| A1 | `io` ranks below `session`; `Data` is `core` | the frozen `Problem(name, DataLoader&)` and `Problem::write_*` make the spec's order unattainable; all four `io → session` includes are `→ Data.hpp` |
| A2 | `CliConfig` (B-01) is promoted to a library `Config`; O-21 is no longer "next campaign" | one run description for four spellings and for `hpc` |
| A3 | Execution target is a `Problem` setting with one resolver; G-04 becomes part of it | charter: `problem(device=…)`; three vocabularies today |
| A4 | `PairRange` is the unit of work for every executor (generalises C-07) | one seam for threads, GPU, MPI, SLURM shards, checkpoints |
| A5 | A codegen report joins counters and oracles; SIMD stays killed | charter: inspect the ASM, not only the clock |
| A6 | Compatibility policy (§2) and a break register; C-09, O-09 and B-14 are re-examined under it | charter: a break needs a solid reason; the older binding rule "1.x shims stay" |
| A7 | Two new waves: W9 interface, W10 scale-out. The open science tracks of the previous plan (R2 derivations, R3 findings, R5 fences, R6 release gate, R7 WASM) are carried into `PLAN.md` | the spec deferred the interface; the old plan is retired |
| A8 | The development machine is a Mac: Metal is local, CUDA is `[BLOCKED-ENV]` here | new computer |
| A9 | Verified findings from the 2026-09-21 sweep and the record audit become rows `S-01…S-13` | they were in no ledger |
| A10 | Floors must be measured on every supported platform | 5 of 131 tests fail on macOS on floors alone |
| A11 | `DistanceConfig` carries semantics only; execution never enters the cache identity | four-lens review 2026-09-22: `distance_strategy` and the device index are hashed today |
| A12 | The oracle is one concrete `PackedOracle` for filled-matrix algorithms; the concept only where a lazy view or a fake is needed; `row(i)` is a gather | dense and mmap read the same packed triangle; no contiguous row exists |
| A13 | Fillers are a core-owned Strategy with one factory in `backends/`; the resolver rejects infeasible bands (R1) | `Problem.cpp`/`api.cpp` backend `#ifdef`s; finite `max()` sentinel reaches the objective |
| A14 | `Config` composes the existing option structs; Python Tier-1 becomes `run(Config)`; `Result` owns values | Python re-implements `api.cpp`; `Result` copies share a mutable `Problem` |
| A15 | `DTWC_FP_MODEL=strict|fast`; conformance pinned under `strict`; flags `PRIVATE` | reassociation at directory scope makes every reduction width-dependent |
| A16 | Scores take the oracle; O(N·k) medoid silhouette; typed error for the full silhouette on a matrix-free result | scores allocate N² today |
| A17 | `dtw_path` and the `euclidean` / `window_fraction` interop tokens are 2.0; MSM/TWE reach the pairwise surface in all three languages; nothing on the not-planned list (ERP, LCSS, EDR, ShapeDTW, Itakura) is added without clustering evidence | time-series review 2026-09-22 |
| A18 | PRE-TAG class in §2: the MATLAB surface and 2.0-only Python names are fixable at no user cost until the tag | `git ls-tree v1.0.0` |
