# R1 documentation truth audit — 2026-07-23

## Scope and base

- Branch: `Claude`
- Base commit: `cd7d37b` (`docs: close R1 TODO reconciliation`)
- Scope: README, Hugo source pages, `docs/api-contract-2.0.md`, CHANGELOG,
  and documentation-bearing source/build comments.
- Constraint: non-behavioral documentation/record changes only.

## Preregistered acceptance band

1. Every corrected load-bearing claim names current code, an executable result,
   a committed baseline, or a verified citation. Unsupported prediction is
   removed or explicitly tagged inferred.
2. Known stale claims from the TODO reconciliation are resolved:
   Float64 default; live metric-dispatch home; removed SIMD surface; accurate
   UCR/kernel performance ranges; current FastCLARA and F7 behavior.
3. Rebuild `build/highs-1151`, then run the real CLI documentation gate:

   ```text
   python scripts/check_docs_contract.py --cli build/highs-1151/bin/dtwc_cl.exe
   ```

   Decisive band: exit 0 and exact terminal line
   `documentation contract checks passed`. Generated-source drift, frozen
   contract governance, migration behavior, rc1 CHANGELOG structure, HPC error
   messages, Tier-1 signatures/method registries, live CLI flags, and Float64
   CLI documentation are all in this gate.
4. Run `scripts/check_site_links.py` only against a site freshly generated from
   the edited sources. Decisive band: exit 0 and exact line
   `all internal site links resolve`. If Hugo cannot run locally, record
   `[BLOCKED-ENV]` with the tool probe verbatim. A pass over the pre-existing
   ignored `docs/public/` tree is advisory and cannot close the fresh-site gate.
5. `git diff --check` emits no errors. The final diff contains no program
   behavior changes.

## Environment probe

```text
hugo=NOT_FOUND
go=NOT_FOUND
node=C:\Program Files\nodejs\node.exe
v24.16.0
npx=C:\Program Files\nodejs\npx.ps1
11.13.0
cli_exists=True
tracked_public=0
!! docs/public/
```

**[BLOCKED-ENV]** A fresh Hugo site and therefore the decisive internal-link
gate cannot be produced on this host: both `hugo` and its `go` build fallback
are absent. Continue with source-level audit, the live CLI contract gate, and
an explicitly advisory check of the existing ignored site. Hosted CI remains
operator-owned and is not claimed.

Advisory only, against the pre-existing ignored `docs/public/` tree:

```text
all internal site links resolve
```

This does not supersede the `[BLOCKED-ENV]` fresh-render result.

## D1 — generated LR-core source moved without its gate

Baseline decisive command:

```text
.venv/Scripts/python.exe scripts/check_docs_contract.py --cli build/highs-1151/bin/dtwc_cl.exe
```

Baseline failure:

```text
RuntimeError: missing Phase 4 implementation outcome in PLAN.md
subprocess.CalledProcessError: Command '['C:\\D\\git\\dtw-cpp\\.venv\\Scripts\\python.exe', 'C:\\D\\git\\dtw-cpp\\scripts\\generate_docs.py', '--check']' returned non-zero exit status 1.
```

[confirmed] PLAN v2 names
`.claude/PLAN-archive-2026-07-20-phases0-9.md` as the verbatim home of the old
Phase 4 implementation outcome; the generator still searches only live
`PLAN.md`.

Registered repair band:

- move only that generator input to the named archive;
- `generate_docs.py --check` exits zero and prints
  `generated documentation is current`;
- no generated page changes (the archive is verbatim, so this is a
  digit-identical source-location repair);
- the full docs-contract gate then reaches the next independent check or passes.

Repair results:

```text
generated documentation is current
```

Full gate after the repair:

```text
generated documentation is current
documentation contract checks passed
```

`git status --short` named only this run-log and `scripts/generate_docs.py`; no
generated page changed.

**D1 verdict: PASS.** The generated output is digit-identical and the full live
CLI contract gate now reaches completion.

## D2 — precision contract still narrates the pre-flip state

Current source evidence:

```text
dtwc/settings.hpp:30:using default_data_t = double;
dtwc/dtwc_cl.cpp:662:  std::string dtype_str = "float64";
dtwc/dtwc_cl.cpp:664:      "Series data type: float64 (default, full precision) or float32 (2x memory saving) (aliases: f32, f64, float, double)")
```

[confirmed] All public distance-helper default templates use
`dtwc::settings::default_data_t` (`dtwc/distance.hpp:35-233`). All Python
single-pair distance bindings now accept contiguous NumPy views
(`python/src/_dtwcpp_core.cpp:529-601`).

A second stale claim is independently false: the contract says Float32 series
are always accumulated in double. The unified kernels store recurrence buffers
and combine cells in template scalar `T`
(`dtwc/core/dtw_kernel.hpp:238-283`); the f32 dispatcher casts the final
float result to the public double return
(`dtwc/core/dtw_dispatch.cpp`, `make_*<float>`). Distance-matrix entries are
double, but Float32 recurrence arithmetic is Float32.

Registered repair band:

- present tense describes the implemented Float64 default and ndarray bindings;
- migration history remains explicit without saying the old state exists now;
- precision language distinguishes recurrence precision from double result
  storage and does not promise Float32 numerical identity;
- regenerating the derived Tier-2 page changes only the corresponding source
  projection;
- `generate_docs.py --check` and the full live-CLI docs contract gate pass.

Registered stale-output probe after editing the source contract:

```text
stale or missing generated documentation:
  docs\content\api\tier-2.md
run: python scripts/generate_docs.py
```

Final frozen-contract inventory:

```text
stale_marker_hits=0
reviewer_resolutions=8
implementation_findings=9
finding_ids=18,19,20,21,22,23,24,25,26
```

Final decisive gate:

```text
generated documentation is current
documentation contract checks passed
```

**D6 verdict: PASS.** The inherited guard failed on all eight registered stale
marker classes. The corrected contract preserves and names all nine
implementation gaps F18–F26, adjudicates exactly eight reviewer decisions, and
has zero registered pre-implementation markers. Derived Tier-1, Tier-2, and
migration pages are current. Fresh rendered-link evidence remains
`[BLOCKED-ENV]`; no stale-site result is promoted as evidence.

## D7 — GPU backend page overstates correctness, routing, and performance

[confirmed] Source audit registered five runtime findings before page repair:
F27 squared-L2 LB uses L1 excess; F28 Metal full-DTW envelopes can be too
narrow; F29 unequal-length GPU LB truncation is unsafe; F30 explicit GPU options
silently degrade; F31 operational Metal failures escape the public
`DeviceError` taxonomy.

Registered documentation repair band:

- add an unconditional drift guard before editing and show it fails on the
  inherited stale phrases;
- scope current GPU LB pruning to equal-length L1 series with an LB envelope
  that covers the actual DTW window; name F27–F29 rather than asserting
  universal `LB <= DTW`;
- say the direct pruning API returns a thresholded matrix with `+inf` for
  pruned pairs, not an exact distance matrix;
- describe the actual shared `dtwc::KernelOverride`, backend-specific option
  fields/defaults, hint semantics, runtime Metal memory threshold, and
  F30/F31 loudness gaps;
- state that `Problem::lb_strategy` and `DistanceMatrixStrategy::Auto` are
  CPU-only today; neither silently selects GPU pruning;
- distinguish envelope preprocessing `O(N·L·r)`, pairwise LB
  `O(N²·L)`, and survivor DTW work;
- remove numeric tables/cutoffs with no tracked originating result artifact.
  The traceable Apple M2 Max table may remain only as historical/advisory and
  must name its raw JSON/benchmark record;
- describe the CUDA work as inspired by/adapted from cuDTW++, not a verified
  direct port;
- the real-CLI documentation contract gate exits zero with exact final line
  `documentation contract checks passed`; `git diff --check` is clean.

Fresh Hugo rendering remains under the D0 `[BLOCKED-ENV]` result.

Deliberate red after adding the reachable GPU-page guard, before page repair:

```text
generated documentation is current
Traceback (most recent call last):
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 348, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 340, in main
    assert_gpu_backend_page()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 254, in assert_gpu_backend_page
    raise AssertionError(f"GPU backend page retains stale claims: {present}")
AssertionError: GPU backend page retains stale claims: ['MetalKernelOverride', '`max_length_hint > 0` skips the runtime length scan', '`Problem::lower_bound_strategy` (coming in a later commit)', 'Selecting `DistanceMatrixStrategy::Auto` on a build with both backends enabled picks CUDA', 'CPU with DistanceMatrixStrategy::Pruned', 'always satisfies $$\\mathrm{LB}_{\\mathrm{Keogh}} \\le \\mathrm{DTW}$$', 'The CUDA reference implementation in DTWC++ is a direct port', '**103×**', '**7.6×**', '`N < 20`, `L < 100`', 'Unified memory removes H2D/D2H', 'per-pair envelope + LB cost is O(L)']
```

Final decisive gate:

```text
generated documentation is current
documentation contract checks passed
```

**D7 verdict: PASS.** All 12 registered stale phrase/classes are rejected by
the permanent guard. The page names F27–F31, limits current GPU LB claims to
equal-length L1 plus an admissible matching window, identifies thresholded
`+inf` output, and states CPU-only Auto/lower-bound routing. Untraceable
regtile/LB timing tables and unsupported crossover advice are gone. The
retained Apple M2 Max table is explicitly historical/advisory and names
`benchmarks/mac_metal_benchmarks.md` plus the raw
`benchmarks/results/mac_m2max/metal_vs_cpu.json`.

## D8 — examples, multivariate/scores pages, and source comments drifted

Registered repair band:

- add one unconditional guard over the named pages/comments before edits and
  demonstrate failure on the inherited deprecated names and overbroad claims;
- all current examples use canonical `set_n_clusters` and canonical snake_case
  score names; no nonexistent `cluster_labels` member remains;
- multivariate docs enumerate the actually implemented variant routes,
  `MVMode::Dependent`/`Independent` constraints, metric scope, MSM/TWE
  rejection, Python `ndim` construction, and low-level-only LB primitives;
- score docs state live formulas and edge behavior for silhouette, DBI, Dunn,
  inertia, CH, ARI, and NMI without unsupported popularity/speed claims;
- source comments state delegation rather than zero-overhead claims, remove
  session/task narration, list the selected floating-point relaxations without
  calling them all/full `-ffast-math`, remove dead SIMD/Highway prose, and
  describe TWE's finite sentinel rather than claiming the build is
  `-ffast-math`;
- the real-CLI documentation contract gate exits zero with exact final line
  `documentation contract checks passed`; `git diff --check` is clean.

Deliberate red gate before the documentation corrections:

```text
generated documentation is current
Traceback (most recent call last):
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 415, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 407, in main
    assert_remaining_docs_truth()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 321, in assert_remaining_docs_truth
    raise AssertionError(f"remaining docs retain stale claims: {present}")
AssertionError: remaining docs retain stale claims: ['set_numberOfClusters', 'set_number_of_clusters', 'daviesBouldinIndex', 'dunnIndex', 'calinskiHarabaszIndex', 'adjustedRandIndex', 'normalizedMutualInformation', 'davies_bouldin_index', 'dunn_index', 'calinski_harabasz_index', 'adjusted_rand_index', 'normalized_mutual_information', 'cluster_labels', 'All DTW variants have `_mv` counterparts', 'Zero Overhead for Univariate', 'without performance penalty', 'loading added consistently later', 'task R1', 'zero overhead', 'all other fast-math optimizations', 'Full safe fast-math subset', 'DTWC_ENABLE_SIMD', 'Highway', 'the build is -ffast-math']
```

The first canonical rebuild wrapper yielded after two minutes with no build
output:

```text
command timed out after 124116 milliseconds
```

The child Ninja build remained active and completed. The verification rerun
then returned exit 0:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

Final decisive gate:

```text
generated documentation is current
documentation contract checks passed
```

Secondary gates:

```text
.venv/Scripts/python.exe -m py_compile scripts/check_docs_contract.py
exit 0; no output

git diff --check
exit 0; no diagnostics
```

**D8 verdict: PASS.** The inherited red named all 24 stale marker classes.
The final real-CLI contract gate rejected none and required the live
multivariate route limits, score edge cases, and floating-point/sentinel
comments.

The generator named only the expected projection. Regeneration and final gates:

```text
generated documentation updated
generated documentation is current
generated documentation is current
documentation contract checks passed
```

`git diff --check` emitted no errors; only the contract, its generated Tier-2
projection, the Unreleased correction, and this run-log changed.

**D2 verdict: PASS.** Present-tense defaults and binding types now match current
source. The contract also states the previously omitted Float32 recurrence
precision instead of mistaking a double result container for double
accumulation.

## D3 — README feature, benchmark, architecture, and option drift

[confirmed] The real CLI exposes seven canonical variants and eight canonical
non-`auto` methods. Current source defines the CUDA default architecture list
as `60;70;75;80;86;89;90` (`CMakeLists.txt:145`) and 18 Boolean project options
across the root, dependency, and standard-settings files.

The inherited README instead says five variants, three algorithms, omits
sm_60/sm_75, and omits seven options. Its 9–11× exact lower-bound matrix claim
is contradicted by the registered LB-cascade experiment
(`.claude/baselines/2026-07-08-lb-cascade.md`): abandoned pairs are recomputed,
so the exact matrix path did more work. The 12×/1.7×/42× headline and 0.003%
Float32 maximum have no tracked originating result artifact; they are
unconfirmed, not asserted false.

Registered repair band:

- feature/method/variant counts and CUDA defaults match the live CLI/source;
- every live Boolean project option plus the two documented cache selectors
  appears once in the option table with its actual default;
- untracked numeric claims and the falsified exact-matrix pruning speedup are
  removed, not replaced by new estimates;
- Float32 and `--ram-limit` scope match the precision/F7 contracts;
- example TOML/YAML no longer call Float32 the default;
- `check_docs_contract.py --cli` and `git diff --check` pass.

Source/table and real-binary audits:

```text
source_options=18 table_entries=20 table_unique=20
missing=
unexpected=
duplicates=
method_phrase=True
variant_phrase=True
cuda_default=True
stale_float_default=False
```

The 20 table rows are all 18 Boolean options plus the two documented cache
selectors `DTWC_ARCH_LEVEL` and `DTWC_CUDA_ARCH_LIST`.

Final gate:

```text
generated documentation is current
documentation contract checks passed
```

`git diff --check` emitted no errors.

**D3 verdict: PASS.** README counts/defaults are live-source-backed, the option
inventory is complete and duplicate-free, and untracked/falsified performance
headlines are absent.

## D4 — method catalog advertises unsupported and incomplete capabilities

[confirmed] `MetricType` contains only L1, L2, and SquaredL2
(`dtwc/core/dtw_options.hpp`), but `metrics.md` documents Huber. The variants
page omits live MSM/TWE. The algorithms page omits OneBatchPAM, LR-core, and
TADPole, gives FasterPAM the wrong venue, and states the old complexity factors.

Registered repair band:

- first add a source-doc drift check that rejects Huber and requires all live
  metric, variant, and CLI algorithm names plus the corrected FasterPAM
  venue/DOI;
- before the page edit, that new guard must fail on the inherited catalog
  (deliberate red);
- after the edit, `check_docs_contract.py --cli` passes;
- lower-bound prose distinguishes admissible threshold/NN pruning from the
  known-pessimizing exact-matrix early-abandon-plus-recompute route;
- MSM/TWE scope is explicitly univariate and unbanded; no unimplemented Python
  direct function is advertised;
- no unsupported performance or scale threshold is introduced.

Deliberate red after adding the reachable guard, before page repair:

```text
generated documentation is current
AssertionError: metrics.md advertises unsupported Huber metric
```

Final gate:

```text
generated documentation is current
documentation contract checks passed
```

The guard now requires exact section headings for DDTW/WDTW/ADTW/Soft-DTW/MSM/
TWE and all eight non-auto CLI clustering methods, rejects a Huber section/table
row and the inherited exact-matrix speed phrase, and pins FasterPAM to
*Information Systems* plus DOI `10.1016/j.is.2021.101804`.

**D4 verdict: PASS.** The inherited unsupported catalog fails; the corrected
catalog passes. Fresh rendered-link evidence remains under the registered Hugo
`[BLOCKED-ENV]` rather than being inferred from this source gate.

## D5 — configuration reference covers only part of the live CLI

[confirmed] The real CLI help contains 52 unique long flags. The inherited
configuration page contains only 34 and uses stale method/variant vocabularies.
TOML is parsed by CLI11; the optional YAML loader is a manual subset whose
canonical keys are the `set_if_unset(...)` calls in `dtwc/dtwc_cl.cpp`.

Registered repair band:

- add a gate comparing every live long flag to the configuration page; it must
  fail on the inherited page before edits (deliberate red);
- the corrected page distinguishes TOML's ordinary CLI-option mapping from
  command-line action/config selectors and from YAML's smaller manual subset;
- every canonical YAML `set_if_unset` key appears in the page;
- current method/variant vocabularies and the known YAML-overrides-CLI bug are
  explicit;
- adjacent CLI precision prose distinguishes Float32 recurrence from double
  result storage, and mmap checkpoint prose says version 3 rather than 2;
- the full live-CLI docs gate passes with 52/52 flags.

Deliberate red after the coverage guard, before page repair:

```text
AssertionError: configuration reference drift:
  live but undocumented: ['--batch-size', '--batch-weighting', '--benders', '--column', '--data-precision', '--data-type', '--dc', '--dtype', '--gpu-dtype', '--help', '--mmap-threshold', '--msm-c', '--mv-mode', '--n-clusters', '--ram-limit', '--resume', '--twe-lambda', '--twe-nu', '--version']
  documented but not live: ['--clusters']
```

Contract-source edits then produced exactly the expected generated drift:

```text
stale or missing generated documentation:
  docs\content\api\tier-2.md
  docs\content\guides\migration.md
run: python scripts/generate_docs.py
```

Final exact inventory:

```text
live_flags=52 documented_flags=52
missing_flags=
dead_flags=
yaml_keys=35 missing_yaml=
v2_doc_hits=0
```

Final gate:

```text
generated documentation is current
documentation contract checks passed
```

**D5 verdict: PASS.** Live CLI flags are covered 52/52, all 35 canonical manual
YAML keys are named, and mmap documentation consistently describes v3. The
audit also exposed a separate product defect: `--resume` loads
`ckpt_result` at `dtwc/dtwc_cl.cpp:1398-1405`, but that object has no later
consumer. Documentation now says so; R3 must pin and repair the behavior.

## D6 — frozen API contract mixes shipped behavior with pre-implementation text

[confirmed] A source-by-source audit found nine unfulfilled frozen-2.0 promises
in addition to F17. The only approved 2.1 deferrals remain the three post-freeze
MATLAB Tier-1 methods and C++ HPC transport. The contract also retains eight
“before FROZEN” reviewer questions despite its FROZEN header, stale `[new bind]`
and “unbound today” annotations for delivered bindings, and future tense for
implemented LR-core/result-writeback behavior.

Registered repair band:

- add one unique R3 finding for each of the nine independently repairable
  implementation gaps, with current source anchors and a first regression gate;
- preserve the frozen promise for every gap—no wording may reclassify an
  unfulfilled 2.0 requirement as intended or as an approved 2.1 defer;
- resolve all eight reviewer questions to the current shipped behavior and
  rename the section from open questions to adjudicated decisions;
- distinguish historical `[introduced-2.0]` provenance from current live/gap
  status; remove stale `[new bind]`, “unbound today”, “before FROZEN”, and
  “on adversarial sign-off” text;
- correct precision, Env ownership, checkpoint-v2 directory layout,
  passive `CheckpointOptions`, Python-copy/view behavior, and delivered
  LR-core/result-writeback descriptions against current code;
- add reachable drift assertions before the contract edit and show a deliberate
  red on the inherited document;
- regenerate the derived Tier-1/Tier-2/migration pages;
- the full real-CLI documentation contract gate exits zero with exact final
  line `documentation contract checks passed`; `git diff --check` is clean.

The frozen-contract governance decision is recorded in `PLAN.md` before any
contract edit. Fresh Hugo rendering remains under the registered
`[BLOCKED-ENV]` result above.

Deliberate red after adding the reachable audit guard, before contract repair:

```text
generated documentation is current
Traceback (most recent call last):
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 300, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 287, in main
    assert_contract_audit_state()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\D\git\dtw-cpp\scripts\check_docs_contract.py", line 76, in assert_contract_audit_state
    raise AssertionError(
        f"frozen contract retains pre-implementation markers: {present}"
    )
AssertionError: frozen contract retains pre-implementation markers: ['[new bind]', 'unbound today', 'before FROZEN', 'On adversarial sign-off', 'to be backed by `Env`', '**Reserved:** `Method::LRCore`', 'today it does not', 'directory checkpoint = `distances.csv` + `metadata.txt`']
```

The source-contract repair then produced exactly the registered generated drift:

```text
stale or missing generated documentation:
  docs\content\api\tier-1.md
  docs\content\api\tier-2.md
  docs\content\guides\migration.md
run: python scripts/generate_docs.py
```
