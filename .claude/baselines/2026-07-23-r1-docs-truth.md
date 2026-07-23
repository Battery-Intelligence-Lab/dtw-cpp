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
