# Obsolete / stale-record sweep — 2026-09-02

Scope: dangling references to removed APIs, dead code, stale project records,
stale test/CMake registrations, docs drift, `.mailmap`. Branch `Claude`.
No commits, no pushes. Off-limits files were inspected but never edited.

## Headline

The tree is much cleaner than the brief assumed: **zero** live references to
`rapidcsv`, `readCSV*`, `pybind11` (outside one skill), the removed
`core::*Dist` functors, the removed `medoid_utils` helpers, or the F15
per-compiler hash profiles. The real rot is in the **agent-facing records**
(`.claude/skills/*`, `.claude/TODO.md`, style guides) and two dangling doc
links. 13 items fixed, 6 reported, 4 kept-deliberate.

## Findings

| file:line | what | action |
|---|---|---|
| `.claude/skills/python-wrapper-skill.md:2,13` | Frontmatter + title tell a future agent to **generate pybind11 bindings**; repo is nanobind (`python/src/_dtwcpp_core.cpp`). Highest derail risk found. | **fixed** — description rewritten to HISTORICAL, blockquote STATUS banner added ("do not emit `pybind11/*` / `PYBIND11_MODULE`; extend the nanobind module") |
| `.claude/skills/python-wrapper-skill.md:19` | `Python >= 3.8`; `pyproject.toml:20` is `requires-python = ">=3.9"` | **fixed** |
| `.claude/skills/python-wrapper-skill.md:18`, `matlab-wrapper-skill.md:18` | "**C++17** or newer"; project minimum is C++20 (`.claude/CLAUDE.md` rule 7, `cpp-style.md`) | **fixed** (both) |
| `.claude/skills/python-wrapper-skill.md:622`, `matlab-wrapper-skill.md:722` | `import dtwc` — no such package; it is `dtwcpp` | **fixed** → `import dtwcpp as dtwc` |
| same two files, "Example (DTW clustering workflow)" | Snippets use `dtwc.DataLoader` / `dtwcpp.DataLoader`, which exist in **neither** binding (`grep -n DataLoader python/dtwcpp/__init__.py python/src/_dtwcpp_core.cpp` → empty) | **fixed** — 3-line note marking the block illustrative, pointing at `dtwcpp.load` / `dtwc.load` |
| `.claude/skills/update-docs-skill.md:60` | `dtwc_cluster(data,'k',5,...)` — no such MATLAB function anywhere (`grep -rn dtwc_cluster bindings/matlab examples/matlab docs/content` → empty); real signature is `dtwc.cluster(data, k, ...)` (`bindings/matlab/+dtwc/cluster.m:4`, k **positional**) | **fixed** → `dtwc.cluster(data, 5, 'method','clara')` |
| `.claude/skills/update-docs-skill.md:68` | Points at `bindings/matlab/examples/example_quickstart.m`; file lives at `examples/matlab/example_quickstart.m` | **fixed** |
| `CONTRIBUTING.md:34` → `docs/content/contributing/guideline.md:35` | Link to `…/tree/main/develop/TODO.md`; `develop/` holds only `contributors.md`, `conventions.md` — public 404. `scripts/check_site_links.py` only validates *local* hrefs, so it never caught this. | **fixed** in the **source** (`CONTRIBUTING.md`), regenerated |
| `docs/content/contributing/_index.md:14` | Same dead `develop/TODO.md` link (hand-written page, not generated) | **fixed** |
| `docs/sources/lr-core-derivation.md:73` → `docs/content/math/lr-core.md:86` | Cites `.claude/TODO.md:110` for the "odd-cycle cutting planes" item; line 110 is now `UP01/quickcpplib`, and the item is row **M01** (already falsified) | **fixed** in the source, regenerated |
| `.claude/TODO.md` B03 | Claimed `*.mexw64` ignore protection still OPEN; `.gitignore:67-69` has all three MEX globs and `check_repo_hygiene.py` asserts them | **fixed** → DONE, CLOSED-BY `8debf1d`, `4797c97` |
| `.claude/TODO.md` D03 | Claimed duplicate docs logos still OPEN; `docs/docs_logo.png` no longer exists (deleted in `4797c97`), `docs/static/docs_logo.png` is sole source, both consumers asserted | **fixed** → DONE, CLOSED-BY `4797c97` |
| `.claude/cpp-style.md:16` | Lists `distMat` as an example of "private, trailing `_`"; it is a **public** member with no underscore (`dtwc/Problem.hpp:135`) | **fixed** → `dtw_fn_`, `data_`, `series_storage_owner_` |
| `.claude/python-style.md:55-69` | Package sketch listed 3 modules; `python/dtwcpp/` has 14, incl. `distance.py` which `design.md` calls the public pairwise surface | **fixed** — minimal expansion, no restructure |
| `dtwc/mpi/mpi_distance_matrix.cpp:96` | `TODO(Phase 5, perf — PLAN.md "Phase 5 — Speed & algorithms program")`; that heading is no longer in `PLAN.md` (v2.1 is R0–R7) — it is in `.claude/PLAN-archive-2026-07-20-phases0-9.md:460` | **fixed** — comment repointed to the archive + PLAN R5. Comment-only. |
| `.mailmap` | Does not exist. `git log --format="%an <%ae>"` → `Kasper Westman <Kasper Westman>` (2 commits: `73084e1`, `95bd137`). GitHub handle is **KaWest** (from `6c66fdb` subject "Merge PR #32 (KaWest:Claude)"). His numeric GitHub id is **not derivable** from the repo, so the modern `<id>+KaWest@users.noreply.github.com` form cannot be written without fabricating. | **reported** — see proposed line below |
| `dtwc/algorithms/detail/medoid_utils.hpp:5-6` | Tombstone says `compute_nearest_and_second` "was removed", but a live function of that exact name is defined at `dtwc/algorithms/fast_pam.cpp:95` and called 6×. The tombstone means the `detail::` copy; a reader can misread it as fast_pam's being dead. | **reported** — editing this header forces a full rebuild (pulled in by `dtwc/dtwc.hpp:34`) for a comment; low value, real confusion risk |
| `.claude/skills/matlab-wrapper-skill.md:712,729,746` + `python-wrapper-skill.md:619,636,653` | `data/ECG200` does not exist (`data/` = benchmark, dummy, test, test_kasper, test_parquet) | **kept** — inside the blocks now marked illustrative; swapping in a real path would imply the whole snippet runs, which it does not |
| `dtwc/dtwc_cl.cpp` (YAML, ~30 refs) + `CMakeLists.txt:48` + `cmake/Dependencies.cmake:119-138` + `examples/cpp/config.yaml` | YAML is **not** obsolete: `DTWC_ENABLE_YAML` is a live optional dep (default OFF) with a real `--yaml-config` path and a CLI test at `tests/unit/unit_test_cli_args.cpp:125`. The "YAML → TOML" migration made TOML the default, not the only format. | **kept-deliberate** |
| `.claude/design.md:26` "DTW is **latency-bound**" | Conflicts with the "memory-bound (0.125 FLOP/byte)" line in my session memory — but `.claude/LESSONS.md:156` records "**no PMU artifact proves a memory-bound bottleneck**". design.md is the better-evidenced statement. | **kept** — not changed; flagging the memory note as the unsupported one |
| `[[deprecated]]` camelCase aliases in `Problem.hpp`, `DataLoader.hpp`, `scores.hpp` | 2.x shims, removed in 3.0 | **kept-deliberate** |

## Clean sweeps (empty results, so no action)

```
grep -rn -E "rapidcsv|readCSV|readTimeSeriesCSV|readCSVColumn|SpanSquaredL2Cost|
             kLibcxx|kLibstdcxx|kRelaxed|kPrecise"   <explicit paths>   → empty
grep -rn -E "assign_to_nearest|find_cluster_medoid"  <explicit paths>   → only the
             medoid_utils.hpp tombstone
grep -rn -E "pybind11|rapidcsv|readCSV|SpanSquaredL2|core::.*Dist" python/  → empty
grep -rn "#if 0" dtwc tests python bindings benchmarks examples
             → 1 hit, a *comment* recording their removal (test_scores_adversarial.cpp:451)
TODO|FIXME|XXX|HACK in dtwc/  → 2 real (dtwc_cl.cpp:1029 live, mpi:96 fixed above)
                                + 2 false positives in dtwc/extern/nanoarrow
unused headers under dtwc/    → none (comm of all dtwc headers vs the full
                                #include set across dtwc tests python bindings
                                benchmarks examples → empty)
unused CMake options          → none (every option()/cmake_dependent_option name
                                is read somewhere)
tests/CMakeLists.txt sources  → all exist (3 apparent misses are
                                ${CMAKE_CURRENT_BINARY_DIR}/f22_*.cpp generated files)
F15_TEST_SUPPORT floor (tests/CMakeLists.txt:376) → still emitted by
                                unit_test_deterministic_series.cpp:385; NOT stale
scripts/*.py|ps1|sh hardcoded paths → 30 apparent misses, all false positives
                                (ROOT-relative joins), verified individually
.claude/commands/*.md          → clean
```

## Commands run (post-edit)

| command | result |
|---|---|
| `uv run python scripts/generate_docs.py` | `generated documentation updated` |
| `uv run python scripts/check_docs_contract.py` | `generated documentation is current` / `documentation contract checks passed` — **PASS**, no `quickstart.cpp` failure |
| `uv run python scripts/check_supply_chain_pins.py` | `supply-chain pins verified`, 39/39 + 7/7 + 1/1, exit 0 |
| `uv run python scripts/check_repo_hygiene.py` | `VERDICT=PASS`; `targeted_duplicate_groups=0`, `required_ignore_targets=23/23`, `asset_routes=4/4` — this is the direct proof for the B03/D03 closures |
| `uv run python scripts/check_record_hygiene.py` | `record hygiene checks passed` |
| `cmake --build build/nollfio` | full relink, then `ninja: no work to do`, exit 0 |

**Trap worth recording:** `docs/content/contributing/guideline.md` and
`docs/content/math/lr-core.md` are **generated** (`scripts/generate_docs.py:181,218`)
from `CONTRIBUTING.md` and `docs/sources/lr-core-derivation.md`. Editing the
generated page is silently reverted by the next contract check. Only
`docs/content/contributing/_index.md` and `docs/content/_index.md` differ:
the latter is generated (`generate_docs.py:162`), the former is not.

## Proposed CHANGELOG bullets (Unreleased) — not applied

- Docs: fixed two dead `develop/TODO.md` links in the contributing pages (the
  file has never existed at that path) and repointed the LR-core derivation's
  stale `.claude/TODO.md:110` citation at row M01.
- Docs: agent skill records refreshed — the Python wrapper skill is marked
  historical (the project ships nanobind, not pybind11), the MATLAB doc-update
  snippet now uses the real `dtwc.cluster(data, k, ...)` signature, and both
  wrapper skills state the C++20 / Python 3.9 minimums.

## For you to route (off-limits to me)

1. **`.mailmap` (needs your GitHub lookup).** Nothing to fix in-tree, but two
   commits carry `Kasper Westman <Kasper Westman>`. Once you have his numeric
   id from `https://api.github.com/users/KaWest`, add:
   `Kasper Westman <<id>+KaWest@users.noreply.github.com> Kasper Westman <Kasper Westman>`
   I did not write a guessed address.
2. **`dtwc/algorithms/detail/medoid_utils.hpp:5-6`** — tombstone/live-symbol
   name collision described above. Header-only comment fix; costs a full
   rebuild, so it belongs in someone else's batch.
3. **`dtwc/dtwc_cl.cpp:1029`** — live `TODO: This always overrides CLI values if
   the YAML key exists.` A real precedence bug (the comment at line 1023 claims
   "CLI flags take precedence" while the code does the opposite for YAML).
   Owned by the agent currently in that file. Worth a finding of its own.
4. **`.claude/skills/python-wrapper-skill.md`** — I banner-marked it rather than
   deleting 671 lines. If you would rather it go, note
   `scripts/check_docs_contract.py:2022-2023` pins both wrapper skills into the
   camelCase-hygiene check and would need the paths dropped.

## Claim I most expect to be wrong

That the `dtwc_cluster` / `DataLoader` snippets in the wrapper skills were meant
as *illustration* rather than as claims about the shipped API. I treated them as
templates and annotated instead of rewriting; if they were meant to document the
real surface, they need rewriting against `bindings/matlab/+dtwc/` and
`python/dtwcpp/`, not a disclaimer.
