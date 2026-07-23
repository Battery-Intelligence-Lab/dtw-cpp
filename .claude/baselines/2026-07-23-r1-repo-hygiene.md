# R1 repository hygiene — 2026-07-23

## Scope and base

- Branch: `Claude`
- Base commit: `c2cc098` (`docs: reconcile durable research records`)
- Scope: tracked generated/orphan artifacts, duplicate documentation assets,
  future generated-file ignores, the public Codecov badge URL, Phase-8
  compatibility disclosures, and the branch/operator handoff record.
- Constraint: repository and documentation hygiene only; no core/runtime
  behavior changes. Data files and untracked build directories remain untouched.

## Preregistered acceptance band

The deterministic checker is added before any cleanup mutation. Its decisive
final run must print:

```text
banned_tracked_paths=0
unexpected_zero_byte_files=0
targeted_duplicate_groups=0
asset_routes=4/4
required_ignore_targets=23/23
high_confidence_secret_hits=0
codecov_badge_query_hits=0
changelog_structure=PASS
seed_compatibility_markers=2/2
VERDICT=PASS
```

Additional hard requirements:

1. The two retained `_autorun` benchmark evidence files are explicitly
   unignored while future `_autorun/*.json` output is ignored. The obsolete
   `benchmarks/baselines/*.json` rule is absent because decisive evidence lives
   only under `.claude/baselines/`.
2. Doxygen and Hugo route to the retained assets under `docs/static/`; all four
   registered asset routes exist.
3. The only permitted tracked zero-byte file is
   `python/dtwcpp/py.typed`, whose zero-byte marker form is intentional.
4. A conservative high-confidence scan of current tracked text finds no private
   key block or AWS, GitHub, OpenAI, Slack, GitLab, npm, PyPI, or Google API
   token shape. The Codecov badge contains no query string.
5. `CHANGELOG.md` has exactly one Unreleased heading and one `2.0.0rc1`
   heading; both Phase-8 F7 breaking changes remain present. It explicitly
   discloses the `portable-v1` vendor-mapping supersession and the seed-42
   default transition without promising that seed 29 recreates rc1 output.
6. The five registered non-data artifacts are absent. The orphaned
   `data/test/AllGestureWiimoteX_dist_50.csv` remains byte-identical because
   the absolute data-read-only rule overrides its zero-reference evidence.
7. The handoff records the current branch divergence, stale-ref caveat,
   operator-only fetch/push/PR/merge sequence, and every preserved local build
   root. Original global sub-band: **no remote or merge operation is
   performed**. This is retained as registered rather than silently narrowed;
   the external ref event below makes its verdict **FALSIFIED**. The separate
   authorization/compliance band is that this campaign agent issues no
   remote/merge command.
8. Final commands:

   ```text
   .venv/Scripts/python.exe -m py_compile scripts/check_repo_hygiene.py
   .venv/Scripts/python.exe scripts/check_repo_hygiene.py
   .venv/Scripts/python.exe scripts/check_docs_contract.py --cli build/highs-1151/bin/dtwc_cl.exe
   git diff --check
   ```

## Inherited-state result

Python syntax compilation exited zero with no output. The checker then exited
one and printed:

```text
banned_tracked_paths=5
unexpected_zero_byte_files=2
targeted_duplicate_groups=3
asset_routes=3/4
required_ignore_targets=0/23
high_confidence_secret_hits=0
codecov_badge_query_hits=1
changelog_structure=PASS
seed_compatibility_markers=0/2
VERDICT=FAIL
ERROR: banned tracked paths remain: ['benchmarks/baselines/.gitkeep', 'benchmarks/results/_autorun/bench_dtw_baseline_20260412_221108.json', 'docs/docs_logo.png', 'media/cluster_matrix_formation4.svg', 'media/Merged_document.png']
ERROR: unexpected tracked zero-byte files remain: ['benchmarks/baselines/.gitkeep', 'benchmarks/results/_autorun/bench_dtw_baseline_20260412_221108.json']
ERROR: targeted duplicate groups remain: [('docs/docs_logo.png', 'docs/static/docs_logo.png'), ('media/cluster_matrix_formation4.svg', 'docs/static/method/cluster_matrix_formation4.svg'), ('media/Merged_document.png', 'docs/static/method/dtw_image.png')]
ERROR: registered asset routes failed: (False, True, True, True)
ERROR: required ignore targets missing: ['/tools/emsdk/', '/tools/node/', '/web/pkg/', 'node_modules/', '/web/.vite/', '/web/coverage/', '/web/playwright-report/', '/web/test-results/', '*.pyd', '*.mexw64', '*.mexa64', '*.mexmaci64', '*.mexmaca64', '/.pytest_cache/', '/.ruff_cache/', '/.mypy_cache/', '/htmlcov/', '/.coverage', '/coverage.xml', '/CMakeUserPresets.json', '/.claude/scheduled_tasks.lock', '.env.*', '!.env.example']
ERROR: required benchmark ignore routes missing: ['/benchmarks/results/_autorun/*.json', '!/benchmarks/results/_autorun/bench_dtw_baseline_20260412_221116.json', '!/benchmarks/results/_autorun/bench_metal_dtw_20260412_212302.json']
ERROR: obsolete benchmarks/baselines/*.json ignore rule remains
ERROR: the public Codecov badge URL still contains a query string
ERROR: CHANGELOG omits one or more registered seed compatibility disclosures
```

This is the registered inherited red. Its zero secret-shape count and passing
CHANGELOG structure do not offset the eight failing hygiene subjects.

## Tracked-file census and disposition

The pre-mutation target census printed:

```text
benchmarks/baselines/.gitkeep	0	ZERO	tracked=True
benchmarks/results/_autorun/bench_dtw_baseline_20260412_221108.json	0	ZERO	tracked=True
docs/docs_logo.png	76526	4CE78596704E8A5F3EE4F46C8FAF9C5B754488158A29002CEAFCB2FD210BA051	tracked=True
docs/static/docs_logo.png	76526	4CE78596704E8A5F3EE4F46C8FAF9C5B754488158A29002CEAFCB2FD210BA051	tracked=True
media/cluster_matrix_formation4.svg	38347	02F3D582EA09A87D9BF0D4C8490B5BE76C761C3E2438B0DFA4C59EE68A3AA9EA	tracked=True
docs/static/method/cluster_matrix_formation4.svg	38347	02F3D582EA09A87D9BF0D4C8490B5BE76C761C3E2438B0DFA4C59EE68A3AA9EA	tracked=True
media/Merged_document.png	179978	7C8FBF6B1847159B5B93EDEB1DCC654041AA8FEC741987FCC664E623CFFAEAC0	tracked=True
docs/static/method/dtw_image.png	179978	7C8FBF6B1847159B5B93EDEB1DCC654041AA8FEC741987FCC664E623CFFAEAC0	tracked=True
python/dtwcpp/py.typed	0	ZERO	tracked=True
data/test/AllGestureWiimoteX_dist_50.csv	17638	FC8173362CD03DD4D13DBF32E4FD773DE1013DD1AF28E7C6E99B95A58019D283	tracked=True
```

**[confirmed]** Commit `4797c97` removes only the two unintended zero-byte
artifacts and the three byte-identical duplicate assets. Doxygen now names
`docs/static/docs_logo.png`; Hugo and the two method pages already route to
the retained static copies. The only old-logo string outside historical
records is now the permanent checker's banned-path fixture, not a consumer.

**[confirmed]** The compiled-artifact census printed:

```text
tracked_compiled_binary_count=0
```

**[confirmed]** The data fixture remains byte-identical at SHA-256
`FC8173362CD03DD4D13DBF32E4FD773DE1013DD1AF28E7C6E99B95A58019D283`.
It has no live code/document consumer; it is retained because the absolute
data-read-only rule is stronger than orphan cleanup.

## Ignore routes, badge, and render environment

The actual Git ignore probes pass: future `_autorun` JSON and direct benchmark
JSON are ignored; the two retained `_autorun` evidence files are not; future
`benchmarks/baselines/*.json` is not ignored; `.env` and `.env.local` are
ignored while `.env.example` is not. The permanent checker requires both the
registered lines and these ordered Git semantics.

The documentation/secret-tool probe printed:

```text
doxygen=NOT_FOUND
hugo=NOT_FOUND
go=NOT_FOUND
gitleaks=NOT_FOUND
trufflehog=NOT_FOUND
detect-secrets=NOT_FOUND
```

Fresh Doxygen/Hugo rendering is therefore `[BLOCKED-ENV]`; the deterministic
asset routes and existing real-CLI documentation contract are the named
fallbacks.

The tokenless and former query-bearing Codecov badge endpoints were probed
without writing a file:

```text
badge_plain_status=200
badge_tokenized_status=200
badge_plain_chars=2274
badge_tokenized_chars=2274
badge_content_identical=True
```

The public README now uses the tokenless URL. Removing a query value from the
current tree cannot remove it from reachable Git history; an operator should
revoke/rotate it if Codecov identifies it as a scoped credential.

## Secret-shape scan

The installed scanners were absent as shown above. A first revision-by-revision
`git log -G` fallback produced no result before:

```text
command timed out after 124046 milliseconds
```

That timeout is not evidence. The fallback instead read every unique reachable
blob through one `git cat-file --batch` process and applied the same
conservative byte patterns, including encrypted private-key headers and
fine-grained GitHub PATs. Its pre-`4797c97` snapshot was:

```text
reachable_objects=10255
scanned_reachable_blobs=4559
history_high_confidence_hits=0
```

After `4797c97`, the same read-only scan was rerun against the enlarged
reachable set:

```text
reachable_objects=10268
scanned_reachable_blobs=4564
history_high_confidence_hits=0
```

This confirms absence of the registered high-confidence ASCII shapes; it is
not a claim that arbitrary entropy or every vendor's credential format was
classified.

## Preserved build inventory

No build path is staged or deleted. The three ignored local roots contain 32
repo-source configure caches:

```text
build_root=build repo_configure_caches=29
build_root=build_arrow_test repo_configure_caches=1
build_root=build_python repo_configure_caches=2
```

- `build/` is the multi-recipe root: canonical `highs-1151`, llfio-OFF,
  Arrow/PyArrow-23, CUDA, MEX, sanitizer, configuration, YAML, and historical
  verification recipes.
- `build_arrow_test/` is the standalone Debug Arrow+HiGHS configure.
- `build_python/` contains the active CPython 3.13 and final-wheel MSVC
  configurations.

These roots encode verified recipes. Disposal remains an operator decision.

## Branch-state snapshot and operator plan

Immediately after committed hygiene snapshot `4797c97`, and before the
remote-tracking update recorded below, the read-only probe printed:

```text
main_vs_Claude=0	548
origin_main_vs_Claude=0	548
origin_Claude_vs_Claude=0	51
main_vs_origin_main=0	0
origin_Claude_vs_origin_main=497	0
 640 files changed, 153326 insertions(+), 2144 deletions(-)
b0cb297 refs/remotes/origin/main@{2026-04-04 00:34:34 +0100}: fetch --progress --prune --recurse-submodules=on-demand origin: fast-forward
c36ba27 refs/remotes/origin/Claude@{2026-07-13 15:12:28 +0100}: update by push
```

The remote-reference counts are explicitly stale until an operator fetches.
After R0–R6 are clean, the proposed operator sequence is: fetch/prune; recompute
ancestry/divergence; rerun release gates against the fetched base; push
`Claude`; open a review PR; require hosted gates; and use a fast-forward merge
only if the fresh ancestry check proves it valid. Any rebase, merge commit,
force update, tag, or publication remains an operator decision.

During final record review, `origin/Claude` changed without a command from this
agent. The 2026-07-23 18:15:38 +01:00 read-only probe printed:

```text
HEAD=4797c97020a6b9bd9fb78e43da53cf0b30eadbb4
Claude=4797c97020a6b9bd9fb78e43da53cf0b30eadbb4
origin_Claude=4797c97020a6b9bd9fb78e43da53cf0b30eadbb4
origin_main=b0cb297cb633094f3b7c0a65e2e04fc9f951a72b
origin_Claude_vs_Claude=0	0
origin_main_vs_Claude=0	548
4797c97020a6b9bd9fb78e43da53cf0b30eadbb4 origin/Claude@{2026-07-23 18:14:56 +0100}: update by push
c36ba27e04bd3f8859765526b70596a20d1920db origin/Claude@{2026-07-13 15:12:28 +0100}: update by push
```

**[confirmed]** The local reflog says `update by push`; `origin/Claude` moved
from `c36ba27` to `4797c97`. **FALSIFIED:** the preregistered global statement
that no remote operation would occur. **[confirmed]** This campaign agent
issued no push/fetch command. The attribution probe printed:

```text
github_desktop_process_count=4
github_desktop_pids=5684,40592,50008,88212
github_desktop_start_times=2026-07-23T17:26:07+01:00,2026-07-23T17:26:07+01:00,2026-07-23T17:26:07+01:00,2026-07-23T17:26:06+01:00
git_exe_process_count=0
non_sample_hook_count=0
```

GitHub Desktop was therefore an extant alternate Git client before the ref
update; no evidence identifies whether it or another actor performed the push.
Actor/cause remains **[inferred: unknown]**, not “external/background” as a
fact. If the update was unintended, rollback is an operator decision:
restore the remote branch to the desired reviewed commit through the hosting
service or an explicitly authorized ref update; this campaign will not
force-push. The proposed merge plan remains operator-only and must begin by
re-reading the server state.

## Adversarial checker review

The first apparent post-cleanup green was rejected four times before commit:

1. worktree existence could hide an index-tracked deleted path, and a missing
   CHANGELOG heading could raise before `VERDICT`;
2. encrypted PKCS#8 and fine-grained GitHub PAT shapes were absent;
3. NUL-free non-UTF-8 blobs were skipped before byte regexes ran;
4. set membership could not prove ordered `.gitignore` exception semantics.

The corrected checker reads index entries and blob identities directly,
scans its own staged blob, applies byte patterns to every NUL-free blob, handles
malformed headings as a FAIL, and executes filter-aware ordered ignore probes.
The independent final review verdict was `GREEN`.

## Final gates

The PLAN, handoff, LESSONS entry, and this run-log were staged before the
decisive run, so the checker scanned their index blobs too.

Python syntax compilation exited zero with no output:

```text
.venv/Scripts/python.exe -m py_compile scripts/check_repo_hygiene.py
```

The permanent checker printed the exact registered band:

```text
banned_tracked_paths=0
unexpected_zero_byte_files=0
targeted_duplicate_groups=0
asset_routes=4/4
required_ignore_targets=23/23
high_confidence_secret_hits=0
codecov_badge_query_hits=0
changelog_structure=PASS
seed_compatibility_markers=2/2
VERDICT=PASS
```

The real-CLI documentation gate printed:

```text
generated documentation is current
documentation contract checks passed
```

Because the closeout adds a new checker lesson, the companion durable-record
gate was rerun and printed:

```text
record hygiene checks passed
```

`git diff --cached --check` and the preregistered `git diff --check` each
exited zero with no output. `git add` had emitted the repository's existing
LF-to-CRLF conversion warnings for the four documentation records; those are
not whitespace-error diagnostics from either diff check.

## Verdict

**PASS [confirmed] for the repository-hygiene implementation; FALSIFIED for
the global no-remote-operation sub-band.** Evidence is the exact checker band
above, commit `4797c97`, the byte/hash census, the batched history scan, the
branch/build inventories, the real-CLI documentation gate, and the independent
review. This campaign agent's no-remote-command authorization band passes, but
the reflog proves a remote update occurred and its actor is unknown. R1 is
closed because the branch-state task requires truthful recording, not repair
of an external ref. Fresh Doxygen/Hugo output remains the named
`[BLOCKED-ENV]` limitation.
