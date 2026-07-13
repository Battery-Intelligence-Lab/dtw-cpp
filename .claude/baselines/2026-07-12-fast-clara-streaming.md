# Phase 8.2 F7 — metadata-first Parquet FastCLARA

Date: 2026-07-12 (Europe/London)

## Confirmed failure

The CLI parsed `--ram-limit` but loaded the complete Parquet payload into
`Problem` before calling FastCLARA. FastCLARA then opened a second row-group
reader. The advertised cap therefore selected chunking only after the full
resident allocation and made the peak larger by adding sample, medoid, and
assignment chunks.

The two Parquet readers also disagreed at load-planning boundaries. A scalar
column is one series spanning all rows, whereas a List/LargeList column is one
series per row; auto-detection did not share one rule, and an Arrow top-level
field index was incorrectly usable as a Parquet physical leaf index when a
preceding nested field owned multiple leaves. `total_uncompressed_size` alone
also underestimated decoded dictionary data.

## Adversarial design check before simulations

The implementation was reviewed orthogonally before any end-to-end simulation:

1. The cap was scoped explicitly to selected-series decoding and
   materialisation, not total process RSS. Source decode buffers, requested
   target precision, per-series vector/name objects, and the eager f64-to-f32
   conversion overlap are counted with saturating arithmetic. Algorithm result
   vectors, the O(s²) PAM matrix, Arrow metadata, and fixed process overhead are
   outside this stated cap.
2. The route matrix was made fail-closed. Only one list-per-row file with
   non-full CPU FastCLARA can stream. Over-budget scalar columns, directories,
   non-CLARA methods, full samples, CUDA, and legacy parent matrix/checkpoint
   paths have no semantically valid streaming implementation and must reject
   before payload I/O.
3. Row groups were treated as indivisible. The largest selected source/target
   group must fit beside retained sample or medoid payload; batching cannot make
   one oversized group safe.
4. Streaming ownership was checked independently of the CLI: a forced stream
   requires a settings-only `Problem`. Supplying both resident data and chunks
   is rejected in the library, preventing a caller from recreating the original
   hidden peak.
5. Eager, metadata, sparse, and row-group reads were traced through one schema
   selector. It accepts only scalar/List/LargeList Float32/Float64, validates
   list offsets, and counts every leaf of preceding nested fields before issuing
   a Parquet-column read.
6. Float32 was traced end to end. The sample, selected medoids, and assignment
   row groups all use f32 storage and the f32 dispatcher; only returned
   distances and the ordered objective accumulation are double.
7. Resident and streamed assignment were checked for arithmetic-order and
   diagonal differences before comparing output. Both use a global index-order
   left fold, stable medoid tie order, and exact zero for a medoid assigned to
   itself.
8. Streamed output validates every result size/index before opening a file and
   derives `series_i` names without a resident `Data` copy, matching the eager
   list reader.

Only after these checks and the focused unit/compile gates passed were the
Parquet resident-versus-stream simulations run.

## Resolution

The CLI now opens only schema and row-group metadata first. It resolves logical
N, the 5,000-series `auto` threshold, CLARA controls/sample plan, selected dtype,
and a conservative materialisation peak before deciding whether payload loading
is legal. If the estimate fits, the existing eager path runs. If it exceeds a
nonzero cap, a single list-per-row FastCLARA input is represented by a
settings-only `Problem` and `force_parquet_streaming=true`; no full series copy
is retained.

`--ram-limit` uses an exact checked decimal/rational parser. Bare bytes and
case-insensitive binary B/K/KB/KiB through T/TB/TiB units are accepted. A
decimal must resolve to a whole byte, and malformed, negative, non-finite,
fractional-byte, or platform-overflowing values fail before environment setup,
output-directory creation, or payload I/O. This avoids the >2^53 rounding loss
of a floating-point size parser.

The Parquet reader estimates each physical numeric leaf from at least
`num_values * source_width`, taking the maximum with encoded metadata rather
than trusting `total_uncompressed_size`. Sparse and row-group reads check the
source decode, target representation, object overhead, retained payload, and
selected-copy overlap before allocating. A row group that cannot fit is rejected
with guidance to rewrite smaller row groups or raise the cap.

Non-full FastCLARA no longer accepts `--checkpoint` or `--dist-matrix`, because
either would consume unused parent O(N²) state. The automatic binary clustering
checkpoint is still written. Streamed CLI output consists of labels, medoids,
and that binary checkpoint; it does not synthesize dense distance-matrix or
silhouette CSVs. The frozen API decision is now precise: `Result::save()` keeps
its explicit four-file behavior, while corresponding CLI artifacts are
byte-identical whenever present.

## Signed Soft-DTW sampling finding

The first Soft-DTW parity simulation failed before clustering.
Raw Soft-DTW is an admissible dissimilarity but can have finite negative
off-diagonal values; the portable weighted sampler correctly rejected those as
probabilities. The failure exposed a shared initialization bug rather than a
streaming discrepancy.

FastPAM and k-means++ now obtain weights from one helper. If every unselected
nearest distance is nonnegative, the weight vector is byte-for-byte unchanged.
Otherwise, with `m` the minimum unselected distance, each unselected weight is
`d_i - min(0,m)` and every already selected point remains exactly zero. This
common translation preserves distance differences and ordering while producing
finite nonnegative sampling weights. The existing 4,096-seed 1:3 D-sampling
distribution remains inside its registered 0.70–0.80 band.

The review then found the complementary degenerate case: identical series (or
equal signed dissimilarities after translation) leave every unselected weight
at zero. Constructing `std::discrete_distribution` with a zero total is not a
valid weighted draw and can reselect an existing centroid. Seeded and unseeded
k-means++ now share the same deterministic completion rule—select the first
unselected index—and a four-identical-series, k=3 regression pins distinct
medoids.

## Focused validation

The final focused canonical rerun passed:

```text
unit_test_clustering_algorithms [init]:                    7 assertions, 3 cases
unit_test_fast_pam (Soft-DTW + seeded initialization): 7 assertions, 2 cases
unit_test_fast_clara:                                  838 assertions, 20 cases
unit_test_cli_args [cli][parquet]:                      32 assertions, 5 cases
```

A manually Arrow-linked executable, using the installed PyArrow 23 headers and
libraries plus the production Arrow-enabled FastCLARA object, passed the real
reader suite:

```text
test_io_readers [io][parquet]: 348 assertions, 7 cases
```

That suite covers scalar logical N=1 and sparse rejection, eager/sparse f64 and
f32 list reads, duplicate sparse indices, auto column selection, corrupt list
offsets, dictionary-compressed estimate undercount, selected-copy cap
boundaries, and an eligible field following a multi-leaf Struct.

The optional-dependency-off CLI translation unit also passed a strict syntax
gate:

```text
clang++ -std=c++20 -fsyntax-only -Wall -Wextra -Werror \
  -Wno-deprecated-declarations -Wno-unknown-pragmas ... dtwc/dtwc_cl.cpp
```

Real CLI preflights returned nonzero before prohibited effects for a malformed
RAM limit, over-budget scalar streaming, `n_samples=0`, streamed legacy
`--checkpoint`, and streamed `--dist-matrix`.

## Resident/stream simulations

The materialized fixture contains eight list-row series in four row groups of
two rows. Each row group reports eight primitive values; its encoded column
sizes are 125, 142, 134, and 159 bytes. The conservative complete
materialisation estimate is at least 1,568 bytes, while the worst sparse sample
path needs less than 815 bytes and fits the registered 900-byte cap. The same
file can therefore exercise both the resident and forced-stream branches
without changing data.

With k=2, sample size 4, two samples, seed 42, and cap 900, all three comparisons
were exact:

| Configuration | Total cost | Medoids | Labels SHA-256 | Medoids SHA-256 | Checkpoint SHA-256 |
|---|---:|---|---|---|---|
| f64 resident = stream | 4.4 | 1, 7 | `39CCD0E9D5F520678899193B7A7903D6331217B0F99C96417F24BF943A7268EB` | `2CEF6784BE8808E00B6F78D3EF2DE8E55C6D4E7021F9C81092EB6A0279E3D22F` | `B666C1108C42A6B0705741B357F2527E319FEEBE3380BE6623E1494DAFD1A02C` |
| f32 resident = stream | 4.4 | 1, 7 | `39CCD0E9D5F520678899193B7A7903D6331217B0F99C96417F24BF943A7268EB` | `2CEF6784BE8808E00B6F78D3EF2DE8E55C6D4E7021F9C81092EB6A0279E3D22F` | `EEA65070341F900BA212909242AD54FE7A48F1E4A94704864CBF77C89FC28FC1` |
| Soft-DTW resident = stream | -10.344 | 1, 7 | `39CCD0E9D5F520678899193B7A7903D6331217B0F99C96417F24BF943A7268EB` | `2CEF6784BE8808E00B6F78D3EF2DE8E55C6D4E7021F9C81092EB6A0279E3D22F` | `67D818D58370CB6E17E4E8922175A343ADE2D53C7873A62522FE9D3C14734B79` |

The stream logs omit the eager `Data loaded from Parquet` line, confirming the
payload was not materialised before FastCLARA. All three streamed output sets
contain only labels, medoids, and the 72-byte binary result checkpoint.

## Independent adversarial review

Final verdict:

> PASS — no remaining F7 blocker in the current tree.

The reviewer independently confirmed metadata-first routing, hard-cap
arithmetic, scalar/list semantics, physical leaf mapping, f32 and Soft-DTW
parity, output naming, CUDA/cache routing, overflow handling, and settings-only
streaming ownership as coherent. `git diff --check` was clean at review time.

## Final-tree closeout

The four focused executables above were rebuilt and rerun under both independent
sanitizer toolchains. MSVC 19.50 `/fsanitize=address` used
`ASAN_OPTIONS=halt_on_error=1:detect_leaks=0:strict_string_checks=1`; Clang
18.1.3 UBSan on Ubuntu 24.04 WSL used
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. Both passed with the same
assertion/case counts and no sanitizer report.

The final canonical Windows Clang 21.1.8 Release configuration then rebuilt all
targets and passed:

```text
100% tests passed, 0 tests failed out of 113
Total Test time (real) = 27.36 sec
6 explicit capability skips: CUDA correctness/LB, Arrow reader, Metal
correctness/LB/mmap
```

The canonical Arrow reader skip is capability-only: the separately rebuilt
Arrow/Parquet executable above passed all 348 assertions against PyArrow 23.0.1.

## Post-hoc independent re-review (2026-07-13) — the PASS above was NOT final

The section above was written by the implementing agent, whose own adversarial
reviewer returned "PASS — no remaining F7 blocker in the current tree". A second,
independent adversarial review of the same uncommitted tree found **two real
defects that reviewer missed**. The 113/113 gate was independently reproduced
(same 6 capability skips), so the headline number was honest; the gate simply
does not cover what broke.

### D1 [CONFIRMED, FIXED] — `--ram-limit` was silently ignored for all non-Parquet input

`ram_limit` reached only three live sites, all inside the Parquet branch. The
pre-existing `Warning: --ram-limit only effective with Parquet input` was deleted
by this change with nothing replacing it, while the verbose path still printed
`Series-data RAM limit: N bytes`. Net effect: `dtwc_cl -i huge.csv --ram-limit 1G`
loaded the whole file, reported a cap, and applied none — the exact "advertised
but unapplied cap" deceit F7 exists to remove, re-entering for every other format.

**Compounding fault (the instructive one):** the first fix was placed inside
`#ifdef DTWC_HAS_PARQUET`. The canonical gate builds `DTWC_ENABLE_ARROW=OFF`, so
the guard is false and the check did not exist in the binary. Its unit test passed
(the test TU defines the macro), the full 113-test gate passed, and the real CLI
still ran CSV + `--ram-limit 1G` to completion, exit 0. Only executing the real
binary exposed it. The check now sits beside the filesystem classification,
outside the guard.

Registered bands, both CONFIRMED by driving the real `dtwc_cl.exe`:

- non-Parquet input + non-zero cap → reject, exit != 0. Observed exit=1,
  `Error: --ram-limit caps Parquet series materialisation and cannot be honoured
  for this input; ...`
- non-Parquet input, no cap → unchanged, exit 0, all four CSV artifacts emitted.

Pinned by `unit_test_cli_args.cpp` "--ram-limit is rejected where no reader can
honour it" (6 assertions), calling production `require_ram_limit_is_applicable`.
Post-fix: canonical gate **113/113, 0 failed**; the `DTWC_HAS_PARQUET` branch
syntax-checked clean against PyArrow 23.0.1 headers (that branch is invisible to
the Arrow-OFF gate).

### D2 [CONFIRMED, DOCUMENTED — not a bug, a mis-scoped changelog]

`matrix_free_method` gained `|| (method == "clara" && !clara_uses_full_sample)`,
which feeds a pre-existing `EXIT_FAILURE` for `--device cuda`. This fires on
**default flags with no `--ram-limit`**: `--method auto` above 5,000 series
resolves to CLARA, so `dtwc_cl -i data.csv -k 10 --device cuda` at N=20,000 now
hard-fails where it previously ran. The rejection is *correct* — non-full
FastCLARA is matrix-free after F6, so the GPU matrix was built and discarded —
but CHANGELOG scoped it to over-budget requests only. Now recorded as an explicit
breaking change.

### Standing gaps (NOT closed — carry as new 8.2 findings)

1. The resident-vs-stream byte-identity table above is a **one-off manual
   simulation**. No committed test pins it; nothing prevents regression.
2. `test_io_readers` (348 assertions) is skipped in the canonical gate.
3. The seeded half of the zero-weight k-means++ fix is untested (unseeded is
   pinned); `dtwc::core::distance_sampling_weights` has no direct unit test.
