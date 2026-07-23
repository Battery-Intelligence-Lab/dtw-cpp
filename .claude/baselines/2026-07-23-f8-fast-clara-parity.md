# R3-F8 FastCLARA resident/stream parity — 2026-07-23

## Scope and base

- Branch: `Claude`
- Base commit: `e0e2db8` (`docs: close DTW derivation task`)
- Subject: the real `dtwc_cl` Parquet FastCLARA route in resident and
  RAM-forced streaming modes for Float64 Standard DTW, Float32 Standard DTW,
  and Float64 Soft-DTW.
- The historical one-off comparison in
  `.claude/baselines/2026-07-12-fast-clara-streaming.md` is fixture-selection
  evidence only. It is not the F8 verdict.
- This registration precedes copying the binary fixture into tracked test
  data, adding the integration test, rebuilding either gate, or executing any
  decisive F8 CLI comparison.

No killed idea is reopened. The PLAN/archive/LESSONS search found only the
binding requirement that F8 turn the historical simulation into a permanent
real-binary gate.

## Registered fixture

The read-only source artifact is
`build/phase8-f7-arrow/list.parquet`. The tracked copy will be
`tests/fixtures/fast_clara_streaming_8x4.parquet` and must be byte-identical:

```text
size=1451
sha256=2F259F418A6BB9C62CA0004CB334C05E8309213F5A76DC890F83C15D5BDA3CA8
```

The historical artifact records one `List<Float64>` column named `series`,
eight rows, and four row groups of two rows. Its series are:

```text
[0.0, 0.1, 0.0, 0.2]
[0.2, 0.1, 0.3, 0.2]
[-0.1, 0.0, 0.1, 0.0]
[0.3, 0.2, 0.4, 0.3]
[10.0, 10.2, 9.9, 10.1]
[9.8, 10.0, 10.1, 9.9]
[10.3, 10.1, 10.4, 10.2]
[9.7, 9.9, 9.8, 10.0]
```

The permanent gate must verify the fixture SHA before running. This pins the
schema, row-group layout, and values without generating a potentially
different Parquet encoding at test time.

## Registered real-CLI matrix

Every run uses:

```text
--input <fixture>
--output <unique-directory>
--column series
--method clara
--n-clusters 2
--sample-size 4
--n-samples 2
--seed 42
--device cpu
--name parity
--verbose
```

The three explicit configurations are:

| ID | Additional arguments | Registered total cost |
|---|---|---:|
| `f64-standard` | `--dtype float64 --variant standard` | `4.3999999999999986` |
| `f32-standard` | `--dtype float32 --variant standard` | `4.4000012278556824` |
| `f64-softdtw` | `--dtype float64 --variant softdtw --sdtw-gamma 0.7` | `-10.343994478252078` |

For each configuration, the resident run has no `--ram-limit`; the streaming
run adds exactly `--ram-limit 900`. The current ABI's conservative resident
estimates are 1,568 bytes for the Float64 runs and 2,016 bytes for the Float32
conversion-overlap path. The largest exhaustive four-of-eight sparse peaks are
687 and 591 bytes, respectively. Thus 900 bytes is the registered route
discriminator for all three configurations. It may not be relaxed after a
failure.

## Registered artifacts

Each invocation must emit exactly these three files and no dense
distance/silhouette artifact:

```text
parity_labels.csv
parity_medoids.csv
parity_checkpoint.bin
```

The expected label and medoid payloads, after normalizing only CRLF to LF, are:

```text
name,cluster
series_0,0
series_1,0
series_2,0
series_3,0
series_4,1
series_5,1
series_6,1
series_7,1
```

```text
cluster,medoid_index,medoid_name
0,1,series_1
1,7,series_7
```

The binary checkpoint size is exactly 72 bytes. On the registered Windows
artifact, the complete expected SHA-256 ledger is:

| Configuration | Labels | Medoids | Checkpoint |
|---|---|---|---|
| `f64-standard` | `39CCD0E9D5F520678899193B7A7903D6331217B0F99C96417F24BF943A7268EB` | `2CEF6784BE8808E00B6F78D3EF2DE8E55C6D4E7021F9C81092EB6A0279E3D22F` | `B666C1108C42A6B0705741B357F2527E319FEEBE3380BE6623E1494DAFD1A02C` |
| `f32-standard` | `39CCD0E9D5F520678899193B7A7903D6331217B0F99C96417F24BF943A7268EB` | `2CEF6784BE8808E00B6F78D3EF2DE8E55C6D4E7021F9C81092EB6A0279E3D22F` | `EEA65070341F900BA212909242AD54FE7A48F1E4A94704864CBF77C89FC28FC1` |
| `f64-softdtw` | `39CCD0E9D5F520678899193B7A7903D6331217B0F99C96417F24BF943A7268EB` | `2CEF6784BE8808E00B6F78D3EF2DE8E55C6D4E7021F9C81092EB6A0279E3D22F` | `67D818D58370CB6E17E4E8922175A343ADE2D53C7873A62522FE9D3C14734B79` |

The portable permanent contract is resident/stream byte identity within each
configuration, exact normalized CSV content, exact 72-byte checkpoint size,
and three distinct configuration checkpoint payloads. The Windows decisive
run additionally requires the exact hash ledger above.

### Pre-run decimal correction

Commit `1c96b3a` initially copied two rounded decimal renderings from the
handoff summary. Before any decisive F8 execution, direct decoding of the
already SHA-pinned historical checkpoints at the binary format's
`total_cost` offset 24 produced:

```text
build/phase8-f7-arrow/f64-v3-resident/parity_checkpoint.bin offset24_R=4.3999999999999986 G17=4.3999999999999986
build/phase8-f7-arrow/f32-v3-resident/parity_checkpoint.bin offset24_R=4.4000012278556824 G17=4.4000012278556824
build/phase8-f7-arrow/softdtw-v4-resident/parity_checkpoint.bin offset24_R=-10.343994478252078 G17=-10.343994478252078
```

The table above is corrected to those exact binary values. The fixture,
commands, 900-byte route discriminator, output bytes, and registered hashes
did not change. This is an evidence transcription correction, not post-run
tuning.

## Acceptance band

F8 passes only if all of the following hold:

1. The inherited tree is red for the named coverage gap: the tracked fixture
   and `test_fast_clara_parquet_parity` CTest target are both absent before the
   implementation.
2. The tracked fixture has the exact registered size and SHA-256.
3. The permanent test is added only when Parquet support and the real
   `dtwc_cl` target are present. Arrow-OFF builds gain neither a passing stub
   nor a capability skip.
4. All six real CLI processes exit exactly zero.
5. Every resident log contains `Data loaded from Parquet: 8 series` and omits
   `Parquet metadata selected streaming:`. Every streaming log contains
   `Parquet metadata selected streaming: 8 series` and omits
   `Data loaded from Parquet: 8 series`.
6. All 18 expected files exist, every invocation emits exactly three files,
   and all nine resident/stream artifact pairs are byte-identical.
7. Labels and medoids equal the registered normalized payloads; each
   checkpoint is exactly 72 bytes; the three configuration checkpoints are
   pairwise distinct. On Windows, all nine hashes equal the registered ledger.
8. The three resident and three streaming summaries report the registered
   configuration costs (full-precision verification may come from checkpoint
   decoding; the CLI's six-significant-digit display is not the arbiter).
9. The test prints an execution marker with `runs=6`, `parity=9/9`, and
   `configs=3/3 distinct`; CTest must show the target ran, with no skip message.
10. The complete Arrow-ON gate passes with `test_io_readers` and F8 executing
    rather than skipping. The canonical Arrow-OFF gate remains 114/114, zero
    failed, with exactly its six registered capability skips.
11. `git diff --check`, documentation/record/repository hygiene, and an
    independent adversarial review pass. PLAN/handoff closure is a separate
    documentation commit.

Any nonzero CLI exit, wrong route marker, missing/extra artifact, resident /
stream byte difference, stale configuration payload, or skipped F8 target is
**FALSIFIED**. There are at most two implementation repair attempts; the
fixture, 900-byte discriminator, seed, configuration matrix, and equality
requirements do not move after execution.

## Rollback and expected-risk claim

The implementation rollback will be the single F8 fixture/test commit. The
claim most expected to be wrong is that the current Arrow-ON build links a
fresh real CLI with all runtime DLLs discoverable; a historical hand-linked
executable is not acceptable evidence. If the environment cannot produce that
binary, record `[BLOCKED-ENV]` with the verbatim build/runtime probe and proceed
to the next PLAN item without weakening this band.
