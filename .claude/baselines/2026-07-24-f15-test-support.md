# F15 deterministic test-support registration and baseline — 2026-07-24

## Scope and base

- Base: `e79fab31a3807fe3e2a7a83c60a455b4da91b880`
  (`docs: close F14 CSV wire contract`).
- `git status --short` and `git diff --check` produced no output.
- Historical claim source:
  `.claude/summaries/handoff-2026-06-01-adversarial-audit.md:36`.
- R1 falsification:
  `.claude/baselines/2026-07-23-r1-todo-reconciliation.md:178,228`.
- Preflight source and raw artifacts:
  `build/f15-generator-preflight-e79fab3/`.
- Subject: deterministic C++ test/benchmark data construction and
  production-backed dense symmetric CPU-reference traversal. No production
  algorithm, public API, file format, or benchmark timing is changed.

The historical statement was:

```text
bench random_series + cpu oracle byte-identical across 5 bench + 3 test files -> belong in test_util.hpp.
```

Verdict: **FALSIFIED [confirmed]**. The five benchmark bodies and three
accelerator-test bodies are two internally exact but mutually different
families. The phrase also conflates dense matrices, packed LB values, and an
MPI timing loop.

## Confirmed inventory

### Exact generator families eligible for sharing

The five benchmark-local scalar bodies have whitespace-normalized SHA-256
`662469A708AB21DFB0AFF9302D1409EDA2318364C803615E43184B0A57C51EAD`.
Each constructs a fresh `std::mt19937(seed)`, draws
`uniform_real_distribution<double>(-1,1)`, and returns one
`vector<double>(length)`:

1. `benchmarks/bench_cuda_dtw.cpp:29-37`
2. `benchmarks/bench_dtw_baseline.cpp:27-35`
3. `benchmarks/bench_metal_dtw.cpp:24-31`
4. `benchmarks/bench_mmap_access.cpp:33-41`
5. `benchmarks/bench_mpi_dtw.cpp:37-45`

Their wrappers intentionally retain different default base seeds: 100, 200,
or 300. Every row uses `base_seed+i`, which constructs a fresh engine per
row. `tests/unit/unit_test_mpi.cpp:45-53` is a sixth semantically identical
scalar body and is included in the extraction even though the old count
missed it.

The three accelerator-test matrix bodies have whitespace-normalized SHA-256
`74FE0D116846930DDD0CF9CA945DF6640F1EA4D1B781670C46ACA1366708C8CD`.
Each constructs one `std::mt19937(seed)`, draws
`uniform_real_distribution<double>(-10,10)`, and fills a row-major
`vector<vector<double>>(N, vector<double>(L))` continuously across row
boundaries:

1. `tests/unit/test_cuda_correctness.cpp:44-57`
2. `tests/unit/test_cuda_lb_keogh.cpp:47-60`
3. `tests/unit/test_metal_correctness.cpp:44-57`

Resetting the engine per row is not equivalent to this family.

### Intentional generator variants retained separately

These are inventory, not consolidation targets:

| Path | Range/type/shape distinction |
|---|---|
| `tests/unit/test_metal_lb_keogh.cpp:41-52` | continuous `N*L` double, `[-5,5]` |
| `tests/unit/test_metal_mmap.cpp:33-43` | continuous `N*L` double, `[-1,1]` |
| `benchmarks/bench_f32_vs_f64.cpp:20-30` | continuous `N*L`, `[0,10]`, distribution type is `T` |
| `tests/unit/core/unit_test_mmap_data_store.cpp:42-55` | continuous `N*L` double plus `Data` names |
| `tests/unit/unit_test_accuracy.cpp:33-40` | fresh scalar double, `[-10,10]` |
| `tests/unit/unit_test_simd.cpp:109-117` | fresh scalar double, `[-10,10]`, independent SIMD tests |
| adversarial/property helpers | caller-owned engines and configurable ranges; draw-stream continuity is fixture semantics |
| random-walk/Gaussian/NaN helpers | distribution cache and conditional draw order are fixture semantics |
| Python/MATLAB generators | PCG64 or MATLAB RNG domains; never C++ byte-equivalent |

`tests/test_util.hpp:23-40` is deliberately not the destination. Its existing
`get_random_data` uses mutable global `randGenerator`, ragged lengths,
integer-valued draws, and no invocation-local seed.

### CPU references and independent oracles

The full dense CPU-reference bodies in CUDA correctness, Metal correctness,
and Metal LB have normalized SHA-256
`6FEF88740D6F9F44E082075FF9B47990FC2113F09D9BADCAE7D1C1F7EC7B46EA`.
They create a row-major `N*N` double matrix, leave the diagonal zero, compute
only `i<j` with production `dtwFull_L<double>`, and mirror each result.

The CUDA and Metal banded bodies have normalized SHA-256
`8370D535A0EF9594394A5A2F2152415844B6FC636B0E601E4C0B2E2793097F72`
and use the same traversal with production `dtwBanded<double>`.
`test_cuda_kernel_override.cpp:118-130` and the MPI correctness loops have the
same dense traversal semantics and are included.

These are backend references, not independent DTW mathematics. The shared
utility may own zero-diagonal upper-triangle/mirror traversal only; callers
continue to supply the distance function.

The following remain separate:

- CUDA LB's packed `N(N-1)/2` envelope/reference order;
- the MPI benchmark's timing-only loop, which discards distances;
- query-row and `K*N` cross layouts;
- `gpu_fixed_band_oracle.hpp` full-matrix DP and path enumeration;
- canonical band, ADTW, barycenter, SIMD, p-median, FastPAM, TADPole, and
  external literal arbiters whose independence is load-bearing.

No generator-specific killed idea exists. SIMD implementation remains killed;
this work does not reopen it.

## Independent byte and value preflight

The preflight implements the inherited bodies literally, writes raw arithmetic
types in row-major order, and computes L1 DTW through a full `(n+1)*(m+1)`
matrix with explicit `|i-j|<=band`. It also calls production
`dtwFull_L`/`dtwBanded` as a separate rolling-buffer computation.

### Windows-MSVC-STL shared fingerprints

Both compiler profiles produced:

```text
benchmark-seed42.bin bytes=40 sha256=70CC88A06F050E253AF62FF5A73D2AE2D659A287A286D88EEB1871C7C45E13EB
benchmark-rows-3x4-base100.bin bytes=96 sha256=B930ACBCF449361A1370E185F5610D9B4BDEBD71FAC07772C3E3449AB29C1733
mmap-3x4-seed42.bin bytes=96 sha256=46E1D8BD4EBBC2E08F5016A93619DF70ADBA6DE3ACF67488BA6F375EB5F0A171
f32-3x4-seed42.bin bytes=48 sha256=174D5B8D0735D566CD2048BEFBE665BE7F7D73CD84B3B19E38D2EBC352D55C2F
f64-3x4-seed42.bin bytes=96 sha256=7536D3D489A5ECEC8A4FE382C59FE51B1DC9A23C0A3E0CC7CA4CEB97EA5B9EA2
```

The five benchmark seed-42 binary64 masks are:

```text
3FE2FA8F6CD7F876,BFE4429ABC432780,3FE1E67511EED8FC,3FC8CB2C149941A8,BFBBBBCF0DB01E60
```

### Clang Release relaxed profile

Compiler: clang 21.1.8 with the repository Release flags, including
`-freciprocal-math -fassociative-math -fno-signed-zeros`.

```text
accelerator-3x4-seed42.bin bytes=96 sha256=BFACE25683F745B965459B36DDA75C9C4EDA8035EF4AFBED4722BC2E1F34758B
metal-lb-3x4-seed42.bin bytes=96 sha256=90C0B86BA19710967CF936D15EB877C6A58E8E910EB2D8146C8B2495460B2C00
full-oracle-3x3.bin bytes=72 sha256=700163C5AC813BABD00845295CEEC6D3FD2C8DD13CEE7A918B442EE2440549B3
band0-oracle-3x3.bin bytes=72 sha256=494C9E299092EED43414A2ABEA110CC4A05D579F3B8162257D4D637F45C933EB
production_full_equal=1
production_band0_equal=1
```

Accelerator binary64 masks:

```text
4017B933480DF694,C01953416B53F160,40166012566A8F3B,3FFEFDF719BF9212,BFF15561688E12FC,C0200041BE89C29E,BFEA14A986DAD398,C00A9B4B96F5696D,C01C92166F9FE92C,4008246451B14AA3,C021BE586FDF2D38,4011C288EB7D97C0
```

Full and band-0 matrices:

```text
full_bits=0000000000000000,40345AE633486486,40362F13B6D21132,40345AE633486486,0000000000000000,40380D70B39D2896,40362F13B6D21132,40380D70B39D2896,0000000000000000
band0_bits=0000000000000000,40345AE633486486,4043B21174E2226B,40345AE633486486,0000000000000000,40407337C4287B6E,4043B21174E2226B,40407337C4287B6E,0000000000000000
```

### MSVC Release precise profile

Compiler: MSVC 19.50.35723 with `/O2 /fp:precise /fp:contract`.
Clang without the repository's relaxed Release flags produced this same
profile.

```text
accelerator-3x4-seed42.bin bytes=96 sha256=C53236B99DC783C3AC0129B58652C25C95ED72F7922A39B369A9E89AF94824B0
metal-lb-3x4-seed42.bin bytes=96 sha256=6BA2FF1570AF16CE0752E1EFE401F17196774785CAC8B11A6A6C7FB054175CD0
full-oracle-3x3.bin bytes=72 sha256=E79B3ACEAF951A614278879049CA66B887AEC388754DA9005308897ACB12526C
band0-oracle-3x3.bin bytes=72 sha256=2405A565FEEBAF9998D3A1A778105911EC8DEA996A6EE5EFC58A9997D8432ABD
production_full_equal=1
production_band0_equal=1
```

Accelerator binary64 masks:

```text
4017B933480DF694,C01953416B53F160,40166012566A8F3C,3FFEFDF719BF9210,BFF15561688E1300,C0200041BE89C29E,BFEA14A986DAD3A0,C00A9B4B96F5696C,C01C92166F9FE92C,4008246451B14AA4,C021BE586FDF2D38,4011C288EB7D97C0
```

Full and band-0 matrices:

```text
full_bits=0000000000000000,40345AE633486486,40362F13B6D21132,40345AE633486486,0000000000000000,40380D70B39D2898,40362F13B6D21132,40380D70B39D2898,0000000000000000
band0_bits=0000000000000000,40345AE633486486,4043B21174E2226C,40345AE633486486,0000000000000000,40407337C4287B6D,4043B21174E2226C,40407337C4287B6D,0000000000000000
```

### Linux libstdc++ Release profile

A pre-run portability review identified that the C++ standard does not freeze
`uniform_real_distribution`'s engine-to-real mapping. The literal preflight
was therefore compiled inside WSL Ubuntu 24.04, still writing only beneath the
repository's ignored `build/f15-generator-preflight-e79fab3/` directory:

```text
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Ubuntu clang version 18.1.3 (1ubuntu1)
```

Both compilers used libstdc++ and the repository's Release relaxations:

```text
-O3 -DNDEBUG -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math -fno-signaling-nans
```

GCC 13.3 and Clang 18.1 produced the same complete profile:

```text
benchmark-seed42.bin bytes=40 sha256=194FB0E76C52FCD84F09960547EEDC6A43788FDCC89F739DF44E3C49AF7B16E0
benchmark-rows-3x4-base100.bin bytes=96 sha256=1F9E6847BA0FFC7943EBCA024827CD6B5A890B8911BFF4F6C59211F3C28892AB
accelerator-3x4-seed42.bin bytes=96 sha256=5D3594B036ED60CAA686D8472C630488B86290BAD1805806FBB38894B96F2C53
full-oracle-3x3.bin bytes=72 sha256=7DE312EABCFFB71D857BF97B9CFCE9C08A6F855E7CE25B863342BE46CBC38B73
band0-oracle-3x3.bin bytes=72 sha256=E81034CE0654315D254D07FA518DDCE7472A542A4EECBD740935E9DFF891022E
production_full_equal=1
production_band0_equal=1
```

Benchmark masks:

```text
3FE2FA8F6CD7F878,BFE4429ABC43277F,3FE1E67511EED8FE,3FC8CB2C149941B0,BFBBBBCF0DB01E50
```

Accelerator masks:

```text
4017B933480DF696,C01953416B53F15F,40166012566A8F3E,3FFEFDF719BF9220,BFF15561688E12F0,C0200041BE89C29E,BFEA14A986DAD3A0,C00A9B4B96F5696C,C01C92166F9FE92A,4008246451B14AA8,C021BE586FDF2D37,4011C288EB7D97C0
```

Full and band-0 matrices:

```text
full_bits=0000000000000000,40345AE633486488,40362F13B6D21132,40345AE633486488,0000000000000000,40380D70B39D2896,40362F13B6D21132,40380D70B39D2896,0000000000000000
band0_bits=0000000000000000,40345AE633486488,4043B21174E2226B,40345AE633486488,0000000000000000,40407337C4287B6E,4043B21174E2226B,40407337C4287B6E,0000000000000000
```

The band-0 result differs from the full result in all three profiles, so the
fixture is non-degenerate. A decisive test must accept one complete coherent
Windows-relaxed, Windows-precise, or Linux-libstdc++ profile only; mixing
generator or oracle values between rows fails.

## Inherited executable baseline

Six benchmark targets built successfully in
`build/baseline-2026-07-06`. Their executable sizes were:

```text
bench_cuda_dtw.exe      363520
bench_dtw_baseline.exe 7492096
bench_f32_vs_f64.exe    393216
bench_metal_dtw.exe    7471104
bench_mmap_access.exe  7550464
bench_mpi_dtw.exe        27648
```

Registration inventories were:

```text
F15_BENCH_LIST target=bench_dtw_baseline exit=0 registrations=72
F15_BENCH_LIST target=bench_cuda_dtw exit=0 registrations=1
F15_BENCH_LIST target=bench_metal_dtw exit=0 registrations=14
F15_BENCH_LIST target=bench_mmap_access exit=0 registrations=14
F15_BENCH_LIST target=bench_f32_vs_f64 exit=0 registrations=10
MPI not available in this build.
F15_BENCH_LIST target=bench_mpi_dtw exit=1 capability_unavailable=1
```

The rebuilt local RTX 4000 Ada CUDA subjects ran for real:

```text
All tests passed (7827 assertions in 61 test cases)
All tests passed (688 assertions in 8 test cases)
F15_CUDA_BASELINE correctness_exit=0 lb_exit=0
```

The CPU-focused subjects reported:

```text
All tests passed (283 assertions in 39 test cases)
All tests passed (7029 assertions in 16 test cases)
MPI not enabled (DTWC_ENABLE_MPI=OFF). Skipping MPI tests.
F15_CPU_BASELINE accuracy_exit=0 simd_exit=0 mpi_exit=0 mpi_capability_unavailable=1
```

The first CUDA build invocation was **INVALID**: CMake regenerated, then the
parent PowerShell environment invoked `cl.exe` without MSVC standard-library
paths and failed before the subject:

```text
fatal error C1083: Cannot open include file: 'cstddef': No such file or directory
```

The identical targets built inside `vcvars64` and produced the passing CUDA
outputs above.

The first CPU wrapper was also **INVALID** despite green subjects because it
expected the custom MPI stub to return 4. The executable actually returns 0
after the exact skip text quoted above. The corrected wrapper requires both
exit 0 and that exact capability-unavailable message.

## Registered implementation boundary

Create a self-contained test-only header outside the incompatible legacy
`tests/test_util.hpp` with exactly:

1. fixed-range scalar benchmark-series generation;
2. benchmark row-set generation using fresh `base_seed+i` engines;
3. fixed-range accelerator `N*L` generation with one continuous engine;
4. generic zero-diagonal upper-triangle/mirror matrix assembly accepting a
   caller-supplied distance function.

Use it from the five historical benchmark files, the MPI scalar test, the
three historical accelerator tests, the exact full/banded CUDA/Metal/Metal-LB
reference wrappers, CUDA override dense reference, and MPI dense-reference
checks. Keep every intentional variant and independent oracle named above
local.

The permanent non-skipping unit target must:

- compare normalized raw IEEE bytes to the exact fingerprints above;
- identify exactly `relaxed`, `precise`, or `libstdcxx` and reject any mixed
  profile;
- prove scalar, per-row, and continuous schedules separately;
- compare production full/banded dense references digit-for-digit with an
  independent full-matrix oracle on the non-degenerate fixture;
- assert exact dense size, zero diagonal, upper-triangle call count, and
  mirrored indices;
- read the named source consumers and reject reintroduced local bodies or
  bypasses;
- print
  `F15_TEST_SUPPORT profile=<relaxed|precise> scalar=ran row_seeded=ran continuous=ran dense=ran source_audit=ran skips=0`.

It must run without CUDA, Metal, MPI, llfio, Arrow, HiGHS, or Gurobi.

## Registered mutation matrix

Each execution changes one subject, rebuilds only the focused target or named
consumer, requires nonzero test/audit exit, and restores the exact base before
the next:

1. benchmark scalar lower endpoint `-1` to `0`;
2. scalar seed `seed` to `seed+1`;
3. row-set seed schedule `base+i` to constant `base`;
4. accelerator upper endpoint `10` to `5`;
5. accelerator continuous engine to one fresh engine per row;
6. accelerator row-major draw loop to column-major draw order;
7. dense traversal omits the mirrored write;
8. dense traversal evaluates the diagonal;
9. dense traversal returns packed upper-triangle storage;
10. CUDA full reference calls banded band 0;
11. CUDA banded reference ignores its band and calls full DTW;
12. one benchmark restores and uses a local scalar generator body;
13. one accelerator test restores and uses a local matrix generator body.

M01-M09 run the permanent focused target. M10-M11 run the real CUDA
correctness target and must fail its existing CPU/GPU comparison. M12-M13 run
the focused source audit. All 13/13 must be killed.

## Acceptance band

F15 passes only if:

1. implementation uses at most two attempts without relaxing any fingerprint,
   source, mutation, or executable band;
2. the focused target has zero skips, prints the exact marker, and passes at
   least 35 assertions in at least 5 cases;
3. all registered generator and oracle hashes match one coherent supported
   profile, and independent/production matrices are digit-identical;
4. source audit finds no historical local generator bodies, all named
   consumers reach the shared header, and independent/intentional variants
   remain local;
5. all six benchmark targets rebuild; registration counts do not fall below
   72/1/14/14/10 for baseline/CUDA/Metal/mmap/f32, while MPI-OFF retains its
   exact explicit message and exit 1;
6. real CUDA correctness remains at least 7,827 assertions / 61 cases and
   CUDA LB remains at least 688 / 8, with no skip;
7. CPU accuracy/SIMD remain at least 283/39 and 7,029/16; MPI-OFF prints its
   exact capability message rather than claiming execution;
8. all 13 mutation executions fail;
9. the live supply-chain checker remains 39/39 actions, 7/7 archives, one
   Arrow pin, and 27 tracked CMake manifests;
10. fresh full gates pass at the post-F15 inventories: canonical 119/119 with
    six capability skips, llfio-OFF 119/119 with nine, and Arrow 121/121 with
    eight. The F15 target must execute in all three;
11. `git diff --check` is clean and only the registered test/benchmark support
    boundary changed.

Real Metal is unavailable on this Windows host. F15 changes no Metal backend;
the shared generator/reference utilities execute in the non-optional focused
target, while Metal call-site reachability is source- and compile-audited.
No real-Metal runtime claim is made.

Rollback is a local revert of the dedicated F15 implementation commit and its
separate registration/closure docs commits. No push, tag, publication, SSH, or
HPC action is authorized.

The claim most likely to be wrong is completeness of the named consolidation
boundary. The inventory above deliberately distinguishes exact duplicates
from purpose-specific generators and independent arbiters; the final source
audit must repeat it before closure.
