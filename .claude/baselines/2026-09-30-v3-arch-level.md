# 2026-09-30 — V3: released wheels and archives target x86-64-v3

**Question:** with `-march=x86-64-v3` / `/arch:AVX2` instead of no flag (SSE2), (1) does the flag reach every
wheel and archive build, (2) does `cpp_conformance` stay digit-identical on each compiler, (3) how much faster is
the lanes fill (P1 expected about 2×)?

**Answer:** (1) yes, (2) yes on clang and MSVC, and on GCC the conformance rows hold but `unit_test_dtw_kernel_lanes`
fails under GCC's default FMA contraction (cause confirmed, below), (3) **no: about 1.0× unbanded, 1.17× banded**
[confirmed]; the 2× expectation is falsified for `double` lanes.

## Machine

Intel Core Ultra 9 285 (AVX2, FMA, no AVX-512), Windows 11, 127.5 GiB; clang 21.1.8 (MSVC ABI) and MSVC 19.50 (VS 18);
GCC 13.3 in WSL Ubuntu 24.04 (HiGHS, Gurobi, llfio OFF there). Load during the timed runs: mean 95 % (`typeperf`).
Base = design-2.0 `dbf26b9`; tip = pb/V3. "SSE2" = the base code with `-DDTWC_ENABLE_NATIVE_ARCH=OFF`, which is exactly what
a wheel and an archive were.

## Flag reaches the builds [confirmed]

- Wheel (`uv pip install --reinstall -v`, VS 18 generator, cl 14.50): configure prints `Architecture level: v3
  (/arch:AVX2)`; all 22 `CL.exe` lines in the log carry `/arch:AVX2`, the `dtw_lanes.cpp` one `/O2 /GL /arch:AVX2
  /fp:precise /fp:contract`.
- cibuildwheel's `cmake.args` replace `pyproject.toml`'s (scikit-build-core 0.12.2 settings reader, probed), but both
  list `-DDTWC_BUILD_PYTHON=ON`, so the `DTWC_BUILD_PYTHON` default reaches every wheel path. macOS wheels are arm64 only.
- Script-mode harness over the real block (x86-64, arm64 Linux, arm64 macOS, universal2, MSVC ARM64, sub-project,
  `v4` on arm64, bad value): v3 adds no flag off x86-64, `v4` off x86-64 and a bad value are configure errors.
- The native tree's `compile_commands.json` is byte-identical before and after the change.

## Conformance [confirmed]

`ctest -R cpp_conformance`, and `DTWC_CONFORMANCE_REGEN=1` then `diff --strip-trailing-cr` against the tracked reference
(17-digit scores, labels, medoids; 15 lines), restored afterwards.

| build | ctest | reference diff | FMA in the linked test exe |
| --- | --- | --- | --- |
| clang native (base) | pass | none | 3952 |
| clang x86-64-v3 | pass | none | 3637 |
| clang SSE2 (base) | pass | none | 0 |
| MSVC `/arch:AVX2 /fp:contract` | pass | none | 9 |
| MSVC SSE2 (base) | pass | none | 0 |
| GCC 13.3 x86-64-v3 (`-ffp-contract=fast`, its default in C++) | pass | none | 187 |

Serial ctest: base clang-win 123 = 120 pass + 3 skips (`test_cuda_correctness`, `test_metal_correctness`,
`test_metal_mmap`), 0 fail; clang v3 the same; MSVC AVX2 122 = 119 + 3 skips, 0 fail. Python from a fresh venv:
base 1191 passed / 19 skipped, tip 1191 / 19, 0 failed.

## GCC: lanes are not bitwise the per-pair kernels under FMA [confirmed]

`unit_test_dtw_kernel_lanes` (squared L2, float and double): x86-64-v3 with GCC's default contraction fails 16 of 175
assertions; the same build with `-ffp-contract=off` passes 175/175 and has 0 FMA; the base SSE2 build passes. Clang's
default (`on`, one expression) and MSVC `/fp:contract` pass at v3/AVX2. The Linux x86-64 wheel is a GCC build. Not fixed
here (FP flags are Volkan's). Also failing in that WSL tree at base, so not from this change: `test_deprecated_shims_warn`,
`test_error_taxonomy`.

## Lanes speed, single thread pinned to logical CPU 22 (a P-core) [confirmed]

`dtw_lanes.cpp.obj` packed ops: SSE2 `minpd`/`addpd`/`subpd`/`mulpd` 16/8/8/4 on xmm; v3 `vminpd`/`vaddpd`/`vsubpd`/`vmulpd`
16/8/8/6 on ymm; no library call in either, no FMA in the v3 lanes object.

`BM_fillDistanceMatrix`, `OMP_NUM_THREADS=1`, `start /affinity 400000`, 5 interleaved rounds, median of 3 repetitions,
CPU time, ratio SSE2 / tip per round, median [min–max]:

| subject | SSE2 ms | v3 ms | SSE2 / v3 | SSE2 / native |
| --- | --- | --- | --- | --- |
| 100 × 1000, unbanded | 1109 | 1078 | 0.99 [0.96–1.12] | 1.03 [0.99–1.03] |
| 50 × 1000, band 50 | 29.3 | 25.1 | 1.17 [1.15–1.21] | 1.19 [1.15–1.31] |

The pinned kernel alone (`p1_kernel_ab.cpp` without the deleted EAP call, built at `-march=x86-64`, `x86-64-v3`, `native`),
lanes ns/cell, L = 1000: unbanded 0.203 / 0.226 / 0.199 (x86-64), 0.203 / 0.212 / 0.208 (v3), 0.202 / 0.195 / 0.197
(native); band 50: 0.209 / 0.201 (x86-64), 0.178 / 0.173 (v3). Same ratios as the fill.

**Interpretation [inferred]:** W = 8 doubles is 4 independent xmm chains or 2 ymm chains, and each DP row step waits
for `min` then `add` of the row above (4 + 4 cycles). 8 lanes × 0.203 ns = 1.6 ns per row step, which is 8 cycles at
about 5 GHz (clock not measured). The kernel is bound by that dependency chain, not by vector width. Next decisive test:
W = 16 doubles, which keeps four ymm chains in flight.
