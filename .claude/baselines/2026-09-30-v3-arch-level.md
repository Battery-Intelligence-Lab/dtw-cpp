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

## V4: GCC after the tests follow the FP ruling (2026-10-01, pb/V4 on V3's tip 41a1be1) [confirmed]

Volkan, 2026-10-01: no bit-for-bit equivalence between compilers; last-bits differences are fine while the clustering is
unchanged. No FP flag changed; contraction stays on (GCC default, clang `on`, MSVC `/fp:contract`).

- **Tree:** V3's GCC tree (`~/dtwc_v3/build-gcc-v3` in WSL Ubuntu 24.04, g++ 13.3, x86-64-v3, no `-ffp-contract` flag,
  HiGHS, Gurobi, llfio OFF), source synced from the pb/V4 worktree. The linked `unit_test_dtw_kernel_lanes` holds 190
  `vfmadd`/`vfnmadd`/`vfmsub` instructions, so contraction is in effect.
- **Bound:** `tests/support/dtw_route_bound.hpp`, `|a - b| <= 2 * (nx + ny - 1) * eps(T) * max(|a|, |b|)`. On this tree over the
  configurations of the test x 20 seeds (scratch program `C:/D/git/wt/V4-lanes-ratio.cpp`): lanes vs `dtwBanded`, double,
  L1 0 of 6400 pairs differ, squared L2 1 of 6400; float, L1 0 of 12800, squared L2 1489 of 12800; worst
  `|a - b| / ((2n - 1) * eps * max)` 0.0043 (double), 0.33 (float), against the bound's 2. With `-ffp-contract=off` nothing
  differs. A per-pair oracle with a band one wider fails 67 of 175 assertions, so the bound still bites.
- **GCC serial ctest, base (V3 log `V3-gcc-v3-ctest.log`, not re-run):** 122 tests, 3 failed (`test_deprecated_shims_warn`,
  `unit_test_dtw_kernel_lanes`, `test_error_taxonomy`), 3 skipped. **Head:** 122 = 119 passed + 3 skipped
  (`test_cuda_correctness`, `test_metal_correctness`, `test_metal_mmap`), 0 failed (`C:/D/git/wt/V4-gcc-head-ctest.log`).
  `cpp_conformance` passes; `DTWC_CONFORMANCE_REGEN=1` then `diff --strip-trailing-cr` against the tracked reference shows no
  difference (scores to 17 digits, labels, medoids; 15 lines), restored afterwards.
- **clang-win serial ctest:** base 123 = 120 passed + 3 skipped, head the same, 0 failed.
- **`test_deprecated_shims_warn`:** GCC did warn for all six shims; the script's patterns match ASCII `'`, and GCC quotes with
  U+2018/U+2019 in a UTF-8 locale (`LANG=en_US.UTF-8`: bytes `e2 80 98`; `LC_ALL=C`: `'`). The script now sets `LC_ALL=C`.
- **`test_error_taxonomy`:** the row "a checkpoint root that is a symlink" is compiled off Windows only. W5d (`6a96642`) deleted
  the symlink rejection with the generation directories it protected; `save_checkpoint` through a symlinked root now
  succeeds and writes into the target (probe `C:/D/git/wt/V4-symlink-probe.cpp`). The row asserted removed behaviour and was removed;
  rejecting a symlinked root again would be a product change.
- **After the sync with design-2.0 (73e7c12: V3, P5, E1) [confirmed]:** P5's `unit_test_one_batch_pam` case compared the lanes table's
  `total_cost` bit for bit with the per-pair scan. On the GCC tree (x86-64-v3, default contraction, 220 FMA instructions in the
  linked test) the original check failed 6 assertions, all "float32, equal lengths, band 4, squared L2" (labels equal in every
  case); it now uses `dtw_routes_agree` and passes (10638 assertions). The L1 `==` checks and the original
  `unit_test_distance_matrix_properties` A/B check passed on GCC unchanged (L1 has no multiply to contract); they are moved onto
  the bound anyway because the two sides are different code. GCC full serial ctest 113 = 110 passed + 3 skipped
  (`test_cuda_correctness`, `test_metal_correctness`, `test_metal_mmap`), 0 failed; clang-win 114 = 111 + 3, 0 failed
  (`test_codegen_no_calls` is clang-only). `cpp_conformance` passes on both and `DTWC_CONFORMANCE_REGEN=1` reproduces the tracked
  reference with no difference on both (restored afterwards). `unit_test_variant_distmat` runs 3 of 12 cases on GCC (llfio OFF).
