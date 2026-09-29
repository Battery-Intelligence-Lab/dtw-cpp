# 2026-09-29 — P1: SIMD lanes in the CPU fill

**Question:** with W = 64 / sizeof(T) equal-length pairs per call in SIMD lanes (`core::dtw_kernel_lanes`,
bitwise the per-pair kernels), is the 24-thread brute-force fill at least 2× faster than the per-pair path it
replaces (unbanded: the EAPruned kernel per pair; banded: `dtwBanded`)?

**Answer: PASS**, on a machine at 98–100 % load from other work (advisory): 8.2×, 3.1× and 15.3× on the three
subjects, every matrix bitwise identical. On the way it turned out that the lane loop the probe showed packed
was **not** packed in the shipped build: lld-link's LTO backend runs no SLP vectoriser, so under clang's ThinLTO
the loop linked as eight scalar chains. That form already passed the band (5.4×, 2.4×, 7.3×); compiling the
lane binding natively (`3c95dc7`) packs it and gives the numbers above. MSVC `cl` does not pack the loop at all.

## Band (registered 2026-09-29, before the first timed run)

Subjects, each measured base against tip:

1. `BM_fillDistanceMatrix/100/1000/-1` (24 threads, unbanded);
2. `BM_fillDistanceMatrix/50/1000/50` (24 threads, band 50);
3. the fill of one real UCR dataset of equal lengths: ECG5000 `ECG5000_TEST.tsv` (N = 4500, L = 140), unbanded
   (the default band), 24 threads, through `Problem::fill_distance_matrix`.

**PASS** if lanes are ≥ 2× faster on every one of the three; **FALSIFIED** otherwise, and then lanes stay only
where they win, with EAP's pruning rate on the losing data (cells computed / cells total) as the reason.
Statistic: the median over interleaved rounds (order alternating) of the paired per-round ratio
t_per-pair / t_lanes, with the lowest and highest round; ≥ 5 rounds. Precondition: the matrices are bitwise
identical (UCR: an FNV-1a hash of the packed upper triangle), else the timing is void. Reported alongside, no
band: the pinned single-thread kernel ratio on a P-core (8 × `BM_dtwFull_L/1000` / `BM_dtwLanes/1000/-1`, and
the banded pair). Wall-clock on this shared machine is advisory; the band is judged on the medians.

(The kernel ratio was in the end measured with `p1_kernel_ab.cpp`, below: `BM_dtwLanes` instantiates the kernel
in the ThinLTO benchmark TU, so on Windows it timed the unpacked form; it was not committed.)

## Machine (`scripts/machine_facts.py --build-dir build`)

| Fact | Value |
| --- | --- |
| OS | Windows 11 (AMD64) |
| CPU | Intel Core Ultra 9 285, 24 cores / 24 threads (8 P + 16 E), AVX2, no AVX-512 |
| Memory | 127.5 GiB |
| Compiler | Clang 21.1.8 (x86_64-pc-windows-msvc, MSVC STL 14.50), LLD 21.1.8 (lld-link) |
| Build | `build/` of the P1 worktree: Release, Ninja, `-O3 -march=native`, IPO (ThinLTO), `DTWC_FP_MODEL=fast`; HiGHS, Gurobi, llfio, benchmarks ON |
| Repository | pb/P1: base `f705329` (= pb/K1), tips `940cd8a` (run 1) and `3c95dc7` (run 2) |

Load [confirmed, `typeperf` total CPU, 5 s samples over each run]: run 1 median 100 % (quartiles 96–100, min 34);
run 2 min 98 %, median 100 %. Top processes: COMSOL, then four Wolfram kernels, MATLAB, other agents' builds.

## Commands

Sources in `2026-09-29-p1-lanes-fill/`. Base binary: `bench_dtw_baseline.exe` built at `f705329` and kept as
`bench_base_f705329.exe`; tip: the same target at the tip. The ECG5000 fill runs through `p1_fill_ucr.cpp`, linked
by `build_driver.sh` against the build's `dtwc++.lib`: the per-pair binary at `e34c37f` (fill code identical to
`f705329`, lane function bound but not called), the lanes binary at `940cd8a` (run 1) and `3c95dc7` (run 2).

```sh
bash run_band.sh                      # 9 rounds of both fills, 5 rounds of ECG5000 (2 fills per process, the warm one counts),
                                      # then kernel_pinned.bat: p1_kernel_ab.exe L band 11 on logical CPU 22 (/affinity 400000)
uv run --no-project python analyse_band.py <out dir>
```

`p1_kernel_ab.cpp` is compiled with the lane TU's own flags (`-O3 -march=native`, the FP subset, `-fno-lto`),
static CRT.

## Numbers [confirmed]

Fills, 24 threads; ms (Google Benchmark median of 5 repetitions per run); median [lowest–highest round].

| subject | run | per-pair | lanes | per-pair / lanes |
| --- | --- | --- | --- | --- |
| `BM_fillDistanceMatrix/100/1000/-1` | 1 (`940cd8a`, unpacked) | 1072.6 [964.5–1410.7] | 201.1 [178.8–263.1] | **5.42** [5.19–5.97] |
| `BM_fillDistanceMatrix/100/1000/-1` | 2 (`3c95dc7`, packed) | 2062.8 [1774.8–2406.4] | 245.5 [219.5–272.7] | **8.21** [7.76–10.04] |
| `BM_fillDistanceMatrix/50/1000/50` | 1 | 10.85 [10.47–14.86] | 4.563 [3.954–5.816] | **2.38** [1.93–3.69] |
| `BM_fillDistanceMatrix/50/1000/50` | 2 | 19.46 [18.58–23.38] | 6.253 [5.840–7.336] | **3.13** [2.77–3.29] |
| ECG5000 fill, wall (s) | 1 | 40.63 [35.07–42.01] | 5.173 [4.786–6.556] | **7.25** [6.41–8.19] |
| ECG5000 fill, wall (s) | 2 | 66.09 [58.29–71.35] | 3.911 [3.660–5.339] | **15.32** [12.41–18.05] |
| ECG5000 fill, process CPU (s) | 2 | 700.7 [697.1–709.3] | 46.3 [45.8–47.7] | 15.15 [14.73–15.39] |

The ECG5000 matrix hashes to `bf0331039340190b` in all 20 fills of both runs, per-pair and lanes alike. The
per-pair fills slowed from run 1 to run 2 with the machine; the paired ratios are what the band reads.

Pinned single-thread kernel, run 2 (`p1_kernel_ab`, CPU 22, 11 interleaved rounds, the benchmark's uniform
[−1, 1) series), ns per DP cell, median; every lane bitwise equal to both per-pair kernels:

| L, band | per-pair of the fill (EAP / `dtwBanded`) | `dtwFull_L` / `dtwBanded` | lanes | ratio to per-pair | ratio to linear/banded |
| --- | --- | --- | --- | --- | --- |
| 1000, full | 6.395 | 1.940 | 0.261 | 26.3 [15.7–33.5] | **8.2** [5.0–12.7] |
| 1000, 100 | 2.509 | 2.550 | 0.250 | 10.9 [4.2–13.2] | **9.3** [4.4–21.3] |
| 140, full | 7.557 | 1.839 | 0.203 | 35.5 [23.5–42.3] | **8.6** [4.4–10.2] |

Under this load the scalar kernels ran 1.84–1.94 ns/cell against K1's quiet 1.36, so the ratios are advisory
[inferred: the load inflates them]. The unpacked lanes of run 1 measured 0.36 ns/cell (`BM_dtwLanes/1000/-1`,
2.89–2.96 ms per 8 pairs) where the PF-5 probe, run beside it on the same core, gave 0.170–0.177 for its W = 8
lanes; the library kernel compiled without LTO matched the probe (0.18 ns/cell, A/B in one binary).

## Why run 1 was unpacked [confirmed]

- `llvm-objdump` of the linked ThinLTO benchmark: the `dtw_kernel_lanes<double, L1Dist, StandardCell>` body has
  0 `vminpd` and 32 `vminsd` (one spill reload). The non-LTO compile of the same TU packs it (two ymm chains).
- `/lldsavetemps`: the post-import bitcode (`*.3.import.bc`) compiled by clang's own `-O2`/`-O3` pipeline gives
  24 `vminpd`; lld-link's backend (`/opt:lldlto=2` or `=3`) gives none, and no SLP remark. The loop vectoriser does
  run there (16 "vectorized loop" remarks).
- A four-statement `__restrict` SLP probe (`a[k] = b[k] + c[k]`) packs without LTO (`vaddpd`) and stays scalar under
  lld-link ThinLTO. So the missing pass is the linker's pipeline, not the kernel [inferred cause: the pipeline
  tuning options leave SLP off in lld's COFF LTO configuration; not read in LLVM's source].
- `#pragma clang loop unroll(disable)` gets the loop vectoriser to pack it under LTO, but `diag`/`left` then live on
  the stack and every lane step pays a store-to-load round trip; `vectorize(enable)` alone changes nothing.

Fix (`3c95dc7`): `resolve_dtw_block_fn` lives in `core/dtw_lanes.cpp`, compiled with `-fno-lto` when clang targets
the MSVC ABI. Its object is native COFF (24 `vminpd`, 16 `vminps` over the four instantiations); every other TU,
`dtw_dispatch.cpp` included, is still ThinLTO bitcode.

## The lane loop as shipped [confirmed, `llvm-objdump -d -M intel build/bin/CMakeFiles/dtwc++.dir/core/dtw_lanes.cpp.obj`]

`dtw_kernel_lanes<double, L1Dist, StandardCell>`, the i-loop unrolled ×2: two ymm chains (lanes 0–3 in ymm7,
lanes 4–7 carried through ymm6/ymm8), `x[i]` broadcast, no call, no spill (the abs mask is rematerialised):

```asm
1230: vbroadcastsd ymm9, qword ptr [rbx + 8*rsi]   ; x[i]
      vsubpd       ymm10, ymm9, ymm3               ; x[i] - y[j][0..3]
      vbroadcastsd ymm11, qword ptr [rip]          ; abs mask
      vandpd       ymm10, ymm10, ymm11             ; |.|
      vmovupd      ymm12, ymmword ptr [rbp - 0x60] ; up = dp[i, j-1][0..3]
      vmovupd      ymm13, ymmword ptr [rbp - 0x40] ; up [4..7]
      vminpd       ymm14, ymm12, ymm5              ; min(diag, up)
      vmovupd      ymm5, ymmword ptr [rbp - 0x20]
      vminpd       ymm6, ymm13, ymm6
      vmovupd      ymm15, ymmword ptr [rbp]
      vminpd       ymm7, ymm7, ymm14               ; min(., left), left carried
      vaddpd       ymm7, ymm10, ymm7               ; + cost
      vmovupd      ymmword ptr [rbp - 0x60], ymm7  ; dp[i, j][0..3]
      vsubpd       ymm9, ymm9, ymm4
      vandpd       ymm9, ymm9, ymm11
      vminpd       ymm6, ymm8, ymm6
      vaddpd       ymm6, ymm9, ymm6
      vmovupd      ymmword ptr [rbp - 0x40], ymm6  ; dp[i, j][4..7]
      ...                                          ; the same for i + 1
      add rsi, 2 / sub rbp, -0x80 / vmovapd ymm6, ymm15 / cmp r13, rsi / jne 1230
```

Squared L2 is the same with `vmulpd ymm, ymm, ymm` in place of the abs, and the product added by a separate
`vaddpd` (no FMA, as in the per-pair kernel); float runs 16 lanes as two ymm chains of `vminps`/`vaddps`. The
per-pair kernels are untouched; `test_codegen_no_calls` covers the lane kernel too (`inner_loops=160 calls=0`).

## MSVC cl [confirmed]

`build-msvc` (MSVC 19.50.35723, Ninja, Release: `/O2 /Ob2 /arch:AVX2 /fp:precise /fp:contract /GL`; CUDA, Arrow,
LLFIO, HiGHS and Gurobi OFF): `unit_test_dtw_kernel_lanes` passes, 175 assertions in 3 test cases, so under
`/fp:contract` too every lane is bitwise the per-pair kernel. The `/FA` listing of `dtw_lanes.cpp` (its cl command
without `/GL`) has **no packed op**: 0 `vminpd`/`vminps`, 49 `vminsd`, 97 `vminss`. The lane loop is unrolled into
eight scalar chains, `left` in registers, but each lane goes through `std::min`'s reference with a `cmovbe` on two
stack addresses, as K1 saw; `/Qvec-report:2` says `loop not vectorized due to reason '1200'` (loop-carried data
dependence) for `dtw_kernel.hpp(606)`, the lane loop. The eight chains still overlap (ILP), but cl gets no SIMD.

## Bit identity [confirmed]

- `unit_test_dtw_kernel_lanes`: every lane against `dtwBanded` (and `dtwFull_eap` unbanded) for double and float,
  L ∈ {1, 2, 7, 100, 1000}, band ∈ {full, 0, 1, L/10}, L1 and squared L2, random walks and tied integers; the
  whole filled matrix against the per-pair function over every pair for equal lengths, float32, all-mixed and
  two-length data, and a partially computed matrix whose known entries must survive. Mutants: dropping `left`
  from the lane min fails 64 of 168 kernel assertions; writing lane results to swapped slots fails 6 of the 7
  fill cases (all but all-mixed, which runs no lanes); dropping the equal-length check fails the two mixed cases.
- `cpp_conformance` passes with the reference untouched; its data are 27 series of one length (band 3), so rows
  0–18 of its fill take lane blocks [inferred from the data and the fill's code path].
- The ECG5000 hash above.

## Observations

- The unbanded fill's per-pair kernel, EAPruned, costs 6.4–7.6 ns per cell on these series against 1.8–1.9 for
  `dtwFull_L` (table above): on uniform noise its data-dependent branches cost more than its pruning saves
  [inferred]. The per-pair tail of each row still runs it.
- With W = 8 the per-pair tail is 7 % of the pairs at N = 100 and 14 % at N = 50; it is most of what separates
  run 2's 3.1× (band 50, N = 50) from the kernel's 9× [inferred]. Padding the tail with a repeated column, as the
  probe did, would remove it; the brief keeps it per pair.
- The codegen gate (`scripts/codegen_report.py`) compiles without LTO, so it cannot see this class of loss.
