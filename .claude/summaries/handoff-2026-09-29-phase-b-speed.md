# Handoff — 2026-09-29 — phase B continued; the CPU kernel and fill made fast (Windows)

## Base

Branch `design-2.0`, HEAD `830568a`, tree clean (untracked build dirs only). Session base `4dcc39d`. Volkan's
instruction this session is in CHARTER (2026-09-29): no unnecessary abstraction; fast code with decent assembly, SIMD
where needed. Worktrees `C:/D/git/wt/<id>` on branches `pb/<id>`; briefs for agents in the session scratchpad
(`brief_common.md`, `brief_integrator.md`, `gates_*.md`, `brief_*.md`), not in the repo.

## Done (merged into design-2.0, one gated merge each)

- K1 `4441969`: no DP cell makes a library call. `std::min({…})` was `__std_min_d` on the MSVC STL; nested min plus a
  register-carried `dp[i-1, j]`; `test_codegen_no_calls` guards it (`baselines/2026-09-29-k1-dp-cell-no-call.md`).
- Y2 `959dc5b`: W5d — one `core::DistanceMatrix` (heap or mapped `.dtwm`), header-only llfio in one .cpp, typed
  checkpoint outcomes; wheels and release archives ship llfio ON. Fable review: merge, no confirmed defect.
- Y3 `b260415`: W6a (`index_t`), W6c (`run_openmp` per-thread failure slots, `parse_ram_limit`, `GpuPrecision`),
  W6d (v1 `cluster_by_kMedoidsPAM` shim back, `dtwc_main` and non-v1 forwarders gone).
- P1 `d61c499`: the CPU fill runs W = 64/sizeof(T) equal-length pairs per call in SIMD lanes, bitwise equal to the
  per-pair kernels (`baselines/2026-09-29-p1-lanes-fill.md`); `core/dtw_lanes.cpp` is `-fno-lto` under clang/MSVC ABI.
- Records: Windows CPU and GPU baselines, PF-5 probe (PASS), P2 (EAP vs linear), W4a (CUDA kernel A/B) — all under
  `.claude/baselines/2026-09-29-*`; DECISIONS §1/§3, LESSONS, MAP refreshed.

## Verified by me

- `4dcc39d`, pinned P-core 12: `BM_dtwFull_L/1000` 7.225 ms (7.2 ns/cell), fill `100/1000/-1` 1402 ms; the inner loop
  of `dtw_kernel_linear<StandardCell>` (build flags, `-S`) calls `__std_min_d` per cell. My probe: nested min + carry
  1.3 ns/cell, bitwise identical (`baselines/2026-09-29-windows-kernel/min_carry_probe.cpp`).
- GPU fill baseline from `build/cuda-verify-0928` (FP32 via Auto): 121 Gcell/s at N 100, L 1000.
- Read: K1's kernel diff (the carried value is the one reloaded before; ScratchMatrix column-major), Y2's
  `DistanceMatrix` header, MSVC `fast` FP flags = `/fp:precise /fp:contract` ("for performance", no measurement),
  EAP is 2.0-born (`git grep dtwFull_eap v1.0.0` = 0).
- At `830568a`: `ctest --test-dir build -C Release -j1` → "100% tests passed, 0 tests failed out of 126"; skipped
  test_cuda_correctness, test_metal_correctness, test_metal_mmap (MAY_SKIP). check_docs VERDICT=PASS; check_pins
  cmake=18 actions=37 failures=0; generate_docs --check current. pytest not re-run by me.

## Reported by agents, unverified

- Integrator gates at each merge: ctest 126 = 123 pass + 3 MAY_SKIP after P1; pytest 1135 / 19 / 0 (fresh venv,
  `--reinstall`); conformance digit-identical at every merge; Arrow 6/6 after Y2 and Y3.
- K1: linear kernel 1.36 ms, banded 0.257 ms (pinned); fill band FALSIFIED (1.36×: the unbanded fill ran EAP).
- P1: fill 14.5× unbanded, 5.1× band 50, 15.3× ECG5000 (machine loaded). cl packs nothing: Windows wheels (built by
  cl) get unpacked lanes.
- P2: linear beats EAP on 7/7 UCR datasets (1.48–3.57×), bitwise equal on 2800 pairs.
- W4a: no CUDA variant within 5 %; bug: FP32 L = 4095–4096 "invalid argument" (static shared memory ignored).

## Decisions

- Volkan: none new this session beyond the CHARTER entry. Orchestrator rulings (DECISIONS §3): PF-5 lanes pulled
  forward (P1); Y2 review point 4 (no per-lookup identity check); Y3 deletions; EAP killed (P2).
- Awaiting Volkan: (1) build the Windows wheels with clang-cl so pip users get packed lanes (needs libomp shipped);
  (2) drop `/fp:contract` from MSVC `fast` (likely cause of the 9 MSVC cross-path last-bit differences Z1 turned into
  tolerances; blocks packed lanes under cl); (3) a VERSIONINFO resource for `dtwc_cl.exe` (may reduce Sophos ML
  false positives for users).

## Next steps (PLAN)

1. B: X2S (sync pb/X2 with design-2.0 + OneBatchPAM k < N tie fix) is running → integrate X2 (Debug CLI tests if
   Sophos quarantines the Release binary, per the 09-28 ruling) → X3 sync + merge.
2. B: P3 (per-pair unbanded → linear kernel; EAP deleted) running → integrate; then drop MAP §7's EAP-slack line.
3. C: W4d (CUDA cleanup + the shared-memory bug) running in `wt/W4a` on `pb/W4d` → integrate (CUDA build gate).
4. B: Y4 (MEX + MATLAB catch-up for Y1/Y2/Y3/X2, then `matlab_suite`), W6e/W6f, race-free sweep, tracker-id comment
   sweep, `test_run_resolution` HiGHS guard.

## Open questions

- Linux/macOS: does ld.lld / ld64 LTO keep SLP vectorisation (the lanes' packing)? Only CI can show it.
- The CI legs for llfio ON wheels (manylinux, delocate) run only after Volkan pushes.

## Status honesty

Windows only. I ran benchmarks, probes, the serial ctest and the three gate scripts at `830568a`; the per-merge gates
(pytest, Arrow, conformance) were run by integrator agents. Not run this session: MATLAB (the MEX has not compiled since Y1), macOS/Metal, Linux, TSan.
