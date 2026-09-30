# Handoff — 2026-09-29 — phase B continued; the CPU kernel and fill made fast (Windows)

## Base

Branch `design-2.0`, HEAD `a073113` (this handoff's interim version, committed). Uncommitted: this final version
and the CHARTER line. Session base `4dcc39d`. Stopped on Volkan's request; no agent runs. Worktrees `C:/D/git/wt/<id>`.

## Done (merged into design-2.0, one gated merge each)

- K1 `4441969`: no DP cell makes a library call. `std::min({…})` was `__std_min_d` on the MSVC STL; nested min plus a
  register-carried `dp[i-1, j]`; `test_codegen_no_calls` guards it (`baselines/2026-09-29-k1-dp-cell-no-call.md`).
- Y2 `959dc5b`: W5d — one `core::DistanceMatrix` (heap or mapped `.dtwm`), header-only llfio in one .cpp, typed
  checkpoint outcomes; wheels and release archives ship llfio ON. Adversarial review (the one agent run on Fable
  this session — Volkan has since said not to use Fable): merge, no confirmed defect.
- Y3 `b260415`: W6a (`index_t`), W6c (`run_openmp` per-thread failure slots, `parse_ram_limit`, `GpuPrecision`),
  W6d (v1 `cluster_by_kMedoidsPAM` shim back, `dtwc_main` and non-v1 forwarders gone).
- P1 `d61c499`: the CPU fill runs W = 64/sizeof(T) equal-length pairs per call in SIMD lanes, bitwise equal to the
  per-pair kernels (`baselines/2026-09-29-p1-lanes-fill.md`); `core/dtw_lanes.cpp` is `-fno-lto` under clang/MSVC ABI.
- Records: CPU/GPU baselines, PF-5 (PASS), P2, W4a in `.claude/baselines/2026-09-29-*`; DECISIONS, LESSONS, MAP.

## Verified by me

- `4dcc39d`, pinned P-core 12: `BM_dtwFull_L/1000` 7.225 ms (7.2 ns/cell), fill `100/1000/-1` 1402 ms; the inner loop
  of `dtw_kernel_linear<StandardCell>` (build flags, `-S`) calls `__std_min_d` per cell. My probe: nested min + carry
  1.3 ns/cell, bitwise identical. GPU fill (FP32 Auto): 121 Gcell/s at N 100, L 1000.
- Read: K1's kernel diff, Y2's `DistanceMatrix` header, MSVC `fast` = `/fp:precise /fp:contract` ("for
  performance", unmeasured); EAP is 2.0-born (`git grep dtwFull_eap v1.0.0` = 0).
- At `830568a`: `ctest --test-dir build -C Release -j1` → "100% tests passed, 0 tests failed out of 126"; skipped
  test_cuda_correctness, test_metal_correctness, test_metal_mmap (MAY_SKIP). check_docs VERDICT=PASS; check_pins
  cmake=18 actions=37 failures=0; generate_docs --check current. pytest not re-run by me.

## Reported by agents, unverified

- Integrators: after P1 ctest 126 = 123 + 3 MAY_SKIP, pytest 1135 / 19 / 0 (`--reinstall`); conformance
  digit-identical at every merge; Arrow 6/6 after Y2 and Y3.
- K1: linear 1.36 ms, banded 0.257 ms (pinned); its fill band FALSIFIED (1.36×: the unbanded fill ran EAP).
- P1: fill 14.5× unbanded, 5.1× band 50, 15.3× ECG5000 (loaded machine); cl packs nothing (Windows wheels use cl).
- P2: linear beats EAP on 7/7 UCR datasets (1.48–3.57×), bitwise equal. W4a: no CUDA variant within 5 %; bug:
  FP32 L = 4095–4096 "invalid argument" (static shared memory ignored).

## Decisions

- Volkan (CHARTER 2026-09-29): "no unnecessary abstraction … make sure the code works fast and generates decent
  assembly like SIMD where needed"; later: "Please do not use it [Fable]. Also stop now and write a handoff".
- Orchestrator rulings (DECISIONS §3): PF-5 lanes pulled forward (P1); no per-lookup identity check (Y2 review
  point 4); Y3's deletions; EAPruned killed (P2).
- Awaiting Volkan: (1) Windows wheels built with clang-cl, so pip users get packed lanes (libomp shipped);
  (2) drop `/fp:contract` from MSVC `fast` (likely cause of Z1's 9 MSVC last-bit mismatches; blocks cl packing);
  (3) a VERSIONINFO resource for `dtwc_cl.exe` (may lower Sophos ML false positives for users).

## Stopped mid-unit (on Volkan's request, ~21:45)

- X2S, worktree `C:/D/git/wt/X2`, branch `pb/X2`: merge of design-2.0 at `d61c499` committed as `ea717f7` (227 files,
  no conflict markers in the tree) and the OneBatchPAM k < N tie fix as `9edfad2` (test first, per its brief). The
  Release tree built after both (`build/bin/dtwc_cl.exe` 21:41). NO gate ran on the synced tree: not ctest, not
  conformance, not pytest, not Arrow. The agent stopped while re-checking `dtwc/cli/run.cpp` ("design-2.0's
  structure plus X2's two changes"): confirm the merge kept X2's own run.cpp edits (`git diff abfd08d 31b4fc9 --
  dtwc/cli/run.cpp`) before gating. Whether Sophos flags the new Release binary is unknown.
- W4d (`wt/W4a`, CUDA build in `build-cuda`, branch `pb/W4d`) and P3 (`wt/P3`, `pb/P3`): both at `830568a`, no change.
- Agent briefs are in the session scratchpad (`brief_*.md`, `gates_*.md`); PLAN rows P3 and W4d carry their substance.

## Next steps (PLAN)

1. B: finish X2S — verify run.cpp, then the gates (serial ctest; CLI tests on `build-dbg` if Sophos quarantines the
   Release binary, per the 09-28 ruling; conformance; pytest `--reinstall`; Arrow in a `build-arrow` tree) → integrate
   X2 → X3 sync (take Y3's GPU-precision versions) + merge.
2. B: P3 — per-pair unbanded Standard DTW on the linear kernel, EAPruned deleted (P2 evidence); then MAP §7's EAP line.
3. C: W4d — CUDA forcing machinery out, the FP32 L = 4095–4096 shared-memory bug fixed with a test first.
4. B: Y4 (MEX + MATLAB catch-up for Y1/Y2/Y3/X2, then `matlab_suite`), W6e/W6f, race-free sweep, tracker-id comment
   sweep, `test_run_resolution` HiGHS guard.
5. No agent runs on Fable (memory `feedback_no_fable.md`); at most four agents at once.

## Open questions

- Do ld.lld / ld64 LTO keep the lanes' SLP packing, and do llfio-ON wheels build on manylinux / macOS? CI after a push.

## Status honesty

Windows only. I ran benchmarks, probes, the serial ctest and the gate scripts at `830568a`; per-merge pytest, Arrow
and conformance ran in integrator agents. Not run: MATLAB (MEX broken since Y1), macOS/Metal, Linux, TSan.
