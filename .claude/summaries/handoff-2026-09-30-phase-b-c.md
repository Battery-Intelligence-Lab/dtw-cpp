# Handoff — 2026-09-30/10-01 — phases B–D nearly closed, E and F under way (Windows)

## Base

Branch `design-2.0`, HEAD after this file's commit (on `5ff4320`). Session base `a073113`; first commit `cf0cf4a`.
Interim version (10-01 ~02:30): W13b, C3, W8a (Opus) and W12c-sync (Sonnet) were running. Agent briefs and gates:
session scratchpad `…/79bd8991-b0e9-441b-bc55-95584dcfaf55/scratchpad/` (`brief_*.md`, `gates_*.md`, inventories).

## Done (merged into design-2.0, one gated merge each)

- 09-30 morning: X2 `4de2ce9`, P3 `5e15459`, Y4 `93eefe1`, G1 `1bb9413`, R1 `1ea8327`, W6e `fe1bf9b`, W6m `e053594`,
  F1 `b5c7048`, W4d `febd25f` (see the 09-30 rulings in DECISIONS §3).
- Bindings and outputs: B1 `4968d44`, B2 `e0085fd`, B3 `65edcf7`; X3b `6e346a7` (-Werror=switch, C4062); H1 `1a079cd`.
- Counts: W11a `cfeac4b` (index_t), W11b `8a9db51` (np.int64 / exact doubles), W11c `d8d831d` (last 32-bit guards).
- GPU: W13a `62d5822` (packed output, int64 chunks), C1 `ceb7f91` (any L; global wavefront), C2 `dd33a0e` (route rule).
- Speed: P4 `69be49c` (OneBatch assignment 21.9×), P5 `bac9120` (OneBatch table on lanes 4.61×), E1 `770816e`
  (DistanceConfig; dist_by_ind an O(1) read; PAM swap 5.6–5.9×; no lock or atomic left in dtwc/).
- Tests: W12a `6c6f3d4` (−4,217), W6f `ce34bf8` (Problem API table; 3 probe files), W12d `eeea331` (parity table).
- Build/HPC: V3 `169db40` (x86-64-v3 wheels/archives; one DTWC_ARCH_LEVEL), S1 `c54e375` (ARC scripts: CUDA floor,
  native GPU-node build), F2 `a965ad5` (DTWC_CL_PATH), V4 `d6a9d54` (cross-route checks within a path-length bound).
- Records: CHARTER (09-30 and 10-01 quotes), DECISIONS §3 rulings, LESSONS, runbook rule 2, PLAN marks.

## Verified by me

- W6f deletions applied by me (`88c4a73`, `78b27ef`): the final test file differs from the agent's verified copy only by
  its header; branch build warning-free, 120 registered, the 5 affected tests pass.
- W12c deletions applied by me (`8d75fde`, −2,717): the oracle table covers the v1 entry points, DTW/ADTW early abandon
  (off, above, at, below), hand values; tree builds, 110 registered. S1's native line fixed by me (`0268df1`, dry run).
- Integrator logs: last green at `5ff4320` — clang ctest 111 = 108 + 3 MAY_SKIP; CUDA tree 110 / 0 (test_cuda_correctness
  60 cases, 7202 assertions); pytest 1183 / 19 / 0 (at W12d); matlab_suite 149 / 148 / 1 incomplete (at V3); Arrow 5 / 5.

## Reported by agents, unverified

- V3: conformance digit-identical on clang, MSVC and GCC 13.3 at v3; lanes at v3 ~1.0× unbanded, 1.17× banded vs SSE2
  (under load). V4: GCC 13.3 v3 ctest 113 / 0 failed after the sync; worst route-bound ratio 0.0043 (double), 0.33 (float).
- E1 integrator: CLI GPU vs CPU on data/dummy, labels and medoids byte-identical.

## Decisions

- Volkan 09-30: "Yes, all three" (W11c/W12a/W6f deletions); CPU floor "x86-64-v3 (Recommended)"; ARC: `device=hpc` with
  `gpu_device=`, or detect on the node; "Yes, please delete the trivial tests, we don't need to write tests just for
  writing tests."
- Volkan 10-01 (FP): "We don't need bit-by-bit equivalence between compilers. So they could differ minimally like 1e-9
  epsilon or something. However, this shouldn't change the clustering results. … then go for the speed of course."
  Contraction stays on every compiler; the MSVC `/fp:contract` question is closed.
- Awaiting Volkan: clang-cl Windows wheels; VERSIONINFO for `dtwc_cl.exe`; whether the CI python job should build dtwc_cl.

## Next steps (PLAN)

1. F: W12c merge after its sync (folds its inline bound into `dtw_routes_agree`); W12b after W13b.
2. E: W13b (finite scan at intake) → merge; W8a (one reader) → merge; then W7c (brief ready: `brief_w7c.md`), W7d–g,
   W8b–c, W9 (after W7), W10.
3. C: C3 (FP64 Shared kernel at 4 blocks/SM, own band); lead: 16 double lanes; W4e Metal (macOS CI).
4. The tracker-id comment sweep when no code unit is in flight; HEAD baseline on a quiet machine.

## Open questions

- Metal edits (W4d, W13a) compiled blind here; macOS CI after a push is their gate. GCC failures fixed by V4 were never
  seen by CI (Linux CI builds Debug, without the arch flag).
- `.github/workflows/python-tests.yml` runs pytest with no dtwc_cl (F2 note); CI not run here.

## Status honesty

Windows only (clang, MSVC for the MEX/CUDA), plus GCC 13.3 in WSL for V3/V4. Not run: macOS/Metal, native Linux, CI.
Timing under load is [inferred]. A weekly API limit stopped three agents once (10-01); they were relaunched.
