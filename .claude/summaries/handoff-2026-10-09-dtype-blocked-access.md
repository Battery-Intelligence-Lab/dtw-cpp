# Handoff 2026-10-09 — dtype at load and block-wise memory access (Windows, session 79bd8991)

## Base

- Branch `design-2.0` at `a15fe9f0`. Uncommitted and untouched: Volkan's own edits to `LICENSE`, `dtwc/dtwc.hpp`,
  `dtwc/Problem_IO.cpp`, `dtwc/dtwc_cl.cpp` (never stage, stash, restore or reset them).
- Worktrees: `C:/D/git/wt/SW` (pb/SW, implementer running), `MV` (pb/MV, FALSIFIED 10-08, kept), `WM` (old, detached).
  Merged units' worktrees were removed (AL, BF, CS, PE, DT, AU, RL); their branches stay.
- `.claude/baselines/` and `CHARTER.md` are on disk only (`.gitignore`); this session's records are there:
  `2026-10-09-{al-clara-lanes,bf-blocked-fill,cs-clara-one-pass,pe-stream-ram-limit,dt-dtype-at-load}.md` + folders.

## Done

- AU `6521b01e` (`@author Volkan Kumtepeli` first in every source file); DT `999457d1` (`dtype` at load).
- `.claude` privacy `034d8994`, prune `9ab2b222`; `CITATIONS.md` moved to `docs/`.
- AL `6359acba`: FastCLARA's CPU assignment through the lanes (clang 3.4–3.8× at 24 threads, 7.8× on one; GCC
  4.8–5.7×, 10.0×; cl 1.29× under a band).
- BF `ec8d10e6`: blocked fill, range checkpoints, automatic `ram_limit` (mapped 22.6× at a 4 GiB cap, ≥ 37× at 2 GiB;
  checkpoints 2 M bytes against 80 M); GCC s1t FALSIFIED (0.892×, 0.951×), e3 FALSIFIED (0.405, 0.450 LLC misses per
  entry against ≤ 0.4), c1's closing save 0.04 % over: merged by Volkan's ruling.
- CS `8af0336f`: CLARA reads its data once (streams 3.4–3.6×, in RAM 1.00–1.86×); the stream refuses NaN, ±inf, empty
  and too-long rows (it published a cost of 1.8e308). The orchestrator finished this merge (see Verified).
- PE `cc45f4b4`: the stream holds `--ram-limit` (P1 at 1G 0.71 GB, was 1.47 GB) and runs 1.55–1.73× faster.
- Records: `77f15736`, `ef26e61b`, `8422a4b7`, `a15fe9f0` (PLAN, DECISIONS).

## Verified by me

- AL, BF, CS and PE timed runs: read their `report.md` files (kits under `C:/D/git/wt/tmp/<unit>/`); gate log
  `C:/D/git/wt/tmp/timing_al_bf.log`; all runs at the < 12 % gate with the load recorded.
- BF e3: both ETW attempts tabulated with `etw_table.py` (attempt 1 `kit/etw_attempt1/`, attempt 2 `kit/etw/`; the 05:27
  run measured s7 only, `kit/etw_0527_s7only/`, because the .cmd skips cases whose counters exist).
- GCC s1t per round (base steady 38–41 ms, BF bimodal) from `BF/kit/raw_wsl_attempt1.txt`.
- CS merge: integrator aborted on AL's clara harness (`--ram-limit 500K` refused in 38/48 streamed runs: CS design §5);
  I redid the merge, `cmp` of CHANGELOG against the integrator's tested resolution was identical, committed `8af0336f`.
- Read BF's fill (Problem.cpp), `DistanceMatrix::write/prefetch_rows/flush_rows`, CS's overlap loop, PE's pricing.

## Reported by agents, unverified

- Integrators' gate counts at each merge (clang 95, Arrow 98, CUDA 94, matlab_suite 141, pytest 895–896/19/0, CLI
  references identical); not re-run by me.
- PE's memory table and CS's/PE's 360-run identity grids; SW's design claims (15–35× predicted at 24 threads).

## Decisions

- Volkan, question tool: "Run now, load recorded" (timing gate < 12 % when the scanners hold 8–10 %); a second ETW
  attempt; "Merge now, misses recorded" (BF); "Next, before SW" (PE). All in DECISIONS §3.
- Orchestrator: CS's refusal of a `--ram-limit` too small for the samples' labels and a group accepted as designed;
  PE's band part 3b moved, before its run, from the 8G pair (routes differ) to 5000M (both stream).

## Next steps (PLAN "Now")

1. SW: implementer on pb/SW (`brief_sw_impl.md` in the scratchpad). Then c0 calibration, timing (≈ 2.8 h at the gate),
   Volkan's ETW `profile_sw_etw.cmd` (clear its output folder first), integrator.
2. PE-L: agent running (`brief_pel.md`): P1/P3 under cgroups in WSL with Arrow from pyarrow's manylinux wheel.
3. FM: design after SW's code (it changes the swap's matrix reads); BF proposes a rows-written field for `.dtwm` v5.
4. MS, then the candidates (BF's 8-column tasks for small GCC fills, …).

## Open questions (Volkan)

- GitHub history purge of baselines, CHARTER and the Kasper folder (a force push: his).
- The Mac: restore `baselines/` and `CHARTER.md` after pulling (`git restore --source=67af5651 --worktree -- …`).
- Whether to commit his four uncommitted edits; RA's notice wording; which other `.claude` files to untrack.

## Status honesty

- Built, run and tested this session: every merged unit's gates (by integrators), the timed kits (by me, at the
  < 12 % gate, not the < 6 % quiet gate: Sophos and Tanium never let it open after 02:13).
- Not run: macOS (Metal, codegen) after RL/DT/AL/BF/CS/PE; Linux release archives; ARC; PE on Linux (PE-L running).
