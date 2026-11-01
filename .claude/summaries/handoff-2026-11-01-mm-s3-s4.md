# Handoff — 2026-11-01 (Windows) — MM S3 and S4 closed: recorded, reviewed, fixed, re-checked; S5 next

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (DECISIONS §3, 10-21). At this session's
  start a64b44be = origin (Volkan's 10-28 merge of the Mac's 4-line handoff edit e48141ec; no code). Tree clean.
- `pb/MM` 819e8703, local only, worktree `C:/D/git/wt/MM` (clean): S1–S4 and their fixes. `wt/FM` (merged) stays for
  E1; `wt/TL` (merged) holds a build tree: ask before removing either.

## Done

- MM S3 14cf7b2b (its agent's reply was lost when the 10-22 session ended; recorded from its logs) and its review fix
  dd7ba144 (an error the PAMs or a block raised on accepted values came before a value refusal), re-checked clean
  (`baselines/2026-10-22-mm-s3-check.md`).
- MM S4 62fab80d: an Arrow IPC file beyond `--ram-limit` streamed through FastCLARA a record batch at a time, Parquet's
  stream through the same template. Two reviews (A the reader's memory bounds, B the stream and planner); fixes
  8dab1e78 (two blockers: the names read without llfio unpriced, per-batch objects priced at 0; six should-fix) and
  819e8703 (the walk's growth, every price measured both ways, `core::release_free_memory()`); both reviewers
  re-checked each (`baselines/2026-11-01-mm-s4-stream.md`).
- LESSONS: a session end can drop a reply after its commit (gates log as they run); a memory price holds only where it
  was measured. Pushed on the way: 23b29677, 2916a50c, 81899574.

## Verified by me

- Each commit's tree is its gated tree: 14cf7b2b/56c71754, dd7ba144/c71c0eb0, 62fab80d/6294a4b3, 8dab1e78/9cfee09b,
  819e8703/8eb85847; their summary logs read.
- Every finding of the four reviews: its cited lines opened before a fix was briefed. S4's `decode_bytes() = 0` sound
  by reading `batch_bytes` and the template's block price.
- My own clang-tree `ctest -j1`: 96/96 at 14cf7b2b and dd7ba144; 98/98 at 62fab80d, 8dab1e78 and 819e8703 (skips
  CUDA, Metal ×2; the three IPC tests ran). The IDE's clangd errors on `arrow_ipc.cpp` were a stale index.
- Corrections: `handoff-2026-10-22-mm-s1.md` and the 10-23 compaction summary call S3 running; it had committed.

## Reported by agents, unverified

- Every commit's six-tree gates, pytest, G6 and mutations (logs read, not re-run); every memory number (Windows
  private bytes, 24 threads, the reviewers' and implementers' runs).
- Reviewer A, for P1: Linux `madvise(MADV_WILLNEED)` may be capped at max(io_pages, ra_pages) a request.

## Decisions

- Asked through the question tool (10-31): "Close S3, then S4 (Recommended)".
- My calls, not asked: S3's B1 fixed as a blocker; S3's `bad_alloc` ordering nit not fixed (machine memory, not input);
  S4's `decode_bytes() = 0` accepted; S4 closed with the lane kernel's scratch carried to S7 (unpriced on every route
  since before MM; a bound needs each series' length as read); `release_free_memory()` kept (reviewer A: sound in a
  host, 0.21 ms the first call; the Windows bound fails without it).

## Next steps

1. PLAN MM S5 (GPU, interim; design §6): brief, implement, review. The CUDA tree here; G8b, Metal, on the Mac.
2. PLAN MM S6–S10. S7 prices the lane kernel's per-thread scratch on every route (2 × 128 B a time step a thread for
   doubles; CLARA streamed at 300M peaked 860 MB at 24 threads on 100,000-value series, PAM in RAM at 340M 939.6 MB).
   Merge design-2.0 into pb/MM before S8.
3. PLAN FM's E1 (Volkan runs `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` elevated); then FM's scratch and `wt/FM`.
4. PLAN "Before the 2.0 tag": the [CI] Linux wheel jobs; `cpp_conformance` under GCC and MSVC; L2b; F43; Linux archives.

## Open questions

- For Volkan (asked 11-01): should a top-level call release its threads' kernel scratch on return? 615 MB stayed in the
  OpenMP workers after a 100,000-value run; MAP §7 says thread-local scratch grows and never shrinks.
- Carried: a report of nanoarrow 0.8.0's trust of offsets upstream; whether MM and HG ship in 2.0; the history purge;
  ext4.vhdx; which `.claude` files to untrack.
- Disk: `C:/D/git/wt/tmp` (MM's tars and `.dmr`; `review_s3/`, `review_s4a/`, `review_s4b/`, `s3r/`, `s4r/`, `s4r2/`),
  `wt/FM` 4.6 GB, `wt/TL` 0.6 GB: deleted only on Volkan's yes.

## Status honesty

- Run by me: the clang-tree ctests above. Every other number is from agents' logs, read by me.
- Not run: any timing (S10's), Metal, CUDA by me, G11; pytest at dd7ba144 (no binding, reader or default). CI on
  design-2.0's tip is unread: `gh` is not authenticated here and the GitHub MCP server did not connect this session.
