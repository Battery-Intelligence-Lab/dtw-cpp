# Handoff — 2026-10-22 (Windows) — MM S1 proven, hardened and force-inlined; S2 started

## Base

- `design-2.0` at the commit holding this file (after 9f403021), pushed fast-forward with it (Volkan's 10-21 ruling);
  origin was 59202f68. Tree clean.
- `pb/MM` 84d84101 (S1 70d512cd, its fixes 4d1e0723 and 84d84101), local only, worktree `C:/D/git/wt/MM`. S2's
  implementer is working there from `C:/D/git/wt/tmp/MM/brief_s2.md`. `wt/FM` (pb/FM 2ef3e075, merged) stays for E1.

## Done

- Docs audit fixed and merged 485dbeb7 (pb/DX 671ae09f); `verify_results.py` names UCR rows from 1 (59202f68).
- F42 closed, it does not reproduce (79ee8310; `baselines/2026-10-21-f42-cuda-auto.md`); its worktree removed.
- MM S1, the Arrow IPC reader, 70d512cd (implementer agent). The adversarial review found two blockers (nanoarrow
  0.8.0 trusts a file's buffer offsets and lengths) and three minor findings: fixed 4d1e0723 (tests first). GCC's
  out-of-line `Data::series` force-inlined 84d84101. The re-check is clean; its minor residuals go first in S2.
- Records: `baselines/2026-10-21-mm-s1-reader.md`; PLAN aa120a96, 9f403021; DECISIONS §3 aa120a96; LESSONS 3c83159c.
- Stopped: S1's agent's `wait_builds.sh`, spinning since 17:13 on `bc` (Git Bash has none), and a stuck command of mine.

## Verified by me

- S1's gates from its logs (`C:/D/git/wt/tmp/MM/logs/final_*`), all older than the commit, no file newer than the
  builds: clang 96, Arrow 99 (`test_io_readers`, F8 parity 9/9), no-llfio 96, MEX 97 (`matlab_suite: 142 run, 142
  passed`), CUDA 95, GCC 95, `test_arrow_ipc` run in all six; pytest fresh venv 915/19/0; docs gates green.
- Vendored nanoarrow: SHA-512 equals the ASF's `.sha512` (fetched again); upstream `bundle.py` re-run gives all 40
  committed files identical. Mutation: name-offset check removed → `test_arrow_ipc` failed at line 208 only; restored.
- GCC inline reports regenerated (`logs/inl_base.txt`, `inl_s1.txt`): misses at `inline-unit-growth` 150 → 152;
  accessor-callee misses 11 → 14; out-of-line accessor calls in the fill tile 1 → 4 (base's one, since FM).
- B1 and B2 read in `nanoarrow_ipc.c:27632-27634`, `:27413` and `nanoarrow.c:3196-3200`.
- Fix diffs read; `data_.series(i)` is exactly `Problem::series`'s body (`Problem.hpp:313`). Final gate logs
  (`final_fix_*`) and G6 from the kit's `.s` files: no accessor call left in Problem.cpp; tile lambdas 6 → 5 calls;
  TADPole and one_batch_pam gain only duplicated indirect `std::function` sites, no new callee.
- My own rebuild of the committed files in `build`, serial ctest: 96/96, `test_arrow_ipc` 120 assertions in 6 cases,
  codegen 88 loops 0 calls (`logs/verify_fix_clang_*`).
- Correction to `handoff-2026-10-21-fm-timed.md`: its "14:5x mine" push was at 14:31 (920a4fe4, 705ada87; origin's
  reflog), and 59202f68 was pushed from here at 15:51.

## Reported by agents, unverified

- Pre-fix runs (`logs/fix_prefix_summary.log`): B1 and B2 accepted with wild spans under clang 21, cl 19.50 and GCC
  13.3; S3 crashed (SIGSEGV) under cl and GCC, refused by nanoarrow under clang.
- The 01:45 rewrite of `Data.hpp` and `Problem.cpp`, after the gates, was a `git stash` / `pop` around clang-tidy, same
  tree id 7a86bbb4 after it; my rebuild re-proves the clang tree only.
- The reviewer's re-check of 4d1e0723 (trace only): every finding closes, no new memory hole. TADPole's extra call
  sites are code copies, no more calls per pair (inferred; not timed).

## Decisions

- Volkan 10-21 (question tool), on the GCC call: "Force-inline now" (DECISIONS §3). Pushing after a merge or a
  committed handoff is his ruling of 10-21.
- My calls, not asked: the reader validates only the columns it reads; S1's minor residuals form S2's first,
  separate commit; S2's brief adds the decode-path check the design predates (buffer bounds before decompression).

## Next steps

1. PLAN MM S2 (in flight): check its report against its logs (the decode path's hostile-input design, G4b, G11 via
   the Linux wheel's `nm -D`; the Linux MEX and the Mac's `nm -gU` stay open), then review its reader changes.
2. PLAN MM S3–S10, per `C:/D/git/wt/tmp/MM/design.md` §6; S3 grows Problem.cpp, so check G6 under GCC.
3. PLAN FM's E1: Volkan runs `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` elevated; then ask about FM's scratch
   (29.6 GB) and removing `wt/FM` and `pb/FM`.
4. PLAN "Before the 2.0 tag" [here] items in machine slots: `cpp_conformance` under GCC and MSVC; L2b; F43; Linux
   archives and libgomp. `DTWC_ALWAYS_INLINE` awaits Linux nvcc (ARC's next GPU job) and Apple clang (the Mac).

## Open questions

- Report nanoarrow 0.8.0's trust of file offsets and lengths upstream (apache/arrow-nanoarrow)? An outside post:
  Volkan's call. Also: whether MM and HG ship in 2.0; ext4.vhdx; the history purge; which `.claude` files to untrack.

## Status honesty

- Run by me: the S1 mutation (clang tree), the GCC inline reports (WSL), the vendored re-bundle and checksum fetch,
  the clang rebuild and ctest of 84d84101. Run by agents: six trees twice, pytest twice, matlab_suite twice, G6 kits.
- Not run: G11 on any binary (S2's), Metal or macOS, any timing (another session's 8-way study loads this machine).
