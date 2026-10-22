# Handoff — 2026-10-22 (Windows) — MM S1 and S2 proven and hardened; the Linux wheel imports; S3 started

## Base

- `design-2.0` at the commit holding this file, pushed fast-forward with it (Volkan's 10-21 ruling); TL merged
  e3cba47d. Tree clean.
- `pb/MM` fd8a7450, local only, worktree `C:/D/git/wt/MM`; merges into design-2.0 cleanly (`git merge-tree`, 10-22).
  S3's implementer works there from `C:/D/git/wt/tmp/MM/brief_s3.md`. `wt/FM` (pb/FM, merged) stays for E1; `wt/TL`
  (pb/TL 571c40f9, merged) holds a build tree: ask before removing either.

## Done

- MM S1, the reader: 70d512cd, hardened 4d1e0723 (nanoarrow 0.8.0 trusts a file's offsets and lengths), GCC force-inline
  84d84101 (Volkan's ruling). S2, codecs and decoding: 61b88767 (S1's residuals), 71d9d4b1, 25f65d34 (declared
  decoded sizes capped, after an automated security review), fd8a7450 (its review: a V4 union, the footer length).
  Records: `baselines/2026-10-21-mm-s1-reader.md`, `2026-10-22-mm-s2-codecs.md`.
- TL: the Linux wheel could not be imported (llfio's initial-exec thread-local made the module STATIC_TLS, since before
  MM); llfio's own `WG14_SIGNALS_HAVE_ASYNC_SAFE_THREAD_LOCAL=0` on ELF, 571c40f9, merged e3cba47d
  (`baselines/2026-10-22-tl-static-tls.md`).
- Earlier today: docs audit merged 485dbeb7; `verify_results.py` 59202f68; F42 closed (79ee8310). LESSONS 3c83159c.

## Verified by me

- S1 and each fix: the gate logs on their committed trees (six trees serial ctest, `matlab_suite` 142/142, pytest
  915/19/0, docs gates). The vendored nanoarrow against the ASF checksum and a re-bundle (40/40 identical). A mutation
  of the name check (`test_arrow_ipc` failed at the case only). GCC inline reports. My own rebuild and ctest of 84d84101.
- S2's commits: every build after the last edit; six trees, `test_arrow_ipc` run in each (252–266 assertions in
  fd8a7450's tree), F8 9/9, pytest 915/19/0. lz4 1.10.0 and zstd 1.5.7 pins equal the published `.sha256` (fetched
  again). G11 from its log (0 codec symbols exported; 175 without the visibility defines).
- The decode path, by reading: the owner keeps the bytes iff a batch is viewed; `field_index` matches nanoarrow's
  numbering; the review's blockers in nanoarrow (V4 union buffers `nanoarrow_ipc.c:27227`, `:27700`, `:27820`; the
  footer's int32 sum `:27025`, `:27184`).
- TL: base and fixed wheels in WSL (STATIC_TLS 1 → 0, initial-exec relocations 1 → 0, import exit 1 → 0); the merge's
  tree equals the gated tree (024e49be).
- Correction to `handoff-2026-10-21-fm-timed.md` (pruned): its "14:5x" push was at 14:31.

## Reported by agents, unverified

- Pre-fix outcomes: B1/B2 wild spans and S3 SIGSEGV (cl, GCC) in S1; in S2 a 2^63−8 offset SIGSEGV (clang, GCC) and the
  V4 union fixture's crash. The reviewer's traces (all re-checked clean). TL's heap fallback is compiled, not exercised.

## Decisions

- Volkan 10-21: "Force-inline now" (DECISIONS §3); push after a merge or a committed handoff.
- My calls, not asked:
  - the reader validates only the columns it reads, and refuses metadata V4 with a union;
  - the minor review leftovers go to the next owner of the reader (S4: the footer's byte order);
  - TL as its own unit, merged.

## Next steps

1. PLAN MM S3 (in flight): check it against its logs, including G6 under GCC, clang and cl (the predicate loop), and
   the lowest-index error across the fill, the fused pass and the stream; then a fresh adversarial review.
2. PLAN MM S4–S10 (design §6). Merge design-2.0 into pb/MM before S8 (S8 builds wheels; pb/MM lacks TL's fix).
3. PLAN FM's E1 (Volkan runs `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` elevated); then FM's scratch and `wt/FM`.
4. PLAN "Before the 2.0 tag": the [CI] Linux wheel jobs; `cpp_conformance` under GCC and MSVC; L2b; F43; Linux archives.

## Open questions

- Report nanoarrow 0.8.0's trust of file offsets and lengths upstream? An outside post: Volkan's call. Whether MM and
  HG ship in 2.0; ext4.vhdx; the history purge; which `.claude` files to untrack.

## Status honesty

- Run by me: the S1 mutation, GCC inline reports, the vendored re-bundle, the checksum fetches, the clang rebuild and
  ctest of 84d84101, TL's readelf and import in WSL. Run by agents: every other gate.
- Not run: G11 on the Linux MEX or the Mac; Metal; any timing (another session's 8-way study loads this machine).
