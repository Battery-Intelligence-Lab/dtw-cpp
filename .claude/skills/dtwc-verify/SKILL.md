---
name: dtwc-verify
description: Build DTWC++ and run the full verification set — serial ctest, the gate scripts — and report the result honestly. Use before claiming a change is green, before a commit, at a wave exit gate, or when asked whether the tests pass.
---

# Verify

"Tests pass" is a claim about a command you ran. If you did not run it, say so.

## Steps

1. **Build** (macOS shown; `MAP.md` §3 has the other platforms):

   ```sh
   cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp
   cmake --build --preset clang-macos
   ```

2. **Test serially.** `-j1` is the evidence run — concurrent tests collide on shared artefacts, so
   a parallel pass is not evidence:

   ```sh
   ctest --test-dir build -C Release -j1 --output-on-failure
   ```

3. **Read the skips.** CTest scores a skipped test as a pass. Every test goes through
   `dtwc_add_test`, which passes a test only when Catch2 printed "All tests passed" with at least one
   assertion in at least one case — a failure, a skip or an empty run fails. Only a `MAY_SKIP`
   test (a device or capability the build lacks) may show as Skipped; read its SKIP message.

4. **Gates**, all three (the secret scan is gitleaks in CI):

   ```sh
   python3 scripts/check_docs.py --cli build/bin/dtwc_cl   # doc flags vs live --help; harness self-check
   python3 scripts/generate_docs.py --check                  # generated pages are fresh
   python3 scripts/check_pins.py
   ```

5. **Before calling anything "pre-existing", stash and re-run it.** `git stash -u`, run the failing
   subject, restore. A failure you inherited and a failure you caused look identical in the log.

6. **Report** with the command and the counts: "129/131 pass, 2 CUDA skips" —
   never "tests pass". If a suite was not run, name it as not run. If two runs disagree, that is a
   third computation, not a coin toss.

## Rules

- A change that claims to be a no-op must produce **digit-identical** conformance output on this
  machine. Across machines the contract is correctness within tolerance, not bit-identity (D-19) —
  `std::exp` and `std::log` differ between glibc, Apple libm and UCRT.
- Optional dependencies stay optional: if you touched the build, configure once without them.
- Numbers go to `.claude/baselines/` verbatim, tagged `[confirmed]` or `[inferred]`.
- Commit only a proven step, locally on `design-2.0` (Volkan, 2026-09-28); never push.
