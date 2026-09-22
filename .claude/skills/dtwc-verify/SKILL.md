---
name: dtwc-verify
description: Build DTWC++ and run the full verification set — serial ctest, the three gate scripts, the layer ratchet — and report the result honestly. Use before claiming a change is green, before a commit, at a wave exit gate, or when asked whether the tests pass.
---

# Verify

"Tests pass" is a claim about a command you ran. If you did not run it, say so.

## Steps

1. **Build** (macOS shown; `PLAN.md` §2.1 has the other platforms):

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
   `dtwc_add_test`, which carries a pass floor from `tests/floors.cmake`; a test whose subject did
   not run fails its floor rather than passing quietly. When a floor fails but the binary prints
   "All tests passed", the floor is the finding — usually a count measured on another platform.

4. **Gates**, all three, plus the ratchet:

   ```sh
   python3 scripts/check_record_hygiene.py
   python3 scripts/check_repo_hygiene.py
   python3 scripts/check_docs_contract.py
   uv run --no-project python scripts/repo_map.py layers     # upward edges may only fall
   ```

5. **Before calling anything "pre-existing", stash and re-run it.** `git stash -u`, run the failing
   subject, restore. A failure you inherited and a failure you caused look identical in the log.

6. **Report** with the command and the counts: "126/131 pass, 2 CUDA skips, 5 floor failures" —
   never "tests pass". If a suite was not run, name it as not run. If two runs disagree, that is a
   third computation, not a coin toss.

## Rules

- A change that claims to be a no-op must produce **digit-identical** conformance output on this
  machine. Across machines the contract is correctness within tolerance, not bit-identity (D-19) —
  `std::exp` and `std::log` differ between glibc, Apple libm and UCRT.
- Optional dependencies stay optional: if you touched the build, configure once without them.
- Numbers go to `.claude/baselines/` verbatim, tagged `[confirmed]` or `[inferred]`.
- Do not commit unless Volkan asked.
