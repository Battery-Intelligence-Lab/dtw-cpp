# R3-F11 tracked CMake archive integrity — 2026-07-23

## Scope and base

- Branch: `Claude`
- Base commit: `658a9cd` (`docs: route supply-chain audit findings`)
- Subject: every tracked `CPMAddPackage(URL ...)` remote archive declaration,
  including `examples/cpp/example_project/CMakeLists.txt`.
- Existing GitHub Action SHA validation and the exact Arrow URL/digest ledger
  remain load-bearing and may not be weakened.
- The adjacent workflow-download/container policy and the malformed
  `cmake>=3.26` shell command are separate findings F34 and F35. They are not
  evidence that F11 passed or failed.

The PLAN archive, live PLAN killed-ideas section, and `.claude/LESSONS.md` were
searched before this branch. No killed idea is reopened. The relevant prior
rule is Task 0.12: remote build inputs must be immutable or content-verified,
and optional dependencies remain optional.

## Registered inherited inventory

The pre-implementation tracked-CMake audit found 25 tracked `CMakeLists.txt` /
`*.cmake` files and seven `CPMAddPackage(URL ...)` archive declarations:

| Manifest | Archive | Same-block SHA-256 |
|---|---|---|
| `cmake/Dependencies.cmake` | Catch2 v3.13.0 | `650795f6501af514f806e78c554729847b98db6935e69076f36bb03ed2e985ef` |
| `cmake/Dependencies.cmake` | HiGHS v1.15.1 | `a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16` |
| `cmake/Dependencies.cmake` | CLI11 v2.6.2 | `c6ea6b2e5608b3ea8617999bd5f47420c71b2ebdb8dc4767c1034d1da5785711` |
| `cmake/Dependencies.cmake` | Eigen 5.0.1 | `e4de6b08f33fd8b8985d2f204381408c660bffa6170ac65b68ae1bd3cd575c0a` |
| `cmake/Dependencies.cmake` | yaml-cpp 0.9.0 | `25cb043240f828a8c51beb830569634bc7ac603978e0f69d6b63558dadefd49a` |
| `cmake/Dependencies.cmake` | Arrow 19.0.1 | `4c898504958841cc86b6f8710ecb2919f96b5e10fa8989ac10ac4fca8362d86a` |
| `examples/cpp/example_project/CMakeLists.txt` | `refs/heads/documentation_update.zip` | **missing** |

Thus the registered inherited archive result is exactly 6/7 verified, one
mutable, one unhashed. The current checker is expected to false-green because
it hard-codes only Arrow and does not enumerate the example manifest.

The two tracked CPM bootstrap scripts use a different, already content-checked
form, `file(DOWNLOAD ... EXPECTED_HASH SHA256=${CPM_HASH_SUM})`; they are part
of F34's wider acquisition inventory, not silently counted as
`CPMAddPackage(URL ...)` declarations here.

## Registered replacement identity

The example will consume the committed 2.0.0rc1 boundary:

```text
commit=eda1b92bc89ee51568b052a6af86f615d336de3c
version=2.0.0rc1
url=https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/eda1b92bc89ee51568b052a6af86f615d336de3c.zip
size=4928286
sha256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696
```

This commit is the PLAN's named rc1 boundary and an ancestor of the published
`origin/Claude` snapshot. Its `examples/cpp/example_project/CMakeLists.txt`
and `main.cpp` Git blobs are byte-identical to the current files:

```text
current_example_cmake_blob=3e024a0fb598ad46c4a966155732cbfc1c250226
rc1_example_cmake_blob=3e024a0fb598ad46c4a966155732cbfc1c250226
current_example_main_blob=f123274e2ccd692154bcafb3eb10ba4b38cfa02b
rc1_example_main_blob=f123274e2ccd692154bcafb3eb10ba4b38cfa02b
rc1_version=2.0.0rc1
```

Two independent HTTPS routes returned identical bytes before registration:

```text
url=https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/eda1b92bc89ee51568b052a6af86f615d336de3c.zip
size=4928286
sha256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696
url=https://codeload.github.com/Battery-Intelligence-Lab/dtw-cpp/zip/eda1b92bc89ee51568b052a6af86f615d336de3c
size=4928286
sha256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696
```

Those downloads select the candidate identity; they are exploration, not the
F11 verdict. The fresh example configure below is the live CPM/hash proof.

## Acceptance band

F11 passes only if all of the following hold:

1. On the registered base, `python scripts/check_supply_chain_pins.py` exits
   zero and prints `supply-chain pins verified` despite the exact mutable,
   unhashed example declaration. Record that result as **FALSIFIED** for
   repo-wide tracked-CMake coverage, not as a passing supply-chain verdict.
2. Implement the generic checker and its tests *before* editing the example.
   The modified production CLI, run against the real inherited tracked tree,
   must exit 1, name
   `examples/cpp/example_project/CMakeLists.txt:12`, and print counters
   `verified=6 total=7 mutable=1 unhashed=1 verdict=FAIL`. This is the PLAN's
   first real gate and proves `main()` reaches the generic scanner; a helper
   fixture alone cannot satisfy it.
3. The permanent checker enumerates paths from the Git index rather than
   recursively entering untracked `build*/_deps`. It finds exactly 25 tracked
   CMake manifests and all seven `CPMAddPackage(URL ...)` declarations.
4. URL and `URL_HASH SHA256=<64 hex>` must occur as active directives in the
   same CPM call. A hash in another package, a comment, or prose does not
   satisfy the declaration. `refs/heads/` and other registered mutable-branch
   archive forms fail even if an attacker supplies a digest.
5. The existing 39/39 full-SHA workflow-action check and the exact Arrow
   URL/digest check remain active.
6. The example URL equals the registered full-commit archive and its
   `URL_HASH` equals the registered SHA-256. No branch, tag alias, shortened
   commit, or post-run digest substitution is accepted.
7. A permanent focused Python suite runs at least nine tests with zero failure
   or skip. It must kill, by name: the exact inherited example fixture; a
   missing, shortened, and comment-only URL hash; a branch URL retaining a
   valid hash; removal of a currently pinned main archive hash; Arrow digest
   drift; and a mutable workflow action. At least one valid hashed archive must
   pass the same parser.
8. The real checker exits zero on the repaired tree and prints counters derived
   from successful checks:

   ```text
   WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
   CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
   ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
   supply-chain pins verified
   ```

9. Before the live download, `build/f11-example-pin` must be absent; do not
   reuse or clean another build tree. Unset `CPM_SOURCE_CACHE` so neither an
   environment cache nor a prior package source can satisfy the request. Run
   these exact PowerShell commands from the repository root:

   ```powershell
   if (Test-Path 'build/f11-example-pin') { throw 'F11 fresh build directory already exists' }
   $env:CPM_SOURCE_CACHE = $null
   cmake -S examples/cpp/example_project -B build/f11-example-pin -G Ninja -DCMAKE_BUILD_TYPE=Release -DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_HIGHS_GPU=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_ARROW=OFF -DDTWC_ENABLE_CUDA=OFF -DDTWC_ENABLE_METAL=OFF -DDTWC_ENABLE_MPI=OFF -DDTWC_ENABLE_YAML=OFF -DDTWC_BUILD_TESTING=OFF -DBUILD_TESTING=OFF -DDTWC_BUILD_BENCHMARK=OFF -DDTWC_BUILD_EXAMPLES=OFF -DDTWC_BUILD_MATLAB=OFF -DDTWC_BUILD_PYTHON=OFF
   ```

   It must perform the registered download/hash check and exit zero. Because
   the project's configuration summary is top-level-only, the reachable
   nested-consumer version proof is instead exact content `2.0.0rc1` in
   `build/f11-example-pin/_deps/dtw-cpp-src/VERSION` plus at least one active
   `DTWC_VERSION_STRING=\"2.0.0rc1\"` definition in the generated
   `build.ninja`. Then
   `cmake --build build/f11-example-pin --target dtwc++ --parallel` must exit
   zero. A cached or historical hand-linked artifact is not acceptable.
10. The existing C++ supply-chain test executes rather than skipping. The full
   canonical gate remains 114/114, zero failed, with exactly the six registered
   capability skips.
11. `git diff --check`, generated documentation, documentation contract,
    record hygiene, repository hygiene, supply-chain checker, and an
    independent adversarial review all pass. The F11 PLAN/handoff closure is a
    separate documentation commit.

Any missed tracked CPM URL, wrong count, branch archive, absent/malformed/
comment-only hash, weakened Arrow/action check, failed fresh configure/build,
skipped subject, or canonical regression is **FALSIFIED**. There are at most
two implementation repair attempts. The commit, URL, digest, seven-declaration
inventory, same-block rule, and mutation requirements do not move after the
first decisive execution.

## Rollback and expected-risk claim

The implementation rollback will be the single F11 checker/example/test
commit. The claim most expected to be wrong is that a fresh nested CPM
configure can consume the rc1 archive with every optional dependency disabled;
the archive bytes and API compatibility are confirmed, but that exact
standalone configure route has not yet run.
