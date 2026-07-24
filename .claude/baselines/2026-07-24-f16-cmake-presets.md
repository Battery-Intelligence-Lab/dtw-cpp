# F16 portable CMake preset contract - 2026-07-24

## Scope and base

- Base: `07f186ad7976191aca65271b8c7674d69a84d877`
  (`docs: close F15 deterministic test support`).
- `git status --short` produced no output. Ignored probe artifacts are retained
  under `build/f16-preset-preflight-07f186a/`.
- Subject: the tracked CMake preset floor, Windows Clang discovery, host
  applicability, and current user-facing CMake/C++ requirements.
- Out of scope: preset names, shared `${sourceDir}/build`, generator choices,
  build/test topology, the Visual Studio 17 generator, optional-dependency
  defaults, WASM, and the deliberately pinned macOS system compiler.

No preset-specific killed idea exists. This task does not extend F11's killed
hand-written CMake URL grammar. The permanent gate checks fixed scalar
metadata; CMake's own JSON/preset parser and real clean configures remain the
behavioral arbiters.

## Confirmed inherited state

The live metadata probe printed:

```text
F16_INHERITED_METADATA preset=3.21.0 root=3.26 pyproject=3.26 clang_cxx=C:/Program Files/LLVM/bin/clang++.exe verdict=FALSIFIED
```

Verdict: **FALSIFIED [confirmed]**. `CMakePresets.json` claims 3.21.0 while
root `CMakeLists.txt` and `pyproject.toml` require 3.26. Preset schema version 6
itself was introduced after 3.21. The Windows Clang preset names one
developer-install location instead of a compiler discoverable through `PATH`.

The inherited `test_supply_chain_pinning` subject passes only its older
coverage:

```text
All tests passed (20 assertions in 3 test cases)
F16_INHERITED_STATIC_GATE exit=0
```

It therefore does not detect either F16 defect.

### Clean configure controls

The Windows probe removed both installed LLVM directories from `PATH` and
prepended an alternate in-repository junction. The inherited preset still
ignored that location and selected its encoded path:

```text
-- Check for working CXX compiler: C:/Program Files/LLVM/bin/clang++.exe - skipped
-- Check for working C compiler: C:/Program Files/LLVM/bin/clang.exe - skipped
-- Configuring done (31.3s)
-- Generating done (0.2s)
F16_INHERITED_CLANG_CONFIGURE exit=0
CMAKE_CXX_COMPILER:STRING=C:/Program Files/LLVM/bin/clang++.exe
CMAKE_C_COMPILER:FILEPATH=C:/Program Files/LLVM/bin/clang.exe
CMAKE_GENERATOR:INTERNAL=Ninja
```

The WSL Ubuntu 24.04 control already uses the `gcc-linux` PATH token and
configured cleanly:

```text
Preset CMake variables:

  CMAKE_BUILD_TYPE="Release"
  CMAKE_CXX_COMPILER="g++"

-- Check for working CXX compiler: /usr/bin/g++ - skipped
-- Check for working C compiler: /usr/bin/cc - skipped
-- Configuring done (22.2s)
-- Generating done (0.3s)
CMAKE_CXX_COMPILER:STRING=/usr/bin/g++
CMAKE_C_COMPILER:FILEPATH=/usr/bin/cc
CMAKE_GENERATOR:INTERNAL=Ninja
```

Both probes disabled testing, examples, benchmarks, Python, MATLAB, Gurobi,
HiGHS, llfio, Arrow, MPI, CUDA, Metal, and YAML. They exercised the real root
configure/generate route, not a fixture project.

Before implementation, an explicit `-DCMAKE_CXX_COMPILER=clang++` control used
the same sanitized PATH and alternate LLVM-root junction. This proves the
registered token can drive the real root configure on this host:

```text
-- Check for working CXX compiler: C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/bin/clang++.exe - skipped
-- Check for working C compiler: C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/bin/clang.exe - skipped
-- Auto-detected LLVM libomp at C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/lib/libomp.lib
-- Configuring done (32.4s)
-- Generating done (0.3s)
F16_EXPLICIT_TOKEN_CONTROL exit=0
```

The inherited presets are unconditioned. CMake 4.2.3 on Windows and CMake
3.28.3 in WSL both listed all five configure presets, all three build presets,
and both test presets, including host-incompatible choices. Replacing the
absolute Windows path with a bare token without a host condition would let
`clang-win` silently select Linux Clang.

### Environment exclusions

The installed Visual Studio is version 18 only. The unchanged `msvc` preset
requires Visual Studio 17 and the clean probe reported verbatim:

```text
CMake Error at CMakeLists.txt:20 (project):
  Generator

    Visual Studio 17 2022

  could not find any instance of Visual Studio.

F16_INHERITED_MSVC_CONFIGURE exit=1
```

This is `[BLOCKED-ENV]`, not an F16 regression and not authority to change the
widely supported VS 2022 generator. Real macOS execution is also unavailable
on this Windows/WSL host. F16 preserves the existing `/usr/bin/clang{,++}`
macOS ABI safeguard byte-for-byte and makes no new macOS runtime claim.

## Tracked consumers and preserved compatibility

- `.github/workflows/macos-unit.yml` invokes `clang-macos` and then the shared
  `build` directory.
- `README.md`, `benchmarks/mac_metal_benchmarks.md`, and the troubleshooting
  record advertise `clang-macos`; the recorded Homebrew LLVM/HiGHS ABI failure
  requires `/usr/bin/clang++`.
- No tracked live command invokes `clang-win`, `clang-win-debug`, `msvc`, or
  `gcc-linux`, but their names are public IDE/user entry points and remain.
- `scripts/slurm/slurm_remote.sh` uploads `CMakePresets.json`; ARC configures
  through explicit `-S/-B` arguments and is behaviorally unchanged.
- Ignored `CMakeUserPresets.json` remains the home for personal absolute
  compiler locations.

Current user-facing floor drift is present in:

```text
docs/content/getting-started/installation.md: CMake 3.21, C++17
docs/content/getting-started/matlab.md: CMake 3.15+, C++17
docs/content/getting-started/mpi-cuda-setup.md: CMake 3.21+
```

The active build requires CMake 3.26 and C++20. Historical CHANGELOG release
prose is provenance and will not be rewritten.

## Registered implementation boundary

Attempt 1 may make only these behavioral changes:

1. set preset `cmakeMinimumRequired` to exactly 3.26.0;
2. set `clang-win`'s complete C++ compiler scalar to exactly `clang++` and
   describe PATH discovery;
3. condition Windows configure/build/test presets on `hostSystemName ==
   Windows`, `gcc-linux` on `Linux`, and macOS configure/build/test presets on
   `Darwin`;
4. update the three current user-facing CMake/C++ floor statements;
5. extend the existing native build-metadata subject and CTest metadata. Do
   not add a test target or CMake manifest.

The static guard must bind one active preset-floor object, root's first active
minimum command, scikit-build's floor, the complete Windows compiler scalar,
the absence of a hidden/default compiler override, exact host conditions, the
macOS system compiler exception, and the unchanged preset references. CMake
configuration must independently parse the JSON with `string(JSON)`.

The permanent direct marker is:

```text
F16_CMAKE_PRESETS floor=3.26.0 compiler=clang++ host_conditions=ran metadata_guard=ran skips=0
```

CTest must clear the inherited skip return code, reject a boundary-aware skip
token, require that marker, and require at least 35 assertions in at least 4
test cases. No Python runtime may be required by this native gate.

## Registered mutation matrix

Each mutation runs alone against restored attempt-1 source:

1. preset minor 26 to 25;
2. root, preset, and pyproject floors all 26 to 25, proving equality cannot
   lower the hard 3.26 floor;
3. pyproject floor 26 to 25;
4. `clang++` to `D:/some-machine/LLVM/bin/clang++.exe`;
5. add a duplicate `CMAKE_CXX_COMPILER` scalar beside the valid token;
6. add an inherited absolute compiler scalar to hidden `default`;
7. remove the Windows configure condition;
8. remove one Windows build/test condition;
9. replace the macOS `/usr/bin/clang++` pin with `clang++`;
10. make the preset JSON syntactically invalid while leaving sentinel text.

M01-M09 must make the already-built direct native subject exit nonzero.
M10 must make both CMake's own `--list-presets=all` parser and the existing
canonical configure (which executes the `string(JSON)` guard) exit nonzero.
Restore exact bytes before the next mutation. All 10/10 must be killed.

## Implementation attempt ledger

### Attempt 1 - FALSIFIED

Command:

```text
cmake --build build/highs-1151 --target test_supply_chain_pinning
```

CMake's preset JSON and host inventory had already parsed successfully, but
the configure-time root-floor comparison failed before compilation:

```text
CMake Error at tests/CMakeLists.txt:105 (message):
  F16 CMake floor drift: preset=3.26.0, root=3.14, expected=3.26.0

ninja: error: rebuilding 'build.ninja': subcommand failed
```

Verdict: **FALSIFIED [confirmed]**. No native test or assertion/case band ran.
The guard incorrectly treated late `CMAKE_MINIMUM_REQUIRED_VERSION` as the
root's immutable floor; after configured dependencies, its observed value was
3.14. Attempt 2 may only replace that observation with the already-registered
root first-command invariant. No preset value, host condition, marker,
mutation, configure, or acceptance band changes.

## Acceptance band

F16 passes only if:

1. at most two implementation attempts are used and no registered band is
   relaxed;
2. the direct and CTest metadata subjects print the exact marker, execute at
   least 35 assertions / 4 cases, have zero skips, and exit 0;
3. Windows `--list-presets=all` exposes exactly configure
   `clang-win`, `clang-win-debug`, `msvc`; build `clang-win`,
   `clang-win-debug`; test `clang-win`;
4. WSL Linux exposes exactly configure `gcc-linux` and no build/test preset;
   wrong-host `gcc-linux` on Windows and `clang-win` on WSL both exit nonzero
   with `Could not use disabled preset`;
5. fresh Windows `clang-win` Release and `clang-win-debug` Debug configures,
   with installed LLVM paths removed from `PATH` and only the alternate
   in-repository LLVM junction supplied, both exit 0, print `Configuring done`
   and `Generating done`, select Ninja and the requested build type, and cache
   C/C++ compilers below that alternate junction rather than `Program Files`;
6. a fresh Windows `clang-win` configure with every LLVM location removed
   from `PATH` exits nonzero and explicitly names missing `clang++`; no fallback
   compiler is accepted;
7. fresh WSL `gcc-linux` configures with all optional capabilities/build
   surfaces off, exits 0, prints `Configuring done` and `Generating done`, and
   caches `/usr/bin/g++`, `/usr/bin/cc`, Ninja, and Release;
8. all 10 mutations fail their registered discriminator;
9. supply-chain inventories remain 39 workflow actions, 7 archives, one Arrow
   pin, and 27 tracked CMake manifests; its 63-test Python suite passes;
10. fresh canonical, llfio-OFF, and Arrow-ON gates remain 119/119, 119/119,
    and 121/121 with 6/9/8 capability skips;
11. `git diff --check` is clean, preset names/binary directory/generators and
    the macOS compiler pin are unchanged, and no unrelated build design enters
    the commit.

Rollback is a local revert of the dedicated F16 implementation commit and its
separate registration/closure docs commits. No push, tag, publication, SSH,
or HPC action is authorized.

The claim most likely to be wrong is portability beyond the two executable
hosts. Windows alternate-PATH and WSL probes validate discovery semantics;
host conditions plus the preserved macOS system path are source-confirmed,
but only a real macOS run would confirm that platform again.

## Attempt 2 - portable repair retained, full band FALSIFIED

Attempt 2 changed only the root-floor observation permitted by attempt 1:
the configure-time check now reads the root's exact first command rather than
the mutable late `CMAKE_MINIMUM_REQUIRED_VERSION`. The focused direct and real
CTest subjects then printed:

```text
F16_CMAKE_PRESETS floor=3.26.0 compiler=clang++ host_conditions=ran metadata_guard=ran skips=0
All tests passed (81 assertions in 4 test cases)
F16_POST_MUTATION_CTEST exit=0
```

This exceeds the registered 35-assertion / 4-case floor with zero skips.
`git diff --check` was clean. The implementation was committed independently
as `7aef30da81aadbdfaeb9962b186f6d70aca4f7be`
(`fix: make CMake presets portable`).

The first post-mutation CTest selector incorrectly used
`^supply_chain_pinning$`, exited 0, and printed:

```text
No tests were found!!!
```

That wrapper is **INVALID [confirmed]** and supplies no evidence. `ctest -N`
identified the real subject as test 66, `test_supply_chain_pinning`; the
corrected exact selector produced the marker and 81/4 summary quoted above.

### Host-aware preset inventory

Windows exposed only the three Windows configure presets (`clang-win`,
`clang-win-debug`, `msvc`), the two Windows build presets (`clang-win`,
`clang-win-debug`), and the `clang-win` test preset:

```text
F16_PRESET_LIST host=windows configure=3 build=2 test=1 exit=0
```

The wrong-host Linux preset failed on Windows:

```text
CMake Error: Could not use disabled preset "gcc-linux"
```

WSL exposed only the `gcc-linux` configure preset and no build or test preset.
Its wrong-host Windows preset failed:

```text
F16_PRESET_LIST_host_linux_configure_1_build_0_test_0_exit_0
F16_WRONG_HOST_host_linux_preset_clang-win_exit_1
CMake Error: Could not use disabled preset "clang-win"
```

Verdict: **PASS [confirmed]** for the exact host inventories and both
wrong-host discriminators.

### Clean compiler-discovery probes

Fresh Release and Debug Windows configures removed installed LLVM roots from
`PATH` and exposed only the alternate in-repository LLVM junction. Both
subjects exited 0 and printed `Configuring done` and `Generating done`. Their
cache evidence was:

```text
F16_CACHE build/f16-preset-preflight-07f186a/attempt2-clang-release
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_CXX_COMPILER:STRING=C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/bin/clang++.exe
CMAKE_C_COMPILER:FILEPATH=C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/bin/clang.exe
OpenMP_omp_LIBRARY:FILEPATH=C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/lib/libomp.lib
CMAKE_GENERATOR:INTERNAL=Ninja
F16_CACHE build/f16-preset-preflight-07f186a/attempt2-clang-debug
CMAKE_BUILD_TYPE:STRING=Debug
CMAKE_CXX_COMPILER:STRING=C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/bin/clang++.exe
CMAKE_C_COMPILER:FILEPATH=C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/bin/clang.exe
OpenMP_omp_LIBRARY:FILEPATH=C:/D/git/dtw-cpp/build/f16-preset-preflight-07f186a/toolchain/llvm-alt/lib/libomp.lib
CMAKE_GENERATOR:INTERNAL=Ninja
```

Verdict: **PASS [confirmed]** for the registered compiler/libomp fields. Both
caches selected C, C++, and `OpenMP_omp_LIBRARY` below the alternate junction,
not `Program Files`. CMake still discovered auxiliary LLVM tools such as
`llvm-ar`, `llvm-ranlib`, and `clang-scan-deps` under the installed
`Program Files` tree; F16 did not register or claim relocation of those tools.

With every LLVM location removed, the two real fresh configure subjects both
exited 1 and printed:

```text
The CMAKE_CXX_COMPILER:
  clang++
is not a full path and was not found in the PATH.
```

They also printed `No CMAKE_C_COMPILER could be found.`, while both partial
caches retained `CMAKE_CXX_COMPILER:UNINITIALIZED=clang++`. The two PowerShell
wrapper classifiers themselves printed `diagnostic=False` because redirected
native stderr was represented as PowerShell error records rather than captured
in the wrapper's `$output` string. Those wrapper predicates are
**FALSIFIED [confirmed]**; the directly observed CMake subjects nevertheless
meet the registered negative condition. No third wrapper was attempted.

The first WSL wrapper was **INVALID [confirmed]** before the subject ran:

```text
bash: -c: line 1: syntax error near unexpected token '('
```

Its unquoted grep expression lost shell quoting through `wsl.exe`. The second
and final wrapper ran the fresh subject, exited 0, and printed `Configuring
done (31.0s)` and `Generating done (0.4s)`. Its cache is:

```text
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_CXX_COMPILER:STRING=/usr/bin/g++
CMAKE_C_COMPILER:FILEPATH=/usr/bin/cc
CMAKE_GENERATOR:INTERNAL=Ninja
```

Verdict: **PASS [confirmed]** for the real Windows and WSL compiler-discovery
subjects; the failed wrapper predicates remain recorded and are not evidence.

### Registered mutation verdicts

Each M01-M09 mutation ran alone against the already-built real native subject,
returned Catch2 exit 42, and was inverted immediately. Restoration checks
after M02, corrected M09, and M10 reproduced the registered source diff hash
`f1fd6d182e8059537af3eb363ff00af4a3bd2d30`; the final post-mutation focused
subject also passed.

M09's first inverse patch matched the earlier Windows `clang++` scalar instead
of the mutated macOS scalar. The restoration hash exposed
`7b40a8103c42cf674e2a7d68cff19615b7ce4df9`; a contextual inverse restored the
registered hash before M10. No test or configure subject ran against that
intermediate mismatch.

| Mutation | Direct result |
|---|---:|
| M01 preset floor 26 -> 25 | 42 |
| M02 root+preset+pyproject 26 -> 25 | 42 |
| M03 pyproject floor 26 -> 25 | 42 |
| M04 Windows compiler -> drive-absolute path | 42 |
| M05 duplicate compiler scalar | 42 |
| M06 inherited hidden-default compiler | 42 |
| M07 remove Windows configure condition | 42 |
| M08 remove Windows build condition | 42 |
| M09 remove the macOS system-compiler pin | 42 |

M10 appended a comma after the complete top-level JSON value while retaining
all static sentinels. CMake's own preset parser correctly failed:

```text
CMake Error: Could not read presets from C:/D/git/dtw-cpp:
CMakePresets.json:147: Extra non-whitespace after JSON value.
},
 ^
F16_MUTATION_M10_LIST exit=1 expected=nonzero
```

Before the decisive separated commands, a combined list/configure wrapper
timed out with `command timed out after 24022 milliseconds` without yielding
subject output. The first separated configure wrapper also timed out with
`command timed out after 23428 milliseconds`; no child CMake process remained.
Both wrappers are **INVALID [confirmed]** and are not used as evidence. The
same independent configure-time subject under the final 60-second wrapper
completed and did not fail:

```text
-- Configuring done (19.4s)
-- Generating done (3.0s)
-- Build files have been written to: C:/D/git/dtw-cpp/build/highs-1151
F16_MUTATION_M10_CONFIGURE exit=0 expected=nonzero
```

Verdict: **FALSIFIED [confirmed]**. CMake's `string(JSON)` accepted a valid
leading JSON value plus trailing non-whitespace, so it is not a fail-closed
whole-document syntax arbiter. M01-M09 are red and CMake's real preset parser
is red, but the registered requirement that both M10 subjects fail is unmet.
The two-attempt cap forbids a rescue patch; no registered band was relaxed.

An independent source review also found that a future top-level preset
`toolchainFile` containing a UNC path could evade the current cache-variable
and drive-letter checks. That mutation was not part of the registered F16
matrix and was not executed after the attempt cap. It is therefore
**[inferred]**, not closure evidence; F38 owns both fail-closed preset-metadata
residuals.

### Supply-chain and full-suite regression gates

The supply-chain subjects at implementation commit `7aef30d` printed:

```text
63 passed in 0.23s
WORKFLOW_ACTION_PIN_GATE verified=39 total=39 verdict=PASS
CMAKE_ARCHIVE_PIN_GATE verified=7 total=7 mutable=0 unhashed=0 verdict=PASS
ARROW_ARCHIVE_PIN_GATE verified=1 total=1 verdict=PASS
TRACKED_CMAKE_MANIFESTS total=27
supply-chain pins verified
F16_SUPPLY pytest_exit=0 checker_exit=0
```

The three configured build gates then passed:

| Gate | Result | Capability skips |
|---|---:|---:|
| canonical `build/highs-1151` | 119/119, 0 failed | 6 |
| llfio-OFF `build/nollfio` | 119/119, 0 failed | 9 |
| Arrow-ON `build/arrow-pyarrow-23` | 121/121, 0 failed | 8 |

The canonical six were CUDA x2, Arrow reader x1, and Metal x3. The llfio-OFF
nine additionally skipped mmap x2 and HiGHS/Benders x1. The Arrow-ON eight
ran the Arrow reader and all three integration entries, and skipped mmap x2,
CUDA x2, Metal x3, and HiGHS/Benders x1. All builds and CTest invocations
exited 0. The only reported llfio-OFF/Arrow build warning was:

```text
clang++: warning: optimization flag '-fno-signaling-nans' is not supported [-Wignored-optimization-argument]
```

### Final verdict

The user-facing F16 repair is **retained [confirmed]** in `7aef30d`: the
developer-absolute Windows compiler path is gone, all three active CMake floors
are 3.26, host-specific preset visibility is enforced, alternate-PATH Windows
and WSL clean configures pass, the missing-compiler probe fails loudly, and
all regression inventories hold.

Formal F16 closure is **FALSIFIED [confirmed]** because registered M10 did not
make the configure-time `string(JSON)` subject fail. Per the binding
two-attempt rule, leave F16 open, route replacement fail-closed metadata
architecture uniquely to F38, and continue at F17. The unchecked F16 box
preserves the falsified acceptance state but owns no further implementation.
Rollback remains a local revert of `7aef30d` plus the separate F16
documentation commits; no remote action was taken.

The claim most likely to be wrong is unchanged: macOS portability is
source-confirmed but not runtime-confirmed on this Windows/WSL host. The
specific additional claim most likely to be wrong is that F38 needs a new
architecture rather than a narrowly documented official-parser invocation;
an executed F38 mutation gate would decide that.
