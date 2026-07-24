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
