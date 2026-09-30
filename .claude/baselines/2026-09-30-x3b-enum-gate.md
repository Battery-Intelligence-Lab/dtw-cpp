# 2026-09-30 — X3b: `-Werror=switch` / `/we4062` bite; the enum validators' removal changes no result

**Question:** after `validate_<Enum>` and the `default: throw` tails go, does a missed enumerator fail the build
under clang and under MSVC, and does anything observable change? Head `4903d8e` on `e9fffed` (tip of pb/W4d).

**Band, registered before the runs:** with `DummyMutation` appended to `core::DTWVariant`, every switch over it
fails to compile (expected six: `dtw.cpp`, `dtw_dispatch.cpp`, `variant_validation.hpp`, `distance.hpp` twice,
`Problem.cpp` `variant_name`); the unmutated tree builds; serial ctest keeps its 121 names and the 3 MAY_SKIP
skips; `cpp_conformance` passes against the untouched `tests/conformance/conformance_reference.txt`; pytest fails
nothing. **Result: all held.**

## The gate bites `[confirmed]`

clang-win (Release), `cmake --build build --target dtwc++ -- -k 0`, log `wt/X3b-mutation-clang.log`:

    dtwc/core/dtw.cpp:56:11: error: enumeration value 'DummyMutation' not handled in switch [-Werror,-Wswitch]
    dtwc/core/dtw_dispatch.cpp:353:11: ...   dtwc/core\../distance.hpp:196:11: ...   dtwc/core\../distance.hpp:57:11: ...
    dtwc\core/variant_validation.hpp:101:11: ...   dtwc/Problem.cpp:118:11: ...      (same message, six distinct switches)

MSVC 19.50.35723 (Visual Studio 18 2026; the `msvc` preset names "Visual Studio 17 2022", which is not installed here,
so `cmake --preset msvc -B build-msvc -G "Visual Studio 18 2026"`), target `dtwc++`, log `wt/X3b-msvc-mutation.log`:

    dtwc\Problem.cpp(126,3): error C4062: enumerator 'dtwc::core::DTWVariant::DummyMutation' in switch of enum 'dtwc::core::DTWVariant' is not handled
    dtwc\core\dtw.cpp(111,3) / dtw_dispatch.cpp(361,3) / variant_validation.hpp(127,3) / distance.hpp(212,3) and (65,3): the same

The same six switches. The mutation was reverted (`git status` clean); the clean MSVC tree then builds
(`dtwc++.lib`, 0 C4062 in `wt/X3b-msvc-clean-rebuild.log`), as did the first clean build before the mutation.

Not gated: `.cu` (nvcc) and `.mm` (Objective-C++) translation units, because the flag is set for `COMPILE_LANGUAGE:CXX`
only. `cuda_dtw.cu` was compiled once by hand with the W4a `build-cuda` nvcc command line (exit 0, log
`wt/X3b-cuda-syntax.log`); `metal_dtw.*` could not be compiled on Windows.

## No result changes `[confirmed]`

- Serial `ctest -j1` (`Release`), base `e9fffed`: 121 tests, 118 passed, 3 skipped (`test_cuda_correctness`,
  `test_metal_correctness`, `test_metal_mmap`). Head: the same 121 names, the same 3 skips, 0 failed
  (`wt/X3b-base-ctest.log`, `wt/X3b-head-ctest2.log`). No ctest name was added or removed.
- Catch2 cases removed (they only fed integers cast to an enum): `unit_test_invalid_distance_enums` 9 → 1,
  `unit_test_invalid_public_selectors` 10 → 5, `unit_test_problem_encapsulation` 3 → 2 (its one remaining
  rejection was of a cast enum); one section of `test_problem_metric` and one assertion block of
  `unit_test_distance_semantics`.
- `cpp_conformance` passed; the reference file is untouched.
- Python, fresh wheel from the head tree: 1103 passed, 19 skipped, 0 failed (`wt/X3b-head-pytest.log`); 1122 collected.
  `tests/python/` is byte-identical to the base, so its 1122 collected tests are the base's. The base suite was not
  run. `test_invalid_distance_enums.py` (40 cases) is the nanobind pin: out-of-range enums and raw ints stay rejected.
- `check_docs.py` PASS, `check_pins.py` 0 failures, `generate_docs.py --check` current.
