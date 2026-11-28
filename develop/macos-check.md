# macOS / Apple Silicon & Metal Verification Handoff (v2.0.0)

Run this checklist once on an Apple Silicon Mac from a clean checkout of `v2.0-fast`.
It verifies everything that cannot be compiled or executed on Windows/Linux (Objective-C++ `metal_dtw.mm`, runtime MSL shader compilation and execution, ARM64 NEON SIMD lanes and AppleClang codegen, strict-FP conformance, and the macOS `cp312-abi3` wheel).

Everything on Windows Clang 21, Windows MSVC + CUDA 13, WSL GCC 13, and WSL Clang 18 is already green through `cffbd58f`.

**Rules for every step:**
- **Never trust a silent skip.** CTest counts Catch2 `SKIP` as a pass. Where a step says **RAN**, confirm the test names and assertion counts appear with zero `SKIPPED` lines.
- **Keep working tree clean.** All scratch build/output directories below live under `build*/` or `.venv-mac/` (all git-ignored).
- **No attribution footers** in code, comments, or commit messages.

---

## What Changed Since the Last Mac Run

If anything fails, these are the exact macOS-reachable changes being compiled and run on Apple hardware for the first time:

1. **Metal kernels & zero-copy host path (`dtwc/metal/metal_dtw.mm`, `dtwc/metal/metal_dtw.hpp`, `dtwc/Problem.cpp`):**
   - `e2246edc`: Dynamic `pairs_per_tg` passed via `[[buffer(9)]]` to `dtw_regtile_w4` / `dtw_regtile_w8` (replaces hardcoded `PAIRS_PER_TG = 8` so hardware `maxTotalThreadsPerThreadgroup` / threadgroup memory clamping cannot skip pairs).
   - `e2246edc`: Zero-copy `compute_distance_matrix_metal(n_series, SeriesAt, opts, ParallelFor, out)` overload called directly from `dtwc/Problem.cpp:1506` with OpenMP FP64→FP32 staging and FP32→FP64 output widening (no owned copy of views).
   - `e2246edc`: Double-buffered `scratch[2]` for multi-chunk `wavefront_global` and `banded_row` dispatches with per-command-buffer error checking.
   - `bc30b29f` (CRIT-5): Early exit for `|La - Lb| > band` in `dtw_wavefront_body` and `dtw_banded_row` (tested directly in Step 3).
2. **Unified GPU policy & CLI/streaming paths (`dtwc/core/gpu_policy.hpp`, `dtwc/cdist.cpp`, `dtwc/cli/run.cpp`, `dtwc/algorithms/detail/fast_clara_stream.hpp`, `dtwc/algorithms/fast_clara.cpp`):**
   - Shared `kGpuBackendName = "Metal"` capability checks in `dtwc/core/gpu_policy.hpp`.
   - `cdist` / `cdist_f32` Metal refusal (`"cross-collection rectangular cdist is not implemented on Metal"`) and `--fallback` CPU routing.
   - `gpu_staging_bytes` Metal accounting in `dtwc/cli/run.cpp:185-201`; `on_gpu = false` in `fast_clara_stream.hpp` (Metal assigns CLARA medoids on CPU).
   - `cffbd58f`: Generic lambdas extracted to plain function templates in `fast_clara_stream.hpp` and `fast_clara.cpp`.
3. **ARM64 NEON CPU kernels & codegen (`8799f44f`, `cffbd58f`):**
   - `dtwc/core/dtw_kernel.hpp:611-654`: `LanesCell` using `std::fmin` (`fminnm`), `dtw_lanes<double> = 16` / `dtw_lanes<float> = 32`, 2-column blocking, expanded SIMD lanes (`ADTW`, `WDTW`, `ZeroCost`, `AROW`), branchless MSM (`dtwc/core/msm.hpp`), peeled TWE (`dtwc/core/twe.hpp`), Soft-DTW backward gradient.
   - `test_codegen_no_calls` (`scripts/codegen_report.py --no-calls`) runs on AppleClang + Ninja Release to confirm zero stray function calls inside any DP loop.

---

## Step 0 — Prerequisites & Clean Checkout

```bash
git fetch origin
git checkout v2.0-fast
git pull --ff-only origin v2.0-fast
git log -n 3 --oneline
```

Verify tools (Xcode Command Line Tools, Homebrew `cmake`, `ninja`, `libomp`, and `uv`):

```bash
xcode-select -p
brew install cmake ninja libomp uv
sw_vers && clang++ --version | head -n 1
```

> **Note:** The `clang-macos` preset explicitly pins `/usr/bin/clang` and `/usr/bin/clang++` (Apple Clang) so Homebrew LLVM cannot shadow it and cause a `libc++` mismatch with HiGHS. Passing `OpenMP_ROOT="$(brew --prefix libomp)"` lets CMake and `uv build` find keg-only Homebrew `libomp` without needing `brew link --force libomp`.

---

## Step 1 — Configure & Build (Release + HiGHS + Metal)

Metal (`DTWC_ENABLE_METAL`) defaults to `ON` on macOS.

```bash
cmake --preset clang-macos \
  -DDTWC_BUILD_TESTING=ON \
  -DDTWC_ENABLE_HIGHS=ON \
  -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build -j
```

Confirm CMake configured with Metal enabled (`DTWC_HAS_METAL` active) and zero build errors.

If a build error occurs, check `dtwc/metal/metal_dtw.mm` first (every C++ translation unit already passed `clang++ -fsyntax-only -DDTWC_HAS_METAL` off-Mac, so compile errors are most likely in Objective-C++ syntax or AppleClang `-fno-objc-arc` handling). Runtime MSL shader errors do not fail the build — they cause `Metal backend is available` to `SKIP` in Step 2, which counts as a failure.

---

## Step 2 — Full CTest Suite & Proof Every Metal/NEON Gate RAN

Run the full CTest suite serially (GPU tests share one Metal device):

```bash
ctest --test-dir build -C Release --output-on-failure
```

Every CTest entry must pass. Then explicitly verify that Metal, zero-copy GPU views, Apple Silicon `cpuinfo`/NEON, ARM64 codegen, CLI device matrix, and conformance actually **RAN** without skipping:

```bash
./build/bin/dtwc_unit_gpu "[metal]" -d yes
./build/bin/dtwc_unit_gpu "[gpu]" -d yes
./build/bin/dtwc_unit_core "[machine]" -d yes
ctest --test-dir build -C Release -R "^(test_codegen_no_calls|test_cli_device_matrix|cpp_conformance)$" -V
```

**Required outcomes:**
- `dtwc_unit_gpu "[metal]" -d yes`:
  - `Metal backend is available`, `test_metal_correctness` (`regtile_w4`, `regtile_w8`, `wavefront`, `wavefront_global`, `banded_row`, `float32`), `test_metal_mmap` (dense, mmap, squared-L2, float32 mmap, `fast_clara`), and `FX-1: Metal rejects a GPU index and a precision it cannot honour` all **RAN**.
  - **Zero `SKIPPED` lines.** Ends with `All tests passed (N assertions in M test cases)`.
- `dtwc_unit_gpu "[gpu]" -d yes`:
  - `A view on a GPU fills the matrix the owned store does` **RAN** and passed (exercises the new zero-copy `SeriesAt` + OpenMP Metal overload in `dtwc/Problem.cpp:1506`).
- `dtwc_unit_core "[machine]" -d yes`:
  - Passes with Apple CPU model and `NEON` detected (`cpuinfo` enabled).
- `test_codegen_no_calls`:
  - Prints `CODEGEN_NO_CALLS ... missing=none calls=0 verdict=PASS`.
- `test_cli_device_matrix`:
  - Prints `gpu=metal device=present ... skips=0` (verifies CLI PAM/CLARA/KMedoids/LR-core/Hierarchical/MIP on Metal, `gpu:1` and `cpu+gpu` refusals, and CLARA's `"medoids on the CPU (Metal has no kernel for the assignment)"` message).
- `cpp_conformance`: passes.

---

## Step 3 — Add & Verify the CRIT-5 Metal Infeasible-Band Regression Test (`bc30b29f`)

`dtw_banded_row` and `dtw_wavefront_body` previously lacked an early exit for `|La - Lb| > band`, causing an out-of-bounds write before the scratch buffer (when the longer series had the lower index) or a stale finite cost. The existing fixed-band test puts the long filler last, so it misses this trigger.

Insert the following `TEST_CASE` into `tests/unit/test_metal_correctness.cpp` **immediately above its final line `#endif // DTWC_HAS_METAL`** (placing it below `#endif` breaks non-Metal builds):

```cpp
TEST_CASE("Metal gives no path for a pair whose length gap exceeds the band with the longer series first",
          "[metal][banded]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");

  // The longer series at the lower index: banded_row wrote before its scratch buffer and the wavefront returned a
  // stale finite cost. Every banded route must give the CPU's no-path value. banded_row takes a band of at most 512
  // with band * 20 < max_L; otherwise the threadgroup wavefront, unless its 3 * max_L floats exceed the 32 KB of
  // threadgroup memory, where wavefront_global takes the batch.
  struct Route {
    std::size_t longer;
    int band;
    const char *kernel_name;
  };
  const std::vector<double> shorter{ 0, 1, 0, 2, 0 };
  for (const auto &route : { Route{ 12, 1, "wavefront" }, Route{ 64, 1, "banded_row" },
                             Route{ 8192, 600, "wavefront_global" } }) {
    const std::vector<std::vector<double>> series{ fixed_band::filler(route.longer), shorter };
    dtwc::metal::MetalDistMatOptions opts;
    opts.band = route.band;
    const auto gpu = gpu_fill(series, opts);
    const double cpu = dtwc::dtwBanded(series[0], series[1], route.band);
    INFO("kernel=" << gpu.kernel_used << " longer=" << route.longer << " band=" << route.band);
    REQUIRE(gpu.kernel_used == route.kernel_name);
    REQUIRE(cpu == fixed_band::public_no_path_sentinel);
    REQUIRE(gpu.matrix[1] == cpu);
    REQUIRE(gpu.matrix[2] == cpu);
  }
}
```

Rebuild and run both the banded filter and the full `[metal]` filter (record the final `[metal]` assertion/case counts for Step 9):

```bash
cmake --build build -j
./build/bin/dtwc_unit_gpu "[metal][banded]" -d yes
./build/bin/dtwc_unit_gpu "[metal]" -d yes
```

*(Optional negative proof, takes ~15 s)* Verify the new test actually catches the bug when `bc30b29f` is absent (`bc30b29f` reverts cleanly on `cffbd58f`):

```bash
git revert --no-commit bc30b29f
cmake --build build -j
./build/bin/dtwc_unit_gpu "[metal][banded]"   # MUST FAIL on the new case
git revert --abort                            # restores metal_dtw.mm, keeps the uncommitted test case
cmake --build build -j
./build/bin/dtwc_unit_gpu "[metal][banded]" -d yes   # MUST PASS again
```

> Use `git revert --abort` (not `git checkout -- dtwc/metal/metal_dtw.mm`), otherwise `REVERT_HEAD` stays behind and turns the Step 9 commit into a revert.

---

## Step 4 — End-to-End CLI Parity (Metal GPU vs CPU) & System Diagnostics

All outputs go under `build/mac-check` (git-ignored):

```bash
o=build/mac-check
rm -rf "$o"
for m in pam clara; do
  ./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 -m "$m" --device cpu -o "$o/cpu_$m" --name r
  ./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 -m "$m" --device gpu -o "$o/gpu_$m" --name r
  cmp "$o/cpu_$m/r_labels.csv" "$o/gpu_$m/r_labels.csv" \
    && cmp "$o/cpu_$m/r_medoids.csv" "$o/gpu_$m/r_medoids.csv" \
    && echo "$m: labels and medoids equal"
done
```

Metal computes in FP32, so the distance matrices differ in the last digits, while labels and medoids on `clara_256.arrow` must be byte-identical.

Check verbose `System Diagnostics` and `--fallback`:

```bash
./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 --device gpu -o "$o/diag" --name r -v
./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 --device gpu --variant wdtw --fallback -o "$o/fb" --name r
```

Confirm:
- `System Diagnostics` names the Apple Silicon chip, prints `SIMD: NEON (CPU), NEON (build)`, and prints `GPU: Metal: ...`.
- The `--fallback` run exits `0` and emits a single `[dtwc] gpu requested: ...; running on cpu (fallback)` line on stderr.

---

## Step 5 — Strict-FP Conformance & `DTWC_DEV_MODE=ON` (`-Werror`)

1. **Strict-FP conformance** (mirrors `.github/workflows/macos-unit.yml`'s `conformance-strict` job):

```bash
cmake --preset clang-macos -B build-strict \
  -DDTWC_BUILD_TESTING=ON \
  -DDTWC_ENABLE_HIGHS=ON \
  -DDTWC_FP_MODEL=strict \
  -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build-strict --target cpp_conformance -j
ctest --test-dir build-strict -C Release -R '^cpp_conformance$' --output-on-failure
```

2. **Dev-mode `-Werror` check** (verifies AppleClang + Objective-C++ `metal_dtw.mm` builds warning-free):

```bash
cmake --preset clang-macos -B build-dev \
  -DDTWC_BUILD_TESTING=ON \
  -DDTWC_ENABLE_HIGHS=ON \
  -DDTWC_DEV_MODE=ON \
  -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build-dev --target dtwc_cl dtwc_unit_gpu dtwc_unit_core -j
```

*(You may `rm -rf build-strict build-dev` right after both pass.)*

---

## Step 6 — Python Stable-ABI Wheel (`cp312-abi3`) & Full Pytest Gate

Builds the stable-ABI wheel on Python 3.12 and runs it on Python 3.13 (proving `cp312-abi3` portability, live Metal execution, OpenMP threading, and `highspy` MIP):

```bash
rm -rf .venv-mac build/mac-wheel
uv venv .venv-mac --python 3.13
OpenMP_ROOT="$(brew --prefix libomp)" uv build --wheel --python 3.12 -o build/mac-wheel
uv pip install --python .venv-mac/bin/python build/mac-wheel/*.whl \
  numpy scipy h5py pyarrow scikit-learn matplotlib pandas pytest

# Verify abi3 filename, live Metal, OpenMP, and highspy MIP wiring
.venv-mac/bin/python -c "
import pathlib, highspy, dtwcpp
so = pathlib.Path(dtwcpp._dtwcpp_core.__file__).name
print('ext:', so)
assert so == '_dtwcpp_core.abi3.so', so
assert not dtwcpp.HIGHS_AVAILABLE
print('parallel:', dtwcpp.test.parallelisation())
print('gpu_info:', dtwcpp.gpu_info())
assert dtwcpp.gpu_available(), 'Metal GPU not detected in wheel'
assert dtwcpp.gpu_info().startswith('Metal:'), dtwcpp.gpu_info()
"

# Verify live Metal test explicitly RAN (not skipped)
.venv-mac/bin/python -m pytest tests/python/test_device.py -k "metal" -v

# Full Python + conformance suite (DTWC_REQUIRE_HIGHSPY=1 prevents silent MIP skips)
DTWC_CL_PATH="$PWD/build/bin/dtwc_cl" DTWC_REQUIRE_HIGHSPY=1 \
  .venv-mac/bin/python -m pytest tests/python tests/conformance -q -rs
```

**Expected outcomes:**
- `test_gpu_problem_fills_through_metal` **PASSED** (not skipped).
- Full pytest suite passes (~744+ passed; inspect `-rs` so every skip is legitimate on macOS, e.g. CUDA-only or Gurobi).

---

## Step 7 — Optional Checks (Run if Available on This Mac)

### A. MATLAB MEX (skip if MATLAB is not installed on this Mac)

```bash
cmake -S . -B build-mex \
  -DDTWC_BUILD_MATLAB=ON \
  -DDTWC_BUILD_TESTING=ON \
  -DDTWC_ENABLE_HIGHS=ON \
  -DDTWC_ENABLE_GUROBI=OFF \
  -DDTWC_ENABLE_YAML=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build-mex --target dtwc_mex -j
ctest --test-dir build-mex -C Release -R '^matlab_suite$' --output-on-failure
```

### B. Portable macOS 13.3+ `libomp` (only if testing distributable archive/wheel packaging locally)

Homebrew's `libomp` sets `minos` to the host macOS version. To test the portable runtime used by `release-artifacts.yml` and `python-wheels.yml`, run `bash scripts/build_libomp_macos.sh build/libomp-13.3` and point `-DOpenMP_ROOT="$PWD/build/libomp-13.3"`. Not required for standard local verification.

---

## Step 8 — Troubleshooting Map (Where to Look First)

| Symptom | First place to inspect |
|---|---|
| Compile error in `metal_dtw.mm` | `dtwc/metal/metal_dtw.mm` (built with `-fno-objc-arc`); check lines 716–793 (`compute_distance_matrix_metal` `SeriesAt`/`ParallelFor` overload) and 923–1069 (`pairs_per_tg`, `scratch[2]`, `out.visit`). |
| `Metal backend is available` skips | Check the printed `newLibraryWithSource` MSL compiler error in `dtwc/metal/metal_dtw.mm` lines 55–558 (especially `dtw_regtile_w4` / `dtw_regtile_w8` `[[buffer(9)]]` and `dtw_wavefront_body` / `dtw_banded_row` early returns). |
| `regtile_w4` / `regtile_w8` wrong values or zeros | `dtwc/metal/metal_dtw.mm:479`, `:528`, `:547`, `:930-935`, `:1011-1014` (`pairs_per_tg` binding at `buffer(9)`). |
| `test_codegen_no_calls` fails on a Soft-DTW `bl _exp` / `bl _log` line | `scripts/codegen_report.py:218` strips `#` comments before `libm_target(...)`; AppleClang ARM64 assembly uses `;` comments. If an inline `;` comment trails `bl _exp`, split on `;` as well as `#`. |
| `dtwc_unit_core "[oracle]"` mismatch on ARM64 | `dtwc/core/dtw_kernel.hpp:611-654` (`LanesCell` `std::fmin` and 128-byte AArch64 lanes) and `dtwc/core/msm.hpp` / `dtwc/core/twe.hpp`. |
| `DTWC_DEV_MODE=ON` warning in `metal_dtw.mm` | Likely an implicit float/size conversion (e.g. `cells_budget = 5e9` at line 880) or Objective-C++ warning flag under AppleClang. |
| `OpenMP` not found in CMake or `uv build` | Ensure `OpenMP_ROOT="$(brew --prefix libomp)"` is set and `brew --prefix libomp` points to a valid directory containing `include/omp.h` and `lib/libomp.dylib`. |

---

## Step 9 — Record the Outcome & Push Back to `v2.0-fast`

1. Clean up temporary directories if desired (`rm -rf .venv-mac build-strict build-dev build-mex`).
2. Check `git status` — only `tests/unit/test_metal_correctness.cpp` (modified) and `develop/macos-check.md` (to be removed) should be changed (plus any bugfix commits if Step 1–6 uncovered an issue).
3. **If something failed:** fix it in its own commit with the failing output in the commit body, rerun Steps 1–6 until completely green, and only then make the closing commit below.
4. **Once everything passes:** stage only the Step 3 test case and the removal of this file, recording the exact hardware and gate numbers in the commit body:

```bash
git add tests/unit/test_metal_correctness.cpp
git rm develop/macos-check.md
git commit -m "test(metal): verify infeasible-band guards on all Metal kernels and complete macOS check" -m "<record: Apple chip, macOS/Xcode version, ctest X/X, [metal] N assertions in M cases, [gpu] view parity, codegen_no_calls PASS, strict conformance PASS, DEV_MODE clean, CLI pam+clara GPU=CPU parity, pytest X passed / Y skipped with live Metal>"
```

5. If `origin/v2.0-fast` moved while running on the Mac, rebase and rerun Step 2 before pushing:

```bash
git pull --rebase origin v2.0-fast
git push origin HEAD:v2.0-fast
```
