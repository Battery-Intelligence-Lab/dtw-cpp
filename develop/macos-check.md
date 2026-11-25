# macOS / Metal check before tagging 2.0.0

The Metal backend is compiled only on Apple hardware, and every Metal edit since the design-2.0 work was written on
Windows and Linux without being built. Run these steps once on a Mac (Apple silicon), from a clean checkout of
`v2.0-fast`, then record the outcome as step 7 says and delete this file in that commit.

Rule for every step: a test that is skipped proves nothing (ctest counts a skip as a pass). Where a step says RAN,
show the test's own output with its assertion count and no `SKIPPED` line.

## 1. Build

```bash
brew install ninja libomp
cmake --preset clang-macos -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_HIGHS=ON -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build -j
```

Metal is on by default on Apple. A compile error in `dtwc/metal/metal_dtw.mm` or in a Metal test is the first thing
to fix: those files have never been compiled since the edits listed in step 5. Off the Mac, every C++ translation
unit of a Windows build (512) passes `clang++ -fsyntax-only -DDTWC_HAS_METAL`, so a build error most likely sits in
the Objective-C++ of `metal_dtw.mm`. Its kernels (MSL source) are compiled at run time by `newLibraryWithSource`: a
shader error is no build error, it makes `Metal backend is available` skip with the Metal compiler's message, and
step 2 counts that skip as a failure.

## 2. The full suite

```bash
ctest --test-dir build -C Release --output-on-failure   # serial: the GPU tests share one device
```

Every entry must pass. Then show that the Metal tests RAN:

```bash
./build/bin/dtwc_unit_gpu "[metal]" -d yes
```

`-d yes` names every test case it ran, with its time. `Metal backend is available` and the cases of
`test_metal_correctness` and `test_metal_mmap` must be among them; no `SKIPPED` may appear (a skip prints its reason,
such as the Metal compiler's message or `DTWC_HAS_METAL not defined`, which is what Windows prints: two skipped
cases); and the last line must read `All tests passed (N assertions in M test cases)`.

## 3. The guard for a pair whose length gap exceeds the band (review CRIT-5, commit bc30b29f)

`dtw_banded_row` and the wavefront body had no early exit for `|La - Lb| > band`, which the CUDA kernels have: with
the longer series at the lower index, `banded_row` wrote before its scratch buffer, and the wavefront returned a
stale finite cost. Only a direct backend call reaches it (Problem, the CLI and the bindings refuse such data first).
The existing fixed-band test puts the long filler last, so it misses the trigger. Add this case to
`tests/unit/test_metal_correctness.cpp` just above its last line, `#endif // DTWC_HAS_METAL`: it uses the Metal-only
`gpu_fill` and `fixed_band` helpers defined there, and below that line it would break every build without Metal.

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

Rebuild (`cmake --build build -j`) and run `./build/bin/dtwc_unit_gpu "[metal][banded]" -d yes`: the new case
must pass on all three routes, beside the existing route test. Then rerun `./build/bin/dtwc_unit_gpu "[metal]" -d yes`:
its counts, with the new case, are the ones step 7 records.

Checked off the Mac: the file's Metal branch, with this case, passes `clang++ -fsyntax-only -DDTWC_HAS_METAL`; the
CPU half holds (`dtwBanded` gives the sentinel for the three pairs, in either order); and the kernels named are the
ones `metal_dtw.mm` selects (regtile, the fourth, is unbanded only, where every pair has a path). A failure here is
therefore Metal's, unless the route assertion fails, which means the route selection changed: adjust the lengths or
bands. The test name has no comma, which Catch2 would read as a separator.

If quick, show the case fails without the guards: `git revert --no-commit bc30b29f` (it applies cleanly to this
branch), rebuild, and run `./build/bin/dtwc_unit_gpu "[metal][banded]"`: the new case must fail. Then restore the
guards with `git revert --abort` (it keeps the uncommitted test case and clears the revert state; a `git checkout`
of the file would leave that state behind and turn the step-7 commit into a revert), rebuild, and rerun the same
filter: it must pass before step 4.

## 4. GPU results equal the CPU's

```bash
o=build/mac-check   # under build/, which git ignores
for m in pam clara; do
  ./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 -m $m --device cpu -o $o/cpu_$m --name r
  ./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 -m $m --device gpu -o $o/gpu_$m --name r
  cmp $o/cpu_$m/r_labels.csv $o/gpu_$m/r_labels.csv && cmp $o/cpu_$m/r_medoids.csv $o/gpu_$m/r_medoids.csv \
    && echo "$m: labels and medoids equal"
done
```

Metal computes in float32, so the distance matrices differ in the last digits; labels and medoids must be equal
(CUDA in float32, `--gpu-precision fp32`, gives the CPU's labels and medoids on this file for both methods).

## 5. Metal-reachable code changed without a Metal build

Covered by steps 2 and 4 if they pass; look here first if they fail:

- the Metal route of `Problem`'s fill in `dtwc/Problem.cpp` (a view's owned copy goes to
  `metal::compute_distance_matrix_metal`);
- `gpu_staging_bytes` in `dtwc/cli/run.cpp` (the Metal formula: the floats plus a float32 matrix);
- `dtwc/algorithms/detail/fast_clara_stream.hpp` (`on_gpu` is false on Metal: the assignment runs on the CPU);
- `tests/unit/test_fill_request.cpp` (Auto means float32 on Metal) and `tests/unit/io/test_ipc_stream.cpp`;
- `tests/unit/test_metal_mmap.cpp`.

## 6. The rest of the Mac build

- cpuinfo builds on arm64; `./build/bin/dtwc_unit_core "[machine]" -d yes` passes (with cpuinfo on,
  `tests/unit/core/test_machine.cpp` asserts a known CPU model; 34 assertions in 2 test cases on Windows); and
  `./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 -o build/mac-check/v --name r -v` prints a
  `System Diagnostics` block whose CPU line names the Apple chip and whose SIMD line reads `NEON (CPU), NEON (build)`.
- `cpp_conformance` passes (part of step 2).
- The Python package (highspy comes with it as a dependency; `OpenMP_ROOT` lets the wheel's build find Homebrew's
  libomp, which is keg-only):

  ```bash
  rm -rf .venv-mac build/mac-wheel && uv venv .venv-mac --python 3.13
  OpenMP_ROOT="$(brew --prefix libomp)" uv build --wheel --python 3.12 -o build/mac-wheel
  uv pip install --python .venv-mac/bin/python build/mac-wheel/*.whl numpy scipy h5py pyarrow scikit-learn \
    matplotlib pandas pytest
  .venv-mac/bin/python -c "import dtwcpp, pathlib; n = pathlib.Path(dtwcpp._dtwcpp_core.__file__).name; print(n); assert n == '_dtwcpp_core.abi3.so'"
  DTWC_CL_PATH="$PWD/build/bin/dtwc_cl" .venv-mac/bin/python -m pytest tests/python tests/conformance -q -rs
  ```

  The wheel is built on Python 3.12 and run on 3.13: a stable-ABI wheel (`cp312-abi3`, module `_dtwcpp_core.abi3.so`)
  must serve both. It has been checked on Windows and Linux, not yet on macOS. pytest: 731 passed and 18 skipped on
  Windows; another split is fine where `-rs` shows, for each skip, a reason that holds here.

## 7. Record the outcome

If every step passes: commit the step-3 test case and the removal of this file, staging those two paths only
(`git add tests/unit/test_metal_correctness.cpp && git rm develop/macos-check.md`), as
`test(metal): a pair whose length gap exceeds the band gets no path on every Metal kernel`, noting in the commit body
what ran (the `[metal]` counts with the new case, the step-4 comparison). If something fails: fix it in its own
commit with the failing output in the body, rerun the steps, and only then make the commit above. The branch may
have moved while the steps ran: `git pull --rebase origin v2.0-fast` (rerun step 2 if it brought commits), then push
with `git push origin HEAD:v2.0-fast`, a fast-forward.
