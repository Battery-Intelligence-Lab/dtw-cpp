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
unit of a Windows build (513) passes `clang++ -fsyntax-only -DDTWC_HAS_METAL`, so a build error most likely sits in
the Objective-C++ of `metal_dtw.mm`. Its kernels (MSL source) are compiled at run time by `newLibraryWithSource`: a
shader error is no build error, it makes `Metal backend is available` skip with the Metal compiler's message, and
step 2 counts that skip as a failure.

## 2. The full suite

```bash
ctest --test-dir build -C Release --output-on-failure   # serial: the GPU tests share one device
```

Every entry must pass. Then show that the Metal tests RAN:

```bash
./build/bin/dtwc_unit_gpu "[metal]" --success | tail -3
```

`Metal backend is available` must pass (not skip), and `test_metal_correctness` and `test_metal_mmap` must report
assertions, not `DTWC_HAS_METAL not defined`.

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
  // stale finite cost. Both must give the CPU's no-path value. At band 1, banded_row takes a batch whose longest
  // series has more than 20 samples (band * 20 < max_L), the threadgroup wavefront a shorter one.
  struct Route {
    std::size_t longer;
    const char *kernel_name;
  };
  const std::vector<double> shorter{ 0, 1, 0, 2, 0 };
  for (const auto &route : { Route{ 12, "wavefront" }, Route{ 64, "banded_row" } }) {
    const std::vector<std::vector<double>> series{ fixed_band::filler(route.longer), shorter };
    dtwc::metal::MetalDistMatOptions opts;
    opts.band = 1;
    const auto gpu = gpu_fill(series, opts);
    const double cpu = dtwc::dtwBanded<double>(series[0], series[1], 1);
    INFO("kernel=" << gpu.kernel_used << " longer=" << route.longer);
    REQUIRE(gpu.kernel_used == route.kernel_name);
    REQUIRE(cpu == fixed_band::public_no_path_sentinel);
    REQUIRE(gpu.matrix[1] == cpu);
    REQUIRE(gpu.matrix[2] == cpu);
  }
}
```

Rebuild (`cmake --build build -j`) and run `./build/bin/dtwc_unit_gpu "[metal][banded]" --success`: the new case
must pass for both lengths, beside the existing route test.

Checked off the Mac: the file's Metal branch, with this case, passes `clang++ -fsyntax-only -DDTWC_HAS_METAL`; the
CPU half holds (`dtwBanded` gives the sentinel for both lengths, in either order); and the kernels named are the
ones `metal_dtw.mm` selects. A failure here is therefore Metal's, unless the route assertion fails, which means the
route selection changed: adjust the lengths. The test name has no comma, which Catch2 would read as a separator.

If quick, show the case fails without the guards: `git revert --no-commit bc30b29f` (it applies cleanly to this
branch), rebuild, and run `./build/bin/dtwc_unit_gpu "[metal][banded]"`: the new case must fail. Then restore the
guards with `git revert --abort` (it keeps the uncommitted test case and clears the revert state; a `git checkout`
of the file would leave that state behind and turn the step-7 commit into a revert), rebuild, and rerun the same
filter: it must pass before step 4.

## 4. GPU results equal the CPU's

```bash
for m in pam clara; do
  ./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 -m $m --device cpu -o out_cpu_$m --name r
  ./build/bin/dtwc_cl -i tests/data/reader/clara_256.arrow -k 4 -m $m --device gpu -o out_gpu_$m --name r
  cmp out_cpu_$m/r_labels.csv out_gpu_$m/r_labels.csv && cmp out_cpu_$m/r_medoids.csv out_gpu_$m/r_medoids.csv
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

- cpuinfo builds on arm64; `./build/bin/dtwc_unit_core "[machine]"` passes (with cpuinfo on,
  `tests/unit/core/test_machine.cpp` asserts a known CPU model; 34 assertions in 2 test cases on Windows); and
  `./build/bin/dtwc_cl -v ...` prints sensible machine lines.
- `cpp_conformance` passes (part of step 2).
- The Python package: `uv venv .venv-mac && uv pip install --python .venv-mac/bin/python . numpy scipy h5py pyarrow
  scikit-learn matplotlib pandas pytest`, then
  `DTWC_CL_PATH="$PWD/build/bin/dtwc_cl" .venv-mac/bin/python -m pytest tests/python tests/conformance -q`
  (774 passed, 19 skipped on Windows; a different skip count is fine if each skip names a reason that holds here).

## 7. Record the outcome

If every step passes: add the step-3 test case, delete this file, and commit both as
`test(metal): a pair whose length gap exceeds the band gets no path on every Metal kernel`, noting in the commit body
what ran (assertion counts of `[metal]`, the step-4 comparison). If something fails: fix it in its own commit with the
failing output in the body, rerun the steps, and only then make the commit above. Push to `v2.0-fast` as a
fast-forward.
