# 2026-10-02 — W9b: Python on the C++ core, files in Python

Unit W9b, branch pb/W9b from 558e09a6 (design-2.0 after W9a and W8c). Windows 11, MSVC 19.50 wheel builds
(VS 18, `CMAKE_GENERATOR` unset), clang 21 `build/` tree. The machine was shared with other agents' builds:
every wall-clock number is [inferred]; sizes, maps, counts and test results are [confirmed].

## The extension module (`_dtwcpp_core.cp312-win_amd64.pyd`) [confirmed]

| tree | bytes | dtwc++.lib members loaded (link /VERBOSE) |
|---|---:|---|
| base 558e09a6 | 6,136,832 | 25, among them api, run, config (CLI11), read_data (DataLoader), parse_number (fast_float), Problem_IO |
| da4fdb09, 7ab98ce0 (same C++) | 6,063,104 | 21: api, run, config, read_data gone; parse_number loaded (Problem_IO.obj references it) and discarded whole by /OPT:REF |

Symbols kept in the image (`link /MAP`, Publics + Static symbols), base -> head: config.obj 738 -> 0 (CLI11 411 -> 0,
fkYAML 2 -> 0); read_data.obj 757 -> 0 (DataLoader 8 -> 0); parse_number.obj 50 -> 0 (fast_float 45 -> 0); run.obj 4 -> 0;
api.obj 14 -> 0; Problem::write_clusters / write_silhouettes / write_distance_matrix, io::write_csv, io::read_csv -> 0;
the to_chars precision tables (`__d2fixed_buffered_n`, `__d2exp_buffered_n`) 4 -> 0. Still kept: Problem_IO's
writeBestRep, writeMedoids and print_clusters (34 symbols): Problem::cluster() reaches the first two through Lloyd's
run-artifact branch (taken only under cluster_and_process), and print_clusters is bound.

## Timing [inferred, shared machine]

`dtwcpp.cluster(X, k=5)`, X = 200 x 1,000 standard normal (seed 0), 5 calls after a warm-up, base and head wheels
interleaved twice: median 0.562 / 0.598 s (base), 0.562 / 0.571 s (head); cost 111825.98465320448 and medoids
{16, 60, 96, 180, 197} identical. `dtwcpp.load(csv).as_data()` on the same data written by `np.savetxt`: 0.024 s with the
C++ reader (base), 0.082 s with the Python reader (head; 0.178 s before the one-pass row check), series bit-identical,
as on `data/dummy` (skip_rows 1, skip_cols 1).

## Commands

```sh
# wheel + map + link log (the map flags on the module link only; LINK=/MAP in the environment is rewritten by MSYS)
uv build --wheel --python 3.12 --out-dir dist-py -C build-dir=<dir> -C build.verbose=true \
  "-C" "cmake.define.CMAKE_MODULE_LINKER_FLAGS=-MAP -VERBOSE" .
# per-object symbol counts of a map
uv run --no-project python mapcount.py <map> "fast_float@@" "DataLoader@dtwc" "CLI@@" ...
```

## Gates at 7ab98ce0 [confirmed]

Serial ctest (5e710ba4; the later commits touch no C++ that `build/` compiles): 95 = 92 passed + 3 MAY_SKIP
(test_cuda_correctness, test_metal_correctness, test_metal_mmap). The 25 CLI runs: byte-identical to base (156 files).
Python gate (fresh venv, `uv pip install --reinstall "W9b[test,dev,io]"`): 933 passed / 19 skipped / 0 failed (base
1,094 / 19 / 0: 199 ids removed, 38 added, by name in the commit messages); its `.pyd` is 6,063,104 bytes, as the mapped
build's. `tests/conformance/test_conformance.py` 2 passed; examples/python 01-09 exit 0.
At d6129e1f (Python-only changes after that gate: the reader fallback and the review fixes, the changed `.py` files
copied into the gate venv, the `.pyd` unchanged): pytest 938 passed / 19 skipped / 0 failed (5 new key-kind cases).
