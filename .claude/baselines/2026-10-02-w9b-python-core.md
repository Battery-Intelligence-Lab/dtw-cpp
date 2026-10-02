# 2026-10-02 — W9b: Python on the C++ core

Unit W9b, branch pb/W9b from 558e09a6 (design-2.0 after W9a and W8c). Windows 11, MSVC 19.50 wheel builds
(VS 18, `CMAKE_GENERATOR` unset), clang 21 `build/` tree. The machine was shared with other agents' builds:
every wall-clock number is [inferred]; sizes, maps, counts and test results are [confirmed].

Volkan's 10-02 ruling ("Prioritise reading the same file in the same way in all languages if possible. So you can
bind some reader and probably assume pyarrow reads things same I guess.") brought the C++ text reader and the C++
writers back into the binding at 98190ae4; d6129e1f was the version with a Python reader and Python writers.

## The extension module (`_dtwcpp_core.cp312-win_amd64.pyd`) [confirmed]

| tree | bytes | dtwc++.lib members loaded (link /VERBOSE) |
|---|---:|---|
| base 558e09a6 | 6,136,832 | 25, among them api, run, config (CLI11), read_data (DataLoader), parse_number (fast_float), Problem_IO |
| d6129e1f (Python reader and writers) | 6,063,104 | 21: api, run, config, read_data gone; parse_number loaded but discarded whole by /OPT:REF |
| 98190ae4 (C++ reader and writers) | 6,156,288 | 22: api, run, config gone; read_data back |

Symbols kept in the image (`link /MAP`, Publics + Static symbols), base -> d6129e1f -> 98190ae4:
config.obj 738 -> 0 -> 0 (CLI11 411 -> 0 -> 0, fkYAML 2 -> 0 -> 0); run.obj 4 -> 0 -> 0; api.obj 14 -> 0 -> 0;
read_data.obj 757 -> 0 -> 750 (DataLoader 8 -> 0 -> 8); fast_float 45 -> 0 -> 45; Problem::write_clusters 15 -> 0 -> 15,
write_silhouettes 20 -> 0 -> 20, write_distance_matrix 8 -> 0 -> 8, io::write_csv 6 -> 0 -> 6, io::read_csv 74 -> 0 -> 74,
detail::write_result_files 0 -> 0 -> 48 (moved from run.cpp to Problem_IO.cpp); the to_chars tables 4 -> 0 -> 4.

## Timing [inferred, shared machine]

- `dtwcpp.cluster(X, k=5)`, X = 200 x 1,000 standard normal (seed 0), 5 calls after a warm-up, wheels interleaved
  twice: median 0.562 / 0.598 s (base), 0.562 / 0.571 s (d6129e1f's code path, unchanged since); cost
  111825.98465320448 and medoids {16, 60, 96, 180, 197} identical.
- `dtwcpp.load(csv).as_data()`, the same data written by `np.savetxt`, median of 5: base 0.023 / 0.022 s, 98190ae4
  0.021 / 0.021 s (the C++ reader in both); d6129e1f's Python reader took 0.082 s. Series bit-identical, as on
  `data/dummy` (skip_rows 1, skip_cols 1).

## Gates [confirmed]

- Serial ctest at 55f2d3eb (the last C++ change): 95 = 92 passed + 3 MAY_SKIP (test_cuda_correctness,
  test_metal_correctness, test_metal_mmap); the 25 CLI runs byte-identical to base (156 files).
- The Ctrl-Z case: unit_test_fileOperations' new case failed against the text-mode reader (no exception, 2 of 2
  assertions) and passes in binary mode; the base wheel reads `tests/data/reader/ctrl_z.csv` without error.
- pytest at 98190ae4 (the exact gate: fresh venv, `uv pip install --reinstall "W9b[test,dev,io]"`; its `.pyd`
  6,156,288 bytes, as the mapped build's): 939 passed / 20 skipped / 0 failed (the 20th skip is the pandas form: pandas is not installed;
  with pandas 3.0.6 in a scratch venv the 4 forms pass). Base 1,094 / 19 / 0; collected 1,113 -> 959: 199 ids removed,
  45 added, by name in the commit messages. `test_conformance.py` 2 passed; examples/python 01-09 exit 0.

## Commands

```sh
# wheel + map + link log (the map flags on the module link only; LINK=/MAP in the environment is rewritten by MSYS)
uv build --wheel --python 3.12 --out-dir dist-py -C build-dir=<dir> -C build.verbose=true \
  "-C" "cmake.define.CMAKE_MODULE_LINKER_FLAGS=-MAP -VERBOSE" .
# per-object symbol counts of a map
uv run --no-project python mapcount.py <map> "fast_float@@" "DataLoader@dtwc" "CLI@@" ...
```
