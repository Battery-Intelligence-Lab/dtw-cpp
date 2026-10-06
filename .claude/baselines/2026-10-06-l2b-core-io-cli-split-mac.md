# 2026-10-06 — L2b: dtwc_core, dtwc_io and the CLI; FILE_SET HEADERS per folder; the bindings link the core (Mac)

Unit L2b from design-2.0 f196c57b. Apple M5 Pro, macOS 26.6.2, Apple clang 21.0.0, CMake 4.4.3, Ninja, MATLAB R2026a,
Python 3.12.14 (uv). Trees: `build/` (clang-macos preset), `build-matlab/` (W9e's command), `build-arrow/` (the preset's
flags, `-DDTWC_ENABLE_ARROW=ON` through a shim over pyarrow 25.0.1's Arrow C++ as Windows's `build/arrow-pyarrow-23`,
and the MEX), the wheel (`uv build --wheel`, build dir kept at one path). Commits: 12b3bce3 (file sets), 2b1e384d
(dtwc_core, dtwc_cli), 95b10a20 (dtwc_io), acf86f93 (docs, CHANGELOG), this record. Sizes, maps, counts and bytes
are [confirmed] by the command named; times are [inferred]. Registered before the runs: the base's ctest names and
statuses, the 25 CLI runs diff -r empty, the D-19 ulp only, no run/api/CLI11/fkYAML/Arrow in the MEX or the extension
with the text reader and writers present, matlab_suite 142/141/0/1, pytest 974/11/0; with Arrow, neither binding links it.

## Targets

`dtwc_core` STATIC: algorithms, mip objects, Problem, Problem_IO, scores, checkpoint, the text reader, parse_number,
arrow_c_data + nanoarrow, Metal/CUDA. `dtwc_cli` STATIC: config.cpp, run.cpp, api.cpp; CLI11 PUBLIC, fkYAML PRIVATE.
`dtwc_io` STATIC, only when Arrow is found: io/read_arrow.cpp, algorithms/fast_clara_parquet.cpp; Arrow and
DTWC_HAS_ARROW/PARQUET PUBLIC. `dtwc++` INTERFACE over all three, owning dtwc.hpp.

## Results

| check | base f196c57b | after |
|---|---|---|
| serial ctest `build/` | 95: 94 + test_cuda_correctness skipped | the same names and statuses at each commit |
| CODEGEN_NO_CALLS | inner_loops=108 calls=0 PASS | the same |
| 25 CLI runs (W9e's set) | 156 files; `--band 10` exits 1, 24 exit 0 | diff -r empty at 2b1e384d and 95b10a20 |
| Arrow-off CLI, 16 Parquet/Arrow inputs | 16 refusals, exit 1, no output dir | diff -r empty |
| MEX (`build-matlab`) | 4,516,640 bytes; 20 libdtwc++.a members; map 12,767 symbols / 3,700,812 B | 2b1e384d byte-identical; 95b10a20 4,517,344, 20 libdtwc_core.a members, 12,778 / 3,701,030 B |
| matlab_suite | 142 run, 141 passed, 0 failed, 1 incomplete | the same at 2b1e384d and 95b10a20 |
| extension installed / `uv build` / wheel | 1,203,728 / 1,203,696 / 576,019; 23 members | 2b1e384d identical; 95b10a20 1,204,352 / 1,204,320 / 576,077, 23 libdtwc_core.a members |
| pytest, fresh venv `.[test,dev,io,mip]` matplotlib pandas | 974 / 11 / 0 | 974 / 11 / 0 at 2b1e384d and 95b10a20 |
| docs gates | — | check_docs PASS (392 flags, 59 pages), check_pins 0, generate_docs current |

Both bindings, demangled `nm -U`, after: `dtwc::run(`, `dtwc::load(`, `dtwc::cluster(`, `dtwc::Result::`, `dtwc::cli::`,
`CLI::`, `fkyaml`, `arrow::`, `parquet::` 0; `dtwc::read_data(` 1; `dtwc::detail::write_result_files(` 3 (MEX), 1
(extension). The extension keeps arrow_c_data + nanoarrow (`DtwcNanoarrow` 61 symbols, as at base): Python's
`data_from_arrow_c_array` is behind `load()` of Parquet/Arrow via pyarrow and every Arrow-array input. The MEX has
neither, as at base. +704 / +624 bytes: read_data.cpp.o and fast_clara.cpp.o hold the shared helpers out of line.

## Arrow on (`build-arrow`, the pyarrow 25.0.1 shim; Mac only)

| check | base | after (95b10a20) |
|---|---|---|
| ctest | 99: 97 + 1 skip + test_io_readers failed (17/18 cases) | the same 99 names and statuses |
| MEX | 4,617,104 B, links libarrow.2500 + libparquet.2500, 22 members, `arrow::` 33 | 4,517,344 B, byte-identical to the Arrow-off MEX |
| extension, `-DDTWC_ENABLE_ARROW=ON` | 1,259,968 B, links libarrow + libparquet, `arrow::` 24 | 1,204,320 B, byte-identical to the Arrow-off extension |
| 16 CLI runs: IPC (pam, -v, clara), Parquet (file, --column, auto, folder, fixture), streamed CLARA f64/f32, 5 errors | 84 files: 11 exit 0, 5 exit 1 | diff -r empty |
| the 25 text CLI runs | equal to the Arrow-off base's | diff -r empty |

The failing case is base's: "Parquet: a streamed Result::save … refuses the matrix" (test_io_readers.cpp:666, no throw);
Windows's pyarrow 23 tree passes it (handoff 10-06) [inferred: Arrow 25 versus the case's 900-byte limit; not
established]. Warnings: 17 Arrow-24 `ReadRowGroups is deprecated`, at base and after.

## Compile commands (per source, flags sorted, against base re-configured) and objects

A re-configure flips Catch2 shared (HiGHS caches `BUILD_SHARED_LIBS=ON`), so the base is base re-configured. 12b3bce3:
254/254 identical (`BASE_DIRS .` keeps `-I…/dtwc/.`). 2b1e384d, 95b10a20 (Arrow off): 231 identical; the 20 core sources
lose `-DDTWC_HAS_YAML` and the CLI11/fkYAML `-isystem` dirs; api.cpp, config.cpp, run.cpp lose `-DDTWC_ENABLE_HIGHS`, the
QUICKCPPLIB definitions, `-I dtwc/extern`, `-I dtwc/mip/.` and llfio's dirs (none used). Objects: 25 of 26 identical to
base's at 2b1e384d (its message says 26: that comparison read dtwc++.dir's stale objects; corrected in 95b10a20's);
distance_matrix.cpp.o differs in three `!srcloc` values (inline-asm source offsets shifted 24 bytes by the dropped
`#define DTWC_HAS_YAML 1`; bisected to that flag; preprocessed source identical; the MEX linking it was byte-identical).

## Headers, and a parent project

83 headers under dtwc/: 2 vendored (not public) and 81 ours, of which 41 were in a `target_sources` and 40 in none (L2a
said 41; counted by parsing every dtwc/**/CMakeLists.txt). Each of the 40 has an includer (`git grep` of its name): none
deleted, 0 unlisted now. The mip headers are dtwc_core's (`<mip/...>`); mip-solvers keeps its `PUBLIC .` include for its
sources and Problem.cpp's `"mip.hpp"`. dtwc_io's three headers are listed only in a build with Arrow. A scratch project
with `add_subdirectory(<worktree> dtwc)` and `target_link_libraries(consumer PRIVATE dtwc++)` calling
`dtwc::cluster(dtwc::load(data/dummy, 1, 1), 3, "pam")` via `<dtwc.hpp>` builds and prints `cost=148362 medoids=3`.
`build-ex` (examples and benchmarks ON): the 5 examples and 6 benchmarks link through dtwc++, 0 warnings;
example_quickstart prints `labels: 0x9 1x9 2x9`, `medoids: 4 13 22`, `mean silhouette: 0.96895`. What calls dtwc_cli
(run, cluster, load, cli::bind) gets it through dtwc++: examples/cpp/quickstart.cpp, f14_result_save_writer, 14 tests.

## Deviations from the brief, and why

1. arrow_c_data.cpp and nanoarrow stay in dtwc_core: the wheel needs them (above); they link no Arrow.
2. FastCLARA's Parquet stream (fast_clara.cpp's `#ifdef DTWC_HAS_PARQUET` code, verbatim but for four `detail::`) is
   dtwc_io's `algorithms::fast_clara_parquet`: in the core it put Parquet in every binding of an Arrow build.
   `algorithms::fast_clara` refuses `force_parquet_streaming` in every build, in an Arrow-off build's words.
3. dtwc_cli is STATIC, not OBJECT: through an INTERFACE dtwc++ an object library passes no objects (or, with
   `$<TARGET_OBJECTS>`, every one into every test). Tests and examples get run/api/config through dtwc++.
4. dtwc_io exists only with Arrow (no source otherwise); the CLI's entry is `format_of` (dtwc_io's `arrow_format`
   first) and `read_input` (the one `if`), each behind `#ifdef DTWC_HAS_ARROW`.
5. A C++ call of `dtwc::read_data` on Parquet in an Arrow build gets "This binary was built without Parquet support":
   the words stay byte-identical for the Arrow-off CLI.
6. cuda/ and metal/ CMakeLists list their headers (every build) besides the target name: item 3 asks every folder to.

Review: an adversarial agent was launched over f196c57b..acf86f93; its report had not arrived when this was committed.
**Not proven here:** Windows (MSVC, clang-cl), Linux/GCC (GNU ld; CMake orders libdtwc_cli, libdtwc_io, libdtwc_core
here), CUDA, an IPC-only Arrow build, a CPM static Arrow, Gurobi, CI; the Arrow-on proof is the Mac's shim, not Windows's.

## Commands

```sh
cmake -S . -B build-arrow -G Ninja -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_TESTING=ON -DOpenMP_ROOT=/opt/homebrew/opt/libomp -DDTWC_ENABLE_ARROW=ON \
  -DArrow_DIR=$S/pyarrow-config -DParquet_DIR=$S/pyarrow-config -DDTWC_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/Applications/MATLAB_R2026a.app
# pyarrow-config: Arrow::arrow_shared / Parquet::parquet_shared, IMPORTED SHARED over pyarrow's lib{arrow,parquet}.2500.dylib
CMAKE_ARGS="-DOpenMP_ROOT=… [-DDTWC_ENABLE_ARROW=ON -DArrow_DIR=… -DParquet_DIR=…]" uv build --wheel --python 3.12 \
  -o $S/dist -C build-dir=$S/wheel-build    # maps: `ninja -t commands` link line re-run with -Wl,-map (byte-identical)
```

## Windows queue: the Arrow-on proof, after the merge

```sh
cmake --build build/arrow-pyarrow-23 && ctest --test-dir build/arrow-pyarrow-23 -C Release --output-on-failure -j 1
build/arrow-pyarrow-23/bin/dtwc_cl.exe -i tests/fixtures/fast_clara_streaming_8x4.parquet -k 2 --method pam -o $TEMP/l2b-pq
build/arrow-pyarrow-23/bin/dtwc_cl.exe -i tests/fixtures/fast_clara_streaming_8x4.parquet -k 2 --method clara \
  --ram-limit 900 --sample-size 4 -v -o $TEMP/l2b-stream          # prints "FastCLARA: streaming from Parquet"
P=C:/D/git/dtw-cpp/build/arrow-pyarrow-23/pyarrow-config
cmake -S . -B build/l2b-mex-arrow -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
  -DDTWC_BUILD_MATLAB=ON -DDTWC_BUILD_TESTING=ON -DMatlab_ROOT_DIR="C:/Program Files/MATLAB/R2024b" \
  -DDTWC_ENABLE_ARROW=ON -DArrow_DIR=$P -DParquet_DIR=$P
cmake --build build/l2b-mex-arrow --target dtwc_mex && ctest --test-dir build/l2b-mex-arrow -R matlab_suite -V
dumpbin /dependents build/l2b-mex-arrow/bin/dtwc_mex.mexw64          # no arrow.dll / parquet.dll
unset CMAKE_GENERATOR; CMAKE_ARGS="-DDTWC_ENABLE_ARROW=ON -DArrow_DIR=$P -DParquet_DIR=$P" \
  uv build --wheel --python 3.12 --out-dir dist-l2b -C build-dir=C:/D/git/wt/tmp/l2b-py
dumpbin /dependents C:/D/git/wt/tmp/l2b-py/python/Release/_dtwcpp_core.cp312-win_amd64.pyd   # no arrow.dll / parquet.dll
```
