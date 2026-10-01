# 2026-10-01 — L1: what the Python extension and the MATLAB MEX carry

Question (Volkan, 2026-10-01): both bindings link the static library `dtwc++`, which holds the readers, the CLI
and the MIP code; how many bytes of each binary belong to which component, and what pulls `cli/` and `io/` in?
Measurement only; no product code changed. Branch pb/L1, base bf2992da (design-2.0). Every number below is
[confirmed] by the command that produced it unless marked [inferred].

## What was built [confirmed]

| id | binary | configuration | file bytes |
|---|---|---|---:|
| A | `build-mex/bin/dtwc_mex.mexw64` | the way `build/mex` is configured: clang 21.1.8, Ninja, Release `-O3`, ThinLTO (`-flto=thin`, lld-link), `-DDTWC_BUILD_MATLAB=ON -DDTWC_BUILD_TESTING=ON`, HiGHS + Gurobi (`C:/gurobi1301`) + llfio ON, R2024b | 7,914,496 |
| B | `build-py/python/Release/_dtwcpp_core.cp312-win_amd64.pyd` | the wheel: `uv build --wheel` (scikit-build-core 1.1.0, CMake 4.2.3, `CMAKE_GENERATOR` unset so Visual Studio 18 2026, MSVC 19.50.35723, `/GL` `/LTCG:incremental`, `/OPT:REF /OPT:ICF`), `cmake.args` of pyproject.toml (HiGHS + llfio ON, Gurobi OFF), Python 3.12.12 | 6,115,328 |
| C | `build-mex-ci/bin/dtwc_mex.mexw64` | the way `.github/workflows/matlab-mex.yml` configures it (see 2026-09-30-y4-matlab.md): MSVC 19.50, VS 18, `-DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_BUILD_TESTING=OFF`, R2024b | 878,592 |

B is byte-identical to the `.pyd` that `uv pip install` put in the venv (`cmp`). Relinking A with the map flags
gave the same size as the first link; relinking B by hand with the recorded link line gave the same size
(6,115,328). Section raw sizes (bytes): A `.text` 6,537,216, `.rdata` 1,102,336, `.data` 41,472 (+ 213,907 bss),
`.pdata` 214,528, `.tls` 7,680, `.reloc` 9,728, `.rsrc` 512. B `.text` 4,844,544, `.rdata` 836,608, `.data`
208,896 (+ 42,264 bss), `.pdata` 204,288, `.reloc` 19,456. C `.text` 565,248, `.rdata` 260,096, `.data` 28,672,
`.pdata` 20,992, `.reloc` 2,048.

## How the bytes were attributed

Linker maps: A `lld-link /map /lldmap` (the lld map carries sizes; ThinLTO objects keep the name of their source
member, `dtwc_mex.mexw64.lto.dtwc++.lib<member>.obj`); B and C `link /MAP` (sizes are the difference to the next
address, `Static symbols` included; `Lib:Object` names the origin object). "File bytes" = code + `.rdata` +
initialised `.data` + `.pdata`; bss is memory only and listed separately. Rules, applied in this order:
1. HiGHS = the 216 members of `highs.lib` (205 linked in A, 203 in B); Gurobi = the 15 members of `GurobiCXX.lib`;
   nanobind = `nanobind-static.lib`; runtime = `msvcprt`/`MSVCRT`/`ucrt`/`vcruntime`/`VCOMP` objects and import stubs.
2. For a `dtwc++.lib` member or the binding TU, the symbol name moves a chunk out of its member: `CLI::` and
   `fkyaml` to cli/; `llfio_v2`, `outcome_v2`, `quickcpplib`, `ntkernel` to checkpoint + llfio; `fast_float` and
   the header-only text readers (`text_io_detail`, `load_batch_file`, `load_folder`, `readFile`, `DataLoader`,
   `io::read_csv`) to io/; `write_csv`, `write_distance_matrix_csv`, `distance_matrix_csv_token` to Problem_IO.
3. Otherwise by member: core = Problem, dtw*, distance_matrix, env, initialisation, scores; algorithms =
   fast_pam, fast_clara, one_batch_pam, barycenter, hierarchical, tadpole; mip/ = lagrangian_root, mip_Highs,
   mip_Gurobi; io/ = parse_number, arrow_c_data; cli/ = config, run; plus Problem_IO, api, checkpoint, nanoarrow.
4. `.pdata` has no per-object entries in the MSVC map and no per-function ones in the lld map: it is shared out
   by the number of functions each component owns [inferred, 204-215 KB in total].
5. A: lld lists no owner for the merged string literals (248,676 bytes between the last listed `.rdata` chunk and
   the export table). Each NUL-terminated string was looked up among the string literals of the sources of each
   component (`strpool.py`): checkpoint + llfio 96,390 (the NT-status message table of ntkernel_error_category),
   HiGHS 107,600, core + algorithms + glue 16,633, Gurobi 5,710, cli/ 4,371, io/ 2,443, mip/ 1,745, not found
   13,785. Import tables and chunk padding are in "PE structure, padding".

## Table A — MEX, build/mex configuration (7.914 MB)

| component | MB | % of file | text | rdata | data | pdata |
|---|---:|---:|---:|---:|---:|---:|
| HiGHS | 6.196 | 78.3 | 5.411 | 0.595 | 0.015 | 0.175 |
| dtwc core | 0.439 | 5.5 | 0.351 | 0.058 | 0.017 | 0.010 |
| checkpoint + llfio | 0.256 | 3.2 | 0.106 | 0.140 | 0.001 | 0.005 |
| binding glue (dtwc_mex.cpp) | 0.195 | 2.5 | 0.161 | 0.026 | 0.001 | 0.007 |
| C++ runtime/STL/CRT (static part, import stubs) | 0.161 | 2.0 | 0.025 | 0.134 | 0.000 | 0.001 |
| mip/ | 0.140 | 1.8 | 0.120 | 0.017 | 0.001 | 0.003 |
| cli/ + CLI11 + fkYAML | 0.138 | 1.7 | 0.105 | 0.024 | 0.004 | 0.005 |
| algorithms | 0.138 | 1.7 | 0.116 | 0.018 | 0.000 | 0.003 |
| Gurobi C++ API (GurobiCXX.lib) | 0.075 | 0.9 | 0.046 | 0.025 | 0.000 | 0.003 |
| io/ readers + fast_float | 0.041 | 0.5 | 0.024 | 0.017 | 0.000 | 0.001 |
| api.cpp (Tier-1 facade, Result::save) | 0.030 | 0.4 | 0.024 | 0.005 | 0.000 | 0.001 |
| Problem_IO + result writers | 0.029 | 0.4 | 0.022 | 0.006 | 0.000 | 0.001 |
| unattributed strings | 0.014 | 0.2 | | 0.014 | | |
| PE structure, padding, import tables | 0.063 | 0.8 | | | | |
| nanoarrow, nanobind | 0 | 0 | not linked (arrow_c_data, nanoarrow, barycenter, dtw.cpp members absent) | | | |
| total | 7.914 | 100.0 | | | | |

## Table B — Python extension, the wheel (6.115 MB)

| component | MB | % of file | text | rdata | data | pdata |
|---|---:|---:|---:|---:|---:|---:|
| HiGHS | 4.611 | 75.4 | 3.897 | 0.378 | 0.173 | 0.164 |
| dtwc core | 0.367 | 6.0 | 0.208 | 0.135 | 0.016 | 0.007 |
| binding glue (_dtwcpp_core.cpp) | 0.297 | 4.9 | 0.227 | 0.050 | 0.008 | 0.012 |
| C++ runtime/STL/CRT (static part, import stubs) | 0.187 | 3.1 | 0.028 | 0.157 | 0.000 | 0.002 |
| checkpoint + llfio | 0.166 | 2.7 | 0.120 | 0.042 | 0.002 | 0.003 |
| algorithms | 0.107 | 1.8 | 0.093 | 0.011 | 0.000 | 0.003 |
| nanobind runtime | 0.106 | 1.7 | 0.086 | 0.015 | 0.002 | 0.003 |
| mip/ | 0.094 | 1.5 | 0.077 | 0.009 | 0.003 | 0.005 |
| io/ readers + fast_float | 0.056 | 0.9 | 0.038 | 0.016 | 0.000 | 0.002 |
| cli/ + CLI11 + fkYAML | 0.054 | 0.9 | 0.039 | 0.008 | 0.005 | 0.002 |
| nanoarrow | 0.017 | 0.3 | 0.012 | 0.005 | 0.000 | 0.000 |
| Problem_IO + result writers | 0.016 | 0.3 | 0.012 | 0.003 | 0.000 | 0.001 |
| api.cpp (Tier-1 facade, Result::save) | 0.009 | 0.1 | 0.007 | 0.001 | 0.000 | 0.000 |
| PE structure, padding | 0.027 | 0.4 | | 0.005 | | |
| total | 6.115 | 100.0 | | | | |

## Table C — MEX, matlab-mex.yml configuration (0.879 MB)

| component | MB | % of file | text | rdata | data | pdata |
|---|---:|---:|---:|---:|---:|---:|
| dtwc core | 0.237 | 27.0 | 0.185 | 0.031 | 0.016 | 0.005 |
| binding glue (dtwc_mex.cpp) | 0.172 | 19.5 | 0.133 | 0.028 | 0.006 | 0.005 |
| C++ runtime/STL/CRT (static part, import stubs) | 0.164 | 18.7 | 0.017 | 0.145 | 0.000 | 0.001 |
| cli/ + CLI11 + fkYAML | 0.087 | 9.9 | 0.062 | 0.017 | 0.005 | 0.003 |
| algorithms | 0.071 | 8.1 | 0.061 | 0.008 | 0.000 | 0.002 |
| io/ readers + fast_float | 0.048 | 5.4 | 0.032 | 0.015 | 0.000 | 0.001 |
| api.cpp (Tier-1 facade, Result::save) | 0.035 | 4.0 | 0.027 | 0.007 | 0.001 | 0.001 |
| mip/ (lagrangian_root, solver stubs) | 0.034 | 3.8 | 0.030 | 0.003 | 0.000 | 0.001 |
| Problem_IO + result writers | 0.015 | 1.7 | 0.011 | 0.003 | 0.000 | 0.001 |
| checkpoint (llfio OFF) | 0.010 | 1.1 | 0.008 | 0.002 | 0.000 | 0.000 |
| PE structure, padding | 0.006 | 0.7 | | | | |
| total | 0.879 | 100.0 | | | | |

HiGHS is absent from C (OFF). The static STL object `xcharconv_ryu_tables.obj` is 114,456 B (A), 114,464 B (B),
114,464 B (C) of the runtime row: 1.4 %, 1.9 % and 13.0 % of the three files. It is referenced by
`std::__d2fixed_buffered_n` and `std::__d2exp_buffered_n`, which the precision form of `std::to_chars(double)` at
`dtwc/core/matrix_io.hpp:81` (`distance_matrix_csv_token`, the CSV distance-matrix writer) calls.

## Linked objects against the library

A links 21 of the 25 `dtwc++.lib` members (not linked: arrow_c_data, barycenter, dtw, nanoarrow) and 205 of 216
HiGHS objects: 6.2 MB of HiGHS is linked out of a 22.7 MB `highs.lib`. B links 24 of 25 (not linked: dtw) and 203
HiGHS objects. `dtw_lanes.cpp` is native code in A (`-fno-lto` for that file in dtwc/CMakeLists.txt).

## DLLs and installed size [confirmed]

- Neither binding ships a DLL. The wheel holds 29 files, none a DLL: `_dtwcpp_core.cp312-win_amd64.pyd` 6,115,328,
  the python files and slurm scripts 183,356, dist-info and licences 57,819; wheel file 2,504,648 B (zip), 6,356,503 B unpacked.
  Installed (`uv pip install dist-py/*.whl`): `site-packages/dtwcpp` 6,298,684 B + `dtwcpp-2.0.0rc1.dist-info`
  58,506 B = 6,357,190 B. The wheel's only dependency, numpy 2.5.3, installs 20,567,546 B + `numpy.libs` 21,164,112 B.
  `h5py` and `pyarrow` are the `io` extra, not installed by default.
- MEX install payload (`install()` in bindings/matlab/CMakeLists.txt): `dtwc_mex.mexw64` + `+dtwc` (38 files,
  70,310 B); `highs_extras.dll` in `build/bin` is not imported by the MEX.
- Imports (`llvm-readobj --coff-imports`), all loaded from the host, none shipped:
  A: libmex.dll 1.23 MB, libmx.dll 4.15 MB, libiomp5md.dll 1.99 MB (MATLAB R2024b `bin/win64`), MSVCP140.dll
  (MATLAB's private 589 KB), VCRUNTIME140.dll, VCRUNTIME140_1.dll, 9 UCRT api-sets, KERNEL32, USER32, ADVAPI32,
  and `gurobi130.dll` (38.68 MB, from the Gurobi install: a Gurobi-ON MEX fails to load without it).
  The LLVM `libomp.dll` (778 KB) is not imported (`/NODEFAULTLIB:libomp.lib`).
  B: python312.dll, MSVCP140.dll, VCRUNTIME140(_1).dll, **VCOMP140.DLL** (213,064 B in System32, from the VC++
  redistributable), 9 UCRT api-sets, KERNEL32, USER32, ADVAPI32. (The tag is cp312, not abi3.)
  C: libmex.dll, libmx.dll, MSVCP140.dll, VCOMP140.DLL, VCRUNTIME140(_1).dll, 9 UCRT api-sets, KERNEL32.

## Reference chains into cli/ and io/ [confirmed unless marked]

Three independent views agree. (1) Object level: A, `llvm-nm` strong definitions of the bitcode members; B, the
full `link /VERBOSE` log (Found / Referenced in / Loaded). (2) Function level: the reference graph of the linked
image (disassembly of `.text` with `llvm-objdump -d`, rip-relative and relocated pointers, nodes = map chunks) from
`mexFunction` / `PyInit__dtwcpp_core` plus the `.CRT$XCU` global-constructor arrays. (3) The maps themselves.
A "reach" counts only functions whose own name says cli/ or io/ (a `std::` template kept from that member's COMDAT
is not).

MEX (A; C links the same 21 `dtwc++.lib` members, checked in its map):
- `tier1_cluster` (`cmd_tier1_cluster`, dtwc_mex.cpp:1294, calls `dtwc::load` :1307 and `dtwc::cluster` :1314)
  -> `dtwc::cluster` (dtwc/api.cpp:274/:280, `Config config` :283, `return run(config...)` :292/:297) -> `dtwc::run`
  (dtwc/cli/run.cpp:714) -> `execute` (run.cpp:352, 47,082 B) -> `text_io_detail::open_text_file`,
  `load_batch_file` (fileOperations.hpp, inlined) -> `io::parse_number` (io/parse_number.cpp) -> fast_float;
  -> `Problem::write_distance_matrix` (Problem_IO.cpp); -> `save_checkpoint` (checkpoint.cpp). Live cli/ code
  reachable from an entry point: 4 functions, 49,158 B, only through `tier1_cluster`.
- `Problem_read_distance_matrix` (dtwc_mex.cpp:963) -> `Problem::read_distance_matrix` (Problem_IO.cpp:238) ->
  `io::read_csv` (core/matrix_io.hpp) -> `io::parse_number` -> fast_float (3 functions, 14,609 B).
- `Result_save` (:1348) -> `Result::save` (api.cpp); `Result_score` (:1338), `Result_distance_matrix` (:1358) ->
  api.cpp; `set_device` (`cmd_set_device` :783) / `get_device` (:790) -> `dtwc::device` (api.cpp:144, :154).
- `save_checkpoint` (:986) / `load_checkpoint` (:995) -> checkpoint.cpp -> `DistanceMatrix::write/read`. Every
  command that can destroy a distance matrix (`Problem_set_variant`, `_set_device`, `_fill_distance_matrix`, ...)
  reaches `DistanceMatrix::Mapping::~Mapping` and the llfio unmap code (42 functions, 41,699 B).
- config.cpp: `run.cpp:717/724` calls `device_text` (cli/config.cpp:156), the one symbol that makes `config.obj` a
  link input (`llvm-nm`: run.cpp.obj -> config.cpp.obj via `dtwc::device_text(Config const&)`). The CLI11 headers
  define namespace-scope objects (`CLI::ExistingFile`, ... Validators.hpp:225; `CLI::detail::escapedChars`, ...)
  whose dynamic initialisers sit in `.CRT$XCU`, a root the linker never drops: 76 `CLI::` functions, 27,695 B,
  are live in A and reachable from no binding entry point (`mexFunction` reaches 0 `CLI::`, `fkyaml`,
  `parse_config` or `cli::bind` functions). fkYAML code: 0 B.

Python (B):
- `device` / `device(name)` (python/src/_dtwcpp_core.cpp:201, :207) -> `dtwc::device` (api.cpp). api.obj also
  holds `dtwc::cluster`, which calls `dtwc::run`: link log `Found dtwc::run / Referenced in dtwc++.lib(api.obj) /
  Loaded dtwc++.lib(run.obj)`, then `Found dtwc::device_text / Referenced in dtwc++.lib(run.obj) / Loaded
  dtwc++.lib(config.obj)`. No Python function calls `cluster`, `run` or `load`: 0 entry nodes reach a cli/ dtwc
  function; run.obj keeps 412 B live. config.obj keeps 82 `CLI::` functions, 25,460 B, live through the same
  `.CRT$XCU` initialisers (`CLI::ExistingFile`, `ExistingDirectory`, `ExistingPath`, `NonexistentPath`, ...), and
  no binding function reaches them (the only 2 hits, 56 B, are folded `std::function` stubs whose type names
  mention CLI11 validators).
- `_read_data` (_dtwcpp_core.cpp:214) -> `DataLoader::load` (DataLoader.hpp) -> `load_batch_file` / `load_folder`
  (fileOperations.hpp) -> `text_io_detail::parse_numeric_row` -> `io::parse_number` (io/parse_number.cpp, link log:
  `Referenced in _dtwcpp_core.obj`) -> fast_float: 21 io/ text-reader functions 13,760 B, fast_float 12 functions
  14,720 B, all inlined into the binding TU.
- `data_from_arrow_c_array` (:690) -> `io::data_from_arrow` / `data_from_arrow_stream` (io/arrow_c_data.cpp) ->
  nanoarrow.c (link log: `Loaded dtwc++.lib(nanoarrow.obj)` from `arrow_c_data.obj`): 3 + 7 functions.
- `write_clusters`, `write_silhouettes`, `write_distance_matrix`, ... (:993-1005) -> Problem_IO.cpp (7 functions,
  8,640 B); the writers call `std::to_chars` (matrix_io.hpp:81), hence the 114 KB STL table.

## Could not attribute, and limits

- Header-only code (CLI11, llfio, the text readers, std:: templates) is attributed by symbol name or by the first
  object that supplied its COMDAT; STL template instantiations stay with the component that needs them, so the
  "C++ runtime/STL" row holds only the statically linked library objects and import stubs. ThinLTO and `/GL` move
  inlined code across members; per-member numbers are accurate to the chunk, not to the source file [inferred].
- 13,785 B of A's string pool (STL exception texts, strings broken by macros, 703 strings under 12 chars).
- `.pdata` is distributed, not read (rule 4). Import tables, chunk and section padding, headers, `.reloc`, `.rsrc`
  are the "PE structure" row.
- bss (A 213,907 B, B 42,264 B, C 2 KB) is memory only.
- Sizes of one TU change with the compiler: A (clang -O3, ThinLTO) and B/C (MSVC /O2, /GL) differ for the same
  source (core 0.439 vs 0.367 vs 0.237 MB); only C is what CI ships for MATLAB.
- Not measured: the Gurobi-OFF clang MEX, Debug builds, abi3 wheel, other platforms, run-time memory.

## Commands

```sh
# worktree
git -C C:/D/git/dtw-cpp worktree add C:/D/git/wt/L1 -b pb/L1 bf2992d
export CPM_SOURCE_CACHE=C:/D/cpm-cache; export TMP='C:\D\git\wt\tmp\L1' TEMP='C:\D\git\wt\tmp\L1'
# A: first configure with the recipe of build/mex, then relink with map flags (matlab_add_mex links a SHARED target)
cmake -S . -B build-mex -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
  -DDTWC_BUILD_MATLAB=ON -DDTWC_BUILD_TESTING=ON -DMatlab_ROOT_DIR="C:/Program Files/MATLAB/R2024b"
cmake --build build-mex --target dtwc_mex -j 8
cmake -S . -B build-mex "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,/map:build-mex/dtwc_mex.map -Wl,/lldmap:build-mex/dtwc_mex.lldmap -Wl,/mapinfo:exports"
cmake --build build-mex --target dtwc_mex -j 8
# B: CMAKE_GENERATOR unset; LINK makes link.exe write <target>.map
unset CMAKE_GENERATOR; export CMAKE_BUILD_PARALLEL_LEVEL=8 LINK='/MAP /MAPINFO:EXPORTS'
uv venv C:/D/git/wt/venv/L1 --python 3.12 --clear
uv build --wheel --python 3.12 --out-dir dist-py -C build-dir=C:/D/git/wt/L1/build-py -C build.verbose=true .
uv pip install --python C:/D/git/wt/venv/L1/Scripts/python.exe dist-py/dtwcpp-2.0.0rc1-cp312-cp312-win_amd64.whl
# B, object-level reasons: the recorded link line (build-py/python/_dtwcpp_core.dir/Release/_dtwcpp_core.tlog/link.command.1.tlog)
#   re-run under vcvarsall x64 with /VERBOSE added, output to build-py/why/ (/WHYEXTRACT is not a link.exe option)
# C: as matlab-mex.yml, default generator (VS 18), LINK='/MAP /MAPINFO:EXPORTS'
cmake -S . -B build-mex-ci -DDTWC_BUILD_MATLAB=ON -DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_HIGHS=OFF \
  -DDTWC_ENABLE_GUROBI=OFF -DDTWC_BUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DMatlab_ROOT_DIR="C:/Program Files/MATLAB/R2024b"
cmake --build build-mex-ci --config Release --target dtwc_mex
# inspection
llvm-readobj --sections --coff-imports --coff-basereloc <binary>; llvm-ar t bin/highs.lib; llvm-nm --no-demangle <bitcode .obj>
llvm-objdump -d --no-show-raw-insn -M intel -j .text <binary>; undname.exe <mangled>
```

Scratch scripts (not committed; `C:/D/git/wt/tmp/L1/`): `lldparse.py`, `msvcparse.py`, `an.py` (rules), `strpool.py`,
`final.py` (tables), `graph.py` (reference graph), `chains.py`/`entries.py`/`initchain.py` (reachability),
`extract.py` (llvm-nm closure). Logs: `C:/D/git/wt/L1-mex-build.log`, `L1-mex-relink.log`, `L1-py-wheel.log`,
`L1-py-verbose.log`, `L1-mexci-build.log`.

## Conclusion and next decisive test

HiGHS is 75-78 % of the two binaries that carry it (A, B). cli/ + CLI11 + io/ readers + fast_float are 2.2 % of A,
1.8 % of B and 15.3 % of C, the MEX CI ships; none of them is large next to HiGHS. The avoidable part is
structural: `device()` and `cluster()` share api.cpp, `dtwc::run` and `device_text` pull cli/config.cpp, and
CLI11's header-level validator objects keep 25-28 KB of CLI11 code alive in every binding, none of it called.
[inferred] Python: cutting the `cluster()` -> `run()` edge (a separate object for `cluster`, or `device()` moved out
of api.cpp) removes run.obj and config.obj (config.obj has a single referrer, run.obj; run.obj a single one,
api.obj). MEX: it calls `run()`, so only moving `device_text` out of config.cpp removes config.obj. The decisive test
is to relink B (and A) with that edge cut and read the map. The 114 KB STL Ryu table (13 % of C) exists for the CSV
writer at matrix_io.hpp:81 and is the largest single non-solver item in C.
