# Third-party notices for DTWC++ binary distributions

DTWC++ itself is BSD-3-Clause; see `LICENSE`. This file is the attribution notice for the
third-party code that is **compiled into, or shipped beside, a binary artefact we distribute**.
It is included in every such artefact:

- the native CLI archives (`dtwc-<version>-<system>-<arch>.tar.gz` / `.zip`), at
  `share/doc/dtwc/THIRD_PARTY_LICENSES.md`;
- the Python wheels and sdist (`dtwcpp`), via `license-files` in `pyproject.toml`.

A machine-generated inventory of the CPM-resolved packages for one specific build is written to
`third_party.txt` by the `write-licenses` target. That file records what a given build resolved;
**this file** records the obligations. Where the two disagree, this file governs.

## What is in which artefact

| Component | Version | Licence | CLI archive | Python wheel |
| --- | --- | --- | --- | --- |
| Apache Arrow nanoarrow, its IPC extension and flatcc | 0.8.0 | Apache-2.0 (four flatcc headers MIT or BSD-3-Clause) | yes — vendored, compiled in | yes |
| lz4 (its library, `lib/`) | 1.10.0 | BSD-2-Clause | yes — compiled in | yes |
| Zstandard (its decoder) | 1.5.7 | BSD-3-Clause | yes — compiled in | yes |
| fast_float | 8.3.0 | Apache-2.0 **OR** MIT **OR** BSL-1.0 | yes — vendored, compiled in | yes |
| HiGHS | 1.15.1 | MIT | yes — `lib/libhighs.*` | no — the wheel links no HiGHS; `method="mip"` solves with highspy, a dependency of dtwcpp that the wheel does not contain |
| CLI11 | 2.6.2 | BSD-3-Clause | yes — headers compiled in | yes — headers compiled in: the module links the CLI library (`dtwc::run`, whose run report prints the command line's key table) |
| fkYAML | 0.4.4 | MIT | yes, when `DTWC_ENABLE_YAML=ON` (default) | no — the wheels are built with `DTWC_ENABLE_YAML=OFF` |
| nanobind | ≥ 2.4.0 | BSD-3-Clause | no | yes — runtime compiled in |
| LLVM OpenMP runtime (`libomp`) | 23.1.1 | Apache-2.0 WITH LLVM-exception | macOS — `lib/`; Windows — `bin/libomp.dll`, beside `dtwc_cl.exe`; see note below | macOS wheels — bundled by `delocate`; Windows wheels — `libomp.dll` in `dtwcpp.libs/`, bundled by `delvewheel` |
| Microsoft Visual C++ runtime (`msvcp140*`, `vcruntime140*`, `concrt140`) and its OpenMP runtime (`vcomp140`) | that of the Visual Studio that built the artefact | Microsoft Distributable Code — see below | Windows only — `bin/`, beside `dtwc_cl.exe` and `dtwc_cl_msvc.exe`; `vcomp140` for `dtwc_cl_msvc.exe` only | Windows wheels — `msvcp140` in `dtwcpp.libs/`, bundled by `delvewheel` |
| llfio, quickcpplib, outcome, ntkernel-error-category (Windows) | pinned commits | Apache-2.0 **OR** BSL-1.0 | yes — headers compiled in (`DTWC_ENABLE_LLFIO=ON`) | yes — headers compiled in |
| wg14_signals (llfio's signal guard) | pinned commit | Apache-2.0 | as llfio | as llfio |
| span-lite, byte-lite (via quickcpplib) | pinned commits | BSL-1.0 | as llfio | as llfio |
| cpuinfo (pytorch/cpuinfo) | pinned commit | BSD-2-Clause | yes — compiled in (`DTWC_ENABLE_CPUINFO=ON`) | yes — compiled in (`DTWC_ENABLE_CPUINFO=ON`): the module links the CLI library, whose run report reads it |

**Not redistributed**, and therefore not covered here: Catch2 (BSL-1.0) and google/benchmark
(Apache-2.0), which are test- and benchmark-only; libpfm4 (MIT), which google/benchmark links only
when `DTWC_BENCHMARK_PMU=ON` — a bare-metal Linux benchmarking build, never a released artefact;
CPM.cmake and CPMLicenses.cmake (MIT), which are build-time CMake helpers; Apache Arrow C++
(Apache-2.0), which is opt-in and linked only if a user builds against their own installation; and
Gurobi, which is proprietary, supplied by the user under their own licence, and never bundled.

**Note on the OpenMP runtime.** On Linux the OpenMP runtime is the one belonging to the user's
toolchain and is not redistributed by us. On Windows, which ships neither runtime, the wheels and the
archive's `dtwc_cl.exe`, which clang-cl builds, carry LLVM's `libomp.dll`; the archive's
`dtwc_cl_msvc.exe`, which MSVC builds, has MSVC's `vcomp140.dll` beside it, with the Visual C++
runtime, as Microsoft's Distributable Code (section below). On macOS we bundle LLVM's `libomp` too.
LLVM's `libomp` is Apache-2.0 WITH LLVM-exception; its terms are reproduced below. For macOS the
release workflows build it unmodified from the LLVM 23.1.1 source release
(`scripts/build_libomp_macos.sh`, SHA-256-pinned) for the artefacts' minimum macOS; the Windows wheels
and archive take `bin/libomp.dll` from LLVM 23.1.1's official release binary for Windows x64,
unmodified (`scripts/fetch_llvm_windows.sh`, SHA-256-pinned), whose clang-cl compiles them. We
deliberately never redistribute GCC's `libgomp`, which is GPL-3.0 WITH GCC-exception.

---

## Apache Arrow nanoarrow 0.8.0 — Apache-2.0

A namespaced amalgamation of nanoarrow (`nanoarrow.h`, `nanoarrow.c`), its IPC extension (`nanoarrow_ipc.h`,
`nanoarrow_ipc.c`) and the flatcc runtime that reads Arrow IPC messages (`flatcc.c`, `flatcc/`) are vendored at
`dtwc/extern/nanoarrow/` and compiled unconditionally into `dtwc++`. They are the ASF release archive
`apache-arrow-nanoarrow-0.8.0.tar.gz` bundled, unmodified, by its own
`ci/scripts/bundle.py --symbol-namespace DtwcNanoarrow --with-ipc --with-flatcc`. SHA-512 of the archive, as the ASF
publishes it, and SHA-256 of each bundled file as committed (LF line ends):

```
98f9f4c8dada0175e39e02d2baa01d0f63ad94636925cd289cbffa423de26bf0ede437aaa1ec10ff91e7d375e72cfddd950d040602520ab7891ab4c6337ce4f7  apache-arrow-nanoarrow-0.8.0.tar.gz
46b9c6e2b3d98e344dab0e537fe8b595e4baf34c81955bdbb9e3304b83814556  nanoarrow.h
647bcf7fe799815acf7263e9969a93c2f3a64bf3f49b8b5bfd3845c2e134fe01  nanoarrow.c
da83285ce5deb36f9680917899cf0cc688d6853af88d6fe696aef5c6c89cdd87  nanoarrow_ipc.h
ee1b5db2899bd568423f9ec816ad6027f053a68a306b842995e2e6937ed9c503  nanoarrow_ipc.c
5f5c30467b76b1fa4c287b23a5de885cf4a95b79a60dcd75df29a4818d64f918  flatcc.c
```

The complete Apache License 2.0 text, as distributed by the project, is shipped alongside this file,
together with the upstream `NOTICE.txt` — in the CLI archive at `share/doc/dtwc/nanoarrow/`, and in
the wheel under `dtwcpp-<version>.dist-info/licenses/dtwc/extern/nanoarrow/`. Its `LICENSE.txt` also names flatcc
(Apache-2.0, Copyright 2015-2023 Mikkel F. Jørgensen, dvide.com). The attribution notice
required by Apache-2.0 §4(d) is reproduced here in full:

```
Apache Arrow nanoarrow
Copyright 2023 The Apache Software Foundation

This product includes software developed at
The Apache Software Foundation (http://www.apache.org/).
```

Four of flatcc's portability headers carry their own terms, kept in each file. `flatcc/portable/punaligned.h`
(Copyright (c) 2016), `pmemaccess.h` and `prestrict.h` (Copyright (c) 2024), all Mikkel Fahnøe Jørgensen, dvide.com,
are MIT:

```
(MIT License)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
- The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.
- The Software is provided "as is", without warranty of any kind, express or
  implied, including but not limited to the warranties of merchantability,
  fitness for a particular purpose and noninfringement. In no event shall the
  authors or copyright holders be liable for any claim, damages or other
  liability, whether in an action of contract, tort or otherwise, arising from,
  out of or in connection with the Software or the use or other dealings in the
  Software.
```

`flatcc/portable/pstdint.h`, a `<stdint.h>` for compilers that lack one, is BSD-3-Clause:

```
Copyright (c) 2005-2016 Paul Hsieh
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## lz4 1.10.0 — BSD-2-Clause

lz4's library sources `lz4.c`, `lz4frame.c`, `lz4hc.c` and `xxhash.c` decode the LZ4_FRAME-compressed record
batches of Arrow IPC files; they are compiled into `dtwc++` from the release archive
`https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz` (SHA-256
`537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b`, as the release publishes it), unmodified. Its
`lib/LICENSE`:

```
LZ4 Library
Copyright (c) 2011-2020, Yann Collet
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Zstandard 1.5.7 — BSD-3-Clause

Zstandard's decoder (`lib/decompress/` and the files of `lib/common/` it calls) decodes the ZSTD-compressed record
batches of Arrow IPC files; it is compiled into `dtwc++` from the release archive
`https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz` (SHA-256
`eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3`, as the release publishes it), unmodified.
Zstandard is dual-licensed, BSD-3-Clause or GPL-2.0; DTWC++ takes it under BSD-3-Clause, its `LICENSE`:

```
BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook, nor Meta, nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Apache Arrow testing data — Apache-2.0 (sdist and source only)

Two test files of the Apache Arrow project's `arrow-testing` repository, at commit
`9ff285c88565f0f6abc855918c6a342e70e4909c`, are in the source tree and the sdist, unmodified, at
`tests/data/reader/arrow-testing/` (they are compiled into nothing): `data/arrow-ipc-stream/integration/` its
`1.0.0-bigendian/generated_primitive.arrow_file` (SHA-256
`80a8b0877d098632f98eb7d186662dbc6613c27851f40f1899ad9a8d4db8ba49`) and `1.0.0-littleendian/generated_primitive.arrow_file`
(SHA-256 `7aab19247177ea959506d7b192e8e21235f082267c72f394bb788c615ce03f54`). The repository's Apache License 2.0
text is beside them, as `LICENSE.txt`; the repository has no NOTICE file.

## fast_float 8.3.0 — Apache-2.0 OR MIT OR BSL-1.0

The single-header release `fast_float.h` of fast_float 8.3.0 (tag commit
`b0ab987b3dfdde13fa1915f65ef2a5c068d9208c`, SHA-256
`f23d93a4d1adf052e7b50e2a55ac54feeef91e188b71d35ae3038597e2659b90`) is vendored at
`dtwc/extern/fast_float/` and compiled unconditionally into `dtwc++`, through
`dtwc/io/parse_number.cpp` only. Copyright (c) 2021 The fast_float authors. It is offered under
your choice of three licences; DTWC++ elects the **Boost Software License 1.0**, whose text is
reproduced in full in the llfio section below. The three upstream licence texts ship alongside this
file — in the CLI archive at `share/doc/dtwc/fast_float/`, and in the wheel under
`dtwcpp-<version>.dist-info/licenses/dtwc/extern/fast_float/`.

## HiGHS 1.15.1 — MIT

The CLI archive ships it as `lib/libhighs.*`, and the MATLAB MEX links it statically; the MIT
licence below covers both. The Python wheel links no HiGHS: `method="mip"` solves with highspy,
which dtwcpp depends on and the wheel does not contain.

```
MIT License

Copyright (c) 2026 HiGHS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CLI11 2.6.2 — BSD-3-Clause

Its headers are compiled into the CLI archives, the Python wheels and the MATLAB MEX, which link the CLI library.

```
CLI11 2.6.2 Copyright (c) 2017-2026 University of Cincinnati, developed by Henry
Schreiner under NSF AWARD 1414736. All rights reserved.

Redistribution and use in source and binary forms of CLI11, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## fkYAML 0.4.4 — MIT

```
MIT License

Copyright (c) 2023-2026 Kensuke Fukutani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## nanobind — BSD-3-Clause

```
Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## LLVM OpenMP runtime (`libomp`) — Apache-2.0 WITH LLVM-exception

Bundled in the macOS artefacts and the Windows wheels and archive. The complete Apache License 2.0
text is shipped alongside this file, at the `nanoarrow/` path given in the nanoarrow section above.
The LLVM exception to it reads, in full:

```
---- LLVM Exceptions to the Apache 2.0 License ----

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
```

## Microsoft Visual C++ runtime — Microsoft Distributable Code

Bundled only in Windows artefacts, because Windows does not ship these files and CPython ships only
`vcruntime140*.dll`. The CLI archive carries in `bin/`, beside `dtwc_cl.exe` and `dtwc_cl_msvc.exe`,
`msvcp140.dll`, `msvcp140_1.dll`, `msvcp140_2.dll`, `msvcp140_atomic_wait.dll`,
`msvcp140_codecvt_ids.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`, `concrt140.dll` and the OpenMP
runtime `vcomp140.dll`, which only `dtwc_cl_msvc.exe` imports; CMake's `InstallRequiredSystemLibraries`
copies them from the building Visual Studio's `VC\Redist\MSVC` folder. The wheels carry
`msvcp140.dll` in `dtwcpp.libs/`, copied unchanged by `delvewheel` under a hash-suffixed name
(`msvcp140-<hash>.dll`) so that no other copy in the Python process stands in for it. These files
are Microsoft's and are not covered by DTWC++'s licence.

Microsoft lets licensed Visual Studio users copy and distribute them, unmodified, with their programs.
The source is the "Distributable Code" section of the Visual Studio licence terms (for Microsoft Visual
Studio Community 2026, last updated 1 October 2025,
<https://visualstudio.microsoft.com/license-terms/vs2026-ga-community/>) and the Distributable List it
refers to (<https://aka.ms/vs/18/redistribution>; for Visual Studio 2022,
<https://aka.ms/vs/17/redistribution>), which covers every file under `VC\Redist` except the
`debug_nonredist` folders. In summary (the terms govern): whoever distributes the files must add
significant primary functionality to them, have distributors and end users agree to terms that protect
them at least as much as the licence terms do, and indemnify Microsoft against claims related to the
distribution or use of the application; they may not modify the files, distribute pre-release ones, use
Microsoft's trademarks to suggest that the application comes from or is endorsed by Microsoft, or put
the files under a licence that requires their source to be disclosed or lets others modify them.
Whoever uses or redistributes these files from a DTWC++ artefact does so on the same terms: unmodified,
as part of DTWC++, and without suggesting Microsoft's endorsement.

## llfio and its header-only dependencies — Apache-2.0 OR BSL-1.0, Apache-2.0, BSL-1.0

Compiled into the CLI archives and the Python wheels (`DTWC_ENABLE_LLFIO=ON`; the MATLAB MEX is
built without it), and optional in a source build. llfio,
quickcpplib, outcome and ntkernel-error-category are offered under your choice of two licences, and
DTWC++ elects the **Boost Software License 1.0**; span-lite and byte-lite are BSL-1.0 only;
wg14_signals is Apache-2.0 only, so its terms (the Apache License 2.0 text shipped for nanoarrow)
apply as well. The Boost Software License 1.0, reproduced in full:

```
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## cpuinfo — BSD-2-Clause

pytorch/cpuinfo reads the CPU's model, its cores and its SIMD extensions for the machine facts that `dtwc_cl -v`
and the run report print. Its library (`src/`, `include/cpuinfo.h`) is compiled into the CLI archives, the Python
wheels and the MATLAB MEX (`DTWC_ENABLE_CPUINFO=ON`, the default) from the archive of commit `e9320ce3398899fbecc986c82f49b5d59b80dff7`
(`https://github.com/pytorch/cpuinfo/archive/e9320ce3398899fbecc986c82f49b5d59b80dff7.tar.gz`, SHA-256
`0f19bffed603d591240b6e25e8def96d203ccf59db09c28cd15c92bf856f5e90`), unmodified; its `deps/clog` is not built. Its
`LICENSE`:

```
Copyright (c) 2019 Google LLC
Copyright (c) 2017-2018 Facebook Inc.
Copyright (C) 2012-2017 Georgia Institute of Technology
Copyright (C) 2010-2012 Marat Dukhan

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
