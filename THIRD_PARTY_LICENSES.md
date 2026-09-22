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
| Eigen | 5.0.1 | MPL-2.0 (plus MPL2-compatible third-party files) | yes — headers compiled in | yes |
| Apache Arrow nanoarrow | 0.8.0 | Apache-2.0 | yes — vendored, compiled in | yes |
| HiGHS | 1.15.1 | MIT | yes — `lib/libhighs.*` | yes — linked in |
| CLI11 | 2.6.2 | BSD-3-Clause | yes — headers compiled in | no |
| fkYAML | 0.4.4 | MIT | yes, when `DTWC_ENABLE_YAML=ON` (default) | no |
| nanobind | ≥ 2.4.0 | BSD-3-Clause | no | yes — runtime compiled in |
| LLVM OpenMP runtime (`libomp`) | as installed | Apache-2.0 WITH LLVM-exception | macOS only — see note below | macOS wheels — bundled by `delocate` |
| llfio, quickcpplib | pinned commits | Apache-2.0 **OR** BSL-1.0 | only if built with `DTWC_ENABLE_LLFIO=ON`; **OFF** in released artefacts | no |

**Not redistributed**, and therefore not covered here: Catch2 (BSL-1.0) and google/benchmark
(Apache-2.0), which are test- and benchmark-only; CPM.cmake and CPMLicenses.cmake (MIT), which are
build-time CMake helpers; Apache Arrow C++ (Apache-2.0), which is opt-in and linked only if a user
builds against their own installation; and Gurobi, which is proprietary, supplied by the user under
their own licence, and never bundled.

**Note on the OpenMP runtime.** On Linux and Windows the OpenMP runtime is the one belonging to the
user's toolchain and is not redistributed by us. On macOS we bundle LLVM's `libomp`, which is
Apache-2.0 WITH LLVM-exception; its terms are reproduced below. We deliberately never redistribute
GCC's `libgomp`, which is GPL-3.0 WITH GCC-exception.

---

## Eigen 5.0.1 — MPL-2.0

Eigen is primarily licensed under the Mozilla Public License 2.0. Some files within it carry
third-party code under BSD or other MPL2-compatible licences (upstream `COPYING.BSD`,
`COPYING.APACHE`, `COPYING.MINPACK`). No part of the Eigen we use is LGPL-licensed.

The full MPL-2.0 text is at <https://www.mozilla.org/MPL/2.0/>.

**MPL-2.0 §3.2 — availability of Source Code Form.** DTWC++ binaries include Eigen in Executable
Form. The corresponding Source Form is the unmodified upstream release, which we do not patch, and
is available at:

> <https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.bz2>
> SHA-256 `e4de6b08f33fd8b8985d2f204381408c660bffa6170ac65b68ae1bd3cd575c0a`

## Apache Arrow nanoarrow 0.8.0 — Apache-2.0

A namespaced amalgamation of nanoarrow (`nanoarrow.h`, `nanoarrow.c`) is vendored at
`dtwc/extern/nanoarrow/` and compiled unconditionally into `dtwc++`.

The complete Apache License 2.0 text, as distributed by the project, is shipped alongside this file,
together with the upstream `NOTICE.txt` — in the CLI archive at `share/doc/dtwc/nanoarrow/`, and in
the wheel under `dtwcpp-<version>.dist-info/licenses/dtwc/extern/nanoarrow/`. The attribution notice
required by Apache-2.0 §4(d) is reproduced here in full:

```
Apache Arrow nanoarrow
Copyright 2023 The Apache Software Foundation

This product includes software developed at
The Apache Software Foundation (http://www.apache.org/).
```

## HiGHS 1.15.1 — MIT

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

Bundled only in macOS artefacts. The complete Apache License 2.0 text is shipped alongside this
file, at the `nanoarrow/` path given in the nanoarrow section above. The LLVM exception to it
reads, in full:

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

## llfio and quickcpplib — Apache-2.0 OR BSL-1.0

Optional, and **off** in every released artefact (`DTWC_ENABLE_LLFIO=OFF` in both the CLI release
workflow and the wheel build). If you build with it enabled and redistribute the result, both
projects are offered under your choice of two licences; DTWC++ elects the **Boost Software License
1.0**, reproduced in full:

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
