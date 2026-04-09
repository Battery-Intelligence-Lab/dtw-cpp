DTW-C++
===========================
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06881/status.svg)](https://doi.org/10.21105/joss.06881)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13551469.svg)](https://doi.org/10.5281/zenodo.13551469)
[![Website](https://img.shields.io/website?url=https%3A%2F%2FBattery-Intelligence-Lab.github.io%2Fdtw-cpp%2F)](https://Battery-Intelligence-Lab.github.io/dtw-cpp/)



[![Ubuntu unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/Ubuntu%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![macOS unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/macOS%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![Windows unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/Windows%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![Python tests](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/Battery-Intelligence-Lab/dtw-cpp/branch/main/graph/badge.svg?token=K739SRV4QG)](https://codecov.io/gh/Battery-Intelligence-Lab/dtw-cpp)

![Contributors](https://img.shields.io/github/contributors/Battery-Intelligence-Lab/dtw-cpp)
![Last update](https://img.shields.io/github/last-commit/Battery-Intelligence-Lab/dtw-cpp/develop)
![Issues](https://img.shields.io/github/issues/Battery-Intelligence-Lab/dtw-cpp)
![Forks](https://img.shields.io/github/forks/Battery-Intelligence-Lab/dtw-cpp)
![Stars](https://img.shields.io/github/stars/Battery-Intelligence-Lab/dtw-cpp)

![GitHub all releases](https://img.shields.io/github/downloads/Battery-Intelligence-Lab/dtw-cpp/total) 
[![](https://img.shields.io/badge/license-BSD--3--like-5AC451.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/LICENSE)

There is separate [detailed documentation](https://Battery-Intelligence-Lab.github.io/dtw-cpp/) available for this project; this `readme.md` file only gives a short summary. 

Introduction
===========================
DTW-C++ is a high-performance C++ library for Dynamic Time Warping (DTW) distance computation and time series clustering, with Python and MATLAB bindings.

**Key features:**

- **5 DTW variants**: Standard, Derivative (DDTW), Weighted (WDTW), Amerced (ADTW), Soft-DTW
- **Missing data support**: NaN-aware DTW (DTW-AROW)
- **3 clustering algorithms**: FastPAM k-medoids, FastCLARA (scalable), MIP (globally optimal)
- **LB pruning**: LB_Kim + LB_Keogh early-abandon for 9-11x faster distance matrices
- **Multi-language**: C++ core, Python (sklearn-compatible), MATLAB MEX bindings
- **Parallelism**: OpenMP threads, MPI distributed, CUDA GPU (optional)
- **Runtime float32 precision**: 2x memory saving with 0.003% max DTW error
- **RAM-aware streaming**: `--ram-limit` enables chunked CLARA for datasets exceeding memory
- **Checkpointing**: Save/resume long-running distance matrix computations
- **I/O**: CSV, HDF5, Parquet, Arrow IPC (zero-copy mmap) — auto-detected from extension

**Performance**: Beats aeon by 12x and dtaidistance by 1.7x on pairwise distance matrix construction. Full end-to-end clustering is 42x faster than aeon/tslearn.
<p align="center"><img src="./media/Merged_document.png" alt="DTW" width="60%"/></center></p>

Installation
===========================

### C++ (CMake)

```bash
cmake -S . -B build -DDTWC_BUILD_TESTING=ON
cmake --build build --config Release -j
cd build && ctest -C Release
```

For maintainer-style builds with strict warnings:

```bash
cmake -S . -B build-dev -DDTWC_DEV_MODE=ON -DDTWC_BUILD_TESTING=ON
cmake --build build-dev --config Debug -j
ctest --test-dir build-dev -C Debug
```

To add sanitizer instrumentation in a developer build, pass the specific maintainer option you want, for example `-Ddtwc_ENABLE_SANITIZER_ADDRESS=ON`.

### Python

```bash
pip install .          # install from source
# or for development:
pip install -e ".[test]"
pytest tests/python/ -v
```

### Optional dependencies

**MPI** (distributed distance matrix across multiple nodes):

```bash
# Linux
sudo apt install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi

# Windows: download MS-MPI from
# https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
# Install BOTH msmpisetup.exe (runtime) AND msmpisdk.msi (SDK)

# Build with MPI
cmake -S . -B build -DDTWC_ENABLE_MPI=ON -DDTWC_BUILD_TESTING=ON
cmake --build build --config Release -j
mpiexec -n 4 ./build/bin/unit_test_mpi
```

**CUDA** (GPU-accelerated batch DTW):

```bash
# Linux
sudo apt install cuda-toolkit-12-6  # or download from nvidia.com

# Windows: download CUDA Toolkit from
# https://developer.nvidia.com/cuda-downloads

# Verify
nvcc --version
nvidia-smi

# Build with CUDA
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON
cmake --build build --config Release -j
```

**MATLAB** (MEX bindings):

```bash
cmake -S . -B build -DDTWC_BUILD_MATLAB=ON
cmake --build build --config Release -j
# Requires MATLAB with C++ MEX compiler configured
```

### HPC / Supercomputer builds

DTW-C++ targets heterogeneous HPC clusters with a mix of CPU and GPU generations.

**Portable CPU build** — safe for all modern HPC CPUs (Broadwell, Haswell, Cascade Lake, Sapphire/Emerald Rapids, Rome, Genoa, Turin). Compile once on the login node, run on any compute node:

```bash
cmake -S . -B build -DDTWC_ARCH_LEVEL=v3
cmake --build build --config Release -j
```

**AVX-512 build** — for homogeneous clusters with Cascade Lake Xeon, Sapphire/Emerald Rapids, Genoa, or Turin nodes:

```bash
cmake -S . -B build -DDTWC_ARCH_LEVEL=v4
cmake --build build --config Release -j
```

**CUDA multi-arch build** — covers the full common HPC GPU fleet (V100→H100) by default. To target specific GPUs:

```bash
# Default: compiles for V100, A100, RTX Ampere, L40s, H100 (sm 70/80/86/89/90)
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON

# Single-arch build for A100-only cluster (faster compile):
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80

# Add P100 (sm_60) if needed:
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON -DDTWC_CUDA_ARCH_LIST="60;70;80;86;89;90"
```

**SIMD** is ON by default for standalone builds. Google Highway compiles for SSE4, AVX2, and AVX-512 in a single binary and dispatches at runtime — no need to rebuild for different node types.

**OpenMP on many-core NUMA nodes** (e.g. 288-core AMD Turin): bind threads to cores to avoid cross-NUMA memory traffic:

```bash
export OMP_NUM_THREADS=288
export OMP_PROC_BIND=close
export OMP_PLACES=cores
./build/bin/dtwc_main ...
```

### All CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `DTWC_BUILD_TESTING` | OFF | Build unit tests (Catch2) |
| `DTWC_BUILD_BENCHMARK` | OFF | Build benchmarks (Google Benchmark) |
| `DTWC_BUILD_PYTHON` | OFF | Build Python bindings (nanobind) |
| `DTWC_BUILD_MATLAB` | OFF | Build MATLAB MEX bindings |
| `DTWC_DEV_MODE` | OFF | Enable developer-only warnings, analyzers, and expose sanitizer options |
| `DTWC_ENABLE_MPI` | OFF | Enable MPI distributed computing |
| `DTWC_ENABLE_CUDA` | OFF | Enable CUDA GPU acceleration |
| `DTWC_ENABLE_ARROW` | OFF | Enable Apache Arrow IPC + Parquet I/O (via `find_package` or CPM) |
| `DTWC_ENABLE_GUROBI` | ON | Enable Gurobi MIP solver (optional) |
| `DTWC_ENABLE_HIGHS` | ON | Enable HiGHS MIP solver (optional) |
| `DTWC_ENABLE_SIMD` | ON* | Enable Highway SIMD with runtime ISA dispatch (*ON for standalone builds, OFF for sub-projects/Python wheels) |
| `DTWC_ENABLE_NATIVE_ARCH` | ON | Tune for host CPU (`-march=native`); disable for portable binaries |
| `DTWC_ARCH_LEVEL` | `""` | Override native arch: `v3` (AVX2+FMA, all modern HPC CPUs), `v4` (AVX-512) |
| `DTWC_CUDA_ARCH_LIST` | `70;80;86;89;90` | CUDA architectures when `CMAKE_CUDA_ARCHITECTURES` is not set |

Citation
===========================

APA style: 
```
Kumtepeli, V., Perriment, R., & Howey, D. A. (2024). DTW-C++: Fast dynamic time warping and clustering of time series data. Journal of Open Source Software, 9(101), 6881. https://doi.org/10.21105/joss.06881
```

BibTeX: 
```
@article{Kumtepeli2024,
author = {Kumtepeli, Volkan and Perriment, Rebecca and Howey, David A.},
doi = {10.21105/joss.06881},
journal = {Journal of Open Source Software},
month = sep,
number = {101},
pages = {6881},
title = {{DTW-C++: Fast dynamic time warping and clustering of time series data}},
url = {https://joss.theoj.org/papers/10.21105/joss.06881},
volume = {9},
year = {2024}
}
```

Contributors
===========================
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section --><!-- prettier-ignore-start --><!-- markdownlint-disable -->
<table>
	<tbody>
		<tr>
			<td style="text-align:center; vertical-align:top"><a href="https://github.com/beckyperriment"><img alt="Becky Perriment" src="https://avatars.githubusercontent.com/u/93582518?v=4?s=100" style="width:100px" /><br />
			<sub><strong>Becky Perriment</strong></sub></a><br />
			<a href="https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/develop/contributors.md#core-contributors">💡💻👀⚠️</a></td>
			<td style="text-align:center; vertical-align:top"><a href="https://github.com/ElektrikAkar"><img alt="Volkan Kumtepeli" src="https://avatars.githubusercontent.com/u/8674942?v=4?s=100" style="width:100px" /><br />
			<sub><strong>Volkan Kumtepeli</strong></sub></a><br />
			<a href="https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/develop/contributors.md#core-contributors">💡💻👀⚠️🚇🐢</a></td>
			<td style="text-align:center; vertical-align:top"><a href="http://howey.eng.ox.ac.uk"><img alt="David Howey" src="https://avatars.githubusercontent.com/u/2247552?v=4?s=100" style="width:100px" /><br />
			<sub><strong>David Howey</strong></sub></a><br />
			<a href="https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/develop/contributors.md#core-contributors">💡👀</a></td>
		</tr>
	</tbody>
</table>
<!-- markdownlint-restore --><!-- prettier-ignore-end --><!-- ALL-CONTRIBUTORS-LIST:END -->
