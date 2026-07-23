DTW-C++
===========================
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06881/status.svg)](https://doi.org/10.21105/joss.06881)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13551469.svg)](https://doi.org/10.5281/zenodo.13551469)
[![Website](https://img.shields.io/website?url=https%3A%2F%2FBattery-Intelligence-Lab.github.io%2Fdtw-cpp%2F)](https://Battery-Intelligence-Lab.github.io/dtw-cpp/)



[![Ubuntu unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/Ubuntu%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![macOS unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/macOS%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![Windows unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/Windows%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![Python tests](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/Battery-Intelligence-Lab/dtw-cpp/branch/main/graph/badge.svg)](https://codecov.io/gh/Battery-Intelligence-Lab/dtw-cpp)

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

- **7 elastic distances**: Standard DTW, DDTW, WDTW, ADTW, Soft-DTW, MSM, TWE
- **Missing data support**: NaN-aware DTW (DTW-AROW)
- **8 CLI-selectable clustering methods**: PAM/FastPAM, OneBatchPAM, FastCLARA, k-medoids, MIP, LR-core, hierarchical, TADPole
- **Lower bounds**: Keogh/Webb bounds and admissible TADPole pair pruning
- **Multi-language**: C++ core, Python (sklearn-compatible), MATLAB MEX bindings
- **Parallelism**: OpenMP threads, MPI distributed, CUDA and Metal GPUs (optional)
- **Runtime precision**: Float64 by default; explicit Float32 halves series-storage bytes and uses Float32 recurrence arithmetic
- **RAM-aware streaming**: `--ram-limit` bounds Parquet series materialisation and streams supported one-list-row-per-series non-full FastCLARA workloads
- **Checkpointing**: Save/resume long-running distance matrix computations
- **I/O**: CSV, HDF5, Parquet, Arrow IPC, and native `.dtws`, gated by compiled capabilities and auto-detected from extension

Recorded, workload-specific measurements are published in the
[UCR benchmark ledger](benchmarks/ucr_benchmark_results.md); do not extrapolate
one machine or dataset into a universal speedup.

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

#### macOS (Apple Clang + Homebrew libomp)

Apple Clang on macOS ships without OpenMP; install `libomp` via Homebrew. Ninja is required by the `clang-macos` preset.

```bash
brew install ninja libomp
brew link --force libomp
cmake --preset clang-macos -DDTWC_BUILD_TESTING=ON
cmake --build build --config Release -j
cd build && ctest -C Release
```

The `clang-macos` preset pins `/usr/bin/clang++` to avoid libc++ ABI conflicts when Homebrew LLVM is also installed. If OpenMP auto-detection fails, pass explicit hints:

```bash
cmake --preset clang-macos \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"
```

Gurobi on macOS installs to `/Library/gurobi<version>/macos_universal2/` — `FindGUROBI.cmake` auto-detects this location, or set `GUROBI_HOME` explicitly.

### Python

We recommend [`uv`](https://docs.astral.sh/uv/) for Python (faster and more reproducible than pip):

```bash
uv pip install .          # install from source
# or for development:
uv pip install -e ".[test]"
pytest tests/python/ -v
```

`pip install .` works too if you prefer — both use the same `pyproject.toml` (scikit-build-core + nanobind).

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
# Default: P100, V100, Turing, A100, RTX Ampere, Ada/L40s, H100 (sm 60/70/75/80/86/89/90)
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON

# Single-arch build for A100-only cluster (faster compile):
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80

# Override the default list if a narrower fleet is required:
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON -DDTWC_CUDA_ARCH_LIST="80;90"
```

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
| `DTWC_BUILD_EXAMPLES` | OFF | Build example programs |
| `DTWC_BUILD_TESTING` | OFF | Build unit tests (Catch2) |
| `DTWC_BUILD_BENCHMARK` | OFF | Build benchmarks (Google Benchmark) |
| `DTWC_BUILD_PYTHON` | OFF | Build Python bindings (nanobind) |
| `DTWC_BUILD_MATLAB` | OFF | Build MATLAB MEX bindings |
| `DTWC_DEV_MODE` | OFF | Enable developer-only warnings, analyzers, and expose sanitizer options |
| `DTWC_ALLOW_SEQUENTIAL` | OFF | Explicitly permit a build without OpenMP; otherwise missing OpenMP is an error |
| `DTWC_ENABLE_MPI` | OFF | Enable MPI distributed computing |
| `DTWC_ENABLE_CUDA` | OFF | Enable CUDA GPU acceleration |
| `DTWC_ENABLE_METAL` | ON | Enable the Metal backend on Apple platforms |
| `DTWC_ENABLE_ARROW` | OFF | Enable Apache Arrow IPC + Parquet I/O (system packages or CPM) |
| `DTWC_ENABLE_YAML` | OFF | Enable YAML configuration files via yaml-cpp |
| `DTWC_ENABLE_LLFIO` | ON | Enable llfio-backed memory-mapped distance matrices |
| `DTWC_ENABLE_GUROBI` | ON | Enable Gurobi MIP solver (optional) |
| `DTWC_ENABLE_HIGHS` | ON | Enable HiGHS MIP solver (optional) |
| `DTWC_HIGHS_GPU` | OFF | Build the optional HiGHS PDLP CUDA backend |
| `DTWC_ENABLE_NATIVE_ARCH` | ON | Tune for host CPU (`-march=native`); disable for portable binaries |
| `DTWC_REPRODUCIBLE_BUILD` | OFF | Strip source/build paths from supported compiler outputs |
| `DTWC_ARCH_LEVEL` | `""` | Override native arch: `v3` (AVX2+FMA, all modern HPC CPUs), `v4` (AVX-512) |
| `DTWC_CUDA_ARCH_LIST` | `60;70;75;80;86;89;90` | CUDA architectures when `CMAKE_CUDA_ARCHITECTURES` is not set |

AI-assisted workflow (Claude Code)
===========================

DTWC++ ships with Claude Code slash commands in `.claude/commands/` so users can drive the library through their AI assistant:

| Command | Purpose |
|---------|---------|
| `/cluster` | Full pipeline: load data, pick method, cluster, evaluate, save |
| `/distance` | Compute DTW distances (single pair or full pairwise matrix) |
| `/evaluate` | Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI |
| `/convert` | Convert between CSV, Parquet, Arrow IPC, HDF5 |
| `/visualize` | Plot clusters, silhouette, distance-matrix heatmap, warping path |
| `/help` | Algorithm selection, variant guide, parameter tuning reference |
| `/troubleshoot` | Diagnose build errors, runtime crashes, performance |

Example: from within Claude Code, type `/cluster data/ecg.csv -k 5 --variant wdtw` and the assistant generates and runs a complete Python script using the DTWC++ API, then reports results and suggests next steps.

Commands are discovered automatically when Claude Code is launched from the repository root. See [`docs/content/getting-started/ai-commands.md`](docs/content/getting-started/ai-commands.md) for details.

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
