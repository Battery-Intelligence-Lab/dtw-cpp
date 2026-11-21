---
title: Installation
weight: 1
---

# Using DTW-C++

DTW-C++ can be used as a C++ library, from the command line, or via its Python and MATLAB bindings:

- Write a C++ program against the library, starting from `examples/cpp` (`-DDTWC_BUILD_EXAMPLES=ON`).
- Run the `dtwc_cl` command-line executable after building from source or unpacking a release archive.
- Link the `dtwc++` CMake target in an external C++ project via `add_subdirectory(dtw-cpp)` or [CPM](https://github.com/cpm-cmake/) (`dtwc++` links `dtwc_core`, the algorithms and the CSV/TSV and Arrow IPC readers; `dtwc_cli`, `dtwc::run` and `dtwc::cluster`; and, in a build with `-DDTWC_ENABLE_ARROW=ON`, `dtwc_io`, the Parquet reader: link `dtwc_core` alone for the algorithms without the others). DTW-C++ has no built-in data or results folder: pass your data paths to the loaders, and set a `Problem`'s output folder with `set_output_folder` (the default is `./results/`, relative to the working directory).

## CPU requirement

The Python wheels and the CLI archives of a release are built for x86-64-v3: an x86-64 CPU with AVX2 and FMA, which is Intel Haswell (2013) or later and AMD Excavator (2015) or later, so every Ryzen. On an older CPU they stop with an illegal-instruction error; some low-end Pentium, Celeron and Atom parts lack AVX2 whatever their date. Apple Silicon and Linux arm64 builds are not affected.

A build from source tunes for the machine it is built on (`-DDTWC_ARCH_LEVEL=native`, the default for C++ builds), so it runs only there and on CPUs that have the same instructions. `-DDTWC_ARCH_LEVEL=v3` builds for the floor above, which runs on every CPU that meets it; `-DDTWC_ARCH_LEVEL=v4` targets AVX-512. A Python build from source (`pip install .`) targets `v3` like the wheels; [Python Installation](#python-installation) shows how to tune it for the build machine.

## The Windows CLI archive

A release's Windows archive (`dtwc-<version>-Windows-AMD64.zip`) holds two builds of the command line interface in `bin/`: `dtwc_cl.exe`, built by clang-cl, and `dtwc_cl_msvc.exe`, built by MSVC. In our tests on an Intel Core Ultra 9 285, `dtwc_cl.exe` filled the distance matrices of equal-length series about 4× faster than `dtwc_cl_msvc.exe` with 24 threads (about 8× on one thread), so we recommend `dtwc_cl.exe`; if it gives you trouble, `dtwc_cl_msvc.exe` takes the same arguments. Each runs from the unpacked folder without the Visual C++ Redistributable: `bin/` holds the runtime DLLs they import, the Visual C++ runtime and each one's OpenMP runtime (LLVM's `libomp.dll` for `dtwc_cl.exe`, MSVC's `vcomp140.dll` for `dtwc_cl_msvc.exe`).

## Dependencies

Install before building from source:
- [CMake](https://cmake.org/) 3.26 or newer
- A C++20 compiler (GCC, Clang, or MSVC)
- [OpenMP](https://www.openmp.org/) (bundled with GCC and MSVC; for Clang on Linux install `libomp-<ver>-dev`, where `<ver>` is your Clang version; on macOS see [below](#macos))
- Gurobi (optional and off by default, free for academic use: HiGHS is the MIP solver unless you pass `-DDTWC_ENABLE_GUROBI=ON`, which stops with an error if Gurobi is not found; see installation guides for [Ubuntu](https://www.youtube.com/watch?v=yNmeG6Wom1o), [macOS](https://www.youtube.com/watch?v=ZcL-NmckTxQ), [Windows](https://www.youtube.com/watch?v=z7t0p5J9YcQ), and [further information](https://support.gurobi.com/hc/en-us/sections/360010017231-Platforms-and-Installation))

Fetched automatically via CPM when needed:
- [HiGHS](https://highs.dev/), the open-source MIP solver (`-DDTWC_ENABLE_HIGHS=ON`, default)
- [CLI11](https://github.com/CLIUtils/CLI11) (command-line parser)
- [cpuinfo](https://github.com/pytorch/cpuinfo) (`-DDTWC_ENABLE_CPUINFO=ON`; when `OFF`, CPU facts in `dtwc_cl -v` read `unknown`)
- [lz4](https://github.com/lz4/lz4) and [zstd](https://github.com/facebook/zstd) (compressed Arrow IPC input)
- [fkYAML](https://github.com/fktn-k/fkYAML) (`-DDTWC_ENABLE_YAML=ON`, YAML `--config` files)
- [llfio](https://github.com/ned14/llfio) with quickcpplib, outcome, span-lite, byte-lite, wg14_signals, and on Windows ntkernel-error-category (`-DDTWC_ENABLE_LLFIO=ON`, memory-mapped distance matrices)
- [CPMLicenses.cmake](https://github.com/cpm-cmake/CPMLicenses.cmake) (`write-licenses` target)
- [Catch2](https://github.com/catchorg/Catch2/) (`-DDTWC_BUILD_TESTING=ON`) and [Google Benchmark](https://github.com/google/benchmark) (`-DDTWC_BUILD_BENCHMARK=ON`)
- [Apache Arrow](https://github.com/apache/arrow) (`-DDTWC_ENABLE_ARROW=ON`, when no system Arrow is found)

## Building from source

Install the prerequisites for your operating system (below), then clone the repository (or download it as a [*.zip file](https://github.com/battery-intelligence-lab/dtw-cpp/archive/refs/heads/main.zip)) and build with CMake:

```bash
git clone https://github.com/battery-intelligence-lab/dtw-cpp.git
cd dtw-cpp
cmake -S . -B build
cmake --build build --config Release -j4
./build/bin/dtwc_cl   # the command line interface
```

`-j4` runs four parallel build jobs; on a machine with more cores you may use up to twice the number of processors (on Linux, `-j$(( $(nproc) * 2 -1))`). If the program crashes, build with `cmake --build build --config Debug` for a better message. For further information on using CMake, see the [CMake guide](https://cmake.org/cmake/help/git-stage/index.html).

### Linux (Debian / Ubuntu 20.04+)

DTWC++ requires CMake 3.26 or newer and a C++20-capable compiler. Distribution packages that do not meet those floors must be upgraded before configuring the project. Install a newer GCC when the distribution compiler is too old; CI builds with GCC 11 and 12.

```bash
sudo apt update
sudo apt install -y build-essential cmake-extras cmake
```

### macOS

Install the latest version of [Xcode](https://developer.apple.com/support/xcode/), then the command line tools, [Homebrew](https://brew.sh/) and CMake:

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install cmake
brew install libomp llvm && brew link --force libomp
```

> **Note:** Apple Clang does not include OpenMP support. Install it via Homebrew:
> `brew install libomp && brew link --force libomp`
> This applies to both Intel and Apple Silicon Macs.

> **Building for other Macs:** Homebrew installs the `libomp` bottle built for your macOS
> release, and that release is its minimum, so a binary that bundles it will not start on an
> older macOS. The released wheels and CLI archives (macOS 13.3 or later) instead bundle a runtime
> built by `bash scripts/build_libomp_macos.sh <prefix>`; configure with
> `-DOpenMP_ROOT=<prefix> -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3` to do the same.

### Windows

Install [CMake](https://cmake.org/download/#latest) and either MSVC/Clang via [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/) (even if you do not use the IDE) or MinGW GCC via [Chocolatey](https://docs.chocolatey.org/en-us/choco/setup/#non-administrative-install) (`choco install mingw`), then select your generator:

```bash
cmake -S . -B build -G "MinGW Makefiles"        # if you use MinGW GCC
cmake -S . -B build -G "Visual Studio 17 2022"  # if you use Visual Studio's compiler
cmake --build build --config Release -j4
```

Visual Studio Community can also [open the folder directly](https://docs.microsoft.com/en-us/visualstudio/ide/develop-code-in-visual-studio-without-projects-or-solutions?view=vs-2019), without using CMake.

## Python Installation

The easiest way to use DTW-C++ from Python:

```bash
pip install dtwcpp
```

The wheel links no HiGHS: exact MIP clustering (`method="mip"`) and LR-core's Kelley root (`lr_root="kelley"`) solve with highspy, which `pip install dtwcpp` installs with it.

The macOS and Windows wheels carry LLVM's OpenMP runtime (`libomp`). A process that also loads Intel's OpenMP runtime (`libiomp5`, as PyTorch and MKL builds do) can abort with "OMP: Error #15"; keep such work in a separate process.

To build from source with GPU support:

```bash
pip install . --config-settings=cmake.define.DTWC_ENABLE_CUDA=ON
```

A build from source targets x86-64-v3 like the wheels (see [CPU requirement](#cpu-requirement)). To tune it for the machine it is built on instead:

```bash
pip install . --config-settings=cmake.define.DTWC_ARCH_LEVEL=native
```

## GPU Acceleration

GPU acceleration is optional. For NVIDIA GPUs on Linux or Windows (`-DDTWC_ENABLE_CUDA=ON`), see [CUDA Setup](cuda-setup.md). On macOS, CUDA is unavailable and the GPU backend is Metal (`-DDTWC_ENABLE_METAL`, on by default on Apple platforms; see [GPU backends](../../method/gpu-backends/)).

## Visual Studio Code

Install [Visual Studio Code](https://code.visualstudio.com/download) with the `C/C++ Extension Pack` and `CMake Tools` extensions, open the `dtw-cpp` folder, select a 64-bit kit when prompted, and build `dtwc_cl` or an example target (`-DDTWC_BUILD_EXAMPLES=ON`).

## Importing as a library

For a standalone CMake consumer template, see [`examples/cpp/example_project`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/tree/main/examples/cpp/example_project).
