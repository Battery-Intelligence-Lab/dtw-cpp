---
title: Installation
weight: 1
---

# Using DTW-C++ 

DTW-C++ can be used as a C++ library, from the command line, or via its Python bindings. The sections below cover building from source for C++ use. For Python, see the Python section below.

- Write a C++ program against the library, starting from the programs in `examples/cpp` (built with `-DDTWC_BUILD_EXAMPLES=ON`).
- Use DTW-C++ from the command line interface, by using the `dtwc_cl` executable after compilation. 
- Use DTW-C++ as an external library in your C++ project by linking the `dtwc++` target in your project (it links `dtwc_core`, the algorithms and the CSV/TSV reader; `dtwc_cli`, `dtwc::run` and `dtwc::cluster`; and, in a build with `-DDTWC_ENABLE_ARROW=ON`, `dtwc_io`, the Parquet and Arrow IPC readers: link `dtwc_core` alone for the algorithms without the others). Download the source code to your folder of preference, include the line `add_subdirectory(dtw-cpp)` in your `CMakeLists.txt` file. Then link your library. Alternatively, you may also use [CPM](https://github.com/cpm-cmake/) to interactively download and include DTW-C++. DTW-C++ has no built-in data or results folder: pass your data paths to the loaders, and set a `Problem`'s output folder with `set_output_folder` (the default is `./results/`, relative to the working directory).

## CPU requirement

The Python wheels and the CLI archives of a release are built for x86-64-v3: an x86-64 CPU with AVX2 and FMA, which is Intel Haswell (2013) or later and AMD Excavator (2015) or later, so every Ryzen. On an older CPU they stop with an illegal-instruction error; some low-end Pentium, Celeron and Atom parts lack AVX2 whatever their date. Apple Silicon and Linux arm64 builds are not affected.

A build from source tunes for the machine it is built on (`-DDTWC_ARCH_LEVEL=native`, the default for C++ builds), so it runs only there and on CPUs that have the same instructions. `-DDTWC_ARCH_LEVEL=v3` builds for the floor above, which runs on every CPU that meets it; `-DDTWC_ARCH_LEVEL=v4` targets AVX-512. A Python build from source (`pip install .`) targets `v3` like the wheels; [Python Installation](#python-installation) shows how to tune it for the build machine.

## Dependencies

DTW-C++ aims to be easily compilable and usable; therefore, it includes only a few libraries where most of the dependencies are automatically installed.  

There are several pre-requisite installations required to compile and run DTW-C++.

The following dependencies need to be manually installed by the user if they do not already exist:
- [CMake](https://cmake.org/)
- A suitable compiler (Clang, GCC, MSVC, etc.)
- Gurobi (optional and off by default: HiGHS is the MIP solver unless you configure with `-DDTWC_ENABLE_GUROBI=ON`, which stops with an error when Gurobi is not found)
- [OpenMP](https://www.openmp.org/) this should come with GCC and MSVC libraries; however to install it with Clang, you may install `libomp-xx-dev` where `xx` is your clang version. 

The following dependencies are installed by the CPM package manager:
- [HiGHS](https://highs.dev/) as an open source MIP solver alternative to Gurobi.
- [CLI11](https://github.com/CLIUtils/CLI11) (for command line interface)
- [Catch2](https://github.com/catchorg/Catch2/) (for testing)

### CMake and compilers

[CMake](https://cmake.org/) is a metabuild system required to provide the correct compilation commands to the compiler. It will be explained how to install CMake and compilers depending on your operating system below. 

### Gurobi

Gurobi is a powerful optimisation solver that is free for academic use. Gurobi is off by default and HiGHS is the MIP solver; to link Gurobi configure with `-DDTWC_ENABLE_GUROBI=ON`, which stops with an error if Gurobi is not found. Please see the following guidelines for installation on [Ubuntu](https://www.youtube.com/watch?v=yNmeG6Wom1o), [macOS](https://www.youtube.com/watch?v=ZcL-NmckTxQ), [Windows](https://www.youtube.com/watch?v=z7t0p5J9YcQ), and [further information](https://support.gurobi.com/hc/en-us/sections/360010017231-Platforms-and-Installation)


## Building from the source

DTW-C++ aims to be compatible with different compilers and platforms. You may easily install DTW-C++ using CMake (although it is not an absolute requirement). Therefore, you need a suitable C++ compiler (preferably [GCC](https://gcc.gnu.org/)) and [CMake](https://cmake.org/) to follow this installation guide.   

### Linux (Debian / Ubuntu 20.04+)

DTWC++ requires CMake 3.26 or newer and a C++20-capable compiler. Distribution packages that do not meet those floors must be upgraded before configuring the project. Please refer to [install a newer version of GCC](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/) when the distribution compiler is too old.

1. Install the essential libraries for building the project

    ```bash
    sudo apt update
    sudo apt install -y build-essential cmake-extras cmake
    ```

2. Clone the repository using the following command or download it as a [*.zip file](https://github.com/battery-intelligence-lab/dtw-cpp/archive/refs/heads/main.zip): 
    ```bash
    git clone https://github.com/battery-intelligence-lab/dtw-cpp.git
    ```
3. After downloading the source files, you need to create a build folder and go into the build folder:
    ```bash
    cd dtw-cpp  # Go into the main directory of source files.
    mkdir build # Create a build folder if it is not present.
    cd build # Go into the build directory. 
    ```
4. Then, create Makefiles by running:
    ```bash
    cmake -G "Unix Makefiles" .. 
    ```
5. Compile the files. Here `-j4` command specifies the number of parallel jobs (threads) to use during the build process and `4` is given as example. For a more powerful computer with many cores you may opt for up to double number of the processors you have. Programmatically you may also use `-j$(( $(nproc) * 2 -1))` where `$(nproc)` denotes number of processors you have. 
    ```bash
    cmake --build . -j4 --config Release
    ```
6. After this, the `dtwc_cl` executable will be ready to run in the `dtw-cpp/bin` folder:
    ```bash
    cd ../bin
    ./dtwc_cl # to run the command line interface
    ```

```note
In case you encounter sudden crash of the program, you may also try to complile the program with ```--config Debug```, where you can receive a better message for the crash. For further information on using CMake, please refer to the [CMake guide](https://cmake.org/cmake/help/git-stage/index.html).
```

### macOS 

1. Install the latest version of [Xcode](https://developer.apple.com/support/xcode/).
2. Install command line tools, [Homebrew](https://brew.sh/) and CMake by executing following commands on the terminal: 
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

3. Clone the repository using the following command or download it as a [*.zip file](https://github.com/battery-intelligence-lab/dtw-cpp/archive/refs/heads/main.zip): 
    ```bash
    git clone https://github.com/battery-intelligence-lab/dtw-cpp.git
    ```
4. After downloading the source files, you need to create a build folder and go into the build folder:
    ```bash
    cd dtw-cpp  # Go into the main directory of source files.
    mkdir build # Create a build folder if it is not present.
    cd build # Go into the build directory. 
    ```
5. Then, create Makefiles by running:
    ```bash
    cmake .. 
    ```
6. Compile the files. Here `-j4` command specifies the number of parallel jobs (threads) to use during the build process and `4` is given as example. For a more powerful computer with many cores you may opt for up to double number of the processors you have. Programmatically you may also use `-j$(( $(nproc) * 2 -1))` where `$(nproc)` denotes number of processors you have. 
    ```bash
    cmake --build . -j4 --config Release
    ```
7. After this, the `dtwc_cl` executable will be ready to run in the `dtw-cpp/bin` folder:
    ```bash
    cd ../bin
    ./dtwc_cl # to run the command line interface
    ```

```note
In case you encounter sudden crash of the program, you may also try to complile the program with ```--config Debug```, where you can receive a better message for the crash. For further information on using CMake, please refer to the [CMake guide](https://cmake.org/cmake/help/git-stage/index.html).
```

### Windows

On Windows platforms, you probably need to install CMake and a C++ compiler:

1. Install the latest version of the [CMake binary](https://cmake.org/download/#latest).
2. You can then install a compiler of your choice. We suggest installing MVSC and/or Clang via [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/) even though you do not use the IDE which comes with its own compiler. Otherwise, for `GCC`, easiest way is to install package manager [chocolatey](https://docs.chocolatey.org/en-us/choco/setup/#non-administrative-install) then run `choco install mingw` command. 
3. Download the repository as a [*.zip file](https://github.com/battery-intelligence-lab/dtw-cpp/archive/refs/heads/main.zip), or if you have [git](https://git-scm.com/download/win) installed you may run the following command in the git bash: 
    ```bash
    git clone https://github.com/battery-intelligence-lab/dtw-cpp.git
    ```
3. After downloading the source files, you need to create a build folder and go into the build folder. You may type the following commands into the command line:
    ```bash
    cd dtw-cpp  # Go into the main directory of source files.
    mkdir build # Create a build folder if it is not present.
    cd build # Go into the build directory. 
    ```
4. Then, create compilation files by running:
    ```bash
    cmake -G "MinGW Makefiles" .. # if you use MinGW GCC compiler.
    cmake -G "Visual Studio 16 2019" ..  # if you use Visual Studio's compiler.
    ```
5. Compile the files. Here `-j4` command specifies the number of parallel jobs (threads) to use during the build process and `4` is given as example. For a more powerful computer with many cores you may opt for up to double number of the processors you have. 
    ```bash
    cmake --build . -j4 --config Release
    ```
6. After this, the `dtwc_cl` executable will be ready to run in the `dtw-cpp/bin` folder:
    ```bash
    cd ../bin
    ./dtwc_cl # to run the command line interface
    ```

```note
In case you encounter sudden crash of the program, you may also try to complile the program with ```--config Debug```, where you can receive a better message for the crash. For further information on using CMake, please refer to the [CMake guide](https://cmake.org/cmake/help/git-stage/index.html).
```

```note
If you are using Visual Studio Community, you may also open the folder in Visual Studio directly, without using CMake. See [this page](https://docs.microsoft.com/en-us/visualstudio/ide/develop-code-in-visual-studio-without-projects-or-solutions?view=vs-2019) for detailed explanation.
```

## Python Installation

The easiest way to use DTW-C++ from Python:

```bash
pip install dtwcpp
```

Exact MIP clustering (`method="mip"`) solves with highspy: `pip install "dtwcpp[mip]"`.

To build from source with GPU support:

```bash
pip install . --config-settings=cmake.define.DTWC_ENABLE_CUDA=ON
```

A build from source targets x86-64-v3 like the wheels (see [CPU requirement](#cpu-requirement)). To tune it for the machine it is built on instead:

```bash
pip install . --config-settings=cmake.define.DTWC_ARCH_LEVEL=native
```

See [CUDA Setup](cuda-setup.md) for detailed GPU configuration.

## GPU Acceleration

CUDA GPU acceleration is optional. See the dedicated [CUDA Setup Guide](cuda-setup.md) for installation instructions.

> **Note:** CUDA is not available on macOS. Use Linux or Windows with an NVIDIA GPU.

## Visual Studio Code

Visual Studio Code (VScode) is one of the powerful editors and we personally prefer using this editor. To use this editor: 

1. Download and install [Visual Studio Code](https://code.visualstudio.com/download)
2. Install `CMake` and a suitable compiler and download using the above guidelines for our operating system. 
3. Download the `DTW-C++` code using `git` or `zip`.
4. Open VScode and install extensions `C/C++ Extension Pack` and `CMake Tools`. 
5. Open the `dtw-cpp` folder with the VScode. 
6. Let the VScode to configure the folder. Now it will scan the kits where you can select a suitable kit (use the 64-bit kits). 
7. It will compile all targets; select `dtwc_cl`, or an example target (with `DTWC_BUILD_EXAMPLES=ON`), to run.

## Importing as a library

DTW-C++ is designed to be used both as a standalone application and a library where you can add into your existing C++ code. You may copy and use the [example project on our GitHub page](https://github.com/Battery-Intelligence-Lab/dtw-cpp/tree/main/examples/example_project)
