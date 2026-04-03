---
title: MPI and CUDA Setup
weight: 4
---

# MPI and CUDA Setup

DTWC++ supports optional MPI (for distributed-memory parallelism) and CUDA (for GPU acceleration) backends. Both are **optional** -- the core library builds and runs without them. This guide covers installation and CMake configuration for each.

## Requirements

| Feature | Minimum Version | Notes |
|---------|----------------|-------|
| MPI | MPI-3.0 (any implementation) | MS-MPI on Windows, OpenMPI or MPICH on Linux/macOS |
| CUDA | 11.0+ | Requires NVIDIA GPU with compute capability 6.0 or newer |
| GPU | NVIDIA Kepler or newer | AMD ROCm support is planned for future releases |

---

## MPI Installation

### Windows (Microsoft MPI)

MS-MPI is the standard MPI implementation on Windows.

1. Download from [https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).
2. Install **both** components:
   - `msmpisetup.exe` -- the runtime (needed to run MPI programs)
   - `msmpisdk.msi` -- the SDK (headers and import libraries, needed to build MPI programs)
3. Verify the installation in a Command Prompt or PowerShell:
   ```
   mpiexec --version
   ```
   This should print the MS-MPI version (e.g. `Microsoft MPI Version 10.x`).
4. The installer automatically sets the following environment variables:
   - `MSMPI_INC` -- path to MPI headers
   - `MSMPI_LIB32` -- path to 32-bit import libraries
   - `MSMPI_LIB64` -- path to 64-bit import libraries

   CMake finds MS-MPI via `find_package(MPI)` using these variables. If CMake cannot find MPI, ensure these environment variables are set (you may need to restart your terminal after installation).

### Linux (Ubuntu / Debian)

Install either OpenMPI or MPICH -- both work. Pick one, not both.

```bash
# OpenMPI (most common)
sudo apt update
sudo apt install -y libopenmpi-dev openmpi-bin

# OR MPICH
sudo apt update
sudo apt install -y libmpich-dev mpich
```

Verify:
```bash
mpirun --version
# or
mpiexec --version
```

### Linux (RHEL / CentOS / Fedora)

```bash
sudo dnf install openmpi-devel
```

On RHEL-family systems, MPI is installed as a module. You must load it before use:

```bash
module load mpi/openmpi-x86_64
```

To make this permanent, add the `module load` line to your `~/.bashrc`.

Verify:
```bash
mpirun --version
```

### macOS

Install via [Homebrew](https://brew.sh/):

```bash
# OpenMPI (recommended)
brew install open-mpi

# OR MPICH
brew install mpich
```

Verify:
```bash
mpirun --version
```

### Verify CMake Detection

After installing MPI, confirm that CMake can find it:

```bash
cmake -S . -B build -DDTWC_ENABLE_MPI=ON
```

Look for output like:
```
-- Found MPI_CXX: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
```

If CMake reports `Could NOT find MPI`, check that the MPI compiler wrappers (`mpicxx`, `mpic++`) are on your `PATH`.

---

## CUDA Installation

### Windows

1. Download the CUDA Toolkit from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
   - Select: Windows > x86_64 > your Windows version > exe (local).
2. Run the installer with default options. This installs the `nvcc` compiler, cuBLAS, and other CUDA libraries.
3. Add CUDA to your system `PATH` if the installer did not do so automatically:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
   ```
   Replace `v12.x` with your installed version.
4. Verify:
   ```
   nvcc --version
   ```
   This should print the CUDA compilation tools version.
5. Verify that your GPU driver is installed and the GPU is detected:
   ```
   nvidia-smi
   ```

CMake finds CUDA via `enable_language(CUDA)` or `find_package(CUDAToolkit)`.

### Linux (Ubuntu / Debian)

**Option 1: NVIDIA repository (recommended -- provides the latest version)**

```bash
# For Ubuntu 22.04 (adjust URL for your version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
```

**Option 2: Ubuntu packages (older version, simpler setup)**

```bash
sudo apt install -y nvidia-cuda-toolkit
```

After installation, ensure CUDA is on your `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Add these lines to your `~/.bashrc` to make them permanent.

> **Tip:** If `nvcc` is not on your PATH, set the `CUDA_PATH` environment variable:
> `export CUDA_PATH=/usr/local/cuda`
> CMake will use this to locate `nvcc` automatically.

Verify:
```bash
nvcc --version
nvidia-smi
```

### Linux (RHEL / CentOS / Fedora)

Follow the [NVIDIA CUDA installation guide for RHEL](https://developer.nvidia.com/cuda-downloads) to add the NVIDIA repository, then:

```bash
sudo dnf install -y cuda-toolkit-12-6
```

Set up `PATH` and `LD_LIBRARY_PATH` as described in the Ubuntu section above.

### macOS

CUDA is **not supported** on macOS. Apple dropped NVIDIA driver support starting with macOS 10.14 (Mojave). There is no workaround -- if you need GPU acceleration, use a Linux or Windows machine with an NVIDIA GPU.

### Verify CMake Detection

After installing CUDA, confirm that CMake can find it:

```bash
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON
```

Look for output like:
```
-- Found CUDAToolkit: /usr/local/cuda/include (found version "12.6.x")
```

---

## Building DTWC++ with MPI and/or CUDA

```bash
# MPI only
cmake -S . -B build -DDTWC_ENABLE_MPI=ON
cmake --build build --config Release

# CUDA only
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON
cmake --build build --config Release

# Both MPI and CUDA
cmake -S . -B build -DDTWC_ENABLE_MPI=ON -DDTWC_ENABLE_CUDA=ON
cmake --build build --config Release
```

To run an MPI-enabled test or binary:

```bash
mpiexec -n 4 ./build/bin/test_mpi_distmat
```

Replace `-n 4` with the number of MPI processes you want to launch.

---

## Troubleshooting

### MPI

| Problem | Cause | Fix |
|---------|-------|-----|
| `Could NOT find MPI` in CMake | MPI compiler wrappers not on PATH | Ensure `mpicxx` is on your PATH. On RHEL, run `module load mpi/openmpi-x86_64` first. |
| `mpiexec: command not found` | Runtime not installed or not on PATH | On Windows, install `msmpisetup.exe` (runtime). On Linux, install the `-bin` package (e.g. `openmpi-bin`). |
| Linker errors referencing `MPI_Init` | SDK/headers installed but not the library, or architecture mismatch | On Windows, ensure `msmpisdk.msi` is installed. Check 32-bit vs 64-bit consistency. |
| `There are not enough slots available` | Requesting more MPI ranks than cores | Use `mpiexec --oversubscribe -n 8 ...` (OpenMPI) or reduce `-n`. |
| Crash at `MPI_Init` on WSL | MS-MPI is a Windows binary, not usable under WSL | Install OpenMPI or MPICH natively inside WSL: `sudo apt install libopenmpi-dev`. |

### CUDA

| Problem | Cause | Fix |
|---------|-------|-----|
| `Could NOT find CUDAToolkit` in CMake | CUDA not on PATH or not installed | Ensure `/usr/local/cuda/bin` is on your PATH and `nvcc --version` works. |
| `nvcc --version` works but CMake still fails | CMake too old to detect your CUDA version | Upgrade CMake to 3.21+ (required by DTWC++ anyway). |
| `nvidia-smi` shows driver but `nvcc` is missing | Only the GPU driver is installed, not the toolkit | Install the full CUDA Toolkit (the driver alone is not enough for compilation). |
| `no CUDA-capable device is detected` | No NVIDIA GPU, or driver not loaded | Check `lspci | grep -i nvidia`. Install or update the NVIDIA driver. |
| Compilation error: `unsupported gpu architecture` | GPU compute capability too old for the CUDA version | Either use an older CUDA Toolkit or set `-DCMAKE_CUDA_ARCHITECTURES=60` (or your GPU's compute capability). |
| CUDA not available on macOS | Apple does not ship NVIDIA drivers | No fix -- use Linux or Windows with an NVIDIA GPU. |
| `undefined reference to cudaXxx` | CUDA libraries not linked | Ensure `LD_LIBRARY_PATH` includes `/usr/local/cuda/lib64`. |

### General Tips

- **Check your GPU's compute capability** at [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). DTWC++ requires compute capability 6.0 or higher.
- **Driver vs Toolkit**: The NVIDIA driver and CUDA Toolkit are separate installs. You need both. `nvidia-smi` shows the driver; `nvcc --version` shows the toolkit.
- **Multiple CUDA versions**: If you have multiple CUDA versions installed, set `CMAKE_CUDA_COMPILER` explicitly:
  ```bash
  cmake -S . -B build -DDTWC_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
  ```
- **WSL2 with CUDA**: CUDA works under WSL2 with the Windows NVIDIA driver. Install only the CUDA Toolkit inside WSL (not the driver). See [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).
