---
title: CUDA Setup
weight: 4
---

# CUDA Setup

DTWC++ supports an optional CUDA backend for GPU acceleration. It is **optional** -- the core library builds and runs without it. This guide covers installation and CMake configuration.

## Requirements

| Feature | Minimum Version | Notes |
|---------|----------------|-------|
| CUDA | 11.0+ | Requires NVIDIA GPU with compute capability 6.0 or newer |
| GPU | NVIDIA Kepler or newer | AMD ROCm support is planned for future releases |

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

## Building DTWC++ with CUDA

```bash
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON
cmake --build build --config Release
```

---

## Troubleshooting

### CUDA

| Problem | Cause | Fix |
|---------|-------|-----|
| `Could NOT find CUDAToolkit` in CMake | CUDA not on PATH or not installed | Ensure `/usr/local/cuda/bin` is on your PATH and `nvcc --version` works. |
| `nvcc --version` works but CMake still fails | CMake too old to detect your CUDA version | Upgrade CMake to 3.26+ (required by DTWC++). |
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
