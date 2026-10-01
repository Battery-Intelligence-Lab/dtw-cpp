// C3 probe (scratch, not part of the library): for every kernel in a cubin
// extracted from a build, its registers, static shared bytes, and the blocks per
// SM the driver grants a 256-thread block with the dynamic shared memory the fill
// launches the wavefront with at each length (wavefront_buffer_count(L) * L *
// sizeof(T)); for the other kernels, at no dynamic shared memory.
// Usage: occupancy <file.cubin> <L>...
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                                   \
  do {                                                                             \
    CUresult r_ = (x);                                                             \
    if (r_ != CUDA_SUCCESS) {                                                      \
      const char *s_ = nullptr;                                                    \
      cuGetErrorString(r_, &s_);                                                   \
      std::fprintf(stderr, "%s failed: %s\n", #x, s_ ? s_ : "?");                  \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

static size_t buffer_count(size_t L)
{
  if (L <= 512) return 5;
  if (L > 1024 && L <= 2048) return 2;
  return 3;
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: occupancy <file.cubin> <L>...\n");
    return 1;
  }
  CHECK(cuInit(0));
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK(cuCtxSetCurrent(ctx));
  int sm_shared = 0, reserved = 0;
  CHECK(cuDeviceGetAttribute(&sm_shared, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev));
  CHECK(cuDeviceGetAttribute(&reserved, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, dev));
  std::printf("device: SM shared %d B, reserved per block %d B\n", sm_shared, reserved);

  CUmodule mod;
  CHECK(cuModuleLoad(&mod, argv[1]));
  unsigned count = 0;
  CHECK(cuModuleGetFunctionCount(&count, mod));
  std::vector<CUfunction> fns(count);
  CHECK(cuModuleEnumerateFunctions(fns.data(), count, mod));
  for (CUfunction f : fns) {
    const char *name = nullptr;
    CHECK(cuFuncGetName(&name, f));
    int regs = 0, sshared = 0;
    CHECK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f));
    CHECK(cuFuncGetAttribute(&sshared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f));
    int blocks0 = 0;
    CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks0, f, 256, 0));
    std::printf("%s REG %d static_smem %d blocks/SM(256 thr, no dyn smem) %d\n", name, regs, sshared, blocks0);
    // The Shared wavefront (mode 1): blocks per SM at each length's launch.
    if (std::strstr(name, "dtw_wavefront_kernel") && std::strstr(name, "WavefrontE1E")) {
      const size_t bytes = std::strstr(name, "kernelId") ? 8 : 4;
      for (int a = 2; a < argc; ++a) {
        const size_t L = std::strtoull(argv[a], nullptr, 10);
        const size_t dyn = buffer_count(L) * L * bytes;
        int blocks = 0;
        CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, f, 256, dyn));
        std::printf("  L %zu dyn_smem %zu blocks/SM %d\n", L, dyn, blocks);
      }
    }
  }
  CHECK(cuModuleUnload(mod));
  CHECK(cuDevicePrimaryCtxRelease(dev));
  return 0;
}
