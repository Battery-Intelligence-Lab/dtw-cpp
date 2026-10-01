// C4 probe (scratch tool, not part of the library): does select_kernel's route
// above L = 2048 agree with the driver's occupancy API? For both precisions and
// every L in [2049, 10000] it asks the driver how many 256-thread blocks of the
// Shared wavefront kernel (extracted from a cubin of cuda_dtw.cu, opted in to the
// device's maximum dynamic shared memory as the fill does) fit an SM with
// wavefront_buffer_count(L) * L * sizeof(T) dynamic bytes, and asks select_kernel
// (the header of the tree this probe is compiled against) with the device's SM
// shared memory and the kernel's static bytes plus the reserved bytes. The rule
// is right where it returns Wavefront exactly when the driver grants >= 3 blocks.
// Usage: boundary <file.cubin>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kernel_selection.hpp"

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

int main(int argc, char **argv)
{
  if (argc != 2) {
    std::fprintf(stderr, "usage: boundary <file.cubin>\n");
    return 1;
  }
  CHECK(cuInit(0));
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK(cuCtxSetCurrent(ctx));
  int sm_shared = 0, reserved = 0, optin = 0;
  CHECK(cuDeviceGetAttribute(&sm_shared, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev));
  CHECK(cuDeviceGetAttribute(&reserved, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, dev));
  CHECK(cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
  std::printf("device: SM shared %d B, reserved per block %d B, opt-in per block %d B\n", sm_shared, reserved, optin);

  CUmodule mod;
  CHECK(cuModuleLoad(&mod, argv[1]));
  unsigned count = 0;
  CHECK(cuModuleGetFunctionCount(&count, mod));
  std::vector<CUfunction> fns(count);
  CHECK(cuModuleEnumerateFunctions(fns.data(), count, mod));
  int rc = 0;
  for (const bool fp64 : { false, true }) {
    CUfunction shared = nullptr;
    for (CUfunction f : fns) {
      const char *name = nullptr;
      CHECK(cuFuncGetName(&name, f));
      if (std::strstr(name, "dtw_wavefront_kernel") && std::strstr(name, "WavefrontE1E") &&
          (std::strstr(name, "kernelId") != nullptr) == fp64)
        shared = f;
    }
    if (!shared) { std::fprintf(stderr, "no Shared wavefront kernel for FP%d\n", fp64 ? 64 : 32); return 1; }
    int static_bytes = 0;
    CHECK(cuFuncGetAttribute(&static_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, shared));
    CHECK(cuFuncSetAttribute(shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin - static_bytes));
    const std::size_t sample = fp64 ? 8 : 4;
    std::size_t last = 0;
    int mismatches = 0, first_global = 0, last_shared = 0;
    for (std::size_t L = 2049; L <= 10000; ++L) {
      const std::size_t dyn = dtwc::cuda::detail::wavefront_buffer_count(L) * L * sample;
      int blocks = 0;
      CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, shared, 256, dyn));
      const auto path = dtwc::cuda::detail::select_kernel(L, sample, sm_shared, std::size_t(static_bytes) + reserved);
      const bool rule_shared = path == dtwc::cuda::detail::KernelPath::Wavefront;
      if (rule_shared) last_shared = int(L);
      else if (!first_global) first_global = int(L);
      if (rule_shared != (blocks >= 3)) {
        ++mismatches;
        if (mismatches <= 12 || blocks != int(last))
          std::printf("  FP%d L %zu: rule %s, driver blocks/SM %d\n", fp64 ? 64 : 32, L,
                      rule_shared ? "Wavefront" : "WavefrontGlobal", blocks);
      }
      last = std::size_t(blocks);
    }
    std::printf("FP%d: static %d B; rule's last shared length %d, first global %d; L 2049..10000 mismatches with driver >= 3 blocks: %d\n",
                fp64 ? 64 : 32, static_bytes, last_shared, first_global, mismatches);
    // blocks/SM around the boundary, for the record
    for (std::size_t L : { std::size_t(2749), std::size_t(2750), std::size_t(2751), std::size_t(2754), std::size_t(2757), std::size_t(2758) }) {
      int blocks = 0;
      CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, shared, 256, dtwc::cuda::detail::wavefront_buffer_count(L) * L * sample));
      if (!fp64) std::printf("  FP32 L %zu: driver blocks/SM %d\n", L, blocks);
    }
    if (mismatches) rc = 2;
  }
  CHECK(cuModuleUnload(mod));
  CHECK(cuDevicePrimaryCtxRelease(dev));
  return rc;
}
