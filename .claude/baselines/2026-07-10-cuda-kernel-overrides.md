# CUDA kernel override dispatch (Task 8.1-M50)

Date: 2026-07-10

Platform: Windows 10, NVIDIA RTX 4000 Ada Generation (compute capability 8.9,
20,475 MiB), driver 596.72, CUDA 13.0.48, MSVC 19.50.35723.0, Ninja,
Release, `CMAKE_CUDA_ARCHITECTURES=89`

Scope: strict CUDA `KernelOverride` selection, pairwise/one-vs-N/K-vs-N
dispatch, truthful actual-kernel/fallback reporting, and forced-kernel memory
safety

## Registration and red evidence

The defect was registered in PLAN before production work (`63c9b76`). The host
selector/discriminator contract was preregistered in `10bae86`, normalized in
`04dfa84`, and extended to both one-vs-N overloads plus K-vs-N in `c5a19d6`.

In a clean non-CUDA Release checkout, the preregistered contract produced two
cases: the host case failed because `cuda/kernel_selection.hpp` did not exist,
and the real-device case was the expected CUDA-capability skip. A disposable
host-only prototype then passed all 42 selector assertions before production
code was changed; no device result was inferred from that prototype.

## Dispatch decision

The host-only seam returns one exact launch path plus an explicit fallback bit:

| Request | Supported CUDA range | Actual path |
|---|---:|---|
| `Auto` | all | Warp through 32, RegTileW4 through 128, RegTileW8 through 256, then Wavefront |
| `Wavefront` | all | Wavefront |
| `RegTile` | through 256 | RegTileW4/RegTileW8; above 256, truthful Auto fallback |
| `WavefrontGlobal` | none | truthful Auto fallback |
| `BandedRow` | none | truthful Auto fallback |

CUDA has no distinct global-wavefront or row-major banded implementation, so
neither unsupported selector is ever reported as forced. Pairwise, one-vs-N by
index, one-vs-N by external query, and K-vs-N all consume the same selection.
Each result reports the kernel that actually launched and whether a supported
public request fell back. Early returns and a fully LB-pruned matrix report
`kernel_used="none"` and `kernel_override_fell_back=false`.

Invalid `CUDAPrecision` and `KernelOverride` values are validated as the first
two executable statements at every option-bearing CUDA root. Precision
selection uses an exhaustive switch with no permissive default.

## Adversarial source audit before device execution

The forced paths extend kernels below their old heuristic range, so both were
checked structurally before running them:

- RegTile at lengths at or below 32 has at least one valid column lane. All 32
  lanes still execute every full-mask shuffle; inactive lanes retain initialized
  infinity state, the result lane is within the template's 128/256-column cap,
  and per-warp shared staging is allocated from the real `max_L`.
- Wavefront at short lengths uses the existing preload/three-buffer layout.
  Dynamic shared memory is `5 * max_L * sizeof(T)` plus the optional band arrays,
  cooperative loads are bounded by the true row/column lengths, and all block
  barriers are reached uniformly. The final one-cell anti-diagonal is written
  before result extraction.

The largest registered device case has length 300 and requests about 12 KiB of
dynamic shared memory per wavefront block. Approximately 17 GiB of device
memory was free before execution. These checks made the subsequent runs tests
of concrete invariants rather than timing-based dispatch guesses.

## Green validation

The production implementation is `54615fb`; `dea098c` independently pins all
four no-launch roots and the fully-pruned reset.

Host-only gate:

```powershell
cmake --build build --target test_cuda_kernel_override -j 4
build/bin/test_cuda_kernel_override.exe "[host]" --reporter compact
```

```text
All tests passed (42 assertions in 1 test case)
```

The CUDA build was a detached clean checkout at `dea098c`, not the shared
working tree. It was configured with CUDA 13.0, `sm_89`, optional solvers and
bindings disabled, and `-allow-unsupported-compiler` for the installed MSVC
host toolchain. The CUDA translation unit and focused executable compiled
successfully.

```powershell
build/m51-red-clean/out-cuda-m50/bin/test_cuda_kernel_override.exe "[no_launch][m50]" --reporter compact
build/m51-red-clean/out-cuda-m50/bin/test_cuda_kernel_override.exe "[m50]" --reporter compact
```

```text
no-launch/final-prune gate: 13/13 assertions in 2 test cases
complete M50 gate:         532/532 assertions in 5 test cases
```

The complete gate forces Wavefront and RegTile where Auto would choose another
family, checks supported and unsupported boundaries, exercises both row APIs
and K-vs-N, and compares every launched FP64 result with an independent CPU DTW
oracle without using elapsed time as evidence.

CUDA address and shared-memory hazard gates over every M50 device case:

```powershell
compute-sanitizer --tool memcheck --error-exitcode 99 build/m51-red-clean/out-cuda-m50/bin/test_cuda_kernel_override.exe "[device][m50]" --reporter compact
compute-sanitizer --tool racecheck --error-exitcode 99 build/m51-red-clean/out-cuda-m50/bin/test_cuda_kernel_override.exe "[device][m50]" --reporter compact
```

```text
device tests under memcheck:  482/482 assertions in 3 test cases
ERROR SUMMARY: 0 errors

device tests under racecheck: 482/482 assertions in 3 test cases
RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

The integrated M47 first-statement selector boundary was also rebuilt in the
same CUDA configuration:

```text
unit_test_invalid_distance_enums "[m47][gpu]": 16/16 assertions in 1 test case
```

## Commits

- `63c9b76` -- register M50 in PLAN
- `10bae86`, `04dfa84`, `c5a19d6` -- preregister and extend the contract
- `54615fb` -- implement strict, shared CUDA override dispatch and reporting
- `dea098c` -- pin no-launch and fully-pruned reporting
