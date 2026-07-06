# Handoff 2026-06-01 — Full-repo adversarial audit

One-line: Multi-agent adversarial audit of whole repo (security/speed/correctness/UX/dedup/portability); 60 agents, ~3M tokens, every high/critical finding independently verified against source.

## Method
- 3 background workflows: (1) 13-domain reviewers + verify, (2) 7 dedup scanners (failed: StructuredOutput), (3) recovery 16 text-output reviewers + skeptic verify.
- Workflow 1 returned cuda/metal/build-ci/scripts (39/42 confirmed). 9 domains + dedup failed schema -> re-run as text in workflow 3 (all 16 returned).
- Verify pass: 95 CONFIRMED, 11 ADJUSTED, 1 REFUTED. Zero-tolerance citation check (open cited file:line, quote code).

## Confirmed CRITICAL (verified real)
1. CUDA double-buffer wavefront drops anti-diagonal cells when max_L>2048 -> SILENT WRONG DTW. cuda_dtw.cu:280-303 + 1800-1822. MAX_SI=8 * block 256 = 2048 cap; 3-buffer path correct. Hits user's 8K-sample target.
2. Metal decode_pair FP32 sqrt -> wrong/OOB pair decode for N>~4096. metal_dtw.mm:57-68. CUDA/MPI use FP64; Metal regressed to float. OOB device read/write at N~100k.
3. Metal in-kernel num_pairs=N*(N-1)/2 int32 overflow N>=~46341. metal_dtw.mm:102,220,341,761 + KVN trunc 2004.
4. mmap_distance_matrix.hpp:120-125 integer-overflow size check -> attacker N -> OOB read. packed_size n*(n+1)/2 overflows size_t.
5. mmap_data_store.hpp:229-241 interior offsets never validated (only sentinel) -> OOB / underflow.
6. dtwc_mex.cpp:91-101 no mxIsDouble guard anywhere -> non-double input NULL-derefs (crashes MATLAB).
7. mip_Highs.cpp:198-199 status check is assert() = NDEBUG no-op -> non-optimal solve extracts garbage / empty centroids_ind -> UB. Same Gurobi catch path 120-125.

## Confirmed HIGH (themes)
- CUDA int32 index overflow result_matrix[si*N+sj] N>46341 (cu:202 etc); decode_pair row_start int32 (cu:73). Adjacent series math already uses long long — index left int.
- Build supply chain: codecov bash uploader curl<() on PR w/ token (documentation.yml:115); llfio GIT_TAG develop moving branch (Dependencies.cmake:130); quickcpplib clone HEAD --depth1 patched+executed at configure (162-176); all CPM URL tarballs no URL_HASH; llvm.sh wget|sudo (ubuntu-unit.yml:47). llfio REQUIRED violates "optional deps only" rule.
- io-security: arrow_ipc_reader / parquet_reader cast to DoubleArray with no Float64 check + no bounds on list offsets; ndim=0 div-by-zero; crc32 is integrity-only, NOT auth (treated as tamper check).
- CPU dispatch: dtw_runtime SoftDTW [[fallthrough]] -> returns Standard L1 silently (dtw.cpp:51-62); SoftDTW gamma>0 check skipped on dispatch path (NaN poison); default_data_t=float halves precision on public helpers (settings.hpp:29).
- algorithms: fast_pam inner loop O(N^2 k) — FastPAM1 trick NOT implemented despite header claim (fast_pam.cpp:187); fast_clara in-RAM assign serial (no omp) + mt19937 vs mt19937_64 between RAM/chunked -> same seed diverges; all algos `int N=static_cast<int>(size())` truncates.
- mpi-parallel: pruned_distance_matrix atomic_min on nn_dist[j] contradicts "writes only nn_dist[i]" comment (409-451); MPI duplicate N^2 buffer no MPI_IN_PLACE; static_cast<int>(N*N) overflow; contiguous-block triangular partition imbalance; scores DBI/Dunn no Nc<2 guard.
- cli-ux: --metric silently ignored on CPU path (only CUDA consumes it); std::stoi(device.substr(5)) uncaught -> terminate on bad cuda:N; --device case-sensitive rfind silent CPU fallback.
- types: is_integer/is_zero/is_one open half-lines misclassify negatives (types_util.hpp, but DEAD — no callers); TimeSeries view() drops ndim -> multivariate round-trip corruption (time_series.hpp:64).

## Redundancy / portability (user's special ask — all verified)
- decode_pair copied 3x divergent: CUDA int/if, MPI size_t/while, Metal float/sqrt. Only correct one is MPI. SSOT candidate.
- DTW band-bounds formula 3 incompatible forms (CPU half-open [lo,hi)+1; CUDA inclusive no+1; Metal (k±band)/2) + 4th copy in unit_test_adtw.cpp:355. Cross-backend divergence risk.
- Cost functors L1Dist/SquaredL2Dist/MVL1Dist/dispatch_metric duplicated dtwc::core (dtw_cost.hpp) vs dtwc::detail (warping.hpp); core::dispatch_metric ZERO call sites (dead). 
- nearest-medoid scan hand-copied 4x (fast_pam 46-79 file-local, clarans 90-101 & 176-186, fast_clara 80-94) while detail/medoid_utils.hpp helpers exist but only tests call them.
- parquet_reader.hpp vs parquet_chunk_reader.hpp: check_arrow byte-identical, find_column near-identical, offset-walk dup; reader path lacks FLOAT branch chunk has (latent crash).
- CSV setprecision(15) write loop 4x (matrix_io.hpp:41/121/139 + Problem_IO.cpp:163 reimpl of operator<<).
- bench random_series + cpu oracle byte-identical across 5 bench + 3 test files -> belong in test_util.hpp.
- fast_clara assign_all_points f64/f32 twins; Problem.cpp brute-force f32/f64 branch twins.
- PORTABILITY: CMakePresets.json:20 hardcodes C:/Program Files/LLVM/bin/clang++.exe; cmake floor mismatch 3.26 (listfiles) vs 3.21 (preset+pyproject); DTWC_ENABLE_SIMD referenced (README/benchmarks) but never option()'d + no Highway CPM dep -> dead unbuildable branch.

## Fixes APPLIED this session (verified by re-read, no build available for C++)
- .github/dependabot.yml: added pip ecosystem (was actions-only). YAML parse OK.
- scripts/slurm/env.example: fixed slurm_remote.py->.sh; labelled 5 dead vars (GATEWAY/SSH_KEY/PASSWORD/HOME_FOLDER/DATA_FOLDER) NOT IMPLEMENTED (grep-confirmed 0 refs in slurm_remote.sh). Removes false-security on SLURM_PASSWORD.

## NOT applied (need build/network — proposed as patches in report)
- All CUDA/Metal fixes (no GPU toolchain on win32 box; blind .cu/.mm edits would violate verify-before-claim).
- llfio SHA pin / codecov action pin (need network for known-good SHA).
- C++ source guards (mmap bounds, mip status, mxIsDouble, SoftDTW dispatch) — patches specified, need build to verify.

## Open questions
- OPEN: pin llfio to which SHA? Needs maintainer decision + network.
- OPEN: is default_data_t=float deliberate for Parquet f32 path or an accident? Affects fix.
- OPEN: MV "L2" = per-channel-L1-sum intentional or bug? (dtw_cost.hpp:90-97)
