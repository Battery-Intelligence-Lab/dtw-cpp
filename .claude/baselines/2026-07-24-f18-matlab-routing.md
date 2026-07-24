# F18 MATLAB estimator metric/device routing - 2026-07-24

## Scope and base

- Base: `1e4131ebc2d839b0abb1ce78681962e53f652dde`
  (`docs: record F17 resume verdict`).
- `git status --short` produced no output.
- Subject: `dtwc.DTWClustering.fit` and `fit_predict`, the MATLAB estimator
  named by PLAN F18.
- Out of scope: a new metric field on the frozen C++/MATLAB `Problem`
  contract, changes to the separate public `dtwc.compute_distance_matrix`
  signature/behavior, `predict` (still explicitly unimplemented), iterative
  clustering changes, and the functional MATLAB `dtwc.cluster` device-routing
  defect now owned by F40.

No F18-specific killed idea exists. The archive's M23 decision is binding:
accepted estimator parameters must execute literally, unsupported cross-products
must fail before compute, and a precomputed metric/backend matrix may be injected
into `Problem` rather than widening its public state.

## Confirmed inherited state

`bindings/matlab/+dtwc/DTWClustering.m` stores `Metric` in the constructor but
never reads it in `fit`. A nonempty `Device` calls `dtwc.device`, then each
restart creates a default `Problem` and calls FastPAM. The new `Problem` retains
`DistanceMatrixStrategy::Auto`; Auto resolves only to Pruned or BruteForce, and
neither `Problem::fill_distance_matrix` nor FastPAM consults `Env`.

The existing MATLAB constructor-only test therefore proves property storage and
global device validation, not estimator routing. MATLAB `Problem` has no metric
property/setter. Python's retained estimator implementation provides the narrow
design precedent: compute a matrix once when the metric is non-L1 or the
effective device is GPU, then inject it into every restart.

The adjacent functional MATLAB `dtwc.cluster` path has the same confirmed device
defect: it changes or reads global Env, creates a default `Problem`, and never
routes the selected device into distance computation. It exposes no metric
parameter. F40 owns that separate Tier-1 function; F18 must not overclaim it.

## Fresh inherited artifacts and environment

An ordinary OpenMP MEX was rebuilt from this base:

```powershell
cmake --build build/mex-verify-msvc --target dtwc_mex -j 2
```

```text
MEX_FILE path=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64 bytes=1501696 utc=2026-07-24T08:57:56.5113398Z
CPU_MEX_SHA256=4DEB4335AC50C2D757599A5C20BED463BA1A5C9B59C4AF1A4C586AB736FCDB1E
```

A new ignored build directory combines the proven MSVC/nvcc recipe with MATLAB:

```text
-- ║  Compiler:  MSVC 19.50.35723.0
-- ║  OpenMP:    ON (spec 2019)
-- ║  CUDA:      ON (13.0.48)
-- ║  Metal:     OFF
-- ║  Testing:   OFF
```

It targets only Ada `sm_89`, uses CUDA 13's required
`-allow-unsupported-compiler`, builds MATLAB R2024b with llfio/solvers/Arrow
OFF, and preserves every pre-existing build directory. The inherited binary is:

```powershell
cmd.exe /d /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake -S . -B build/mex-cuda-f18 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER=\"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe\" -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DDTWC_CUDA_ARCH_LIST=89 -DDTWC_ENABLE_CUDA=ON -DDTWC_ENABLE_METAL=OFF -DDTWC_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=\"C:\Program Files\MATLAB\R2024b\" -DDTWC_BUILD_TESTING=OFF -DBUILD_TESTING=OFF -DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_ARROW=OFF"
cmd.exe /d /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build build/mex-cuda-f18 --target dtwc_mex -j 2"
```

```text
MEX_FILE path=C:\D\git\dtw-cpp\build\mex-cuda-f18\bin\dtwc_mex.mexw64 bytes=1172480 utc=2026-07-24T09:05:17.1727329Z
CUDA_MEX_SHA256=8B9CA00807BFBCC715F998EC91FDBDD8C7496542052C57A79AF65740778C1BC6
```

Local GPU identity is:

```text
GPU 0: NVIDIA RTX 4000 Ada Generation (UUID: GPU-bf7d6c84-9fc8-80cd-0488-d6253428587e)
0, NVIDIA RTX 4000 Ada Generation, 8.9
```

Nsight Compute is locally available:

```text
NVIDIA (R) Nsight Compute Command Line Profiler
Version 2025.3.0.0 (build 36273991) (public-release)
```

`--set basic` is not usable without performance-counter permission. Exploratory
`--set none` profiling needs no such permission and still names launches. A
real CUDA-test positive control was:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\target\windows-desktop-win7-x64\ncu.exe' `
  --set none --target-processes all --launch-count 1 `
  --print-summary per-kernel `
  'build/cuda-verify/bin/test_cuda_correctness.exe'
```

It exited 0 after 7,827 assertions in 61 cases and printed:

```text
==PROF== Profiling "dtw_regtile_kernel" - 0 (1/1): 0%....50%....100% - 1 pass
void dtw_regtile_kernel<double, 4>(const T1 *, const int *, T1 *, int, int, int, bool, int, const int *) (2, 1, 1)x(256, 1, 1), Device 0, CC 8.9, Invocations 1
```

The inherited estimator profile reproduced below also exited 0 despite printing:

```text
==WARNING== No kernels were profiled.
```

The profiler output, not its exit status, is therefore the arbiter.

## Non-degenerate independent oracle

The registered fixture is four length-two scalar series with all eight values
distinct:

```text
X = [0 1;
     3 8;
     5 2;
     6 4]
k = 2
```

For two length-two series, exhaustive enumeration has three monotone DTW paths:
diagonal, horizontal-then-diagonal, and vertical-then-diagonal. Every
off-diagonal local term here is strictly positive, so the diagonal is the
unique optimum for every distinct pair. The exact independent command was:

```powershell
@'
import itertools
X = [(0, 1), (3, 8), (5, 2), (6, 4)]
paths = [
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
]
for name, local in [
    ("L1", lambda a, b: abs(a - b)),
    ("SquaredL2", lambda a, b: (a - b) ** 2),
]:
    D = [[0] * len(X) for _ in X]
    unique = True
    for i, j in itertools.combinations(range(len(X)), 2):
        path_costs = [
            sum(local(X[i][a], X[j][b]) for a, b in path)
            for path in paths
        ]
        D[i][j] = D[j][i] = min(path_costs)
        unique = unique and path_costs.count(min(path_costs)) == 1
    medoids = list(itertools.combinations(range(len(X)), 2))
    costs = [
        sum(min(D[i][m] for m in pair) for i in range(len(X)))
        for pair in medoids
    ]
    order = sorted(range(len(costs)), key=costs.__getitem__)
    print(f"{name}_MATRIX={D}")
    print(f"{name}_PATHS_UNIQUE={int(unique)}")
    print(f"{name}_MEDOID_SETS={medoids}")
    print(
        f"{name}_MEDOID_COSTS={costs} OPTIMUM={medoids[order[0]]} "
        f"NEXT={costs[order[1]]} GAP={costs[order[1]]-costs[order[0]]}"
    )
'@ | uv run --no-sync python -
```

It produced:

```text
L1_MATRIX=[[0, 10, 6, 9], [10, 0, 8, 7], [6, 8, 0, 3], [9, 7, 3, 0]]
L1_PATHS_UNIQUE=1
L1_MEDOID_SETS=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
L1_MEDOID_COSTS=[13, 11, 10, 9, 12, 13] OPTIMUM=(1, 2) NEXT=10 GAP=1
SquaredL2_MATRIX=[[0, 58, 26, 45], [58, 0, 40, 25], [26, 40, 0, 5], [45, 25, 5, 0]]
SquaredL2_PATHS_UNIQUE=1
SquaredL2_MEDOID_SETS=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SquaredL2_MEDOID_COSTS=[51, 45, 30, 31, 50, 51] OPTIMUM=(0, 3) NEXT=31 GAP=1
```

L1 costs have the input amplitude's units; SquaredL2 costs have squared
amplitude units. No cross-metric numeric comparison is used as an inequality.
Enumerating all six two-medoid sets in the order
`{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}` gives:

| Metric | Six costs | Unique optimum | Next | Gap |
|---|---|---:|---:|---:|
| L1 | `13,11,10,9,12,13` | `{1,2}` at 9 | 10 | 1 |
| SquaredL2 | `51,45,30,31,50,51` | `{0,3}` at 30 | 31 | 1 |

The medoid sets are disjoint, each is unique, and each has a positive unit gap.
Literal matrices injected into independent public `Problem` objects produced:

```text
EXPLICIT_L1 labels=[1 2 1 1 ] medoids=[3 2 ] cost=9 iterations=1 converged=1
EXPLICIT_SQUARED labels=[2 1 1 1 ] medoids=[4 1 ] cost=30 iterations=1 converged=1
```

The fresh inherited estimator is **FALSIFIED [confirmed]**:

```text
ESTIMATOR_L1 labels=[1 2 1 1 ] medoids=[3 2 ] cost=9
ESTIMATOR_SQUARED_EUCLIDEAN labels=[1 2 1 1 ] medoids=[3 2 ] cost=9
```

The squared request reproduces every L1 field instead of the independent
SquaredL2 result.

The complete R2024b rerun that printed and asserted those records was:

```powershell
$matlab = 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe'
$batch = @"
restoredefaultpath;
cd('C:/D/git/dtw-cpp');
addpath('bindings/matlab');
addpath('build/mex-verify-msvc/bin');
clear dtwc_mex;
paths=which('dtwc_mex','-all');
if iscell(paths)
  for i=1:numel(paths), fprintf('MEX_ALL_%d=%s\n',i,paths{i}); end
else
  fprintf('MEX_ALL_1=%s\n',paths);
end
X=[0 1;3 8;5 2;6 4];
D1=[0 10 6 9;10 0 8 7;6 8 0 3;9 7 3 0];
D2=[0 58 26 45;58 0 40 25;26 40 0 5;45 25 5 0];
p1=dtwc.Problem('f18_l1');
p1.set_data(X);
p1.set_distance_matrix(D1);
r1=dtwc.fast_pam(p1,2,'Seed',42);
fprintf('EXPLICIT_L1 labels=['); fprintf('%d ',r1.labels);
fprintf('] medoids=['); fprintf('%d ',r1.medoid_indices);
fprintf('] cost=%.17g iterations=%d converged=%d\n',r1.total_cost,r1.iterations,r1.converged);
p2=dtwc.Problem('f18_sq');
p2.set_data(X);
p2.set_distance_matrix(D2);
r2=dtwc.fast_pam(p2,2,'Seed',42);
fprintf('EXPLICIT_SQUARED labels=['); fprintf('%d ',r2.labels);
fprintf('] medoids=['); fprintf('%d ',r2.medoid_indices);
fprintf('] cost=%.17g iterations=%d converged=%d\n',r2.total_cost,r2.iterations,r2.converged);
e1=dtwc.DTWClustering('NClusters',2,'Metric','l1','NInit',1);
e1=e1.fit(X);
fprintf('ESTIMATOR_L1 labels=['); fprintf('%d ',e1.Labels);
fprintf('] medoids=['); fprintf('%d ',e1.MedoidIndices);
fprintf('] cost=%.17g\n',e1.TotalCost);
e2=dtwc.DTWClustering('NClusters',2,'Metric','squared_euclidean','NInit',1);
e2=e2.fit(X);
fprintf('ESTIMATOR_SQUARED_EUCLIDEAN labels=['); fprintf('%d ',e2.Labels);
fprintf('] medoids=['); fprintf('%d ',e2.MedoidIndices);
fprintf('] cost=%.17g\n',e2.TotalCost);
assert(strcmp(which('dtwc_mex'),'C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64'));
assert(isequal(r1.labels,int32([1 2 1 1])) && isequal(r1.medoid_indices,int32([3 2])) && r1.total_cost==9 && r1.iterations==1 && r1.converged);
assert(isequal(r2.labels,int32([2 1 1 1])) && isequal(r2.medoid_indices,int32([4 1])) && r2.total_cost==30 && r2.iterations==1 && r2.converged);
assert(isequal(e1.Labels,int32([1 2 1 1])) && isequal(e1.MedoidIndices,int32([3 2])) && e1.TotalCost==9);
assert(isequal(e2.Labels,int32([1 2 1 1])) && isequal(e2.MedoidIndices,int32([3 2])) && e2.TotalCost==9);
"@
& $matlab -batch $batch
```

The current five-suite inventory was rerun through that fresh OpenMP MEX on
both installed MATLAB versions. The complete wrapper below was executed with
`$release='R2024b'` and again with `$release='R2025b'`:

```powershell
$release = 'R2024b' # then R2025b
$matlab = "C:\Program Files\MATLAB\$release\bin\matlab.exe"
$batch = @"
restoredefaultpath;
cd('C:/D/git/dtw-cpp');
addpath('bindings/matlab');
addpath('build/mex-verify-msvc/bin');
clear dtwc_mex;
paths=which('dtwc_mex','-all');
fprintf('MATLAB_VERSION=$release\n');
if iscell(paths)
  for i=1:numel(paths), fprintf('MEX_ALL_%d=%s\n',i,paths{i}); end
else
  fprintf('MEX_ALL_1=%s\n',paths);
end
files={
  fullfile(pwd,'tests','matlab','test_dtwc.m'), ...
  fullfile(pwd,'tests','matlab','test_mex_input_validation.m'), ...
  fullfile(pwd,'tests','matlab','test_contract_parity.m'), ...
  fullfile(pwd,'tests','matlab','test_test_api.m'), ...
  fullfile(pwd,'tests','conformance','test_conformance.m')};
results=runtests(files);
total=numel(results);
passed=sum([results.Passed]);
failed=sum([results.Failed]);
incomplete=sum([results.Incomplete]);
inc=results([results.Incomplete]);
fprintf('MATLAB_TOTAL=%d PASSED=%d FAILED=%d INCOMPLETE=%d\n',total,passed,failed,incomplete);
for i=1:numel(inc), fprintf('INCOMPLETE_NAME=%s\n',inc(i).Name); end
omp=dtwc.test.parallelisation();
fprintf('OMP_AVAILABLE=%d OMP_MAX=%d OMP_ENGAGED=%d OMP_PASS=%d OMP_REASON=%s\n',omp.available,omp.max_threads,omp.threads_engaged,omp.pass,omp.reason);
assert(strcmp(which('dtwc_mex'),'C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64'));
assert(total==82 && passed==81 && failed==0 && incomplete==1);
assert(numel(inc)==1 && strcmp(inc.Name,'test_test_api/test_parallelisation_serial_is_honest'));
assert(omp.available && omp.pass && omp.max_threads==24 && omp.threads_engaged==24 && isempty(omp.reason));
"@
& $matlab -batch $batch
```

Both complete wrappers exited 0 and printed:

```text
MATLAB_TOTAL=82 PASSED=81 FAILED=0 INCOMPLETE=1
INCOMPLETE_NAME=test_test_api/test_parallelisation_serial_is_honest
OMP_AVAILABLE=1 OMP_MAX=24 OMP_ENGAGED=24 OMP_PASS=1 OMP_REASON=
```

The sole incomplete result was:

```text
test_test_api/test_parallelisation_serial_is_honest              X       Filtered by assumption.
```

The first R2024b wrapper used `assert(all([results.Passed]))`, rejected that
intentional opposite-flavor filter, and exited 1 despite zero failed tests. It
is not acceptance evidence; the exact-count R2024b and R2025b reruns above both
exited 0. This confirms that the inherited `61/61` working-rule total was stale,
not a missing-test regression.

Separate `which('dtwc_mex','-all')` batches produced one path each:

```text
MATLAB_VERSION=R2024b
MEX_ALL_1=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
MATLAB_VERSION=R2025b
MEX_ALL_1=C:\D\git\dtw-cpp\build\mex-verify-msvc\bin\dtwc_mex.mexw64
```

## Real inherited CUDA discriminator

The profiled process performed no explicit GPU probe or direct GPU Problem call.
Its only GPU-capable operation was
`DTWClustering(...,'Metric','squared_euclidean','Device','gpu').fit(X)` through
the fresh CUDA-enabled MEX. The exact command was:

```powershell
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\target\windows-desktop-win7-x64\ncu.exe'
$matlab = 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe'
$batch = "restoredefaultpath; cd('C:/D/git/dtw-cpp'); " +
  "addpath('bindings/matlab'); addpath('build/mex-cuda-f18/bin'); " +
  "clear dtwc_mex; fprintf('MEX=%s\n',which('dtwc_mex')); " +
  "dtwc.device('cpu'); X=[0 1;3 8;5 2;6 4]; " +
  "e=dtwc.DTWClustering('NClusters',2,'Metric','squared_euclidean','Device','gpu','NInit',1); " +
  "e=e.fit(X); " +
  "fprintf('INHERITED_F18 labels=['); fprintf('%d ',e.Labels); " +
  "fprintf('] medoids=['); fprintf('%d ',e.MedoidIndices); " +
  "fprintf('] cost=%.17g device=%s\n',e.TotalCost,dtwc.device()); " +
  "assert(strcmp(which('dtwc_mex'),'C:\D\git\dtw-cpp\build\mex-cuda-f18\bin\dtwc_mex.mexw64')); " +
  "assert(isequal(e.Labels,int32([1 2 1 1]))&&isequal(e.MedoidIndices,int32([3 2]))&&e.TotalCost==9&&strcmp(dtwc.device(),'gpu'));"
& $ncu --set none --target-processes all --launch-count 1 $matlab -batch $batch
```

It exited 0 and printed:

```text
==WARNING== No metrics to collect found in sections.
MEX=C:\D\git\dtw-cpp\build\mex-cuda-f18\bin\dtwc_mex.mexw64
INHERITED_F18 labels=[1 2 1 1 ] medoids=[3 2 ] cost=9 device=gpu
==WARNING== No kernels were profiled.
```

This is the registered defect: Env reports `gpu`, but the estimator executes
the CPU-L1 result and launches no DTWC CUDA kernel.

Separate `which('dtwc_mex','-all')` batches proved the same CUDA MEX is the sole
resolution under both installed releases:

```text
MATLAB_VERSION=R2024b
CUDA_MEX_ALL_1=C:\D\git\dtw-cpp\build\mex-cuda-f18\bin\dtwc_mex.mexw64
MATLAB_VERSION=R2025b
CUDA_MEX_ALL_1=C:\D\git\dtw-cpp\build\mex-cuda-f18\bin\dtwc_mex.mexw64
```

An HPC safety probe inspected only repository-root `.env` metadata, never its
contents:

```text
ENV_METADATA exists=1 bytes=267
```

Because `Env::select_hpc()` can proceed from that file to
`std::system("ssh ...")`, an unshimmed inherited or mutation test is prohibited.

## Corrected contract registered before implementation

1. `Metric='l1'` remains the default. The estimator also accepts
   `Metric='squared_euclidean'`, lower-case normalized like Python; unknown
   tokens raise `dtwc:invalidArgument` before data, device, or distance work.
2. Squared Euclidean is supported only with the Standard variant and
   `MissingStrategy='error'`. Incompatible requests fail loudly instead of
   precomputing a matrix that silently replaces the requested recurrence.
3. Explicit local CPU/GPU requests delegate to `dtwc.device`; an empty value
   reads the active global device. Explicit `hpc` is the safety exception: the
   estimator trims and normalizes it, then rejects it before `dtwc.device` can
   inspect credentials or invoke SSH. An already-active global `hpc` value is
   read and rejected without another transport attempt. Both paths raise
   `dtwc:deviceError` with the exact message:
   `MATLAB DTWClustering does not implement device='hpc'; no SSH/SLURM
   transport was attempted.`
4. CPU-L1 retains the inherited lazy `Problem` route. CPU-SquaredL2 and
   supported GPU requests compute exactly one matrix before the restart loop
   and inject it into every `Problem` after all semantic setters. This preserves
   one matrix production at `NInit=2` and prevents any setter from invalidating
   an injected matrix.
5. GPU estimator execution supports L1 and SquaredL2 only for Standard DTW with
   `MissingStrategy='error'`; other cross-products fail before device mutation
   or distance work. CUDA receives the active Env ordinal and normalized metric.
   Metal uses the normalized metric, accepts only ordinal zero, and otherwise
   raises `dtwc:deviceError`.
6. Optional-backend availability and producer results are fail-closed. Before
   allocation it uses checked `N*N` and pair-count arithmetic and rejects
   backend integer narrowing. Before injection it checks backend availability,
   result dimension, and `N*N` storage. For `N>1` it additionally requires the
   full unpruned pair count, zero pruned pairs, and a kernel string that is
   nonempty and not `none`; a valid singleton matrix has zero pairs and no
   kernel. Empty/zero backend results raise `dtwc:deviceError`; operational
   backend exceptions remain under F31's wider taxonomy owner and are not
   relabeled as an F18 closure.
7. No `Problem.metric` field is added. The explicit oracle route remains a
   literal metric matrix plus `Problem.set_distance_matrix`.

The routed matrix producer is an estimator-internal MEX command called by
`DTWClustering.fit`; its focused test may call that hidden command to compare
the exact matrices. F18 does not widen or reroute the separate public
`dtwc.compute_distance_matrix` wrapper.

The local environment probe is:

```text
Microsoft Windows NT 10.0.26200.0
DTWC_ENABLE_METAL:BOOL=OFF
```

Real Metal MATLAB estimator execution is therefore **[BLOCKED-ENV]** and is not
claimed by F18. F18 registers a fail-closed source/mutation gate for the guarded
Metal branch. F41, not F12, owns the future Apple+MATLAB runtime proof because
F12's retained Metal suite covers core fixed-band geometry rather than this
estimator/MEX path.

## Decisive bands

Two new cases in `test_contract_parity.m` raise the five-suite OpenMP inventory
from 82 collected / 81 passed / 0 failed / 1 expected opposite-flavor
capability skip to:

- **84 collected, 83 passed, 0 failed, 1 incomplete** on MATLAB R2024b;
- **84 collected, 83 passed, 0 failed, 1 incomplete** on MATLAB R2025b.

The sole incomplete case must be exactly
`test_parallelisation_serial_is_honest`, filtered because this is the OpenMP
MEX. The new cases must both run and emit exactly:

```text
F18_MATLAB_METRIC subject=DTWClustering.fit+fit_predict oracle=exhaustive_paths matrices=2/2 problem_routes=2/2 fit_routes=2/2 fit_predict=1/1 case_norm=1/1 distinct=3/3 ninit=2 skips=0
F18_MATLAB_VALIDATION unknown_metric=1/1 unknown_precedence=1/1 squared_variant=1/1 squared_missing=1/1 skips=0
```

The metric case requires both exact computed matrices, both literal-`Problem`
routes, both `fit` routes at `NInit=2`, and uppercase
`Metric='SQUARED_EUCLIDEAN'` through `fit_predict`. The validation case requires
`Metric='not_a_metric'`, `Device='not_a_device'`, and empty data to raise
`dtwc:invalidArgument` with exact message
`Unknown Metric 'not_a_metric'. Expected one of: l1, squared_euclidean.`;
this proves metric validation wins before input/device effects. It also requires
`dtwc:invalidArgument` for CPU SquaredL2 plus WDTW and CPU SquaredL2 plus
`zero_cost`.

### Offline HPC poison gate

One isolated R2024b process uses the ordinary MEX and a repo-local fixture under
`build/f18-hpc-poison`. Before the first MEX load, the gate points
`DTWC_REPO_ROOT` at a generated fake repository containing a valid fake `.env`
and prepends a generated `ssh.cmd` to `PATH`. `where.exe ssh` must resolve that
shim first. The shim records its exact arguments and returns success; it never
opens a socket or invokes another executable.

The process first sets global CPU and calls an estimator with explicit
`Device=' HpC '`. It must trim/case-normalize that token, receive the exact
estimator-specific error above, retain empty labels/medoids and `NaN` cost,
leave Env at CPU, and record zero fake-SSH calls. It then deliberately selects
`dtwc.device('hpc')`; the poison shim count must become exactly one and Env must
become HPC. An estimator with `Device=''` must receive the same exact error,
publish no result, and leave the count at one. The unchanged value-class result
is not evidence of when computation stopped; the guarded-source marker below
separately requires both HPC checks before matrix production/materialisation and
restart entry. Only then may the gate print:

```text
F18_MATLAB_HPC explicit_normalized=1/1 explicit_rejected=1/1 explicit_no_ssh=1/1 explicit_device_unchanged=1/1 active_selected=1/1 active_rejected=1/1 active_no_extra_ssh=1/1 no_result=2/2 skips=0
```

No ordinary-suite or mutation process may exercise an unshimmed estimator HPC
path.

### Real CUDA gate

The permanent PowerShell/Nsight gate runs the R2024b-built CUDA MEX under both
R2024b and R2025b. It runs two isolated profiles per version; it does not run
the ordinary five-suite inventory against this flavor because that suite
deliberately expects GPU unavailable.

The valid profile:

1. sets global GPU and fits L1 at `NInit=2` with `Device=''`, requiring the
   exact L1 labels/medoids/cost and active GPU;
2. resets global CPU, then fits uppercase SquaredL2 at `NInit=2` with explicit
   `Device='gpu:0'`, requiring the exact squared labels/medoids/cost and active
   GPU.

The summed profiler rows must name only production `dtw_*kernel` launches,
`Device 0`, `CC 8.9`, and exactly two invocations total: one matrix production
per fit despite two restarts. The Nsight command must be uncapped: it omits
`--launch-count` and `--launch-skip`, and the gate rejects either option before
launch so an inside-loop four-launch mutant cannot be truncated to the expected
two. It uses `--set none --print-summary per-kernel`, sums every matching DTWC
row, and requires `No kernels were profiled` absent.

The no-kernel profile begins with global GPU, then fits L1 with explicit
`Device='cpu'`; it must switch Env to CPU and reproduce the exact L1 result.
GPU-L1 plus WDTW and GPU-L1 plus `zero_cost` must then each raise
`dtwc:invalidArgument` before device mutation, leaving Env CPU. Nsight must
print `No kernels were profiled` and no DTWC kernel.

Each version may print its marker only after both child exit codes, exact
`which('dtwc_mex','-all')` path, numerical results, device transitions,
rejections, and profiler checks pass:

```text
F18_MATLAB_CUDA version=<R2024b|R2025b> subject=DTWClustering.fit profiles=2/2 mex_paths=2/2 routes=3/3 gpu_results=2/2 gpu_kernels=2/2 invocations=2/2 overrides=2/2 gpu_rejections=2/2 no_kernel=1/1 skips=0
```

After both releases:

```text
F18_MATLAB_CUDA subject=DTWClustering.fit versions=2/2 profiles=4/4 valid_invocations=4/4 no_kernel_profiles=2/2 skips=0
```

The profiled processes must not call `dtwc.test.gpu` or a direct GPU `Problem`;
the named kernels must be reachable only from the estimator under test.

### Guarded-source gate

The permanent PowerShell gate also reads only the live estimator and MEX source.
It must prove the CPU-L1 route remains lazy, CPU-SquaredL2 production precedes
the restart loop, injection follows every semantic setter, and explicit/active
HPC rejection precedes input materialisation, matrix production, and restart
entry. Both optional GPU branches must have checked allocation/index arithmetic
and availability/result fail-closed guards with no CPU fallthrough. CUDA
consumes the Env ordinal and normalized metric; Metal consumes the normalized
metric while rejecting a nonzero ordinal. A symmetric distance matrix cannot
distinguish a transposed row-major copy, so the exact
`out[i + j*N] = matrix[i*N + j]` ownership/layout expression is source-pinned.
Only then may it print:

```text
F18_MATLAB_SOURCE subject=DTWClustering+dtwc_mex metric_before_effects=1/1 hpc_before_compute=2/2 cpu_l1_lazy=1/1 cpu_squared_outside=1/1 inject_after_setters=1/1 checked_bounds=1/1 cuda_guard=1/1 cuda_metric=1/1 cuda_ordinal=1/1 cuda_result=1/1 metal_guard=1/1 metal_metric=1/1 metal_ordinal=1/1 metal_result=1/1 no_cpu_fallback=2/2 matrix_copy=1/1 skips=0
```

Both MEX builds must be rebuilt after source changes, and every MATLAB process
must start with `restoredefaultpath`, add `bindings/matlab` first, add the fresh
binary directory last, clear `dtwc_mex`, and print `which('dtwc_mex','-all')`.

The ordinary docs generation/contract checks and all three C++ suites remain:

- canonical: 120/120, zero failed, 6 capability skips;
- llfio-OFF: 120/120, zero failed, 9 capability skips;
- Arrow-ON: 122/122, zero failed, 8 capability skips.

F39's already-registered supply-chain failure is not relabeled green. The exact
expected invariant is 62 passed / 1 failed, with 39 action pins, 7 archive pins,
1 Arrow pin, 28 tracked CMake manifests, and the sole failure still
`assert 28 == 27`. Any additional failure or inventory delta falsifies F18.

## Registered mutations

Thirty-five executions in fourteen mutation classes are fixed before
implementation.
Each non-equivalent mutant must make its named subject fail before exact source
restoration:

| Class | Executions | Fixed mutations and required killed observations |
|---|---:|---|
| C01 metric consumption | 3 | ignore `Metric` in `fit` → squared fit becomes L1; force the CPU routed producer to L1 → exact squared matrix fails; omit injection only from restart 2 → its L1 cost 9 defeats squared cost 30 |
| C02 precompute placement | 2 | move CPU-Squared production inside `NInit` → source placement fails; move GPU production inside `NInit` → each valid profile reports four rather than two launches |
| C03 GPU reachability/metric | 4 | retain the inherited Env-only route → no kernels/squared L1 result; replace CUDA producer with CPU producer → source and Nsight fail; force CUDA `use_squared_l2=false` → squared cost becomes 9; send only GPU-L1 through lazy `Problem` → one rather than two launches |
| C04 effective device/overrides | 3 | treat empty `Device` as CPU → active-global GPU route launches none; ignore explicit GPU after the registered CPU reset → squared route stays CPU; ignore explicit CPU after global GPU → the no-kernel control remains GPU |
| C05 metric tokens/order | 3 | map unknown metric to L1 → validation accepts it; remove lower-case normalization → uppercase `fit_predict` fails; move metric validation below input/device handling → the empty-data/invalid-device precedence case returns the wrong diagnostic and the source field fails |
| C06 CPU cross-products | 2 | permit SquaredL2+WDTW; permit SquaredL2+`zero_cost` → corresponding ordinary validation fails |
| C07 GPU cross-products | 2 | permit GPU-L1+WDTW; permit GPU-L1+`zero_cost` → corresponding no-kernel rejection fails |
| C08 HPC boundary | 4 | delegate explicit HPC before rejection → poison count/device/message fails; reject only literal lower-case `hpc` without trim/case normalization → mixed token reaches the poison shim; allow active HPC to continue locally → active rejection fails; move active-HPC rejection after matrix/restart work → `hpc_before_compute` fails |
| C09 `fit_predict` | 1 | bypass the fitted squared route and return L1 labels → exact uppercase squared labels fail |
| C10 Metal guarded branch | 3 | replace Metal producer with CPU; force Metal squared flag false; remove Metal nonzero-ordinal rejection → the corresponding source field fails |
| C11 CUDA ordinal | 1 | replace the Env ordinal with literal zero → `cuda_ordinal` source field fails |
| C12 backend fail-closed | 4 | remove checked `N*N`/pair/narrowing arithmetic; remove CUDA availability guard; remove Metal availability guard; remove producer cardinality plus conditional `N>1` pair/pruning/nonempty-non-`none` kernel validation → corresponding source guard fails |
| C13 lazy default | 1 | eagerly precompute CPU-L1 → `cpu_l1_lazy` source field fails |
| C14 boundary ordering/layout | 2 | inject before semantic setters → source order fails and the matrix is invalidated before clustering; copy row-major producer bytes into `out[i*N+j]` instead of MATLAB column-major `out[i+j*N]` → `matrix_copy` fails |

The explicit-HPC mutant runs only inside the poison-shim process. No mutation
may contact a real SSH executable.

Implementation is capped at two attempts. No band, oracle, marker, count, or
mutation may be weakened after a run. A falsified band is recorded and F18
remains open.

## Rollback and weakest claim

Rollback is a local revert of the future F18 implementation commit; no data file
or persistent cache is changed. The claim most likely to be wrong is that the
R2024b-built CUDA MEX can execute and unload after profiled estimator kernels
under R2025b. Both releases resolve that exact MEX, but only the post-repair
profile can confirm cross-version CUDA execution. Real Metal estimator
reachability remains explicitly unconfirmed under F41.
