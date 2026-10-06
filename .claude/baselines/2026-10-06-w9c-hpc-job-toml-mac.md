# 2026-10-06 — W9c on macOS: `hpc` submits one `job.toml`; `gpu_device=` names the GPU

Machine as `2026-10-05-macos-design-2-0.md` (M5 Pro, Apple clang 21, libomp; no Arrow, CUDA, Gurobi). From `fed1f37d`:
16 commits `e9d54d8b`..`387c2cd8` (the transport A–E, eight review fixes, finding 3 being step 2, and steps 3–5) and
this record. `[confirmed]` = run here; `[inferred]` = not run.

## Commands

```sh
cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp && cmake --build --preset clang-macos
ctest --test-dir build -C Release -j1 --output-on-failure
uv venv $S/venv --python 3.12      # $S: this session's scratchpad/w9c, outside the repo; fresh for base and final
CMAKE_ARGS="-DOpenMP_ROOT=/opt/homebrew/opt/libomp" uv pip install --python $S/venv/bin/python ".[test,dev,io]" matplotlib
cd $S && DTWC_CL_PATH=<worktree>/build/bin/dtwc_cl $S/venv/bin/python -m pytest <worktree>/tests/python -q -p no:cacheprovider -rs --junitxml=<run>.xml
uv run --no-project python scripts/check_docs.py --cli build/bin/dtwc_cl; scripts/check_pins.py; scripts/generate_docs.py --check
```

## Python suite, reconciled by test id (junit XML) `[confirmed]`

| Run | Passed | Skipped | Failed | Ids |
|---|---|---|---|---|
| base `fed1f37d`, fresh venv | 970 | 12 | 0 | 982 |
| final `387c2cd8`, fresh venv (wheel built from the tree) | 922 | 12 | 0 | 934 |

93 ids removed, 45 added, no outcome changed (982 − 93 + 45 = 934); the 12 skips are base's. Removed (each with its cover
in the deleting commit): test_hpc TestBuildCommand 10, TestDTWClusteringHpcDispatch 30, TestFindDtwcBinary 3,
TestSlurmRunner 11, TestSlurmLastMile 21, TestLocalRoundTrip 4, TestPackagedWrapper 2; test_api 7; test_variant_domains 5.
Added: TestJobToml 18, TestSlurmLastMile 22, TestParseLabelsCSV 1, TestLocalRoundTrip 2, TestPackagedWrapper 2. All 53
bash-driven cases ran, and the 5 that run dtwc_cl. Per step: A 248/1, B 894/12, C 911/12, D 914/12, final 922/12.

## ctest, docs gates, lint `[confirmed]`

`100% tests passed out of 95`, `test_cuda_correctness` skipped (MAY_SKIP), 41.68 s, at `861e18eb`; no C++ changed since
(`git diff --stat fed1f37d..HEAD -- dtwc CMakeLists.txt cmake tests/unit tests/integration benchmarks bindings python/src`
is empty). `check_docs.py`: `DOCS flags checked=398 pages=59 live=60 VERDICT=PASS`; `check_pins.py`: `PINS cmake=18
actions=37 failures=0`; `generate_docs.py --check`: current. ruff 0.16.10 per touched file, base vs now: no file gains a
finding. No shellcheck here; `/bin/bash -n` (3.2) passes every script. Lines, `git diff --shortstat fed1f37d..387c2cd8`:
32 files, +1181 / −2630; `_hpc.py` 623 → 344, `cluster_generic.slurm` 137 → 33, `slurm_remote.sh` 780 → 653, four smoke
jobs (618) → `smoke.slurm` (103), `test_hpc.py` 1492 → 1243 (advisor: −900/+80; the kept injection tests and the GPU,
smoke, build and review cases hold the difference).

## The new tests bite (mutation of the venv's copy, then restored) `[confirmed]`

Each fails its tests: the job script without `cd "$DTWC_JOB"`; with `|| true` after dtwc_cl; the writer not escaping
`"`; download-cluster in `results/<id>`; the bash table reader dropping the constraint (6); the Python reader without
the floor (5); a GPU build without its request; the smoke GPU mode without its request; `cluster()` without `apply()`
(2) or the k ≤ N check (1); no name check; no `src/logs` mkdir; the labels mapped by name without n; no CR strip; an
empty GPU type read as `*` (2); no `--partition`; an ignored `SLURM_GPU_GRES`.

## One-off proofs `[confirmed]`

smoke.slurm on the upload layout (synthetic UCR-format Coffee 28×286, Beef 30×470): cpu 0, checkpoint 0 (resumed),
gpu 1 (Metal refuses fp64), parquet 1 (no Arrow here), bogus 1. build-arc.sh with stub nvidia-smi (8.0), cmake, nproc:
`DTWC_NATIVE_CPU` unset/ON adds CUDA and CPU `native`, OFF only CUDA, after `-DDTWC_ARCH_LEVEL=v3`. The missing-build
hint under `/bin/bash` 3.2 names `build htc-gpu --gpu-device a6000`. dtwc_cl's `--print-config` writes back Python's
line unchanged for strings with `"`, `\`, tab, U+0001, U+007F, `é`, `δ`, `#`, `,`, `[x]` and floats 0.17, 1e-05, 1e+16,
2.5e-310, inf, nan (`"123"`, `"true"`, `123456789.0`, `-0.0`: the same value, another form); `delimiter = "\t"` is one
tab; an unknown key: `INI was not able to parse newer-key`, exit 110, no output directory.

## git grep: the deleted names (tree minus `.claude/` and CHANGELOG.md) `[confirmed]`

0 hits: build_dtwc_command, find_dtwc_binary, _validate_remote_configuration, _normalize_choice, _variant_validation,
normalize_variant_parameters, _validate_submission_envelope, _validate_restart_schedule, _normalize_cli_int,
_METHOD_ALIASES, _SAFE_JOB_NAME, _SAFE_REMOTE_PATH, _HPC_KEYS, _hpc_remote_device, submit_cluster, submit-cluster,
cmd_submit_cluster, is_finite_number, submit-cpu, submit-gpu, submit-checkpoint, submit-parquet,
{cpu,gpu,checkpoint,parquet}_test.slurm, "unsupported by the remote CPU CLI". `SLURM_GPU_GRES` has 3, all its refusal
(wrapper, test case, contract §6.2).

## gpu_device → sbatch request (`python/dtwcpp/_slurm/gpu_devices.txt`, read by Python and the wrapper)

| gpu_device | cc | Request | Source |
|---|---|---|---|
| (none) | ≥ 8.0 | `--gres=gpu:1 --constraint=gpu_cc:8.0\|gpu_cc:8.6\|gpu_cc:8.9\|gpu_cc:9.0` | the rows below |
| a100 | 8.0 | `--gres=gpu:a100:1` | job-scheduling L239–245 (types at L245); systems L199 |
| a6000 | 8.6 | `--gres=gpu:1 --constraint=gpu_cc:8.6` | job-scheduling L247, L253, L268; systems L211 |
| l40s | 8.9 | `--gres=gpu:1 --constraint=gpu_cc:8.9` | same; systems L223 |
| h100 | 9.0 | `--gres=gpu:1 --constraint=gpu_cc:9.0` | same; systems L213, L219, L221 |
| p100, v100, rtx8000, rtx | 6.0–7.5 | refused, DeviceError | systems L197–209; types L245 |

Sources, `arc-user-guide.readthedocs.io/en/latest/_sources/`: `job-scheduling.rst.txt` (sha256 `e5bb868b…`, 280 lines;
L233 `--gres=gpu:1`, L239 `--gres=gpu:v100:1`, L245 "Available devices are P100, V100, RTX (Titan RTX), RTX8000, and
A100", L247–257 the constraint forms, L264–268 gpu_gen/gpu_sku values naming no A6000, L40S or H100) and
`arc-systems.rst.txt` (sha256 `cf369d04…`, 258 lines; L195–223 GPUs and compute capabilities, L106–137 partitions:
`short` holds the GPU nodes, `interactive` is htc-g[048-049] with V100s; L155–182 CPUs: A100 nodes Cascade Lake and
Rome, H100 nodes Ice Lake and Sapphire Rapids). `[inferred]`: that ARC tags those nodes `gpu_cc:8.6/8.9/9.0` exactly.

## The ARC leg (Volkan, from the checkout with `.env`; nothing ran against ARC)

```sh
bash scripts/slurm/slurm_remote.sh test
bash scripts/slurm/slurm_remote.sh ssh "sinfo -M htc -N -h -o '%N %G %f' | sort -u -k2"  # gpu_cc:8.6/8.9/9.0, gres a100?
bash scripts/slurm/slurm_remote.sh upload                   # creates src/logs too
bash scripts/slurm/slurm_remote.sh build htc-cpu
bash scripts/slurm/slurm_remote.sh build htc-gpu
bash scripts/slurm/slurm_remote.sh build htc-gpu --gpu-device a100   # log: "native CUDA architecture, portable CPU", build-a100
bash scripts/slurm/slurm_remote.sh status                   # until the builds leave the queue
DTWC_REPO_ROOT=$PWD python -c "
import numpy as np, dtwcpp
x = [np.random.default_rng(i).standard_normal(64) * 0.1 + (0.0 if i < 10 else 9.0) for i in range(20)]
for d, g in (('hpc', None), ('hpc:gpu', None), ('hpc:gpu', 'a100')):
    print(d, g, dtwcpp.cluster(x, k=2, device=d, method='pam', **({'gpu_device': g} if g else {})).labels)"
bash scripts/slurm/slurm_remote.sh ssh "tail -n 20 logs/cluster_<id>.out"   # node (A100: htc-g015..019), job.toml
mkdir -p results/hpc/w9c_unknown && printf 'input = "input.tsv"\nn-clusters = 2\nnewer-key = 1\n' > results/hpc/w9c_unknown/job.toml
printf '0\t1\n1\t0\n' > results/hpc/w9c_unknown/input.tsv && bash scripts/slurm/slurm_remote.sh submit-job results/hpc/w9c_unknown
bash scripts/slurm/slurm_remote.sh ssh "cat logs/cluster_<id>.err"          # INI was not able to parse newer-key
bash scripts/slurm/slurm_remote.sh submit-smoke gpu
```

Expect three 10/10 splits (A100 on htc-g015..019), the unknown key refused, the smoke job on a GPU of cc ≥ 8.0.

## Deviations from the brief

1. The GPU table is a data file beside the wrapper, read by `_hpc.gpu_request` and by bash (its `build --gpu-device`,
   `submit-benchmark-gpu`, `submit-smoke gpu` need it without Python); A6000/L40S/H100 asked for by `gpu_cc:`.
2. `submit-job <rundir> [--gpu | --gpu-device <type>]`, `download-cluster <job-id>`, `results/cluster_<id>`: the name
   never crosses the shell, so any file name works (empty, `.`, `..`, `/`, `\` refused). A cluster path is absolute.
3. C++ checks values (`apply()` on a scratch Problem); what only the cluster's GPU judges fails the job. No `.env`,
   bash or wrapper: DeviceError (subclass of the old RuntimeError). Tests take DTWC_CL_PATH only (unset: CLI cases skip).
4. A `--gpu-device` build keeps the CPU portable (DTWC_NATIVE_CPU=OFF), narrowing DECISIONS 09-30's "native CPU flags"
   to hand builds; submit-job and GPU builds use the `.env` partition; smoke jobs fail on a failed run.

## Merged on the main tree (orchestrator, `834b7904` = `7ce98523` (M1, its review, W9e merged) + `pb/W9c`), all `[confirmed]`

Conflicts resolved by hand: CHANGELOG (both entries kept), the contract's `skip_rows` row (W9c's Python cell, W9e's
MATLAB cell), `tier-1.md` regenerated. `build/`: zero warnings; `ctest -j1` 100 % of 95, the CUDA skip, 34.91 s;
conformance the one silhouette ulp (D-19); `check_docs` PASS, `check_pins` 0 failures, `generate_docs --check`
current. Fresh venv (`.[test,dev,io,mip]`, matplotlib, pandas) with `DTWC_REQUIRE_HIGHSPY=1`: first **2 failed,
924 passed, 11 skipped**: M1's `test_mip.py` (landed after W9c's base) called `_hpc.find_dtwc_binary`, which W9c
deleted; it now takes the conftest `dtwc_cl` fixture (DTWC_CL_PATH). After that one edit: **926 passed, 11 skipped,
0 failed, 81.41 s** (the targeted `test_mip` + `test_wheel_smoke` + `test_index_types` run: 34 passed).
`python-wheels.yml`'s smoke now also asserts `gpu_devices.txt` ships in the wheel (not run here).
