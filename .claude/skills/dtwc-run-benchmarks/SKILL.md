---
name: dtwc-run-benchmarks
description: Run DTWC++ benchmarks on this machine (or prepare an HPC job) and record the result with the hardware it ran on — CPU, GPU, memory, compiler, build flags. Use when asked to benchmark, measure performance, check a speed-up, compare devices, or reproduce a number. Also use before claiming any performance change.
---

# Run the benchmarks

A benchmark number without its machine is not evidence. Every record this skill produces names the
CPU, the GPU, the memory, the compiler and the resolved `DTWC_*` options, because the same code is
routinely 10× apart on the M2 Max and on an ARC node, and nobody can tell afterwards which one a
bare number came from.

Records are **per machine**. Never put two machines in one table and call the ratio a speed-up.

## Steps

1. **Capture the machine first**, not after — a crashed run still deserves its context:

   ```sh
   uv run --no-project python scripts/machine_facts.py --build-dir build --with-dtwcpp
   ```

   `--with-dtwcpp` adds the two engagement probes. Read them, do not just paste them.

2. **Prove the thing you are measuring is switched on.** `dtwc::test::parallelisation()` reports
   `threads_engaged` — distinct threads that actually ran, not a flag — and `dtwc::test::gpu()`
   reports `validated`, which checks a real DTW matrix against the CPU oracle. A benchmark of a
   build that silently fell back to one thread is worse than no benchmark, because it will be
   believed. If `threads_engaged` is 1 on an OpenMP build, stop and fix that first.

3. **Build Release with the benchmarks on** (they are `OFF` by default):

   ```sh
   cmake --preset clang-macos -DDTWC_BUILD_BENCHMARK=ON -DOpenMP_ROOT=/opt/homebrew/opt/libomp
   cmake --build --preset clang-macos
   ```

   Targets: `bench_dtw_baseline` (the DTW kernels and matrix fill), `bench_mmap_access` (dense vs
   mmap, and vector-of-vector vs flat), `bench_f32_vs_f64`, `bench_openmp_schedule`,
   `bench_cuda_dtw`, `bench_metal_dtw` (both have custom mains that inject GPU device info),
   `bench_mpi_dtw`, `benchmark_UCR` (the long sweep; needs the UCR archive under
   `settings::paths::data`, which nothing downloads for you).

4. **Run through the existing driver** — it already writes JSON, timestamps the filename under
   `benchmarks/results/_autorun/` and blanks the hostname:

   ```sh
   scripts/run_bench.sh build/benchmarks/bench_dtw_baseline
   ```

   Pass `--benchmark_filter=<regex>` to narrow. Do not hand-roll a `--benchmark_out` invocation.

5. **Make the number mean something.** Check `cpu_scaling_enabled` and `load_avg` in the JSON
   `context` block; on a laptop, run on mains power. Repeat at least three times and report the
   median with the spread. One run is an anecdote.

   On **bare-metal Linux** prefer counters over wall-clock (X-24):

   ```sh
   cmake -S . -B build -DDTWC_BUILD_BENCHMARK=ON -DDTWC_BENCHMARK_PMU=ON
   scripts/run_bench.sh build/bin/bench_dtw_baseline \
       --benchmark_perf_counters=CYCLES,INSTRUCTIONS
   ```

   Anywhere else the configure stops with a message saying why — macOS has no `perf_event_open`
   (that is `xctrace`'s job), and GitHub runners and usually WSL2 do not expose the PMU. Do not work
   around it by asking a non-PMU build for counters: Google Benchmark accepts the flag, prints one
   line to stderr and writes a normal JSON with no counter fields, exit 0. `run_bench.sh` catches
   that, renames the file `*.no-counters.json` and exits 65 — if you see that, the record is
   wall-clock and must be reported as wall-clock.

6. **Write the record** to `.claude/baselines/YYYY-MM-DD-<machine>-<subject>.md`:
   - the machine table from step 1, first, verbatim;
   - the exact commands, copy-pasteable;
   - the numbers verbatim, each tagged `[confirmed]` (you ran it) or `[inferred]`;
   - the registered band if one was registered before the run, and whether it held. **FALSIFIED is
     a result** — record it with the same care as a win.

7. **HPC.** `scripts/slurm/slurm_remote.sh submit-benchmark-cpu` and `submit-benchmark-gpu
   [a100|l40s|h100]` exist, and `python/dtwcpp/_hpc.py` can generate the command line. **Generate
   and hand over — never submit, ssh or poll.** That is Volkan's action. The SLURM jobs write their
   own `benchmark_meta.json`; `benchmarks/aggregate_results.py` merges it with the timings.

## Rules

- `docs/content/benchmarks/ucr.md` is generated and carries **release evidence only**. A local or
  advisory measurement never goes there, and `scripts/generate_docs.py` owns the file.
- Wall-clock is advisory; counters decide. Say which one you have.
- If you cannot capture the machine, say so in the record rather than leaving the table out.
- Comparing against dtaidistance / tslearn / aeon goes through `benchmarks/bench_cross_library.py`
  so the competitors' conventions (L2 local cost, sqrt, window fraction) are converted, not assumed.
