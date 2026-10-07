lr-omp timing kit  (unit lr-omp, pb/lr-omp; record: .claude/baselines/2026-10-06-mac-lr-omp.md in the worktree)

ONE COMMAND         ./run.sh            the sweep (needs a QUIET machine: nothing else building or running; the report says NOT QUIET otherwise)
                    ./run.sh smoke      a check that the kit runs, N = 100, 200, one repeat (its numbers are not evidence)

WHAT IT RUNS        base b36fad43 against head, interleaved (A B, B A, ...), REPEATS = 3, 18 threads:
                    N in {100, 200, 400, 800, 1600, 3200} x regimes {noise, line} x HiGHS {OFF = the wheel's subgradient root, ON = Kelley}
                    per cell: the root (warm, and the first call's ratio as `cold x`), the LR phase (lagrangian_root_exact), Problem::cluster(),
                    and dtwc_cl's wall time; the iteration and node counts; whether every base / head pair was bit-identical; the crossover N;
                    the four registered bands B1-B4. Output: stdout, and runs/<stamp>/{report.txt,results.json,cells.jsonl}.
                    Rough cost: an hour (the line regime's serial base at N = 1600 / 3200 is minutes per run): REPEATS=2, SIZES="400 800 1600",
                    E2E=0 shorten it.

KNOBS (environment) REPEATS SIZES REGIMES FLAVORS THREADS MIN_MS E2E
                    HEAD_MIN_N     unset: the head probe uses the threshold its object was built with (the product's constant when the probes
                                   were built; the report header and every transcript's `min_n_object` say which).
                                   Set (e.g. 1 = every N forks, for the crossover sweep): the probe's LR_OMP_MIN_N knob.
                                   The head dtwc_cl (the e2e leg) has the product's compiled-in constant, so after a threshold change rebuild the trees.
                    HEAD_LOOP2     unset: both loops as the product.  0: evaluate_dual's second loop (g[j], N*k work) stays serial in the head probe.

BUILD               uv run --no-project python build_kit.py [--if-stale | base_on base_off head_on head_off]
                    run.sh calls --if-stale: the four probes are rebuilt when the probe source, either lagrangian_root.cpp, or any of the four
                    trees' libdtwc_core.a / libdtwc_cli.a changed (bin/.stamp). Four probes from `ninja -t commands dtwc_cl` of four trees
                    (paths at the top of build_kit.py):
                      base  b36fad43 export  /private/tmp/.../scratchpad/lr-omp/base-src, base-build (HiGHS ON), base-build-nohighs (OFF)
                      head  this worktree's  build/ (HiGHS ON) and build-nohighs/ (OFF)

FILES               run.sh run.py build_kit.py gen_sweep.py lrcore_probe.cpp  (data/ = generated inputs, bin/ = probes, runs/ = outputs)
