# PLAN archive — 2026-08-09 closed-finding prose

Verbatim finding bodies moved out of `PLAN.md` by the 2026-08-09 rule-12
slimming pass. The source snapshot is commit `d2e8834`; the live plan retains
status digests and authoritative baseline pointers. The blocks below are
immutable historical text and are not reflowed or corrected.

## F17 (moved verbatim)

- [ ] **F17 — CLI `--resume` reads and discards clustering state.**
      `dtwc/dtwc_cl.cpp:1398-1405` loads a binary `ClusteringResult` into the
      block-local `ckpt_result`, prints its metadata, and has no later consumer;
      the ordinary clustering path then runs and overwrites the automatic
      checkpoint. First gate: drive the real CLI from a valid binary checkpoint
      with deliberately distinguishable labels, medoids, cost, and iteration
      count; the inherited CLI must fail an assertion that resumed state affects
      the result rather than merely producing the verbose “Loaded checkpoint”
      line. Define the supported continuation semantics before repair—never
      silently relabel a read-and-discard operation as resume.
      **REPAIR RETAINED / CLOSURE FALSIFIED 2026-07-24:** `fb853eb` replays
      all five fields of a completed binary result (v1 = generic completed
      result, not iteration state), skips every clustering method, preserves
      the source binary, rejects unusable requested state, and kills all
      twelve registered mutants; closure falsified only by the frozen
      supply-chain manifest sub-band (observed 28 vs frozen 27, caused by the
      new legitimate CMake driver). Attempts exhausted — evidence-only
      checkbox; F39 uniquely owns inventory reconciliation. Registered bands
      and both attempt records:
      `.claude/baselines/2026-07-24-f17-cli-resume.md`.

## F18 (moved verbatim)

- [ ] **F18 — MATLAB estimator accepts routing options that do not reach its
      `Problem`.** `DTWClustering.Metric` is stored but never read by `fit`;
      `Device` updates global `Env`, but each repetition creates a default
      `Problem` whose `distance_strategy` remains `Auto`, and `fast_pam` does not
      consult `Env` (`bindings/matlab/+dtwc/DTWClustering.m:53-155`,
      `dtwc/Problem.cpp:778-787`). First gates: a non-degenerate fixture whose
      L1 and squared-L2 medoids/cost differ must make the estimator match the
      corresponding explicit `Problem` route, and a fresh CUDA-enabled MEX must
      prove `Device='gpu'` reaches CUDA dispatch rather than merely validating
      the global device. The current constructor-only parity test is not a gate.
      **ATTEMPTS EXHAUSTED / FALSIFIED 2026-07-24:** attempt 1 failed to
      compile; attempt 2 passed the ordinary metric/validation, guarded-source,
      and offline HPC poison markers, but the first R2024b valid CUDA profile
      crashed `0xc0000005` under `CUDAPrecision::Auto` before any kernel row
      (forced FP32/FP64 both return exact distance 10; forcing a precision
      would be rescue-tuning). Product files rolled back; red-first cases and
      the permanent runner retained in `625b5b7`. F40 owns the functional
      `dtwc.cluster` route, F41 real-Metal reachability, F42 the
      Auto-selector crash — F42 is a prerequisite for reopening F18.
      Registered fixture, oracles, and bands:
      `.claude/baselines/2026-07-24-f18-matlab-routing.md`.

## F20 (moved verbatim)

- [ ] **F20 — `Problem::set_storage_policy` is an advisory no-op for storage
      routing.** The setter only validates and stores an enum
      (`dtwc/Problem.hpp:339-345`); heap/mmap selection is owned independently
      by `DataLoader` (`dtwc/DataLoader.hpp:200-203,276-323`). This does not
      satisfy the frozen §2.1/§6.3 promise that the `Problem` setting overrides
      local series storage. First gate: derive and register which subsequent
      load/set-data operation the setter governs, then force Heap and Mmap on a
      payload above a deterministic threshold; backing mode must differ while
      series bytes and downstream distances remain identical. The inherited
      setter must fail by leaving both routes unchanged.
      **REPAIR RETAINED / CLOSURE FALSIFIED 2026-07-24:** `aa9781c` routes
      owning `Problem::set_data(Data)` (the registered governed boundary,
      shared by C++/Python/MATLAB; non-retroactive; `set_view_data` stays an
      explicit non-owning bypass) and loader construction through the shared
      series-storage router with loud unsupported-route rejection
      (Float32/llfio-OFF/CUDA/Metal mapped); `f1ef5e0` makes derived DTW
      closures move-stable. Focused llfio-ON 963/5 and llfio-OFF 606/5 pass,
      mutations 11/11 killed, full gates 122/122, 122/122, 124/124 — but the
      six-binding band is FALSIFIED at 5/6: the R2024b MATLAB llfio-ON MEX
      crashes `0xc0000005` in LLFIO's first `std::mutex` lock (VS 14.50
      constexpr-mutex vs R2024b's private MSVCP140 14.36; R2025b passes)
      before any I/O. Both attempts consumed — evidence-only checkbox; F43
      owns the toolset-runtime incompatibility, F44 the cache-path taxonomy
      escape. Registered bands and full evidence:
      `.claude/baselines/2026-07-24-f20-storage-policy.md`.

## F23 (moved verbatim)

- [x] **F23 — Python lacks the frozen binary result-checkpoint bindings.**
      CLOSED 2026-07-30: GIL-safe `save_binary_checkpoint`/
      `load_binary_checkpoint` bound over the frozen native binary-v1 codec
      with typed `dtwcpp.IOError`; exact 72-byte/10-field/3-error marker 3/3,
      parity 157/157, registered 1,046/1,048 inventories with only the F39
      red. F56 owns the disclosed non-UTF-8/surrogate path residual. Full
      registration prose archived verbatim in
      `.claude/PLAN-archive-2026-07-30-decisions.md`. Evidence:
      `.claude/baselines/2026-07-30-f23-python-binary-checkpoint.md`.
