---
description: "Adversarial read-only code-quality audit of the DTWC++ repository. Runs 13 concrete checks plus a hardening appendix, handles multi-config builds, and catches docs/code drift and misleading test coverage. Never edits tracked source files; diagnosis only."
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# Check Code Quality

Independent senior-engineer audit of DTWC++. Opinionated, reproducible, read-only. Every finding cites the exact grep/bash command that produced it so the user can re-run.

This skill does **not** rubber-stamp. If a check is ambiguous, mark it WARN with remediation, not PASS.

## Input

`$ARGUMENTS` = optional scope. Default: whole repo. Accepts a subdirectory (e.g. `dtwc/core`) or comma-separated list.

Inside the skill, set:
```bash
REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
SCOPE="${ARGUMENTS:-$REPO}"
SCOPE_GREP="${SCOPE//,/ }"
# Canonical source directories (repo has NO dtwc/src/ subtree — code lives directly under dtwc/)
SRC_DIRS="dtwc dtwc/core dtwc/algorithms dtwc/io dtwc/mip dtwc/cuda dtwc/metal dtwc/mpi dtwc/enums dtwc/types"
HOT_DIRS="dtwc dtwc/core dtwc/algorithms dtwc/cuda dtwc/metal"
```

Every `rg` below must be run with `cd "$REPO"` so relative paths resolve. Global checks (licenses, CITATIONS.md) always run repo-wide.

## Tool availability check (run first, WARN on missing)

```bash
for tool in git rg cmake ctest awk uv; do
  command -v "$tool" >/dev/null 2>&1 || echo "MISSING_TOOL: $tool"
done
NM_TOOL=""
for tool in nm llvm-nm; do
  command -v "$tool" >/dev/null 2>&1 && NM_TOOL="$tool" && break
done
[ -n "$NM_TOOL" ] || echo "MISSING_TOOL: nm_like"
# `fd` is optional — all fd lines below have a `find` fallback.
```

If `git` or `rg` is missing, abort with `FAIL`. If `cmake`/`ctest`/`nm_like` missing, downgrade the checks that need them to WARN with reason.

## Philosophical baseline (apply when judging every sub-check)

**1. Jason Turner (C++ Best Practices)**
- naked `new` / `delete` in core
- raw owning pointers instead of `std::unique_ptr` / `std::span`
- non-`const` member functions that obviously don't mutate
- missing `noexcept` on move ctors / swap / trivial accessors
- missing `[[nodiscard]]` on pure query functions
- `std::endl` where `'\n'` would do (forced flush in hot path)
- `using namespace` in a header
- plain `enum` instead of `enum class`
- hand-rolled rule-of-five when rule-of-zero suffices

**2. Klaus Iglberger (C++ Software Design)**
- `virtual` in hot inner-loop types (prefer CRTP / policy templates)
- inheritance used purely for code reuse (should be composition)
- god class: >500 LOC / >15 public methods
- function doing >1 level of abstraction
- Strategy done with a long `if … else if` chain instead of dispatch
- orthogonal concerns tangled (I/O inside a numeric kernel)

**3. FFmpeg-style performance**
- heap allocation inside the innermost DTW / warping loop
- `std::string` / `std::stringstream` built per-iteration
- gather/scatter access patterns where stride-1 would work
- missing `__restrict__` / `[[gnu::hot]]` on inner kernels
- missing SIMD where it is obvious (column sweep, min3)
- un-benchmarked claims of "fast"

**4. Rust-zealot safety**
- unchecked `reinterpret_cast`, C-style casts in new code
- signed-integer overflow relied upon, `int` against `size()`
- uninitialized POD read
- lambda captures by reference that outlive the frame
- file-exists-then-open (TOCTOU)
- missing bounds check at public API (only `assert`, off in Release)

**5. Defense-industry / MISRA / JSF**
- hostnames, usernames, absolute home paths embedded in JSON / CSV / benchmarks
- `fmt::format` / `printf` with user-supplied format string
- `std::system` with any concatenated string
- silent catch-all: `catch (...) { /* nothing */ }`
- non-deterministic ordering in safety-relevant output (unseeded RNG, unordered_map iteration)
- PII leakage into telemetry / benchmarks / CSVs

## Execution contract

1. Run checks A–M in order. Section N is informational only.
2. Each verdict-bearing section emits `Status: PASS | WARN | FAIL`, a Findings block, and Remediation.
3. After check M, emit aggregate `## Verdict` using the worst status seen. Emit a machine-parseable JSON trailer so downstream tools can pipe it.
4. Do NOT edit tracked source files. Builds/tests may write inside the existing build directory or system temp, but never intentionally write into the tracked source tree.

---

## A. Test quality (tautologies, skipped, always-pass)

```bash
# Build/test — do NOT silence errors.
cd "$REPO" || exit 1
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "WARN: $BUILD_DIR not configured"
else
  BUILD_ARGS=()
  CTEST_ARGS=()
  if rg -n '^CMAKE_CONFIGURATION_TYPES:STRING=' "$BUILD_DIR/CMakeCache.txt" >/dev/null 2>&1; then
    BUILD_ARGS+=(--config Release)
    CTEST_ARGS+=(-C Release)
  fi

  cmake --build "$BUILD_DIR" "${BUILD_ARGS[@]}" -j 8 2>&1 | tail -40

  CTEST_OUTPUT="$(ctest --test-dir "$BUILD_DIR" "${CTEST_ARGS[@]}" -j 4 --output-on-failure 2>&1)"
  printf '%s\n' "$CTEST_OUTPUT" | tail -80

  FAILED_TESTS="$(printf '%s\n' "$CTEST_OUTPUT" \
    | awk '/The following tests FAILED:/{flag=1; next} /^Errors while running CTest/{flag=0} flag && $2=="-" {print $3}')"

  for t in $FAILED_TESTS; do
    echo "=== SERIAL_RERUN: $t ==="
    ctest --test-dir "$BUILD_DIR" "${CTEST_ARGS[@]}" --output-on-failure -R "^${t}$" 2>&1 | tail -60
  done
fi
```

Tautology / skip scans (ripgrep's default regex engine does NOT support backreferences — use `--pcre2`):

```bash
cd "$REPO"
# Catch2 / GoogleTest tautologies — PCRE2 needed for backrefs
rg -n --pcre2 --type cpp 'REQUIRE\(\s*(\w+)\s*==\s*\1\s*\)' tests/ bindings/ python/
rg -n --pcre2 --type cpp 'ASSERT_(EQ|TRUE)\(\s*(\w+)\s*,\s*\2\s*\)' tests/ bindings/
rg -n --type cpp '\bREQUIRE\(\s*true\s*\)' tests/
rg -n --type cpp '\bREQUIRE\(\s*1\s*==\s*1\s*\)' tests/
# Commented-out assertions
rg -n --type cpp '^\s*//\s*(REQUIRE|ASSERT|EXPECT)' tests/
# Skipped / disabled (Catch2 tag `[.]` hides tests from default run)
rg -n --type cpp 'GTEST_SKIP|SKIP\(|TEST_CASE\([^,]+,\s*"\[\.' tests/
# Python: trivial asserts (repo puts Python tests in tests/python/ and python/)
rg -n --pcre2 --type py 'assert\s+(\w+)\s*==\s*\1\b' tests/ python/
rg -n --type py 'assert\s+True\b|pytest\.skip(?!IfInstalled)' tests/ python/
```

Status rules:
- any `REQUIRE(x == x)` / `ASSERT_EQ(x,x)` → **FAIL**
- `REQUIRE(true)` or `assert True` without a justifying `// WHY:` comment on the previous or same line → **FAIL**
- a failed/crashed test that also fails on serial rerun → **FAIL**
- a test that fails only under parallel ctest and passes on serial rerun → **WARN** (flake / shared-state issue)
- skipped tests with no justification string in the skip call → **WARN**
- tests tagged `[.]` (hidden from default run) → **WARN**

## B. Coverage honesty (untested public APIs)

Use a symbol index from the compiled library — far more reliable than string-grepping headers.

```bash
cd "$REPO"
# Public symbols from the compiled library (non-local text/data)
LIB_PATH=""
for candidate in \
  "$BUILD_DIR/bin/libdtwc++.a" \
  "$BUILD_DIR/bin/Release/dtwc++.lib" \
  "$BUILD_DIR/bin/Debug/dtwc++.lib" \
  "$BUILD_DIR/bin/dtwc++.lib"; do
  [ -f "$candidate" ] && LIB_PATH="$candidate" && break
done

NM_TOOL=""
for tool in nm llvm-nm; do
  command -v "$tool" >/dev/null 2>&1 && NM_TOOL="$tool" && break
done

if [ -n "$LIB_PATH" ] && [ -n "$NM_TOOL" ]; then
  PUBLIC_SYMS="$("$NM_TOOL" --defined-only -g "$LIB_PATH" 2>/dev/null \
    | awk '$2 ~ /[TWDR]/ {print $3}' \
    | c++filt \
    | grep -E '^dtwc::' \
    | sort -u)"
  TEST_SYMS="$(rg -o --no-filename --type cpp '\bdtwc::\w+(::\w+)*\b' tests/ 2>/dev/null | sort -u)"
  UNCOVERED="$(comm -23 <(printf '%s\n' "$PUBLIC_SYMS") <(printf '%s\n' "$TEST_SYMS"))"
  PUB_N="$(printf '%s\n' "$PUBLIC_SYMS" | sed '/^$/d' | wc -l | awk '{print $1}')"
  UNC_N="$(printf '%s\n' "$UNCOVERED" | sed '/^$/d' | wc -l | awk '{print $1}')"
  printf 'public_symbols=%s uncovered=%s\n' "$PUB_N" "$UNC_N"
  printf '%s\n' "$UNCOVERED" | head -30
else
  echo "WARN: compiled library or nm-like tool missing — coverage check degraded."
fi
```

Status rules:
- >20% of public symbols have zero test reference → **FAIL**
- 5–20% uncovered → **WARN**
- <5% uncovered → **PASS** (surface the list as informational)

## C. Performance documentation vs roofline

```bash
cd "$REPO"
rg -n -i 'gflops|gb/s|roofline|peak (bandwidth|flops)|arithmetic intensity' \
  benchmarks/ docs/ README.md CHANGELOG.md
# Benchmark result artefacts (fallback to find if fd missing)
(command -v fd >/dev/null && fd -t f -e json -e csv . benchmarks/results) \
  || find benchmarks/results -type f \( -name '*.json' -o -name '*.csv' \) 2>/dev/null \
  | head
```

Status rules:
- measured throughput absent AND no roofline analysis anywhere → **FAIL**
- throughput present, roofline / peak-machine comparison missing → **WARN**
- measured GFLOPS + ratio vs peak on target arch → **PASS**

## D. No heap allocation in hot loops

Two-pass scan (adversarial fix: old awk depth counter broke on nested braces). First enumerate for-loop lines with context, then filter in a separate pass.

```bash
cd "$REPO"
# List for-headers with 40 lines of context so we can see the loop body.
rg -n --type cpp -A 40 '^\s*for\s*\(' $HOT_DIRS 2>/dev/null \
  | rg -n '(\.push_back|\.emplace_back|std::vector<|std::string\(|std::make_unique|std::make_shared|\bnew\s+[A-Za-z_])' \
  | rg -v 'thread_local|static\s+(std::vector|std::array)|^\s*//' \
  | head -40
```

Additionally:

```bash
# Known-good scratch pattern — must appear in each hot kernel file
rg -n --type cpp 'thread_local\s+(static\s+)?std::vector|thread_local.*ScratchMatrix' \
  dtwc/warping.hpp dtwc/core/dtw_kernel.hpp dtwc/warping_adtw.hpp dtwc/warping_wdtw.hpp dtwc/warping_missing.hpp
```

Status rules:
- any match in `dtwc/warping*.hpp` or `dtwc/core/dtw_kernel.hpp` after the `rg -v` filter → **FAIL**, point at file:line
- matches in `dtwc/algorithms/` (CLARA/PAM outer loops) without `// amortised` comment → **WARN**
- hot kernels have thread_local scratch → confirms **PASS** criterion

## E. Duplication / DRY

```bash
cd "$REPO"
# N-gram of ≥20-line verbatim blocks across dtwc/
rg -n --type cpp -U '^[^\n]{20,}\n[^\n]{20,}\n[^\n]{20,}\n[^\n]{20,}\n[^\n]{20,}' $SRC_DIRS \
  | sort | uniq -c | sort -rn | awk '$1>=2 {print}' | head -20
# Structurally similar function signatures (stripped of digits)
rg -n --type cpp '^\s*[A-Za-z_:<>&\*\s]+\s+\w+\s*\([^;)]*\)\s*(const)?\s*\{' $SRC_DIRS \
  | sed -E 's/[0-9]+//g; s/\s+/ /g' | sort | uniq -c | sort -rn | head -20
```

Status rules:
- ≥20-line verbatim block in ≥2 files (excluding templated specializations on different types) → **FAIL**
- near-duplicate function signatures (same shape, different name) → **WARN**
- else → **PASS**

Exemption: Catch2-generated macros (`INTERNAL_CATCH_*`) and CUDA template specialisations (per-arch) do not count — strip them before scoring.

## F. Robust user-facing API

```bash
cd "$REPO"
# Throwing validation at boundaries
rg -n --type cpp 'throw\s+std::(invalid_argument|out_of_range|domain_error|length_error|runtime_error)' $SRC_DIRS
# Public functions with >5 positional args — but exempt those whose 1st arg is an Options/Settings/Config struct
rg -nU --type cpp '^\s*[A-Za-z_:<>&\*\s]+\s+(\w+)\s*\(\s*(?!(?:const\s+)?(?:dtwc::)?\w*(Options|Settings|Config)\b)[^;{)]*(,[^,;{)]*){5,}\)' \
  $SRC_DIRS --pcre2 2>/dev/null \
  | rg -v '__global__|__device__|<<<' | head -30
# Public API using only `assert` (killed in Release)
rg -n --type cpp '^\s*assert\(' $SRC_DIRS bindings/ python/
```

Status rules:
- public API with zero throws AND only `assert` guards → **FAIL**
- function with >5 positional args, first arg not an Options struct → **WARN**
- otherwise → **PASS**

## G. Docs + citations

```bash
cd "$REPO"
test -f .claude/CITATIONS.md && echo "citations present" || echo "MISSING"
# Code mentions of references
rg -n --type cpp -i 'reference:|@cite\b|\bSchmidt\s*&\s*Hundt|Keogh\s*&\s*Ratanamahatana' $SRC_DIRS docs/
# CITATIONS.md content
rg -n '^\s*-\s|\bDOI\b|\bdoi:|10\.\d{4,}' .claude/CITATIONS.md 2>/dev/null | head -20
```

Status rules:
- `.claude/CITATIONS.md` missing or empty → **FAIL**
- "Reference:" in code with no matching DOI/author in CITATIONS.md → **FAIL** (name offending file:line)
- uncited numeric claim (GFLOPS, big-O, error bound) → **WARN**

## H. Not reinventing the wheel

```bash
cd "$REPO"
rg -n --type cpp 'class\s+Matrix\b|struct\s+Matrix\b' $SRC_DIRS
rg -n --type cpp '\bsprintf\s*\(|\bsnprintf\b.*%[df]' $SRC_DIRS
rg -n --type cpp 'class\s+Logger\b|struct\s+Logger\b' $SRC_DIRS
rg -n --type cpp 'class\s+(SmallVector|Optional|Expected|Span)\b' $SRC_DIRS
# What deps are on offer?
rg -n 'CPMAddPackage|FetchContent_Declare|find_package' cmake/ CMakeLists.txt 2>/dev/null
```

Status rules:
- hand-rolled Matrix/Logger/Optional AND Eigen/spdlog/fmt already linked → **FAIL**
- hand-rolled helper with a trivial std-lib equivalent (`std::span`, `std::optional`, `<bit>`) → **WARN**
- domain-specific type (`core::DenseDistanceMatrix`) → **PASS**

## I. License compliance

```bash
cd "$REPO"
find . -maxdepth 3 -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' \) 2>/dev/null | head -20
# Enumerate declared deps
rg -n 'CPMAddPackage|FetchContent_Declare|find_package' cmake/ CMakeLists.txt 2>/dev/null \
  | awk -F'[ (]' '{for(i=1;i<=NF;i++) if ($i=="find_package" || $i=="CPMAddPackage" || $i=="FetchContent_Declare") print}'
```

For each dep, note license (check its upstream repo or vendored `LICENSE`):
- AGPL / GPLv3 in a non-GPL project → **FAIL**
- LGPL via static link w/o relink-exemption wording → **WARN**
- Unknown / missing LICENSE on vendored dep → **WARN**
- MIT / BSD / Apache-2.0 / Boost / MPL-2.0 → **PASS**

Gurobi is commercial — must be gated and NOT linked by default. Verify:
```bash
rg -n 'find_package\s*\(\s*(CUDA|Gurobi|HiGHS|Arrow|OpenMP|Metal)\s+.*REQUIRED' cmake/ CMakeLists.txt
```
Any `REQUIRED` on an optional dep → **FAIL** (breaks CLAUDE.md non-negotiable #3).

## J. Python/MATLAB wrapper parity

```bash
cd "$REPO"
# Does a cross-language bench exist?
(command -v fd >/dev/null && fd -t f 'test_cross_language*|bench_cross*|parity*' tests/ bindings/ python/) \
  || find tests bindings python -type f \( -name 'test_cross_language*' -o -name 'bench_cross*' -o -name 'parity*' \) 2>/dev/null

# Run it under uv (per CLAUDE.md #6: never pip)
if [ -f tests/integration/test_cross_language.py ]; then
  uv run pytest tests/integration/test_cross_language.py -q 2>&1 | tail -20 \
    || echo "cross-language bench present but failed or uv missing"
else
  echo "no cross-language bench"
fi
```

Status rules:
- bench exists and wrappers within ±10% of native C++ → **PASS**
- gap 10–100% → **WARN**
- gap ≥2× OR bench missing entirely → **FAIL** (library implicitly claims parity by shipping bindings)

## K. External benchmarking vs other DTW libs

```bash
cd "$REPO"
rg -n -i '\btslearn\b|\bdtaidistance\b|\bfastdtw\b|matlab.*dtw\b|file\s*exchange|rust-dtw' \
  benchmarks/ docs/ README.md CHANGELOG.md
```

Required: ≥1 comparison with MATLAB built-in `dtw`, `tslearn.metrics.dtw`, `dtaidistance`, `fastdtw`, `rust-dtw`. If none, propose methodology: same dataset (UCR ECG200 / UWave), same band, wall-clock on identical hardware, report mean ± std over ≥5 runs, include CPU model (NOT hostname — see L).

Status rules:
- ≥1 external comparison with numbers + methodology → **PASS**
- prose mentions but no measured numbers → **WARN**
- zero mention → **FAIL**

## L. DTWC++ project-specific invariants (HARD RULES)

Any violation is **FAIL**, no softening. User directives from memory/CLAUDE.md.

```bash
cd "$REPO"

# L1) No host_name in ANY committed JSON under benchmarks/ or docs/
rg -n '"host_name"\s*:\s*"[^"]+"' benchmarks/ docs/ 2>/dev/null

# L2) Benchmark WRITERS must not emit hostname (or must blank it).
rg -n 'gethostname|hostname\s*\(|HOST_NAME_MAX|uname.*nodename|socket\.gethostname|\.hostname\b' \
  benchmarks/ $SRC_DIRS python/ bindings/ 2>/dev/null

# L3) Tests must not write CSVs into repo root or repo-level `results/`
rg -n --type cpp '("|\b)\.\.?/[^"]*\.csv|"[A-Za-z0-9_]+\.csv"' tests/
find "$REPO" -maxdepth 1 -type f -name '*.csv' 2>/dev/null
find "$REPO/results" -maxdepth 1 -type f -name '*.csv' 2>/dev/null
# Must also be covered by .gitignore
rg -n 'results/|\*\.csv' .gitignore 2>/dev/null

# L4) No user-home / absolute-user paths in any committed artefact
rg -n '/home/|/Users/[^/]+/' benchmarks/results/ docs/ 2>/dev/null \
  | rg -v '/Users/\$USER|/Users/<user>|/Users/me'

# L5) Optional deps stay optional — core must build without CUDA/Metal/HiGHS/Gurobi/Arrow/OpenMP
rg -n 'find_package\s*\(\s*(CUDA|Gurobi|HiGHS|Arrow|OpenMP|Metal|MPI)\s+.*REQUIRED' cmake/ CMakeLists.txt

# L6) No naked new/delete in core (non-negotiable #7)
rg -n --type cpp '\bnew\s+[A-Za-z_]|\bdelete\s+[a-zA-Z_]' $SRC_DIRS \
  | rg -v '//|operator new|operator delete|placement'

# L7) CHANGELOG updated for any Unreleased change touching public API
test -f CHANGELOG.md && rg -n '^#\s*Unreleased|^##\s*Unreleased' CHANGELOG.md | head -3 \
  || echo "no Unreleased section found"
```

Status rules (each is an independent hard gate):
- L1 any hit → **FAIL**, point at file:line with the hostname string redacted to `REDACTED`
- L2 writer grep matches AND no matching post-process (sed/strip step in the writer) → **FAIL**
- L3 any CSV in repo root or `results/` that is not `.gitignore`d → **FAIL**
- L4 absolute-user path in committed results → **FAIL**
- L5 REQUIRED on optional dep → **FAIL**
- L6 naked `new`/`delete` in `$SRC_DIRS` → **FAIL**
- L7 public-API change without Unreleased entry (cross-check against `git diff HEAD~10 CHANGELOG.md`) → **WARN**

## M. Truthfulness / drift

```bash
cd "$REPO"
# Internal architecture docs vs shipped implementation
rg -n 'Separate functions per variant|std::function dispatch in Problem|Soft-DTW is a separate algorithm|WDTW/ADTW need recurrence changes' .claude/design.md
rg -n 'dtw_kernel_|resolve_dtw_fn|AROWCell|SoftCell' \
  dtwc/core/dtw_kernel.hpp dtwc/core/dtw_dispatch.cpp .claude/summaries/handoff-2026-04-13-phase4-wheel-fix.md

# User-facing docs vs implementation
rg -n 'CLI flags take precedence|command-line flags take precedence' \
  docs/content/getting-started/configuration.md docs/content/getting-started/cli.md
rg -n 'TODO: This always overrides CLI values if the YAML key exists' dtwc/dtwc_cl.cpp

# Test / algorithm naming drift
rg -n 'fast_pam' tests/unit/adversarial/test_fast_pam_adversarial.cpp tests/unit/algorithms/unit_test_fast_pam.cpp
rg -n 'Method::Kmedoids|cluster_by_kMedoidsLloyd' \
  tests/unit/adversarial/test_fast_pam_adversarial.cpp tests/unit/algorithms/unit_test_fast_pam.cpp
```

Status rules:
- user-facing docs claim behaviour that the code explicitly contradicts → **FAIL**
- a test file/title claims algorithm X but drives algorithm Y, creating false coverage confidence → **FAIL**
- stale internal design docs contradict shipped architecture → **WARN**
- else → **PASS**

## N. Senior-eng hardening (informational, one-pass)

Report status, don't block PASS/FAIL on these unless dire:

```bash
cd "$REPO"
# Sanitizers wired into CMake
rg -n -i 'asan|ubsan|-fsanitize|SANITIZE' cmake/ CMakeLists.txt | head
# -Werror discipline
rg -n 'Werror|WARNINGS_AS_ERRORS' cmake/ CMakeLists.txt | head
# CI matrix
find .github/workflows -name '*.yml' -o -name '*.yaml' 2>/dev/null | head
# compile_commands freshness (required for clang-tidy / IDE)
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
  stat -c "%y %n" "$BUILD_DIR/compile_commands.json" 2>/dev/null \
    || stat -f "%Sm %N" "$BUILD_DIR/compile_commands.json" 2>/dev/null
fi
# clang-tidy config
test -f .clang-tidy && head -3 .clang-tidy
# Reproducible build flags
rg -n 'SOURCE_DATE_EPOCH|-frandom-seed|-ffile-prefix-map' cmake/ CMakeLists.txt
```

Report each as present/absent. No PASS/FAIL — surface as `Info:` bullets so the reader can prioritise.

---

## Output shape (strict)

```
# Code Quality Audit — DTWC++ <ISO-date>
Scope: <path or "whole repo">
Tool availability: git=OK rg=OK cmake=OK ctest=OK nm_like=OK uv=OK fd=MISSING

## A. Test quality
Status: PASS|WARN|FAIL
Findings:
- <file:line> <what>
Remediation:
- <one bullet>

## B. Coverage
...

## L. Project-specific invariants
...

## M. Truthfulness / drift
...

## N. Hardening (info only)
- sanitizers: present|absent
- -Werror: present|absent
- CI matrix: <count> workflows
- compile_commands.json: fresh|stale|missing
- .clang-tidy: present|absent
- reproducible-build flags: present|absent

## Verdict: PASS|WARN|FAIL
Summary: <3 sentences, blunt>

<!-- MACHINE-READABLE TRAILER -->
```json
{
  "verdict": "FAIL",
  "sections": {
    "A": "PASS", "B": "WARN", "C": "FAIL", "D": "PASS",
    "E": "PASS", "F": "WARN", "G": "PASS", "H": "PASS",
    "I": "PASS", "J": "WARN", "K": "FAIL", "L": "PASS",
    "M": "FAIL"
  },
  "tool_availability": {"git": true, "rg": true, "cmake": true, "ctest": true, "nm_like": true, "fd": false}
}
```
```

Aggregation rule: any FAIL in A–M → FAIL; else any WARN → WARN; else PASS. Section N is informational and does NOT feed the verdict.

No emojis. No congratulatory filler. If verdict is PASS, the summary names the single weakest area. If FAIL, the summary lists the top 3 blockers in priority order.

## Self-restraint

- Each finding gets ≤2 remediation bullets. No prescriptions longer than that.
- Do not read files outside the declared scope unless a cross-reference demands it (e.g. CITATIONS.md referenced from a hot-path header).
- Do not edit tracked files. Build/test output under `$BUILD_DIR` or system temp is acceptable; do not intentionally write under the tracked source tree.
- If a check cannot be run (tool missing, build/ not primed), emit `Status: WARN — could not evaluate: <reason>` rather than silently skipping or guessing.
- If a grep returns zero hits because its paths are wrong, that is a **skill bug**, not a clean result — cross-check against `find $SCOPE -type d` when a check claims PASS with zero output.
