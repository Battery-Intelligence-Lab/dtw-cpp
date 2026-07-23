#!/usr/bin/env python3
"""Reject stale claims in the durable research and lessons records."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require_absent(paths: tuple[str, ...]) -> None:
    present = [path for path in paths if (ROOT / path).exists()]
    if present:
        raise AssertionError(f"retired records were recreated: {present}")


def require_markers(label: str, text: str, markers: tuple[str, ...]) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{label} omits required current markers: {missing}")


def reject_markers(label: str, text: str, markers: tuple[str, ...]) -> None:
    present = [marker for marker in markers if marker in text]
    if present:
        raise AssertionError(f"{label} retains stale claims: {present}")


def check_unimodular() -> None:
    text = read(".claude/UNIMODULAR.md")
    freshness_lines = [
        line for line in text.splitlines()
        if line.startswith("> **Freshness (2026-07-23):")
    ]
    if len(freshness_lines) != 1:
        raise AssertionError("UNIMODULAR must contain exactly one current freshness header")
    freshness = freshness_lines[0]
    missing_commits = [
        commit for commit in ("85eabcd", "fd49ff4", "8375aef", "789cc52")
        if commit not in freshness
    ]
    if missing_commits:
        raise AssertionError(
            f"UNIMODULAR freshness header omits commit anchors: {missing_commits}"
        )
    require_markers(
        "UNIMODULAR",
        text,
        (
            "historical proposal",
            "implementation roadmap is historical",
            "Geoffrion's theorem (1974)",
            "constructed cycle family",
            "let `c = 1`",
            "`c - Σs ∈ {-1,0,1}`",
            "Kelley",
            "y-only",
            "continuous assignment-cost surrogates",
            "fix_tol = 1e-9",
            "LagrangianParams::max_nodes",
            "`certified_optimal=false`",
            "LR-core is not the bound engine for `benders.cpp`",
            "800,000,000 bytes",
            "40/40",
            "80.3%",
            "73.3%",
            "77.8%",
            "`.claude/PLAN-archive-2026-07-20-phases0-9.md:428-434`",
            "P3 remains unmeasured",
            "Cristian Duran-Mateluna",
            "10.1016/j.ejor.2022.11.033",
        ),
    )
    reject_markers(
        "UNIMODULAR",
        text,
        (
            "To exhibit a genuine gap, consider p = 4",
            "Lower bound on gap | 2",
            "Upper bound on gap | ~2.675",
            "p <= ~10",
            "Each fractional extreme point corresponds to an odd cycle",
            "Variables on the cycle take value 1/2",
            "Expected: integer ~80-90%",
            "Jain and Vazirani, 2001",
            "O(pk) per cut generation",
            "mip_detect_symmetry",
            "A[i,i] >= A[i+1,i+1]",
            "min  theta\n",
            "(N-1)*A[i,i]",
            "canonical tracked derivation",
            "Only odd cycles among facilities break total unimodularity",
            "Only odd cycles violate TU",
            "`|1 − (m_i mod 2)·(±1)| ≤ 1`",
            "exactly the odd-cycle obstruction",
            "LR-core is its bound engine",
            "Lagrangian bound = LP bound",
            "Current best 2.675+eps",
            "dual bound = LP bound",
            'empirical observation that its LP relaxation is "almost always integral"',
            "Benders core finish",
            "run Benders (`benders.cpp`)",
            "D packed = 400 MB f64",
            "Task 4.1 (`dtwc/mip/lagrangian_root",
            "Duran-Mateluna, G.",
            "(`PLAN-archive-2026-07-20-phases0-9.md:428-434`)",
        ),
    )


def check_lessons() -> None:
    text = read(".claude/LESSONS.md")
    require_markers(
        "LESSONS",
        text,
        (
            "uv add/remove",
            "fresh `.pyd` + `libomp.dll`",
            "BM_dtwFull_L/1000",
            "2967",
            "1924",
            "BM_dtwBanded/1000/50",
            "258",
            "146",
            "BM_wdtwBanded_g/1000/50",
            "450",
            "159",
            "2.95×–8.06×",
            "operation-count estimate",
            "1.57×–1.90×",
            "uses only 10 random length-500",
            "mixed environment",
            "metric-only",
            "20.58×",
            "Float64 is the default",
            "F25",
            "which('dtwc_mex','-all')",
            "61/61",
            "`test_parallelisation` engaging 24 OpenMP threads",
            "v1.15.1",
            "PyArrow 23",
            "architecture-specific",
            "AArch64 build/runtime remains unverified",
            "F9 is closed",
            "390 assertions",
            "11 Arrow/Parquet cases",
            "7c71602",
            "833f570",
            "0c91c9b",
            "e323197",
            "ffb7a8d",
            "13.718%",
        ),
    )
    reject_markers(
        "LESSONS",
        text,
        (
            "pip install -e .",
            "MIP gap bounds don't formally apply",
            "installed `dtwcpp` is a non-editable wheel",
            "`OMP_PROC_BIND` / `OMP_PLACES` are no-ops on Darwin",
            "Only 3% of L1 bandwidth used",
            "DTW is latency-bound** (10-cycle recurrence)",
            "Branchless scalar matches explicit SIMD for DTW",
            "2.3×–8.1×",
            "i*(i+1)/2` = 1-2 cycles",
            "`std::min({a,b,c})` is catastrophically slow",
            "Mmap is safe as default",
            "llfio > mio",
            "DTW dominates I/O by 10-100x",
            "Arrow IPC = same speed as .dtws",
            "Always use LargeListArray",
            "Full scan with 8 cores",
            "Float32 DTW speed = identical to float64",
            "Float32 benefit is purely memory",
            "Negligible for medoid selection",
            "Default to float32",
            "Assert `!is_view()`",
            "MEX longjmp skips destructors",
            "Check output, not exit code",
            "mexLock() prevents shutdown crashes",
            "OpenMP 24/24",
            "Stable ABI, 5-10x smaller binaries",
            "Verified by Codex",
            "HiGHS: row-major. Gurobi: column-major",
            "Check `TARGET X` instead of `X_FOUND`",
            "Use vcpkg/conda for proper install",
            "defended only by a hand-linked PyArrow build",
            "All arc nodes support v4",
            "CUDA kernel not yet ported",
            "first CI run remains open",
            "gap ≤13.7%",
            "even 12000 iters",
            "never within two orders of Kelley",
            "on a totally-unimodular problem",
        ),
    )


def check_citations() -> None:
    text = read(".claude/CITATIONS.md")
    exact_once = (
        "Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms",
        "10.1016/j.is.2021.101804",
        "arXiv:2008.05171",
        "TC-DTW: Accelerating Multivariate Dynamic Time Warping Through Triangle Inequality and Point Clustering",
        "arXiv:2101.07731",
        "Generalizing DTW to the multi-dimensional case requires an adaptive approach",
        "10.1007/s10618-016-0455-0",
        "https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html",
    )
    bad_counts = {marker: text.count(marker) for marker in exact_once if text.count(marker) != 1}
    if bad_counts:
        raise AssertionError(f"CITATIONS canonical-entry counts differ from one: {bad_counts}")
    require_markers(
        "CITATIONS",
        text,
        (
            "Information Systems*, 101, 101804",
            "Cristian Duran-Mateluna",
            "10.1016/j.ejor.2022.11.033",
            "10.1137/1.9781611972719.1",
            "mexErrMsgIdAndTxt",
            "matlab/apiref/mexerrmsgidandtxt.html",
            "creating-c-mex-files.html",
            "cuda/archive/13.0.0/cuda-c-programming-guide/index.html",
            "cuda/archive/13.0.0/cuda-runtime-api/structcudaDeviceProp.html",
            "Zemin Chao",
            "arXiv:2603.14899",
            "Daniel Shen",
            "Min Chi",
            "Ron Shapira Weber",
            "Oren Freifeld",
            "arXiv:2602.17206",
            "releases/tag/v1.15.1",
            "tree/v1.15.1/highs/pdlp",
            "blob/v1.15.1/CMakeLists.txt",
            "format/CDataInterface.html",
            "PyCapsuleInterface.html",
            "apache-arrow-nanoarrow-0.8.0",
            "10.1109/MLSP.2012.6349714",
        ),
    )
    reject_markers(
        "CITATIONS",
        text,
        (
            "This function calls `longjmp`",
            "*JMLR*, 22(1), 4653-4688",
            "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
            "https://docs.nvidia.com/cuda/cuda-runtime-api/",
            "Shen, Y., & Chen, Y.",
            "https://github.com/ERGO-Code/HiGHS —",
            " + .../format/",
            "immutable source",
            "immutable release",
        ),
    )


def check_plan() -> None:
    text = read("PLAN.md")
    require_markers(
        "PLAN",
        text,
        (
            "MISSING.md` / `READ.md` were retired by `0449f7c`",
            "do not recreate them",
            "2.95–8.06×",
            "the memory-bound explanation remains",
            "**[inferred]** pending D17 counters",
        ),
    )
    reject_markers(
        "PLAN",
        text,
        (
            "UNIMODULAR.md / MISSING.md / READ.md:\n      add a one-line freshness header",
            "recorded 2.3–8.1×",
            "measured 2.3–8.1×",
            "DTW ≈ 0.125 FLOP/byte, memory-bound",
        ),
    )


def check_supporting_records() -> None:
    faster_pam = read(".claude/baselines/2026-07-08-faster-pam-bench.md")
    require_markers(
        "FasterPAM baseline",
        faster_pam,
        (
            "**Record correction (2026-07-23):**",
            "2.95×–8.06×",
            "non-monotone",
            "Why NOT ≥10× remains inferred",
        ),
    )
    reject_markers(
        "FasterPAM baseline",
        faster_pam,
        (
            "2.3×–8.1×",
            "growing with k**",
            "confirming memory — not allocation — is the bottleneck",
            "unit_test_fast_clara:619",
        ),
    )

    baseline = read(".claude/baselines/2026-07-23-r1-record-hygiene.md")
    require_markers(
        "record-hygiene baseline",
        baseline,
        (
            "**[confirmed]**",
            "Python 3.13.7",
            "package=C:\\D\\git\\dtw-cpp\\python\\dtwcpp\\__init__.py",
            "2.95×–8.06×",
            "The first post-edit checker green was not accepted as final evidence.",
            "**[inferred]** Commit `420f764`",
            "With no raw transcript",
        ),
    )
    reject_markers(
        "record-hygiene baseline",
        baseline,
        (
            "- [confirmed]",
            "HiGHS' immutable",
            "pins the identities and immutable/versioned URL text",
            "Its 199.6 MB to 9.7 MB fixture result is 20.58×",
        ),
    )


def main() -> int:
    require_absent((".claude/MISSING.md", ".claude/READ.md"))
    check_unimodular()
    check_lessons()
    check_citations()
    check_plan()
    check_supporting_records()
    print("record hygiene checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
