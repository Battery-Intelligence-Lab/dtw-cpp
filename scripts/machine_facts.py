#!/usr/bin/env python3
"""Describe the machine and the build a benchmark ran on.

Benchmark numbers are only meaningful next to the machine that produced them, and a
record typed by hand is a record that drifts.  This prints the same facts on macOS,
Linux and Windows: CPU, memory, GPU, OS, toolchain and the resolved DTWC_* options.

Build flags come from the CMake cache, not from a guess, so what is reported is what
was compiled.

    uv run --no-project python scripts/machine_facts.py               # markdown table
    uv run --no-project python scripts/machine_facts.py --json
    uv run --no-project python scripts/machine_facts.py --build-dir build-cuda
    uv run --no-project python scripts/machine_facts.py --hostname    # opt in

Stdlib only.  Every probe is best-effort: a fact we cannot read is reported as
"unknown" rather than omitted, so a gap in the record is visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

UNKNOWN = "unknown"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    """Run a probe command; return stripped stdout, or "" on any failure."""
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def _human_bytes(n: int) -> str:
    return f"{n / (1024 ** 3):.1f} GiB"


# --------------------------------------------------------------------------- CPU / RAM


def cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or UNKNOWN
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name") or line.startswith("Model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        # aarch64 has no "model name"; fall back to the implementer/part pair.
        return platform.processor() or platform.machine() or UNKNOWN
    if system == "Windows":
        name = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"])
        return name.splitlines()[0].strip() if name else (platform.processor() or UNKNOWN)
    return platform.processor() or UNKNOWN


def cpu_counts() -> tuple[str, str]:
    """(physical, logical) as strings."""
    logical = str(os.cpu_count() or UNKNOWN)
    system = platform.system()
    physical = UNKNOWN
    if system == "Darwin":
        physical = _sysctl("hw.physicalcpu") or UNKNOWN
        logical = _sysctl("hw.logicalcpu") or logical
    elif system == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            # One "core id" set per physical package.
            pairs = set(re.findall(r"physical id\s*:\s*(\d+)[\s\S]*?core id\s*:\s*(\d+)", text))
            if pairs:
                physical = str(len(pairs))
        except OSError:
            pass
    elif system == "Windows":
        cores = _run(["powershell", "-NoProfile", "-Command",
                      "(Get-CimInstance Win32_Processor | "
                      "Measure-Object -Property NumberOfCores -Sum).Sum"])
        physical = cores.strip() or UNKNOWN
    return physical, logical


def ram_total() -> str:
    system = platform.system()
    if system == "Darwin":
        raw = _sysctl("hw.memsize")
        return _human_bytes(int(raw)) if raw.isdigit() else UNKNOWN
    if system == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return _human_bytes(int(line.split()[1]) * 1024)
        except (OSError, ValueError, IndexError):
            pass
        return UNKNOWN
    if system == "Windows":
        raw = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        return _human_bytes(int(raw)) if raw.strip().isdigit() else UNKNOWN
    return UNKNOWN


# --------------------------------------------------------------------------- GPU


def gpus() -> list[str]:
    """Every GPU we can name, CUDA first.  Empty list means none was found."""
    found: list[str] = []
    smi = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader"])
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            name, mem, cap = parts[0], parts[1], parts[2]
            driver = f", driver {parts[3]}" if len(parts) > 3 else ""
            found.append(f"{name} ({mem}, compute {cap}{driver})")
    if platform.system() == "Darwin" and not found:
        # Apple Silicon: the GPU is part of the SoC, and the working-set limit is the
        # number that matters for our Metal backend.  dtwc::metal::metal_device_info()
        # reports it exactly; use it when the module is importable (see --with-dtwcpp).
        chip = _sysctl("machdep.cpu.brand_string")
        if chip:
            found.append(f"{chip} integrated GPU (Metal)")
    return found


# --------------------------------------------------------------------------- toolchain


def cmake_cache_facts(build_dir: Path) -> dict[str, object]:
    """Compiler, build type, flags and every resolved DTWC_* option."""
    facts: dict[str, object] = {
        "build_dir": str(build_dir),
        "present": False,
        "options": {},
    }
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return facts
    facts["present"] = True
    wanted = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
              "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CXX_STANDARD", "CMAKE_GENERATOR")
    options: dict[str, str] = {}
    try:
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            name, _, entry_type = key.partition(":")
            if name in wanted:
                facts[name.lower()] = value.strip()
            elif name.startswith("DTWC_") and entry_type != "INTERNAL":
                # INTERNAL entries are CMake's own UI hints (DTWC_ARCH_LEVEL-STRINGS),
                # not options anyone set.
                options[name] = value.strip()
    except OSError:
        return facts
    facts["options"] = dict(sorted(options.items()))

    # CMake records the compiler it actually probed; that beats running `--version`
    # on whatever is first on PATH today.
    for path in glob.glob(str(build_dir / "CMakeFiles" / "*" / "CMakeCXXCompiler.cmake")):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cid = re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)', text)
        ver = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)', text)
        if cid and ver:
            facts["compiler"] = f"{cid.group(1)} {ver.group(1)}"
            break
    facts.setdefault("compiler", UNKNOWN)
    return facts


def git_facts() -> dict[str, str]:
    head = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]) or UNKNOWN
    branch = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"]) or UNKNOWN
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    version = UNKNOWN
    version_file = REPO_ROOT / "VERSION"
    if version_file.is_file():
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return {
        "head": head,
        "branch": branch,
        "tree": "dirty" if status else "clean",
        "version": version,
    }


def dtwcpp_probes() -> dict[str, object]:
    """Ask the installed package what it can actually do — engagement, not flags.

    dtwc::test::parallelisation() counts distinct threads that really ran, and
    dtwc::test::gpu() validates a real DTW matrix against the CPU oracle.  A benchmark
    recorded while OpenMP silently ran on one thread is worse than no benchmark.
    """
    try:
        import dtwcpp  # noqa: PLC0415 - optional, probed on purpose
    except ImportError as exc:
        return {"importable": False, "reason": str(exc)}
    out: dict[str, object] = {"importable": True, "version": getattr(dtwcpp, "__version__", UNKNOWN)}
    for flag in ("OPENMP_AVAILABLE", "CUDA_AVAILABLE", "METAL_AVAILABLE",
                 "HIGHS_AVAILABLE", "MPI_AVAILABLE"):
        out[flag.lower()] = getattr(dtwcpp, flag, None)
    try:
        out["parallelisation"] = dtwcpp.test.parallelisation()
    except Exception as exc:  # noqa: BLE001 - a probe must never fail the record
        out["parallelisation"] = {"error": str(exc)}
    try:
        out["gpu"] = dtwcpp.test.gpu()
    except Exception as exc:  # noqa: BLE001
        out["gpu"] = {"error": str(exc)}
    return out


# --------------------------------------------------------------------------- assembly


def collect(build_dir: Path, *, with_hostname: bool, with_dtwcpp: bool) -> dict[str, object]:
    physical, logical = cpu_counts()
    facts: dict[str, object] = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": cpu_model(),
        "cpu_cores_physical": physical,
        "cpu_threads_logical": logical,
        "ram_total": ram_total(),
        "gpus": gpus(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "python": platform.python_version(),
        "git": git_facts(),
        "build": cmake_cache_facts(build_dir),
    }
    if with_hostname:
        facts["hostname"] = platform.node()
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    if slurm:
        facts["slurm"] = {
            k: slurm[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
                                  "SLURM_CPUS_PER_TASK", "SLURM_NTASKS") if k in slurm
        }
    if with_dtwcpp:
        facts["dtwcpp"] = dtwcpp_probes()
    return facts


def _build_flags(build: dict[str, object]) -> str:
    """The compiler flags CMake recorded, for the configured build type.

    Only the cache-level flags: DTWC++ attaches `-march=native`, the FP model and
    friends to the `project_options` / `dtwc++` targets, which never appear in
    CMakeCache.txt. The DTWC_* option list below is what carries those, so read
    the two together rather than treating this row as the whole command line.
    """
    build_type = str(build.get("cmake_build_type", "") or "")
    parts = [
        str(build.get("cmake_cxx_flags", "") or ""),
        str(build.get(f"cmake_cxx_flags_{build_type.lower()}", "") or ""),
    ]
    flags = " ".join(part for part in parts if part).strip()
    standard = build.get("cmake_cxx_standard")
    if standard:
        flags = f"{flags} (C++{standard})".strip()
    return f"`{flags}`" if flags else UNKNOWN


def as_markdown(facts: dict[str, object]) -> str:
    build = facts.get("build", {})
    assert isinstance(build, dict)
    rows = [
        ("Host", facts.get("hostname", "(omitted)")),
        ("OS", facts["os"]),
        ("CPU", f"{facts['cpu']} — {facts['cpu_cores_physical']} cores / "
                f"{facts['cpu_threads_logical']} threads"),
        ("Memory", facts["ram_total"]),
        ("GPU", "; ".join(facts["gpus"]) if facts["gpus"] else "none detected"),
        ("Compiler", build.get("compiler", UNKNOWN)),
        ("Build type", build.get("cmake_build_type", UNKNOWN)),
        ("Build flags", _build_flags(build)),
        ("Generator", build.get("cmake_generator", UNKNOWN)),
        # The env var only; the count that actually ran is proven by the
        # --with-dtwcpp parallelisation probe below, which is what a benchmark
        # record should cite. Unset does NOT reliably mean "one thread per core".
        ("OMP_NUM_THREADS", facts["omp_num_threads"]),
        ("Repository", f"{facts['git']['branch']} @ {facts['git']['head']} "  # type: ignore[index]
                       f"({facts['git']['tree']}), VERSION {facts['git']['version']}"),  # type: ignore[index]
    ]
    if "slurm" in facts:
        slurm = facts["slurm"]
        assert isinstance(slurm, dict)
        rows.append(("SLURM", ", ".join(f"{k.removeprefix('SLURM_').lower()}={v}"
                                        for k, v in slurm.items())))

    lines = ["| Fact | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]

    options = build.get("options", {})
    assert isinstance(options, dict)
    if options:
        on = [k for k, v in options.items() if v.upper() in ("ON", "TRUE", "1", "YES")]
        off = [k for k, v in options.items() if v.upper() in ("OFF", "FALSE", "0", "NO")]
        other = {k: v for k, v in options.items() if k not in on and k not in off}
        lines.append("")
        lines.append(f"**Options ON:** {', '.join(on) if on else '(none)'}")
        lines.append("")
        lines.append(f"**Options OFF:** {', '.join(off) if off else '(none)'}")
        if other:
            lines.append("")
            lines.append("**Options set:** "
                         + ", ".join(f"`{k}={v}`" for k, v in sorted(other.items())))
    elif not build.get("present"):
        lines.append("")
        lines.append(f"**Build flags unknown** — no CMakeCache.txt under `{build.get('build_dir')}`; "
                     "pass `--build-dir` so the record says what was compiled.")

    probes = facts.get("dtwcpp")
    if isinstance(probes, dict) and probes.get("importable"):
        par = probes.get("parallelisation")
        gpu = probes.get("gpu")
        lines.append("")
        if isinstance(par, dict) and "threads_engaged" in par:
            lines.append(f"**Parallelisation proven:** {par.get('threads_engaged')} threads engaged "
                         f"of {par.get('max_threads')} — pass={par.get('pass')}")
        if isinstance(gpu, dict) and "backend" in gpu:
            lines.append(f"**GPU proven:** backend={gpu.get('backend') or 'none'}, "
                         f"device={gpu.get('device_name') or 'n/a'}, "
                         f"validated={gpu.get('validated')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", default="build", type=Path,
                        help="build directory whose CMakeCache.txt describes the flags (default: build)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a markdown table")
    parser.add_argument("--hostname", action="store_true",
                        help="include the hostname (omitted by default, as run_bench.sh strips it)")
    parser.add_argument("--with-dtwcpp", action="store_true",
                        help="also run dtwcpp.test.parallelisation() and .gpu() engagement probes")
    args = parser.parse_args()

    build_dir = args.build_dir if args.build_dir.is_absolute() else REPO_ROOT / args.build_dir
    facts = collect(build_dir, with_hostname=args.hostname, with_dtwcpp=args.with_dtwcpp)
    print(json.dumps(facts, indent=2) if args.json else as_markdown(facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
