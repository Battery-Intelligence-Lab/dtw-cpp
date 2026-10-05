#!/usr/bin/env python3
"""Codegen report for the DTW hot paths (X-04).

Answers one question with evidence instead of opinion: *which loops does the
compiler actually vectorise, and which does it refuse to?* That is the input
D-17 needs before anyone argues for hand-written SIMD again, because the case
for SIMD is only as good as the gap between what the compiler already does and
what a kernel could do.

Rather than compile a synthetic probe, this replays the **real** compile command
out of compile_commands.json with clang's vectorisation remarks switched on, so
the report describes the code that actually ships -- same flags, same floating
point model, same architecture tuning. The object goes to a temporary file and
is discarded; nothing in the build tree is touched.

Prints

  CODEGEN_REPORT tool=<cc> tus=<n> vectorized=<n> missed=<n>

then one line per loop: file:line:col and whether it was vectorised. A manual
tool, not a gate: the answer depends on the compiler and the host. Exit 0 when a
report was produced, 2 when it could not be (no clang, no compile_commands.json,
no matching TU) -- never silently green. --no-calls (test_codegen_no_calls) fails
instead, exit 1, when a call sits inside any loop of a dtwc function, however deep.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# clang remark lines look like:
#   path/file.cpp:123:45: remark: vectorized loop (vectorization width: 2, ...) [-Rpass=loop-vectorize]
#   path/file.cpp:123:45: remark: loop not vectorized: ... [-Rpass-missed=loop-vectorize]
REMARK = re.compile(
    r"^(?P<file>(?:[A-Za-z]:)?[^:]+):(?P<line>\d+):(?P<col>\d+):\s+remark:\s+(?P<msg>.*?)"
    r"\s*\[-R(?P<kind>pass|pass-missed|pass-analysis)=(?P<pass>[\w-]+)\]\s*$"
)
REMARK_FLAGS = [
    "-Rpass=loop-vectorize",
    "-Rpass-missed=loop-vectorize",
    "-Rpass-analysis=loop-vectorize",
    # StandardProjectSettings.cmake adds -fcolor-diagnostics, which wraps every
    # remark in ANSI escapes and makes them unparseable. This comes later on the
    # command line and wins.
    "-fno-color-diagnostics",
]
# Flags that would stop the compile emitting the remarks we came for, or that
# write somewhere in the build tree we are not entitled to disturb.
DROP_PREFIXES = ("-o", "-MF", "-MT", "-MD", "-MMD", "-MQ", "-c")
# ThinLTO defers optimisation to link time: with -flto the -c step emits bitcode
# and the loop vectoriser has simply not run yet, so -Rpass reports nothing at
# all. The probe therefore compiles with LTO off. What the report then describes
# is per-TU codegen of these kernels under the project's real flags -- which is
# the question X-04 asks -- and not the final post-link code of an LTO build.
LTO_PREFIX = "-flto"
# --no-calls reads clang's loop comments. A loop's header block says "This Loop Header", or
# "This Inner Loop Header" when the loop has no sub-loop; every other block of a loop says
# "in Loop: Header=BBn", naming the innermost loop it is in. Either is loop code, at any depth:
# the banded kernel's column loop has a child loop, and a call there is a call per column.
BLOCK = re.compile(r"^(?:\.?LBB(\w+):|\s*(?:#|;|//) %bb\.\d+:)")
FUNCTION = re.compile(r'^(?![.Ll])("[^"]+"|\S+):')
CALL = re.compile(r"^\s+(?:call|callq|bl|blr)\s")
LOOP = re.compile(r"This (?:Inner )?Loop Header|in Loop: Header=BB")


def compile_commands(build_dir: Path) -> list[dict]:
    db = build_dir / "compile_commands.json"
    if not db.is_file():
        raise FileNotFoundError(db)
    return json.loads(db.read_text(encoding="utf-8"))


def rebuild_argv(entry: dict, out_obj: Path, source: Path,
                 mode: tuple[str, ...] = (*REMARK_FLAGS, "-c")) -> list[str]:
    """The entry's own command, compiling `source` instead, with remarks on.

    The flags are borrowed wholesale from a real library translation unit so the
    probe is compiled exactly as the shipped code is -- include paths, -O level,
    DTWC_FP_MODEL flags, architecture tuning and all. Only the source file and
    the output path are replaced.
    """
    nt = sys.platform == "win32"  # keeps backslash paths, which a POSIX split eats; quotes are \"
    argv = ([a.replace('\\"', '"') if nt else a for a in shlex.split(entry["command"], posix=not nt)]
            if "command" in entry else list(entry["arguments"]))
    own_source = entry["file"]
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in DROP_PREFIXES:
            # "-o obj" and "-MF dep" carry their value in the next argument.
            skip_next = arg in ("-o", "-MF", "-MT", "-MQ")
            continue
        if arg.startswith("-o") and len(arg) > 2:
            continue
        if arg.startswith(LTO_PREFIX):
            continue
        # Drop the donor's own source, however it is spelled in the command.
        if arg == own_source or Path(arg).name == Path(own_source).name:
            continue
        cleaned.append(arg)
    return [*cleaned, *mode, str(source), "-o", str(out_obj)]


def collect(entry: dict, out_obj: Path, source: Path) -> tuple[list[dict], str]:
    argv = rebuild_argv(entry, out_obj, source)
    proc = subprocess.run(
        argv, cwd=entry.get("directory", str(ROOT)),
        capture_output=True, text=True, errors="replace",
    )
    found: list[dict] = []
    for line in proc.stderr.splitlines():
        m = REMARK.match(line.strip())
        if not m or m.group("pass") != "loop-vectorize":
            continue
        if m.group("kind") == "pass-analysis":
            continue  # explanatory only; the state is pass vs pass-missed
        try:
            # Remark paths arrive like ".../dtwc/./core/dtw_kernel.hpp"; resolve
            # normalises that. Anything outside the repo is a standard-library or
            # SDK header — real remarks, but not ours to pin.
            rel = str(Path(m.group("file")).resolve().relative_to(ROOT))
        except ValueError:
            continue
        found.append(
            {
                "file": rel,
                "line": int(m.group("line")),
                "col": int(m.group("col")),
                "vectorized": m.group("kind") == "pass",
            }
        )
    return found, argv[0]


def key(rec: dict) -> str:
    return f"{rec['file']}:{rec['line']}:{rec['col']}"


def loop_calls(asm: str) -> tuple[int, list[str]]:
    """(innermost-loop count, calls inside any loop) over the dtwc functions in `asm`."""
    fn = block = ""
    inner, in_loop, calls = set(), set(), []
    for n, line in enumerate(asm.splitlines()):
        if m := FUNCTION.match(line):
            fn = m.group(1)
        if m := BLOCK.match(line):
            block = m.group(1) or f"line{n}"
        if "dtwc" not in fn:
            continue
        if LOOP.search(line):
            in_loop.add(block)
        if "This Inner Loop Header" in line:
            inner.add(block)
        if CALL.match(line):
            calls.append((block, f"{fn}: {line.strip()}"))
    return len(inner), [c for b, c in calls if b in in_loop]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build")
    ap.add_argument(
        "--match",
        default=r"dtwc/core/dtw_dispatch\.cpp",
        help="regex picking the translation unit whose flags the probe borrows",
    )
    ap.add_argument(
        "--probe",
        type=Path,
        default=ROOT / "scripts" / "codegen_probe.cpp",
        help="the probe TU that instantiates the kernels (see its header comment)",
    )
    ap.add_argument("--no-calls", action="store_true",
                    help="fail if a call sits inside any loop of a probe kernel")
    args = ap.parse_args()

    try:
        db = compile_commands(args.build_dir)
    except FileNotFoundError as exc:
        print(f"CODEGEN_REPORT tool=none tus=0 vectorized=0 missed=0 verdict=FAIL")
        print(f"  no compile_commands.json at {exc}", file=sys.stderr)
        return 2

    pattern = re.compile(args.match)
    entries = [e for e in db if pattern.search(e["file"]) and "_deps" not in e["file"]]
    if not entries:
        print("CODEGEN_REPORT tool=none tus=0 vectorized=0 missed=0 verdict=FAIL")
        print(f"  no translation unit matched {args.match!r}", file=sys.stderr)
        return 2
    if args.no_calls:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "probe.s"
            argv = rebuild_argv(entries[0], out, args.probe, ("-S", "-fno-color-diagnostics"))
            proc = subprocess.run(argv, cwd=entries[0].get("directory", str(ROOT)),
                                  capture_output=True, text=True, errors="replace")
            loops, bad = loop_calls(out.read_text(errors="replace") if out.is_file() else "")
        # A listing with no loop proves nothing: loops > 0 shows the probe instantiated the kernels.
        ok = proc.returncode == 0 and loops > 0 and not bad
        print(f"CODEGEN_NO_CALLS tool={Path(argv[0]).name} inner_loops={loops} calls={len(bad)} "
              f"verdict={'PASS' if ok else 'FAIL'}", *bad, sep="\n  ")
        print(proc.stderr[-2000:] if proc.returncode else "", file=sys.stderr, end="")
        return 0 if ok else 1

    records: list[dict] = []
    tool = "none"
    with tempfile.TemporaryDirectory() as tmp:
        for i, entry in enumerate(entries):
            found, cc = collect(entry, Path(tmp) / f"probe{i}.o", args.probe)
            tool = Path(cc).name
            records.extend(found)

    if not records:
        print(
            f"CODEGEN_REPORT tool={tool} tus={len(entries)} vectorized=0 missed=0 "
            f"verdict=FAIL"
        )
        print(
            "  the compile produced no loop-vectorize remarks at all. Either the\n"
            "  compiler is not clang, the flags were dropped, or the probe no\n"
            "  longer instantiates anything with a loop in it -- this report\n"
            "  cannot tell you anything, so it does not claim to.",
            file=sys.stderr,
        )
        return 2

    records.sort(key=key)
    today = {key(r): r["vectorized"] for r in records}
    vectorized = sum(1 for v in today.values() if v)
    print(
        f"CODEGEN_REPORT tool={tool} tus={len(entries)} vectorized={vectorized} "
        f"missed={len(today) - vectorized}"
    )
    for k, v in today.items():
        print(f"  {k}: {'vectorized' if v else 'not vectorized'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
