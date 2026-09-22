#!/usr/bin/env python3
"""Report whether Problem::dist_by_ind survives as an out-of-line call inside
the FastPAM SWAP kernels of a linked binary (ledger A-06 / core map Q10).

Disassembles the binary with llvm-objdump or objdump (--disassemble --demangle),
walks the function blocks whose demangled name matches --callers, and counts
call/branch instructions whose text names the callee. Prints

  IPO_INLINING binary=<name> tool=<objdump> ipo=<on|off|unknown> callers=<n> dist_by_ind_calls=<n> symbol_present=<0|1>

Report-only: this script does not decide anything, it produces the marker that
W4 reads. Exit 0 when a disassembler produced inspectable caller blocks, 2 when
no disassembler is available or the binary carries no matching symbols (a
stripped binary cannot be inspected, which is why the CTest registration is
Linux/macOS only).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# llvm-objdump and GNU objdump both emit "<address> <name>:" block headers. The
# name is already demangled because of the flags below, so it carries spaces and
# punctuation; match lazily up to the trailing ">:".
FUNCTION_HEADER = re.compile(r"^[0-9a-fA-F]+\s+<(.+)>:$")
# x86-64 call/jmp, arm64 bl/blr. A tail call is a jmp/b to the callee and still
# means the call was not inlined, so both forms count.
CALL = re.compile(r"\b(call[a-z]*|bl|blr|jmp|b)\b")


def disassemble(binary: Path) -> tuple[str, str]:
    """Return (tool, disassembly text), or ("none", "") if nothing worked."""
    for tool in ("llvm-objdump", "objdump"):
        exe = shutil.which(tool)
        if not exe:
            continue
        if tool == "objdump":
            flags = ["-d", "--no-show-raw-insn", "-C"]
        else:
            flags = ["-d", "--no-show-raw-insn", "--demangle"]
        out = subprocess.run(
            [exe, *flags, str(binary)],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if out.returncode == 0 and out.stdout:
            return tool, out.stdout
    return "none", ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument(
        "--callers",
        default=r"fast_pam|swap|omp_outlined|_omp_fn|nearest_and_second",
        help="regex over demangled function names to inspect",
    )
    # "swap" alone also matches std::__function::__value_func::swap and friends,
    # which have nothing to do with FastPAM. Requiring the project namespace as
    # well keeps the caller count meaningful; pass --scope '' to widen it.
    parser.add_argument(
        "--scope",
        default="dtwc",
        help="additional regex a caller name must also match (default: dtwc)",
    )
    parser.add_argument("--callee", default="dist_by_ind")
    parser.add_argument(
        "--ipo", default="unknown", help="on|off|unknown, copied into the marker"
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="list each caller that still calls the callee, before the marker",
    )
    args = parser.parse_args()

    tool, text = disassemble(args.binary)
    if tool == "none":
        print(
            f"IPO_INLINING binary={args.binary.name} tool=none ipo={args.ipo} "
            f"callers=0 dist_by_ind_calls=0 symbol_present=0"
        )
        return 2

    callers = re.compile(args.callers)
    scope = re.compile(args.scope) if args.scope else None
    current = ""
    in_caller = False
    inspected = 0
    calls = 0
    symbol_present = 0
    per_caller: dict[str, int] = {}

    for line in text.splitlines():
        header = FUNCTION_HEADER.match(line)
        if header:
            current = header.group(1)
            if args.callee in current:
                symbol_present = 1
            in_caller = bool(callers.search(current)) and (
                scope is None or bool(scope.search(current))
            )
            if in_caller:
                inspected += 1
            continue
        if in_caller and args.callee in line and CALL.search(line):
            calls += 1
            per_caller[current] = per_caller.get(current, 0) + 1

    if args.breakdown:
        for name, count in sorted(per_caller.items(), key=lambda kv: -kv[1]):
            # Demangled C++ names run to hundreds of characters; the tail is the
            # argument list, the head is the function actually being named.
            print(f"  {count:3d}  {name[:110]}")

    print(
        f"IPO_INLINING binary={args.binary.name} tool={tool} ipo={args.ipo} "
        f"callers={inspected} dist_by_ind_calls={calls} "
        f"symbol_present={symbol_present}"
    )
    return 0 if inspected > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
