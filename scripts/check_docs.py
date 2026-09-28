#!/usr/bin/env python3
"""Two documentation gates that compare what users read with what the code does.

1. Every dtwc_cl flag the user docs show exists in the live `dtwc_cl --help`.
   Scope: docs/content/**/*.md, README.md, .claude/commands/*.md. A flag "the
   docs show" is a -x / --name token in one of three places, and nowhere else:
     a. after a `dtwc_cl` word in a shell command, inside a fenced block or an
        inline code span, up to the next |, &&, ||, ; or # comment;
     b. an inline code span that starts with `--` (a flag named in prose);
     c. an inline code span that starts `-x, --name` (an option-list row).
   Other programs' flags (cmake -D..., ctest -j1, #SBATCH --gres) sit in none
   of these, so prose and foreign commands cannot raise false positives.
2. cmake/DtwcTest.cmake still fails a test that skips without MAY_SKIP.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FLAG = re.compile(r"(?<![\w-])--?[A-Za-z][\w-]*")
DTWC_CL = re.compile(r"(?:^|[\s/\\\"'=])dtwc_cl(?:\.exe)?(?=[\s\"',\]]|$)")
SEGMENT_END = re.compile(r"\|\||&&|[|;]|(?:^|\s)#")
SPAN = re.compile(r"`([^`\n]+)`")
BARE_FLAGS = re.compile(r"--[A-Za-z]|-[A-Za-z], --[A-Za-z]")
FENCE = re.compile(r"^\s*(```|~~~)")


def command_flags(command: str) -> list[str]:
    """Flags passed to dtwc_cl in one logical shell line (rule a)."""
    found = []
    for match in DTWC_CL.finditer(command):
        args = command[match.end():]
        end = SEGMENT_END.search(args)
        found += FLAG.findall(args[:end.start()] if end else args)
    return found


def span_flags(span: str) -> list[str]:
    if BARE_FLAGS.match(span):  # rules b and c
        return FLAG.findall(span)
    return command_flags(span)


def documented_flags(path: Path) -> list[tuple[int, str]]:
    """(line, flag) for every dtwc_cl flag the page shows."""
    found: list[tuple[int, str]] = []
    in_block, pending, start = False, "", 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            in_block, pending = not in_block, ""
            continue
        if not in_block:
            found += [(number, f) for s in SPAN.findall(line) for f in span_flags(s)]
            continue
        # Join continuation lines: POSIX \, PowerShell `, cmd ^.
        if not pending:
            start = number
        stripped = line.rstrip()
        if stripped.endswith(("\\", "`", "^")):
            pending += stripped[:-1] + " "
            continue
        found += [(start, f) for f in command_flags(pending + line)]
        pending = ""
    return found


def check_cli_flags(binary: Path) -> list[str]:
    help_text = subprocess.run([str(binary), "--help"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=True).stdout
    live = set(FLAG.findall(help_text))
    pages = [*sorted((ROOT / "docs/content").rglob("*.md")), ROOT / "README.md",
             *sorted((ROOT / ".claude/commands").glob("*.md"))]
    errors, count = [], 0
    for page in pages:
        for line, flag in documented_flags(page):
            count += 1
            if flag not in live:
                errors.append(f"{page.relative_to(ROOT).as_posix()}:{line}: "
                              f"{flag} is not in `dtwc_cl --help`")
    print(f"DOCS flags checked={count} pages={len(pages)} live={len(live)}")
    return errors


def check_harness() -> list[str]:
    harness = " ".join((ROOT / "cmake/DtwcTest.cmake").read_text(encoding="utf-8").split())
    required = (
        # A skip line fails a non-MAY_SKIP test, and the pass regex is the
        # subject's marker then Catch2's summary, which Catch2 prints only when
        # no case failed or skipped.
        'FAIL_REGULAR_EXPRESSION "${DTWC_TEST_SKIP_REGEX}" PASS_REGULAR_EXPRESSION "${_pass}"',
        'set(_pass "${ARG_MARKER}(.|[\\r\\n])*${_summary}")',
        "All tests passed \\\\(",
    )
    return [f"cmake/DtwcTest.cmake no longer fails a non-MAY_SKIP test that skips: "
            f"missing {text!r}" for text in required if text not in harness]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", type=Path, required=True, help="built dtwc_cl binary")
    args = parser.parse_args()
    errors = check_cli_flags(args.cli.resolve()) + check_harness()
    for error in errors:
        print(error, file=sys.stderr)
    print("VERDICT=" + ("FAIL" if errors else "PASS"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
