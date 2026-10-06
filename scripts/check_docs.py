#!/usr/bin/env python3
"""Three documentation gates that compare what users read with what the code does.

1. Every dtwc_cl flag the user docs show exists in the live `dtwc_cl --help`.
   Scope: docs/content/**/*.md, README.md, .claude/commands/*.md. A flag "the
   docs show" is a -x / --name token in one of three places, and nowhere else:
     a. after a `dtwc_cl` word in a shell command, inside a fenced block or an
        inline code span, up to the next |, &&, ||, ; or # comment;
     b. an inline code span that starts with `--` (a flag named in prose);
     c. an inline code span that starts `-x, --name` (an option-list row).
   Other programs' flags (cmake -D..., ctest -j1, #SBATCH --gres) sit in none
   of these, so prose and foreign commands cannot raise false positives.
2. The other way: every flag the live `--help` prints, short forms and aliases
   included, is named by a table row of the CLI reference, REFERENCE below (the
   page with one row per flag; configuration.md calls it the source of truth).
   A mention in prose or in an example does not stand in for the row. A hidden
   flag (v1.0.0's spellings) is exempt because `--help` does not print it, not
   because a list says so. `dtwc_cl` has no subcommands; if its help ever lists
   some, this fails, because their flags are in their own help, which is not read.
3. The gates still bite: cmake/DtwcTest.cmake fails a test that skips without
   MAY_SKIP, and gate 2's reader names a flag that has no table row.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/content/getting-started/cli.md"
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


def documented_flags(lines: list[str]) -> list[tuple[int, str]]:
    """(line, flag) for every dtwc_cl flag the lines of a page show."""
    found: list[tuple[int, str]] = []
    in_block, pending, start = False, "", 0
    for number, line in enumerate(lines, 1):
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


def undocumented(live: set[str], page: list[str]) -> list[str]:
    """The live flags that no table row of the page names (gate 2)."""
    rows = {n for n, line in enumerate(page, 1) if line.lstrip().startswith("|")}
    return sorted(live - {flag for n, flag in documented_flags(page) if n in rows})


def check_cli_flags(binary: Path) -> list[str]:
    help_text = subprocess.run([str(binary), "--help"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=True).stdout
    live = set(FLAG.findall(help_text))
    pages = [*sorted((ROOT / "docs/content").rglob("*.md")), ROOT / "README.md",
             *sorted((ROOT / ".claude/commands").glob("*.md"))]
    errors, count = [], 0
    for page in pages:
        for line, flag in documented_flags(page.read_text(encoding="utf-8").splitlines()):
            count += 1
            if flag not in live:
                errors.append(f"{page.relative_to(ROOT).as_posix()}:{line}: "
                              f"{flag} is not in `dtwc_cl --help`")
    reference = REFERENCE.relative_to(ROOT).as_posix()
    missing = undocumented(live, REFERENCE.read_text(encoding="utf-8").splitlines())
    errors += [f"{reference}: {flag} is in `dtwc_cl --help` "
               f"but no table row of this page names it" for flag in missing]
    if "SUBCOMMANDS:" in help_text:
        errors.append("`dtwc_cl --help` lists SUBCOMMANDS: this gate reads the top-level help "
                      "only, so their flags are in neither direction's check")
    print(f"DOCS flags checked={count} pages={len(pages)} live={len(live)} "
          f"undocumented={len(missing)}")
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


def check_reverse_bites() -> list[str]:
    """Gate 2 on a made-up page: a flag with a table row passes, and one that only prose
    and an example name is reported."""
    row = ["| `-a, --alpha <int>` | the flag in its row | 1 |"]
    prose = ["Prose names `--beta`, and so does an example:", "```", "dtwc_cl --beta", "```"]
    got = (undocumented({"-a", "--alpha"}, row),
           undocumented({"-a", "--alpha", "--beta"}, row + prose))
    if got == ([], ["--beta"]):
        return []
    return [f"scripts/check_docs.py reads a table row wrongly: {got}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", type=Path, required=True, help="built dtwc_cl binary")
    args = parser.parse_args()
    errors = check_cli_flags(args.cli.resolve()) + check_reverse_bites() + check_harness()
    for error in errors:
        print(error, file=sys.stderr)
    print("VERDICT=" + ("FAIL" if errors else "PASS"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
