#!/usr/bin/env python3
"""Typed-throw ratchet for dtwc/ (ledger X-16).

The 2.0 contract is that a request which cannot be honoured raises a *typed*
dtwc error, not a bare standard-library exception, because callers across three
language bindings have to tell failures apart. A great many `throw std::...`
sites predate that rule. Rewriting them all at once is not on; letting the count
grow back is what this stops.

The rule is one-directional: the number of `throw std::` sites in dtwc/ may fall,
never rise. Lower it by converting sites to typed errors, then lower CEILING to
match in the same commit.

Counting is deliberately literal so the number is reproducible: comments and
string literals are stripped first, so a `throw std::` written in prose or inside
an error message does not count. Vendored code under dtwc/extern/ is not ours and
is skipped.

Prints

  TYPED_THROW_RATCHET files=<n> untyped=<n> ceiling=<n> verdict=<PASS|FAIL>

Exit 0 when at or under the ceiling, 1 when over, 2 when the tree could not be
scanned at all.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "dtwc"
SUFFIXES = {".hpp", ".cpp", ".h", ".cc", ".cu", ".cuh", ".mm"}
SKIP_DIRS = {"extern"}  # vendored: nanoarrow's amalgamation is not ours to police

# Measured 2026-09-22 on design-2.0: 259 sites across 40 files, after stripping
# comments and strings (a raw grep says 260; one of those is inside a comment).
# The ledger row said 256, which no longer matched the tree -- the point of
# recording the number with a script is that it stops being folklore.
# This may only ever be lowered.
CEILING = 259

THROW = re.compile(r"\bthrow\s+std::")


# A character literal is short: 'x', '\n', '\\', '\0', u8'a'. Anything longer is
# not one, and treating a stray apostrophe as an opening quote is how a stripper
# silently eats the rest of a file -- see the note in strip_comments_and_strings.
CHAR_LITERAL = re.compile(r"'(\\.|[^\\'])'")
RAW_STRING_OPEN = re.compile(r'R"([^(\s\\]{0,16})\(')


def _blank(chunk: str) -> str:
    """Same length, same line breaks, no content."""
    return "".join("\n" if ch == "\n" else " " for ch in chunk)


def strip_comments_and_strings(text: str) -> str:
    """Blank out //, /* */, "...", raw strings and char literals.

    Replaces removed spans with spaces, keeping line numbers intact.

    Two cases matter here and both were found by disagreeing with grep rather
    than by reasoning:

    * **Raw strings.** dtwc/metal/metal_dtw.mm embeds its Metal shader as
      R"(...)". A naive scanner does not know the delimiter form and loses track
      of where the string ends.
    * **Apostrophes.** That shader's own comments say "thread's" and "row's".
      Treated as opening quotes they swallow everything to the next apostrophe,
      which silently hid all 18 throw sites in that file. So a quote only opens a
      character literal when what follows actually looks like one.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if c == "/" and nxt == "/":
            start = i
            while i < n and text[i] != "\n":
                i += 1
            out.append(_blank(text[start:i]))
            continue

        if c == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(_blank(text[i:end]))
            i = end
            continue

        raw = RAW_STRING_OPEN.match(text, i)
        if raw:
            closer = f'){raw.group(1)}"'
            end = text.find(closer, raw.end())
            end = n if end == -1 else end + len(closer)
            out.append(_blank(text[i:end]))
            i = end
            continue

        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(_blank(text[i:j]))
            i = j
            continue

        if c == "'":
            m = CHAR_LITERAL.match(text, i)
            if m:
                out.append(_blank(m.group(0)))
                i = m.end()
                continue
            # Not a character literal: a digit separator (1'000) or prose inside
            # something we are not otherwise stripping. Keep it and move on.
            out.append(c)
            i += 1
            continue

        out.append(c)
        i += 1
    return "".join(out)


def sources(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix not in SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--breakdown", action="store_true", help="list per-file counts")
    ap.add_argument(
        "--ceiling", type=int, default=CEILING, help="override the recorded ceiling"
    )
    args = ap.parse_args()

    if not SOURCE_DIR.is_dir():
        print("TYPED_THROW_RATCHET files=0 untyped=0 ceiling=0 verdict=FAIL")
        print(f"  no source tree at {SOURCE_DIR}", file=sys.stderr)
        return 2

    per_file: dict[str, int] = {}
    scanned = 0
    for path in sources(SOURCE_DIR):
        scanned += 1
        code = strip_comments_and_strings(
            path.read_text(encoding="utf-8", errors="replace")
        )
        hits = len(THROW.findall(code))
        if hits:
            per_file[str(path.relative_to(ROOT))] = hits

    if scanned == 0:
        print("TYPED_THROW_RATCHET files=0 untyped=0 ceiling=0 verdict=FAIL")
        print(f"  {SOURCE_DIR} held no source files -- nothing was checked", file=sys.stderr)
        return 2

    total = sum(per_file.values())
    verdict = "PASS" if total <= args.ceiling else "FAIL"

    if args.breakdown:
        for name, count in sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {count:4d}  {name}")

    print(
        f"TYPED_THROW_RATCHET files={scanned} untyped={total} "
        f"ceiling={args.ceiling} verdict={verdict}"
    )
    if verdict == "FAIL":
        print(
            f"  {total - args.ceiling} new `throw std::` site(s) in dtwc/.\n"
            f"  2.0 raises typed dtwc errors so callers in C++, Python and MATLAB can\n"
            f"  tell failures apart. Convert the new site(s), or -- if you genuinely\n"
            f"  lowered the count -- lower CEILING in this script to match.\n"
            f"  Run with --breakdown to see where they are.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
