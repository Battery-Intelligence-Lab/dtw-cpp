#!/usr/bin/env python3
"""Fail when CI actions or the optional Arrow source are mutable/unverified."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", re.MULTILINE)
ARROW_ARCHIVE = (
    "https://github.com/apache/arrow/archive/refs/tags/"
    "apache-arrow-19.0.1.tar.gz"
)
ARROW_SHA256 = "4c898504958841cc86b6f8710ecb2919f96b5e10fa8989ac10ac4fca8362d86a"


def mutable_action_references() -> list[str]:
    failures: list[str] = []
    workflows = sorted((ROOT / ".github/workflows").glob("*.y*ml"))
    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        for match in USES.finditer(text):
            spec = match.group(1)
            if spec.startswith("./") or spec.startswith("docker://"):
                continue
            reference = spec.rsplit("@", 1)[-1] if "@" in spec else ""
            if FULL_COMMIT.fullmatch(reference) is None:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{workflow.relative_to(ROOT)}:{line}: {spec}"
                )
    return failures


def arrow_pin_error() -> str | None:
    dependencies = (ROOT / "cmake/Dependencies.cmake").read_text(encoding="utf-8")
    block = re.search(
        rf'URL\s+"{re.escape(ARROW_ARCHIVE)}"(?P<body>.*?)\n\s*OPTIONS',
        dependencies,
        flags=re.DOTALL,
    )
    if block is None:
        return f"Arrow archive URL is missing or changed: {ARROW_ARCHIVE}"
    digest = re.search(r"URL_HASH\s+SHA256=([0-9A-Fa-f]{64})", block.group("body"))
    if digest is None:
        return "Arrow archive is missing URL_HASH SHA256"
    if digest.group(1).lower() != ARROW_SHA256:
        return (
            "Arrow archive SHA256 drift: "
            f"expected {ARROW_SHA256}, found {digest.group(1).lower()}"
        )
    return None


def main() -> int:
    failures = mutable_action_references()
    arrow_error = arrow_pin_error()
    if failures or arrow_error:
        if failures:
            print("mutable GitHub Action references:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
        if arrow_error:
            print(arrow_error, file=sys.stderr)
        return 1
    print("supply-chain pins verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
