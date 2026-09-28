#!/usr/bin/env python3
"""Every third-party download is pinned by content: SHA-256 or a 40-hex commit.

CMake: each CPMAddPackage / FetchContent_Declare / ExternalProject_Add / file(DOWNLOAD)
in a tracked CMake file carries URL_HASH (or EXPECTED_HASH) SHA256=<64 hex> or
GIT_TAG <40 hex>. A ${VAR} counts only when VAR is set once, to a literal, in a
tracked CMake file; any other ${...} inside CPMAddPackage is refused, because a
pin nobody can read statically is not a pin. Workflows: every `uses:` is @<40 hex>.
REVIEWED is a second key: bumping Arrow means editing this file as well.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = {"Arrow": ("https://github.com/apache/arrow/archive/refs/tags/apache-arrow-19.0.1.tar.gz",
                      "4c898504958841cc86b6f8710ecb2919f96b5e10fa8989ac10ac4fca8362d86a")}
CALL = re.compile(r"\b(CPMAddPackage|FetchContent_Declare|ExternalProject_Add|file(?=\(\s*DOWNLOAD\b))\s*\(", re.I)
PIN = re.compile(r"(URL_HASH|EXPECTED_HASH)\s+SHA256=[0-9A-Fa-f]{64}\b|\bGIT_TAG\s+[0-9a-f]{40}\b")
SET = re.compile(r"^\s*set\(\s*(\w+)\s+\"?([^\s\"$()]+)\"?\s*\)", re.M)


def calls(text):
    text = re.sub(r"#[^\n]*", "", text)  # comments; no dependency argument contains '#'
    for match in CALL.finditer(text):
        depth, end = 1, match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(text[end], 0)
            end += 1
        yield text.count("\n", 0, match.start()) + 1, match[1], text[match.end():end - 1]


def literals(texts):
    """Variables set exactly once, to one literal value."""
    values = {}
    for text in texts:
        for var, value in SET.findall(text):
            values.setdefault(var, []).append(value)
    return {var: vals[0] for var, vals in values.items() if len(vals) == 1}


files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*CMakeLists.txt", "*.cmake"],
                       capture_output=True, text=True, check=True).stdout.splitlines()
texts = {name: (ROOT / name).read_text(encoding="utf-8") for name in files}
repo_literals = literals(texts.values())

failures, total, named = [], 0, {}
for name, text in texts.items():
    literal = {**repo_literals, **literals([text])}  # a file's own value wins
    for line, command, args in calls(text):
        total += 1
        args = re.sub(r"\$\{(\w+)\}", lambda m: literal.get(m[1], m[0]), args)
        where = f"{name}:{line}: {command}"
        if command == "CPMAddPackage" and "${" in args:
            failures.append(f"{where} has a ${{...}} that is not a literal set once in this repo")
        if not PIN.search(args):
            failures.append(f"{where} pins neither SHA256=<64 hex> nor GIT_TAG <40 hex>")
        package = re.search(r"\bNAME\s+(\S+)", args)
        if package and package[1] in REVIEWED:
            named.setdefault(package[1], []).append(args)
for package, (url, digest) in REVIEWED.items():
    blocks = named.get(package, [])
    if len(blocks) != 1 or url not in blocks[0] or f"SHA256={digest}" not in blocks[0]:
        failures.append(f"{package}: expected one block with the reviewed {url} and SHA256={digest}")

actions = [(wf.name, ref) for wf in sorted((ROOT / ".github/workflows").glob("*.y*ml"))
           for ref in re.findall(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", wf.read_text(encoding="utf-8"), re.M)]
failures += [f".github/workflows/{wf}: uses {ref} is not pinned to a 40-hex commit"
             for wf, ref in actions if not ref.startswith("./") and not re.search(r"@[0-9a-f]{40}$", ref)]

print(*failures, sep="\n", file=sys.stderr)
print(f"PINS cmake={total} actions={len(actions)} failures={len(failures)}")
sys.exit(1 if failures else 0)
