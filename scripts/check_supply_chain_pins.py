#!/usr/bin/env python3
"""Verify tracked CMake archive, workflow-action, and exact dependency pins."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", re.MULTILINE)
SHA256 = re.compile(r"[0-9A-Fa-f]{64}\Z")
BRACKET_OPEN = re.compile(r"\[(?P<equals>=*)\[")
VERSION_REF = re.compile(r"v?\d+(?:[._-][0-9A-Za-z]+)*\Z", re.IGNORECASE)
REGISTERED_CMAKE_MANIFEST_TOTAL = 28  # fb853eb added tests/integration/test_cli_resume_state.cmake (2026-07-24)
CPM_PARSE_KEYWORDS = {
    "BITBUCKET_REPOSITORY",
    "CUSTOM_CACHE_KEY",
    "DOWNLOAD_COMMAND",
    "DOWNLOAD_NAME",
    "DOWNLOAD_ONLY",
    "EXCLUDE_FROM_ALL",
    "FIND_PACKAGE_ARGUMENTS",
    "FORCE",
    "GITHUB_REPOSITORY",
    "GITLAB_REPOSITORY",
    "GIT_REPOSITORY",
    "GIT_SHALLOW",
    "GIT_TAG",
    "NAME",
    "NO_CACHE",
    "OPTIONS",
    "PATCHES",
    "SOURCE_DIR",
    "SOURCE_SUBDIR",
    "SVN_REPOSITORY",
    "SYSTEM",
    "URL",
    "VERSION",
}
CPM_URL_ALTERNATE_SOURCE_DIRECTIVES = {
    "BITBUCKET_REPOSITORY",
    "CVS_REPOSITORY",
    "DOWNLOAD_COMMAND",
    "FIND_PACKAGE_ARGUMENTS",
    "GITHUB_REPOSITORY",
    "GITLAB_REPOSITORY",
    "GIT_REPOSITORY",
    "HG_REPOSITORY",
    "SOURCE_DIR",
    "SVN_REPOSITORY",
}

ARROW_ARCHIVE = (
    "https://github.com/apache/arrow/archive/refs/tags/"
    "apache-arrow-19.0.1.tar.gz"
)
ARROW_SHA256 = "4c898504958841cc86b6f8710ecb2919f96b5e10fa8989ac10ac4fca8362d86a"
EXAMPLE_MANIFEST = Path("examples/cpp/example_project/CMakeLists.txt")
EXAMPLE_ARCHIVE = (
    "https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/"
    "eda1b92bc89ee51568b052a6af86f615d336de3c.zip"
)
EXAMPLE_SHA256 = "d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696"
REGISTERED_ARCHIVE_IDENTITIES = (
    (
        "cmake/Dependencies.cmake",
        "Catch2",
        "https://github.com/catchorg/Catch2/archive/refs/tags/v3.13.0.tar.gz",
        "650795f6501af514f806e78c554729847b98db6935e69076f36bb03ed2e985ef",
    ),
    (
        "cmake/Dependencies.cmake",
        "highs",
        "https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.15.1.tar.gz",
        "a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16",
    ),
    (
        "cmake/Dependencies.cmake",
        "CLI11",
        "https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.2.tar.gz",
        "c6ea6b2e5608b3ea8617999bd5f47420c71b2ebdb8dc4767c1034d1da5785711",
    ),
    (
        "cmake/Dependencies.cmake",
        "Eigen",
        "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.bz2",
        "e4de6b08f33fd8b8985d2f204381408c660bffa6170ac65b68ae1bd3cd575c0a",
    ),
    (
        "cmake/Dependencies.cmake",
        "yaml-cpp",
        "https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz",
        "25cb043240f828a8c51beb830569634bc7ac603978e0f69d6b63558dadefd49a",
    ),
    (
        "cmake/Dependencies.cmake",
        "Arrow",
        ARROW_ARCHIVE,
        ARROW_SHA256,
    ),
    (
        EXAMPLE_MANIFEST.as_posix(),
        "dtw-cpp",
        EXAMPLE_ARCHIVE,
        EXAMPLE_SHA256,
    ),
)
REGISTERED_CMAKE_ARCHIVE_TOTAL = len(REGISTERED_ARCHIVE_IDENTITIES)


@dataclass(frozen=True)
class ArchivePin:
    path: Path
    line: int
    name: str | None
    url: str
    digest: str | None

    @property
    def literal_remote(self) -> bool:
        return (
            re.fullmatch(r"https?://[^\s;$]+", self.url, re.IGNORECASE)
            is not None
        )

    @property
    def mutable(self) -> bool:
        if not self.literal_remote:
            return False

        parsed = urlsplit(self.url)
        host = (parsed.hostname or "").lower()
        path = unquote(parsed.path)
        lower_path = path.lower()
        if "/refs/heads/" in lower_path:
            return True

        if host in {"github.com", "www.github.com"}:
            match = re.fullmatch(
                r"/[^/]+/[^/]+/archive/(?P<ref>.+)"
                r"\.(?:zip|tar\.gz|tar\.bz2)",
                path,
                re.IGNORECASE,
            )
            if match is not None:
                reference = match.group("ref")
                return (
                    not reference.lower().startswith("refs/tags/")
                    and FULL_COMMIT.fullmatch(reference.lower()) is None
                )

        if host == "codeload.github.com":
            match = re.fullmatch(
                r"/[^/]+/[^/]+/(?:zip|tar\.gz)/(?P<ref>.+)",
                path,
                re.IGNORECASE,
            )
            if match is not None:
                reference = match.group("ref")
                return (
                    not reference.lower().startswith("refs/tags/")
                    and FULL_COMMIT.fullmatch(reference.lower()) is None
                )

        if host in {"api.github.com", "github.com", "www.github.com"}:
            match = re.fullmatch(
                r"(?:/repos)?/[^/]+/[^/]+/(?:zipball|tarball)"
                r"(?:/(?P<ref>.*))?",
                path,
                re.IGNORECASE,
            )
            if match is not None:
                reference = match.group("ref")
                return (
                    reference is None
                    or FULL_COMMIT.fullmatch(reference.lower()) is None
                )

        if host == "gitlab.com":
            match = re.fullmatch(
                r"/.+/-/archive/(?P<ref>[^/]+)/.+",
                path,
                re.IGNORECASE,
            )
            if match is not None:
                reference = match.group("ref")
                return (
                    FULL_COMMIT.fullmatch(reference.lower()) is None
                    and VERSION_REF.fullmatch(reference) is None
                )
            if re.fullmatch(
                r"/api/v4/projects/.+/repository/archive"
                r"(?:\.(?:zip|tar|tar\.gz|tar\.bz2))?",
                path,
                re.IGNORECASE,
            ):
                references = parse_qs(parsed.query).get("sha", [])
                return (
                    len(references) != 1
                    or FULL_COMMIT.fullmatch(references[0].lower()) is None
                )

        return False

    @property
    def hashed(self) -> bool:
        return self.digest is not None and SHA256.fullmatch(self.digest) is not None

    @property
    def verified(self) -> bool:
        return self.literal_remote and self.hashed and not self.mutable


@dataclass(frozen=True)
class _CMakeToken:
    value: str
    start: int
    kind: str = "argument"


def _has_cmake_expansion(value: str) -> bool:
    return re.search(r"\$(?:\{|ENV\{|CACHE\{|<)", value, re.IGNORECASE) is not None


def workflow_action_references_in_text(path: Path, text: str) -> tuple[list[str], int]:
    failures: list[str] = []
    total = 0
    for match in USES.finditer(text):
        total += 1
        spec = match.group(1)
        if spec.startswith("./") or spec.startswith("docker://"):
            continue
        reference = spec.rsplit("@", 1)[-1] if "@" in spec else ""
        if FULL_COMMIT.fullmatch(reference) is None:
            line = text.count("\n", 0, match.start()) + 1
            failures.append(f"{path.as_posix()}:{line}: {spec}")
    return failures, total


def workflow_action_pin_results(root: Path = ROOT) -> tuple[list[str], int]:
    failures: list[str] = []
    total = 0
    workflows = sorted((root / ".github/workflows").glob("*.y*ml"))
    for workflow in workflows:
        relative = workflow.relative_to(root)
        file_failures, file_total = workflow_action_references_in_text(
            relative, workflow.read_text(encoding="utf-8")
        )
        failures.extend(file_failures)
        total += file_total
    return failures, total


def mutable_action_references(root: Path = ROOT) -> list[str]:
    """Compatibility wrapper for the original checker seam."""
    return workflow_action_pin_results(root)[0]


def tracked_cmake_files(root: Path = ROOT) -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--",
            "*CMakeLists.txt",
            "*.cmake",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths = [
        Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    ]
    return sorted(paths, key=lambda path: path.as_posix())


def _cmake_tokens(text: str) -> list[_CMakeToken]:
    """Tokenize the CMake syntax needed to inspect command arguments."""
    tokens: list[_CMakeToken] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue

        if char == "#":
            bracket = BRACKET_OPEN.match(text, index + 1)
            if bracket is None:
                newline = text.find("\n", index + 1)
                index = len(text) if newline < 0 else newline + 1
                continue
            closing = f"]{bracket.group('equals')}]"
            end = text.find(closing, bracket.end())
            if end < 0:
                line = text.count("\n", 0, index) + 1
                raise ValueError(f"unterminated CMake bracket comment at line {line}")
            index = end + len(closing)
            continue

        if char in "()":
            tokens.append(
                _CMakeToken(
                    value=char,
                    start=index,
                    kind="left_paren" if char == "(" else "right_paren",
                )
            )
            index += 1
            continue

        if char == '"':
            start = index
            index += 1
            value: list[str] = []
            while index < len(text):
                if text[index] == '"':
                    index += 1
                    break
                if text[index] == "\\" and index + 1 < len(text):
                    value.append(text[index + 1])
                    index += 2
                    continue
                value.append(text[index])
                index += 1
            else:
                line = text.count("\n", 0, start) + 1
                raise ValueError(f"unterminated CMake quoted argument at line {line}")
            tokens.append(_CMakeToken(value="".join(value), start=start))
            continue

        bracket = BRACKET_OPEN.match(text, index)
        if bracket is not None:
            start = index
            closing = f"]{bracket.group('equals')}]"
            end = text.find(closing, bracket.end())
            if end < 0:
                line = text.count("\n", 0, start) + 1
                raise ValueError(f"unterminated CMake bracket argument at line {line}")
            tokens.append(
                _CMakeToken(value=text[bracket.end() : end], start=start)
            )
            index = end + len(closing)
            continue

        start = index
        value = []
        while index < len(text):
            char = text[index]
            if char.isspace() or char in "()#":
                break
            if char == "\\" and index + 1 < len(text):
                value.append(text[index + 1])
                index += 2
                continue
            value.append(char)
            index += 1
        if not value:
            line = text.count("\n", 0, start) + 1
            raise ValueError(
                f"unsupported CMake token {text[start]!r} at line {line}"
            )
        tokens.append(_CMakeToken(value="".join(value), start=start))
    return tokens


def _cmake_language_cpm_arguments(
    arguments: list[_CMakeToken],
    text: str,
    depth: int = 0,
) -> Iterable[list[_CMakeToken]]:
    if not arguments:
        return
    semicolon = next(
        (token for token in arguments if ";" in token.value),
        None,
    )
    if semicolon is not None:
        line = text.count("\n", 0, semicolon.start) + 1
        raise ValueError(
            f"semicolon-expanded cmake_language arguments at line {line} "
            "are not auditable"
        )
    if depth > 8:
        line = text.count("\n", 0, arguments[0].start) + 1
        raise ValueError(
            f"nested cmake_language calls exceed the audit limit at line {line}"
        )
    if _has_cmake_expansion(arguments[0].value):
        line = text.count("\n", 0, arguments[0].start) + 1
        raise ValueError(
            f"dynamic cmake_language operation at line {line} cannot be classified"
        )

    operation = arguments[0].value.upper()
    call_index: int | None = None
    if operation == "CALL":
        call_index = 0
    elif operation == "DEFER":
        cursor = 1
        defer_options = {"DIRECTORY", "ID", "ID_VAR"}
        defer_queries = {"GET_CALL_IDS", "GET_CALL", "CANCEL_CALL"}
        while cursor < len(arguments):
            token = arguments[cursor]
            line = text.count("\n", 0, token.start) + 1
            if _has_cmake_expansion(token.value):
                raise ValueError(
                    f"dynamic cmake_language(DEFER ...) at line {line} "
                    "cannot be classified"
                )
            keyword = token.value.upper()
            if keyword in defer_options:
                if cursor + 1 >= len(arguments):
                    raise ValueError(
                        f"cmake_language(DEFER {keyword}) at line {line} "
                        "is missing its value"
                    )
                operand = arguments[cursor + 1]
                if _has_cmake_expansion(operand.value):
                    operand_line = text.count("\n", 0, operand.start) + 1
                    raise ValueError(
                        "dynamic cmake_language(DEFER ...) at line "
                        f"{operand_line} cannot be classified"
                    )
                cursor += 2
                continue
            if keyword == "CALL":
                call_index = cursor
                break
            if keyword in defer_queries:
                return
            raise ValueError(
                f"unsupported cmake_language(DEFER) option "
                f"{token.value!r} at line {line}"
            )
    elif operation == "EVAL":
        line = text.count("\n", 0, arguments[0].start) + 1
        raise ValueError(
            f"cmake_language(EVAL ...) at line {line} is not "
            "auditable by the archive pin gate"
        )

    if call_index is None or call_index + 1 >= len(arguments):
        return
    called_command = arguments[call_index + 1].value
    if _has_cmake_expansion(called_command):
        line = text.count("\n", 0, arguments[call_index + 1].start) + 1
        raise ValueError(
            f"dynamic cmake_language({operation} ... CALL ...) command "
            f"at line {line} cannot be classified"
        )
    called_arguments = arguments[call_index + 2 :]
    if called_command.lower() == "cpmaddpackage":
        yield called_arguments
    elif called_command.lower() == "cmake_language":
        yield from _cmake_language_cpm_arguments(
            called_arguments,
            text,
            depth + 1,
        )


def _cpm_call_arguments(text: str) -> Iterable[list[_CMakeToken]]:
    tokens = _cmake_tokens(text)
    index = 0
    while index + 1 < len(tokens):
        if (
            tokens[index].kind == "argument"
            and tokens[index + 1].kind == "left_paren"
        ):
            command = tokens[index].value.lower()
            depth = 1
            end = index + 2
            while end < len(tokens) and depth:
                if tokens[end].kind == "left_paren":
                    depth += 1
                elif tokens[end].kind == "right_paren":
                    depth -= 1
                end += 1
            if depth:
                line = text.count("\n", 0, tokens[index].start) + 1
                raise ValueError(
                    f"unterminated {tokens[index].value} at line {line}"
                )
            arguments = tokens[index + 2 : end - 1]
            if command == "cpmaddpackage":
                yield arguments
            elif command == "cmake_language" and arguments:
                yield from _cmake_language_cpm_arguments(arguments, text)
            index = end
            continue
        index += 1


def cmake_archive_pins_in_text(path: Path, text: str) -> list[ArchivePin]:
    pins: list[ArchivePin] = []
    for arguments in _cpm_call_arguments(text):
        semicolon = next(
            (token for token in arguments if ";" in token.value),
            None,
        )
        if semicolon is not None:
            line = text.count("\n", 0, semicolon.start) + 1
            raise ValueError(
                f"{path.as_posix()}:{line}: semicolon-expanded "
                "CPMAddPackage arguments are not auditable"
            )

        url_indices = [
            index
            for index, token in enumerate(arguments)
            if token.kind == "argument" and token.value.upper() == "URL"
        ]
        if url_indices:
            alternate_source = next(
                (
                    token
                    for token in arguments
                    if token.value.upper()
                    in CPM_URL_ALTERNATE_SOURCE_DIRECTIVES
                ),
                None,
            )
            if alternate_source is not None:
                line = text.count("\n", 0, alternate_source.start) + 1
                raise ValueError(
                    f"{path.as_posix()}:{line}: URL archive contains "
                    "alternate source directive "
                    f"{alternate_source.value.upper()}"
                )

        dynamic_argument = next(
            (
                token
                for token in arguments
                if _has_cmake_expansion(token.value)
            ),
            None,
        )
        if dynamic_argument is not None:
            line = text.count("\n", 0, dynamic_argument.start) + 1
            raise ValueError(
                f"{path.as_posix()}:{line}: dynamic CPMAddPackage "
                "arguments cannot be classified"
            )

        if not url_indices:
            continue

        if len(url_indices) != 1:
            line = text.count("\n", 0, arguments[url_indices[1]].start) + 1
            raise ValueError(
                f"{path.as_posix()}:{line}: each CPMAddPackage call must "
                "contain exactly one URL directive"
            )

        url_index = url_indices[0]
        name_indices = [
            index
            for index, token in enumerate(arguments)
            if token.kind == "argument" and token.value.upper() == "NAME"
        ]
        if len(name_indices) > 1:
            line = text.count("\n", 0, arguments[name_indices[1]].start) + 1
            raise ValueError(
                f"{path.as_posix()}:{line}: each CPMAddPackage call must "
                "contain at most one NAME directive"
            )
        name = None
        if name_indices:
            name_index = name_indices[0]
            if (
                name_index + 1 >= len(arguments)
                or arguments[name_index + 1].value.upper()
                in CPM_PARSE_KEYWORDS
            ):
                line = text.count("\n", 0, arguments[name_index].start) + 1
                raise ValueError(
                    f"{path.as_posix()}:{line}: CPMAddPackage NAME is "
                    "missing its value"
                )
            name = arguments[name_index + 1].value

        segment_end = len(arguments)
        for index in range(url_index + 1, len(arguments)):
            if arguments[index].value.upper() in CPM_PARSE_KEYWORDS:
                segment_end = index
                break
        url_segment = arguments[url_index + 1 : segment_end]

        digest = None
        if len(url_segment) > 1:
            integrity_keyword = url_segment[1].value.upper()
            if integrity_keyword not in {"URL_HASH", "URL_MD5"}:
                line = text.count("\n", 0, url_segment[1].start) + 1
                raise ValueError(
                    f"{path.as_posix()}:{line}: multiple URL values in one "
                    "CPMAddPackage call are not auditable"
                )
            if len(url_segment) != 3:
                line = text.count("\n", 0, url_segment[1].start) + 1
                raise ValueError(
                    f"{path.as_posix()}:{line}: malformed {integrity_keyword} "
                    "arguments in CPMAddPackage URL list"
                )
            if (
                integrity_keyword == "URL_HASH"
                and url_segment[2].value.upper().startswith("SHA256=")
            ):
                digest = url_segment[2].value.split("=", 1)[1]

        token = arguments[url_index]
        url = "<missing>"
        if url_segment:
            url = url_segment[0].value
        line = text.count("\n", 0, token.start) + 1
        pins.append(
            ArchivePin(
                path=path,
                line=line,
                name=name,
                url=url,
                digest=digest,
            )
        )
    return pins


def tracked_cmake_archive_pins(
    root: Path = ROOT,
) -> tuple[list[ArchivePin], int]:
    manifests = tracked_cmake_files(root)
    pins: list[ArchivePin] = []
    for relative in manifests:
        pins.extend(
            cmake_archive_pins_in_text(
                relative, (root / relative).read_text(encoding="utf-8")
            )
        )
    return pins, len(manifests)


def archive_pin_failures(pins: Iterable[ArchivePin]) -> list[str]:
    failures: list[str] = []
    for pin in pins:
        prefix = f"{pin.path.as_posix()}:{pin.line}"
        if not pin.literal_remote:
            failures.append(
                f"{prefix}: URL must be one literal http(s) archive: {pin.url}"
            )
        if pin.mutable:
            failures.append(f"{prefix}: mutable remote archive URL: {pin.url}")
        if not pin.hashed:
            failures.append(
                f"{prefix}: missing or invalid URL_HASH SHA256=<64 hex>: {pin.url}"
            )
    return failures


def _archive_identity(
    pin: ArchivePin,
) -> tuple[str, str | None, str, str | None]:
    digest = pin.digest.lower() if pin.digest is not None else None
    return pin.path.as_posix(), pin.name, pin.url, digest


def _format_archive_identity(
    identity: tuple[str, str | None, str, str | None],
) -> str:
    path, name, url, digest = identity
    return f"{path}|{name or '<missing>'}|{url}|{digest or '<missing>'}"


def registered_archive_inventory_failures(
    pins: Iterable[ArchivePin],
    path: Path | None = None,
) -> list[str]:
    path_text = path.as_posix() if path is not None else None
    expected = Counter(
        identity
        for identity in REGISTERED_ARCHIVE_IDENTITIES
        if path_text is None or identity[0] == path_text
    )
    observed = Counter(
        _archive_identity(pin)
        for pin in pins
        if path is None or pin.path == path
    )
    failures: list[str] = []
    for identity, count in sorted(
        (expected - observed).items(),
        key=lambda item: _format_archive_identity(item[0]),
    ):
        failures.extend(
            [
                "missing registered CMake archive identity: "
                f"{_format_archive_identity(identity)}"
            ]
            * count
        )
    for identity, count in sorted(
        (observed - expected).items(),
        key=lambda item: _format_archive_identity(item[0]),
    ):
        failures.extend(
            [
                "unexpected tracked CMake archive identity: "
                f"{_format_archive_identity(identity)}"
            ]
            * count
        )
    return failures


def _exact_pin_error(
    pins: Iterable[ArchivePin],
    *,
    path: Path,
    expected_url: str,
    expected_digest: str,
    expected_name: str,
    label: str,
) -> str | None:
    pin_list = list(pins)
    candidates = [
        pin
        for pin in pin_list
        if pin.path == path and pin.name == expected_name
    ]
    if len(candidates) != 1:
        if len(candidates) > 1:
            return (
                f"{label} package NAME {expected_name} occurs "
                f"{len(candidates)} times"
            )
        return f"{label} archive URL is missing or changed: {expected_url}"
    candidate = candidates[0]
    if candidate.url != expected_url:
        return f"{label} archive URL is missing or changed: {expected_url}"
    digest = candidate.digest
    if digest is None or SHA256.fullmatch(digest) is None:
        return f"{label} archive is missing URL_HASH SHA256"
    if digest.lower() != expected_digest:
        return (
            f"{label} archive SHA256 drift: "
            f"expected {expected_digest}, found {digest.lower()}"
        )
    inventory_failures = registered_archive_inventory_failures(
        pin_list,
        path=path,
    )
    if inventory_failures:
        return (
            f"{label} manifest archive inventory changed: "
            f"{inventory_failures[0]}"
        )
    return None


def arrow_pin_error(pins: Iterable[ArchivePin] | None = None) -> str | None:
    if pins is None:
        pins = tracked_cmake_archive_pins(ROOT)[0]
    return _exact_pin_error(
        pins,
        path=Path("cmake/Dependencies.cmake"),
        expected_url=ARROW_ARCHIVE,
        expected_digest=ARROW_SHA256,
        expected_name="Arrow",
        label="Arrow",
    )


def example_pin_error(pins: Iterable[ArchivePin]) -> str | None:
    return _exact_pin_error(
        pins,
        path=EXAMPLE_MANIFEST,
        expected_url=EXAMPLE_ARCHIVE,
        expected_digest=EXAMPLE_SHA256,
        expected_name="dtw-cpp",
        label="Example",
    )


def main() -> int:
    action_failures, action_total = workflow_action_pin_results()
    try:
        pins, manifest_total = tracked_cmake_archive_pins()
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as error:
        print(f"CMake archive scan failed: {error}", file=sys.stderr)
        return 1

    archive_failures = archive_pin_failures(pins)
    registered_inventory_failures = registered_archive_inventory_failures(pins)
    arrow_error = arrow_pin_error(pins)
    example_error = example_pin_error(pins)
    inventory_error = None
    if manifest_total != REGISTERED_CMAKE_MANIFEST_TOTAL:
        inventory_error = (
            "tracked CMake manifest inventory changed: "
            f"manifests={manifest_total} "
            f"expected={REGISTERED_CMAKE_MANIFEST_TOTAL}"
        )

    mutable_total = sum(pin.mutable for pin in pins)
    unhashed_total = sum(not pin.hashed for pin in pins)
    archive_verified = sum(pin.verified for pin in pins)
    action_verified = action_total - len(action_failures)
    arrow_verified = 0 if arrow_error else 1
    failures = bool(
        action_failures
        or archive_failures
        or registered_inventory_failures
        or arrow_error
        or example_error
        or inventory_error
    )
    action_verdict = "FAIL" if action_failures else "PASS"
    archive_verdict = (
        "FAIL"
        if (
            archive_failures
            or registered_inventory_failures
            or example_error
            or inventory_error
        )
        else "PASS"
    )
    arrow_verdict = "FAIL" if arrow_error else "PASS"

    if action_failures:
        print("mutable GitHub Action references:", file=sys.stderr)
        for failure in action_failures:
            print(f"  {failure}", file=sys.stderr)
    if archive_failures:
        print("mutable or unhashed tracked CMake archives:", file=sys.stderr)
        for failure in archive_failures:
            print(f"  {failure}", file=sys.stderr)
    if registered_inventory_failures:
        print("tracked CMake archive identity drift:", file=sys.stderr)
        for failure in registered_inventory_failures:
            print(f"  {failure}", file=sys.stderr)
    if arrow_error:
        print(arrow_error, file=sys.stderr)
    if example_error:
        print(example_error, file=sys.stderr)
    if inventory_error:
        print(inventory_error, file=sys.stderr)

    print(
        "WORKFLOW_ACTION_PIN_GATE "
        f"verified={action_verified} total={action_total} verdict={action_verdict}"
    )
    print(
        "CMAKE_ARCHIVE_PIN_GATE "
        f"verified={archive_verified} total={len(pins)} "
        f"mutable={mutable_total} unhashed={unhashed_total} "
        f"verdict={archive_verdict}"
    )
    print(
        "ARROW_ARCHIVE_PIN_GATE "
        f"verified={arrow_verified} total=1 verdict={arrow_verdict}"
    )
    print(f"TRACKED_CMAKE_MANIFESTS total={manifest_total}")
    if failures:
        return 1
    print("supply-chain pins verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
