"""Mutation controls for the tracked CMake archive pin checker."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import check_supply_chain_pins as pins  # noqa: E402


VALID_URL = (
    "https://example.invalid/project/archive/"
    "0123456789abcdef0123456789abcdef01234567.zip"
)
VALID_HASH = "a" * 64


def cpm_block(
    url: str = VALID_URL,
    hash_line: str | None = None,
    name: str = "fixture",
) -> str:
    if hash_line is None:
        hash_line = f"  URL_HASH SHA256={VALID_HASH}\n"
    return (
        "CPMAddPackage(\n"
        f"  NAME {name}\n"
        f'  URL "{url}"\n'
        f"{hash_line}"
        ")\n"
    )


def parsed(text: str, path: str = "fixture/CMakeLists.txt"):
    return pins.cmake_archive_pins_in_text(Path(path), text)


def test_exact_inherited_example_fixture_is_rejected():
    text = cpm_block(
        "https://github.com/Battery-Intelligence-Lab/dtw-cpp/"
        "archive/refs/heads/documentation_update.zip",
        hash_line="",
    )
    failures = pins.archive_pin_failures(
        parsed(text, "examples/cpp/example_project/CMakeLists.txt")
    )
    assert len(failures) == 2
    assert "mutable remote archive URL" in failures[0]
    assert "missing or invalid URL_HASH" in failures[1]


def test_valid_hashed_commit_archive_is_accepted():
    archive_pins = parsed(cpm_block())
    assert len(archive_pins) == 1
    assert archive_pins[0].verified
    assert pins.archive_pin_failures(archive_pins) == []


def test_missing_url_hash_is_rejected():
    failures = pins.archive_pin_failures(parsed(cpm_block(hash_line="")))
    assert len(failures) == 1
    assert "missing or invalid URL_HASH" in failures[0]


def test_shortened_url_hash_is_rejected():
    failures = pins.archive_pin_failures(
        parsed(cpm_block(hash_line="  URL_HASH SHA256=abcd\n"))
    )
    assert len(failures) == 1
    assert "missing or invalid URL_HASH" in failures[0]


def test_comment_only_url_hash_is_rejected():
    failures = pins.archive_pin_failures(
        parsed(cpm_block(hash_line=f"  # URL_HASH SHA256={VALID_HASH}\n"))
    )
    assert len(failures) == 1
    assert "missing or invalid URL_HASH" in failures[0]


def test_bracket_comment_url_hash_is_rejected():
    failures = pins.archive_pin_failures(
        parsed(
            cpm_block(
                hash_line=(
                    "  #[[\n"
                    f"  URL_HASH SHA256={VALID_HASH}\n"
                    "  ]]\n"
                )
            )
        )
    )
    assert len(failures) == 1
    assert "missing or invalid URL_HASH" in failures[0]


def test_hash_text_in_another_argument_is_rejected():
    failures = pins.archive_pin_failures(
        parsed(
            cpm_block(
                hash_line=(
                    f'  OPTIONS "URL_HASH SHA256={VALID_HASH}"\n'
                )
            )
        )
    )
    assert len(failures) == 1
    assert "missing or invalid URL_HASH" in failures[0]


def test_url_hash_inside_options_is_rejected():
    failures = pins.archive_pin_failures(
        parsed(
            "CPMAddPackage(\n"
            "  NAME fixture\n"
            f'  URL "{VALID_URL}"\n'
            "  OPTIONS\n"
            "  URL_HASH\n"
            f"  SHA256={VALID_HASH}\n"
            ")\n"
        )
    )
    assert len(failures) == 1
    assert "missing or invalid URL_HASH" in failures[0]


def test_branch_url_with_valid_hash_is_rejected():
    branch_url = "https://example.invalid/project/archive/refs/heads/main.zip"
    failures = pins.archive_pin_failures(parsed(cpm_block(branch_url)))
    assert len(failures) == 1
    assert "mutable remote archive URL" in failures[0]


def test_github_shorthand_branch_with_valid_hash_is_rejected():
    branch_url = "https://github.com/example/project/archive/main.zip"
    failures = pins.archive_pin_failures(parsed(cpm_block(branch_url)))
    assert len(failures) == 1
    assert "mutable remote archive URL" in failures[0]


def test_gitlab_branch_with_valid_hash_is_rejected():
    branch_url = (
        "https://gitlab.com/example/project/-/archive/main/project-main.tar.gz"
    )
    failures = pins.archive_pin_failures(parsed(cpm_block(branch_url)))
    assert len(failures) == 1
    assert "mutable remote archive URL" in failures[0]


@pytest.mark.parametrize(
    "branch_url",
    [
        "https://api.github.com/repos/example/project/zipball/main",
        "https://api.github.com/repos/example/project/zipball",
        (
            "https://api.github.com/repos/example/project/"
            "zipball/feature%2Fbranch"
        ),
        (
            "https://gitlab.com/api/v4/projects/1/repository/"
            "archive.zip?sha=main"
        ),
    ],
)
def test_api_branch_archive_with_valid_hash_is_rejected(branch_url):
    failures = pins.archive_pin_failures(parsed(cpm_block(branch_url)))
    assert len(failures) == 1
    assert "mutable remote archive URL" in failures[0]


def test_inline_cpm_arguments_are_scanned():
    archive_pins = parsed(
        f'CPMAddPackage(NAME fixture URL "{VALID_URL}" '
        f"URL_HASH SHA256={VALID_HASH})\n"
    )
    assert len(archive_pins) == 1
    assert archive_pins[0].verified


def test_repeated_url_directives_are_rejected():
    with pytest.raises(ValueError, match="exactly one URL directive"):
        parsed(
            "CPMAddPackage(\n"
            "  NAME fixture\n"
            f'  URL "{VALID_URL}"\n'
            f'  URL "{VALID_URL.replace("project", "mirror")}"\n'
            f"  URL_HASH SHA256={VALID_HASH}\n"
            ")\n"
        )


def test_multiple_url_values_are_rejected():
    with pytest.raises(ValueError, match="multiple URL values"):
        parsed(
            "CPMAddPackage(\n"
            "  NAME fixture\n"
            f'  URL "{VALID_URL}" '
            f'"{VALID_URL.replace("project", "mirror")}"\n'
            f"  URL_HASH SHA256={VALID_HASH}\n"
            ")\n"
        )


def test_dynamic_mirror_url_is_rejected():
    with pytest.raises(ValueError, match="dynamic CPMAddPackage"):
        parsed(
            "CPMAddPackage(\n"
            "  NAME fixture\n"
            f'  URL "{VALID_URL}" ${{MIRROR_URL}}\n'
            f"  URL_HASH SHA256={VALID_HASH}\n"
            ")\n"
        )


def test_semicolon_expanded_arguments_are_rejected():
    with pytest.raises(ValueError, match="semicolon-expanded"):
        parsed(
            "CPMAddPackage("
            f"URL;{VALID_URL};URL_HASH;SHA256={VALID_HASH}"
            ")\n"
        )


@pytest.mark.parametrize(
    "dynamic_arguments",
    [
        "${ARCHIVE_ARGS}",
        "$ENV{ARCHIVE_ARGS}",
        "$CACHE{ARCHIVE_ARGS}",
    ],
)
def test_fully_dynamic_cpm_arguments_are_rejected(dynamic_arguments):
    with pytest.raises(ValueError, match="dynamic CPMAddPackage"):
        parsed(f"CPMAddPackage({dynamic_arguments})\n")


def test_dynamic_arguments_beside_a_literal_url_are_rejected():
    with pytest.raises(ValueError, match="dynamic CPMAddPackage"):
        parsed(
            "CPMAddPackage(\n"
            "  ${ARCHIVE_ARGS}\n"
            "  NAME fixture\n"
            f'  URL "{VALID_URL}"\n'
            f"  URL_HASH SHA256={VALID_HASH}\n"
            ")\n"
        )


@pytest.mark.parametrize(
    ("directive", "value"),
    [
        (
            "DOWNLOAD_COMMAND",
            "git clone --branch main https://example.invalid/project.git",
        ),
        ("SOURCE_DIR", "vendor/project"),
        ("GITHUB_REPOSITORY", "example/project"),
        ("GITLAB_REPOSITORY", "example/project"),
        ("BITBUCKET_REPOSITORY", "example/project"),
        ("GIT_REPOSITORY", "https://example.invalid/project.git"),
        ("SVN_REPOSITORY", "https://example.invalid/project/svn"),
        ("HG_REPOSITORY", "https://example.invalid/project/hg"),
        ("CVS_REPOSITORY", "https://example.invalid/project/cvs"),
        ("FIND_PACKAGE_ARGUMENTS", "CONFIG"),
    ],
)
def test_url_archive_rejects_alternate_source_directives(directive, value):
    with pytest.raises(ValueError, match="alternate source directive"):
        parsed(
            cpm_block(
                hash_line=(
                    f"  URL_HASH SHA256={VALID_HASH}\n"
                    f"  {directive} {value}\n"
                )
            )
        )


def test_cmake_language_call_is_scanned():
    archive_pins = parsed(
        "cmake_language(CALL CPMAddPackage\n"
        "  NAME fixture\n"
        f'  URL "{VALID_URL}"\n'
        f"  URL_HASH SHA256={VALID_HASH}\n"
        ")\n"
    )
    assert len(archive_pins) == 1
    assert archive_pins[0].verified


def test_cmake_language_defer_call_is_scanned():
    archive_pins = parsed(
        "cmake_language(DEFER CALL CPMAddPackage\n"
        "  NAME fixture\n"
        f'  URL "{VALID_URL}"\n'
        f"  URL_HASH SHA256={VALID_HASH}\n"
        ")\n"
    )
    assert len(archive_pins) == 1
    assert archive_pins[0].verified


@pytest.mark.parametrize("option", ["ID", "ID_VAR"])
def test_cmake_language_defer_option_operand_named_call_is_scanned(option):
    archive_pins = parsed(
        f"cmake_language(DEFER {option} call CALL CPMAddPackage\n"
        "  NAME fixture\n"
        f'  URL "{VALID_URL}"\n'
        f"  URL_HASH SHA256={VALID_HASH}\n"
        ")\n"
    )
    assert len(archive_pins) == 1
    assert archive_pins[0].verified


def test_nested_cmake_language_call_is_scanned():
    archive_pins = parsed(
        "cmake_language(CALL cmake_language CALL CPMAddPackage\n"
        "  NAME fixture\n"
        f'  URL "{VALID_URL}"\n'
        f"  URL_HASH SHA256={VALID_HASH}\n"
        ")\n"
    )
    assert len(archive_pins) == 1
    assert archive_pins[0].verified


@pytest.mark.parametrize(
    "dynamic_command",
    [
        "${PACKAGE_COMMAND}",
        "$ENV{PACKAGE_COMMAND}",
        "$CACHE{PACKAGE_COMMAND}",
    ],
)
def test_dynamic_cmake_language_call_is_rejected(dynamic_command):
    with pytest.raises(ValueError, match=r"dynamic cmake_language\(CALL"):
        parsed(
            f"cmake_language(CALL {dynamic_command}\n"
            f'  URL "{VALID_URL}"\n'
            f"  URL_HASH SHA256={VALID_HASH}\n"
            ")\n"
        )


def test_dynamic_cmake_language_operation_is_rejected():
    with pytest.raises(ValueError, match="dynamic cmake_language operation"):
        parsed(
            "cmake_language(${OPERATION} CPMAddPackage\n"
            f'  URL "{VALID_URL}"\n'
            f"  URL_HASH SHA256={VALID_HASH}\n"
            ")\n"
        )


def test_cmake_language_eval_of_cpm_is_rejected():
    with pytest.raises(ValueError, match=r"cmake_language\(EVAL"):
        parsed(
            'cmake_language(EVAL CODE "CPMAddPackage('
            f"URL {VALID_URL} URL_HASH SHA256={VALID_HASH}"
            ')")\n'
        )


@pytest.mark.parametrize(
    "arguments",
    [
        (
            "CALL;CPMAddPackage;NAME;hidden;URL;"
            "https://github.com/example/project/archive/refs/heads/main.zip"
        ),
        (
            "DEFER;CALL;CPMAddPackage;NAME;hidden;URL;"
            "https://github.com/example/project/archive/refs/heads/main.zip"
        ),
        "EVAL;CODE;message",
    ],
)
def test_semicolon_expanded_cmake_language_arguments_are_rejected(arguments):
    with pytest.raises(ValueError, match="semicolon-expanded cmake_language"):
        parsed(f"cmake_language({arguments})\n")


def test_hash_from_another_cpm_call_does_not_satisfy_url():
    archive_pins = parsed(
        cpm_block(hash_line="")
        + cpm_block(
            url=VALID_URL.replace("project", "second"),
            hash_line=f"  URL_HASH SHA256={VALID_HASH}\n",
        )
    )
    failures = pins.archive_pin_failures(archive_pins)
    assert len(archive_pins) == 2
    assert len(failures) == 1
    assert "missing or invalid URL_HASH" in failures[0]


def test_dynamic_url_is_rejected_fail_closed():
    with pytest.raises(ValueError, match="dynamic CPMAddPackage"):
        parsed(
            "CPMAddPackage(\n"
            "  NAME fixture\n"
            "  URL ${ARCHIVE_URL}\n"
            f"  URL_HASH SHA256={VALID_HASH}\n"
            ")\n"
        )


def test_removing_main_dependency_hash_is_rejected():
    dependencies = (ROOT / "cmake/Dependencies.cmake").read_text(encoding="utf-8")
    dependencies = dependencies.replace(
        "URL_HASH SHA256=650795f6501af514f806e78c554729847b98db6935e69076f36bb03ed2e985ef",
        "",
        1,
    )
    archive_pins = pins.cmake_archive_pins_in_text(
        Path("cmake/Dependencies.cmake"), dependencies
    )
    failures = pins.archive_pin_failures(archive_pins)
    assert len(failures) == 1
    assert "Catch2/archive/refs/tags/v3.13.0.tar.gz" in failures[0]


def test_arrow_digest_drift_remains_rejected():
    dependencies = (ROOT / "cmake/Dependencies.cmake").read_text(encoding="utf-8")
    dependencies = dependencies.replace(pins.ARROW_SHA256, "0" * 64, 1)
    archive_pins = pins.cmake_archive_pins_in_text(
        Path("cmake/Dependencies.cmake"), dependencies
    )
    assert pins.arrow_pin_error(archive_pins) == (
        "Arrow archive SHA256 drift: "
        f"expected {pins.ARROW_SHA256}, found {'0' * 64}"
    )


def test_arrow_exact_pin_cannot_be_satisfied_by_a_decoy_package():
    path = Path("cmake/Dependencies.cmake")
    archive_pins = parsed(
        cpm_block(
            url=VALID_URL.replace("project", "arrow-drift"),
            name="Arrow",
        )
        + cpm_block(
            url=pins.ARROW_ARCHIVE,
            hash_line=f"  URL_HASH SHA256={pins.ARROW_SHA256}\n",
            name="decoy",
        ),
        path.as_posix(),
    )
    assert pins.arrow_pin_error(archive_pins) == (
        "Arrow archive URL is missing or changed: "
        f"{pins.ARROW_ARCHIVE}"
    )


def test_arrow_exact_name_cannot_be_transferred_to_a_download_only_decoy():
    path = Path("cmake/Dependencies.cmake")
    archive_pins = parsed(
        cpm_block(
            url=VALID_URL.replace("project", "arrow-drift"),
            name="actual-arrow",
        )
        + cpm_block(
            url=pins.ARROW_ARCHIVE,
            hash_line=(
                f"  URL_HASH SHA256={pins.ARROW_SHA256}\n"
                "  DOWNLOAD_ONLY YES\n"
            ),
            name="Arrow",
        ),
        path.as_posix(),
    )
    assert pins.arrow_pin_error(archive_pins) is not None


def test_mutable_workflow_action_remains_rejected():
    failures, total = pins.workflow_action_references_in_text(
        Path(".github/workflows/fixture.yml"),
        "steps:\n  - uses: actions/checkout@v6\n",
    )
    assert total == 1
    assert failures == [
        ".github/workflows/fixture.yml:2: actions/checkout@v6"
    ]


def test_live_tracked_cmake_inventory_is_complete():
    archive_pins, manifest_total = pins.tracked_cmake_archive_pins(ROOT)
    assert manifest_total == 28
    assert len(archive_pins) == 7


def test_inventory_diagnostics_sort_missing_and_string_names():
    archive_pins = [
        pins.ArchivePin(
            path=Path("fixture/CMakeLists.txt"),
            line=1,
            name=None,
            url=VALID_URL,
            digest=VALID_HASH,
        ),
        pins.ArchivePin(
            path=Path("fixture/CMakeLists.txt"),
            line=2,
            name="X",
            url=VALID_URL.replace("project", "second"),
            digest=VALID_HASH,
        ),
    ]
    failures = pins.registered_archive_inventory_failures(archive_pins)
    assert any("|<missing>|" in failure for failure in failures)
    assert any("|X|" in failure for failure in failures)


def test_production_main_rejects_changed_manifest_inventory(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        pins,
        "tracked_cmake_archive_pins",
        lambda: ([], pins.REGISTERED_CMAKE_MANIFEST_TOTAL - 1),
    )
    monkeypatch.setattr(
        pins,
        "workflow_action_pin_results",
        lambda: ([], 39),
    )
    assert pins.main() == 1
    assert "tracked CMake manifest inventory changed" in capsys.readouterr().err


def exact_example_pin(
    url: str = pins.EXAMPLE_ARCHIVE,
    digest: str = pins.EXAMPLE_SHA256,
):
    return parsed(
        cpm_block(
            url=url,
            hash_line=f"  URL_HASH SHA256={digest}\n",
            name="dtw-cpp",
        ),
        pins.EXAMPLE_MANIFEST.as_posix(),
    )


def test_exact_example_pin_is_accepted():
    archive_pins = exact_example_pin()
    assert pins.archive_pin_failures(archive_pins) == []
    assert pins.example_pin_error(archive_pins) is None


def test_example_tag_alias_is_rejected():
    archive_pins = exact_example_pin(
        "https://github.com/Battery-Intelligence-Lab/dtw-cpp/"
        "archive/refs/tags/2.0.0rc1.zip"
    )
    assert pins.example_pin_error(archive_pins) == (
        "Example archive URL is missing or changed: "
        f"{pins.EXAMPLE_ARCHIVE}"
    )


def test_example_short_commit_is_rejected():
    archive_pins = exact_example_pin(
        "https://github.com/Battery-Intelligence-Lab/dtw-cpp/"
        "archive/eda1b92bc89e.zip"
    )
    assert pins.example_pin_error(archive_pins) == (
        "Example archive URL is missing or changed: "
        f"{pins.EXAMPLE_ARCHIVE}"
    )


def test_example_digest_drift_is_rejected():
    archive_pins = exact_example_pin(digest="0" * 64)
    assert pins.example_pin_error(archive_pins) == (
        "Example archive SHA256 drift: "
        f"expected {pins.EXAMPLE_SHA256}, found {'0' * 64}"
    )


def test_example_exact_pin_cannot_be_satisfied_by_a_decoy_package():
    archive_pins = parsed(
        cpm_block(
            url=VALID_URL.replace("project", "dtwc-drift"),
            name="dtw-cpp",
        )
        + cpm_block(
            url=pins.EXAMPLE_ARCHIVE,
            hash_line=f"  URL_HASH SHA256={pins.EXAMPLE_SHA256}\n",
            name="decoy",
        ),
        pins.EXAMPLE_MANIFEST.as_posix(),
    )
    assert pins.example_pin_error(archive_pins) == (
        "Example archive URL is missing or changed: "
        f"{pins.EXAMPLE_ARCHIVE}"
    )


def test_example_exact_name_cannot_be_transferred_to_a_download_only_decoy():
    archive_pins = parsed(
        cpm_block(
            url=VALID_URL.replace("project", "dtwc-drift"),
            name="actual-dtw",
        )
        + cpm_block(
            url=pins.EXAMPLE_ARCHIVE,
            hash_line=(
                f"  URL_HASH SHA256={pins.EXAMPLE_SHA256}\n"
                "  DOWNLOAD_ONLY YES\n"
            ),
            name="dtw-cpp",
        ),
        pins.EXAMPLE_MANIFEST.as_posix(),
    )
    assert pins.example_pin_error(archive_pins) is not None
