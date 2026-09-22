"""Unpack a DTWC++ release archive outside the checkout and run its CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


FIXTURE = """0,1,2,1,0
0.1,1.1,2.1,1.1,0.1
10,11,12,11,10
10.1,11.1,12.1,11.1,10.1
"""

# Notices that redistribution obliges us to ship in every archive.
REQUIRED_DOCS = (
    "share/doc/dtwc/LICENSE",
    "share/doc/dtwc/THIRD_PARTY_LICENSES.md",
    "share/doc/dtwc/nanoarrow/LICENSE.txt",
    "share/doc/dtwc/nanoarrow/NOTICE.txt",
)

# A dependency path is acceptable if it is resolved relative to the executable
# (so it travels with the archive) or belongs to the OS itself.
PORTABLE_PREFIXES = ("@rpath/", "@loader_path/", "@executable_path/", "$ORIGIN/")
SYSTEM_PREFIXES = ("/usr/lib/", "/System/", "/lib/", "/lib64/", "/usr/lib64/")


def dependency_paths(binary: Path) -> list[str]:
    """Absolute, non-system libraries the binary will try to load."""
    if sys.platform == "darwin":
        lines = subprocess.run(
            ["otool", "-L", str(binary)], text=True, capture_output=True, check=True
        ).stdout.splitlines()[1:]
        found = [line.strip().partition(" (compatibility")[0].strip() for line in lines]
    elif sys.platform.startswith("linux"):
        lines = subprocess.run(
            ["ldd", str(binary)], text=True, capture_output=True, check=True
        ).stdout.splitlines()
        found = []
        for line in lines:
            _, sep, rhs = line.partition("=>")
            if sep:
                found.append(rhs.strip().partition(" (0x")[0].strip())
    else:  # Windows resolves DLLs beside the .exe; no equivalent to inspect here.
        return []
    return [
        path
        for path in found
        if path
        and path.startswith("/")
        and not path.startswith(SYSTEM_PREFIXES)
        and not path.startswith(PORTABLE_PREFIXES)
    ]


def check_self_contained(binary: Path, root: Path) -> None:
    """Fail if the archive only runs on a machine that looks like the builder's.

    Running the CLI is not enough on its own: the build machine still has every
    absolute path the binary was linked against, so a dependency that will be
    missing for a user resolves fine here.
    """
    missing = [name for name in REQUIRED_DOCS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"archive is missing required notices: {missing}")

    escaping = [path for path in dependency_paths(binary) if not path.startswith(str(root))]
    if escaping:
        raise RuntimeError(
            "archive is not self-contained — these are absolute paths on the build "
            f"machine and will not exist for a user: {escaping}\n"
            "Bundle the library under lib/ and rewrite the load command to @rpath "
            "(see the APPLE block in CMakeLists.txt)."
        )


def find_archive(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    candidates = sorted(path.glob("dtwc-*.zip")) + sorted(path.glob("dtwc-*.tar.gz"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one DTWC++ archive in {path}, found {candidates}")
    return candidates[0].resolve()


def extract(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination)
    else:
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(destination, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="archive file or directory containing one")
    args = parser.parse_args()
    archive = find_archive(args.archive)

    with tempfile.TemporaryDirectory(prefix="dtwc-release-smoke-") as scratch_text:
        scratch = Path(scratch_text)
        extract(archive, scratch)
        executable_name = "dtwc_cl.exe" if archive.suffix == ".zip" else "dtwc_cl"
        executables = list(scratch.rglob(executable_name))
        if len(executables) != 1:
            raise RuntimeError(f"expected one {executable_name}, found {executables}")

        check_self_contained(executables[0].resolve(), executables[0].resolve().parent.parent)

        fixture = scratch / "fixture.csv"
        output = scratch / "result"
        fixture.write_text(FIXTURE, encoding="utf-8")
        completed = subprocess.run(
            [
                str(executables[0].resolve()),
                "--input", str(fixture.resolve()),
                "--output", str(output.resolve()),
                "--name", "archive_smoke",
                "--n-clusters", "2",
                "--method", "pam",
                "--max-iter", "3",
                "--device", "cpu",
            ],
            cwd=scratch,
            text=True,
            capture_output=True,
            check=True,
        )
        labels = output / "archive_smoke_labels.csv"
        if not labels.exists():
            raise RuntimeError(
                f"CLI returned success but did not create {labels}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        print(f"release archive smoke OK: {archive.name} -> {labels}")


if __name__ == "__main__":
    main()
