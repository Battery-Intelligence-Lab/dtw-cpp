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
    "share/doc/dtwc/fast_float/LICENSE-APACHE",
    "share/doc/dtwc/fast_float/LICENSE-BOOST",
    "share/doc/dtwc/fast_float/LICENSE-MIT",
)

# A dependency path is acceptable if it is resolved relative to the executable
# (so it travels with the archive) or belongs to the OS itself.
PORTABLE_PREFIXES = ("@rpath/", "@loader_path/", "@executable_path/", "$ORIGIN/")
SYSTEM_PREFIXES = ("/usr/lib/", "/System/", "/lib/", "/lib64/", "/usr/lib64/")

# The oldest macOS the archive promises to start on: CMAKE_OSX_DEPLOYMENT_TARGET
# in release-artifacts.yml. dyld refuses any Mach-O whose minimum is newer than
# the running system, so ONE bundled library built for the builder's macOS
# (Homebrew's libomp carries minos 26.0) makes the whole archive unusable below
# it. Running the CLI here cannot notice: the builder is new enough.
MACOS_DEPLOYMENT_TARGET = (13, 3)
MACHO_MAGICS = {
    bytes.fromhex(magic)
    for magic in ("feedface", "cefaedfe", "feedfacf", "cffaedfe", "cafebabe", "bebafeca")
}


def macos_minimums(binary: Path) -> list[tuple[int, ...]]:
    """Every minimum macOS a Mach-O file records, one per architecture slice."""
    text = subprocess.run(
        ["otool", "-arch", "all", "-l", str(binary)], text=True, capture_output=True, check=True
    ).stdout
    found = []
    for command in text.split("Load command")[1:]:
        fields = dict(
            parts for parts in (line.split(None, 1) for line in command.splitlines()) if len(parts) == 2
        )
        # LC_BUILD_VERSION records `minos`; the older LC_VERSION_MIN_MACOSX, `version`.
        key = {"LC_BUILD_VERSION": "minos", "LC_VERSION_MIN_MACOSX": "version"}.get(fields.get("cmd"))
        if key in fields:
            found.append(tuple(int(part) for part in fields[key].strip().split(".")))
    return found


def check_macos_minimum(root: Path) -> None:
    """Fail unless every Mach-O file in the archive starts on the promised macOS."""
    target = ".".join(map(str, MACOS_DEPLOYMENT_TARGET))
    offenders = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        with path.open("rb") as stream:
            if stream.read(4) not in MACHO_MAGICS:
                continue
        minimums = macos_minimums(path)
        if not minimums or max(minimums) > MACOS_DEPLOYMENT_TARGET:
            shown = ", ".join(".".join(map(str, v)) for v in minimums) or "none recorded"
            offenders.append(f"{path.relative_to(root)} (minos {shown})")
    if offenders:
        raise RuntimeError(
            f"archive does not start on macOS {target}: {offenders}\n"
            "Build the dependency for that target — scripts/build_libomp_macos.sh "
            "builds the OpenMP runtime; pass its prefix as OpenMP_ROOT."
        )


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

        root = executables[0].resolve().parent.parent
        check_self_contained(executables[0].resolve(), root)
        if sys.platform == "darwin":
            check_macos_minimum(root)

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
