"""Unpack a DTWC++ release archive outside the checkout and run its CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tarfile
import tempfile
import zipfile


FIXTURE = """0,1,2,1,0
0.1,1.1,2.1,1.1,0.1
10,11,12,11,10
10.1,11.1,12.1,11.1,10.1
"""


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
