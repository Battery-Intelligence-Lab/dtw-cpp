"""Unpack a DTWC++ release archive outside the checkout and run its CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
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

# DLLs that Windows itself provides (10 and later); the api-ms-win-* API sets resolve
# to the kernel and the Universal CRT. The Visual C++ runtime (vcruntime140*,
# msvcp140*, concrt140, vcomp140) and LLVM's libomp.dll are not among them: a machine
# without the Visual C++ Redistributable has none, so the archive carries each one
# its binaries import, beside dtwc_cl.exe, where the loader looks first.
WINDOWS_SYSTEM_DLLS = {"kernel32.dll", "user32.dll", "advapi32.dll", "ucrtbase.dll"}


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


def pe_imports(binary: Path) -> list[str]:
    """The DLLs a PE file imports, at load time or delay-loaded: what `dumpbin /dependents` lists.

    Read here rather than through dumpbin, which is on PATH only inside a Visual Studio
    environment; the release workflow runs this script outside one.
    """
    data = binary.read_bytes()
    header = struct.unpack_from("<I", data, 0x3C)[0]  # "PE\0\0", then the COFF header
    sections, optional_size = struct.unpack_from("<2xH12xH", data, header + 4)
    optional = header + 24
    pe32_plus = struct.unpack_from("<H", data, optional)[0] == 0x20B
    directory_count = struct.unpack_from("<I", data, optional + (108 if pe32_plus else 92))[0]
    directories = optional + (112 if pe32_plus else 96)
    section_table = optional + optional_size

    def offset(rva: int) -> int:
        for entry in range(section_table, section_table + 40 * sections, 40):
            size, address, _, raw = struct.unpack_from("<4I", data, entry + 8)
            if address <= rva < address + size:
                return rva - address + raw
        raise RuntimeError(f"{binary}: RVA {rva:#x} lies in no section")

    names = []
    # Directory 1 lists the import descriptors (20 bytes, name RVA at 12), directory 13
    # the delay-load ones (32 bytes, name RVA at 4); a zero name ends each list.
    for directory, stride, name_at in ((1, 20, 12), (13, 32, 4)):
        if directory >= directory_count:
            continue
        rva = struct.unpack_from("<I", data, directories + 8 * directory)[0]
        if not rva:
            continue
        descriptor = offset(rva)
        while name := struct.unpack_from("<I", data, descriptor + name_at)[0]:
            start = offset(name)
            names.append(data[start:data.index(b"\0", start)].decode("ascii"))
            descriptor += stride
    return names


def check_windows_imports(root: Path, executable_dir: Path) -> None:
    """Fail unless every DLL the archive's binaries import is beside dtwc_cl.exe or part of Windows.

    Running the CLI cannot notice a missing runtime: the build machine has the Visual
    C++ Redistributable installed, and a user's machine may not.
    """
    bundled = {path.name.lower() for path in executable_dir.iterdir() if path.suffix.lower() == ".dll"}
    known = bundled | WINDOWS_SYSTEM_DLLS
    binaries = sorted(path for path in root.rglob("*") if path.suffix.lower() in (".exe", ".dll"))
    missing = [
        f"{binary.relative_to(root)} imports {name}"
        for binary in binaries
        for name in pe_imports(binary)
        if name.lower() not in known and not name.lower().startswith("api-ms-win-")
    ]
    if missing:
        raise RuntimeError(
            f"archive is not self-contained; Windows does not ship these DLLs: {missing}\n"
            "Install the runtime beside dtwc_cl.exe (InstallRequiredSystemLibraries in CMakeLists.txt)."
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
    else:  # Windows records DLL names, not paths: check_windows_imports.
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
        if sys.platform == "win32":
            check_windows_imports(root, executables[0].resolve().parent)

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
                "--mmap-threshold", "0",  # the distance matrix is mapped (llfio)
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
        matrix = output / "archive_smoke.dtwm"  # 4 series: 48-byte header, 10 doubles
        if not matrix.exists() or matrix.stat().st_size != 48 + 10 * 8:
            raise RuntimeError(
                f"the archive's CLI did not map its distance matrix to {matrix}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        print(f"release archive smoke OK: {archive.name} -> {labels}, {matrix.name}")


if __name__ == "__main__":
    main()
