#!/usr/bin/env bash
# Fetch LLVM's official Windows x64 release, unmodified, for the clang-cl builds.
#
# The Windows wheels are compiled by its clang-cl and linked by its lld-link, and they
# bundle its OpenMP runtime (bin/libomp.dll), so compiler and runtime come from one
# release: the one scripts/build_libomp_macos.sh builds the macOS libomp from.
# clang-cl and lld-link find Visual Studio's headers, libraries and Windows SDK
# themselves; no vcvars is needed.
#
# Usage:  bash scripts/fetch_llvm_windows.sh <dir>
#         then configure with CC and CXX set to <dir>/bin/clang-cl.exe.
# Needs:  curl, sha256sum, tar and xz (Git for Windows has them). Licence of
#         libomp.dll: Apache-2.0 WITH LLVM-exception (THIRD_PARTY_LICENSES.md).
set -euo pipefail

LLVM_VERSION=23.1.1
LLVM_SHA256=c54ac8146b420fe72e11e6fdd56498d6818011ad23267196b6ab37b5ac9264c3
ASSET="clang+llvm-${LLVM_VERSION}-x86_64-pc-windows-msvc.tar.xz"
LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${ASSET}"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <dir>" >&2
    exit 2
fi
mkdir -p "$1"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

curl -fsSL --retry 5 --retry-all-errors -o "${WORK}/${ASSET}" "${LLVM_URL}"
echo "${LLVM_SHA256}  ${WORK}/${ASSET}" | sha256sum -c -
# --force-local: a drive letter's colon is not a remote host.
tar --force-local -xJf "${WORK}/${ASSET}" -C "$1" --strip-components=1

# What the build takes from the release; a missing one stops here.
ls -l "$1/bin/clang-cl.exe" "$1/bin/lld-link.exe" "$1/bin/libomp.dll" "$1/lib/libomp.lib"
"$1/bin/clang-cl.exe" --version
