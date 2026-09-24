#!/usr/bin/env bash
# Build LLVM's OpenMP runtime (libomp) for an older macOS than the build host.
#
# The CLI archive and the macOS wheel both bundle the libomp they link against.
# Homebrew's libomp is built for the runner's own macOS (`minos 26.0` on
# macos-latest), so bundling it makes both artefacts refuse to start below
# macOS 26 whatever deployment target the rest of the build uses. This builds
# the same LLVM release Homebrew ships (same source tarball, same SHA-256) for
# MACOSX_DEPLOYMENT_TARGET (default 13.3, the target of release-artifacts.yml
# and python-wheels.yml).
#
# Usage:  bash scripts/build_libomp_macos.sh <install-prefix>
#         then configure DTWC++ with -DOpenMP_ROOT=<install-prefix>.
# Needs:  curl, CMake >= 3.20, a C/C++ compiler, python3 (the LLVM runtimes
#         build requires it). Licence of the result: Apache-2.0 WITH
#         LLVM-exception (THIRD_PARTY_LICENSES.md).
set -euo pipefail

LLVM_VERSION=23.1.1
LLVM_SHA256=ebe9be46fe8756d58c5b198ffad0fa2a766257add81a4dc52179bfacc7888ee6
LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz"
TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.3}"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <install-prefix>" >&2
    exit 2
fi
mkdir -p "$1"
PREFIX="$(CDPATH='' cd -- "$1" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

curl -fsSL --retry 5 --retry-all-errors -o "${WORK}/llvm.tar.xz" "${LLVM_URL}"
echo "${LLVM_SHA256}  ${WORK}/llvm.tar.xz" | shasum -a 256 -c -

# LLVM 22 stopped publishing per-project source tarballs and LLVM 23 removed the
# standalone openmp build, so this is the runtimes build over the five subtrees
# of the monorepo it reads (third-party/unittest: openmp adds its unit tests
# unconditionally; they are configured but never built or installed).
SRC="llvm-project-${LLVM_VERSION}.src"
tar -xJf "${WORK}/llvm.tar.xz" -C "${WORK}" \
    "${SRC}/cmake" "${SRC}/llvm/cmake" "${SRC}/openmp" "${SRC}/runtimes" \
    "${SRC}/third-party/unittest"

# An absolute install name, as Homebrew's libomp has: the CLI's install step
# rewrites exactly that name to @rpath, and delocate resolves it when it copies
# the runtime into the wheel. No aliases: a libgomp.dylib here could shadow GCC's.
cmake -S "${WORK}/${SRC}/runtimes" -B "${WORK}/build" \
    -DLLVM_ENABLE_RUNTIMES=openmp \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${TARGET}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_NAME_DIR="${PREFIX}/lib" \
    -DLIBOMP_INSTALL_ALIASES=OFF
cmake --build "${WORK}/build" --target omp --parallel "$(sysctl -n hw.ncpu)"
cmake --install "${WORK}/build"
otool -l "${PREFIX}/lib/libomp.dylib" | grep -A4 LC_BUILD_VERSION
