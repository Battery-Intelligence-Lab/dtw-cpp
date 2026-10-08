#!/bin/sh
# clang_tree.sh configure|build|ctest [args]: the RL worktree's clang tree C:/D/git/wt/RL/build, configured as the
# main checkout's build/ (its CMakeCache: clang++ 21.1.8, Ninja Release, Gurobi, HiGHS, llfio, YAML OFF, Arrow OFF,
# benchmarks, tests, native arch, IPO), deps from the CPM cache C:/D/cpm-cache (as MV's clang_tree.sh).
set -e
W=C:/D/git/wt/RL
export TMP='C:\D\git\wt\tmp\rl' TEMP='C:\D\git\wt\tmp\rl'
mkdir -p C:/D/git/wt/tmp/rl
unset CMAKE_GENERATOR
case "$1" in
configure)
  cmake -S $W -B $W/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
    "-DCMAKE_C_COMPILER=C:/Program Files/LLVM/bin/clang.exe" -DCMAKE_MAKE_PROGRAM=C:/D/git/dtw-cpp/.venv/Scripts/ninja.exe \
    -DCPM_SOURCE_CACHE=C:/D/cpm-cache -DDTWC_BUILD_TESTING=ON -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_GUROBI=ON \
    -DGUROBI_HOME=C:/gurobi1301/win64 -DDTWC_ENABLE_YAML=OFF -DDTWC_ENABLE_ARROW=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ;;
build) shift; cmake --build $W/build -j 8 "$@" ;;
ctest) shift; ctest --test-dir $W/build -C Release -j1 --output-on-failure "$@" ;;
*) echo "usage: clang_tree.sh configure|build|ctest"; exit 2 ;;
esac
