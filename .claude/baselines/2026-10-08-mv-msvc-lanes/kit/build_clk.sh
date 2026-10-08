#!/bin/sh
# build_clk.sh: bin/clk.exe, the clock of CPU 12 for the cl rounds (tools/clk.cpp, clang++ -O2: cl has no x64
# inline assembly). It prints add-imm, add-reg and imul-chain clocks; the driver takes add-reg (this core folds a
# chain of add-immediate at rename, XL's tools/clk.cpp).
set -e
cd "$(dirname "$0")"
mkdir -p bin
"/c/Program Files/LLVM/bin/clang++.exe" -O2 -std=c++20 tools/clk.cpp -o bin/clk.exe
echo "built bin/clk.exe"
