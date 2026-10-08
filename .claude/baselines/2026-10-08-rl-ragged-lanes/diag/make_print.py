"""Diagnostic only (kit): diag/src_<v>/dtwc, a copy of src/<v>/dtwc whose dtw_kernel_lanes prints, once per thread, the
addresses mod 4096 of its rolling column s, its packed series Y, x, and a local on its stack, to stderr.
usage: make_print.py <v>"""
import pathlib
import shutil
import sys

KIT = pathlib.Path(__file__).resolve().parents[1]
v = sys.argv[1]
dst = KIT / "diag" / f"src_{v}" / "dtwc"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(KIT / "src" / v / "dtwc", dst)
p = dst / "core" / "dtw_kernel.hpp"
s = p.read_bytes().decode("utf-8")
anchor = "  Row *s = s_buf.data();\n"
if anchor not in s:
    anchor = anchor.replace("\n", "\r\n")
assert s.count(anchor) == 1
x_name = "x" if "const T *x = xs.data();" in s or "const T *x," in s else "x"
probe = (anchor +
         "  { thread_local bool rl_printed = false; int rl_local = 0;\n"
         "    if (!rl_printed) { rl_printed = true;\n"
         "      std::fprintf(stderr, \"lanes W %zu: s %% 4096 = %zu, Y %% 4096 = %zu, x %% 4096 = %zu, stack %% 4096 = %zu\\n\",\n"
         "                   W, std::size_t(reinterpret_cast<std::uintptr_t>(s) % 4096),\n"
         "                   std::size_t(reinterpret_cast<std::uintptr_t>(Y) % 4096),\n"
         "                   std::size_t(reinterpret_cast<std::uintptr_t>(x) % 4096),\n"
         "                   std::size_t(reinterpret_cast<std::uintptr_t>(&rl_local) % 4096)); } }\n")
s = s.replace(anchor, probe)
s = s.replace("#include <vector>", "#include <vector>\n#include <cstdio>\n#include <cstdint>", 1)
p.write_bytes(s.encode("utf-8"))
print(f"wrote {p}")
