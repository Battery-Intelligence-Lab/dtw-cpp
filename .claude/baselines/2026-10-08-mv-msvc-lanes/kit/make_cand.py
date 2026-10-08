# -*- coding: utf-8 -*-
"""make_cand.py <dir with dtwc/>: apply unit MV's candidate edits to <dir>/dtwc (dtw_kernel.hpp,
core/CMakeLists.txt). Each edit must match exactly once; the files' line endings are kept."""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "dtwc"
CRLF, LF = chr(13) + chr(10), chr(10)


def edit(rel, pairs):
    p = root / rel
    raw = p.read_bytes().decode("utf-8")
    crlf = CRLF in raw
    s = raw.replace(CRLF, LF)
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:70])
        s = s.replace(old, new)
    p.write_bytes((s.replace(LF, CRLF) if crlf else s).encode("utf-8"))


edit("core/dtw_kernel.hpp", [
("""// `d * d + m` into an FMA in one kernel and not the other (GCC does by default;
// the two then differ in the last bits). The loop over the lanes is the one that
""",
"""// `d * d + m` into an FMA in one kernel and not the other (GCC does by default,
// cl in its packed lanes under /fp:contract; the two then differ in the last
// bits). The loop over the lanes is the one that
"""),
("""/// disagree. Elsewhere this is StandardCell: x86's fmin is three instructions (minpd,
/// cmpunordpd, blendvpd) where its min is one. The per-pair kernels keep std::min
/// on AArch64 too: their scalar fminnm chain measured slower.
""",
"""/// disagree. The per-pair kernels keep std::min on AArch64 too: their scalar
/// fminnm chain measured slower. Elsewhere the min is std::min's own comparison,
/// returned by value: the same operand as std::min for every input, NaN and ±0
/// included, but cl packs it where it leaves std::min's reference scalar
/// (/Qvec-report reason 1200). x86's fmin is three instructions (minpd, cmpunordpd,
/// blendvpd) where this min is one.
"""),
("""#else
using LanesCell = StandardCell;
#endif
""",
"""#else
struct LanesCell : StandardCell {
  template <typename T>
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    const T m = up < diag ? up : diag;   // std::min(diag, up)
    return (left < m ? left : m) + cost; // std::min(m, left)
  }
};
#endif
"""),
("""/// floats) on AArch64 and on x86-64 under clang (clang++, clang-cl), where 64 left
/// the pipes waiting on that chain; 64 bytes, one cache line, elsewhere and under
/// cl, whose lanes are scalar and ran slower at 128 (baselines
/// 2026-10-06-mac-arm-lanes, 2026-10-08-xl-x86-lanes-128). GCC's x86-64 lanes stay
/// at 64 bytes until timed, though its f64 lanes are packed at 128 and scalar at 64.
#if defined(__aarch64__) || (defined(__x86_64__) && defined(__clang__))
""",
"""/// floats) on AArch64 and on x86-64 under clang (clang++, clang-cl), where 64 left
/// the pipes waiting on that chain, and under cl, which packs a loop of 12 or more
/// iterations only (/Qvec-report reason 1303: 8 doubles stay scalar); 64 bytes, one
/// cache line, elsewhere (baselines 2026-10-06-mac-arm-lanes,
/// 2026-10-08-xl-x86-lanes-128, 2026-10-08-mv-msvc-lanes). GCC's x86-64 lanes stay
/// at 64 bytes until timed, though its f64 lanes are packed at 128 and scalar at 64.
#if defined(__aarch64__) || (defined(__x86_64__) && defined(__clang__)) || defined(_M_X64)
"""),
])
edit("core/CMakeLists.txt", [
("""  set_source_files_properties(dtw_lanes.cpp TARGET_DIRECTORY dtwc_core PROPERTIES COMPILE_OPTIONS -fno-lto)
endif()
""",
"""  set_source_files_properties(dtw_lanes.cpp TARGET_DIRECTORY dtwc_core PROPERTIES COMPILE_OPTIONS -fno-lto)
endif()
# GCC unrolls a loop of up to 16 iterations completely before its loop vectoriser runs, and its
# basic-block vectoriser cannot pack the selects of the unrolled lane loop (x86-64's LanesCell, a min
# by value): capped at 8, a loop of 16 lanes stays a loop and is packed. AArch64's fminnm lanes are
# left as they were.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND _dtwc_x86_64)
  set_source_files_properties(dtw_lanes.cpp TARGET_DIRECTORY dtwc_core PROPERTIES
    COMPILE_OPTIONS --param=max-completely-peel-times=8)
endif()
"""),
])
print("applied to", root)
