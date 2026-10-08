"""Diagnostic only (kit): diag/src_<name>/dtwc, a copy of src/B2/dtwc whose dtw_kernel_lanes has one B feature taken
out or changed, to see which one moves the L 8000 one-thread time. usage: make_variant.py <name>
  notake   no take lambda and no per-column end test: every lane copied out after the last column (equal lengths only)
  nopad    the packing's padding loop removed (equal lengths only)
"""
import pathlib
import shutil
import sys

KIT = pathlib.Path(__file__).resolve().parents[1]
name = sys.argv[1]
dst = KIT / "diag" / f"src_{name}" / "dtwc"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(KIT / "src" / "B2" / "dtwc", dst)
p = dst / "core" / "dtw_kernel.hpp"
s = p.read_bytes().decode("utf-8").replace("\r\n", "\n")


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


if name == "notake":
    sub("""  auto take = [&out, &ny, s, nx, ny_max](std::size_t columns) {
    std::size_t next = ny_max;
    for (std::size_t w = 0; w < W; ++w) {
      if (ny[w] == columns) out[w] = s[nx - 1].v[w];
      if (ny[w] > columns && ny[w] < next) next = ny[w];
    }
    return next;
  };
  std::size_t end = take(1);
""", "")
    sub("""    if (j + 1 == end) end = take(j + 1);
  }
  return out;""", """  }
  for (std::size_t w = 0; w < W; ++w) out[w] = s[nx - 1].v[w];
  return out;""")
elif name == "nopad":
    sub("""    for (; t < ny_max; ++t) Y[t].v[w] = yw[ny[w] - 1];
""", "")
else:
    sys.exit("unknown variant")
p.write_bytes(s.encode("utf-8"))
print(f"wrote {p}")
