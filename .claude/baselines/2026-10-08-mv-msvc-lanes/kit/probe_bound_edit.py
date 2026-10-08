"""One-off: add the route-bound count to xprobe.cpp's check_lanes (unit MV). Run once from the kit directory."""
import pathlib

p = pathlib.Path("xprobe.cpp")
s = p.read_text(encoding="utf-8")
edits = [
    ("// its last column in the lanes past it, as the fill does; every output against the per-pair route, bitwise.\n",
     "// its last column in the lanes past it, as the fill does; every output against the per-pair route, bitwise, and\n"
     "// (MV) a mismatch against tests/support/dtw_route_bound.hpp: |a - b| <= 2 (2n - 1) eps(T) max(|a|, |b|), inf only\n"
     "// equal to inf; `max dev` is the largest |a - b| / ((2n - 1) eps(T) max(|a|, |b|)) over the mismatches (bound: 2).\n"),
    ("  long long outputs = 0, vs_pair = 0, nonfinite = 0;\n"
     "  for (int seed = 0; seed < seeds; ++seed)\n"
     "    for (int kind = 0; kind < 4; ++kind) {\n"
     "      std::vector<std::uint64_t> hrow(1001, kFnv0);\n"
     "#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs, vs_pair, nonfinite)\n",
     "  long long outputs = 0, vs_pair = 0, nonfinite = 0, outside = 0;\n"
     "  double max_dev = 0;\n"
     "  for (int seed = 0; seed < seeds; ++seed)\n"
     "    for (int kind = 0; kind < 4; ++kind) {\n"
     "      std::vector<std::uint64_t> hrow(1001, kFnv0);\n"
     "#pragma omp parallel for schedule(dynamic, 1) reduction(+ : outputs, vs_pair, nonfinite, outside) \\\n"
     "  reduction(max : max_dev)\n"),
    ("            const double p = pair(x, y[w]);\n"
     "            vs_pair += std::memcmp(&p, &res[w], sizeof(double)) != 0;\n"
     "            nonfinite += !std::isfinite(res[w]);\n",
     "            const double p = pair(x, y[w]);\n"
     "            const bool differ = std::memcmp(&p, &res[w], sizeof(double)) != 0;\n"
     "            vs_pair += differ;\n"
     "            nonfinite += !std::isfinite(res[w]);\n"
     "            if (differ) {\n"
     "              if (std::isinf(p) || std::isinf(res[w])) {\n"
     "                ++outside;\n"
     "              } else {\n"
     "                const double unit = double(2 * n - 1) * double(std::numeric_limits<T>::epsilon())\n"
     "                                    * std::max(std::abs(p), std::abs(res[w]));\n"
     "                const double dev = std::abs(p - res[w]) / unit;\n"
     "                outside += !(dev <= 2.0);\n"
     "                max_dev = std::max(max_dev, dev);\n"
     "              }\n"
     "            }\n"),
    ('              "mismatches %lld  non-finite outputs %lld\\n",\n'
     '              tname, sq ? "Sq" : "L1", W, seeds, outputs, (unsigned long long)hash, vs_pair, nonfinite);\n',
     '              "mismatches %lld  non-finite outputs %lld  outside the route bound %lld  max dev %.3f\\n",\n'
     '              tname, sq ? "Sq" : "L1", W, seeds, outputs, (unsigned long long)hash, vs_pair, nonfinite, outside,\n'
     '              max_dev);\n'),
]
for old, new in edits:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("edited")
