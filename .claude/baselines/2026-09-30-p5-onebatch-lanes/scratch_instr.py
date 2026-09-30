# Scratch (never committed): time the batch-table fill and hash the N x m table, in any tree.
import sys
p = sys.argv[1]
raw = open(p, newline="").read()
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")
old = "  FixedBatchDistances distances(prob, std::move(sample));\n"
assert s.count(old) == 1
new = """  const auto scratch_t0 = std::chrono::steady_clock::now();
  FixedBatchDistances distances(prob, std::move(sample));
  {
    const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - scratch_t0).count();
    std::uint64_t h = 1469598103934665603ull;
    for (double v : distances.raw) { std::uint64_t b; std::memcpy(&b, &v, 8); for (int i = 0; i < 8; ++i) { h ^= (b >> (8 * i)) & 0xff; h *= 1099511628211ull; } }
    std::fprintf(stderr, "OBP fill_s=%.6f table_hash=%016llx evals=%llu\\n", fill_s, (unsigned long long)h, (unsigned long long)distances.evaluations);
  }
"""
s = s.replace(old, new)
s = s.replace("#include <algorithm>\n", "#include <algorithm>\n#include <chrono>\n#include <cstdio>\n#include <cstring>\n", 1)
open(p, "w", newline="").write(s.replace("\n", "\r\n") if crlf else s)
