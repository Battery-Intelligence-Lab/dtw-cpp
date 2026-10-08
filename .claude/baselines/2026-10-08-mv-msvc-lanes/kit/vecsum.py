"""vecsum.py <vec-report .txt>: the C5001/C5002 lines of cl's /Qvec-report:2 that point into dtw_kernel.hpp, per
dtw_kernel_lanes instantiation (f64|f32 x lambda_1 squared | lambda_2 L1), as `line: reason`."""
import re, sys
cur, out = None, {}
for ln in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if ln.startswith("--- Analyzing function:"):
        m = re.match(r"--- Analyzing function: class std::array<(double|float),\d+> __cdecl dtwc::core::dtw_kernel_lanes<.*?<lambda_(\d)>", ln)
        cur = (("f64" if m.group(1) == "double" else "f32") + (" L1" if m.group(2) == "2" else " Sq")) if m else None
        continue
    if cur is None:
        continue
    m = re.search(r"dtw_kernel\.hpp\((\d+)\) : info (C500[12]): loop (?:not vectorized due to reason '(\d+)'|(vectorized))", ln)
    if m:
        out.setdefault(cur, []).append(f"{m.group(1)}:{m.group(3) or 'VECTORIZED'}")
for k in sorted(out):
    print(f"{k}: " + " ".join(out[k]))
