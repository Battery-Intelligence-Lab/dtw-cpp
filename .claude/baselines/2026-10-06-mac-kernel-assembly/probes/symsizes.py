# Usage: symsizes.py <nm -n output file> <regex>  -> prints addr, size, demangled-short for text symbols matching regex
import sys, re, subprocess
lines = [l.split(None, 2) for l in open(sys.argv[1]) if l.strip()]
addrs = sorted({int(a, 16) for a, t, s in lines})
nxt = {a: b for a, b in zip(addrs, addrs[1:])}
pat = re.compile(sys.argv[2])
for a, t, s in lines:
    if pat.search(s):
        ai = int(a, 16)
        size = nxt.get(ai, ai) - ai
        d = subprocess.run(['c++filt'], input=s, capture_output=True, text=True).stdout.strip()
        d = re.sub(r'std::__1::span<[^<>]*(<[^<>]*>)?[^<>]*>', 'SPAN', d)
        d = re.sub(r'std::__1::function<void \(.*?\)>', 'FN', d)
        print(f'{a} {size:6d} {d[:200]}')
