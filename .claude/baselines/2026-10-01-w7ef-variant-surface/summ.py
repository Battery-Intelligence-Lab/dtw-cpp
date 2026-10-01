import csv, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for case in dict.fromkeys(r['case'] for r in rows):
    b = sorted(float(r['fill_ms']) for r in rows if r['case'] == case and r['binary'] == 'base')
    h = sorted(float(r['fill_ms']) for r in rows if r['case'] == case and r['binary'] == 'head')
    mb, mh = st.median(b), st.median(h)
    limit = mb + (b[-1] - b[0])
    print(f"{case}: base {b} median {mb:.1f} spread {b[-1]-b[0]:.1f}; head {h} median {mh:.1f}; head/base {mh/mb:.3f}; band {'PASS' if mh <= limit else 'FAIL'} (limit {limit:.1f})")
