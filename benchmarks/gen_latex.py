#!/usr/bin/env python3
"""Generate LaTeX benchmark table from UCR benchmark JSON."""
import json
import sys
import os

data_path = os.path.join(os.environ.get("TEMP", "/tmp"), "ucr_all_benchmark.json")
with open(data_path) as f:
    data = json.load(f)

xeon = data["xeon16"]
genoa = data["genoa168"]
l40s = data["l40s"]
h100 = data["h100"]
common = sorted(set(xeon) & set(genoa) & set(l40s) & set(h100))

tx = sum(xeon[n]["elapsed_s"] for n in common)
tg = sum(genoa[n]["elapsed_s"] for n in common)
tl = sum(l40s[n]["elapsed_s"] for n in common)
th = sum(h100[n]["elapsed_s"] for n in common)

BS = "\\\\"  # double backslash

lines = []
lines.append(r"\documentclass[11pt,landscape]{article}")
lines.append(r"\usepackage[margin=1.2cm]{geometry}")
lines.append(r"\usepackage{booktabs}")
lines.append(r"\usepackage{longtable}")
lines.append(r"\usepackage{xcolor}")
lines.append(r"\usepackage{colortbl}")
lines.append(r"\definecolor{fastrow}{HTML}{E8F5E9}")
lines.append(r"\pagestyle{empty}")
lines.append(r"\begin{document}")
lines.append(r"\begin{center}")
lines.append(r"{\Large\bfseries DTWC++ UCR Archive Benchmark Results}" + BS + "[4pt]")
lines.append(r"{\normalsize Lloyd k-medoids, DTW L1, float64, Oxford ARC HTC, 2026-04-09}" + BS + "[12pt]")
lines.append(r"\end{center}")
lines.append("")

# Summary table
lines.append(r"\begin{center}")
lines.append(r"\begin{tabular}{llrrr}")
lines.append(r"\toprule")
lines.append(r"Platform & Processor & Total (s) & Total (min) & Speedup " + BS)
lines.append(r"\midrule")
lines.append(f"CPU (baseline) & Intel Xeon Platinum 8268, 16 cores & {tx:.1f} & {tx/60:.1f} & 1.0$\\times$ {BS}")
lines.append(f"CPU (Genoa) & AMD EPYC 9645, 168 cores & {tg:.1f} & {tg/60:.1f} & {tx/tg:.1f}$\\times$ {BS}")
lines.append(f"GPU (L40S) & NVIDIA L40S (Ada Lovelace) & {tl:.1f} & {tl/60:.1f} & {tx/tl:.1f}$\\times$ {BS}")
lines.append(f"GPU (H100) & NVIDIA H100 NVL (Hopper) & {th:.1f} & {th/60:.1f} & {tx/th:.1f}$\\times$ {BS}")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{center}")
lines.append(r"\vspace{6pt}")

# Full results
lines.append(r"\begin{longtable}{lrrrrrrrr}")
lines.append(r"\toprule")
hdr = r"\textbf{Dataset} & \textbf{$N$} & \textbf{Length} & \textbf{$k$} & \textbf{Xeon 16c (s)} & \textbf{EPYC 168c (s)} & \textbf{L40S (s)} & \textbf{H100 (s)} & \textbf{H100 $\times$} " + BS
lines.append(hdr)
lines.append(r"\midrule")
lines.append(r"\endfirsthead")
lines.append(r"\toprule")
lines.append(hdr)
lines.append(r"\midrule")
lines.append(r"\endhead")
lines.append(r"\midrule")
lines.append(r"\multicolumn{9}{r}{\textit{Continued on next page}} " + BS)
lines.append(r"\endfoot")
lines.append(r"\bottomrule")
lines.append(r"\endlastfoot")

for name in common:
    x = xeon[name]["elapsed_s"]
    g = genoa[name]["elapsed_s"]
    l = l40s[name]["elapsed_s"]
    h = h100[name]["elapsed_s"]
    sp = x / h if h > 0.01 else 0
    length = str(xeon[name]["series_length"])
    safe = name.replace("_", r"\_")
    rc = r"\rowcolor{fastrow}" if x > 10 else ""
    lines.append(f"{rc}{safe} & {xeon[name]['n_series']} & {length} & {xeon[name]['k']} & {x:.2f} & {g:.2f} & {l:.2f} & {h:.2f} & {sp:.1f} {BS}")

lines.append(r"\end{longtable}")
lines.append("")
lines.append(r"\vspace{6pt}")
lines.append(r"\noindent\textbf{Notes:} "
             r"All 128 UCR datasets clustered with 0 failures. "
             r"Green rows: datasets taking $>$10\,s on the 16-core Xeon baseline. "
             r"H100 $\times$: speedup of NVIDIA H100 over 16-core Xeon. "
             r"GPU runs use CUDA for distance matrix computation, CPU for clustering iteration. "
             r"Distance matrices saved for CPU/GPU numerical verification.")
lines.append(r"\end{document}")

out = "benchmarks/ucr_benchmark_results.tex"
with open(out, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {out} ({len(common)} datasets)")
