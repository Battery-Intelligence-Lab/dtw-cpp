---
title: Getting Started
weight: 1
---

# Getting started

Check the [installation and dependencies guide](installation.md#dependencies) first, then choose how to run DTWC++: from the [command line](cli.md), [Python](python.md), [MATLAB](matlab.md), or C++ (`examples/cpp`, [Examples](examples.md)). A C++ program against the library gives the most freedom; the command line is the simplest.

Supported input formats are detailed in [Supported data formats](supported-data.md). Output files (written to `--output`, `./results` by default) are:
- `<name>_distance_matrix.csv` — pairwise DTW distances, written when the method materialised the full matrix (`onebatch`, `tadpole`, non-full `clara`, and streamed runs do not).
- `<name>_labels.csv` and `<name>_medoids.csv` — cluster assignments and medoid centres. The total [clustering cost](../method/mip.md) is printed to stdout when the run ends (`Problem::write_clusters` writes it at the bottom of its own CSV).
- `<name>_silhouettes.csv` — per-series silhouette scores (written alongside the distance matrix when `k > 1`); their mean can be taken as the total silhouette score of the clustering.
- `<name>_report.txt` — the run report: the run's machine, memory plan, stages and result in plain text (`--no-report` writes none).
