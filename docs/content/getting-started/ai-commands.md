---
title: AI-Assisted Workflow (Claude Code)
weight: 7
---

# AI-Assisted Workflow (Claude Code)

DTWC++ includes a set of [Claude Code](https://claude.ai/code) slash commands that let you drive the library through natural-language prompts. Instead of writing Python or CLI commands by hand, you ask your AI assistant to cluster, evaluate, or visualize — the command expands into a complete, reproducible script using the DTWC++ API.

## Availability

The commands live in `.claude/commands/`. They are discovered automatically when Claude Code is launched from the DTWC++ repository root. Each command is a markdown file with a YAML frontmatter that specifies the tools Claude may use (`Read`, `Write`, `Bash`, etc.).

No separate installation is required — they ship with the repository.

## Command reference

### `/cluster` — Full clustering pipeline

Handles the end-to-end workflow: load data, pick a method and variant, run clustering, compute evaluation metrics, save outputs.

```text
/cluster data/ecg.csv -k 5 --variant wdtw
/cluster experiment.parquet             # silhouette-based k search
/cluster huge_dataset.parquet --method clara --ram-limit 8G
```

The command auto-detects:
- Input format from extension (CSV, Parquet, Arrow IPC, HDF5)
- Method from N (PAM for N ≤ 5000, CLARA above)
- Missing data (auto-switches to `missing_strategy=arow`)

### `/distance` — Compute DTW distances

Either a single pair of series with all variants compared, or the full pairwise matrix.

```text
/distance series1.csv series2.csv       # compare DTW/DDTW/WDTW/ADTW/Soft-DTW
/distance data.parquet --matrix          # pairwise NxN matrix
/distance data.parquet --matrix --device cuda
```

### `/evaluate` — Clustering quality metrics

Given data and labels, compute silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn, inertia. If ground-truth labels are supplied, also ARI and NMI.

```text
/evaluate data.csv labels.csv
/evaluate data.csv labels.csv --ground-truth gt.csv
```

Produces an `evaluation.json` and a recommendation ("silhouette < 0.25 — try different k or variant").

### `/convert` — Format conversion

```text
/convert data.csv data.parquet           # 5-20× smaller
/convert data.parquet data.arrow         # zero-copy mmap
```

Uses the `dtwc-convert` Python CLI if available, otherwise falls back to `dtwcpp.io` Python APIs.

### `/visualize` — Plots

```text
/visualize clusters data.csv labels.csv medoids.csv
/visualize silhouette silhouette.csv labels.csv
/visualize distance-matrix distances.csv labels.csv
/visualize warping-path data.csv 0 3        # alignment between series 0 and 3
/visualize elbow data.csv                    # cost vs k for k=2..10
```

### `/help` — Reference

```text
/help how do I pick k?
/help what variant for shape matching?
/help soft-dtw gamma
```

Returns a focused answer with a runnable code snippet.

### `/troubleshoot` — Diagnosis

Paste an error message or describe a symptom; the command diagnoses the root cause and provides the fix.

```text
/troubleshoot ModuleNotFoundError: dtwcpp
/troubleshoot std::bad_alloc during clustering
/troubleshoot clusters are all the same label
```

## Example workflow

A typical first-time user session:

```text
/help how do I choose between PAM and CLARA?
/convert data.csv data.parquet
/cluster data.parquet -k 4
/evaluate data.parquet clustering_output/labels.csv
/visualize clusters data.parquet clustering_output/labels.csv clustering_output/medoids.csv
```

The assistant reports timing, metrics, and saved file paths at each step, and suggests the next command.

## Design principles

- **Progressive disclosure**: commands ask for the minimum needed info, fill sensible defaults, explain why.
- **Script generation**: complex operations produce a `*.py` script you can inspect, rerun, or modify.
- **Graceful fallback**: Python preferred; falls back to `dtwc_cl` CLI when Python is unavailable.
- **Cross-referencing**: every command suggests related ones.
- **Read-only by default** for `/help` and `/troubleshoot` — they never modify your data or files.

## Extending commands

To add your own command, create a new file in `.claude/commands/` with frontmatter:

```markdown
---
description: "One-line description (used for discovery)"
allowed-tools:
  - Read
  - Write
  - Bash
---

# My Command

Instructions for Claude to follow when the user invokes `/my-command`.
```

See the shipped commands for patterns and conventions.
