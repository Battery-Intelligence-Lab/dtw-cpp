---
title: Command Line Interface (CLI)
weight: 2
---

# Command Line Interface (CLI)

DTW-C++ provides a full-featured CLI tool for time series clustering. After compiling the software using the [installation instructions](installation.md), run the `bin/dtwc_cl` executable.

## Features

- **Multiple clustering methods**: FastPAM, FastCLARA, Lloyd's k-medoids, MIP (exact), and hierarchical clustering
- **DTW variants**: Standard, DDTW, WDTW, ADTW, and Soft-DTW
- **Distance metrics**: L1 (default) and squared Euclidean
- **GPU acceleration**: CUDA support for distance matrix computation
- **Configuration files**: TOML (native) and YAML (optional) configuration support
- **Checkpointing**: Save and resume distance matrix computation
- **Flexible I/O**: CSV input with configurable row/column skipping, multiple output formats

## Quick Start

```bash
# Basic clustering with 5 clusters
dtwc_cl -i data.csv -k 5

# Use TOML configuration file
dtwc_cl --config config.toml

# FastCLARA on a large dataset with GPU acceleration
dtwc_cl -i data.csv -k 10 --method clara --device cuda -v
```

## Command Reference

### Core Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input <path>` | Input file or folder. Supported: `.csv`, `.tsv`, `.parquet`, `.arrow`, `.ipc`, `.feather`, `.dtws` | — |
| `-o, --output <path>` | Output directory | `./results` |
| `--name <string>` | Problem name (used in output filenames) | `dtwc` |
| `-k, --clusters <int>` | Number of clusters | 3 |
| `-v, --verbose` | Verbose output | off |
| `--column <name>` | Parquet column to use as time series (required for Parquet single-file mode) | — |
| `--dtype <string>` | Data type for in-memory storage. Aliases: `--data-precision`, `--data-type`, `f32`, `fp32`, `float`, `f64`, `fp64`, `double` | `float32` |
| `--ram-limit <size>` | Memory budget, e.g. `2G`, `500M`, `128G` (parsed; used for chunked processing) | — |

### Clustering Method

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --method <string>` | Clustering method | `pam` |
| `--max-iter <int>` | Maximum iterations | 100 |
| `--n-init <int>` | Number of random restarts (PAM/kMedoids) | 1 |

Available methods: `pam`, `clara`, `kmedoids`, `mip`, `hierarchical` (alias: `hclust`).

### DTW Options

| Flag | Description | Default |
|------|-------------|---------|
| `-b, --band <int>` | Sakoe-Chiba band width (-1 = full DTW) | -1 |
| `--metric <string>` | Pointwise distance metric | `l1` |
| `--variant <string>` | DTW variant | `standard` |

Available metrics: `l1`, `squared_euclidean` (aliases: `sqeuclidean`, `l2sq`).

Available variants: `standard`, `ddtw`, `wdtw`, `adtw`, `softdtw` (alias: `soft-dtw`).

### DTW Variant Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--wdtw-g <float>` | WDTW logistic weight steepness | 0.05 |
| `--adtw-penalty <float>` | ADTW non-diagonal step penalty | 1.0 |
| `--sdtw-gamma <float>` | Soft-DTW smoothing parameter | 1.0 |

### CLARA-Specific Options

| Flag | Description | Default |
|------|-------------|---------|
| `--sample-size <int>` | Subsample size (-1 = auto) | -1 |
| `--n-samples <int>` | Number of independent subsamples | 5 |
| `--seed <int>` | Random seed for reproducibility | 42 |

### Hierarchical Clustering Options

| Flag | Description | Default |
|------|-------------|---------|
| `--linkage <string>` | Linkage criterion: `single`, `complete`, `average` | `average` |

### MIP Solver Options

| Flag | Description | Default |
|------|-------------|---------|
| `--solver <string>` | MIP solver: `highs`, `gurobi` | `highs` |
| `--mip-gap <float>` | Optimality gap tolerance | 1e-5 |
| `--time-limit <int>` | Solver time limit in seconds (-1 = unlimited) | -1 |
| `--no-warm-start` | Disable FastPAM warm start | off |
| `--numeric-focus <int>` | Gurobi NumericFocus (0-3) | 1 |
| `--mip-focus <int>` | Gurobi MIPFocus (0-3) | 2 |
| `--verbose-solver` | Show MIP solver log output | off |
| `--benders <string>` | Benders decomposition: `auto`, `on`, `off` | `auto` |

### CSV Parsing

| Flag | Description | Default |
|------|-------------|---------|
| `--skip-rows <int>` | Number of header rows to skip | 0 |
| `--skip-cols <int>` | Number of leading columns to skip | 0 |

### Distance Matrix and Checkpointing

| Flag | Description | Default |
|------|-------------|---------|
| `--dist-matrix <path>` | Path to precomputed distance matrix CSV | — |
| `--checkpoint <path>` | Checkpoint directory for save/resume | — |
| `--resume` | Resume from checkpoint (distance matrix cache + clustering state) | off |
| `--mmap-threshold <int>` | N above which to use memory-mapped distance matrix (0=always) | 50000 |

### GPU Options

| Flag                       | Description                                  | Default |
|----------------------------|----------------------------------------------|---------|
| `-d, --device <string>`    | Compute device: `cpu`, `cuda`, `cuda:N`      | `cpu`   |
| `--gpu-precision <string>` | GPU kernel precision. Alias: `--gpu-dtype`. Values: `auto`, `fp32`/`f32`/`float32`, `fp64`/`f64`/`float64`/`double` | `auto` |

### Configuration Files

| Flag | Description |
|------|-------------|
| `--config <path>` | TOML configuration file (CLI11 native) |
| `--yaml-config <path>` | YAML configuration file (requires `-DDTWC_ENABLE_YAML=ON`) |

See [Configuration Files](configuration.md) for full details and examples.

## Output Files

The CLI writes the following files to the output directory:

| File | Content |
|------|---------|
| `<name>_labels.csv` | Point name and cluster assignment |
| `<name>_medoids.csv` | Cluster ID, medoid index, and medoid name |
| `<name>_silhouettes.csv` | Point name, cluster, and silhouette score |
| `<name>_distmatrix.csv` | Full pairwise distance matrix |

## Examples

### Basic k-medoids clustering

```bash
dtwc_cl -i data.csv -k 5 --method pam --max-iter 200
```

### DDTW with Sakoe-Chiba band

```bash
dtwc_cl -i data.csv -k 3 --variant ddtw --band 10
```

### Weighted DTW with custom steepness

```bash
dtwc_cl -i data.csv -k 5 --variant wdtw --wdtw-g 0.1
```

### Scalable clustering with FastCLARA

```bash
dtwc_cl -i large_dataset.csv -k 20 --method clara --sample-size 500 --n-samples 10
```

### Hierarchical clustering with single linkage

```bash
dtwc_cl -i data.csv -k 5 --method hierarchical --linkage single
```

### MIP exact solution with Gurobi

```bash
dtwc_cl -i data.csv -k 3 --method mip --solver gurobi --mip-gap 1e-6 --time-limit 300
```

### GPU-accelerated distance matrix

```bash
dtwc_cl -i data.csv -k 5 --device cuda --gpu-precision fp32 -v
```

### Using a TOML configuration file

```bash
dtwc_cl --config config.toml
```

### Checkpoint and resume

```bash
# Start with checkpointing
dtwc_cl -i data.csv -k 5 --checkpoint ./checkpoints

# If interrupted, resume from the same checkpoint
dtwc_cl -i data.csv -k 5 --checkpoint ./checkpoints --resume
```

All flags are case-insensitive for enum values (e.g., `--method PAM` works the same as `--method pam`).
