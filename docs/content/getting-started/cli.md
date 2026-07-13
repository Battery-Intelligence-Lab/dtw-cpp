---
title: Command Line Interface (CLI)
weight: 2
---

# Command Line Interface (CLI)

DTW-C++ provides a full-featured CLI tool for time series clustering. After compiling the software using the [installation instructions](installation.md), run the `bin/dtwc_cl` executable.

## Features

- **Multiple clustering methods**: FastPAM, OneBatchPAM, FastCLARA, Lloyd's k-medoids, MIP, LR-core, hierarchical, and TADPole
- **DTW variants**: Standard, DDTW, WDTW, ADTW, Soft-DTW, MSM, and TWE
- **Distance metrics**: L1 (default) and squared Euclidean
- **GPU acceleration**: CUDA support for distance matrix computation
- **Configuration files**: TOML (native) and YAML (optional) configuration support
- **Checkpointing**: Save and resume distance matrix computation
- **Flexible I/O**: CSV/TSV plus optional Arrow/Parquet input, configurable row/column skipping, and stable CSV outputs

## Quick Start

```bash
# Basic clustering with 5 clusters
dtwc_cl -i data.csv -k 5

# Use TOML configuration file
dtwc_cl --config config.toml

# Matrix-free FastCLARA on a large dataset
dtwc_cl -i data.csv -k 10 --method clara --device cpu -v
```

## Command Reference

### Core Options

| Flag | Description | Default |
|------|-------------|---------|
| `-h, --help` | Print the live command reference | — |
| `--version` | Print the version from the repository `VERSION` source of truth | — |
| `-i, --input <path>` | Input file or folder. CSV/TSV and `.dtws` are core; Parquet/Arrow IPC/Feather require an Arrow-enabled build | — |
| `-o, --output <path>` | Output directory | `./results` |
| `--name <string>` | Problem name (used in output filenames) | `dtwc` |
| `-k, --n-clusters <int>` | Number of clusters | 3 |
| `-v, --verbose` | Verbose output | off |
| `--column <name>` | Parquet scalar/list Float32 or Float64 column. If omitted, the first eligible top-level column is selected | — |
| `--dtype <string>` | Data type for in-memory storage. Flag aliases: `--data-precision`, `--data-type`; value aliases include `f32`, `fp32`, `float`, `f64`, `fp64`, `double` | `float64` |
| `--ram-limit <size>` | Conservative Parquet series-materialization budget, e.g. `2GiB`, `500M`, `1.5G`; see below | unlimited |

### Clustering Method

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --method <string>` | Clustering method | `pam` |
| `--max-iter <int>` | Maximum iterations | 100 |
| `--n-init <int>` | Number of random restarts (PAM/kMedoids) | 1 |

Available methods: `auto`, `pam`, `onebatch` (alias `obp`), `clara`,
`kmedoids`, `mip`, `lrcore` (alias `lr`), `hierarchical` (alias `hclust`),
and `tadpole`.

### DTW Options

| Flag | Description | Default |
|------|-------------|---------|
| `-b, --band <int>` | Sakoe-Chiba band width (-1 = full DTW) | -1 |
| `--metric <string>` | Pointwise distance metric | `l1` |
| `--variant <string>` | DTW variant | `standard` |
| `--missing-strategy <string>` | Missing-data strategy: `error`, `zero_cost`, `arow`, `interpolate` (aliases: `zero-cost`, `zerocost`) | `error` |

Available metrics: `l1`, `squared_euclidean` (aliases: `sqeuclidean`, `l2sq`).

Available variants: `standard`, `ddtw`, `wdtw`, `adtw`, `softdtw` (alias
`soft-dtw`), `msm`, and `twe`.

### DTW Variant Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--wdtw-g <float>` | WDTW logistic weight steepness | 0.05 |
| `--adtw-penalty <float>` | ADTW non-diagonal step penalty | 1.0 |
| `--sdtw-gamma <float>` | Soft-DTW smoothing parameter | 1.0 |
| `--msm-c <float>` | MSM split/merge cost | 1.0 |
| `--twe-nu <float>` | TWE stiffness | 0.001 |
| `--twe-lambda <float>` | TWE edit penalty | 1.0 |
| `--mv-mode <string>` | Multivariate mode: `dependent`, `independent` | `dependent` |

### Sampling-method options

| Flag | Description | Default |
|------|-------------|---------|
| `--sample-size <int>` | Subsample size (-1 = auto) | -1 |
| `--n-samples <int>` | Number of independent subsamples | 5 |
| `--seed <int>` | Invocation-local seed for PAM, OneBatchPAM, and CLARA | 42 |

The default seed is identical across the C++, Python, MATLAB, sklearn, and CLI
seed-aware routes. Supplying `--seed` does not consume the legacy process-global
FastPAM engine. Valid CLI seeds are integers from 0 through `UINT_MAX`.

### RAM-limited Parquet FastCLARA

For Parquet input, a nonzero `--ram-limit` is applied before the selected
column payload is loaded. The CLI reads schema and row-group metadata, resolves
`--method auto` from the logical series count, and estimates the peak needed to
decode and materialise the selected column at the requested `--dtype`. If that
estimate fits, the ordinary resident reader is used. If it does not fit, the
only supported streaming route is non-full FastCLARA over one Parquet file
whose selected column is `List<Float32/Float64>` or
`LargeList<Float32/Float64>` with one list cell per series.

The cap is fail-closed:

- over-budget scalar-column, Parquet-directory, and non-CLARA requests stop
  before payload materialisation and explain how to convert the input or raise
  the limit;
- Parquet row groups are indivisible. A sample or assignment row group that
  cannot fit beside retained series fails with guidance to rewrite smaller row
  groups or raise the limit;
- a CLARA sample size that resolves to all N series is rejected while the file
  is over budget, because its full-data PAM fallback cannot stream;
- non-full FastCLARA is a CPU matrix-free schedule, so `--device cuda` is
  rejected before the Parquet payload is read; and
- `--dtype f32` keeps the sample, medoid, and assignment chunks in Float32;
  distances and the accumulated objective remain double precision.

`--ram-limit` is a conservative cap for series decoding/materialisation, not a
hard operating-system RSS limit: algorithm result arrays, the subsample PAM
matrix, library metadata, and fixed process overhead are outside it. Units are
binary and case-insensitive: `K`/`KB`/`KiB` through `T`/`TB`/`TiB`. A decimal is
accepted only when it resolves exactly to a whole number of bytes. Zero means
unlimited; malformed, negative, fractional-byte, or overflowing values are
errors rather than silently disabling the limit.

The cap governs Parquet series materialisation and nothing else. No other reader
can honour it, so a nonzero `--ram-limit` on CSV/TSV, HDF5, Arrow IPC, `.dtws`,
or a CSV directory is a hard error, not a warning: those formats materialise
their series unconditionally, and accepting the flag would report a budget that
is never applied. Drop the flag, or convert the series to a list-per-row Parquet
file to stream them under the cap.

### OneBatchPAM and TADPole options

| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size <int>` | OneBatchPAM objective batch size (-1 = logarithmic auto) | -1 |
| `--batch-weighting <string>` | `uniform`, `debiased`, or nearest-neighbour weighting `nniw` | `nniw` |
| `--dc <float>` | TADPole density cutoff (omitted/negative = deterministic auto-selection) | auto |

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

TADPole's pruning schedule avoids eagerly filling all pairs, but every
exact/fallback distance still uses the packed cache. It therefore follows
`--mmap-threshold` just like matrix-based methods. OneBatchPAM is the exception:
its fixed O(Nm) table never allocates the Problem distance matrix. If the
threshold is reached in a binary built without LLFIO, the CLI exits before a
heap allocation and tells you to enable LLFIO, raise the threshold only when the
packed matrix fits in RAM, or select `onebatch`.

Non-full FastCLARA is also parent-matrix-free: only its current subsample PAM
owns an O(s²) matrix, while full-data assignment evaluates N×k configured DTW
distances directly. It therefore rejects `--checkpoint` and `--dist-matrix`,
which would otherwise load or save unused O(N²) state. If `--sample-size`
resolves to N, FastCLARA deliberately becomes one full-data PAM run and the
ordinary distance-storage/checkpoint rules apply. RAM-limited streaming always
uses a non-full sample and still writes the automatic binary clustering-result
checkpoint.

The mmap cache resumes automatically only when its version-2 fingerprint matches
the exact data and distance configuration. A legacy version-1 or mismatched cache
fails loudly and must be deleted/renamed and recomputed. When the threshold
selects mmap, `--checkpoint` and `--dist-matrix` are incompatible because they
require a dense CSV matrix; the CLI rejects the combination before opening
either path. CUDA mmap runs must select explicit `--gpu-precision fp32` or
`fp64`; the hardware-dependent `auto` setting is not a stable cache identity.

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
| `<name>_silhouettes.csv` | Point name, cluster, and silhouette score (only when a full distance matrix is materialised) |
| `<name>_distance_matrix.csv` | Full pairwise distance matrix (only when materialised) |
| `<name>_checkpoint.bin` | Automatic binary clustering-result checkpoint |

RAM-limited Parquet streaming writes labels, medoids, and the binary result
checkpoint, but deliberately does not materialise the dense matrix merely to
produce distance or silhouette CSVs. List rows use the stable names
`series_0`, `series_1`, and so on. For the same seed/configuration, streamed and
resident list-column runs produce byte-identical label, medoid, and binary
checkpoint files.

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

### RAM-limited Parquet FastCLARA

```bash
# `series` is List<Float32/Float64> or LargeList<Float32/Float64>, one row per series
dtwc_cl -i large_dataset.parquet --column series -k 20 --method clara \
  --sample-size 500 --n-samples 10 --ram-limit 2GiB
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
