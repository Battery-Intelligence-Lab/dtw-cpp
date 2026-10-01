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
- **GPU acceleration**: the distance matrix on CUDA (NVIDIA) or Metal (macOS), with `--device gpu`
- **Configuration files**: TOML or YAML (CLI11 native); `--print-config` writes the settings as a file `--config` reads
- **Checkpointing**: Save and resume distance matrix computation
- **Flexible I/O**: CSV/TSV plus optional Arrow/Parquet input, configurable row/column skipping, and stable CSV outputs

## Quick Start

```bash
# Basic clustering with 5 clusters
dtwc_cl -i data.csv -k 5

# Use a TOML or YAML configuration file
dtwc_cl --config config.toml
dtwc_cl --config config.yaml

# Matrix-free FastCLARA on a large dataset
dtwc_cl -i data.csv -k 10 --method clara --device cpu -v

# Write the settings to a file, and run that file later (flags still beat it)
dtwc_cl -i data.csv -k 10 --method clara --print-config > job.toml
dtwc_cl --config job.toml
```

`dtwc_cl` reads its flags and any `--config` file into one `dtwc::Config` and runs
it with `dtwc::run`, the pipeline the C++ Tier-1 `dtwc::cluster` also runs, so the
two agree on every default, check and method choice.

## Command Reference

### Core Options

| Flag | Description | Default |
|------|-------------|---------|
| `-h, --help` | Print the live command reference | — |
| `--version` | Print the version from the repository `VERSION` source of truth | — |
| `-i, --input <path>` | Input file or folder. CSV/TSV are core; Parquet/Arrow IPC/Feather require an Arrow-enabled build | — |
| `-o, --output <path>` | Output directory (`""` writes no file) | `./results` |
| `--name <string>` | Problem name (used in output filenames) | the input's file or folder name |
| `-k, --n-clusters <int>` | Number of clusters (required) | — |
| `-v, --verbose` | Verbose output | off |
| `--column <name>` | Parquet scalar/list Float32 or Float64 column. If omitted, the first eligible top-level column is selected | — |
| `--dtype <string>` | Data type for in-memory storage. Flag aliases: `--data-precision`, `--data-type`; value aliases include `f32`, `fp32`, `float`, `f64`, `fp64`, `double` | `float64` |
| `--ram-limit <size>` | Conservative Parquet series-materialization budget, e.g. `2GiB`, `500M`, `1.5G`; see below | unlimited |

The option names of DTW-C++ 1.0 (Nc, probName, in, out, skipRows, skipCols, maxIter, Nrep, bandwidth, distMat and
their other spellings) still work, hidden from `--help`: each prints one warning naming its 2.0 flag. Its range of
cluster counts, i..j, is refused: run `dtwc_cl` once per k.

### Clustering Method

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --method <string>` | Clustering method | `auto` |
| `--max-iter <int>` | Maximum iterations (at least 1; `0` exits 1 before the data is read) | 100 |
| `--n-init <int>` | Number of random restarts (PAM/kMedoids; at least 1) | 1 |

Available methods: `auto`, `pam`, `onebatch` (alias `obp`), `clara`,
`kmedoids`, `mip`, `lrcore` (alias `lr`), `hierarchical` (alias `hclust`),
and `tadpole`. On `--device cpu`, `auto` runs `pam` for up to 5,000 series and
`clara` above that; on `--device gpu` it runs `pam` at any size, since the GPU
fills the matrix PAM reads.

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
FastPAM engine. Valid CLI seeds are integers from 0 through `UINT64_MAX`.

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
- non-full FastCLARA is a CPU matrix-free schedule, so `--device gpu` is
  rejected before the Parquet payload is read; and
- `--dtype f32` keeps the sample, medoid, and assignment chunks in Float32;
  DTW recurrence arithmetic is Float32, while returned distances and the
  accumulated objective are stored in double.

`--ram-limit` is a conservative cap for series decoding/materialisation, not a
hard operating-system RSS limit: algorithm result arrays, the subsample PAM
matrix, library metadata, and fixed process overhead are outside it. Units are
binary and case-insensitive: `K`/`KB`/`KiB` through `T`/`TB`/`TiB`. A fraction
of a byte rounds up, so a nonzero value never disables the limit. Zero means
unlimited; malformed, negative, or overflowing values are errors.

The cap governs Parquet series materialisation and nothing else. No other reader
can honour it, so a nonzero `--ram-limit` on CSV/TSV, HDF5, Arrow IPC,
or a CSV directory is a hard error, not a warning: those formats materialise
their series unconditionally, and accepting the flag would report a budget that
is never applied. Drop the flag, or convert the series to a list-per-row Parquet
file to stream them under the cap.

### OneBatchPAM and TADPole options

| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size <int>` | OneBatchPAM objective batch size (-1 = logarithmic auto, raised to k); an explicit size below k is an error | -1 |
| `--dc <float>` | TADPole density cutoff (omitted/negative = deterministic auto-selection) | auto |

### Hierarchical Clustering Options

| Flag | Description | Default |
|------|-------------|---------|
| `--linkage <string>` | Linkage criterion: `single`, `complete`, `average` | `average` |

### MIP Solver Options

| Flag | Description | Default |
|------|-------------|---------|
| `--solver <string>` | MIP solver: `highs`, `gurobi`. On a build without Gurobi, `gurobi` exits 1 naming the flag rather than solving with HiGHS | `highs` |
| `--mip-gap <float>` | Optimality gap tolerance | 1e-5 |
| `--time-limit <int>` | Solver time limit in seconds (-1 = unlimited) | -1 |
| `--no-warm-start` | Disable FastPAM warm start | off |
| `--numeric-focus <int>` | Gurobi NumericFocus (0-3) | 1 |
| `--mip-focus <int>` | Gurobi MIPFocus (0-3) | 2 |
| `--verbose-solver` | Show MIP solver log output | off |
| `--lr-max-nodes <int>` | Branch-and-bound node cap of `--method lrcore` | 2000000 |

### CSV Parsing

| Flag | Description | Default |
|------|-------------|---------|
| `--skip-rows <int>` | Number of header rows to skip | 0 |
| `--skip-cols <int>` | Number of leading columns to skip | 0 |
| `--delimiter <char>` | Field delimiter, one character | inferred: tab for `.tsv`/`.txt`, else `,` |

These three apply to CSV/TSV input only; any of them on Parquet or Arrow IPC
input is an error rather than an option silently ignored.

### Distance Matrix and Checkpointing

| Flag | Description | Default |
|------|-------------|---------|
| `--dist-matrix <path>` | Path to precomputed distance matrix CSV | — |
| `--checkpoint <path>` | Checkpoint directory: `<name>.dtwm` is saved there and resumed from | — |
| `--checkpoint-interval <rows>` | Needs `--checkpoint`: save the checkpoint every N completed distance-matrix rows | 0 (save once, at the end) |
| `--mmap-threshold <int>` | N above which to use memory-mapped distance matrix (0=always) | 50000 |

The checkpoint is one file, `<dir>/<name>.dtwm` ([Checkpointing](checkpointing.md)). Without `--checkpoint-interval` (or with `0`) it is written once, after clustering. With a non-zero interval, `fill_distance_matrix` saves it after every `<rows>` completed matrix rows, so an interrupted run resumes from the last block instead of recomputing the whole matrix; it requires `--checkpoint <dir>` and exits 1 without it, before any data is read. A save of a matrix in RAM writes all N(N+1)/2 doubles, so choose an interval whose block (about `<rows>` * N DTW computations) costs much more than one save. A checkpoint for other data or settings, or a damaged one, stops the run with exit status 1 before anything is computed; it is never recomputed over.

A `--dist-matrix` file that cannot be loaded (missing, unreadable, empty, not square, not symmetric, or with a row count other than the number of input series) and a checkpoint that cannot be saved are errors: `dtwc_cl` exits 1 with a message naming the option and the path, rather than warning and carrying on. The `--checkpoint` directory is created, or found not to be a directory, before any data is read; the end-of-run save comes after the result files, so a save that fails there (a full disk) leaves the results written.

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
uses a non-full sample.

The mapped matrix is `<name>.dtwm`, in the `--checkpoint` directory when one is
given (it is then the checkpoint) and in the output directory otherwise. It
resumes automatically when its magic, version, exact length, N and SHA-256
fingerprint of the data and distance settings match; a file of other data or
settings is `InvalidInput` and a damaged or earlier-layout one `IOError`, both
before any cached distance is read. When the threshold selects mmap,
`--dist-matrix` (a CSV matrix) is rejected before either path is opened. CUDA
mmap and checkpoint runs must select explicit `--gpu-precision fp32` or `fp64`:
CUDA's `auto` depends on the GPU, so it is not a stable cache identity (Metal's
`auto` is FP32).

### GPU Options

| Flag                       | Description                                  | Default |
|----------------------------|----------------------------------------------|---------|
| `-d, --device <string>`    | Compute device: `cpu`, `gpu`, `gpu:N` (`cuda`, `cuda:N` are the same; CUDA on NVIDIA, Metal on macOS) | `cpu`   |
| `--gpu-precision <string>` | GPU kernel precision. Alias: `--gpu-dtype`. Values: `auto`, `fp32`/`f32`/`float32`, `fp64`/`f64`/`float64`/`double` | `auto` |

On `--device gpu` the GPU fills the distance matrix, so the methods that read one
run there: `pam`, `kmedoids`, `mip`, `lrcore`, `hierarchical`, and `clara` when its
sample covers every series. `onebatch`, `tadpole` and a `clara` sample smaller
than N compute on the CPU as they go, so they exit 1 on `gpu` naming the methods
that use it. A variant other than `standard`, a missing-data strategy, `--dtype
float32`, and on Metal a GPU index other than 0 or `--gpu-precision fp64`, exit 1
before the input is read; a build without a GPU backend refuses `gpu` naming the
build flag. `--device hpc` exits 1: `dtwc_cl` computes where it runs, and a SLURM
job is submitted with `bash scripts/slurm/slurm_remote.sh submit-cluster` or
Python's `dtwcpp.cluster(..., device="hpc")` ([SLURM](slurm.md)).

### Configuration Files

| Flag | Description |
|------|-------------|
| `--config <path>` | TOML or YAML configuration file (CLI11 native) |
| `--print-config` | Write every setting, one `key = value` line each, as a TOML file `--config` reads back, and exit |

See [Configuration Files](configuration.md) for full details and examples.

## Output Files

The CLI writes the following files to the output directory:

| File | Content |
|------|---------|
| `<name>_labels.csv` | Point name and cluster assignment |
| `<name>_medoids.csv` | Cluster ID, medoid index, and medoid name |
| `<name>_silhouettes.csv` | Point name, cluster, and silhouette score (only when a full distance matrix is materialised) |
| `<name>_distance_matrix.csv` | Full pairwise distance matrix (only when materialised) |

If an output file cannot be written in full (an unwritable directory, a full
disk, a file-size quota), `dtwc_cl` exits 1 and names the file; a file named in
that message is incomplete. A silhouette that is undefined (fewer than two clusters
realised) is only a warning; `-k 1` writes no silhouettes file.

RAM-limited Parquet streaming writes labels and medoids, but deliberately does
not materialise the dense matrix merely to produce distance or silhouette CSVs.
List rows use the stable names `series_0`, `series_1`, and so on. For the same
seed/configuration, streamed and resident list-column runs produce
byte-identical label and medoid files.

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
dtwc_cl -i data.csv -k 5 --device gpu --gpu-precision fp32 -v
```

### Using a TOML or YAML configuration file

```bash
dtwc_cl --config config.toml
dtwc_cl --config config.yaml
```

### Distance checkpoint

```bash
# Save <name>.dtwm to ./checkpoints and resume from it; a restart uses the same command
dtwc_cl -i data.csv -k 5 --checkpoint ./checkpoints
dtwc_cl -i data.csv -k 5 --checkpoint ./checkpoints
```

All flags are case-insensitive for enum values (e.g., `--method PAM` works the same as `--method pam`).
