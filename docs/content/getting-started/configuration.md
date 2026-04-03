---
title: Configuration Files
weight: 9
---

# Configuration Files

The DTWC++ command-line tool supports configuration files so you can avoid repeating long option lists. All CLI flags can be specified as configuration keys.

## TOML configuration

TOML is the default configuration format, supported natively through CLI11. No extra build flags are required.

```bash
dtwc_cl --config config.toml
```

Command-line flags override values from the configuration file, so you can use a config file for defaults and override specific options on the command line:

```bash
dtwc_cl --config config.toml -k 10 --method clara
```

### Full TOML example

```toml
# DTWC++ Configuration File
#
# Usage: dtwc_cl --config examples/config.toml
#
# All options can also be set via command line flags.
# Command line flags override values from this file.

# Input CSV file or folder containing time series data.
# Each row is one time series; columns are time steps.
input = "data/dummy"

# Output directory for results (labels, medoids, silhouettes, distance matrix).
output = "./results"

# Problem name used in output filenames (e.g., "myrun_labels.csv").
name = "example"

# Number of clusters.
clusters = 5

# Clustering method: "pam", "clara", "kmedoids", "mip"
method = "pam"

# Sakoe-Chiba band width. -1 = full (unconstrained) DTW.
band = -1

# Distance metric: "l1", "squared_euclidean"
metric = "l1"

# DTW variant: "standard", "ddtw", "wdtw", "adtw", "softdtw"
variant = "standard"

# Maximum iterations for iterative algorithms.
max-iter = 100

# Number of random restarts (PAM/kMedoids Lloyd).
n-init = 1

# CSV parsing: rows and columns to skip (e.g., for headers or index columns).
skip-rows = 0
skip-cols = 0

# Verbose output.
verbose = true

# --- DTW variant parameters ---
# WDTW logistic weight steepness (only used when variant = "wdtw").
wdtw-g = 0.05

# ADTW non-diagonal step penalty (only used when variant = "adtw").
adtw-penalty = 1.0

# Soft-DTW smoothing parameter (only used when variant = "softdtw").
sdtw-gamma = 1.0

# --- CLARA-specific options (only used when method = "clara") ---
# Subsample size. -1 = auto (40 + 2*k).
sample-size = -1

# Number of subsamples to try.
n-samples = 5

# Random seed for CLARA reproducibility.
seed = 42

# --- Checkpointing ---
# Uncomment to enable checkpoint save/resume:
# checkpoint = "./checkpoints"

# --- MIP solver (only used when method = "mip") ---
# Solver: "highs", "gurobi"
solver = "highs"

# MIP optimality gap tolerance (default: 1e-5).
mip-gap = 1e-5

# MIP solver time limit in seconds (-1 = unlimited).
time-limit = -1

# Gurobi NumericFocus (0-3, default: 1). Higher = more numeric care, slower.
numeric-focus = 1

# Gurobi MIPFocus (0=balanced, 1=find feasible, 2=prove optimal, 3=bound).
mip-focus = 2

# Show detailed MIP solver log output.
verbose-solver = false

# Uncomment to disable FastPAM warm start for MIP:
# no-warm-start = true

# --- Compute device ---
# "cpu" (default), "cuda" (GPU), or "cuda:N" for specific GPU device.
# GPU mode computes the distance matrix on the GPU before clustering.
# Only works with variant = "standard". Requires CUDA build.
device = "cpu"

# GPU precision: "auto" (FP32 on consumer GPUs, FP64 on HPC GPUs),
# "fp32" (fastest, ~1e-5 relative error), "fp64" (bit-identical to CPU).
precision = "auto"

# --- Precomputed distance matrix ---
# Uncomment to load a precomputed distance matrix:
# dist-matrix = "path/to/distance_matrix.csv"
```

## YAML configuration

YAML configuration requires building with the optional yaml-cpp dependency:

```bash
cmake .. -DDTWC_ENABLE_YAML=ON
```

Then use the `--yaml-config` flag:

```bash
dtwc_cl --yaml-config config.yaml
```

As with TOML, command-line flags take precedence over values in the YAML file.

### Full YAML example

```yaml
# DTWC++ Configuration File (YAML format)
#
# Usage: dtwc_cl --yaml-config examples/config.yaml
# Requires: build with -DDTWC_ENABLE_YAML=ON
#
# All options can also be set via command line flags.
# Command line flags override values from this file.
# Key names use kebab-case, matching CLI flags and TOML keys.

# Input CSV file or folder containing time series data.
input: data/dummy

# Output directory for results.
output: ./results

# Problem name used in output filenames.
name: example

# Number of clusters.
clusters: 5

# Clustering method: pam, clara, kmedoids, mip, hierarchical
method: pam

# Sakoe-Chiba band width. -1 = full (unconstrained) DTW.
band: -1

# Distance metric: l1, squared_euclidean
metric: l1

# DTW variant: standard, ddtw, wdtw, adtw, softdtw
variant: standard

# Maximum iterations for iterative algorithms.
max-iter: 100

# Number of random restarts (PAM/kMedoids Lloyd).
n-init: 1

# CSV parsing.
skip-rows: 0
skip-cols: 0

# Verbose output.
verbose: true

# DTW variant parameters.
wdtw-g: 0.05
adtw-penalty: 1.0
sdtw-gamma: 1.0

# CLARA-specific options (only used when method = "clara").
sample-size: -1
n-samples: 5
seed: 42

# MIP solver (only used when method = "mip").
solver: highs
mip-gap: 1.0e-5
time-limit: -1
numeric-focus: 1
mip-focus: 2
verbose-solver: false
# no-warm-start: true  # Uncomment to disable FastPAM warm start

# Hierarchical linkage (only used when method = "hierarchical").
linkage: average

# Compute device.
device: cpu
precision: auto
```

## CLI flag reference

The following table maps CLI flags to configuration file keys. All keys use kebab-case in both TOML and YAML.

### Core options

| CLI flag | Config key | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `-i`, `--input` | `input` | string | required | Input CSV file or folder |
| `-o`, `--output` | `output` | string | `"./results"` | Output directory |
| `--name` | `name` | string | `"dtwc"` | Problem name for output filenames |
| `-k`, `--clusters` | `clusters` | int | `3` | Number of clusters |
| `-m`, `--method` | `method` | string | `"pam"` | Clustering method: `pam`, `clara`, `kmedoids`, `mip`, `hierarchical` |
| `-b`, `--band` | `band` | int | `-1` | Sakoe-Chiba band width (`-1` = full DTW) |
| `--metric` | `metric` | string | `"l1"` | Distance metric: `l1`, `squared_euclidean` |
| `--variant` | `variant` | string | `"standard"` | DTW variant: `standard`, `ddtw`, `wdtw`, `adtw`, `softdtw` |
| `--max-iter` | `max-iter` | int | `100` | Maximum iterations |
| `--n-init` | `n-init` | int | `1` | Number of random restarts |
| `-v`, `--verbose` | `verbose` | bool | `false` | Verbose output |

### CSV parsing

| CLI flag | Config key | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `--skip-rows` | `skip-rows` | int | `0` | Header rows to skip |
| `--skip-cols` | `skip-cols` | int | `0` | Leading columns to skip |

### DTW variant parameters

| CLI flag | Config key | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `--wdtw-g` | `wdtw-g` | float | `0.05` | WDTW logistic weight steepness |
| `--adtw-penalty` | `adtw-penalty` | float | `1.0` | ADTW non-diagonal step penalty |
| `--sdtw-gamma` | `sdtw-gamma` | float | `1.0` | Soft-DTW smoothing parameter |

### CLARA options

| CLI flag | Config key | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `--sample-size` | `sample-size` | int | `-1` | Subsample size (`-1` = auto) |
| `--n-samples` | `n-samples` | int | `5` | Number of subsamples |
| `--seed` | `seed` | int | `42` | Random seed |

### Hierarchical clustering

| CLI flag | Config key | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `--linkage` | `linkage` | string | `"average"` | Linkage: `single`, `complete`, `average` |

### MIP solver

| CLI flag | Config key | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `--solver` | `solver` | string | `"highs"` | MIP solver: `highs`, `gurobi` |
| `--mip-gap` | `mip-gap` | float | `1e-5` | Optimality gap tolerance |
| `--time-limit` | `time-limit` | int | `-1` | Time limit in seconds (`-1` = unlimited) |
| `--no-warm-start` | `no-warm-start` | bool | `false` | Disable FastPAM warm start |
| `--numeric-focus` | `numeric-focus` | int | `1` | Gurobi NumericFocus (0-3) |
| `--mip-focus` | `mip-focus` | int | `2` | Gurobi MIPFocus (0-3) |
| `--verbose-solver` | `verbose-solver` | bool | `false` | Show MIP solver log |

### GPU and I/O

| CLI flag | Config key | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `-d`, `--device` | `device` | string | `"cpu"` | Compute device: `cpu`, `cuda`, `cuda:N` |
| `--precision` | `precision` | string | `"auto"` | GPU precision: `auto`, `fp32`, `fp64` |
| `--dist-matrix` | `dist-matrix` | string | | Path to precomputed distance matrix CSV |
| `--checkpoint` | `checkpoint` | string | | Checkpoint directory for save/resume |
