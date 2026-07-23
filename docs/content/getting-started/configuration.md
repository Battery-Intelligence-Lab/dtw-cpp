---
title: Configuration Files
weight: 9
---

# Configuration Files

`dtwc_cl` supports native CLI11 TOML configuration and an optional YAML subset.
They do not have identical coverage or precedence, so choose the format
deliberately.

The [live CLI reference](../cli/) is the source of truth for option semantics,
accepted values, defaults, and aliases.

## TOML

TOML support needs no optional dependency:

```bash
dtwc_cl --config examples/cpp/config.toml
```

For ordinary options, the TOML key is the canonical long flag without the
leading `--`; for example, `--n-clusters 5` becomes `n-clusters = 5`.
Command-line values override TOML values. Deprecated keys `clusters` and
`restart` remain accepted with warnings, but new files should use
`n-clusters` and `resume`.

Canonical keys and aliases represented by the live CLI are:

| Area | Long flags / TOML keys |
|---|---|
| Input/output | `--input`, `--output`, `--name`, `--column`, `--dtype`, `--data-precision`, `--data-type`, `--skip-rows`, `--skip-cols`, `--ram-limit` |
| Clustering | `--n-clusters`, `--method`, `--max-iter`, `--n-init`, `--seed` |
| Distances | `--band`, `--metric`, `--variant`, `--missing-strategy`, `--mv-mode`, `--wdtw-g`, `--adtw-penalty`, `--sdtw-gamma`, `--msm-c`, `--twe-nu`, `--twe-lambda` |
| FastCLARA/OneBatch/TADPole | `--sample-size`, `--n-samples`, `--batch-size`, `--batch-weighting`, `--dc` |
| Hierarchical | `--linkage` |
| Exact solvers | `--solver`, `--mip-gap`, `--time-limit`, `--no-warm-start`, `--numeric-focus`, `--mip-focus`, `--verbose-solver`, `--benders` |
| Device/storage | `--device`, `--gpu-precision`, `--gpu-dtype`, `--dist-matrix`, `--checkpoint`, `--resume`, `--mmap-threshold` |
| Diagnostics | `--verbose` |

Use canonical keys (`dtype`, `gpu-precision`) rather than their aliases in new
files. The repository's complete example is
[`examples/cpp/config.toml`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/examples/cpp/config.toml).

`checkpoint` enables the dense CSV distance checkpoint, which loads on startup
and saves on completion. The separate `resume` key maps the live `--resume`
flag, but the current CLI only reads and reports its binary result checkpoint;
it does not restore algorithm state from that object. Do not rely on `resume`
until the open CLI defect is repaired.

These four flags control the command invocation rather than the clustering
payload and are passed on the command line: `--help`, `--version`, `--config`,
and `--yaml-config`.

Current method values are `auto`, `pam`, `onebatch`, `clara`, `kmedoids`,
`mip`, `lrcore`, `hierarchical`, and `tadpole`. Current variants are
`standard`, `ddtw`, `wdtw`, `adtw`, `softdtw`, `msm`, and `twe`.

## YAML

YAML requires yaml-cpp:

```bash
cmake -S . -B build -DDTWC_ENABLE_YAML=ON
dtwc_cl --yaml-config examples/cpp/config.yaml
```

Build the configured tree using the platform recipe in the installation guide
before invoking `dtwc_cl`.

The YAML loader currently runs after CLI11 parsing and overwrites a supported
command-line value when the same key is present. This is a known precedence
bug. Until it is fixed, treat YAML as the source of truth for keys present in
the YAML file.

The manually mapped canonical YAML keys are:

| Area | YAML keys |
|---|---|
| Core | `input`, `output`, `name`, `n-clusters`, `method`, `band`, `metric`, `variant`, `max-iter`, `n-init`, `verbose` |
| Device/storage | `device`, `dtype`, `ram-limit`, `gpu-precision`, `resume` |
| Distance semantics | `wdtw-g`, `adtw-penalty`, `sdtw-gamma`, `msm-c`, `twe-nu`, `twe-lambda`, `mv-mode`, `missing-strategy` |
| Sampling/hierarchy | `sample-size`, `n-samples`, `seed`, `linkage` |
| Solver | `solver`, `mip-gap`, `time-limit`, `no-warm-start`, `numeric-focus`, `mip-focus`, `verbose-solver` |

Deprecated YAML keys `clusters` and `restart` are accepted with warnings; the
canonical key wins when both are present.

YAML does not currently map `column`, CSV skip controls, OneBatchPAM controls,
`dc`, `benders`, `checkpoint`, `dist-matrix`, or `mmap-threshold`. Supply those
through the CLI/TOML route instead of assuming the YAML reader consumed them.
The repository example is
[`examples/cpp/config.yaml`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/examples/cpp/config.yaml).
