---
title: Configuration Files
weight: 9
---

# Configuration Files

`dtwc_cl` supports native CLI11 TOML configuration.

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
| Device/storage | `--device`, `--gpu-precision`, `--gpu-dtype`, `--dist-matrix`, `--checkpoint`, `--checkpoint-interval`, `--resume`, `--mmap-threshold` |
| Diagnostics | `--verbose` |

Use canonical keys (`dtype`, `gpu-precision`) rather than their aliases in new
files. The repository's complete example is
[`examples/cpp/config.toml`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/examples/cpp/config.toml).

`checkpoint` enables the dense CSV distance checkpoint, which loads on startup
and saves on completion. `checkpoint-interval` additionally publishes a
generation every N completed distance-matrix rows and requires `checkpoint`.
The separate `resume` key maps the live `--resume`
flag: it validates and replays the completed binary result selected by the same
`output` and `name`, restores all result fields, skips clustering, and preserves
the binary file. It is not algorithm-state continuation, does not add
`max-iter`, and assumes the same input order/configuration because binary v1 has
no semantic fingerprint. Missing or incompatible state is a hard error.

These three flags control the command invocation rather than the clustering
payload and are passed on the command line: `--help`, `--version`, and
`--config`.

Current method values are `auto`, `pam`, `onebatch`, `clara`, `kmedoids`,
`mip`, `lrcore`, `hierarchical`, and `tadpole`. Current variants are
`standard`, `ddtw`, `wdtw`, `adtw`, `softdtw`, `msm`, and `twe`.
