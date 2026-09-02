---
title: Configuration Files
weight: 9
---

# Configuration Files

`dtwc_cl --config` reads a TOML or a YAML file. Both go through CLI11's own
configuration interface, so the two formats are interchangeable and share one
set of keys, defaults, validators, and deprecation warnings. The format is
detected from the file content, not the extension.

The [live CLI reference](../cli/) is the source of truth for option semantics,
accepted values, defaults, and aliases.

## Precedence

**A value given on the command line always wins over the same key in a
configuration file.** The file supplies only the options you did not pass
explicitly. A key that matches no live option is an error, not a comment: the
run stops instead of silently ignoring the setting.

## TOML

TOML support needs no optional dependency:

```bash
dtwc_cl --config examples/cpp/config.toml
```

For ordinary options, the TOML key is the canonical long flag without the
leading `--`; for example, `--n-clusters 5` becomes `n-clusters = 5`.
Deprecated keys `clusters` and `restart` remain accepted with warnings, but new
files should use `n-clusters` and `resume`.

## YAML

YAML needs the optional fkYAML dependency, enabled by default
(`-DDTWC_ENABLE_YAML=ON`). In a build configured with `-DDTWC_ENABLE_YAML=OFF`,
a YAML file is refused with `built without YAML support; use TOML` rather than
being misread:

```bash
dtwc_cl --config examples/cpp/config.yaml
```

Keys are spelled exactly as in TOML; only the syntax differs
(`n-clusters: 5` instead of `n-clusters = 5`). A YAML sequence supplies repeated
values for one option, and a nested mapping is read the way a TOML table is.
Anything with no CLI11 equivalent — a nested sequence, a non-mapping document,
several documents in one file — is rejected with the offending key path. The
example above mirrors
[`examples/cpp/config.toml`](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/examples/cpp/config.toml)
key for key.

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
