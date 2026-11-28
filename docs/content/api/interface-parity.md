---
title: Cross-language interface map
weight: 2
---

# Cross-language interface map

This page is the quickest way to compare the current C++, Python, MATLAB, and
CLI surfaces without reading each binding page independently.

## Interface diagram

<svg viewBox="0 0 980 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Cross-language interface flow">
  <style>
    .box { fill: #f8fafc; stroke: #0f172a; stroke-width: 2; rx: 14; ry: 14; }
    .lang { fill: #dbeafe; }
    .data { fill: #dcfce7; }
    .cluster { fill: #fef3c7; }
    .result { fill: #fee2e2; }
    .label { font: 600 18px sans-serif; fill: #0f172a; }
    .small { font: 500 14px sans-serif; fill: #334155; }
    .arrow { stroke: #334155; stroke-width: 2.5; fill: none; marker-end: url(#arrowhead); }
  </style>
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#334155" />
    </marker>
  </defs>

  <rect class="box lang" x="20" y="20" width="180" height="70" />
  <text class="label" x="44" y="48">CLI / C++ / Python / MATLAB</text>
  <text class="small" x="44" y="73">settings entry points</text>

  <rect class="box data" x="250" y="20" width="190" height="70" />
  <text class="label" x="309" y="48">Data</text>
  <text class="small" x="280" y="73">arrays, matrices, files</text>

  <rect class="box" x="490" y="20" width="180" height="70" />
  <text class="label" x="546" y="48">Problem</text>
  <text class="small" x="515" y="73">variant, band, metric</text>

  <rect class="box cluster" x="720" y="20" width="220" height="70" />
  <text class="label" x="784" y="48">Algorithms</text>
  <text class="small" x="744" y="73">FastPAM, CLARA, hierarchy</text>

  <rect class="box result" x="720" y="145" width="220" height="70" />
  <text class="label" x="797" y="173">Outputs</text>
  <text class="small" x="751" y="198">scores, labels, medoids, files</text>

  <rect class="box" x="490" y="145" width="180" height="70" />
  <text class="label" x="514" y="173">distance.* facade</text>
  <text class="small" x="514" y="198">pairwise DTW helpers</text>

  <path class="arrow" d="M200 55 H250" />
  <path class="arrow" d="M440 55 H490" />
  <path class="arrow" d="M670 55 H720" />
  <path class="arrow" d="M580 90 V145" />
  <path class="arrow" d="M670 180 H720" />
</svg>

## Generated API reference

Use the generated Doxygen site for the structural view:

- [Full class index](../../Doxygen/annotated.html)
- [Full namespace index](../../Doxygen/namespaces.html)
- [C++ `dtwc::Problem` class page with collaboration/call graphs](../../Doxygen/classdtwc_1_1_problem.html)
- [C++ `dtwc::distance` namespace](../../Doxygen/namespacedtwc_1_1distance.html)
- [Python `dtwcpp.distance` namespace](../../Doxygen/namespacedtwcpp_1_1distance.html)
- [Python `DTWClustering` class page](../../Doxygen/classdtwcpp_1_1__clustering_1_1_d_t_w_clustering.html)
- [MATLAB `Problem.m` page](../../Doxygen/_problem_8m.html)

The generated Python reference is limited to the checked-in Python sources.
`dtwcpp.Problem` comes from the extension module, so Doxygen does not emit a
standalone Python class page for it. Collaboration and call graphs are linked
from the generated class pages when Graphviz is available in the documentation
build.

## Shared flow

The CLI and `cluster()` in C++, Python and MATLAB are config-first: their settings become a `dtwc::Config`
that `dtwc::run` runs. The Tier-2 APIs (`Problem`, the algorithms, the estimators) are Problem-first:

```text
settings -> data -> Problem -> clustering algorithm -> scores / outputs
```

## Surface map

| Concern | CLI | C++ | Python | MATLAB |
|---------|-----|-----|--------|--------|
| Config file input | TOML or YAML (`--config`; `--print-config` writes one) | `dtwc::Config` read by `cli::bind` / `parse_config`, run by `dtwc::run` | none: the typed API takes the settings | none: the typed API takes the settings |
| Stateful clustering object | `dtwc::Config` -> `dtwc::run` -> `Problem` | `dtwc::Problem` | `dtwcpp.Problem` and `dtwcpp.DTWClustering` | `dtwc.Problem` and `dtwc.DTWClustering` |
| Pairwise distance | none (the variant flags set a clustering's distance) | `dtwc::distance::dtw` | `dtwcpp.distance.dtw` | `dtwc.distance.dtw` |
| Cross-collection distance | none | `dtwc::cdist` / `dtwc::cdist_f32` | `dtwcpp.cdist(X, Y=None, ...)` | `dtwc.cdist` / `dtwc.distance.cdist` |
| Soft-DTW gradient | none | `dtwc::soft_dtw_gradient` | `dtwcpp.soft_dtw_gradient` | `dtwc.soft_dtw_gradient` |
| FastPAM entry point | `--method pam` | `dtwc::fast_pam(prob, ...)` | `dtwcpp.fast_pam(prob, ...)` or `DTWClustering.fit()` | `dtwc.fast_pam(prob, ...)` or `DTWClustering.fit()` |
| OneBatchPAM | `--method onebatch` | `dtwc::algorithms::one_batch_pam` | `dtwcpp.one_batch_pam` | `dtwc.cluster(..., 'Method', 'onebatch')` |
| LR-core exact solver | `--method lrcore` | `Problem::set_method(Method::LRCore)` / `LR_core_clustering` | `dtwcpp.cluster(..., method="lrcore")` / `Method.LRCore` | `dtwc.cluster(..., 'Method', 'lrcore')` |
| TADPole density peaks | `--method tadpole --dc` | `dtwc::algorithms::tadpole` | `dtwcpp.cluster(..., method="tadpole")` | `dtwc.cluster(..., 'Method', 'tadpole')` |
| Hierarchical clustering | `--method hierarchical` | `build_dendrogram` / `cut_dendrogram` | same names | same names |
| DTW barycentres | none | `dtw_barycenter` / `barycenter_kmeans` | `dtw_barycenter` / `barycenter_kmeans` | none |
| Missing-data strategy | `--missing-strategy` | `Problem::set_missing_strategy(...)` | `Problem.missing_strategy`, `DTWClustering(missing_strategy=...)` | `Problem.set_missing_strategy(...)`, `DTWClustering('MissingStrategy', ...)` |
| Multivariate loading | Arrow IPC schema `ndim` | `dtwc::load(..., ndim)` / `Data(..., ndim)` | `dtwcpp.load(X_3d, ndim=...)` / `Data.from_numpy_3d` | `dtwc.load(X, 'Ndim', ...)` / `Problem.set_data(..., ndim)` |
| Precision (`dtype`, `matrix_dtype`) | `--dtype`, `--matrix-dtype` | `load(..., dtype)` / `Problem::set_matrix_dtype` | `load(..., dtype=...)`, `matrix_dtype=...` | `load(..., 'Dtype', ...)`, `'MatrixDtype'` |
| RAM budget | `--ram-limit` | `Config::ram_limit` / `Problem::set_ram_limit` | `ram_limit=` / `Problem.ram_limit` | `'RamLimit'` / `Problem.set_ram_limit` |
| A `Problem`'s device & fallback | `--device`, `--fallback` | `Problem::set_device(Device, index, fallback)` | `Problem(..., device=, fallback=)` | `dtwc.Problem(name, 'Device', d, 'Fallback', f)` |
| Run report | written to `<name>_report.txt` (`--no-report` disables) | `Result::report()` | `Result.report` | `Result.report` |

## Pairwise distance vs stateful `Problem`

Use `distance.dtw` for single-pair distances (Python and MATLAB validate settings through the shared C++ name tables):

```cpp
double d = dtwc::distance::dtw(x, y, 10);
```

```python
d = dtwcpp.distance.dtw(x, y, band=10)
```

```matlab
d = dtwc.distance.dtw(x, y, 'Band', 10);
```

Use `Problem` when the workflow is stateful:

- load data
- set variant / missing strategy / band / device
- run clustering
- compute scores or export results

## Similar Problem flow

### C++

```cpp
#include <dtwc.hpp>

std::vector<std::vector<double>> series = {
  {1.0, 2.0, 3.0},
  {1.1, 2.1, 3.1},
  {8.0, 9.0, 10.0}
};
std::vector<std::string> names = {"a", "b", "c"};

dtwc::Problem prob("demo");
prob.set_data(dtwc::Data(std::move(series), std::move(names)));
prob.set_band(10);
prob.set_n_clusters(2);

auto result = dtwc::fast_pam(prob, 2);
auto sil = dtwc::scores::silhouette(prob);
```

### Python

```python
import dtwcpp

prob = dtwcpp.Problem("demo")
prob.set_data(series, names)
prob.band = 10
prob.set_n_clusters(2)

result = dtwcpp.fast_pam(prob, n_clusters=2, max_iter=100)
sil = dtwcpp.silhouette(prob)
```

### MATLAB

```matlab
prob = dtwc.Problem('demo');
prob.set_data(series);
prob.set_band(10);
prob.set_n_clusters(2);

result = dtwc.fast_pam(prob, 2);
sil = dtwc.silhouette(prob);
```

