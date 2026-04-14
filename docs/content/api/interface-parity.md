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

- [Full class index](../Doxygen/annotated.html)
- [Full namespace index](../Doxygen/namespaces.html)
- [C++ `dtwc::Problem` class page with collaboration/call graphs](../Doxygen/classdtwc_1_1_problem.html)
- [C++ `dtwc::distance` namespace](../Doxygen/namespacedtwc_1_1distance.html)
- [Python `dtwcpp.distance` namespace](../Doxygen/namespacedtwcpp_1_1distance.html)
- [Python `DTWClustering` class page](../Doxygen/classdtwcpp_1_1__clustering_1_1_d_t_w_clustering.html)
- [MATLAB `Problem.m` page](../Doxygen/_problem_8m.html)

The generated Python reference is limited to the checked-in Python sources.
`dtwcpp.Problem` comes from the extension module, so Doxygen does not emit a
standalone Python class page for it.

## Generated structure snapshots

These images come directly from the generated Doxygen output:

### C++ `dtwc::Problem` collaboration graph

<img src="../Doxygen/classdtwc_1_1_problem__coll__graph.svg" alt="C++ Problem collaboration graph from Doxygen" />

### Python `dtwcpp.DTWClustering` collaboration graph

<img src="../Doxygen/classdtwcpp_1_1__clustering_1_1_d_t_w_clustering__coll__graph.svg" alt="Python DTWClustering collaboration graph from Doxygen" />

## Shared flow

The recommended flow is intentionally similar across the stateful APIs:

```text
settings -> data -> Problem -> clustering algorithm -> scores / outputs
```

The CLI is already config-first. The library APIs are currently Problem-first.

## Surface map

| Concern | CLI | C++ | Python | MATLAB |
|---------|-----|-----|--------|--------|
| Config file input | TOML, YAML | not first-class yet | not first-class yet | not first-class yet |
| Stateful clustering object | implicit CLI config -> `Problem` | `dtwc::Problem` | `dtwcpp.Problem` and `dtwcpp.DTWClustering` | `dtwc.Problem` and `dtwc.DTWClustering` |
| Pairwise distance namespace | flags only | `dtwc::distance::*` | `dtwcpp.distance.*` | `dtwc.distance.*` |
| FastPAM entry point | `--method pam` | `dtwc::fast_pam(prob, ...)` | `dtwcpp.fast_pam(prob, ...)` or `DTWClustering.fit()` | `dtwc.fast_pam(prob, ...)` or `DTWClustering.fit()` |
| Hierarchical clustering | `--method hierarchical` | `build_dendrogram` / `cut_dendrogram` | same names | same names |
| Missing-data strategy | CLI support is still incomplete | `Problem.missing_strategy` | `Problem.missing_strategy`, `DTWClustering(missing_strategy=...)` | `Problem.set_missing_strategy(...)`, `DTWClustering('MissingStrategy', ...)` |

## Recommended usage split

Use the `distance` namespace for pairwise distances:

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
prob.band = 10;
prob.set_numberOfClusters(2);

auto result = dtwc::fast_pam(prob, 2);
auto sil = dtwc::scores::silhouette(prob);
```

### Python

```python
import dtwcpp

prob = dtwcpp.Problem("demo")
prob.set_data(series, names)
prob.band = 10
prob.set_number_of_clusters(2)

result = dtwcpp.fast_pam(prob, n_clusters=2, max_iter=100)
sil = dtwcpp.silhouette(prob)
```

### MATLAB

```matlab
prob = dtwc.Problem('demo');
prob.set_data(series);
prob.Band = 10;
prob.set_n_clusters(2);

result = dtwc.fast_pam(prob, 2);
sil = dtwc.silhouette(prob);
```

## Current divergences

These are real API differences today, not just documentation differences:

1. The CLI is the only surface with first-class TOML/YAML configuration loading.
2. Python `Problem` is extension-backed, so Doxygen does not emit a standalone `dtwcpp.Problem` page the way it can for pure-Python classes.
3. MATLAB still implements the `dtwc.distance.*` surface as package functions over `dtwc_mex`, while Python uses a pure-Python namespace module and C++ uses headers/templates.
4. C++ examples in older docs historically used `DataLoader` + legacy Lloyd clustering more than the newer `Problem` + algorithm function flow.

## Direction of travel

The intended convergence is:

```text
config/settings -> Problem -> data -> cluster -> scores/results
```

with `distance.*` used for pairwise distances in all languages, and config-file
loading added consistently later rather than invented differently in each
binding.

