---
title: Examples
weight: 5
---

# Examples

These examples use the `Problem`-centric flow, mirroring the Python and MATLAB
surfaces as closely as the C++ API allows.

## Pairwise DTW distance

```cpp
#include <dtwc.hpp>

#include <cstdlib>
#include <vector>

int main()
{
  std::vector<double> x{1.0, 2.0, 3.0, 4.0};
  std::vector<double> y{1.2, 2.1, 2.9, 4.2};

  const double d_standard = dtwc::distance::dtw(x, y, 10);

  dtwc::core::DTWVariantParams params;
  params.variant = dtwc::core::DTWVariant::WDTW;
  params.wdtw_g = 0.05;

  const double d_weighted = dtwc::distance::dtw(
    x, y, params, 10, dtwc::core::MetricType::L1);

  return (d_standard >= 0.0 && d_weighted >= 0.0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

## FastPAM clustering

```cpp
#include <dtwc.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main()
{
  std::vector<std::vector<double>> series{
    {1.0, 2.0, 3.0, 4.0},
    {1.1, 2.1, 3.1, 4.1},
    {9.0, 8.0, 7.0, 6.0},
    {9.2, 8.1, 7.1, 6.2}
  };
  std::vector<std::string> names{"a", "b", "c", "d"};

  dtwc::Problem prob("demo");
  prob.set_data(dtwc::Data(std::move(series), std::move(names)));
  prob.set_band(10);
  prob.set_n_clusters(2);

  auto result = dtwc::fast_pam(prob, 2, 100);

  std::cout << "Total cost: " << result.total_cost << '\n';
  std::cout << "Iterations: " << result.iterations << '\n';
  return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

## Cross-collection distances (`cdist`) and Soft-DTW gradients

```cpp
#include <dtwc.hpp>

#include <iostream>
#include <vector>

int main()
{
  dtwc::Data X({ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }, { "x0", "x1" });
  dtwc::Data Y({ { 1.1, 2.1, 3.1 }, { 7.0, 8.0, 9.0 } }, { "y0", "y1" });

  // Dense row-major M x N cross-collection matrix
  std::vector<double> dists = dtwc::cdist(X, Y);

  // Soft-DTW gradient with respect to first series
  std::vector<double> grad = dtwc::soft_dtw_gradient(X.series(0), Y.series(0), /*gamma=*/1.0);
  std::cout << "X[0] vs Y[0]: " << dists[0] << ", grad[0]: " << grad[0] << '\n';
}
```

## Exact clustering (`LRCore`) and DTW barycentres

```cpp
#include <dtwc.hpp>

#include <vector>

int main()
{
  std::vector<std::vector<double>> series{
    { 0.0, 0.2, 0.4, 0.2 },
    { 0.1, 0.3, 0.5, 0.3 },
    { 5.0, 5.2, 5.4, 5.2 },
    { 5.1, 5.3, 5.5, 5.3 }
  };

  dtwc::Problem prob("bary");
  prob.set_data(dtwc::Data(series, { "s0", "s1", "s2", "s3" }));

  // Exact y-space branch-and-bound without an external MIP solver
  prob.set_n_clusters(2);
  prob.set_method(dtwc::Method::LRCore);
  auto exact = prob.cluster();

  // Compute a single barycentre (SSG / DBA / SoftDTW) or barycentre k-means
  dtwc::algorithms::BarycenterClusteringOptions opts;
  opts.n_clusters = 2;
  opts.barycenter.method = dtwc::algorithms::BarycenterMethod::SSG;
  auto bres = dtwc::algorithms::barycenter_kmeans(prob, opts);
}
```
