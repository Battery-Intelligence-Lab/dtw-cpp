---
title: Examples
weight: 5
---

# Examples

These examples use the current `Problem`-centric flow rather than the older
loader-heavy examples. The goal is to mirror the Python and MATLAB surfaces as
closely as the C++ API allows today.

## Pairwise DTW distance

```cpp
#include <dtwc.hpp>

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

#include <iostream>
#include <string>
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
  prob.band = 10;
  prob.set_numberOfClusters(2);

  auto result = dtwc::fast_pam(prob, 2, 100);

  std::cout << "Total cost: " << result.total_cost << '\n';
  std::cout << "Iterations: " << result.iterations << '\n';
  return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
}
```
