/*!
 * @file example_new_features.cpp
 * @brief Demonstrates new DTWC++ features: DTW variants, missing data,
 *        FastCLARA, checkpointing, and distance metrics.
 *
 * Build:
 *   cmake --build . --target example_new_features
 * or:
 *   g++ -std=c++17 -O2 -I../dtwc example_new_features.cpp -o example_new_features
 *
 * @date 29 Mar 2026
 */

#include <dtwc.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int main()
{
  using dtwc::data_t;

  // -------------------------------------------------------------------------
  // 1. DTW variants on sine waves
  // -------------------------------------------------------------------------
  std::cout << "=== DTW Variants ===\n";

  const int L = 200;
  std::vector<data_t> x(L), y(L);
  for (int i = 0; i < L; ++i) {
    double t = 4.0 * M_PI * i / (L - 1);
    x[i] = std::sin(t);
    y[i] = std::sin(t + 0.5);  // phase-shifted
  }

  auto d_std = dtwc::dtwBanded<data_t>(x, y, -1);
  auto d_ddtw = dtwc::ddtwBanded<data_t>(x, y, -1);
  auto d_wdtw = dtwc::wdtwBanded<data_t>(x, y, -1, 0.05);
  auto d_adtw = dtwc::adtwBanded<data_t>(x, y, -1, 1.0);
  auto d_soft = dtwc::soft_dtw<data_t>(x, y, 1.0);

  std::cout << "  Standard DTW:  " << d_std << "\n"
            << "  DDTW:          " << d_ddtw << "\n"
            << "  WDTW (g=0.05): " << d_wdtw << "\n"
            << "  ADTW (p=1.0):  " << d_adtw << "\n"
            << "  Soft-DTW:      " << d_soft << "\n\n";

  // -------------------------------------------------------------------------
  // 2. DTW with missing data (NaN-aware)
  // -------------------------------------------------------------------------
  std::cout << "=== Missing Data DTW ===\n";

  std::vector<data_t> x_miss = {1.0, 2.0, NAN, 4.0, 5.0};
  std::vector<data_t> y_miss = {1.5, 2.5, 3.5, 4.5, 5.5};

  auto d_missing = dtwc::dtwMissing_banded<data_t>(x_miss, y_miss, -1);
  auto d_normal = dtwc::dtwBanded<data_t>(x_miss, y_miss, -1);

  std::cout << "  With missing data handler: " << d_missing << "\n"
            << "  Standard DTW (NaN leaks):  " << d_normal << "\n\n";

  // -------------------------------------------------------------------------
  // 3. FastCLARA — scalable clustering
  // -------------------------------------------------------------------------
  std::cout << "=== FastCLARA Clustering ===\n";

  // Generate 100 series in 3 clusters
  const int n_series = 100;
  const int series_len = 30;
  const int k = 3;
  std::mt19937 rng(42);
  std::normal_distribution<data_t> noise(0.0, 2.0);

  std::vector<std::vector<data_t>> all_series;
  std::vector<std::string> names;
  for (int c = 0; c < k; ++c) {
    data_t center = static_cast<data_t>(c * 30);
    for (int i = 0; i < n_series / k; ++i) {
      std::vector<data_t> s(series_len);
      for (auto &v : s)
        v = center + noise(rng);
      all_series.push_back(std::move(s));
      names.push_back("s" + std::to_string(all_series.size() - 1));
    }
  }

  // Use FastCLARA (avoids O(N^2) distance matrix)
  dtwc::Problem prob("clara_demo");
  dtwc::Data data(std::vector<std::vector<data_t>>(all_series),
                   std::vector<std::string>(names));
  prob.set_data(std::move(data));

  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.sample_size = 50;  // subsample size
  opts.n_samples = 5;
  opts.random_seed = 42;

  auto result = dtwc::algorithms::fast_clara(prob, opts);

  std::cout << "  Total cost:  " << result.total_cost << "\n"
            << "  Iterations:  " << result.iterations << "\n"
            << "  Converged:   " << (result.converged ? "yes" : "no") << "\n"
            << "  Medoids:     ";
  for (auto m : result.medoid_indices)
    std::cout << m << " ";
  std::cout << "\n";

  // Count cluster sizes
  std::vector<int> sizes(k, 0);
  for (auto label : result.labels)
    ++sizes[label];
  std::cout << "  Cluster sizes: ";
  for (auto sz : sizes)
    std::cout << sz << " ";
  std::cout << "\n\n";

  // -------------------------------------------------------------------------
  // 4. Checkpointing — save and resume distance matrices
  // -------------------------------------------------------------------------
  std::cout << "=== Checkpointing ===\n";

  // Build a small problem and fill the distance matrix
  const int n_small = 10;
  std::vector<std::vector<data_t>> small_series(
      all_series.begin(), all_series.begin() + n_small);
  std::vector<std::string> small_names(
      names.begin(), names.begin() + n_small);

  dtwc::Problem prob2("ckpt_demo");
  dtwc::Data data2(std::move(small_series), std::move(small_names));
  prob2.set_data(std::move(data2));
  prob2.fillDistanceMatrix();

  std::string ckpt_path = "./example_checkpoint";
  dtwc::save_checkpoint(prob2, ckpt_path);
  std::cout << "  Checkpoint saved to: " << ckpt_path << "\n";

  // Resume in a fresh Problem
  dtwc::Problem prob3("resumed");
  // Must reload data with same series in same order
  std::vector<std::vector<data_t>> small_series2(
      all_series.begin(), all_series.begin() + n_small);
  std::vector<std::string> small_names2(
      names.begin(), names.begin() + n_small);
  dtwc::Data data3(std::move(small_series2), std::move(small_names2));
  prob3.set_data(std::move(data3));

  bool loaded = dtwc::load_checkpoint(prob3, ckpt_path);
  std::cout << "  Checkpoint loaded: " << (loaded ? "yes" : "no") << "\n";
  std::cout << "  Distance matrix filled: "
            << (prob3.isDistanceMatrixFilled() ? "yes" : "no") << "\n";

  // Clean up checkpoint directory
  std::filesystem::remove_all(ckpt_path);
  std::cout << "  Cleaned up checkpoint.\n\n";

  std::cout << "All examples completed successfully.\n";
  return EXIT_SUCCESS;
}
