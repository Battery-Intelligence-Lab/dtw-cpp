#include <dtwc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv)
{
  const std::filesystem::path csv = argc > 1
    ? argv[1]
    : "tests/conformance/data/conformance_series.csv";

  dtwc::device("cpu");
  const auto data = dtwc::load(csv, 0, 0, ',', "quickstart");
  const auto result = dtwc::cluster(data, 3, "pam", 3, "cpu", 100);

  // Canonicalise cluster IDs so the printed answer is independent of medoid order.
  std::vector<dtwc::index_t> medoids = result.medoids();
  std::sort(medoids.begin(), medoids.end());
  std::vector<dtwc::index_t> labels(result.labels().size());
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const dtwc::index_t assigned = result.medoids().at(
      static_cast<std::size_t>(result.labels().at(i)));
    labels[i] = std::lower_bound(medoids.begin(), medoids.end(), assigned) - medoids.begin();
  }

  const std::vector<dtwc::index_t> expected = {
    0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2
  };
  if (labels != expected || medoids != std::vector<dtwc::index_t>{4, 13, 22})
    throw std::runtime_error("quickstart result differs from the conformance fixture");

  std::cout << "labels: 0x9 1x9 2x9\nmedoids: 4 13 22\n";
  std::cout << "mean silhouette: " << result.score("silhouette") << '\n';
}
