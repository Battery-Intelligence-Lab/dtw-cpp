/*!
 * @file tier1.cpp
 * @brief One clustering run as a dtwc::Config, the settings dtwc_cl's flags, a TOML
 *        file and Python's keywords all fill, run on series already in memory.
 *
 * Build with -DDTWC_BUILD_EXAMPLES=ON; with testing on, ctest runs it (example_tier1).
 */

#include <dtwc.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main()
{
  // Six short series in two groups, near 0 and near 9, and their names.
  std::vector<std::vector<dtwc::data_t>> series{ { 0.0, 0.1, 0.2 }, { 0.1, 0.0, 0.2 }, { 0.2, 0.2, 0.0 },
                                                 { 9.0, 9.1, 9.2 }, { 9.1, 9.0, 9.2 }, { 9.2, 9.2, 9.0 } };
  std::vector<std::string> names{ "a1", "a2", "a3", "b1", "b2", "b3" };

  dtwc::Config config;   // every setting starts at dtwc_cl's default
  config.k = 2;          // --n-clusters, the one setting a run needs
  config.method = dtwc::Method::PAM;
  config.band = 1;       // --band: a Sakoe-Chiba window of one step
  config.output.clear(); // write no files; Result::save(dir) writes them on request

  const dtwc::Result result = dtwc::run(config, dtwc::Data{ std::move(series), std::move(names) });

  const auto &labels = result.labels();
  if (labels[0] != labels[1] || labels[1] != labels[2] || labels[3] != labels[4]
      || labels[4] != labels[5] || labels[0] == labels[3]) {
    std::cerr << "example_tier1: the two groups were not recovered\n";
    return 1;
  }
  std::cout << "medoids:";
  for (const dtwc::index_t m : result.medoids()) std::cout << ' ' << m;
  std::cout << "\ncost: " << result.cost() << "\nmean silhouette: " << result.score("silhouette")
            << "\nexample_tier1: two groups recovered\n";
}
