/**
 * @file test_cluster_first.cpp
 * @brief A Problem holds a clustering only after a clustering wrote it.
 *
 * @details `set_n_clusters` used to size and zero-fill `clusters_ind` and
 *   `centroids_ind`, so a Problem that had only been sized looked clustered:
 *   `find_total_cost()` and `write_clusters()` read the zeros (or, on a Problem
 *   never sized, an empty vector: an access violation in Python and in MATLAB).
 *   A Problem is clustered when it holds one label per series and one medoid per
 *   cluster; every call that reads the whole clustering checks that once and
 *   raises InvalidInput otherwise. The per-element accessors stay unchecked.
 */

#include <dtwc.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
namespace fs = std::filesystem;
using dtwc::Problem;
using dtwc::test_support::ScratchDirectory;

namespace {

const std::vector<std::vector<double>> kSeries{ { 0, 1, 2, 1 }, { 0, 1, 2, 2 }, { 0, 1, 1, 1 },
                                                { 9, 8, 9, 9 }, { 9, 9, 8, 9 }, { 8, 9, 9, 9 } };
const std::vector<std::string> kNames{ "s0", "s1", "s2", "s3", "s4", "s5" };

Problem six_series(const fs::path &out)
{
  Problem prob("cluster_first");
  prob.set_data(dtwc::Data(std::vector(kSeries), std::vector(kNames)));
  prob.set_verbose(false);
  prob.set_output_folder(out);
  return prob;
}

/// Every call that reads the whole clustering, by the name its error carries.
struct Reader
{
  std::string who;
  std::function<void(Problem &)> call;
};

std::vector<Reader> whole_clustering_readers()
{
  namespace scores = dtwc::scores;
  return {
    { "find_total_cost", [](Problem &p) { (void)p.find_total_cost(); } },
    { "print_clusters", [](Problem &p) { p.print_clusters(); } },
    { "write_clusters", [](Problem &p) { p.write_clusters(); } },
    { "write_medoid_members", [](Problem &p) { p.write_medoid_members(0); } },
    { "calculate_medoids", [](Problem &p) { p.calculate_medoids(); } },
    { "silhouette", [](Problem &p) { (void)scores::silhouette(p); } },
    { "davies_bouldin", [](Problem &p) { (void)scores::davies_bouldin(p); } },
    { "dunn", [](Problem &p) { (void)scores::dunn(p); } },
    { "inertia", [](Problem &p) { (void)scores::inertia(p); } },
    { "calinski_harabasz", [](Problem &p) { (void)scores::calinski_harabasz(p); } },
  };
}

void require_every_reader_refuses(Problem &prob)
{
  for (const auto &reader : whole_clustering_readers()) {
    CAPTURE(reader.who);
    REQUIRE_THROWS_MATCHES(reader.call(prob), dtwc::InvalidInput,
                           MessageMatches(ContainsSubstring(reader.who + ": ")
                                          && ContainsSubstring("cluster it first")));
  }
}

std::size_t files_in(const fs::path &dir)
{
  std::size_t n = 0;
  for (const auto &entry : fs::directory_iterator(dir)) { (void)entry; ++n; }
  return n;
}

} // namespace

TEST_CASE("set_n_clusters sizes nothing: the outputs stay empty until a clustering writes them",
          "[problem][cluster-first]")
{
  ScratchDirectory dir{ "cluster_first_empty" };
  auto prob = six_series(dir.path);
  CHECK(prob.labels().empty());
  CHECK(prob.medoids().empty());

  prob.set_n_clusters(2);
  CHECK(prob.labels().empty());
  CHECK(prob.medoids().empty());

  // Initial medoids are a starting point, not a clustering.
  std::vector<int> initial{ 0, 3 };
  prob.set_clusters(initial);
  CHECK(prob.labels().empty());
  CHECK(prob.medoids() == std::vector<int>{ 0, 3 });

  prob.cluster();
  CHECK(prob.labels().size() == 6);
  CHECK(prob.medoids().size() == 2);
}

TEST_CASE("set_view_data sizes no output either", "[problem][cluster-first]")
{
  std::vector<std::span<const double>> spans(kSeries.begin(), kSeries.end());
  Problem view("cluster_first_view");
  view.set_n_clusters(2);
  std::vector<std::string_view> names(kNames.begin(), kNames.end());
  view.set_view_data(dtwc::Data(std::move(spans), std::move(names), 1));
  CHECK(view.labels().empty());
  CHECK(view.medoids().empty());
}

TEST_CASE("every whole-clustering reader refuses a Problem never clustered",
          "[problem][cluster-first]")
{
  ScratchDirectory dir{ "cluster_first_fresh" };
  auto prob = six_series(dir.path);
  require_every_reader_refuses(prob);
  CHECK(files_in(dir.path) == 0); // the refusal comes before any file is opened
}

TEST_CASE("every whole-clustering reader refuses a Problem that was only sized",
          "[problem][cluster-first]")
{
  ScratchDirectory dir{ "cluster_first_sized" };
  auto prob = six_series(dir.path);
  prob.set_n_clusters(2);
  require_every_reader_refuses(prob);
  CHECK(files_in(dir.path) == 0);
}

TEST_CASE("every whole-clustering reader refuses a Problem given only initial medoids",
          "[problem][cluster-first]")
{
  ScratchDirectory dir{ "cluster_first_initial" };
  auto prob = six_series(dir.path);
  prob.set_n_clusters(2);
  std::vector<int> initial{ 0, 3 };
  prob.set_clusters(initial);
  require_every_reader_refuses(prob);
  CHECK(files_in(dir.path) == 0);
}

TEST_CASE("a clustering goes stale when the cluster count changes", "[problem][cluster-first]")
{
  ScratchDirectory dir{ "cluster_first_stale" };
  auto prob = six_series(dir.path);
  prob.set_n_clusters(2);
  prob.cluster();
  REQUIRE_NOTHROW(prob.find_total_cost());

  SECTION("fewer clusters than medoids held")
  {
    prob.set_n_clusters(1);
    require_every_reader_refuses(prob);
  }
  SECTION("more clusters than medoids held")
  {
    prob.set_n_clusters(3);
    require_every_reader_refuses(prob);
  }
  SECTION("series added since the clustering")
  {
    auto more = kSeries;
    more.push_back({ 0, 1, 2, 3 });
    auto names = kNames;
    names.push_back("s6");
    prob.set_data(dtwc::Data(std::move(more), std::move(names)));
    require_every_reader_refuses(prob);
  }
  CHECK(files_in(dir.path) == 0);
}

TEST_CASE("the refusal names the counts it found", "[problem][cluster-first]")
{
  ScratchDirectory dir{ "cluster_first_counts" };
  auto prob = six_series(dir.path);
  prob.set_n_clusters(2);
  REQUIRE_THROWS_MATCHES(prob.find_total_cost(), dtwc::InvalidInput,
                         MessageMatches(ContainsSubstring("0 labels for N = 6")
                                        && ContainsSubstring("0 medoids for k = 2")));
}

TEST_CASE("set_n_clusters then cluster still clusters, for every method that runs here",
          "[problem][cluster-first]")
{
  const auto method = GENERATE(dtwc::Method::Kmedoids, dtwc::Method::LRCore, dtwc::Method::TADPole);
  CAPTURE(method);
  ScratchDirectory dir{ "cluster_first_control" };
  auto prob = six_series(dir.path);
  prob.set_n_clusters(2);
  prob.set_method(method);
  prob.cluster();

  REQUIRE(prob.labels().size() == 6);
  REQUIRE(prob.medoids().size() == 2);
  // Independent oracle: the cost recomputed from the series with the one-shot DTW.
  double expected = 0.0;
  for (std::size_t i = 0; i < kSeries.size(); ++i) {
    const auto medoid = static_cast<std::size_t>(prob.medoids()[static_cast<std::size_t>(prob.labels()[i])]);
    expected += dtwc::dtwFull_L<double>(kSeries[i], kSeries[medoid]);
  }
  CHECK(prob.find_total_cost() == expected);
  CHECK(prob.centroid_of(4) == prob.medoids()[static_cast<std::size_t>(prob.labels()[4])]);
  CHECK_NOTHROW(prob.write_clusters());
  CHECK(files_in(dir.path) == 1);
}
