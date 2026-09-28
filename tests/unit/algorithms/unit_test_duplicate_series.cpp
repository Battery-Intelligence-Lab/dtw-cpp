/**
 * @file unit_test_duplicate_series.cpp
 * @brief Duplicate series as medoids: every route publishes k non-empty clusters.
 *
 * @details Four series {a, a, b, c} with k = 4. The optimum opens every series
 * (cost 0), so both copies of `a` are medoids and each is at distance 0 from the
 * other. A nearest-medoid scan whose ties go to the first slot labels the second
 * copy with the first copy's slot: its own cluster is then empty (a published
 * k-clustering with k - 1 clusters), and an exact backend that checks "every
 * medoid is in its own cluster" throws on a valid optimum. The oracle is the
 * definition: each medoid carries its own slot, so no cluster is empty.
 */

#include <dtwc.hpp>
#include <algorithms/fast_clara.hpp>
#include <algorithms/fast_pam.hpp>
#include <algorithms/one_batch_pam.hpp>
#include <mip/mip.hpp>

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <string>
#include <vector>

using namespace dtwc;

namespace {

Problem duplicate_problem(const std::vector<double> &values)
{
  std::vector<std::vector<data_t>> series;
  std::vector<std::string> names;
  for (std::size_t i = 0; i < values.size(); ++i) {
    series.push_back({ values[i] });
    names.push_back("s" + std::to_string(i));
  }
  Problem prob("duplicate_series");
  prob.set_data(Data(std::move(series), std::move(names)));
  return prob;
}

/// Each medoid carries its own slot, hence no slot is empty.
void require_self_labelled(const std::vector<int> &medoids, const std::vector<int> &labels, int k)
{
  REQUIRE(medoids.size() == static_cast<std::size_t>(k));
  REQUIRE(std::set<int>(medoids.begin(), medoids.end()).size() == static_cast<std::size_t>(k));
  for (int slot = 0; slot < k; ++slot) {
    INFO("slot " << slot << " medoid " << medoids[static_cast<std::size_t>(slot)]);
    CHECK(labels[static_cast<std::size_t>(medoids[static_cast<std::size_t>(slot)])] == slot);
  }
}

const std::vector<double> kFour = { 0.0, 0.0, 5.0, 10.0 }; // {a, a, b, c}

} // namespace

TEST_CASE("FasterPAM labels a duplicate medoid with its own slot", "[duplicates][pam]")
{
  auto prob = duplicate_problem(kFour);
  const auto result = fast_pam(prob, 4);
  CHECK(result.total_cost == 0.0);
  require_self_labelled(result.medoid_indices, result.labels, 4);
}

TEST_CASE("OneBatchPAM labels a duplicate medoid with its own slot", "[duplicates][onebatch]")
{
  auto prob = duplicate_problem(kFour);
  algorithms::OneBatchPAMOptions options;
  options.n_clusters = 4;
  const auto result = algorithms::one_batch_pam(prob, options);
  require_self_labelled(result.medoid_indices, result.labels, 4);
}

TEST_CASE("CLARA labels a duplicate medoid with its own slot", "[duplicates][clara]")
{
  SECTION("k = N delegates to FasterPAM")
  {
    auto prob = duplicate_problem(kFour);
    algorithms::CLARAOptions options;
    options.n_clusters = 4;
    const auto result = algorithms::fast_clara(prob, options);
    require_self_labelled(result.medoid_indices, result.labels, 4);
  }
  SECTION("a sample holding both copies runs CLARA's own assignment")
  {
    // sample_size = k makes every sampled series a medoid; the seed loop finds a
    // sample that holds both copies of `a` (series 0 and 1).
    const std::vector<double> five = { 0.0, 0.0, 5.0, 10.0, 20.0 };
    bool reached = false;
    for (unsigned seed = 0; seed < 64 && !reached; ++seed) {
      auto prob = duplicate_problem(five);
      algorithms::CLARAOptions options;
      options.n_clusters = 4;
      options.sample_size = 4;
      options.n_samples = 1;
      options.random_seed = seed;
      const auto result = algorithms::fast_clara(prob, options);
      const std::set<int> medoids(result.medoid_indices.begin(), result.medoid_indices.end());
      if (medoids.count(0) == 0 || medoids.count(1) == 0) continue;
      reached = true;
      INFO("seed " << seed);
      require_self_labelled(result.medoid_indices, result.labels, 4);
    }
    REQUIRE(reached);
  }
}

TEST_CASE("Lloyd k-medoids labels a duplicate medoid with its own slot", "[duplicates][kmedoids]")
{
  auto prob = duplicate_problem(kFour);
  prob.set_n_clusters(4);
  prob.set_method(Method::Kmedoids);
  prob.cluster();
  require_self_labelled(prob.centroids_ind, prob.clusters_ind, 4);
}

TEST_CASE("LR-core publishes the zero-cost optimum with duplicate medoids", "[duplicates][lrcore]")
{
  auto prob = duplicate_problem(kFour);
  prob.set_n_clusters(4);
  prob.set_method(Method::LRCore);
  REQUIRE_NOTHROW(prob.cluster());
  require_self_labelled(prob.centroids_ind, prob.clusters_ind, 4);
}

TEST_CASE("MIP publishes the zero-cost optimum with duplicate medoids", "[duplicates][mip]")
{
  if (!highs_solver_available()) {
    SUCCEED("HiGHS is not built here.");
    return;
  }
  auto prob = duplicate_problem(kFour);
  prob.set_n_clusters(4);
  prob.set_method(Method::MIP);
  REQUIRE(prob.set_solver(Solver::HiGHS));
  REQUIRE_NOTHROW(prob.cluster());
  require_self_labelled(prob.centroids_ind, prob.clusters_ind, 4);
}
