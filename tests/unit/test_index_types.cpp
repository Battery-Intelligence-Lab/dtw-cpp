/**
 * @file test_index_types.cpp
 * @brief Pins the integer types of the public counts and indices (DECISIONS §2
 *        rule 2): counts of series and clusters, labels, medoids and
 *        dist_by_ind indices are index_t; band, max_iter, n_init and n_samples
 *        stay int; seeds are uint64_t.
 */

#include <dtwc.hpp>
#include <algorithms/detail/fast_clara_plan.hpp>
#include <algorithms/tadpole.hpp>
#include <cli/config.hpp>
#include <core/medoid_assignment_policy.hpp>
#include <mip/decode_assignment.hpp>
#include <mip/lagrangian_root.hpp>
#include <mip/nearest_medoid.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using dtwc::index_t;
using dtwc::Problem;
using dtwc::core::ClusteringResult;
namespace alg = dtwc::algorithms;

template <typename A, typename B>
constexpr bool same = std::is_same_v<A, B>;

static_assert(same<index_t, std::int64_t>);

// Problem and Data: counts and indices.
static_assert(same<decltype(std::declval<const dtwc::Data &>().size()), index_t>);
static_assert(same<decltype(std::declval<const Problem &>().size()), index_t>);
static_assert(same<decltype(std::declval<const Problem &>().n_clusters()), index_t>);
static_assert(same<decltype(&Problem::set_n_clusters), void (Problem::*)(index_t)>);
static_assert(same<decltype(&Problem::centroid_of), index_t (Problem::*)(index_t) const>);
static_assert(same<decltype(&Problem::dist_by_ind), dtwc::data_t (Problem::*)(index_t, index_t)>);

// ClusteringResult counts.
static_assert(same<decltype(std::declval<const ClusteringResult &>().n_clusters()), index_t>);
static_assert(same<decltype(std::declval<const ClusteringResult &>().n_points()), index_t>);
static_assert(same<decltype(ClusteringResult::iterations), int>);

// Algorithm entry points and options: counts index_t, tuning int, seeds uint64_t.
static_assert(same<decltype(&dtwc::fast_pam), ClusteringResult (*)(Problem &, index_t, int)>);
static_assert(same<decltype(&dtwc::fast_pam_seeded),
                   ClusteringResult (*)(Problem &, index_t, std::uint64_t, int)>);
static_assert(same<decltype(&alg::tadpole),
                   ClusteringResult (*)(Problem &, index_t, double, bool, alg::TADPoleStats *)>);
static_assert(same<decltype(&alg::cut_dendrogram),
                   ClusteringResult (*)(const alg::Dendrogram &, Problem &, index_t)>);

static_assert(same<decltype(alg::CLARAOptions::n_clusters), index_t>);
static_assert(same<decltype(alg::CLARAOptions::sample_size), index_t>);
static_assert(same<decltype(alg::CLARAOptions::n_samples), int>);
static_assert(same<decltype(alg::CLARAOptions::max_iter), int>);
static_assert(same<decltype(alg::CLARAOptions::random_seed), std::uint64_t>);
static_assert(same<decltype(alg::detail::ClaraPlan::n_points), index_t>);
static_assert(same<decltype(alg::detail::ClaraPlan::sample_size), index_t>);

static_assert(same<decltype(alg::OneBatchPAMOptions::n_clusters), index_t>);
static_assert(same<decltype(alg::OneBatchPAMOptions::batch_size), index_t>);
static_assert(same<decltype(alg::OneBatchPAMOptions::max_iter), int>);
static_assert(same<decltype(alg::OneBatchPAMOptions::random_seed), std::uint64_t>);

static_assert(same<decltype(alg::BarycenterClusteringOptions::n_clusters), index_t>);
static_assert(same<decltype(alg::BarycenterClusteringOptions::max_iter), int>);

static_assert(same<decltype(alg::DendrogramStep::cluster_a), index_t>);
static_assert(same<decltype(alg::DendrogramStep::cluster_b), index_t>);
static_assert(same<decltype(alg::DendrogramStep::new_size), index_t>);
static_assert(same<decltype(alg::Dendrogram::n_points), index_t>);
static_assert(same<decltype(alg::HierarchicalOptions::max_points), index_t>);

// The finite-distance policy reports medoid slots and indices as index_t.
static_assert(same<decltype(&dtwc::core::detail::require_finite_medoid_distance),
                   double (*)(double, std::string_view, std::size_t, index_t, index_t)>);
static_assert(same<decltype(&dtwc::core::detail::require_finite_candidate_distance),
                   double (*)(double, std::string_view, std::size_t, index_t)>);

// Labels and medoids: one container type from every producer to every reader.
using indices = std::vector<index_t>;
static_assert(same<decltype(ClusteringResult::labels), indices>);
static_assert(same<decltype(ClusteringResult::medoid_indices), indices>);
static_assert(same<decltype(Problem::clusters_ind), indices>);
static_assert(same<decltype(Problem::centroids_ind), indices>);
static_assert(same<decltype(std::declval<const Problem &>().labels()), const indices &>);
static_assert(same<decltype(std::declval<const Problem &>().medoids()), const indices &>);
static_assert(same<decltype(std::declval<const dtwc::Result &>().labels()), const indices &>);
static_assert(same<decltype(std::declval<const dtwc::Result &>().medoids()), const indices &>);
static_assert(same<decltype(alg::BarycenterClusteringResult::labels), indices>);
static_assert(same<decltype(&alg::dtw_barycenter),
                   std::vector<dtwc::data_t> (*)(const Problem &, const indices &, std::size_t,
                                                 const alg::BarycenterOptions &)>);
static_assert(same<decltype(&dtwc::scores::adjusted_rand), double (*)(const indices &, const indices &)>);
static_assert(same<decltype(&dtwc::scores::normalized_mutual_info),
                   double (*)(const indices &, const indices &)>);
static_assert(same<decltype(dtwc::mip::LagrangianResult::medoids), indices>);
static_assert(same<decltype(dtwc::mip::LagrangianResult::labels), indices>);
static_assert(same<decltype(dtwc::mip::LagrangianResult::core), indices>);
static_assert(same<decltype(dtwc::mip::LagrangianResult::n_core), index_t>);
static_assert(same<decltype(&dtwc::mip::decode_assignment),
                   ClusteringResult (*)(std::span<const double>, std::size_t, index_t, bool,
                                        std::string_view)>);
static_assert(same<decltype(dtwc::mip::NearestMedoid::position), index_t>);
static_assert(same<decltype(&dtwc::mip::lagrangian_root),
                   dtwc::mip::LagrangianResult (*)(const double *, index_t, index_t, double)>);

// set_clusters: the index_t overload takes a braced list; v1's non-const
// std::vector<int>& overload cannot bind one, so the call is not ambiguous.
static_assert(requires(Problem &p) { p.set_clusters({ 0, 2 }); });

// Loaders, the Tier-1 surface and the CLI Config: row, column and series counts
// are index_t, tuning values int, the seed uint64_t.
static_assert(same<decltype(dtwc::LoadOptions::Ndata), index_t>);
static_assert(same<decltype(dtwc::LoadOptions::start_row), index_t>);
static_assert(same<decltype(dtwc::LoadOptions::start_col), index_t>);
static_assert(same<decltype(dtwc::LoadOptions::verbose), int>);
static_assert(same<decltype(std::declval<const dtwc::Dataset &>().skip_cols()), index_t>);
static_assert(same<decltype(std::declval<const dtwc::Dataset &>().skip_rows()), index_t>);
static_assert(same<decltype(dtwc::Config::k), index_t>);
static_assert(same<decltype(dtwc::Config::sample_size), index_t>);
static_assert(same<decltype(dtwc::Config::batch_size), index_t>);
static_assert(same<decltype(dtwc::Config::skip_rows), index_t>);
static_assert(same<decltype(dtwc::Config::skip_cols), index_t>);
static_assert(same<decltype(dtwc::Config::seed), std::uint64_t>);
static_assert(same<decltype(dtwc::Config::max_iter), int>);
static_assert(same<decltype(dtwc::Config::n_init), int>);
static_assert(same<decltype(dtwc::Config::n_samples), int>);
static_assert(same<decltype(dtwc::Config::band), int>);
// Two int skips still bind the skips: the deleted (source, n, char) trap admits
// only a char, since an int converts to index_t and to char alike.
static_assert(requires(const std::filesystem::path &p) { dtwc::load(p, 0, 1); });
static_assert(requires(dtwc::Dataset::series_type s) { dtwc::load(s, 0, 1); });

// The one source break, on purpose: a std::vector<int> no longer assigns to the
// public outputs (CHANGELOG: declare the vector as std::vector<dtwc::index_t>).
static_assert(!std::is_assignable_v<indices &, const std::vector<int> &>);
static_assert(!std::is_assignable_v<decltype((std::declval<Problem &>().clusters_ind)),
                                    const std::vector<int> &>);

} // namespace

TEST_CASE("Option counts hold values above INT_MAX", "[index_t]")
{
  constexpr index_t big = index_t{ 1 } << 40;
  alg::CLARAOptions clara;
  clara.n_clusters = big;
  clara.sample_size = big;
  alg::OneBatchPAMOptions one_batch;
  one_batch.n_clusters = big;
  one_batch.batch_size = big;
  alg::HierarchicalOptions hierarchical;
  hierarchical.max_points = big;
  CHECK(clara.n_clusters == big);
  CHECK(clara.sample_size == big);
  CHECK(one_batch.n_clusters == big);
  CHECK(one_batch.batch_size == big);
  CHECK(hierarchical.max_points == big);
}

TEST_CASE("set_clusters takes an index_t list and v1's std::vector<int>", "[index_t]")
{
  std::vector<std::vector<dtwc::data_t>> series{ { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 } };
  std::vector<std::string> names{ "a", "b", "c", "d" };
  Problem prob;
  prob.set_data(dtwc::Data(std::move(series), std::move(names)));
  prob.set_n_clusters(2);

  prob.set_clusters({ 1, 3 });
  CHECK(prob.medoids() == std::vector<index_t>{ 1, 3 });

  std::vector<int> v1_medoids{ 0, 2 };
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  prob.set_clusters(v1_medoids);
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
  CHECK(prob.medoids() == std::vector<index_t>{ 0, 2 });
}
