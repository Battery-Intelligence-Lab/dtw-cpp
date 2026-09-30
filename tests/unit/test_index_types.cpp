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
#include <core/medoid_assignment_policy.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

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
