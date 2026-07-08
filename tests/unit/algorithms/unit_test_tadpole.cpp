/**
 * @file unit_test_tadpole.cpp
 * @brief TADPole density-peaks clustering — admissibility, oracle, pruning bench.
 *
 * @details TADPole (Begum et al. KDD 2015 / arXiv:1612.00637) is density-peaks
 *   clustering (Rodriguez & Laio, Science 2014) with admissible LB/UB DTW pruning.
 *   Its Theorem 1 guarantees the clustering labels are EXACTLY those of brute-force
 *   density-peaks (all DTW computed). We test that guarantee three ways:
 *     (1) an INDEPENDENT brute-force density-peaks oracle re-implemented here from
 *         the documented conventions (CLAUDE.md §4 — a second implementation);
 *     (2) prune ON vs prune OFF from the SAME call — the pruning flag toggles only
 *         whether a DTW is skipped, so any label/medoid/cost difference falsifies
 *         admissibility;
 *     (3) the Euclidean UB and LB_Keogh really bound DTW (case B/C correctness).
 *
 * Registered bands (stated BEFORE the runs):
 *   BAND-ADMISSIBLE [HARD] — tadpole(prune=true) and tadpole(prune=false) return
 *       DIGIT-IDENTICAL labels + medoids (integers) and total_cost within
 *       1e-9·max(1,cost), for every tested (data, dc, k, band). A difference
 *       FALSIFIES the pruning's admissibility.
 *   BAND-ORACLE     [HARD] — tadpole(prune=true) labels + medoids EQUAL the
 *       independent brute-force density-peaks oracle's, same conditions.
 *   BAND-BOUNDS     [HARD] — for equal-length pairs, LB_Keogh(band) ≤ DTW(band) ≤
 *       Σ_t|x_t−y_t| (diagonal Euclidean UB). Violating the UB would let case B
 *       count a non-neighbour; violating the LB would let case C drop a neighbour.
 *   BAND-PRUNE      [ADVISORY→HARD floor, [.] bench] — on equal-length clustered
 *       data (N=200, len=64, band=10%, dc≈2nd percentile) the fraction of the
 *       N(N−1)/2 brute-force DTWs avoided is ≥ 0.50 (paper reports 80–88%). The
 *       ≥0.50 floor is a HARD assertion; the exact % is advisory (machine/data).
 *
 * @author Volkan Kumtepeli
 * @date 08 Jul 2026
 */

#include <dtwc.hpp>
#include <algorithms/tadpole.hpp>
#include <core/lower_bound_impl.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace {

constexpr double kInf = std::numeric_limits<double>::max();

/// Equal-length (UCR-style) data with `k_true` well-separated sinusoidal clusters,
/// interleaved by index (clusters are NOT index-contiguous, a harder layout).
/// Deterministic given `seed`.
Problem make_clusters(int N, int k_true, int len, int band, unsigned seed = 1)
{
  std::mt19937 rng(seed);
  std::normal_distribution<double> jit(0.0, 0.4);
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;
  vecs.reserve(N);
  names.reserve(N);
  for (int i = 0; i < N; ++i) {
    const int g = i % k_true;                 // interleaved cluster id
    const double base = g * 100.0;            // clusters 100 apart ⇒ well separated
    const double freq = 0.20 + 0.05 * g;
    const double phase = 0.3 * g;
    std::vector<data_t> ts(static_cast<std::size_t>(len));
    for (int t = 0; t < len; ++t)
      ts[t] = base + 10.0 * std::sin(freq * t + phase) + jit(rng);
    vecs.emplace_back(std::move(ts));
    names.push_back("s" + std::to_string(i));
  }
  Problem prob("tadpole_test");
  prob.set_data(Data(std::move(vecs), std::move(names)));
  prob.set_band(band);
  return prob;
}

/// Independent brute-force density-peaks (NO pruning, NO shared code with
/// tadpole.cpp), using the SAME documented conventions so labels are comparable.
core::ClusteringResult oracle_dp(Problem& prob, int k, double dc)
{
  const int N = static_cast<int>(prob.size());
  auto D = [&](int i, int j) { return prob.dist_by_ind(i, j); };

  std::vector<int> rho(N, 0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (i != j && D(i, j) < dc) ++rho[i];

  auto higher = [&](int a, int b) { return rho[a] > rho[b] || (rho[a] == rho[b] && a < b); };

  std::vector<double> delta(N, kInf);
  std::vector<int> parent(N, -1);
  for (int i = 0; i < N; ++i) {
    double best = kInf;
    int bp = -1;
    for (int q = 0; q < N; ++q) {
      if (q == i || !higher(q, i)) continue;
      const double d = D(i, q);
      if (d < best) { best = d; bp = q; }
    }
    delta[i] = best;
    parent[i] = bp;
  }
  int densest = -1;
  for (int i = 0; i < N; ++i)
    if (parent[i] < 0) { densest = i; break; }
  if (densest >= 0) {
    double mx = 0.0;
    for (int i = 0; i < N; ++i)
      if (i != densest && delta[i] < kInf) mx = std::max(mx, delta[i]);
    delta[densest] = mx;
  }

  std::vector<double> g(N);
  for (int i = 0; i < N; ++i) g[i] = static_cast<double>(rho[i]) * delta[i];
  std::vector<int> bg(N);
  std::iota(bg.begin(), bg.end(), 0);
  std::sort(bg.begin(), bg.end(), [&](int a, int b) { return g[a] != g[b] ? g[a] > g[b] : a < b; });

  std::vector<int> label(N, -1), col(k);
  std::vector<char> isc(N, 0);
  for (int c = 0; c < k; ++c) { isc[bg[c]] = 1; label[bg[c]] = c; col[c] = bg[c]; }

  std::vector<int> order(N);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return rho[a] != rho[b] ? rho[a] > rho[b] : a < b; });
  for (int r = 0; r < N; ++r) {
    const int p = order[r];
    if (isc[p]) continue;
    const int par = parent[p];
    label[p] = (par >= 0 && label[par] >= 0) ? label[par] : 0;
  }

  core::ClusteringResult res;
  res.labels = label;
  res.medoid_indices = col;
  double cost = 0.0;
  for (int i = 0; i < N; ++i)
    if (i != col[label[i]]) cost += D(i, col[label[i]]);
  res.total_cost = cost;
  return res;
}

int pct_band(int len, double frac) { return std::max(1, static_cast<int>(frac * len)); }

} // namespace


TEST_CASE("TADPole: pruning is admissible (prune on == prune off)", "[tadpole][admissible]")
{
  // BAND-ADMISSIBLE [HARD]: identical labels/medoids, cost within 1e-9·max(1,cost).
  struct Cfg { int N, k_true, len, band, k; double dc; };
  const std::vector<Cfg> cfgs = {
    { 40, 3, 32, 4, 3, -1.0 },   // auto dc
    { 60, 4, 40, 6, 4, -1.0 },
    { 50, 2, 24, 3, 2, -1.0 },
    { 45, 3, 30, 4, 5, -1.0 },   // k > k_true
    { 40, 3, 32, 4, 1, -1.0 },   // k = 1
  };

  for (const auto& c : cfgs) {
    Problem prob = make_clusters(c.N, c.k_true, c.len, c.band);
    const double dc = (c.dc > 0.0) ? c.dc : algorithms::tadpole_auto_dc(prob, 3.0);
    INFO("N=" << c.N << " k=" << c.k << " band=" << c.band << " dc=" << dc);

    // Fresh problems so the internal DTW cache does not leak between runs.
    Problem pa = make_clusters(c.N, c.k_true, c.len, c.band);
    Problem pb = make_clusters(c.N, c.k_true, c.len, c.band);

    algorithms::TADPoleStats sp{}, sb{};
    auto pruned = algorithms::tadpole(pa, c.k, dc, /*prune=*/true, &sp);
    auto brute  = algorithms::tadpole(pb, c.k, dc, /*prune=*/false, &sb);

    REQUIRE(pruned.labels == brute.labels);
    REQUIRE(pruned.medoid_indices == brute.medoid_indices);
    REQUIRE_THAT(pruned.total_cost,
                 WithinAbs(brute.total_cost, 1e-9 * std::max(1.0, brute.total_cost)));

    // Brute path computes every pair; pruned path skips some (equal-length data).
    REQUIRE(sb.pruned_fraction() == 0.0);
    REQUIRE(sp.pruned_fraction() > 0.0);
    REQUIRE(sp.total_pairs == static_cast<std::size_t>(c.N) * (c.N - 1) / 2);
  }
}

TEST_CASE("TADPole: matches independent brute-force density-peaks oracle", "[tadpole][oracle]")
{
  // BAND-ORACLE [HARD]: labels + medoids equal a second, independent implementation.
  struct Cfg { int N, k_true, len, band, k; };
  const std::vector<Cfg> cfgs = {
    { 36, 3, 28, 4, 3 },
    { 48, 4, 32, 5, 4 },
    { 30, 2, 20, 2, 2 },
    { 40, 3, 24, 3, 6 },
  };
  for (const auto& c : cfgs) {
    Problem pref = make_clusters(c.N, c.k_true, c.len, c.band);
    const double dc = algorithms::tadpole_auto_dc(pref, 3.0);
    INFO("N=" << c.N << " k=" << c.k << " dc=" << dc);

    Problem po = make_clusters(c.N, c.k_true, c.len, c.band);
    Problem pt = make_clusters(c.N, c.k_true, c.len, c.band);
    auto oracle = oracle_dp(po, c.k, dc);
    auto got = algorithms::tadpole(pt, c.k, dc, /*prune=*/true);

    REQUIRE(got.labels == oracle.labels);
    REQUIRE(got.medoid_indices == oracle.medoid_indices);
    REQUIRE_THAT(got.total_cost, WithinAbs(oracle.total_cost, 1e-9 * std::max(1.0, oracle.total_cost)));
  }
}

TEST_CASE("TADPole: LB_Keogh <= DTW <= Euclidean UB on equal-length pairs", "[tadpole][bounds]")
{
  // BAND-BOUNDS [HARD]: the case-B upper bound and case-C lower bound are valid.
  const int len = 40, band = pct_band(len, 0.1);
  Problem prob = make_clusters(50, 3, len, band);
  std::vector<core::Envelope> envs(prob.size());
  for (std::size_t i = 0; i < prob.size(); ++i)
    envs[i] = core::compute_envelope(prob.series(i), band);

  std::size_t checks = 0;
  for (int i = 0; i < static_cast<int>(prob.size()); ++i)
    for (int j = i + 1; j < static_cast<int>(prob.size()); ++j) {
      auto x = prob.series(i);
      auto y = prob.series(j);
      const double dtw = prob.dist_by_ind(i, j);
      double ub = 0.0;
      for (std::size_t t = 0; t < x.size(); ++t) ub += std::abs(x[t] - y[t]);
      const double lb = core::lb_keogh_symmetric(x, envs[i], y, envs[j]);
      REQUIRE(lb <= dtw + 1e-9);
      REQUIRE(dtw <= ub + 1e-9);
      ++checks;
    }
  REQUIRE(checks > 0);
}

TEST_CASE("TADPole: recovers well-separated cluster structure", "[tadpole][quality]")
{
  // Two clusters 100 apart, interleaved by index: a correct clusterer must put all
  // even indices in one label and all odd in the other (not self-consistent noise).
  const int N = 40, len = 32, band = pct_band(len, 0.1);
  Problem prob = make_clusters(N, 2, len, band);
  const double dc = algorithms::tadpole_auto_dc(prob, 5.0);

  Problem pt = make_clusters(N, 2, len, band);
  auto res = algorithms::tadpole(pt, 2, dc, /*prune=*/true);

  // All members of ground-truth group g (indices i with i%2==g) share one label.
  const int lab_even = res.labels[0];
  const int lab_odd = res.labels[1];
  REQUIRE(lab_even != lab_odd);
  for (int i = 0; i < N; ++i)
    REQUIRE(res.labels[i] == (i % 2 == 0 ? lab_even : lab_odd));
}

TEST_CASE("TADPole: edge cases (k=1, k=N, N=1, identical series)", "[tadpole][edge]")
{
  SECTION("k = 1 puts everything in one cluster") {
    Problem prob = make_clusters(30, 3, 24, 3);
    auto res = algorithms::tadpole(prob, 1, algorithms::tadpole_auto_dc(prob, 3.0), true);
    REQUIRE(res.medoid_indices.size() == 1);
    for (int l : res.labels) REQUIRE(l == 0);
  }
  SECTION("k = N gives each point its own cluster") {
    const int N = 12;
    Problem prob = make_clusters(N, 3, 20, 3);
    auto res = algorithms::tadpole(prob, N, 1.0, true);
    std::vector<int> ls = res.labels;
    std::sort(ls.begin(), ls.end());
    ls.erase(std::unique(ls.begin(), ls.end()), ls.end());
    REQUIRE(static_cast<int>(ls.size()) == N);
  }
  SECTION("N = 1") {
    std::vector<std::vector<data_t>> v{ { 1.0, 2.0, 3.0 } };
    std::vector<std::string> nm{ "only" };
    Problem prob("one");
    prob.set_data(Data(std::move(v), std::move(nm)));
    auto res = algorithms::tadpole(prob, 1, 1.0, true);
    REQUIRE(res.labels.size() == 1);
    REQUIRE(res.labels[0] == 0);
  }
  SECTION("identical series: prune == brute, no crash") {
    const int N = 20, len = 16, band = 2;
    std::vector<std::vector<data_t>> v;
    std::vector<std::string> nm;
    for (int i = 0; i < N; ++i) {
      std::vector<data_t> s(len);
      for (int t = 0; t < len; ++t) s[t] = std::sin(0.3 * t);
      v.push_back(std::move(s));
      nm.push_back("id" + std::to_string(i));
    }
    Problem pa("idA"), pb("idB");
    pa.set_data(Data(std::vector<std::vector<data_t>>(v), std::vector<std::string>(nm)));
    pb.set_data(Data(std::move(v), std::move(nm)));
    pa.set_band(band);
    pb.set_band(band);
    auto ra = algorithms::tadpole(pa, 3, 0.5, /*prune=*/true);
    auto rb = algorithms::tadpole(pb, 3, 0.5, /*prune=*/false);
    REQUIRE(ra.labels == rb.labels);
    REQUIRE(ra.medoid_indices == rb.medoid_indices);
  }
}

TEST_CASE("TADPole: unequal-length series fall back to exact (still admissible)", "[tadpole][varlen]")
{
  // Variable-length series: LB/UB pruning is disabled per pair; result must still
  // equal the oracle (correctness holds, only pruning degrades).
  const int N = 30;
  std::vector<std::vector<data_t>> v;
  std::vector<std::string> nm;
  std::mt19937 rng(7);
  std::normal_distribution<double> jit(0.0, 0.4);
  for (int i = 0; i < N; ++i) {
    const int g = i % 3;
    const int len = 20 + (i % 4);                 // variable length
    std::vector<data_t> s(static_cast<std::size_t>(len));
    for (int t = 0; t < len; ++t) s[t] = g * 50.0 + std::sin(0.25 * t) + jit(rng);
    v.push_back(std::move(s));
    nm.push_back("v" + std::to_string(i));
  }
  Problem po("vo"), pt("vt");
  po.set_data(Data(std::vector<std::vector<data_t>>(v), std::vector<std::string>(nm)));
  pt.set_data(Data(std::move(v), std::move(nm)));
  po.set_band(3);
  pt.set_band(3);
  const double dc = algorithms::tadpole_auto_dc(po, 5.0);
  auto oracle = oracle_dp(po, 3, dc);
  auto got = algorithms::tadpole(pt, 3, dc, /*prune=*/true);
  REQUIRE(got.labels == oracle.labels);
  REQUIRE(got.medoid_indices == oracle.medoid_indices);
}

TEST_CASE("TADPole: >=50% of brute-force DTW calls pruned", "[.][tadpole][bench]")
{
  // BAND-PRUNE: hard floor 0.50 on the fraction of N(N-1)/2 DTWs avoided.
  const int N = 200, len = 64;
  const int band = pct_band(len, 0.1);
  Problem prob = make_clusters(N, 5, len, band);
  const double dc = algorithms::tadpole_auto_dc(prob, 2.0);

  Problem pt = make_clusters(N, 5, len, band);
  algorithms::TADPoleStats st{};
  auto res = algorithms::tadpole(pt, 5, dc, /*prune=*/true, &st);

  const double frac = st.pruned_fraction();
  std::printf("[tadpole][bench] N=%d len=%d band=%d dc=%.4g | dtw_calls=%zu / %zu pairs | "
              "pruned=%.1f%% (LB=%zu UB=%zu)\n",
              N, len, band, dc, st.dtw_calls, st.total_pairs, 100.0 * frac,
              st.pruned_by_lb, st.pruned_by_ub);
  REQUIRE(res.labels.size() == static_cast<std::size_t>(N));
  REQUIRE(frac >= 0.50);
}
