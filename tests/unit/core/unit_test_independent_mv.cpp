/**
 * @file unit_test_independent_mv.cpp
 * @brief Independent multivariate DTW (DTW_I, Task 5.6): aeon oracle, DTW_I<=DTW_D
 *        arbiter, Problem wiring, bind-time rejection, univariate no-op.
 *
 * @details Independent-DTW (Shokoohi-Yekta et al., DMKD 2017) runs a separate
 *          univariate DTW on each channel and sums the per-channel distances:
 *          DTW_I = Σ_c DTW(x[:,c], y[:,c]). This is distinct from the existing
 *          dependent DTW_D (dtwFull_L_mv / dtwBanded_mv), which forces one
 *          warping path shared by all channels. Neither dominates for clustering
 *          accuracy (the paper's whole point) — both are provided.
 *
 *          REGISTERED BANDS (fixed before running):
 *            - BAND-ORACLE [HARD]: dtw_independent_mv(SquaredL2) == aeon 1.5.0
 *              Σ_c dtw_distance(channel_c) to rel 1e-9 on 10 non-degenerate 2-/3-
 *              channel pairs (equal + unequal lengths). aeon's DTW uses a
 *              squared-Euclidean local cost, matching our SquaredL2 metric.
 *            - BAND-INEQ [HARD]: DTW_I <= DTW_D always (same band, additive
 *              per-channel cost; each channel is free to pick its own path).
 *              Checked L1 + SquaredL2, unbanded + banded, on random MV pairs.
 *            - BAND-DEINTERLEAVE [HARD]: dtw_independent_mv equals the manual
 *              sum of univariate dtwFull_eap over channels extracted with an
 *              independent de-interleave (guards the strided channel copy).
 *            - BAND-WIRING [HARD]: a Problem with mv_mode=Independent, ndim>1
 *              fills a matrix equal to the direct kernel, and different from the
 *              dependent matrix on at least one entry (genuinely distinct modes).
 *            - BAND-REJECT [HARD]: Independent + non-Standard variant, or
 *              Independent + a missing-data strategy, throws InvalidInput at
 *              bind time (serial — before the parallel fill).
 *
 *          Oracle values generated from aeon 1.5.0 on reproducible LCG series
 *          (generator in .claude/baselines/2026-07-08-independent-mv.md).
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 4.8
 * @date 2026-07-08
 */

#include <warping.hpp>
#include <Problem.hpp>
#include <error.hpp>
#include <core/dtw_options.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dtwc::dtw_independent_mv;
using dtwc::dtwFull_eap;
using dtwc::dtwFull_L_mv;
using dtwc::dtwBanded_mv;
using MT = dtwc::core::MetricType;

struct MVPair {
  std::size_t ndim, nx, ny;
  std::vector<double> x, y;   // interleaved: v[t*ndim + c]
  double dtw_i_sqL2;          // aeon Σ_c dtw_distance(channel_c), squared-L2
};

// aeon 1.5.0 reference (squared-Euclidean local cost == our SquaredL2).
static const std::vector<MVPair> AEON_REF = {
{2,5,5, {3.643353,-1.218004,-2.575912,2.629961,-3.579928,-3.541556,-2.517864,-3.922131,4.662100,3.070695}, {-4.496874,0.641769,2.659375,-2.134752,-3.348354,-3.309982,1.877314,0.473046,-3.771407,4.637188}, 166.4177560811},
{2,8,8, {-2.492317,2.646326,-1.988657,3.217216,0.257265,0.295637,-2.944611,-4.348879,-4.478458,3.930136,-3.053858,3.497761,-1.890225,0.235423,4.568044,0.807629}, {-0.632544,4.506099,3.246630,-1.547497,0.488839,0.527211,1.450567,0.046299,-2.911965,-4.503371,-3.342597,3.209022,1.999476,4.125124,2.077072,-1.683343}, 51.4551569260},
{2,7,11, {1.372012,-3.489345,-1.401402,3.804470,4.094458,4.132830,-3.371358,-4.775626,-3.619017,4.789578,2.107997,-1.340385,0.674603,2.800252}, {3.231785,-1.629572,3.833885,-0.960243,4.326032,4.364404,1.023820,-0.380448,-2.052523,-3.643929,1.819257,-1.629124,4.564304,-3.310048,-3.964444,2.275141,-4.354864,-3.106677,-2.598490,-3.917958,-1.663119,-4.565953}, 159.0252828529},
{3,6,6, {-4.763658,0.374985,-4.486372,-0.814147,4.391725,-0.402402,-2.068349,-2.029977,-1.991605,-3.798105,4.797627,3.393360,-2.759575,-4.350981,4.057614,-2.730149,3.821470,0.373088}, {-2.903885,2.234758,-2.626599,4.421140,-0.372988,4.832885,-1.836775,-1.798403,-1.760031,0.597073,-0.807195,-2.211462,-1.193082,-2.784487,-4.375893,-3.018888,3.532730,0.084349}, 125.6385743553},
{3,10,10, {-0.899329,4.239314,-0.622042,-0.226893,4.978980,0.184853,1.768844,1.807216,1.845588,-4.224852,4.370880,2.966613,-1.900133,-3.491539,4.917055,2.431705,-1.016676,-4.465058,-4.195740,-2.070092,0.055557,-3.556505,2.683080,-1.077335,-3.838810,-2.590623,-1.342437,0.312334,-1.007134,-2.326601}, {0.960444,-3.900912,1.237731,-4.991606,0.214267,-4.579861,2.000418,2.038790,2.077161,0.170326,-1.233942,-2.638209,-0.333640,-1.925046,-3.516451,2.142966,-1.305415,-4.753797,-0.306039,1.819609,3.945257,3.952523,0.192108,-3.568307,-4.717487,-3.469300,-2.221113,3.508033,2.188566,0.869098}, 236.9575024213},
{3,9,13, {2.965001,-1.896356,3.242287,0.360362,-4.433765,0.772107,-4.393963,-4.355591,-4.317219,-4.651599,3.944133,2.539866,-1.040692,-2.632097,-4.223503,-2.406440,4.145178,0.696797,-1.630911,0.494737,2.620385,0.401979,-3.358436,2.881149,0.979878,2.228065,3.476252}, {4.824774,-0.036583,-4.897940,-4.404351,0.801522,-3.992606,-4.162390,-4.124018,-4.085646,-0.256421,-1.660689,-3.064956,0.525801,-1.065604,-2.657010,-2.695180,3.856439,0.408057,2.258789,4.384437,-3.489915,-2.088993,4.150592,0.390177,0.101202,1.349389,2.597576,1.561295,0.241827,-1.077640,-2.513325,4.583841,1.681007,-2.083549,3.557072,-0.802306,4.668219,-4.286722,-3.241663}, 329.1755016537},
{2,12,12, {-3.170670,1.967974,0.947617,-3.846510,-0.556770,-0.518398,4.921654,3.517386,-0.181250,-1.772656,2.755414,-0.692967,0.933917,3.059565,4.360463,0.600048,-4.201433,-2.953246,-3.581143,-4.900610,4.207391,1.304557,-1.749889,3.890733}, {-1.310897,3.827747,-3.817096,1.388776,-0.325197,-0.286825,-0.683168,-2.087436,1.385243,-0.206163,2.466675,-0.981707,4.823617,-3.050734,1.869491,-1.890924,4.919891,-3.831922,-0.385444,-1.704911,-2.796727,4.300439,1.978592,-2.380786}, 148.1750879773},
{2,4,4, {0.693660,-4.167697,1.534872,-3.259256,3.280422,3.318794,4.494907,3.090639}, {2.553433,-2.307924,-3.229841,1.976031,3.511996,3.550368,-1.109915,-2.514183}, 91.8637480181},
{3,15,15, {4.557989,-0.303367,4.835276,2.122127,-2.672001,2.533872,-2.882385,-2.844013,-2.805641,4.068160,2.663892,1.259625,1.537633,-0.053773,-1.645178,3.079123,-0.369259,-3.817640,-3.936426,-1.810778,0.314870,2.277430,-1.482985,4.756600,-4.564055,-3.315868,-2.067682,2.525380,1.205913,-0.113554,3.640587,0.737753,-2.165081,-3.625606,2.015015,-2.344363,-2.166956,-1.121897,-0.076838,2.637733,3.340321,4.042910,-0.247137,3.151928,-3.449006}, {-3.582238,1.556406,-3.304951,-2.642587,2.563286,-2.230841,-2.650811,-2.612439,-2.574067,-1.536662,-2.940930,-4.345197,3.104126,1.512720,-0.078685,2.790383,-0.657998,-4.106380,-0.046726,2.078922,4.204570,-0.213541,-3.973956,2.265628,4.557268,-4.194545,-2.946358,-4.278920,4.401612,3.082145,-3.363531,3.733635,0.830801,0.102875,-4.256504,1.384118,-1.814140,-0.769081,0.275978,-1.830220,-1.127631,-0.425042,4.144718,-2.456216,0.942849}, 444.0507858283},
{2,20,16, {-1.577681,3.560962,2.709381,-2.084746,0.954808,0.993180,3.641413,2.237145,2.397074,0.805669,-1.759023,4.792596,-1.371598,0.754050,-3.764086,2.475499,0.254633,1.502820,0.578642,-0.740825,3.357185,0.454351,0.436535,-3.922843,2.338924,3.383983,2.896600,3.599189,-0.340606,3.058459,2.014813,-0.527756,4.419825,2.198531,-4.850910,3.608113,4.395110,-2.576073,3.917195,-3.670422}, {0.282092,-4.579265,-2.055332,3.150541,1.186382,1.224754,-1.963409,-3.367677,3.963567,2.372162,-2.047762,4.503856,2.518102,4.643751,3.744942,-0.015473,-0.624043,0.624144,3.774341,2.454874,-3.646933,3.450233,4.165016,-0.194362,2.691741,3.736799,-1.571352,-0.868763,4.051249,-2.549685,-3.401195,4.056236}, 285.2654323728},
};

// =========================================================================
//  BAND-ORACLE [HARD] — matches aeon 1.5.0 independent DTW (squared-L2)
// =========================================================================

TEST_CASE("Independent MV DTW matches aeon 1.5.0 reference", "[indep][mv][oracle]")
{
  for (const auto& p : AEON_REF) {
    const double got = dtw_independent_mv<double>(
      p.x.data(), p.nx, p.y.data(), p.ny, p.ndim, -1, MT::SquaredL2);
    REQUIRE_THAT(got, WithinRel(p.dtw_i_sqL2, 1e-9) || WithinAbs(p.dtw_i_sqL2, 1e-9));
  }
}

// =========================================================================
//  BAND-DEINTERLEAVE [HARD] — strided channel copy is correct
// =========================================================================

TEST_CASE("Independent MV DTW equals manual per-channel sum", "[indep][mv][deinterleave]")
{
  for (const auto& p : AEON_REF) {
    for (MT m : { MT::L1, MT::SquaredL2 }) {
      // Independent de-interleave in the test (not the library's copy).
      double manual = 0.0;
      std::vector<double> cx(p.nx), cy(p.ny);
      for (std::size_t c = 0; c < p.ndim; ++c) {
        for (std::size_t t = 0; t < p.nx; ++t) cx[t] = p.x[t * p.ndim + c];
        for (std::size_t t = 0; t < p.ny; ++t) cy[t] = p.y[t * p.ndim + c];
        manual += dtwFull_eap<double>(cx.data(), p.nx, cy.data(), p.ny, m);
      }
      const double got = dtw_independent_mv<double>(
        p.x.data(), p.nx, p.y.data(), p.ny, p.ndim, -1, m);
      REQUIRE_THAT(got, WithinRel(manual, 1e-12) || WithinAbs(manual, 1e-12));
    }
  }
}

// =========================================================================
//  BAND-INEQ [HARD] — DTW_I <= DTW_D (same band, additive per-channel cost)
// =========================================================================

static std::vector<double> rnd_mv(std::mt19937& g, std::size_t steps, std::size_t ndim)
{
  std::uniform_real_distribution<double> d(-5.0, 5.0);
  std::vector<double> s(steps * ndim);
  for (auto& v : s) v = d(g);
  return s;
}

TEST_CASE("DTW_I canonical band uses the finite no-path sentinel", "[indep][mv][band]")
{
  constexpr std::size_t ndim = 2;
  constexpr auto max_value = std::numeric_limits<double>::max();
  const std::vector<double> x{0.0, 10.0};
  const std::vector<double> y{
      1.0, 11.0,
      2.0, 12.0,
      3.0, 13.0
  };

  REQUIRE(dtw_independent_mv<double>(
              x.data(), 1, y.data(), 3, ndim, 1, MT::L1)
          == max_value);
  REQUIRE(dtw_independent_mv<double>(
              y.data(), 3, x.data(), 1, ndim, 1, MT::SquaredL2)
          == max_value);
  REQUIRE(dtw_independent_mv<double>(
              x.data(), 1, y.data(), 3, ndim, 2, MT::L1)
          == 12.0);
  REQUIRE(dtw_independent_mv<double>(
              x.data(), 1, y.data(), 3, ndim, 2, MT::SquaredL2)
          == 28.0);
}

TEST_CASE("DTW_I <= DTW_D (independent has per-channel freedom)", "[indep][mv][ineq]")
{
  std::mt19937 g(20260708u);
  for (int rep = 0; rep < 200; ++rep) {
    std::uniform_int_distribution<std::size_t> nd(2, 4), ln(3, 20);
    const std::size_t ndim = nd(g), nx = ln(g), ny = ln(g);
    auto x = rnd_mv(g, nx, ndim), y = rnd_mv(g, ny, ndim);
    for (MT m : { MT::L1, MT::SquaredL2 }) {
      // Unbanded.
      const double di = dtw_independent_mv<double>(x.data(), nx, y.data(), ny, ndim, -1, m);
      const double dd = dtwFull_L_mv<double>(x.data(), nx, y.data(), ny, ndim, -1, m);
      REQUIRE(di <= dd + 1e-9);
      // Use the same feasible canonical band for both routes.
      const auto gap = (nx > ny) ? (nx - ny) : (ny - nx);
      const int band = static_cast<int>(std::max<std::size_t>(2, gap));
      const double dib = dtw_independent_mv<double>(x.data(), nx, y.data(), ny, ndim, band, m);
      const double ddb = dtwBanded_mv<double>(x.data(), nx, y.data(), ny, ndim, band, -1, m);
      REQUIRE(dib <= ddb + 1e-9);
    }
  }
}

// =========================================================================
//  Hand-computed 2-channel example (L1)
// =========================================================================

TEST_CASE("Independent MV DTW hand example (2 channels, L1)", "[indep][mv][hand]")
{
  // ndim=2, 2 timesteps. x: t0=(0,0), t1=(1,1); y: t0=(0,0), t1=(3,3).
  // Each channel is [0,1] vs [0,3]; univariate DTW_L1 = 2 (align 0-0, then 1
  // maps to the nearer of {0,3}=... path cost: D11=min(0,3,1)+|1-3|=0+2=2).
  // Two identical channels -> DTW_I = 2 + 2 = 4.
  const std::vector<double> x{ 0.0, 0.0, 1.0, 1.0 };
  const std::vector<double> y{ 0.0, 0.0, 3.0, 3.0 };
  const double di = dtw_independent_mv<double>(x.data(), 2, y.data(), 2, 2, -1, MT::L1);
  REQUIRE_THAT(di, WithinAbs(4.0, 1e-12));
}

// =========================================================================
//  BAND-WIRING [HARD] — Problem mv_mode=Independent dispatch + distinct modes
// =========================================================================

static dtwc::Data make_mv_data(std::mt19937& g, int n, std::size_t steps, std::size_t ndim)
{
  dtwc::Data data;
  data.ndim = ndim;
  for (int i = 0; i < n; ++i) {
    data.p_vec.push_back(rnd_mv(g, steps, ndim));
    data.p_names.push_back("s" + std::to_string(i));
  }
  return data;
}

TEST_CASE("Problem mv_mode=Independent matches direct kernel and differs from Dependent",
          "[indep][mv][wiring]")
{
  std::mt19937 g(42u);
  const std::size_t ndim = 2, steps = 10;
  const int N = 6;
  auto base = make_mv_data(g, N, steps, ndim);

  // Independent Problem.
  dtwc::Problem prob_i;
  prob_i.set_data(dtwc::Data(base));
  dtwc::core::DTWVariantParams vp_i;
  vp_i.variant = dtwc::core::DTWVariant::Standard;
  vp_i.mv_mode = dtwc::core::MVMode::Independent;
  prob_i.set_variant(vp_i);
  prob_i.fill_distance_matrix();

  // Dependent Problem (default mode).
  dtwc::Problem prob_d;
  prob_d.set_data(dtwc::Data(base));
  prob_d.fill_distance_matrix();

  bool any_diff = false;
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      const double ref = dtw_independent_mv<double>(
        base.p_vec[i].data(), steps, base.p_vec[j].data(), steps, ndim, prob_i.band);
      REQUIRE_THAT(prob_i.dist_by_ind(i, j), WithinRel(ref, 1e-10) || WithinAbs(ref, 1e-10));
      if (std::abs(prob_i.dist_by_ind(i, j) - prob_d.dist_by_ind(i, j)) > 1e-6)
        any_diff = true;
    }
  REQUIRE(any_diff);  // Independent and Dependent are genuinely different modes.
}

// =========================================================================
//  BAND-REJECT [HARD] — unsupported Independent combinations throw at bind
// =========================================================================

TEST_CASE("Independent MV mode rejects non-Standard variant / missing strategy",
          "[indep][mv][reject]")
{
  std::mt19937 g(7u);
  auto base = make_mv_data(g, 4, 6, 2);

  // The binding is resolved eagerly by set_variant() (which rebinds the DTW
  // function), so the rejection fires there — serial, before any parallel fill.

  // Independent + DDTW -> InvalidInput.
  {
    dtwc::Problem prob;
    prob.set_data(dtwc::Data(base));
    dtwc::core::DTWVariantParams vp;
    vp.variant = dtwc::core::DTWVariant::DDTW;
    vp.mv_mode = dtwc::core::MVMode::Independent;
    REQUIRE_THROWS_AS(prob.set_variant(vp), dtwc::InvalidInput);
  }
  // Independent + a missing-data strategy -> InvalidInput. missing_strategy is
  // set before set_variant so the rebind sees the combination.
  {
    dtwc::Problem prob;
    prob.set_data(dtwc::Data(base));
    prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
    dtwc::core::DTWVariantParams vp;
    vp.variant = dtwc::core::DTWVariant::Standard;
    vp.mv_mode = dtwc::core::MVMode::Independent;
    REQUIRE_THROWS_AS(prob.set_variant(vp), dtwc::InvalidInput);
  }
}

// =========================================================================
//  Univariate no-op — Independent mode on ndim=1 == scalar DTW
// =========================================================================

TEST_CASE("Independent MV DTW reduces to scalar DTW for ndim=1", "[indep][mv][univariate]")
{
  std::mt19937 g(99u);
  for (int rep = 0; rep < 20; ++rep) {
    std::uniform_int_distribution<std::size_t> ln(3, 30);
    const std::size_t nx = ln(g), ny = ln(g);
    auto x = rnd_mv(g, nx, 1), y = rnd_mv(g, ny, 1);
    for (MT m : { MT::L1, MT::SquaredL2 }) {
      const double di = dtw_independent_mv<double>(x.data(), nx, y.data(), ny, 1, -1, m);
      const double sc = dtwFull_eap<double>(x.data(), nx, y.data(), ny, m);
      REQUIRE_THAT(di, WithinRel(sc, 1e-12) || WithinAbs(sc, 1e-12));
    }
  }
}
