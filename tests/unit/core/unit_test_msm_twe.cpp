/**
 * @file unit_test_msm_twe.cpp
 * @brief MSM + TWE elastic distances (Task 5.5): aeon oracle, metric props, bench.
 *
 * @details MSM (Stefan et al. 2013) and TWE (Marteau 2009) are metric elastic
 *          distances added as DTW variants — the 2024 KAIS clustering
 *          evaluation ranks MSM the best clustering distance (DTW ≈ Euclidean),
 *          so they are a QUALITY lever. The load-bearing contract is matching
 *          the reference implementation (aeon 1.5.0) exactly.
 *
 *          REGISTERED BANDS (fixed before running):
 *            - BAND-ORACLE [HARD]: |ours - aeon| <= 1e-10*max(1,|aeon|) on 20
 *              non-degenerate pairs (equal + unequal lengths), MSM c=1.0 and
 *              TWE nu=0.001, lambda=1.0 (aeon 1.5.0 defaults, window=None).
 *            - BAND-METRIC [HARD]: d>=0; d(x,x)=0; symmetry d(x,y)==d(y,x);
 *              triangle inequality d(x,z) <= d(x,y)+d(y,z)+1e-9 (both are metrics).
 *            - BAND-WIRING [HARD]: a Problem with variant=MSM/TWE fills a matrix
 *              whose entries equal core::msm_distance / twe_distance directly.
 *            - BAND-SPEED [ADVISORY, [.] bench]: per-pair kernel wall-time
 *              MSM/TWE <= 1.3x plain DTW DP (dtwFull_L, same DP structure).
 *
 *          Oracle values were generated from aeon 1.5.0 on reproducible LCG
 *          series (generator in .claude/baselines/2026-07-08-msm-twe.md).
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 4.8
 * @date 2026-07-08
 */

#include <core/msm.hpp>
#include <core/twe.hpp>
#include <warping.hpp>
#include <Problem.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using dtwc::core::msm_distance;
using dtwc::core::twe_distance;

struct Pair { std::vector<double> x, y; double msm, twe; };

static const std::vector<Pair> AEON_REF = {
  { {3.643353,-2.575912,-3.579928,-2.517864,4.6621}, {-4.496874,2.659375,-3.348354,1.877314,-3.771407}, 22.40824, 32.390973 },
  { {-1.218004,2.629961,-3.541556,-3.922131,3.070695,-1.664094,-2.329405,-3.150855}, {0.641769,-2.134752,-3.309982,0.473046,4.637188,-1.952833,1.560295,4.358173}, 21.828826999999997, 32.158626 },
  { {3.92064,-2.164167,-3.503184,4.673601,1.479289,4.887525,-0.203757,3.08873,-0.617191,-4.539648,0.102134,-4.841494}, {-4.219587,3.07112,-3.27161,-0.931221,3.045782,4.598786,3.685944,0.597758,-1.495868,-1.343948,3.098017,-1.113013}, 33.85975799999999, 58.668567 },
  { {-0.940717,3.041706,-3.464812,3.269334,-0.112116,1.439144,1.921892,-0.671685,0.630995,4.140885,-2.8007,0.799128,4.921178,2.674559,0.697812,2.699506}, {0.919056,-1.723007,-3.233238,-2.335489,1.454377,1.150404,-4.188408,-3.162657,-0.247681,-2.663416,0.195183,4.527608,-4.726006,-1.793393,-4.910333,-2.716502}, 42.683192999999996, 67.028286 },
  { {4.197926,-1.752422,-3.42644,1.865066,-1.703522,-2.009238,4.04754}, {-3.942301,3.482865,-3.194866,-3.739756,-0.137029,-2.297977,-2.06276,3.076928,1.000506,-3.982883,-2.707651}, 24.576482, 38.413773 },
  { {-0.663431,3.453451,-3.388068,0.460799,-3.294928,4.542381,-3.826812,1.807484,3.127369,1.50195,1.393632,2.08037,-2.988704}, {1.196342,-1.311262,-3.156494,4.855977,-1.728434,4.253641,0.062889,-0.683488,2.248693}, 25.715892000000004, 41.453623 },
  { {4.475213,-1.340676,-3.349696,-0.943469,-4.886333,1.093999,-1.701163,-1.952931,4.375556,0.182483,-1.509202,-2.279008,-1.943645,4.782325,0.895008,-4.928202,0.256514,2.782678,-3.367401,-2.752976}, {-3.665014,3.89461,-3.118122,3.451709,-3.31984,0.80526,2.188537,-4.443903,3.49688,3.378182,1.486681,1.449473,-1.590829,0.314373,-4.713137,-0.34421,2.493499,-3.08033,-0.464647,1.543473}, 43.46475299999999, 72.663057 },
  { {-0.386144,3.865196,-3.311324,-2.347736,3.522261,-2.354382}, {1.473629,-0.899517,-3.07975,2.047442,-4.911246,-2.643122,4.314185,1.795682,4.745066,2.058715,-1.416153,-2.909906,-0.54577,1.016962}, 26.336392, 41.335440000000006 },
  { {4.752499,-0.928931,-3.272952,-3.752004,1.930856,4.197236,2.550133,0.526239,-3.12807,-2.456452}, {-3.387728,4.306356,-3.041379,0.643174,3.497349,3.908497,-3.560167,-1.964733,-4.006747,0.739247}, 26.470015, 45.758331 },
  { {-0.108857,4.276941,-3.23458,4.843729,0.33945,0.748855,4.675781,-3.234176,-1.879883,-3.775919,-0.217704,4.642856,1.191531,-3.109909,1.092204}, {1.750916,-0.487772,-3.003007,-0.761093,1.905943,0.460116,-1.434518,4.274852,-2.75856,-0.58022,2.778179,-1.628663,1.544347,2.422139,-4.515941}, 34.998518999999995, 58.412893 },
  { {-4.970214,-0.517186,-3.196209,3.439461,-1.251955,-2.699527,-3.19857,3.005408,-0.631696}, {-3.110441,4.718101,-2.964635,-2.165361,0.314538,-2.988266,0.69113,0.514437,-1.510373}, 18.124132999999997, 31.067923000000008 },
  { {0.168429,4.688686,-3.157837,2.035194,-2.843361,3.852092,-1.072922,-0.755007,0.616491,3.585146,3.976628}, {2.028202,-0.076027,-2.926263,-3.569628,-1.276868,3.563353,2.816778}, 23.365663999999995, 39.81754 },
  { {-4.692927,-0.105441,-3.119465,0.630926,-4.434767,0.40371,1.052726,-4.515422,1.864677,2.265679,1.073794,1.56472,4.326708,-1.002143,1.2894,-0.183618,-3.071249,3.536815}, {-2.833154,-4.870154,-2.887891,-4.973896,-2.868274,0.114971,4.942427,2.993606,0.986001,-4.538622,4.069676,-4.706799}, 42.630838000000004, 68.19040000000001 },
  { {0.445716,-4.899569,-3.081093,-0.773341}, {2.305489,0.335718,-2.849519,3.621837}, 10.414999000000002, 14.43882 },
  { {-4.415641,0.306304,-3.042721,-2.177609,2.382422,3.506948,-4.695977,-2.036252,4.361051,-0.373256,-4.731874,2.845963,-3.583175,0.403035,-1.91247,4.731243,2.486163,0.45486,0.863136,-3.453914,3.483558,3.175294,-3.051789,-0.048651,0.064027}, {-2.555868,-4.458409,-2.811147,2.217569,3.948915,3.218208,-0.806277,-4.527224,3.482375,2.822443,-1.735992,-3.425556,-3.230359,-4.064918,2.479385,-0.684764,4.723148,4.591853,3.76589,0.842535,0.990236,-0.402367,2.269696,4.02539,3.074983}, 46.73697000000001, 80.53898399999999 },
  { {0.723002,-4.487823,-3.004349,-3.581876,0.791017,0.058566,-2.570329,4.203333}, {2.582775,0.747463,-2.772775,0.813302,2.35751,-0.230173,1.319371,1.712361,4.730562,1.502976,-4.638826,2.215065,-2.1853}, 28.136705999999997, 40.91377899999999 },
  { {-4.138354,0.718049,-2.965977,-4.986144,-0.800389,-3.389815,-0.444681,0.442917,-3.142575,-3.012191,-0.537542,4.127206,-1.493057,1.808212}, {-2.278581,-4.046664,-2.734403,-0.590966,0.766104,-3.678555,3.44502,-2.048054,-4.021252,0.183508,2.45834,-2.144313,-1.140241,-2.65974}, 27.546471000000004, 46.646989999999995 },
  { {1.000289,-4.076078,-2.927605,3.609589,-2.391795,3.161803,1.680968,-3.317498,-1.894388,-4.331658,-3.440376,-0.232173,-0.447998,2.510801,-1.715274,-2.896465,-4.177718}, {2.860062,1.159208,-2.696031,-1.995233,-0.825301,2.873064,-4.429332,4.191531,-2.773065,-1.135959}, 31.512796, 49.016833 },
  { {-3.861068,1.129794,-2.889233,2.205322,-3.9832,-0.286578}, {-2.001295,-3.634919,-2.657659,-3.399501,-2.416707,-0.575318}, 14.316116000000001, 26.628783999999996 },
  { {1.277576,-3.664333,-2.850861,0.801054,4.425394,-3.73496,-4.067736,-0.838328,0.601986,3.029407,0.753956,1.04907,1.642119,3.915978,-4.917143,2.018397,1.379694,2.749975,-3.992778,-1.392,3.356481,-1.727687}, {3.137349,1.570954,-2.619287,-4.803768,-4.008113,-4.023699,-0.178035,-3.3293,-0.276691,-3.774894,3.749838,4.777551,1.994935,-0.551974,-0.525288,-3.397611,3.616679,-3.113033,-1.090024}, 44.947512, 74.346361 },
};

// =========================================================================
//  BAND-ORACLE [HARD]
// =========================================================================

TEST_CASE("MSM matches aeon 1.5.0 reference (c=1.0)", "[msm][oracle]")
{
  for (const auto& p : AEON_REF) {
    const double got = msm_distance(p.x, p.y, 1.0);
    REQUIRE_THAT(got, WithinRel(p.msm, 1e-10) || WithinAbs(p.msm, 1e-10));
  }
}

TEST_CASE("TWE matches aeon 1.5.0 reference (nu=0.001, lambda=1.0)", "[twe][oracle]")
{
  for (const auto& p : AEON_REF) {
    const double got = twe_distance(p.x, p.y, 0.001, 1.0);
    REQUIRE_THAT(got, WithinRel(p.twe, 1e-10) || WithinAbs(p.twe, 1e-10));
  }
}

// =========================================================================
//  BAND-METRIC [HARD]
// =========================================================================

static std::vector<double> rnd(std::mt19937& g, std::size_t n)
{
  std::uniform_real_distribution<double> d(-5.0, 5.0);
  std::vector<double> s(n);
  for (auto& v : s) v = d(g);
  return s;
}

TEST_CASE("MSM/TWE metric properties", "[msm][twe][metric]")
{
  std::mt19937 g(20260708u);
  using Fn = double (*)(const std::vector<double>&, const std::vector<double>&);
  Fn fns[] = {
    +[](const std::vector<double>& a, const std::vector<double>& b) { return msm_distance(a, b, 1.0); },
    +[](const std::vector<double>& a, const std::vector<double>& b) { return twe_distance(a, b, 0.001, 1.0); }
  };
  for (int rep = 0; rep < 200; ++rep) {
    std::uniform_int_distribution<std::size_t> len(2, 25);
    auto x = rnd(g, len(g)), y = rnd(g, len(g)), z = rnd(g, len(g));
    for (auto fn : fns) {
      const double dxy = fn(x, y), dyx = fn(y, x), dxz = fn(x, z), dyz = fn(y, z);
      REQUIRE(dxy >= 0.0);                                              // non-negative
      REQUIRE_THAT(fn(x, x), WithinAbs(0.0, 1e-12));                    // identity
      REQUIRE_THAT(dxy, WithinRel(dyx, 1e-12) || WithinAbs(dyx, 1e-12)); // symmetry
      REQUIRE(dxz <= dxy + dyz + 1e-9);                                // triangle inequality
    }
  }
}

TEST_CASE("MSM/TWE edge cases", "[msm][twe][edge]")
{
  std::vector<double> a{ 2.0 }, b{ 5.0 }, c{ 1.0, 2.0, 3.0, 4.0 };
  REQUIRE_THAT(msm_distance(a, a, 1.0), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(twe_distance(a, a, 0.001, 1.0), WithinAbs(0.0, 1e-12));
  REQUIRE(msm_distance(a, b, 1.0) > 0.0);
  REQUIRE(twe_distance(a, b, 0.001, 1.0) > 0.0);
  REQUIRE_THAT(msm_distance(a, c, 1.0), WithinRel(msm_distance(c, a, 1.0), 1e-12));
  REQUIRE_THAT(twe_distance(a, c, 0.001, 1.0), WithinRel(twe_distance(c, a, 0.001, 1.0), 1e-12));
}

// =========================================================================
//  BAND-WIRING [HARD] — Problem dispatch matches the direct kernel
// =========================================================================

TEST_CASE("Problem variant=MSM/TWE matrix matches direct kernel", "[msm][twe][wiring]")
{
  std::mt19937 g(42u);
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (int i = 0; i < 6; ++i) {
    series.push_back(rnd(g, 12 + static_cast<std::size_t>(i)));
    names.push_back("s" + std::to_string(i));
  }

  auto check = [&](dtwc::core::DTWVariant var, auto kernel) {
    dtwc::Problem prob;
    prob.set_data(dtwc::Data(std::vector<std::vector<double>>(series),
                             std::vector<std::string>(names)));
    dtwc::core::DTWVariantParams vp; vp.variant = var;
    prob.set_variant(vp);
    prob.fill_distance_matrix();
    for (std::size_t i = 0; i < series.size(); ++i)
      for (std::size_t j = i + 1; j < series.size(); ++j) {
        const double ref = kernel(series[i], series[j]);
        REQUIRE_THAT(prob.dist_by_ind(static_cast<int>(i), static_cast<int>(j)),
                     WithinRel(ref, 1e-10) || WithinAbs(ref, 1e-10));
      }
  };
  check(dtwc::core::DTWVariant::MSM,
        [](const std::vector<double>& a, const std::vector<double>& b) { return msm_distance(a, b, 1.0); });
  check(dtwc::core::DTWVariant::TWE,
        [](const std::vector<double>& a, const std::vector<double>& b) { return twe_distance(a, b, 0.001, 1.0); });
}

// =========================================================================
//  BAND-SPEED [ADVISORY, hidden bench] — MSM/TWE within 1.3x plain DTW DP
// =========================================================================

TEST_CASE("MSM/TWE speed vs plain DTW DP", "[.][msm][twe][bench]")
{
  std::mt19937 g(7u);
  const std::size_t lens[] = { 128, 256, 512, 1024 };
  constexpr int PAIRS = 40, ROUNDS = 3;
  std::printf("\n  len   dtw(ms)   msm(ms)  msm/dtw   twe(ms)  twe/dtw\n");
  for (auto n : lens) {
    std::vector<std::vector<double>> A, B;
    for (int p = 0; p < PAIRS; ++p) { A.push_back(rnd(g, n)); B.push_back(rnd(g, n)); }
    auto timeit = [&](auto fn) {
      double best = 1e18;
      for (int r = 0; r < ROUNDS; ++r) {
        volatile double s = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < PAIRS; ++p) s += fn(A[p], B[p]);
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        (void)s;
      }
      return best;
    };
    { volatile double s = 0;
      for (int p = 0; p < PAIRS; ++p) {
        s += dtwc::dtwFull_L<double>(A[p], B[p]);
        s += msm_distance(A[p], B[p], 1.0);
        s += twe_distance(A[p], B[p], 0.001, 1.0);
      } (void)s; }
    const double dtw = timeit([](const std::vector<double>& a, const std::vector<double>& b) { return dtwc::dtwFull_L<double>(a, b); });
    const double msm = timeit([](const std::vector<double>& a, const std::vector<double>& b) { return msm_distance(a, b, 1.0); });
    const double twe = timeit([](const std::vector<double>& a, const std::vector<double>& b) { return twe_distance(a, b, 0.001, 1.0); });
    std::printf("  %4zu  %8.3f  %8.3f  %6.2fx  %8.3f  %6.2fx\n",
                n, dtw, msm, msm / dtw, twe, twe / dtw);
  }
  std::printf("  REGISTERED BAND-SPEED: MSM/TWE <= 1.30x plain DTW DP (advisory).\n\n");
  SUCCEED();
}
