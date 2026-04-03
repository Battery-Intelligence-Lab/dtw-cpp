/**
 * @file dtwc_mex.cpp
 * @brief MATLAB MEX gateway for DTWC++ library (comprehensive OOP bindings).
 *
 * @details Uses the legacy C MEX API (mex.h / matrix.h) with R2018a+
 *          interleaved complex (mxGetDoubles). mexLock() prevents DLL unload
 *          while handle objects exist. mexAtExit drains all handles.
 *
 *          String-dispatched ~45 commands. See the plan document for full list.
 *
 * @date 02 Apr 2026
 */

#include "mex.h"
#include "matrix.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../dtwc/dtwc.hpp"
#include "../../dtwc/algorithms/fast_pam.hpp"
#include "../../dtwc/algorithms/fast_clara.hpp"
#include "../../dtwc/algorithms/clarans.hpp"
#include "../../dtwc/algorithms/hierarchical.hpp"
#include "../../dtwc/scores.hpp"
#include "../../dtwc/core/z_normalize.hpp"
#include "../../dtwc/warping_ddtw.hpp"
#include "../../dtwc/warping_wdtw.hpp"
#include "../../dtwc/warping_adtw.hpp"
#include "../../dtwc/warping_missing.hpp"
#include "../../dtwc/warping_missing_arow.hpp"
#include "../../dtwc/soft_dtw.hpp"

#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

// =========================================================================
//  HandleManager: counter-based handle map for shared_ptr<T>
// =========================================================================

template <typename T>
class HandleManager {
  static std::unordered_map<uint64_t, std::shared_ptr<T>> map_;
  static uint64_t counter_;

public:
  static uint64_t create(std::shared_ptr<T> obj) {
    uint64_t h = ++counter_;
    map_[h] = std::move(obj);
    return h;
  }

  static std::shared_ptr<T>& get(uint64_t h) {
    auto it = map_.find(h);
    if (it == map_.end())
      throw std::invalid_argument("Invalid handle: " + std::to_string(h));
    return it->second;
  }

  static void destroy(uint64_t h) {
    auto it = map_.find(h);
    if (it != map_.end())
      map_.erase(it);
  }

  static void drain() {
    map_.clear();
  }

  static size_t size() { return map_.size(); }
};

template <typename T>
std::unordered_map<uint64_t, std::shared_ptr<T>> HandleManager<T>::map_;
template <typename T>
uint64_t HandleManager<T>::counter_ = 0;

// =========================================================================
//  Helpers: MATLAB <-> C++ type conversion
// =========================================================================

/// MATLAB double vector/row -> std::vector<double>
static std::vector<double> to_std_vector(const mxArray *mx) {
  const double *data = mxGetDoubles(mx);
  size_t n = mxGetNumberOfElements(mx);
  return std::vector<double>(data, data + n);
}

/// MATLAB N x L matrix -> vector of series (each row is one series)
static std::vector<std::vector<double>> matrix_to_series(const mxArray *mx) {
  size_t N = mxGetM(mx);  // rows = number of series
  size_t L = mxGetN(mx);  // cols = series length
  const double *data = mxGetDoubles(mx);

  // MATLAB is column-major: data[row + col*N]
  std::vector<std::vector<double>> series(N, std::vector<double>(L));
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < L; ++j)
      series[i][j] = data[i + j * N];

  return series;
}

/// std::vector<int> -> MATLAB 1xN int32 row vector (1-based indexing)
static mxArray *ivec_to_mx_1based(const std::vector<int> &v) {
  mxArray *mx = mxCreateNumericMatrix(1, v.size(), mxINT32_CLASS, mxREAL);
  int32_t *out = static_cast<int32_t *>(mxGetData(mx));
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = static_cast<int32_t>(v[i] + 1);  // 0-based -> 1-based
  return mx;
}

/// Extract scalar double from mxArray
static double get_scalar(const mxArray *mx) {
  return mxGetScalar(mx);
}

/// Extract uint64 handle from mxArray
static uint64_t get_handle(const mxArray *mx) {
  if (mxIsUint64(mx)) {
    uint64_t *p = static_cast<uint64_t *>(mxGetData(mx));
    return p[0];
  }
  // Accept double as well (MATLAB defaults to double)
  return static_cast<uint64_t>(mxGetScalar(mx));
}

/// Extract string from mxArray (char array or string)
static std::string get_string(const mxArray *mx) {
  char *str = mxArrayToString(mx);
  if (!str) return "";
  std::string result(str);
  mxFree(str);
  return result;
}

/// Build a ClusteringResult MATLAB struct from a C++ ClusteringResult
static mxArray *clustering_result_to_mx(const dtwc::core::ClusteringResult &result) {
  const char *field_names[] = { "labels", "medoid_indices", "total_cost", "iterations", "converged" };
  mxArray *s = mxCreateStructMatrix(1, 1, 5, field_names);

  mxSetField(s, 0, "labels", ivec_to_mx_1based(result.labels));
  mxSetField(s, 0, "medoid_indices", ivec_to_mx_1based(result.medoid_indices));
  mxSetField(s, 0, "total_cost", mxCreateDoubleScalar(result.total_cost));

  // iterations as int32
  mxArray *iter_mx = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
  *static_cast<int32_t *>(mxGetData(iter_mx)) = static_cast<int32_t>(result.iterations);
  mxSetField(s, 0, "iterations", iter_mx);

  // converged as logical
  mxArray *conv_mx = mxCreateLogicalScalar(result.converged);
  mxSetField(s, 0, "converged", conv_mx);

  return s;
}

/// Store clustering result back into Problem (CRITICAL for scoring functions)
static void store_result_in_problem(dtwc::Problem &prob, const dtwc::core::ClusteringResult &result) {
  int k = result.n_clusters();
  prob.set_numberOfClusters(k);
  prob.centroids_ind = result.medoid_indices;
  prob.clusters_ind = result.labels;
}

/// Build a Dendrogram MATLAB struct
static mxArray *dendrogram_to_mx(const dtwc::algorithms::Dendrogram &dend) {
  const char *field_names[] = { "merges", "n_points" };
  mxArray *s = mxCreateStructMatrix(1, 1, 2, field_names);

  size_t n_merges = dend.merges.size();
  mxArray *merges_mx = mxCreateDoubleMatrix(n_merges, 4, mxREAL);
  double *out = mxGetDoubles(merges_mx);
  // Column-major: out[row + col * n_merges]
  for (size_t i = 0; i < n_merges; ++i) {
    const auto &step = dend.merges[i];
    out[i + 0 * n_merges] = static_cast<double>(step.cluster_a + 1); // 1-based
    out[i + 1 * n_merges] = static_cast<double>(step.cluster_b + 1); // 1-based
    out[i + 2 * n_merges] = step.distance;
    out[i + 3 * n_merges] = static_cast<double>(step.new_size);
  }
  mxSetField(s, 0, "merges", merges_mx);

  mxArray *np_mx = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
  *static_cast<int32_t *>(mxGetData(np_mx)) = static_cast<int32_t>(dend.n_points);
  mxSetField(s, 0, "n_points", np_mx);

  return s;
}

/// Reconstruct a C++ Dendrogram from a MATLAB struct
static dtwc::algorithms::Dendrogram mx_to_dendrogram(const mxArray *mx) {
  dtwc::algorithms::Dendrogram dend;

  mxArray *merges_mx = mxGetField(mx, 0, "merges");
  mxArray *np_mx = mxGetField(mx, 0, "n_points");

  if (!merges_mx || !np_mx)
    throw std::invalid_argument("Invalid dendrogram struct: missing 'merges' or 'n_points' field.");

  dend.n_points = static_cast<int>(mxGetScalar(np_mx));

  size_t n_merges = mxGetM(merges_mx);
  const double *data = mxGetDoubles(merges_mx);
  dend.merges.resize(n_merges);
  for (size_t i = 0; i < n_merges; ++i) {
    dend.merges[i].cluster_a = static_cast<int>(data[i + 0 * n_merges]) - 1; // 1-based -> 0-based
    dend.merges[i].cluster_b = static_cast<int>(data[i + 1 * n_merges]) - 1;
    dend.merges[i].distance = data[i + 2 * n_merges];
    dend.merges[i].new_size = static_cast<int>(data[i + 3 * n_merges]);
  }

  return dend;
}

/// Parse missing strategy string -> enum
static dtwc::core::MissingStrategy parse_missing_strategy(const std::string &s) {
  if (s == "error") return dtwc::core::MissingStrategy::Error;
  if (s == "zero_cost") return dtwc::core::MissingStrategy::ZeroCost;
  if (s == "arow") return dtwc::core::MissingStrategy::AROW;
  if (s == "interpolate") return dtwc::core::MissingStrategy::Interpolate;
  throw std::invalid_argument("Unknown missing strategy: '" + s + "'. "
    "Valid: 'error', 'zero_cost', 'arow', 'interpolate'.");
}

/// Parse distance strategy string -> enum
static dtwc::DistanceMatrixStrategy parse_distance_strategy(const std::string &s) {
  if (s == "auto") return dtwc::DistanceMatrixStrategy::Auto;
  if (s == "brute_force") return dtwc::DistanceMatrixStrategy::BruteForce;
  if (s == "pruned") return dtwc::DistanceMatrixStrategy::Pruned;
  if (s == "gpu") return dtwc::DistanceMatrixStrategy::GPU;
  throw std::invalid_argument("Unknown distance strategy: '" + s + "'. "
    "Valid: 'auto', 'brute_force', 'pruned', 'gpu'.");
}

/// Parse linkage string -> enum
static dtwc::algorithms::Linkage parse_linkage(const std::string &s) {
  if (s == "single") return dtwc::algorithms::Linkage::Single;
  if (s == "complete") return dtwc::algorithms::Linkage::Complete;
  if (s == "average") return dtwc::algorithms::Linkage::Average;
  throw std::invalid_argument("Unknown linkage: '" + s + "'. Valid: 'single', 'complete', 'average'.");
}

// =========================================================================
//  Problem lifecycle commands
// =========================================================================

static void cmd_Problem_new(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  std::string name;
  if (nrhs > 1 && mxIsChar(prhs[1])) {
    name = get_string(prhs[1]);
  }
  auto prob = std::make_shared<dtwc::Problem>(name);
  prob->verbose = false;
  uint64_t h = HandleManager<dtwc::Problem>::create(prob);

  plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *static_cast<uint64_t *>(mxGetData(plhs[0])) = h;
}

static void cmd_Problem_delete(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_delete requires a handle.");
  uint64_t h = get_handle(prhs[1]);
  HandleManager<dtwc::Problem>::destroy(h);
}

static void cmd_Problem_get_info(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_info requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  const char *field_names[] = { "name", "size", "band", "verbose", "dist_filled" };
  mxArray *s = mxCreateStructMatrix(1, 1, 5, field_names);

  mxSetField(s, 0, "name", mxCreateString(prob.name.c_str()));
  mxSetField(s, 0, "size", mxCreateDoubleScalar(static_cast<double>(prob.size())));
  mxSetField(s, 0, "band", mxCreateDoubleScalar(static_cast<double>(prob.band)));
  mxSetField(s, 0, "verbose", mxCreateLogicalScalar(prob.verbose));
  mxSetField(s, 0, "dist_filled", mxCreateLogicalScalar(prob.isDistanceMatrixFilled()));

  plhs[0] = s;
}

// =========================================================================
//  Problem property get/set commands
// =========================================================================

static void cmd_Problem_set_data(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_data requires handle and data matrix.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  auto series = matrix_to_series(prhs[2]);
  const size_t N = series.size();
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) names[i] = std::to_string(i);

  dtwc::Data data(std::move(series), std::move(names));
  prob.set_data(std::move(data));
}

static void cmd_Problem_set_band(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_band requires handle and band value.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.band = static_cast<int>(get_scalar(prhs[2]));
}

static void cmd_Problem_get_band(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_band requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(prob.band));
}

static void cmd_Problem_set_verbose(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_verbose requires handle and bool.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.verbose = mxIsLogicalScalarTrue(prhs[2]);
}

static void cmd_Problem_set_max_iter(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_max_iter requires handle and value.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.maxIter = static_cast<int>(get_scalar(prhs[2]));
}

static void cmd_Problem_set_n_repetition(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_n_repetition requires handle and value.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.N_repetition = static_cast<int>(get_scalar(prhs[2]));
}

static void cmd_Problem_set_n_clusters(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_n_clusters requires handle and k.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_numberOfClusters(static_cast<int>(get_scalar(prhs[2])));
}

static void cmd_Problem_set_missing_strategy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_missing_strategy requires handle and string.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  std::string s = get_string(prhs[2]);
  prob.missing_strategy = parse_missing_strategy(s);
}

static void cmd_Problem_set_distance_strategy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_distance_strategy requires handle and string.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  std::string s = get_string(prhs[2]);
  prob.distance_strategy = parse_distance_strategy(s);
}

static void cmd_Problem_set_variant(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_variant requires handle and variant string.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  std::string variant = get_string(prhs[2]);

  dtwc::core::DTWVariantParams params = prob.variant_params;

  if (variant == "standard") params.variant = dtwc::core::DTWVariant::Standard;
  else if (variant == "ddtw") params.variant = dtwc::core::DTWVariant::DDTW;
  else if (variant == "wdtw") {
    params.variant = dtwc::core::DTWVariant::WDTW;
    if (nrhs > 3) params.wdtw_g = get_scalar(prhs[3]);
  }
  else if (variant == "adtw") {
    params.variant = dtwc::core::DTWVariant::ADTW;
    if (nrhs > 3) params.adtw_penalty = get_scalar(prhs[3]);
  }
  else if (variant == "softdtw") {
    params.variant = dtwc::core::DTWVariant::SoftDTW;
    if (nrhs > 3) params.sdtw_gamma = get_scalar(prhs[3]);
  }
  else {
    throw std::invalid_argument("Unknown variant: '" + variant + "'. "
      "Valid: 'standard', 'ddtw', 'wdtw', 'adtw', 'softdtw'.");
  }

  prob.set_variant(params);
}

static void cmd_Problem_get_size(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_size requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(prob.size()));
}

static void cmd_Problem_get_cluster_size(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_cluster_size requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(prob.cluster_size()));
}

static void cmd_Problem_get_name(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_name requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateString(prob.name.c_str());
}

static void cmd_Problem_get_centroids(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_centroids requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = ivec_to_mx_1based(prob.centroids_ind);
}

static void cmd_Problem_get_clusters(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_clusters requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = ivec_to_mx_1based(prob.clusters_ind);
}

static void cmd_Problem_is_distance_matrix_filled(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_is_distance_matrix_filled requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateLogicalScalar(prob.isDistanceMatrixFilled());
}

// =========================================================================
//  Problem method commands
// =========================================================================

static void cmd_Problem_fill_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_fill_distance_matrix requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.fillDistanceMatrix();
}

static void cmd_Problem_dist_by_ind(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 4) throw std::invalid_argument("Problem_dist_by_ind requires handle, i, j.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  // Convert from MATLAB 1-based to C++ 0-based
  int i = static_cast<int>(get_scalar(prhs[2])) - 1;
  int j = static_cast<int>(get_scalar(prhs[3])) - 1;
  double d = prob.distByInd(i, j);
  plhs[0] = mxCreateDoubleScalar(d);
}

static void cmd_Problem_cluster(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_cluster requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.cluster();
}

static void cmd_Problem_find_total_cost(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_find_total_cost requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  double cost = prob.findTotalCost();
  plhs[0] = mxCreateDoubleScalar(cost);
}

static void cmd_Problem_get_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_distance_matrix requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  const auto &dm = prob.distance_matrix();
  size_t N = dm.size();
  mxArray *result = mxCreateDoubleMatrix(N, N, mxREAL);
  double *out = mxGetDoubles(result);

  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      out[i + j * N] = dm.get(i, j);  // column-major

  plhs[0] = result;
}

static void cmd_Problem_set_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_distance_matrix requires handle and matrix.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  size_t N = mxGetM(prhs[2]);
  if (N != mxGetN(prhs[2]))
    throw std::invalid_argument("Distance matrix must be square.");
  if (N != prob.size())
    throw std::invalid_argument("Distance matrix size does not match problem size.");

  auto &dm = prob.distance_matrix();
  dm.resize(N);
  const double *data = mxGetDoubles(prhs[2]);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      dm.set(i, j, data[i + j * N]);  // column-major

  prob.set_distance_matrix_filled(true);
}

// =========================================================================
//  Stateless DTW distance functions
// =========================================================================

static void cmd_dtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("dtw_distance requires x and y.");
  auto x = to_std_vector(prhs[1]);
  auto y = to_std_vector(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::dtwBanded<double>(x, y, band));
}

static void cmd_ddtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("ddtw_distance requires x and y.");
  auto x = to_std_vector(prhs[1]);
  auto y = to_std_vector(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::ddtwBanded<double>(x, y, band));
}

static void cmd_wdtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("wdtw_distance requires x and y.");
  auto x = to_std_vector(prhs[1]);
  auto y = to_std_vector(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  double g = 0.05;
  if (nrhs > 4) g = get_scalar(prhs[4]);
  plhs[0] = mxCreateDoubleScalar(dtwc::wdtwBanded<double>(x, y, band, g));
}

static void cmd_adtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("adtw_distance requires x and y.");
  auto x = to_std_vector(prhs[1]);
  auto y = to_std_vector(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  double penalty = 1.0;
  if (nrhs > 4) penalty = get_scalar(prhs[4]);
  plhs[0] = mxCreateDoubleScalar(dtwc::adtwBanded<double>(x, y, band, penalty));
}

static void cmd_soft_dtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("soft_dtw_distance requires x and y.");
  auto x = to_std_vector(prhs[1]);
  auto y = to_std_vector(prhs[2]);
  double gamma = 1.0;
  if (nrhs > 3) gamma = get_scalar(prhs[3]);
  plhs[0] = mxCreateDoubleScalar(dtwc::soft_dtw<double>(x, y, gamma));
}

static void cmd_soft_dtw_gradient(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("soft_dtw_gradient requires x and y.");
  auto x = to_std_vector(prhs[1]);
  auto y = to_std_vector(prhs[2]);
  double gamma = 1.0;
  if (nrhs > 3) gamma = get_scalar(prhs[3]);
  auto grad = dtwc::soft_dtw_gradient<double>(x, y, gamma);

  mxArray *result = mxCreateDoubleMatrix(1, grad.size(), mxREAL);
  double *out = mxGetDoubles(result);
  for (size_t i = 0; i < grad.size(); ++i) out[i] = grad[i];
  plhs[0] = result;
}

static void cmd_dtw_distance_missing(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("dtw_distance_missing requires x and y.");
  auto x = to_std_vector(prhs[1]);
  auto y = to_std_vector(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::dtwMissing_banded<double>(x, y, band));
}

static void cmd_dtw_arow_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("dtw_arow_distance requires x and y.");
  auto x = to_std_vector(prhs[1]);
  auto y = to_std_vector(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::dtwAROW_banded<double>(x, y, band));
}

static void cmd_compute_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("compute_distance_matrix requires a data matrix.");
  auto series = matrix_to_series(prhs[1]);
  const size_t N = series.size();
  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 2) band = static_cast<int>(get_scalar(prhs[2]));

  // Use Problem + fillDistanceMatrix() for OpenMP parallelism and LB pruning
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) names[i] = std::to_string(i);

  dtwc::Problem prob("matlab_distmat");
  prob.band = band;
  prob.verbose = false;
  dtwc::Data data(std::move(series), std::move(names));
  prob.set_data(std::move(data));
  prob.fillDistanceMatrix();

  // Copy from Problem's distance matrix to MATLAB output (column-major)
  mxArray *result = mxCreateDoubleMatrix(N, N, mxREAL);
  double *out = mxGetDoubles(result);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      out[i + j * N] = prob.distByInd(static_cast<int>(i), static_cast<int>(j));

  plhs[0] = result;
}

static void cmd_derivative_transform(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("derivative_transform requires a vector.");
  auto x = to_std_vector(prhs[1]);
  auto dx = dtwc::derivative_transform<double>(x);
  mxArray *result = mxCreateDoubleMatrix(1, dx.size(), mxREAL);
  double *out = mxGetDoubles(result);
  for (size_t i = 0; i < dx.size(); ++i) out[i] = dx[i];
  plhs[0] = result;
}

static void cmd_z_normalize(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("z_normalize requires a vector.");
  auto x = to_std_vector(prhs[1]);
  auto result_vec = dtwc::core::z_normalized<double>(x.data(), x.size());
  mxArray *result = mxCreateDoubleMatrix(1, result_vec.size(), mxREAL);
  double *out = mxGetDoubles(result);
  for (size_t i = 0; i < result_vec.size(); ++i) out[i] = result_vec[i];
  plhs[0] = result;
}

// =========================================================================
//  Algorithm commands
// =========================================================================

static void cmd_fast_pam(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("fast_pam requires handle and k.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  int k = static_cast<int>(get_scalar(prhs[2]));
  int max_iter = 100;
  if (nrhs > 3) max_iter = static_cast<int>(get_scalar(prhs[3]));

  auto result = dtwc::fast_pam(prob, k, max_iter);

  // Store results back into Problem (CRITICAL for scoring)
  store_result_in_problem(prob, result);

  plhs[0] = clustering_result_to_mx(result);
}

static void cmd_fast_clara(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("fast_clara requires handle and k.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = static_cast<int>(get_scalar(prhs[2]));
  if (nrhs > 3) opts.sample_size = static_cast<int>(get_scalar(prhs[3]));
  if (nrhs > 4) opts.n_samples = static_cast<int>(get_scalar(prhs[4]));
  if (nrhs > 5) opts.max_iter = static_cast<int>(get_scalar(prhs[5]));
  if (nrhs > 6) opts.random_seed = static_cast<unsigned>(get_scalar(prhs[6]));

  auto result = dtwc::algorithms::fast_clara(prob, opts);
  store_result_in_problem(prob, result);
  plhs[0] = clustering_result_to_mx(result);
}

static void cmd_clarans(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("clarans requires handle and k.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  dtwc::algorithms::CLARANSOptions opts;
  opts.n_clusters = static_cast<int>(get_scalar(prhs[2]));
  if (nrhs > 3) opts.num_local = static_cast<int>(get_scalar(prhs[3]));
  if (nrhs > 4) opts.max_neighbor = static_cast<int>(get_scalar(prhs[4]));
  if (nrhs > 5) opts.max_dtw_evals = static_cast<int64_t>(get_scalar(prhs[5]));
  if (nrhs > 6) opts.random_seed = static_cast<unsigned>(get_scalar(prhs[6]));

  auto result = dtwc::algorithms::clarans(prob, opts);
  store_result_in_problem(prob, result);
  plhs[0] = clustering_result_to_mx(result);
}

static void cmd_build_dendrogram(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("build_dendrogram requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  dtwc::algorithms::HierarchicalOptions opts;
  if (nrhs > 2) opts.linkage = parse_linkage(get_string(prhs[2]));
  if (nrhs > 3) opts.max_points = static_cast<int>(get_scalar(prhs[3]));

  auto dend = dtwc::algorithms::build_dendrogram(prob, opts);
  plhs[0] = dendrogram_to_mx(dend);
}

static void cmd_cut_dendrogram(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 4) throw std::invalid_argument("cut_dendrogram requires dendrogram struct, handle, and k.");

  auto dend = mx_to_dendrogram(prhs[1]);
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[2]));
  int k = static_cast<int>(get_scalar(prhs[3]));

  auto result = dtwc::algorithms::cut_dendrogram(dend, prob, k);
  store_result_in_problem(prob, result);
  plhs[0] = clustering_result_to_mx(result);
}

// =========================================================================
//  Scoring commands
// =========================================================================

static void cmd_silhouette(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("silhouette requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  auto sil = dtwc::scores::silhouette(prob);

  mxArray *result = mxCreateDoubleMatrix(1, sil.size(), mxREAL);
  double *out = mxGetDoubles(result);
  for (size_t i = 0; i < sil.size(); ++i) out[i] = sil[i];
  plhs[0] = result;
}

static void cmd_davies_bouldin_index(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("davies_bouldin_index requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::daviesBouldinIndex(prob));
}

static void cmd_dunn_index(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("dunn_index requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::dunnIndex(prob));
}

static void cmd_inertia(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("inertia requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::inertia(prob));
}

static void cmd_calinski_harabasz_index(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("calinski_harabasz_index requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::calinskiHarabaszIndex(prob));
}

static void cmd_adjusted_rand_index(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("adjusted_rand_index requires two label vectors.");
  auto mx1 = prhs[1];
  auto mx2 = prhs[2];

  size_t n1 = mxGetNumberOfElements(mx1);
  size_t n2 = mxGetNumberOfElements(mx2);
  if (n1 != n2) throw std::invalid_argument("Label vectors must have the same length.");

  // Accept both double and int32, convert to 0-based C++ int
  std::vector<int> labels1(n1), labels2(n2);
  if (mxIsInt32(mx1)) {
    int32_t *p = static_cast<int32_t *>(mxGetData(mx1));
    for (size_t i = 0; i < n1; ++i) labels1[i] = static_cast<int>(p[i] - 1);
  } else {
    const double *p = mxGetDoubles(mx1);
    for (size_t i = 0; i < n1; ++i) labels1[i] = static_cast<int>(p[i] - 1);
  }
  if (mxIsInt32(mx2)) {
    int32_t *p = static_cast<int32_t *>(mxGetData(mx2));
    for (size_t i = 0; i < n2; ++i) labels2[i] = static_cast<int>(p[i] - 1);
  } else {
    const double *p = mxGetDoubles(mx2);
    for (size_t i = 0; i < n2; ++i) labels2[i] = static_cast<int>(p[i] - 1);
  }

  plhs[0] = mxCreateDoubleScalar(dtwc::scores::adjustedRandIndex(labels1, labels2));
}

static void cmd_normalized_mutual_information(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("normalized_mutual_information requires two label vectors.");
  auto mx1 = prhs[1];
  auto mx2 = prhs[2];

  size_t n1 = mxGetNumberOfElements(mx1);
  size_t n2 = mxGetNumberOfElements(mx2);
  if (n1 != n2) throw std::invalid_argument("Label vectors must have the same length.");

  std::vector<int> labels1(n1), labels2(n2);
  if (mxIsInt32(mx1)) {
    int32_t *p = static_cast<int32_t *>(mxGetData(mx1));
    for (size_t i = 0; i < n1; ++i) labels1[i] = static_cast<int>(p[i] - 1);
  } else {
    const double *p = mxGetDoubles(mx1);
    for (size_t i = 0; i < n1; ++i) labels1[i] = static_cast<int>(p[i] - 1);
  }
  if (mxIsInt32(mx2)) {
    int32_t *p = static_cast<int32_t *>(mxGetData(mx2));
    for (size_t i = 0; i < n2; ++i) labels2[i] = static_cast<int>(p[i] - 1);
  } else {
    const double *p = mxGetDoubles(mx2);
    for (size_t i = 0; i < n2; ++i) labels2[i] = static_cast<int>(p[i] - 1);
  }

  plhs[0] = mxCreateDoubleScalar(dtwc::scores::normalizedMutualInformation(labels1, labels2));
}

// =========================================================================
//  Legacy "cluster" command (stateless, backward-compatible)
// =========================================================================

static void cmd_cluster_legacy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3)
    throw std::invalid_argument("cluster requires data matrix and k.");

  auto series = matrix_to_series(prhs[1]);
  int k = static_cast<int>(get_scalar(prhs[2]));

  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));

  int maxIter = 100;
  if (nrhs > 5) maxIter = static_cast<int>(get_scalar(prhs[5]));

  const size_t N = series.size();
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) names[i] = std::to_string(i);

  dtwc::Problem prob("matlab_clustering");
  prob.band = band;
  prob.maxIter = maxIter;
  prob.verbose = false;

  dtwc::Data data(std::move(series), std::move(names));
  prob.set_data(std::move(data));

  auto result = dtwc::fast_pam(prob, k, maxIter);

  plhs[0] = ivec_to_mx_1based(result.labels);
  if (nlhs > 1) plhs[1] = ivec_to_mx_1based(result.medoid_indices);
  if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(result.total_cost);
}

// =========================================================================
//  Cleanup callback - called when MEX is unloaded (clear mex / exit)
// =========================================================================

static void cleanup_at_exit() {
  HandleManager<dtwc::Problem>::drain();
}

static bool first_call = true;

// =========================================================================
//  MEX entry point - longjmp-safe error handling
// =========================================================================

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  if (first_call) {
    mexLock();  // Prevent DLL unload while handles exist
    mexAtExit(cleanup_at_exit);
    first_call = false;
  }

  if (nrhs < 1 || !mxIsChar(prhs[0]))
    mexErrMsgIdAndTxt("dtwc:invalidInput",
                      "First argument must be a command string.");

  std::string cmd = get_string(prhs[0]);

  // longjmp-safe: catch C++ exceptions, exit scope, THEN call mexErrMsgIdAndTxt
  std::string error_id, error_msg;
  try {
    // Problem lifecycle
    if (cmd == "Problem_new") cmd_Problem_new(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_delete") cmd_Problem_delete(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_info") cmd_Problem_get_info(nlhs, plhs, nrhs, prhs);
    // Problem properties
    else if (cmd == "Problem_set_data") cmd_Problem_set_data(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_band") cmd_Problem_set_band(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_band") cmd_Problem_get_band(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_verbose") cmd_Problem_set_verbose(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_max_iter") cmd_Problem_set_max_iter(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_n_repetition") cmd_Problem_set_n_repetition(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_n_clusters") cmd_Problem_set_n_clusters(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_missing_strategy") cmd_Problem_set_missing_strategy(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_distance_strategy") cmd_Problem_set_distance_strategy(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_variant") cmd_Problem_set_variant(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_size") cmd_Problem_get_size(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_cluster_size") cmd_Problem_get_cluster_size(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_name") cmd_Problem_get_name(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_centroids") cmd_Problem_get_centroids(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_clusters") cmd_Problem_get_clusters(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_is_distance_matrix_filled") cmd_Problem_is_distance_matrix_filled(nlhs, plhs, nrhs, prhs);
    // Problem methods
    else if (cmd == "Problem_fill_distance_matrix") cmd_Problem_fill_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_dist_by_ind") cmd_Problem_dist_by_ind(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_cluster") cmd_Problem_cluster(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_find_total_cost") cmd_Problem_find_total_cost(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_distance_matrix") cmd_Problem_get_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_distance_matrix") cmd_Problem_set_distance_matrix(nlhs, plhs, nrhs, prhs);
    // Stateless DTW functions
    else if (cmd == "dtw_distance") cmd_dtw_distance(nlhs, plhs, nrhs, prhs);
    else if (cmd == "ddtw_distance") cmd_ddtw_distance(nlhs, plhs, nrhs, prhs);
    else if (cmd == "wdtw_distance") cmd_wdtw_distance(nlhs, plhs, nrhs, prhs);
    else if (cmd == "adtw_distance") cmd_adtw_distance(nlhs, plhs, nrhs, prhs);
    else if (cmd == "soft_dtw_distance") cmd_soft_dtw_distance(nlhs, plhs, nrhs, prhs);
    else if (cmd == "soft_dtw_gradient") cmd_soft_dtw_gradient(nlhs, plhs, nrhs, prhs);
    else if (cmd == "dtw_distance_missing") cmd_dtw_distance_missing(nlhs, plhs, nrhs, prhs);
    else if (cmd == "dtw_arow_distance") cmd_dtw_arow_distance(nlhs, plhs, nrhs, prhs);
    else if (cmd == "compute_distance_matrix") cmd_compute_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "derivative_transform") cmd_derivative_transform(nlhs, plhs, nrhs, prhs);
    else if (cmd == "z_normalize") cmd_z_normalize(nlhs, plhs, nrhs, prhs);
    // Algorithms
    else if (cmd == "fast_pam") cmd_fast_pam(nlhs, plhs, nrhs, prhs);
    else if (cmd == "fast_clara") cmd_fast_clara(nlhs, plhs, nrhs, prhs);
    else if (cmd == "clarans") cmd_clarans(nlhs, plhs, nrhs, prhs);
    else if (cmd == "build_dendrogram") cmd_build_dendrogram(nlhs, plhs, nrhs, prhs);
    else if (cmd == "cut_dendrogram") cmd_cut_dendrogram(nlhs, plhs, nrhs, prhs);
    // Scoring
    else if (cmd == "silhouette") cmd_silhouette(nlhs, plhs, nrhs, prhs);
    else if (cmd == "davies_bouldin_index") cmd_davies_bouldin_index(nlhs, plhs, nrhs, prhs);
    else if (cmd == "dunn_index") cmd_dunn_index(nlhs, plhs, nrhs, prhs);
    else if (cmd == "inertia") cmd_inertia(nlhs, plhs, nrhs, prhs);
    else if (cmd == "calinski_harabasz_index") cmd_calinski_harabasz_index(nlhs, plhs, nrhs, prhs);
    else if (cmd == "adjusted_rand_index") cmd_adjusted_rand_index(nlhs, plhs, nrhs, prhs);
    else if (cmd == "normalized_mutual_information") cmd_normalized_mutual_information(nlhs, plhs, nrhs, prhs);
    // Legacy backward-compatible command
    else if (cmd == "cluster") cmd_cluster_legacy(nlhs, plhs, nrhs, prhs);
    // System capability check
    else if (cmd == "system_check") {
      const char *fields[] = {"openmp", "openmp_threads", "cuda", "cuda_info", "mpi"};
      mxArray *info = mxCreateStructMatrix(1, 1, 5, fields);
#ifdef _OPENMP
      mxSetField(info, 0, "openmp", mxCreateLogicalScalar(true));
      mxSetField(info, 0, "openmp_threads", mxCreateDoubleScalar(omp_get_max_threads()));
#else
      mxSetField(info, 0, "openmp", mxCreateLogicalScalar(false));
      mxSetField(info, 0, "openmp_threads", mxCreateDoubleScalar(1));
#endif
#ifdef DTWC_HAS_CUDA
      mxSetField(info, 0, "cuda", mxCreateLogicalScalar(dtwc::cuda::cuda_available()));
      std::string ci = dtwc::cuda::cuda_device_info(0);
      mxSetField(info, 0, "cuda_info", mxCreateString(ci.c_str()));
#else
      mxSetField(info, 0, "cuda", mxCreateLogicalScalar(false));
      mxSetField(info, 0, "cuda_info", mxCreateString("not compiled (rebuild with -DDTWC_ENABLE_CUDA=ON)"));
#endif
#ifdef DTWC_HAS_MPI
      mxSetField(info, 0, "mpi", mxCreateLogicalScalar(true));
#else
      mxSetField(info, 0, "mpi", mxCreateLogicalScalar(false));
#endif
      plhs[0] = info;
    }
    else {
      throw std::invalid_argument("Unknown command: '" + cmd + "'.");
    }
  } catch (const std::invalid_argument &e) {
    error_id = "dtwc:invalidArgument"; error_msg = e.what();
  } catch (const std::out_of_range &e) {
    error_id = "dtwc:outOfRange"; error_msg = e.what();
  } catch (const std::runtime_error &e) {
    error_id = "dtwc:runtime"; error_msg = e.what();
  } catch (const std::exception &e) {
    error_id = "dtwc:internal"; error_msg = e.what();
  } catch (...) {
    error_id = "dtwc:internal"; error_msg = "Unknown C++ exception";
  }

  // All C++ RAII objects are destroyed before longjmp
  if (!error_msg.empty())
    mexErrMsgIdAndTxt(error_id.c_str(), "%s", error_msg.c_str());
}
