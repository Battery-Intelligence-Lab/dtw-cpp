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
 * @author Volkan Kumtepeli
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
#include "../../dtwc/env.hpp"          // dtwc::Env / device() (contract §1.1, §6)
#include "../../dtwc/error.hpp"        // dtwc::InvalidInput/SolverError/DeviceError/IOError (§5)
#include "../../dtwc/checkpoint.hpp"   // save/load_checkpoint (contract §2.7)
#include "../../dtwc/test_api.hpp"     // dtwc::test::parallelisation()/gpu() (Task 3.3)
#include "../../dtwc/mip/pdlp_lp.hpp" // dtwc::mip::pdlp_lp_bound (cross-language parity)
#include "../../dtwc/core/distance_semantics.hpp" // parse_metric_token (checkpoint + metric routes)
#include "../../dtwc/core/pruned_distance_matrix.hpp" // exact metric-aware matrix builder

#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
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
//  Input validation guards (fix: audit CRITICAL #6 -- mxGetDoubles NULL-deref)
//
//  In the R2018a+ interleaved-complex API, mxGetDoubles() returns NULL when the
//  array is not a REAL DOUBLE (int32/single/logical/char, complex, or sparse).
//  The previous code fed that NULL straight into the DTW kernels, NULL-dereffing
//  and crashing MATLAB. Every entry point now validates class / complexity /
//  shape BEFORE any data-pointer access and throws std::invalid_argument, which
//  mexFunction maps to mexErrMsgIdAndTxt("dtwc:invalidArgument", ...).
// =========================================================================

/// Require a full (non-sparse), real, double, non-empty, <=2-D array.
/// Throws std::invalid_argument BEFORE any mxGetDoubles() access.
static void require_real_double(const mxArray *mx, const char *arg_name) {
  if (mx == nullptr)
    throw std::invalid_argument(std::string(arg_name) + ": argument is missing.");
  if (mxIsComplex(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be real, not complex.");
  if (mxIsSparse(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be a full (non-sparse) array.");
  if (!mxIsDouble(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be of class 'double' (got '"
      + std::string(mxGetClassName(mx)) + "'); convert with double(...) in MATLAB.");
  if (mxIsEmpty(mx))
    throw std::invalid_argument(std::string(arg_name) + " must not be empty.");
  if (mxGetNumberOfDimensions(mx) != 2)
    throw std::invalid_argument(std::string(arg_name) + " must be 1-D or 2-D (got an N-D array).");
}

/// Require a label vector: real, non-empty, full, class int32 OR double.
/// The ARI/NMI entry points intentionally accept both int32 and double labels.
static void require_label_vector(const mxArray *mx, const char *arg_name) {
  if (mx == nullptr)
    throw std::invalid_argument(std::string(arg_name) + ": argument is missing.");
  if (mxIsComplex(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be real, not complex.");
  if (mxIsSparse(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be a full (non-sparse) array.");
  if (!mxIsInt32(mx) && !mxIsDouble(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be an int32 or double vector (got '"
      + std::string(mxGetClassName(mx)) + "').");
  if (mxIsEmpty(mx))
    throw std::invalid_argument(std::string(arg_name) + " must not be empty.");
}

// =========================================================================
//  Helpers: MATLAB <-> C++ type conversion
// =========================================================================

/// MATLAB double vector/row -> std::vector<double>
static std::vector<double> to_std_vector(const mxArray *mx, const char *arg_name = "input") {
  require_real_double(mx, arg_name);
  const double *data = mxGetDoubles(mx);
  size_t n = mxGetNumberOfElements(mx);
  return std::vector<double>(data, data + n);
}

/// MATLAB N x L matrix -> vector of series (each row is one series)
static std::vector<std::vector<double>> matrix_to_series(const mxArray *mx, const char *arg_name = "data") {
  require_real_double(mx, arg_name);
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

/// Require a char/string mxArray BEFORE any dereference (string entry points).
/// Throws std::invalid_argument -> mapped to dtwc:invalidArgument by mexFunction.
static void require_char(const mxArray *mx, const char *arg_name) {
  if (mx == nullptr)
    throw std::invalid_argument(std::string(arg_name) + ": argument is missing.");
  if (!mxIsChar(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be a char row vector (string).");
}

/// MATLAB cell array of numeric vectors -> vector of ragged double series.
/// Validates the container is a cell AND every element is a real, full, double,
/// non-empty vector BEFORE any data-pointer access (audit CRITICAL #6 guard).
static std::vector<std::vector<double>> cell_to_series(const mxArray *mx, const char *arg_name = "data") {
  if (mx == nullptr)
    throw std::invalid_argument(std::string(arg_name) + ": argument is missing.");
  if (!mxIsCell(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be a cell array of numeric vectors.");
  const size_t N = mxGetNumberOfElements(mx);
  if (N == 0)
    throw std::invalid_argument(std::string(arg_name) + " (cell array) must not be empty.");

  std::vector<std::vector<double>> series;
  series.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    const mxArray *cell = mxGetCell(mx, i);
    const std::string elem = std::string(arg_name) + "{" + std::to_string(i + 1) + "}";
    // Full validation BEFORE mxGetDoubles: reject non-double/complex/sparse/empty/N-D.
    require_real_double(cell, elem.c_str());
    const double *data = mxGetDoubles(cell);
    const size_t n = mxGetNumberOfElements(cell);
    series.emplace_back(data, data + n);
  }
  return series;
}

/// MATLAB cell array of char/string -> vector of std::string names.
/// Validates cell-ness and element char-ness BEFORE dereference.
static std::vector<std::string> cell_to_names(const mxArray *mx, size_t expected_N,
                                              const char *arg_name = "names") {
  if (!mxIsCell(mx))
    throw std::invalid_argument(std::string(arg_name) + " must be a cell array of strings.");
  const size_t N = mxGetNumberOfElements(mx);
  if (N != expected_N)
    throw std::invalid_argument(std::string(arg_name) + " length (" + std::to_string(N)
      + ") must match the number of series (" + std::to_string(expected_N) + ").");
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) {
    const mxArray *cell = mxGetCell(mx, i);
    if (cell == nullptr || !mxIsChar(cell))
      throw std::invalid_argument(std::string(arg_name) + "{" + std::to_string(i + 1)
        + "} must be a char row vector.");
    char *s = mxArrayToString(cell);
    names[i] = (s ? std::string(s) : std::string());
    if (s) mxFree(s);
  }
  return names;
}

/// std::vector<int> -> MATLAB 1xN int32 row vector (1-based indexing)
static mxArray *ivec_to_mx_1based(const std::vector<int> &v) {
  mxArray *mx = mxCreateNumericMatrix(1, v.size(), mxINT32_CLASS, mxREAL);
  int32_t *out = static_cast<int32_t *>(mxGetData(mx));
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = static_cast<int32_t>(v[i] + 1);  // 0-based -> 1-based
  return mx;
}

/// Extract scalar double from mxArray (validates numeric/logical + non-empty).
static double get_scalar(const mxArray *mx, const char *arg_name = "argument") {
  if (mx == nullptr || (!mxIsNumeric(mx) && !mxIsLogical(mx)))
    throw std::invalid_argument(std::string(arg_name) + " must be a numeric scalar.");
  if (mxIsEmpty(mx))
    throw std::invalid_argument(std::string(arg_name) + " must not be empty.");
  if (mxGetNumberOfElements(mx) != 1)
    throw std::invalid_argument(std::string(arg_name) + " must be a scalar.");
  return mxGetScalar(mx);
}

/// Decode the public CUDA precision selector without any out-of-range or
/// non-integral floating-to-integer conversion.
static int get_cuda_precision(const mxArray *mx) {
  const double value = get_scalar(mx, "precision");
  if (!std::isfinite(value) || std::floor(value) != value
      || value < 0.0 || value > 2.0) {
    throw dtwc::InvalidInput("Invalid CUDA precision value.");
  }
  return static_cast<int>(value);
}

/// Convert one MATLAB double to int without ever invoking an out-of-range or
/// non-integral float-to-int conversion (both are undefined behaviour, and a
/// NaN/Inf label silently produced a garbage cluster id before this guard).
static int exact_int_from_double(double value, const char *arg_name) {
  constexpr double int_min = static_cast<double>(
    std::numeric_limits<int>::min());
  constexpr double int_max = static_cast<double>(
    std::numeric_limits<int>::max());
  if (!std::isfinite(value) || std::floor(value) != value
      || value < int_min || value > int_max) {
    throw std::invalid_argument(
      std::string(arg_name) + " must be a finite integer in the C++ int range.");
  }
  return static_cast<int>(value);
}

static int get_exact_int(const mxArray *mx, const char *arg_name) {
  return exact_int_from_double(get_scalar(mx, arg_name), arg_name);
}

/// Shift a validated 1-based MATLAB index down to 0-based. INT_MIN is exactly
/// representable as a double and therefore passes exact_int_from_double, so the
/// callers' bare `- 1` was signed overflow (undefined behaviour) at exactly the
/// boundary these helpers exist to make safe.
static int to_0based(int value, const char *arg_name) {
  if (value == std::numeric_limits<int>::min())
    throw std::invalid_argument(
      std::string(arg_name) + " = " + std::to_string(value)
      + " has no 0-based representation in the C++ int range.");
  return value - 1;
}

static int exact_int_1based_to_0based(double value, const char *arg_name) {
  return to_0based(exact_int_from_double(value, arg_name), arg_name);
}

/// Decode a 1-based MATLAB label/index vector (int32 or double) to 0-based ints.
/// Every double element goes through exact_int_from_double, so NaN/Inf/fractional
/// entries are rejected instead of being cast with undefined behaviour.
static std::vector<int> label_vector_to_0based(const mxArray *mx, const char *arg_name) {
  require_label_vector(mx, arg_name);
  const size_t n = mxGetNumberOfElements(mx);
  std::vector<int> out(n);
  if (mxIsInt32(mx)) {
    const int32_t *p = static_cast<const int32_t *>(mxGetData(mx));
    for (size_t i = 0; i < n; ++i) out[i] = to_0based(static_cast<int>(p[i]), arg_name);
  } else {
    const double *p = mxGetDoubles(mx);
    for (size_t i = 0; i < n; ++i)
      out[i] = exact_int_1based_to_0based(p[i], arg_name);
  }
  return out;
}

/// Decode a MATLAB double seed without invoking an out-of-range float-to-int
/// conversion. MATLAB represents every integer exactly only through flintmax.
static std::uint64_t get_random_seed(
  const mxArray *mx,
  std::uint64_t max_seed = (std::uint64_t{1} << 53) - 1)
{
  const double seed = get_scalar(mx, "seed");
  if (!std::isfinite(seed) || seed < 0.0 || std::floor(seed) != seed
      || seed > static_cast<double>(max_seed)) {
    throw std::invalid_argument(
      "seed must be a finite integer in [0, " + std::to_string(max_seed) + "].");
  }
  return static_cast<std::uint64_t>(seed);
}

/// Extract uint64 handle from mxArray
static uint64_t get_handle(const mxArray *mx) {
  if (mx == nullptr)
    throw std::invalid_argument("handle argument is missing.");
  if (mxIsUint64(mx)) {
    if (mxIsEmpty(mx))
      throw std::invalid_argument("handle must not be empty.");
    uint64_t *p = static_cast<uint64_t *>(mxGetData(mx));
    return p[0];
  }
  // Accept double as well (MATLAB defaults to double)
  return static_cast<uint64_t>(get_scalar(mx, "handle"));
}

/// Extract string from mxArray (char array or string)
static std::string get_string(const mxArray *mx) {
  char *str = mxArrayToString(mx);
  if (!str) return "";
  std::string result(str);
  mxFree(str);
  return result;
}

/// Optional trailing string argument; "" when absent or empty.
static std::string optional_string(int nrhs, const mxArray *prhs[], int index,
                                   const char *arg_name) {
  if (nrhs <= index || mxIsEmpty(prhs[index])) return {};
  require_char(prhs[index], arg_name);
  return get_string(prhs[index]);
}

/// Optional trailing integer argument, validated exactly (no UB cast).
static int optional_int(int nrhs, const mxArray *prhs[], int index,
                        const char *arg_name, int fallback) {
  if (nrhs <= index || mxIsEmpty(prhs[index])) return fallback;
  return get_exact_int(prhs[index], arg_name);
}

/// One delimiter character; 0 keeps the extension-derived default.
static char parse_delimiter(const std::string &value) {
  if (value.empty()) return static_cast<char>(0);
  if (value.size() != 1)
    throw std::invalid_argument("delimiter must be a single character.");
  return value[0];
}

/// Optional metric token; absent means L1, matching C++/Python defaults.
static dtwc::core::MetricType optional_metric(int nrhs, const mxArray *prhs[],
                                              int index) {
  const std::string token = optional_string(nrhs, prhs, index, "metric");
  if (token.empty()) return dtwc::core::MetricType::L1;
  return dtwc::core::parse_metric_token(token);
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

  // Guard before data access: 'merges' must be a real double matrix (empty is a
  // valid single-point dendrogram, so non-empty is NOT required here).
  if (mxIsComplex(merges_mx) || mxIsSparse(merges_mx) || !mxIsDouble(merges_mx))
    throw std::invalid_argument("dendrogram.merges must be a real, full, double matrix.");

  // The loop below reads column 3 (data[i + 3 * n_merges]), so the column count
  // is load-bearing: an Nx3 or transposed 'merges' over-reads the heap. Only the
  // empty single-point dendrogram may have no columns.
  const size_t n_merge_cols = mxGetN(merges_mx);
  if (!mxIsEmpty(merges_mx) && n_merge_cols != 4)
    throw std::invalid_argument("dendrogram.merges must have exactly 4 columns "
      "[cluster_a, cluster_b, distance, new_size] (got "
      + std::to_string(n_merge_cols) + ").");

  dend.n_points = get_exact_int(np_mx, "dendrogram.n_points");

  size_t n_merges = mxIsEmpty(merges_mx) ? 0 : mxGetM(merges_mx);
  const double *data = mxGetDoubles(merges_mx);
  dend.merges.resize(n_merges);
  for (size_t i = 0; i < n_merges; ++i) {
    dend.merges[i].cluster_a = exact_int_1based_to_0based(
      data[i + 0 * n_merges], "dendrogram.merges(:,1)"); // 1-based -> 0-based
    dend.merges[i].cluster_b = exact_int_1based_to_0based(
      data[i + 1 * n_merges], "dendrogram.merges(:,2)");
    dend.merges[i].distance = data[i + 2 * n_merges];
    dend.merges[i].new_size = exact_int_from_double(
      data[i + 3 * n_merges], "dendrogram.merges(:,4)");
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
  if (s == "cuda") return dtwc::DistanceMatrixStrategy::CUDA;
  if (s == "metal") return dtwc::DistanceMatrixStrategy::Metal;
  throw std::invalid_argument("Unknown distance strategy: '" + s + "'. "
    "Valid: 'auto', 'brute_force', 'pruned', 'cuda', 'metal'.");
}

/// Parse linkage string -> enum
static dtwc::algorithms::Linkage parse_linkage(const std::string &s) {
  if (s == "single") return dtwc::algorithms::Linkage::Single;
  if (s == "complete") return dtwc::algorithms::Linkage::Complete;
  if (s == "average") return dtwc::algorithms::Linkage::Average;
  throw std::invalid_argument("Unknown linkage: '" + s + "'. Valid: 'single', 'complete', 'average'.");
}

/// Parse clustering method string -> enum (contract §2.1 set_method).
static dtwc::Method parse_method(const std::string &s) {
  if (s == "kmedoids" || s == "pam" || s == "auto") return dtwc::Method::Kmedoids;
  if (s == "mip") return dtwc::Method::MIP;
  throw std::invalid_argument("Unknown method: '" + s + "'. Valid: 'kmedoids', 'mip'.");
}

/// Parse MIP solver string -> enum (contract §2.1 set_solver).
static dtwc::Solver parse_solver(const std::string &s) {
  if (s == "highs") return dtwc::Solver::HiGHS;
  if (s == "gurobi") return dtwc::Solver::Gurobi;
  throw std::invalid_argument("Unknown solver: '" + s + "'. Valid: 'highs', 'gurobi'.");
}

/// Parse lower-bound strategy string -> enum (contract §2.1 set_lb_strategy).
static dtwc::LowerBoundStrategy parse_lb_strategy(const std::string &s) {
  if (s == "auto") return dtwc::LowerBoundStrategy::Auto;
  if (s == "none") return dtwc::LowerBoundStrategy::None;
  if (s == "kim") return dtwc::LowerBoundStrategy::Kim;
  if (s == "keogh") return dtwc::LowerBoundStrategy::Keogh;
  if (s == "kim_keogh" || s == "kimkeogh") return dtwc::LowerBoundStrategy::KimKeogh;
  if (s == "enhanced") return dtwc::LowerBoundStrategy::Enhanced;
  if (s == "webb") return dtwc::LowerBoundStrategy::Webb;
  throw std::invalid_argument("Unknown lb_strategy: '" + s + "'. "
    "Valid: 'auto', 'none', 'kim', 'keogh', 'kim_keogh', 'enhanced', 'webb'.");
}

/// Parse storage policy string -> enum (contract §2.1 set_storage_policy).
static dtwc::core::StoragePolicy parse_storage_policy(const std::string &s) {
  if (s == "auto") return dtwc::core::StoragePolicy::Auto;
  if (s == "heap") return dtwc::core::StoragePolicy::Heap;
  if (s == "mmap") return dtwc::core::StoragePolicy::Mmap;
  throw std::invalid_argument("Unknown storage_policy: '" + s + "'. "
    "Valid: 'auto', 'heap', 'mmap'.");
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
  prob->set_verbose(false);
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

  const char *field_names[] = {
    "name", "size", "band", "verbose", "max_iter", "n_repetitions",
    "dist_filled"
  };
  mxArray *s = mxCreateStructMatrix(1, 1, 7, field_names);

  mxSetField(s, 0, "name", mxCreateString(prob.name().c_str()));
  mxSetField(s, 0, "size", mxCreateDoubleScalar(static_cast<double>(prob.size())));
  mxSetField(s, 0, "band", mxCreateDoubleScalar(static_cast<double>(prob.band)));
  mxSetField(s, 0, "verbose", mxCreateLogicalScalar(prob.verbose()));
  mxSetField(s, 0, "max_iter", mxCreateDoubleScalar(prob.max_iter()));
  mxSetField(
    s, 0, "n_repetitions", mxCreateDoubleScalar(prob.n_repetitions()));
  mxSetField(s, 0, "dist_filled", mxCreateLogicalScalar(prob.is_distance_matrix_filled()));

  plhs[0] = s;
}

// =========================================================================
//  Problem property get/set commands
// =========================================================================

static void cmd_Problem_set_data(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Args: handle, data, [names cell], [ndim]. `data` is either an N x L real
  // double matrix (each row a series) OR a cell array of numeric row vectors
  // (ragged / variable-length series, contract §2.1 "data (owning)").
  if (nrhs < 3) throw std::invalid_argument("Problem_set_data requires handle and data.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  // Ragged (cell array) vs rectangular (matrix). Both paths validate class /
  // complexity / shape BEFORE any mxGetDoubles() access (audit CRITICAL #6).
  std::vector<std::vector<double>> series =
    mxIsCell(prhs[2]) ? cell_to_series(prhs[2]) : matrix_to_series(prhs[2]);
  const size_t N = series.size();

  // Optional series names (cell array of char). Empty ([]) => auto-derive "0..N-1".
  std::vector<std::string> names;
  if (nrhs > 3 && !mxIsEmpty(prhs[3])) {
    names = cell_to_names(prhs[3], N, "names");
  } else {
    names.resize(N);
    for (size_t i = 0; i < N; ++i) names[i] = std::to_string(i);
  }

  // Optional ndim (multivariate interleaved layout). Default 1 (univariate).
  // Data::validate_ndim() throws if any series flat-size is not divisible by ndim.
  size_t ndim = 1;
  if (nrhs > 4 && !mxIsEmpty(prhs[4])) {
    const double nd = get_scalar(prhs[4], "ndim");
    if (nd < 1.0) throw std::invalid_argument("ndim must be a positive integer.");
    ndim = static_cast<size_t>(nd);
  }

  dtwc::Data data(std::move(series), std::move(names), ndim);
  prob.set_data(std::move(data));
}

static void cmd_Problem_set_band(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_band requires handle and band value.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_band(static_cast<int>(get_scalar(prhs[2])));
}

static void cmd_Problem_get_band(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_band requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(prob.band));
}

static void cmd_Problem_set_verbose(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_verbose requires handle and bool.");
  if (!mxIsLogical(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
    throw std::invalid_argument("verbose must be a logical scalar.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_verbose(mxIsLogicalScalarTrue(prhs[2]));
}

static void cmd_Problem_set_max_iter(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_max_iter requires handle and value.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_max_iter(static_cast<int>(get_scalar(prhs[2])));
}

static void cmd_Problem_set_n_repetition(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_n_repetition requires handle and value.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_n_repetitions(static_cast<int>(get_scalar(prhs[2])));
}

static void cmd_Problem_set_n_clusters(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_n_clusters requires handle and k.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_n_clusters(static_cast<int>(get_scalar(prhs[2])));
}

static void cmd_Problem_set_missing_strategy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_missing_strategy requires handle and string.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  std::string s = get_string(prhs[2]);
  prob.set_missing_strategy(parse_missing_strategy(s));
}

static void cmd_Problem_set_distance_strategy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_distance_strategy requires handle and string.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  std::string s = get_string(prhs[2]);
  prob.set_distance_strategy(parse_distance_strategy(s));
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
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(prob.n_clusters()));
}

static void cmd_Problem_get_name(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_name requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateString(prob.name().c_str());
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
  plhs[0] = mxCreateLogicalScalar(prob.is_distance_matrix_filled());
}

// =========================================================================
//  Problem method commands
// =========================================================================

static void cmd_Problem_fill_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_fill_distance_matrix requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.fill_distance_matrix();
}

static void cmd_Problem_dist_by_ind(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 4) throw std::invalid_argument("Problem_dist_by_ind requires handle, i, j.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  // Convert from MATLAB 1-based to C++ 0-based
  int i = static_cast<int>(get_scalar(prhs[2])) - 1;
  int j = static_cast<int>(get_scalar(prhs[3])) - 1;
  double d = prob.dist_by_ind(i, j);
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
  double cost = prob.find_total_cost();
  plhs[0] = mxCreateDoubleScalar(cost);
}

static void cmd_Problem_get_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_distance_matrix requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  const auto &dm = prob.dense_distance_matrix();
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

  require_real_double(prhs[2], "distance_matrix");
  size_t N = mxGetM(prhs[2]);
  if (N != mxGetN(prhs[2]))
    throw std::invalid_argument("Distance matrix must be square.");
  if (N != prob.size())
    throw std::invalid_argument("Distance matrix size does not match problem size.");

  auto &dm = prob.dense_distance_matrix();
  dm.resize(N);
  const double *data = mxGetDoubles(prhs[2]);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      dm.set(i, j, data[i + j * N]);  // column-major

}

// =========================================================================
//  Device / Env commands (contract §1.1, §6 — delegate to dtwc::Env)
// =========================================================================

/// Normalised device name string ("cpu"/"gpu"/"gpu:N"/"hpc") from the singleton Env.
static std::string current_device_string() {
  std::string s = dtwc::to_string(dtwc::env().device());
  const int idx = dtwc::env().device_index();
  if (dtwc::env().device() == dtwc::Device::GPU && idx > 0)
    s += ":" + std::to_string(idx);
  return s;
}

/// set_device(name) -> normalised name. Delegates to dtwc::env().set_device(),
/// which throws dtwc::DeviceError (mapped to dtwc:deviceError) on any unknown
/// name / gpu-without-backend / hpc .env failure — NEVER a silent fallback.
static void cmd_set_device(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("set_device requires a device-name string.");
  require_char(prhs[1], "device");   // validate BEFORE mxArrayToString deref
  const std::string name = get_string(prhs[1]);
  dtwc::env().set_device(name);
  plhs[0] = mxCreateString(current_device_string().c_str());
}

/// get_device() -> normalised name of the currently selected device.
static void cmd_get_device(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  plhs[0] = mxCreateString(current_device_string().c_str());
}

// =========================================================================
//  dtwc.test introspection API (Task 3.3) — struct with the SAME field names
//  as C++ dtwc::test::* and Python dtwcpp.test.*. Both take no data arguments
//  (like get_device / system_check), so there is nothing to require_*-validate.
// =========================================================================

/// test_parallelisation() -> struct {available, max_threads, threads_engaged, pass, reason}.
/// Runs a REAL OpenMP region via dtwc::test::parallelisation(); on the serial MEX
/// build (DTWC_SEQUENTIAL_BUILD) available=false with a non-empty reason.
static void cmd_test_parallelisation(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  const dtwc::test::ParallelReport r = dtwc::test::parallelisation();
  const char *fields[] = { "available", "max_threads", "threads_engaged", "pass", "reason" };
  mxArray *s = mxCreateStructMatrix(1, 1, 5, fields);
  mxSetField(s, 0, "available", mxCreateLogicalScalar(r.available));
  mxSetField(s, 0, "max_threads", mxCreateDoubleScalar(static_cast<double>(r.max_threads)));
  mxSetField(s, 0, "threads_engaged", mxCreateDoubleScalar(static_cast<double>(r.threads_engaged)));
  mxSetField(s, 0, "pass", mxCreateLogicalScalar(r.pass));
  mxSetField(s, 0, "reason", mxCreateString(r.reason.c_str()));
  plhs[0] = s;
}

/// test_gpu() -> struct {available, backend, device_name, validated, pass, reason}.
/// Executes a tiny real GPU kernel + CPU-oracle validation via dtwc::test::gpu();
/// when no GPU backend is compiled in, available=false with a naming reason.
static void cmd_test_gpu(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  const dtwc::test::GpuReport r = dtwc::test::gpu();
  const char *fields[] = { "available", "backend", "device_name", "validated", "pass", "reason" };
  mxArray *s = mxCreateStructMatrix(1, 1, 6, fields);
  mxSetField(s, 0, "available", mxCreateLogicalScalar(r.available));
  mxSetField(s, 0, "backend", mxCreateString(r.backend.c_str()));
  mxSetField(s, 0, "device_name", mxCreateString(r.device_name.c_str()));
  mxSetField(s, 0, "validated", mxCreateLogicalScalar(r.validated));
  mxSetField(s, 0, "pass", mxCreateLogicalScalar(r.pass));
  mxSetField(s, 0, "reason", mxCreateString(r.reason.c_str()));
  plhs[0] = s;
}

// =========================================================================
//  Problem: MIP surface, solver, strategies, output folder, CUDA dispatch
// =========================================================================

static void cmd_Problem_set_method(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_method requires handle and method string.");
  require_char(prhs[2], "method");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_method(parse_method(get_string(prhs[2])));
}

static void cmd_Problem_set_solver(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_solver requires handle and solver string.");
  require_char(prhs[2], "solver");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const bool ok = prob.set_solver(parse_solver(get_string(prhs[2])));
  plhs[0] = mxCreateLogicalScalar(ok);  // false => requested solver not compiled in
}

static void cmd_Problem_set_lb_strategy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_lb_strategy requires handle and strategy string.");
  require_char(prhs[2], "lb_strategy");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const auto candidate = parse_lb_strategy(get_string(prhs[2]));
  prob.set_lb_strategy(candidate);
}

static void cmd_Problem_set_storage_policy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_storage_policy requires handle and policy string.");
  require_char(prhs[2], "storage_policy");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const auto candidate = parse_storage_policy(get_string(prhs[2]));
  prob.set_storage_policy(candidate);
}

static void cmd_Problem_set_output_folder(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_output_folder requires handle and folder string.");
  require_char(prhs[2], "output_folder");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_output_folder(std::filesystem::path(get_string(prhs[2])));
}

/// set_mip_settings(struct): reads any subset of the MIPSettings fields present.
static void cmd_Problem_set_mip_settings(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_mip_settings requires handle and a struct.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const mxArray *s = prhs[2];
  if (!mxIsStruct(s))
    throw std::invalid_argument("mip_settings must be a struct (fields: mip_gap, time_limit_sec, "
      "warm_start, numeric_focus, mip_focus, verbose_solver, max_benders_iter, benders, "
      "lr_max_nodes).");

  dtwc::MIPSettings m = prob.mip_settings; // start from current, override present fields
  if (mxArray *f = mxGetField(s, 0, "mip_gap"))        m.mip_gap        = get_scalar(f, "mip_gap");
  if (mxArray *f = mxGetField(s, 0, "time_limit_sec")) m.time_limit_sec = static_cast<int>(get_scalar(f, "time_limit_sec"));
  if (mxArray *f = mxGetField(s, 0, "warm_start"))     m.warm_start     = (get_scalar(f, "warm_start") != 0.0);
  if (mxArray *f = mxGetField(s, 0, "numeric_focus"))  m.numeric_focus  = static_cast<int>(get_scalar(f, "numeric_focus"));
  if (mxArray *f = mxGetField(s, 0, "mip_focus"))      m.mip_focus      = static_cast<int>(get_scalar(f, "mip_focus"));
  if (mxArray *f = mxGetField(s, 0, "verbose_solver")) m.verbose_solver = (get_scalar(f, "verbose_solver") != 0.0);
  if (mxArray *f = mxGetField(s, 0, "max_benders_iter")) m.max_benders_iter = static_cast<int>(get_scalar(f, "max_benders_iter"));
  if (mxArray *f = mxGetField(s, 0, "lr_max_nodes"))
    m.lr_max_nodes = exact_int_from_double(get_scalar(f, "lr_max_nodes"), "lr_max_nodes");
  if (mxArray *f = mxGetField(s, 0, "benders")) {
    if (!mxIsChar(f)) throw std::invalid_argument("mip_settings.benders must be a string ('auto'/'on'/'off').");
    m.benders = get_string(f);
  }
  prob.mip_settings = m;
}

/// get_mip_settings() -> struct mirroring MIPSettings (round-trip / introspection).
static void cmd_Problem_get_mip_settings(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_mip_settings requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const auto &m = prob.mip_settings;
  const char *fields[] = { "mip_gap", "time_limit_sec", "warm_start", "numeric_focus",
                           "mip_focus", "verbose_solver", "max_benders_iter", "benders",
                           "lr_max_nodes" };
  mxArray *s = mxCreateStructMatrix(1, 1, 9, fields);
  mxSetField(s, 0, "mip_gap", mxCreateDoubleScalar(m.mip_gap));
  mxSetField(s, 0, "time_limit_sec", mxCreateDoubleScalar(m.time_limit_sec));
  mxSetField(s, 0, "warm_start", mxCreateLogicalScalar(m.warm_start));
  mxSetField(s, 0, "numeric_focus", mxCreateDoubleScalar(m.numeric_focus));
  mxSetField(s, 0, "mip_focus", mxCreateDoubleScalar(m.mip_focus));
  mxSetField(s, 0, "verbose_solver", mxCreateLogicalScalar(m.verbose_solver));
  mxSetField(s, 0, "max_benders_iter", mxCreateDoubleScalar(m.max_benders_iter));
  mxSetField(s, 0, "benders", mxCreateString(m.benders.c_str()));
  mxSetField(s, 0, "lr_max_nodes", mxCreateDoubleScalar(static_cast<double>(m.lr_max_nodes)));
  plhs[0] = s;
}

/// set_cuda_settings(device_id, precision) — CUDA dispatch passthrough (contract §2.1).
/// precision: 0 = Auto, 1 = FP32, 2 = FP64 (see CUDASettings docs).
/// set_checkpoint(handle, struct) -- writes Problem::checkpoint, which
/// fill_distance_matrix() consumes (contract 2.7). Fields are optional; the
/// current value is kept for any field the struct omits.
static void cmd_Problem_set_checkpoint(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3)
    throw std::invalid_argument("Problem_set_checkpoint requires handle and a struct.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const mxArray *s = prhs[2];
  if (!mxIsStruct(s))
    throw std::invalid_argument("checkpoint must be a struct (fields: directory, "
      "save_interval, enabled).");

  dtwc::CheckpointOptions options = prob.checkpoint;
  if (mxArray *f = mxGetField(s, 0, "directory")) {
    require_char(f, "checkpoint.directory");
    options.directory = get_string(f);
  }
  if (mxArray *f = mxGetField(s, 0, "save_interval"))
    options.save_interval =
      exact_int_from_double(get_scalar(f, "save_interval"), "checkpoint.save_interval");
  if (mxArray *f = mxGetField(s, 0, "enabled"))
    options.enabled = (get_scalar(f, "enabled") != 0.0);
  prob.checkpoint = options;
}

/// get_checkpoint() -> struct mirroring CheckpointOptions (round-trip).
static void cmd_Problem_get_checkpoint(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_checkpoint requires a handle.");
  const auto &options = HandleManager<dtwc::Problem>::get(get_handle(prhs[1]))->checkpoint;
  const char *fields[] = { "directory", "save_interval", "enabled" };
  mxArray *s = mxCreateStructMatrix(1, 1, 3, fields);
  mxSetField(s, 0, "directory", mxCreateString(options.directory.c_str()));
  mxSetField(s, 0, "save_interval",
             mxCreateDoubleScalar(static_cast<double>(options.save_interval)));
  mxSetField(s, 0, "enabled", mxCreateLogicalScalar(options.enabled));
  plhs[0] = s;
}

static void cmd_Problem_set_cuda_settings(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_cuda_settings requires handle and device_id.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  auto settings = prob.cuda_settings;
  settings.device_id = get_exact_int(prhs[2], "device_id");
  if (nrhs > 3) settings.precision = get_cuda_precision(prhs[3]);
  prob.set_cuda_settings(settings);
}

/// get_cuda_settings() -> struct mirroring CUDASettings (round-trip).
static void cmd_Problem_get_cuda_settings(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_cuda_settings requires a handle.");
  const auto &settings = HandleManager<dtwc::Problem>::get(get_handle(prhs[1]))->cuda_settings;
  const char *fields[] = { "device_id", "precision" };
  mxArray *s = mxCreateStructMatrix(1, 1, 2, fields);
  mxSetField(s, 0, "device_id", mxCreateDoubleScalar(static_cast<double>(settings.device_id)));
  mxSetField(s, 0, "precision", mxCreateDoubleScalar(static_cast<double>(settings.precision)));
  plhs[0] = s;
}

static void cmd_Problem_refresh_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_refresh_distance_matrix requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.refresh_distance_matrix();
}

static void cmd_Problem_read_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_read_distance_matrix requires handle and path.");
  require_char(prhs[2], "path");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.read_distance_matrix(std::filesystem::path(get_string(prhs[2])));
}

static void cmd_Problem_max_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_max_distance requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(prob.max_distance()));
}

static void cmd_Problem_n_clusters(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_n_clusters requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(prob.n_clusters()));
}

// =========================================================================
//  Checkpoint / resume commands (contract §2.7)
// =========================================================================

/// Reconstruct a core::ClusteringResult from a MATLAB result struct.
/// Converts 1-based labels/medoid_indices back to 0-based at the MEX boundary.
static dtwc::core::ClusteringResult mx_to_clustering_result(const mxArray *mx) {
  if (!mxIsStruct(mx))
    throw std::invalid_argument("result must be a struct with fields labels, medoid_indices, "
      "total_cost, iterations, converged.");
  dtwc::core::ClusteringResult r;

  const mxArray *lab = mxGetField(mx, 0, "labels");
  const mxArray *med = mxGetField(mx, 0, "medoid_indices");
  if (!lab || !med)
    throw std::invalid_argument("result struct is missing 'labels' or 'medoid_indices'.");
  r.labels = label_vector_to_0based(lab, "result.labels");
  r.medoid_indices = label_vector_to_0based(med, "result.medoid_indices");
  if (mxArray *f = mxGetField(mx, 0, "total_cost")) r.total_cost = get_scalar(f, "total_cost");
  if (mxArray *f = mxGetField(mx, 0, "iterations")) r.iterations = static_cast<int>(get_scalar(f, "iterations"));
  if (mxArray *f = mxGetField(mx, 0, "converged")) r.converged = (get_scalar(f, "converged") != 0.0);
  return r;
}

static void cmd_save_checkpoint(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("save_checkpoint requires handle and directory path.");
  require_char(prhs[2], "path");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  // The metric is part of the checkpoint identity: an L1 and a SquaredL2 matrix
  // over the same data are different numbers (C++/Python default to L1 too).
  dtwc::save_checkpoint(prob, get_string(prhs[2]), optional_metric(nrhs, prhs, 3));
}

static void cmd_load_checkpoint(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("load_checkpoint requires handle and directory path.");
  require_char(prhs[2], "path");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const bool ok = dtwc::load_checkpoint(prob, get_string(prhs[2]),
                                       optional_metric(nrhs, prhs, 3));
  plhs[0] = mxCreateLogicalScalar(ok);
}

static void cmd_save_binary_checkpoint(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("save_binary_checkpoint requires a result struct and a file path.");
  require_char(prhs[2], "path");
  const auto result = mx_to_clustering_result(prhs[1]);
  dtwc::save_binary_checkpoint(result, std::filesystem::path(get_string(prhs[2])));
}

static void cmd_load_binary_checkpoint(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("load_binary_checkpoint requires a file path.");
  require_char(prhs[1], "path");
  dtwc::core::ClusteringResult result;
  const bool ok = dtwc::load_binary_checkpoint(result, std::filesystem::path(get_string(prhs[1])));
  if (!ok)
    throw std::runtime_error("load_binary_checkpoint: file not found or invalid header: "
      + get_string(prhs[1]));
  plhs[0] = clustering_result_to_mx(result);  // 0-based -> 1-based inside
}

// =========================================================================
//  Stateless DTW distance functions
// =========================================================================

static void cmd_dtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("dtw_distance requires x and y.");
  require_real_double(prhs[1], "x");
  require_real_double(prhs[2], "y");
  const double *x = mxGetDoubles(prhs[1]);
  size_t nx = mxGetNumberOfElements(prhs[1]);
  const double *y = mxGetDoubles(prhs[2]);
  size_t ny = mxGetNumberOfElements(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::dtwBanded<double>(x, nx, y, ny, band));
}

static void cmd_ddtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("ddtw_distance requires x and y.");
  require_real_double(prhs[1], "x");
  require_real_double(prhs[2], "y");
  const double *x = mxGetDoubles(prhs[1]);
  size_t nx = mxGetNumberOfElements(prhs[1]);
  const double *y = mxGetDoubles(prhs[2]);
  size_t ny = mxGetNumberOfElements(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::ddtwBanded<double>(x, nx, y, ny, band));
}

static void cmd_wdtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("wdtw_distance requires x and y.");
  require_real_double(prhs[1], "x");
  require_real_double(prhs[2], "y");
  const double *x = mxGetDoubles(prhs[1]);
  size_t nx = mxGetNumberOfElements(prhs[1]);
  const double *y = mxGetDoubles(prhs[2]);
  size_t ny = mxGetNumberOfElements(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  double g = 0.05;
  if (nrhs > 4) g = get_scalar(prhs[4]);
  plhs[0] = mxCreateDoubleScalar(dtwc::wdtwBanded<double>(x, nx, y, ny, band, g));
}

static void cmd_adtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("adtw_distance requires x and y.");
  require_real_double(prhs[1], "x");
  require_real_double(prhs[2], "y");
  const double *x = mxGetDoubles(prhs[1]);
  size_t nx = mxGetNumberOfElements(prhs[1]);
  const double *y = mxGetDoubles(prhs[2]);
  size_t ny = mxGetNumberOfElements(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  double penalty = 1.0;
  if (nrhs > 4) penalty = get_scalar(prhs[4]);
  plhs[0] = mxCreateDoubleScalar(dtwc::adtwBanded<double>(x, nx, y, ny, band, penalty));
}

static void cmd_soft_dtw_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("soft_dtw_distance requires x and y.");
  auto x = to_std_vector(prhs[1], "x");
  auto y = to_std_vector(prhs[2], "y");
  double gamma = 1.0;
  if (nrhs > 3) gamma = get_scalar(prhs[3]);
  plhs[0] = mxCreateDoubleScalar(dtwc::soft_dtw<double>(x, y, gamma));
}

static void cmd_soft_dtw_gradient(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("soft_dtw_gradient requires x and y.");
  auto x = to_std_vector(prhs[1], "x");
  auto y = to_std_vector(prhs[2], "y");
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
  auto x = to_std_vector(prhs[1], "x");
  auto y = to_std_vector(prhs[2], "y");
  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::dtwMissing_banded<double>(x, y, band));
}

static void cmd_dtw_arow_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("dtw_arow_distance requires x and y.");
  auto x = to_std_vector(prhs[1], "x");
  auto y = to_std_vector(prhs[2], "y");
  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::dtwAROW_banded<double>(x, y, band));
}

static void cmd_compute_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("compute_distance_matrix requires a data matrix.");
  auto series = matrix_to_series(prhs[1]);
  const size_t N = series.size();
  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 2) band = static_cast<int>(get_scalar(prhs[2]));

  // Use Problem + fill_distance_matrix() for OpenMP parallelism and LB pruning
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) names[i] = std::to_string(i);

  dtwc::Problem prob("matlab_distmat");
  prob.band = band;
  prob.set_verbose(false);
  dtwc::Data data(std::move(series), std::move(names));
  prob.set_data(std::move(data));
  prob.fill_distance_matrix();

  // Copy from Problem's distance matrix to MATLAB output (column-major)
  mxArray *result = mxCreateDoubleMatrix(N, N, mxREAL);
  double *out = mxGetDoubles(result);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      out[i + j * N] = prob.dist_by_ind(static_cast<int>(i), static_cast<int>(j));

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

  // Omitted seed preserves the mutable legacy Tier-2 behaviour. MATLAB Tier-1
  // passes the shared default explicitly, so it never consumes global RNG state.
  auto result = nrhs > 4
    ? dtwc::fast_pam_seeded(
        prob, k, get_random_seed(prhs[4]), max_iter)
    : dtwc::fast_pam(prob, k, max_iter);

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
  if (nrhs > 6)
    opts.random_seed = static_cast<unsigned>(
      get_random_seed(prhs[6], std::numeric_limits<unsigned>::max()));

  auto result = dtwc::algorithms::fast_clara(prob, opts);
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
  if (nrhs > 6)
    opts.random_seed = static_cast<unsigned>(
      get_random_seed(prhs[6], std::numeric_limits<unsigned>::max()));

  auto result = dtwc::algorithms::clarans(prob, opts);
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
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::davies_bouldin(prob));
}

static void cmd_dunn_index(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("dunn_index requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::dunn(prob));
}

static void cmd_inertia(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("inertia requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::inertia(prob));
}

static void cmd_calinski_harabasz_index(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("calinski_harabasz_index requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::calinski_harabasz(prob));
}

static void cmd_adjusted_rand_index(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("adjusted_rand_index requires two label vectors.");
  const std::vector<int> labels1 = label_vector_to_0based(prhs[1], "labels_1");
  const std::vector<int> labels2 = label_vector_to_0based(prhs[2], "labels_2");
  if (labels1.size() != labels2.size())
    throw std::invalid_argument("Label vectors must have the same length.");

  plhs[0] = mxCreateDoubleScalar(dtwc::scores::adjusted_rand(labels1, labels2));
}

static void cmd_normalized_mutual_information(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("normalized_mutual_information requires two label vectors.");
  const std::vector<int> labels1 = label_vector_to_0based(prhs[1], "labels_1");
  const std::vector<int> labels2 = label_vector_to_0based(prhs[2], "labels_2");
  if (labels1.size() != labels2.size())
    throw std::invalid_argument("Label vectors must have the same length.");

  plhs[0] = mxCreateDoubleScalar(dtwc::scores::normalized_mutual_info(labels1, labels2));
}

// =========================================================================
//  LP-relaxation bound (PDLP) — cross-language parity with the Python binding
// =========================================================================

static void cmd_pdlp_gpu_available(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  (void)nlhs; (void)nrhs; (void)prhs;
  plhs[0] = mxCreateLogicalScalar(dtwc::mip::pdlp_gpu_available());
}

static void cmd_pdlp_lp_bound(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  (void)nlhs;
  if (nrhs < 3)
    throw std::invalid_argument("pdlp_lp_bound requires a square distance matrix and k.");
  require_real_double(prhs[1], "D");
  const size_t rows = mxGetM(prhs[1]);
  const size_t cols = mxGetN(prhs[1]);
  if (rows != cols)
    throw std::invalid_argument("pdlp_lp_bound: D must be square (got "
      + std::to_string(rows) + "x" + std::to_string(cols) + ").");
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("pdlp_lp_bound: D is larger than the int index range.");
  const int N = static_cast<int>(rows);
  const int k = get_exact_int(prhs[2], "k");

  // MATLAB stores column-major; the C++ routine indexes D[i*N + j] row-major.
  const double *src = mxGetDoubles(prhs[1]);
  std::vector<double> D(rows * cols);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      D[i * cols + j] = src[i + j * rows];

  // Name/value options carry the C++ PdlpParams field names verbatim.
  dtwc::mip::PdlpParams params;
  if (((nrhs - 3) % 2) != 0)
    throw std::invalid_argument("pdlp_lp_bound: options must be name/value pairs.");
  for (int a = 3; a + 1 < nrhs; a += 2) {
    require_char(prhs[a], "option name");
    const std::string name = get_string(prhs[a]);
    if (name == "variant") {
      require_char(prhs[a + 1], "variant");
      params.variant = get_string(prhs[a + 1]);
    } else if (name == "tol") {
      params.tol = get_scalar(prhs[a + 1], "tol");
    } else if (name == "iteration_limit") {
      params.iteration_limit = static_cast<long>(get_exact_int(prhs[a + 1], "iteration_limit"));
    } else if (name == "use_gpu") {
      params.use_gpu = (get_scalar(prhs[a + 1], "use_gpu") != 0.0);
    } else if (name == "verbose") {
      params.verbose = (get_scalar(prhs[a + 1], "verbose") != 0.0);
    } else {
      throw std::invalid_argument("pdlp_lp_bound: unknown option '" + name
        + "'. Valid: variant, tol, iteration_limit, use_gpu, verbose.");
    }
  }

  const dtwc::mip::PdlpResult result = dtwc::mip::pdlp_lp_bound(D.data(), N, k, params);

  const char *fields[] = { "lp_bound", "solved", "iterations", "gpu_used" };
  mxArray *out = mxCreateStructMatrix(1, 1, 4, fields);
  mxSetField(out, 0, "lp_bound", mxCreateDoubleScalar(result.lp_bound));
  mxSetField(out, 0, "solved", mxCreateLogicalScalar(result.solved));
  mxSetField(out, 0, "iterations", mxCreateDoubleScalar(static_cast<double>(result.iterations)));
  mxSetField(out, 0, "gpu_used", mxCreateLogicalScalar(result.gpu_used));
  plhs[0] = out;
}

// =========================================================================
//  Tier-1 API (contract 1.3 / 1.4): ONE C++ route
//
//  cluster.m parses arguments and calls tier1_cluster once. Method routing,
//  the k <= N guard, the per-call device override, max_iter, and the
//  skip_cols/skip_rows source semantics are all decided by dtwc::cluster(),
//  so the MATLAB layer has no routing logic that can drift from C++.
// =========================================================================

/// tier1_cluster(source, k, method, band, device, max_iter,
///               skip_cols, skip_rows, delimiter, name) -> struct.
/// `source` is a char path, an N x L real double matrix, or a cell of series.
static void cmd_tier1_cluster(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3)
    throw std::invalid_argument("tier1_cluster requires a data source and k.");
  const int k = get_exact_int(prhs[2], "k");
  const std::string method = optional_string(nrhs, prhs, 3, "method");
  const int band = optional_int(nrhs, prhs, 4, "band", dtwc::settings::DEFAULT_BAND);
  const std::string device = optional_string(nrhs, prhs, 5, "device");
  const int max_iter = optional_int(nrhs, prhs, 6, "max_iter", 100);
  const int skip_cols = optional_int(nrhs, prhs, 7, "skip_cols", 0);
  const int skip_rows = optional_int(nrhs, prhs, 8, "skip_rows", 0);
  const char delimiter = parse_delimiter(optional_string(nrhs, prhs, 9, "delimiter"));
  const std::string name = optional_string(nrhs, prhs, 10, "name");

  const dtwc::Dataset dataset = mxIsChar(prhs[1])
    ? dtwc::load(std::filesystem::path(get_string(prhs[1])), skip_cols, skip_rows,
                 delimiter, name)
    : dtwc::load(mxIsCell(prhs[1]) ? cell_to_series(prhs[1], "data")
                                   : matrix_to_series(prhs[1], "data"),
                 skip_cols, skip_rows, delimiter, name);

  auto result = std::make_shared<dtwc::Result>(dtwc::cluster(
    dataset, k, method.empty() ? std::string("pam") : method, band, device, max_iter));
  const uint64_t h = HandleManager<dtwc::Result>::create(result);

  const char *fields[] = { "handle", "labels", "medoid_indices", "total_cost",
                           "device", "name" };
  mxArray *s = mxCreateStructMatrix(1, 1, 6, fields);
  mxArray *handle_mx = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *static_cast<uint64_t *>(mxGetData(handle_mx)) = h;
  mxSetField(s, 0, "handle", handle_mx);
  mxSetField(s, 0, "labels", ivec_to_mx_1based(result->labels()));
  mxSetField(s, 0, "medoid_indices", ivec_to_mx_1based(result->medoids()));
  mxSetField(s, 0, "total_cost", mxCreateDoubleScalar(result->cost()));
  mxSetField(s, 0, "device", mxCreateString(result->device().c_str()));
  mxSetField(s, 0, "name", mxCreateString(dataset.name().c_str()));
  plhs[0] = s;
}

static void cmd_Result_delete(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Result_delete requires a handle.");
  HandleManager<dtwc::Result>::destroy(get_handle(prhs[1]));
}

/// Result_score(handle, name) -> scalar. dtwc::Result::score owns the name set.
static void cmd_Result_score(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3)
    throw std::invalid_argument("Result_score requires a handle and a score name.");
  require_char(prhs[2], "score");
  const auto &res = *HandleManager<dtwc::Result>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(res.score(get_string(prhs[2])));
}

/// Result_save(handle, directory). The C++ writer owns the four CSVs, so the
/// series names it emits are the dataset's, not MATLAB ordinals.
static void cmd_Result_save(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3)
    throw std::invalid_argument("Result_save requires a handle and a directory.");
  require_char(prhs[2], "directory");
  const auto &res = *HandleManager<dtwc::Result>::get(get_handle(prhs[1]));
  res.save(std::filesystem::path(get_string(prhs[2])));
}

/// DTWClustering_compute_distance_matrix(X, band, metric) -> N x N matrix.
/// The estimator's non-L1 route: the same exact builder the Python estimator
/// uses when metric != 'l1' (Problem's lazy matrix is intrinsically L1).
static void cmd_DTWClustering_compute_distance_matrix(
  int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2)
    throw std::invalid_argument(
      "DTWClustering_compute_distance_matrix requires a data matrix.");
  const auto series = matrix_to_series(prhs[1]);
  const int band = optional_int(nrhs, prhs, 2, "band", dtwc::settings::DEFAULT_BAND);
  const dtwc::core::MetricType metric = optional_metric(nrhs, prhs, 3);

  const size_t N = series.size();
  std::vector<double> row_major(N * N, 0.0);
  dtwc::core::compute_distance_matrix_pruned(series, row_major.data(), band, metric);

  mxArray *out = mxCreateDoubleMatrix(N, N, mxREAL);
  double *dst = mxGetDoubles(out);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      dst[i + j * N] = row_major[i * N + j];
  plhs[0] = out;
}

// =========================================================================
//  Legacy "cluster" command (stateless, backward-compatible)
// =========================================================================

static void cmd_cluster_legacy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3)
    throw std::invalid_argument("cluster requires data matrix and k.");

  auto series = matrix_to_series(prhs[1]);
  int k = static_cast<int>(get_scalar(prhs[2]));

  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));

  int max_iter = 100;
  if (nrhs > 5) max_iter = static_cast<int>(get_scalar(prhs[5]));

  const size_t N = series.size();
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) names[i] = std::to_string(i);

  dtwc::Problem prob("matlab_clustering");
  prob.band = band;
  prob.set_max_iter(max_iter);
  prob.set_verbose(false);

  dtwc::Data data(std::move(series), std::move(names));
  prob.set_data(std::move(data));

  auto result = dtwc::fast_pam(prob, k, max_iter);

  plhs[0] = ivec_to_mx_1based(result.labels);
  if (nlhs > 1) plhs[1] = ivec_to_mx_1based(result.medoid_indices);
  if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(result.total_cost);
}

// =========================================================================
//  Cleanup callback - called when MEX is unloaded (clear mex / exit)
// =========================================================================

static void cleanup_at_exit() {
  // Results own their Problem through a shared_ptr and are never registered in
  // HandleManager<Problem>, so the two drains are independent; Results go first
  // only so a Result's Problem is released before the Problem table is walked.
  HandleManager<dtwc::Result>::drain();
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
    // Device / Env (contract §1.1, §6)
    if (cmd == "version") {
      if (nlhs > 0) plhs[0] = mxCreateString(DTWC_VERSION_STRING);
    }
    else if (cmd == "default_random_seed") {
      if (nlhs > 0)
        plhs[0] = mxCreateDoubleScalar(
          static_cast<double>(dtwc::settings::DEFAULT_RANDOM_SEED));
    }
    else if (cmd == "set_device") cmd_set_device(nlhs, plhs, nrhs, prhs);
    else if (cmd == "get_device") cmd_get_device(nlhs, plhs, nrhs, prhs);
    // dtwc.test introspection API (Task 3.3)
    else if (cmd == "test_parallelisation") cmd_test_parallelisation(nlhs, plhs, nrhs, prhs);
    else if (cmd == "test_gpu") cmd_test_gpu(nlhs, plhs, nrhs, prhs);
    // Problem lifecycle
    else if (cmd == "Problem_new") cmd_Problem_new(nlhs, plhs, nrhs, prhs);
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
    // Problem: 2.0 config setters (method / solver / strategies / output / MIP / CUDA)
    else if (cmd == "Problem_set_method") cmd_Problem_set_method(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_solver") cmd_Problem_set_solver(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_lb_strategy") cmd_Problem_set_lb_strategy(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_storage_policy") cmd_Problem_set_storage_policy(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_output_folder") cmd_Problem_set_output_folder(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_mip_settings") cmd_Problem_set_mip_settings(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_mip_settings") cmd_Problem_get_mip_settings(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_cuda_settings") cmd_Problem_set_cuda_settings(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_cuda_settings") cmd_Problem_get_cuda_settings(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_checkpoint") cmd_Problem_set_checkpoint(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_checkpoint") cmd_Problem_get_checkpoint(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_n_clusters") cmd_Problem_n_clusters(nlhs, plhs, nrhs, prhs);
    // Problem methods
    else if (cmd == "Problem_fill_distance_matrix") cmd_Problem_fill_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_refresh_distance_matrix") cmd_Problem_refresh_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_read_distance_matrix") cmd_Problem_read_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_max_distance") cmd_Problem_max_distance(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_dist_by_ind") cmd_Problem_dist_by_ind(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_cluster") cmd_Problem_cluster(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_find_total_cost") cmd_Problem_find_total_cost(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_distance_matrix") cmd_Problem_get_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_distance_matrix") cmd_Problem_set_distance_matrix(nlhs, plhs, nrhs, prhs);
    // Checkpoint / resume (contract §2.7)
    else if (cmd == "save_checkpoint") cmd_save_checkpoint(nlhs, plhs, nrhs, prhs);
    else if (cmd == "load_checkpoint") cmd_load_checkpoint(nlhs, plhs, nrhs, prhs);
    else if (cmd == "save_binary_checkpoint") cmd_save_binary_checkpoint(nlhs, plhs, nrhs, prhs);
    else if (cmd == "load_binary_checkpoint") cmd_load_binary_checkpoint(nlhs, plhs, nrhs, prhs);
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
    // LP-relaxation bound (PDLP)
    else if (cmd == "pdlp_lp_bound") cmd_pdlp_lp_bound(nlhs, plhs, nrhs, prhs);
    else if (cmd == "pdlp_gpu_available") cmd_pdlp_gpu_available(nlhs, plhs, nrhs, prhs);
    // Tier-1 route (contract 1.3 / 1.4): dtwc::cluster owns every decision
    else if (cmd == "tier1_cluster") cmd_tier1_cluster(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Result_score") cmd_Result_score(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Result_save") cmd_Result_save(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Result_delete") cmd_Result_delete(nlhs, plhs, nrhs, prhs);
    else if (cmd == "DTWClustering_compute_distance_matrix")
      cmd_DTWClustering_compute_distance_matrix(nlhs, plhs, nrhs, prhs);
    // Legacy backward-compatible command
    else if (cmd == "cluster") cmd_cluster_legacy(nlhs, plhs, nrhs, prhs);
    // System capability check
    else if (cmd == "system_check") {
      const char *fields[] = {"openmp", "openmp_threads", "cuda", "cuda_info",
                              "metal", "metal_info", "mpi"};
      mxArray *info = mxCreateStructMatrix(1, 1, 7, fields);
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
#ifdef DTWC_HAS_METAL
      mxSetField(info, 0, "metal", mxCreateLogicalScalar(dtwc::metal::metal_available()));
      std::string mi = dtwc::metal::metal_device_info();
      mxSetField(info, 0, "metal_info", mxCreateString(mi.c_str()));
#else
      mxSetField(info, 0, "metal", mxCreateLogicalScalar(false));
      mxSetField(info, 0, "metal_info", mxCreateString("not compiled (macOS only)"));
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
  }
  // Error taxonomy (contract §5): map the dtwc leaf types FIRST, then keep the
  // std fallbacks below them. dtwc::InvalidInput/SolverError/DeviceError/IOError all
  // derive from dtwc::Error : std::runtime_error, so they MUST be caught before
  // std::runtime_error. The std::invalid_argument catch is preserved below so the
  // 19 pinned input-validation cases (require_* -> std::invalid_argument) keep
  // firing dtwc:invalidArgument unchanged.
  catch (const dtwc::InvalidInput &e) {
    error_id = "dtwc:invalidArgument"; error_msg = e.what();
  } catch (const dtwc::SolverError &e) {
    error_id = "dtwc:solverError"; error_msg = e.what();
  } catch (const dtwc::DeviceError &e) {
    error_id = "dtwc:deviceError"; error_msg = e.what();
  } catch (const dtwc::IOError &e) {
    error_id = "dtwc:ioError"; error_msg = e.what();
  } catch (const dtwc::Error &e) {
    error_id = "dtwc:error"; error_msg = e.what();
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
