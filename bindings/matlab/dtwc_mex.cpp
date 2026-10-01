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

#ifdef DTWC_MEX_MATLAB_LIBOMP
// On macOS this MEX runs on the libomp MATLAB ships, and on Windows with clang on
// its libiomp5md (bindings/matlab/CMakeLists.txt). Neither has
// __kmpc_dispatch_deinit, which Clang calls after every dynamic or guided loop:
// R2026a's libomp predates it, so dyld refused to load the MEX, and libiomp5md
// leaves it undefined at link time. LLVM's host runtime gives it an empty body in
// every release that has it (19.1.0 to 23.1.1, openmp/runtime/src/kmp_dispatch.cpp),
// so this is the same behaviour.
extern "C" void __kmpc_dispatch_deinit(void * /*loc*/, int /*gtid*/) {}
#endif

#include "../../dtwc/dtwc.hpp"
#include "../../dtwc/algorithms/fast_pam.hpp"
#include "../../dtwc/algorithms/fast_clara.hpp"
#include "../../dtwc/algorithms/hierarchical.hpp"
#include "../../dtwc/scores.hpp"
#include "../../dtwc/core/z_normalize.hpp"
#include "../../dtwc/warping_ddtw.hpp"
#include "../../dtwc/soft_dtw.hpp"
#include "../../dtwc/base/env.hpp"     // detail::parse_device: the device-name grammar
#include "../../dtwc/base/error.hpp"   // dtwc::InvalidInput/SolverError/DeviceError/IOError (§5)
#include "../../dtwc/checkpoint.hpp"   // save/load_checkpoint (contract §2.7)
#include "../../dtwc/test_api.hpp"     // dtwc::test::parallelisation()/gpu() (Task 3.3)
#include "../../dtwc/base/names.hpp"   // parse_name over the C++ name tables

#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <utility>

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

/// std::vector<index_t> -> MATLAB 1xN double row vector (1-based indexing). MATLAB's
/// own numbers are doubles, and a double holds every index exactly up to 2^53.
static mxArray *ivec_to_mx_1based(const std::vector<dtwc::index_t> &v) {
  mxArray *mx = mxCreateDoubleMatrix(1, v.size(), mxREAL);
  double *out = mxGetDoubles(mx);
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = static_cast<double>(v[i] + 1);  // 0-based -> 1-based
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

/// Convert one MATLAB double to `Int` (`int` for the parameters C++ still takes as
/// an int, `index_t` for counts and indices) without ever invoking an out-of-range
/// or non-integral float-to-int conversion (both are undefined behaviour, and a
/// NaN/Inf label silently produced a garbage cluster id before this guard).
/// The smallest `Int` is minus a power of two, exact as a double, and the largest
/// `Int` + 1 is its negation, so `value >= -lowest` is the exact upper bound.
template <class Int = int>
static Int exact_int_from_double(double value, const char *arg_name) {
  constexpr double lowest = static_cast<double>(std::numeric_limits<Int>::min());
  if (!std::isfinite(value) || std::floor(value) != value
      || value < lowest || value >= -lowest) {
    throw std::invalid_argument(
      std::string(arg_name) + " must be a finite integer in the C++ "
      + (sizeof(Int) == sizeof(int) ? "int" : "index_t") + " range.");
  }
  return static_cast<Int>(value);
}

template <class Int = int>
static Int get_exact_int(const mxArray *mx, const char *arg_name) {
  return exact_int_from_double<Int>(get_scalar(mx, arg_name), arg_name);
}

/// Shift a validated 1-based MATLAB index down to 0-based. The first index is 1:
/// a smaller value is no index (0 would address the element before the first,
/// and the lowest index_t - 1 is signed overflow), so it is an error, not a shift.
static dtwc::index_t to_0based(dtwc::index_t value, const char *arg_name) {
  if (value < 1)
    throw std::invalid_argument(
      std::string(arg_name) + " = " + std::to_string(value)
      + " is not a 1-based index (the first index is 1).");
  return value - 1;
}

static dtwc::index_t exact_int_1based_to_0based(double value, const char *arg_name) {
  return to_0based(exact_int_from_double<dtwc::index_t>(value, arg_name), arg_name);
}

/// Decode a MATLAB label vector (double, as the results return them, or int32) to
/// index_t. A label is a name, not a position: any integer is one (0 and negatives
/// included), unshifted. Every double element goes through exact_int_from_double,
/// so NaN/Inf/fractional entries are rejected instead of being cast with undefined
/// behaviour.
static std::vector<dtwc::index_t> label_vector(const mxArray *mx, const char *arg_name) {
  require_label_vector(mx, arg_name);
  const size_t n = mxGetNumberOfElements(mx);
  std::vector<dtwc::index_t> out(n);
  if (mxIsInt32(mx)) {
    const int32_t *p = static_cast<const int32_t *>(mxGetData(mx));
    for (size_t i = 0; i < n; ++i) out[i] = p[i];
  } else {
    const double *p = mxGetDoubles(mx);
    for (size_t i = 0; i < n; ++i)
      out[i] = exact_int_from_double<dtwc::index_t>(p[i], arg_name);
  }
  return out;
}

/// Decode a MATLAB double seed without invoking an out-of-range float-to-int
/// conversion. MATLAB represents every integer exactly only through flintmax, so
/// that is the largest seed a double can carry into the C++ uint64.
static std::uint64_t get_random_seed(const mxArray *mx) {
  constexpr std::uint64_t max_seed = (std::uint64_t{1} << 53) - 1;
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
  // Accept double as well (MATLAB defaults to double). A negative id wraps to a
  // value no handle has, so the lookup fails with "Invalid handle".
  return static_cast<uint64_t>(get_exact_int(mx, "handle"));
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
template <class Int = int>
static Int optional_int(int nrhs, const mxArray *prhs[], int index,
                        const char *arg_name, Int fallback) {
  if (nrhs <= index || mxIsEmpty(prhs[index])) return fallback;
  return get_exact_int<Int>(prhs[index], arg_name);
}

/// One delimiter character; 0 keeps the extension-derived default.
static char parse_delimiter(const std::string &value) {
  if (value.empty()) return static_cast<char>(0);
  if (value.size() != 1)
    throw std::invalid_argument("delimiter must be a single character.");
  return value[0];
}

/// Optional metric name; absent means L1, matching C++/Python defaults.
static dtwc::core::MetricType optional_metric(int nrhs, const mxArray *prhs[],
                                              int index) {
  const std::string token = optional_string(nrhs, prhs, index, "metric");
  if (token.empty()) return dtwc::core::MetricType::L1;
  return dtwc::parse_name(dtwc::core::metric_names, token, "metric");
}

/// The distance settings a caller names: the dtwc_cl keys, in MATLAB's case.
enum class DistanceKey { Variant, Band, Metric, MissingStrategy, WdtwG, AdtwPenalty, SdtwGamma, MsmC, TweNu, TweLambda };

constexpr dtwc::Name<DistanceKey> distance_keys[]{
  { "Variant", DistanceKey::Variant },     { "Band", DistanceKey::Band },
  { "Metric", DistanceKey::Metric },       { "MissingStrategy", DistanceKey::MissingStrategy },
  { "WdtwG", DistanceKey::WdtwG },         { "AdtwPenalty", DistanceKey::AdtwPenalty },
  { "SdtwGamma", DistanceKey::SdtwGamma }, { "MsmC", DistanceKey::MsmC },
  { "TweNu", DistanceKey::TweNu },         { "TweLambda", DistanceKey::TweLambda },
};

/// The distance configuration the name-value pairs prhs[first..nrhs) name, over the
/// C++ defaults and read with the C++ name tables; core::validate checks it where it
/// is used (distance::dtw, Problem::set_distance).
static dtwc::core::DistanceConfig distance_config(int nrhs, const mxArray *prhs[], int first) {
  if ((nrhs - first) % 2 != 0)
    throw std::invalid_argument("distance settings come in name-value pairs.");
  using namespace dtwc::core;
  DistanceConfig c;
  for (int i = first; i < nrhs; i += 2) {
    require_char(prhs[i], "a distance setting's name");
    const std::string name = get_string(prhs[i]);
    const mxArray *value = prhs[i + 1];
    const auto text = [&] {
      require_char(value, name.c_str());
      return get_string(value);
    };
    switch (dtwc::parse_name(distance_keys, name, "distance setting")) {
    case DistanceKey::Variant: c.variant.variant = dtwc::parse_name(variant_names, text(), "variant"); break;
    case DistanceKey::Band: c.band = get_exact_int(value, "Band"); break;
    case DistanceKey::Metric: c.metric = dtwc::parse_name(metric_names, text(), "metric"); break;
    case DistanceKey::MissingStrategy:
      c.missing = dtwc::parse_name(missing_strategy_names, text(), "missing strategy");
      break;
    case DistanceKey::WdtwG: c.variant.wdtw_g = get_scalar(value, "WdtwG"); break;
    case DistanceKey::AdtwPenalty: c.variant.adtw_penalty = get_scalar(value, "AdtwPenalty"); break;
    case DistanceKey::SdtwGamma: c.variant.sdtw_gamma = get_scalar(value, "SdtwGamma"); break;
    case DistanceKey::MsmC: c.variant.msm_c = get_scalar(value, "MsmC"); break;
    case DistanceKey::TweNu: c.variant.twe_nu = get_scalar(value, "TweNu"); break;
    case DistanceKey::TweLambda: c.variant.twe_lambda = get_scalar(value, "TweLambda"); break;
    }
  }
  return c;
}

/// Build a ClusteringResult MATLAB struct from a C++ ClusteringResult
static mxArray *clustering_result_to_mx(const dtwc::core::ClusteringResult &result) {
  const char *field_names[] = { "labels", "medoid_indices", "total_cost", "iterations", "converged" };
  mxArray *s = mxCreateStructMatrix(1, 1, 5, field_names);

  mxSetField(s, 0, "labels", ivec_to_mx_1based(result.labels));
  mxSetField(s, 0, "medoid_indices", ivec_to_mx_1based(result.medoid_indices));
  mxSetField(s, 0, "total_cost", mxCreateDoubleScalar(result.total_cost));

  mxSetField(s, 0, "iterations", mxCreateDoubleScalar(result.iterations));

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

  mxSetField(s, 0, "n_points", mxCreateDoubleScalar(static_cast<double>(dend.n_points)));

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

  dend.n_points = get_exact_int<dtwc::index_t>(np_mx, "dendrogram.n_points");

  size_t n_merges = mxIsEmpty(merges_mx) ? 0 : mxGetM(merges_mx);
  const double *data = mxGetDoubles(merges_mx);
  dend.merges.resize(n_merges);
  for (size_t i = 0; i < n_merges; ++i) {
    dend.merges[i].cluster_a = exact_int_1based_to_0based(
      data[i + 0 * n_merges], "dendrogram.merges(:,1)"); // 1-based -> 0-based
    dend.merges[i].cluster_b = exact_int_1based_to_0based(
      data[i + 1 * n_merges], "dendrogram.merges(:,2)");
    dend.merges[i].distance = data[i + 2 * n_merges];
    dend.merges[i].new_size = exact_int_from_double<dtwc::index_t>(
      data[i + 3 * n_merges], "dendrogram.merges(:,4)");
  }

  return dend;
}

// =========================================================================
//  Problem lifecycle commands
// =========================================================================

/// Problem::set_device from a device name, parsed by dtwc::detail::parse_device,
/// the grammar behind dtwc::device(name) and the CLI's --device.
static void set_problem_device(dtwc::Problem &prob, const mxArray *device) {
  require_char(device, "device");
  const auto [selected, index] = dtwc::detail::parse_device(get_string(device));
  prob.set_device(selected, index);
}

static void cmd_Problem_new(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  std::string name;
  if (nrhs > 1 && mxIsChar(prhs[1])) {
    name = get_string(prhs[1]);
  }
  auto prob = std::make_shared<dtwc::Problem>(name);
  prob->set_verbose(false);
  // Select the device before the handle exists: a rejected name leaks nothing.
  if (nrhs > 2) set_problem_device(*prob, prhs[2]);
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
    const int nd = get_exact_int(prhs[4], "ndim");
    if (nd < 1) throw std::invalid_argument("ndim must be a positive integer.");
    ndim = static_cast<size_t>(nd);
  }

  dtwc::Data data(std::move(series), std::move(names), ndim);
  prob.set_data(std::move(data));
}

static void cmd_Problem_set_band(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_band requires handle and band value.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_band(get_exact_int(prhs[2], "band"));
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
  prob.set_max_iter(get_exact_int(prhs[2], "max_iter"));
}

static void cmd_Problem_set_n_repetition(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_n_repetition requires handle and value.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_n_repetitions(get_exact_int(prhs[2], "n_repetitions"));
}

static void cmd_Problem_set_n_clusters(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_n_clusters requires handle and k.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_n_clusters(get_exact_int<dtwc::index_t>(prhs[2], "k"));
}

static void cmd_Problem_set_missing_strategy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_missing_strategy requires handle and string.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  prob.set_missing_strategy(dtwc::parse_name(
    dtwc::core::missing_strategy_names, get_string(prhs[2]), "missing strategy"));
}

static void cmd_Problem_set_device(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_device requires handle and device name.");
  set_problem_device(*HandleManager<dtwc::Problem>::get(get_handle(prhs[1])), prhs[2]);
}

static void cmd_Problem_set_variant(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_variant requires handle and variant string.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  using dtwc::core::DTWVariant;
  const DTWVariant variant = dtwc::parse_name(
    dtwc::core::variant_names, get_string(prhs[2]), "variant");

  dtwc::core::DTWVariantParams params = prob.variant_params();
  params.variant = variant;
  // The one optional scalar is the parameter of WDTW, ADTW and Soft-DTW; the
  // other variants (MSM and TWE keep their defaults) take none here.
  if (nrhs > 3) {
    if (variant == DTWVariant::WDTW) params.wdtw_g = get_scalar(prhs[3]);
    else if (variant == DTWVariant::ADTW) params.adtw_penalty = get_scalar(prhs[3]);
    else if (variant == DTWVariant::SoftDTW) params.sdtw_gamma = get_scalar(prhs[3]);
  }

  prob.set_variant(params);
}

static void cmd_Problem_set_distance(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_set_distance requires a handle.");
  HandleManager<dtwc::Problem>::get(get_handle(prhs[1]))->set_distance(distance_config(nrhs, prhs, 2));
}

static void cmd_Problem_get_size(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_size requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(prob.size()));
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
  // Problem::dist_by_ind is the unchecked hot path: this boundary owns the range check.
  const auto n = prob.size();
  const auto index = [&](int arg, const char *name) {
    const dtwc::index_t i = to_0based(get_exact_int<dtwc::index_t>(prhs[arg], name), name);
    if (i >= n)
      throw std::invalid_argument(std::string(name) + " = " + std::to_string(i + 1)
        + " is outside 1..N (N = " + std::to_string(n) + ").");
    return i;
  };
  const dtwc::index_t i = index(2, "i");
  const dtwc::index_t j = index(3, "j");
  prob.fill_distance_matrix(); // a no-op once filled: Problem::dist_by_ind reads the matrix
  plhs[0] = mxCreateDoubleScalar(prob.dist_by_ind(i, j));
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

  const auto &dm = std::as_const(prob).distance_matrix();
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

  auto &dm = prob.writable_distance_matrix();
  dm.resize(N);
  const double *data = mxGetDoubles(prhs[2]);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      dm.set(i, j, data[i + j * N]);  // column-major
  // A complete matrix is filled; NaN entries are computed on first use.
  if (dm.all_computed("Problem.set_distance_matrix")) prob.fill_distance_matrix();

}

// =========================================================================
//  Device commands (contract §1.1, §6 — delegate to dtwc::device)
// =========================================================================

/// set_device(name) -> canonical name ("cpu"/"gpu"/"gpu:N"), exactly as C++
/// dtwc::device(name) returns it. It throws dtwc::DeviceError (mapped to
/// dtwc:deviceError) on an unknown name, on hpc (which C++ and MATLAB do not
/// have) and on gpu without a backend — NEVER a silent fallback.
static void cmd_set_device(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("set_device requires a device-name string.");
  require_char(prhs[1], "device");   // validate BEFORE mxArrayToString deref
  plhs[0] = mxCreateString(dtwc::device(get_string(prhs[1])).c_str());
}

/// get_device() -> canonical name of the currently selected device (dtwc::device()).
static void cmd_get_device(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  plhs[0] = mxCreateString(dtwc::device().c_str());
}

// =========================================================================
//  dtwc.test introspection API (Task 3.3) — struct with the SAME field names
//  as C++ dtwc::test::* and Python dtwcpp.test.*. Both take no data arguments
//  (like get_device / gpu_info), so there is nothing to require_*-validate.
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
  prob.set_method(dtwc::parse_name(dtwc::method_names, get_string(prhs[2]), "method"));
}

static void cmd_Problem_set_solver(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_solver requires handle and solver string.");
  require_char(prhs[2], "solver");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const bool ok = prob.set_solver(
    dtwc::parse_name(dtwc::solver_names, get_string(prhs[2]), "solver"));
  plhs[0] = mxCreateLogicalScalar(ok);  // false => requested solver not compiled in
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
      "warm_start, numeric_focus, mip_focus, verbose_solver, lr_max_nodes).");

  dtwc::MIPSettings m = prob.mip_settings; // start from current, override present fields
  if (mxArray *f = mxGetField(s, 0, "mip_gap"))        m.mip_gap        = get_scalar(f, "mip_gap");
  if (mxArray *f = mxGetField(s, 0, "time_limit_sec")) m.time_limit_sec = get_exact_int(f, "time_limit_sec");
  if (mxArray *f = mxGetField(s, 0, "warm_start"))     m.warm_start     = (get_scalar(f, "warm_start") != 0.0);
  if (mxArray *f = mxGetField(s, 0, "numeric_focus"))  m.numeric_focus  = get_exact_int(f, "numeric_focus");
  if (mxArray *f = mxGetField(s, 0, "mip_focus"))      m.mip_focus      = get_exact_int(f, "mip_focus");
  if (mxArray *f = mxGetField(s, 0, "verbose_solver")) m.verbose_solver = (get_scalar(f, "verbose_solver") != 0.0);
  if (mxArray *f = mxGetField(s, 0, "lr_max_nodes"))
    m.lr_max_nodes = get_exact_int(f, "lr_max_nodes");
  prob.mip_settings = m;
}

/// get_mip_settings() -> struct mirroring MIPSettings (round-trip / introspection).
static void cmd_Problem_get_mip_settings(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("Problem_get_mip_settings requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  const auto &m = prob.mip_settings;
  const char *fields[] = { "mip_gap", "time_limit_sec", "warm_start", "numeric_focus",
                           "mip_focus", "verbose_solver", "lr_max_nodes" };
  mxArray *s = mxCreateStructMatrix(1, 1, 7, fields);
  mxSetField(s, 0, "mip_gap", mxCreateDoubleScalar(m.mip_gap));
  mxSetField(s, 0, "time_limit_sec", mxCreateDoubleScalar(m.time_limit_sec));
  mxSetField(s, 0, "warm_start", mxCreateLogicalScalar(m.warm_start));
  mxSetField(s, 0, "numeric_focus", mxCreateDoubleScalar(m.numeric_focus));
  mxSetField(s, 0, "mip_focus", mxCreateDoubleScalar(m.mip_focus));
  mxSetField(s, 0, "verbose_solver", mxCreateLogicalScalar(m.verbose_solver));
  mxSetField(s, 0, "lr_max_nodes", mxCreateDoubleScalar(static_cast<double>(m.lr_max_nodes)));
  plhs[0] = s;
}

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
      get_exact_int(f, "checkpoint.save_interval");
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

/// set_gpu_precision(name) — what a GPU computes in, by the `--gpu-precision` names.
static void cmd_Problem_set_gpu_precision(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("Problem_set_gpu_precision requires handle and precision name.");
  require_char(prhs[2], "precision");
  HandleManager<dtwc::Problem>::get(get_handle(prhs[1]))
    ->set_gpu_precision(dtwc::parse_name(dtwc::gpu_precision_names, get_string(prhs[2]), "gpu precision"));
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

// =========================================================================
//  DTW distance
// =========================================================================

/// dtw(x, y, name, value, ...): dtwc::distance::dtw, the checked boundary. The
/// configuration is checked first (core::validate), then x and y: NaN or ±inf is
/// dtwc:invalidArgument naming x or y and the 0-based position, and a missing-data
/// strategy reads NaN as missing.
static void cmd_dtw(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("dtw requires x and y.");
  require_real_double(prhs[1], "x");
  require_real_double(prhs[2], "y");
  const auto c = distance_config(nrhs, prhs, 3);
  plhs[0] = mxCreateDoubleScalar(dtwc::distance::dtw<double>(
    std::span<const double>(mxGetDoubles(prhs[1]), mxGetNumberOfElements(prhs[1])),
    std::span<const double>(mxGetDoubles(prhs[2]), mxGetNumberOfElements(prhs[2])), c.variant, c.band,
    c.metric, c.missing));
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

static void cmd_compute_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("compute_distance_matrix requires a data matrix.");
  auto series = matrix_to_series(prhs[1]);
  const dtwc::index_t N = std::ssize(series);
  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 2) band = get_exact_int(prhs[2], "band");

  // Use Problem + fill_distance_matrix() for OpenMP parallelism and LB pruning
  std::vector<std::string> names(N);
  for (dtwc::index_t i = 0; i < N; ++i) names[i] = std::to_string(i);

  dtwc::Problem prob("matlab_distmat");
  prob.set_band(band);
  prob.set_verbose(false);
  dtwc::Data data(std::move(series), std::move(names));
  prob.set_data(std::move(data));
  prob.fill_distance_matrix();

  // Copy from Problem's distance matrix to MATLAB output (column-major)
  mxArray *result = mxCreateDoubleMatrix(N, N, mxREAL);
  double *out = mxGetDoubles(result);
  for (dtwc::index_t i = 0; i < N; ++i)
    for (dtwc::index_t j = 0; j < N; ++j)
      out[i + j * N] = prob.dist_by_ind(i, j);

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
  const auto k = get_exact_int<dtwc::index_t>(prhs[2], "k");
  int max_iter = 100;
  if (nrhs > 3) max_iter = get_exact_int(prhs[3], "max_iter");

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
  opts.n_clusters = get_exact_int<dtwc::index_t>(prhs[2], "k");
  if (nrhs > 3) opts.sample_size = get_exact_int<dtwc::index_t>(prhs[3], "sample_size");
  if (nrhs > 4) opts.n_samples = get_exact_int(prhs[4], "n_samples");
  if (nrhs > 5) opts.max_iter = get_exact_int(prhs[5], "max_iter");
  if (nrhs > 6) opts.random_seed = get_random_seed(prhs[6]);

  auto result = dtwc::algorithms::fast_clara(prob, opts);
  plhs[0] = clustering_result_to_mx(result);
}

static void cmd_build_dendrogram(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("build_dendrogram requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));

  dtwc::algorithms::HierarchicalOptions opts;
  if (nrhs > 2)
    opts.linkage = dtwc::parse_name(dtwc::algorithms::linkage_names, get_string(prhs[2]), "linkage");
  if (nrhs > 3) opts.max_points = get_exact_int<dtwc::index_t>(prhs[3], "max_points");

  auto dend = dtwc::algorithms::build_dendrogram(prob, opts);
  plhs[0] = dendrogram_to_mx(dend);
}

static void cmd_cut_dendrogram(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 4) throw std::invalid_argument("cut_dendrogram requires dendrogram struct, handle, and k.");

  auto dend = mx_to_dendrogram(prhs[1]);
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[2]));
  const auto k = get_exact_int<dtwc::index_t>(prhs[3], "k");

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

static void cmd_davies_bouldin(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("davies_bouldin requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::davies_bouldin(prob));
}

static void cmd_dunn(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("dunn requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::dunn(prob));
}

static void cmd_inertia(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("inertia requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::inertia(prob));
}

static void cmd_calinski_harabasz(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) throw std::invalid_argument("calinski_harabasz requires a handle.");
  auto &prob = *HandleManager<dtwc::Problem>::get(get_handle(prhs[1]));
  plhs[0] = mxCreateDoubleScalar(dtwc::scores::calinski_harabasz(prob));
}

static void cmd_adjusted_rand(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("adjusted_rand requires two label vectors.");
  const auto labels1 = label_vector(prhs[1], "labels_1");
  const auto labels2 = label_vector(prhs[2], "labels_2");
  if (labels1.size() != labels2.size())
    throw std::invalid_argument("Label vectors must have the same length.");

  plhs[0] = mxCreateDoubleScalar(dtwc::scores::adjusted_rand(labels1, labels2));
}

static void cmd_normalized_mutual_info(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3) throw std::invalid_argument("normalized_mutual_info requires two label vectors.");
  const auto labels1 = label_vector(prhs[1], "labels_1");
  const auto labels2 = label_vector(prhs[2], "labels_2");
  if (labels1.size() != labels2.size())
    throw std::invalid_argument("Label vectors must have the same length.");

  plhs[0] = mxCreateDoubleScalar(dtwc::scores::normalized_mutual_info(labels1, labels2));
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
  const auto k = get_exact_int<dtwc::index_t>(prhs[2], "k");
  const std::string method = optional_string(nrhs, prhs, 3, "method");
  const int band = optional_int(nrhs, prhs, 4, "band", dtwc::settings::DEFAULT_BAND);
  const std::string device = optional_string(nrhs, prhs, 5, "device");
  const int max_iter = optional_int(nrhs, prhs, 6, "max_iter", 100);
  const auto skip_cols = optional_int<dtwc::index_t>(nrhs, prhs, 7, "skip_cols", 0);
  const auto skip_rows = optional_int<dtwc::index_t>(nrhs, prhs, 8, "skip_rows", 0);
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

/// Result_distance_matrix(handle) -> N x N matrix: dtwc::Result::distance_matrix
/// (row-major) unpacked into MATLAB's column-major layout.
static void cmd_Result_distance_matrix(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2)
    throw std::invalid_argument("Result_distance_matrix requires a handle.");
  const auto &res = *HandleManager<dtwc::Result>::get(get_handle(prhs[1]));
  const std::vector<double> flat = res.distance_matrix();
  const size_t n = res.labels().size();
  mxArray *out = mxCreateDoubleMatrix(n, n, mxREAL);
  double *dst = mxGetDoubles(out);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      dst[i + j * n] = flat[i * n + j];
  plhs[0] = out;
}

// =========================================================================
//  Legacy "cluster" command (stateless, backward-compatible)
// =========================================================================

static void cmd_cluster_legacy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 3)
    throw std::invalid_argument("cluster requires data matrix and k.");

  auto series = matrix_to_series(prhs[1]);
  const auto k = get_exact_int<dtwc::index_t>(prhs[2], "k");

  int band = dtwc::settings::DEFAULT_BAND;
  if (nrhs > 3) band = get_exact_int(prhs[3], "band");

  int max_iter = 100;
  if (nrhs > 5) max_iter = get_exact_int(prhs[5], "max_iter");

  const size_t N = series.size();
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) names[i] = std::to_string(i);

  dtwc::Problem prob("matlab_clustering");
  prob.set_band(band);
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
    // Device (contract §1.1, §6)
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
    else if (cmd == "gpu_available") plhs[0] = mxCreateLogicalScalar(dtwc::gpu_available());
    else if (cmd == "gpu_info") plhs[0] = mxCreateString(dtwc::gpu_info().c_str());
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
    else if (cmd == "Problem_set_verbose") cmd_Problem_set_verbose(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_max_iter") cmd_Problem_set_max_iter(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_n_repetition") cmd_Problem_set_n_repetition(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_n_clusters") cmd_Problem_set_n_clusters(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_missing_strategy") cmd_Problem_set_missing_strategy(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_device") cmd_Problem_set_device(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_variant") cmd_Problem_set_variant(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_distance") cmd_Problem_set_distance(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_size") cmd_Problem_get_size(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_name") cmd_Problem_get_name(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_centroids") cmd_Problem_get_centroids(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_clusters") cmd_Problem_get_clusters(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_is_distance_matrix_filled") cmd_Problem_is_distance_matrix_filled(nlhs, plhs, nrhs, prhs);
    // Problem: 2.0 config setters (method / solver / output / MIP / GPU precision)
    else if (cmd == "Problem_set_method") cmd_Problem_set_method(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_solver") cmd_Problem_set_solver(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_output_folder") cmd_Problem_set_output_folder(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_mip_settings") cmd_Problem_set_mip_settings(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_get_mip_settings") cmd_Problem_get_mip_settings(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Problem_set_gpu_precision") cmd_Problem_set_gpu_precision(nlhs, plhs, nrhs, prhs);
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
    // Stateless DTW functions
    else if (cmd == "dtw") cmd_dtw(nlhs, plhs, nrhs, prhs);
    else if (cmd == "soft_dtw_gradient") cmd_soft_dtw_gradient(nlhs, plhs, nrhs, prhs);
    else if (cmd == "compute_distance_matrix") cmd_compute_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "derivative_transform") cmd_derivative_transform(nlhs, plhs, nrhs, prhs);
    else if (cmd == "z_normalize") cmd_z_normalize(nlhs, plhs, nrhs, prhs);
    // Algorithms
    else if (cmd == "fast_pam") cmd_fast_pam(nlhs, plhs, nrhs, prhs);
    else if (cmd == "fast_clara") cmd_fast_clara(nlhs, plhs, nrhs, prhs);
    else if (cmd == "build_dendrogram") cmd_build_dendrogram(nlhs, plhs, nrhs, prhs);
    else if (cmd == "cut_dendrogram") cmd_cut_dendrogram(nlhs, plhs, nrhs, prhs);
    // Scoring
    else if (cmd == "silhouette") cmd_silhouette(nlhs, plhs, nrhs, prhs);
    else if (cmd == "davies_bouldin") cmd_davies_bouldin(nlhs, plhs, nrhs, prhs);
    else if (cmd == "dunn") cmd_dunn(nlhs, plhs, nrhs, prhs);
    else if (cmd == "inertia") cmd_inertia(nlhs, plhs, nrhs, prhs);
    else if (cmd == "calinski_harabasz") cmd_calinski_harabasz(nlhs, plhs, nrhs, prhs);
    else if (cmd == "adjusted_rand") cmd_adjusted_rand(nlhs, plhs, nrhs, prhs);
    else if (cmd == "normalized_mutual_info") cmd_normalized_mutual_info(nlhs, plhs, nrhs, prhs);
    // Tier-1 route (contract 1.3 / 1.4): dtwc::cluster owns every decision
    else if (cmd == "tier1_cluster") cmd_tier1_cluster(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Result_score") cmd_Result_score(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Result_save") cmd_Result_save(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Result_distance_matrix") cmd_Result_distance_matrix(nlhs, plhs, nrhs, prhs);
    else if (cmd == "Result_delete") cmd_Result_delete(nlhs, plhs, nrhs, prhs);
    // Legacy backward-compatible command
    else if (cmd == "cluster") cmd_cluster_legacy(nlhs, plhs, nrhs, prhs);
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
