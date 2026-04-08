/**
 * @file Problem.cpp
 * @brief Implementation of the DTWC (Dynamic Time Warping Clustering) problem encapsulated in a class.
 *
 * @details This file includes the implementation of the Problem class, which contains methods for clustering,
 * initializing clusters, calculating distances, and other functionalities related to the DTWC problem.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#include "Problem.hpp"
#include "mip.hpp"             // for MIP_clustering_byGurobi, MIP_clustering_byBenders
#include "parallelisation.hpp" // for run
#include "scores.hpp"          // for silhouette
#include "settings.hpp"        // for data_t, randGenerator, band, isDebug
#include "warping.hpp"         // for dtwBanded, dtwFull
#include "core/matrix_io.hpp"  // for io::write_csv, io::read_csv, operator<<

#ifdef DTWC_HAS_CUDA
#include "cuda/cuda_dtw.cuh"   // GPU distance matrix computation
#endif
#include "warping_ddtw.hpp"    // for ddtwBanded
#include "warping_wdtw.hpp"    // for wdtwBanded
#include "warping_adtw.hpp"    // for adtwBanded
#include "soft_dtw.hpp"        // for soft_dtw
#include "types/Range.hpp"     // for Range
#include "initialisation.hpp"  // For initialisation functions
#include "core/pruned_distance_matrix.hpp" // for fill_distance_matrix_pruned
#include "missing_utils.hpp"               // for has_missing, interpolate_linear
#include "warping_missing.hpp"             // for dtwMissing_banded
#include "warping_missing_arow.hpp"        // for dtwAROW_banded


#include <algorithm> // for max_element, min, min_element, sample
#include <cmath>     // for sqrt, floor
#include <iomanip>   // for operator<<, setprecision
#include <iostream>  // for cout
#include <iterator>  // for back_insert_iterator, back_inserter
#include <limits>    // for numeric_limits
#include <random>    // for mt19937, discrete_distribution, unifo...
#include <string>    // for allocator, char_traits, operator+
#include <utility>   // for pair
#include <vector>    // for vector, operator==

namespace dtwc {

/**
 * @brief Resizes data structures based on the current number of clusters.
 *
 * @details Adjusts the size of cluster_members, centroids_ind, and clusters_ind arrays based on the current
 * value of Nc (number of clusters).
 */
void Problem::resize()
{
  clusters_ind.resize(size());
  centroids_ind.resize(cluster_size());
}

/**
 * @brief Sets the number of clusters for the problem.
 *
 * @param Nc_ The number of clusters to set.
 * @throws std::runtime_error if the size of candidate_centroids is not equal to Nc.
 */
void Problem::set_numberOfClusters(int Nc_)
{
  Nc = Nc_;
  resize();
}

/**
 * @brief Sets the initial centroids for clustering.
 *
 * @param candidate_centroids A vector containing the indices of candidate centroids.
 */
void Problem::set_clusters(std::vector<int> &candidate_centroids)
{
  if (candidate_centroids.size() != static_cast<size_t>(Nc))
    throw std::runtime_error("Set cluster has failed as number of centroids is not same as the number of indices in candidate centroids vector.\n");

  centroids_ind = candidate_centroids;
}

/**
 * @brief Sets the solver to be used for clustering.
 *
 * @param solver_ The solver to use.
 * @return True if the solver is set successfully,
 * False otherwise (e.g., if Gurobi is not available and a default solver is used instead).
 */
bool Problem::set_solver(Solver solver_)
{
  if (solver_ == Solver::Gurobi) {
#ifdef DTWC_ENABLE_GUROBI
    mipSolver = Solver::Gurobi;
    return true;
#else
    std::cout << "Solver Gurobi is not available; therefore using default solver\n";
    mipSolver = settings::DEFAULT_MIP_SOLVER;
    return false;
#endif
  }

  mipSolver = solver_;
  return true;
}

/**
 * @brief Prints the current distance matrix to the standard output.
 * @details Outputs the distance matrix in a human-readable format, useful for debugging and verification.
 */
void Problem::printDistanceMatrix() const
{
  visit_distmat([](const auto &m) { std::cout << m << '\n'; });
}

/**
 * @brief Refreshes the distance matrix.
 * @details Resets state and rebinds the DTW function. Does NOT allocate the
 * dense N×N matrix — that is deferred to fillDistanceMatrix() so that
 * large-N algorithms (e.g. FastCLARA) can load data without forcing
 * quadratic memory usage.
 *
 * If the matrix was previously allocated (e.g. from a prior fillDistanceMatrix()
 * call), it is reset to size 0 so that stale entries are not reused after a
 * variant or data change.
 */
void Problem::refreshDistanceMatrix()
{
  visit_distmat([](auto &m) {
    if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
      if (m.size() != 0)
        m.resize(0); // Release old data; re-allocation deferred to fillDistanceMatrix().
    }
    // MmapDistanceMatrix: no-op (mmap is pre-allocated and persistent).
  });
  rebind_dtw_fn();
}

void Problem::refresh_variant_caches()
{
  wdtw_weights_cache_.clear();

  if (variant_params.variant != core::DTWVariant::WDTW || data.p_vec.empty())
    return;

  const auto g = static_cast<data_t>(variant_params.wdtw_g);

  // Precompute WDTW weights for every unique max_dev that can arise.
  // max_dev = max(len_x, len_y) for univariate, max(steps_x, steps_y)-1 for MV.
  // We precompute for every unique series length so the parallel DTW lambda
  // never mutates the cache. Thread-safe by design: no insertion after this point.
  if (data.ndim > 1) {
    for (const auto &series : data.p_vec) {
      const size_t steps = series.size() / data.ndim;
      if (steps == 0) continue;
      const size_t max_dev = steps - 1;
      wdtw_weights_cache_.try_emplace(max_dev, wdtw_weights<data_t>(static_cast<int>(max_dev), g));
    }
    return;
  }

  for (const auto &series : data.p_vec) {
    const size_t max_dev = series.size();
    wdtw_weights_cache_.try_emplace(max_dev, wdtw_weights<data_t>(static_cast<int>(max_dev), g));
  }
}

/**
 * @brief Rebind the DTW distance function based on current variant_params and band.
 */
void Problem::rebind_dtw_fn()
{
  // Capture `this` so that `band` and variant params are read at invocation time,
  // not at lambda creation time. This ensures that changing `prob.band = 50` after
  // construction correctly affects subsequent DTW calls without requiring a rebind.
  using namespace core;
  refresh_variant_caches();

  // Missing-data strategies override the variant dispatch
  if (missing_strategy == MissingStrategy::ZeroCost) {
    if (data.ndim > 1) {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        return dtwMissing_banded_mv(x.data(), x.size() / data.ndim,
                                    y.data(), y.size() / data.ndim,
                                    data.ndim, band);
      };
    } else {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        return dtwMissing_banded(x, y, band);
      };
    }
    return;
  }

  if (missing_strategy == MissingStrategy::Interpolate) {
    dtw_fn_ = [this](const auto &x, const auto &y) {
      auto xi = has_missing(x) ? interpolate_linear(x) : std::vector<data_t>(x.begin(), x.end());
      auto yi = has_missing(y) ? interpolate_linear(y) : std::vector<data_t>(y.begin(), y.end());
      return dtwBanded<data_t>(xi, yi, band);
    };
    return;
  }

  if (missing_strategy == MissingStrategy::AROW) {
    // TODO: AROW MV (multivariate) is deferred. The AROW recurrence is more complex
    // than zero-cost and requires a dedicated MV AROW _impl to handle per-channel
    // missing data correctly. For now, AROW always uses the scalar path regardless of
    // data.ndim. This means ndim>1 series are treated as a flat vector for AROW.
    dtw_fn_ = [this](const auto &x, const auto &y) {
      return dtwAROW_banded(x, y, band);
    };
    return;
  }

  // Standard variant dispatch (for Error strategy and unsupported missing strategies)
  switch (variant_params.variant) {
  case DTWVariant::DDTW:
    if (data.ndim > 1) {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        thread_local std::vector<data_t> dx, dy;
        derivative_transform_mv_inplace(x, data.ndim, dx);
        derivative_transform_mv_inplace(y, data.ndim, dy);
        return dtwBanded_mv(dx.data(), dx.size() / data.ndim,
                            dy.data(), dy.size() / data.ndim,
                            data.ndim, band);
      };
    } else {
      dtw_fn_ = [this](const auto &x, const auto &y) { return ddtwBanded(x, y, band); };
    }
    break;
  case DTWVariant::WDTW:
    if (data.ndim > 1) {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        const size_t x_steps = x.size() / data.ndim;
        const size_t y_steps = y.size() / data.ndim;
        if (x_steps == 0 || y_steps == 0)
          return std::numeric_limits<data_t>::max();

        const auto max_dev = std::max(x_steps, y_steps) - size_t{1};
        // Cache is populated by refresh_variant_caches() before parallel fill.
        // Read-only lookup here — no race by design.
        auto it = wdtw_weights_cache_.find(max_dev);
        if (it == wdtw_weights_cache_.end()) {
          // Cache miss (e.g. DBA centroid with novel length). Compute on stack.
          // This path is serial only (centroid update), never inside parallel fill.
          const auto g = static_cast<data_t>(variant_params.wdtw_g);
          auto w = wdtw_weights<data_t>(static_cast<int>(max_dev), g);
          return wdtwBanded_mv(x.data(), x_steps, y.data(), y_steps,
                               data.ndim, w, band);
        }
        return wdtwBanded_mv(x.data(), x_steps,
                             y.data(), y_steps,
                             data.ndim, it->second, band);
      };
    } else {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        const auto max_dev = std::max(x.size(), y.size());
        // Cache is populated by refresh_variant_caches(). Read-only in parallel.
        auto it = wdtw_weights_cache_.find(max_dev);
        if (it == wdtw_weights_cache_.end()) {
          // Serial-only fallback for novel lengths (DBA centroids).
          const auto g = static_cast<data_t>(variant_params.wdtw_g);
          auto w = wdtw_weights<data_t>(static_cast<int>(max_dev), g);
          return wdtwBanded(x, y, w, band);
        }
        return wdtwBanded(x, y, it->second, band);
      };
    }
    break;
  case DTWVariant::ADTW:
    if (data.ndim > 1) {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        return adtwBanded_mv(x.data(), x.size() / data.ndim,
                             y.data(), y.size() / data.ndim,
                             data.ndim, band, static_cast<data_t>(variant_params.adtw_penalty));
      };
    } else {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        return adtwBanded(x, y, band, static_cast<data_t>(variant_params.adtw_penalty));
      };
    }
    break;
  case DTWVariant::SoftDTW:
    // Soft-DTW can produce negative values; the distance matrix sentinel (-1.0)
    // is safe because soft_dtw results are only stored when explicitly computed.
    dtw_fn_ = [this](const auto &x, const auto &y) {
      return soft_dtw(x, y, static_cast<data_t>(variant_params.sdtw_gamma));
    };
    break;
  case DTWVariant::Standard:
  default:
    if (data.ndim > 1) {
      dtw_fn_ = [this](const auto &x, const auto &y) {
        return dtwBanded_mv(x.data(), x.size() / data.ndim,
                            y.data(), y.size() / data.ndim,
                            data.ndim, band);
      };
    } else {
      dtw_fn_ = [this](const auto &x, const auto &y) { return dtwBanded(x, y, band); };
    }
    break;
  }
}

void Problem::set_variant(core::DTWVariant v)
{
  variant_params.variant = v;
  refreshDistanceMatrix(); // calls rebind_dtw_fn() internally
}

void Problem::set_variant(core::DTWVariantParams params)
{
  variant_params = params;
  refreshDistanceMatrix(); // calls rebind_dtw_fn() internally
}

void Problem::use_mmap_distance_matrix(const std::filesystem::path &cache_path)
{
  const size_t N = data.size();
  if (std::filesystem::exists(cache_path)) {
    distMat = core::MmapDistanceMatrix::open(cache_path);
    auto &m = std::get<core::MmapDistanceMatrix>(distMat);
    if (m.size() != N)
      throw std::runtime_error("Mmap cache N=" + std::to_string(m.size())
                               + " != data N=" + std::to_string(N));
  } else {
    distMat = core::MmapDistanceMatrix(cache_path, N);
  }
}

/**
 *@brief Retrieves or calculates the distance between two points by their indices.
 *@param i Index of the first point.
 *@param j Index of the second point.
 *@return The distance between the two points.
 *
 *@note Thread safety: the lazy-alloc + compute path is NOT thread-safe.
 *      Call fillDistanceMatrix() before entering any parallel region.
 *      After that, all calls are read-only lookups (no race by design).
 */
double Problem::distByInd(int i, int j)
{
  if (i == j) return 0.0;

  const size_t N = data.size();

  // Lazily allocate the dense matrix on first individual distance request.
  // MmapDistanceMatrix is pre-allocated at creation, so only Dense needs this.
  // The critical section ensures thread safety if called from a parallel region
  // before fillDistanceMatrix(). Double-check pattern: fast path skips the lock.
  bool needs_init = visit_distmat([&](const auto &m) { return m.size() != N; });
  if (needs_init) {
#ifdef _OPENMP
    #pragma omp critical(distByInd_init)
#endif
    {
      visit_distmat([&](auto &m) {
        if (m.size() != N) {
          if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
            m.resize(N);
          }
        }
      });
    }
    rebind_dtw_fn();
  }

  bool computed = visit_distmat([&](const auto &m) { return m.is_computed(i, j); });
  if (computed)
    return visit_distmat([&](const auto &m) { return m.get(i, j); });

  const double d = dtw_fn_(p_vec(i), p_vec(j));
  visit_distmat([&](auto &m) { m.set(i, j, d); });
  return d;
}

/**
 * @brief Determines whether the pruned distance matrix strategy is applicable.
 * @details The pruned strategy requires Standard DTW variant (LB_Kim/LB_Keogh
 *          are only valid for the standard recurrence with L1 metric).
 * @return true if pruned strategy can be used.
 */
static bool pruned_strategy_applicable(const Problem &prob)
{
  // LB_Keogh is a valid lower bound for Standard DTW and ADTW: ADTW penalties
  // only increase cost, so LB_Keogh(x,y) <= DTW(x,y) <= ADTW(x,y,penalty).
  const bool supported_variant = prob.variant_params.variant == core::DTWVariant::Standard
                               || prob.variant_params.variant == core::DTWVariant::ADTW;
  return supported_variant
      && prob.band >= 0
      && prob.size() >= 64;
}

/**
 * @brief Fills the distance matrix using brute-force parallel computation.
 * @details Original implementation: parallel loop over all upper-triangle pairs
 *          using the bound dtw_fn_ (supports all DTW variants).
 */
void Problem::fillDistanceMatrix_BruteForce()
{
  const size_t N = data.size();

  // Resize (Dense only — mmap is pre-allocated at creation).
  visit_distmat([&](auto &m) {
    if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
      m.resize(N);
    }
  });

  // Set diagonal to 0
  visit_distmat([&](auto &m) {
    for (size_t i = 0; i < N; ++i)
      if (!m.is_computed(i, i))
        m.set(i, i, 0.0);
  });

  // Lock-free by design: each thread owns a disjoint set of rows.
  // The row-based partitioning ensures no two threads write the same (i,j) pair.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int ii = 0; ii < static_cast<int>(N); ++ii) {
    const size_t i = static_cast<size_t>(ii);
    const auto &series_i = p_vec(i);
    for (size_t j = i + 1; j < N; ++j) {
      bool computed = visit_distmat([&](const auto &m) { return m.is_computed(i, j); });
      if (!computed)
        visit_distmat([&](auto &m) { m.set(i, j, dtw_fn_(series_i, p_vec(j))); });
    }
  }
}

/**
 * @brief Fills the distance matrix by computing distances between all pairs of points.
 * @details Uses a strategy-based approach:
 *   - Auto: selects Pruned for Standard DTW variant, BruteForce otherwise.
 *   - BruteForce: parallel brute-force (all variants).
 *   - Pruned: parallel with LB_Kim + LB_Keogh early-abandon (Standard DTW only).
 *   - GPU: reserved for CUDA path (selected externally, e.g., via CLI).
 */
void Problem::fillDistanceMatrix()
{
  if (isDistanceMatrixFilled()) return;

  // Allocate the dense N×N matrix on first call (deferred from set_data / refreshDistanceMatrix).
  // MmapDistanceMatrix is pre-allocated at creation, so only Dense needs this.
  visit_distmat([&](auto &m) {
    if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
      if (m.size() != data.size())
        m.resize(data.size());
    }
  });

  // Re-bind the DTW function in case missing_strategy was changed after construction
  // (e.g., user sets prob.missing_strategy = ZeroCost after prob.set_data(...)).
  rebind_dtw_fn();

  if (verbose)
    std::cout << "Distance matrix is being filled!" << '\n';

  // Pre-scan for NaN if strategy is Error
  if (missing_strategy == core::MissingStrategy::Error) {
    for (size_t i = 0; i < data.size(); ++i) {
      if (has_missing(p_vec(i))) {
        throw std::runtime_error(
          "fillDistanceMatrix: NaN detected in series '" + data.p_names[i]
          + "' (index " + std::to_string(i)
          + "). Set missing_strategy to ZeroCost, AROW, or Interpolate to handle missing data.");
      }
    }
  }

  // Resolve Auto strategy
  DistanceMatrixStrategy effective = distance_strategy;
  if (effective == DistanceMatrixStrategy::Auto) {
    if (pruned_strategy_applicable(*this))
      effective = DistanceMatrixStrategy::Pruned;
    else
      effective = DistanceMatrixStrategy::BruteForce;
  }

  // Disable LB pruning if dataset has missing values (LB bounds are invalid with NaN)
  if (missing_strategy != core::MissingStrategy::Error
      && missing_strategy != core::MissingStrategy::Interpolate) {
    if (effective == DistanceMatrixStrategy::Pruned) {
      for (size_t i = 0; i < data.size(); ++i) {
        if (has_missing(p_vec(i))) {
          effective = DistanceMatrixStrategy::BruteForce;
          break;
        }
      }
    }
  }

  switch (effective) {
  case DistanceMatrixStrategy::Pruned: {
    auto stats = core::fill_distance_matrix_pruned(*this, band);
    if (verbose) {
      std::cout << "Pruned strategy: " << stats.total_pairs << " pairs, "
                << stats.early_abandoned << " early-abandoned, "
                << "pruning ratio: " << stats.pruning_ratio() << '\n';
    }
    break;
  }
  case DistanceMatrixStrategy::GPU:
#ifdef DTWC_HAS_CUDA
  {
    if (!dtwc::cuda::cuda_available()) {
      if (verbose) std::cout << "No CUDA GPU detected, falling back to CPU.\n";
      fillDistanceMatrix_BruteForce();
      break;
    }

    dtwc::cuda::CUDADistMatOptions cuda_opts;
    cuda_opts.band = band;
    cuda_opts.device_id = cuda_settings.device_id;
    if (cuda_settings.precision_mode == 1)
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP32;
    else if (cuda_settings.precision_mode == 2)
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP64;
    cuda_opts.use_squared_l2 = false;
    cuda_opts.verbose = verbose;

    auto cuda_result = dtwc::cuda::compute_distance_matrix_cuda(data.p_vec, cuda_opts);

    visit_distmat([&](auto &m) {
      if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
        m.resize(cuda_result.n);
      }
      for (size_t i = 0; i < cuda_result.n; ++i)
        for (size_t j = i; j < cuda_result.n; ++j)
          m.set(i, j, cuda_result.matrix[i * cuda_result.n + j]);
    });

    if (verbose)
      std::cout << "GPU distance matrix: " << cuda_result.pairs_computed
                << " pairs in " << std::setprecision(3)
                << cuda_result.gpu_time_sec * 1000 << " ms\n";
    break;
  }
#else
    if (verbose) std::cout << "CUDA not compiled in, falling back to CPU brute-force.\n";
    [[fallthrough]];
#endif
  case DistanceMatrixStrategy::BruteForce:
  default:
    fillDistanceMatrix_BruteForce();
    break;
  }

  if (verbose)
    std::cout << "Distance matrix has been filled!" << '\n';
}
/**
 * @brief Performs clustering based on the specified method.
 * @details Chooses between different clustering methods (K-medoids or MIP) and performs the clustering accordingly.
 */
void Problem::cluster()
{
  switch (method) {
  case Method::Kmedoids:
    cluster_by_kMedoidsLloyd();
    break;
  case Method::MIP:
    cluster_by_MIP();
    break;
  }
}

/**
 * @brief Executes the clustering process and additional post-processing tasks.
 * @details Performs clustering, then prints and writes the cluster results, including silhouettes, to files.
 */
void Problem::cluster_and_process()
{
  cluster();
  printClusters(); // Prints to screen.
  writeDistanceMatrix();
  writeClusters(); // Prints to file.
  writeSilhouettes();
}

/**
 *@brief Clusters the data using Mixed Integer Programming (MIP) based on the chosen solver.
 *@details Uses either Gurobi or HiGHS solver for MIP clustering, depending on the solver set in the Problem instance.
 */
void Problem::cluster_by_MIP()
{
  // Auto-dispatch to Benders decomposition for large N
  const bool use_benders = (mip_settings.benders == "on") ||
    (mip_settings.benders == "auto" && data.size() > 200);

  if (use_benders) {
    MIP_clustering_byBenders(*this);
    return;
  }

  switch (mipSolver) {
  case Solver::Gurobi:
    MIP_clustering_byGurobi(*this);
    break;
  case Solver::HiGHS:
    MIP_clustering_byHiGHS(*this);
    break;
  }
}


/**
 * @brief Assigns each data point to the nearest cluster centroid.
 * @details Iterates over each data point, calculating its distance to each centroid, and assigns it to the nearest one.
 */
void Problem::assignClusters()
{
  auto assignClustersTask = [this](size_t i_p) //!< i_p  and i_c in [0, Np)
  {
    const int ip = static_cast<int>(i_p);
    auto minIt = std::min_element(centroids_ind.begin(), centroids_ind.end(), [this, ip](int ic_1, int ic_2) {
      return distByInd(ip, ic_1) < distByInd(ip, ic_2);
    });
    clusters_ind[i_p] = static_cast<int>(std::distance(centroids_ind.begin(), minIt));
  };

  clusters_ind.resize(data.size()); // Resize before assigning.
  run(assignClustersTask, data.size());
}

/**
 * @brief Calculates the pairwise distances within each cluster.
 * @details Iterates through each data point, determining its cluster and calculating the distance to other points
 * within the same cluster. This method populates the distance matrix with these intra-cluster distances.
 */
void Problem::distanceInClusters()
{
  auto distanceInClustersTask = [&, N = size()](size_t i_p) {
    const int clusterNo{ clusters_ind[i_p] };
    for (size_t i{ i_p }; i < N; i++)
      if (clusters_ind[i] == clusterNo) // If they are in the same cluster
        distByInd(static_cast<int>(i_p), static_cast<int>(i));
  };

  run(distanceInClustersTask, size());
}

/**
 * @brief Calculates and updates the medoids of each cluster.
 * @details This function iterates through each data point and calculates the total cost of designating that point
 * as the medoid of its cluster. The point with the minimum total cost is set as the new medoid for that cluster.
 */
void Problem::calculateMedoids()
{
  std::vector<double> pointCosts(size());

  auto findBetterMedoidTask = [&](size_t i_p) // i_p is point index.
  {
    double sum{ 0 };
    for (const auto i : Range(size()))
      if (clusters_ind[i] == clusters_ind[i_p]) // If they are in the same cluster
        sum += distByInd(static_cast<int>(i_p), static_cast<int>(i));

    pointCosts[i_p] = sum;
  };

  run(findBetterMedoidTask, size());

  std::vector<double> clusterCosts(cluster_size(), std::numeric_limits<double>::max());
  for (const auto i : Range(size()))
    if (pointCosts[i] < clusterCosts[clusters_ind[i]]) {
      clusterCosts[clusters_ind[i]] = pointCosts[i];
      centroids_ind[clusters_ind[i]] = static_cast<int>(i);
    }
}

/**
 * @brief Performs the clustering using the Lloyd k-medoids algorithm.
 * @details Executes the Lloyd k-medoids algorithm (alternating assign + update medoids within clusters)
 * with multiple repetitions, each time initializing medoids randomly.
 * The repetition yielding the lowest total cost is chosen as the best solution.
 */
void Problem::cluster_by_kMedoidsLloyd()
{
  fillDistanceMatrix(); // Ensure all distances computed before parallel clustering.

  int best_rep = 0;
  double best_cost = std::numeric_limits<data_t>::max();

  for (int i_rand = 0; i_rand < N_repetition; i_rand++) {
    std::cout << "Metoid initialisation is started.\n";
    init();

    std::cout << "Metoid initialisation is finished. "
              << Nc << " medoids are initialised.\n"
              << "Start clustering:\n";

    auto [status, total_cost] = cluster_by_kMedoidsLloyd_single(i_rand);

    if (status == 0)
      std::cout << "Medoids are same for last two iterations, algorithm is converged!\n";
    else if (status == -1)
      std::cout << "Maximum iteration is reached before medoids are converged!\n";

    if (total_cost < best_cost) {
      best_cost = total_cost;
      best_rep = i_rand;
    }
    std::cout << "Tot cost: " << total_cost << " best cost: " << best_cost << " i rand: " << i_rand << '\n';
  }

  writeBestRep(best_rep);
}

/**
 * @brief Executes a single run of the Lloyd k-medoids clustering.
 * @details This function performs a single run of the Lloyd k-medoids algorithm, updating the medoids and clusters,
 * and calculating the total cost for this run.
 * @param rep The current repetition number.
 * @return A pair containing the status (whether the algorithm converged or not) and the total cost of clustering for this repetition.
 */
std::pair<int, double> Problem::cluster_by_kMedoidsLloyd_single(int rep)
{
  if (centroids_ind.empty()) init(); //<! Initialise if not initialised.

  auto oldmedoids = centroids_ind;

  int status = -1;
  std::vector<std::vector<int>> centroids_all;

  for (int i = 0; i < maxIter; i++) {

    std::cout << "Medoids: ";
    for (auto medoid : centroids_ind)
      std::cout << get_name(medoid) << ' ';

    centroids_all.push_back(centroids_ind);

    assignClusters();

    std::cout << " Iteration: " << i << " completed with cost: " << std::setprecision(10)
              << findTotalCost() << ".\n"; // Uses clusters_ind to find cost.

    printClusters();
    distanceInClusters(); // Just populates distance matrix ahead.
    calculateMedoids();   // Changes centroids_ind

    if (oldmedoids == centroids_ind) {
      status = 0;
      break;
    }

    oldmedoids = centroids_ind;
  }

  const double total_cost = findTotalCost();
  std::cout << "Procedure is completed with cost: " << total_cost << '\n';
  writeMedoids(centroids_all, rep, total_cost);
  return std::pair(status, total_cost);
}

/**
 * @brief Calculates the total cost of the current clustering solution.
 * @details Computes the sum of the distances between each point and its closest medoid.
 * This serves as a measure of the quality of the current clustering solution.
 * @return The total cost of the clustering.
 */
double Problem::findTotalCost()
{
  double sum = 0;
  for (const auto idx : Range(size())) {
    const int i = static_cast<int>(idx);
    if constexpr (settings::isDebug)
      std::cout << "Distance between " << i << " and closest cluster " << clusters_ind[i]
                << " which is: " << distByInd(i, centroid_of(i)) << "\n";

    sum += distByInd(i, centroid_of(i)); // #TODO should cost be square or like this?
  }

  return sum;
}

} // namespace dtwc
