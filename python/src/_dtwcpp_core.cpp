/**
 * @file _dtwcpp_core.cpp
 * @brief nanobind Python bindings for DTWC++.
 *
 * @details Exposes DTW distance functions, clustering algorithms,
 *          distance matrix, and scoring to Python with zero-copy
 *          numpy integration where possible.
 *
 * @date 28 Mar 2026
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>

#include <dtwc.hpp>
#include <checkpoint.hpp>
#include <warping.hpp>
#include <warping_ddtw.hpp>
#include <warping_wdtw.hpp>
#include <warping_adtw.hpp>
#include <warping_missing.hpp>
#include <soft_dtw.hpp>
#include <algorithms/fast_pam.hpp>
#include <algorithms/fast_clara.hpp>
#include <scores.hpp>
#include <core/z_normalize.hpp>
#include <core/dtw_options.hpp>
#include <core/pruned_distance_matrix.hpp>

#include <vector>
#include <string>

namespace nb = nanobind;
using namespace nb::literals; // for _a arg names

NB_MODULE(_dtwcpp_core, m) {
  m.doc() = "DTWC++ — Fast Dynamic Time Warping and Clustering (C++ core)";

  // =========================================================================
  // Enums
  // =========================================================================

  nb::enum_<dtwc::Method>(m, "Method")
    .value("Kmedoids", dtwc::Method::Kmedoids)
    .value("MIP", dtwc::Method::MIP);

  nb::enum_<dtwc::Solver>(m, "Solver")
    .value("Gurobi", dtwc::Solver::Gurobi)
    .value("HiGHS", dtwc::Solver::HiGHS);

  nb::enum_<dtwc::core::ConstraintType>(m, "ConstraintType")
    .value("NONE", dtwc::core::ConstraintType::None)
    .value("SakoeChibaBand", dtwc::core::ConstraintType::SakoeChibaBand);

  nb::enum_<dtwc::core::MetricType>(m, "MetricType")
    .value("L1", dtwc::core::MetricType::L1)
    .value("L2", dtwc::core::MetricType::L2)
    .value("SquaredL2", dtwc::core::MetricType::SquaredL2);

  nb::enum_<dtwc::core::DTWVariant>(m, "DTWVariant")
    .value("Standard", dtwc::core::DTWVariant::Standard)
    .value("DDTW", dtwc::core::DTWVariant::DDTW)
    .value("WDTW", dtwc::core::DTWVariant::WDTW)
    .value("ADTW", dtwc::core::DTWVariant::ADTW)
    .value("SoftDTW", dtwc::core::DTWVariant::SoftDTW);

  // =========================================================================
  // DTWVariantParams
  // =========================================================================

  nb::class_<dtwc::core::DTWVariantParams>(m, "DTWVariantParams")
    .def(nb::init<>())
    .def_rw("variant", &dtwc::core::DTWVariantParams::variant)
    .def_rw("wdtw_g", &dtwc::core::DTWVariantParams::wdtw_g)
    .def_rw("adtw_penalty", &dtwc::core::DTWVariantParams::adtw_penalty)
    .def_rw("sdtw_gamma", &dtwc::core::DTWVariantParams::sdtw_gamma);

  // =========================================================================
  // ClusteringResult
  // =========================================================================

  nb::class_<dtwc::core::ClusteringResult>(m, "ClusteringResult")
    .def(nb::init<>())
    .def_rw("labels", &dtwc::core::ClusteringResult::labels)
    .def_rw("medoid_indices", &dtwc::core::ClusteringResult::medoid_indices)
    .def_rw("total_cost", &dtwc::core::ClusteringResult::total_cost)
    .def_rw("iterations", &dtwc::core::ClusteringResult::iterations)
    .def_rw("converged", &dtwc::core::ClusteringResult::converged)
    .def_prop_ro("n_clusters", &dtwc::core::ClusteringResult::n_clusters)
    .def_prop_ro("n_points", &dtwc::core::ClusteringResult::n_points)
    .def("__repr__", [](const dtwc::core::ClusteringResult &r) {
      return "ClusteringResult(k=" + std::to_string(r.n_clusters())
             + ", cost=" + std::to_string(r.total_cost)
             + ", iters=" + std::to_string(r.iterations)
             + ", converged=" + (r.converged ? "True" : "False") + ")";
    });

  // =========================================================================
  // DenseDistanceMatrix
  // =========================================================================

  nb::class_<dtwc::core::DenseDistanceMatrix>(m, "DenseDistanceMatrix")
    .def(nb::init<>())
    .def(nb::init<size_t>(), "n"_a)
    .def("resize", &dtwc::core::DenseDistanceMatrix::resize, "n"_a)
    .def("get", &dtwc::core::DenseDistanceMatrix::get, "i"_a, "j"_a)
    .def("set", &dtwc::core::DenseDistanceMatrix::set, "i"_a, "j"_a, "value"_a)
    .def("is_computed", &dtwc::core::DenseDistanceMatrix::is_computed, "i"_a, "j"_a)
    .def_prop_ro("size", &dtwc::core::DenseDistanceMatrix::size)
    .def("max", &dtwc::core::DenseDistanceMatrix::max)
    .def("to_numpy", [](dtwc::core::DenseDistanceMatrix &dm) {
      size_t n = dm.size();
      size_t shape[2] = {n, n};
      return nb::ndarray<nb::numpy, double>(dm.raw(), 2, shape, nb::handle());
    }, nb::rv_policy::reference_internal,
       "Return a numpy view of the distance matrix (zero-copy).")
    .def("write_csv", &dtwc::core::DenseDistanceMatrix::write_csv, "path"_a)
    .def("read_csv", &dtwc::core::DenseDistanceMatrix::read_csv, "path"_a)
    .def("__repr__", [](const dtwc::core::DenseDistanceMatrix &dm) {
      return "DenseDistanceMatrix(n=" + std::to_string(dm.size()) + ")";
    });

  // =========================================================================
  // DTW distance functions
  // =========================================================================

  m.def("dtw_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                            nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                            int band, const std::string &metric) {
    nb::gil_scoped_release release;
    auto mt = dtwc::core::MetricType::L1;
    if (metric == "squared_euclidean" || metric == "sqeuclidean")
        mt = dtwc::core::MetricType::SquaredL2;
    return dtwc::dtwBanded<double>(x.data(), x.size(), y.data(), y.size(), band, -1.0, mt);
  }, "x"_a, "y"_a, "band"_a = -1, "metric"_a = "l1",
     "Compute DTW distance (zero-copy from numpy).\n\n"
     "metric: 'l1' (default) or 'squared_euclidean'.\n"
     "band=-1 for full DTW, band>0 for Sakoe-Chiba banded DTW.");

  m.def("ddtw_distance", [](const std::vector<double> &x,
                              const std::vector<double> &y,
                              int band) {
    nb::gil_scoped_release release;
    return dtwc::ddtwBanded<double>(x, y, band);
  }, "x"_a, "y"_a, "band"_a = -1,
     "Compute Derivative DTW distance.");

  m.def("wdtw_distance", [](const std::vector<double> &x,
                              const std::vector<double> &y,
                              int band, double g) {
    nb::gil_scoped_release release;
    return dtwc::wdtwBanded<double>(x, y, band, g);
  }, "x"_a, "y"_a, "band"_a = -1, "g"_a = 0.05,
     "Compute Weighted DTW distance with logistic weight steepness g.");

  m.def("adtw_distance", [](const std::vector<double> &x,
                              const std::vector<double> &y,
                              int band, double penalty) {
    nb::gil_scoped_release release;
    return dtwc::adtwBanded<double>(x, y, band, penalty);
  }, "x"_a, "y"_a, "band"_a = -1, "penalty"_a = 1.0,
     "Compute Amerced DTW distance with non-diagonal step penalty.");

  m.def("soft_dtw_distance", [](const std::vector<double> &x,
                                 const std::vector<double> &y,
                                 double gamma) {
    nb::gil_scoped_release release;
    return dtwc::soft_dtw<double>(x, y, gamma);
  }, "x"_a, "y"_a, "gamma"_a = 1.0,
     "Compute Soft-DTW distance (differentiable).");

  m.def("soft_dtw_gradient", [](const std::vector<double> &x,
                                 const std::vector<double> &y,
                                 double gamma) {
    nb::gil_scoped_release release;
    return dtwc::soft_dtw_gradient<double>(x, y, gamma);
  }, "x"_a, "y"_a, "gamma"_a = 1.0,
     "Compute Soft-DTW gradient w.r.t. first series x.");

  m.def("dtw_distance_missing", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                                    nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                                    int band, const std::string &metric) {
    nb::gil_scoped_release release;
    auto mt = dtwc::core::MetricType::L1;
    if (metric == "squared_euclidean" || metric == "sqeuclidean")
        mt = dtwc::core::MetricType::SquaredL2;
    return dtwc::dtwMissing_banded<double>(x.data(), x.size(), y.data(), y.size(), band, -1.0, mt);
  }, "x"_a, "y"_a, "band"_a = -1, "metric"_a = "l1",
     "DTW distance with missing data support (NaN = missing).\n\n"
     "NaN values in either series are treated as missing; pairs where\n"
     "one or both values are NaN contribute zero cost.\n"
     "metric: 'l1' (default) or 'squared_euclidean'.\n"
     "band=-1 for full DTW, band>0 for Sakoe-Chiba banded DTW.");

  // =========================================================================
  // Utility functions
  // =========================================================================

  m.def("derivative_transform", [](const std::vector<double> &x) {
    return dtwc::derivative_transform<double>(x);
  }, "x"_a, "Compute derivative transform for DDTW.");

  m.def("z_normalize", [](std::vector<double> x) {
    dtwc::core::z_normalize(x.data(), x.size());
    return x;
  }, "x"_a, "Return z-normalized copy (zero mean, unit stddev).");

  // =========================================================================
  // Data
  // =========================================================================

  nb::class_<dtwc::Data>(m, "Data")
    .def(nb::init<>())
    .def(nb::init<std::vector<std::vector<dtwc::data_t>> &&,
                   std::vector<std::string> &&>(),
         "series"_a, "names"_a)
    .def_rw("p_vec", &dtwc::Data::p_vec)
    .def_rw("p_names", &dtwc::Data::p_names)
    .def_prop_ro("size", &dtwc::Data::size);

  // =========================================================================
  // Problem class
  // =========================================================================

  nb::class_<dtwc::Problem>(m, "Problem")
    .def(nb::init<>())
    .def("__init__", [](dtwc::Problem *p, const std::string &name) {
      new (p) dtwc::Problem(name);
    }, "name"_a)
    // Properties for public fields
    .def_rw("method", &dtwc::Problem::method)
    .def_rw("max_iter", &dtwc::Problem::maxIter)
    .def_rw("n_repetition", &dtwc::Problem::N_repetition)
    .def_rw("band", &dtwc::Problem::band)
    .def_rw("variant_params", &dtwc::Problem::variant_params)
    .def_rw("name", &dtwc::Problem::name)
    .def_rw("clusters_ind", &dtwc::Problem::clusters_ind)
    .def_rw("centroids_ind", &dtwc::Problem::centroids_ind)
    // Getters
    .def_prop_ro("size", &dtwc::Problem::size)
    .def_prop_ro("cluster_size", &dtwc::Problem::cluster_size)
    .def("is_distance_matrix_filled", &dtwc::Problem::isDistanceMatrixFilled)
    .def("max_distance", &dtwc::Problem::maxDistance)
    .def("dist_by_ind", &dtwc::Problem::distByInd, "i"_a, "j"_a)
    // Setters
    .def("set_number_of_clusters", &dtwc::Problem::set_numberOfClusters, "n_clusters"_a)
    .def("set_variant", nb::overload_cast<dtwc::core::DTWVariant>(&dtwc::Problem::set_variant), "variant"_a)
    .def("set_data", [](dtwc::Problem &p, std::vector<std::vector<double>> series,
                         std::vector<std::string> names) {
      dtwc::Data d(std::move(series), std::move(names));
      p.set_data(std::move(d));
    }, "series"_a, "names"_a, "Set time series data.")
    // Distance matrix
    .def("fill_distance_matrix", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.fillDistanceMatrix();
    }, "Compute all pairwise DTW distances.")
    .def("distance_matrix_numpy", [](dtwc::Problem &prob) {
      prob.fillDistanceMatrix();
      const size_t n = prob.size();
      double* ptr = new double[n * n];
      for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
          ptr[i * n + j] = prob.distByInd(static_cast<int>(i), static_cast<int>(j));

      nb::capsule owner(ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
      return nb::ndarray<nb::numpy, double>(ptr, {n, n}, owner);
    }, "Fill distance matrix in C++ (OpenMP parallel) and return as numpy array.")
    .def("refresh_distance_matrix", &dtwc::Problem::refreshDistanceMatrix)
    // Clustering
    .def("cluster", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.cluster();
    }, "Run clustering (Lloyd k-medoids or MIP).")
    .def("find_total_cost", &dtwc::Problem::findTotalCost)
    .def("assign_clusters", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.assignClusters();
    })
    .def("calculate_medoids", [](dtwc::Problem &p) {
      nb::gil_scoped_release release;
      p.calculateMedoids();
    })
    // I/O
    .def("print_clusters", &dtwc::Problem::printClusters)
    .def("write_clusters", &dtwc::Problem::writeClusters)
    .def("write_distance_matrix", nb::overload_cast<>(&dtwc::Problem::writeDistanceMatrix, nb::const_))
    .def("write_silhouettes", &dtwc::Problem::writeSilhouettes)
    .def("set_distance_matrix_from_numpy",
         [](dtwc::Problem &p, nb::ndarray<const double, nb::ndim<2>, nb::c_contig> dm) {
           const size_t n = dm.shape(0);
           if (dm.shape(1) != n)
               throw std::runtime_error("Expected square matrix");
           if (n != p.size())
               throw std::runtime_error("Matrix size doesn't match Problem data size");
           auto &mat = p.distance_matrix();
           mat.resize(n);
           const double* data = dm.data();
           for (size_t i = 0; i < n; ++i)
               for (size_t j = i; j < n; ++j)
                   mat.set(i, j, data[i * n + j]);
           p.set_distance_matrix_filled(true);
         }, "dm"_a,
         "Load a precomputed NxN distance matrix (e.g. from GPU computation).")
    .def("__repr__", [](const dtwc::Problem &p) {
      return "Problem(name='" + p.name + "', n=" + std::to_string(p.size())
             + ", k=" + std::to_string(p.cluster_size()) + ")";
    });

  // =========================================================================
  // Distance matrix convenience function
  // =========================================================================

  m.def("compute_distance_matrix", [](const std::vector<std::vector<double>> &series,
                                        int band, const std::string &metric,
                                        bool use_pruning) {
    auto mt = dtwc::core::MetricType::L1;
    if (metric == "squared_euclidean" || metric == "sqeuclidean")
        mt = dtwc::core::MetricType::SquaredL2;

    const size_t n = series.size();
    double* ptr = new double[n * n]();  // zero-init

    // Release GIL only for the compute-heavy section
    {
      nb::gil_scoped_release release;

      if (use_pruning && (mt == dtwc::core::MetricType::L1 || mt == dtwc::core::MetricType::L2)) {
        // LB-pruned version: precomputes envelopes + summaries,
        // uses early-abandon DTW guided by LB_Kim / LB_Keogh thresholds.
        dtwc::core::compute_distance_matrix_pruned(series, ptr, band, mt);
      } else {
        // Standard unpruned version (for non-L1 metrics or when pruning disabled).
        // Lock-free by design: each thread owns a disjoint set of rows (outer loop i).
        // Writes to ptr[i*n+j] and ptr[j*n+i] never collide across threads because
        // no two threads share the same i value.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 16)
        #endif
        for (int i = 0; i < static_cast<int>(n); ++i) {
            for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
                double d = (band >= 0)
                    ? dtwc::dtwBanded<double>(series[i], series[j], band, -1.0, mt)
                    : dtwc::dtwFull_L<double>(series[i], series[j], -1.0, mt);
                ptr[i * n + j] = d;
                ptr[j * n + i] = d;
            }
        }
      }
    }  // GIL re-acquired here

    nb::capsule owner(ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double>(ptr, {n, n}, owner);
  }, "series"_a, "band"_a = -1, "metric"_a = "l1", "use_pruning"_a = true,
     "Compute pairwise DTW distance matrix entirely in C++.\n\n"
     "Returns NxN numpy array. Uses OpenMP parallelism when available.\n"
     "When use_pruning=True (default), uses LB_Kim and LB_Keogh lower\n"
     "bounds with early-abandon DTW for faster computation (L1 metric only).\n"
     "Much faster than calling dtw_distance in a Python loop.");

  // =========================================================================
  // FastPAM
  // =========================================================================

  m.def("fast_pam", [](dtwc::Problem &prob, int n_clusters, int max_iter) {
    dtwc::core::ClusteringResult result;
    {
      nb::gil_scoped_release release;
      result = dtwc::fast_pam(prob, n_clusters, max_iter);
    }
    // Store results back into Problem so silhouette(prob) and DBI(prob) work
    prob.set_numberOfClusters(n_clusters);
    prob.centroids_ind = result.medoid_indices;
    prob.clusters_ind = result.labels;
    return result;
  }, "prob"_a, "n_clusters"_a, "max_iter"_a = 100,
     "Run FastPAM k-medoids clustering (Schubert & Rousseeuw 2021).\n\n"
     "Results are also stored back into prob, so silhouette(prob) and\n"
     "davies_bouldin_index(prob) work after this call.");

  // =========================================================================
  // FastCLARA
  // =========================================================================

  nb::class_<dtwc::algorithms::CLARAOptions>(m, "CLARAOptions")
    .def(nb::init<>())
    .def_rw("n_clusters", &dtwc::algorithms::CLARAOptions::n_clusters)
    .def_rw("sample_size", &dtwc::algorithms::CLARAOptions::sample_size)
    .def_rw("n_samples", &dtwc::algorithms::CLARAOptions::n_samples)
    .def_rw("max_iter", &dtwc::algorithms::CLARAOptions::max_iter)
    .def_rw("random_seed", &dtwc::algorithms::CLARAOptions::random_seed)
    .def("__repr__", [](const dtwc::algorithms::CLARAOptions &o) {
      return "CLARAOptions(k=" + std::to_string(o.n_clusters)
             + ", sample_size=" + std::to_string(o.sample_size)
             + ", n_samples=" + std::to_string(o.n_samples)
             + ", max_iter=" + std::to_string(o.max_iter)
             + ", seed=" + std::to_string(o.random_seed) + ")";
    });

  m.def("fast_clara", [](dtwc::Problem &prob, int n_clusters, int sample_size,
                           int n_samples, int max_iter, unsigned seed) {
    dtwc::algorithms::CLARAOptions opts;
    opts.n_clusters = n_clusters;
    opts.sample_size = sample_size;
    opts.n_samples = n_samples;
    opts.max_iter = max_iter;
    opts.random_seed = seed;
    nb::gil_scoped_release release;
    return dtwc::algorithms::fast_clara(prob, opts);
  }, "prob"_a, "n_clusters"_a, "sample_size"_a = -1,
     "n_samples"_a = 5, "max_iter"_a = 100, "seed"_a = 42,
     "Run FastCLARA scalable k-medoids clustering.\n\n"
     "Runs FastPAM on random subsamples and assigns all points to the\n"
     "best medoids found. Avoids O(N^2) memory of full PAM.\n\n"
     "Parameters:\n"
     "  prob: Problem with data loaded.\n"
     "  n_clusters: Number of clusters (k).\n"
     "  sample_size: Subsample size (-1 = auto: 40 + 2*k).\n"
     "  n_samples: Number of subsamples to try (default 5).\n"
     "  max_iter: Max PAM iterations per subsample (default 100).\n"
     "  seed: Random seed for reproducibility (default 42).");

  // =========================================================================
  // Checkpointing
  // =========================================================================

  nb::class_<dtwc::CheckpointOptions>(m, "CheckpointOptions")
    .def(nb::init<>())
    .def_rw("directory", &dtwc::CheckpointOptions::directory)
    .def_rw("save_interval", &dtwc::CheckpointOptions::save_interval)
    .def_rw("enabled", &dtwc::CheckpointOptions::enabled)
    .def("__repr__", [](const dtwc::CheckpointOptions &o) {
      return "CheckpointOptions(dir='" + o.directory
             + "', interval=" + std::to_string(o.save_interval)
             + ", enabled=" + (o.enabled ? "True" : "False") + ")";
    });

  m.def("save_checkpoint", &dtwc::save_checkpoint, "prob"_a, "path"_a,
        "Save distance matrix checkpoint to directory.\n\n"
        "Creates distances.csv and metadata.txt in the given directory.\n"
        "The directory is created if it does not exist.");

  m.def("load_checkpoint", &dtwc::load_checkpoint, "prob"_a, "path"_a,
        "Load distance matrix checkpoint from directory.\n\n"
        "Returns True if checkpoint was loaded successfully, False otherwise.\n"
        "Validates that matrix dimensions match the Problem's data size.\n"
        "Sets distance matrix filled flag if all pairs are computed.");

  // =========================================================================
  // Scores
  // =========================================================================

  m.def("silhouette", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::silhouette(prob);
  }, "prob"_a, "Compute silhouette scores for each data point.");

  m.def("davies_bouldin_index", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::daviesBouldinIndex(prob);
  }, "prob"_a, "Compute Davies-Bouldin Index.");

  // =========================================================================
  // CUDA (optional)
  // =========================================================================

#ifdef DTWC_HAS_CUDA
  m.def("cuda_available", &dtwc::cuda::cuda_available,
        "Check if a CUDA-capable GPU is available.");

  m.def("cuda_device_info", &dtwc::cuda::cuda_device_info,
        "device_id"_a = 0,
        "Get a human-readable string describing the CUDA device.");

  m.def("compute_distance_matrix_cuda",
        [](const std::vector<std::vector<double>> &series,
           int band, bool use_squared_l2, int device_id, bool verbose) {
          dtwc::cuda::CUDADistMatOptions opts;
          opts.band = band;
          opts.use_squared_l2 = use_squared_l2;
          opts.device_id = device_id;
          opts.verbose = verbose;
          nb::gil_scoped_release release;
          auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
          size_t n = result.n;
          double* data = new double[n * n];
          std::copy(result.matrix.begin(), result.matrix.end(), data);
          nb::gil_scoped_acquire acquire;
          nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
          return nb::ndarray<nb::numpy, double>(data, {n, n}, owner);
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "device_id"_a = 0, "verbose"_a = false,
        "Compute NxN DTW distance matrix on GPU.\n\n"
        "Returns NxN numpy array of DTW distances.");

  m.def("compute_lb_keogh_cuda",
        [](const std::vector<std::vector<double>> &series,
           int band, int device_id) {
          nb::gil_scoped_release release;
          auto result = dtwc::cuda::compute_lb_keogh_cuda(series, band, device_id);
          size_t np = result.lb_values.size();
          double* data = new double[np];
          std::copy(result.lb_values.begin(), result.lb_values.end(), data);
          nb::gil_scoped_acquire acquire;
          nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
          return nb::ndarray<nb::numpy, double>(data, {np}, owner);
        },
        "series"_a, "band"_a, "device_id"_a = 0,
        "Compute LB_Keogh lower bounds for all N*(N-1)/2 pairs on GPU.\n\n"
        "Returns flat array of symmetric LB_Keogh values (upper triangle).\n"
        "Requires band >= 0 (Sakoe-Chiba constraint).");

  m.attr("CUDA_AVAILABLE") = true;
#else
  m.def("cuda_available", []() { return false; },
        "Check if CUDA GPU is available.");

  m.def("cuda_device_info", [](int) { return std::string("CUDA not available (not compiled)"); },
        "device_id"_a = 0,
        "Get CUDA device info string.");

  m.def("compute_distance_matrix_cuda",
        [](const std::vector<std::vector<double>> &, int, bool, int, bool) -> nb::object {
          throw std::runtime_error("CUDA support not compiled. Rebuild with -DDTWC_ENABLE_CUDA=ON");
        },
        "series"_a, "band"_a = -1, "use_squared_l2"_a = false,
        "device_id"_a = 0, "verbose"_a = false,
        "Compute NxN DTW distance matrix on GPU (requires CUDA build).");

  m.def("compute_lb_keogh_cuda",
        [](const std::vector<std::vector<double>> &, int, int) -> nb::object {
          throw std::runtime_error("CUDA support not compiled. Rebuild with -DDTWC_ENABLE_CUDA=ON");
        },
        "series"_a, "band"_a, "device_id"_a = 0,
        "Compute LB_Keogh lower bounds on GPU (requires CUDA build).");

  m.attr("CUDA_AVAILABLE") = false;
#endif
}
