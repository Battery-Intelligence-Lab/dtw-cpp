/**
 * @file _dtwcpp_core.cpp
 * @brief nanobind Python bindings for DTWC++.
 *
 * @details Exposes DTW distance functions, clustering algorithms,
 *          distance matrix, and scoring to Python with zero-copy
 *          numpy integration where possible.
 *
 * @author Volkan Kumtepeli
 * 
 * @date 28 Mar 2026
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <dtwc.hpp>
#include <checkpoint.hpp>
#include <warping.hpp>
#include <warping_ddtw.hpp>
#include <warping_wdtw.hpp>
#include <warping_adtw.hpp>
#include <warping_missing.hpp>
#include <warping_missing_arow.hpp>
#include <soft_dtw.hpp>
#include <algorithms/fast_pam.hpp>
#include <algorithms/fast_clara.hpp>
#include <algorithms/clarans.hpp>
#include <algorithms/hierarchical.hpp>
#include <scores.hpp>
#include <core/z_normalize.hpp>
#include <core/dtw_options.hpp>
#include <core/pruned_distance_matrix.hpp>
#include <core/matrix_io.hpp>

#include <Eigen/Core>

#include <cstring>
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

  nb::enum_<dtwc::core::MissingStrategy>(m, "MissingStrategy")
    .value("Error", dtwc::core::MissingStrategy::Error)
    .value("ZeroCost", dtwc::core::MissingStrategy::ZeroCost)
    .value("AROW", dtwc::core::MissingStrategy::AROW)
    .value("Interpolate", dtwc::core::MissingStrategy::Interpolate);

  nb::enum_<dtwc::DistanceMatrixStrategy>(m, "DistanceMatrixStrategy")
    .value("Auto", dtwc::DistanceMatrixStrategy::Auto)
    .value("BruteForce", dtwc::DistanceMatrixStrategy::BruteForce)
    .value("Pruned", dtwc::DistanceMatrixStrategy::Pruned)
    .value("GPU", dtwc::DistanceMatrixStrategy::GPU);

  // =========================================================================
  // Linkage (hierarchical clustering)
  // =========================================================================

  nb::enum_<dtwc::algorithms::Linkage>(m, "Linkage")
    .value("Single", dtwc::algorithms::Linkage::Single)
    .value("Complete", dtwc::algorithms::Linkage::Complete)
    .value("Average", dtwc::algorithms::Linkage::Average);

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
  // MIPSettings
  // =========================================================================

  nb::class_<dtwc::MIPSettings>(m, "MIPSettings")
    .def(nb::init<>())
    .def_rw("mip_gap", &dtwc::MIPSettings::mip_gap,
            "Relative MIP gap tolerance (default 1e-5).")
    .def_rw("time_limit_sec", &dtwc::MIPSettings::time_limit_sec,
            "Solver time limit in seconds (-1 = unlimited).")
    .def_rw("warm_start", &dtwc::MIPSettings::warm_start,
            "Run FastPAM first and feed as MIP start (default True).")
    .def_rw("numeric_focus", &dtwc::MIPSettings::numeric_focus,
            "Gurobi NumericFocus (0-3, default 1).")
    .def_rw("mip_focus", &dtwc::MIPSettings::mip_focus,
            "Gurobi MIPFocus (0=balanced, 1=feasible, 2=optimal, 3=bound).")
    .def_rw("verbose_solver", &dtwc::MIPSettings::verbose_solver,
            "Show solver log output (default False).")
    .def("__repr__", [](const dtwc::MIPSettings &s) {
      return "MIPSettings(gap=" + std::to_string(s.mip_gap)
             + ", time_limit=" + std::to_string(s.time_limit_sec)
             + ", warm_start=" + (s.warm_start ? "True" : "False")
             + ", numeric_focus=" + std::to_string(s.numeric_focus)
             + ", mip_focus=" + std::to_string(s.mip_focus)
             + ", verbose=" + (s.verbose_solver ? "True" : "False") + ")";
    });

  // =========================================================================
  // DendrogramStep
  // =========================================================================

  nb::class_<dtwc::algorithms::DendrogramStep>(m, "DendrogramStep")
    .def(nb::init<>())
    .def_rw("cluster_a", &dtwc::algorithms::DendrogramStep::cluster_a,
            "First merged cluster index.")
    .def_rw("cluster_b", &dtwc::algorithms::DendrogramStep::cluster_b,
            "Second merged cluster index.")
    .def_rw("distance", &dtwc::algorithms::DendrogramStep::distance,
            "Merge distance.")
    .def_rw("new_size", &dtwc::algorithms::DendrogramStep::new_size,
            "Size of the merged cluster.")
    .def("__repr__", [](const dtwc::algorithms::DendrogramStep &s) {
      return "DendrogramStep(a=" + std::to_string(s.cluster_a)
             + ", b=" + std::to_string(s.cluster_b)
             + ", dist=" + std::to_string(s.distance)
             + ", size=" + std::to_string(s.new_size) + ")";
    });

  // =========================================================================
  // Dendrogram
  // =========================================================================

  nb::class_<dtwc::algorithms::Dendrogram>(m, "Dendrogram")
    .def(nb::init<>())
    .def_rw("merges", &dtwc::algorithms::Dendrogram::merges,
            "List of N-1 DendrogramStep merge records.")
    .def_rw("n_points", &dtwc::algorithms::Dendrogram::n_points,
            "Number of original data points.")
    .def("__repr__", [](const dtwc::algorithms::Dendrogram &d) {
      return "Dendrogram(n_points=" + std::to_string(d.n_points)
             + ", merges=" + std::to_string(d.merges.size()) + ")";
    });

  // =========================================================================
  // HierarchicalOptions
  // =========================================================================

  nb::class_<dtwc::algorithms::HierarchicalOptions>(m, "HierarchicalOptions")
    .def(nb::init<>())
    .def_rw("linkage", &dtwc::algorithms::HierarchicalOptions::linkage,
            "Linkage criterion (Single, Complete, or Average).")
    .def_rw("max_points", &dtwc::algorithms::HierarchicalOptions::max_points,
            "Hard guard: throws if N exceeds this (default 2000).")
    .def("__repr__", [](const dtwc::algorithms::HierarchicalOptions &o) {
      std::string linkage_str;
      switch (o.linkage) {
        case dtwc::algorithms::Linkage::Single: linkage_str = "Single"; break;
        case dtwc::algorithms::Linkage::Complete: linkage_str = "Complete"; break;
        case dtwc::algorithms::Linkage::Average: linkage_str = "Average"; break;
      }
      return "HierarchicalOptions(linkage=" + linkage_str
             + ", max_points=" + std::to_string(o.max_points) + ")";
    });

  // =========================================================================
  // CLARANSOptions
  // =========================================================================

  nb::class_<dtwc::algorithms::CLARANSOptions>(m, "CLARANSOptions")
    .def(nb::init<>())
    .def_rw("n_clusters", &dtwc::algorithms::CLARANSOptions::n_clusters,
            "Number of clusters (k).")
    .def_rw("num_local", &dtwc::algorithms::CLARANSOptions::num_local,
            "Number of random restarts (default 2).")
    .def_rw("max_neighbor", &dtwc::algorithms::CLARANSOptions::max_neighbor,
            "Max non-improving swaps per restart (-1 = auto).")
    .def_rw("max_dtw_evals", &dtwc::algorithms::CLARANSOptions::max_dtw_evals,
            "Hard budget on total DTW computations (-1 = no limit).")
    .def_rw("random_seed", &dtwc::algorithms::CLARANSOptions::random_seed,
            "RNG seed for determinism (default 42).")
    .def("__repr__", [](const dtwc::algorithms::CLARANSOptions &o) {
      return "CLARANSOptions(k=" + std::to_string(o.n_clusters)
             + ", num_local=" + std::to_string(o.num_local)
             + ", max_neighbor=" + std::to_string(o.max_neighbor)
             + ", max_dtw_evals=" + std::to_string(o.max_dtw_evals)
             + ", seed=" + std::to_string(o.random_seed) + ")";
    });

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
    .def("to_numpy", [](const dtwc::core::DenseDistanceMatrix &dm) {
      // Expand packed triangular storage to full N*N numpy array.
      const Eigen::MatrixXd full = dtwc::io::to_full_matrix(dm);
      const size_t n = dm.size();
      // Eigen is column-major; numpy expects row-major (C order).
      // Use nb::ndarray with explicit strides to handle this, or just copy
      // row-major since the matrix is symmetric (col-major == row-major for symmetric).
      double *ptr = new double[n * n];
      // Copy from Eigen column-major to row-major (symmetric, so identical)
      std::memcpy(ptr, full.data(), n * n * sizeof(double));
      nb::capsule owner(ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
      return nb::ndarray<nb::numpy, double>(ptr, {n, n}, owner);
    }, "Return a numpy array of the full N*N distance matrix.")
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

  m.def("dtw_arow_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                                  nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                                  int band, const std::string &metric) {
    nb::gil_scoped_release release;
    auto mt = dtwc::core::MetricType::L1;
    if (metric == "squared_euclidean" || metric == "sqeuclidean")
        mt = dtwc::core::MetricType::SquaredL2;
    if (band >= 0)
      return dtwc::dtwAROW_banded<double>(x.data(), x.size(), y.data(), y.size(), band, mt);
    else
      return dtwc::dtwAROW_L<double>(x.data(), x.size(), y.data(), y.size(), mt);
  }, "x"_a, "y"_a, "band"_a = -1, "metric"_a = "l1",
     "DTW-AROW distance with diagonal-only alignment for missing values.\n\n"
     "When x[i] or y[j] is NaN, the warping path is restricted to the\n"
     "diagonal direction only (one-to-one alignment), preventing free\n"
     "stretching through missing regions.\n"
     "Reference: Yurtman et al. (ECML-PKDD 2023).\n"
     "metric: 'l1' (default) or 'squared_euclidean'.\n"
     "band=-1 for full DTW-AROW, band>0 for Sakoe-Chiba banded DTW-AROW.");

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
    .def_rw("ndim", &dtwc::Data::ndim,
            "Number of features (dimensions) per timestep (default 1).")
    .def_prop_ro("size", &dtwc::Data::size)
    .def("series_length", &dtwc::Data::series_length, "i"_a,
         "Return the number of timesteps for series i (flat size / ndim).")
    .def("validate_ndim", &dtwc::Data::validate_ndim,
         "Validate that all series flat sizes are divisible by ndim.\n\n"
         "Raises RuntimeError if any series has incompatible size.");

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
    .def_rw("missing_strategy", &dtwc::Problem::missing_strategy,
            "Strategy for handling NaN values (Error, ZeroCost, AROW, Interpolate).")
    .def_rw("distance_strategy", &dtwc::Problem::distance_strategy,
            "Distance matrix computation strategy (Auto, BruteForce, Pruned, GPU).")
    .def_rw("mip_settings", &dtwc::Problem::mip_settings,
            "MIP solver tuning parameters.")
    .def_rw("verbose", &dtwc::Problem::verbose,
            "Print progress messages for long-running operations.")
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
      {
        nb::gil_scoped_release release;
        prob.fillDistanceMatrix();
      }
      // Use to_full_matrix() to expand packed triangular → full NxN.
      const auto &dm = prob.dense_distance_matrix();
      const Eigen::MatrixXd full = dtwc::io::to_full_matrix(dm);
      const size_t n = dm.size();
      double *ptr = new double[n * n];
      // Symmetric matrix: col-major == row-major, safe to memcpy.
      std::memcpy(ptr, full.data(), n * n * sizeof(double));
      nb::capsule owner(ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
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
           auto &mat = p.dense_distance_matrix();
           mat.resize(n);
           const double* data = dm.data();
           for (size_t i = 0; i < n; ++i)
               for (size_t j = i; j < n; ++j)
                   mat.set(i, j, data[i * n + j]);
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

  m.def("dunn_index", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::dunnIndex(prob);
  }, "prob"_a,
     "Compute Dunn Index (min inter-cluster distance / max intra-cluster diameter).\n\n"
     "Higher values indicate better-separated, compact clusters.");

  m.def("inertia", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::inertia(prob);
  }, "prob"_a,
     "Compute inertia (total within-cluster distance sum to medoids).");

  m.def("calinski_harabasz_index", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    return dtwc::scores::calinskiHarabaszIndex(prob);
  }, "prob"_a,
     "Compute Calinski-Harabasz Index (medoid-adapted).\n\n"
     "Higher values indicate better-defined clusters.");

  m.def("adjusted_rand_index", [](const std::vector<int> &labels_true,
                                    const std::vector<int> &labels_pred) {
    return dtwc::scores::adjustedRandIndex(labels_true, labels_pred);
  }, "labels_true"_a, "labels_pred"_a,
     "Compute Adjusted Rand Index between two label assignments.\n\n"
     "Returns a value in [-0.5, 1.0]; 1.0 indicates perfect agreement,\n"
     "0.0 indicates random labeling.");

  m.def("normalized_mutual_information", [](const std::vector<int> &labels_true,
                                              const std::vector<int> &labels_pred) {
    return dtwc::scores::normalizedMutualInformation(labels_true, labels_pred);
  }, "labels_true"_a, "labels_pred"_a,
     "Compute Normalized Mutual Information between two label assignments.\n\n"
     "Returns a value in [0.0, 1.0]; 1.0 indicates perfect agreement.");

  // =========================================================================
  // Hierarchical clustering
  // =========================================================================

  m.def("build_dendrogram", [](dtwc::Problem &prob,
                                 const dtwc::algorithms::HierarchicalOptions &opts) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::build_dendrogram(prob, opts);
  }, "prob"_a, "opts"_a = dtwc::algorithms::HierarchicalOptions{},
     "Build a hierarchical dendrogram from a Problem.\n\n"
     "Requires distance matrix to be filled (call fill_distance_matrix() first).\n"
     "Returns a Dendrogram containing N-1 merge steps in merge order.\n"
     "Throws RuntimeError if N > opts.max_points (default 2000).");

  m.def("cut_dendrogram", [](const dtwc::algorithms::Dendrogram &dend,
                               dtwc::Problem &prob, int k) {
    nb::gil_scoped_release release;
    return dtwc::algorithms::cut_dendrogram(dend, prob, k);
  }, "dendrogram"_a, "prob"_a, "k"_a,
     "Cut a dendrogram to produce k flat clusters.\n\n"
     "Returns a ClusteringResult with labels, medoid_indices, and total_cost.\n"
     "Does NOT mutate Problem.clusters_ind or Problem.centroids_ind.");

  // =========================================================================
  // CLARANS
  // =========================================================================

  m.def("clarans", [](dtwc::Problem &prob, const dtwc::algorithms::CLARANSOptions &opts) {
    dtwc::core::ClusteringResult result;
    {
      nb::gil_scoped_release release;
      result = dtwc::algorithms::clarans(prob, opts);
    }
    // Store results back into Problem so scoring functions work
    prob.set_numberOfClusters(opts.n_clusters);
    prob.centroids_ind = result.medoid_indices;
    prob.clusters_ind = result.labels;
    return result;
  }, "prob"_a, "opts"_a,
     "Run CLARANS randomized k-medoids clustering.\n\n"
     "Experimental bounded mid-ground algorithm. Tests random\n"
     "(medoid_out, x_in) swaps, accepting only strictly improving ones.\n"
     "Results are stored back into prob for scoring.\n\n"
     "Reference: Ng & Han (2002), IEEE TKDE 14(5).");

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

  // =========================================================================
  // Capability detection: OpenMP, MPI
  // =========================================================================

#ifdef _OPENMP
  m.attr("OPENMP_AVAILABLE") = true;
  m.def("openmp_max_threads", []() {
    return omp_get_max_threads();
  }, "Return the maximum number of OpenMP threads available.");
#else
  m.attr("OPENMP_AVAILABLE") = false;
  m.def("openmp_max_threads", []() { return 1; },
        "Return 1 (OpenMP not compiled in).");
#endif

#ifdef DTWC_HAS_MPI
  m.attr("MPI_AVAILABLE") = true;
#else
  m.attr("MPI_AVAILABLE") = false;
#endif

  m.def("system_info", []() {
    std::string info;
    info += "DTWC++ System Information\n";
#ifdef _OPENMP
    info += "  OpenMP: available (" + std::to_string(omp_get_max_threads()) + " threads)\n";
#else
    info += "  OpenMP: not available\n";
#endif
#ifdef DTWC_HAS_CUDA
    if (dtwc::cuda::cuda_available())
      info += "  CUDA:   available (" + dtwc::cuda::cuda_device_info(0) + ")\n";
    else
      info += "  CUDA:   compiled but no GPU detected\n";
#else
    info += "  CUDA:   not compiled (rebuild with -DDTWC_ENABLE_CUDA=ON)\n";
#endif
#ifdef DTWC_HAS_MPI
    info += "  MPI:    available\n";
#else
    info += "  MPI:    not compiled (rebuild with -DDTWC_ENABLE_MPI=ON)\n";
#endif
    return info;
  }, "Return a string summarizing available backends and capabilities.");
}
