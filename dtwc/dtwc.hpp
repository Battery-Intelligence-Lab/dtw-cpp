/**
 * @file dtwc.hpp
 * @brief Main header to include to use DTWC++ library. Please only include this file.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 15 Dec 2021
 */

#pragma once

// NaN marks missing values and pruned or uncomputed distances (design §9).
// Under -ffinite-math-only (implied by -ffast-math) the compiler may assume no
// NaN exists and fold every such test away, so answers would be silently wrong.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "DTWC++ cannot be compiled with -ffinite-math-only (implied by -ffast-math): NaN marks missing values and pruned distances. Add -fno-finite-math-only after -ffast-math."
#endif

#include "base/settings.hpp"
#include "base/random_engine.hpp" //!< dtwc::randGenerator, split out of settings.hpp (X-12)
#include "api.hpp"
#include "fileOperations.hpp"
#include "Problem.hpp"
#include "checkpoint.hpp"
#include "DataLoader.hpp"
#include "io/read_data.hpp"
#include "distance.hpp"
#include "scores.hpp"
#include "utility.hpp"
#include "warping.hpp"
#include "warping_ddtw.hpp"
#include "warping_wdtw.hpp"
#include "warping_adtw.hpp"
#include "warping_missing.hpp"
#include "warping_missing_arow.hpp"
#include "base/missing_utils.hpp"
#include "soft_dtw.hpp"
#include "algorithms/fast_pam.hpp"
#include "algorithms/fast_clara.hpp"
#include "algorithms/one_batch_pam.hpp"
#include "algorithms/barycenter.hpp"
#include "algorithms/hierarchical.hpp"

// Phase 1: Core types (binding-friendly, Armadillo-independent headers)
#include "core/clustering_result.hpp"
#include "core/distance_matrix.hpp"
#include "core/matrix_io.hpp"
#include "core/dtw_options.hpp"
#include "core/lower_bound_impl.hpp"
#include "core/time_series.hpp"
#include "core/z_normalize.hpp"

#ifdef DTWC_HAS_CUDA
#include "cuda/cuda_dtw.cuh"
#endif

#ifdef DTWC_HAS_METAL
#include "metal/metal_dtw.hpp"
#endif
