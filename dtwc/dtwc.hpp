/**
 * @file dtwc.hpp
 * @brief Main header to include to use DTWC++ library. Please only include this file.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 15 Dec 2021
 */

#pragma once

#include "settings.hpp"
#include "fileOperations.hpp"
#include "Problem.hpp"
#include "DataLoader.hpp"
#include "scores.hpp"
#include "utility.hpp"
#include "warping.hpp"
#include "algorithms/fast_pam.hpp"

// Phase 1: Core types (binding-friendly, Armadillo-independent headers)
#include "core/clustering_result.hpp"
#include "core/distance_matrix.hpp"
#include "core/distance_metric.hpp"
#include "core/dtw.hpp"
#include "core/dtw_options.hpp"
#include "core/lower_bounds.hpp"
#include "core/scratch_matrix.hpp"
#include "core/time_series.hpp"
#include "core/z_normalize.hpp"
