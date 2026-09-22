/**
 * @file types.hpp
 * @brief Header for types folder.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 19 Dec 2022
 */

#pragma once

#include "Range.hpp"
// element_types.hpp / types_util.hpp used to be included here. They only ever
// held `dtwc::solver` sparse-matrix helpers, whose sole consumer is mip_Highs.cpp,
// so they moved to mip/solver_types.hpp (C-12).