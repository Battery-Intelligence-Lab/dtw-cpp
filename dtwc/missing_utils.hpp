/**
 * @file missing_utils.hpp
 * @brief Compatibility forwarder — this header moved to dtwc/base/missing_utils.hpp.
 *
 * @details The foundation layer now lives in its own directory (ledger C-11), so
 * that the folder a header sits in matches the layer the dependency checker
 * assigns it. This forwarder keeps the old path working for one release.
 *
 * Every include inside this repository was updated to the new path when the move
 * happened, so the message below can only be reached by code outside it.
 */

#pragma once

#pragma message("dtwc/missing_utils.hpp has moved to dtwc/base/missing_utils.hpp. The old path still works in this release and will be removed in the next one; please include \"base/missing_utils.hpp\" instead.")

#include "base/missing_utils.hpp"
