/**
 * @file random_engine.hpp
 * @brief Compatibility forwarder — this header moved to dtwc/base/random_engine.hpp.
 *
 * @details The foundation layer now lives in its own directory (ledger C-11), so
 * that the folder a header sits in matches the layer the dependency checker
 * assigns it. This forwarder keeps the old path working for one release.
 *
 * Every include inside this repository was updated to the new path when the move
 * happened, so the message below can only be reached by code outside it.
 */

#pragma once

#pragma message("dtwc/random_engine.hpp has moved to dtwc/base/random_engine.hpp. The old path still works in this release and will be removed in the next one; please include \"base/random_engine.hpp\" instead.")

#include "base/random_engine.hpp"
