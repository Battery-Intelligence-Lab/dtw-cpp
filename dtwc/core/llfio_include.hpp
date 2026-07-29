/**
 * @file llfio_include.hpp
 * @brief Contain compiler diagnostic changes made by LLFIO dependencies.
 */

#pragma once

// quickcpplib/ringbuffer_log.hpp installs an unbalanced Clang ignore for
// -Wdeprecated-declarations on Windows. Preserve the caller's diagnostic state
// across the complete third-party include.
#if defined(__clang__)
#  pragma clang diagnostic push
#endif

#include <llfio/v2.0/llfio.hpp>

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
