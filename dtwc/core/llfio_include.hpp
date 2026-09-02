/**
 * @file llfio_include.hpp
 * @brief Contain compiler diagnostic changes made by LLFIO dependencies.
 */

#pragma once

// quickcpplib/ringbuffer_log.hpp installs an unbalanced Clang ignore for
// -Wdeprecated-declarations on Windows. Preserve the caller's diagnostic state
// across the complete third-party include. Apple libc++ also deprecates
// std::char_traits<std::byte> inside LLFIO headers; ignore that only while
// the dependency is parsed so a later -Werror=deprecated-declarations still
// diagnoses DTWC++ [[deprecated]] names (F22 / F45).
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <llfio/v2.0/llfio.hpp>

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
