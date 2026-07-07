/**
 * @file error.hpp
 * @brief DTWC++ exception taxonomy.
 *
 * A small, header-only hierarchy rooted at dtwc::Error, which derives from
 * std::runtime_error. Every DTWC++ library error derives from Error, so a
 * caller can:
 *   - catch dtwc::Error         -> handle any DTWC++ failure, or
 *   - catch std::runtime_error  -> handle it alongside the standard library,
 *   - catch std::exception      -> handle everything.
 *
 * The message carried by what() is the entire payload: no error codes, no
 * macros, no extra machinery. Constructors are inherited from
 * std::runtime_error, so every type is constructed from a message string and
 * preserves it verbatim through what().
 *
 * Categories:
 *   - InvalidInput : the caller supplied bad arguments or data (a validation
 *                    failure at a public API boundary).
 *   - SolverError  : an optimisation solver (HiGHS/Gurobi) rejected the model,
 *                    failed to run, or returned a non-optimal status.
 *   - DeviceError  : a compute-device (CPU/CUDA/...) selection or operation
 *                    failed.
 *   - IOError      : a file or data-source read/write failed.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 07 Jul 2026
 */

#pragma once

#include <stdexcept>

namespace dtwc {

/**
 * @brief Base class for all DTWC++ exceptions.
 *
 * Derives from std::runtime_error so that pre-existing
 * `catch (const std::runtime_error &)` / `catch (const std::exception &)`
 * handlers keep working unchanged.
 */
class Error : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error; // message-preserving constructors
};

/// The caller supplied invalid input (bad arguments, empty/mismatched data, ...).
class InvalidInput : public Error
{
public:
  using Error::Error;
};

/// An optimisation solver rejected/failed the model or returned a non-optimal status.
class SolverError : public Error
{
public:
  using Error::Error;
};

/// A compute-device selection or operation failed.
class DeviceError : public Error
{
public:
  using Error::Error;
};

/// A file or data-source I/O operation failed.
class IOError : public Error
{
public:
  using Error::Error;
};

} // namespace dtwc
