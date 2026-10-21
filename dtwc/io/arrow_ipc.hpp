/**
 * @file arrow_ipc.hpp
 * @brief dtwc::io::read_arrow_ipc: the Arrow IPC reader of every build, on the vendored nanoarrow IPC extension (no
 *        Arrow C++): a file is read in place, each series a view of its bytes.
 * @author Volkan Kumtepeli
 */

#pragma once

#include "../Data.hpp"
#include "../core/storage.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace dtwc::io {

/**
 * @brief Every series of the Arrow IPC file `path`, in the file format or the stream format, as dtwc::read_data
 *        describes them, read in place: the file is mapped (llfio), or read once into one buffer in a build without
 *        llfio, and each series is a view of those bytes, which the Data's owner() keeps alive.
 *
 * `dtype` is the precision the series are stored in: a file holding the other precision is copied, each value cast
 * once. A record batch that cannot be viewed (compressed, big-endian, or a buffer not aligned to its values) is an
 * IOError naming why, as is a file that is not Arrow IPC, or is truncated or corrupt; a null series or value is
 * InvalidInput. An IOError names the file.
 */
Data read_arrow_ipc(const std::filesystem::path &path, core::Precision dtype = core::Precision::Float64);

namespace detail {

/// read_arrow_ipc's reader on the bytes of one Arrow IPC file, which `owner` keeps alive: the Data's views point into
/// `bytes`, and its owner() holds `owner`. Its errors name no file.
Data read_ipc(std::span<const std::byte> bytes, std::shared_ptr<const void> owner, core::Precision dtype);

} // namespace detail

} // namespace dtwc::io
