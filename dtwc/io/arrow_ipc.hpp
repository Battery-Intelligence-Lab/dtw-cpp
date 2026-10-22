/**
 * @file arrow_ipc.hpp
 * @brief dtwc::io::read_arrow_ipc: the Arrow IPC reader of every build, on the vendored nanoarrow IPC extension (no
 *        Arrow C++): a file is read in place, each series a view of its bytes, or of its batch decoded.
 * @author Volkan Kumtepeli
 */

#pragma once

#include "../Data.hpp"
#include "../core/storage.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace dtwc::io {

/**
 * @brief Every series of the Arrow IPC file `path`, in the file format or the stream format, as dtwc::read_data
 *        describes them, read in place: the file is mapped (llfio), or read once into one buffer in a build without
 *        llfio, and each series is a view of those bytes, which the Data's owner() keeps alive.
 *
 * A record batch that cannot be viewed (compressed with LZ4_FRAME or ZSTD, big-endian, or a buffer not aligned to its
 * values) is decoded, the batches in parallel, into buffers the owner holds, which its series view. `dtype` is the
 * precision the series are stored in: a file holding the other precision is copied, each value cast once. A file that
 * is not Arrow IPC, or is truncated or corrupt, is an IOError naming why; a null series or value is InvalidInput. An
 * IOError names the file.
 */
Data read_arrow_ipc(const std::filesystem::path &path, core::Precision dtype = core::Precision::Float64);

namespace detail {

/// read_arrow_ipc's reader on the bytes of one Arrow IPC file, which `owner` keeps alive: the Data's views point into
/// `bytes` (into an 8-aligned copy the owner holds, if `bytes` is not 8-aligned), and its owner() holds `owner`. Its
/// errors name no file.
Data read_ipc(std::span<const std::byte> bytes, std::shared_ptr<const void> owner, core::Precision dtype);

/// read_ipc's decoding of a record batch it cannot view, on any column: each record batch's column `column`, decoded
/// (decompressed, in this machine's byte order, aligned) and checked as read_ipc decodes a used column, as the bytes of
/// each of its buffers, depth first. For the big-endian test: arrow-testing's files in that byte order hold no 'data'.
std::vector<std::vector<std::vector<std::byte>>> decoded_buffers(std::span<const std::byte> bytes,
                                                                 std::string_view column);

} // namespace detail

} // namespace dtwc::io
