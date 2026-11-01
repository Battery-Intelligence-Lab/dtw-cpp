/**
 * @file arrow_ipc.hpp
 * @brief dtwc::io::read_arrow_ipc: the Arrow IPC reader of every build, on the vendored nanoarrow IPC extension (no
 *        Arrow C++): a file is read in place, each series a view of its bytes, or of its batch decoded.
 * @author Volkan Kumtepeli
 */

#pragma once

#include "../Data.hpp"
#include "../base/settings.hpp"
#include "../core/storage.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
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

/**
 * @brief An Arrow IPC file read a record batch at a time: what FastCLARA's stream (algorithms::detail::
 *        fast_clara_chunked) reads a file larger than its memory limit with, and what run's planner prices a read by.
 *
 * The constructor walks the file as read_arrow_ipc does, with its checks and texts (both magics, the footer against
 * the messages, each batch's buffers inside its body, each compressed buffer's declared decoded bytes against its
 * lengths), and decodes nothing. The file is mapped (llfio), or without llfio read a message at a time, the walk once
 * and each batch again when it is read (each with a file read). A batch is read as read_arrow_ipc reads it: viewed
 * where it lies or decoded, its rows' values and names checked; read again, it must be the batch the walk found, else
 * the file changed while it was read (IOError). Its series are views of T (read in the other precision, its values
 * cast once into a buffer the views point into), its rows named by nothing named series_<row>, owned by the Data.
 * An IOError names the file.
 */
class IpcFile
{
public:
  /// For algorithms::detail::fast_clara_chunked: a batch's record batches are read and checked by a team of threads.
  static constexpr bool decodes_with_team = true;
  static constexpr const char *piece = "record batch", *pieces = "record batches"; ///< what its refusals call them
  /// check(j, batch, first_row): the j-th record batch read_row_groups reads, the file's rows from first_row on.
  using PieceCheck = std::function<void(std::size_t, const Data &, index_t)>;

  explicit IpcFile(const std::filesystem::path &path);
  ~IpcFile();
  IpcFile(const IpcFile &) = delete;
  IpcFile &operator=(const IpcFile &) = delete;

  index_t num_rows() const;     ///< the series, every record batch's rows
  int num_row_groups() const;   ///< the record batches
  bool is_f32() const;          ///< whether the values are Float32
  std::size_t ndim() const;     ///< features a time step: the schema's metadata 'ndim', else 1

  /// What read_arrow_ipc(path, dtype) holds at once outside the page cache: a view's detail::view_series_bytes a
  /// series, the batches it decodes, without llfio the whole file it reads, in the other precision the copy; and with
  /// `gpu` the copy a GPU uploads of a view (a view's values are the page cache's, or the decoded batches').
  std::size_t resident_bytes(core::Precision dtype, bool gpu) const;
  /// Whether read_arrow_ipc(path, dtype) holds values outside the page cache: decoded batches, the file read without
  /// llfio, or a copy in the other precision. A stream holds a batch of them at a time; a mapped view, none.
  bool streams(core::Precision dtype) const;

  /// The series of the file's rows `rows` (each in [0, num_rows())), in that order, as an owned copy of T: the record
  /// batches holding them read in parallel, each whole, as many at a time as `ram_budget` holds beside the rows copied
  /// so far, and released once their rows are copied. InvalidInput, before any batch is read, if a record batch is
  /// larger than ram_budget (the stream reads each whole) or the rows' objects and one batch exceed it; and if the rows
  /// copied leave no room for one more batch.
  template <typename T>
  Data read_rows(std::vector<index_t> rows, std::size_t ram_budget) const;
  /// Record batches [first, first + count) as one view Data of T: read in parallel, each handed to check() by the
  /// thread that read it; the lowest failing batch's error is rethrown (run_openmp), as a serial read raises it.
  template <typename T>
  Data read_row_groups(int first, int count, const PieceCheck &check) const;
  /// The record batches a batch of read_row_groups() holds within `ram_budget`: InvalidInput if not one.
  int row_groups_per_batch(std::size_t ram_budget, bool f32) const;
  /// The most a record batch holds from its read until its batch is freed, as Float32 (`f32`) or data_t: its message
  /// read from the file (without llfio), its decoded columns, its values cast into the other precision, its series'
  /// spans and name views and their copy in the batch's Data, and the names series_<row> made for rows with none.
  std::size_t largest_row_group_bytes(bool f32) const;
  /// What reading holds beside the record batches it hands over, which are priced whole (largest_row_group_bytes):
  /// nothing, whatever the team's size.
  std::size_t decode_bytes() const { return 0; }
  /// The bytes `rows` owned series of `values` values hold (dtwc::detail::owned_series_bytes): read_rows' copies.
  static std::size_t series_bytes(std::size_t values, std::size_t rows, bool f32)
  {
    return dtwc::detail::owned_series_bytes(values, rows, f32);
  }
  /// The bytes series_names() returns: a string a series, and with a 'name' column its bytes and the allocator's
  /// header a name.
  std::size_t series_names_bytes() const;
  /// Every series' name, as read_arrow_ipc names it, the 'name' column alone read: what a streamed run writes.
  std::vector<std::string> series_names() const;

  struct State;

private:
  std::unique_ptr<const State> state_;
};

extern template Data IpcFile::read_rows<float>(std::vector<index_t>, std::size_t) const;
extern template Data IpcFile::read_rows<data_t>(std::vector<index_t>, std::size_t) const;
extern template Data IpcFile::read_row_groups<float>(int, int, const PieceCheck &) const;
extern template Data IpcFile::read_row_groups<data_t>(int, int, const PieceCheck &) const;

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
