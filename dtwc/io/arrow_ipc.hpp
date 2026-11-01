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
 * lengths), and decodes nothing. The file is mapped (llfio), or without llfio read in pieces: the walk reads each
 * message's header and, of a batch to be decoded, each used buffer's declared length and frame descriptor, and a batch
 * is read whole when it is read (each with a file read). A batch is read as read_arrow_ipc reads it: viewed where it
 * lies or decoded, its rows' values and names checked; read again, it must be the batch the walk found, else the file
 * changed while it was read (IOError). Its series are views of T (read in the other precision, its values cast once
 * into a buffer the views point into), its rows named by nothing named series_<row>, owned by the Data. An IOError
 * names the file.
 *
 * The prices: a record batch from its read until it is freed (largest_row_group_bytes); what a thread holds while it
 * reads one, the codec's scratch for the file's largest frame and nanoarrow's decoders (decode_bytes holds one, and
 * row_groups_per_batch and read_rows a team's others); the walk's facts, held while the file is open (decode_bytes).
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
  /// series, the batches it decodes and the scratch of the threads decoding them, each batch's objects, without llfio
  /// the whole file it reads, in the other precision the copy; and with `gpu` the copy a GPU uploads of a view (a
  /// view's values are the page cache's, or the decoded batches').
  std::size_t resident_bytes(core::Precision dtype, bool gpu) const;
  /// Whether read_arrow_ipc(path, dtype) holds values outside the page cache: decoded batches, the file read without
  /// llfio, or a copy in the other precision. A stream holds a batch of them at a time; a mapped view, none.
  bool streams(core::Precision dtype) const;

  /// The series of the file's rows `rows` (each in [0, num_rows())), in that order, as an owned copy of T: the record
  /// batches holding them read in parallel, each whole by a thread with its scratch, as many at a time as `ram_budget`
  /// holds beside the walk, the rows copied so far and each batch's copies (at most its values and names), and
  /// released once their rows are copied. InvalidInput, before any batch is read, if a record batch is larger than
  /// ram_budget (the stream reads each whole) or the rows' objects and one batch with its copies exceed it; and if the
  /// rows copied leave no room for one more batch.
  template <typename T>
  Data read_rows(std::vector<index_t> rows, std::size_t ram_budget) const;
  /// Record batches [first, first + count) as one view Data of T: read in parallel, each handed to check() by the
  /// thread that read it; the lowest failing batch's error is rethrown (run_openmp), as a serial read raises it.
  template <typename T>
  Data read_row_groups(int first, int count, const PieceCheck &check) const;
  /// The record batches a batch of read_row_groups() holds within `ram_budget`, read by a thread each up to the team's
  /// size, each thread but one (decode_bytes) with its scratch: InvalidInput if not one.
  int row_groups_per_batch(std::size_t ram_budget, bool f32) const;
  /// The most a record batch holds from its read until its batch is freed, as Float32 (`f32`) or data_t: its message
  /// read from the file (without llfio), its decoded columns, its values cast into the other precision, its series'
  /// spans and name views and their copy in the batch's Data, the names series_<row> made for rows with none, and its
  /// objects (its Data, owner and nanoarrow's arrays, the stream's check of it).
  std::size_t largest_row_group_bytes(bool f32) const;
  /// What reading holds beside the record batches it hands over, which are priced whole (largest_row_group_bytes): the
  /// walk's facts, held while the file is open, and what one thread holds while it reads a batch (reader_bytes);
  /// row_groups_per_batch and read_rows price a team's other threads.
  std::size_t decode_bytes() const;
  /// What a thread holds while it reads a record batch, freed before its next: the codec's scratch for the file's
  /// largest frame (zstd's context; LZ4F's context and block buffers, from each frame's descriptor), and nanoarrow's
  /// decoders and views of the schema's fields.
  std::size_t reader_bytes() const;
  /// What a record batch's objects hold from its read until it is freed, beyond what its price counts by its rows and
  /// bytes, on every path that holds batches: the stream's (its Data and their vectors, its owner and the owner's
  /// vectors, its decoded columns' nanoarrow arrays and their private data, the template's RowCheck, its share in the
  /// block's Data), read_rows' (the batch's Data while its rows are copied), and the in-RAM read's (a decoded batch's
  /// arrays until the read ends, the walk's facts of it). Measured 2026-11-01 on Windows' private bytes (the UCRT
  /// heap, clang 21), files of 200,000 record batches of one series, a block of every batch read by 24 threads (its
  /// peak): at most 1,690 bytes a batch beyond its other prices (LZ4 with a 'name' column), 1,480 without one, 590
  /// uncompressed.
  static constexpr std::size_t batch_object_bytes = 2048;
  /// What nanoarrow allocates for each field of the schema (the root and every column, depth first) in a thread that
  /// reads a record batch, freed after it: the array view and the array of the schema its decoder sets (SetSchema),
  /// and of a decoded batch the batch's view. Measured 2026-11-01 as above: 846 bytes a field on one thread (203
  /// fields, LZ4), about 400 over 24 threads.
  static constexpr std::size_t field_bytes = 1024;
  /// What a decoded record batch's view of the schema holds a field in read_arrow_ipc, until it returns: nanoarrow
  /// 0.8's ArrowArrayView (208 bytes on a 64-bit target), its place in its parent's children, the allocator's header.
  /// Measured 2026-11-01 as above: 230 bytes a field (96 batches of 203 fields, one thread).
  static constexpr std::size_t view_field_bytes = 320;
  /// What the walk's decoded schema holds a field while the file is open: nanoarrow's ArrowSchema (72 bytes), its
  /// place in its parent's children, its format and name strings, each with the allocator's header (about 160 bytes,
  /// read in nanoarrow 0.8).
  static constexpr std::size_t schema_field_bytes = 256;
  /// The bytes `rows` owned series of `values` values hold (dtwc::detail::owned_series_bytes): read_rows' copies.
  static std::size_t series_bytes(std::size_t values, std::size_t rows, bool f32)
  {
    return dtwc::detail::owned_series_bytes(values, rows, f32);
  }
  /// The bytes series_names() returns: a string a series, and with a 'name' column its bytes and the allocator's header
  /// a name.
  std::size_t series_names_bytes() const;
  /// What series_names() holds beside what it returns, read by one thread: the walk's facts and, with a 'name' column,
  /// the most reading one record batch's names holds (without llfio its header and the bytes the column spans; the
  /// column decoded; the batch's objects) with the thread's scratch.
  std::size_t names_read_bytes() const;
  /// Every series' name, as read_arrow_ipc names it, the 'name' column alone read (without llfio, of each record batch
  /// its header and the bytes the column spans), by as many threads as `ram_budget` holds beside the names and the
  /// walk: what a streamed run writes. InvalidInput if `ram_budget` holds not one (series_names_bytes() and
  /// names_read_bytes() more than it).
  std::vector<std::string> series_names(std::size_t ram_budget) const;

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
/// `bytes` (into an 8-aligned copy the owner holds, if `bytes` is not 8-aligned), and its owner() holds `owner`. With
/// `mapped`, `bytes` are a file's mapping: a Data of views of them alone is mapped (Data::is_mapped). Its errors name
/// no file.
Data read_ipc(std::span<const std::byte> bytes, std::shared_ptr<const void> owner, core::Precision dtype,
              bool mapped = false);

/// read_ipc's decoding of a record batch it cannot view, on any column: each record batch's column `column`, decoded
/// (decompressed, in this machine's byte order, aligned) and checked as read_ipc decodes a used column, as the bytes of
/// each of its buffers, depth first. For the big-endian test: arrow-testing's files in that byte order hold no 'data'.
std::vector<std::vector<std::vector<std::byte>>> decoded_buffers(std::span<const std::byte> bytes,
                                                                 std::string_view column);

} // namespace detail

} // namespace dtwc::io
