/**
 * @file arrow_ipc.cpp
 * @brief dtwc::io::read_arrow_ipc and io::IpcFile (see arrow_ipc.hpp): the messages walked and decoded by nanoarrow's
 *        IPC extension, each record batch's buffers viewed where they lie.
 * @author Volkan Kumtepeli
 */

#include "arrow_ipc.hpp"

#include "read_data.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"
#include "../core/distance_matrix.hpp" // available_memory

#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"

#ifdef DTWC_HAS_MMAP
// As in core/distance_matrix.cpp: quickcpplib leaves an unbalanced Clang ignore for -Wdeprecated-declarations on
// Windows, and Apple libc++ deprecates std::char_traits<std::byte> inside llfio.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <llfio/v2.0/llfio.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace dtwc::io {
namespace {

/// An array nanoarrow decoded into buffers it allocated, released on every path.
struct Array
{
  ArrowArray arrow{};
  Array() = default;
  Array(Array &&other) noexcept : arrow(other.arrow) { other.arrow.release = nullptr; }
  Array &operator=(Array &&) = delete;
  ~Array()
  {
    if (arrow.release != nullptr) arrow.release(&arrow);
  }
};

/// A view whose children nanoarrow allocated, reset on every path.
struct View
{
  ArrowArrayView arrow{};
  View() = default;
  View(const View &) = delete;
  View &operator=(const View &) = delete;
  ~View() { ArrowArrayViewReset(&arrow); }
};

/// What a view of the file holds: the bytes' keep-alive (none once every batch is decoded), the decoded columns, the
/// names made for the series that have none, and the values of a record batch read in the other precision.
struct Owner
{
  std::shared_ptr<const void> bytes;
  std::vector<Array> decoded;
  std::vector<char> names;
  std::shared_ptr<const void> values;
};

/// nanoarrow's failure as an IOError naming the step; some of its returns set no message. ENOMEM, an allocation the
/// system refused it, is std::bad_alloc, as every refused allocation is.
void check(ArrowErrorCode code, const ArrowError &error, const std::string &step)
{
  if (code == ENOMEM) throw std::bad_alloc();
  if (code != NANOARROW_OK)
    throw IOError(step + ": "
                  + (error.message[0] != '\0' ? std::string(error.message)
                                              : "nanoarrow error " + std::to_string(code)));
}

/// A schema and a decoder, released on every path, the decoder first (as nanoarrow's own reader releases them).
struct Schema
{
  ArrowSchema arrow{};
  Schema() = default;
  Schema(const Schema &) = delete;
  Schema &operator=(const Schema &) = delete;
  ~Schema()
  {
    if (arrow.release != nullptr) arrow.release(&arrow);
  }
};
struct Decoder
{
  ArrowIpcDecoder ipc{};
  Decoder()
  {
    if (ArrowIpcDecoderInit(&ipc) != NANOARROW_OK) throw std::bad_alloc();
  }
  Decoder(const Decoder &) = delete;
  Decoder &operator=(const Decoder &) = delete;
  ~Decoder() { ArrowIpcDecoderReset(&ipc); }
};

/// One message: its header at byte `at` of the file, `header` bytes long, then its body of `body` bytes.
struct Message
{
  std::int64_t at = 0, header = 0, body = 0;
  std::int64_t end() const { return at + header + body; }
};

ArrowBufferView view_of(std::span<const std::byte> bytes)
{
  ArrowBufferView view;
  view.data.data = bytes.data();
  view.size_bytes = static_cast<std::int64_t>(bytes.size());
  return view;
}

/// The bytes of an Arrow IPC file: all of them in memory (mapped, or read into one buffer), or read from the file a
/// range at a time (an IpcFile without llfio). Used by one thread at a time.
class Source
{
public:
  explicit Source(std::span<const std::byte> memory)
    : memory_(memory), size_(static_cast<std::int64_t>(memory.size()))
  {}
  Source(const std::filesystem::path &path, std::int64_t size) : path_(&path), size_(size) {}
  std::int64_t size() const { return size_; }
  bool in_memory() const { return path_ == nullptr; }

  /// The bytes [first, last) of the file: a view of memory, or read into `buffer` (a vector's allocation, so aligned
  /// for any value), which holds them until its next use.
  std::span<const std::byte> range(std::int64_t first, std::int64_t last, std::vector<std::byte> &buffer) const
  {
    if (in_memory()) return memory_.subspan(static_cast<std::size_t>(first), static_cast<std::size_t>(last - first));
    buffer.resize(static_cast<std::size_t>(last - first));
    read(first, last, buffer.data());
    return buffer;
  }

  /// The bytes [first, last) of a file not in memory, read into `into`.
  void read(std::int64_t first, std::int64_t last, std::byte *into) const
  {
    if (!file_.is_open()) file_.open(*path_, std::ios::binary);
    file_.clear();
    file_.seekg(first);
    file_.read(reinterpret_cast<char *>(into), static_cast<std::streamsize>(last - first));
    if (!file_ || file_.gcount() != static_cast<std::streamsize>(last - first))
      throw IOError("only " + std::to_string(file_.gcount()) + " of the " + std::to_string(last - first)
                    + " bytes at byte " + std::to_string(first) + " could be read: the file changed while it was read");
  }

private:
  std::span<const std::byte> memory_;
  const std::filesystem::path *path_ = nullptr;
  std::int64_t size_ = 0;
  mutable std::ifstream file_;
};

/// A record batch's body as nanoarrow places its buffers, at `view` (each buffer at its offset from the header added to
/// view's address): its bytes in memory; or, of a file read without llfio, a place that holds none of them (`file`
/// set), the bytes a check needs read from the file, the body's first at byte `at`.
struct Body
{
  ArrowBufferView view{};
  const Source *file = nullptr;
  std::int64_t at = 0;

  /// The first `n` bytes of `buffer`, one of the body's buffers proven inside it (fewer if it is shorter): in memory,
  /// or read from the file into `scratch`.
  std::span<const std::byte> head(const ArrowBufferView &buffer, std::int64_t n, std::vector<std::byte> &scratch) const
  {
    const std::int64_t size = std::min(n, buffer.size_bytes);
    if (file == nullptr) return { static_cast<const std::byte *>(buffer.data.data), static_cast<std::size_t>(size) };
    const auto offset = static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(buffer.data.data)
                                                  - reinterpret_cast<std::uintptr_t>(view.data.data));
    return file->range(at + offset, at + offset + size, scratch);
  }
};

/// The body of `message`, whose bytes are `bytes` (its own: its header, then its body).
Body body_of(std::span<const std::byte> bytes, const Message &message)
{
  return { view_of(bytes.subspan(static_cast<std::size_t>(message.header))) };
}

/// The body of `message`, of the file `source` reads without llfio, none of it read: placed for nanoarrow, which only
/// adds the header's buffer offsets to its address (a walk sets no codec and no byte order), as the message read whole
/// into a vector is placed, at a 16-aligned address with its body its header's length on, so that a buffer is aligned
/// here as it is there. The address, 64, lies in the first page, which no process maps: a read through it would fault.
/// The checks read the bytes they need from the file.
Body unread_body(const Source &source, const Message &message)
{
  Body body;
  body.view.data.data =
    reinterpret_cast<const void *>(std::uintptr_t{ 64 } + static_cast<std::uintptr_t>(message.header));
  body.view.size_bytes = message.body;
  body.file = &source;
  body.at = message.at + message.header;
  return body;
}

/// The bytes [first, last) that hold the messages: the file format's between its leading magic and its footer, the
/// stream format's all of them.
std::pair<std::int64_t, std::int64_t> message_bytes(const Source &source)
{
  const std::int64_t size = source.size();
  std::vector<std::byte> head_buffer, tail_buffer;
  const auto head = source.range(0, std::min<std::int64_t>(size, 8), head_buffer);
  const auto holds = [](std::span<const std::byte> bytes, std::size_t at, std::string_view text) {
    return bytes.size() >= at + text.size() && std::memcmp(bytes.data() + at, text.data(), text.size()) == 0;
  };
  constexpr std::string_view magic = "ARROW1";
  if (holds(head, 0, magic)) {
    // The magic and 2 bytes of padding, the stream, the footer, its length (a little-endian int32), the magic.
    const auto tail = size >= 18 ? source.range(size - 10, size, tail_buffer) : std::span<const std::byte>();
    if (size < 18 || !holds(tail, 4, magic))
      throw IOError("an Arrow IPC file ends with the ARROW1 magic it begins with, and this one does not: it is "
                    "truncated");
    std::uint32_t length = 0;
    for (std::size_t i = 4; i-- > 0;) length = (length << 8) | std::to_integer<std::uint32_t>(tail[i]);
    const auto footer = static_cast<std::int64_t>(static_cast<std::int32_t>(length));
    // nanoarrow adds the 10 bytes after the footer to its length in an int32 (VerifyFooter, DecodeFooter).
    if (footer <= 0 || footer > size - 18 || footer > std::numeric_limits<std::int32_t>::max() - 10)
      throw IOError("the Arrow IPC footer's length, " + std::to_string(footer) + " bytes, does not fit in the file's "
                    + std::to_string(size) + ": it is truncated or corrupt");
    return { 8, size - 10 - footer };
  }
  if (holds(head, 0, std::string_view("\xFF\xFF\xFF\xFF", 4))) return { 0, size };
  if (holds(head, 0, "FEA1"))
    throw IOError("a Feather version 1 file (magic FEA1), which is not Arrow IPC: write it as Feather version 2, which "
                  "is (pyarrow.feather.write_feather(table, path, version=2, compression=\"uncompressed\"))");
  throw IOError("not an Arrow IPC file: it begins with neither the ARROW1 magic of the file format nor the 0xFFFFFFFF "
                "of the stream format");
}

/// The message at byte `at` of [at, last), its header verified (flatcc) and decoded, and `window` its bytes, which the
/// decoder's state points into: in memory the whole message, [at, its end); in a file only its header, read into
/// `buffer` (its body is read where a check needs it, unread_body); none at `last`, or at the end-of-stream marker.
std::optional<Message> next_message(Decoder &decoder, const Source &source, std::int64_t at, std::int64_t last,
                                    std::vector<std::byte> &buffer, std::span<const std::byte> &window)
{
  if (at == last) return std::nullopt;
  const std::string step = "Arrow IPC message at byte " + std::to_string(at);
  const bool memory = source.in_memory();
  ArrowError error{};
  std::int32_t prefix = 0;
  // nanoarrow reads a message's first 8 bytes for its length, then its header alone.
  const ArrowErrorCode peeked = ArrowIpcDecoderPeekHeader(
    &decoder.ipc, view_of(source.range(at, memory ? last : std::min(last, at + 8), buffer)), &prefix, &error);
  if (peeked == ENODATA) return std::nullopt;
  check(peeked, error, step);
  // The header's length, from the file, which nanoarrow's verifier trusts: at least its 8-byte prefix, inside the file.
  const std::int64_t header = decoder.ipc.header_size_bytes;
  if (header < 8 || header > last - at)
    throw IOError(step + ": its header of " + std::to_string(header) + " bytes does not fit in the "
                  + std::to_string(last - at) + " left: the file is truncated or corrupt");
  const std::span<const std::byte> bytes = source.range(at, memory ? last : at + header, buffer);
  const ArrowBufferView data = view_of(bytes);
  check(ArrowIpcDecoderVerifyHeader(&decoder.ipc, data, &error), error, step);
  check(ArrowIpcDecoderDecodeHeader(&decoder.ipc, data, &error), error, step);
  const Message message{ at, header, decoder.ipc.body_size_bytes };
  if (message.body < 0 || message.body > last - at - message.header)
    throw IOError(step + ": its body of " + std::to_string(message.body) + " bytes runs past the "
                  + std::to_string(last - at - message.header) + " left: the file is truncated or corrupt");
  window = memory ? bytes.first(static_cast<std::size_t>(message.end() - at)) : bytes;
  return message;
}

/// The header of a message the walk found at `message`, in `bytes`, the message's own bytes, verified and decoded
/// into `decoder` again: the walk's lengths, else the file changed while it was read.
void redecode(Decoder &decoder, std::span<const std::byte> bytes, const Message &message, const std::string &step)
{
  ArrowError error{};
  std::int32_t prefix = 0;
  const ArrowBufferView data = view_of(bytes);
  check(ArrowIpcDecoderPeekHeader(&decoder.ipc, data, &prefix, &error), error, step);
  if (decoder.ipc.header_size_bytes == message.header) {
    check(ArrowIpcDecoderVerifyHeader(&decoder.ipc, data, &error), error, step);
    check(ArrowIpcDecoderDecodeHeader(&decoder.ipc, data, &error), error, step);
  }
  if (decoder.ipc.header_size_bytes != message.header || decoder.ipc.body_size_bytes != message.body)
    throw IOError(step + "'s message is not the one the file was walked with: the file changed while it was read");
}

/// The schema's columns the series come from, checked with the Parquet reader's texts: 'data', a List or LargeList of
/// Float32/Float64, and 'name', Utf8 or LargeUtf8 (-1: none), and the metadata 'ndim'.
struct Columns
{
  std::int64_t data = -1, name = -1;
  bool f32 = false;
  std::size_t ndim = 1;
};

/// An IOError naming `where` unless each buffer of `view` and of its children lies in the record batch's `body` and,
/// with `lengths`, each length fits in it: nanoarrow trusts the file's buffer offsets and lengths, and its validation
/// sizes the buffers from those lengths. A batch to be decoded is checked without its lengths (a compressed body can
/// be smaller than its values), which are bounded once it is decoded.
void check_in_body(const ArrowArrayView *view, const ArrowBufferView &body, const std::string &where,
                   bool lengths = true)
{
  // Each element of the columns read (a list, its values, a string) takes at least a byte of the body.
  if (lengths && (view->length < 0 || view->length > body.size_bytes))
    throw IOError(where + ": a length of " + std::to_string(view->length) + " does not fit in the batch's body of "
                  + std::to_string(body.size_bytes) + " bytes: the file is corrupt");
  const auto size = static_cast<std::uintptr_t>(body.size_bytes);
  for (const ArrowBufferView &buffer : view->buffer_views) {
    // The buffer's offset in the body, modulo 2^64 (a body placed for a read of part of it may begin below address 0).
    const auto at =
      reinterpret_cast<std::uintptr_t>(buffer.data.data) - reinterpret_cast<std::uintptr_t>(body.data.data);
    if (buffer.size_bytes != 0
        && (buffer.size_bytes < 0 || at > size || static_cast<std::uintptr_t>(buffer.size_bytes) > size - at))
      throw IOError(where + ": a buffer of " + std::to_string(buffer.size_bytes)
                    + " bytes is not inside the batch's body of " + std::to_string(body.size_bytes)
                    + " bytes: the file is corrupt");
  }
  for (std::int64_t i = 0; i < view->n_children; ++i) check_in_body(view->children[i], body, where, lengths);
}

/// The bytes [first, last) of the body that the buffers of `view` and its children span, each inside it
/// (check_in_body); {0, 0} with none.
std::pair<std::int64_t, std::int64_t> span_of(const ArrowArrayView *view, const ArrowBufferView &body)
{
  std::pair<std::int64_t, std::int64_t> span{ std::numeric_limits<std::int64_t>::max(), 0 };
  const auto add = [&](const auto &self, const ArrowArrayView *node) -> void {
    for (const ArrowBufferView &buffer : node->buffer_views)
      if (buffer.size_bytes > 0) {
        const auto at = static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(buffer.data.data)
                                                  - reinterpret_cast<std::uintptr_t>(body.data.data));
        span = { std::min(span.first, at), std::max(span.second, at + buffer.size_bytes) };
      }
    for (std::int64_t i = 0; i < node->n_children; ++i) self(self, node->children[i]);
  };
  add(add, view);
  return span.second == 0 ? std::pair<std::int64_t, std::int64_t>{ 0, 0 } : span;
}

/// Whether the offsets or values (buffer 1) of `view` and of its children start at an address their elements can be
/// read from; validity bits and characters are bytes.
bool aligned(const ArrowArrayView *view)
{
  const auto bytes = static_cast<std::uintptr_t>(view->layout.element_size_bits[1] / 8);
  if (bytes > 1 && reinterpret_cast<std::uintptr_t>(view->buffer_views[1].data.data) % bytes != 0) return false;
  for (std::int64_t i = 0; i < view->n_children; ++i)
    if (!aligned(view->children[i])) return false;
  return true;
}

/// The index nanoarrow's decoder gives column `c`: every field depth first, the batch's root not counted.
std::int64_t field_index(const ArrowSchema &schema, std::int64_t c)
{
  const auto fields = [](const auto &self, const ArrowSchema *field) -> std::int64_t {
    std::int64_t n = 1;
    for (std::int64_t i = 0; i < field->n_children; ++i) n += self(self, field->children[i]);
    return n;
  };
  std::int64_t index = 0;
  for (std::int64_t i = 0; i < c; ++i) index += fields(fields, schema.children[i]);
  return index;
}

/// A column the series come from: its index among the batch's columns, its field index (field_index), and the name its
/// errors give it.
struct Use
{
  std::int64_t column, field;
  const char *name;
};
using Used = std::vector<Use>;

/// The used columns: 'data', and 'name' if the file has one.
Used used_of(const ArrowSchema &schema, const Columns &columns)
{
  Used used{ { columns.data, field_index(schema, columns.data), "data" } };
  if (columns.name >= 0) used.push_back({ columns.name, field_index(schema, columns.name), "name" });
  return used;
}

/// An IOError naming `where` unless the length of `used`, a column the series come from, is the batch's `rows`.
void check_rows(const ArrowArrayView *used, std::int64_t rows, const std::string &where)
{
  if (used->length != rows)
    throw IOError(where + ": its length of " + std::to_string(used->length) + " is not the batch's "
                  + std::to_string(rows) + ": the file is corrupt");
}

/// Record batch `k`'s views of every column, nanoarrow's pointers into its body; none when a used buffer is not aligned
/// to its elements (the batch is decoded instead, into aligned buffers). The columns the series come from are checked
/// to lie in the body, as long as the batch, aligned and, with `validate`, validated (DEFAULT: every buffer size, and
/// the first and last offset of each list and string column, read as integers); the others are never read. A walk of
/// a file read without llfio, whose body is unread (unread_body), validates when the batch is read.
ArrowArrayView *view_batch(Decoder &decoder, const ArrowBufferView &body, const Used &columns, std::size_t k,
                           bool validate = true)
{
  const std::string step = "Arrow IPC record batch " + std::to_string(k);
  ArrowError error{};
  ArrowArrayView *view = nullptr;
  check(ArrowIpcDecoderDecodeArrayView(&decoder.ipc, body, -1, &view, &error), error, step);
  for (const Use &use : columns) {
    const std::string where = step + ", column '" + use.name + "'";
    ArrowArrayView *used = view->children[use.column];
    check_in_body(used, body, where);
    check_rows(used, view->length, where);
    if (!aligned(used)) return nullptr;
    if (validate) check(ArrowArrayViewValidate(used, NANOARROW_VALIDATION_LEVEL_DEFAULT, &error), error, where);
  }
  return view;
}

/// `view`, made from the column's schema, filled from `array`, which nanoarrow decoded: each buffer's size is the
/// size it allocated.
void view_decoded(ArrowArrayView *view, ArrowArray *array)
{
  view->length = array->length;
  view->offset = array->offset;
  view->null_count = array->null_count;
  for (std::int64_t i = 0; i < std::min<std::int64_t>(array->n_buffers, NANOARROW_MAX_FIXED_BUFFERS); ++i) {
    const ArrowBuffer *buffer = ArrowArrayBuffer(array, i);
    view->buffer_views[i].data.data = buffer->data;
    view->buffer_views[i].size_bytes = buffer->size_bytes;
  }
  for (std::int64_t i = 0; i < view->n_children; ++i) view_decoded(view->children[i], array->children[i]);
}

/// The bytes of `view`'s buffers and its children's.
std::int64_t decoded_bytes(const ArrowArrayView *view)
{
  std::int64_t bytes = 0;
  for (const ArrowBufferView &buffer : view->buffer_views) bytes += buffer.size_bytes;
  for (std::int64_t i = 0; i < view->n_children; ++i) bytes += decoded_bytes(view->children[i]);
  return bytes;
}

/// An IOError naming `where` unless each length of `view` and of its children is in [0, `bytes`], the column's decoded
/// bytes: each element of a column the series come from takes at least one, and nanoarrow's validation sizes the
/// buffers from the lengths, in a product that wraps for lengths of 2^57 and more.
void check_lengths(const ArrowArrayView *view, std::int64_t bytes, const std::string &where)
{
  if (view->length < 0 || view->length > bytes)
    throw IOError(where + ": a length of " + std::to_string(view->length) + " does not fit in the column's "
                  + std::to_string(bytes) + " decoded bytes: the file is corrupt");
  for (std::int64_t i = 0; i < view->n_children; ++i) check_lengths(view->children[i], bytes, where);
}

/// `a + b`, at most INT64_MAX: a sum of declared sizes, which a file can make as large as it likes.
std::int64_t sum(std::int64_t a, std::int64_t b)
{
  return b > std::numeric_limits<std::int64_t>::max() - a ? std::numeric_limits<std::int64_t>::max() : a + b;
}

/// The bytes buffer `i` of `view` takes from its lengths alone, at most INT64_MAX: a validity bitmap's bits, a list's
/// or a string's offsets, a Float32/Float64 column's values, a Utf8 column's characters at most what its 32-bit
/// offsets reach; -1 where nothing bounds it (a LargeUtf8 column's characters, priced as they declare).
std::int64_t needed_bytes(const ArrowArrayView *view, int i)
{
  constexpr std::int64_t most = std::numeric_limits<std::int64_t>::max();
  const std::int64_t n = view->length, width = view->layout.element_size_bits[i] / 8;
  const ArrowBufferType type = view->layout.buffer_type[i];
  if (type == NANOARROW_BUFFER_TYPE_VALIDITY) return n / 8 + (n % 8 != 0);
  if (type == NANOARROW_BUFFER_TYPE_DATA_OFFSET && width > 0) return n < most / width ? (n + 1) * width : most;
  if (type != NANOARROW_BUFFER_TYPE_DATA) return -1;
  if (width > 0 && (view->storage_type == NANOARROW_TYPE_FLOAT || view->storage_type == NANOARROW_TYPE_DOUBLE))
    return n <= most / width ? n * width : most;
  return view->storage_type == NANOARROW_TYPE_STRING ? std::int64_t{ std::numeric_limits<std::int32_t>::max() } : -1;
}

/// What one heap block of `bytes` bytes adds to them in private bytes (0 for none), measured 2026-11-01 on Windows' heap (the
/// UCRT's process heap, clang 21; malloc of 1 byte to 100 MB, many blocks of each size): up to 16,384 bytes its
/// bucket's rounding and its header, at most an eighth and 18 bytes (10.04 % at 10,000 bytes and 17.6 bytes at 1, the
/// most measured); to 262,128 its place in the heap's segments, at most a 64th and 64 bytes (1/70 at 255,000 the most);
/// above, a block of its own pages: their page tables (8 bytes a 4,096-byte page) and at most 4,700 bytes more, its last
/// page with the heap's header (4,624 the most measured, 262,143 bytes to 100 MB).
std::uint64_t heap_block_added(std::uint64_t bytes)
{
  if (bytes == 0) return 0;
  return bytes <= 16384 ? bytes / 8 + 18 : bytes <= 262128 ? bytes / 64 + 64 : bytes / 512 + 4700;
}

/// A heap block of `bytes` a file declares (heap_block_added), at most INT64_MAX.
std::int64_t heap_block_bytes(std::int64_t bytes)
{
  return bytes <= 0 ? 0 : sum(bytes, static_cast<std::int64_t>(heap_block_added(static_cast<std::uint64_t>(bytes))));
}

/// What the codecs allocate to decode one buffer beside its decoded bytes, each call freeing it before the next (read
/// in lz4 1.10.0's and zstd 1.5.7's sources, measured 2026-11-01), each a heap block: zstd's decompression context,
/// which ZSTD_decompress makes and frees a call (ZSTD_HEAPMODE, 1 by default; ZSTD_estimateDCtxSize()), and LZ4F's
/// context (lz4frame.c's LZ4F_dctx_s, 288 bytes on a 64-bit target).
constexpr std::int64_t zstd_context_bytes = 95968, lz4_context_bytes = 288;

/// The scratch nanoarrow's codec holds while it decodes a buffer whose frame begins with `frame` (its bytes after the
/// 8-byte declared length), as heap blocks: zstd's context; LZ4F's context and, for a frame descriptor
/// LZ4F_decodeHeader takes (the magic, version 01, the reserved bits clear, a block size ID of 4 to 7), its two block
/// buffers (lz4frame.c, dstage_init): in, the frame's largest block (BD: 64 KB, 256 KB, 1 MB or 4 MB) and its 4-byte
/// checksum; out, the block and 128 KB more for linked blocks (FLG's block independence clear).
std::int64_t codec_bytes(ArrowIpcCompressionType codec, std::span<const std::byte> frame)
{
  if (codec == NANOARROW_IPC_COMPRESSION_TYPE_ZSTD) return heap_block_bytes(zstd_context_bytes);
  const auto at = [&](std::size_t i) { return std::to_integer<unsigned>(frame[i]); };
  if (frame.size() < 6 || at(0) != 0x04 || at(1) != 0x22 || at(2) != 0x4D || at(3) != 0x18 || (at(4) >> 6) != 1
      || (at(4) & 0x02) != 0 || (at(5) & 0x8F) != 0 || ((at(5) >> 4) & 7) < 4)
    return heap_block_bytes(lz4_context_bytes);
  const std::int64_t block = std::int64_t{ 1 } << (8 + 2 * ((at(5) >> 4) & 7));
  const bool linked = (at(4) & 0x20) == 0;
  return heap_block_bytes(lz4_context_bytes) + heap_block_bytes(block + 4)
         + heap_block_bytes(block + (linked ? 128 * 1024 : 0));
}

/// What decoding a column's buffers allocates: their decoded bytes, each a heap block (heap_block_bytes), and the most
/// the codec holds to decode one of them; and of a string column, the bytes of its characters.
struct Declared
{
  std::int64_t bytes = 0, scratch = 0, characters = 0;
};

/// What nanoarrow allocates to decode the buffers of `view` and its children, each proven inside the body `body`, added
/// to `out`, each buffer a heap block (heap_block_bytes): a compressed buffer's declared decoded length (its first 8
/// bytes, little-endian, as nanoarrow reads them; -1: stored as it is, and copied), refused beyond Arrow's 64-byte
/// padding of the bytes its lengths allow (needed_bytes: Arrow C++'s writer keeps a sliced array's buffer whole up to
/// that padding, so a batch of 32 rows of a 256-row column declares its 32-byte validity bitmap), and the codec's
/// scratch for its frame (codec_bytes, from the frame's descriptor after the length); an uncompressed buffer's own
/// size, which the file holds. At most INT64_MAX.
void declared_bytes(const ArrowArrayView *view, ArrowIpcCompressionType codec, const Body &body,
                    const std::string &where, Declared &out, std::vector<std::byte> &scratch)
{
  if (view->length < 0)
    throw IOError(where + ": a length of " + std::to_string(view->length) + " is negative: the file is corrupt");
  constexpr std::int64_t most = std::numeric_limits<std::int64_t>::max();
  const bool compressed = codec != NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  const bool strings = view->storage_type == NANOARROW_TYPE_STRING || view->storage_type == NANOARROW_TYPE_LARGE_STRING;
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; ++i) {
    const ArrowBufferView &buffer = view->buffer_views[i];
    const bool characters = strings && view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA;
    if (!compressed || buffer.size_bytes < 8) { // below 8 bytes nanoarrow refuses it, allocating nothing
      out.bytes = sum(out.bytes, compressed ? 0 : heap_block_bytes(buffer.size_bytes));
      if (characters && !compressed) out.characters = buffer.size_bytes;
      continue;
    }
    // The declared length, then the frame's descriptor (LZ4: its magic, FLG and BD bytes).
    const auto head = body.head(buffer, 8 + 6, scratch);
    std::uint64_t word = 0;
    for (std::size_t b = 8; b-- > 0;) word = (word << 8) | std::to_integer<std::uint64_t>(head[b]);
    const auto declared = static_cast<std::int64_t>(word);
    if (declared < -1) continue; // nanoarrow refuses it, allocating nothing
    if (declared == -1) {
      out.bytes = sum(out.bytes, heap_block_bytes(buffer.size_bytes - 8));
      if (characters) out.characters = buffer.size_bytes - 8;
      continue;
    }
    if (const std::int64_t need = needed_bytes(view, i);
        need >= 0 && declared > (need > most - 63 ? most : (need + 63) / 64 * 64))
      throw IOError(where + ": a buffer declares " + std::to_string(declared) + " decoded bytes, more than Arrow's "
                    "64-byte padding beyond the " + std::to_string(need) + " bytes its lengths allow: refused before "
                    "any decode");
    out.bytes = sum(out.bytes, heap_block_bytes(declared));
    if (characters) out.characters = declared;
    out.scratch = std::max(out.scratch, codec_bytes(codec, head.subspan(8)));
  }
  for (std::int64_t i = 0; i < view->n_children; ++i)
    declared_bytes(view->children[i], codec, body, where, out, scratch);
}

/// What the walk learns of a record batch it cannot view: its rows (its root's length, unbounded: its columns bound it
/// once decoded), the bytes decoding its used columns will allocate and the most the codec holds decoding one of their
/// buffers, the values of its 'data' column, and its 'name' column's decoded bytes, its characters' bytes and the
/// body's bytes its buffers span (0 without one), each from its header's lengths and its buffers' declared sizes, and
/// whether it holds a validity buffer (without one no name is null).
struct Proof
{
  std::int64_t rows = 0, declared = 0, scratch = 0, values = 0, names = 0, characters = 0, name_span = 0;
  bool name_nulls = true;
};

/// The record batch `decoder` last decoded, which cannot be viewed (compressed, in the other byte order, or a used
/// buffer not aligned), its body `body`: each buffer of its used columns proven before nanoarrow reads it, and the
/// bytes decoding them will allocate (Proof).
///
/// nanoarrow trusts the file's buffer offsets and lengths, and its DecodeArray reads each buffer of the column it
/// decodes (decompressing, swapping or copying it) at the offset the file gives, having allocated the size the buffer
/// declares, before any check of ours could run. A pass with the codec set to NONE and no byte order makes each buffer
/// a view where the file puts it and reads none: DecodeArrayView of a used column's field index, the walk its
/// DecodeArray makes, so it reads exactly the buffers proven here (a pass over the whole batch can differ: it skips a
/// pre-1.0 union's extra buffer, which a one-column walk does not). Each used column's views must lie in the body
/// (bounds only: a compressed body can be smaller than its values) and, to be swapped where it lies, be aligned to its
/// elements; then the declared sizes are read from inside the body (declared_bytes), from the file when it is not in
/// memory (unread_body).
Proof prove_batch(Decoder &decoder, const Body &body, const Used &used, bool swapped, const std::string &step)
{
  ArrowError error{};
  const auto codec = decoder.ipc.codec;
  decoder.ipc.codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  // The batch's rows, its root's length, from a pass over every column; then each used column from the pass its
  // DecodeArray makes, the same walk from the same field index, so the buffers proven are the buffers it reads.
  ArrowArrayView *root = nullptr;
  std::vector<ArrowArrayView *> views(used.size());
  ArrowErrorCode code = ArrowIpcDecoderDecodeArrayView(&decoder.ipc, body.view, -1, &root, &error);
  for (std::size_t c = 0; c < used.size() && code == NANOARROW_OK; ++c)
    code = ArrowIpcDecoderDecodeArrayView(&decoder.ipc, body.view, used[c].field, &views[c], &error);
  decoder.ipc.codec = codec;
  check(code, error, step);
  Proof proof;
  std::vector<std::byte> scratch;
  for (std::size_t c = 0; c < used.size(); ++c) {
    const std::string where = step + ", column '" + used[c].name + "'";
    check_in_body(views[c], body.view, where, false);
    if (swapped && !aligned(views[c]))
      throw IOError(step + ": a buffer is not aligned to its elements, so it cannot be byte-swapped (Arrow IPC "
                           "writers align every buffer to 8 bytes: rewrite the file with one)");
    Declared declared;
    declared_bytes(views[c], codec, body, where, declared, scratch);
    proof.declared = sum(proof.declared, declared.bytes);
    proof.scratch = std::max(proof.scratch, declared.scratch);
    if (std::string_view(used[c].name) == "data") {
      proof.values = views[c]->n_children > 0 ? views[c]->children[0]->length : 0;
    } else {
      proof.names = declared.bytes;
      proof.characters = declared.characters;
      proof.name_nulls = views[c]->buffer_views[0].size_bytes != 0;
      const auto [first, last] = span_of(views[c], body.view);
      proof.name_span = last - first;
    }
  }
  proof.rows = root->length;
  return proof;
}

/// A record batch that cannot be viewed, decoded: the arrays of its used columns, whose buffers hold the values and
/// names the series view, and a view of the batch through them.
struct Decoded
{
  std::vector<Array> arrays;
  View batch;
};

/// What a read of a record batch must find of what the walk proved of it: its used columns' declared bytes, and at most
/// its codec's scratch.
struct Expect
{
  std::int64_t declared = 0, scratch = 0;
};

/// Record batch `k`, which the walk found at `message` and which cannot be viewed, decoded into `out` by a decoder of
/// its own (a decoder holds the batch it last decoded, so none is shared between threads): each column of `used` into
/// an array nanoarrow allocates, decompressed, in this machine's byte order and aligned. `header` holds the message's
/// header (or the whole message), `body` its body's bytes, of every buffer of `used` at least. Its buffers are proven
/// first (prove_batch), and must be what the walk found (`walked`, if it walked the file), which the whole file's were
/// checked against. Decoded, each length is bounded by the column's decoded bytes (nanoarrow's validation sizes the
/// buffers from the lengths, in a product that wraps at 2^57), the column must be as long as the batch, and it is
/// validated at DEFAULT.
void decode_batch(const ArrowSchema &schema, ArrowIpcEndianness endianness, std::span<const std::byte> header,
                  const Body &body, const Message &message, const Used &used, std::size_t k,
                  std::optional<Expect> walked, Decoded &out)
{
  const std::string step = "Arrow IPC record batch " + std::to_string(k);
  ArrowError error{};
  Decoder decoder;
  // SetSchema reads the schema, as every decoder does at once.
  check(ArrowIpcDecoderSetSchema(&decoder.ipc, const_cast<ArrowSchema *>(&schema), &error), error, step);
  redecode(decoder, header, message, step); // the walk verified it: decode it again
  const Proof proof = prove_batch(decoder, body, used, endianness != ArrowIpcSystemEndianness(), step);
  if (walked && (proof.declared != walked->declared || proof.scratch > walked->scratch))
    throw IOError(step + " declares other buffers than when the file was walked: the file changed while it was read");
  check(ArrowIpcDecoderSetEndianness(&decoder.ipc, endianness), error, step);
  check(ArrowArrayViewInitFromSchema(&out.batch.arrow, &schema, &error), error, step);
  out.batch.arrow.length = proof.rows;
  out.arrays.reserve(used.size());
  for (const Use &use : used) {
    const std::string where = step + ", column '" + use.name + "'";
    Array &array = out.arrays.emplace_back();
    check(ArrowIpcDecoderDecodeArray(&decoder.ipc, body.view, use.field, &array.arrow, NANOARROW_VALIDATION_LEVEL_NONE,
                                     &error),
          error, where);
    if (array.arrow.release == nullptr) // nanoarrow returns OK with a released array when it runs out of memory
      throw IOError(where + ": nanoarrow decoded no array"
                    + (error.message[0] != '\0' ? ": " + std::string(error.message) : std::string()));
    ArrowArrayView *view = out.batch.arrow.children[use.column];
    view_decoded(view, &array.arrow);
    check_lengths(view, decoded_bytes(view), where);
    check_rows(view, proof.rows, where);
    check(ArrowArrayViewValidate(view, NANOARROW_VALIDATION_LEVEL_DEFAULT, &error), error, where);
  }
}

/// A record batch of the walk: its message, whether it is viewed where it lies (else decoded), its rows, the bytes
/// decoding it will allocate (0 when viewed) and the codec's scratch for its largest frame, the values of its 'data'
/// column, and of its 'name' column whether a name may be null (it holds a validity buffer), the bytes (its buffers',
/// or decoded), the body's bytes its buffers span, and the most bytes a name takes: its longest name where the walk
/// read its offsets (a batch viewed in memory), else its characters' bytes (-1 without the column).
struct Batch
{
  Message message;
  bool viewed = false, name_nulls = true;
  std::int64_t rows = 0, declared = 0, scratch = 0, values = 0, names = 0, name_span = 0, longest_name = -1;
};

/// The longest row `column` holds, a list or string column viewed in memory and validated (its offsets inside their
/// buffer), from its offsets, at most `bound` (its values, or its characters' bytes): a row whose offsets run
/// backwards or past them is refused when it is read.
std::int64_t longest_span(const ArrowArrayView *column, std::int64_t bound)
{
  const bool large =
    column->storage_type == NANOARROW_TYPE_LARGE_STRING || column->storage_type == NANOARROW_TYPE_LARGE_LIST;
  const ArrowBufferView &offsets = column->buffer_views[1];
  std::int64_t longest = 0;
  for (std::int64_t i = column->offset; i < column->offset + column->length; ++i)
    longest = std::max(longest, large ? offsets.data.as_int64[i + 1] - offsets.data.as_int64[i]
                                      : std::int64_t{ offsets.data.as_int32[i + 1] } - offsets.data.as_int32[i]);
  return std::min(longest, bound);
}

/// Whether `a` and `b` give every field the same type, name, nullability and metadata.
bool same_schema(const ArrowSchema *a, const ArrowSchema *b)
{
  const auto text = [](const char *chars) { return std::string_view(chars != nullptr ? chars : ""); };
  const auto metadata = [](const char *chars) {
    return chars != nullptr ? std::string_view(chars, static_cast<std::size_t>(ArrowMetadataSizeOf(chars)))
                            : std::string_view();
  };
  if (text(a->format) != text(b->format) || text(a->name) != text(b->name) || a->flags != b->flags
      || metadata(a->metadata) != metadata(b->metadata) || a->n_children != b->n_children)
    return false;
  for (std::int64_t i = 0; i < a->n_children; ++i)
    if (!same_schema(a->children[i], b->children[i])) return false;
  return true;
}

/// What the walk learns of the messages before it holds any fact of them (count_batches).
struct Count
{
  std::size_t batches = 0; ///< the record batches up to the end, or up to the first message the walk refuses
  std::int64_t end = 0;    ///< the byte after the last of them
  std::string footer;      ///< why the file format's footer is not those messages', or empty
};

/// Why the footer's block `block` is not record batch `k`'s message `message`, or empty.
std::string block_mismatch(const ArrowIpcFileBlock &block, const Message &message, std::size_t k)
{
  if (block.offset != message.at)
    return "the Arrow IPC footer puts record batch " + std::to_string(k) + " at byte " + std::to_string(block.offset)
           + ", and the file's messages put it at byte " + std::to_string(message.at) + ": the file is corrupt";
  if (block.metadata_length != message.header || block.body_length != message.body)
    return "the Arrow IPC footer gives record batch " + std::to_string(k) + " a header of "
           + std::to_string(block.metadata_length) + " bytes and a body of " + std::to_string(block.body_length)
           + ", and the file's messages give it " + std::to_string(message.header) + " and "
           + std::to_string(message.body) + ": the file is corrupt";
  return {};
}

/// The record batch messages from byte `at` to `last`, counted before the walk holds anything of them: each header
/// read, verified and decoded (next_message, by `decoder`) and kept nowhere, so that the walk holds its facts in one
/// allocation of their size (grown a batch at a time, a vector held up to 2.5 times them while it grew: its old block
/// beside the new, MSVC's growth by half). With a footer (the file format's, from byte `last` on), the footer is checked
/// here, before the facts are held, beside which its decoded blocks (24 bytes each, nanoarrow's) would lie: verified
/// (flatcc) and decoded, its byte order and schema the stream's (`endianness`, `schema`), and its record batch blocks
/// those messages, each at its message's byte with its header's and body's lengths. A reader that seeks by the footer
/// (pyarrow) reads what the footer says: the values byte-swapped under another byte order, other values from the same
/// bytes under another schema, or the rows in another order under blocks in another order. A message the walk refuses
/// ends the count, which it then refuses in turn, after what it finds before it; the footer's refusal (Count::footer)
/// waits for the walk's end, as when the footer was checked after it.
Count count_batches(Decoder &decoder, const Source &source, std::int64_t at, std::int64_t last, bool footer,
                    const ArrowSchema &schema, ArrowIpcEndianness endianness)
{
  Count count;
  Decoder reader; // the footer's: decoding a footer sets its decoder's byte order and schema
  const ArrowIpcFileBlock *block = nullptr;
  std::size_t blocks = 0;
  if (footer) {
    try {
      ArrowError error{};
      std::vector<std::byte> buffer; // the footer, its length and ARROW1, read once (without llfio), freed decoded
      const ArrowBufferView bytes = view_of(source.range(last, source.size(), buffer));
      check(ArrowIpcDecoderVerifyFooter(&reader.ipc, bytes, &error), error, "Arrow IPC footer");
      check(ArrowIpcDecoderDecodeFooter(&reader.ipc, bytes, &error), error, "Arrow IPC footer");
    } catch (const IOError &refused) {
      count.footer = refused.what();
    }
    if (count.footer.empty() && reader.ipc.endianness != endianness)
      count.footer = "the Arrow IPC footer's byte order is not the stream's, so a reader that follows the footer swaps "
                     "every value: the file is corrupt";
    if (count.footer.empty() && !same_schema(&reader.ipc.footer->schema, &schema))
      count.footer = "the Arrow IPC footer's schema is not the stream's, so a reader that follows the footer reads other "
                     "columns: the file is corrupt";
    if (count.footer.empty()) {
      const ArrowBuffer &decoded = reader.ipc.footer->record_batch_blocks;
      block = reinterpret_cast<const ArrowIpcFileBlock *>(decoded.data);
      blocks = static_cast<std::size_t>(decoded.size_bytes) / sizeof(ArrowIpcFileBlock);
    }
  }
  std::string mismatch; // the first block that is not its message's
  std::vector<std::byte> buffer;
  std::span<const std::byte> window;
  try {
    for (auto message = next_message(decoder, source, at, last, buffer, window); message;
         message = next_message(decoder, source, at, last, buffer, window)) {
      if (decoder.ipc.message_type != NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH) break; // the walk refuses it
      const std::size_t k = count.batches++;
      if (k < blocks && mismatch.empty()) mismatch = block_mismatch(block[k], *message, k);
      at = message->end();
    }
  } catch (const IOError &) {
    // The walk refuses this message, after the batches before it.
  }
  count.end = at;
  if (block != nullptr)
    count.footer = blocks != count.batches
                     ? "the Arrow IPC footer lists " + std::to_string(blocks) + " record batches, and the file's "
                         "messages hold " + std::to_string(count.batches) + ": the file is corrupt"
                     : mismatch;
  return count;
}

ArrowType type_of(const ArrowSchema *schema)
{
  ArrowSchemaView view;
  ArrowError error{};
  check(ArrowSchemaViewInit(&view, schema, &error), error, "Arrow IPC schema");
  return view.type;
}

/// Whether `schema` or a field below it is a union.
bool has_union(const ArrowSchema *schema)
{
  if (const ArrowType type = type_of(schema); type == NANOARROW_TYPE_SPARSE_UNION || type == NANOARROW_TYPE_DENSE_UNION)
    return true;
  for (std::int64_t i = 0; i < schema->n_children; ++i)
    if (has_union(schema->children[i])) return true;
  return false;
}

std::string type_text(const ArrowSchema *schema)
{
  char text[256];
  ArrowSchemaToString(schema, text, sizeof text, 1);
  return text;
}

Columns columns_of(const ArrowSchema &schema)
{
  Columns columns;
  // Features per time step: a whole positive integer, nothing else ("2x" is not 2, "-1" does not wrap, and 0 would
  // divide every series length by zero).
  ArrowStringView ndim = ArrowCharView(nullptr);
  ArrowError error{};
  check(ArrowMetadataGetValue(schema.metadata, ArrowCharView("ndim"), &ndim), error, "Arrow IPC schema metadata");
  if (ndim.data != nullptr) {
    const std::string_view text(ndim.data, static_cast<std::size_t>(ndim.size_bytes));
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), columns.ndim);
    if (ec != std::errc{} || end != text.data() + text.size() || columns.ndim == 0)
      throw IOError("schema metadata 'ndim' must be a positive integer, got '" + std::string(text)
                    + "'. Set it to the number of features per timestep (1 for univariate data).");
  }
  for (std::int64_t i = 0; i < schema.n_children; ++i) {
    const std::string_view field = schema.children[i]->name != nullptr ? schema.children[i]->name : "";
    if (field == "data" && columns.data < 0) columns.data = i;
    if (field == "name" && columns.name < 0) columns.name = i;
  }
  if (columns.data < 0)
    throw IOError("no 'data' column; write the series as a List or LargeList of Float32/Float64 named 'data'.");
  const ArrowSchema *data = schema.children[columns.data];
  const ArrowType list = type_of(data);
  const ArrowType value = (list == NANOARROW_TYPE_LIST || list == NANOARROW_TYPE_LARGE_LIST) && data->n_children == 1
                            ? type_of(data->children[0])
                            : NANOARROW_TYPE_NA;
  if (value != NANOARROW_TYPE_FLOAT && value != NANOARROW_TYPE_DOUBLE)
    throw IOError("the 'data' column must be a List or LargeList of Float32/Float64, got " + type_text(data) + ".");
  columns.f32 = value == NANOARROW_TYPE_FLOAT;
  if (columns.name >= 0) {
    const ArrowSchema *name = schema.children[columns.name];
    if (const ArrowType type = type_of(name); type != NANOARROW_TYPE_STRING && type != NANOARROW_TYPE_LARGE_STRING)
      throw IOError("the 'name' column must be Utf8 or LargeUtf8, got " + type_text(name)
                    + ". Write the names as strings, or drop the column.");
  }
  return columns;
}

/// The schema message at byte `first`, decoded into `schema` and set on `decoder`: the byte order of the file's buffers
/// (nanoarrow swaps a batch's buffers only after SetEndianness, as its own reader calls it), and the byte after it.
std::pair<ArrowIpcEndianness, std::int64_t> read_schema(Decoder &decoder, const Source &source, std::int64_t first,
                                                        std::int64_t last, Schema &schema)
{
  std::vector<std::byte> buffer;
  std::span<const std::byte> window;
  const auto head = next_message(decoder, source, first, last, buffer, window);
  if (!head || decoder.ipc.message_type != NANOARROW_IPC_MESSAGE_TYPE_SCHEMA)
    throw IOError("an Arrow IPC stream begins with its schema, and this one does not");
  const ArrowIpcEndianness endianness = decoder.ipc.endianness;
  ArrowError error{};
  check(ArrowIpcDecoderDecodeSchema(&decoder.ipc, &schema.arrow, &error), error, "Arrow IPC schema");
  check(ArrowIpcDecoderSetSchema(&decoder.ipc, &schema.arrow, &error), error, "Arrow IPC schema");
  return { endianness, head->end() };
}

/// An Arrow IPC file walked (walk()): its schema, the columns the series come from, the byte order of its buffers,
/// every record batch, the bytes [first, last) its messages lie in, and the most values a series holds, from the
/// offsets of every batch where each is viewed in memory (-1 where one is viewed otherwise or decoded). No batch is
/// decoded. `decoded_for` says why the first batch that is decoded cannot be viewed (IpcFile::storage names it);
/// null if every batch is viewed.
struct Walk
{
  Schema schema;
  Columns columns;
  Used used;
  ArrowIpcEndianness endianness = NANOARROW_IPC_ENDIANNESS_UNINITIALIZED;
  std::vector<Batch> batches;
  std::int64_t first = 0, last = 0, longest_values = 0;
  const char *decoded_for = nullptr;
};

/// The file `source` holds walked into `out`: its record batches counted and the file format's footer checked against
/// them (count_batches), then each record batch's message, viewed where it lies (uncompressed, in this machine's byte
/// order, its used buffers aligned) with its rows, or else proven (prove_batch) to be decoded later; the end-of-stream
/// marker's place checked. A file not in memory (read without llfio) is read a header at a time (twice: counted, then
/// walked) and, of a batch to be decoded, each used buffer's declared length and frame descriptor: no body is read.
void walk(const Source &source, Walk &out)
{
  std::tie(out.first, out.last) = message_bytes(source);
  Decoder decoder; // the walk's, which views batches in this machine's byte order only
  std::int64_t at = 0;
  std::tie(out.endianness, at) = read_schema(decoder, source, out.first, out.last, out.schema);
  out.columns = columns_of(out.schema.arrow);
  out.used = used_of(out.schema.arrow, out.columns);
  const bool unions = has_union(&out.schema.arrow);
  const Count count = count_batches(decoder, source, at, out.last, out.first != 0, out.schema.arrow, out.endianness);
  out.batches.reserve(count.batches);
  std::vector<std::byte> buffer;
  std::span<const std::byte> window;
  for (auto message = next_message(decoder, source, at, out.last, buffer, window); message;
       message = next_message(decoder, source, at, out.last, buffer, window)) {
    const std::size_t k = out.batches.size();
    if (decoder.ipc.message_type != NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH)
      throw IOError("Arrow IPC message at byte " + std::to_string(message->at)
                    + " is a second schema; a file holds one, before its record batches");
    const bool plain =
      decoder.ipc.codec == NANOARROW_IPC_COMPRESSION_TYPE_NONE && out.endianness == ArrowIpcSystemEndianness();
    // A pre-1.0 union carries a validity buffer nanoarrow skips over a whole batch, and not in the walk of one column.
    if (unions && decoder.ipc.metadata_version < NANOARROW_IPC_METADATA_VERSION_V5)
      throw IOError("Arrow IPC record batch " + std::to_string(k) + " is in metadata version V4 and its schema holds a "
                    "union, whose pre-1.0 layout misplaces the buffers of later columns in nanoarrow's decode of one "
                    "column: rewrite the file with Arrow 1.0 or later");
    const bool memory = source.in_memory();
    const Body body = memory ? body_of(window, *message) : unread_body(source, *message);
    // A file not in memory is read here a header at a time (its body unread): the batch read validates a viewed batch.
    const ArrowArrayView *view = plain ? view_batch(decoder, body.view, out.used, k, memory) : nullptr;
    Batch batch{ *message, view != nullptr };
    if (view != nullptr) {
      batch.rows = view->length;
      const ArrowArrayView *series = view->children[out.columns.data];
      batch.values = series->children[0]->length;
      if (out.longest_values >= 0)
        out.longest_values = memory ? std::max(out.longest_values, longest_span(series, batch.values)) : -1;
      if (out.columns.name >= 0) {
        const ArrowArrayView *names = view->children[out.columns.name];
        for (const ArrowBufferView &buffer : names->buffer_views) batch.names += buffer.size_bytes;
        const auto [first, last] = span_of(names, body.view);
        batch.name_span = last - first;
        batch.longest_name =
          memory ? longest_span(names, names->buffer_views[2].size_bytes) : names->buffer_views[2].size_bytes;
        batch.name_nulls = names->buffer_views[0].size_bytes != 0;
      }
    } else {
      if (out.decoded_for == nullptr)
        out.decoded_for = decoder.ipc.codec == NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME ? "lz4"
                          : decoder.ipc.codec == NANOARROW_IPC_COMPRESSION_TYPE_ZSTD    ? "zstd"
                          : out.endianness != ArrowIpcSystemEndianness()                ? "big-endian"
                                                                                        : "misaligned";
      const Proof proof = prove_batch(decoder, body, out.used, out.endianness != ArrowIpcSystemEndianness(),
                                      "Arrow IPC record batch " + std::to_string(k));
      out.longest_values = -1;
      batch.rows = proof.rows;
      batch.declared = proof.declared;
      batch.scratch = proof.scratch;
      batch.values = proof.values;
      batch.names = proof.names;
      batch.name_span = proof.name_span;
      if (out.columns.name >= 0) {
        batch.longest_name = proof.characters;
        batch.name_nulls = proof.name_nulls;
      }
    }
    out.batches.push_back(batch);
    at = message->end();
  }
  if (out.batches.size() != count.batches || at != count.end)
    throw IOError("the Arrow IPC file's messages are not those it was counted with: the file changed while it was "
                  "read");
  // The file format's end-of-stream marker, if it has one, is its last message (just before the footer): a marker
  // before a record batch would drop it unread. The stream format's messages begin at byte 0, the file format's at 8.
  if (out.first != 0 && at != out.last && at != out.last - 8)
    throw IOError("Arrow IPC message at byte " + std::to_string(at) + " is an end-of-stream marker, but the file's "
                  "messages run on to its footer at byte " + std::to_string(out.last) + ": the file is corrupt");
  if (!count.footer.empty()) throw IOError(count.footer);
}

/// Name `label`'s row r (the batch's row `row`, for the error) gives a series, a view into the names; none when null.
std::optional<std::string_view> row_name(const ArrowArrayView *label, std::int64_t r, std::int64_t row)
{
  if (label == nullptr || (label->null_count != 0 && ArrowArrayViewIsNull(label, r))) return std::nullopt;
  const std::int64_t i = label->offset + r;
  const auto [start, end] =
    label->storage_type == NANOARROW_TYPE_LARGE_STRING
      ? std::pair<std::int64_t, std::int64_t>(label->buffer_views[1].data.as_int64[i],
                                              label->buffer_views[1].data.as_int64[i + 1])
      : std::pair<std::int64_t, std::int64_t>(label->buffer_views[1].data.as_int32[i],
                                              label->buffer_views[1].data.as_int32[i + 1]);
  const std::int64_t chars = label->buffer_views[2].size_bytes;
  if (start < 0 || end < start || end > chars)
    throw IOError("data_from_arrow: name offset [" + std::to_string(start) + ", " + std::to_string(end)
                  + ") of the series at row " + std::to_string(row) + " is outside the names [0, "
                  + std::to_string(chars) + ").");
  return std::string_view(label->buffer_views[2].data.as_char + start, static_cast<std::size_t>(end - start));
}

/// The rows of a record batch, the file's rows from `first` on: a span of each series' values and a view of its name
/// where they lie (the body, or a decoded column), today's verdicts on a null and on offsets outside their values row
/// by row; a row named by no 'name' value is marked in `unnamed`, at its row less `marks`.
template <typename T>
void view_rows(const ArrowArrayView *batch, const Columns &columns, std::int64_t first,
               std::vector<std::span<const T>> &series, std::vector<std::string_view> &names,
               std::vector<bool> &unnamed, std::int64_t marks)
{
  const ArrowArrayView *list = batch->children[columns.data];
  const ArrowArrayView *values = list->children[0];
  const ArrowArrayView *label = columns.name >= 0 ? batch->children[columns.name] : nullptr;
  const T *data = reinterpret_cast<const T *>(values->buffer_views[1].data.data);
  for (std::int64_t r = 0; r < batch->length; ++r) {
    const std::int64_t row = first + r;
    if (list->null_count != 0 && ArrowArrayViewIsNull(list, r))
      throw InvalidInput("data_from_arrow: null series at row " + std::to_string(row)
                         + " (drop or fill nulls before clustering).");
    // A slice's offsets start at its offset, which ArrowArrayViewListChildOffset, unlike the other accessors, does
    // not add. Validation read only the first and the last.
    const std::int64_t start = ArrowArrayViewListChildOffset(list, list->offset + r);
    const std::int64_t end = ArrowArrayViewListChildOffset(list, list->offset + r + 1);
    if (start < 0 || end < start || end > values->length)
      throw IOError("data_from_arrow: list offset [" + std::to_string(start) + ", " + std::to_string(end)
                    + ") of the series at row " + std::to_string(row) + " is outside the values [0, "
                    + std::to_string(values->length) + ").");
    if (values->null_count != 0)
      for (std::int64_t j = start; j < end; ++j)
        if (ArrowArrayViewIsNull(values, j))
          throw InvalidInput("data_from_arrow: null value inside series at row " + std::to_string(row)
                             + " (drop or fill nulls before clustering).");
    series.emplace_back(data + values->offset + start, static_cast<std::size_t>(end - start));
    const auto name = row_name(label, r, row);
    names.push_back(name.value_or(std::string_view()));
    if (!name) unnamed[static_cast<std::size_t>(row - marks)] = true;
  }
}

/// The names series_<first + i> of the rows `unnamed` marks, made in one buffer sized first (views into a growing one
/// would dangle) and viewed by names[i]: the buffer, which the views' owner keeps.
std::vector<char> name_unnamed(std::vector<std::string_view> &names, const std::vector<bool> &unnamed,
                               std::int64_t first)
{
  char digits[24];
  std::size_t chars = 0;
  for (std::size_t i = 0; i < unnamed.size(); ++i)
    if (unnamed[i])
      chars += 7 + static_cast<std::size_t>(
                     std::to_chars(digits, digits + sizeof digits, first + static_cast<std::int64_t>(i)).ptr - digits);
  std::vector<char> made(chars);
  char *out = made.data();
  for (std::size_t i = 0; i < unnamed.size(); ++i)
    if (unnamed[i]) {
      char *const name = out;
      out = std::to_chars(std::copy_n("series_", 7, out), made.data() + chars, first + static_cast<std::int64_t>(i))
              .ptr;
      names[i] = std::string_view(name, static_cast<std::size_t>(out - name));
    }
  return made;
}

/// The view Data of the series in `batches`, each viewed where it lies or, if `decoded`, in its decoded columns: their
/// spans and name views, and the names series_<row> of the rows named by nothing. The owner keeps the bytes only if a
/// batch is viewed in them; the Data's values are mapped (Data::is_mapped) if `mapped`, the bytes a file's mapping, and
/// every batch is viewed.
template <typename T>
Data view_batches(Decoder &decoder, std::span<const std::byte> bytes, const std::vector<Batch> &batches,
                  std::vector<Decoded> &decoded, const Columns &columns, const Used &used,
                  std::shared_ptr<const void> keep, bool mapped)
{
  std::int64_t rows = 0; // each batch's rows are bounded by its body or its decoded bytes
  for (const Batch &batch : batches) rows += batch.rows;
  const auto n = static_cast<std::size_t>(rows);
  std::vector<std::span<const T>> series;
  std::vector<std::string_view> names;
  std::vector<bool> unnamed(n);
  series.reserve(n);
  names.reserve(n);
  bool viewed = false, decoded_any = false;
  for (std::size_t k = 0; k < batches.size(); ++k) {
    const ArrowArrayView *batch = &decoded[k].batch.arrow;
    if (batches[k].viewed) {
      const Message &message = batches[k].message;
      const auto own = bytes.subspan(static_cast<std::size_t>(message.at),
                                     static_cast<std::size_t>(message.header + message.body));
      redecode(decoder, own, message, "Arrow IPC record batch " + std::to_string(k)); // the walk verified it
      batch = view_batch(decoder, body_of(own, message).view, used, k);
      if (batch == nullptr)
        throw IOError("Arrow IPC record batch " + std::to_string(k)
                      + " can no longer be viewed where it lies: the file changed while it was read");
      viewed = true;
    } else {
      decoded_any = true;
    }
    if (series.size() + static_cast<std::size_t>(batch->length) > n)
      throw IOError("Arrow IPC record batch " + std::to_string(k)
                    + " holds more rows than when the file was walked: the file changed while it was read");
    view_rows(batch, columns, static_cast<std::int64_t>(series.size()), series, names, unnamed, 0);
  }
  if (series.size() != n)
    throw IOError("the Arrow IPC file's record batches hold " + std::to_string(series.size()) + " rows, and its walk "
                  + "counted " + std::to_string(n) + ": the file changed while it was read");
  std::vector<char> made = name_unnamed(names, unnamed, 0);
  std::size_t count = 0; // one allocation of the arrays, not one grown an array at a time
  for (const Decoded &batch : decoded) count += batch.arrays.size();
  std::vector<Array> arrays;
  arrays.reserve(count);
  for (Decoded &batch : decoded)
    for (Array &array : batch.arrays) arrays.push_back(std::move(array)); // the structs move, their buffers stay
  auto owner = std::make_shared<Owner>(
    Owner{ viewed ? std::move(keep) : nullptr, std::move(arrays), std::move(made), nullptr }); // the chars stay too
  return Data(std::move(series), std::move(names), columns.ndim, std::move(owner), mapped && !decoded_any);
}

#ifdef DTWC_HAS_MMAP
/// The file at `path` mapped (llfio): its bytes, and the mapping, which keeps them. A map the system refuses memory for
/// (a cap on address space) is std::bad_alloc.
std::pair<std::shared_ptr<const void>, std::span<const std::byte>> map_file(const std::filesystem::path &path)
{
  namespace llfio = LLFIO_V2_NAMESPACE;
  const auto value = [](auto &&result, const char *step) {
    if (!result && result.error() == std::errc::not_enough_memory) throw std::bad_alloc();
    if (!result) throw IOError(std::string(step) + ": " + result.error().message());
    return std::forward<decltype(result)>(result).value();
  };
  auto file = std::make_shared<llfio::mapped_file_handle>(value(
    llfio::mapped_file_handle::mapped_file(0, {}, path, llfio::file_handle::mode::read,
                                           llfio::file_handle::creation::open_existing),
    "it could not be opened"));
  const auto length = static_cast<std::size_t>(value(file->maximum_extent(), "its length could not be read"));
  const std::span<const std::byte> bytes(length == 0 ? nullptr : reinterpret_cast<const std::byte *>(file->address()),
                                         length);
  return { std::move(file), bytes };
}
#endif

} // namespace

Data detail::read_ipc(std::span<const std::byte> bytes, std::shared_ptr<const void> owner, core::Precision dtype,
                      bool mapped)
{
  if (reinterpret_cast<std::uintptr_t>(bytes.data()) % 8 != 0) {
    // flatcc verifies a header's tables at their alignment in memory, which the file gives relative to its start: an
    // 8-aligned copy, which the owner holds, is read instead.
    auto copy = std::make_shared<const std::vector<std::byte>>(bytes.begin(), bytes.end());
    const std::span<const std::byte> aligned(*copy);
    return read_ipc(aligned, std::move(copy), dtype);
  }
  // The walk: each record batch's message, viewed where it lies with its rows, so the views are sized before any is
  // made, or else decoded below.
  const Source source(bytes);
  Walk walked;
  walk(source, walked);
  const std::vector<Batch> &batches = walked.batches;
  // What decoding will allocate, before any batch is decoded (each buffer a heap block): the decoded batches are held
  // together, so a file that declares more than the memory this process may use now cannot be decoded without
  // exhausting it. Where that memory is unknown (a build without llfio), no limit, as everywhere it is read.
  std::int64_t declared = 0;
  for (const Batch &batch : batches) declared = sum(declared, batch.declared);
  if (const std::size_t memory = declared > 0 ? core::available_memory() : 0;
      memory != 0 && static_cast<std::uint64_t>(declared) > memory)
    throw IOError("the Arrow IPC file's batches that cannot be viewed declare buffers that take "
                  + std::to_string(declared) + " bytes decoded, more than the " + std::to_string(memory)
                  + " bytes of memory this process may use now: free memory, or write the file uncompressed, so that "
                  + "it is viewed in place");

  // Every batch that cannot be viewed, decoded in parallel over disjoint batches; a failure is the lowest failing
  // batch's, as a serial decode raises it.
  std::vector<std::size_t> todo;
  for (std::size_t k = 0; k < batches.size(); ++k)
    if (!batches[k].viewed) todo.push_back(k);
  std::vector<Decoded> decoded(batches.size());
  const auto decode = [&](std::size_t j) {
    const std::size_t k = todo[j];
    const Message &message = batches[k].message;
    const auto own = bytes.subspan(static_cast<std::size_t>(message.at),
                                   static_cast<std::size_t>(message.end() - message.at));
    decode_batch(walked.schema.arrow, walked.endianness, own, body_of(own, message), message, walked.used, k,
                 Expect{ batches[k].declared, batches[k].scratch }, decoded[k]);
  };
  run_openmp(decode, todo.size(), todo.size() > 1);

  Decoder decoder; // views the batches the walk viewed
  ArrowError error{};
  check(ArrowIpcDecoderSetSchema(&decoder.ipc, &walked.schema.arrow, &error), error, "Arrow IPC schema");
  Data view = walked.columns.f32 ? view_batches<float>(decoder, bytes, batches, decoded, walked.columns, walked.used,
                                                       std::move(owner), mapped)
                                 : view_batches<data_t>(decoder, bytes, batches, decoded, walked.columns, walked.used,
                                                        std::move(owner), mapped);
  if (view.precision == dtype) return view;
  return dtwc::detail::owned_copy(view, dtype); // the request names other bytes: each value cast once
}

std::vector<std::vector<std::vector<std::byte>>> detail::decoded_buffers(std::span<const std::byte> bytes,
                                                                         std::string_view column)
{
  const Source source(bytes);
  const auto [first, last] = message_bytes(source);
  Schema schema;
  Decoder decoder;
  const auto [endianness, after_schema] = read_schema(decoder, source, first, last, schema);
  const std::string name(column);
  std::int64_t index = -1;
  for (std::int64_t i = 0; i < schema.arrow.n_children; ++i)
    if (schema.arrow.children[i]->name != nullptr && name == schema.arrow.children[i]->name) index = i;
  if (index < 0) throw IOError("no column '" + name + "'");
  std::vector<std::vector<std::vector<std::byte>>> out;
  std::vector<std::byte> buffer;
  std::span<const std::byte> window;
  std::int64_t at = after_schema;
  for (auto message = next_message(decoder, source, at, last, buffer, window); message;
       message = next_message(decoder, source, at, last, buffer, window)) {
    Decoded batch;
    const Used used{ { index, field_index(schema.arrow, index), name.c_str() } };
    decode_batch(schema.arrow, endianness, window, body_of(window, *message), *message, used, out.size(), std::nullopt,
                 batch);
    auto &buffers = out.emplace_back();
    const auto collect = [&buffers](const auto &self, ArrowArray *array) -> void {
      for (std::int64_t i = 0; i < array->n_buffers; ++i) {
        const ArrowBuffer *buffer = ArrowArrayBuffer(array, i);
        const auto *data = reinterpret_cast<const std::byte *>(buffer->data);
        buffers.emplace_back(data, data + buffer->size_bytes);
      }
      for (std::int64_t i = 0; i < array->n_children; ++i) self(self, array->children[i]);
    };
    collect(collect, &batch.arrays.front().arrow);
    at = message->end();
  }
  return out;
}

Data read_arrow_ipc(const std::filesystem::path &path, core::Precision dtype)
{
  try {
#ifdef DTWC_HAS_MMAP
    auto [file, bytes] = map_file(path);
    return detail::read_ipc(bytes, std::move(file), dtype, true);
#else
    // No mapping: the file read once into one buffer, which the views point into.
    std::ifstream in(path, std::ios::binary);
    std::error_code ec;
    const auto length = std::filesystem::file_size(path, ec);
    if (!in || ec) throw IOError("it could not be opened" + (ec ? ": " + ec.message() : std::string()));
    auto buffer = std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(length));
    in.read(reinterpret_cast<char *>(buffer->data()), static_cast<std::streamsize>(length));
    if (in.gcount() != static_cast<std::streamsize>(length))
      throw IOError("only " + std::to_string(in.gcount()) + " of its " + std::to_string(length) + " bytes were read");
    const std::span<const std::byte> bytes(*buffer);
    return detail::read_ipc(bytes, std::move(buffer), dtype);
#endif
  } catch (...) {
    dtwc::detail::rethrow_naming_the_file(path);
  }
}

// ---- IpcFile: the file read a record batch at a time ----

/// The file, walked: mapped (its bytes `memory`, kept by `mapping`), or read from `path` a range at a time.
struct IpcFile::State
{
  std::filesystem::path path;
  std::shared_ptr<const void> mapping;
  std::span<const std::byte> memory;
  std::int64_t size = 0;
  Walk walk;
  std::vector<index_t> first_row; ///< record batch k's first row, and the rows past the last
  std::int64_t declared = 0, values = 0, names = 0; ///< every batch's: decoded, of 'data', of 'name'
  std::size_t decoded = 0;    ///< the record batches decoded, not viewed
  std::size_t fields = 0;     ///< the schema's fields, depth first, its root among them
  std::size_t reader = 0;     ///< what a thread holds while it reads a record batch (reader_bytes)
  std::size_t names_read = 0; ///< the most a read of one record batch's names holds (batch_names_bytes)

  bool mapped() const { return mapping != nullptr; }
};

namespace {

/// One heap block of `bytes` (heap_block_added), as a size: a price of rows and values, as unsigned as they.
std::size_t heap(std::size_t bytes) { return bytes + static_cast<std::size_t>(heap_block_added(bytes)); }

/// A string's heap block for `chars` characters: none within its own buffer (15 characters in each standard library's
/// string), else its characters, its terminator and the rounding of its capacity (at most 15 more).
std::size_t string_bytes(std::size_t chars) { return chars <= 15 ? 0 : heap(chars + 16); }

/// The fields of `schema`, it and its children's, depth first.
std::size_t fields_of(const ArrowSchema &schema)
{
  std::size_t n = 1;
  for (std::int64_t i = 0; i < schema.n_children; ++i) n += fields_of(*schema.children[i]);
  return n;
}

/// The threads OpenMP gives a parallel region: a team's size at most.
std::size_t thread_count() { return static_cast<std::size_t>(std::max(1, get_max_threads())); }

/// The bytes of the name series_<i> made for a row of a file of `n` rows that has none, at most: what a price counts.
std::size_t made_name_bytes(index_t n) { return 7 + std::to_string(std::max<index_t>(n - 1, 0)).size(); }

/// The objects a record batch's decoded columns hold (nanoarrow's arrays and their private data, their place in an
/// owner): IpcFile::decoded_data_bytes, and IpcFile::decoded_name_bytes with a 'name' column; none when it is viewed.
std::size_t decoded_object_bytes(const IpcFile::State &state, const Batch &batch)
{
  if (batch.viewed) return 0;
  return IpcFile::decoded_data_bytes + (state.walk.columns.name >= 0 ? IpcFile::decoded_name_bytes : 0);
}

/// IpcFile::largest_row_group_bytes for one record batch, each heap block as the heap holds it (heap_block_bytes): its
/// message read from the file (without llfio) and its decoded columns; its spans and its name views (16 bytes a row
/// each), its made names (a made name a row at most, none where a 'name' column has no nulls), the rows it marks
/// unnamed while it is read; in the other
/// precision (T Float32 with `f32`) its values cast, beside which the spans of the stored values live until it is
/// read; its share of the block's spans and name views; its objects (batch_object_bytes, decoded_object_bytes); and
/// the two names the stream's check of it copies (its shortest and longest rows'), each at most its longest name where
/// the walk read its offsets (a batch viewed in memory), else its characters' bytes, or a made name.
std::size_t batch_bytes(const IpcFile::State &state, const Batch &batch, bool f32)
{
  const auto rows = static_cast<std::size_t>(batch.rows);
  const std::size_t made = made_name_bytes(state.first_row.back());
  const std::size_t views = rows * sizeof(std::span<const float>);
  const bool cast = f32 != state.walk.columns.f32;
  const std::size_t name = std::max(static_cast<std::size_t>(std::max<std::int64_t>(batch.longest_name, 0)), made);
  return (state.mapped() ? 0 : heap(static_cast<std::size_t>(batch.message.end() - batch.message.at)))
         + static_cast<std::size_t>(batch.declared) + (cast ? 3 : 2) * heap(views)
         + heap((state.walk.columns.name < 0 || batch.name_nulls ? rows : 0) * made)
         + heap((rows + 63) / 64 * 8)
         + (cast ? heap(static_cast<std::size_t>(batch.values) * (f32 ? sizeof(float) : sizeof(data_t))) : 0)
         + 2 * (views + views / 512) + IpcFile::batch_object_bytes + decoded_object_bytes(state, batch)
         + 2 * string_bytes(name);
}

/// What reading record batch `batch`'s names alone holds beside the names it makes (read_names), each heap block as the
/// heap holds it: without llfio its header and the bytes its 'name' column spans in the body (with 16 to place them);
/// decoded, the column and its objects.
std::size_t batch_names_bytes(const IpcFile::State &state, const Batch &batch)
{
  const std::size_t read = state.mapped() ? 0
                                          : heap(static_cast<std::size_t>(batch.message.header))
                                              + heap(static_cast<std::size_t>(batch.name_span) + 16);
  return read + (batch.viewed ? 0 : static_cast<std::size_t>(batch.names) + IpcFile::decoded_name_bytes);
}

/// What the walk holds for as long as the file is open: its facts of every record batch, and its schema.
std::size_t walk_bytes(const IpcFile::State &state)
{
  return state.walk.batches.capacity() * sizeof(Batch) + state.first_row.capacity() * sizeof(index_t)
         + state.fields * IpcFile::schema_field_bytes;
}

/// The bytes the 'name' column's names take as strings beyond their objects: their bytes and the allocator's header
/// each; none without the column (series_<i> fits a string's own buffer).
std::size_t name_heap_bytes(const IpcFile::State &state, std::size_t bytes, std::size_t rows)
{
  return state.walk.columns.name >= 0 ? bytes + rows * dtwc::detail::series_block_overhead_bytes : 0;
}

/// Record batch k of `state`'s file as a view Data of T, read as read_arrow_ipc reads it: its message in the mapping,
/// or read from the file; viewed where it lies or decoded; in the other precision, its values cast once into a buffer
/// of T; the rows named by nothing named series_<row>. Its owner holds what it views.
template <typename T, typename Stored>
Data read_batch_as(const IpcFile::State &state, std::size_t k)
{
  const Walk &walked = state.walk;
  const Batch &batch = walked.batches[k];
  const Message &message = batch.message;
  const std::string step = "Arrow IPC record batch " + std::to_string(k);
  auto owner = std::make_shared<Owner>();
  std::span<const std::byte> bytes;
  if (state.mapped()) {
    bytes = state.memory.subspan(static_cast<std::size_t>(message.at),
                                 static_cast<std::size_t>(message.end() - message.at));
    owner->bytes = state.mapping;
  } else {
    auto buffer = std::make_shared<std::vector<std::byte>>();
    bytes = Source(state.path, state.size).range(message.at, message.end(), *buffer);
    owner->bytes = std::move(buffer);
  }
  std::optional<Decoder> decoder; // a viewed batch's: its views are the decoder's state
  Decoded decoded;
  const ArrowArrayView *view = nullptr;
  if (batch.viewed) {
    ArrowError error{};
    check(ArrowIpcDecoderSetSchema(&decoder.emplace().ipc, const_cast<ArrowSchema *>(&walked.schema.arrow), &error),
          error, step);
    redecode(*decoder, bytes, message, step);
    view = view_batch(*decoder, body_of(bytes, message).view, walked.used, k);
    if (view == nullptr)
      throw IOError(step + " can no longer be viewed where it lies: the file changed while it was read");
  } else {
    decode_batch(walked.schema.arrow, walked.endianness, bytes, body_of(bytes, message), message, walked.used, k,
                 Expect{ batch.declared, batch.scratch }, decoded);
    view = &decoded.batch.arrow;
  }
  if (view->length != batch.rows)
    throw IOError(step + " holds " + std::to_string(view->length) + " rows, and when the file was walked "
                  + std::to_string(batch.rows) + ": the file changed while it was read");
  const auto rows = static_cast<std::size_t>(batch.rows);
  const std::int64_t first = state.first_row[k];
  std::vector<std::span<const Stored>> stored;
  std::vector<std::string_view> names;
  std::vector<bool> unnamed(rows);
  stored.reserve(rows);
  names.reserve(rows);
  view_rows(view, walked.columns, first, stored, names, unnamed, first);
  // Whole time steps, as Data::validate_ndim checks them, the file's row named (a Data names its own index).
  for (std::size_t r = 0; r < rows; ++r)
    if (stored[r].size() % walked.columns.ndim != 0)
      throw InvalidInput("Series " + std::to_string(first + static_cast<std::int64_t>(r)) + " has flat size "
                         + std::to_string(stored[r].size()) + " which is not divisible by ndim="
                         + std::to_string(walked.columns.ndim));
  owner->names = name_unnamed(names, unnamed, first);
  owner->decoded.reserve(decoded.arrays.size());
  for (Array &array : decoded.arrays) owner->decoded.push_back(std::move(array));
  if constexpr (std::is_same_v<T, Stored>) {
    return Data(std::move(stored), std::move(names), walked.columns.ndim, std::move(owner));
  } else {
    // The other precision: every value cast once, into one buffer the spans view.
    std::size_t total = 0;
    for (const auto &values : stored) total += values.size();
    auto cast = std::make_shared<std::vector<T>>();
    cast->reserve(total);
    std::vector<std::span<const T>> series;
    series.reserve(rows);
    for (const auto &values : stored) {
      const std::size_t at = cast->size();
      for (const Stored value : values) cast->push_back(static_cast<T>(value));
      series.emplace_back(cast->data() + at, values.size());
    }
    owner->values = std::move(cast);
    return Data(std::move(series), std::move(names), walked.columns.ndim, std::move(owner));
  }
}

template <typename T>
Data read_batch(const IpcFile::State &state, std::size_t k)
{
  return state.walk.columns.f32 ? read_batch_as<T, float>(state, k) : read_batch_as<T, data_t>(state, k);
}

/// Record batch k's 'name' column alone, read as read_batch reads the batch, each row's name, or series_<row>: in the
/// mapping or, without llfio, its header and the bytes the column's buffers span read from the file, the rest of the
/// body unread (unread_body), placed at the address residue the message read whole would give them, so that a buffer
/// is aligned here as it is there. Viewed or decoded as the walk found it, and checked against the walk (else the file
/// changed while it was read: an IOError).
void read_names(const IpcFile::State &state, std::size_t k, std::vector<std::string> &names)
{
  const Walk &walked = state.walk;
  const Batch &batch = walked.batches[k];
  const Message &message = batch.message;
  const std::string step = "Arrow IPC record batch " + std::to_string(k);
  const Used used{ walked.used[1] };
  Decoder decoder;
  ArrowError error{};
  check(ArrowIpcDecoderSetSchema(&decoder.ipc, const_cast<ArrowSchema *>(&walked.schema.arrow), &error), error, step);
  std::vector<std::byte> header_bytes, span_bytes;
  std::span<const std::byte> header;
  Body body;
  if (state.mapped()) {
    header = state.memory.subspan(static_cast<std::size_t>(message.at),
                                  static_cast<std::size_t>(message.end() - message.at));
    body = body_of(header, message);
  } else {
    const Source source(state.path, state.size);
    header = source.range(message.at, message.at + message.header, header_bytes);
    redecode(decoder, header, message, step);
    // Where the column's buffers lie, from the header alone (no codec, no byte order: nothing read), inside the body.
    const Body unread = unread_body(source, message);
    const auto codec = decoder.ipc.codec;
    decoder.ipc.codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
    ArrowArrayView *column = nullptr;
    const ArrowErrorCode code =
      ArrowIpcDecoderDecodeArrayView(&decoder.ipc, unread.view, used[0].field, &column, &error);
    decoder.ipc.codec = codec;
    check(code, error, step);
    check_in_body(column, unread.view, step + ", column 'name'", false);
    const auto [first, last] = span_of(column, unread.view);
    const auto pad = static_cast<std::size_t>((message.header + first) % 16);
    span_bytes.resize(pad + static_cast<std::size_t>(last - first));
    if (last > first) source.read(unread.at + first, unread.at + last, span_bytes.data() + pad);
    body.view.data.data = reinterpret_cast<const void *>(reinterpret_cast<std::uintptr_t>(span_bytes.data() + pad)
                                                         - static_cast<std::uintptr_t>(first));
    body.view.size_bytes = message.body;
  }
  Decoded decoded;
  const ArrowArrayView *view = nullptr;
  if (batch.viewed) {
    if (state.mapped()) redecode(decoder, header, message, step);
    view = view_batch(decoder, body.view, used, k);
    if (view == nullptr)
      throw IOError(step + " can no longer be viewed where it lies: the file changed while it was read");
  } else {
    decode_batch(walked.schema.arrow, walked.endianness, header, body, message, used, k,
                 Expect{ batch.names, batch.scratch }, decoded);
    view = &decoded.batch.arrow;
  }
  if (view->length != batch.rows)
    throw IOError(step + " holds " + std::to_string(view->length) + " rows, and when the file was walked "
                  + std::to_string(batch.rows) + ": the file changed while it was read");
  const ArrowArrayView *label = view->children[walked.columns.name];
  for (std::int64_t r = 0; r < batch.rows; ++r) {
    const std::int64_t row = state.first_row[k] + r;
    const auto name = row_name(label, r, row);
    names[static_cast<std::size_t>(row)] = name ? std::string(*name) : "series_" + std::to_string(row);
  }
}

} // namespace

IpcFile::IpcFile(const std::filesystem::path &path)
{
  try {
    auto state = std::make_unique<State>();
    state->path = path;
#ifdef DTWC_HAS_MMAP
    std::tie(state->mapping, state->memory) = map_file(path);
    state->size = static_cast<std::int64_t>(state->memory.size());
    walk(Source(state->memory), state->walk);
#else
    std::error_code ec;
    const auto length = std::filesystem::file_size(path, ec);
    if (ec || !std::ifstream(path, std::ios::binary))
      throw IOError("it could not be opened" + (ec ? ": " + ec.message() : std::string()));
    state->size = static_cast<std::int64_t>(length);
    walk(Source(state->path, state->size), state->walk);
#endif
    state->first_row.reserve(state->walk.batches.size() + 1);
    state->first_row.push_back(0);
    std::int64_t scratch = 0;
    for (const Batch &batch : state->walk.batches) {
      state->first_row.push_back(sum(state->first_row.back(), batch.rows));
      state->declared = sum(state->declared, batch.declared);
      state->values = sum(state->values, batch.values);
      state->names = sum(state->names, batch.names);
      state->decoded += batch.viewed ? 0 : 1;
      scratch = std::max(scratch, batch.scratch);
    }
    state->fields = fields_of(state->walk.schema.arrow);
    state->reader = static_cast<std::size_t>(scratch) + state->fields * field_bytes;
    if (state->walk.columns.name >= 0)
      for (const Batch &batch : state->walk.batches)
        state->names_read = std::max(state->names_read, batch_names_bytes(*state, batch));
    state_ = std::move(state);
  } catch (...) {
    dtwc::detail::rethrow_naming_the_file(path);
  }
}

IpcFile::~IpcFile() = default;

index_t IpcFile::num_rows() const { return state_->first_row.back(); }

int IpcFile::num_row_groups() const { return static_cast<int>(state_->walk.batches.size()); }

bool IpcFile::is_f32() const { return state_->walk.columns.f32; }

std::size_t IpcFile::ndim() const { return state_->walk.columns.ndim; }

std::size_t IpcFile::num_values() const { return static_cast<std::size_t>(state_->values); }

std::size_t IpcFile::longest_steps() const
{
  const std::int64_t longest = state_->walk.longest_values;
  return longest < 0 ? 0 : static_cast<std::size_t>(longest) / state_->walk.columns.ndim;
}

bool IpcFile::streams(core::Precision dtype) const
{
  return !state_->mapped() || state_->decoded > 0 || (dtype == core::Precision::Float32) != is_f32();
}

std::string IpcFile::storage(core::Precision dtype) const
{
  const State &state = *state_;
  if ((dtype == core::Precision::Float32) != is_f32())
    return "a copy as " + std::string(name_of(core::precision_names, dtype));
  if (state.decoded > 0) {
    const std::size_t batches = state.walk.batches.size();
    const std::string count = state.decoded < batches ? std::to_string(state.decoded) + " of " : std::string();
    return std::string("decoded (") + state.walk.decoded_for + ", " + count + std::to_string(batches)
           + (batches == 1 ? " record batch)" : " record batches)");
  }
  return state.mapped() ? "mapped" : "read into memory (no llfio)";
}

std::size_t IpcFile::resident_bytes(core::Precision dtype, bool gpu) const
{
  const State &state = *state_;
  const auto n = static_cast<std::size_t>(num_rows());
  const auto values = static_cast<std::size_t>(state.values);
  const std::size_t names = name_heap_bytes(state, static_cast<std::size_t>(state.names), n);
  // read_arrow_ipc decodes its batches with up to a thread each, each holding a reader's scratch (and views the rest
  // with one decoder), and holds until it returns each batch's objects (its walk's facts among them), a decoded batch's
  // columns' objects and its view of the schema, and its walk's schema; without llfio, the file read into one block.
  const std::size_t readers = std::min(thread_count(), std::max<std::size_t>(state.decoded, 1));
  const std::size_t decoded_objects =
    resident_data_bytes + (state.walk.columns.name >= 0 ? resident_name_bytes : 0) + state.fields * view_field_bytes;
  std::size_t bytes = n * dtwc::detail::view_series_bytes + static_cast<std::size_t>(state.declared)
                      + (state.mapped() ? 0 : heap(static_cast<std::size_t>(state.size)))
                      + state.walk.batches.size() * resident_batch_bytes
                      + state.decoded * decoded_objects + state.fields * schema_field_bytes + readers * state.reader;
  const bool f32 = dtype == core::Precision::Float32;
  if (f32 != is_f32()) bytes += series_bytes(values, n, f32) + names; // the copy in the other precision
  else if (gpu) bytes += series_bytes(values, n, false) + names;      // a view's upload copy
  return bytes;
}

std::size_t IpcFile::largest_row_group_bytes(bool f32) const
{
  std::size_t largest = 0;
  for (const Batch &batch : state_->walk.batches) largest = std::max(largest, batch_bytes(*state_, batch, f32));
  return largest;
}

std::size_t IpcFile::decode_bytes() const { return walk_bytes(*state_) + state_->reader; }

std::size_t IpcFile::open_bytes() const { return walk_bytes(*state_); }

std::size_t IpcFile::reader_bytes() const { return state_->reader; }

int IpcFile::row_groups_per_batch(std::size_t ram_budget, bool f32) const
{
  const std::size_t largest = largest_row_group_bytes(f32);
  if (largest > ram_budget)
    throw InvalidInput("io::IpcFile: --ram-limit leaves " + std::to_string(ram_budget) + " bytes for record batches, "
                       "but one record batch holds " + std::to_string(largest) + " bytes once read; rewrite the file "
                       "in smaller record batches or raise the limit");
  // `count` record batches are read by min(count, threads) threads, each but the first (decode_bytes holds its scratch)
  // holding a reader's scratch beside its batch.
  const std::size_t scratch = state_->reader, threads = thread_count();
  std::size_t count = (ram_budget + scratch) / (largest + scratch);
  if (count >= threads) count = (ram_budget - (threads - 1) * scratch) / largest;
  return static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(num_row_groups(), count)));
}

template <typename T>
Data IpcFile::read_rows(std::vector<index_t> rows, std::size_t ram_budget) const
{
  static_assert(std::is_same_v<T, data_t> || std::is_same_v<T, float>);
  if (rows.empty()) return Data{};
  constexpr bool f32 = std::is_same_v<T, float>;
  const State &state = *state_;
  try {
    // The pass reads every record batch whole: one larger than the limit is refused before any is read.
    const std::size_t largest = largest_row_group_bytes(f32);
    if (largest > ram_budget) {
      std::size_t k = 0;
      while (batch_bytes(state, state.walk.batches[k], f32) != largest) ++k;
      throw InvalidInput("io::IpcFile::read_rows: record batch " + std::to_string(k) + " of the Arrow IPC file holds "
                         + std::to_string(largest) + " bytes once read (its "
                         + std::to_string(state.walk.batches[k].rows) + " series), more than --ram-limit="
                         + std::to_string(ram_budget) + ", and FastCLARA's stream reads each record batch whole; raise "
                         "the limit, or rewrite the file in smaller record batches");
    }
    std::vector<std::size_t> order(rows.size()); // rows[order[i]] ascending
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return rows[a] < rows[b]; });
    // The record batches holding a requested row, each with its requests order[from, to).
    struct Group
    {
      std::size_t batch, from, to;
    };
    std::vector<Group> groups;
    for (std::size_t i = 0; i < order.size();) {
      const auto k = static_cast<std::size_t>(
        std::upper_bound(state.first_row.begin(), state.first_row.end(), rows[order[i]]) - state.first_row.begin() - 1);
      std::size_t to = i;
      while (to < order.size() && rows[order[to]] < state.first_row[k + 1]) ++to;
      groups.push_back({ k, i, to });
      i = to;
    }
    std::vector<std::vector<T>> vecs(rows.size());
    std::vector<std::string> names(rows.size());
    // Held throughout: what reading holds (decode_bytes: the walk, one reader's scratch), and the series' objects;
    // their values as they are copied.
    const std::size_t reading = decode_bytes();
    std::size_t retained = series_bytes(0, rows.size(), f32);
    const std::size_t threads = thread_count();
    for (std::size_t next = 0; next < groups.size();) {
      // As many record batches read at once as the limit holds beside the rows copied so far: each read whole, by a
      // thread with its scratch (the first's in decode_bytes), beside the copies of its rows, at most its values and
      // its names.
      std::size_t team = 0, wave = 0;
      for (; team < threads && next + team < groups.size(); ++team) {
        const Group &group = groups[next + team];
        const Batch &batch = state.walk.batches[group.batch];
        const std::size_t member =
          largest + (team > 0 ? state.reader : 0) + series_bytes(static_cast<std::size_t>(batch.values), 0, f32)
          + name_heap_bytes(state, static_cast<std::size_t>(batch.names), group.to - group.from);
        if (reading + retained + wave + member > ram_budget) {
          if (team == 0)
            throw InvalidInput("io::IpcFile::read_rows: the " + std::to_string(rows.size()) + " selected series hold "
                               + std::to_string(retained) + " bytes so far and reading the file "
                               + std::to_string(reading) + " (its walk and a reader's scratch); a record batch of "
                               + "their rows read whole and its rows copied need " + std::to_string(member)
                               + " more, " + std::to_string(reading + retained + member)
                               + " in all, exceeding --ram-limit=" + std::to_string(ram_budget) + "; raise the limit");
          break;
        }
        wave += member;
      }
      std::vector<std::size_t> copied(team);
      const auto read = [&](std::size_t j) {
        const Group &group = groups[next + j];
        const Data batch = read_batch<T>(state, group.batch);
        for (std::size_t i = group.from; i < group.to; ++i) {
          const std::size_t original = order[i];
          const auto local = static_cast<std::size_t>(rows[original] - state.first_row[group.batch]);
          const auto values = [&] {
            if constexpr (f32)
              return batch.series_f32(local);
            else
              return batch.series(local);
          }();
          vecs[original].assign(values.begin(), values.end());
          names[original] = std::string(batch.name(local));
          copied[j] += series_bytes(values.size(), 0, f32) + name_heap_bytes(state, names[original].size(), 1);
        }
      };
      run_openmp(read, team, team > 1);
      for (const std::size_t bytes : copied) retained += bytes;
      next += team;
    }
    return Data(std::move(vecs), std::move(names), state.walk.columns.ndim);
  } catch (...) {
    dtwc::detail::rethrow_naming_the_file(state.path);
  }
}

template <typename T>
Data IpcFile::read_row_groups(int first, int count, const PieceCheck &check) const
{
  const State &state = *state_;
  try {
    std::vector<Data> batches(static_cast<std::size_t>(count));
    const auto read = [&](std::size_t j) {
      const std::size_t k = static_cast<std::size_t>(first) + j;
      batches[j] = read_batch<T>(state, k);
      check(j, batches[j], state.first_row[k]);
    };
    run_openmp(read, batches.size(), batches.size() > 1);
    // One Data: the batches' spans and names, its owner holding theirs.
    std::size_t rows = 0;
    for (const Data &batch : batches) rows += static_cast<std::size_t>(batch.size());
    std::vector<std::span<const T>> series;
    std::vector<std::string_view> names;
    series.reserve(rows);
    names.reserve(rows);
    auto owners = std::make_shared<std::vector<std::shared_ptr<const void>>>();
    for (const Data &batch : batches) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(batch.size()); ++i) {
        if constexpr (std::is_same_v<T, float>)
          series.push_back(batch.series_f32(i));
        else
          series.push_back(batch.series(i));
        names.push_back(batch.name(i));
      }
      owners->push_back(batch.owner());
    }
    return Data(std::move(series), std::move(names), state.walk.columns.ndim, std::move(owners));
  } catch (...) {
    dtwc::detail::rethrow_naming_the_file(state.path);
  }
}

std::size_t IpcFile::series_names_bytes() const
{
  const auto n = static_cast<std::size_t>(num_rows());
  return n * sizeof(std::string) + name_heap_bytes(*state_, static_cast<std::size_t>(state_->names), n);
}

std::size_t IpcFile::names_read_bytes() const
{
  return walk_bytes(*state_) + (state_->walk.columns.name >= 0 ? state_->names_read + state_->reader : 0);
}

std::vector<std::string> IpcFile::series_names(std::size_t ram_budget) const
{
  const State &state = *state_;
  std::vector<std::string> names(static_cast<std::size_t>(num_rows()));
  if (state.walk.columns.name < 0) {
    for (std::size_t i = 0; i < names.size(); ++i) names[i] = "series_" + std::to_string(i);
    return names;
  }
  try {
    // A team the budget holds beside the names and the walk, each thread reading a batch's names alone; not one is
    // refused (run's planner prices one: names_read_bytes).
    const std::size_t held = series_names_bytes() + walk_bytes(state), each = state.names_read + state.reader;
    if (ram_budget < held + each)
      throw InvalidInput("io::IpcFile::series_names: the " + std::to_string(names.size()) + " names and the file's "
                         "walk hold " + std::to_string(held) + " bytes and reading one record batch's names needs "
                         + std::to_string(each) + " more, " + std::to_string(held + each)
                         + " in all, exceeding the " + std::to_string(ram_budget) + " the limit leaves them; raise "
                         "the limit");
    const std::size_t team = std::min((ram_budget - held) / each, thread_count());
    const auto read = [&](std::size_t k) { read_names(state, k, names); };
    run_openmp(read, state.walk.batches.size(), team > 1 && state.walk.batches.size() > 1, 4, static_cast<int>(team));
  } catch (...) {
    dtwc::detail::rethrow_naming_the_file(state.path);
  }
  return names;
}

template Data IpcFile::read_rows<float>(std::vector<index_t>, std::size_t) const;
template Data IpcFile::read_rows<data_t>(std::vector<index_t>, std::size_t) const;
template Data IpcFile::read_row_groups<float>(int, int, const PieceCheck &) const;
template Data IpcFile::read_row_groups<data_t>(int, int, const PieceCheck &) const;

} // namespace dtwc::io
