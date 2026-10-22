/**
 * @file arrow_ipc.cpp
 * @brief dtwc::io::read_arrow_ipc (see arrow_ipc.hpp): the messages walked and decoded by nanoarrow's IPC extension,
 *        each record batch's buffers viewed where they lie.
 * @author Volkan Kumtepeli
 */

#include "arrow_ipc.hpp"

#include "read_data.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"

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
#else
#include <fstream>
#endif

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
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

/// What a view of the file holds: the bytes' keep-alive (none once every batch is decoded), the decoded columns, and
/// the names made for the series that have none.
struct Owner
{
  std::shared_ptr<const void> bytes;
  std::vector<Array> decoded;
  std::vector<char> names;
};

/// nanoarrow's failure as an IOError naming the step; some of its returns set no message.
void check(ArrowErrorCode code, const ArrowError &error, const std::string &step)
{
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

ArrowBufferView bytes_view(std::span<const std::byte> bytes, std::int64_t first, std::int64_t last)
{
  ArrowBufferView view;
  view.data.data = bytes.data() + first;
  view.size_bytes = last - first;
  return view;
}

/// The bytes [first, last) that hold the messages: the file format's between its leading magic and its footer, the
/// stream format's all of them.
std::pair<std::int64_t, std::int64_t> message_bytes(std::span<const std::byte> bytes)
{
  const auto size = static_cast<std::int64_t>(bytes.size());
  const auto holds = [&](std::int64_t at, std::string_view text) {
    return at >= 0 && size - at >= static_cast<std::int64_t>(text.size())
           && std::memcmp(bytes.data() + at, text.data(), text.size()) == 0;
  };
  constexpr std::string_view magic = "ARROW1";
  if (holds(0, magic)) {
    // The magic and 2 bytes of padding, the stream, the footer, its length (a little-endian int32), the magic.
    if (size < 18 || !holds(size - 6, magic))
      throw IOError("an Arrow IPC file ends with the ARROW1 magic it begins with, and this one does not: it is "
                    "truncated");
    std::uint32_t length = 0;
    for (std::int64_t i = size - 7; i >= size - 10; --i)
      length = (length << 8) | std::to_integer<std::uint32_t>(bytes[static_cast<std::size_t>(i)]);
    const auto footer = static_cast<std::int64_t>(static_cast<std::int32_t>(length));
    if (footer <= 0 || footer > size - 18)
      throw IOError("the Arrow IPC footer's length, " + std::to_string(footer) + " bytes, does not fit in the file's "
                    + std::to_string(size) + ": it is truncated or corrupt");
    return { 8, size - 10 - footer };
  }
  if (holds(0, std::string_view("\xFF\xFF\xFF\xFF", 4))) return { 0, size };
  if (holds(0, "FEA1"))
    throw IOError("a Feather version 1 file (magic FEA1), which is not Arrow IPC: write it as Feather version 2, which "
                  "is (pyarrow.feather.write_feather(table, path, version=2, compression=\"uncompressed\"))");
  throw IOError("not an Arrow IPC file: it begins with neither the ARROW1 magic of the file format nor the 0xFFFFFFFF "
                "of the stream format");
}

/// The message at byte `at` of [at, last), its header verified (flatcc) and decoded; none at `last`, or at the
/// end-of-stream marker.
std::optional<Message> next_message(Decoder &decoder, std::span<const std::byte> bytes, std::int64_t at,
                                    std::int64_t last)
{
  if (at == last) return std::nullopt;
  const std::string step = "Arrow IPC message at byte " + std::to_string(at);
  ArrowError error{};
  const ArrowBufferView data = bytes_view(bytes, at, last);
  std::int32_t prefix = 0;
  const ArrowErrorCode peeked = ArrowIpcDecoderPeekHeader(&decoder.ipc, data, &prefix, &error);
  if (peeked == ENODATA) return std::nullopt;
  check(peeked, error, step);
  // The header's length, from the file, which nanoarrow's verifier trusts: at least its 8-byte prefix, inside the file.
  const std::int64_t header = decoder.ipc.header_size_bytes;
  if (header < 8 || header > last - at)
    throw IOError(step + ": its header of " + std::to_string(header) + " bytes does not fit in the "
                  + std::to_string(last - at) + " left: the file is truncated or corrupt");
  check(ArrowIpcDecoderVerifyHeader(&decoder.ipc, data, &error), error, step);
  check(ArrowIpcDecoderDecodeHeader(&decoder.ipc, data, &error), error, step);
  const Message message{ at, header, decoder.ipc.body_size_bytes };
  if (message.body < 0 || message.body > last - at - message.header)
    throw IOError(step + ": its body of " + std::to_string(message.body) + " bytes runs past the "
                  + std::to_string(last - at - message.header) + " left: the file is truncated or corrupt");
  return message;
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
  const auto first = reinterpret_cast<std::uintptr_t>(body.data.data);
  const auto last = first + static_cast<std::uintptr_t>(body.size_bytes);
  for (const ArrowBufferView &buffer : view->buffer_views) {
    const auto at = reinterpret_cast<std::uintptr_t>(buffer.data.data);
    if (buffer.size_bytes != 0
        && (buffer.size_bytes < 0 || at < first || at > last
            || static_cast<std::uintptr_t>(buffer.size_bytes) > last - at))
      throw IOError(where + ": a buffer of " + std::to_string(buffer.size_bytes)
                    + " bytes is not inside the batch's body of " + std::to_string(body.size_bytes)
                    + " bytes: the file is corrupt");
  }
  for (std::int64_t i = 0; i < view->n_children; ++i) check_in_body(view->children[i], body, where, lengths);
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

/// The columns the series come from, each with the name its errors give it: 'data', and 'name' if the file has one.
using Used = std::vector<std::pair<std::int64_t, const char *>>;

Used used_of(const Columns &columns)
{
  Used used{ { columns.data, "data" } };
  if (columns.name >= 0) used.emplace_back(columns.name, "name");
  return used;
}

/// An IOError naming `where` unless the length of `used`, a column the series come from, is the batch's `rows`.
void check_rows(const ArrowArrayView *used, std::int64_t rows, const std::string &where)
{
  if (used->length != rows)
    throw IOError(where + ": its length of " + std::to_string(used->length) + " is not the batch's "
                  + std::to_string(rows) + ": the file is corrupt");
}

/// Record batch `k`'s views of every column, nanoarrow's pointers into the body; none when a used buffer is not aligned
/// to its elements (the batch is decoded instead, into aligned buffers). The columns the series come from are checked
/// to lie in the body, as long as the batch, aligned, and validated (DEFAULT: every buffer size, and the first and last
/// offset of each list and string column, read as integers); the others are never read.
ArrowArrayView *view_batch(Decoder &decoder, std::span<const std::byte> bytes, const Message &message,
                           const Columns &columns, std::size_t k)
{
  const std::string step = "Arrow IPC record batch " + std::to_string(k);
  ArrowError error{};
  ArrowArrayView *view = nullptr;
  const ArrowBufferView body = bytes_view(bytes, message.at + message.header, message.end());
  check(ArrowIpcDecoderDecodeArrayView(&decoder.ipc, body, -1, &view, &error), error, step);
  for (const auto &[column, name] : used_of(columns)) {
    const std::string where = step + ", column '" + name + "'";
    ArrowArrayView *used = view->children[column];
    check_in_body(used, body, where);
    check_rows(used, view->length, where);
    if (!aligned(used)) return nullptr;
    check(ArrowArrayViewValidate(used, NANOARROW_VALIDATION_LEVEL_DEFAULT, &error), error, where);
  }
  return view;
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

/// A record batch that cannot be viewed, decoded: the arrays of its used columns, whose buffers hold the values and
/// names the series view, and a view of the batch through them.
struct Decoded
{
  std::vector<Array> arrays;
  View batch;
};

/// Record batch `k` at `message`, which cannot be viewed (compressed, in the other byte order, or a used buffer not
/// aligned), decoded into `out` by a decoder of its own (a decoder holds the batch it last decoded, so none is shared
/// between threads): each column of `used` into an array nanoarrow allocates, decompressed, in this machine's byte
/// order and aligned.
///
/// nanoarrow trusts the file's buffer offsets and lengths, and its DecodeArray reads each buffer of the column it
/// decodes (decompressing, swapping or copying it) at the offset the file gives, before any check of ours could run.
/// So every buffer it will read is proven inside the body first: a pass with the codec set to NONE and no byte order
/// makes each buffer a view where the file puts it and reads none, and each used column's views must lie in the body
/// (bounds only: a compressed body can be smaller than its values), and, to be swapped where it lies, be aligned to its
/// elements. DecodeArray of a column then reads those buffers and no others: the same FieldNodes and Buffers of the
/// same verified header, of that column's fields only. Decoded, each length is bounded by the column's decoded bytes,
/// the column must be as long as the batch, and it is validated at DEFAULT.
void decode_batch(const ArrowSchema &schema, ArrowIpcEndianness endianness, std::span<const std::byte> bytes,
                  const Message &message, const Used &used, std::size_t k, Decoded &out)
{
  const std::string step = "Arrow IPC record batch " + std::to_string(k);
  ArrowError error{};
  Decoder decoder;
  // SetSchema reads the schema, as every decoder does at once.
  check(ArrowIpcDecoderSetSchema(&decoder.ipc, const_cast<ArrowSchema *>(&schema), &error), error, step);
  (void)next_message(decoder, bytes, message.at, message.end()); // the walk verified it: decode it again
  const ArrowBufferView body = bytes_view(bytes, message.at + message.header, message.end());
  const auto codec = decoder.ipc.codec;
  decoder.ipc.codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  ArrowArrayView *raw = nullptr;
  check(ArrowIpcDecoderDecodeArrayView(&decoder.ipc, body, -1, &raw, &error), error, step);
  decoder.ipc.codec = codec;
  const std::int64_t rows = raw->length;
  const bool swapped = endianness != ArrowIpcSystemEndianness();
  for (const auto &[column, name] : used) {
    check_in_body(raw->children[column], body, step + ", column '" + name + "'", false);
    if (swapped && !aligned(raw->children[column]))
      throw IOError(step + ": a buffer is not aligned to its elements, so it cannot be byte-swapped (Arrow IPC "
                           "writers align every buffer to 8 bytes: rewrite the file with one)");
  }
  check(ArrowIpcDecoderSetEndianness(&decoder.ipc, endianness), error, step);
  check(ArrowArrayViewInitFromSchema(&out.batch.arrow, &schema, &error), error, step);
  out.batch.arrow.length = rows;
  out.arrays.reserve(used.size());
  for (const auto &[column, name] : used) {
    const std::string where = step + ", column '" + name + "'";
    Array &array = out.arrays.emplace_back();
    check(ArrowIpcDecoderDecodeArray(&decoder.ipc, body, field_index(schema, column), &array.arrow,
                                     NANOARROW_VALIDATION_LEVEL_NONE, &error),
          error, where);
    if (array.arrow.release == nullptr) // nanoarrow returns OK with a released array when it runs out of memory
      throw IOError(where + ": nanoarrow decoded no array"
                    + (error.message[0] != '\0' ? ": " + std::string(error.message) : std::string()));
    ArrowArrayView *view = out.batch.arrow.children[column];
    view_decoded(view, &array.arrow);
    check_lengths(view, decoded_bytes(view), where);
    check_rows(view, rows, where);
    check(ArrowArrayViewValidate(view, NANOARROW_VALIDATION_LEVEL_DEFAULT, &error), error, where);
  }
}

/// A record batch of the walk: its message, whether it is viewed where it lies (else decoded), and its rows.
struct Batch
{
  Message message;
  bool viewed = false;
  std::int64_t rows = 0;
};

/// The file format's footer verified (flatcc) and decoded, its record batch blocks the batches the walk found: a
/// reader that seeks by the footer reads those.
void check_footer(std::span<const std::byte> bytes, const std::vector<Batch> &batches)
{
  Decoder decoder;
  ArrowError error{};
  const ArrowBufferView file = bytes_view(bytes, 0, static_cast<std::int64_t>(bytes.size()));
  check(ArrowIpcDecoderVerifyFooter(&decoder.ipc, file, &error), error, "Arrow IPC footer");
  check(ArrowIpcDecoderDecodeFooter(&decoder.ipc, file, &error), error, "Arrow IPC footer");
  const ArrowBuffer &blocks = decoder.ipc.footer->record_batch_blocks;
  const auto *block = reinterpret_cast<const ArrowIpcFileBlock *>(blocks.data);
  const auto n = static_cast<std::size_t>(blocks.size_bytes) / sizeof(ArrowIpcFileBlock);
  if (n != batches.size())
    throw IOError("the Arrow IPC footer lists " + std::to_string(n) + " record batches, and the file's messages hold "
                  + std::to_string(batches.size()) + ": the file is corrupt");
  for (std::size_t k = 0; k < n; ++k)
    if (block[k].offset != batches[k].message.at)
      throw IOError("the Arrow IPC footer puts record batch " + std::to_string(k) + " at byte "
                    + std::to_string(block[k].offset) + ", and the file's messages put it at byte "
                    + std::to_string(batches[k].message.at) + ": the file is corrupt");
}

ArrowType type_of(const ArrowSchema *schema)
{
  ArrowSchemaView view;
  ArrowError error{};
  check(ArrowSchemaViewInit(&view, schema, &error), error, "Arrow IPC schema");
  return view.type;
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

/// The offsets [start, end) of string `r` of a Utf8 or LargeUtf8 view.
std::pair<std::int64_t, std::int64_t> string_offsets(const ArrowArrayView *view, std::int64_t r)
{
  const std::int64_t i = view->offset + r;
  if (view->storage_type == NANOARROW_TYPE_LARGE_STRING)
    return { view->buffer_views[1].data.as_int64[i], view->buffer_views[1].data.as_int64[i + 1] };
  return { view->buffer_views[1].data.as_int32[i], view->buffer_views[1].data.as_int32[i + 1] };
}

/// The rows of a record batch, from row `first` on: a span of each series' values and a view of its name where they
/// lie (the body, or a decoded column), today's verdicts on a null and on offsets outside their values row by row; a
/// row named by no 'name' value is marked in `unnamed`.
template <typename T>
void view_rows(const ArrowArrayView *batch, const Columns &columns, std::int64_t first,
               std::vector<std::span<const T>> &series, std::vector<std::string_view> &names,
               std::vector<bool> &unnamed)
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
    if (label != nullptr && !(label->null_count != 0 && ArrowArrayViewIsNull(label, r))) {
      const auto [name_start, name_end] = string_offsets(label, r);
      const std::int64_t chars = label->buffer_views[2].size_bytes;
      if (name_start < 0 || name_end < name_start || name_end > chars)
        throw IOError("data_from_arrow: name offset [" + std::to_string(name_start) + ", " + std::to_string(name_end)
                      + ") of the series at row " + std::to_string(row) + " is outside the names [0, "
                      + std::to_string(chars) + ").");
      names.emplace_back(label->buffer_views[2].data.as_char + name_start,
                         static_cast<std::size_t>(name_end - name_start));
    } else {
      names.emplace_back();
      unnamed[static_cast<std::size_t>(row)] = true;
    }
  }
}

/// The view Data of the series in `batches`, each viewed where it lies or, if `decoded`, in its decoded columns: their
/// spans and name views, and the names series_<row> of the rows named by nothing, made in one buffer sized first (views
/// into a growing one would dangle). The owner keeps the bytes only if a batch is viewed in them.
template <typename T>
Data view_batches(Decoder &decoder, std::span<const std::byte> bytes, const std::vector<Batch> &batches,
                  std::vector<Decoded> &decoded, const Columns &columns, std::shared_ptr<const void> keep)
{
  std::int64_t rows = 0; // each batch's rows are bounded by its body or its decoded bytes
  for (const Batch &batch : batches) rows += batch.rows;
  const auto n = static_cast<std::size_t>(rows);
  std::vector<std::span<const T>> series;
  std::vector<std::string_view> names;
  std::vector<bool> unnamed(n);
  series.reserve(n);
  names.reserve(n);
  bool viewed = false;
  for (std::size_t k = 0; k < batches.size(); ++k) {
    const ArrowArrayView *batch = &decoded[k].batch.arrow;
    if (batches[k].viewed) {
      const Message &message = batches[k].message;
      (void)next_message(decoder, bytes, message.at, message.end()); // the walk verified it: decode it again
      batch = view_batch(decoder, bytes, message, columns, k);
      if (batch == nullptr)
        throw IOError("Arrow IPC record batch " + std::to_string(k)
                      + " can no longer be viewed where it lies: the file changed while it was read");
      viewed = true;
    }
    if (series.size() + static_cast<std::size_t>(batch->length) > n)
      throw IOError("Arrow IPC record batch " + std::to_string(k)
                    + " holds more rows than when the file was walked: the file changed while it was read");
    view_rows(batch, columns, static_cast<std::int64_t>(series.size()), series, names, unnamed);
  }
  if (series.size() != n)
    throw IOError("the Arrow IPC file's record batches hold " + std::to_string(series.size()) + " rows, and its walk "
                  + "counted " + std::to_string(n) + ": the file changed while it was read");
  char digits[24];
  std::size_t chars = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (unnamed[i]) chars += 7 + static_cast<std::size_t>(std::to_chars(digits, digits + sizeof digits, i).ptr - digits);
  std::vector<char> made(chars);
  char *out = made.data();
  for (std::size_t i = 0; i < n; ++i)
    if (unnamed[i]) {
      char *const name = out;
      out = std::to_chars(std::copy_n("series_", 7, out), made.data() + chars, i).ptr;
      names[i] = std::string_view(name, static_cast<std::size_t>(out - name));
    }
  std::vector<Array> arrays;
  for (Decoded &batch : decoded)
    for (Array &array : batch.arrays) arrays.push_back(std::move(array)); // the structs move, their buffers stay
  auto owner = std::make_shared<Owner>(
    Owner{ viewed ? std::move(keep) : nullptr, std::move(arrays), std::move(made) }); // the chars stay too
  return Data(std::move(series), std::move(names), columns.ndim, std::move(owner));
}

/// The schema message at byte `first`, decoded into `schema` and set on `decoder`: the byte order of the file's buffers
/// (nanoarrow swaps a batch's buffers only after SetEndianness, as its own reader calls it), and the byte after it.
std::pair<ArrowIpcEndianness, std::int64_t> read_schema(Decoder &decoder, std::span<const std::byte> bytes,
                                                        std::int64_t first, std::int64_t last, Schema &schema)
{
  const auto head = next_message(decoder, bytes, first, last);
  if (!head || decoder.ipc.message_type != NANOARROW_IPC_MESSAGE_TYPE_SCHEMA)
    throw IOError("an Arrow IPC stream begins with its schema, and this one does not");
  const ArrowIpcEndianness endianness = decoder.ipc.endianness;
  ArrowError error{};
  check(ArrowIpcDecoderDecodeSchema(&decoder.ipc, &schema.arrow, &error), error, "Arrow IPC schema");
  check(ArrowIpcDecoderSetSchema(&decoder.ipc, &schema.arrow, &error), error, "Arrow IPC schema");
  return { endianness, head->end() };
}

} // namespace

Data detail::read_ipc(std::span<const std::byte> bytes, std::shared_ptr<const void> owner, core::Precision dtype)
{
  if (reinterpret_cast<std::uintptr_t>(bytes.data()) % 8 != 0) {
    // flatcc verifies a header's tables at their alignment in memory, which the file gives relative to its start: an
    // 8-aligned copy, which the owner holds, is read instead.
    auto copy = std::make_shared<const std::vector<std::byte>>(bytes.begin(), bytes.end());
    const std::span<const std::byte> aligned(*copy);
    return read_ipc(aligned, std::move(copy), dtype);
  }
  const auto [first, last] = message_bytes(bytes);
  Schema schema;
  Decoder decoder; // the walk's, which views batches in this machine's byte order only
  const auto [endianness, after_schema] = read_schema(decoder, bytes, first, last, schema);
  const Columns columns = columns_of(schema.arrow);

  // The walk: each record batch's message, viewed where it lies (uncompressed, in this machine's byte order, its used
  // buffers aligned) with its rows, so the views are sized before any is made, or else decoded below.
  std::vector<Batch> batches;
  std::int64_t at = after_schema;
  for (auto message = next_message(decoder, bytes, at, last); message;
       message = next_message(decoder, bytes, at, last)) {
    const std::size_t k = batches.size();
    if (decoder.ipc.message_type != NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH)
      throw IOError("Arrow IPC message at byte " + std::to_string(message->at)
                    + " is a second schema; a file holds one, before its record batches");
    const bool plain =
      decoder.ipc.codec == NANOARROW_IPC_COMPRESSION_TYPE_NONE && endianness == ArrowIpcSystemEndianness();
    const ArrowArrayView *view = plain ? view_batch(decoder, bytes, *message, columns, k) : nullptr;
    batches.push_back({ *message, view != nullptr, view != nullptr ? view->length : 0 });
    at = message->end();
  }
  // The file format's end-of-stream marker, if it has one, is its last message (just before the footer): a marker
  // before a record batch would drop it unread. The stream format's messages begin at byte 0, the file format's at 8.
  if (first != 0 && at != last && at != last - 8)
    throw IOError("Arrow IPC message at byte " + std::to_string(at) + " is an end-of-stream marker, but the file's "
                  "messages run on to its footer at byte " + std::to_string(last) + ": the file is corrupt");
  if (first != 0) check_footer(bytes, batches);

  // Every batch that cannot be viewed, decoded in parallel over disjoint batches; a failure is the lowest failing
  // batch's, as a serial decode raises it.
  std::vector<std::size_t> todo;
  for (std::size_t k = 0; k < batches.size(); ++k)
    if (!batches[k].viewed) todo.push_back(k);
  std::vector<Decoded> decoded(batches.size());
  const Used used = used_of(columns);
  const auto decode = [&](std::size_t j) {
    const std::size_t k = todo[j];
    decode_batch(schema.arrow, endianness, bytes, batches[k].message, used, k, decoded[k]);
    batches[k].rows = decoded[k].batch.arrow.length;
  };
  run_openmp(decode, todo.size(), todo.size() > 1);

  Data view = columns.f32 ? view_batches<float>(decoder, bytes, batches, decoded, columns, std::move(owner))
                          : view_batches<data_t>(decoder, bytes, batches, decoded, columns, std::move(owner));
  if (view.precision == dtype) return view;
  return dtwc::detail::owned_copy(view, dtype); // the request names other bytes: each value cast once
}

std::vector<std::vector<std::vector<std::byte>>> detail::decoded_buffers(std::span<const std::byte> bytes,
                                                                         std::string_view column)
{
  const auto [first, last] = message_bytes(bytes);
  Schema schema;
  Decoder decoder;
  const auto [endianness, after_schema] = read_schema(decoder, bytes, first, last, schema);
  const std::string name(column);
  std::int64_t index = -1;
  for (std::int64_t i = 0; i < schema.arrow.n_children; ++i)
    if (schema.arrow.children[i]->name != nullptr && name == schema.arrow.children[i]->name) index = i;
  if (index < 0) throw IOError("no column '" + name + "'");
  std::vector<std::vector<std::vector<std::byte>>> out;
  std::int64_t at = after_schema;
  for (auto message = next_message(decoder, bytes, at, last); message;
       message = next_message(decoder, bytes, at, last)) {
    Decoded batch;
    decode_batch(schema.arrow, endianness, bytes, *message, { { index, name.c_str() } }, out.size(), batch);
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
    namespace llfio = LLFIO_V2_NAMESPACE;
    const auto value = [](auto &&result, const char *step) {
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
    return detail::read_ipc(bytes, std::move(file), dtype);
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

} // namespace dtwc::io
