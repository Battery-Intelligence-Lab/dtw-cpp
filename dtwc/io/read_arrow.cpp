/**
 * @file read_arrow.cpp
 * @brief dtwc::io::read_arrow and arrow_format (see read_arrow.hpp): the readers that need Arrow, in dtwc_io.
 */

#include "read_arrow.hpp"

#include "arrow_c_data.hpp"
#include "../base/error.hpp"
#include "../fileOperations.hpp"
#ifdef DTWC_HAS_PARQUET
#include "parquet_chunk_reader.hpp"
#endif

#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

#include <charconv>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace dtwc::io {
namespace {

void check_arrow(const arrow::Status &status, const char *what)
{
  if (!status.ok()) throw IOError(std::string(what) + ": " + status.ToString());
}

/// Arrow C++ maps the file and exports its 'data' and 'name' columns, every record
/// batch, as one C stream, which the one Arrow converter reads: the values are
/// copied once, into the Data.
Data read_arrow_ipc(const fs::path &path)
{
  // Arrow takes a UTF-8 path on every platform; path::string() is the ANSI code page on Windows.
  auto file = arrow::io::MemoryMappedFile::Open(path_to_utf8(path), arrow::io::FileMode::READ);
  check_arrow(file.status(), "Arrow IPC open");
  auto reader = arrow::ipc::RecordBatchFileReader::Open(*file);
  check_arrow(reader.status(), "Arrow IPC footer");
  const auto schema = (*reader)->schema();

  // Features per time step: a whole positive integer, nothing else ("2x" is not
  // 2, "-1" does not wrap, and 0 would divide every series length by zero).
  std::size_t ndim = 1;
  if (const auto &metadata = schema->metadata(); metadata != nullptr) {
    if (const int key = metadata->FindKey("ndim"); key >= 0) {
      const std::string &text = metadata->value(key);
      const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), ndim);
      if (ec != std::errc{} || end != text.data() + text.size() || ndim == 0)
        throw IOError("schema metadata 'ndim' must be a positive integer, got '" + text
                      + "'. Set it to the number of features per timestep (1 for univariate data).");
    }
  }

  // The series are the 'data' column, named by the 'name' column when there is one.
  // A column of another type is a bad file (IOError), as it is for Parquet.
  const int data = schema->GetFieldIndex("data");
  if (data < 0)
    throw IOError("no 'data' column; write the series as a List or LargeList of Float32/Float64 named 'data'.");
  const auto &data_type = schema->field(data)->type();
  const bool list = data_type->id() == arrow::Type::LIST || data_type->id() == arrow::Type::LARGE_LIST;
  const auto value = list ? data_type->field(0)->type()->id() : arrow::Type::NA;
  if (value != arrow::Type::FLOAT && value != arrow::Type::DOUBLE)
    throw IOError("the 'data' column must be a List or LargeList of Float32/Float64, got " + data_type->ToString()
                  + ".");
  std::vector<int> columns{ data };
  if (const int name = schema->GetFieldIndex("name"); name >= 0) {
    const auto &type = schema->field(name)->type();
    if (type->id() != arrow::Type::STRING && type->id() != arrow::Type::LARGE_STRING)
      throw IOError("the 'name' column must be Utf8 or LargeUtf8, got " + type->ToString()
                    + ". Write the names as strings, or drop the column.");
    columns.push_back(name);
  }

  auto table = (*reader)->ToTable();
  check_arrow(table.status(), "Arrow IPC read");
  auto selected = (*table)->SelectColumns(columns);
  check_arrow(selected.status(), "Arrow IPC columns");
  ArrowArrayStream stream;
  check_arrow(arrow::ExportRecordBatchReader(std::make_shared<arrow::TableBatchReader>(*selected), &stream),
              "Arrow IPC export");
  return data_from_arrow_stream(&stream, ndim);
}

} // namespace

std::optional<InputFormat> arrow_format(const fs::path &path)
{
#ifdef DTWC_HAS_PARQUET
  if (!parquet_files(path).empty()) return InputFormat::Parquet;
#endif
  if (is_arrow_ipc(path)) return InputFormat::ArrowIPC;
  return std::nullopt; // text, or Parquet in a build without it: the core's reader takes it or refuses it
}

Data read_arrow(const fs::path &path, const std::string &column, index_t skip_cols, index_t skip_rows)
{
  const auto format = arrow_format(path);
  if (!format) return read_data(path, skip_cols, skip_rows, '\0', column);
  require_reader_options(format, skip_cols, skip_rows, '\0', column);
  try {
#ifdef DTWC_HAS_PARQUET
    if (*format == InputFormat::Parquet) {
      std::vector<std::vector<data_t>> series;
      std::vector<std::string> names;
      for (const auto &file : parquet_files(path))
        ParquetChunkReader(file, column, skip_cols, skip_rows).read_all(series, names);
      return Data(std::move(series), std::move(names));
    }
#endif
    return read_arrow_ipc(path);
  } catch (...) {
    dtwc::detail::rethrow_naming_the_file(path); // dtwc::io::detail is the Parquet reader's
  }
}

} // namespace dtwc::io
