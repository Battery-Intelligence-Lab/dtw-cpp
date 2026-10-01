/**
 * @file read_data.cpp
 * @brief dtwc::read_data and the input classification dtwc_cl shares with it (see read_data.hpp).
 */

#include "read_data.hpp"

#include "../DataLoader.hpp"
#include "../base/error.hpp"
#include "../fileOperations.hpp"

#ifdef DTWC_HAS_ARROW
#include "arrow_c_data.hpp"

#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

#include <charconv>
#endif
#ifdef DTWC_HAS_PARQUET
#include "parquet_chunk_reader.hpp"
#endif

#include <exception>
#include <string>
#include <system_error>
#include <utility>

namespace dtwc {
namespace {

std::string lower_extension(const fs::path &path)
{
  return text_io_detail::lower_ascii(path_to_utf8(path.extension()));
}

bool is_parquet_file(const fs::path &path)
{
  const auto ext = lower_extension(path);
  return ext == ".parquet" || ext == ".pq";
}

#ifdef DTWC_HAS_ARROW
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
  std::vector<int> columns{ schema->GetFieldIndex("data") };
  if (columns.front() < 0)
    throw IOError("no 'data' column; write the series as a List or LargeList of Float32/Float64 named 'data'.");
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
  return io::data_from_arrow_stream(&stream, ndim);
}
#endif

} // namespace

std::vector<fs::path> parquet_files(const fs::path &path)
{
  std::error_code ec;
  if (!fs::is_directory(path, ec)) return is_parquet_file(path) ? std::vector<fs::path>{ path } : std::vector<fs::path>{};
  std::vector<fs::path> files;
  try {
    files = sorted_directory_files(path);
  } catch (const fs::filesystem_error &e) {
    throw IOError("load: cannot list '" + path_to_utf8(path) + "': " + e.what());
  }
  std::erase_if(files, [](const fs::path &file) { return !is_parquet_file(file); });
  return files;
}

// Each rejection lives in the build that lacks the reader: a check compiled only into the build that has it would let
// the file reach the text reader in exactly the build that cannot read it.
InputFormat input_format(const fs::path &path)
{
  if (!parquet_files(path).empty()) {
#ifdef DTWC_HAS_PARQUET
    return InputFormat::Parquet;
#else
    throw IOError("Parquet input (.parquet/.pq) requires a build with Arrow/Parquet (-DDTWC_ENABLE_ARROW=ON). This "
                  "binary was built without Parquet support; convert the input to CSV/TSV or use an Arrow-enabled "
                  "build.");
#endif
  }
  const auto ext = lower_extension(path);
  if (ext == ".arrow" || ext == ".ipc" || ext == ".feather") {
#ifdef DTWC_HAS_ARROW
    return InputFormat::ArrowIPC;
#else
    throw IOError("Arrow IPC input (.arrow/.ipc/.feather) requires a build with Arrow (-DDTWC_ENABLE_ARROW=ON). This "
                  "binary was built without Arrow support; convert the input to CSV/TSV or use an Arrow-enabled "
                  "build.");
#endif
  }
  return InputFormat::Text;
}

void require_reader_options(std::optional<InputFormat> format, index_t skip_cols, index_t skip_rows, char delimiter,
                            std::string_view column)
{
  if (!column.empty() && format != InputFormat::Parquet)
    throw InvalidInput("--column selects a Parquet column and cannot be honoured for this input; drop --column, or "
                       "pass a .parquet/.pq file or directory.");
  if ((skip_cols != 0 || skip_rows != 0 || delimiter != '\0') && format != InputFormat::Text)
    throw InvalidInput("--skip-rows, --skip-cols and --delimiter (skip_rows, skip_cols, delimiter) are CSV/TSV parsing "
                       "options and cannot be honoured for this input; drop them, or pass a text input.");
}

Data read_data(const fs::path &path, index_t skip_cols, index_t skip_rows, char delimiter, const std::string &column)
{
  const InputFormat format = input_format(path);
  require_reader_options(format, skip_cols, skip_rows, delimiter, column);
  try {
#ifdef DTWC_HAS_PARQUET
    if (format == InputFormat::Parquet) {
      std::vector<std::vector<data_t>> series;
      std::vector<std::string> names;
      for (const auto &file : parquet_files(path)) io::ParquetChunkReader(file, column).read_all(series, names);
      return Data(std::move(series), std::move(names));
    }
#endif
#ifdef DTWC_HAS_ARROW
    if (format == InputFormat::ArrowIPC) return read_arrow_ipc(path);
#endif
    DataLoader loader{ path }; // text: input_format() refused every format this build has no reader for
    loader.start_column(skip_cols).start_row(skip_rows).verbosity(0);
    if (delimiter != '\0') loader.delimiter(delimiter);
    return loader.load();
  } catch (const IOError &e) {
    throw IOError("load: failed to read '" + path_to_utf8(path) + "': " + e.what());
  } catch (const Error &) {
    throw;
  } catch (const std::exception &e) {
    throw IOError("load: failed to read '" + path_to_utf8(path) + "': " + e.what());
  }
}

} // namespace dtwc
