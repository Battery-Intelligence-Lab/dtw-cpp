/**
 * @file read_arrow.cpp
 * @brief dtwc::io::read_arrow and arrow_format (see read_arrow.hpp): Parquet, read with Arrow C++, in dtwc_io; Arrow
 *        IPC goes to the core's reader.
 * @author Volkan Kumtepeli
 */

#include "read_arrow.hpp"

#include "../fileOperations.hpp"
#ifdef DTWC_HAS_PARQUET
#include "parquet_chunk_reader.hpp"
#endif

#include <string>
#include <utility>
#include <vector>

namespace dtwc::io {
namespace {

#ifdef DTWC_HAS_PARQUET
/// The series of a Parquet file, or of a folder's files in order, each value as T.
template <typename T>
Data read_parquet(const fs::path &path, const std::string &column, index_t skip_cols, index_t skip_rows)
{
  std::vector<std::vector<T>> series;
  std::vector<std::string> names;
  for (const auto &file : parquet_files(path))
    ParquetChunkReader(file, column, skip_cols, skip_rows).read_all(series, names);
  return Data(std::move(series), std::move(names));
}
#endif

} // namespace

std::optional<InputFormat> arrow_format(const fs::path &path)
{
#ifdef DTWC_HAS_PARQUET
  if (!parquet_files(path).empty()) return InputFormat::Parquet;
#endif
  if (is_arrow_ipc(path)) return InputFormat::ArrowIPC;
  return std::nullopt; // text, or Parquet in a build without it: the core's reader takes it or refuses it
}

Data read_arrow(const fs::path &path, const std::string &column, index_t skip_cols, index_t skip_rows,
                core::Precision dtype)
{
#ifdef DTWC_HAS_PARQUET
  if (arrow_format(path) == InputFormat::Parquet) {
    require_reader_options(InputFormat::Parquet, skip_cols, skip_rows, '\0', column);
    try {
      return dtype == core::Precision::Float32 ? read_parquet<float>(path, column, skip_cols, skip_rows)
                                               : read_parquet<data_t>(path, column, skip_cols, skip_rows);
    } catch (...) {
      dtwc::detail::rethrow_naming_the_file(path); // dtwc::io::detail is the Parquet reader's
    }
  }
#endif
  return read_data(path, skip_cols, skip_rows, '\0', column, dtype); // Arrow IPC, text: the core's reader
}

} // namespace dtwc::io
