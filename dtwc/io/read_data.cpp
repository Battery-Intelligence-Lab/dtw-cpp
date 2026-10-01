/**
 * @file read_data.cpp
 * @brief dtwc::read_data and the input classification dtwc_cl shares with it (see read_data.hpp).
 */

#include "read_data.hpp"

#include "../DataLoader.hpp"
#include "../base/error.hpp"
#include "../fileOperations.hpp"

#ifdef DTWC_HAS_ARROW
#include "arrow_ipc_reader.hpp"
#endif
#ifdef DTWC_HAS_PARQUET
#include "parquet_reader.hpp"
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
    if (format == InputFormat::Parquet)
      return fs::is_directory(path) ? io::load_parquet_directory(path, column) : io::load_parquet_file(path, column);
#endif
#ifdef DTWC_HAS_ARROW
    if (format == InputFormat::ArrowIPC) { // copied out of the map
      auto source = io::ArrowIPCDataSource::open(path);
      std::vector<std::vector<data_t>> series(source.size());
      for (std::size_t i = 0; i < series.size(); ++i) {
        const auto values = source.series(i);
        series[i].assign(values.begin(), values.end());
      }
      return Data(std::move(series), source.all_names(), source.ndim());
    }
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
