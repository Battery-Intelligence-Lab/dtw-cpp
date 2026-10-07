/**
 * @file read_data.cpp
 * @brief dtwc::read_data and the input classification dtwc_cl shares with it (see read_data.hpp).
 */

#include "read_data.hpp"

#include "../DataLoader.hpp"
#include "../base/error.hpp"
#include "../fileOperations.hpp"

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
    throw IOError("load: failed to read '" + path_to_utf8(path) + "': cannot list it: " + e.what());
  }
  std::erase_if(files, [](const fs::path &file) { return !is_parquet_file(file); });
  return files;
}

bool is_arrow_ipc(const fs::path &path)
{
  std::error_code ec;
  if (fs::is_directory(path, ec)) return false; // a folder is text whatever its name
  const auto ext = lower_extension(path);
  return ext == ".arrow" || ext == ".ipc" || ext == ".feather";
}

// This reader has no Arrow, in any build: Parquet and Arrow IPC are refused here, never read as text. dtwc_cl reads
// them through dtwc_io (read_arrow.hpp) in a build with Arrow, which it asks first.
InputFormat input_format(const fs::path &path)
{
  if (!parquet_files(path).empty())
    throw IOError("Parquet input (.parquet/.pq) requires a build with Arrow/Parquet (-DDTWC_ENABLE_ARROW=ON). This "
                  "binary was built without Parquet support; convert the input to CSV/TSV or use an Arrow-enabled "
                  "build.");
  if (is_arrow_ipc(path))
    throw IOError("Arrow IPC input (.arrow/.ipc/.feather) requires a build with Arrow (-DDTWC_ENABLE_ARROW=ON). This "
                  "binary was built without Arrow support; convert the input to CSV/TSV or use an Arrow-enabled "
                  "build.");
  return InputFormat::Text;
}

void require_reader_options(std::optional<InputFormat> format, index_t skip_cols, index_t skip_rows, char delimiter,
                            std::string_view column)
{
  if (skip_rows < 0 || skip_cols < 0) // here, for every format: the text reader would read skip_rows -1 as 0
    throw InvalidInput("--skip-rows and --skip-cols (skip_rows, skip_cols) count leading rows and columns and must be "
                       "non-negative, got " + std::to_string(skip_rows) + " and " + std::to_string(skip_cols) + ".");
  if (!column.empty() && format != InputFormat::Parquet)
    throw InvalidInput("--column selects a Parquet column and cannot be honoured for this input; drop --column, or "
                       "pass a .parquet/.pq file or directory.");
  if (delimiter != '\0' && format != InputFormat::Text)
    throw InvalidInput("--delimiter (delimiter) splits CSV/TSV text into fields and cannot be honoured for this input; "
                       "drop it, or pass a text input.");
  if ((skip_cols != 0 || skip_rows != 0) && format != InputFormat::Text && format != InputFormat::Parquet)
    throw InvalidInput("--skip-rows and --skip-cols (skip_rows, skip_cols) drop the leading rows and columns of CSV/TSV "
                       "text or Parquet and cannot be honoured for this input; drop them.");
}

Data read_data(const fs::path &path, index_t skip_cols, index_t skip_rows, char delimiter, const std::string &column)
{
  const InputFormat format = input_format(path);
  require_reader_options(format, skip_cols, skip_rows, delimiter, column);
  try {
    DataLoader loader{ path }; // text: input_format() refused Parquet and Arrow IPC
    loader.start_column(skip_cols).start_row(skip_rows).verbosity(0);
    if (delimiter != '\0') loader.delimiter(delimiter);
    return loader.load();
  } catch (...) {
    detail::rethrow_naming_the_file(path);
  }
}

void detail::rethrow_naming_the_file(const fs::path &path)
{
  try {
    throw;
  } catch (const IOError &e) {
    throw IOError("load: failed to read '" + path_to_utf8(path) + "': " + e.what());
  } catch (const Error &) {
    throw;
  } catch (const std::exception &e) {
    throw IOError("load: failed to read '" + path_to_utf8(path) + "': " + e.what());
  }
}

std::string detail::default_name(const fs::path &input)
{
  // "data/" names its folder; "." and ".." name nothing a file could be called after.
  // UTF-8, which every writer turns back into a path with utf8_to_path(), losslessly.
  const std::string stem = path_to_utf8((input.has_filename() ? input : input.parent_path()).stem());
  return stem.empty() || stem == "." || stem == ".." ? "dataset" : stem;
}

} // namespace dtwc
