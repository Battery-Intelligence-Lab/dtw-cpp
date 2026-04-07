/**
 * @file fileOperations.hpp
 * @brief Functions for file operations
 *
 * @details This header file declares various functions for performing file operations such as
 * reading and writing data to/from files. It includes functions to handle comma-separated values (CSV) files,
 * read data into vectors or Armadillo matrices, and save matrices to files.
 * It provides the functionality to ignore Byte Order Marks (BOM) in text files,
 * read specific rows and columns from files, and handle data from directories or batch files.
 *
 * @date 21 Jan 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include "settings.hpp" // for resultsPath

#include <cassert>    // for assert
#include <chrono>     // for filesystem
#include <cstdlib>    // for size_t
#include <filesystem> // for operator<<, path, operator/, directory_iterator
#include <iostream>   // for operator<<, ifstream, basic_ostream, operator>>
#include <string>     // for string, getline, to_string
#include <utility>    // for pair
#include <vector>     // for vector
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept> // for std::runtime_error

#include <rapidcsv.h>

namespace dtwc {

namespace fs = std::filesystem;

/**
 * @brief Ignores Byte Order Mark (BOM) in UTF-8 encoded files.
 *
 * @param in Reference to the input stream to process.
 */
inline void ignoreBOM(std::istream &in)
{
  char BOMchars[] = { '\xEF', '\xBB', '\xBF' };
  int seek = 0;
  char c = '.';
  while (in >> c) {
    if (BOMchars[seek] != c) {
      in.putback(c);
      break;
    }
    seek++;
  }
  in.clear(); // Clear EOF flag if end of file was reached
}

/**
 * @brief Reads a file and returns the data as a vector of a specified type.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param name Path of the file to read.
 * @param start_row Starting row index for reading the data (default is 0).
 * @param start_col Starting column index for reading the data (default is 0).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::vector<data_t> A vector containing the read data.
 */
template <typename data_t>
auto readFile(const fs::path &name, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::ifstream in(name, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    throw std::runtime_error("Error in readFile: File " + name.string() + " could not be opened.");
  }

  ignoreBOM(in);

  // https://stackoverflow.com/questions/70497719/read-from-comma-separated-file-into-vector-of-objects
  std::string line{};
  char c = '.';

  for (int i = 0; i < start_row; i++) // Skip first start_row rows to start from start_row.
    std::getline(in, line);


  data_t temp, p_i;
  std::vector<data_t> p;
  p.reserve(10000);

  while (std::getline(in, line)) {
    std::istringstream iss(line);

    for (int i = 0; i < start_col; i++) // Skip first start_col columns to start from start_col.
    {
      iss >> temp;
      if (delimiter != ' ' && delimiter != '\t') // These we do not need to remove from stream.
        iss >> c;
    }

    iss >> p_i; // Finally we got it!  % #TODO does not work for many-column arrays.
    p.push_back(p_i);
  }

  p.shrink_to_fit();
  return p;
}

/**
 * @brief Loads all files from a given folder and returns their data as vectors along with file names.
 *
 * @tparam data_t The data type of the elements to be read.
 * @tparam Tpath Type of the folder path (auto-deduced).
 * @param folder_path Path of the folder containing the files.
 * @param Ndata Maximum number of data points to read from each file (default is -1, read all data).
 * @param verbose Verbosity level for logging output (default is 1).
 * @param start_row Starting row index for reading the data (default is 0).
 * @param start_col Starting column index for reading the data (default is 0).
 * @param delimiter Delimiter character used in the files (default is ',').
 * @return std::pair<std::vector<std::vector<data_t>>, std::vector<std::string>> A pair containing vectors of data and corresponding file names.
 */
template <typename data_t, typename Tpath>
auto load_folder(Tpath &folder_path, int Ndata = -1, int verbose = 1, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::cout << "Reading data:" << '\n';

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  int i_data = 0;
  for (const auto &entry : fs::directory_iterator(folder_path)) {

    auto p = readFile<data_t>(entry.path(), start_row, start_col, delimiter);

    if (verbose >= 2 || (verbose == 1 && p.empty()))
      std::cout << entry.path() << "\tSize: " << p.size() << '\n';

    p_vec.push_back(std::move(p));
    p_names.push_back(entry.path().stem().string());

    i_data++;
    if (i_data == Ndata) break;
  }

  std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}

/**
 * @brief Loads batch data from a single file and returns the data as vectors.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param file_path Path of the file containing batch data.
 * @param Ndata Maximum number of data points to read (default is -1, read all data).
 * @param verbose Verbosity level for logging output (default is 1).
 * @param start_row Starting row index for reading the data (default is 0).
 * @param start_col Starting column index for reading the data (default is 0).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::pair<std::vector<std::vector<data_t>>, std::vectorstd::string> A pair containing vectors of data and corresponding identifiers.
 */
template <typename data_t>
auto load_batch_file(fs::path &file_path, int Ndata = -1, int verbose = 1, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::cout << "Reading data:" << '\n';

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  auto myAbsPath = fs::absolute(file_path);

  std::ifstream in(file_path, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    throw std::runtime_error("Error in load_batch_file: File " + file_path.string() + " could not be opened.");
  }

  std::string line;
  int line_no{ 0 };
  int n_rows{ 0 };
  while ((Ndata == -1 || n_rows < Ndata) && std::getline(in, line)) //!< Read file.
  {
    if (line_no++ < start_row) // Skip first rows.
      continue;

    n_rows++;

    std::vector<data_t> p;
    p.reserve(10000);
    std::istringstream in_line(line);
    data_t temp, p_i;
    char c;

    for (int i = 0; i < start_col; i++) // Skip first start_col columns to start from start_col.
    {
      in_line >> temp;
      if (delimiter != ' ' && delimiter != '\t') // These we do not need to remove from stream.
        in_line >> c;
    }

    while (in_line >> p_i) {
      p.push_back(p_i);
      if (delimiter != ' ' && delimiter != '\t') // These we do not need to remove from stream.
        in_line >> c;
    }

    p.shrink_to_fit();

    if (verbose >= 2 || (verbose == 1 && p.empty()))
      std::cout << file_path << '\t' << "data: " << n_rows << " Size: " << p.size() << '\n';

    p_vec.push_back(std::move(p));
    p_names.push_back(std::to_string(n_rows));
  }

  std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}

// ============================================================================
// RapidCSV-based functions for robust multi-column CSV parsing
// ============================================================================

/**
 * @brief Reads a multi-column CSV file and returns data as a 2D vector using rapidcsv.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param file_path Path of the CSV file to read.
 * @param has_header Whether the first row is a header (default is false).
 * @param has_row_names Whether the first column contains row names (default is false).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::vector<std::vector<data_t>> A 2D vector where each inner vector is a row.
 */
template <typename data_t>
auto readCSV(const fs::path &file_path, bool has_header = false, bool has_row_names = false, char delimiter = ',')
{
  if (!fs::exists(file_path)) {
    throw std::runtime_error("Error in readCSV: File " + file_path.string() + " does not exist.");
  }

  rapidcsv::LabelParams labels(
    has_header ? 0 : -1,    // Row header index (-1 = no header)
    has_row_names ? 0 : -1  // Column header index (-1 = no row names)
  );
  rapidcsv::SeparatorParams sep(delimiter);

  rapidcsv::Document doc(file_path.string(), labels, sep);

  std::vector<std::vector<data_t>> result;
  const size_t numRows = doc.GetRowCount();
  result.reserve(numRows);

  for (size_t i = 0; i < numRows; ++i) {
    result.push_back(doc.GetRow<data_t>(i));
  }

  return result;
}

/**
 * @brief Reads a multi-column CSV file where each row is a time series.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param file_path Path of the CSV file to read.
 * @param max_rows Maximum number of rows to read (-1 = all rows).
 * @param has_header Whether the first row is a header (default is false).
 * @param label_col Column index containing labels (-1 = no labels, use row numbers).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::pair<std::vector<std::vector<data_t>>, std::vector<std::string>> Data and names.
 */
template <typename data_t>
auto readTimeSeriesCSV(const fs::path &file_path, int max_rows = -1, bool has_header = false,
                       int label_col = -1, char delimiter = ',')
{
  if (!fs::exists(file_path)) {
    throw std::runtime_error("Error in readTimeSeriesCSV: File " + file_path.string() + " does not exist.");
  }

  rapidcsv::LabelParams labels(has_header ? 0 : -1, -1);
  rapidcsv::SeparatorParams sep(delimiter);

  rapidcsv::Document doc(file_path.string(), labels, sep);

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  const size_t numRows = doc.GetRowCount();
  const size_t rowsToRead = (max_rows < 0) ? numRows : std::min(static_cast<size_t>(max_rows), numRows);

  p_vec.reserve(rowsToRead);
  p_names.reserve(rowsToRead);

  for (size_t i = 0; i < rowsToRead; ++i) {
    auto row = doc.GetRow<std::string>(i);

    // Extract name from label column or use row number
    std::string name;
    if (label_col >= 0 && static_cast<size_t>(label_col) < row.size()) {
      name = row[label_col];
    } else {
      name = std::to_string(i + 1);
    }
    p_names.push_back(name);

    // Convert remaining columns to data
    std::vector<data_t> series;
    series.reserve(row.size());
    for (size_t j = 0; j < row.size(); ++j) {
      if (static_cast<int>(j) == label_col) continue; // Skip label column
      try {
        if constexpr (std::is_same_v<data_t, double>) {
          series.push_back(std::stod(row[j]));
        } else if constexpr (std::is_same_v<data_t, float>) {
          series.push_back(std::stof(row[j]));
        } else if constexpr (std::is_same_v<data_t, int>) {
          series.push_back(std::stoi(row[j]));
        } else {
          series.push_back(static_cast<data_t>(std::stod(row[j])));
        }
      } catch (const std::exception &) {
        // Skip non-numeric values
      }
    }
    p_vec.push_back(std::move(series));
  }

  return std::pair(std::move(p_vec), std::move(p_names));
}

/**
 * @brief Reads a single column from a CSV file.
 *
 * @tparam data_t The data type of the elements to be read.
 * @param file_path Path of the CSV file to read.
 * @param column Column index to read (0-based).
 * @param has_header Whether the first row is a header (default is false).
 * @param delimiter Delimiter character used in the file (default is ',').
 * @return std::vector<data_t> A vector containing the column data.
 */
template <typename data_t>
auto readCSVColumn(const fs::path &file_path, size_t column, bool has_header = false, char delimiter = ',')
{
  if (!fs::exists(file_path)) {
    throw std::runtime_error("Error in readCSVColumn: File " + file_path.string() + " does not exist.");
  }

  rapidcsv::LabelParams labels(has_header ? 0 : -1, -1);
  rapidcsv::SeparatorParams sep(delimiter);

  rapidcsv::Document doc(file_path.string(), labels, sep);

  return doc.GetColumn<data_t>(column);
}

} // namespace dtwc