// Vk: 2022.01.21

#pragma once

#include "settings.hpp"
#include "utility.hpp"
#include "dataTypes.hpp"


#include <vector>
#include <filesystem>
#include <cstdlib>
#include <string>
#include <limits>
#include <sstream>
#include <fstream>
#include <cassert>

namespace dtwc {

namespace fs = std::filesystem;

inline void ignoreBOM(std::ifstream &in)
{
  char c = '.';
  do {
    in >> c;
  } while (c < 45); //!< Ignore byte order mark (BOM) at the beginning

  in.putback(c);
}


template <typename data_t>
void generic_reader(fs::path path)
{
  std::ifstream in(path, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    std::cerr << "Error in generic_reader. File " << path << " could not be opened.\n";
    throw 2;
  }

  ignoreBOM(in);

  std::vector<data_t> p;
  std::vector<size_t> rowStarts;
}

template <typename data_t>
auto readFile(const fs::path &name, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::ifstream in(name, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    std::cerr << "Error in readFile. File " << name << " could not be opened.\n";
    throw 2;
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


template <typename data_t, typename Tpath>
auto load_folder(Tpath &folder_path, int Ndata = -1, bool print = false, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::cout << "Reading data:" << std::endl;

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  int i_data = 0;
  for (const auto &entry : fs::directory_iterator(folder_path)) {

    auto p = readFile<data_t>(entry.path(), start_row, start_col, delimiter);

    if (print || p.empty())
      std::cout << entry.path() << "\tSize: " << p.size() << '\n';

    assert(p.size() > 2);

    p_vec.push_back(std::move(p));
    p_names.push_back(entry.path().stem().string());

    i_data++;
    if (i_data == Ndata) break;
  }

  std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}

template <typename data_t>
auto load_batch_file(fs::path &file_path, int Ndata = -1, bool print = false, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::cout << "Reading data:" << std::endl;

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  std::ifstream in(file_path, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    std::cerr << "Error in readFile. File " << file_path << " could not be opened.\n";
    throw 2;
  }

  std::string line;
  int n_rows{ 0 };
  while ((Ndata == -1 || n_rows < Ndata) && std::getline(in, line)) //!< Read file.
  {
    if (n_rows < start_row) // Skip first rows.
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

    if (print || p.empty()) // It should say if p is empty!
      std::cout << file_path << '\t' << "data: " << n_rows << " Size: " << p.size() << '\n';

    p_vec.push_back(std::move(p));
    p_names.push_back(std::to_string(n_rows));
  }

  std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}


template <typename data_t>
void writeMatrix(dtwc::VecMatrix<data_t> &matrix, const std::string &name, fs::path out_folder = settings::resultsPath)
{
  std::ofstream myFile(out_folder / name, std::ios_base::out);

  if (!myFile.good()) // check if we could open the file
  {
    std::cerr << "Error in writeMatrix. File " << out_folder / name << " could not be opened.\n"
              << "Please ensure that you have the folder " << out_folder << " and file is not open in any other program.\n";
    throw 2;
  }

  for (int i = 0; i < matrix.rows(); i++) {
    for (int j = 0; j < (matrix.cols() - 1); j++)
      myFile << matrix(i, j) << ',';

    myFile << matrix(i, (matrix.cols() - 1)) << '\n';
  }

  myFile.close();
}

template <typename data_t>
void readMatrix(dtwc::VecMatrix<data_t> &matrix, const fs::path &name)
{
  std::ifstream in(name, std::ios_base::in);
  if (!in.good()) // check if we could open the file
    std::cout << "File " << name << " is not found. Matrix will not be written.\n";

  matrix.data.clear();
  std::string x_str;
  data_t x;
  while (in >> x) {
    if (in.peek() == ',')
      in.ignore();

    matrix.data.push_back(x);
  }

  if (matrix.size() != matrix.data.size())
    std::cout << "Warning! Given file and sizes are not compatible.\n";

  if (matrix.data.size() < matrix.size())
    throw 1;
}


} // namespace dtwc