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

template <typename data_t, typename Tpath>
std::vector<std::vector<data_t>> readCSVfromFolder(const Tpath &path)
{
  auto Nfiles = (std::size_t)std::distance(fs::directory_iterator{ path }, fs::directory_iterator{});
  std::vector<std::vector<data_t>> allFiles(Nfiles);
}


template <typename data_t, typename T>
std::vector<data_t> readFile(const T &name)
{
  std::ifstream in(name, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    std::cerr << "Error in readFile. File " << name << " could not be opened.\n";
    throw 2;
  }

  char c = '.';

  while (in >> c) // Ignore byte order mark (BOM) at the beginning
    if (c == '0')
      break;

  data_t x, y;
  // in >> y; // Read the 0.

  std::vector<data_t> p;

  p.reserve(10000);

  while (in >> x >> c >> y) // Read file.
    p.push_back(y);

  p.shrink_to_fit();

  return p;
}


template <typename data_t, typename Tpath>
auto load_data(Tpath &path, int Ndata = -1, bool print = false, fs::path out_folder = settings::resultsPath)
{
  std::cout << "Reading data:" << std::endl;

  std::ofstream out(out_folder + "dataOrder.csv", std::ios_base::out);
  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  int i_data = 0;
  for (const auto &entry : fs::directory_iterator(path)) {

    auto p = readFile<data_t>(entry.path());

    if (print) {
      std::cout << entry.path() << '\t'
                << "Size: " << p.size() << " Capacity: " << p.capacity() << '\n';
    }


    assert(p.size() > 2);

    p_vec.push_back(std::move(p));

    out << i_data << ',' << entry.path().filename() << '\n';
    p_names.push_back(entry.path().stem().string());

    i_data++;
    if (i_data == Ndata)
      break;
  }

  std::cout << p_vec.size() << " battery data is read.\n";

  return std::pair(p_vec, p_names);
}

template <typename data_t, typename Tpath>
auto load_tsv(Tpath &file_path, int Ndata = -1, bool print = false)
{
  std::cout << "Reading data:" << std::endl;

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  int i_data = 0;

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
    n_rows++;
    std::vector<data_t> p;
    p.reserve(10000);
    std::istringstream in_line(line);
    double x_i;

    in_line >> x_i; // Ignore first element since it is cluster no for verification.
    while (in_line >> x_i) {
      p.push_back(x_i);
    }

    p.shrink_to_fit();

    if (print) {
      std::cout << file_path << '\t' << "data: " << n_rows
                << " Size: " << p.size() << " Capacity: " << p.capacity() << '\n';
    }

    p_vec.push_back(std::move(p));
    p_names.push_back(std::to_string(n_rows));
  }

  std::cout << p_vec.size() << " battery data is read.\n";

  return std::pair(p_vec, p_names);
}


template <typename data_t>
void writeMatrix(dtwc::VecMatrix<data_t> &matrix, const std::string &name, fs::path out_folder = settings::resultsPath)
{
  std::ofstream myFile(out_folder + name, std::ios_base::out);

  for (int i = 0; i < matrix.rows(); i++) {
    for (int j = 0; j < (matrix.cols() - 1); j++)
      myFile << matrix(i, j) << ',';

    myFile << matrix(i, (matrix.cols() - 1)) << '\n';
  }

  myFile.close();
}


template <typename data_t>
void readMatrix(dtwc::VecMatrix<data_t> &matrix, const std::string &name)
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