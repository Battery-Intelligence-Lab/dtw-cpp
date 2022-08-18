#include <iostream>
#include <vector>
#include <array>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <thread>
#include <iterator>
#include <cassert>

#include "dtwc.hpp"


int main()
{

  using namespace dtwc;


  std::clock_t t_start(std::clock());

  std::cout << ROOT_FOLDER << '\n';


  auto [p_vec, p_names] = load_data<Tdata>(settings::path, settings::Ndata_max);


  std::cout << "Data loading finished at " << get_duration(t_start) << "\n";

  dtwc::VecMatrix<Tdata> DTWdist(p_vec.size(), p_vec.size(), -1); // For distance memoization.

  // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating

  auto DTWdistByInd = [&DTWdist, p_vec = p_vec](int i, int j) {
    if (DTWdist(i, j) < 0) {
      if constexpr (settings::band == 0) {
        DTWdist(j, i) = DTWdist(i, j) = dtwFun2<Tdata>(p_vec[i], p_vec[j]);
      } else {
        DTWdist(j, i) = DTWdist(i, j) = dtwFunBanded_Act<Tdata>(p_vec[i], p_vec[j], settings::band);
      }
    }
    return DTWdist(i, j);
  };

  fillDistanceMatrix(DTWdistByInd, p_vec.size()); // Otherwise takes time.

  std::string DistMatrixName = "DTW_matrix.csv";
  writeMatrix(DTWdist, DistMatrixName);
  // DTWdist.print();
  std::cout << "Finished all tasks in " << get_duration(t_start) << "\n";

  std::cout << "Band used " << settings::band << "\n\n\n";
}
