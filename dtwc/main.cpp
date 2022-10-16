#include "dtwc.hpp"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>

int main()
{
  using namespace dtwc;
  dtwc::Clock clk;

  int Ndata_max = 10; // Load 10 data maximum.

  auto [p_vec, p_names] = load_data<Tdata, true>(settings::path, Ndata_max);

  std::cout << "Data loading finished at " << clk << "\n";

  dtwc::VecMatrix<Tdata> DTWdist(p_vec.size(), p_vec.size(), -1); // For distance memoization.

  // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating

  auto DTWdistByInd = [&DTWdist, p_vec = p_vec](int i, int j) {
    if (DTWdist(i, j) < 0) {
      if constexpr (settings::band == 0) {
        DTWdist(j, i) = DTWdist(i, j) = dtwFun_L<Tdata>(p_vec[i], p_vec[j]);
      } else {
        DTWdist(j, i) = DTWdist(i, j) = dtwFunBanded_Act<Tdata>(p_vec[i], p_vec[j], settings::band); // dtwFunBanded_Act_L faster and more accurate.
      }
    }
    return DTWdist(i, j);
  };

  fillDistanceMatrix(DTWdistByInd, p_vec.size()); // Otherwise takes time.

  std::string DistMatrixName = "DTW_matrix.csv";
  writeMatrix(DTWdist, DistMatrixName);
  // DTWdist.print();
  std::cout << "Finished all tasks in " << clk << "\n";
  std::cout << "Band used " << settings::band << "\n\n\n";
}
