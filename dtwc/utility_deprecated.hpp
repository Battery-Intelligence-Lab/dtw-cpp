/*
 * utility_deprecated.hpp
 *
 * deprecated utility functions

 * Created on: 15 Dec 2021
 * Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include <iostream>
#include <vector>
#include <array>

#include <fstream>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <thread>
#include <iterator>
#include <memory>
#include <execution>
#include <tuple>
#include <iomanip>

#include "settings.hpp"
#include "dataTypes.hpp"
#include "fileOperations.hpp"


namespace dtwc {

template <typename Tdata, typename Tsequence>
int getMedoidIndex(const std::vector<Tsequence> &sequences)
{
  VecMatrix mutualDistances(sequences.size());

  int ind{ -1 };
  Tdata minInertia = maxValue<Tdata>;
  for (int i = 0; i < sequences.size(); i++) {
    Tdata sum = 0;
    for (int j = 0; j < sequences.size(); j++) {
      if (i <= j) // When equal dtwFun should give zero by comparing pointers.
        mutualDistances(i, j) = dtwFun2(sequences[i], sequences[j]);

      sum += mutualDistances(i, j) * mutualDistances(i, j);
    }

    if (sum < minInertia) {
      ind = i;
      minInertia = sum;
    }
  }

  return ind;
}

template <typename Tdata, typename Tsequence>
void updateDBA(std::vector<Tdata> &mean, const std::vector<Tsequence> &sequences)
{
  std::vector<Tdata> newMean(mean.size());
  std::vector<unsigned short> Nmean(mean.size());
}


// template <typename Tdata>
// Tdata dtwFun5(const std::vector<Tdata> &x, const std::vector<Tdata> &y)
// {
//   // Using backtracking
//   static thread_local VecMatrix<Tdata> C(x.size(), y.size());
//   Tdata z = maxValue<Tdata>;

//   if (&x == &y)
//     return 0; // If they are the same data then distance is 0.

//   int mx = x.size();
//   int my = y.size();

//   C.resize(mx, my);

//   auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

//   if ((mx != 0) && (my != 0)) {
//     for (int j = 0; j < my; j++)
//       for (int i = 0; i < mx; i++)
//         C(i, j) = distance(x[i], y[j]);

//     int i = mx - 1;
//     int j = my - 1;

//     auto sum = C(i, j);

//     while (i > 0 && j > 0) {
//       if (C(i - 1, j - 1) <= C(i - 1, j)) {
//         if (C(i - 1, j - 1) <= C(i, j - 1)) // so minimum is diagonal
//         {
//           --i;
//           --j;
//         } else
//           --j;
//       } else
//         --i;

//       sum += C(i, j);
//     }

//     while (i > 0) {
//       --i;
//       sum += C(i, j);
//     }

//     while (j > 0) {
//       --j;
//       sum += C(i, j);
//     }

//     return sum;
//   }

//   return z;
// }

// template <typename Tdata>
// Tdata dtwFun2_wFill(const std::vector<Tdata> &x, const std::vector<Tdata> &y)
// {
//   static thread_local VecMatrix<Tdata> C(x.size(), y.size());
//   Tdata z = maxValue<Tdata>;

//   int mx = x.size();
//   int my = y.size();

//   C.resize(mx, my);

//   std::fill(C.data.begin(), C.data.end(), maxValue<Tdata>);

//   auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

//   if ((mx != 0) && (my != 0)) {
//     C(0, 0) = distance(x[0], y[0]);

//     for (int i = 1; i < mx; i++)
//       C(i, 0) = C(i - 1, 0) + distance(x[i], y[0]);

//     for (int j = 1; j < my; j++)
//       C(0, j) = C(0, j - 1) + distance(x[0], y[j]);

//     for (int i = 1; i < mx; i++)
//       for (int j = 1; j < my; j++) {
//         const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
//         C(i, j) = minimum + distance(x[i], y[j]);
//       }

//     return C(mx - 1, my - 1);
//   }

//   return z;
// }

// template <typename Tdata>
// Tdata dtwFun2_2(const std::vector<Tdata> &x, const std::vector<Tdata> &y)
// {
//   static thread_local VecMatrix<Tdata> C(x.size(), y.size());
//   Tdata z = maxValue<Tdata>;

//   int mx = x.size();
//   int my = y.size();

//   C.resize(mx, my);

//   auto distance = [&](int i, int j) { return std::abs(x[i] - y[j]); };

//   if ((mx != 0) && (my != 0)) {
//     C(0, 0) = distance(0, 0);

//     for (int i = 1; i < mx; i++)
//       C(i, 0) = C(i - 1, 0) + distance(i, 0);

//     for (int j = 1; j < my; j++)
//       C(0, j) = C(0, j - 1) + distance(0, j);

//     for (int i = 1; i < mx; i++)
//       for (int j = 1; j < my; j++) {
//         const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
//         C(i, j) = minimum + distance(i, j);
//       }

//     return C(mx - 1, my - 1);
//   }

//   return z;
// }

// template <typename Tdata>
// Tdata dtwFun2_1(const std::vector<Tdata> &x, const std::vector<Tdata> &y)
// {
//   static thread_local VecMatrix<Tdata> C(x.size(), y.size());
//   Tdata z = maxValue<Tdata>;

//   int mx = x.size();
//   int my = y.size();

//   C.resize(mx, my);

//   auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

//   if ((mx != 0) && (my != 0)) {

//     for (int i = 0; i < mx; i++)
//       for (int j = 0; j < my; j++)
//         C(i, j) = distance(x[i], y[j]);

//     for (int i = 1; i < mx; i++)
//       C(i, 0) = C(i, 0) + C(i - 1, 0);

//     for (int j = 1; j < my; j++)
//       C(0, j) = C(0, j - 1) + C(0, j);

//     for (int i = 1; i < mx; i++)
//       for (int j = 1; j < my; j++) {
//         const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
//         C(i, j) = minimum + C(i, j);
//       }

//     return C(mx - 1, my - 1);
//   }

//   return z;
// }

// template <typename Tdata>
// Tdata dtwFunBanded(const std::vector<Tdata> &x, const std::vector<Tdata> &y, int band = 100)
// {
//   static thread_local BandMatrix<Tdata> C(x.size(), y.size(), band, band);
//   Tdata z = maxValue<Tdata>;

//   const int mx = x.size();
//   const int my = y.size();

//   C.resize(mx, my, band, band);

//   auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

//   if ((mx != 0) && (my != 0)) {
//     C(0, 0) = distance(x[0], y[0]);

//     for (int i = 1; i < std::min(mx, band + 1); i++)
//       C(i, 0) = C(i - 1, 0) + distance(x[i], y[0]);

//     for (int j = 1; j < std::min(my, band + 1); j++)
//       C(0, j) = C(0, j - 1) + distance(x[0], y[j]);

//     for (int j = 1; j < my; j++) {
//       int i = std::max(1, j - band);
//       {
//         const auto minimum = std::min({ C.at(i, j - 1), C.at(i - 1, j - 1) });
//         C(i, j) = minimum + distance(x[i], y[j]);
//       }

//       for (i++; i < std::min(mx - 1, j + band); i++) {
//         const auto minimum = std::min({ C.at(i - 1, j), C.at(i, j - 1), C.at(i - 1, j - 1) });
//         C(i, j) = minimum + distance(x[i], y[j]);
//       }

//       {
//         const auto minimum = std::min({ C.at(i - 1, j), C.at(i - 1, j - 1) });
//         C(i, j) = minimum + distance(x[i], y[j]);
//       }
//     }

//     return C(mx - 1, my - 1);
//   }

//   return z;
// }


// template <typename Tdata = float>
// Tdata dtwFunBanded_Itakura(const std::vector<Tdata> &x, const std::vector<Tdata> &y, int band = 100, double max_slope = 1.0)
// {
//   // Actual banding with skewness.
//   SkewedBandMatrix<Tdata> C(x.size(), y.size(), band + 1, band + 1); // static thread_local
//   Tdata z = maxValue<Tdata>;

//   const int mx = x.size();
//   const int my = y.size();

//   double min_slope = 1 / max_slope;

//   C.resize(mx, my, band + 1, band + 1); // +1 is for inf values.

//   std::fill(C.CompactMat.data.begin(), C.CompactMat.data.end(), maxValue<Tdata>);
//   auto distance = [](Tdata x, Tdata y) { return std::abs(x - y); };

//   if ((mx != 0) && (my != 0)) {
//     auto slope = static_cast<double>(mx) / static_cast<double>(my); // #CHECK: (mx-1)/(my-1) ?????

//     C(0, 0) = distance(x[0], y[0]);

//     for (int j = 1; j < my; j++) {
//       const int j_mod = std::round(j * slope);

//       const int j_mod_min_beg = std::round(j * slope * min_slope);
//       const int j_mod_max_beg = std::round(j * slope * max_slope);

//       const int j_mod_min_end = (mx - 1) - std::round((my - 1 - j) * slope * max_slope);
//       const int j_mod_max_end = (mx - 1) - std::round((my - 1 - j) * slope * min_slope);

//       int i = std::max(1, j_mod - band);
//       {
//         const auto minimum = std::min({ C(i, j - 1), C(i - 1, j - 1) });
//         C(i, j) = minimum + distance(x[i], y[j]);
//       }

//       for (i++; i < std::min({ mx - 1, j_mod + band, j_mod_max_beg + 1 }); i++) {
//         const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
//         C(i, j) = minimum + distance(x[i], y[j]);
//       }

//       {
//         const auto minimum = std::min({ C(i - 1, j), C(i - 1, j - 1) });
//         C(i, j) = minimum + distance(x[i], y[j]);
//       }
//     }

//     return C(mx - 1, my - 1);
//   }

//   return z;
// }


void fillDistanceMatrix(auto &distByInd, int N)
{
  auto distanceAllTask = [&](int i_p) {
    for (int i = 0; i <= i_p; i++)
      distByInd(i_p, i);

    auto i_p_p = N - i_p - 1;
    for (int i = 0; i <= i_p_p; i++)
      distByInd(i_p_p, i);
  };

  const int N_2 = (N + 1) / 2;

  run(distanceAllTask, N_2);
}

template <typename Tdata, typename Tdist>
auto kMedoidsPAM(std::vector<std::vector<Tdata>> &p_vec, std::vector<int> &centroids_ind, Tdist distByInd, std::vector<std::string> &p_names, int rep, int maxIter = 100)
{
  std::vector<unsigned short> closestCluster(p_vec.size());
  std::vector<std::vector<int>> clusterMembers(centroids_ind.size()); //
  std::vector<Tdata> clusterCosts(centroids_ind.size(), maxValue<Tdata>);
  std::vector<int> oldmedoids(centroids_ind);

  auto assignClustersTask = [&](int i_p) // i_p -> index of points
  {
    double cost = std::numeric_limits<double>::max();
    for (size_t i = 0; i < centroids_ind.size(); i++) {
      auto new_cost = distByInd(i_p, centroids_ind[i]);

      if (new_cost < cost) {
        cost = new_cost;
        closestCluster[i_p] = i;
      }
    }
  };

  auto distanceInClustersTask = [&](int i_p) {
    const int clusterNo = closestCluster[i_p];
    for (auto otherPointInd : clusterMembers[clusterNo])
      if (i_p <= otherPointInd)
        distByInd(i_p, otherPointInd);
  };

  auto distributeClusters = [&clusterMembers, &closestCluster]() {
    for (auto &member : clusterMembers)
      member.clear();

    for (size_t i = 0; i < closestCluster.size(); i++)
      clusterMembers[closestCluster[i]].push_back(i);
  };

  auto findBetterMedoidTask = [&](int i_p) // i_p is point index.
  {
    const auto i_c = closestCluster[i_p];
    Tdata sum{ 0 };
    for (auto member : clusterMembers[i_c])
      sum += distByInd(i_p, member);

    if (sum < clusterCosts[i_c]) {
      clusterCosts[i_c] = sum;
      centroids_ind[i_c] = i_p;
    }
  };

  int status = -1;
  std::ofstream medoidsFile(settings::resultsPath / "medoids_rep_" + std::to_string(rep) + ".csv", std::ios_base::out);
  for (int i = 0; i < maxIter; i++) {
    std::cout << "Medoids: ";
    for (auto medoid : centroids_ind) {
      std::cout << medoid << ' ';
      if constexpr (settings::writeAsFileNames)
        medoidsFile << p_names[medoid] << ',';
      else
        medoidsFile << medoid << ',';
    }

    medoidsFile << '\n';

    run(assignClustersTask, p_vec.size()); // uses centroids_ind sets closestCluster
    distributeClusters();                  // uses closestCluster populates clusterMembers.
    std::cout << " Iteration: " << i << " completed with cost: " << std::setprecision(10)
              << findTotalCost(closestCluster, centroids_ind, distByInd) << ".\n"; // Uses closestCluster to find cost.

    writeMedoidMembers(clusterMembers, p_names, i, rep);

    run(distanceInClustersTask, p_vec.size()); // Just populates distByInd matrix ahead.
    run(findBetterMedoidTask, p_vec.size());   // Changes centroids_ind

    if (aremedoidsSame(oldmedoids, centroids_ind)) {
      status = 0;
      break;
    }

    oldmedoids = centroids_ind;
  }

  auto total_cost = findTotalCost(closestCluster, centroids_ind, distByInd);
  std::cout << "Procedure is completed with cost: " << total_cost << '\n';
  medoidsFile << "Procedure is completed with cost: " << total_cost << '\n';

  medoidsFile.close();

  return std::tuple(status, total_cost);
}
}; // namespace dtwc