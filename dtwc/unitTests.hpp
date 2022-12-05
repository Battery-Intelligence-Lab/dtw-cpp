// Vk: 2021.01.21

#pragma once

#include "utility.hpp"
#include "initialisation.hpp"

#include <iostream>
#include <vector>
#include <array>
#include <filesystem>
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

#include <cassert>


namespace dtwc::test {

void test_DTWfunction()
{

  // std::vector<double> asd{ 0.8147,0.9058,0.1270 };
  // std::vector<double> bsd{ 0.5469 ,0.9575,0.9649,0.1576, 0.9706 };

  // std::cout << "Ground truth:\n";

  // std::cout << "Test cost dtwFun: " << dtwFun(p_vec[0], p_vec[5]) << '\n';
  // //std::cout << "Test cost dtwFun2: " << dtwFun2(p_vec[0], p_vec[5]) << '\n';
  // std::cout << "Test cost dtwFun5: " << dtwFun5(p_vec[0], p_vec[5]) << '\n';
  // std::cout << "Test cost dtwFunBanded_Act: " << dtwFunBanded_Act(p_vec[0], p_vec[5], 400) << '\n';


  // std::cout << "Test cost: " << dtwFun(p_vec[5], p_vec[0]) << '\n';
  // std::cout << "Test cost: " << distByInd(5, 0) << '\n';
  // std::cout << "Test cost: " << dtwFun5(p_vec[5], p_vec[0]) << '\n';


  // std::cout << "Test cost: " << dtwFun2(p_vec[0], p_vec[0]) << '\n';


  // std::cout << "Test cost: " << dtwFun_recursive(p_vec[0], p_vec[5]) << '\n';

  // std::cout << "Test cost: " << dtwFun3(p_vec[0], p_vec[5]) << '\n';
  // std::cout << "Test cost: " << dtwFun3_1(p_vec[0], p_vec[5]) << '\n'; // Slower compared to vector allocation.
  // std::cout << "Test cost: " << dtwFunBanded_2(p_vec[0], p_vec[5], 500) << '\n';

  // std::cout << "Test cost dtwFunBanded_Act buggy: " << dtwFunBanded_Act(p_vec[0], p_vec[3], 500) << '\n';
  // std::cout << "Test cost dtwFunBanded_Act buggy: " << dtwFunBanded_Act(p_vec[9], p_vec[3], 300) << '\n';
  // std::cout << "Test cost dtwFun buggy: " << dtwFun(p_vec[9], p_vec[3]) << '\n';
  // std::cout << "Test cost dtwFunBanded_Act2 buggy: " << dtwFunBanded_Act2(p_vec[9], p_vec[3], 300) << '\n';
  // std::cout << "Test cost dtwFunBanded_Itakura buggy: " << dtwFunBanded_Itakura(p_vec[9], p_vec[3], 300) << '\n';
}


void test_initialisation()
{
  // dtwc::Initialisation::init_random(p_vec, 3);
  // auto centroids_vec = dtwc::Initialisation::init_random(p_vec, 3);
  // auto centroids_vec = dtwc::Initialisation::init_Kmeanspp(p_vec, N_k, dtwFun2<data_t>);
}

} // namespace dtwc::test