#include "dtwc.hpp"
#include "examples.hpp"
#include "../benchmark/benchmark_main.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>


int main()
{
  // Here are some examples. You can either take the contents of example functions into main or modify and run them.
  dtwc::Clock clk; // Create a clock object

  // dtwc::examples::cluster_byKmeans_single();
  // dtwc::examples::cluster_byMIP_single();
 // dtwc::examples::cluster_byMIP_multiple();

  dtwc::benchmarks::run_all();
  std::cout << "Finished all tasks " << clk << "\n";
  //  dtwc::examples::cluster_byKmeans_single(); // -> Not properly working
}
