/*
 * benchmark_main.hpp
 *
 *  Benchmark file
 *
 *  Created on: 17 Aug 2022
 *   Author(s): Volkan Kumtepeli, Rebecca Perriment
 */

#pragma once

#include "../dtwc/dtwc.hpp"

#include <string>
#include <vector>
#include <map>
#include <utility>

namespace dtwc::benchmarks {


inline auto get_UCR_2018_files()
{
  std::vector<fs::path> UCR_list{};
  std::map<std::string, int> Nc_list{};


  std::ifstream summary_file(settings::root_folder / "data/benchmark/UCR_DataSummary.csv", std::ios_base::in);
  // #TODO addwarning i fi
  std::string summary_line{}, summary_name;

  int Nc_now;
  char c{ '.' };

  std::getline(summary_file, summary_line);

  while (std::getline(summary_file, summary_line)) {
    std::istringstream iss(summary_line);

    for (int i = 0; i < 3; i++)
      std::getline(iss, summary_name, ',');


    for (int i = 0; i < 3; i++)
      iss >> Nc_now >> c;

    Nc_list[summary_name] = Nc_now;
  }


  auto directories = fs::recursive_directory_iterator(settings::root_folder / "data/benchmark/UCRArchive_2018");
  for (const auto &entry : directories) {
    if (entry.is_regular_file()) {                   // Check if the entry is a regular file
      std::string file_path = entry.path().string(); // Get the file path as a string
      // Check if the file path ends with "_TEST.tsv"
      if (file_path.length() >= 9 && file_path.substr(file_path.length() - 9) == "_TEST.tsv") {
        UCR_list.push_back(entry.path());
      }
    }
  }


  return std::pair(UCR_list, Nc_list);
}


inline void UCR_2018()
{
  auto [UCR_list, Nc_list] = get_UCR_2018_files();

  dtwc::DataLoader dl;
  dl.startColumn(1); // For not reading first column of *.tsv files;

  fs::path out_folder = settings::resultsPath / "benchmark";

  std::ofstream timing_file(out_folder / "timing_all.csv", std::ios_base::out);

  timing_file << "Name,fillDistanceMatrix,MIPclustering,writeSilhouettes\n";
  std::string reportName = "MILP_results";
  // UCR_list
  std::vector<fs::path> dataofInterest{
    // (settings::root_folder / "data/benchmark/UCRArchive_2018/UMD/UMD_TEST.tsv"),
    //   (settings::root_folder / "data/benchmark/UCRArchive_2018/TwoPatterns/TwoPatterns_TEST.tsv")
    (settings::root_folder / "data/benchmark/UCRArchive_2018/Plane/Plane_TEST.tsv"),
    // (settings::root_folder / "data/benchmark/UCRArchive_2018/AllGestureWiimoteX/AllGestureWiimoteX_TEST.tsv"),
    // (settings::root_folder / "data/benchmark/UCRArchive_2018/AllGestureWiimoteZ/AllGestureWiimoteZ_TEST.tsv")

  };
  size_t solved = 0;
  for (auto &file_path :  UCR_list ) { //dataofInterest
    dl.path(file_path);
    auto stem_str = file_path.stem().string();
    dtwc::Problem prob{ "sqr_" + stem_str, dl }; // Create a problem.
    int Nc = Nc_list[stem_str.substr(0, stem_str.length() - 5)];

    std::cout << "Now, number " << solved << " " << file_path << " is being solved.\n";
    solved++;
    if (solved < 123) // We already calculated this part
      continue;

    if (prob.data.size() > 4000) // DOnt calculate large data it is not good. For example Crop.
      continue;

    prob.set_numberOfClusters(Nc); // Nc = number of clusters.

    dtwc::Clock clk; // Create a clock object

    prob.output_folder = out_folder;

    prob.fillDistanceMatrix();
    prob.writeDistanceMatrix();


    auto time_1 = clk.duration();

    std::cout << "Finished calculating distances " << clk << std::endl;
    std::cout << "Band used " << settings::band << "\n\n\n";


     //prob.cluster_by_kMedoidsPAM_repetetive(1);
    prob.cluster_by_MIP(); // Uses MILP to do clustering.

    auto time_2 = clk.duration();
    //std::cout << "Finished MIP clustering " << clk << '\n';
   // std::cout << "Band used " << settings::band << "\n\n\n";

    //prob.printClusters(); // Prints to screen.
    //prob.writeClusters(); // Prints to file.
   // prob.writeSilhouettes();

    auto time_3 = clk.duration();
    timing_file << stem_str << ',' << time_1 << ',' << time_2 << ',' << time_3 << '\n';
  }
}


inline void run_all()
{
  UCR_2018();
}

} // namespace dtwc::benchmarks