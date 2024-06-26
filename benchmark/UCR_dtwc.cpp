/*
 * UCR_dtwc.cpp
 *
 *  Benchmark file
 *
 *  Created on: 17 Aug 2022
 *   Author(s): Volkan Kumtepeli, Rebecca Perriment
 */

#include <dtwc.hpp>

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <filesystem>

using namespace dtwc;

inline auto get_UCR_2018_files()
{
  std::vector<fs::path> UCR_list{};
  std::map<std::string, int> Nc_list{};


  std::ifstream summary_file(settings::root_folder / "data/benchmark/UCR_DataSummary.csv", std::ios_base::in);

  if (!summary_file.is_open()) {
    std::cerr << fs::absolute(settings::root_folder / "data/benchmark/UCR_DataSummary.csv")
              << " could not be opened!\n";
    throw 11;
  }

  std::string summary_line{}, summary_name;

  int Nc_now{};
  char c{ '.' };

  std::getline(summary_file, summary_line);

  while (std::getline(summary_file, summary_line)) {
    std::istringstream iss(summary_line);

    for (int i = 0; i < 3; i++)
      std::getline(iss, summary_name, ',');


    for (int i = 0; i < 3; i++)
      iss >> Nc_now >> c;


    if (Nc_now == 0) {
      std::cerr << summary_name << " cluster info could not be read!\n";
      throw 11;
    }

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
    //(settings::root_folder / "data/benchmark/UCRArchive_2018/UMD/UMD_TEST.tsv"),
    // (settings::root_folder / "data/benchmark/UCRArchive_2018/TwoPatterns/TwoPatterns_TEST.tsv")
    //(settings::root_folder / "data/benchmark/UCRArchive_2018/Coffee/Coffee_TEST.tsv"),
    //(settings::root_folder / "data/benchmark/UCRArchive_2018/FaceFour/FaceFour_TEST.tsv"),
    (settings::root_folder / "data/benchmark/UCRArchive_2018/AllGestureWiimoteX/AllGestureWiimoteX_TEST.tsv"),
    //(settings::root_folder / "data/benchmark/UCRArchive_2018/AllGestureWiimoteZ/AllGestureWiimoteZ_TEST.tsv")

  };

  dataofInterest = UCR_list; // Comment this out to do an individual testing.

  size_t solved = 0;
  for (auto &file_path : dataofInterest) {
    dl.path(file_path);
    auto stem_str = file_path.stem().string();
    dtwc::Problem prob{ "sqr_" + stem_str, dl }; // Create a problem.
    prob.output_folder = out_folder;

    int Nc = Nc_list[stem_str.substr(0, stem_str.length() - 5)];

    std::cout << "Now, number " << solved << " " << file_path << " is being solved.\n";
    solved++;

    if (prob.data.size() > 1000) // Don't calculate large data it is not good. For example Crop.
      continue;

    prob.set_numberOfClusters(Nc); // Nc = number of clusters.

    dtwc::Clock clk; // Create a clock object

    prob.fillDistanceMatrix();

    const auto time_1 = clk.duration();

    std::cout << "Finished calculating distances " << clk << std::endl;
    std::cout << "Band used " << settings::band << "\n\n\n";

    prob.N_repetition = 2;

    prob.cluster_by_kMedoidsPAM();
    // prob.cluster_by_MIP(); // Uses MILP to do clustering.

    const auto time_2 = clk.duration();
    std::cout << "Finished MIP clustering " << clk << '\n';
    std::cout << "Band used " << settings::band << "\n\n\n";

    prob.printClusters(); // Prints to screen.
    prob.writeDistanceMatrix();
    prob.writeClusters(); // Prints to file.
    prob.writeSilhouettes();

    const auto time_3 = clk.duration();
    timing_file << stem_str << ',' << time_1 << ',' << time_2 << ',' << time_3 << '\n';
  }
}


int main()
{
  dtwc::Clock clk; // Create a clock object
  UCR_2018();
  std::cout << "Finished benchmarking " << clk << "\n";
}