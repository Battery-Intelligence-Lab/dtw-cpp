// GC fix 2: FastCLARA on a GPU with a parent whose series are a view. Prints
// what ran before the refusal (verbose), then the refusal.
#include <dtwc.hpp>

#include <cstdio>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

int main()
{
  std::vector<std::vector<double>> owned;
  for (int i = 0; i < 40; ++i) owned.push_back(std::vector<double>(20, (i % 2) * 10.0 + 0.01 * i));
  std::vector<std::span<const double>> spans(owned.begin(), owned.end());
  std::vector<std::string> names(owned.size(), "s");
  std::vector<std::string_view> name_views(names.begin(), names.end());
  dtwc::Problem prob("view_parent");
  prob.set_view_data(dtwc::Data(std::move(spans), std::move(name_views), 1));
  prob.set_device(dtwc::Device::GPU);
  prob.set_verbose(true);
  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = 2;
  opts.sample_size = 10;
  opts.n_samples = 2;
  try {
    (void)dtwc::algorithms::fast_clara(prob, opts);
    std::cout << "RAN\n";
  } catch (const dtwc::DeviceError &e) {
    std::cout << "DeviceError: " << e.what() << "\n";
  }
  return 0;
}
