// Vk 2022.03.01
// Timing functions

#include <ctime>
#include <iostream>
#include <cmath>

namespace dtwc {
struct Duration
{
  double d{ 0 };
};

inline auto get_duration(auto t_start)
{
  const double dr = (std::clock() - t_start) / (double)CLOCKS_PER_SEC;
  return Duration{ dr };
}

std::ostream &operator<<(std::ostream &out, const Duration &obj)
{
  out << std::floor(obj.d / 60) << ":" << obj.d - std::floor(obj.d / 60) * 60;
  return out;
}
} // namespace dtwc
