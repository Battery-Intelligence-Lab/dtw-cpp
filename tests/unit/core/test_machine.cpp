/**
 * @file test_machine.cpp
 * @brief machine_facts(): every fact is said, an unknown one reads "unknown"; the CPU's facts come with cpuinfo, the
 *        memory with llfio, the GPU line with a GPU run only. Contracts, never a literal fact of a machine.
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 5.5 (RR1a)
 * @date 03 Nov 2026
 */

#include <Problem.hpp>              // gpu_info
#include <core/distance_matrix.hpp> // available_memory
#include <core/machine.hpp>

#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>

namespace {

/// The value of the line of `text` labelled `label` ("  RAM:   <value>"), without the padding; "(no line)" if none.
std::string value_of(const std::string &text, const std::string &label)
{
  std::istringstream lines(text);
  for (std::string line; std::getline(lines, line);)
    if (line.rfind("  " + label + ":", 0) == 0) {
      const auto at = line.find_first_not_of(' ', label.size() + 3);
      return at == std::string::npos ? "" : line.substr(at);
    }
  return "(no line)";
}

} // namespace

TEST_CASE("machine_facts says every fact, and unknown where it cannot", "[machine]")
{
  const dtwc::core::MachineFacts facts = dtwc::core::machine_facts();
  for (const std::string *field :
       { &facts.host, &facts.os, &facts.cpu, &facts.cpu_simd, &facts.build_simd, &facts.compiler, &facts.libraries })
    CHECK_FALSE(field->empty());
  const std::string text = facts.text();
  INFO(text);
  for (const char *label : { "Host", "OS", "CPU", "Cores", "SIMD", "RAM", "Build" }) {
    const std::string value = value_of(text, label);
    CHECK(value != "(no line)");
    CHECK_FALSE(value.empty());
  }
  CHECK(value_of(text, "CPU") == facts.cpu);

#ifdef DTWC_HAS_CPUINFO
  CHECK(facts.physical_cores >= 1);
  CHECK(facts.logical_cores >= facts.physical_cores);
  CHECK(facts.cpu != "unknown");
  CHECK(facts.cpu_simd != "unknown");
#else
  CHECK(facts.cpu == "unknown");
  CHECK(facts.cpu_simd == "unknown");
  CHECK(facts.physical_cores == 0);
  CHECK(facts.logical_cores == 0);
  CHECK(value_of(text, "Cores") == "unknown");
#endif

#ifdef DTWC_HAS_MMAP
  CHECK(dtwc::core::total_memory() > 0);
  CHECK(dtwc::core::total_memory() >= dtwc::core::available_memory());
  CHECK(facts.ram_total == dtwc::core::total_memory());
#else
  CHECK(dtwc::core::total_memory() == 0);
  CHECK(dtwc::core::available_memory() == 0);
  CHECK(value_of(text, "RAM") == "unknown");
#endif
}

TEST_CASE("machine_facts names the GPU for a GPU run only", "[machine]")
{
  const dtwc::core::MachineFacts on_cpu = dtwc::core::machine_facts(dtwc::Device::CPU);
  CHECK_FALSE(on_cpu.gpu.has_value());
  CHECK(value_of(on_cpu.text(), "GPU") == "(no line)");

  // GPU 0: the line gpu_info() gives (a build without a GPU backend: why there is none).
  const dtwc::core::MachineFacts on_gpu = dtwc::core::machine_facts(dtwc::Device::GPU, 0);
  REQUIRE(on_gpu.gpu.has_value());
  CHECK(*on_gpu.gpu == dtwc::gpu_info());
  CHECK(value_of(on_gpu.text(), "GPU") == dtwc::gpu_info());
}
