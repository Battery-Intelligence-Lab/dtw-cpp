/**
 * @file test_env_device.cpp
 * @brief The process-wide device: dtwc::device(name) / dtwc::device().
 *
 * @details Drives the live entry points. The messages are the §6.1 contract,
 * declared before any check:
 *   - an unknown name lists the valid names,
 *   - `gpu` on a build with no GPU backend names the CMake flag,
 *   - `hpc` / `hpc:gpu` name Python's dtwcpp.device("hpc") and slurm_remote.sh.
 * A rejected request leaves the selection unchanged.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

static const std::string kMsgUnknownFoo =
  "[dtwc] unknown device 'foo'. Valid devices: cpu, gpu, gpu:N (aliases cuda, cuda:N).";

static const std::string kMsgGpuNotBuilt =
  "[dtwc] device='gpu' requested but this build has no GPU backend compiled in.\n"
  "Rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, on macOS, -DDTWC_ENABLE_METAL=ON.\n"
  "This build will not silently fall back to CPU.";

static std::string hpc_message(const std::string &name)
{
  return "[dtwc] device '" + name + "' submits a whole run to a SLURM cluster, which "
         "Python's dtwcpp.device(\"hpc\") and scripts/slurm/slurm_remote.sh do. C++, "
         "MATLAB and dtwc_cl compute where they start: use cpu or gpu.";
}

// The message dtwc::device(name) throws, or a FAIL if it did not throw.
static std::string device_error_message(const std::string &name)
{
  try {
    (void)dtwc::device(name);
  } catch (const dtwc::DeviceError &err) {
    return err.what();
  }
  FAIL("dtwc::device('" + name + "') did not throw dtwc::DeviceError");
  return {};
}

TEST_CASE("dtwc::device: an unknown name lists the valid names", "[env][device]")
{
  REQUIRE(dtwc::device("cpu") == "cpu");
  REQUIRE(device_error_message("foo") == kMsgUnknownFoo);
  REQUIRE(dtwc::device() == "cpu"); // a rejected request changes nothing
}

TEST_CASE("dtwc::device accepts cpu, case-insensitive and trimmed", "[env][device]")
{
  REQUIRE(dtwc::device(" CPU ") == "cpu");
  REQUIRE(dtwc::device() == "cpu");
}

TEST_CASE("dtwc::device gpu without a GPU build names the flag", "[env][device]")
{
  REQUIRE(dtwc::device("cpu") == "cpu");
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  REQUIRE(dtwc::device("cuda:1") == "gpu:1");
  REQUIRE(dtwc::device() == "gpu:1");
  REQUIRE(dtwc::device("cpu") == "cpu");
#else
  // gpu, cuda and gpu:N are one request.
  REQUIRE(device_error_message("gpu") == kMsgGpuNotBuilt);
  REQUIRE(device_error_message("cuda") == kMsgGpuNotBuilt);
  REQUIRE(device_error_message("gpu:0") == kMsgGpuNotBuilt);
  REQUIRE(dtwc::device() == "cpu");
#endif
}

TEST_CASE("dtwc::device hpc names the Python and SLURM routes", "[env][hpc]")
{
  REQUIRE(dtwc::device("cpu") == "cpu");
  REQUIRE(device_error_message("hpc") == hpc_message("hpc"));
  REQUIRE(device_error_message("HPC:gpu") == hpc_message("HPC:gpu"));
  REQUIRE(dtwc::device() == "cpu");
  // The grammar behind --device and Problem::set_device refuses it the same way.
  REQUIRE_THROWS_AS(dtwc::detail::parse_device("hpc"), dtwc::DeviceError);
}
