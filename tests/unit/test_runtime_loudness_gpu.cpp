/**
 * @file test_runtime_loudness_gpu.cpp
 * @brief Explicit GPU strategies fail rather than changing the backend.
 */

#include <Data.hpp>
#include <Problem.hpp>
#include <error.hpp>

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <string>
#include <vector>

static const std::string kMsgCudaNotCompiled =
  "CUDA distance strategy requested but CUDA is not compiled in. "
  "Rebuild with -DDTWC_ENABLE_CUDA=ON. No CPU fallback was attempted.";
static const std::string kMsgMetalNotCompiled =
  "Metal distance strategy requested but Metal is not compiled in. "
  "Rebuild on macOS with -DDTWC_ENABLE_METAL=ON. No CPU fallback was attempted.";

static void require_device_error(const std::function<void()> &call,
                                 const std::string &expected)
{
  try {
    call();
    FAIL("expected dtwc::DeviceError");
  } catch (const dtwc::DeviceError &e) {
    REQUIRE(std::string(e.what()) == expected);
  }
}

static dtwc::Problem make_tiny_problem()
{
  std::vector<std::vector<dtwc::data_t>> vecs;
  std::vector<std::string> names;
  for (int i = 0; i < 4; ++i) {
    vecs.push_back({ double(i), double(i + 1), double(i + 2), double(i + 3) });
    names.push_back("s" + std::to_string(i));
  }
  dtwc::Problem prob("loudness_gpu");
  prob.set_data(dtwc::Data(std::move(vecs), std::move(names)));
  return prob;
}

TEST_CASE("fill_distance_matrix(strategy=Metal) never changes to CPU", "[loudness][gpu]")
{
  auto prob = make_tiny_problem();
  prob.distance_strategy = dtwc::DistanceMatrixStrategy::Metal;
  prob.verbose = false;

#if defined(DTWC_HAS_METAL)
  try {
    prob.fill_distance_matrix();
    REQUIRE(prob.is_distance_matrix_filled());
  } catch (const dtwc::DeviceError &) {
    REQUIRE_FALSE(prob.is_distance_matrix_filled()); // compiled, no live device
  }
#else
  require_device_error([&] { prob.fill_distance_matrix(); }, kMsgMetalNotCompiled);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
#endif
}

TEST_CASE("fill_distance_matrix(strategy=CUDA) never changes to CPU", "[loudness][gpu]")
{
  auto prob = make_tiny_problem();
  prob.distance_strategy = dtwc::DistanceMatrixStrategy::CUDA;
  prob.verbose = false;

#if defined(DTWC_HAS_CUDA)
  try {
    prob.fill_distance_matrix();
    REQUIRE(prob.is_distance_matrix_filled());
  } catch (const dtwc::DeviceError &) {
    REQUIRE_FALSE(prob.is_distance_matrix_filled());
  }
#else
  require_device_error([&] { prob.fill_distance_matrix(); }, kMsgCudaNotCompiled);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
#endif
}
