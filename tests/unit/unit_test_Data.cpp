/**
 * @file unit_test_Data.cpp
 * @brief Unit test file for Data class
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 16 Dec 2023
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <span>
#include <string>
#include <string_view>
#include <vector>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("Data class functionality", "[Data]")
{
  SECTION("Default constructor creates empty Data object")
  {
    Data data;
    REQUIRE(data.size() == 0);
    REQUIRE(data.p_vec.empty());
    REQUIRE(data.p_names.empty());
  }

  SECTION("Parameterized constructor sets data correctly")
  {
    std::vector<std::vector<data_t>> testVec = { { 1, 2, 3 }, { 4, 5, 6 } };
    std::vector<std::string> testNames = { "First", "Second" };

    Data data(std::move(testVec), std::move(testNames));

    REQUIRE(data.size() == 2);
    REQUIRE(data.p_vec.size() == 2);
    REQUIRE(data.p_names.size() == 2);
    REQUIRE(data.p_vec[0].size() == 3);
    REQUIRE(data.p_vec[1].size() == 3);
    REQUIRE(data.p_names[0] == "First");
    REQUIRE(data.p_names[1] == "Second");
  }

  SECTION("Data size corresponds to number of elements")
  {
    Data data;
    data.p_vec = { { 1 }, { 2, 3 }, { 4, 5, 6 } };
    data.p_names = { "A", "B", "C" };

    REQUIRE(data.size() == 3);
  }

  SECTION("Constructor throws assertion error with mismatched vector sizes")
  {
    std::vector<std::vector<data_t>> testVec = { { 1, 2 }, { 3, 4 } };
    std::vector<std::string> testNames = { "One" };

    REQUIRE_THROWS_AS(Data(std::move(testVec), std::move(testNames)), std::exception);
  }

  SECTION("series() returns correct span for heap-mode Data")
  {
    std::vector<std::vector<data_t>> vecs = { { 1.0, 2.0, 3.0 }, { 4.0, 5.0 } };
    std::vector<std::string> names = { "A", "B" };
    Data data(std::move(vecs), std::move(names));

    auto s0 = data.series(0);
    REQUIRE(s0.size() == 3);
    REQUIRE(s0[0] == 1.0);
    REQUIRE(s0[2] == 3.0);

    auto s1 = data.series(1);
    REQUIRE(s1.size() == 2);
    REQUIRE(s1[0] == 4.0);

    REQUIRE(data.series_flat_size(0) == 3);
    REQUIRE(data.series_flat_size(1) == 2);
    REQUIRE(data.name(0) == "A");
    REQUIRE(data.name(1) == "B");
    REQUIRE_FALSE(data.is_view());
  }

  SECTION("series_length() accounts for ndim correctly")
  {
    std::vector<std::vector<data_t>> vecs = { { 1, 2, 3, 4, 5, 6 } };
    std::vector<std::string> names = { "MV" };
    Data data(std::move(vecs), std::move(names), 2);

    REQUIRE(data.series_length(0) == 3); // 6 values / 2 dims = 3 timesteps
    REQUIRE(data.series_flat_size(0) == 6);
  }
}

TEST_CASE("Data view-mode functionality", "[Data][view]")
{
  // Create parent data that will outlive the view.
  std::vector<std::vector<data_t>> parent_vecs = { { 1.0, 2.0, 3.0 }, { 4.0, 5.0 }, { 6.0, 7.0, 8.0, 9.0 } };
  std::vector<std::string> parent_names = { "Alpha", "Beta", "Gamma" };
  Data parent(std::move(parent_vecs), std::move(parent_names));

  SECTION("View-mode Data references parent series without copying")
  {
    // Create a view of series 0 and 2 (skipping 1).
    std::vector<std::span<const data_t>> spans = { parent.series(0), parent.series(2) };
    std::vector<std::string_view> names = { parent.name(0), parent.name(2) };
    Data view(std::move(spans), std::move(names), 1);

    REQUIRE(view.is_view());
    REQUIRE(view.size() == 2);
    REQUIRE(view.series(0).size() == 3);
    REQUIRE(view.series(0)[0] == 1.0);
    REQUIRE(view.series(1).size() == 4);
    REQUIRE(view.series(1)[3] == 9.0);
    REQUIRE(view.name(0) == "Alpha");
    REQUIRE(view.name(1) == "Gamma");
    REQUIRE(view.series_flat_size(0) == 3);
    REQUIRE(view.series_flat_size(1) == 4);
  }

  SECTION("View-mode series() data pointers match parent (zero-copy)")
  {
    std::vector<std::span<const data_t>> spans = { parent.series(1) };
    std::vector<std::string_view> names = { parent.name(1) };
    Data view(std::move(spans), std::move(names), 1);

    // The view's span should point to the exact same memory as the parent's.
    REQUIRE(view.series(0).data() == parent.series(1).data());
  }

  SECTION("View-mode constructor validates ndim")
  {
    // Series of size 3 with ndim=2 should fail (3 not divisible by 2).
    std::vector<std::span<const data_t>> spans = { parent.series(0) }; // size 3
    std::vector<std::string_view> names = { parent.name(0) };
    REQUIRE_THROWS_AS(Data(std::move(spans), std::move(names), 2), std::runtime_error);
  }

  SECTION("View-mode constructor validates size mismatch")
  {
    std::vector<std::span<const data_t>> spans = { parent.series(0), parent.series(1) };
    std::vector<std::string_view> names = { parent.name(0) }; // only 1 name for 2 spans
    REQUIRE_THROWS_AS(Data(std::move(spans), std::move(names), 1), std::runtime_error);
  }
}

TEST_CASE("Data float32 storage", "[Data][float32]")
{
  SECTION("Float32 constructor stores data correctly")
  {
    std::vector<std::vector<float>> vecs = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f } };
    std::vector<std::string> names = { "A", "B" };
    Data data(std::move(vecs), std::move(names));

    REQUIRE(data.is_f32());
    REQUIRE(data.size() == 2);
    REQUIRE(data.series_flat_size(0) == 3);
    REQUIRE(data.series_flat_size(1) == 2);
    REQUIRE(data.series_f32(0).size() == 3);
    REQUIRE(data.series_f32(0)[0] == 1.0f);
    REQUIRE(data.series_f32(1)[1] == 5.0f);
    REQUIRE(data.name(0) == "A");
    REQUIRE(data.name(1) == "B");
    REQUIRE_FALSE(data.is_view());
  }

  SECTION("Float32 series_length accounts for ndim")
  {
    std::vector<std::vector<float>> vecs = { { 1, 2, 3, 4, 5, 6 } };
    std::vector<std::string> names = { "MV" };
    Data data(std::move(vecs), std::move(names), 2);

    REQUIRE(data.is_f32());
    REQUIRE(data.series_length(0) == 3); // 6 values / 2 dims
    REQUIRE(data.series_flat_size(0) == 6);
  }

  SECTION("Float32 validates ndim")
  {
    std::vector<std::vector<float>> vecs = { { 1, 2, 3 } };
    std::vector<std::string> names = { "bad" };
    REQUIRE_THROWS_AS(Data(std::move(vecs), std::move(names), 2), std::runtime_error);
  }
}

TEST_CASE("Data float32 view-mode", "[Data][float32][view]")
{
  // Create parent float32 data that will outlive the view.
  std::vector<std::vector<float>> parent_vecs = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f }, { 6.0f, 7.0f, 8.0f, 9.0f } };
  std::vector<std::string> parent_names = { "Alpha", "Beta", "Gamma" };
  Data parent(std::move(parent_vecs), std::move(parent_names));

  REQUIRE(parent.is_f32());

  SECTION("Float32 view references parent series without copying")
  {
    std::vector<std::span<const float>> spans = { parent.series_f32(0), parent.series_f32(2) };
    std::vector<std::string_view> names = { parent.name(0), parent.name(2) };
    Data view(std::move(spans), std::move(names), 1);

    REQUIRE(view.is_view());
    REQUIRE(view.is_f32());
    REQUIRE(view.size() == 2);
    REQUIRE(view.series_f32(0).size() == 3);
    REQUIRE(view.series_f32(0)[0] == 1.0f);
    REQUIRE(view.series_f32(1).size() == 4);
    REQUIRE(view.series_f32(1)[3] == 9.0f);
    REQUIRE(view.name(0) == "Alpha");
    REQUIRE(view.name(1) == "Gamma");
    REQUIRE(view.series_flat_size(0) == 3);
    REQUIRE(view.series_flat_size(1) == 4);
  }

  SECTION("Float32 view pointers match parent (zero-copy)")
  {
    std::vector<std::span<const float>> spans = { parent.series_f32(1) };
    std::vector<std::string_view> names = { parent.name(1) };
    Data view(std::move(spans), std::move(names), 1);

    REQUIRE(view.series_f32(0).data() == parent.series_f32(1).data());
  }

  SECTION("Float32 view validates size mismatch")
  {
    std::vector<std::span<const float>> spans = { parent.series_f32(0), parent.series_f32(1) };
    std::vector<std::string_view> names = { parent.name(0) }; // 1 name for 2 spans
    REQUIRE_THROWS_AS(Data(std::move(spans), std::move(names), 1), std::runtime_error);
  }

  SECTION("Float32 view validates ndim")
  {
    std::vector<std::span<const float>> spans = { parent.series_f32(0) }; // size 3
    std::vector<std::string_view> names = { parent.name(0) };
    REQUIRE_THROWS_AS(Data(std::move(spans), std::move(names), 2), std::runtime_error);
  }
}

TEST_CASE("Float32 DTW via Problem", "[Problem][float32]")
{
  using namespace dtwc;

  // Create float32 data: 3 series, 2 groups
  std::vector<std::vector<float>> vecs = {
    { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f },   // group A
    { 1.1f, 2.1f, 3.1f, 4.1f, 5.1f },   // group A (close to series 0)
    { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f } // group B (far from A)
  };
  std::vector<std::string> names = { "a1", "a2", "b1" };

  Problem prob("f32_test");
  prob.set_data(Data(std::move(vecs), std::move(names)));

  REQUIRE(prob.data.is_f32());
  REQUIRE(prob.size() == 3);

  // DTW distances should reflect grouping
  double d_close = prob.distByInd(0, 1); // a1 vs a2 — should be small
  double d_far = prob.distByInd(0, 2);   // a1 vs b1 — should be large

  REQUIRE(d_close < d_far);
  REQUIRE(d_close < 1.0); // 5 * 0.1 = 0.5
  REQUIRE(d_far > 10.0);  // very different series
}