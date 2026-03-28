/**
 * @file unit_test_scratch_matrix.cpp
 * @brief Unit tests for ScratchMatrix class.
 *
 * @date 28 Mar 2026
 */

#include <core/scratch_matrix.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace dtwc::core;

TEST_CASE("ScratchMatrix default constructor", "[ScratchMatrix]")
{
  ScratchMatrix<double> m;
  REQUIRE(m.rows() == 0);
  REQUIRE(m.cols() == 0);
  REQUIRE(m.size() == 0);
  REQUIRE(m.empty());
}

TEST_CASE("ScratchMatrix sized constructor", "[ScratchMatrix]")
{
  ScratchMatrix<double> m(3, 4);
  REQUIRE(m.rows() == 3);
  REQUIRE(m.cols() == 4);
  REQUIRE(m.size() == 12);
  REQUIRE_FALSE(m.empty());
}

TEST_CASE("ScratchMatrix valued constructor", "[ScratchMatrix]")
{
  ScratchMatrix<float> m(2, 3, 5.0f);
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE(m(i, j) == 5.0f);
}

TEST_CASE("ScratchMatrix element access and assignment", "[ScratchMatrix]")
{
  ScratchMatrix<int> m(3, 3);
  m(0, 0) = 1;
  m(1, 2) = 42;
  m(2, 0) = -7;

  REQUIRE(m(0, 0) == 1);
  REQUIRE(m(1, 2) == 42);
  REQUIRE(m(2, 0) == -7);
}

TEST_CASE("ScratchMatrix row-major layout", "[ScratchMatrix]")
{
  ScratchMatrix<int> m(2, 3);
  // Fill sequentially
  int val = 0;
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      m(i, j) = val++;

  // Verify row-major layout in raw memory
  const int *raw = m.raw();
  REQUIRE(raw[0] == 0); // m(0,0)
  REQUIRE(raw[1] == 1); // m(0,1)
  REQUIRE(raw[2] == 2); // m(0,2)
  REQUIRE(raw[3] == 3); // m(1,0)
  REQUIRE(raw[4] == 4); // m(1,1)
  REQUIRE(raw[5] == 5); // m(1,2)
}

TEST_CASE("ScratchMatrix resize", "[ScratchMatrix]")
{
  ScratchMatrix<double> m(2, 2);
  m(0, 0) = 1.0;
  m(1, 1) = 2.0;

  m.resize(4, 5);
  REQUIRE(m.rows() == 4);
  REQUIRE(m.cols() == 5);
  REQUIRE(m.size() == 20);
}

TEST_CASE("ScratchMatrix fill", "[ScratchMatrix]")
{
  ScratchMatrix<double> m(3, 4);
  m.fill(99.5);

  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 4; ++j)
      REQUIRE(m(i, j) == 99.5);
}

TEST_CASE("ScratchMatrix const access", "[ScratchMatrix]")
{
  ScratchMatrix<double> m(2, 2);
  m(0, 0) = 3.14;
  m(1, 1) = 2.72;

  const ScratchMatrix<double> &cm = m;
  REQUIRE(cm(0, 0) == 3.14);
  REQUIRE(cm(1, 1) == 2.72);
  REQUIRE(cm.raw() != nullptr);
}
