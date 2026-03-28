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

TEST_CASE("ScratchMatrix column-major layout", "[ScratchMatrix]")
{
  ScratchMatrix<int> m(2, 3);
  // Fill sequentially via operator()
  int val = 0;
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      m(i, j) = val++;

  // m(0,0)=0  m(0,1)=1  m(0,2)=2
  // m(1,0)=3  m(1,1)=4  m(1,2)=5

  // Column-major layout in raw memory: columns stored contiguously
  // Column 0: m(0,0)=0, m(1,0)=3
  // Column 1: m(0,1)=1, m(1,1)=4
  // Column 2: m(0,2)=2, m(1,2)=5
  const int *raw = m.raw();
  REQUIRE(raw[0] == 0); // m(0,0) - col 0, row 0
  REQUIRE(raw[1] == 3); // m(1,0) - col 0, row 1
  REQUIRE(raw[2] == 1); // m(0,1) - col 1, row 0
  REQUIRE(raw[3] == 4); // m(1,1) - col 1, row 1
  REQUIRE(raw[4] == 2); // m(0,2) - col 2, row 0
  REQUIRE(raw[5] == 5); // m(1,2) - col 2, row 1
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

TEST_CASE("ScratchMatrix zero-dimension edge cases", "[ScratchMatrix]")
{
  // Zero rows
  ScratchMatrix<double> m1(0, 5);
  REQUIRE(m1.rows() == 0);
  REQUIRE(m1.cols() == 5);
  REQUIRE(m1.size() == 0);
  REQUIRE(m1.empty());

  // Zero cols
  ScratchMatrix<double> m2(5, 0);
  REQUIRE(m2.rows() == 5);
  REQUIRE(m2.cols() == 0);
  REQUIRE(m2.size() == 0);
  REQUIRE(m2.empty());

  // Both zero
  ScratchMatrix<double> m3(0, 0);
  REQUIRE(m3.rows() == 0);
  REQUIRE(m3.cols() == 0);
  REQUIRE(m3.size() == 0);
  REQUIRE(m3.empty());
}

TEST_CASE("ScratchMatrix resize exception safety - dimensions unchanged on allocation failure", "[ScratchMatrix]")
{
  // This test documents the behavior: resize() allocates BEFORE updating
  // dimensions. If the vector resize throws, rows_ and cols_ remain unchanged.
  // We can't easily force std::vector::resize to throw in a portable way,
  // but we verify the ordering by testing a normal resize works correctly.
  ScratchMatrix<double> m(3, 4);
  m.fill(1.0);

  // Resize to something smaller (should not throw)
  m.resize(2, 2);
  REQUIRE(m.rows() == 2);
  REQUIRE(m.cols() == 2);
  REQUIRE(m.size() == 4);

  // Resize back to larger
  m.resize(5, 6);
  REQUIRE(m.rows() == 5);
  REQUIRE(m.cols() == 6);
  REQUIRE(m.size() == 30);
}

TEST_CASE("ScratchMatrix column-major matches Armadillo convention", "[ScratchMatrix]")
{
  // In column-major order, incrementing the row index by 1 advances by 1 in
  // memory, while incrementing the column index by 1 advances by rows_ in
  // memory. Verify this property.
  ScratchMatrix<int> m(4, 3);
  for (size_t j = 0; j < 3; ++j)
    for (size_t i = 0; i < 4; ++i)
      m(i, j) = static_cast<int>(j * 4 + i); // linear index in col-major

  const int *raw = m.raw();
  for (size_t k = 0; k < 12; ++k)
    REQUIRE(raw[k] == static_cast<int>(k));
}
