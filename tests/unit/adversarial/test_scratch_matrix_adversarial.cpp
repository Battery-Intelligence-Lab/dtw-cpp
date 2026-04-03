/**
 * @file test_scratch_matrix_adversarial.cpp
 * @brief Adversarial tests for ScratchMatrix — written from SPEC, not from code.
 *
 * Tests column-major layout guarantees, edge cases, and thread safety.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <core/scratch_matrix.hpp>

#include <catch2/catch_test_macros.hpp>

#include <thread>
#include <vector>
#include <atomic>
#include <cstring> // for memcmp
#include <numeric> // for iota

using namespace dtwc::core;

// ============================================================================
// Area 1: ScratchMatrix Column-Major Layout
// ============================================================================

TEST_CASE("Adversarial: column-major raw pointer matches operator()(i,j)", "[ScratchMatrix][adversarial]")
{
  // SPEC: data_[j * rows_ + i] == operator()(i, j)
  // Use a non-square matrix to make row-major vs column-major distinguishable.
  constexpr size_t R = 4, C = 7;
  ScratchMatrix<double> m(R, C);

  // Fill with unique values via operator()
  double val = 1.0;
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j)
      m(i, j) = val++;

  // Verify raw pointer at column-major index matches operator()
  const double *raw = m.raw();
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE(raw[j * R + i] == m(i, j));
    }
  }
}

TEST_CASE("Adversarial: row-major indexing would give WRONG values", "[ScratchMatrix][adversarial]")
{
  // If the matrix were row-major, raw[i*cols+j] would equal operator()(i,j).
  // For a non-square matrix with distinct values, this must NOT hold for all (i,j).
  constexpr size_t R = 3, C = 5;
  ScratchMatrix<double> m(R, C);

  // Fill with unique values
  double val = 1.0;
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j)
      m(i, j) = val++;

  // At least one element at row-major index must differ from operator()(i,j)
  // (In fact, for this 3x5 matrix with sequential values, most will differ.)
  int mismatches = 0;
  const double *raw = m.raw();
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      size_t row_major_idx = i * C + j;
      if (row_major_idx < R * C && raw[row_major_idx] != m(i, j))
        ++mismatches;
    }
  }

  // For a non-square matrix with unique sequential values, row-major indexing
  // must disagree with column-major for at least some elements.
  // Specifically, only elements where j*R+i == i*C+j would match,
  // which requires j*(R-1) == i*(C-1). For R=3,C=5 this is rare.
  REQUIRE(mismatches > 0);
}

TEST_CASE("Adversarial: elements within a column are contiguous in memory", "[ScratchMatrix][adversarial]")
{
  // SPEC: column-major means elements (0,j), (1,j), (2,j), ... are contiguous.
  constexpr size_t R = 5, C = 4;
  ScratchMatrix<double> m(R, C);

  for (size_t j = 0; j < C; ++j) {
    for (size_t i = 0; i + 1 < R; ++i) {
      INFO("column=" << j << " row=" << i);
      // Address of m(i+1,j) must be exactly 1 element after m(i,j)
      REQUIRE(&m(i + 1, j) == &m(i, j) + 1);
    }
  }
}

TEST_CASE("Adversarial: elements within a ROW are NOT contiguous (stride = rows)", "[ScratchMatrix][adversarial]")
{
  // In column-major, stepping across columns in the same row has stride = rows_.
  constexpr size_t R = 6, C = 3;
  ScratchMatrix<double> m(R, C);

  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j + 1 < C; ++j) {
      INFO("row=" << i << " col=" << j);
      ptrdiff_t stride = &m(i, j + 1) - &m(i, j);
      REQUIRE(stride == static_cast<ptrdiff_t>(R));
    }
  }
}

TEST_CASE("Adversarial: fill() fills every element", "[ScratchMatrix][adversarial]")
{
  constexpr size_t R = 7, C = 11;
  ScratchMatrix<double> m(R, C);

  // Set some non-42 values first
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j)
      m(i, j) = static_cast<double>(i * 100 + j);

  m.fill(42.0);

  // Check via operator()
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE(m(i, j) == 42.0);
    }
  }

  // Also check via raw pointer (catches any mismatch between logical and physical)
  const double *raw = m.raw();
  for (size_t k = 0; k < R * C; ++k) {
    INFO("raw index=" << k);
    REQUIRE(raw[k] == 42.0);
  }
}

TEST_CASE("Adversarial: resize down then access", "[ScratchMatrix][adversarial]")
{
  ScratchMatrix<double> m(10, 10);
  m.fill(99.0);

  m.resize(5, 5);
  REQUIRE(m.rows() == 5);
  REQUIRE(m.cols() == 5);
  REQUIRE(m.size() == 25);

  // Must be able to write and read all elements in new dimensions
  for (size_t i = 0; i < 5; ++i)
    for (size_t j = 0; j < 5; ++j)
      m(i, j) = static_cast<double>(i + j);

  for (size_t i = 0; i < 5; ++i)
    for (size_t j = 0; j < 5; ++j)
      REQUIRE(m(i, j) == static_cast<double>(i + j));
}

TEST_CASE("Adversarial: resize up after resize down", "[ScratchMatrix][adversarial]")
{
  ScratchMatrix<double> m(5, 5);
  m.fill(1.0);

  m.resize(100, 100);
  REQUIRE(m.rows() == 100);
  REQUIRE(m.cols() == 100);
  REQUIRE(m.size() == 10000);

  // Must be able to write and read at far corners
  m(99, 99) = 777.0;
  m(0, 99) = 888.0;
  m(99, 0) = 999.0;
  REQUIRE(m(99, 99) == 777.0);
  REQUIRE(m(0, 99) == 888.0);
  REQUIRE(m(99, 0) == 999.0);
}

TEST_CASE("Adversarial: resize to zero dimensions does not crash", "[ScratchMatrix][adversarial]")
{
  ScratchMatrix<double> m(10, 10);
  m.fill(5.0);

  // These should not crash or throw
  REQUIRE_NOTHROW(m.resize(0, 0));
  REQUIRE(m.rows() == 0);
  REQUIRE(m.cols() == 0);
  REQUIRE(m.size() == 0);
  REQUIRE(m.empty());

  // Resize back from zero
  REQUIRE_NOTHROW(m.resize(3, 3));
  REQUIRE(m.rows() == 3);
  REQUIRE(m.cols() == 3);
  REQUIRE(m.size() == 9);
}

TEST_CASE("Adversarial: rows() and cols() correct after resize", "[ScratchMatrix][adversarial]")
{
  ScratchMatrix<double> m;
  REQUIRE(m.rows() == 0);
  REQUIRE(m.cols() == 0);

  m.resize(3, 7);
  REQUIRE(m.rows() == 3);
  REQUIRE(m.cols() == 7);

  m.resize(7, 3);
  REQUIRE(m.rows() == 7);
  REQUIRE(m.cols() == 3);

  m.resize(1, 1);
  REQUIRE(m.rows() == 1);
  REQUIRE(m.cols() == 1);
}

TEST_CASE("Adversarial: 1x1 matrix behaves correctly", "[ScratchMatrix][adversarial]")
{
  ScratchMatrix<double> m(1, 1);
  m(0, 0) = 3.14;
  REQUIRE(m(0, 0) == 3.14);
  REQUIRE(m.raw()[0] == 3.14);
  REQUIRE(m.rows() == 1);
  REQUIRE(m.cols() == 1);
  REQUIRE(m.size() == 1);
}

TEST_CASE("Adversarial: tall skinny matrix column-major layout", "[ScratchMatrix][adversarial]")
{
  // Extreme aspect ratio: 100 rows, 2 columns
  constexpr size_t R = 100, C = 2;
  ScratchMatrix<int> m(R, C);

  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j)
      m(i, j) = static_cast<int>(i * C + j);

  const int *raw = m.raw();
  // Column 0 occupies raw[0..99], column 1 occupies raw[100..199]
  for (size_t i = 0; i < R; ++i) {
    REQUIRE(raw[i] == m(i, 0));
    REQUIRE(raw[R + i] == m(i, 1));
  }
}

// ============================================================================
// Area 2: ScratchMatrix Thread Safety
// ============================================================================

TEST_CASE("Adversarial: thread_local ScratchMatrix independence", "[ScratchMatrix][adversarial][threads]")
{
  // Two threads each use their own thread_local ScratchMatrix.
  // They fill with different values and must not interfere.
  constexpr size_t R = 50, C = 50;
  std::atomic<bool> t1_ok{ true };
  std::atomic<bool> t2_ok{ true };

  auto worker = [&](double fill_val, std::atomic<bool> &ok) {
    thread_local ScratchMatrix<double> local_m;
    local_m.resize(R, C);
    local_m.fill(fill_val);

    // Busy work to increase chance of interference
    for (int iter = 0; iter < 100; ++iter) {
      for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
          if (local_m(i, j) != fill_val) {
            ok.store(false);
            return;
          }
        }
      }
      local_m.fill(fill_val); // re-fill in case of interference
    }
  };

  std::thread t1(worker, 1.0, std::ref(t1_ok));
  std::thread t2(worker, 2.0, std::ref(t2_ok));
  t1.join();
  t2.join();

  REQUIRE(t1_ok.load());
  REQUIRE(t2_ok.load());
}

TEST_CASE("Adversarial: concurrent resize on thread_local matrices", "[ScratchMatrix][adversarial][threads]")
{
  constexpr int NUM_THREADS = 4;
  std::vector<std::atomic<bool>> results(NUM_THREADS);
  for (auto &r : results) r.store(true);

  auto worker = [](size_t thread_id, std::atomic<bool> &ok) {
    thread_local ScratchMatrix<double> local_m;

    for (int iter = 0; iter < 50; ++iter) {
      size_t r = (thread_id + 1) * 10 + iter;
      size_t c = (thread_id + 1) * 5 + iter;
      local_m.resize(r, c);
      local_m.fill(static_cast<double>(thread_id));

      if (local_m.rows() != r || local_m.cols() != c) {
        ok.store(false);
        return;
      }

      // Verify all values
      for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
          if (local_m(i, j) != static_cast<double>(thread_id)) {
            ok.store(false);
            return;
          }
        }
      }
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < NUM_THREADS; ++t)
    threads.emplace_back(worker, static_cast<size_t>(t), std::ref(results[t]));

  for (auto &th : threads)
    th.join();

  for (int t = 0; t < NUM_THREADS; ++t) {
    INFO("Thread " << t << " failed");
    REQUIRE(results[t].load());
  }
}

TEST_CASE("Adversarial: after resize, dimensions reflect new size", "[ScratchMatrix][adversarial]")
{
  ScratchMatrix<double> m(10, 10);

  // Fill with known pattern
  for (size_t i = 0; i < 10; ++i)
    for (size_t j = 0; j < 10; ++j)
      m(i, j) = static_cast<double>(i * 10 + j);

  // Resize to different dimensions — old logical layout is gone
  m.resize(20, 20);
  REQUIRE(m.rows() == 20);
  REQUIRE(m.cols() == 20);
  REQUIRE(m.size() == 400);

  // Fill new matrix and verify column-major layout holds for new dimensions
  for (size_t i = 0; i < 20; ++i)
    for (size_t j = 0; j < 20; ++j)
      m(i, j) = static_cast<double>(i + j * 100);

  const double *raw = m.raw();
  for (size_t i = 0; i < 20; ++i) {
    for (size_t j = 0; j < 20; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE(raw[j * 20 + i] == static_cast<double>(i + j * 100));
    }
  }
}

TEST_CASE("Adversarial: resize with one zero dimension", "[ScratchMatrix][adversarial]")
{
  ScratchMatrix<double> m(5, 5);
  m.fill(1.0);

  // Resize to zero rows, non-zero cols
  REQUIRE_NOTHROW(m.resize(0, 10));
  REQUIRE(m.rows() == 0);
  REQUIRE(m.cols() == 10);
  REQUIRE(m.size() == 0);
  REQUIRE(m.empty());

  // Resize to non-zero rows, zero cols
  REQUIRE_NOTHROW(m.resize(10, 0));
  REQUIRE(m.rows() == 10);
  REQUIRE(m.cols() == 0);
  REQUIRE(m.size() == 0);
  REQUIRE(m.empty());
}

TEST_CASE("Adversarial: large matrix fill and column-major spot check", "[ScratchMatrix][adversarial]")
{
  // Larger matrix to stress-test — still fast (1M elements)
  constexpr size_t R = 1000, C = 1000;
  ScratchMatrix<double> m(R, C);
  m.fill(-1.0);

  // Set specific elements and verify via raw pointer
  m(0, 0) = 0.0;
  m(R - 1, 0) = 1.0;
  m(0, C - 1) = 2.0;
  m(R - 1, C - 1) = 3.0;
  m(500, 500) = 4.0;

  const double *raw = m.raw();
  REQUIRE(raw[0 * R + 0] == 0.0);         // m(0,0)
  REQUIRE(raw[0 * R + (R - 1)] == 1.0);   // m(R-1, 0)
  REQUIRE(raw[(C - 1) * R + 0] == 2.0);   // m(0, C-1)
  REQUIRE(raw[(C - 1) * R + (R - 1)] == 3.0); // m(R-1, C-1)
  REQUIRE(raw[500 * R + 500] == 4.0);     // m(500, 500)

  // Verify fill worked for a non-set element
  REQUIRE(raw[1 * R + 1] == -1.0); // m(1, 1) was not overwritten
}
