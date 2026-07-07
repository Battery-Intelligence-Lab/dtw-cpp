/**
 * @file test_decode_pair.cpp
 * @brief Regression tests for the SSOT upper-triangle pair decode and the
 *        64-bit index arithmetic that depends on it.
 *
 * Targets audit 2026-06-01 Criticals #2/#3 and PLAN Tasks 0.2 & 0.7:
 *   - Critical #2: the Metal decode_pair used an FP32 sqrt, over-estimating the
 *     row near row boundaries so the last pair of most rows decoded WRONG for
 *     large N (the single upward `if` could not recover an over-estimate).
 *   - Critical #3 / Task 0.7: every int32 copy overflowed the i*(2N-i-1)
 *     intermediate (and the si*N+sj matrix index) at N >= 46341.
 *
 * The shared dtwc::detail::decode_pair (FP64 seed + int64 correction) must
 * decode correctly at N = 8192 and N = 50000 and match the audited-correct MPI
 * reference formula bit-for-bit. Pure host C++ — buildable with no CUDA / Metal
 * / MPI toolchain, so it runs everywhere.
 *
 * WHY THE UNFIXED CODE FAILS THESE:
 *   - old_metal_fp32_decode() (a verbatim copy of the retired Metal decode) is
 *     asserted to be WRONG for ~5000 of the ~8191 row-boundary indices at
 *     N = 8192 — that assertion pins Critical #2.
 *   - The int32 matrix index for the last pair at N = 46342 is shown to exceed
 *     INT32_MAX — that pins Critical #3 / Task 0.7; the shared decode returns
 *     int64 so the CUDA `si * N_series + sj` sites now compute in 64-bit.
 */

#include <catch2/catch_test_macros.hpp>

#include <detail/decode_pair.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

/// Oracle A: the audited-correct MPI reference formula (FP64 seed + `while`),
/// reproduced as it stood in mpi_distance_matrix.cpp before the SSOT extraction.
/// This is the "correct copy" the plan says the shared decode must match.
std::pair<std::int64_t, std::int64_t> mpi_reference_decode(std::int64_t k, std::int64_t N)
{
  const double Nd = static_cast<double>(N);
  auto i = static_cast<std::int64_t>(std::floor(
      Nd - 0.5 - std::sqrt((Nd - 0.5) * (Nd - 0.5) - 2.0 * static_cast<double>(k))));
  std::int64_t row_start = i * (2 * N - i - 1) / 2;
  while (row_start + (N - i - 1) <= k) {
    row_start += (N - i - 1);
    ++i;
  }
  const std::int64_t j = i + 1 + (k - row_start);
  return { i, j };
}

/// Oracle B: the OLD Metal FP32 decode (int32 + single `if`), reproduced to
/// PIN Critical #2. It must produce WRONG (i, j) for many k at N = 8192.
void old_metal_fp32_decode(int k, int N, int &i, int &j)
{
  float Nf = static_cast<float>(N);
  float kf = static_cast<float>(k);
  i = static_cast<int>(std::floor(Nf - 0.5f - std::sqrt((Nf - 0.5f) * (Nf - 0.5f) - 2.0f * kf)));
  int row_start = i * (2 * N - i - 1) / 2;
  if (row_start + (N - i - 1) <= k) {
    row_start += (N - i - 1);
    ++i;
  }
  j = i + 1 + (k - row_start);
}

/// Independent re-encode: linear index of pair (i, j) with 0 <= i < j < N.
std::int64_t encode_pair(std::int64_t i, std::int64_t j, std::int64_t N)
{
  return i * (2 * N - i - 1) / 2 + (j - i - 1);
}

/// Correctness of the shared decode for one k. Independent of the decode method:
///   (a) matches the MPI reference exactly,
///   (b) round-trips: encode(decode(k)) == k,
///   (c) bounds: 0 <= i < j < N.
bool shared_decode_ok(std::int64_t k, std::int64_t N)
{
  std::int64_t i = -1, j = -1;
  dtwc::detail::decode_pair(k, N, i, j);
  const auto ref = mpi_reference_decode(k, N);
  return i == ref.first && j == ref.second && 0 <= i && i < j && j < N
      && encode_pair(i, j, N) == k;
}

/// Sweep k over [k0, k1) with the given stride; return the first failing k
/// (or -1 if all pass). REGISTERED pass band (zero tolerance): every sampled k
/// must satisfy shared_decode_ok().
std::int64_t first_failing_k(std::int64_t N, std::int64_t k0, std::int64_t k1,
                             std::int64_t stride)
{
  for (std::int64_t k = k0; k < k1; k += stride)
    if (!shared_decode_ok(k, N)) return k;
  return -1;
}

/// Check every row boundary for a sampled row (where the FP-seed correction
/// loop is most likely to be off by one). Returns first failing k or -1.
std::int64_t first_failing_row_boundary(std::int64_t N, std::int64_t i)
{
  if (i < 0 || i >= N - 1) return -1;
  const std::int64_t row_start = encode_pair(i, i + 1, N);
  const std::int64_t row_len = N - i - 1;
  const std::int64_t ks[] = { row_start,                    // (i, i+1)
                              row_start + row_len - 1,       // (i, N-1)
                              row_start + row_len / 2 };      // interior
  for (std::int64_t k : ks)
    if (!shared_decode_ok(k, N)) return k;
  return -1;
}

} // namespace

// ---------------------------------------------------------------------------
// Task 0.2 — decode correctness
// ---------------------------------------------------------------------------

TEST_CASE("decode_pair matches brute-force enumeration for small N", "[decode_pair]")
{
  std::int64_t bad = 0, bad_N = -1, bad_i = -1, bad_j = -1;
  for (std::int64_t N = 2; N <= 200 && bad == 0; ++N) {
    std::int64_t k = 0;
    for (std::int64_t bi = 0; bi < N; ++bi) {
      for (std::int64_t bj = bi + 1; bj < N; ++bj) {
        std::int64_t i = -1, j = -1;
        dtwc::detail::decode_pair(k, N, i, j);
        if (i != bi || j != bj) { ++bad; bad_N = N; bad_i = bi; bad_j = bj; break; }
        ++k;
      }
      if (bad) break;
    }
  }
  CAPTURE(bad_N, bad_i, bad_j);
  REQUIRE(bad == 0);
}

TEST_CASE("decode_pair is correct at N=8192", "[decode_pair]")
{
  // Critical #2 threshold: FP32 loses integer precision above 2^24 ~ 1.68e7,
  // and num_pairs = 8192*8191/2 ~ 3.35e7 exceeds that, so the retired FP32
  // decode went wrong in the upper half of the index range.
  constexpr std::int64_t N = 8192;
  const std::int64_t num_pairs = N * (N - 1) / 2;

  CHECK(first_failing_k(N, 0, num_pairs, 101) == -1); // dense strided sweep
  CHECK(shared_decode_ok(0, N));
  CHECK(shared_decode_ok(num_pairs - 1, N));
  for (std::int64_t k = num_pairs - 64; k < num_pairs; ++k) CHECK(shared_decode_ok(k, N));
  for (std::int64_t i : { std::int64_t(0), std::int64_t(1), std::int64_t(2),
                          std::int64_t(100), N / 4, N / 2, 3 * N / 4,
                          N - 3, N - 2 })
    CHECK(first_failing_row_boundary(N, i) == -1);
}

TEST_CASE("decode_pair is correct at N=50000", "[decode_pair]")
{
  // Critical #3 threshold: i*(2N-i-1) overflows int32 near N=46341; N=50000 is
  // safely past it. num_pairs ~ 1.25e9 is too large to sweep fully, so we test
  // a coarse global stride plus dense row boundaries around the overflow point.
  constexpr std::int64_t N = 50000;
  const std::int64_t num_pairs = N * (N - 1) / 2;

  CHECK(first_failing_k(N, 0, num_pairs, 999983) == -1); // ~1250 coarse samples
  CHECK(shared_decode_ok(0, N));
  CHECK(shared_decode_ok(num_pairs - 1, N));
  for (std::int64_t k = num_pairs - 64; k < num_pairs; ++k) CHECK(shared_decode_ok(k, N));
  for (std::int64_t i : { std::int64_t(0), std::int64_t(1), std::int64_t(2),
                          std::int64_t(46341), std::int64_t(46342),
                          N / 4, N / 2, 3 * N / 4, N - 3, N - 2 })
    CHECK(first_failing_row_boundary(N, i) == -1);
}

TEST_CASE("retired Metal FP32 decode is wrong at N=8192 (pins Critical #2)", "[decode_pair]")
{
  // The retired FP32 decode over-estimates the row near row boundaries (its
  // single upward `if` cannot decrement), so the LAST pair of most rows decodes
  // wrong. Sweeping those boundary k's makes the failure dense (~5000 of ~8191),
  // so this pin is not flaky. N=8192 < 46341 keeps old_metal_fp32_decode free of
  // int32 overflow, isolating the FP32-precision failure and staying UB-free.
  constexpr int N = 8192;

  int fp32_wrong = 0;
  int shared_wrong = 0;
  for (int i = 0; i < N - 1; ++i) {
    const std::int64_t k = encode_pair(i, N - 1, N); // last pair of row i == (i, N-1)
    const auto ref = mpi_reference_decode(k, N);

    int fi = -1, fj = -1;
    old_metal_fp32_decode(static_cast<int>(k), N, fi, fj);
    if (fi != ref.first || fj != ref.second) ++fp32_wrong;

    if (!shared_decode_ok(k, N)) ++shared_wrong;
  }

  CHECK(fp32_wrong > 0);     // the retired FP32 decode really is broken
  CHECK(shared_wrong == 0);  // the SSOT decode is correct everywhere
}

// ---------------------------------------------------------------------------
// Task 0.7 — 64-bit matrix-index arithmetic
// ---------------------------------------------------------------------------

TEST_CASE("last-pair matrix index overflows int32 at N=46342 (pins Task 0.7)", "[decode_pair][index]")
{
  // The CUDA kernels index the NxN result matrix as result_matrix[si*N + sj].
  // With int32 si/N/sj this overflows for N >= 46341; the SSOT decode now
  // returns int64 so the product is evaluated in 64-bit.
  constexpr std::int64_t N = 46342;
  const std::int64_t num_pairs = N * (N - 1) / 2;

  std::int64_t si = -1, sj = -1;
  dtwc::detail::decode_pair(num_pairs - 1, N, si, sj); // last pair == (N-2, N-1)
  REQUIRE(si == N - 2);
  REQUIRE(sj == N - 1);

  // decode_pair returns int64 so this expression is computed in 64 bits.
  static_assert(sizeof(decltype(si * N + sj)) >= 8,
                "matrix index must be 64-bit to avoid overflow at N>=46341");

  const std::int64_t idx = si * N + sj;      // row-major (si, sj)
  const std::int64_t idx_sym = sj * N + si;  // symmetric write (sj, si)
  const std::int64_t int32_max = std::numeric_limits<std::int32_t>::max();

  // Both indices exceed INT32_MAX -> an int32 computation would wrap negative
  // and read/write out of bounds. The 64-bit values are exact.
  CHECK(idx > int32_max);
  CHECK(idx_sym > int32_max);
  CHECK(idx == (N - 2) * N + (N - 1));
  CHECK(idx_sym == (N - 1) * N + (N - 2));
}
