/**
 * @file decode_pair.hpp
 * @brief Single source of truth (SSOT) for decoding a linear upper-triangle
 *        pair index into its (i, j) row/column coordinates.
 *
 * @details The upper triangle of an N x N matrix stores N*(N-1)/2 entries,
 *          enumerated row-major:
 *            k=0 -> (0,1), k=1 -> (0,2), ..., k=N-2 -> (0,N-1),
 *            k=N-1 -> (1,2), ...
 *          Row i starts at linear index  i*(2N - i - 1)/2.
 *
 *          Before 2026-07 three divergent copies of this decode existed
 *          (CUDA: int/if, MPI: size_t/while, Metal: float/sqrt). The Metal
 *          FP32 copy produced wrong / out-of-bounds pairs for N > ~4096, and
 *          every int32 copy overflowed the  i*(2N-i-1)  intermediate at
 *          N >= 46341. This header is the SSOT that fixes both:
 *            - decode_pair(): host + CUDA device, FP64 seed + 64-bit integer
 *              correction loop (matches the audited-correct MPI copy).
 *            - kDecodePairMSL: Metal Shading Language source (integer-only,
 *              64-bit `long`). MSL has no FP64, so it uses an exact integer
 *              square-root seed instead; it produces bit-identical (i, j).
 *
 * @see .claude/summaries/handoff-2026-06-01-adversarial-audit.md (Critical 2, 3)
 */
#pragma once

#include <cstdint>

#include <cmath>
#include <math.h> // guarantees ::sqrt/::floor in the global namespace (host)

#if defined(__CUDACC__)
#define DTWC_DECODE_PAIR_HD __host__ __device__
#else
#define DTWC_DECODE_PAIR_HD
#endif

namespace dtwc::detail {

/**
 * @brief Decode linear upper-triangle index @p k into pair (@p i, @p j),
 *        with 0 <= i < j < N.
 *
 * FP64 seed (exact for N up to ~5.7e7, where every relevant integer is < 2^53)
 * plus a 64-bit integer correction loop, so the intermediate i*(2N-i-1) never
 * overflows (the retired int32 copies wrapped at N >= 46341).
 *
 * @pre 0 <= k < N*(N-1)/2  and  N >= 2.
 */
DTWC_DECODE_PAIR_HD inline void decode_pair(std::int64_t k, std::int64_t N,
                                            std::int64_t &i, std::int64_t &j)
{
  const double Nd = static_cast<double>(N);
  const double kd = static_cast<double>(k);

  // Approximate the row from the quadratic formula, then correct exactly.
  std::int64_t row = static_cast<std::int64_t>(
      floor(Nd - 0.5 - sqrt((Nd - 0.5) * (Nd - 0.5) - 2.0 * kd)));
  if (row < 0) row = 0; // defensive: valid k already yields row >= 0

  // Row `row` starts at linear index row*(2N - row - 1)/2. All arithmetic is
  // 64-bit, so the product never wraps.
  std::int64_t row_start = row * (2 * N - row - 1) / 2;

  // Correct any residual FP rounding upward. The audited-correct MPI copy used
  // `while`; the buggy CUDA/Metal copies used a single `if` that could not
  // recover when the seed underestimated the row by more than one.
  while (row_start + (N - row - 1) <= k) {
    row_start += (N - row - 1);
    ++row;
  }

  i = row;
  j = row + 1 + (k - row_start);
}

/**
 * @brief Metal Shading Language source for the same decode.
 *
 * MSL supports neither `double` nor a wide-enough FP32 mantissa for large N,
 * so this uses an exact 64-bit integer square root (float seed + integer
 * correction). `long` is a 64-bit signed integer in MSL (feature set common to
 * Apple-silicon GPUs). It is exposed as a string so metal_dtw.mm can prepend it
 * to its runtime-compiled kernel library — keeping this the ONLY definition of
 * the decode across all three backends. Produces bit-identical (i, j) to
 * decode_pair() above.
 */
inline constexpr const char *kDecodePairMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// SSOT decode: see dtwc/detail/decode_pair.hpp. Integer-only, overflow-safe.
// Produces bit-identical (i, j) to the FP64 decode above.
static inline void decode_pair(long k, long N, thread long &i, thread long &j)
{
  // Exact 64-bit integer square root: float seed + integer correction.
  const long a = 2 * N - 1;
  const long disc = a * a - 8 * k;         // < 4N^2, fits in 64-bit `long`
  long s = (long)sqrt((float)disc);        // approximate seed
  while ((s + 1) * (s + 1) <= disc) ++s;   // correct up
  while (s * s > disc) --s;                 // correct down -> s = isqrt(disc)

  // row = floor((a - sqrt(disc)) / 2). Using the floored isqrt s, (a - s) / 2
  // can be up to one too high, so correct in BOTH directions (an up-only loop
  // like the retired copies cannot recover from an overestimate).
  long row = (a - s) / 2;
  while (row > 0 && row * (2 * N - row - 1) / 2 > k) --row; // correct down
  long row_start = row * (2 * N - row - 1) / 2;
  while (row_start + (N - row - 1) <= k) {                  // correct up
    row_start += (N - row - 1);
    ++row;
  }
  i = row;
  j = row + 1 + (k - row_start);
}
)MSL";

} // namespace dtwc::detail
