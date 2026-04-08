/**
 * @file crc32.hpp
 * @brief Simple CRC32 utility for binary header validation.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace dtwc::core::detail {

/// Simple CRC32 (no lookup table -- only used on small headers).
inline uint32_t crc32_naive(const uint8_t *data, size_t len)
{
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      if (crc & 1u)
        crc = (crc >> 1) ^ 0xEDB88320u;
      else
        crc >>= 1;
    }
  }
  return ~crc;
}

} // namespace dtwc::core::detail
