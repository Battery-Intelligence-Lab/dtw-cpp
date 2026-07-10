/**
 * @file sha256.hpp
 * @brief Small dependency-free streaming SHA-256 implementation.
 *
 * Used for persistent-cache identities. The digest is a format contract, so it
 * must not depend on std::hash (which is implementation-defined and may be
 * salted) or on host word size.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace dtwc::core::detail {

class Sha256
{
public:
  using Digest = std::array<std::uint8_t, 32>;

  void update(const void *bytes, std::size_t size)
  {
    const auto *input = static_cast<const std::uint8_t *>(bytes);
    total_size_ += static_cast<std::uint64_t>(size);

    if (buffer_size_ != 0) {
      const std::size_t take = std::min(size, block_size - buffer_size_);
      std::memcpy(buffer_.data() + buffer_size_, input, take);
      buffer_size_ += take;
      input += take;
      size -= take;
      if (buffer_size_ == block_size) {
        transform(buffer_.data());
        buffer_size_ = 0;
      }
    }

    while (size >= block_size) {
      transform(input);
      input += block_size;
      size -= block_size;
    }

    if (size != 0) {
      std::memcpy(buffer_.data(), input, size);
      buffer_size_ = size;
    }
  }

  void update(std::span<const std::uint8_t> bytes)
  {
    update(bytes.data(), bytes.size());
  }

  Digest digest() const
  {
    Sha256 finished = *this;
    const std::uint64_t bit_size = total_size_ * 8u;

    std::array<std::uint8_t, block_size> padding{};
    padding[0] = 0x80u;
    const std::size_t padding_size =
      buffer_size_ < 56 ? 56 - buffer_size_ : 120 - buffer_size_;
    finished.update(padding.data(), padding_size);

    std::array<std::uint8_t, 8> encoded_size{};
    for (std::size_t i = 0; i < encoded_size.size(); ++i)
      encoded_size[encoded_size.size() - 1 - i] =
        static_cast<std::uint8_t>(bit_size >> (i * 8));
    finished.update(encoded_size);

    Digest result{};
    for (std::size_t i = 0; i < finished.state_.size(); ++i) {
      result[4 * i + 0] = static_cast<std::uint8_t>(finished.state_[i] >> 24);
      result[4 * i + 1] = static_cast<std::uint8_t>(finished.state_[i] >> 16);
      result[4 * i + 2] = static_cast<std::uint8_t>(finished.state_[i] >> 8);
      result[4 * i + 3] = static_cast<std::uint8_t>(finished.state_[i]);
    }
    return result;
  }

private:
  static constexpr std::size_t block_size = 64;

  std::array<std::uint32_t, 8> state_{
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
  };
  std::array<std::uint8_t, block_size> buffer_{};
  std::size_t buffer_size_{ 0 };
  std::uint64_t total_size_{ 0 };

  static constexpr std::uint32_t rotate_right(std::uint32_t value, unsigned count)
  {
    return (value >> count) | (value << (32u - count));
  }

  void transform(const std::uint8_t *block)
  {
    static constexpr std::array<std::uint32_t, 64> constants{
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };

    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
      words[i] = (static_cast<std::uint32_t>(block[4 * i + 0]) << 24)
               | (static_cast<std::uint32_t>(block[4 * i + 1]) << 16)
               | (static_cast<std::uint32_t>(block[4 * i + 2]) << 8)
               | static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7)
                             ^ rotate_right(words[i - 15], 18)
                             ^ (words[i - 15] >> 3);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17)
                             ^ rotate_right(words[i - 2], 19)
                             ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (std::size_t i = 0; i < words.size(); ++i) {
      const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11)
                               ^ rotate_right(e, 25);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t temp1 = h + sum1 + choose + constants[i] + words[i];
      const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13)
                               ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }
};

} // namespace dtwc::core::detail
