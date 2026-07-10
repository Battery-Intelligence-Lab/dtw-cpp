#include <core/sha256.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

std::string hex_digest(const dtwc::core::detail::Sha256::Digest &digest)
{
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest)
    out << std::setw(2) << static_cast<unsigned>(byte);
  return out.str();
}

} // namespace

TEST_CASE("Sha256 matches standard vectors and streaming boundaries", "[sha256][cache]")
{
  using dtwc::core::detail::Sha256;

  SECTION("empty")
  {
    Sha256 hash;
    REQUIRE(hex_digest(hash.digest()) ==
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  }

  SECTION("abc streamed one byte at a time")
  {
    Sha256 hash;
    for (const char byte : std::string{"abc"})
      hash.update(&byte, 1);
    REQUIRE(hex_digest(hash.digest()) ==
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
  }

  SECTION("multi-block vector")
  {
    const std::string input =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    Sha256 hash;
    hash.update(input.data(), 13);
    hash.update(input.data() + 13, input.size() - 13);
    REQUIRE(hex_digest(hash.digest()) ==
            "248d6a61d20638b8e5c026930c3e6039"
            "a33ce45964ff2167f6ecedd419db06c1");
  }
}
