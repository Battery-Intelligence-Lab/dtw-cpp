/**
 * @file test_env_device.cpp
 * @brief Unit tests for dtwc::Env device selection + `.env` HPC contract (Task 1.3).
 *
 * @details Every test drives the LIVE public entry point `dtwc::Env::set_device()`
 * (and the singleton `dtwc::env()`), per .claude/LESSONS.md "Tests must pin the
 * LIVE code path". The five DeviceError messages are authored VERBATIM in
 * docs/api-contract-2.0.md §6.1/§6.2 (FROZEN); the expected strings below are
 * transcribed from that contract and asserted byte-for-byte against what
 * `set_device()` throws.
 *
 * Coverage:
 *   - unknown device name  → DeviceError listing valid names               (§6.1)
 *   - `gpu` with no GPU backend compiled in → DeviceError naming the flag   (§6.1)
 *       (this build has no DTWC_HAS_CUDA/DTWC_HAS_METAL, so the real error fires)
 *   - device="hpc", three `.env` failure modes, each message verbatim      (§6.2):
 *       (1) no `.env` file        (Env pointed at an empty scratch dir)
 *       (2) `.env` missing a key  (fixture omits SLURM_HOST — the first key)
 *       (3) auth failure          (valid `.env`, injected failing auth probe)
 *   - device="hpc" success path   (valid `.env`, injected passing probe)
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <env.hpp>
#include <error.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

// ===========================================================================
// Registered expected values — VERBATIM from api-contract-2.0.md §6.1/§6.2.
// Registered here BEFORE any check so the run is judged against a pre-declared
// band, never a post-hoc one.
// ===========================================================================

static const std::string kMsgUnknownFoo =
  "[dtwc] unknown device 'foo'. Valid devices: cpu, gpu, gpu:N (aliases cuda, cuda:N), hpc.";

static const std::string kMsgGpuNotBuilt =
  "[dtwc] device='gpu' requested but this build has no GPU backend compiled in.\n"
  "Rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, on macOS, -DDTWC_ENABLE_METAL=ON.\n"
  "This build will not silently fall back to CPU.";

static const std::string kMsgNoEnv =
  "[dtwc] device='hpc' requires a .env file at the repository root, but none was found.\n"
  "Copy scripts/slurm/env.example to .env and set SLURM_HOST, SLURM_USER, and SLURM_REMOTE_BASE.\n"
  "Example .env:\n"
  "  SLURM_HOST=arc-login.arc.ox.ac.uk\n"
  "  SLURM_USER=abcd1234\n"
  "  SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs";

static const std::string kMsgMissingHost =
  "[dtwc] device='hpc': the .env file is missing required key 'SLURM_HOST'.\n"
  "Set it in .env at the repository root. Example .env:\n"
  "  SLURM_HOST=arc-login.arc.ox.ac.uk\n"
  "  SLURM_USER=abcd1234\n"
  "  SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs";

static const std::string kMsgAuthFail =
  "[dtwc] device='hpc': could not authenticate to SLURM host 'arc-login.arc.ox.ac.uk' as user 'abcd1234'.\n"
  "Check that your SSH key is authorized on that host (ssh abcd1234@arc-login.arc.ox.ac.uk must succeed without a password prompt) and that SLURM_HOST and SLURM_USER in .env are correct.";

// ===========================================================================
// Test fixtures — isolated scratch directory + `.env` writer.
// ===========================================================================

static fs::path make_scratch_dir()
{
  static unsigned counter = 0;
  const auto name = "dtwc_env_test_" + std::to_string(counter++) + "_"
                  + std::to_string(static_cast<unsigned long long>(
                      std::hash<std::string>{}(__FILE__)) & 0xffffu);
  const auto dir = fs::temp_directory_path() / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

static void write_file(const fs::path &path, const std::string &contents)
{
  std::ofstream out(path, std::ios::binary);
  out << contents;
}

// Return the message thrown by set_device(name), or FAIL if it did not throw.
static std::string device_error_message(dtwc::Env &e, const std::string &name)
{
  try {
    e.set_device(name);
  } catch (const dtwc::DeviceError &err) {
    return err.what();
  }
  FAIL("set_device('" + name + "') did not throw dtwc::DeviceError");
  return {};
}

// ===========================================================================
// Unknown device name → DeviceError (drives Env::set_device, §6.1).
// ===========================================================================

TEST_CASE("Env::set_device unknown name -> DeviceError listing valid names", "[env][device]")
{
  dtwc::Env e;
  REQUIRE_THROWS_AS(e.set_device("foo"), dtwc::DeviceError);
  REQUIRE(device_error_message(e, "foo") == kMsgUnknownFoo);

  // No silent mutation: a rejected request leaves the device unchanged (CPU).
  REQUIRE(e.device() == dtwc::Device::CPU);
}

// ===========================================================================
// Accepted names parse (drives Env::set_device happy paths, §6.1).
// ===========================================================================

TEST_CASE("Env::set_device accepts cpu (case-insensitive)", "[env][device]")
{
  dtwc::Env e;
  e.set_device("CPU");
  REQUIRE(e.device() == dtwc::Device::CPU);
  REQUIRE(e.device_index() == 0);
}

// ===========================================================================
// gpu on a build with no GPU backend → DeviceError naming the CMake flag.
// LIVE path: this build defines neither DTWC_HAS_CUDA nor DTWC_HAS_METAL, so the
// real compile-time branch throws (no silent CPU fallback). The #if keeps the
// test correct if it is ever compiled in a GPU build.
// ===========================================================================

TEST_CASE("Env::set_device gpu without a GPU build -> DeviceError naming the flag", "[env][device]")
{
  dtwc::Env e;
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  e.set_device("gpu");
  REQUIRE(e.device() == dtwc::Device::GPU);
#else
  REQUIRE_THROWS_AS(e.set_device("gpu"), dtwc::DeviceError);
  REQUIRE(device_error_message(e, "gpu") == kMsgGpuNotBuilt);

  // Aliases cuda / gpu:0 route to the SAME message (gpu ≡ cuda).
  REQUIRE(device_error_message(e, "cuda") == kMsgGpuNotBuilt);
  REQUIRE(device_error_message(e, "gpu:0") == kMsgGpuNotBuilt);

  // No silent fallback: device stays CPU after the rejected GPU request.
  REQUIRE(e.device() == dtwc::Device::CPU);
#endif
}

// ===========================================================================
// device="hpc" — .env failure mode (1): no file (drives Env::select_hpc, §6.2).
// ===========================================================================

TEST_CASE("Env::set_device hpc with no .env file -> DeviceError (verbatim)", "[env][hpc]")
{
  const auto dir = make_scratch_dir(); // empty: contains no .env
  REQUIRE_FALSE(fs::exists(dir / ".env"));

  dtwc::Env e;
  e.set_env_file_dir(dir);

  REQUIRE(device_error_message(e, "hpc") == kMsgNoEnv);
  REQUIRE(e.device() == dtwc::Device::CPU); // no silent fallback

  fs::remove_all(dir);
}

// ===========================================================================
// device="hpc" — .env failure mode (2): missing key (drives Env::select_hpc).
// Fixture omits SLURM_HOST (the first required key) — the pinned case in §6.2.
// ===========================================================================

TEST_CASE("Env::set_device hpc with .env missing SLURM_HOST -> DeviceError (verbatim)", "[env][hpc]")
{
  const auto dir = make_scratch_dir();
  write_file(dir / ".env",
             "SLURM_USER=abcd1234\n"
             "SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs\n");

  dtwc::Env e;
  e.set_env_file_dir(dir);

  REQUIRE(device_error_message(e, "hpc") == kMsgMissingHost);
  REQUIRE(e.device() == dtwc::Device::CPU);

  fs::remove_all(dir);
}

// ===========================================================================
// device="hpc" — .env failure mode (3): auth failure (drives Env::select_hpc).
// A valid .env is present; an injected probe returns false so the check is
// deterministic and offline (no real ssh). Message names the host + user tried.
// ===========================================================================

TEST_CASE("Env::set_device hpc auth failure -> DeviceError naming host+user (verbatim)", "[env][hpc]")
{
  const auto dir = make_scratch_dir();
  write_file(dir / ".env",
             "SLURM_HOST=arc-login.arc.ox.ac.uk\n"
             "SLURM_USER=abcd1234\n"
             "SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs\n");

  dtwc::Env e;
  e.set_env_file_dir(dir);
  e.set_hpc_auth_probe([](const std::string &, const std::string &) { return false; });

  REQUIRE(device_error_message(e, "hpc") == kMsgAuthFail);
  REQUIRE(e.device() == dtwc::Device::CPU);

  fs::remove_all(dir);
}

// ===========================================================================
// device="hpc" — success path: valid .env + a passing probe → Device::HPC.
// Confirms the three checks are gates, not unconditional failures.
// ===========================================================================

TEST_CASE("Env::set_device hpc succeeds with valid .env and a passing probe", "[env][hpc]")
{
  const auto dir = make_scratch_dir();
  write_file(dir / ".env",
             "SLURM_HOST=arc-login.arc.ox.ac.uk\n"
             "SLURM_USER=abcd1234\n"
             "SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs\n");

  dtwc::Env e;
  e.set_env_file_dir(dir);
  e.set_hpc_auth_probe([](const std::string &h, const std::string &u) {
    return h == "arc-login.arc.ox.ac.uk" && u == "abcd1234";
  });

  e.set_device("hpc");
  REQUIRE(e.device() == dtwc::Device::HPC);

  fs::remove_all(dir);
}

// ===========================================================================
// Singleton dtwc::env(): one process-wide instance; set_device/threads live.
// ===========================================================================

TEST_CASE("dtwc::env() is a usable process-wide singleton", "[env]")
{
  REQUIRE(&dtwc::env() == &dtwc::env()); // same instance on every call

  dtwc::env().set_device("cpu");
  REQUIRE(dtwc::env().device() == dtwc::Device::CPU);
  REQUIRE(dtwc::to_string(dtwc::env().device()) == "cpu");

  REQUIRE(dtwc::env().threads() >= 1); // OpenMP max threads, or 1 without OpenMP
}
