/**
 * @file env.cpp
 * @brief Implementation of dtwc::Env — device registry + `.env` HPC credentials.
 *
 * @details See env.hpp. The DeviceError messages emitted here are authored
 * verbatim in docs/api-contract-2.0.md §6.1/§6.2 (FROZEN) and asserted
 * byte-for-byte in tests/unit/test_env_device.cpp — do not reword them without a
 * PLAN.md decision entry updating the contract.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 07 Jul 2026
 */

#include "env.hpp"
#include "error.hpp"

#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <system_error>

#ifdef DTWC_HAS_OPENMP
#include <omp.h>
#endif

#ifndef _WIN32
#include <sys/wait.h> // WIFEXITED / WEXITSTATUS for std::system() return decoding
#endif

namespace fs = std::filesystem;

namespace dtwc {

namespace {

// ---------------------------------------------------------------------------
// Small string helpers.
// ---------------------------------------------------------------------------

std::string to_lower_copy(std::string_view s)
{
  std::string out(s);
  for (auto &c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string trim(std::string_view s)
{
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string_view::npos) return {};
  const auto e = s.find_last_not_of(" \t\r\n");
  return std::string(s.substr(b, e - b + 1));
}

// ---------------------------------------------------------------------------
// DeviceError message builders — VERBATIM from api-contract-2.0.md §6.1/§6.2.
// Kept as functions (single source of truth) so the strings can only change in
// one place; the test hard-codes the same contract strings and asserts equality.
// ---------------------------------------------------------------------------

std::string msg_unknown_device(const std::string &name)
{
  return "[dtwc] unknown device '" + name
       + "'. Valid devices: cpu, gpu, gpu:N (aliases cuda, cuda:N), hpc.";
}

std::string msg_gpu_not_built()
{
  return "[dtwc] device='gpu' requested but this build has no GPU backend compiled in.\n"
         "Rebuild with -DDTWC_ENABLE_CUDA=ON (NVIDIA) or, on macOS, -DDTWC_ENABLE_METAL=ON.\n"
         "This build will not silently fall back to CPU.";
}

std::string msg_no_env_file()
{
  return "[dtwc] device='hpc' requires a .env file at the repository root, but none was found.\n"
         "Copy scripts/slurm/env.example to .env and set SLURM_HOST, SLURM_USER, and SLURM_REMOTE_BASE.\n"
         "Example .env:\n"
         "  SLURM_HOST=arc-login.arc.ox.ac.uk\n"
         "  SLURM_USER=abcd1234\n"
         "  SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs";
}

std::string msg_missing_key(const std::string &key)
{
  return "[dtwc] device='hpc': the .env file is missing required key '" + key + "'.\n"
         "Set it in .env at the repository root. Example .env:\n"
         "  SLURM_HOST=arc-login.arc.ox.ac.uk\n"
         "  SLURM_USER=abcd1234\n"
         "  SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs";
}

std::string msg_auth_failure(const std::string &host, const std::string &user)
{
  return "[dtwc] device='hpc': could not authenticate to SLURM host '" + host
       + "' as user '" + user + "'.\n"
         "Check that your SSH key is authorized on that host (ssh " + user + "@" + host
       + " must succeed without a password prompt) and that SLURM_HOST and SLURM_USER in .env are correct.";
}

// ---------------------------------------------------------------------------
// `.env` parsing.
// ---------------------------------------------------------------------------

/// Parse a `.env` file into KEY→VALUE. Blank lines and `#` comment lines are
/// ignored; each remaining line is split on the first '='. Values are taken
/// verbatim (trimmed) — dotenv semantics, no inline-comment stripping.
std::map<std::string, std::string> parse_env_file(const fs::path &path)
{
  std::map<std::string, std::string> kv;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    const std::string t = trim(line);
    if (t.empty() || t.front() == '#') continue;
    const auto eq = t.find('=');
    if (eq == std::string::npos) continue;
    std::string key = trim(std::string_view(t).substr(0, eq));
    std::string val = trim(std::string_view(t).substr(eq + 1));
    if (!key.empty()) kv.emplace(std::move(key), std::move(val));
  }
  return kv;
}

// ---------------------------------------------------------------------------
// Default HPC authentication probe (unverified on a real cluster — HPC is beta
// per PLAN.md Phase 6). Attempts a non-interactive ssh; a password prompt or a
// non-zero exit is treated as failure. Injectable via Env::set_hpc_auth_probe so
// tests never touch the network.
// ---------------------------------------------------------------------------

bool is_shell_safe(const std::string &s)
{
  // Conservative allowlist for host/user tokens embedded in the ssh command line.
  static constexpr std::string_view allowed =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_@";
  return !s.empty() && s.find_first_not_of(allowed) == std::string::npos;
}

bool default_ssh_auth_probe(const std::string &host, const std::string &user)
{
  // Refuse to interpolate anything shell-unsafe into system() — a token we
  // cannot safely attempt counts as an auth failure (no silent success).
  if (!is_shell_safe(host) || !is_shell_safe(user)) return false;

  const std::string cmd = "ssh -o BatchMode=yes -o ConnectTimeout=10 "
                          "-o StrictHostKeyChecking=accept-new "
                        + user + "@" + host + " true";
  const int rc = std::system(cmd.c_str());
#ifdef _WIN32
  return rc == 0;
#else
  return rc != -1 && WIFEXITED(rc) && WEXITSTATUS(rc) == 0;
#endif
}

} // namespace

// ---------------------------------------------------------------------------
// Free helpers.
// ---------------------------------------------------------------------------

std::string to_string(Device d)
{
  switch (d) {
  case Device::CPU: return "cpu";
  case Device::GPU: return "gpu";
  case Device::HPC: return "hpc";
  }
  return "cpu";
}

// ---------------------------------------------------------------------------
// Env.
// ---------------------------------------------------------------------------

Env::Env()
{
  if (const char *root = std::getenv("DTWC_REPO_ROOT"); root != nullptr && *root != '\0') {
    env_file_dir_ = fs::path(root);
  } else {
    std::error_code ec;
    env_file_dir_ = fs::current_path(ec);
  }
  auth_probe_ = &default_ssh_auth_probe;
}

void Env::set_device(std::string_view name)
{
  const std::string raw = trim(name);
  const std::string lname = to_lower_copy(raw);

  if (lname == "cpu") {
    device_ = Device::CPU;
    device_index_ = 0;
    return;
  }

  if (lname == "hpc") {
    select_hpc(); // sets device_ = HPC only on full success
    return;
  }

  // GPU family: "gpu", "cuda", "gpu:N", "cuda:N".
  bool is_gpu = false;
  int gpu_id = 0;
  if (lname == "gpu" || lname == "cuda") {
    is_gpu = true;
  } else if (lname.rfind("gpu:", 0) == 0 || lname.rfind("cuda:", 0) == 0) {
    const std::string id = lname.substr(lname.find(':') + 1);
    if (!id.empty() && id.find_first_not_of("0123456789") == std::string::npos) {
      try {
        gpu_id = std::stoi(id);
        is_gpu = true;
      } catch (const std::out_of_range &) {
        is_gpu = false; // absurdly large ordinal → treat name as unknown
      }
    }
  }

  if (is_gpu) {
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
    device_ = Device::GPU;
    device_index_ = gpu_id;
    return;
#else
    throw DeviceError(msg_gpu_not_built());
#endif
  }

  throw DeviceError(msg_unknown_device(raw));
}

void Env::select_hpc()
{
  const fs::path env_path = env_file_dir_ / ".env";

  std::error_code ec;
  if (!fs::exists(env_path, ec) || ec) throw DeviceError(msg_no_env_file());

  const auto kv = parse_env_file(env_path);

  // Order fixed by the contract so the "first missing key" is deterministic.
  static constexpr std::array<std::string_view, 3> required{
    "SLURM_HOST", "SLURM_USER", "SLURM_REMOTE_BASE"
  };
  for (const std::string_view key : required) {
    const auto it = kv.find(std::string(key));
    if (it == kv.end() || it->second.empty())
      throw DeviceError(msg_missing_key(std::string(key)));
  }

  const std::string &host = kv.at("SLURM_HOST");
  const std::string &user = kv.at("SLURM_USER");

  const bool authenticated = auth_probe_ ? auth_probe_(host, user)
                                         : default_ssh_auth_probe(host, user);
  if (!authenticated) throw DeviceError(msg_auth_failure(host, user));

  device_ = Device::HPC;
  device_index_ = 0;
}

int Env::threads() const
{
#ifdef DTWC_HAS_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

Env &env()
{
  static Env instance;
  return instance;
}

} // namespace dtwc
