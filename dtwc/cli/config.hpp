/**
 * @file config.hpp
 * @brief cli::bind(), the one key table of dtwc::Config, and the text forms built on it.
 *
 * @details cli::bind() is the one key table: each key is the CLI long name
 * without "--", nested fields keep flat keys (`wdtw-g` sets `variant.wdtw_g`),
 * and enums are read through the Name tables beside them. to_config_text() and
 * parse_config() are built on bind(), so no second field list exists. Settings
 * are only read here; run-time checks belong to the run. The Config itself is
 * the core's (../config.hpp).
 *
 * @date 24 Sep 2026
 */

#pragma once

#include "../config.hpp"
#include "../algorithms/one_batch_pam.hpp"
#include "../base/names.hpp"

#include <string>
#include <utility>
#include <vector>

namespace CLI {
class App;
}

namespace dtwc {
namespace cli {

/// Adds every Config key to `app`, bound to `config`, plus `--config <file>`
/// (TOML or YAML, the same keys; flags beat the file; an unknown key is an error).
/// `--help` shows `config`'s values as the defaults. v1.0.0's option spellings
/// (`--Nc`, `--probName`, `--skipRows`, ...) are hidden ones that each warn once
/// on stderr and yield to the canonical spelling.
/// A value no spelling reads raises during the parse: CLI11's error for a bad
/// choice or number, InvalidInput for `--ram-limit` / `--delimiter` and for
/// v1.0.0's `-k`/`--Nc` range `i..j`, DeviceError for `--device`.
void bind(CLI::App &app, Config &config);

} // namespace cli

/// Every key of `config`, one `key = value` line each in bind() order: enums by
/// canonical name, doubles in shortest round-trip form. parse_config() and
/// `--config` read it back to an equal Config.
/// @throws InvalidInput when a field holds a value no name spells (MetricType::L2).
std::string to_config_text(const Config &config);

/// A Config from (key, value) pairs, as name-value pairs give them: `_` in a
/// key reads as `-` (`max_iter` is `max-iter`), a one-letter key is the short
/// flag (`k` is `n-clusters`); the pairs are read as a config file, so the keys,
/// spellings and precedence are bind()'s.
/// @throws InvalidInput for an unknown key or a value CLI11 rejects.
Config parse_config(const std::vector<std::pair<std::string, std::string>> &pairs);

} // namespace dtwc
