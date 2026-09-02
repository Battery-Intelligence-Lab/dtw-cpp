/**
 * @file config_file.hpp
 * @brief CLI11 configuration reader that accepts TOML or YAML through --config.
 *
 * @details YAML is translated into CLI11 `ConfigItem`s, so CLI11 alone owns
 * precedence (command-line values beat file values), option validation,
 * deprecated-key warnings and --help. Without DTWC_HAS_YAML a YAML file is
 * refused with a typed error rather than misread as TOML.
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#pragma once

#include <CLI/CLI.hpp>
#ifdef DTWC_HAS_YAML
#include <fkYAML/node.hpp>
#endif

#include <cstdint>
#include <istream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dtwc::cli
{

/// Reads a `--config` file in either TOML or YAML and hands CLI11 the items.
class ConfigFile : public CLI::ConfigBase
{
public:
  /// @param app owning application; used only to name the file in diagnostics.
  explicit ConfigFile(const CLI::App *app = nullptr) : app_{ app } {}

  std::vector<CLI::ConfigItem> from_config(std::istream &input) const override
  {
    const std::string text{ std::istreambuf_iterator<char>{ input },
                            std::istreambuf_iterator<char>{} };
    if (is_toml(text)) {
      std::istringstream toml{ text };
      return CLI::ConfigBase::from_config(toml);
    }
    return from_yaml(text);
  }

private:
  /// Sniff rule: the first line that is neither blank nor a `#` comment decides.
  /// `[table]`, or a `=` ahead of any `:`, means TOML; anything else
  /// (`key: value`, `---`, `%YAML`) is YAML.
  static bool is_toml(const std::string &text)
  {
    std::istringstream lines{ text };
    for (std::string line; std::getline(lines, line);) {
      const auto first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos || line[first] == '#') continue;
      if (line[first] == '[') return true;
      const auto equals = line.find('=', first);
      const auto colon = line.find(':', first);
      return equals != std::string::npos && (colon == std::string::npos || equals < colon);
    }
    return true; // an empty file is an empty TOML document
  }

  [[nodiscard]] std::string source() const
  {
    const CLI::Option *opt = app_ != nullptr ? app_->get_config_ptr() : nullptr;
    if (opt != nullptr && !opt->results().empty()) return opt->results().front();
    return "config file";
  }

#ifdef DTWC_HAS_YAML
  static std::string scalar_text(const fkyaml::node &node, const std::string &path)
  {
    if (node.is_string()) return node.get_value<std::string>();
    if (node.is_boolean()) return node.get_value<bool>() ? "true" : "false";
    if (node.is_integer()) return std::to_string(node.get_value<std::int64_t>());
    // A YAML null is NOT an empty value: CLI11 would take "" as a supplied
    // value, so `band: ~` set band 0 and `verbose: ~` turned the flag on.
    if (node.is_null())
      throw CLI::ConfigError("key '" + path
                             + "' is null; omit the key to use the default");
    if (node.is_float_number()) {
      std::ostringstream out;
      out.precision(std::numeric_limits<double>::max_digits10);
      out << node.get_value<double>();
      return out.str();
    }
    throw CLI::ConfigError(path + ": expected a scalar value");
  }

  /// Top-level scalars become bare items; nested mappings become `parents`, the
  /// way CLI11 reads TOML tables; a sequence of scalars becomes repeated inputs.
  static void collect(const fkyaml::node &map,
                      const std::vector<std::string> &parents,
                      std::vector<CLI::ConfigItem> &items)
  {
    for (auto entry : map.map_items()) {
      CLI::ConfigItem item;
      item.parents = parents;
      item.name = scalar_text(entry.key(), CLI::detail::join(parents, ".") + ".<key>");
      const fkyaml::node &value = entry.value();
      std::vector<std::string> path = parents;
      path.push_back(item.name);

      if (value.is_mapping()) {
        collect(value, path, items);
        continue;
      }
      const std::string joined = CLI::detail::join(path, ".");
      if (value.is_sequence())
        for (const fkyaml::node &element : value)
          item.inputs.push_back(scalar_text(element, joined));
      else
        item.inputs.push_back(scalar_text(value, joined));
      items.push_back(std::move(item));
    }
  }
#endif

  [[nodiscard]] std::vector<CLI::ConfigItem> from_yaml([[maybe_unused]] const std::string &text) const
  {
#ifdef DTWC_HAS_YAML
    std::vector<fkyaml::node> docs;
    try {
      docs = fkyaml::node::deserialize_docs(text);
    } catch (const fkyaml::exception &e) {
      throw CLI::ConfigError(source() + ": invalid YAML: " + e.what());
    }
    if (docs.size() != 1)
      throw CLI::ConfigError(source() + ": expected one YAML document, found "
                             + std::to_string(docs.size()));
    if (!docs.front().is_mapping())
      throw CLI::ConfigError(source() + ": top level must be a mapping of option names to values");

    std::vector<CLI::ConfigItem> items;
    collect(docs.front(), {}, items);
    return items;
#else
    throw CLI::ConfigError(source() + ": built without YAML support; use TOML");
#endif
  }

  const CLI::App *app_{ nullptr };
};

} // namespace dtwc::cli
