# YAML CLI config removal + medoid_utils tombstone (2026-09-02)

## Outcome

The hand-rolled `--yaml-config` loader is gone; `--config` (CLI11 TOML) is the
single CLI config mechanism. `DTWC_ENABLE_YAML` / `DTWC_HAS_YAML` / the yaml-cpp
CPM package no longer exist. The `medoid_utils.hpp` tombstone now states what is
true.

## Files changed

- `dtwc/dtwc_cl.cpp` — removed `--yaml-config` option, the `yaml-cpp` include,
  and the 80-line post-parse loader (whose own TODO admitted it overrode
  explicit CLI flags). Required-input error is now
  `Error: --input is required via CLI or config file (TOML)`. Comments that
  described values as "YAML" reworded; the defensive `to_lower`/alias
  normalisation and `validate_cli_*` guards were kept (they still protect the
  dispatch chains).
- `CMakeLists.txt` — dropped the `DTWC_ENABLE_YAML` option and the
  `yaml-cpp::yaml-cpp` link + `DTWC_HAS_YAML` define on `dtwc_cl`.
- `cmake/Dependencies.cmake` — dropped the yaml-cpp CPM block.
- `tests/CMakeLists.txt` + `tests/integration/test_cli_rejects_unknown_option.cmake`
  (new file, staged) — `test_cli_rejects_yaml_config` drives the real binary and
  asserts a non-zero exit plus the flag named back, without pinning CLI11's
  wording. (I first wrote this as a bare `PASS_REGULAR_EXPRESSION` add_test; a
  concurrent agent hardened it into the script form, which is better and is what
  now stands.)
- `tests/unit/unit_test_cli_args.cpp`, `tests/unit/test_supply_chain_pinning.cpp`,
  `tests/unit/algorithms/unit_test_medoid_utils.cpp` — stale YAML wording only.
- `scripts/check_supply_chain_pins.py` — yaml-cpp identity removed
  (`REGISTERED_CMAKE_ARCHIVE_TOTAL` 7 -> 6). `REGISTERED_CMAKE_MANIFEST_TOTAL`
  28 -> 29 for the new integration script; the script counts `git ls-files`, so
  I staged `tests/integration/test_cli_rejects_unknown_option.cmake` (staged
  only, NOT committed) to keep the gate self-consistent. Whoever commits must
  keep that file in the same commit.
- `tests/python/test_supply_chain_pins.py` — `len(archive_pins) == 6`,
  `manifest_total == 29`.
- `scripts/check_docs_contract.py` — removed the `set_if_unset` YAML-key check.
- Docs: `docs/content/getting-started/cli.md`, `.../configuration.md` (YAML
  section deleted), `docs/content/method/missing-data.md`,
  `docs/content/api/interface-parity.md`, `docs/api-contract-2.0.md`,
  `README.md` (option table row).
- Deleted `examples/cpp/config.yaml`.
- `dtwc/algorithms/detail/medoid_utils.hpp` — tombstone now says the *shared*
  helpers were removed from this header and names fast_pam.cpp's file-local
  `compute_nearest_and_second`. The header still declares `validate_medoids`
  (used by fast_pam.cpp, dtwc.hpp, unit tests), so it stays.

## Grep residue

`grep -rn -i yaml dtwc python bindings tests scripts cmake CMakeLists.txt
docs/content README.md .github` (excluding `__pycache__`):

- `tests/CMakeLists.txt` x5 — the new rejection test (intentional).
- `docs/content/getting-started/ai-commands.md:12` — "YAML frontmatter" of
  Claude command files; unrelated to CLI config.
- **`tests/python/test_hpc.py:1223`** — pins the OLD stderr string
  `"(TOML; YAML if built with DTWC_ENABLE_YAML)\n"`. That file is owned by the
  Python agent (out of my edit scope). It must become
  `"Error: --input is required via CLI or config file (TOML)\n"` (single
  string, no continuation) or `TestLocalRoundTrip::
  test_required_input_message_names_toml_first` will fail wherever a local
  `dtwc_cl` binary exists.

No workflow or vendored build set `DTWC_ENABLE_YAML=ON` (checked
`.github/workflows/*`, `examples/`, `cmake/`).

## Verification (build/nollfio, clang+Ninja, Release, llfio OFF, Arrow OFF)

- Reconfigure + full build: exit 0. `grep -ci yaml build/nollfio/build.ninja` = 0;
  no yaml-cpp in `_deps`. Only pre-existing warnings
  (`-fno-signaling-nans`, `getenv`).
- Real binary: `dtwc_cl --yaml-config foo.yaml -i x.csv` ->
  `The following arguments were not expected: --yaml-config foo.yaml`, exit 109.
  `--help` lists only `--config Read TOML configuration file`.
- `ctest -R "cli|supply|hygiene"` -> **4 of 5 passed** (`test_supply_chain_pinning`,
  `unit_test_cli_args`, `test_cli_resume_state`, `test_cli_rejects_yaml_config`).
  No test matches "hygiene". The one failure, `unit_test_cli_checkpoint`, is a
  foreign untracked test (`tests/unit/unit_test_cli_checkpoint.cpp`, the
  checkpoint agent's in-flight work); it fails at its own line 45,
  `REQUIRE(self.has_parent_path())` — an executable-path helper, nothing to do
  with config parsing. **nollfio inventory is now 130 tests** (128 + my
  `test_cli_rejects_yaml_config` + their `unit_test_cli_checkpoint`); AGENTS.md
  floors need +1 for this task in all three matrices.
- `uv run python scripts/check_supply_chain_pins.py` -> PASS
  (`CMAKE_ARCHIVE_PIN_GATE verified=6 total=6`, `TRACKED_CMAKE_MANIFESTS total=29`).
- `uv run pytest tests/python/test_supply_chain_pins.py -q` -> 63 passed.
- `uv run python scripts/check_docs_contract.py --cli build/nollfio/bin/dtwc_cl.exe`
  -> "generated documentation is current", then FAILS with
  `CLI reference drift: live but undocumented: ['--checkpoint-interval'];
  documented but not live: []`. **Not caused by this task**: `--checkpoint-interval`
  is absent from `git show HEAD:dtwc/dtwc_cl.cpp` and arrives via another
  agent's uncommitted CLI addition that is not yet in
  `docs/content/getting-started/cli.md`. My removal shows as
  `documented but not live: []`, i.e. consistent on both sides.

## Contradicted prior record

`.claude/reports/2026-09-02-obsolete-sweep.md:38` classified YAML as
**kept-deliberate** ("`DTWC_ENABLE_YAML` is a live optional dep ... with a real
`--yaml-config` path"). The maintainer's TOML-only preference plus the loader's
own precedence bug overrule that entry; it is now stale.

## Note for other agents

`generate_docs.py --check` was already failing on `docs/content/api/tier-2.md`
before this task (another agent's uncommitted `docs/api-contract-2.0.md` edits).
I ran `uv run python scripts/generate_docs.py` to unblock the contract gate, so
`docs/content/api/tier-1.md` and `tier-2.md` now contain the regenerated
(checkpoint/MIP-settings) text derived from that in-flight contract.

## Proposed CHANGELOG bullets (Unreleased)

- **Removed** the `--yaml-config` CLI option, the `DTWC_ENABLE_YAML` build
  option and the yaml-cpp dependency. `--config` (TOML) is the only CLI
  configuration mechanism; the YAML loader silently overrode explicitly given
  command-line flags. Migrate `config.yaml` files to the equivalent TOML keys
  (same kebab-case names).
- Required-input error text is now
  `Error: --input is required via CLI or config file (TOML)`.
