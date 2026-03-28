# DTWC++ CLI, Config, GUI, and Self-Test Research Report

> Note: Web search/fetch tools were unavailable due to a model configuration issue.
> This report is based on Claude's training knowledge (cutoff May 2025). All claims
> should be verified against current documentation before implementation decisions.

---

## 1. CLI Configuration for Scientific Computing Tools

### 1.1 Format Comparison: TOML vs YAML vs JSON vs INI

| Criterion | TOML | YAML | JSON | INI |
|-----------|------|------|------|-----|
| **Human readability** | Excellent | Excellent | Fair | Good |
| **Comments** | Yes (`#`) | Yes (`#`) | No | Yes (`;` or `#`) |
| **Type safety** | Strong (datetime, arrays, tables) | Weak (implicit typing gotchas) | Moderate | Weak (all strings) |
| **Nesting** | Moderate (tables, inline tables) | Excellent (indentation) | Excellent (braces) | Poor (sections only) |
| **C++ library maturity** | toml++ (header-only, MIT, excellent) | yaml-cpp (stable, MIT) | nlohmann/json (header-only, MIT, ubiquitous) | Many, including CLI11 built-in |
| **Spec stability** | TOML v1.0 is finalized | YAML 1.2 is large and complex | RFC 8259, rock-solid | No formal spec |
| **Footgun risk** | Low | High (Norway problem: `NO` -> `false`, `3.10` -> `3.1`) | Low | Moderate |
| **Scientific tool adoption** | Rust ecosystem (Cargo), Python (pyproject.toml), Julia (Project.toml) | Kubernetes, Ansible, many DevOps tools | Web APIs, VS Code | Git, Windows apps, legacy tools |

**Recommendation for DTWC++: TOML**

Rationale:
- TOML's type system maps well to scientific parameters (integers, floats, booleans, arrays, datetime for run metadata).
- The `toml++` library (by Mark Gillard) is header-only, C++17, MIT-licensed, and has excellent error messages. It is one of the highest-quality C++ TOML parsers available.
- YAML's implicit type coercion is dangerous for scientific configs (silent data corruption).
- JSON lacks comments, which scientists need for documenting parameter choices.
- INI is too limited for nested configuration structures.

### 1.2 How Popular Scientific Tools Handle Config

| Tool | Format | Notes |
|------|--------|-------|
| **GROMACS** | Custom `.mdp` (key=value) | Simple flat format; no nesting. CLI overrides via `-override`. |
| **LAMMPS** | Custom script-like input files | Procedural input scripts, not declarative config. |
| **OpenFOAM** | Custom dictionary format | Hierarchical but proprietary syntax. Steep learning curve. |
| **NAMD** | Custom key-value | Flat config, simple but limited. |
| **Snakemake** | YAML (`config.yaml`) | Hierarchical, with CLI override via `--config key=value`. |
| **Nextflow** | Groovy-based `nextflow.config` | Custom DSL; powerful but non-standard. |
| **tskit / msprime** | Python API + TOML for project config | Modern Python scientific tools increasingly use TOML. |
| **CMake** | Custom + cache variables | `-D` flags override cached values; similar pattern to config+CLI. |

**Key pattern**: Modern tools are converging on TOML or YAML for declarative config, with CLI flags as overrides. Older tools often have custom formats (regrettable — maintenance burden, no ecosystem tooling).

### 1.3 CLI11 Configuration File Support

CLI11 has **built-in configuration file support**. Key facts:

- **Native INI format**: CLI11 reads/writes a simple INI-like format by default. The format supports sections, comments, and vectors.
- **TOML support**: CLI11 v2.x includes a `CLI::ConfigTOML` formatter. You can enable it with:
  ```cpp
  app.config_formatter(std::make_shared<CLI::ConfigTOML>());
  ```
  However, CLI11's TOML support is limited — it handles basic TOML but not the full spec. For full TOML, use `toml++` to parse, then feed values into CLI11 programmatically.
- **Config file flag**: Adding `app.set_config("--config", "default.toml", "Configuration file")` automatically adds a `--config` flag.
- **Priority order**: CLI flags > config file values > default values. This is the standard and correct behavior.
- **Config generation**: `app.config_to_str()` generates a config file from current values, useful for `--generate-config` commands.

**Recommended approach for DTWC++**:
1. Use CLI11 for CLI parsing (already a good fit for C++17 projects).
2. Use `toml++` for full TOML parsing.
3. Bridge them: parse the TOML file first, then set CLI11 defaults from the parsed TOML values before CLI11 parses `argv`. CLI flags then override naturally.

---

## 2. GUI Options for Scientific C++ Libraries

### 2.1 Dear ImGui

- **What**: Immediate-mode GUI library for C++. Renders via OpenGL/Vulkan/DirectX/Metal.
- **License**: MIT — fully compatible with BSD-3.
- **Effort**: Medium. Requires a rendering backend (SDL2+OpenGL is simplest). ~1-2 weeks for a basic parameter editor + result viewer.
- **Pros**:
  - Extremely lightweight, fast, embeddable.
  - Perfect for real-time visualization (plot DTW paths, distance matrices as heatmaps).
  - `implot` extension provides scientific plotting (line plots, heatmaps, scatter plots). MIT licensed.
  - Single-process, no network dependencies.
  - Popular in gamedev and increasingly in scientific/engineering tools (e.g., Tracy profiler, various physics simulators).
  - Easy to add sliders, input fields, combo boxes for parameters.
- **Cons**:
  - Not a "traditional" GUI — no native look-and-feel (custom rendered).
  - Requires graphics context (won't work on headless servers without offscreen rendering).
  - No built-in file dialogs (need `nfd` or `tinyfiledialogs` addon).
  - Distribution: must ship as a compiled binary, not a web page.
- **Best for DTWC++**: Interactive parameter tuning, DTW alignment visualization, distance matrix heatmaps, clustering result inspection. Ideal as an optional `dtwc-gui` binary.

### 2.2 Qt

- **License**: LGPL v3 (or commercial). LGPL is compatible with BSD-3 if dynamically linked, but adds distribution complexity.
- **Effort**: High. Qt is a massive framework. 2-4 weeks for a basic app.
- **Pros**:
  - Native look-and-feel on all platforms.
  - Qt Charts, Qt Data Visualization for scientific plots.
  - Mature ecosystem, good documentation.
  - QCustomPlot (GPL/commercial) or `qwt` (LGPL) for scientific plotting.
- **Cons**:
  - Heavy dependency (~100+ MB).
  - LGPL requires dynamic linking or commercial license.
  - MOC (meta-object compiler) adds build complexity.
  - Overkill for a library that primarily needs parameter editing and result viewing.
- **Verdict for DTWC++**: **Not recommended** as primary GUI. Too heavy, license friction. Consider only if a full desktop application is the goal.

### 2.3 Web-Based GUI (REST API + Web Frontend)

- **Architecture**: C++ backend exposes a REST API (via `cpp-httplib`, `crow`, or `drogon`); frontend is HTML/JS (React/Vue/Plotly.js).
- **License**: All mentioned libraries are MIT or BSD.
- **Effort**: Medium-High. ~2-3 weeks. Need to design API, build frontend, handle CORS, etc.
- **Pros**:
  - Works on headless servers (SSH + port forwarding).
  - Rich plotting via Plotly.js or D3.js.
  - Accessible from any browser.
  - Scientists familiar with Jupyter will find this natural.
- **Cons**:
  - Two-language development (C++ + JS).
  - Security considerations for network-exposed API.
  - More complex deployment.
  - Latency for large data transfers.
- **Verdict for DTWC++**: Good for remote/HPC use cases but high effort. Consider as a future option, not initial implementation.

### 2.4 Jupyter Widgets (ipywidgets) via Python Bindings

- **Architecture**: Python bindings (pybind11) + ipywidgets for interactive controls + matplotlib/plotly for visualization.
- **License**: BSD-3 (ipywidgets), compatible.
- **Effort**: Low-Medium. ~1 week if Python bindings already exist. The Python bindings ARE already planned for DTWC++.
- **Pros**:
  - Scientists already use Jupyter notebooks.
  - No new binary to distribute — works in existing Jupyter installations.
  - Rich ecosystem: matplotlib, plotly, seaborn for visualization.
  - Interactive: sliders for band width, dropdowns for metrics, live re-clustering.
  - `ipywidgets.interact()` can create a basic GUI in ~10 lines of Python.
- **Cons**:
  - Requires Python bindings to be working.
  - Performance limited by Python-C++ round-trips (but fine for interactive use).
  - Not a standalone application.
- **Verdict for DTWC++**: **Strongly recommended as the primary "GUI" strategy.** Leverages the Python bindings that are already planned. Lowest effort, highest scientist adoption.

### 2.5 Gradio

- **License**: Apache 2.0 — compatible with BSD-3.
- **Effort**: Very low. ~2-3 days for a basic interface once Python bindings exist.
- **Pros**:
  - Extremely fast to prototype: `gr.Interface(fn=cluster, inputs=[...], outputs=[...])`.
  - Auto-generates web UI from function signatures.
  - Built-in sharing (Gradio generates a public URL).
  - Good for demos and presentations.
- **Cons**:
  - Limited customization for complex scientific workflows.
  - Designed for ML demos, not deep scientific parameter exploration.
  - Depends on Python bindings.
- **Verdict for DTWC++**: Good for quick demos and shareable prototypes. Not a primary GUI but a nice "show-off" tool. Consider as a complement to ipywidgets.

### 2.6 GUI Recommendation Summary

| Priority | Approach | Effort | When |
|----------|----------|--------|------|
| **1st** | **Jupyter ipywidgets** via Python bindings | Low | After Python bindings are working |
| **2nd** | **Dear ImGui** standalone `dtwc-gui` | Medium | When interactive C++ visualization is needed |
| **3rd** | **Gradio** demo app | Very Low | For demos, papers, sharing |
| **4th** | Web-based (REST + Plotly.js) | High | If HPC/remote access is critical |
| Skip | Qt | High | Not recommended |

---

## 3. Self-Test / Diagnostic Commands

### 3.1 Common Patterns in Scientific Tools

Scientific tools typically implement self-test via one or more of:

1. **`--selftest` / `--check` flag**: Runs a minimal built-in test suite.
   - GROMACS: `gmx check` validates trajectory and topology files.
   - OpenMPI: `ompi_info` reports build configuration and detected hardware.
   - FFTW: `fftw-wisdom` can benchmark and verify FFT correctness.

2. **`--info` / `--version --verbose` flag**: Reports build configuration, detected features, and hardware.
   - Pattern: Print a diagnostic block showing what was compiled in and what's available at runtime.

3. **Built-in micro-benchmark**: Run a small known problem and verify the answer.
   - This is the most robust approach for numerical software.

### 3.2 Recommended `dtwc --selftest` Implementation

```
$ dtwc --selftest

DTWC++ v1.2.0  Self-Test Report
================================

Build Configuration:
  C++ Standard:     C++17
  Compiler:         GCC 12.3.0
  OpenMP:           Compiled in: YES
  Armadillo:        v12.6.4
  HiGHS:            v1.7.0
  Gurobi:           Not available
  CUDA:             Not available

Runtime Environment:
  OpenMP Threads:   8 (of 16 available cores)
  OMP_NUM_THREADS:  8
  Memory:           31.4 GB available

Parallelism Verification:
  Serial DTW (N=100, L=200):       12.3 ms
  Parallel DTW (N=100, L=200, 8T): 2.1 ms  (5.9x speedup)
  Parallelism:                      WORKING

Correctness Checks:
  DTW(identical series):   0.0       PASS
  DTW(known pair):         23.456    PASS (expected 23.456)
  DTW symmetry:            PASS
  DTW triangle inequality: PASS
  Banded DTW <= Full DTW:  PASS
  PAM k=3 (Trace dataset): PASS (matches reference)

All 8 checks passed.
```

### 3.3 OpenMP Verification at Runtime

Key techniques for verifying OpenMP is actually working:

```cpp
#include <omp.h>
#include <set>
#include <atomic>

struct ParallelismCheck {
    bool openmp_compiled;   // Was OpenMP available at compile time?
    bool openmp_working;    // Did multiple threads actually execute?
    int threads_requested;
    int threads_observed;
    double serial_time_ms;
    double parallel_time_ms;
    double speedup;
};

ParallelismCheck verify_parallelism() {
    ParallelismCheck result{};

    #ifdef _OPENMP
    result.openmp_compiled = true;
    result.threads_requested = omp_get_max_threads();

    // Actually verify multiple threads run
    std::atomic<int> thread_count{0};
    std::set<int> thread_ids; // Not atomic, but we only check size after join
    std::vector<int> ids(result.threads_requested, -1);

    #pragma omp parallel
    {
        ids[omp_get_thread_num()] = omp_get_thread_num();
        thread_count.fetch_add(1);
    }

    result.threads_observed = thread_count.load();
    result.openmp_working = (result.threads_observed > 1);

    // Timing comparison (use a real workload, not a sleep)
    // ... benchmark serial vs parallel DTW on a small synthetic dataset
    #else
    result.openmp_compiled = false;
    result.openmp_working = false;
    result.threads_requested = 1;
    result.threads_observed = 1;
    #endif

    return result;
}
```

**Important pitfalls to check**:
- `OMP_NUM_THREADS=1` silently disables parallelism.
- Some HPC environments restrict thread affinity (`OMP_PROC_BIND`, `GOMP_CPU_AFFINITY`).
- Virtual machines may report many cores but throttle parallel execution.
- The timing test is essential: thread creation alone doesn't prove useful parallelism.

### 3.4 Correctness Self-Tests

For a DTW library, the minimal self-test suite should verify:

1. **Identity**: `DTW(x, x) == 0` for any series x.
2. **Known answer**: DTW of two specific short series equals a pre-computed reference value (computed analytically or by a trusted reference implementation).
3. **Symmetry**: `DTW(x, y) == DTW(y, x)`.
4. **Triangle inequality**: DTW does NOT satisfy the triangle inequality in general, but this is worth documenting/testing.
5. **Constraint consistency**: `DTW_banded(x, y, w) >= DTW_full(x, y)` (banding restricts the path, so distance can only increase or stay equal).
6. **Clustering sanity**: PAM on a known dataset with known optimal medoids produces the expected result.

---

## 4. Configuration File + CLI Override Pattern

### 4.1 The Standard Pattern

The universally accepted priority order is:

```
CLI flags  >  Environment variables  >  Config file  >  Built-in defaults
```

This is how virtually all well-designed tools work (Docker, Kubernetes, Git, etc.).

### 4.2 CLI11 Implementation

CLI11 supports this pattern natively. Here is the recommended implementation:

```cpp
#include <CLI/CLI.hpp>

int main(int argc, char** argv) {
    CLI::App app{"DTWC++ - Dynamic Time Warping Clustering"};

    // Config file support
    // This adds --config flag automatically
    app.set_config("--config", "", "Path to TOML configuration file");

    // Option: generate a config template
    bool generate_config = false;
    app.add_flag("--generate-config", generate_config,
                 "Print a default configuration file and exit");

    // Define all options (these work from both CLI and config)
    int num_clusters = 3;
    app.add_option("-k,--clusters", num_clusters, "Number of clusters")
       ->default_val(3);

    std::string method = "pam";
    app.add_option("-m,--method", method, "Clustering method")
       ->check(CLI::IsMember({"pam", "clara", "clarans"}))
       ->default_val("pam");

    double band_ratio = 0.1;
    app.add_option("--band-ratio", band_ratio, "Sakoe-Chiba band ratio")
       ->check(CLI::Range(0.0, 1.0))
       ->default_val(0.1);

    int threads = 0; // 0 = auto
    app.add_option("-j,--threads", threads, "Number of threads (0=auto)")
       ->default_val(0);

    // Parse
    CLI11_PARSE(app, argc, argv);

    if (generate_config) {
        std::cout << app.config_to_str(true, true);
        return 0;
    }

    // ... run clustering
}
```

**Config file (`dtwc.toml`) would look like**:

```toml
# DTWC++ configuration file
# CLI flags override these values

clusters = 5
method = "pam"
band-ratio = 0.15
threads = 4

[dtw]
constraint = "band"       # none, band, itakura
normalize = true           # path-length normalization
metric = "L2"             # L1, L2, cosine

[output]
format = "csv"
verbose = true
save-distance-matrix = true
```

### 4.3 Bridging TOML++ and CLI11 (Full TOML Support)

For richer TOML than CLI11 natively handles (nested tables, arrays of tables), bridge like this:

```cpp
#include <toml++/toml.hpp>
#include <CLI/CLI.hpp>

void apply_toml_defaults(CLI::App& app, const std::string& config_path) {
    if (config_path.empty()) return;

    auto tbl = toml::parse_file(config_path);

    // Walk the TOML table and set CLI11 defaults
    for (auto* opt : app.get_options()) {
        std::string name = opt->get_name(false, true); // long name
        // Remove leading --
        if (name.substr(0, 2) == "--") name = name.substr(2);

        if (auto val = tbl[name]) {
            // Set as default so CLI flags still override
            opt->default_str(val.value_or<std::string>(""));
        }
    }
}

int main(int argc, char** argv) {
    CLI::App app{"DTWC++"};

    std::string config_file;
    app.add_option("--config", config_file, "Config file path");

    // Pre-parse just to get --config value
    app.allow_extras(true);
    app.parse(argc, argv);
    app.allow_extras(false);

    // Apply TOML defaults
    apply_toml_defaults(app, config_file);

    // Full parse with TOML defaults applied
    app.parse(argc, argv);
}
```

### 4.4 Environment Variable Override

CLI11 also supports environment variable fallbacks:

```cpp
app.add_option("-j,--threads", threads, "Number of threads")
   ->envname("DTWC_THREADS")
   ->default_val(0);
```

Priority: `--threads 4` > `DTWC_THREADS=8` > default (0).

### 4.5 Config Search Path Convention

Scientific tools commonly search for config in this order:
1. Explicit `--config path/to/file.toml`
2. `./dtwc.toml` (current working directory)
3. `$XDG_CONFIG_HOME/dtwc/config.toml` (Linux) or `%APPDATA%\dtwc\config.toml` (Windows)
4. Built-in defaults

---

## Consolidated Recommendations for DTWC++

### Immediate (Phase 1)

1. **CLI**: Use **CLI11** (header-only, BSD-3). Already an excellent fit.
2. **Config format**: **TOML** via **toml++** (header-only, MIT).
3. **Config+CLI pattern**: TOML file provides defaults; CLI flags override; `--generate-config` emits a template.
4. **Self-test**: Implement `dtwc --selftest` that checks:
   - Build features (OpenMP, Armadillo, HiGHS, Gurobi)
   - Parallelism actually works (thread count + timing benchmark)
   - Numerical correctness (DTW identity, known-answer, symmetry, constraint consistency)

### Medium-term (Phase 2)

5. **GUI via Jupyter**: Once Python bindings are working, create a `dtwc.widgets` module with ipywidgets for interactive parameter tuning and result visualization.
6. **Gradio demo**: A simple `dtwc-demo.py` script that launches a Gradio interface for quick demos.

### Long-term (Phase 3)

7. **Dear ImGui standalone GUI**: Optional `dtwc-gui` binary for interactive C++ visualization (DTW paths, distance matrix heatmaps, cluster assignments). Only if there's demand beyond what Jupyter provides.

### Libraries to Add as Dependencies

| Library | Role | Header-only | License | Optional? |
|---------|------|-------------|---------|-----------|
| CLI11 | CLI parsing | Yes | BSD-3 | No (CLI binary only) |
| toml++ | Config parsing | Yes | MIT | No (CLI binary only) |
| implot | ImGui scientific plots | Yes (with ImGui) | MIT | Yes |
| pybind11 | Python bindings | Yes | BSD-3 | Yes |
