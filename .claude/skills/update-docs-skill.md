---
name: update-docs
description: After any user-facing change (new feature, CLI flag, API change, format support), update all documentation layers including Hugo website, examples, diagrams, and changelog.
---

# Update Documentation Skill

Trigger after any user-facing change: new feature, CLI flag, API change, format support, configuration option, or performance improvement.

## Documentation Layers (update ALL that apply)

### 1. File Headers
- Every `.hpp`/`.cpp`/`.py` must have `@file`, `@brief`, `@author`, `@date`
- Authors: Volkan Kumtepeli (first), Becky Perriment (CLARA/clustering), Claude (generated)
- Keep `@brief` to one line

### 2. Hugo Website (`docs/content/`)

Key pages — check each for relevance:

| Page | What goes here |
|------|---------------|
| `getting-started/installation.md` | CMake options, dependencies, platform notes |
| `getting-started/cli.md` | Every CLI flag with examples |
| `getting-started/configuration.md` | TOML/YAML config options |
| `getting-started/supported-data.md` | I/O formats, file layouts, compression |
| `getting-started/python.md` | Python API, dtwc-convert, PyPI |
| `getting-started/matlab.md` | MATLAB MEX bindings |
| `getting-started/data-conversion.md` | dtwc-convert tool reference |
| `method/*.md` | Algorithms, DTW variants, metrics, scores |
| `api/*.md` | Doxygen links, API reference |

### 3. Multi-Language Code Examples

Always show examples in **all available languages** using tab-style headers:

```markdown
#### CLI
\`\`\`bash
dtwc_cl --input data.parquet --column Voltage -k 5 --method clara
\`\`\`

#### C++
\`\`\`cpp
#include <dtwc.hpp>
auto data = dtwc::io::load_parquet_file("data.parquet", "Voltage");
dtwc::Problem prob("battery");
prob.set_data(std::move(data));
prob.cluster();
\`\`\`

#### Python
\`\`\`python
import dtwcpp
result = dtwcpp.cluster(data, k=5, method="clara")
\`\`\`

#### MATLAB
\`\`\`matlab
result = dtwc_cluster(data, 'k', 5, 'method', 'clara');
\`\`\`
```

Example files to keep updated:
- `examples/cpp/example_new_features.cpp`
- `examples/python/` (numbered: `01_quickstart.py`, `02_...`, etc.)
- `examples/cpp/config.toml`
- `bindings/matlab/examples/example_quickstart.m`

### 4. Mermaid Diagrams

Use Mermaid for architecture and flow diagrams in Hugo pages. Hugo renders them natively.

**Data flow diagram** (update when I/O formats change):
```markdown
\`\`\`mermaid
graph LR
    A[Parquet] -->|dtwc-convert| B[Arrow IPC]
    A -->|--input .parquet| C[DTWC++ CLI]
    B -->|--input .arrow| C
    D[CSV/TSV] -->|--input .csv| C
    E[HDF5] -->|dtwc-convert| B
    C --> F[Distance Matrix]
    F --> G[Clustering]
    G --> H[Labels + Medoids]
\`\`\`
```

**Architecture diagram** (update when layers change):
```markdown
\`\`\`mermaid
graph TB
    subgraph "Layer 4: Bindings"
        PY[Python nanobind]
        ML[MATLAB MEX]
        CLI[CLI dtwc_cl]
    end
    subgraph "Layer 3: I/O"
        CSV[CSV/TSV]
        PQ[Parquet]
        ARROW[Arrow IPC]
        DTWS[.dtws cache]
    end
    subgraph "Layer 2: Algorithms"
        PAM[FastPAM]
        CLARA[CLARA]
        MIP[MIP k-median]
        HIER[Hierarchical]
    end
    subgraph "Layer 1: Core"
        DTW[DTW variants]
        LB[Lower bounds]
        DM[Distance matrix]
        MMAP[Mmap storage]
    end
    PY --> PAM
    ML --> PAM
    CLI --> PQ
    CLI --> ARROW
    CLI --> CSV
    PAM --> DTW
    CLARA --> DTW
    DTW --> DM
    DM --> MMAP
\`\`\`
```

**Precision decision diagram** (for docs explaining float32 vs float64):
```markdown
\`\`\`mermaid
flowchart TD
    A[Load data] --> B{Series precision?}
    B -->|--precision float32| C[2x memory saving]
    B -->|--precision float64| D[Full precision]
    C --> E[DTW float32 templates]
    D --> F[DTW float64 templates]
    E --> G[Distance matrix always float64]
    F --> G
    G --> H[Clustering]
\`\`\`
```

### 5. CHANGELOG.md

Add entry under `## [Unreleased]` with category:
```markdown
### Added
- Direct Parquet reading via `--input data.parquet --column Voltage`

### Changed
- Default precision changed to float32 (2x memory saving, 0.003% DTW error)

### Fixed
- Arrow IPC writer: LargeListArray prevents int32 overflow at >2B elements
```

### 6. README.md

Update when adding:
- New I/O format support
- New algorithm
- New binding language
- Significant performance improvement

### 7. LESSONS.md (`.claude/LESSONS.md`)

Add benchmark results, gotchas, or non-obvious decisions. One bullet per lesson, bold the key takeaway.

## Execution

Use a **Sonnet subagent** for bulk doc updates to save context:

```
Agent(model="sonnet", prompt="Update docs for [feature]. Read files first. Match existing style. Add Mermaid diagrams where architecture is involved.")
```

## Checklist (copy into todo list)

```
- [ ] File headers (@file, @brief, @author) on new/modified files
- [ ] Hugo docs updated (cli.md, supported-data.md, python.md, etc.)
- [ ] Mermaid diagrams added/updated if architecture changed
- [ ] Code examples in all languages (CLI / C++ / Python / MATLAB)
- [ ] CHANGELOG.md entry
- [ ] README.md updated if significant
- [ ] LESSONS.md updated if benchmark or gotcha
```
