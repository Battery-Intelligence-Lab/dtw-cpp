# Plan: Migrate docs/ from Jekyll to Hugo + Add New Feature Documentation

## Context

The project currently uses Jekyll with `remote_theme: Battery-Intelligence-Lab/bil-jekyll-rtd-theme` for docs hosted on GitHub Pages. The lab has created a new Hugo theme (`bil-hugo-rtd-theme`) as a Hugo Module with conversion scripts, Oxford blue styling, and full feature parity. This plan converts the `docs/` folder in-place from Jekyll to Hugo, updates the GitHub Actions workflow, and adds ~10 new documentation pages for features that have been implemented but not yet documented (DTW variants, multivariate support, Python/MATLAB bindings, new clustering algorithms, quality scores, checkpointing).

## Approach

Replace the Jekyll site structure with Hugo conventions directly in `docs/`. The theme's `scripts/migrate-from-jekyll.sh` handles the mechanical conversion (renaming files, stripping `layout:`, converting `nav_order` to `weight`), but we need manual work for: new `hugo.toml` + `go.mod`, updating the GitHub Actions workflow (Hugo build + Doxygen + LCOV in one job), fixing the duplicate z-normalization section in `2_dtw.md`, fixing broken links in `7_develop/`, writing ~10 new content pages, and updating existing pages (algorithms, metrics) to reflect current state.

---

## Step 1: Hugo scaffolding — config + module init

Create the Hugo project files in `docs/`:

**New files:**
- `docs/hugo.toml` — site config importing the theme module
- `docs/go.mod` — Hugo module declaration (run `hugo mod init`)

**`docs/hugo.toml` content:**
```toml
baseURL = "https://battery-intelligence-lab.github.io/dtw-cpp/"
title = "DTWC++"
languageCode = "en"

[module]
[[module.imports]]
  path = "github.com/Battery-Intelligence-Lab/bil-hugo-rtd-theme"

[params]
  description = "Official documentation of DTWC++ software."
  logo = "docs_logo.png"
  version = "v1.0.0"
  edit_on_github = true
  edit_branch = "main"
  repo_url = "https://github.com/Battery-Intelligence-Lab/dtw-cpp/"
  github_docs_folder = "docs/content"
  search_enabled = true
  code_copy = true
  math_globally = true
  site_author = "DTWC++ authors"

[markup]
  [markup.highlight]
    style = "github"
    noClasses = false
  [markup.goldmark.renderer]
    unsafe = true
  [markup.tableOfContents]
    startLevel = 2
    endLevel = 4
```

**`docs/go.mod`:** Run `cd docs && hugo mod init github.com/Battery-Intelligence-Lab/dtw-cpp/docs`

---

## Step 2: Restructure content — Jekyll dirs to Hugo `content/`

Rename and restructure existing markdown files. The theme's migration script logic applies:
- Numbered prefixes stripped from folders and files
- `index.md` → `_index.md`
- `layout: default` removed
- `nav_order: N` → `weight: N`
- Image references updated

### Directory mapping

| Jekyll path | Hugo path | Notes |
|---|---|---|
| `docs/index.md` | `docs/content/_index.md` | Home page, `nav_exclude: true` |
| `docs/404.md` | `docs/content/404.md` | Keep `nav_exclude: true` |
| `docs/1_getting_started/index.md` | `docs/content/getting-started/_index.md` | `weight: 1` |
| `docs/1_getting_started/1_installation.md` | `docs/content/getting-started/installation.md` | `weight: 1` |
| `docs/1_getting_started/2_cli.md` | `docs/content/getting-started/cli.md` | `weight: 2` |
| `docs/1_getting_started/3_supported_data.md` | `docs/content/getting-started/supported-data.md` | `weight: 3` |
| `docs/1_getting_started/3_mpi_cuda_setup.md` | `docs/content/getting-started/mpi-cuda-setup.md` | `weight: 4` |
| `docs/1_getting_started/4_examples.md` | `docs/content/getting-started/examples.md` | `weight: 5` |
| `docs/2_method/index.md` | `docs/content/method/_index.md` | `weight: 2` |
| `docs/2_method/2_dtw.md` | `docs/content/method/dtw.md` | `weight: 1`, fix duplicate z-norm section |
| `docs/2_method/3_mip.md` | `docs/content/method/mip.md` | `weight: 2` |
| `docs/2_method/4_k-Medoids.md` | `docs/content/method/k-medoids.md` | `weight: 3` |
| `docs/2_method/5_algorithms.md` | `docs/content/method/algorithms.md` | `weight: 4`, UPDATE: add FastCLARA, hierarchical, CLARANS |
| `docs/2_method/6_metrics.md` | `docs/content/method/metrics.md` | `weight: 5` |
| `docs/5_publications/index.md` | `docs/content/publications/_index.md` | `weight: 5` |
| `docs/5_publications/joss_paper.md` | `docs/content/publications/joss-paper.md` | `weight: 1` |
| `docs/6_Doxygen/index.md` | `docs/content/api/_index.md` | `weight: 6`, rename section to "API Documentation" |
| `docs/6_Doxygen/2_doxygen.md` | `docs/content/api/doxygen.md` | `weight: 1` |
| `docs/6_Doxygen/3_lcov_report.md` | `docs/content/api/lcov-report.md` | `weight: 2` |
| `docs/6_Doxygen/4_codecov_report.md` | `docs/content/api/codecov-report.md` | `weight: 3` |
| `docs/7_develop/index.md` | `docs/content/contributing/_index.md` | `weight: 7` |

### Static assets mapping

| Jekyll path | Hugo path |
|---|---|
| `docs/docs_logo.png` | `docs/static/docs_logo.png` |
| `docs/2_method/dtw_image.png` | `docs/static/method/dtw_image.png` |
| `docs/2_method/cluster_matrix_formation4.svg` | `docs/static/method/cluster_matrix_formation4.svg` |

### Content fixes during migration

1. **`dtw.md`**: Remove duplicate z-Normalization section (lines 89-99). Keep the more comprehensive "Z-Normalization" section (lines 52-83).
2. **`dtw.md`**: Fix image path: `<img src="dtw_image.png"` → `<img src="/method/dtw_image.png"`
3. **`mip.md`**: Fix image path: `cluster_matrix_formation4.svg` → `/method/cluster_matrix_formation4.svg`
4. **`algorithms.md`**: Rewrite "Future Algorithms" — CLARA, CLARANS, hierarchical are now implemented.
5. **`contributing/_index.md`**: Fix broken links to CONTRIBUTING.md, conventions.md, contributors.md — include them inline or copy them into content.
6. **`joss-paper.md`**: Keep the iframe embed as-is (Hugo `unsafe = true` in goldmark config allows raw HTML).

---

## Step 3: Copy contributing docs into content

The existing workflow copies `CONTRIBUTING.md`, `develop/conventions.md`, and `develop/contributors.md` into docs at build time. For Hugo, we have two options:

**Approach: Copy at build time in the workflow** (same as current Jekyll approach)
- In the GitHub Actions workflow, before Hugo build: copy these files into `docs/content/contributing/` with appropriate frontmatter prepended
- The `docs/content/contributing/_index.md` links to them

This preserves the single-source-of-truth for CONTRIBUTING.md at the repo root.

---

## Step 4: Delete old Jekyll files

Remove files that are no longer needed:
- `docs/_config.yml`
- `docs/custom.css`
- `docs/header.html` (Doxygen template — keep in place, Doxygen still needs it)
- `docs/.gitignore` (empty)
- All numbered directories (`docs/1_getting_started/`, `docs/2_method/`, etc.) after content is moved

Keep:
- `docs/Doxyfile` (still needed for Doxygen build)
- `docs/doxygen-awesome.css` and `docs/doxygen-awesome-sidebar-only.css` (Doxygen theme)
- `docs/header.html` (Doxygen header template)

---

## Step 5: Write new documentation pages (~10 new pages)

### 5a. `docs/content/method/dtw-variants.md` (weight: 6)
DTW variant documentation covering:
- DDTW (Derivative DTW) — Keogh & Pazzani 2001
- WDTW (Weighted DTW) — Jeong et al. 2011, logistic weight vector
- ADTW (Amerced DTW) — Herrmann & Shifaz 2023, non-diagonal penalty
- Soft-DTW — Cuturi & Blondel 2017, differentiable, gamma parameter
- Comparison table: which variant for which use case
- CLI usage: `--variant ddtw/wdtw/adtw/softdtw`, `--wdtw-g`, `--adtw-penalty`, `--sdtw-gamma`
- Source: `dtwc/warping_ddtw.hpp`, `warping_wdtw.hpp`, `warping_adtw.hpp`, `soft_dtw.hpp`

### 5b. `docs/content/method/missing-data.md` (weight: 7)
Missing data handling:
- `MissingStrategy` enum: Error, ZeroCost, AROW, Interpolate
- ZeroCost DTW — NaN values contribute zero cost (Yurtman et al. 2023)
- DTW-AROW — diagonal-only alignment for missing regions
- Linear interpolation fallback
- Bitwise NaN detection (safe under `-ffast-math`)
- CLI/Python/C++ API examples
- Source: `dtwc/missing_utils.hpp`, `warping_missing.hpp`, `warping_missing_arow.hpp`

### 5c. `docs/content/method/multivariate.md` (weight: 8)
Multivariate time series support:
- `Data.ndim` interleaved layout
- All DTW variants support `_mv` multivariate counterparts
- Per-channel LB_Keogh lower bounds
- API usage in C++, Python, CLI
- Source: `dtwc/Data.hpp`, all `_mv` functions

### 5d. `docs/content/method/scores.md` (weight: 9)
Cluster quality evaluation:
- Silhouette, Davies-Bouldin, Dunn Index, Inertia, Calinski-Harabasz
- External validation: Adjusted Rand Index, Normalized Mutual Information
- When to use which score
- Python/C++ API examples
- Source: `dtwc/scores.hpp`

### 5e. `docs/content/getting-started/python.md` (weight: 6)
Python API quickstart:
- Installation: `pip install dtwcpp`
- DTW distance computation
- Distance matrix computation
- `DTWClustering` sklearn-compatible class: `fit()`, `predict()`, `fit_predict()`, `score()`
- DTW variants, missing data, metrics
- GPU acceleration: `device="cuda"`
- I/O utilities: CSV, HDF5, Parquet
- Source: `python/dtwcpp/__init__.py`, `python/dtwcpp/_clustering.py`, `python/dtwcpp/io.py`

### 5f. `docs/content/getting-started/matlab.md` (weight: 7)
MATLAB bindings:
- Build: `cmake .. -DDTWC_BUILD_MATLAB=ON`
- `dtwc.dtw_distance`, `dtwc.compute_distance_matrix`
- `dtwc.DTWClustering` handle class
- 1-based indexing
- Source: `bindings/matlab/+dtwc/`, `bindings/matlab/dtwc_mex.cpp`

### 5g. `docs/content/getting-started/checkpointing.md` (weight: 8)
Checkpoint save/load:
- Save partial distance matrices to disk
- Resume after crashes
- CLI: `--checkpoint path/`
- Python: `save_checkpoint()`, `load_checkpoint()`, `CheckpointOptions`
- Source: `dtwc/checkpoint.hpp`

### 5h. `docs/content/getting-started/configuration.md` (weight: 9)
Configuration files:
- TOML config (`--config config.toml`) — CLI11 native
- YAML config (`--yaml-config`, requires `-DDTWC_ENABLE_YAML=ON`)
- Full reference of all config keys (maps to CLI flags)
- Example config files from `examples/config.toml` and `examples/config.yaml`

### 5i. Update `docs/content/method/algorithms.md`
Rewrite to reflect current state:
- FastPAM1 (keep existing, it's accurate)
- Lloyd's (keep)
- MIP (keep, add warm-start info and MIPSettings)
- **NEW: FastCLARA** — subsampling + FastPAM, default sample sizing, scalability
- **NEW: Hierarchical** — single/complete/average linkage, `build_dendrogram()` + `cut_dendrogram()`, max_points=2000 guard, Ward excluded for DTW
- **NEW: CLARANS** — experimental, budget controls
- Remove "Future Algorithms" section

### 5j. Update `docs/content/getting-started/cli.md`
Update CLI docs with all new flags:
- `--variant` (ddtw, wdtw, adtw, softdtw)
- `--metric` (l1, squared_euclidean)
- Variant params: `--wdtw-g`, `--adtw-penalty`, `--sdtw-gamma`
- CLARA: `--sample-size`, `--n-samples`, `--seed`
- Hierarchical: `--linkage`
- MIP: `--mip-gap`, `--time-limit`, `--no-warm-start`, etc.
- Checkpointing: `--checkpoint`
- GPU: `-d/--device`, `--precision`
- Config files: `--config`

---

## Step 6: Update GitHub Actions workflow

Replace `docs/documentation.yml` with Hugo build + Doxygen + LCOV. 

**Key changes:**
- Replace `actions/jekyll-build-pages@v1` with Hugo build (setup Go, setup Hugo extended, `hugo mod get`, `hugo --minify`)
- Add Pagefind search index build (`npx pagefind@1.3.0 --site public`)
- Keep Doxygen step — move output into `docs/public/Doxygen/`
- Keep LCOV coverage step — move output into `docs/public/Coverage/`
- Keep contributing file copy step — prepend Hugo frontmatter
- Upload `docs/public/` as pages artifact

**Workflow structure:**
```yaml
name: Build, test and generate docs
on:
  push:
    branches: [develop]
  pull_request:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      # Copy contributing docs with Hugo frontmatter
      - name: Prepare contributing docs
        run: |
          mkdir -p docs/content/contributing
          # Prepend Hugo frontmatter to each file
          ...

      # Hugo build
      - uses: actions/setup-go@v5
        with: { go-version: '1.22' }
      - uses: peaceiris/actions-hugo@v3
        with: { hugo-version: 'latest', extended: true }
      - name: Build Hugo site
        working-directory: docs
        run: |
          hugo mod get -u
          hugo --minify
      - name: Build search index
        working-directory: docs
        run: npx pagefind@1.3.0 --site public --output-subdir _pagefind

      # Doxygen (same as before)
      - name: Build Doxygen
        uses: mattnotmitt/doxygen-action@v1.9.8
        with: { doxyfile-path: 'docs/Doxyfile' }
      - run: mv ./build/Doxygen/html ./docs/public/Doxygen
      - run: cp -r ./media ./docs/public/Doxygen

      # Coverage (same as before)
      - name: Build and test with coverage
        run: ...
      - run: mv ./build_dir/Coverage ./docs/public/Coverage

      - uses: actions/upload-pages-artifact@v3
        with: { path: docs/public }

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: { name: github-pages }
    steps:
      - uses: actions/deploy-pages@v4
```

---

## Step 7: Update docs/.gitignore

Add Hugo build artifacts:
```
public/
resources/
.hugo_build.lock
```

---

## Files to modify (summary)

### Delete
- `docs/_config.yml`
- `docs/custom.css`
- `docs/.gitignore` (replace with Hugo version)
- `docs/1_getting_started/` (entire directory, after content moved)
- `docs/2_method/` (entire directory, after content moved)
- `docs/5_publications/` (entire directory)
- `docs/6_Doxygen/` (entire directory, after content moved — keep Doxyfile and doxygen CSS at docs/ root)
- `docs/7_develop/` (entire directory)
- `docs/index.md` (moved to content/_index.md)
- `docs/404.md` (moved to content/404.md)

### Create new
- `docs/hugo.toml`
- `docs/go.mod` + `docs/go.sum`
- `docs/content/_index.md` (from index.md)
- `docs/content/getting-started/_index.md` + 9 files (5 migrated + 4 new)
- `docs/content/method/_index.md` + 9 files (5 migrated + 4 new)
- `docs/content/publications/_index.md` + 1 file
- `docs/content/api/_index.md` + 3 files
- `docs/content/contributing/_index.md`
- `docs/static/docs_logo.png`
- `docs/static/method/dtw_image.png`
- `docs/static/method/cluster_matrix_formation4.svg`

### Modify
- `.github/workflows/documentation.yml` (Jekyll → Hugo + Doxygen + LCOV)

---

## Verification

1. **Local Hugo build**: `cd docs && hugo mod get && hugo server` — verify site renders with Oxford blue sidebar, navigation tree, breadcrumbs, math rendering
2. **Content check**: Every existing page accessible at new URLs, no broken internal links
3. **New pages**: All ~10 new pages render with correct navigation ordering
4. **Admonitions**: `` ```note `` blocks render as styled admonition boxes (theme supports this via render-codeblock hooks)
5. **Math**: LaTeX `$$...$$` renders via MathJax on all method pages
6. **Images**: DTW image and cluster matrix SVG display correctly
7. **Search**: Pagefind search works (requires `npx pagefind --site public` after build)
8. **404 page**: `/404.html` works
9. **Doxygen link**: `/Doxygen/index.html` redirect works
10. **CI workflow**: Push to a test branch and verify GitHub Actions builds successfully
