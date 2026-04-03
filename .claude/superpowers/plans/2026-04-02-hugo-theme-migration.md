# BIL Hugo RTD Theme — Migration Plan

> **Goal**: Replace `bil-jekyll-rtd-theme` with a Hugo Module theme (`bil-hugo-rtd-theme`) that replicates the Oxford blue ReadTheDocs look, supports remote consumption by multiple repos (markdown-only), and provides a modern foundation for future JS enhancements.

---

## Phase 0: Repository Setup

### 0.1 Create the theme repo

```bash
# Create the repo on GitHub: Battery-Intelligence-Lab/bil-hugo-rtd-theme
# Then locally:
git clone git@github.com:Battery-Intelligence-Lab/bil-hugo-rtd-theme.git
cd bil-hugo-rtd-theme
hugo mod init github.com/Battery-Intelligence-Lab/bil-hugo-rtd-theme
```

This creates `go.mod` — the file that makes the repo a Hugo Module.

### 0.2 Directory structure to create

```
bil-hugo-rtd-theme/
├── go.mod
├── hugo.toml                      # theme defaults (params, menus, markup config)
├── theme.toml                     # theme metadata (for Hugo theme registry)
├── LICENSE
├── README.md
│
├── assets/
│   ├── scss/
│   │   ├── theme.scss             # main entry point, @imports everything
│   │   ├── _variables.scss        # Oxford blue palette, fonts, layout widths
│   │   ├── _root.scss             # CSS custom properties (TOC levels, dark mode ready)
│   │   ├── _font-face.scss        # @font-face for Lato, Roboto Slab
│   │   ├── _direction.scss        # RTL support
│   │   ├── _layout.scss           # sidebar + content grid
│   │   ├── _rest.scss             # reset / normalize
│   │   └── core/
│   │       ├── sidebar.scss       # sidebar navigation styles
│   │       ├── content.scss       # main content area
│   │       ├── header.scss        # mobile header bar
│   │       ├── container.scss     # responsive container
│   │       ├── markdown.scss      # markdown body styling
│   │       ├── title.scss         # page titles
│   │       ├── addons.scss        # toast notifications, extras
│   │       └── toasts.scss        # alert/warning/note boxes
│   │
│   ├── js/
│   │   ├── theme.js               # main JS (nav, TOC, sidebar toggle, scroll restore)
│   │   └── search.js              # search implementation (Pagefind or FlexSearch)
│   │
│   └── lib/                       # vendored CSS libraries
│       └── primer/                # subset of GitHub Primer CSS (base, buttons, forms, utils)
│
├── layouts/
│   ├── _default/
│   │   ├── baseof.html            # root template (replaces default.liquid)
│   │   ├── single.html            # single page (article)
│   │   ├── list.html              # section index page
│   │   └── search.html            # search results page
│   │
│   ├── partials/
│   │   ├── head.html              # <head> contents (meta, CSS, fonts)
│   │   ├── sidebar.html           # sidebar wrapper (logo, search, toctree)
│   │   ├── toctree.html           # recursive navigation tree
│   │   ├── toctree-node.html      # single node in the tree (recursive partial)
│   │   ├── breadcrumbs.html       # breadcrumb trail
│   │   ├── content.html           # content wrapper + in-page TOC
│   │   ├── footer.html            # prev/next buttons, edit-on-github, copyright
│   │   ├── scripts.html           # JS includes at end of body
│   │   └── shortcode-helpers.html # reusable partial bits
│   │
│   ├── shortcodes/
│   │   ├── note.html              # {% note %} equivalent
│   │   ├── warning.html
│   │   ├── danger.html
│   │   ├── tip.html
│   │   └── mermaid.html           # mermaid diagram blocks
│   │
│   └── 404.html
│
├── static/
│   ├── fonts/
│   │   ├── lato-normal.woff2
│   │   ├── lato-bold.woff2
│   │   ├── lato-normal-italic.woff2
│   │   ├── lato-bold-italic.woff2
│   │   ├── Roboto-Slab-Regular.woff2
│   │   └── Roboto-Slab-Bold.woff2
│   └── images/
│       ├── favicon-16x16.png
│       ├── favicon-32x32.png
│       └── favicon-96x96.png
│
├── i18n/
│   └── en.toml                    # search_docs, search_results, etc.
│
├── data/                          # (empty, available for theme data files)
│
└── exampleSite/                   # self-contained demo site for testing
    ├── hugo.toml
    ├── go.mod
    ├── content/
    │   ├── _index.md
    │   ├── getting-started/
    │   │   ├── _index.md
    │   │   ├── installation.md
    │   │   └── cli.md
    │   └── method/
    │       ├── _index.md
    │       └── dtw.md
    └── static/
        └── docs_logo.png
```

---

## Phase 1: Port the SCSS (visual identity)

This is where the Oxford blue RTD look lives. The existing SCSS is well-structured and transfers almost directly to Hugo.

### 1.1 Copy SCSS files

From `bil-jekyll-rtd-theme/_sass/` → `bil-hugo-rtd-theme/assets/scss/`.

**Key mappings:**

| Jekyll source | Hugo destination | Changes needed |
|---|---|---|
| `_sass/theme.scss` | `assets/scss/theme.scss` | Update `@import` paths |
| `_sass/_variables.scss` | `assets/scss/_variables.scss` | None — copy verbatim |
| `_sass/_root.scss` | `assets/scss/_root.scss` | None |
| `_sass/_font-face.scss` | `assets/scss/_font-face.scss` | Update font paths to `/fonts/` |
| `_sass/_layout.scss` | `assets/scss/_layout.scss` | None |
| `_sass/_rest.scss` | `assets/scss/_rest.scss` | None |
| `_sass/core/*.scss` | `assets/scss/core/*.scss` | None |
| `_sass/lib/@primer/css/` | `assets/lib/primer/` | Flatten path, keep contents |
| `_sass/lib/font-awesome/` | `assets/scss/lib/font-awesome/` | See note below |
| `_sass/lib/rouge/` | `assets/scss/lib/rouge/` | Replace with Hugo's Chroma |

**Critical SCSS variables to preserve** (the visual identity):

```scss
// assets/scss/_variables.scss — these define the Oxford blue look
$theme-blue: rgb(0, 32, 72);        // Oxford navy — THE signature color
$theme-red: #e74c3c;
$theme-green: #1abc9c;
$theme-orange: #e67e22;

$theme-black: $theme-blue;           // text color = dark blue, not black
$theme-menu-background: $theme-blue;  // sidebar background
$theme-menu-active-background: $theme-blue;
$theme-menu-width: 300px;

$body-font: "Lato", BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
$head-font: "Roboto-Slab", sans-serif;
```

### 1.2 Font Awesome → Lucide (modernization)

The Jekyll theme uses Font Awesome 4 webfont for ~10 icons (home, bars, plus-square, minus-square, link, chevrons). Replace with inline SVGs or Lucide icons to eliminate the 100KB+ font file:

Icons used in the theme:
- `fa-home` → sidebar title
- `fa-bars` → mobile menu toggle
- `fa-plus-square-o` / `fa-minus-square-o` → tree expand/collapse
- `fa-link` → heading anchor
- `fa-chevron-left` / `fa-chevron-right` → prev/next buttons

Create a partial `layouts/partials/icon.html`:
```go-html-template
{{/* Usage: {{ partial "icon.html" "home" }} */}}
{{ $icons := dict
  "home" `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>`
  "menu" `<svg ...>...</svg>`
  "expand" `<svg ...>...</svg>`
  "collapse" `<svg ...>...</svg>`
  "link" `<svg ...>...</svg>`
  "chevron-left" `<svg ...>...</svg>`
  "chevron-right" `<svg ...>...</svg>`
}}
{{ index $icons . | safeHTML }}
```

### 1.3 Syntax highlighting

Jekyll uses Rouge. Hugo uses Chroma (built-in, no JS needed).

Remove `_sass/lib/rouge/github.scss` and add to `hugo.toml`:

```toml
[markup.highlight]
  style = "github"
  lineNos = false
  noClasses = false
```

Then generate a CSS file: `hugo gen chromastyles --style=github > assets/scss/lib/_chroma.scss`

### 1.4 Hugo Pipes integration

In `layouts/partials/head.html`, process SCSS via Hugo Pipes:

```go-html-template
{{ $opts := dict "targetPath" "css/theme.css" "outputStyle" "compressed" }}
{{ $style := resources.Get "scss/theme.scss" | toCSS $opts | fingerprint }}
<link rel="stylesheet" href="{{ $style.RelPermalink }}" integrity="{{ $style.Data.Integrity }}">
```

This compiles SCSS at build time — no Node.js, no webpack (the Jekyll theme uses webpack for building the minified CSS; Hugo eliminates that entirely).

---

## Phase 2: Port the Templates (Liquid → Go templates)

This is the main rewrite work. Hugo's Go templates are more powerful but syntactically different.

### 2.1 Base template: `layouts/_default/baseof.html`

Replaces `_layouts/default.liquid`. Structure:

```go-html-template
<!DOCTYPE html>
<html lang="{{ .Lang | default "en" }}" dir="ltr">
<head>
  {{ partial "head.html" . }}
</head>
<body class="container">
  {{ partial "sidebar.html" . }}
  <div class="content-wrap">
    {{/* Mobile header (hidden on desktop) */}}
    <div class="header d-flex flex-justify-between p-2 hide-lg hide-xl">
      <button id="toggle" aria-label="Toggle menu" class="btn-octicon p-2 m-0 text-white" type="button">
        {{ partial "icon.html" "menu" }}
      </button>
      <div class="title flex-1 d-flex flex-justify-center">
        <a class="h4 no-underline py-1 px-2 rounded-1" href="{{ "/" | relURL }}">{{ .Site.Title }}</a>
      </div>
    </div>
    <div class="content p-3 p-sm-5">
      {{ partial "breadcrumbs.html" . }}
      <hr>
      <div role="main" itemscope itemtype="https://schema.org/Article">
        <div class="markdown-body" itemprop="articleBody">
          {{ block "main" . }}{{ end }}
        </div>
      </div>
      {{ partial "footer.html" . }}
    </div>
  </div>
  {{ partial "scripts.html" . }}
</body>
</html>
```

### 2.2 Sidebar: `layouts/partials/sidebar.html`

Replaces `_includes/templates/sidebar.liquid`:

```go-html-template
<div class="sidebar-wrap overflow-hidden">
  <div class="sidebar height-full overflow-y-scroll overflow-x-hidden">
    <div class="header d-flex flex-column p-3 text-center">
      <div class="title pb-1">
        <a class="h2 no-underline py-1 px-2 rounded-1" href="{{ "/" | relURL }}"
           title="{{ .Site.Params.description }}">
          {{ partial "icon.html" "home" }} {{ .Site.Title }}
        </a>
      </div>
      {{ with .Site.Params.logo }}
      <div>
        <a href="{{ "/" | relURL }}">
          <img src="{{ . | relURL }}" width="100%" height="auto" alt="{{ $.Site.Title }} logo">
        </a>
      </div>
      {{ end }}
      {{ with .Site.Params.version }}
      <span class="version">{{ . }}</span>
      {{ end }}
      <form class="search pt-2" action="{{ "search/" | relURL }}" method="get" autocomplete="off">
        <input class="form-control input-block input-sm" type="text" name="q"
               placeholder="{{ i18n "search_docs" | default "Search docs..." }}">
      </form>
    </div>
    <div class="toctree py-2" data-spy="affix" role="navigation" aria-label="main navigation">
      {{ partial "toctree.html" . }}
    </div>
  </div>
</div>
```

### 2.3 Toctree (recursive navigation): `layouts/partials/toctree.html`

This is the most complex part. Jekyll's version walks a flat sorted page list. Hugo has a proper page tree via `.Sections` and `.Pages`.

```go-html-template
{{/* Build the sidebar nav tree from Hugo's section hierarchy */}}
<ul>
  {{ range .Site.Sections }}
    {{ partial "toctree-node.html" (dict "page" . "current" $.Page "level" 1) }}
  {{ end }}
</ul>
```

`layouts/partials/toctree-node.html`:

```go-html-template
{{ $page := .page }}
{{ $current := .current }}
{{ $level := .level }}
{{ $isCurrent := eq $page $current }}
{{ $isAncestor := $current.IsDescendant $page }}
{{ $hasChildren := or ($page.Sections) ($page.RegularPages) }}

<li class="toc level-{{ $level }}{{ if or $isCurrent $isAncestor }} current{{ end }}"
    data-level="{{ $level }}">
  <a href="{{ $page.RelPermalink }}"
     class="{{ if $isCurrent }}current{{ end }}">
    {{ $page.LinkTitle | default $page.Title }}
  </a>
  {{ if $hasChildren }}
  <ul>
    {{/* Section pages (subsections) first */}}
    {{ range $page.Sections }}
      {{ partial "toctree-node.html" (dict "page" . "current" $current "level" (add $level 1)) }}
    {{ end }}
    {{/* Regular pages */}}
    {{ range $page.RegularPages }}
      {{ if not .Params.nav_exclude }}
      <li class="toc level-{{ add $level 1 }}{{ if eq . $current }} current{{ end }}"
          data-level="{{ add $level 1 }}">
        <a href="{{ .RelPermalink }}"
           class="{{ if eq . $current }}current{{ end }}">
          {{ .LinkTitle | default .Title }}
        </a>
      </li>
      {{ end }}
    {{ end }}
  </ul>
  {{ end }}
</li>
```

### 2.4 Page ordering

Hugo uses `weight` in front matter to order pages (lower = earlier). Map the existing numbered folders:

| Jekyll path | Hugo path | Front matter |
|---|---|---|
| `1_getting_started/index.md` | `content/getting-started/_index.md` | `weight: 1` |
| `1_getting_started/1_installation.md` | `content/getting-started/installation.md` | `weight: 1` |
| `2_method/2_dtw.md` | `content/method/dtw.md` | `weight: 2` |

The numbered prefix convention is no longer needed — `weight` controls order.

### 2.5 Breadcrumbs: `layouts/partials/breadcrumbs.html`

```go-html-template
<nav aria-label="breadcrumb">
  <ul class="breadcrumb">
    <li><a href="{{ "/" | relURL }}">{{ .Site.Title }}</a></li>
    {{ range .Ancestors.Reverse }}
      {{ if not .IsHome }}
      <li><a href="{{ .RelPermalink }}">{{ .LinkTitle }}</a></li>
      {{ end }}
    {{ end }}
    <li class="active">{{ .LinkTitle }}</li>
  </ul>
</nav>
```

### 2.6 Footer with prev/next: `layouts/partials/footer.html`

```go-html-template
<footer class="d-flex flex-justify-between py-4">
  {{ with .PrevInSection }}
  <a class="btn" href="{{ .RelPermalink }}">
    {{ partial "icon.html" "chevron-left" }} {{ .LinkTitle }}
  </a>
  {{ end }}

  {{ with .NextInSection }}
  <a class="btn" href="{{ .RelPermalink }}">
    {{ .LinkTitle }} {{ partial "icon.html" "chevron-right" }}
  </a>
  {{ end }}
</footer>

{{ if .Site.Params.edit_on_github }}
<div class="edit-on-github text-right text-small py-2">
  <a href="{{ .Site.Params.repo_url }}edit/main/docs/content/{{ .File.Path }}">
    Edit this page on GitHub
  </a>
</div>
{{ end }}
```

### 2.7 Shortcodes

Replace Jekyll `{% include shortcodes/note.liquid %}` with Hugo shortcodes.

`layouts/shortcodes/note.html`:
```go-html-template
<div class="admonition note">
  <p class="admonition-title">Note</p>
  {{ .Inner | markdownify }}
</div>
```

Similarly for `warning.html`, `danger.html`, `tip.html`.

Usage in markdown changes from:
```liquid
{% include shortcodes/note.liquid content="Some note" %}
```
to:
```markdown
{{< note >}}
Some note
{{< /note >}}
```

### 2.8 MathJax and Mermaid

In `layouts/partials/scripts.html`, conditionally load these:

```go-html-template
{{ if .Params.math }}
<script>
  MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] } };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
{{ end }}

{{ if .Params.mermaid }}
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true });
</script>
{{ end }}
```

Enable per-page via front matter (`math: true`, `mermaid: true`) or site-wide via `hugo.toml` params.

---

## Phase 3: JavaScript Modernization

### 3.1 Port theme.js to vanilla JS

The existing `theme.js` (~290 lines) uses jQuery for:
- Toctree expand/collapse toggling
- In-page TOC generation from headings
- Sidebar scroll position persistence (localStorage)
- Search highlighting
- Heading anchor links
- Mobile sidebar toggle

Rewrite without jQuery (~200 lines of modern JS). Key functions:

```javascript
// assets/js/theme.js — no jQuery dependency

document.addEventListener('DOMContentLoaded', () => {
  initToctree();
  initTableOfContents();
  initSidebarToggle();
  initScrollRestore();
  initHeadingAnchors();
  initCodeCopyButtons();  // NEW: modern enhancement
});

function initToctree() {
  // Expand/collapse logic for sidebar navigation
  document.querySelectorAll('.toctree .toc > a').forEach(link => {
    if (link.nextElementSibling?.tagName === 'UL') {
      const expand = document.createElement('span');
      expand.className = 'tree-toggle';
      expand.innerHTML = /* expand SVG icon */;
      expand.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        link.closest('li').classList.toggle('current');
      });
      link.prepend(expand);
    }
  });
}

function initTableOfContents() {
  // Auto-generate in-page TOC from h2-h6 headings in content
  const currentToc = document.querySelector('.toctree li.toc.current');
  if (!currentToc) return;

  const headings = document.querySelectorAll('.markdown-body h2, .markdown-body h3, .markdown-body h4');
  if (headings.length === 0) return;

  const tocList = document.createElement('ul');
  tocList.className = 'content-toc';

  headings.forEach(heading => {
    const li = document.createElement('li');
    const level = parseInt(heading.tagName[1]);
    li.className = `toc level-${level}`;
    const a = document.createElement('a');
    a.href = `#${heading.id}`;
    a.textContent = heading.textContent;
    li.appendChild(a);
    tocList.appendChild(li);
  });

  currentToc.appendChild(tocList);
}

function initScrollRestore() {
  // Persist sidebar scroll position across page navigations
  const sidebar = document.querySelector('.sidebar');
  if (!sidebar) return;

  const saved = sessionStorage.getItem('sidebar-scroll');
  if (saved) sidebar.scrollTop = parseInt(saved);

  sidebar.addEventListener('scroll', () => {
    sessionStorage.setItem('sidebar-scroll', sidebar.scrollTop);
  });
}

function initSidebarToggle() {
  document.getElementById('toggle')?.addEventListener('click', () => {
    document.querySelectorAll('.sidebar-wrap, .content-wrap').forEach(el => {
      el.classList.toggle('shift');
    });
  });
}

function initHeadingAnchors() {
  document.querySelectorAll('.markdown-body :is(h1,h2,h3,h4,h5,h6)').forEach(heading => {
    if (!heading.id) return;
    const anchor = document.createElement('a');
    anchor.href = `#${heading.id}`;
    anchor.className = 'anchor';
    anchor.innerHTML = /* link SVG icon */;
    heading.appendChild(anchor);
  });
}

function initCodeCopyButtons() {
  // NEW: Add copy button to code blocks
  document.querySelectorAll('pre > code').forEach(block => {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(block.textContent);
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = 'Copy', 2000);
    });
    block.parentElement.style.position = 'relative';
    block.parentElement.appendChild(btn);
  });
}
```

### 3.2 Search: Pagefind (recommended)

Pagefind is a static search library that builds an index after Hugo generates the site. Zero bundle cost — JS only loads when the user searches.

**Integration in GitHub Actions:**

```yaml
- run: hugo --minify
  working-directory: docs
- run: npx pagefind --site public --output-subdir _pagefind
  working-directory: docs
```

**Search page template** (`layouts/_default/search.html`):

```go-html-template
{{ define "main" }}
<div id="search"></div>
<link href="{{ "/_pagefind/pagefind-ui.css" | relURL }}" rel="stylesheet">
<script src="{{ "/_pagefind/pagefind-ui.js" | relURL }}"></script>
<script>
  new PagefindUI({
    element: "#search",
    showSubResults: true,
    showImages: false
  });
  // Pre-fill from ?q= parameter
  const q = new URLSearchParams(location.search).get('q');
  if (q) document.querySelector('.pagefind-ui__search-input').value = q;
</script>
{{ end }}
```

---

## Phase 4: Theme Configuration Interface

### 4.1 Theme's `hugo.toml` (defaults)

```toml
# hugo.toml — theme defaults (consumed repos can override any of these)

[params]
  description = ""
  logo = ""                        # path to logo image (e.g., "docs_logo.png")
  version = ""                     # shown below logo in sidebar
  edit_on_github = false
  repo_url = ""                    # e.g., "https://github.com/Battery-Intelligence-Lab/dtw-cpp/"
  github_docs_folder = "docs/content"

  # Feature toggles
  math_globally = false            # enable MathJax on all pages
  mermaid_globally = false         # enable Mermaid on all pages
  code_copy = true                 # copy button on code blocks
  search_enabled = true

  # Colors (override via CSS custom properties for theming)
  # Default: Oxford blue — consumers can override in their hugo.toml
  theme_color = "rgb(0, 32, 72)"

[markup]
  [markup.highlight]
    style = "github"
    noClasses = false
    lineNos = false
  [markup.goldmark.renderer]
    unsafe = true                  # allow raw HTML in markdown (needed for some content)
  [markup.tableOfContents]
    startLevel = 2
    endLevel = 4

[outputFormats.SearchIndex]
  baseName = "data"
  mediaType = "application/json"

[outputs]
  home = ["HTML", "SearchIndex"]
```

### 4.2 Consumer repo's `hugo.toml` (e.g., dtw-cpp)

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
  version = "v2.0.0"
  edit_on_github = true
  repo_url = "https://github.com/Battery-Intelligence-Lab/dtw-cpp/"
  search_enabled = true
```

That's the ENTIRE config a consuming repo needs. No theme files.

---

## Phase 5: Migrate dtw-cpp docs (consumer side)

### 5.1 New directory structure in dtw-cpp

```
docs/
├── hugo.toml
├── go.mod                              # auto: hugo mod init github.com/...
├── go.sum                              # auto: hugo mod get
├── static/
│   ├── docs_logo.png                   # keep existing
│   └── 2_method/
│       ├── cluster_matrix_formation4.svg
│       └── dtw_image.png
└── content/
    ├── _index.md                       # home page
    ├── getting-started/
    │   ├── _index.md                   # section index (weight: 1)
    │   ├── installation.md             # weight: 1
    │   ├── cli.md                      # weight: 2
    │   ├── mpi-cuda-setup.md           # weight: 3
    │   ├── supported-data.md           # weight: 4
    │   └── examples.md                 # weight: 5
    ├── method/
    │   ├── _index.md                   # section index (weight: 2)
    │   ├── dtw.md                      # weight: 1
    │   ├── mip.md                      # weight: 2
    │   ├── k-medoids.md                # weight: 3
    │   ├── algorithms.md               # weight: 4
    │   └── metrics.md                  # weight: 5
    ├── publications/
    │   ├── _index.md                   # weight: 5
    │   └── joss-paper.md
    ├── doxygen/
    │   ├── _index.md                   # weight: 6
    │   ├── doxygen.md
    │   ├── lcov-report.md
    │   └── codecov-report.md
    └── develop/
        └── _index.md                   # weight: 7
```

### 5.2 Front matter migration

Each markdown file needs updated front matter. Example:

**Before** (Jekyll):
```yaml
---
layout: default
title: Installation
nav_order: 1
parent: Getting Started
---
```

**After** (Hugo):
```yaml
---
title: Installation
weight: 1
---
```

Hugo infers the section from directory structure — no `parent` or `layout` needed.

### 5.3 Content changes

- `{{ site.baseurl }}` references → remove (Hugo handles base URL automatically)
- `{% include shortcodes/note.liquid %}` → `{{< note >}}...{{< /note >}}`
- Image paths: `![DTW](dtw_image.png)` → `![DTW](/method/dtw_image.png)` or use page bundles
- Internal links: use Hugo `ref` shortcode: `[Installation]({{< ref "getting-started/installation" >}})`

### 5.4 Remove old Jekyll files from dtw-cpp

Delete:
- `docs/_config.yml`
- `docs/custom.css`
- `docs/header.html`
- `docs/404.md`
- `docs/Doxyfile` (keep if still needed for separate Doxygen build)
- `docs/doxygen-awesome*.css`
- Numbered folder prefixes

---

## Phase 6: GitHub Actions Workflow

### 6.1 Theme repo CI (bil-hugo-rtd-theme)

```yaml
# .github/workflows/ci.yml
name: Theme CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  build-example:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: 'latest'
          extended: true

      - name: Build example site
        working-directory: exampleSite
        run: |
          hugo mod get
          hugo --minify

      - name: Run Pagefind
        working-directory: exampleSite
        run: npx pagefind --site public

      - name: Deploy preview
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: exampleSite/public
```

### 6.2 Consumer repo workflow (e.g., dtw-cpp)

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation
on:
  push:
    branches: [main]
    paths:
      - 'docs/**'

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: 'latest'
          extended: true

      - name: Pull theme and build
        working-directory: docs
        run: |
          hugo mod get -u
          hugo --minify --baseURL "https://battery-intelligence-lab.github.io/dtw-cpp/"

      - name: Build search index
        working-directory: docs
        run: npx pagefind --site public --output-subdir _pagefind

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/public

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

---

## Phase 7: Future Enhancements (post-migration)

These are improvements to add incrementally after the basic port works:

### 7.1 Dark mode

Add CSS custom properties in `_variables.scss`:

```scss
:root {
  --sidebar-bg: #{$theme-blue};
  --sidebar-text: #{$theme-white};
  --content-bg: #fff;
  --content-text: #404040;
  --code-bg: #{$bg-gray};
}

@media (prefers-color-scheme: dark) {
  :root {
    --content-bg: #1a1a2e;
    --content-text: #e0e0e0;
    --code-bg: #2d2d44;
  }
}
```

### 7.2 Version switcher

Hugo supports building multiple versions via branches/tags. Add a dropdown in the sidebar:

```go-html-template
{{ with .Site.Params.versions }}
<select class="version-switcher" onchange="location.href=this.value">
  {{ range . }}
  <option value="{{ .url }}">{{ .label }}</option>
  {{ end }}
</select>
{{ end }}
```

### 7.3 API reference integration

For Doxygen output, mount it as a static subdirectory:

```toml
# hugo.toml in consumer repo
[[module.mounts]]
  source = "static"
  target = "static"
[[module.mounts]]
  source = "../build/doxygen/html"   # Doxygen output
  target = "static/api"
```

Then link from sidebar: `[API Reference](/api/index.html)`

### 7.4 Interactive examples

Add a shortcode for embedded code playgrounds (e.g., Compiler Explorer for C++):

```go-html-template
{{/* layouts/shortcodes/godbolt.html */}}
<iframe
  width="100%" height="400"
  src="https://godbolt.org/e#{{ .Get 0 }}"
  frameborder="0">
</iframe>
```

---

## Implementation Order

| Step | What | Depends on | Estimated effort |
|------|------|-----------|-----------------|
| 1 | Create theme repo, init Hugo module | Nothing | 10 min |
| 2 | Copy + adapt SCSS files | Step 1 | 1–2 hours |
| 3 | Write baseof.html + head.html partial | Step 2 | 1 hour |
| 4 | Write sidebar.html + toctree partials | Step 3 | 2 hours |
| 5 | Write content.html + footer.html | Step 3 | 30 min |
| 6 | Port theme.js to vanilla JS | Step 4 | 2 hours |
| 7 | Set up exampleSite, verify it looks right | Steps 2–6 | 1 hour |
| 8 | Write shortcodes (note/warning/tip/mermaid) | Step 3 | 30 min |
| 9 | Add Pagefind search | Step 7 | 30 min |
| 10 | Theme CI workflow | Step 7 | 30 min |
| 11 | Migrate dtw-cpp content (frontmatter + paths) | Steps 7–8 | 1–2 hours |
| 12 | dtw-cpp docs workflow | Step 11 | 30 min |
| 13 | Visual comparison + polish | Step 11 | 1–2 hours |
| **Total** | | | **~12 hours** |

---

## Checklist for Visual Parity

Before retiring the Jekyll theme, verify:

- [ ] Oxford blue sidebar with white text matches exactly
- [ ] Lato body + Roboto Slab headings render correctly (woff2 loaded)
- [ ] Tree navigation expands/collapses with same behavior
- [ ] In-page TOC appears in sidebar for current page
- [ ] Breadcrumbs show correct path
- [ ] Prev/next navigation works at bottom of pages
- [ ] Mobile responsive: hamburger menu, sidebar slides in
- [ ] Search returns results
- [ ] Code syntax highlighting matches GitHub style
- [ ] Admonition boxes (note, warning, danger, tip) look correct
- [ ] Logo displays in sidebar header
- [ ] External links open in new tab
- [ ] Heading anchor links appear on hover
- [ ] 404 page works

---

## Key Differences from Jekyll Version (for reference)

| Aspect | Jekyll (old) | Hugo (new) |
|---|---|---|
| Template language | Liquid | Go templates |
| Config format | `_config.yml` | `hugo.toml` |
| Section index | `index.md` | `_index.md` (underscore prefix) |
| Page ordering | folder name prefix (`1_`, `2_`) | `weight` front matter |
| Remote theme | `remote_theme:` in config | Hugo Module import in config |
| SCSS processing | webpack (theme-side) | Hugo Pipes (built-in) |
| Syntax highlighting | Rouge (Ruby) | Chroma (built-in Go) |
| Search | JSON + jQuery regex | Pagefind (Rust, static index) |
| Icons | Font Awesome webfont | Inline SVG (zero network requests) |
| JS dependency | jQuery 3.x (~87KB) | None (vanilla JS) |
| Build tool | Ruby + Bundler + webpack | Single `hugo` binary |