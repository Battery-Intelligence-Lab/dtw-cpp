#!/usr/bin/env python3
"""Validate every local href/src in a built Hugo site."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = "href" if tag in {"a", "link"} else "src" if tag in {
            "img", "script", "source"
        } else None
        if attr is None:
            return
        for key, value in attrs:
            if key == attr and value:
                self.values.append((attr, value))


def target_exists(public: Path, url_path: str) -> bool:
    path = public / unquote(url_path.lstrip("/"))
    candidates = [path]
    if url_path.endswith("/"):
        candidates.append(path / "index.html")
    elif not path.suffix:
        candidates.extend((path / "index.html", path.with_suffix(".html")))
    return any(candidate.is_file() for candidate in candidates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("public", type=Path)
    parser.add_argument("--base-path", default="/dtw-cpp/")
    parser.add_argument("--skip-subtree", action="append",
                        default=["Doxygen", "Coverage", "_pagefind"],
                        help="generated subtree whose internal links are owned by its generator")
    args = parser.parse_args()
    public = args.public.resolve()
    if not public.is_dir():
        parser.error(f"site directory does not exist: {public}")

    failures: list[str] = []
    for html_file in public.rglob("*.html"):
        relative_path = html_file.relative_to(public)
        if relative_path.parts and relative_path.parts[0] in set(args.skip_subtree):
            continue
        parser_ = References()
        parser_.feed(html_file.read_text(encoding="utf-8", errors="replace"))
        relative = relative_path.as_posix()
        page_url = "/" + relative
        if page_url.endswith("index.html"):
            page_url = page_url[:-len("index.html")]
        for attr, raw in parser_.values:
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or raw.startswith(("#", "mailto:", "tel:", "data:")):
                continue
            resolved = urlsplit(urljoin(page_url, raw)).path
            if resolved.startswith(args.base_path):
                resolved = "/" + resolved[len(args.base_path):].lstrip("/")
            if resolved in {"", "/"}:
                resolved = "/index.html"
            if not target_exists(public, resolved):
                failures.append(f"{relative}: {attr}={raw!r} -> {resolved}")

    if failures:
        print("broken internal site links:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("all internal site links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
