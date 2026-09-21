#!/usr/bin/env python3
"""Check external and local links in markdown files.

Converts markdown to HTML (via the `markdown` library) and extracts <a>
hrefs with BeautifulSoup, so nested brackets (e.g. badge images inside
links) are handled correctly. Local links are resolved relative to each
source file; external links are checked with HTTP HEAD (GET fallback).

Exit 1 if any broken link is found, 0 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
import requests
from bs4 import BeautifulSoup


USER_AGENT = 'Mozilla/5.0 (modeltest link checker)'
TIMEOUT = 10


def strip_frontmatter(text: str) -> str:
    if not text.startswith('---'):
        return text
    parts = text.split('---', 2)
    return parts[2] if len(parts) >= 3 else text

def md_to_html(text: str) -> str:
    text = strip_frontmatter(text)
    return markdown.Markdown().convert(text)


def extract_hrefs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    hrefs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith("#"):
            hrefs.append(href)
    return hrefs


def strip_fragment(href: str) -> tuple[str, str]:
    """Separa la ruta del anchor fragment (si existe).

    El fragmento es un anchor DENTRO del archivo target y no afecta la
    existencia del archivo; se quita antes de resolver la ruta local.
    """
    if "#" in href:
        path_part, frag = href.split("#", 1)
        return path_part, frag
    return href, ""


def resolve_local(href: str, base: Path) -> Path:
    if href.startswith("/"):
        return Path(href.lstrip("/"))
    return (base / href).resolve()


def check_local(path: Path) -> tuple[bool, str]:
    if path.exists():
        return True, ""
    # fallback: href que omitió la extensión .md
    if not path.suffix:
        alt = path.with_suffix(".md")
        if alt.exists():
            return True, ""
    return False, f"not found: {path}"
def check_external(url: str) -> tuple[bool, str]:
    try:
        r = requests.head(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        if r.status_code == 405:
            r = requests.get(
                url,
                stream=True,
                allow_redirects=True,
                timeout=TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
        if r.status_code == 200:
            return True, str(r.status_code)
        if r.status_code == 404:
            return False, f"HTTP {r.status_code}"
        # other codes (3xx already followed, 403, 5xx...): ok to avoid
        # false positives on rate-limited / temporarily-down sources
        return True, str(r.status_code)
    except requests.exceptions.Timeout:
        return False, "timeout"
    except requests.exceptions.ConnectionError:
        return False, "connection error"
    except requests.exceptions.RequestException as e:
        return False, f"error: {e}"



def check_link(href: str, base: Path) -> tuple[bool, str]:
    if href.startswith("http://") or href.startswith("https://"):
        return check_external(href)
    # enlaces locales: quitar fragmento antes de resolver, el anchor no
    # afecta la existencia del archivo
    path_part, _ = strip_fragment(href)
    return check_local(resolve_local(path_part, base))


def main() -> int:
    root = Path.cwd()
    args = sys.argv[1:]

    if args:
        # Usar las rutas proporcionadas (soporta archivos y directorios)
        files: list[Path] = []
        for arg in args:
            p = Path(arg)
            if not p.exists():
                print(f"Warning: path not found: {arg}", file=sys.stderr)
                continue
            if p.is_dir():
                files.extend(sorted(p.rglob("*.md")))
            else:
                files.append(p)
    else:
        # Por defecto: docs/ + README.md
        files: list[Path] = []
        docs_dir = root / "docs"
        if docs_dir.is_dir():
            files.extend(sorted(docs_dir.rglob("*.md")))
        readme = root / "README.md"
        if readme.is_file():
            files.append(readme)

    if not files:
        print('No markdown files found.', file=sys.stderr)
        return 1

    broken: list[tuple[Path, str, str]] = []
    checked = 0

    for md_file in files:
        base = md_file.parent
        html = md_to_html(md_file.read_text(encoding='utf-8'))
        for href in sorted(set(extract_hrefs(html))):
            checked += 1
            ok, reason = check_link(href, base)
            if not ok:
                broken.append((md_file, href, reason))

    if broken:
        print(f"FAIL: {len(broken)} broken link(s) out of {checked} checked:")
        for src, href, reason in broken:
            try:
                rel = src.relative_to(root)
            except ValueError:
                rel = src.name
            print(f"  [{rel}] {href} — {reason}")
        return 1

    print(f'OK: all {checked} links are valid.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
