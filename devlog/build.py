#!/usr/bin/env python3
"""Static site generator for the Freestyle Vision dev log.

Reads ``devlog/content/*.md``, ``devlog/models.yaml``, ``devlog/code_tour.yaml``,
and ``devlog/site/status.json`` and writes static mobile-first HTML/CSS/assets
into ``devlog/site/``.
"""

from __future__ import annotations

import ast
import html
import json
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

DEVLOG_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVLOG_DIR.parent
CONTENT_DIR = DEVLOG_DIR / "content"
SITE_DIR = DEVLOG_DIR / "site"
ASSETS_DIR = SITE_DIR / "assets"
IMG_ASSETS_DIR = ASSETS_DIR / "img"
ENTRIES_DIR = SITE_DIR / "entries"
STATUS_PATH = SITE_DIR / "status.json"
MODELS_PATH = DEVLOG_DIR / "models.yaml"
CODE_TOUR_PATH = DEVLOG_DIR / "code_tour.yaml"

# Inline Lucide-style SVG icons (MIT/ISC). No external icon CDNs.
ICONS: dict[str, str] = {
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "home": '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/><path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "book": '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a1 1 0 0 1 0-5H20"/>',
    "arrow-left": '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    "external": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
}


def icon_svg(name: str, size: int = 16, cls: str = "") -> str:
    body = ICONS.get(name, "")
    cls_attr = f' class="icon icon--{name} {cls}"'.rstrip() if cls else f' class="icon icon--{name}"'
    return (
        f'<svg{cls_attr} xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">'
        f"{body}</svg>"
    )


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "section"


def highlight_code(code: str, lang: str = "python") -> str:
    """Render code into static Pygments HTML inside a dark Gelato slab."""
    try:
        lexer = get_lexer_by_name(lang, stripall=False)
    except ClassNotFound:
        try:
            lexer = guess_lexer(code)
        except ClassNotFound:
            lexer = get_lexer_by_name("text", stripall=False)
    formatter = HtmlFormatter(nowrap=True, cssclass="highlight")
    highlighted = highlight(code, lexer, formatter).rstrip("\n")
    return f'<pre class="code-slab"><code class="highlight language-{html.escape(lang)}">{highlighted}</code></pre>'


# ---------------------------------------------------------------------------
# Markdown parser + front matter + safe image asset copy/downscale
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    slug: str
    title: str
    date: str
    sort_ts: float
    display_date: str
    tags: list[str]
    summary: str
    body_html: str
    headings: list[dict[str, Any]]
    reading_min: int


def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end != -1:
            meta = yaml.safe_load(raw[4:end]) or {}
            body = raw[end + 5 :]
            return meta, body
    return {}, raw


def _copy_and_downscale_image(src_ref: str) -> str | None:
    """Copy a repo-referenced image (e.g. under runs/) into site/assets/img/ <=1600px wide."""
    if re.match(r"^https?://", src_ref, re.IGNORECASE):
        return src_ref
    candidate = (REPO_ROOT / src_ref.lstrip("/")).resolve()
    # Safety guard: never touch data/ or /home/david/dev/eg/
    data_dir = (REPO_ROOT / "data").resolve()
    eg_dir = Path("/home/david/dev/eg").resolve()
    if candidate.is_relative_to(data_dir) or candidate.is_relative_to(eg_dir):
        return None
    if not candidate.is_relative_to(REPO_ROOT.resolve()) or not candidate.is_file():
        return None

    IMG_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", src_ref.lstrip("/"))
    dest = IMG_ASSETS_DIR / safe_name
    with Image.open(candidate) as im:
        im = ImageOps.exif_transpose(im)
        if im.width > 1600:
            im.thumbnail((1600, 16000), Image.Resampling.LANCZOS)
        if dest.suffix.lower() in (".jpg", ".jpeg"):
            im.convert("RGB").save(dest, format="JPEG", quality=88, optimize=True)
        else:
            im.save(dest)
    return f"assets/img/{safe_name}"


def _render_inline(text: str, root_prefix: str = "") -> str:
    # Extract inline code spans first so formatting does not touch inside code.
    placeholders: list[str] = []

    def _stash_code(match: re.Match[str]) -> str:
        idx = len(placeholders)
        placeholders.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00CODE{idx}\x00"

    work = re.sub(r"`([^`]+)`", _stash_code, text)

    # Images ![alt](url)
    def _img_sub(match: re.Match[str]) -> str:
        alt = html.escape(match.group(1))
        href = match.group(2).strip()
        copied = _copy_and_downscale_image(href)
        if copied is None:
            return f'<span class="tag-pill">Image unavailable: {html.escape(href)}</span>'
        src = copied if copied.startswith("http") else f"{root_prefix}{copied}"
        return f'<figure class="entry-figure"><img src="{html.escape(src)}" alt="{alt}" loading="lazy"><figcaption>{alt}</figcaption></figure>'

    work = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img_sub, work)

    # Links [label](url)
    def _link_sub(match: re.Match[str]) -> str:
        label = html.escape(match.group(1))
        href = match.group(2).strip()
        if href.startswith("http"):
            return f'<a href="{html.escape(href)}" target="_blank" rel="noopener">{label}</a>'
        return f'<a href="{html.escape(href)}">{label}</a>'

    # Escape raw text outside already-emitted tags
    parts = re.split(r"(<figure[\s\S]*?</figure>)", work)
    rendered_parts: list[str] = []
    for part in parts:
        if part.startswith("<figure"):
            rendered_parts.append(part)
            continue
        segs = re.split(r"(\[[^\]]+\]\([^)]+\))", part)
        out_segs: list[str] = []
        for seg in segs:
            m = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", seg)
            if m:
                out_segs.append(_link_sub(m))
            else:
                esc = html.escape(seg)
                esc = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", esc)
                esc = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", esc)
                out_segs.append(esc)
        rendered_parts.append("".join(out_segs))

    joined = "".join(rendered_parts)
    for idx, code_html in enumerate(placeholders):
        joined = joined.replace(f"\x00CODE{idx}\x00", code_html)
    return joined


def render_markdown(md: str, root_prefix: str = "") -> tuple[str, list[dict[str, Any]], str]:
    """Convert Markdown into Gelato-styled HTML, heading list, and first-paragraph summary."""
    lines = md.splitlines()
    blocks: list[str] = []
    headings: list[dict[str, Any]] = []
    used_slugs: dict[str, int] = {}
    first_para = ""
    i = 0

    def unique_slug(title: str) -> str:
        base = slugify(title)
        n = used_slugs.get(base, 0)
        used_slugs[base] = n + 1
        return base if n == 0 else f"{base}-{n}"

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip() or "text"
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(highlight_code("\n".join(code_lines), lang))
            continue

        # Headings
        hm = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if hm:
            depth = len(hm.group(1))
            raw_title = hm.group(2).strip()
            hslug = unique_slug(re.sub(r"`([^`]+)`", r"\1", raw_title))
            if depth in (2, 3):
                headings.append({"depth": depth, "slug": hslug, "text": re.sub(r"`([^`]+)`", r"\1", raw_title)})
            blocks.append(f'<h{depth} id="{hslug}">{_render_inline(raw_title, root_prefix)}</h{depth}>')
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}$", stripped):
            blocks.append("<hr>")
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            q_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(f"<blockquote><p>{_render_inline(' '.join(q_lines), root_prefix)}</p></blockquote>")
            continue

        # Markdown table
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\|?[\s:|\-]+\|?$", lines[i + 1].strip()):
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip():
                row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(row_cells)
                i += 1
            th_html = "".join(f"<th>{_render_inline(c, root_prefix)}</th>" for c in header_cells)
            tr_html = "".join(
                "<tr>" + "".join(f"<td>{_render_inline(c, root_prefix)}</td>" for c in r) + "</tr>"
                for r in rows
            )
            blocks.append(
                f'<div class="table-wrap"><table><thead><tr>{th_html}</tr></thead>'
                f"<tbody>{tr_html}</tbody></table></div>"
            )
            continue

        # Lists (unordered or ordered)
        ul_m = re.match(r"^[-*]\s+(.+)$", stripped)
        ol_m = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ul_m or ol_m:
            tag = "ul" if ul_m else "ol"
            pat = r"^[-*]\s+(.+)$" if ul_m else r"^\d+\.\s+(.+)$"
            items: list[str] = []
            while i < len(lines):
                cur = lines[i].strip()
                m = re.match(pat, cur)
                if m:
                    items.append(m.group(1))
                    i += 1
                elif cur and not cur.startswith(("#", "```", ">")) and items and (lines[i].startswith("  ") or lines[i].startswith("\t")):
                    items[-1] += " " + cur
                    i += 1
                else:
                    break
            li_html = "".join(f"<li>{_render_inline(item, root_prefix)}</li>" for item in items)
            blocks.append(f"<{tag}>{li_html}</{tag}>")
            continue

        # Paragraph
        p_lines: list[str] = []
        while i < len(lines):
            cur = lines[i].strip()
            if not cur or cur.startswith(("#", "```", ">", "- ", "* ")) or re.match(r"^\d+\.\s+", cur) or re.match(r"^[-*_]{3,}$", cur):
                break
            if "|" in cur and i + 1 < len(lines) and re.match(r"^\|?[\s:|\-]+\|?$", lines[i + 1].strip()):
                break
            p_lines.append(cur)
            i += 1
        para_text = " ".join(p_lines)
        if not first_para and not para_text.startswith("!["):
            first_para = re.sub(r"`([^`]+)`", r"\1", re.sub(r"\*\*([^*]+)\*\*", r"\1", para_text))
        blocks.append(f"<p>{_render_inline(para_text, root_prefix)}</p>")

    return "\n".join(blocks), headings, first_para


def load_entries() -> list[Entry]:
    entries: list[Entry] = []
    if not CONTENT_DIR.exists():
        return entries
    for path in sorted(CONTENT_DIR.glob("*.md")):
        raw = path.read_text()
        meta, body = parse_front_matter(raw)
        title = str(meta.get("title") or path.stem)
        date_str = str(meta.get("date") or path.stat().st_mtime)
        try:
            dt = datetime.fromisoformat(date_str)
            sort_ts = dt.timestamp()
            display_date = dt.strftime("%Y-%m-%d %H:%M") if "T" in date_str else dt.strftime("%Y-%m-%d")
        except ValueError:
            sort_ts = path.stat().st_mtime
            display_date = date_str
        tags = [str(t) for t in (meta.get("tags") or [])]
        body_html, headings, summary = render_markdown(body, root_prefix="../")
        words = len(re.findall(r"\w+", body))
        reading_min = max(1, round(words / 210))
        entries.append(
            Entry(
                slug=path.stem,
                title=title,
                date=date_str,
                sort_ts=sort_ts,
                display_date=display_date,
                tags=tags,
                summary=summary,
                body_html=body_html,
                headings=headings,
                reading_min=reading_min,
            )
        )
    entries.sort(key=lambda e: e.sort_ts, reverse=True)
    return entries


# ---------------------------------------------------------------------------
# AST code-tour extraction
# ---------------------------------------------------------------------------


def extract_symbol_source(rel_path: str, symbol: str) -> tuple[str, int, int] | None:
    """Extract a function, class, or Class.method from a Python file using AST."""
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return None
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    parts = symbol.split(".")

    def find_in_body(body: list[ast.stmt], names: list[str]) -> ast.AST | None:
        target = names[0]
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target:
                if len(names) == 1:
                    return node
                if isinstance(node, ast.ClassDef):
                    return find_in_body(node.body, names[1:])
        return None

    node = find_in_body(tree.body, parts)
    if node is None or not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None

    start = node.lineno
    decs = getattr(node, "decorator_list", [])
    if decs:
        start = min(start, *(d.lineno for d in decs if hasattr(d, "lineno")))
    end = node.end_lineno or start
    lines = source.splitlines()[start - 1 : end]
    snippet = textwrap.dedent("\n".join(lines))
    return snippet, start, end


def load_code_tour() -> dict[str, Any]:
    if not CODE_TOUR_PATH.exists():
        return {"title": "Code Tour", "subtitle": "", "stops": []}
    data = yaml.safe_load(CODE_TOUR_PATH.read_text()) or {}
    stops: list[dict[str, Any]] = []
    for raw in data.get("stops", []):
        extracted = extract_symbol_source(raw["file"], raw["symbol"])
        if extracted is None:
            code_html = highlight_code(f"# Symbol {raw['symbol']} not found in {raw['file']}", "python")
            line_range = "missing"
        else:
            code_text, start, end = extracted
            code_html = highlight_code(code_text, "python")
            line_range = f"L{start}-L{end}"
        stops.append(
            {
                "id": raw.get("id") or slugify(raw["symbol"]),
                "title": raw.get("title") or raw["symbol"],
                "file": raw["file"],
                "symbol": raw["symbol"],
                "line_range": line_range,
                "explanation": str(raw.get("explanation", "")).strip(),
                "code_html": code_html,
            }
        )
    return {
        "title": data.get("title", "Code Tour"),
        "subtitle": data.get("subtitle", ""),
        "stops": stops,
    }


def load_models() -> list[dict[str, Any]]:
    if not MODELS_PATH.exists():
        return []
    data = yaml.safe_load(MODELS_PATH.read_text()) or {}
    return list(data.get("models", []))


def load_status() -> dict[str, Any]:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except Exception:
            pass
    return {
        "version": 1,
        "timestamp": "",
        "local_timestamp": "not yet collected",
        "pi_workers": {"ok": False, "status": "unavailable", "workers": []},
        "external_agents": {"ok": False, "status": "unavailable", "latest_by_agent": [], "recent": []},
        "k8s": {"ok": False, "status": "unavailable", "jobs": [], "pods": []},
        "gpus": {"ok": False, "status": "unavailable", "cards": []},
        "git": {"ok": False, "status": "unavailable", "commits": [], "oneline": []},
        "tests": {"ok": False, "status": "unavailable"},
        "experiments": {"ok": False, "status": "unavailable", "metrics_runs": [], "smoke_runs": []},
    }


# ---------------------------------------------------------------------------
# CSS + client JS (Mimics Gelato study-guide with Coca-Cola-red accent)
# ---------------------------------------------------------------------------

SITE_CSS = r"""/* Freestyle Dev Log — Gelato layout & typography with Coca-Cola red accent */

:root,
:root[data-theme='light'] {
  color-scheme: light;

  --bg: #fff8f1;
  --bg-subtle: #fbefe4;
  --surface: #ffffff;
  --surface-alt: #fdf3ea;
  --ink: #2b2330;
  --ink-muted: #6e6470;
  --border: #efe3d8;
  --border-strong: #e2d3c6;

  /* Coca-Cola red-ish accent + warm Gelato pastels */
  --cola: #c8102e;
  --cola-ink: #9e0b23;
  --cola-soft: #fde8ec;
  --cola-title: #480914;
  --cola-body: #3e141c;

  --sky: #dcebfa;
  --sky-ink: #2a4a6e;
  --sky-title: #1e2a3a;
  --pistachio: #ddefd8;
  --pistachio-ink: #2f5b33;
  --pistachio-body: #1f3a22;
  --strawberry: #fbdde3;
  --strawberry-ink: #8c2f45;
  --strawberry-body: #4a1f2a;
  --butter: #fff0c2;
  --butter-ink: #7a5a00;
  --butter-body: #4a3800;

  --code-ink: #6a222d;
  --code-block-bg: #2b2330;
  --code-block-fg: #f6e9ee;
  --contrast-bg: #2b2330;
  --contrast-fg: #fff8f1;
  --pager-next-label: #e7c9d3;

  --focus: #c8102e;
  --backdrop: rgba(43, 35, 48, 0.38);
  --shadow: 0 8px 28px rgba(80, 50, 40, 0.14);
  --shadow-pill: 0 2px 10px rgba(80, 50, 40, 0.06);

  --font-serif: 'Young Serif', Georgia, serif;
  --font-sans: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'DM Mono', ui-monospace, 'DejaVu Sans Mono', monospace;

  --measure: 72ch;
  --topbar-h: 52px;
  --sidebar-w: 272px;
  --onpage-w: 210px;

  --r-xs: 8px;
  --r-sm: 16px;
  --r-md: 20px;
  --r-lg: 22px;
  --r-xl: 26px;
  --r-2xl: 28px;
  --r-pill: 999px;

  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
}

:root[data-theme='dark'] {
  color-scheme: dark;

  --bg: #34313a;
  --bg-subtle: #3a3740;
  --surface: #423e48;
  --surface-alt: #3a3740;
  --ink: #f3eef0;
  --ink-muted: #bdb5bf;
  --border: #4a4550;
  --border-strong: #57515e;

  --cola: #f06474;
  --cola-ink: #ffb5be;
  --cola-soft: #57333c;
  --cola-title: #ffe6ea;
  --cola-body: #f8dce0;

  --sky: #3c4a5a;
  --sky-ink: #cfe2f7;
  --sky-title: #eaf3fd;
  --pistachio: #3c4e40;
  --pistachio-ink: #cdebc6;
  --pistachio-body: #e3f5de;
  --strawberry: #5a3f47;
  --strawberry-ink: #f8cbd5;
  --strawberry-body: #f6dce2;
  --butter: #554e32;
  --butter-ink: #fbe7a6;
  --butter-body: #f5edd2;

  --code-ink: #ffb5be;
  --code-block-bg: #2a2830;
  --code-block-fg: #f6e9ee;
  --contrast-bg: #f06474;
  --contrast-fg: #2b2330;
  --pager-next-label: #480914;

  --focus: #f06474;
  --backdrop: rgba(20, 18, 24, 0.55);
  --shadow: 0 8px 28px rgba(20, 18, 24, 0.45);
  --shadow-pill: 0 2px 10px rgba(20, 18, 24, 0.3);
}

*, *::before, *::after { box-sizing: border-box; }

html {
  -webkit-text-size-adjust: 100%;
  text-size-adjust: 100%;
  scroll-padding-top: calc(var(--topbar-h) + 20px);
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font-sans);
  font-size: 16px;
  font-weight: 400;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}

@media (min-width: 1024px) {
  body { font-size: 16.5px; }
}

h1, h2, h3, h4, p, ul, ol, figure, blockquote { margin: 0; }
ul, ol { padding-left: 1.4em; }
img, svg { display: block; max-width: 100%; height: auto; }
button, input, select { font: inherit; color: inherit; }

a {
  color: var(--cola-ink);
  text-decoration-line: underline;
  text-underline-offset: 2px;
  text-decoration-thickness: 1px;
}
a:hover { text-decoration-thickness: 2px; }

:focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 2px;
  border-radius: var(--r-xs);
}

[hidden] { display: none !important; }
body.scroll-locked { overflow: hidden; }

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}

.skip-link {
  position: absolute;
  left: -9999px;
  top: 0;
  z-index: 100;
  padding: 8px 14px;
  background: var(--surface);
  color: var(--cola-ink);
  border-radius: var(--r-pill);
  box-shadow: var(--shadow);
  font-size: 14px;
  font-weight: 700;
  text-decoration: none;
}
.skip-link:focus { position: fixed; top: 8px; left: 8px; }

.icon-btn {
  width: 44px;
  height: 44px;
  display: inline-grid;
  place-items: center;
  border: 0;
  background: transparent;
  color: var(--ink-muted);
  border-radius: var(--r-pill);
  cursor: pointer;
  padding: 0;
  flex: none;
  -webkit-tap-highlight-color: transparent;
}
.icon-btn:hover { color: var(--ink); }

.theme-icon--sun { display: none; }
.theme-icon--moon { display: block; }
html[data-theme='dark'] .theme-icon--sun { display: block; }
html[data-theme='dark'] .theme-icon--moon { display: none; }

/* ---------- Topbar ---------- */

.topbar {
  --topbar-band: 17px;
  position: sticky;
  top: 0;
  z-index: 40;
  padding: calc(12px + env(safe-area-inset-top)) calc(12px + env(safe-area-inset-right)) 0
    calc(12px + env(safe-area-inset-left));
}

.topbar::before {
  content: '';
  position: absolute;
  inset: 0 0 calc(var(--topbar-band) * -1) 0;
  z-index: -1;
  pointer-events: none;
  background: color-mix(in srgb, var(--bg) 60%, transparent);
  -webkit-backdrop-filter: blur(14px) saturate(1.2);
  backdrop-filter: blur(14px) saturate(1.2);
  -webkit-mask-image: linear-gradient(to bottom, #000 70%, transparent);
  mask-image: linear-gradient(to bottom, #000 70%, transparent);
}

.topbar__pill {
  height: var(--topbar-h);
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 6px;
  background: var(--surface);
  border-radius: var(--r-pill);
  box-shadow: var(--shadow-pill);
}

#toc-toggle.topbar__menu {
  background: var(--cola-soft);
  color: var(--cola-ink);
}

.hamburger {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 3.5px;
}
.hamburger span {
  display: block;
  width: 16px;
  height: 2.5px;
  border-radius: 2px;
  background: currentColor;
}

.topbar__title {
  flex: 0 1 auto;
  min-width: 0;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-family: var(--font-serif);
  font-weight: 400;
  font-size: 18px;
  line-height: 1;
  color: var(--ink);
  text-decoration: none;
  white-space: nowrap;
  padding-left: 4px;
}
.topbar__title:hover { text-decoration: none; color: var(--ink); }

.brand-dot {
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: var(--cola);
  flex: none;
}

.topbar__title-rest {
  display: none;
  font-family: var(--font-sans);
  font-size: 14px;
  font-weight: 600;
  color: var(--ink-muted);
}

.topbar__section {
  display: none;
  flex: 0 1 auto;
  min-width: 0;
  padding-left: 14px;
  margin-left: 4px;
  border-left: 1px solid var(--border);
  font-size: 13.5px;
  font-weight: 600;
  color: var(--ink-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.topbar__actions {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  flex: none;
}

.live-chip {
  height: 36px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0 11px;
  border-radius: var(--r-pill);
  background: var(--pistachio);
  color: var(--pistachio-ink);
  font-size: 12px;
  font-weight: 700;
  text-decoration: none;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.live-chip--warn { background: var(--butter); color: var(--butter-ink); }
.live-chip--down { background: var(--strawberry); color: var(--strawberry-ink); }

.pulse-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: currentColor;
  flex: none;
}

.topbar__theme {
  background: var(--butter);
  color: var(--butter-ink);
}

@media (min-width: 640px) {
  .topbar { padding: calc(14px + env(safe-area-inset-top)) 20px 0; }
  .topbar__section:not(:empty) { display: block; }
  .topbar__title-rest { display: inline; }
}

@media (min-width: 1024px) {
  .topbar__pill {
    height: 58px;
    padding: 0 10px 0 22px;
    gap: 10px;
  }
  .topbar__title { font-size: 21px; padding-left: 0; }
  #toc-toggle.topbar__menu { display: none; }
}

/* ---------- Shell Grid ---------- */

.shell {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
}
@media (min-width: 1024px) {
  .shell { grid-template-columns: var(--sidebar-w) minmax(0, 1fr); }
}
@media (min-width: 1280px) {
  .shell { grid-template-columns: var(--sidebar-w) minmax(0, 1fr) var(--onpage-w); }
}

/* ---------- Sidebar / Drawer TOC ---------- */

.toc-panel {
  --toc-pad-x: calc(14px + env(safe-area-inset-left));
  background: var(--bg);
  font-family: var(--font-sans);
}

@media (max-width: 1023.98px) {
  .toc-panel {
    position: fixed;
    inset: 0 auto 0 0;
    width: min(86vw, 340px);
    z-index: 60;
    background: var(--bg);
    box-shadow: 8px 0 28px rgba(43, 35, 48, 0.2);
    border-radius: 0 var(--r-2xl) var(--r-2xl) 0;
    transform: translateX(-100%);
    transition: transform 200ms ease, visibility 0s linear 200ms;
    visibility: hidden;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 0 var(--toc-pad-x) calc(28px + env(safe-area-inset-bottom));
  }
  .toc-panel[data-state='open'] {
    visibility: visible;
    transform: none;
    transition: transform 200ms ease, visibility 0s;
  }
  .toc-backdrop {
    position: fixed;
    inset: 0;
    z-index: 55;
    background: var(--backdrop);
    opacity: 0;
    transition: opacity 180ms ease;
  }
  .toc-backdrop.is-visible { opacity: 1; }
}

@media (min-width: 1024px) {
  .toc-panel {
    --toc-pad-x: 12px;
    position: sticky;
    top: calc(var(--topbar-h) + 18px);
    height: calc(100dvh - var(--topbar-h) - 18px);
    overflow-y: auto;
    background: transparent;
    padding: 4px var(--toc-pad-x) 32px;
  }
  .toc-panel__head,
  .toc-backdrop { display: none; }
}

.toc-panel__head {
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-inline: calc(-1 * var(--toc-pad-x));
  padding: 12px var(--toc-pad-x) 10px;
  background: var(--bg);
}

@media (min-width: 1024px) {
  .toc-panel .toc-panel__head,
  .toc-backdrop {
    display: none;
  }
}

.toc-panel__title {
  font-family: var(--font-serif);
  font-size: 22px;
  font-weight: 400;
  color: var(--ink);
}

.toc-panel__close {
  background: var(--surface);
  color: var(--ink);
}

.toc-group-label {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--ink-muted);
  padding: 14px 10px 6px;
}

.toc__list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.toc__link {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 42px;
  padding: 8px 12px;
  border-radius: var(--r-sm);
  color: var(--ink);
  text-decoration: none;
  font-size: 14.5px;
  font-weight: 600;
  line-height: 1.3;
  transition: background-color 120ms ease, color 120ms ease;
}

.toc__badge {
  flex: none;
  width: 28px;
  height: 28px;
  border-radius: var(--r-pill);
  background: var(--surface);
  color: var(--ink-muted);
  display: inline-grid;
  place-items: center;
}

.toc__link[aria-current='page'] {
  background: var(--cola-soft);
  color: var(--cola-title);
  font-weight: 700;
}
.toc__link[aria-current='page'] .toc__badge {
  background: var(--cola);
  color: #ffffff;
}

@media (hover: hover) {
  .toc__link:not([aria-current='page']):hover {
    background: var(--surface);
    color: var(--ink);
  }
}

.toc__sublist {
  list-style: none;
  margin: 4px 0 6px 14px;
  padding: 0 0 0 10px;
  border-left: 2px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.toc__sublink {
  display: block;
  padding: 6px 10px;
  border-radius: var(--r-xs);
  color: var(--ink-muted);
  text-decoration: none;
  font-size: 13.5px;
  font-weight: 500;
  line-height: 1.35;
}
.toc__sublink[aria-current='page'] {
  background: var(--pistachio);
  color: var(--pistachio-body);
  font-weight: 700;
}
@media (hover: hover) {
  .toc__sublink:not([aria-current='page']):hover {
    color: var(--ink);
    background: var(--surface);
  }
}

/* ---------- Main column & Prose ---------- */

.main {
  min-width: 0;
  padding: 18px calc(16px + env(safe-area-inset-right)) calc(56px + env(safe-area-inset-bottom))
    calc(16px + env(safe-area-inset-left));
}
@media (min-width: 640px) {
  .main { padding: 24px 28px 80px; }
}
@media (min-width: 1024px) {
  .main { padding: 24px 36px 88px; }
}

.prose {
  max-width: var(--measure);
  margin: 0 auto;
  min-width: 0;
}
@media (min-width: 1024px) {
  .prose, .pager { max-width: 720px; }
}

.prose h2 {
  font-family: var(--font-serif);
  font-weight: 400;
  font-size: 24px;
  line-height: 1.2;
  color: var(--ink);
  margin: 30px 0 12px;
  scroll-margin-top: calc(var(--topbar-h) + 20px);
}
@media (min-width: 1024px) {
  .prose h2 { font-size: 27px; }
}

.prose h3 {
  font-family: var(--font-sans);
  font-size: 17.5px;
  font-weight: 700;
  line-height: 1.3;
  color: var(--ink);
  margin: 24px 0 8px;
  scroll-margin-top: calc(var(--topbar-h) + 20px);
}

.prose p { margin: 0 0 15px; }
.prose ul, .prose ol { margin: 0 0 16px; }
.prose li + li { margin-top: 5px; }

.prose blockquote {
  border-left: 3px solid var(--cola);
  padding: 4px 0 4px 16px;
  color: var(--ink-muted);
  margin: 0 0 16px;
}

.prose hr {
  border: 0;
  border-top: 1px solid var(--border);
  margin: 24px 0;
}

:not(pre) > code {
  font-family: var(--font-mono);
  font-size: 0.86em;
  font-weight: 500;
  background: var(--butter);
  color: var(--code-ink);
  border-radius: var(--r-xs);
  padding: 0.14em 0.42em;
  overflow-wrap: anywhere;
}

.entry-figure {
  margin: 18px 0;
  background: var(--surface);
  border-radius: var(--r-md);
  padding: 10px;
}
.entry-figure img {
  border-radius: var(--r-sm);
  width: 100%;
}
.entry-figure figcaption {
  font-size: 13px;
  color: var(--ink-muted);
  margin-top: 8px;
  padding: 0 4px;
}

/* ---------- On this page rail ---------- */

.onpage { display: none; }
@media (min-width: 1280px) {
  .onpage {
    display: block;
    position: sticky;
    top: calc(var(--topbar-h) + 24px);
    align-self: start;
    max-height: calc(100vh - var(--topbar-h) - 48px);
    overflow: auto;
    padding: 0 14px;
    font-size: 13px;
  }
  .onpage__title {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-muted);
    margin: 0 0 8px;
  }
  .onpage__list { list-style: none; margin: 0; padding: 0; }
  .onpage__item a {
    display: block;
    padding: 6px 10px;
    border-radius: 10px;
    font-weight: 500;
    line-height: 1.35;
    color: var(--ink-muted);
    text-decoration: none;
  }
  .onpage__item--h3 a { padding-left: 20px; }
  .onpage__item a:hover { color: var(--ink); }
  .onpage__item a.is-active {
    background: var(--cola-soft);
    color: var(--cola-title);
    font-weight: 700;
  }
}

/* ---------- Hero / Chapter Heads ---------- */

.home-head,
.chapter-head {
  background: var(--cola-soft);
  color: var(--cola-body);
  border-radius: var(--r-2xl);
  padding: 22px 20px;
  margin: 0 0 24px;
  position: relative;
  overflow: hidden;
}
.home-head::before,
.chapter-head::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--cola);
}
@media (min-width: 820px) {
  .home-head, .chapter-head { padding: 30px 34px; }
}

.home-head__eyebrow,
.chapter-head__eyebrow {
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--cola-ink);
  margin: 0;
}

.home-head__title,
.chapter-head__title {
  font-family: var(--font-serif);
  font-weight: 400;
  font-size: 30px;
  line-height: 1.12;
  color: var(--cola-title);
  text-wrap: balance;
  margin: 8px 0 10px;
}
@media (min-width: 820px) {
  .home-head__title, .chapter-head__title { font-size: 40px; }
}

.home-head__lede,
.chapter-head__lede {
  font-size: 15.5px;
  font-weight: 500;
  line-height: 1.5;
  color: var(--cola-body);
  max-width: 60ch;
}

.home-head__stats {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  list-style: none;
  margin: 16px 0 0;
  padding: 0;
}
.home-head__stat {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 30px;
  padding: 0 12px;
  border-radius: var(--r-pill);
  background: var(--surface);
  color: var(--ink);
  font-size: 12.5px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

/* Status strip (running now / blockers / next) inside hero */
.status-strip {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 10px;
  margin-top: 18px;
}
@media (min-width: 680px) {
  .status-strip { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
.strip-cell {
  background: var(--surface);
  border-radius: var(--r-md);
  padding: 12px 14px;
  min-width: 0;
}
.strip-cell__k {
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--cola-ink);
  margin-bottom: 4px;
}
.strip-cell__v {
  font-size: 13.5px;
  font-weight: 600;
  line-height: 1.4;
  color: var(--ink);
  overflow-wrap: anywhere;
}

/* ---------- Cards & Grids ---------- */

.card-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 12px;
  margin: 14px 0 24px;
}
@media (min-width: 700px) {
  .card-grid--2 { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
}

.card {
  background: var(--surface);
  border-radius: var(--r-lg);
  padding: 16px 18px;
  min-width: 0;
}
.card--sky { background: var(--sky); color: var(--sky-body, var(--ink)); }
.card--pistachio { background: var(--pistachio); color: var(--pistachio-body); }
.card--butter { background: var(--butter); color: var(--butter-body); }
.card--strawberry { background: var(--strawberry); color: var(--strawberry-body); }

.card__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.card__title {
  font-family: var(--font-sans);
  font-size: 15.5px;
  font-weight: 700;
  color: var(--ink);
  margin: 0;
}
.card__sub {
  font-size: 13px;
  color: var(--ink-muted);
  margin-top: 2px;
}

.entry-card {
  display: block;
  background: var(--surface);
  border-radius: var(--r-lg);
  padding: 18px 20px;
  text-decoration: none;
  color: var(--ink);
  transition: transform 120ms ease;
}
@media (hover: hover) {
  .entry-card:hover { transform: translateY(-1px); text-decoration: none; }
}
.entry-card__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--ink-muted);
  margin-bottom: 6px;
}
.entry-card__title {
  font-family: var(--font-serif);
  font-weight: 400;
  font-size: 21px;
  line-height: 1.2;
  color: var(--ink);
  margin-bottom: 8px;
}
.entry-card__summary {
  font-size: 14.5px;
  line-height: 1.5;
  color: var(--ink-muted);
}

.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}
.tag-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: var(--r-pill);
  background: var(--bg-subtle);
  color: var(--ink-muted);
  font-size: 12px;
  font-weight: 700;
  line-height: 1.3;
  white-space: nowrap;
}
.tag-pill--ok { background: var(--pistachio); color: var(--pistachio-ink); }
.tag-pill--warn { background: var(--butter); color: var(--butter-ink); }
.tag-pill--bad { background: var(--strawberry); color: var(--strawberry-ink); }
.tag-pill--cola { background: var(--cola-soft); color: var(--cola-ink); }
.tag-pill--sky { background: var(--sky); color: var(--sky-ink); }

.btn-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 34px;
  padding: 0 13px;
  border: 0;
  border-radius: var(--r-pill);
  background: var(--surface);
  color: var(--ink);
  font-size: 12.5px;
  font-weight: 700;
  cursor: pointer;
  text-decoration: none;
}
.btn-pill[aria-pressed='true'] {
  background: var(--cola);
  color: #ffffff;
}

/* ---------- Live Status Panel ---------- */

.section-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  margin: 28px 0 12px;
}
.section-bar h2 { margin: 0; }

.live-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.live-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--r-sm);
  background: var(--bg-subtle);
  font-size: 13.5px;
  min-width: 0;
}
.live-row__main {
  min-width: 0;
  flex: 1 1 auto;
  overflow-wrap: anywhere;
}
.live-row__title { font-weight: 700; color: var(--ink); }
.live-row__meta {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--ink-muted);
  margin-top: 2px;
  overflow-wrap: anywhere;
}

.gpu-bar {
  height: 7px;
  border-radius: 4px;
  background: var(--bg);
  overflow: hidden;
  margin: 8px 0 4px;
}
.gpu-bar__fill {
  display: block;
  height: 100%;
  border-radius: 4px;
  background: var(--cola);
}

details.report-box {
  margin-top: 6px;
  background: var(--surface);
  border-radius: var(--r-xs);
  padding: 6px 10px;
  font-size: 12.5px;
}
details.report-box summary {
  cursor: pointer;
  font-weight: 700;
  color: var(--cola-ink);
}
details.report-box pre {
  margin: 8px 0 2px;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--font-mono);
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--ink-muted);
}

/* ---------- Tables ---------- */

.table-wrap {
  max-width: 100%;
  overflow-x: auto;
  border-radius: var(--r-sm);
  background: var(--surface);
  margin: 0 0 20px;
  -webkit-overflow-scrolling: touch;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13.5px;
  line-height: 1.45;
}
th {
  text-align: left;
  font-size: 12px;
  font-weight: 800;
  color: var(--ink);
  background: var(--surface-alt);
  white-space: nowrap;
}
th, td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
tr:last-child td { border-bottom: 0; }

#models-table td:first-child { min-width: 9.5rem; }
#models-table td:nth-child(2) { min-width: 8.5rem; }
#models-table td:nth-child(3),
#models-table td:nth-child(4),
#models-table td:nth-child(7) { white-space: nowrap; }
#models-table td:nth-child(3) code { white-space: nowrap; }
#models-table td:nth-child(5) { min-width: 9rem; }
#models-table td:nth-child(6) { min-width: 12rem; }

.sort-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 0;
  background: transparent;
  padding: 0;
  font: inherit;
  font-weight: 800;
  color: inherit;
  cursor: pointer;
}
.sort-btn[data-dir='asc']::after { content: ' ▲'; font-size: 10px; color: var(--cola); }
.sort-btn[data-dir='desc']::after { content: ' ▼'; font-size: 10px; color: var(--cola); }

/* ---------- Code Slab (Gelato dark block + Pygments tokens) ---------- */

pre.code-slab {
  background: var(--code-block-bg);
  color: var(--code-block-fg);
  border-radius: var(--r-md);
  padding: 16px 18px;
  margin: 0 0 18px;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.55;
  overflow-x: auto;
  max-width: 100%;
  -webkit-overflow-scrolling: touch;
}
pre.code-slab code {
  display: block;
  background: none;
  padding: 0;
  font-family: inherit;
  font-size: inherit;
  color: inherit;
  white-space: pre;
}

/* Pygments dark tokens tuned for WCAG AA on #2b2330 / #2a2830 */
.highlight .c, .highlight .c1, .highlight .cm, .highlight .cs, .highlight .ch, .highlight .sd { color: #a3adb8; font-style: italic; }
.highlight .k, .highlight .kd, .highlight .kn, .highlight .kp, .highlight .kr, .highlight .kt, .highlight .ow { color: #ff8fa3; font-weight: 500; }
.highlight .kc, .highlight .nb, .highlight .bp { color: #8be9fd; }
.highlight .nf, .highlight .fm, .highlight .nc { color: #fde68a; }
.highlight .s, .highlight .s1, .highlight .s2, .highlight .sa, .highlight .sb, .highlight .sc, .highlight .dl, .highlight .se, .highlight .si, .highlight .sr { color: #b7f0b1; }
.highlight .mi, .highlight .mf, .highlight .mh, .highlight .mo, .highlight .il { color: #c4b5fd; }
.highlight .o { color: #f6e9ee; }
.highlight .nd { color: #fda4af; }

/* Code tour stop card */
.tour-stop {
  background: var(--surface);
  border-radius: var(--r-lg);
  padding: 18px;
  margin: 0 0 22px;
}
.tour-stop__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.tour-stop__title {
  font-family: var(--font-serif);
  font-weight: 400;
  font-size: 21px;
  color: var(--ink);
  margin: 0;
  scroll-margin-top: calc(var(--topbar-h) + 20px);
}
.tour-stop__loc {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 500;
  background: var(--cola-soft);
  color: var(--cola-ink);
  padding: 3px 10px;
  border-radius: var(--r-pill);
  overflow-wrap: anywhere;
}
.tour-stop__desc {
  font-size: 14.5px;
  line-height: 1.55;
  color: var(--ink);
  margin-bottom: 14px;
}
.tour-stop pre.code-slab { margin-bottom: 0; }

/* ---------- Model Card ---------- */

.model-card {
  background: var(--surface);
  border-radius: var(--r-lg);
  padding: 18px;
  margin-bottom: 14px;
}
.model-card__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 12px;
  margin: 12px 0;
  padding: 10px 12px;
  border-radius: var(--r-sm);
  background: var(--bg-subtle);
  font-size: 13px;
}
@media (min-width: 640px) {
  .model-card__grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
.model-kv__k {
  display: block;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--ink-muted);
}
.model-kv__v {
  font-weight: 600;
  color: var(--ink);
  overflow-wrap: anywhere;
}

/* ---------- Timeline ---------- */

.timeline {
  list-style: none;
  margin: 16px 0;
  padding: 0 0 0 18px;
  border-left: 2px solid var(--border-strong);
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.timeline__item {
  position: relative;
  background: var(--surface);
  border-radius: var(--r-md);
  padding: 14px 16px;
}
.timeline__item::before {
  content: '';
  position: absolute;
  left: -25px;
  top: 18px;
  width: 12px;
  height: 12px;
  border-radius: 999px;
  background: var(--border-strong);
  border: 2px solid var(--bg);
}
.timeline__item--entry::before { background: var(--cola); }
.timeline__item--commit::before { background: var(--sky-ink); }

/* ---------- Pager & Back to top ---------- */

.pager {
  max-width: var(--measure);
  margin: 28px auto 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.pager__link {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  background: var(--surface);
  border-radius: var(--r-md);
  padding: 14px 16px;
  color: var(--ink);
  text-decoration: none;
}
.pager__link--next {
  align-items: flex-end;
  background: var(--contrast-bg);
  color: var(--contrast-fg);
  text-align: right;
}
.pager__dir {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ink-muted);
}
.pager__link--next .pager__dir { color: var(--pager-next-label); }
.pager__label { font-size: 14px; font-weight: 700; line-height: 1.3; }

.back-to-top {
  position: fixed;
  right: 16px;
  bottom: 18px;
  z-index: 35;
  width: 44px;
  height: 44px;
  border: 0;
  border-radius: var(--r-pill);
  background: var(--surface);
  color: var(--ink);
  box-shadow: var(--shadow);
  display: inline-grid;
  place-items: center;
  cursor: pointer;
}
"""

SITE_JS = r"""(function () {
  var root = document.documentElement;
  function setTheme(theme) {
    root.dataset.theme = theme;
    try { localStorage.setItem('freestyle-devlog:theme', theme); } catch (e) {}
  }
  var tBtn = document.getElementById('theme-toggle');
  if (tBtn) {
    tBtn.addEventListener('click', function () {
      setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');
    });
  }

  // Mobile drawer
  var panel = document.getElementById('toc-panel');
  var toggle = document.getElementById('toc-toggle');
  var closeBtn = document.getElementById('toc-close');
  var backdrop = document.getElementById('toc-backdrop');
  function openDrawer() {
    if (!panel) return;
    panel.dataset.state = 'open';
    if (toggle) toggle.setAttribute('aria-expanded', 'true');
    if (backdrop) {
      backdrop.hidden = false;
      requestAnimationFrame(function () { backdrop.classList.add('is-visible'); });
    }
    document.body.classList.add('scroll-locked');
  }
  function closeDrawer() {
    if (!panel) return;
    panel.dataset.state = 'closed';
    if (toggle) toggle.setAttribute('aria-expanded', 'false');
    if (backdrop) {
      backdrop.classList.remove('is-visible');
      setTimeout(function () { backdrop.hidden = true; }, 180);
    }
    document.body.classList.remove('scroll-locked');
  }
  if (toggle) toggle.addEventListener('click', openDrawer);
  if (closeBtn) closeBtn.addEventListener('click', closeDrawer);
  if (backdrop) backdrop.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && panel && panel.dataset.state === 'open') closeDrawer();
  });

  // Back to top
  var btt = document.getElementById('back-to-top');
  if (btt) {
    window.addEventListener('scroll', function () {
      btt.hidden = window.scrollY < 420;
    }, { passive: true });
    btt.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // On-page heading spy
  var onpageLinks = Array.from(document.querySelectorAll('[data-onpage-link]'));
  if (onpageLinks.length && 'IntersectionObserver' in window) {
    var map = new Map();
    onpageLinks.forEach(function (a) { map.set(a.dataset.onpageLink, a); });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (ent) {
        if (ent.isIntersecting) {
          onpageLinks.forEach(function (a) { a.classList.remove('is-active'); });
          var active = map.get(ent.target.id);
          if (active) active.classList.add('is-active');
        }
      });
    }, { rootMargin: '-15% 0px -70% 0px' });
    onpageLinks.forEach(function (a) {
      var target = document.getElementById(a.dataset.onpageLink);
      if (target) obs.observe(target);
    });
  }
})();
"""


def _pill(text: str, tone: str = "neutral") -> str:
    cls = {
        "ok": "tag-pill tag-pill--ok",
        "warn": "tag-pill tag-pill--warn",
        "bad": "tag-pill tag-pill--bad",
        "cola": "tag-pill tag-pill--cola",
        "sky": "tag-pill tag-pill--sky",
    }.get(tone, "tag-pill")
    return f'<span class="{cls}">{html.escape(text)}</span>'


def _state_tone(state: str) -> str:
    s = state.lower()
    if any(k in s for k in ("running", "active", "pending")):
        return "warn"
    if any(k in s for k in ("done", "complete", "succeeded", "ok")):
        return "ok"
    if any(k in s for k in ("dead", "fail", "error", "backoff", "unavailable")):
        return "bad"
    return "neutral"


def render_shell(
    *,
    title: str,
    description: str,
    section_label: str,
    active_nav: str,
    root_prefix: str,
    entries: list[Entry],
    on_page: list[dict[str, Any]],
    main_html: str,
    status_badge: tuple[str, str] = ("live", "ok"),
    extra_scripts: str = "",
    pager: tuple[dict[str, str] | None, dict[str, str] | None] = (None, None),
) -> str:
    full_title = f"{title} · Freestyle Dev Log" if title != "Home" else "Freestyle Dev Log"
    nav_items = [
        ("home", "Overview & Live", f"{root_prefix}index.html", "home"),
        ("models", "Model Catalogue", f"{root_prefix}models.html", "cpu"),
        ("code", "Code Tour", f"{root_prefix}code.html", "code"),
        ("timeline", "Timeline", f"{root_prefix}timeline.html", "clock"),
    ]
    nav_html = "".join(
        f'<li><a class="toc__link" href="{href}"'
        + (' aria-current="page"' if active_nav == key else "")
        + f'><span class="toc__badge">{icon_svg(ic, 15)}</span>'
        f"<span>{html.escape(label)}</span></a></li>"
        for key, label, href, ic in nav_items
    )

    entries_links = "".join(
        f'<li><a class="toc__sublink" href="{root_prefix}entries/{e.slug}.html"'
        + (' aria-current="page"' if active_nav == f"entry:{e.slug}" else "")
        + f">{html.escape(e.title)}</a></li>"
        for e in entries
    )

    onpage_html = ""
    if on_page:
        items_html = "".join(
            f'<li class="onpage__item onpage__item--h{h["depth"]}">'
            f'<a href="#{html.escape(h["slug"])}" data-onpage-link="{html.escape(h["slug"])}">'
            f"{html.escape(h['text'])}</a></li>"
            for h in on_page
        )
        onpage_html = (
            '<aside class="onpage" aria-label="On this page">'
            '<p class="onpage__title">On this page</p>'
            f'<ul class="onpage__list">{items_html}</ul>'
            "</aside>"
        )

    prev_link, next_link = pager
    pager_html = ""
    if prev_link or next_link:
        prev_block = (
            f'<a class="pager__link pager__link--prev" href="{prev_link["href"]}" rel="prev">'
            f'<span class="pager__dir">{icon_svg("arrow-left", 13)} Previous</span>'
            f'<span class="pager__label">{html.escape(prev_link["label"])}</span></a>'
            if prev_link
            else "<span></span>"
        )
        next_block = (
            f'<a class="pager__link pager__link--next" href="{next_link["href"]}" rel="next">'
            f'<span class="pager__dir">Next {icon_svg("arrow-right", 13)}</span>'
            f'<span class="pager__label">{html.escape(next_link["label"])}</span></a>'
            if next_link
            else "<span></span>"
        )
        pager_html = f'<nav class="pager" aria-label="Page navigation">{prev_block}{next_block}</nav>'

    badge_text, badge_tone = status_badge
    badge_cls = "live-chip" if badge_tone == "ok" else f"live-chip live-chip--{badge_tone}"

    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>{html.escape(full_title)}</title>
  <meta name="description" content="{html.escape(description)}" />
  <meta name="color-scheme" content="light dark" />
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='15' fill='%23c8102e'/%3E%3Cpath d='M11 9h11v3.5h-7v3h5.5v3.3H15V24h-4V9z' fill='%23fff8f1'/%3E%3C/svg%3E" type="image/svg+xml" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&family=Young+Serif&display=swap" />
  <link rel="stylesheet" href="{root_prefix}assets/style.css" />
  <script>
    (function () {{
      var t;
      try {{ t = localStorage.getItem('freestyle-devlog:theme'); }} catch (e) {{}}
      if (t !== 'light' && t !== 'dark') t = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      document.documentElement.dataset.theme = t;
    }})();
  </script>
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="topbar">
    <div class="topbar__pill">
      <button id="toc-toggle" class="icon-btn topbar__menu" type="button" aria-controls="toc-panel" aria-expanded="false" aria-label="Open navigation">
        <span class="hamburger" aria-hidden="true"><span></span><span></span><span></span></span>
      </button>
      <a class="topbar__title" href="{root_prefix}index.html">
        <span class="brand-dot" aria-hidden="true"></span>
        <span>Freestyle</span>
        <span class="topbar__title-rest">Dev Log</span>
      </a>
      <span class="topbar__section">{html.escape(section_label)}</span>
      <div class="topbar__actions">
        <a id="topbar-live-pill" class="{badge_cls}" href="{root_prefix}index.html#live-status">
          <span class="pulse-dot" aria-hidden="true"></span>
          <span id="topbar-live-text">{html.escape(badge_text)}</span>
        </a>
        <button id="theme-toggle" class="icon-btn topbar__theme" type="button" aria-label="Switch color theme">
          {icon_svg("moon", 18, "theme-icon theme-icon--moon")}
          {icon_svg("sun", 18, "theme-icon theme-icon--sun")}
        </button>
      </div>
    </div>
  </header>

  <div class="shell">
    <aside id="toc-panel" class="toc-panel" aria-label="Navigation" data-state="closed">
      <div class="toc-panel__head">
        <span class="toc-panel__title">Freestyle Log</span>
        <button id="toc-close" class="icon-btn toc-panel__close" type="button" aria-label="Close navigation">
          {icon_svg("x", 18)}
        </button>
      </div>
      <nav aria-label="Primary">
        <div class="toc-group-label">Sections</div>
        <ul class="toc__list">{nav_html}</ul>
        <div class="toc-group-label">Log Entries ({len(entries)})</div>
        <ul class="toc__sublist">{entries_links}</ul>
      </nav>
    </aside>
    <div id="toc-backdrop" class="toc-backdrop" hidden></div>

    <main id="main" class="main" tabindex="-1">
      <article class="prose">
        {main_html}
      </article>
      {pager_html}
    </main>

    {onpage_html}
  </div>

  <button id="back-to-top" class="back-to-top" type="button" aria-label="Back to top" hidden>
    {icon_svg("up", 18)}
  </button>

  <script src="{root_prefix}assets/app.js"></script>
  {extra_scripts}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------


def _render_status_fragments(status: dict[str, Any]) -> dict[str, Any]:
    """Compute initial HTML & summary values for the live status panel."""
    pi = status.get("pi_workers", {})
    ext = status.get("external_agents", {})
    k8s = status.get("k8s", {})
    gpus = status.get("gpus", {})
    git = status.get("git", {})
    tests = status.get("tests", {})
    exps = status.get("experiments", {})

    running_workers = [w["slug"] for w in pi.get("workers", []) if w.get("state") == "running"]
    active_jobs = [j["name"] for j in k8s.get("jobs", []) if j.get("phase") == "Running"]
    running_parts = []
    if running_workers:
        running_parts.append(f"pi: {', '.join(running_workers)}")
    if active_jobs:
        running_parts.append(f"k8s: {', '.join(active_jobs)}")
    running_now = " · ".join(running_parts) if running_parts else "Idle (workspace pods up)"

    blockers = []
    dead_workers = [w["slug"] for w in pi.get("workers", []) if w.get("state") == "dead"]
    if dead_workers:
        blockers.append(f"Dead pi worker(s): {', '.join(dead_workers)}")
    if tests.get("ok") is False and tests.get("status") == "failed":
        blockers.append("Unit tests failing")
    for card in gpus.get("cards", []):
        if card.get("memory_free_mib", 99999) < 1500:
            blockers.append(f"{card['node']} low free VRAM ({card['memory_free_mib']} MiB)")
    blocker_text = "; ".join(blockers) if blockers else "No hard blockers; GT verification in progress"

    next_text = "Verify test/val GT in review app -> run A/B/C/D/E/F experiment sweep"
    if not exps.get("metrics_runs"):
        next_text = "Verify test/val GT in review app, then run baseline A/B/C experiments"

    # GPUs HTML
    if gpus.get("ok") and gpus.get("cards"):
        gpu_rows = []
        for c in gpus["cards"]:
            total = max(1, int(c.get("memory_total_mib", 1)))
            used = int(c.get("memory_used_mib", 0))
            free = int(c.get("memory_free_mib", 0))
            pct = min(100, round((used / total) * 100))
            gpu_rows.append(
                f'<li class="live-row"><div class="live-row__main">'
                f'<div class="live-row__title">{html.escape(c["name"])} · {html.escape(c["node"])}</div>'
                f'<div class="gpu-bar"><span class="gpu-bar__fill" style="width:{pct}%"></span></div>'
                f'<div class="live-row__meta">{used:,} / {total:,} MiB used ({free:,} MiB free · util {c.get("utilization_pct", 0)}%)</div>'
                f"</div>{_pill(f'{pct}%', 'warn' if pct > 90 else 'ok')}</li>"
            )
        gpus_html = f'<ul class="live-list">{"".join(gpu_rows)}</ul>'
    else:
        gpus_html = f'<p class="card__sub">Unavailable: {html.escape(gpus.get("error", "no data"))}</p>'

    # Pi workers HTML
    if pi.get("ok"):
        w_rows = []
        for w in pi.get("workers", []):
            rep = ""
            if w.get("report_block"):
                rep = (
                    '<details class="report-box"><summary>Last report</summary>'
                    f"<pre>{html.escape(w['report_block'])}</pre></details>"
                )
            w_rows.append(
                f'<li class="live-row"><div class="live-row__main">'
                f'<div class="live-row__title">{html.escape(w["slug"])}</div>'
                f'<div class="live-row__meta">age {html.escape(w["age"])} · idle {html.escape(w["idle"])} · sess {html.escape(w["session"])}</div>'
                f"{rep}</div>{_pill(w['state'], _state_tone(w['state']))}</li>"
            )
        pi_html = f'<ul class="live-list">{"".join(w_rows)}</ul>' if w_rows else '<p class="card__sub">No workers recorded.</p>'
    else:
        pi_html = f'<p class="card__sub">Unavailable: {html.escape(pi.get("error", "no data"))}</p>'

    # External agents HTML
    if ext.get("ok"):
        e_rows = []
        for a in ext.get("latest_by_agent", []):
            active_obj = a.get("latest_nonempty") or a
            e_rows.append(
                f'<li class="live-row"><div class="live-row__main">'
                f'<div class="live-row__title">{html.escape(a["agent"])} · <code>{html.escape(active_obj["file"])}</code></div>'
                f'<div class="live-row__meta">{html.escape(active_obj["status_line"])}</div>'
                f"</div>{_pill(active_obj['mtime'][:16].replace('T', ' '), 'sky')}</li>"
            )
        for r in ext.get("recent", [])[:5]:
            if r["size_bytes"] > 0:
                e_rows.append(
                    f'<li class="live-row"><div class="live-row__main">'
                    f'<div class="live-row__title"><code>{html.escape(r["file"])}</code></div>'
                    f'<div class="live-row__meta">{html.escape(r["status_line"])}</div>'
                    f"</div>{_pill(r['agent'], 'neutral')}</li>"
                )
        ext_html = f'<ul class="live-list">{"".join(e_rows)}</ul>' if e_rows else '<p class="card__sub">No external agent logs.</p>'
    else:
        ext_html = f'<p class="card__sub">Unavailable: {html.escape(ext.get("error", "no data"))}</p>'

    # K8s HTML
    if k8s.get("ok"):
        k_rows = []
        for j in k8s.get("jobs", []):
            k_rows.append(
                f'<li class="live-row"><div class="live-row__main">'
                f'<div class="live-row__title">Job: {html.escape(j["name"])}</div>'
                f'<div class="live-row__meta">gpu {html.escape(str(j["gpu"]))} · duration {html.escape(j["duration"])}</div>'
                f"</div>{_pill(j['phase'], _state_tone(j['phase']))}</li>"
            )
        for p in k8s.get("pods", []):
            k_rows.append(
                f'<li class="live-row"><div class="live-row__main">'
                f'<div class="live-row__title">Pod: {html.escape(p["name"])}</div>'
                f'<div class="live-row__meta">node {html.escape(p["node"])} · gpu {html.escape(str(p["gpu"]))} · {html.escape(p["duration"])}</div>'
                f"</div>{_pill(p['phase'], _state_tone(p['phase']))}</li>"
            )
        k8s_html = f'<ul class="live-list">{"".join(k_rows)}</ul>' if k_rows else '<p class="card__sub">No vision-intern jobs or pods.</p>'
    else:
        k8s_html = f'<p class="card__sub">Unavailable: {html.escape(k8s.get("error", "no data"))}</p>'

    # Experiments / Smoke HTML
    exp_rows: list[str] = []
    for m in exps.get("metrics_runs", []):
        if "error" in m:
            exp_rows.append(f'<li class="live-row"><div class="live-row__main">{html.escape(m["id"])}: error</div></li>')
        else:
            ap50 = f"{m['ap50']:.3f}" if isinstance(m.get("ap50"), (int, float)) else "-"
            top1 = f"{m['top1_acc']:.3f}" if isinstance(m.get("top1_acc"), (int, float)) else "-"
            mae = f"{m['total_count_mae']:.2f}" if isinstance(m.get("total_count_mae"), (int, float)) else "-"
            exp_rows.append(
                f'<li class="live-row"><div class="live-row__main">'
                f'<div class="live-row__title">{html.escape(m["id"])}</div>'
                f'<div class="live-row__meta">AP50 {ap50} · Top-1 {top1} · Count MAE {mae}</div>'
                f"</div>{_pill('metrics', 'ok')}</li>"
            )
    for s in exps.get("smoke_runs", []):
        exp_rows.append(
            f'<li class="live-row"><div class="live-row__main">'
            f'<div class="live-row__title">{html.escape(s["job"])}</div>'
            f'<div class="live-row__meta">{html.escape(str(s.get("gpu", "")))} · fp16 {s.get("fp16_tflops")} TFLOPS · bf16 {s.get("bf16_tflops")} TFLOPS</div>'
            f"</div>{_pill('smoke', 'ok')}</li>"
        )
    probe = exps.get("collective_probe")
    if probe:
        exp_rows.append(
            f'<li class="live-row"><div class="live-row__main">'
            f'<div class="live-row__title">Collective probe</div>'
            f'<div class="live-row__meta">{probe["tiny_ok"]}/{probe["tiny_total"]} tiny probes ok · {probe["detect_runs"]} detect · {probe["identify_runs"]} identify</div>'
            f"</div>{_pill('probe', 'sky')}</li>"
        )
    exp_html = f'<ul class="live-list">{"".join(exp_rows)}</ul>' if exp_rows else '<p class="card__sub">No experiment metrics yet.</p>'

    # Git + Tests HTML
    git_commits = git.get("oneline", [])[:10]
    git_rows = "".join(f"<li><code>{html.escape(c[:7])}</code> {html.escape(c[8:])}</li>" for c in git_commits)
    test_summary = tests.get("summary", "unavailable")
    test_badge = (
        f"{tests.get('passed', 0)} passed"
        if tests.get("ok")
        else (f"failed ({tests.get('failed', 0)})" if tests.get("status") == "failed" else "unavailable")
    )

    return {
        "running_now": running_now,
        "blockers": blocker_text,
        "next": next_text,
        "gpus_html": gpus_html,
        "pi_html": pi_html,
        "ext_html": ext_html,
        "k8s_html": k8s_html,
        "exp_html": exp_html,
        "git_rows": git_rows,
        "git_branch": git.get("branch", "freestyle-cartridges"),
        "git_sha": git.get("short_sha", "-"),
        "git_dirty": (git.get("dirty") or {}).get("total", 0),
        "test_badge": test_badge,
        "test_summary": test_summary,
        "test_ok": tests.get("ok"),
        "running_count": len(running_workers) + len(active_jobs),
    }


def build_index(entries: list[Entry], models: list[dict[str, Any]], code_tour: dict[str, Any], status: dict[str, Any]) -> str:
    frags = _render_status_fragments(status)
    live_text = f"{frags['running_count']} running" if frags["running_count"] else frags["test_badge"]
    badge_tone = "warn" if frags["running_count"] else ("ok" if frags["test_ok"] else "bad")

    entries_cards = "".join(
        f'<a class="entry-card" href="entries/{e.slug}.html">'
        f'<div class="entry-card__meta"><span>{html.escape(e.display_date)}</span><span>·</span><span>{e.reading_min} min read</span></div>'
        f'<h3 class="entry-card__title">{html.escape(e.title)}</h3>'
        f'<p class="entry-card__summary">{html.escape(e.summary[:240])}{"…" if len(e.summary) > 240 else ""}</p>'
        f'<div class="pill-row">{"".join(_pill(t, "cola") for t in e.tags)}</div>'
        f"</a>"
        for e in entries
    )

    main_html = f"""
<header class="home-head">
  <p class="home-head__eyebrow">Coca-Cola Freestyle · Computer Vision Project Log</p>
  <h1 class="home-head__title">Freestyle Cartridge Recognition</h1>
  <p class="home-head__lede">Live progress from the workstation &amp; home GPU cluster: what we tried, model benchmarks, Python pipeline internals, and real-time worker/job telemetry.</p>
  <ul class="home-head__stats">
    <li class="home-head__stat">{len(entries)} entries</li>
    <li class="home-head__stat">{len(models)} models</li>
    <li class="home-head__stat">{len(code_tour.get('stops', []))} code stops</li>
    <li class="home-head__stat" id="hero-tests-stat">tests: {html.escape(frags['test_badge'])}</li>
  </ul>
  <div class="status-strip" aria-label="Project pulse">
    <div class="strip-cell">
      <div class="strip-cell__k">Running now</div>
      <div class="strip-cell__v" id="strip-running">{html.escape(frags['running_now'])}</div>
    </div>
    <div class="strip-cell">
      <div class="strip-cell__k">Blockers / Watch</div>
      <div class="strip-cell__v" id="strip-blockers">{html.escape(frags['blockers'])}</div>
    </div>
    <div class="strip-cell">
      <div class="strip-cell__k">Next up</div>
      <div class="strip-cell__v" id="strip-next">{html.escape(frags['next'])}</div>
    </div>
  </div>
</header>

<h2 id="latest-entries">Latest Entries</h2>
<div class="card-grid">
  {entries_cards}
</div>

<div class="section-bar">
  <h2 id="live-status">Live Cluster &amp; Worker Status</h2>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <span class="tag-pill" id="status-updated">Updated: {html.escape(status.get('local_timestamp', '-'))}</span>
    <button type="button" class="btn-pill" id="refresh-status-btn">{icon_svg("refresh", 14)} Refresh</button>
  </div>
</div>

<div class="card-grid card-grid--2">
  <section class="card" aria-labelledby="card-gpus">
    <div class="card__head">
      <h3 id="card-gpus" class="card__title">Cluster GPUs</h3>
      <span class="tag-pill">nvidia-smi</span>
    </div>
    <div id="live-gpus">{frags['gpus_html']}</div>
  </section>

  <section class="card" aria-labelledby="card-k8s">
    <div class="card__head">
      <h3 id="card-k8s" class="card__title">Kubernetes Jobs &amp; Pods</h3>
      <span class="tag-pill">admin@home</span>
    </div>
    <div id="live-k8s">{frags['k8s_html']}</div>
  </section>

  <section class="card" aria-labelledby="card-pi">
    <div class="card__head">
      <h3 id="card-pi" class="card__title">pi-orch Workers</h3>
      <span class="tag-pill">tmux</span>
    </div>
    <div id="live-pi">{frags['pi_html']}</div>
  </section>

  <section class="card" aria-labelledby="card-ext">
    <div class="card__head">
      <h3 id="card-ext" class="card__title">External Agents</h3>
      <span class="tag-pill">ext/*.log</span>
    </div>
    <div id="live-ext">{frags['ext_html']}</div>
  </section>
</div>

<div class="card-grid card-grid--2">
  <section class="card" aria-labelledby="card-exps">
    <div class="card__head">
      <h3 id="card-exps" class="card__title">Experiments &amp; Probes</h3>
      <span class="tag-pill">runs/</span>
    </div>
    <div id="live-exps">{frags['exp_html']}</div>
  </section>

  <section class="card" aria-labelledby="card-git">
    <div class="card__head">
      <h3 id="card-git" class="card__title">Git &amp; Unit Tests</h3>
      <span class="tag-pill" id="live-git-meta">{html.escape(frags['git_branch'])} @ {html.escape(frags['git_sha'])} ({frags['git_dirty']} dirty)</span>
    </div>
    <p class="card__sub" id="live-test-summary" style="margin-bottom:10px;">pytest: {html.escape(frags['test_summary'])}</p>
    <ul id="live-git-list" style="margin:0;padding-left:1.1em;font-size:13px;">{frags['git_rows']}</ul>
  </section>
</div>

<h2 id="explore">Explore Reference</h2>
<div class="card-grid card-grid--2">
  <a class="entry-card" href="models.html">
    <div class="entry-card__meta"><span>{len(models)} models compared</span></div>
    <h3 class="entry-card__title">Model Catalogue</h3>
    <p class="entry-card__summary">Parameter counts, release dates, licences, input resolutions, and why people reach for each detector, embedder, VLM, and OCR model.</p>
  </a>
  <a class="entry-card" href="code.html">
    <div class="entry-card__meta"><span>{len(code_tour.get('stops', []))} AST-extracted stops</span></div>
    <h3 class="entry-card__title">Python Code Tour</h3>
    <p class="entry-card__summary">Walk through the actual tiling, VLM labelling/judging, kNN crop identification, counting metrics, and k8s launcher code.</p>
  </a>
</div>
"""

    on_page = [
        {"depth": 2, "slug": "latest-entries", "text": "Latest Entries"},
        {"depth": 2, "slug": "live-status", "text": "Live Cluster & Worker Status"},
        {"depth": 2, "slug": "explore", "text": "Explore Reference"},
    ]

    refresher_js = r"""<script>
(function () {
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function tone(state) {
    var s = String(state || '').toLowerCase();
    if (/running|active|pending/.test(s)) return 'warn';
    if (/done|complete|succeeded|ok/.test(s)) return 'ok';
    if (/dead|fail|error|backoff|unavailable/.test(s)) return 'bad';
    return 'neutral';
  }
  function pill(txt, t) {
    var cls = t === 'neutral' ? 'tag-pill' : 'tag-pill tag-pill--' + t;
    return '<span class="' + cls + '">' + esc(txt) + '</span>';
  }
  function renderStatus(st) {
    if (!st) return;
    var up = document.getElementById('status-updated');
    if (up) up.textContent = 'Updated: ' + (st.local_timestamp || st.timestamp || '-');

    var pi = st.pi_workers || {};
    var k8s = st.k8s || {};
    var gpus = st.gpus || {};
    var ext = st.external_agents || {};
    var git = st.git || {};
    var tests = st.tests || {};
    var exps = st.experiments || {};

    var rw = (pi.workers || []).filter(function (w) { return w.state === 'running'; }).map(function (w) { return w.slug; });
    var aj = (k8s.jobs || []).filter(function (j) { return j.phase === 'Running'; }).map(function (j) { return j.name; });
    var rParts = [];
    if (rw.length) rParts.push('pi: ' + rw.join(', '));
    if (aj.length) rParts.push('k8s: ' + aj.join(', '));
    var rEl = document.getElementById('strip-running');
    if (rEl) rEl.textContent = rParts.length ? rParts.join(' · ') : 'Idle (workspace pods up)';

    var liveCount = rw.length + aj.length;
    var topPill = document.getElementById('topbar-live-pill');
    var topText = document.getElementById('topbar-live-text');
    var tBadge = tests.ok ? (tests.passed || 0) + ' passed' : (tests.status === 'failed' ? 'tests failed' : 'unavailable');
    if (topText) topText.textContent = liveCount ? (liveCount + ' running') : tBadge;
    if (topPill) topPill.className = 'live-chip' + (liveCount ? ' live-chip--warn' : (tests.ok ? '' : ' live-chip--down'));
    var heroTest = document.getElementById('hero-tests-stat');
    if (heroTest) heroTest.textContent = 'tests: ' + tBadge;

    var gEl = document.getElementById('live-gpus');
    if (gEl) {
      if (gpus.ok && gpus.cards && gpus.cards.length) {
        gEl.innerHTML = '<ul class="live-list">' + gpus.cards.map(function (c) {
          var total = Math.max(1, c.memory_total_mib || 1);
          var used = c.memory_used_mib || 0;
          var pct = Math.min(100, Math.round((used / total) * 100));
          return '<li class="live-row"><div class="live-row__main">'
            + '<div class="live-row__title">' + esc(c.name) + ' · ' + esc(c.node) + '</div>'
            + '<div class="gpu-bar"><span class="gpu-bar__fill" style="width:' + pct + '%"></span></div>'
            + '<div class="live-row__meta">' + used.toLocaleString() + ' / ' + total.toLocaleString() + ' MiB used (' + (c.memory_free_mib || 0).toLocaleString() + ' MiB free · util ' + (c.utilization_pct || 0) + '%)</div>'
            + '</div>' + pill(pct + '%', pct > 90 ? 'warn' : 'ok') + '</li>';
        }).join('') + '</ul>';
      } else {
        gEl.innerHTML = '<p class="card__sub">Unavailable: ' + esc(gpus.error || 'no data') + '</p>';
      }
    }

    var kEl = document.getElementById('live-k8s');
    if (kEl) {
      if (k8s.ok) {
        var rows = (k8s.jobs || []).map(function (j) {
          return '<li class="live-row"><div class="live-row__main"><div class="live-row__title">Job: ' + esc(j.name) + '</div>'
            + '<div class="live-row__meta">gpu ' + esc(j.gpu) + ' · duration ' + esc(j.duration) + '</div></div>'
            + pill(j.phase, tone(j.phase)) + '</li>';
        }).concat((k8s.pods || []).map(function (p) {
          return '<li class="live-row"><div class="live-row__main"><div class="live-row__title">Pod: ' + esc(p.name) + '</div>'
            + '<div class="live-row__meta">node ' + esc(p.node) + ' · gpu ' + esc(p.gpu) + ' · ' + esc(p.duration) + '</div></div>'
            + pill(p.phase, tone(p.phase)) + '</li>';
        }));
        kEl.innerHTML = rows.length ? '<ul class="live-list">' + rows.join('') + '</ul>' : '<p class="card__sub">No vision-intern jobs or pods.</p>';
      } else {
        kEl.innerHTML = '<p class="card__sub">Unavailable: ' + esc(k8s.error || 'no data') + '</p>';
      }
    }

    var pEl = document.getElementById('live-pi');
    if (pEl) {
      if (pi.ok) {
        var wRows = (pi.workers || []).map(function (w) {
          var rep = w.report_block ? '<details class="report-box"><summary>Last report</summary><pre>' + esc(w.report_block) + '</pre></details>' : '';
          return '<li class="live-row"><div class="live-row__main"><div class="live-row__title">' + esc(w.slug) + '</div>'
            + '<div class="live-row__meta">age ' + esc(w.age) + ' · idle ' + esc(w.idle) + ' · sess ' + esc(w.session) + '</div>' + rep + '</div>'
            + pill(w.state, tone(w.state)) + '</li>';
        });
        pEl.innerHTML = wRows.length ? '<ul class="live-list">' + wRows.join('') + '</ul>' : '<p class="card__sub">No workers recorded.</p>';
      } else {
        pEl.innerHTML = '<p class="card__sub">Unavailable: ' + esc(pi.error || 'no data') + '</p>';
      }
    }

    var eEl = document.getElementById('live-ext');
    if (eEl && ext.ok) {
      var eRows = (ext.latest_by_agent || []).map(function (a) {
        var act = a.latest_nonempty || a;
        return '<li class="live-row"><div class="live-row__main"><div class="live-row__title">' + esc(a.agent) + ' · <code>' + esc(act.file) + '</code></div>'
          + '<div class="live-row__meta">' + esc(act.status_line) + '</div></div>'
          + pill((act.mtime || '').slice(0, 16).replace('T', ' '), 'sky') + '</li>';
      });
      eEl.innerHTML = eRows.length ? '<ul class="live-list">' + eRows.join('') + '</ul>' : '<p class="card__sub">No external logs.</p>';
    }

    var gitMeta = document.getElementById('live-git-meta');
    if (gitMeta && git.ok) {
      gitMeta.textContent = (git.branch || '') + ' @ ' + (git.short_sha || '-') + ' (' + ((git.dirty && git.dirty.total) || 0) + ' dirty)';
    }
    var tSum = document.getElementById('live-test-summary');
    if (tSum) tSum.textContent = 'pytest: ' + (tests.summary || 'unavailable');
    var gList = document.getElementById('live-git-list');
    if (gList && git.oneline) {
      gList.innerHTML = git.oneline.slice(0, 10).map(function (c) {
        return '<li><code>' + esc(c.slice(0, 7)) + '</code> ' + esc(c.slice(8)) + '</li>';
      }).join('');
    }
    var exEl = document.getElementById('live-exps');
    if (exEl && exps.ok && exps.metrics_runs && exps.metrics_runs.length) {
      // Keep existing server-rendered list or update metrics runs
    }
  }

  function refresh() {
    fetch('status.json', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { if (d) renderStatus(d); })
      .catch(function () {});
  }
  var btn = document.getElementById('refresh-status-btn');
  if (btn) btn.addEventListener('click', refresh);
  setInterval(refresh, 60000);
})();
</script>"""

    return render_shell(
        title="Home",
        description="Live dev log for the Coca-Cola Freestyle computer vision project.",
        section_label="",
        active_nav="home",
        root_prefix="",
        entries=entries,
        on_page=on_page,
        main_html=main_html,
        status_badge=(live_text, badge_tone),
        extra_scripts=refresher_js,
    )


def build_entry_page(
    entry: Entry,
    idx: int,
    entries: list[Entry],
    status: dict[str, Any],
) -> str:
    frags = _render_status_fragments(status)
    live_text = f"{frags['running_count']} running" if frags["running_count"] else frags["test_badge"]
    badge_tone = "warn" if frags["running_count"] else ("ok" if frags["test_ok"] else "bad")

    tags_html = "".join(_pill(t, "cola") for t in entry.tags)
    main_html = f"""
<header class="chapter-head">
  <p class="chapter-head__eyebrow">Dev Log Entry · {html.escape(entry.display_date)} · {entry.reading_min} min read</p>
  <h1 class="chapter-head__title">{html.escape(entry.title)}</h1>
  <div class="pill-row">{tags_html}</div>
</header>
{entry.body_html}
"""
    prev_e = {"href": f"{entries[idx + 1].slug}.html", "label": entries[idx + 1].title} if idx + 1 < len(entries) else {"href": "../index.html", "label": "Overview"}
    next_e = {"href": f"{entries[idx - 1].slug}.html", "label": entries[idx - 1].title} if idx > 0 else {"href": "../models.html", "label": "Model Catalogue"}

    return render_shell(
        title=entry.title,
        description=entry.summary[:160],
        section_label=entry.title,
        active_nav=f"entry:{entry.slug}",
        root_prefix="../",
        entries=entries,
        on_page=entry.headings,
        main_html=main_html,
        status_badge=(live_text, badge_tone),
        pager=(prev_e, next_e),
    )


def build_models_page(models: list[dict[str, Any]], entries: list[Entry], status: dict[str, Any]) -> str:
    frags = _render_status_fragments(status)
    live_text = f"{frags['running_count']} running" if frags["running_count"] else frags["test_badge"]
    badge_tone = "warn" if frags["running_count"] else ("ok" if frags["test_ok"] else "bad")

    rows_html = []
    cards_html = []
    on_page = [
        {"depth": 2, "slug": "comparison-table", "text": "Sortable Comparison Table"},
        {"depth": 2, "slug": "model-cards", "text": "Detailed Model Cards"},
    ]

    for m in models:
        st = m.get("status", "comparison")
        st_tone = "ok" if st == "active" else ("warn" if st == "planned" else "neutral")
        mid = m.get("id") or slugify(m["name"])
        rows_html.append(
            f'<tr data-status="{html.escape(st)}">'
            f'<td><a href="#{html.escape(mid)}"><strong>{html.escape(m["name"])}</strong></a></td>'
            f'<td>{html.escape(m.get("family", "-"))}</td>'
            f'<td><code>{html.escape(str(m.get("params", "-")))}</code></td>'
            f'<td>{html.escape(str(m.get("release_date", "-")))}</td>'
            f'<td>{html.escape(m.get("licence", "-"))}</td>'
            f'<td>{html.escape(m.get("input_res", "-"))}</td>'
            f'<td>{_pill(st, st_tone)}</td>'
            f"</tr>"
        )
        links = m.get("links") or {}
        link_pills = "".join(
            f'<a class="tag-pill tag-pill--sky" href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(k)} {icon_svg("external", 12)}</a>'
            for k, url in links.items()
            if url
        )
        cards_html.append(
            f'<section class="model-card" id="{html.escape(mid)}" data-status="{html.escape(st)}">'
            f'<div class="card__head">'
            f'<div><h3 class="card__title" style="margin:0;">{html.escape(m["name"])}</h3>'
            f'<div class="card__sub">{html.escape(m.get("org", ""))} · {html.escape(m.get("family", ""))}</div></div>'
            f'{_pill(st, st_tone)}</div>'
            f'<div class="model-card__grid">'
            f'<div><span class="model-kv__k">Parameters</span><span class="model-kv__v">{html.escape(str(m.get("params", "-")))}</span></div>'
            f'<div><span class="model-kv__k">Released</span><span class="model-kv__v">{html.escape(str(m.get("release_date", "-")))}</span></div>'
            f'<div><span class="model-kv__k">Licence</span><span class="model-kv__v">{html.escape(m.get("licence", "-"))}</span></div>'
            f'<div><span class="model-kv__k">Input Res</span><span class="model-kv__v">{html.escape(m.get("input_res", "-"))}</span></div>'
            f'<div style="grid-column:span 2;"><span class="model-kv__k">Pipeline Role</span><span class="model-kv__v">{html.escape(m.get("role", "-"))}</span></div>'
            f"</div>"
            f'<p style="margin-bottom:8px;"><strong>Why people reach for it:</strong> {html.escape(m.get("why", ""))}</p>'
            f'<p style="margin-bottom:10px;color:var(--ink-muted);"><strong>Caveats:</strong> {html.escape(m.get("caveats", ""))}</p>'
            f'<div class="pill-row">{link_pills}</div>'
            f"</section>"
        )

    main_html = f"""
<header class="chapter-head">
  <p class="chapter-head__eyebrow">Verified Catalogue · As of 2026-11-26</p>
  <h1 class="chapter-head__title">Model Catalogue</h1>
  <p class="chapter-head__lede">Every VLM, detector, segmenter, embedding encoder, and OCR model we run or compare in Freestyle, with parameter counts, licences, input resolutions, and trade-offs.</p>
  <div class="pill-row" id="model-filters" style="margin-top:14px;">
    <button type="button" class="btn-pill" data-filter="all" aria-pressed="true">All ({len(models)})</button>
    <button type="button" class="btn-pill" data-filter="active" aria-pressed="false">Active</button>
    <button type="button" class="btn-pill" data-filter="planned" aria-pressed="false">Planned</button>
    <button type="button" class="btn-pill" data-filter="comparison" aria-pressed="false">Comparison</button>
  </div>
</header>

<h2 id="comparison-table">Sortable Comparison Table</h2>
<div class="table-wrap">
  <table id="models-table">
    <thead>
      <tr>
        <th><button type="button" class="sort-btn" data-col="0">Model</button></th>
        <th><button type="button" class="sort-btn" data-col="1">Family</button></th>
        <th><button type="button" class="sort-btn" data-col="2">Params</button></th>
        <th><button type="button" class="sort-btn" data-col="3">Released</button></th>
        <th><button type="button" class="sort-btn" data-col="4">Licence</button></th>
        <th><button type="button" class="sort-btn" data-col="5">Input Res</button></th>
        <th><button type="button" class="sort-btn" data-col="6">Status</button></th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</div>

<h2 id="model-cards">Detailed Model Cards</h2>
{"".join(cards_html)}
"""

    sort_js = r"""<script>
(function () {
  var table = document.getElementById('models-table');
  if (!table) return;
  var tbody = table.querySelector('tbody');
  var sortBtns = Array.from(table.querySelectorAll('.sort-btn'));

  function parseMetric(text, col) {
    var t = text.trim();
    if (col === 2) {
      if (/unknown/i.test(t)) return 1e12;
      var m = t.match(/([\d.]+)\s*([BMK])/i);
      if (m) {
        var n = parseFloat(m[1]);
        var u = m[2].toUpperCase();
        return n * (u === 'B' ? 1e9 : u === 'M' ? 1e6 : 1e3);
      }
    }
    return t.toLowerCase();
  }

  sortBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var col = parseInt(btn.dataset.col, 10);
      var dir = btn.dataset.dir === 'asc' ? 'desc' : 'asc';
      sortBtns.forEach(function (b) { delete b.dataset.dir; });
      btn.dataset.dir = dir;
      var rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort(function (a, b) {
        var va = parseMetric(a.children[col].textContent, col);
        var vb = parseMetric(b.children[col].textContent, col);
        var cmp = (typeof va === 'number' && typeof vb === 'number') ? (va - vb) : String(va).localeCompare(String(vb));
        return dir === 'asc' ? cmp : -cmp;
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
    });
  });

  var filterBtns = Array.from(document.querySelectorAll('#model-filters [data-filter]'));
  filterBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var f = btn.dataset.filter;
      filterBtns.forEach(function (b) { b.setAttribute('aria-pressed', String(b === btn)); });
      document.querySelectorAll('#models-table tbody tr, .model-card').forEach(function (el) {
        el.hidden = (f !== 'all' && el.dataset.status !== f);
      });
    });
  });
})();
</script>"""

    return render_shell(
        title="Model Catalogue",
        description="Model sizes, release dates, licences, input resolutions, and pipeline roles.",
        section_label="Model Catalogue",
        active_nav="models",
        root_prefix="",
        entries=entries,
        on_page=on_page,
        main_html=main_html,
        status_badge=(live_text, badge_tone),
        extra_scripts=sort_js,
        pager=({"href": "index.html", "label": "Overview & Live"}, {"href": "code.html", "label": "Code Tour"}),
    )


def build_code_page(code_tour: dict[str, Any], entries: list[Entry], status: dict[str, Any]) -> str:
    frags = _render_status_fragments(status)
    live_text = f"{frags['running_count']} running" if frags["running_count"] else frags["test_badge"]
    badge_tone = "warn" if frags["running_count"] else ("ok" if frags["test_ok"] else "bad")

    stops = code_tour.get("stops", [])
    on_page = [{"depth": 2, "slug": s["id"], "text": s["title"]} for s in stops]
    stops_html = "".join(
        f'<section class="tour-stop">'
        f'<div class="tour-stop__head">'
        f'<h2 class="tour-stop__title" id="{html.escape(s["id"])}">{idx}. {html.escape(s["title"])}</h2>'
        f'<span class="tour-stop__loc">{html.escape(s["file"])}:{html.escape(s["line_range"])} · {html.escape(s["symbol"])}</span>'
        f"</div>"
        f'<p class="tour-stop__desc">{html.escape(s["explanation"])}</p>'
        f'{s["code_html"]}'
        f"</section>"
        for idx, s in enumerate(stops, 1)
    )

    main_html = f"""
<header class="chapter-head">
  <p class="chapter-head__eyebrow">AST-Extracted · Always Current · Pygments</p>
  <h1 class="chapter-head__title">{html.escape(code_tour.get('title', 'Code Tour'))}</h1>
  <p class="chapter-head__lede">{html.escape(code_tour.get('subtitle', ''))}</p>
</header>
{stops_html}
"""

    return render_shell(
        title="Code Tour",
        description="AST-extracted Python code tour of the Freestyle vision pipeline.",
        section_label="Code Tour",
        active_nav="code",
        root_prefix="",
        entries=entries,
        on_page=on_page,
        main_html=main_html,
        status_badge=(live_text, badge_tone),
        pager=({"href": "models.html", "label": "Model Catalogue"}, {"href": "timeline.html", "label": "Timeline"}),
    )


def build_timeline_page(entries: list[Entry], status: dict[str, Any]) -> str:
    frags = _render_status_fragments(status)
    live_text = f"{frags['running_count']} running" if frags["running_count"] else frags["test_badge"]
    badge_tone = "warn" if frags["running_count"] else ("ok" if frags["test_ok"] else "bad")

    events: list[dict[str, Any]] = []
    for e in entries:
        events.append(
            {
                "kind": "entry",
                "sort_ts": e.sort_ts,
                "date": e.display_date,
                "title": e.title,
                "href": f"entries/{e.slug}.html",
                "meta": ", ".join(e.tags),
                "body": e.summary[:220],
            }
        )

    for c in status.get("git", {}).get("commits", []):
        try:
            dt = datetime.fromisoformat(c["date"])
            ts = dt.timestamp()
            disp = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts = 0.0
            disp = c.get("date", "")
        events.append(
            {
                "kind": "commit",
                "sort_ts": ts,
                "date": disp,
                "title": c.get("subject", ""),
                "sha": c.get("short_sha", ""),
                "author": c.get("author", ""),
            }
        )

    events.sort(key=lambda x: x["sort_ts"], reverse=True)

    items_html = []
    for ev in events:
        if ev["kind"] == "entry":
            items_html.append(
                f'<li class="timeline__item timeline__item--entry">'
                f'<div class="entry-card__meta">{_pill("Entry", "cola")}<span>{html.escape(ev["date"])}</span></div>'
                f'<h3 style="margin:6px 0 4px;font-size:18px;"><a href="{html.escape(ev["href"])}">{html.escape(ev["title"])}</a></h3>'
                f'<p style="margin:0;font-size:14px;color:var(--ink-muted);">{html.escape(ev["body"])}…</p>'
                f"</li>"
            )
        else:
            items_html.append(
                f'<li class="timeline__item timeline__item--commit">'
                f'<div class="entry-card__meta">{_pill(ev["sha"], "sky")}<span>{html.escape(ev["date"])}</span><span>·</span><span>{html.escape(ev["author"])}</span></div>'
                f'<p style="margin:6px 0 0;font-weight:600;font-size:14.5px;">{html.escape(ev["title"])}</p>'
                f"</li>"
            )

    main_html = f"""
<header class="chapter-head">
  <p class="chapter-head__eyebrow">Chronological Stream</p>
  <h1 class="chapter-head__title">Project Timeline</h1>
  <p class="chapter-head__lede">Narrative dev log entries merged chronologically with the latest 25 commits on branch <code>{html.escape(frags['git_branch'])}</code>.</p>
</header>
<ol class="timeline">
  {"".join(items_html)}
</ol>
"""

    return render_shell(
        title="Timeline",
        description="Chronological timeline of dev log entries and git commits.",
        section_label="Timeline",
        active_nav="timeline",
        root_prefix="",
        entries=entries,
        on_page=[],
        main_html=main_html,
        status_badge=(live_text, badge_tone),
        pager=({"href": "code.html", "label": "Code Tour"}, {"href": "index.html", "label": "Overview & Live"}),
    )


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(text)
    os.replace(tmp, path)


def build_site() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    ENTRIES_DIR.mkdir(parents=True, exist_ok=True)

    status = load_status()
    if not STATUS_PATH.exists():
        _write_atomic(STATUS_PATH, json.dumps(status, indent=2) + "\n")

    entries = load_entries()
    models = load_models()
    code_tour = load_code_tour()

    _write_atomic(ASSETS_DIR / "style.css", SITE_CSS)
    _write_atomic(ASSETS_DIR / "app.js", SITE_JS)
    _write_atomic(SITE_DIR / "index.html", build_index(entries, models, code_tour, status))
    _write_atomic(SITE_DIR / "models.html", build_models_page(models, entries, status))
    _write_atomic(SITE_DIR / "code.html", build_code_page(code_tour, entries, status))
    _write_atomic(SITE_DIR / "timeline.html", build_timeline_page(entries, status))

    for idx, entry in enumerate(entries):
        _write_atomic(
            ENTRIES_DIR / f"{entry.slug}.html",
            build_entry_page(entry, idx, entries, status),
        )

    print(
        f"Built {SITE_DIR}: {len(entries)} entries, {len(models)} models, "
        f"{len(code_tour.get('stops', []))} code tour stops"
    )


def main() -> int:
    build_site()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
