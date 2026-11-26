# Freestyle Dev Log (`devlog/`)

Mobile-first static site + live status feed for following the Coca-Cola Freestyle vision project over Tailscale.

## Files

- `devlog/collect.py` — fault-tolerant collector writing `devlog/site/status.json` (pi-orch workers + last reports, external agent logs, `admin@home` k8s jobs/pods, cluster GPU VRAM, git log/diff, cached unit test count, and experiment metrics).
- `devlog/build.py` — static generator building `devlog/site/` (`index.html`, `entries/*.html`, `models.html`, `code.html`, `timeline.html`, and downscaled `assets/img/`).
- `devlog/content/*.md` — narrative entries with YAML front matter (`title`, `date`, `tags`).
- `devlog/models.yaml` — verified model catalogue rendered as cards and a sortable table.
- `devlog/code_tour.yaml` — curated Python symbols extracted at build time via `ast` and highlighted server-side with Pygments.
- `devlog/serve.sh` — serves `devlog/site` on `100.89.241.57:8787` (tailnet IP only).
- `devlog/loop.sh` — runs `collect.py` + `build.py` every 15 minutes, logging to `devlog/site/_loop.log` (designed for a tmux window).

## Adding an entry

Create `devlog/content/YYYY-MM-DD-short-slug.md`:

```markdown
---
title: Your entry title
date: 2026-11-26T16:00:00-05:00
tags: [experiments, findings]
---

Narrative text here. Images under `runs/` referenced via `![alt](runs/...)`
are copied into `devlog/site/assets/img/` and downscaled to ≤1600 px wide.
```

## Collect & rebuild

```bash
uv run python devlog/collect.py
uv run python devlog/build.py
```

Note: `collect.py` runs `uv run pytest tests/unit -q` only when `git HEAD` changes (caching the result in `devlog/site/.pytest_cache.json`).

## Serving on the tailnet

```bash
# Window 1: serve static site (bound only to Tailscale IP)
./devlog/serve.sh

# Window 2: refresh status.json and rebuild every 15 min
./devlog/loop.sh
```
