# Freestyle Dev Log (`devlog/`)

Mobile-first static site + live status feed for following the Coca-Cola Freestyle vision project over Tailscale.

## Files

- `devlog/collect.py` — fault-tolerant collector writing `devlog/site/status.json` (pi-orch workers + last reports, external agent logs, `admin@home` k8s jobs/pods with start/end times and GPU nodes, cluster GPU VRAM, git log/diff, cached unit test count, and experiment metrics excluding chain-check runs).
- `devlog/build.py` — static generator building `devlog/site/` (`index.html`, `entries/index.html`, `entries/*.html`, `models.html`, `code.html`, `timeline.html`, and downscaled `assets/img/`), including milestone progress, experiment scoreboard SVG, k8s jobs Gantt SVG, activity strip, NY local timestamps + relative ages, and `--watch` polling mode (10 s mtime check).
- `devlog/content/*.md` — narrative entries with YAML front matter (`title`, `date`, `tags`).
- `devlog/milestones.yaml` — orchestrator-maintained project milestone statuses (`done`, `active`, `todo`, `blocked`).
- `devlog/models.yaml` — verified model catalogue rendered as cards and a sortable table.
- `devlog/code_tour.yaml` — curated Python symbols extracted at build time via `ast` and highlighted server-side with Pygments.
- `devlog/serve.sh` — serves `devlog/site` on `100.89.241.57:8787` (tailnet IP only).
- `devlog/loop.sh` — runs `build.py --watch` in the background (10 s polling on `devlog/content`, `milestones.yaml`, `models.yaml`, `code_tour.yaml`, `status.json`) alongside a 15-minute `collect.py` + `build.py` refresh loop, logging to `devlog/site/_loop.log`.

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
# Or poll for content/milestone/status changes every 10s:
uv run python devlog/build.py --watch
```

Note: `collect.py` runs `uv run pytest tests/unit -q` only when `git HEAD` changes (caching the result in `devlog/site/.pytest_cache.json`).

## Serving on the tailnet

```bash
# Window 1: serve static site (bound only to Tailscale IP)
./devlog/serve.sh

# Window 2: refresh status.json and rebuild every 15 min
./devlog/loop.sh
```
