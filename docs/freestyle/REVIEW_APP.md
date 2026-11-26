# Local review / ground-truth annotation app

A fast local web tool for verifying and correcting model-proposed cartridge boxes
and identities, producing human-verified ground truth in `data/gt/`. The human
confirms/corrects boxes; they don't draw from scratch.

## Run

```bash
uv run python -m freestyle.review.server \
    --proposals data/labels/<file>.json \
    --out data/gt/<name>.json \
    --images <image-id-list-file | split name | all> \
    --port 8765 \
    --host 127.0.0.1
```

- `--proposals` — COCO json in the contract format (`docs/freestyle/ARCHITECTURE.md`).
  May be empty for an image. Loaded once at startup; never re-read.
- `--out` — COCO json written under `data/gt/` (human-verified, gitignored). If it
  exists, the app **resumes** from it (boxes + image attributes survive a restart).
- `--images` — `all`, a split name (resolved against `data/splits.json`), or a
  **file listing image ids** (one per line, or JSON with a `images`/`image_ids`/
  `file_names` list). Image ids are paths relative to
  `data/raw/freestyle-vision-data/images/`.
- `--host` — `127.0.0.1` (local) or `0.0.0.0` (LAN/tailnet). Bind to a local UI
  only; never expose to the public internet.

Open `http://127.0.0.1:8765` (or your host). Query params: `?mode=grid` starts in
the audit view (`?mode=edit` for the normal canvas).

## What "verified" means

An image is **verified** when a human has gone through it and clicked `v`
(or the `verified` checkbox / `verified` button). Verifying sets
`image.attributes.verified = true` in the output COCO and stamps the image
as `verified` in the left list. Accepting a proposed box unchanged records
`attributes.source = "human-verified:<orig source>"`; a box created or edited
by the human records `attributes.source = "human"`. Nothing under `data/labels/`
or model output ever counts as ground truth — only the output file under `data/gt/`
written by this app is human-verified.

## Keyboard shortcuts

| key | action |
|-----|--------|
| `n` | draw a new box (drag on the canvas; Esc cancels) |
| `Tab` / `Shift+Tab` | cycle boxes (left→right, top→bottom) |
| `Delete` / `Backspace` | delete the selected box |
| `1`–`9`, `0`, then `q…` letters | set selected box's class (hotkey shown next to each class) |
| `v` | toggle verification of the current image; jumps to the next todo image when marking verified |
| `g` | toggle audit grid view (`?mode=grid`) |
| `f` | fit image to canvas |
| `Space` + drag, or middle-mouse drag | pan |
| wheel | zoom (centered on cursor) |
| `Esc` | cancel drawing / clear selection |
| `Enter` (focus in right panel) | — |
| `dblclick` | fit image to canvas |

Middle mouse / Alt+left-drag pan too. Right-click is suppressed.

## Canvas

- The centre canvas shows the **original full-resolution image**, served as-is.
  Nothing is re-encoded or resized server-side.
- Boxes are coloured by class (colour swatch from the catalog `color_hint`)
  with a small label tag.
- The selected box gets 8 resize handles (corners + edges) and a crop preview.

## Right panel

For the selected box:
- a zoomed crop preview (up to 3x, from the original pixels),
- a class picker: the 18 active catalog classes + `unknown` + `cartridge`,
  with number/letter hotkeys and colour swatches,
- flags: `truncated`, `occluded`, `identity_visible`,
- image-level: `scene` (cabinet / no_cabinet), `verified` checkbox, reviewer note.

## Bottom bar

Per-class counts for the current image — this is what the human checks
against reality. Counts update live.

## Audit view (`?mode=grid`)

Grid of every crop of the current image, grouped and sorted by class, so a
human can spot misclassified crops in seconds. Each cell is labeled with the
class; clicking a cell selects that box (canvas and right panel follow).

## Autosave / atomic writes

Every change (box create/edit/delete, flags, category re-pick, image attributes)
is saved to `--out` by `PUT /api/ann/<idx>` (client-side debounce ~250 ms, flush
on release). The server writes atomically: temp file + `fsync` + `rename`.
The file is valid COCO json at every instant.

## Server endpoints

| endpoint | description |
|----------|-------------|
| `GET /api/images` | image list, status (todo / in_progress / verified), box counts |
| `GET /api/image/<idx>/meta` | one image's metadata |
| `GET /img/<idx>` | raw original bytes (also `/img/<image-id>`) |
| `GET /api/ann/<idx>` | one image's annotations + image attributes |
| `PUT /api/ann/<idx>` | replace one image's annotations + attributes (autosave) |
| `GET /api/classes` | class list (id, hotkey, color, display name) |

Only files under `data/raw/freestyle-vision-data/images/` are served; `--out`
and the images root are path-traversal guarded.

## Files

- `freestyle/review/server.py` — HTTP server + store
- `freestyle/review/static/{index.html,app.js,app.css}` — UI (no build step)
- `tests/unit/freestyle/test_review_server.py` — end-to-end tests over HTTP
