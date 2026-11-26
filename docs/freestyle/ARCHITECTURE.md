# Freestyle cartridge recognition — project contract

Goal: from a photograph taken by the in-door fisheye camera of a Coca-Cola Freestyle
cartridge cabinet, **locate** every cartridge face, **identify** it (catalog id), and
**count** cartridges per catalog id. Built on Vision Intern (label → judge → train),
extended where needed. The final architecture is chosen by experiment, not assumed.

## Data (private, gitignored)

```
data/raw/freestyle-vision-data/        extracted package, read-only (README/GAPS/hardware notes inside)
data/derived/                          anything computed from raw (crops, undistorted copies, tiles)
data/gt/                               HUMAN-VERIFIED ground truth only
data/labels/                           model-generated labels (never GT)
runs/<experiment_id>/                  experiment outputs (predictions, metrics, viz)
```

* Image id = path relative to `data/raw/freestyle-vision-data/images/` with `/` kept,
  e.g. `island-cabinet-01/50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg`, `local/downloads/frame.jpg`.
* Group id (for splits; the independent unit) = the event uuid for server frames,
  `local/blur` for the 12-frame burst, and one group per remaining local file.
* `data/splits.json`: `{"version":1, "groups": {group_id: split}, "images": {image_id: split}}`,
  split ∈ `train | val | test | exclude`. Test groups are frozen once GT exists; never tune on them.

## Classes

From `data/raw/freestyle-vision-data/data/catalog_items.json`, `active: true` rows only
(18 ids, e.g. `nns-sweetener`, `coca-cola-part1`). Plus:
* `cartridge` — generic super-class used by class-agnostic detectors.
* `unknown` — a cartridge whose identity is not determinable from the pixels (blur, occlusion, cut off).
`freestyle/catalog.py` is the single source of class lists, display names, shelf hints, colour hints.

## Annotation format

COCO JSON, boxes `[x, y, w, h]` in ORIGINAL image pixels (no resizing). Per annotation extras:
* `category_id` → catalog id (category `name` = catalog id slug).
* `attributes`: `{"truncated": bool, "occluded": bool, "identity_visible": bool, "source": "human|vlm:<model>|model:<exp>"}`
* optional `quad`: 4 `[x,y]` corner points (the face is a slanted rectangle under fisheye).
Countable cartridge = any cartridge whose front face is at least partially in frame.
Frames with no cabinet in view have zero annotations and `image.attributes.scene = "no_cabinet"`.

## Prediction format (every method emits this)

`runs/<exp>/predictions.json`: COCO results list
`[{"image_id": <coco image id>, "bbox": [x,y,w,h], "score": float, "category_id": int, "id_score": float}]`
plus `runs/<exp>/meta.json` (`{"experiment": ..., "config": ..., "git_sha": ..., "hardware": ..., "seconds_per_image": ..., "gpu_mem_gb": ...}`).

## Evaluation (`freestyle/eval/`)

`uv run python -m freestyle.eval.evaluate --gt data/gt/<split>.json --pred runs/<exp>/predictions.json --out runs/<exp>/metrics.json`
* Localization (class-agnostic): AP50, AP50:95, recall@IoU0.5 at the best-F1 threshold, precision there.
* Identification: top-1 accuracy over GT boxes matched at IoU≥0.5 (excluding GT `unknown`), macro-F1 over classes.
* Counting: per image per class |pred−gt| → per-image total-count MAE, per-class count MAE,
  exact-inventory rate (fraction of images where every class count is correct).
* All metrics reported with the image list and bootstrap 95% CI over groups (tiny test set).

## Experiments

`configs/experiments/<exp>.yaml` — one file per experiment, fully specifies method + params.
GPU experiments run as k8s Jobs from `k8s/` templates (see `k8s/README.md`); CPU/VLM-API ones may run locally.
Results table lives in `docs/freestyle/RESULTS.md` (generated, not hand-edited).
