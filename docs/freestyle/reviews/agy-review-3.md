# Vision Intern Core Dense-Scene Review #3

Independent review of the two Vision Intern core dense-scene commits (`fb8c805` *opt-in dense-scene support* and `289cc0e` *real sampling params, extra_body, finish_reason/usage logging*) across `tools/vlm_client.py`, `tools/vlm_detect.py`, `tools/bbox_utils.py`, `workflows/vlm_label.py`, and `workflows/vlm_judge.py`, evaluated against `HEAD~3` (`bd9696a`) and [docs/freestyle/VISION_INTERN_MAP.md](file:///home/david/dev/vision-intern/docs/freestyle/VISION_INTERN_MAP.md) §6 (`L1–L8, L14, L16`).

---

## Findings (Ranked by Severity)

### 1. High — `_label_local` rejects symlinked images in the source directory (including default mode) and `rglob` skips symlinked subdirectories

- **Location:** [workflows/vlm_label.py:76-80](file:///home/david/dev/vision-intern/workflows/vlm_label.py#L76-L80)
- **Failure scenario:** `_label_local` unconditionally executes `if any(not p.resolve().is_relative_to(image_dir.resolve()) for p in image_files): raise ValueError("image_paths must stay within source")`. Because `.resolve()` dereferences symlinks, any directory containing symlinks to an external image store (such as train/val split directories symlinked from raw frames, as planned in [docs/freestyle/VISION_INTERN_MAP.md:437](file:///home/david/dev/vision-intern/docs/freestyle/VISION_INTERN_MAP.md#L437)) immediately raises `ValueError`—even when `image_paths=None` and `recursive=False` (default behaviour). Conversely, when `recursive=True`, Python 3.12's `Path.rglob("*")` does not follow symlinked subdirectories, causing `FileNotFoundError` when images reside under symlinked subdirs.
- **Evidence (CPU repro via `/tmp/repro_review.py`):**
  ```text
  symlinked image in source dir: CRASHED with ValueError: image_paths must stay within source
  symlinked subdir recursive=True: FileNotFoundError: No images found in /tmp/tmp10d9um49/images
  ```
- **Fix:** Apply the containment check only when `image_paths is not None` using lexical normalization (`os.path.abspath(image_dir / p)` relative to `os.path.abspath(image_dir)`) so symlinked files inside `image_dir` remain valid while `..`/absolute path traversal in `image_paths` is still rejected; use `os.walk(..., followlinks=True)` if symlinked subdirectories should be traversed.

---

### 2. High — Cross-tile `nms` suppresses full boundary boxes in favour of earlier tile-clipped partial boxes

- **Location:** [tools/vlm_detect.py:292](file:///home/david/dev/vision-intern/tools/vlm_detect.py#L292), [tools/bbox_utils.py:122-144](file:///home/david/dev/vision-intern/tools/bbox_utils.py#L122-L144)
- **Failure scenario:** `vlm_detect` collects per-tile detections (none of which have a `"score"` key yet) and calls `nms(detections)` (`iou_threshold=0.5`, `containment_threshold=0.9`). `nms` sorts by `d.get("score", 1.0)` (all `1.0`, preserving top-left-to-bottom-right tile order). When an object straddles a tile boundary, the earlier tile produces a clipped partial box (e.g. `[950, 100, 1000, 200]`) and the overlapping tile produces the full box (`[950, 100, 1050, 200]`). Because `containment(full, clipped) == 1.0 >= 0.9`, `nms` keeps the clipped box from the earlier tile and discards the full box.
- **Evidence (CPU repro via `/tmp/repro_review.py` on 1600×600 image, `tile=1000, tile_overlap=0.2`):**
  ```text
  Cross-tile clipped vs full NMS result: [{'bbox': [950, 100, 1000, 200], 'label': 'cartridge', 'sub_label': ''}]
  ```
- **Fix:** Break score ties in `nms` (or in a cross-tile merge helper) by descending box area—or prefer detections not touching an interior tile boundary—before applying `containment` suppression.

---

### 3. High — `n_samples > 1` consensus and cross-tile `nms` run before label/alias normalization, splitting scores and leaving duplicates

- **Location:** [tools/vlm_detect.py:292,298-308](file:///home/david/dev/vision-intern/tools/vlm_detect.py#L292), [workflows/vlm_label.py:103,191](file:///home/david/dev/vision-intern/workflows/vlm_label.py#L103)
- **Failure scenario:** `vlm_detect` is called without `label_aliases` and groups boxes across tiles (`nms`, `tools/bbox_utils.py:131`) and across samples (`clusters`, `tools/vlm_detect.py:302`) using exact `label.casefold()`. `_normalize_detections` in `workflows/vlm_label.py` runs only *after* `vlm_detect` returns. If sample 0 (or tile 0) outputs `"coke-zero"` and sample 1 (or tile 1) outputs `"coke zero!"` or a catalog alias `"cola"` for the exact same box, `vlm_detect` splits them into two unmerged detections (`score=0.5` each) and `_normalize_detections` then maps both to `"coke-zero"`, yielding duplicate identical boxes with halved consensus scores.
- **Evidence (CPU repro via `/tmp/repro_review.py`):**
  ```text
  n_samples=2 with 'coke-zero' vs 'coke zero!': anns=[([100, 100, 100, 100], 0.5), ([100, 100, 100, 100], 0.5)]
  ```
- **Fix:** Normalize labels (punctuation-stripped canonical keys + `label_aliases`) inside `vlm_detect` (or pass a `label_normalizer` callback from `label_dataset`) prior to cross-tile `nms` and multi-sample consensus clustering.

---

### 4. High — Crop-mode contact sheet omits box boundaries/labels while the prompt asks the judge to verify tightness, and crashes on OOB or inverted boxes

- **Location:** [workflows/vlm_judge.py:288-309](file:///home/david/dev/vision-intern/workflows/vlm_judge.py#L288-L309), [workflows/vlm_judge.py:160-197](file:///home/david/dev/vision-intern/workflows/vlm_judge.py#L160-L197), [workflows/vlm_judge.py:348-354](file:///home/david/dev/vision-intern/workflows/vlm_judge.py#L348-L354)
- **Failure scenario:**
  1. `_crop_sheet` crops a 15%-padded region around each detection and draws only `#i` in black text above the crop without drawing the proposed bounding box or class label on the crop, yet passes it to `_build_judge_prompt_overlay`, which instructs the model: `"Each proposed detection is drawn on the image as a numbered coloured box (#0, #1, …) with its label"` and asks `"2. Is the box reasonably tight around the object?"`. Without a drawn box, the judge cannot verify box tightness or distinguish the target from adjacent objects inside the 15% padding.
  2. `_crop_sheet` computes `box = (max(0, int(x1 - pad_x)), ..., min(image.width, math.ceil(x2 + pad_x)), ...)` without validating `left < right` and `top < bottom`, and runs outside `score_detections`'s `try/except`. Any out-of-bounds (`x1 >= image.width`) or inverted (`x1 > x2`) detection crashes the entire run with `ValueError` inside `image.crop(box)`.
  3. When `judge_mode="crops"` is used without `judge_batch_size` on dense frames (e.g. 30+ boxes), `_crop_sheet` packs all crops into one large grid (e.g. `1656×1430` for 30 boxes) that default `max_dimension=1280` immediately downscales.
- **Evidence (CPU repro via `/tmp/repro_review.py`):**
  ```text
  _crop_sheet(oob=[450, 100, 500, 150]): CRASHED with ValueError: Coordinate 'right' is less than 'left'
  _crop_sheet(inverted=[200, 200, 100, 100]): CRASHED with ValueError: Coordinate 'right' is less than 'left'
  crop prompt snippet: 'You are a quality-control judge for object-detection annotations.\nEach proposed detection is drawn on the image as a numbered coloured box (#0, #'
  ```
- **Fix:** Draw each proposed box (in crop-relative coordinates, after scaling if thumbnailed) and label onto its padded crop (or use a dedicated crop-sheet prompt), clamp/validate `box` in `_crop_sheet` so degenerate/OOB boxes render a blank placeholder instead of raising `ValueError`, and set a sensible default batch size when `judge_mode="crops"` if `judge_batch_size is None`.

---

### 5. High — `backend="transformers"` never returns `VLMResponse` / `finish_reason`, silently disabling truncation detection and auto-split

- **Location:** [tools/vlm_client.py:186-251](file:///home/david/dev/vision-intern/tools/vlm_client.py#L186-L251), [tools/vlm_detect.py:250](file:///home/david/dev/vision-intern/tools/vlm_detect.py#L250), [workflows/vlm_judge.py:378,583](file:///home/david/dev/vision-intern/workflows/vlm_judge.py#L378)
- **Failure scenario:** `_run_openai` returns `VLMResponse(text, finish_reason, usage)`, whereas `_run_transformers_batch` returns `processor.batch_decode(gen, skip_special_tokens=True)` (plain `list[str]`). Consequently `getattr(response, "finish_reason", None)` is always `None` on `backend="transformers"`, silently disabling truncation warnings, `vlm_detect` quadrant auto-split, COCO `images[].truncated`, and `raw_response_log["finish_reason"]`.
- **Evidence (CPU repro via `/tmp/repro_review.py` and inspection of `tools/vlm_client.py:239`):**
  ```text
  transformers backend truncated flag: False
  ```
- **Fix:** In `_run_transformers_batch`, check each generated sequence's token length against `max_tokens` and whether it terminated with `eos_token_id` (when available on the tokenizer/generation config), returning `VLMResponse(text, finish_reason="length" if truncated else "stop")`.

---

### 6. Medium — Judge `_parse_verdicts` uses greedy `\[.*\]` + strict `json.loads`, dropping all verdicts on truncated output or trailing bracketed notes

- **Location:** [workflows/vlm_judge.py:247-253,274](file:///home/david/dev/vision-intern/workflows/vlm_judge.py#L247-L253)
- **Failure scenario:**
  1. Unlike `vlm_detect.parse_bboxes_from_response`, `_parse_verdicts` requires a complete `[...]` span matched by greedy `re.search(r"\[.*\]", text, re.DOTALL)` and parsed via `json.loads`. If a judge response hits `max_tokens` before closing `]`, `_parse_verdicts` returns `{}` and marks **every** box in that batch `"unjudged"` instead of salvaging completed verdict objects.
  2. If the model appends any bracketed text after the JSON array (e.g. `Note: [box 0] looks good.`), the greedy regex captures through the final `]`, `json.loads` raises `JSONDecodeError`, and all verdicts are lost.
  3. Unlike `vlm_detect`, `score_detections` only warns on `finish_reason == "length"` without splitting/retrying the batch.
  4. Line 274 calls `float(item.get("score", 0.0))` without guarding `None` or non-numeric strings, raising an uncaught `TypeError`/`ValueError`.
- **Evidence (CPU repro via `/tmp/repro_review.py`):**
  ```text
  _parse_verdicts trailing bracket note: {}
  _parse_verdicts truncated array: {}
  ```
- **Fix:** Parse from the first `[` with `json.JSONDecoder().raw_decode` (ignoring trailing text), add a per-object `{...}` regex fallback for truncated arrays, guard `float(...)` coercion, and optionally split-and-retry `score_detections` chunks on `finish_reason == "length"`.

---

### 7. Medium — `_normalize_detections` edge cases (`label: None` crash, class-key collisions, order-dependent chained aliases, fragmented `unmatched:` categories)

- **Location:** [workflows/vlm_label.py:40-58,110-113](file:///home/david/dev/vision-intern/workflows/vlm_label.py#L40-L58)
- **Failure scenario:**
  1. A detection dict with `"label": None` passes `None` to `key(raw)` and crashes with `AttributeError: 'NoneType' object has no attribute 'casefold'`.
  2. If `classes` contains two labels that collapse under `re.sub(r"[^\w]+|_", "", value.casefold())` (e.g. `["a_b", "a-b"]`), `canonical` silently overwrites the first with the second while `_label_local` still emits both in `coco["categories"]` (leaving an orphaned category). Similarly, `classes=["Cat", "cat"]` passes `classes_lower = ["cat", "cat"]` to `_label_local`, emitting duplicate category names with distinct IDs.
  3. Because `canonical` is mutated in-place inside `for alias, target in (aliases or {}).items():`, chained aliases are order-dependent: `{"a": "coke", "b": "a"}` succeeds while `{"b": "a", "a": "coke"}` raises `ValueError: Alias target 'a' is not a class`.
  4. When `keep_unmatched=True`, `"unmatched:" + raw` preserves raw casing/punctuation, creating separate COCO categories for `"unmatched:Pepsi"`, `"unmatched:pepsi"`, and `"unmatched:Pepsi!"`.
- **Evidence (CPU repro via `/tmp/repro_review.py`):**
  ```text
  label=None: CRASHED with AttributeError: 'NoneType' object has no attribute 'casefold'
  class key collision ['a_b', 'a-b']: ['a-b', 'a-b']
  chained aliases b->a->coke: ValueError: Alias target 'a' is not a class
  keep_unmatched categories for Pepsi/pepsi/Pepsi!: [{'id': 0, 'name': 'coke'}, {'id': 1, 'name': 'unmatched:Pepsi'}, {'id': 2, 'name': 'unmatched:pepsi'}, {'id': 3, 'name': 'unmatched:Pepsi!'}]
  ```
- **Fix:** Coerce `raw = str(det.get("label") or "")`, deduplicate `classes_lower` and check for `key()` collisions when constructing `canonical`, resolve alias targets against initial classes only (or resolve chains topologically), and normalize `unmatched:` labels consistently if category consolidation is wanted.

---

### 8. Medium — Auto-split disabled for `tile=None`, quadrant splits lack overlap, and `n_samples` clustering uses asymmetric `containment`

- **Location:** [tools/vlm_detect.py:251-259,288,303](file:///home/david/dev/vision-intern/tools/vlm_detect.py#L251-L259)
- **Failure scenario:**
  1. Line 288 passes `retry = tile is not None` to `detect_region`, so full-image detection (`tile=None`) never auto-splits on `finish_reason == "length"`.
  2. When `tile` is set and a tile truncates, lines 254–255 split at `(mx, my)` with zero overlap; any object crossing `mx` or `my` is sliced into two non-overlapping pieces that cannot be merged by `nms`.
  3. For `n_samples > 1`, line 303 merges detections across samples using `containment(c["det"]["bbox"], det["bbox"]) >= 0.7`. A small box `[100, 100, 120, 120]` in sample 0 and a huge box `[0, 0, 1000, 1000]` in sample 1 have `containment == 1.0`, merging into one cluster with `score=1.0` that keeps whichever box appeared in sample 0.
- **Evidence (CPU repro via `/tmp/repro_review.py` and code inspection):**
  ```text
  tile=None truncated: calls=1, truncated=True, warnings=1, dets=[{'bbox': [0, 0, 100, 100], 'label': 'cartridge', 'sub_label': ''}]
  ```
- **Fix:** Allow opt-in or default auto-split when `tile=None`, pad quadrant splits by `tile_overlap` when splitting a truncated tile, and require IoU (or symmetric min-IoU) for multi-sample clustering in `n_samples > 1`.

---

### 9. Medium — `score_detections_batch` and `run_vlm_batch` lack all dense-scene controls

- **Location:** [workflows/vlm_judge.py:392-456](file:///home/david/dev/vision-intern/workflows/vlm_judge.py#L392-L456), [tools/vlm_client.py:332-366](file:///home/david/dev/vision-intern/tools/vlm_client.py#L332-L366)
- **Failure scenario:** The GPU batching path (`score_detections_batch`, used by `jobs/judge_one.py`) and `run_vlm_batch` were not updated: `score_detections_batch` still hardcodes `max_tokens=1024` for non-LFM3 models, has no `judge_batch_size`, `judge_mode="crops"`, `max_dimension`, `image_format`, `jpeg_quality`, or truncation warning, and `run_vlm_batch` lacks sampling/encoding/`raw_response_log` arguments.
- **Evidence:** Direct inspection of [workflows/vlm_judge.py:445-449](file:///home/david/dev/vision-intern/workflows/vlm_judge.py#L445-L449) and [tools/vlm_client.py:332-366](file:///home/david/dev/vision-intern/tools/vlm_client.py#L332-L366).
- **Fix:** Forward `max_tokens`, encoding options, sampling options, `judge_batch_size`, `judge_mode`, and truncation handling into `run_vlm_batch` and `score_detections_batch` (or clearly deprecate/gate `score_detections_batch` for dense scenes).

---

### 10. Medium — `raw_response_log` appends without a thread lock under Hub mode's 16-thread pool

- **Location:** [tools/vlm_client.py:324-328](file:///home/david/dev/vision-intern/tools/vlm_client.py#L324-L328), [workflows/vlm_label.py:199-215](file:///home/david/dev/vision-intern/workflows/vlm_label.py#L199-L215)
- **Failure scenario:** `_label_hub` with `backend="openai"` runs `vlm_detect` across 16 concurrent threads (`LABEL_MAX_WORKERS`), each opening `raw_response_log` and calling `log.write(json.dumps(record) + "\n")` with no lock, risking interleaved/corrupted JSONL lines when payloads exceed the Python/OS write buffer.
- **Evidence:** Direct inspection of [tools/vlm_client.py:324-328](file:///home/david/dev/vision-intern/tools/vlm_client.py#L324-L328) and [workflows/vlm_label.py:199-215](file:///home/david/dev/vision-intern/workflows/vlm_label.py#L199-L215).
- **Fix:** Protect `raw_response_log` writes with a `threading.Lock` (e.g. a module-level `_log_lock`).

---

### 11. Low — Duplicate `image_paths` produce duplicate `file_name` entries in COCO output, and minor default-path semantic shifts

- **Location:** [workflows/vlm_label.py:40-58,76,106](file:///home/david/dev/vision-intern/workflows/vlm_label.py#L40-L58), [tools/vlm_client.py:79,152](file:///home/david/dev/vision-intern/tools/vlm_client.py#L79), [workflows/vlm_judge.py:273,280,481,588](file:///home/david/dev/vision-intern/workflows/vlm_judge.py#L273)
- **Failure scenario & evidence (CPU repro via `/tmp/repro_review.py`):**
  1. Passing `image_paths=["a.png", "./a.png"]` keeps unique COCO `id`s (`0, 1`) and `ann_id`s (`0, 1`), but emits duplicate `file_name: "a.png"` entries in `coco["images"]`.
  2. Default-path semantic shifts compared to `HEAD~3`:
     - `workflows/vlm_label.py:40-58`: Punctuation/underscore stripping in `_normalize_detections` is active even when `label_aliases=None`; `"Coke Zero!"` now matches class `"coke-zero"` (`old_anns=0, new_anns=1`).
     - `workflows/vlm_judge.py:273,280,481`: Missing verdicts changed from `"verdict": "incorrect"` to `"verdict": "unjudged"` (score remains `0.0`, so `threshold` filtering is unchanged, while callers checking `verdict == "incorrect"` observe `"unjudged"`).
     - `tools/vlm_client.py:79`: `_pil_to_data_url` calls `image.convert("RGB")` before JPEG save (byte-identical for `"RGB"`/`"RGBA"`, different bytes for `"L"` grayscale while fixing `"P"` mode crashes).
     - `tools/vlm_client.py:152`: `_run_openai` now always includes `top_p=1` in the API payload (omitted in `HEAD~3`).
- **Fix:** Deduplicate normalized relative paths when `image_paths` is provided, and document the intentional default shifts (`"unjudged"` and punctuation-normalized matching).

---

## Checklist Verification Summary (`a`–`f`)

- **(a) Defaults preserve previous behaviour:**
  - `tools/vlm_detect.py`: Exact output and prompt parity with `HEAD~3` verified on mocked `RGB` (2048×1536) and `RGBA` inputs (`vlm_detect outputs equal: True`, `vlm_detect old prompt == new prompt: True`). `run_vlm` receives the same `max_tokens=4096` plus new default encoding/sampling kwargs (`max_dimension=1280, image_format=None, jpeg_quality=85, temperature=0, top_p=1, seed=None, extra_body=None, raw_response_log=None`).
  - `tools/vlm_client.py`: `_pil_to_data_url` is byte-identical for `RGB` and `RGBA`; differs for grayscale `L` due to `.convert("RGB")` (Finding 11). `_run_openai` additionally sends `top_p=1` and returns `VLMResponse` (a `str` subclass).
  - `workflows/vlm_label.py` & `workflows/vlm_judge.py`: Intentional default changes observed in repro: punctuation-normalized label matching is always on (Finding 11), and missing judge verdicts are `"unjudged"` instead of `"incorrect"` (Finding 11). Pre-existing and new unit tests pass (`199 passed, 9 skipped`).
- **(b) Tiling coordinate remap & edge tiles:**
  - `starts(length)` + `denormalize_bbox(raw, right - x, bottom - y)` + `[bbox[0] + x, bbox[1] + y, ...]` correctly remaps 0-1000 tile-local coordinates on full tiles, non-multiple edge tiles (`2048×1536` with `tile=1024, overlap=0.2` -> `[1900, 1400, 2000, 1500]`), non-square images (`800×1500` with `tile=1024` -> `[400, 1400, 600, 1450]`), and odd quadrant auto-splits (`101×101` -> `[60, 60, 101, 101]`). Broken by cross-tile `nms` preferring clipped edge boxes over full boxes (Finding 2).
- **(c) Truncation detection & auto-split retry:**
  - Works on mocked `VLMResponse(..., "length")` with `tile` set (depth-1 2×2 split, propagating `DetectionList.truncated`). Fails on `backend="transformers"` (Finding 5), does not retry when `tile=None` or split with overlap (Finding 8), and judge truncation drops the batch in `_parse_verdicts` (Finding 6).
- **(d) Judge batching & crop-mode mapping:**
  - In `score_detections` (and `_judge_local` with `judge_mode="crops"` / `overlay=True` / `judge_batch_size`), every box receives exactly one verdict and `detection_idx` stays aligned across batches (`[(100, 0, ...), ..., (104, 4, ...)]` verified in repro). However, `_crop_sheet` omits drawn boxes/labels while using the overlay prompt and crashes on OOB/inverted boxes (Finding 4), `_parse_verdicts` fails on truncated/trailing-bracket responses (Finding 6), and `score_detections_batch` lacks batching/crop/token controls (Finding 9).
- **(e) Label alias matching:**
  - Works for basic punctuation/case normalization and direct aliases, but has edge-case bugs (`label: None` crash, `key()` class collisions, order-dependent chained aliases, fragmented `unmatched:` categories — Finding 7) and runs after `vlm_detect` consensus/NMS (Finding 3).
- **(f) Recursive local dir `file_name` & COCO ID uniqueness:**
  - `img_id` (`0..N-1`), `ann_id` (`0..M-1`), and `category_id` (`0..K-1`, including `keep_unmatched=True`) are unique per run; `file_name` uses `path.relative_to(image_dir).as_posix()`. Fails on symlinked images (even in default non-recursive mode) and skips symlinked subdirs (Finding 1); duplicate `image_paths` / duplicate `classes` create duplicate `file_name` / category name entries (Finding 11 / Finding 7).

---

## STATUS

Review complete. No repository source files were modified; only [docs/freestyle/reviews/agy-review-3.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/agy-review-3.md) was created.

## FILES

- Created: [docs/freestyle/reviews/agy-review-3.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/agy-review-3.md)
- Read (in-scope): [docs/freestyle/VISION_INTERN_MAP.md](file:///home/david/dev/vision-intern/docs/freestyle/VISION_INTERN_MAP.md), [tools/vlm_client.py](file:///home/david/dev/vision-intern/tools/vlm_client.py), [tools/vlm_detect.py](file:///home/david/dev/vision-intern/tools/vlm_detect.py), [tools/bbox_utils.py](file:///home/david/dev/vision-intern/tools/bbox_utils.py), [tools/bbox_viz.py](file:///home/david/dev/vision-intern/tools/bbox_viz.py), [tools/d1.py](file:///home/david/dev/vision-intern/tools/d1.py), [tools/model_options.py](file:///home/david/dev/vision-intern/tools/model_options.py), [tools/registry.py](file:///home/david/dev/vision-intern/tools/registry.py), [workflows/vlm_label.py](file:///home/david/dev/vision-intern/workflows/vlm_label.py), [workflows/vlm_judge.py](file:///home/david/dev/vision-intern/workflows/vlm_judge.py), [workflows/train_rfdetr.py](file:///home/david/dev/vision-intern/workflows/train_rfdetr.py), [jobs/judge_one.py](file:///home/david/dev/vision-intern/jobs/judge_one.py), [tests/unit/test_dense_scenes.py](file:///home/david/dev/vision-intern/tests/unit/test_dense_scenes.py), [tests/unit/test_bbox_utils.py](file:///home/david/dev/vision-intern/tests/unit/test_bbox_utils.py), [tests/unit/test_judge_reasoning.py](file:///home/david/dev/vision-intern/tests/unit/test_judge_reasoning.py), [tests/unit/test_judge_pair.py](file:///home/david/dev/vision-intern/tests/unit/test_judge_pair.py), [docs/freestyle/reviews/agy-review-2.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/agy-review-2.md).
- Extracted to `/tmp/old_core/` (read-only baseline comparison): `git archive HEAD~3 tools workflows`.

## CHECKS

- `uv run pytest tests/unit/` → `199 passed, 9 skipped in 11.09s` (all `tests/unit/test_dense_scenes.py`, `test_bbox_utils.py`, `test_judge_pair.py`, `test_judge_reasoning.py` passed; skipped tests are `pycocotools`-gated eval test and HF Hub/GPU job tests).
- `uv run python /tmp/repro_review.py` → executed cleanly (exit code `0`), comparing `HEAD~3` (`/tmp/old_core`) vs current working tree and reproducing Findings 1–8 and 11 on CPU with mocked `run_vlm` responses.

## NOTES

- No state-changing git commands were executed; `data/` and `/home/david/dev/eg/` were not read; no live VLM endpoint calls were made.
- Temporary baseline files and CPU repro script were placed in `/tmp/old_core` and `/tmp/repro_review.py` only, leaving the repository working tree untouched except for [docs/freestyle/reviews/agy-review-3.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/agy-review-3.md).
