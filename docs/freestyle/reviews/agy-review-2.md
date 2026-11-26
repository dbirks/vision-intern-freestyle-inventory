# Freestyle correctness & API review #2

Independent second review of the Freestyle cartridge pipeline (`freestyle/*`) against [docs/freestyle/ARCHITECTURE.md](file:///home/david/dev/vision-intern/docs/freestyle/ARCHITECTURE.md) after commit `101564a` (*fix review #1 findings*), including verification of `rfdetr==1.12.1` and `transformers==5.24.1` usage against extracted wheel source trees.

## Verification of Review #1 Fixes

| Review #1 Finding | Status after commit `101564a` | Assessment |
|---|---|---|
| 1. Identify split/group leakage | **Incomplete** | `freestyle/identify/cli.py` enforces split-map provenance, **but** `freestyle/pipeline.py:37` stores `self.gallery_groups` and never reads it (`Pipeline.run` / `run_images` never checks group disjointness). `_resolve_groups_and_splits` also rejects negative `exclude` frames inside `train`/`val`/`test` groups from `make_splits.py` (Finding 5). |
| 2. VLM `auto` coordinate mode | **Fixed** | Removed `'auto'` mode; both `VLMGrounder.__init__` and `_convert` (`freestyle/vlm_ground.py:40, 55`) reject anything outside `('norm1000', 'pixel', 'norm1')`. |
| 3. TiledDetector discards identities | **Mostly fixed** | `freestyle/pipeline.py:106-117` resolves `category_id` through `class_names`; minor gap remains for cross-class duplicate suppression across overlapping tiles (`freestyle/infer/detector.py:68-69`, Finding 11). |
| 4. `identity_thr` does not reject | **Fixed in counting, broken upstream in KNN** | `freestyle/pipeline.py:121-122` and `freestyle/count/aggregate.py:18-19` downgrade below-threshold detections to `unknown`, **but** `EmbeddingIdentifier.predict` feeds normalized `predict_proba` (often `1.0` even for tiny cosine similarity) instead of `predict_scores`, defeating rejection (Finding 1). |
| 5. Exact inventory ignores `unknown`/`cartridge` | **Fixed** | `freestyle/eval/metrics.py:582` appends `-1` (`unknown`) column so extra `unknown`/`cartridge` predictions affect `per_class_count_mae` and `exact_inventory_rate`. |
| 6. Silent per-image bootstrap fallback | **Fixed (library/CLI)** | `evaluate_predictions` defaults `allow_image_groups=False` and `load_groups_map` raises `ValueError` on missing groups (`freestyle/eval/metrics.py:114-118, 750`). Note `ARCHITECTURE.md:50` CLI example still fails unless `--groups` or `--split-map` is supplied when GT lacks embedded `group_id`. |
| 7. Threshold selection can use test labels | **Incomplete** | `freestyle/eval/evaluate.py:146-156` checks `val_gt` against `val`, **but** builds `test_images` by filtering `gt_coco["images"]` for `split_images.get(file_name) == "test"`, failing open if `--gt` uses renamed/unmapped `file_name`s (Finding 8). |
| 8. Unlabeled visible border cartridges in tiles | **Flawed** | `freestyle/train/export_tiles.py:74-82` paints solid `mean` rectangles on `< min_visible` boxes in `valid`/`test` too, overwrites overlapping kept cartridges, and crashes on grayscale `"L"` images (Finding 7). |

---

## Findings (Ranked by Severity)

### 1. Critical — `EmbeddingIdentifier.predict` uses normalized `predict_proba`, completely defeating `identity_thr` unknown rejection

- **Location:** [freestyle/pipeline.py:41-49](file:///home/david/dev/vision-intern/freestyle/pipeline.py#L41-L49), [freestyle/pipeline.py:118-122](file:///home/david/dev/vision-intern/freestyle/pipeline.py#L118-L122); cross-module mismatch with [freestyle/identify/classify.py:171-217](file:///home/david/dev/vision-intern/freestyle/identify/classify.py#L171-L217), [freestyle/identify/classify.py:587-613](file:///home/david/dev/vision-intern/freestyle/identify/classify.py#L587-L613), and [freestyle/identify/cli.py:302](file:///home/david/dev/vision-intern/freestyle/identify/cli.py#L302)
- **Failure scenario:** `KNNClassifier._probs_and_scores_from_topk` computes two distinct quantities:
  1. `probs`: `probs[i] /= probs[i].sum()` (normalized vote weights over only the top-`k` neighbors). Whenever all `k` nearest neighbors belong to one class—or only 1 class matches—`probs.max() == 1.0`, **even when cosine similarity is near 0** (e.g. a blank, occluded, or out-of-catalog crop).
  2. `scores`: `(np.sum(pos_sims * match_mask, axis=1) / float(k_eff))`, combining positive cosine similarity magnitude and top-`k` consensus. `KNNClassifier.predict_with_scores`, `calibrate_unknown_threshold`, and `identify/cli.py` all threshold `scores`.
  `EmbeddingIdentifier.predict` instead calls `self.classifier.predict_proba(embeddings)` and assigns `id_score = float(p[i])`. Consequently, an out-of-distribution crop with cosine similarity `0.05` receives `id_score = 1.0`, bypasses `identity_thr=0.5`, and is counted as a confident catalog class instead of `unknown`. Review #1's test (`tests/unit/freestyle/test_pipeline.py:54-67`) tested `Pipeline` with a mock `Detector`, never `EmbeddingIdentifier`.
- **Evidence (CPU repro via `/tmp/agy_review_2_repro.py` Repro 1):**
  ```text
  OOD query (cos_sim=0.05): predict_proba max = 1.0 | predict_with_scores = (['unknown'], [0.05])
  Pipeline out on OOD query: cola id_score= 1.0 counts= {'cola': 1} unknown= 0
  ```
- **Suggested fix:** In `EmbeddingIdentifier.predict`, call `self.classifier.predict_with_scores(embeddings, unknown_thr=None)` (or `predict_scores`) and return those similarity-consensus `scores` as `id_score`, matching `identify/cli.py` and `calibrate_unknown_threshold`. Add a unit test in `tests/unit/freestyle/test_pipeline.py` verifying an orthogonal/low-similarity crop against a single-class gallery gets rejected to `unknown` when `identity_thr=0.5`.

---

### 2. Critical — `Owlv2Backend` crashes in `transformers==5.24.1` on visual exemplar detection (`AttributeError` / `IndexError`) and rescales tile scores to `1.0`

- **Location:** [freestyle/propose/backends.py:207](file:///home/david/dev/vision-intern/freestyle/propose/backends.py#L207), [freestyle/propose/backends.py:230-245](file:///home/david/dev/vision-intern/freestyle/propose/backends.py#L230-L245)
- **Failure scenario:**
  1. Line 207 builds `target_sizes = [(image.height, image.width)]` (a Python `list`). While `post_process_grounded_object_detection` accepts a list, `Owlv2Backend.detect` passes the same `list` to `processor.post_process_image_guided_detection(..., target_sizes=target_sizes)` (line 240). In `transformers==5.24.1`, `Owlv2ImageProcessor.post_process_image_guided_detection` accesses `target_sizes.shape[1]`, which immediately crashes any GPU/CPU exemplar run with `AttributeError: 'list' object has no attribute 'shape'`.
  2. When no raw logits produce nonzero `query_scores`, `post_process_image_guided_detection` `continue`s and returns `[]`, so `[0]` at `backends.py:241` raises `IndexError: list index out of range`.
  3. Even after fixing (1) and (2), `post_process_image_guided_detection` normalizes surviving scores per query image as display alphas `(query_scores - 0.1 * max_score) / (0.9 * max_score)` clipped to `[0, 1]`, forcing the highest box on every non-empty tile to `score ≈ 1.0` regardless of true confidence, which breaks `--score-thr` and cross-tile NMS/ranking.
- **Evidence (quoted from `/tmp/pkgs_src/transformers/transformers/models/owlv2/image_processing_owlv2.py:215, 243-264`):**
  ```python
  215: if target_sizes is not None and target_sizes.shape[1] != 2:
  216:     raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")
  ...
  245:     query_scores = scores[idx]
  246:     if not query_scores.nonzero().numel():
  247:         continue
  ...
  254:     max_score = torch.max(query_scores) + 1e-6
  255:     query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
  ...
  264: return results
  ```
- **Suggested fix:** Pass `target_sizes=torch.tensor(target_sizes)` to `post_process_image_guided_detection`, handle an empty `results` return list (`if not results_list: continue`), and consider computing raw sigmoid `scores` + torchvision NMS rather than using display `query_alphas` (or document/avoid mixing exemplar display-alpha `1.0` scores with text-prompt sigmoid scores).

---

### 3. High — `Sam3Backend` with `box_exemplars` crashes in `bfloat16` on CUDA (`RuntimeError` dtype mismatch)

- **Location:** [freestyle/propose/backends.py:66-75](file:///home/david/dev/vision-intern/freestyle/propose/backends.py#L66-L75), [freestyle/propose/backends.py:486-495](file:///home/david/dev/vision-intern/freestyle/propose/backends.py#L486-L495), [freestyle/propose/backends.py:531-563](file:///home/david/dev/vision-intern/freestyle/propose/backends.py#L531-L563)
- **Failure scenario:** `Sam3Backend._load` loads `Sam3Model` in `torch.bfloat16` whenever `torch.cuda.is_bf16_supported()` is true (`backends.py:486-495`). In `detect()`, `Sam3Processor.__call__` constructs `input_boxes` as `torch.float32`, while `_move_inputs(raw_inputs, model.device, pixel_dtype=model.dtype)` (`backends.py:70-73`) only casts `"pixel_values"` to `bfloat16`, leaving `"input_boxes"` in `float32`. Inside `Sam3GeometryEncoder._encode_boxes`, `self.boxes_direct_project(boxes)` (an `nn.Linear` with `bfloat16` weights) receives `float32` `boxes` and fails on GPU with `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`.
  (Additionally, `backends.py:531, 554` calls `processor.post_process_instance_segmentation(...)`, which bilinearly interpolates every predicted mask to `(image.height, image.width)` only to discard `res["masks"]`; `processor.post_process_object_detection(...)` avoids this wasted memory/work.)
- **Evidence (quoted from `/tmp/pkgs_src/transformers/transformers/models/sam3/processing_sam3.py:196` and `modeling_sam3.py:1234, 1259`):**
  ```python
  # processing_sam3.py:196
  final_boxes = torch.tensor(padded_boxes, dtype=torch.float32)

  # modeling_sam3.py:1234, 1259
  1234: boxes_embed = self.boxes_direct_project(boxes)
  ...
  1259: pos_projection = self.boxes_pos_enc_project(pos_enc)
  ```
- **Suggested fix:** Cast floating geometry inputs (`input_boxes`) as well as `pixel_values` to `model.dtype` in `_move_inputs` / `Sam3Backend.detect`, and switch `post_process_instance_segmentation` to `post_process_object_detection` since only boxes and scores are used.

---

### 4. High — `copy_paste.py --photometric` warps pixels radially (`fisheye_warp`) without updating bounding boxes or quads

- **Location:** [freestyle/augment/copy_paste.py:795](file:///home/david/dev/vision-intern/freestyle/augment/copy_paste.py#L795), [freestyle/augment/copy_paste.py:828-829](file:///home/david/dev/vision-intern/freestyle/augment/copy_paste.py#L828-L829), [freestyle/augment/copy_paste.py:863-865](file:///home/david/dev/vision-intern/freestyle/augment/copy_paste.py#L863-L865), [freestyle/augment/photometric.py:325-392](file:///home/david/dev/vision-intern/freestyle/augment/photometric.py#L325-L392)
- **Failure scenario:** `run_cli(..., apply_photometric=True)` calls `compose_photometric()` with `cfg=None`, which selects `photometric.DEFAULT_PIPELINE`. That default pipeline includes `"fisheye_warp": {"p": 0.3, "k1_range": (-0.35, -0.05)}`. At line 829, `synth_img = photo_fn(synth_img, gen)` warps pixel coordinates radially across the full frame while `synth_anns` retains the pre-warp COCO `bbox` and `quad` coordinates—silently shifting labels off cartridges in ~30% of `--photometric` training images. Separately, line 864 writes `"groups": {}` into the synthetic `splits.json`, causing downstream group-aware scripts (`evaluate.py`, `identify/cli.py`) to reject it.
- **Evidence (CPU repro via `/tmp/agy_review_2_repro.py` Repro 2):**
  ```text
  DEFAULT_PIPELINE has fisheye_warp: {'p': 0.3, 'k1_range': (-0.35, -0.05)}
  copy_paste out images: 10 | out splits.json groups: {}
  fisheye_warp k1=-0.35 at corner x=10 samples source x=-49.5 -> shift=-59.5px while bbox stays unwarped
  ```
- **Suggested fix:** Exclude `fisheye_warp` (`{"fisheye_warp": False, ...}`) from full-image `copy_paste` photometric augmentation unless `bbox`/`quad` coordinates are warped analytically alongside the image, and populate `"groups"` in the output `splits.json` from the parent images' group IDs (e.g. `synthetic:<parent_group>`).

---

### 5. High — Incomplete Review #1 fix in `pipeline.py` (`gallery_groups` unused) and split/group conflict for negative frames in `identify/cli.py`

- **Location:** [freestyle/pipeline.py:31-39](file:///home/david/dev/vision-intern/freestyle/pipeline.py#L31-L39), [freestyle/pipeline.py:89-97](file:///home/david/dev/vision-intern/freestyle/pipeline.py#L89-L97), [freestyle/identify/cli.py:141-146](file:///home/david/dev/vision-intern/freestyle/identify/cli.py#L141-L146), [freestyle/identify/cli.py:275-285](file:///home/david/dev/vision-intern/freestyle/identify/cli.py#L275-L285), [freestyle/make_splits.py:175-187](file:///home/david/dev/vision-intern/freestyle/make_splits.py#L175-L187)
- **Failure scenario:**
  1. Commit `101564a` added `self.gallery_groups = set(map(str, data['groups']))` at `freestyle/pipeline.py:37`, yet `self.gallery_groups` is never referenced anywhere in `pipeline.py`. Calling `Pipeline` via CLI (`--images train`) or `run_images` on an image belonging to a `gallery_groups` group never raises, silently evaluating training images against their own KNN gallery rows.
  2. `make_splits.py:175-186` assigns `groups_map[gid] = "train"` (or `"val"`/`"test"`) for the group while marking negative frames (`scene_label in NEGATIVE_LABELS`) as `images_map[iid] = "exclude"`. When any annotation or image from such a frame reaches `_resolve_groups_and_splits` in `identify/cli.py:141-145`, `split` is resolved from `split_groups` (`"train"`) and fails `image_split != split` (`"exclude" != "train"`), raising `ValueError`.
  3. In `identify/cli.py:275-285`, `run_eval` passes `npz_groups` to `_resolve_groups_and_splits`, which ignores the `npz_groups` parameter completely, and `n_excluded_unknown` (`line 285`) counts `unknown` rows across the entire NPZ rather than the evaluated split/mask.
- **Evidence (CPU repro via `/tmp/agy_review_2_repro.py` Repros 1 & 7):**
  ```text
  # Repro 1:
  Pipeline.gallery_groups stored: {'g_train'} -> reused on same group g_train without error!

  # Repro 7:
  _resolve_groups_and_splits on negative inside train group raised: ValueError Missing or inconsistent split/group for cab/g_train/neg.jpg
  ```
- **Suggested fix:** Check query image group/split against `self.gallery_groups` in `Pipeline.main` / `run_images` (via `split_map` or group resolver) and fail when query groups overlap `gallery_groups` or belong to `train`. In `_resolve_groups_and_splits`, allow `image_split == "exclude"` to override `split_groups[matched]` cleanly (preserving `grp` so group identity stays intact).

---

### 6. High — `propose.run` in split or text-list mode writes string `image_id`s, causing `eval` to silently drop 100% of predictions

- **Location:** [freestyle/propose/run.py:109-130](file:///home/david/dev/vision-intern/freestyle/propose/run.py#L109-L130), [freestyle/propose/run.py:356](file:///home/david/dev/vision-intern/freestyle/propose/run.py#L356); mismatch with [freestyle/eval/metrics.py:139-172](file:///home/david/dev/vision-intern/freestyle/eval/metrics.py#L139-L172) and [docs/freestyle/ARCHITECTURE.md:45](file:///home/david/dev/vision-intern/docs/freestyle/ARCHITECTURE.md#L45)
- **Failure scenario:** `resolve_images` sets `ImageEntry(image_id=name, file_name=name)` when `--images` is a split name (`val`, `test`) or text list, so `run_proposals` writes relative path strings (e.g. `"cab/g1/008.jpg"`) into `predictions[i]["image_id"]`. In contrast, GT COCO JSON uses integer `img["id"]`, and both `freestyle/eval/metrics.py:143-172` and `freestyle/eval/failures.py:326-349` match predictions strictly by `img["id"]` (`str(img["id"])`), not `file_name`. Evaluating a proposal run created with `--images val` against `data/gt/val.json` silently ignores every prediction (`n_pred_ignored_unknown_image == len(predictions)`) and reports `AP50 = 0.0`. (`freestyle/pipeline.py:191-196` solved this exact bug by requiring `--manifest`, whereas `propose.run` has no manifest mapping.)
- **Evidence (CPU repro via `/tmp/agy_review_2_repro.py` Repro 3):**
  ```text
  resolve_images('val') emitted image_id: 'cab/g1/008.jpg'
  eval AP50: 0.0 n_pred_ignored_unknown_image: 1
  ```
- **Suggested fix:** Add a `--manifest` (COCO JSON) argument for split/text-list modes in `propose/run.py` so `entry.image_id` resolves to the COCO `img["id"]` (just like `freestyle/pipeline.py:191-198`), or warn/fail when `split_name` is used without a COCO manifest.

---

### 7. Medium — Review #1 Fix #8 (`export_tiles.py`) masks `valid`/`test` splits identically to `train`, clobbers overlapping kept instances, and crashes on grayscale images

- **Location:** [freestyle/train/export_tiles.py:56](file:///home/david/dev/vision-intern/freestyle/train/export_tiles.py#L56), [freestyle/train/export_tiles.py:74-89](file:///home/david/dev/vision-intern/freestyle/train/export_tiles.py#L74-L89)
- **Failure scenario:**
  1. **Grayscale crash:** Line 56 computes `mean` as a 3-tuple from `sample.convert("RGB")`, while lines 78–80 draw `fill=mean` onto the unconverted `sample`. Any `"L"` (8-bit grayscale) source image with a `< min_visible` box raises `TypeError: color must be int or single-element tuple`.
  2. **Overlapping neighbor corruption:** Because `< min_visible` boxes are filled with `mean` in annotation iteration order without protecting kept (`>= min_visible`) annotations, a masked border instance overlapping a kept neighbor paints solid grey over the kept cartridge's pixels.
  3. **Validation/test distribution shift:** `export_tiles` applies the exact same pixel-smearing rectangle mask to `valid/` and `test/` splits (explicitly asserted by `test_train_export.py:34-50`). RF-DETR's validation `COCOEvalCallback` / best-checkpoint selection therefore scores tiles with artificial flat grey rectangles along cutoffs that do **not** exist during `TiledDetector` full-image inference.
- **Evidence (CPU repro via `/tmp/agy_review_2_repro.py` Repro 4):**
  ```text
  export_tiles grayscale 'L' raised: TypeError color must be int or single-element tuple
  Kept ann 1 pixels [2:8, 6:9] mean after ann 2 masking (originally 240.0): 130.0
  ```
- **Suggested fix:** Convert `sample = img.crop(rect).convert("RGB")` before computing `mean` and drawing; apply `< min_visible` pixel masking only to `train` (or re-composite pixels inside kept annotations' boxes/polygons after masking); and do not artificially erase pixels on `valid`/`test` tiles unless inference applies matching border masking.

---

### 8. Medium — Review #1 Fix #7 (`evaluate.py --select-thr-on`) fails open when `--gt` image paths are missing from `--split-map`

- **Location:** [freestyle/eval/evaluate.py:146-160](file:///home/david/dev/vision-intern/freestyle/eval/evaluate.py#L146-L160)
- **Failure scenario:** In `evaluate.py:152-154`, `test_images` only collects images from `gt_coco` whose `file_name` is present in `split_images` **and** equal to `"test"`:
  ```python
  test_images = [image for image in gt_coco.get("images", [])
                 if split_images.get(image.get("file_name")) == "test"]
  ```
  If `--gt` contains images whose `file_name` is missing from `split_images` (for example a renamed/exported file carrying `attributes.group_id = "g_val"`, or using `id` keys), `test_images` becomes `[]` and `selection_groups & test_groups` silently passes without error even when groups overlap. Additionally, `val_loc = compute_localization_metrics(val_gt, val_pred, ...)` at line 158 never checks `val_loc` for `n_pred_ignored_unknown_image` (if `val_pred` has mismatched `image_id`s, threshold selection silently picks `0.0` or max score).
- **Evidence (CPU repro via `/tmp/agy_review_2_repro.py` Repro 6):**
  ```text
  evaluate.main with overlapping group 'g_val' returned exit code: 0
  ```
- **Suggested fix:** Resolve groups across **all** `gt_coco["images"]` via `load_groups_map(gt_coco, split_data)` (failing closed on unmapped images unless explicitly opting out), check `selection_groups & eval_groups` whenever `--gt` is test or contains non-val splits, and check `val_prepared.n_pred_ignored_unknown_image == 0` when selecting thresholds.

---

### 9. Medium — `VLMGrounder` / `Localizer.predict` lets hallucinated class names crash `run_images`, and `VLMCropIdentifier.contact_sheet` zeros out all `id_score`s

- **Location:** [freestyle/vlm_ground.py:90](file:///home/david/dev/vision-intern/freestyle/vlm_ground.py#L90), [freestyle/vlm_ground.py:139-144](file:///home/david/dev/vision-intern/freestyle/vlm_ground.py#L139-L144), [freestyle/pipeline.py:161-163](file:///home/david/dev/vision-intern/freestyle/pipeline.py#L161-L163); [freestyle/vlm_identify.py:80-86](file:///home/david/dev/vision-intern/freestyle/vlm_identify.py#L80-L86), [freestyle/vlm_identify.py:96-116](file:///home/david/dev/vision-intern/freestyle/vlm_identify.py#L96-L116)
- **Failure scenario:**
  1. `VLMGrounder.label_coco` (`vlm_ground.py:121`) maps out-of-catalog labels safely to `'unknown'`, whereas `VLMGrounder.detect` (`line 90`) and `Localizer.predict` (`line 142`, used by `Pipeline` when `with_identity=True`) emit the raw VLM string unchanged with `id_score = 1.0` (when `n_samples=1`). `Pipeline.run` accepts it because `id_score >= identity_thr`, and `run_images` (`pipeline.py:162-163`) crashes with `ValueError: Missing COCO category ID`.
  2. `VLMCropIdentifier.contact_sheet` (`vlm_identify.py:110`) asks only for `index` and `class` (omitting `confidence`), so `_result` sets `confidence = 0.0` on every returned dict. Any caller feeding `contact_sheet` output into `identity_thr=0.5` counting rejects 100% of crops as `unknown`.
- **Evidence (CPU repro via `/tmp/agy_review_2_repro.py` Repro 5):**
  ```text
  Localizer + run_images hallucinated label raised: ValueError Missing COCO category ID for 'Coke Zero'
  ```
- **Suggested fix:** Validate `d['label']` against `catalog.class_names()` in `Localizer.predict` (mapping unknown strings to `'unknown'` with `id_score = 0.0`, matching `VLMCropIdentifier._result`), and include `confidence` in `contact_sheet`'s prompt schema (or omit/document it).

---

### 10. Medium — `Embedder` with `Siglip2ImageProcessor` silently ignores `do_resize=False`

- **Location:** [freestyle/identify/embed.py:336-357](file:///home/david/dev/vision-intern/freestyle/identify/embed.py#L336-L357)
- **Failure scenario:** `Embedder.preprocess` letterboxes each crop to `self.input_size` (e.g. `512x512` for `google/siglip2-base-patch16-512`) and `_init_hf` passes `do_resize=False` to avoid a second resize inside the HF image processor. However, in `transformers==5.24.1`, both `Siglip2ImageProcessor` and `Siglip2ImageProcessorPil` pop `"do_resize"` inside `_validate_preprocess_kwargs` before `_preprocess` runs, forcing `do_resize=self.do_resize` (`True`) and dynamically resizing again to fit `max_num_patches` (`256` by default in code, i.e., `256x256` unless overridden by the checkpoint's config).
- **Evidence (quoted from `/tmp/pkgs_src/transformers/transformers/models/siglip2/image_processing_siglip2.py:140-143` and `image_processing_pil_siglip2.py:139-143`):**
  ```python
  140: def _validate_preprocess_kwargs(self, **kwargs) -> tuple:
  141:     # Remove do_resize from kwargs to not raise an error as size is None (computed dynamically)
  142:     kwargs.pop("do_resize", None)
  143:     return super()._validate_preprocess_kwargs(**kwargs)
  ```
- **Suggested fix:** Inspect/log the effective processed shape for SigLIP2 checkpoints or pass `max_num_patches` aligned with `(target_w // patch_size) * (target_h // patch_size)` when `Siglip2ImageProcessor` is used so `512x512` inputs are not downsampled by default `max_num_patches`.

---

### 11. Low — Cross-class duplicate detections at tile overlaps (`TiledDetector` / `VLMGrounder`) and `cartridge` GT handling in `eval/metrics.py`

- **Location:** [freestyle/infer/detector.py:68-69](file:///home/david/dev/vision-intern/freestyle/infer/detector.py#L68-L69), [freestyle/vlm_ground.py:92](file:///home/david/dev/vision-intern/freestyle/vlm_ground.py#L92), [freestyle/geometry.py:191-216](file:///home/david/dev/vision-intern/freestyle/geometry.py#L191-L216), [freestyle/eval/metrics.py:478-485](file:///home/david/dev/vision-intern/freestyle/eval/metrics.py#L478-L485)
- **Failure scenario:**
  1. `TiledDetector.predict` calls `merge_tiled`, which applies `class_aware_nms`. When a multi-class detector predicts the same physical cartridge at an overlapping tile seam as class `A` in tile 1 and class `B` in tile 2, neither box suppresses the other, producing duplicate detections (`Repro 8`) that double-count the physical cartridge (`Pipeline` later flags them as `overlap` without deduplicating).
  2. In `freestyle/eval/metrics.py:478-485`, `_per_image_identification_records` excludes `unknown_cat_ids` from `y_true` and `n_unmatched_known_gt`, whereas a GT annotation accidentally carrying `cartridge_cat_ids` is not excluded and gets evaluated as an identity class in `top1_acc`/`macro_f1`.
- **Evidence (CPU repro via `/tmp/agy_review_2_repro.py` Repro 8):**
  ```text
  TiledDetector multi-class overlapping detections: [{'box': [80.0, 10.0, 100.0, 40.0], 'score': 0.9, 'category_id': 0}, {'box': [80.0, 10.0, 100.0, 40.0], 'score': 0.85, 'category_id': 1}]
  ```
- **Suggested fix:** Add a class-agnostic NMS/overlap option for multi-class `merge_tiled` calls in `TiledDetector`/`VLMGrounder` (`with_identity=True`) at high IoU (e.g. `>= 0.7`), and skip `cartridge_cat_ids` alongside `unknown_cat_ids` in `_per_image_identification_records` (or reject `cartridge` in GT COCO).

---

## Cross-Module & Package API Checks Summary

- **Verified compatible with `rfdetr==1.12.1` source (`/tmp/pkgs_src/rfdetr`):**
  - `freestyle/train/train_detector.py:36-42`: `RFDETR{Nano,Small,Medium,Large,Base}(resolution=...)` and `model.train(dataset_dir=..., output_dir=..., epochs, batch_size, grad_accum_steps, lr, multi_scale, aug_config, early_stopping, early_stopping_patience, seed)` match `TrainConfig` (`rfdetr/config.py:1224-1404`) and `RFDETR.train` (`rfdetr/detr.py:1285`).
  - `freestyle/infer/detector.py:18-24`: `RFDETR.from_checkpoint(str(checkpoint))` and `.predict(image, threshold=threshold)` returning `Detections(xyxy, confidence, class_id)` match `rfdetr/detr.py:775, 3150` and `rfdetr/_prediction/core.py:1392-1405` (with the caveat that pretrained 80-class COCO checkpoints use sparse `class_id`s 1–90 while Freestyle-trained checkpoints use contiguous 0-based `class_names` indices). Empty-annotation negative/background tiles in `train/` and `valid/` are handled safely by `ConvertCoco` / `convert_coco_poly_to_mask` (`rfdetr/datasets/coco.py:481-523, 845-880`).
- **Verified compatible with `transformers==5.24.1` source (`/tmp/pkgs_src/transformers`):**
  - `GroundingDinoBackend`: `AutoProcessor`, `GroundingDinoForObjectDetection(..., torch_dtype=torch.float32)`, `processor.post_process_grounded_object_detection(..., input_ids=..., threshold=..., text_threshold=..., target_sizes=[(h, w)])`, and `"text_labels" in res` match `processing_grounding_dino.py:130-212`.
  - `Florence2Backend`: `Florence2ForConditionalGeneration` exists natively in `transformers==5.24.1` (`modeling_florence2.py:808`), `generate(input_ids=..., pixel_values=...)` casts vision features to `inputs_embeds.dtype` (`modeling_florence2.py:740`), and `post_process_generation(..., task=..., image_size=(w, h))` uses `(width, height)` order (`backends.py:431`).
  - `Embedder` (`dinov2`, `dinov3_vit`, `siglip`, `siglip2`): `DINOv3ViTModel.forward` and `Dinov2Model.forward` return `BaseModelOutputWithPooling` (`pooler_output`), and `Siglip2Model.get_image_features` accepts `pixel_values`, `pixel_attention_mask`, `spatial_shapes` returning `BaseModelOutputWithPooling` (`siglip2/modeling_siglip2.py:753-790`), subject to Finding 10 (`Siglip2ImageProcessor` popping `do_resize`).
  - **Broken / incompatible on GPU/CPU:** `Owlv2Backend` exemplar detection (Finding 2) and `Sam3Backend` `box_exemplars` under `bfloat16` (Finding 3).

---

## STATUS

Second independent review completed. No source code modified; created this review report only.

## FILES

- Created: [docs/freestyle/reviews/agy-review-2.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/agy-review-2.md)
- Read (in-scope): [docs/freestyle/ARCHITECTURE.md](file:///home/david/dev/vision-intern/docs/freestyle/ARCHITECTURE.md), [docs/freestyle/reviews/codex-review-1.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/codex-review-1.md), `freestyle/pipeline.py`, `freestyle/geometry.py`, `freestyle/train/*`, `freestyle/infer/*`, `freestyle/count/*`, `freestyle/vlm_ground.py`, `freestyle/vlm_identify.py`, `freestyle/identify/*`, `freestyle/propose/*`, `freestyle/eval/*`, `freestyle/augment/*`, and unit tests under `tests/unit/freestyle/`.
- Read (package sources extracted to `/tmp`): `rfdetr==1.12.1` and `transformers==5.24.1` wheel contents under `/tmp/pkgs_src/`.

## CHECKS

- `uv run pytest tests/unit/freestyle/ -rs -q` → `100 passed, 1 skipped in 10.20s` (`tests/unit/freestyle/test_eval.py:321` skipped because `pycocotools` is not installed in the CPU dev environment).
- `PYTHONPATH=. uv run python /tmp/agy_review_2_repro.py` → executed cleanly (exit code `0`), reproducing Findings 1, 4, 5, 6, 7, 8, 9, and 11 on CPU.
- `pip download --no-deps rfdetr==1.12.1 transformers==5.24.1 -d /tmp/pkgs` + source inspection in `/tmp/pkgs_src/` (since `torch`, `transformers`, and `rfdetr` are not installed in the local CPU `uv` environment).

## NOTES

- All private paths (`/home/david/dev/eg/`, `data/`) were excluded and unread; no git state was changed.
- Because `torch` and `transformers` are absent from the local CPU environment, Findings 2, 3, and 10 are established directly from the `transformers==5.24.1` wheel source code rather than live GPU/HF-checkpoint execution.
