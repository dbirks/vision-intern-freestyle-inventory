# Freestyle Review #5 — `score_all.py` & Quiz Ground-Truth Scoring (`to_coco.py`)

Rigorous pre-human-GT audit of [freestyle/experiments/score_all.py](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py) and [freestyle/quiz/to_coco.py](file:///home/david/dev/vision-intern/freestyle/quiz/to_coco.py) on branch `freestyle-cartridges`, cross-checked against [freestyle/count/pairs.py](file:///home/david/dev/vision-intern/freestyle/count/pairs.py), [freestyle/count/shelves.py](file:///home/david/dev/vision-intern/freestyle/count/shelves.py), [freestyle/eval/matching.py](file:///home/david/dev/vision-intern/freestyle/eval/matching.py), [freestyle/eval/metrics.py](file:///home/david/dev/vision-intern/freestyle/eval/metrics.py), [freestyle/identify/classify.py](file:///home/david/dev/vision-intern/freestyle/identify/classify.py), [freestyle/quiz/build_queue.py](file:///home/david/dev/vision-intern/freestyle/quiz/build_queue.py), [freestyle/quiz/server.py](file:///home/david/dev/vision-intern/freestyle/quiz/server.py), and [docs/freestyle/ARCHITECTURE.md](file:///home/david/dev/vision-intern/docs/freestyle/ARCHITECTURE.md).

All findings were confirmed on CPU with synthetic quiz data and run artifacts using `/tmp/repro_score_all.py` and `/tmp/repro_gallery_and_bootstrap.py` (no code modified, no images under `data/` or `runs/` opened, no git state modified).

---

## Executive Summary of Audit Coverage

| Audit Area | Verdict | Key Issues Found |
|---|---|---|
| **GT construction** (`to_coco.py`, `build_human_gt`) | **Broken on negatives, partials, and partial completion** | `build_human_gt` drops verified `not_cabinet` negatives (Finding 2); `to_coco._identity` silently promotes wrong model guesses on `"partial"` (Finding 5); partially answered images (`0.8 <= fraction < 1.0`) have skipped boxes deleted while keeping full `human_total` (Finding 3). Last-answer-wins & `undo` work cleanly. |
| **Pair rule application** (`prepare_rows`, `infer_pairs`) | **Broken / No-op** | `pair=on` is a 100% no-op for all 7 non-VLM identifiers (`own`, `knn`, `proto`, `knn_trainvlm`, `knn_trainvlm+val`, `linear_trainvlm`, `linear_trainvlm+val`), and on `vlm` it bypasses `identity_thr > 0.5` while ignoring real `id_confidence < 0.5` two-part crops (Finding 1). |
| **Count metrics** (total, top/bottom, per-class, `unknown`/`cartridge`, mismatches) | **Incomplete & biased** | `score_subset` drops predicted-only classes including `'unknown'` from `class:{c}` MAE; `prepare_rows` fails to map generic `'cartridge'` to `'unknown'` (Finding 4); `counts_mismatch` & unverified images corrupt `class:{c}` MAE and `ap50` (Finding 3). |
| **Matching predicted boxes to crops** | **Minor mismatch vs `eval/metrics.py`** | `per_image` calls `match_boxes` without `pred_scores`, letting low-confidence duplicates steal GT matches (Finding 7). `box_iou(..., 'xywh')` in `identify` is consistent with `emb['bboxes']` and `proposals`, though offline `emb` only covers `bootstrap_v2` boxes. |
| **Identifier variants & gallery leakage** | **No test leakage; `val` dual-use caveat** | `test` crops never enter `gallery` or `train_gallery`; `validate_groups` + `gallery(..., held_out_group)` properly hold out the query group during `val` LOGO. However, `val` uses 5-group LOGO for threshold/confidence selection and 6-group `val` at test (Finding 8a). |
| **Val-only threshold selection** | **Tie-break & mismatch bias** | `select_thresholds` breaks ties toward `score_thr = 0.0` (accepting sub-floor test FPs) and `confidence = 0.0` (Finding 6), and calibrates `identity_thr` on `counts_mismatch`/unverified `val` images (Finding 3). |
| **Group bootstrap CIs** | **Sound** | `score_subset` aligns image order with `loc_records` and clusters correctly by `group_id` via `compute_group_bootstrap_ci`, handling empty subsets (`sharp-full`/`partial/blurry`) and `None` -> `nan` cleanly. |
| **Sharp / blurry subsets** | **Minor `no_cabinet` leak** | `audit.json` criteria omit `usable=False`, so a human-marked `not_cabinet` frame with `motion_blur='severe'` still enters `partial/blurry` (Finding 8b). |

---

## Findings (Ranked by Severity)

### 1. Critical — Pair rule (`pair=on`) is a 100% no-op for all 7 non-VLM identifiers and bypasses `identity_thr` on `vlm`

- **Location:** [freestyle/experiments/score_all.py:186-189](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L186-L189), [freestyle/experiments/score_all.py:238-252](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L238-L252); [freestyle/count/pairs.py:33-34](file:///home/david/dev/vision-intern/freestyle/count/pairs.py#L33-L34), [freestyle/count/pairs.py:68-111](file:///home/david/dev/vision-intern/freestyle/count/pairs.py#L68-L111)
- **Failure scenario:**
  1. **Total no-op on `own`, `knn`, `proto`, `knn_trainvlm`, `knn_trainvlm+val`, `linear_trainvlm`, `linear_trainvlm+val`:** `identify()` (`score_all.py:170-174, 221-223`) only sets `class` and `identity_score` (never `family` or `part`). `prepare_rows()` (`score_all.py:242-245`) then runs `re.match(r'(.+)-part([12])$', row['class'])` and sets `family=match.group(1), part=int(match.group(2))`. Consequently:
     - Whenever `family` is set, `part` is **already** `1` or `2`. Inside `infer_pairs(..., mode='apply')` (`pairs.py:109-111`), `if mode == 'apply' and existing not in (1, 2): d['class'] = ...` **never fires**.
     - Whenever `row['class'] == 'unknown'`, the regex fails to match, so `family` remains `None` and `infer_pairs` ignores the box (`pairs.py:68-71`).
     - In `runs/score_all_demo/RESULTS_GT.md`, every single `pair=off` vs `pair=on` row for `knn`, `proto`, `knn_trainvlm`, `knn_trainvlm+val`, `linear_trainvlm`, and `linear_trainvlm+val` (and even `vlm`) has identical numbers.
  2. **`identity_thr` bypass & `id_confidence < 0.5` loss on `vlm`:** For `method == 'vlm'`, `identify()` (`score_all.py:186-189`) sets `family_confidence = part_confidence = id_confidence`.
     - If `identity_thr = 0.75` and `id_confidence = 0.60`, `prepare_rows` (`score_all.py:239-240`) sets `row['class'] = 'unknown'` **without** clearing `family_confidence`, `part`, or `part_confidence`. `infer_pairs` then runs with hardcoded `family_threshold=0.5, part_threshold=0.5` (`pairs.py:33-34`), sees `0.60 >= 0.5`, and overwrites `row['class']` back to `coca-cola-part2` — defeating `identity_thr=0.75`.
     - Conversely, in `data/labels/bootstrap_v2.json`, **103 / 428** family-tagged crops (including `id_family != None, id_part = None` two-part crops) have `id_confidence < 0.5` (e.g., `0.40–0.45`). Because `score_all.py:188` sets `family_confidence = id_confidence`, `infer_pairs` treats those families as `None` and skips them.
- **Evidence (CPU repro via `/tmp/repro_score_all.py` Check 1 & `/tmp/repro_gallery_and_bootstrap.py` Check 12):**
  ```text
  1a non-VLM pair=False: ['coca-cola-part1', 'unknown'] pair=True: ['coca-cola-part1', 'unknown']
  1b VLM @ id_thr=0.75 pair=False: ['coca-cola-part1', 'unknown'] pair=True: ['coca-cola-part1', 'coca-cola-part2']
  E2E vlm pair=on  ID accuracy: 0.0  class:coca-cola-part2: 1.0  (box 3 had id_confidence=0.4 < 0.5)
  ```
- **Suggested fix:**
  - In `score_all.py:238-246`, when `row['identity_score'] < identity_thr`, either clear `family`/`part`/`family_confidence`/`part_confidence` (unless `vlm` had an explicit family stage above a separate family threshold) or pass `family_threshold=identity_thr, part_threshold=identity_thr` to `infer_pairs` so `infer_pairs` cannot override `identity_thr`.
  - To make `pair=on` meaningful for embedding classifiers (`knn`, `proto`, `linear`) and `own`, either: (a) pool class scores/probabilities per two-part brand family (e.g. `p(coca-cola) = p(part1) + p(part2)`) so confident-family / uncertain-part boxes have `family='coca-cola', part=None`, or (b) allow `infer_pairs` (or a `score_all` post-pass) to resolve conflicting/low-margin `part` predictions within a coherent run (`part_confidence` derived from the `part1` vs `part2` margin rather than total confidence).
  - For `vlm`, only pass `part_confidence` when `part in (1, 2)` (it is `0.0` when `part is None`), and decide whether `family_confidence` in `two_stage` comes from Stage 1 vs Stage 2 rather than overloading low `id_confidence` (`0.40`).

---

### 2. High — `build_human_gt` silently drops human-verified `not_cabinet` negative frames

- **Location:** [freestyle/experiments/score_all.py:48-59](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L48-L59) vs [freestyle/quiz/to_coco.py:92-94](file:///home/david/dev/vision-intern/freestyle/quiz/to_coco.py#L92-L94), [freestyle/quiz/to_coco.py:137](file:///home/david/dev/vision-intern/freestyle/quiz/to_coco.py#L137), [freestyle/quiz/to_coco.py:147-148](file:///home/david/dev/vision-intern/freestyle/quiz/to_coco.py#L147-L148)
- **Failure scenario:** When a human answers `"not_cabinet"` on an image's `count` question, `to_coco._image_counts` sets `human_counts = {"top": 0, "bottom": 0, "total": 0}`, `usable = False`, `scene = "no_cabinet"`, drops all model annotations, and marks `verified = True` even if crop questions were skipped or unanswered (`to_coco.py:137`, verified by `tests/unit/freestyle/test_quiz.py:364-375`). However, `build_human_gt` (`score_all.py:48-57`) checks `answered / len(qs) >= 0.8` across **all** questions in `qs` (count + every model crop proposal on that non-cabinet frame), excluding `"skip"` and missing answers. If a frame has 5+ model proposals and the human skips/leaves them unanswered after marking `not_cabinet`, `fraction <= 1/6 < 0.8` and `build_human_gt` excludes the verified negative image with `reason='below_80_percent'`.
- **Evidence (CPU repro via `/tmp/repro_score_all.py` Check 2):**
  ```text
  to_coco verified: True usable: False
  build_human_gt kept: 0 excluded: [{'image_id': 'g1/001.jpg', 'answered': 1, 'questions': 6, 'fraction': 0.16666666666666666, 'reason': 'below_80_percent'}]
  ```
- **Suggested fix:** In `build_human_gt` (`score_all.py:54`), bypass the `fraction < 0.8` filter when `not image['attributes'].get('usable', True)` and `image['attributes']['human_counts'] is not None` (i.e. a human-verified `not_cabinet` frame with `count` answered).

---

### 3. High — Partially answered (`0.8 <= fraction < 1.0`) and `counts_mismatch` images corrupt `class:{c}` MAE, `ap50`, and `select_thresholds`

- **Location:** [freestyle/experiments/score_all.py:54-64](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L54-L64), [freestyle/experiments/score_all.py:255-271](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L255-L271), [freestyle/experiments/score_all.py:285-286](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L285-L286), [freestyle/experiments/score_all.py:304-315](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L304-L315), [freestyle/experiments/score_all.py:335-343](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L335-L343)
- **Failure scenario:**
  1. **Deleted skipped boxes corrupt `ap50` and `class:{c}` MAE:** When an image has `0.8 <= fraction < 1.0` (e.g., 1 skipped crop out of 5), `build_human_gt` keeps the image (`line 54`) and strips the skipped `identity_source == 'model'` annotation (`lines 62-64`), while `counts['classes']` only sums human-answered crops (`to_coco.py:149-151`). A detector that accurately finds all 5 cartridges is penalized on `ap50` (the skipped cartridge box becomes a false positive if it scores above any kept box) and on `class:{c}` MAE (`errors[c] += 1`).
  2. **`counts_mismatch` / unverified `val` images skew `select_thresholds`:** In `select_thresholds` (`score_all.py:335-342`), `class_error` sums `abs(predicted[c] - actual[c])` over **all** `val` images, including unverified and `counts_mismatch` images where `sum(actual.values()) < human_counts['total']`. Step 1 picks `score_thr` to match `human_counts['total']`, so `rows` has more boxes than `actual`, biasing `score_thr` tie-breaking and `identity_thr` / `train_vlm_confidence_thr` selection toward predicting `unknown` if `actual` contains `unknown` or fewer known boxes.
  3. **`aggregate` averages `class:{c}` MAE over `counts_mismatch` / unverified `test` images:** While `exact_inventory` returns `None` when `not counts['verified'] or counts['counts_mismatch']` (`line 270`), `class_errors` (`line 269`) and `aggregate` (`lines 285-286`) still include those incomplete images, penalizing any detector (like `S2_c2_tile1280`) that detects cartridges missed by `bootstrap_v2`.
- **Evidence (CPU repro via `/tmp/repro_score_all.py` Check 3 & `/tmp/repro_gallery_and_bootstrap.py` Check 11):**
  ```text
  # Check 3: Perfect detector on 5/6 answered image (1 skipped crop)
  Perfect detector on 5/6 answered image -> ap50: 0.7999999999999998 class:sprite MAE: 1.0

  # Check 11: Val image with human_total=3, counts_mismatch actual={'sprite':1, 'unknown':2}
  With missing known crop & 2 unknown in mismatch val, picks id_thr: 0.5 class_err: 0
  ```
- **Suggested fix:**
  - Exclude images with `not counts['verified'] or counts['counts_mismatch']` from `class_errors` in `per_image` / `aggregate` and from `class_error` in `select_thresholds` (or require `100%` crop answers + review-app box fixes on `counts_mismatch` images before scoring class inventory / AP50).
  - Do not silently drop skipped boxes from `loc_gt` if the image itself is kept for `ap50`: either require all crop questions to be answered (`verified == True`) for `ap50` and `class:*`, or evaluate `ap50` / `class:*` on the verified non-mismatch subset and report `total_error` / `top_error` / `bottom_error` on all usable answered-count images.

---

### 4. High — `score_subset` omits predicted-only classes (including `'unknown'`) from `class:{c}` MAE, and `prepare_rows` fails to map generic `'cartridge'` to `'unknown'`

- **Location:** [freestyle/experiments/score_all.py:238-241](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L238-L241), [freestyle/experiments/score_all.py:285-286](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L285-L286), [freestyle/experiments/score_all.py:295](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L295) vs [docs/freestyle/ARCHITECTURE.md:53-54](file:///home/david/dev/vision-intern/docs/freestyle/ARCHITECTURE.md#L53-L54) and [freestyle/eval/metrics.py:590](file:///home/david/dev/vision-intern/freestyle/eval/metrics.py#L590)
- **Failure scenario:**
  1. `score_subset` (`line 295`) defines `classes = sorted({c for image in images for c in counts['images'][image['file_name']]['classes']})`. Any class predicted by the model that never appears in that subset's GT — e.g. `'unknown'` predictions on a sharp subset with 0 GT `'unknown'` boxes, or false-positive catalog classes — is completely omitted from `class:{c}` in `metrics` and `RESULTS_GT.md` (even though `per_image` computed its error inside `errors`!).
  2. `prepare_rows` (`lines 238-241`) downgrades `identity_score < identity_thr` to `'unknown'`, **but never maps `row['class'] == 'cartridge'` to `'unknown'`** (unlike `freestyle/pipeline.py:125` and `ARCHITECTURE.md:53`). If `own` or a fallback emits `'cartridge'`, it stays `'cartridge'`, failing `exact_inventory` against GT `'unknown'` and disappearing from `class:{c}` MAE.
- **Evidence (CPU repro via `/tmp/repro_score_all.py` Check 4):**
  ```text
  4a metric keys (sprite predicted, not in GT): ['class:unknown']
  4b exact_inventory for generic 'cartridge' vs GT 'unknown': 0.0 class:unknown MAE: 1.0
  ```
- **Suggested fix:**
  - In `prepare_rows` (`score_all.py:239`), change the condition to `if row['identity_score'] < identity_thr or row['class'] == 'cartridge': row['class'] = 'unknown'`.
  - In `score_subset` (`score_all.py:295`), construct `classes` from the full catalog + `'unknown'` (from `gt['categories']`) or at least the union of GT classes and predicted classes in `records` (`sorted({c for r in records if r['class_errors'] for c in r['class_errors']})`), matching `freestyle/eval/metrics.py:590`.

---

### 5. High — `to_coco._identity` silently promotes unconfirmed wrong model guesses to `identity_source="human"` on `"partial"`

- **Location:** [freestyle/quiz/to_coco.py:55-68](file:///home/david/dev/vision-intern/freestyle/quiz/to_coco.py#L55-L68), [freestyle/quiz/static/index.html:108](file:///home/david/dev/vision-intern/freestyle/quiz/static/index.html#L108), [freestyle/quiz/static/quiz.js:244](file:///home/david/dev/vision-intern/freestyle/quiz/static/quiz.js#L244)
- **Failure scenario:** In the quiz UI, `"Box cuts a cartridge (partial)"` submits `{"result": "partial"}` immediately with no class selection or confirmation step (`quiz.js:244, 299`). In `to_coco._identity` (`lines 60-67`), `"partial"` is not handled explicitly and falls through to line 67 (`return guess, HUMAN, result  # 'correct' keeps the model's guess, verified by a human`). Whenever the model guessed the wrong class (e.g. `hi-c` on a half-cut `coca-cola-part1`) and the human flags the box as `"partial"`, `to_coco.py` records the unconfirmed wrong model guess as `identity_source = "human"`, corrupting `counts['classes']`, `id_accuracy`, and `+val` galleries.
- **Evidence (CPU repro via `/tmp/repro_score_all.py` Check 5):**
  ```text
  partial answer -> category_id: 3 identity_source: human counts classes: {'hi-c': 1}
  ```
- **Suggested fix:** Either (a) map bare `{"result": "partial"}` to `UNKNOWN` (like `"two_plus"`, unless an explicit `"class"` is also provided in the answer payload), or (b) update the quiz UI so `"Box cuts a cartridge (partial)"` acts as a toggle/checkbox modifier attached to `✓ Correct` / class selection rather than a standalone button that implicitly accepts the model's guess. Also explicitly check `if result == "correct":` in `_identity` and reject unrecognized `result` values instead of falling through.

---

### 6. Medium — `select_thresholds` breaks ties toward `score_thr = 0.0` and `confidence = 0.0`

- **Location:** [freestyle/experiments/score_all.py:327-346](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L327-L346), [freestyle/experiments/score_all.py:478-480](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L478-L480)
- **Failure scenario:**
  1. `candidates = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9)` and `key = (best_total, class_error, score_thr, identity_thr)` picks the **lowest** `score_thr` whenever thresholds tie. Because RF-DETR runs (`C1`, `C2`, `D1`, `S1`, `S2`, `S3`, `S5`, `S7`) pre-filter inference at `0.3` (val min scores `0.302–0.313`) and zero-shot `B1`/`B3` start at `0.100`/`0.252`, `score_thr` values `0.0, 0.1, 0.2, 0.3` produce identical predictions on `val`. `select_thresholds` therefore selects **`score_thr = 0.0`** (confirmed in `runs/score_all_demo/results_gt.json` for `B1`, `B3`, `S2`, etc.). Any `test` prediction below `val`'s minimum score is then accepted unthresholded!
  2. The fixed `0.1` step grid skips `[0.25, 0.30]`, whereas `B3` (`val=[0.252, 0.464]`) and `B1` (`val=[0.100, 0.302]`) have steep count/F1 transitions inside `(0.1, 0.3)`.
  3. At `score_all.py:478`, `selection = (*key, confidence if confidence is not None else -1.)` orders `(score_thr, identity_thr)` **before** `confidence`, and breaks ties toward `confidence = 0.0` (the noisiest VLM labels) instead of higher confidence.
- **Evidence (CPU repro via `/tmp/repro_score_all.py` Check 6 & `runs/score_all_demo/results_gt.json`):**
  ```text
  selected key (best_total, class_err, score_thr, id_thr): (0, 0, 0.0, 0.0)
  test total_error due to score_thr=0.0: 2.0
  ```
- **Suggested fix:** Prefer the **highest** (or midpoint) `score_thr` on `(best_total, class_error)` ties (`-score_thr` in `key`), evaluate observed unique validation scores (or a finer grid), and break `train_vlm_confidence_thr` ties toward higher confidence (`-confidence`) rather than `0.0`.

---

### 7. Medium — `per_image` calls `match_boxes` without `pred_scores`, letting low-confidence duplicates steal GT matches

- **Location:** [freestyle/experiments/score_all.py:262](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L262) vs [freestyle/eval/matching.py:129-168](file:///home/david/dev/vision-intern/freestyle/eval/matching.py#L129-L168) and [freestyle/eval/metrics.py:472-479](file:///home/david/dev/vision-intern/freestyle/eval/metrics.py#L472-L479)
- **Failure scenario:** `per_image` calls `match_boxes([r['bbox'] for r in rows], gt_boxes, iou_thr=.5)` without `pred_scores=[r['score'] for r in rows]`. When `pred_scores is None`, `match_greedy` sorts eligible pairs by descending IoU rather than descending detection confidence score (`matching.py:150-168`). A low-confidence duplicate/fragment box (`score=0.35`, IoU `0.85`) steals the GT match from the primary high-confidence box (`score=0.95`, IoU `0.80`), changing `id_correct` relative to `freestyle/eval/metrics.py`.
- **Evidence (CPU repro via `/tmp/repro_score_all.py` Check 7):**
  ```text
  score_all per_image id_correct: 0 vs metrics.py top1_acc: 1.0
  ```
- **Suggested fix:** Pass `pred_scores=[float(r['score']) for r in rows]` to `match_boxes` at `score_all.py:262`.

---

### 8. Medium — `val` dual-use (5-group LOGO vs 6-group gallery) & `audit.json` subset leakage of `not_cabinet` frames

- **Location:** [freestyle/experiments/score_all.py:196-199](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L196-L199), [freestyle/experiments/score_all.py:465-490](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L465-L490)
- **Failure scenario:**
  1. **`val` dual-use & gallery size shift (no test leakage):** Verified in Repro Check 10 that `test` crops never enter `gallery` or `train_gallery`, and `validate_groups` + `gallery(emb, val_gt, image['group_id'])` properly hold out the query image's group during `val`. However: (a) `val` only has 6 groups (5 gallery groups per LOGO fold) while `test` uses all 6 `val` groups (`validation=False`), so `identity_thr` calibrated on 5-group LOGO is applied to a larger 6-group gallery at test time; (b) `val` simultaneously selects `score_thr`, `identity_thr`, and `train_vlm_confidence_thr` while supplying `+val` gallery crops (`knn`, `proto`, `knn_trainvlm+val`, `linear_trainvlm+val`); and (c) `data/splits.json` records a Hamming-2 near-duplicate pair across `train`/`val` (`94562f2f.../001.jpg` vs `d622b414.../055.jpg`), which makes `trainvlm` predictions on that `val` image optimistic.
  2. **`not_cabinet` in `partial/blurry`:** Subsets `sharp-full` and `partial/blurry` filter purely on `audit.json` (`scene_label`, `motion_blur`) without checking `im['attributes']['usable']`. A human-marked `not_cabinet` frame whose VLM audit had `motion_blur == 'severe'` enters `partial/blurry` (`line 489-490`). Also note `sharp-full` (`motion_blur in ('none', 'mild')`) differs from `DATA_AUDIT.md`'s `sharp_full >= 20` Laplacian sharpness bucket (though documented at `score_all.py:367`).
- **Evidence (CPU repro via `/tmp/repro_score_all.py` Check 9 & `/tmp/repro_gallery_and_bootstrap.py` Check 10):**
  ```text
  train_gallery indexes: [0] targets: ['sprite']
  val LOGO (hold out g_val1) indexes: [2]
  test gallery (all val) indexes: [1, 2]
  partial/blurry includes no_cabinet image: ['g1/001.jpg']
  ```
- **Suggested fix:** Gate `sharp` and `partial` subsets with `im['attributes'].get('usable', True)` (or skip `scene == 'no_cabinet'`), and document the 5-vs-6 group LOGO gallery shift and `train`/`val` near-duplicate caveat in `RESULTS_GT.md` / `results_gt.json['limitations']`.

---

### 9. Low — Input-fragility & schema mismatches (`assign_shelves` crash on degenerate boxes, `int(image_id)`, category ID `19` vs `91`, unused `loc_preds`)

- **Location:** [freestyle/experiments/score_all.py:95](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L95), [freestyle/experiments/score_all.py:230-248](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L230-L248), [freestyle/experiments/score_all.py:297-302](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L297-L302), [freestyle/experiments/score_all.py:441-447](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py#L441-L447); [freestyle/quiz/to_coco.py:130](file:///home/david/dev/vision-intern/freestyle/quiz/to_coco.py#L130)
- **Failure scenario:**
  1. **Degenerate box crash (Check 8):** If a detector emits a `w <= 0` or `h <= 0` prediction (especially when `score_thr = 0.0`), `geometry()` -> `assign_shelves()` (`shelves.py:49-50`) raises `ValueError: boxes must be finite and nondegenerate`, aborting the entire `score_all.py` run.
  2. **`int(pred['image_id'])`:** `grouped_predictions` (`line 95`) fails with `ValueError` if predictions use string `file_name`s as `image_id` (unlike `freestyle/eval/metrics.py:170-172`), or if `queue` lacks `coco_image_id` (`to_coco.py:139` falls back to `len(images) + 1`).
  3. **Category ID `19` vs `91`:** `score_all.py:441` builds `category_names` from `gt['val']['categories']` (`unknown = 91`), whereas `bootstrap_v2.json` and `runs/*/predictions.json` encode `unknown` as `category_id = 19`. It currently only resolves `19 -> 'unknown'` via the `.get(..., 'unknown')` fallback at `line 447`.
  4. **Minor bookkeeping:** `loc_preds` (`lines 297, 302`) is accumulated and never used; `to_coco.py:130` counts `"skip"` in `counts[image_id]["answered"]` whereas `score_all.py:48-52` excludes `"skip"` from `answered`; `to_coco.py:158` leaves `identity_visible=True` when a human selects `"unknown"` via `class-grid`.
- **Suggested fix:** Filter non-positive-area / non-finite predicted boxes when loading predictions (or guard `assign_shelves`), key predictions via `_resolve_image_key` / `file_name` fallback like `metrics.py`, remove dead `loc_preds`, and exclude `"skip"` from `counts['answered']` in `to_coco.py:130`.

---

## Verified Correct Behaviour (No Bug Found)

- **Last-answer-wins & `undo` (`to_coco.py:37-52`, `quiz/server.py:203-219, 298-309`):** Re-answering overwrites the prior answer for that `qid` (including re-answering with `skip`), `undo` atomically drops the last JSONL line and replays backwards, and `QuizStore` + `to_coco.load_answers` agree (confirmed in Check 12).
- **`Not a cartridge` & `Box covers 2+ cartridges` (`to_coco.py:61-66, 147-165`):** `not_cartridge` drops the box from GT and counts; `two_plus` maps to `unknown` with `box_covers_2_plus=True, identity_visible=False`, increments `crop_total` by 1, and triggers `counts_mismatch=True` when `human_total > crop_total` (so `exact_inventory` becomes `None`).
- **No test gallery leakage & group bootstrap clustering (`score_all.py:121-162, 306-318, 349-356`):** `validate_groups` overrides `to_coco`'s `group_id_for` (`island-cabinet-01/<uuid>`) with the authoritative `splits.json` group (`<uuid>` / `local/downloads/frame.jpg`), `gallery` and `train_gallery` never admit `test` crops, and `compute_group_bootstrap_ci` clusters test images by `group_id` (e.g. `bb2c251e/000` and `001` resampled together, 11 groups for 12 test images) with `loc_records` aligned 1-to-1 with `images`.

---

## STATUS

Review completed. No repository source files modified; wrote review report to [docs/freestyle/reviews/agy-review-5-score-all.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/agy-review-5-score-all.md) and CPU repro scripts under `/tmp`.

## FILES

- Created: [docs/freestyle/reviews/agy-review-5-score-all.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/agy-review-5-score-all.md)
- Created (repro scripts in `/tmp`): `/tmp/repro_score_all.py`, `/tmp/repro_gallery_and_bootstrap.py`
- Read (in-scope): [freestyle/experiments/score_all.py](file:///home/david/dev/vision-intern/freestyle/experiments/score_all.py), [freestyle/quiz/to_coco.py](file:///home/david/dev/vision-intern/freestyle/quiz/to_coco.py), [freestyle/quiz/build_queue.py](file:///home/david/dev/vision-intern/freestyle/quiz/build_queue.py), [freestyle/quiz/server.py](file:///home/david/dev/vision-intern/freestyle/quiz/server.py), [freestyle/quiz/static/index.html](file:///home/david/dev/vision-intern/freestyle/quiz/static/index.html), [freestyle/quiz/static/quiz.js](file:///home/david/dev/vision-intern/freestyle/quiz/static/quiz.js), [freestyle/count/pairs.py](file:///home/david/dev/vision-intern/freestyle/count/pairs.py), [freestyle/count/shelves.py](file:///home/david/dev/vision-intern/freestyle/count/shelves.py), [freestyle/count/aggregate.py](file:///home/david/dev/vision-intern/freestyle/count/aggregate.py), [freestyle/eval/matching.py](file:///home/david/dev/vision-intern/freestyle/eval/matching.py), [freestyle/eval/metrics.py](file:///home/david/dev/vision-intern/freestyle/eval/metrics.py), [freestyle/eval/evaluate.py](file:///home/david/dev/vision-intern/freestyle/eval/evaluate.py), [freestyle/identify/classify.py](file:///home/david/dev/vision-intern/freestyle/identify/classify.py), [freestyle/identify/cli.py](file:///home/david/dev/vision-intern/freestyle/identify/cli.py), [freestyle/experiments/entry.py](file:///home/david/dev/vision-intern/freestyle/experiments/entry.py), [freestyle/experiments/schema.py](file:///home/david/dev/vision-intern/freestyle/experiments/schema.py), [freestyle/pipeline.py](file:///home/david/dev/vision-intern/freestyle/pipeline.py), [freestyle/catalog.py](file:///home/david/dev/vision-intern/freestyle/catalog.py), [freestyle/review/server.py](file:///home/david/dev/vision-intern/freestyle/review/server.py), [tests/unit/freestyle/test_score_all.py](file:///home/david/dev/vision-intern/tests/unit/freestyle/test_score_all.py), [tests/unit/freestyle/test_quiz.py](file:///home/david/dev/vision-intern/tests/unit/freestyle/test_quiz.py), [docs/freestyle/ARCHITECTURE.md](file:///home/david/dev/vision-intern/docs/freestyle/ARCHITECTURE.md), [docs/freestyle/QUIZ_APP.md](file:///home/david/dev/vision-intern/docs/freestyle/QUIZ_APP.md), [docs/freestyle/GT_PROTOCOL.md](file:///home/david/dev/vision-intern/docs/freestyle/GT_PROTOCOL.md), [docs/freestyle/REPORT.md](file:///home/david/dev/vision-intern/docs/freestyle/REPORT.md), [docs/freestyle/reviews/agy-review-2.md](file:///home/david/dev/vision-intern/docs/freestyle/reviews/agy-review-2.md), plus JSON/YAML/NPZ metadata under `data/` and `runs/` (no images opened).

## CHECKS

- `uv run pytest tests/unit/freestyle/test_score_all.py tests/unit/freestyle/test_quiz.py -q` → `32 passed in 2.86s`.
- `PYTHONPATH=. uv run python /tmp/repro_score_all.py` → exit code `0`, `ALL 9 REPRO CHECKS PASSED.` (confirmed Findings 1, 2, 3, 4, 5, 6, 7, 8b, 9).
- `PYTHONPATH=. uv run python /tmp/repro_gallery_and_bootstrap.py` → exit code `0`, `ALL GALLERY/BOOTSTRAP/E2E REPRO CHECKS PASSED.` (confirmed Findings 1, 3, 8a, plus last-answer-wins/undo/gallery-isolation behavior).

## NOTES

- No images under `data/` or `runs/` were opened; only JSON/YAML/NPZ/Markdown metadata was inspected.
- No git state-changing commands were executed; no live endpoints were contacted; no source code was modified.
- Note that `freestyle/experiments/score_all.py` already handles `tests/unit/freestyle/test_score_all.py` cleanly in the current tree (`32 passed` across `test_score_all.py` and `test_quiz.py`), even though [docs/freestyle/REPORT.md:326](file:///home/david/dev/vision-intern/docs/freestyle/REPORT.md#L326) mentions an earlier in-progress failure.
