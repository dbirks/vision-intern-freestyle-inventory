# Coca-Cola Freestyle Cartridge Recognition — Final Project Report

**Branch:** `freestyle-cartridges`  
**Contract:** [`docs/freestyle/ARCHITECTURE.md`](ARCHITECTURE.md)  
**Scope & provenance:** Clean-room implementation built on Vision Intern (`workflows/`, `tools/`) and `data/raw/freestyle-vision-data/`, using the home Kubernetes GPU cluster (`k8s/`) and the LLM Collective OpenAI-compatible endpoint (`freestyle/collective.py`).

> [!IMPORTANT]
> **Human ground truth is pending.** Every quantitative experiment result currently in `runs/*/metrics.json` and `runs/*/failures/failures.json` is scored against **model-generated bootstrap proposals** (`data/labels/bootstrap_v2.json`), **not** human-verified ground truth. Sections marked `<!-- GT-PENDING: ... -->` await completion of the phone-first GT quiz and the resulting `docs/freestyle/RESULTS_GT.md` output from `freestyle.experiments.score_all`.

---

## 1. Executive Summary

The goal of this project is to locate, identify (across 18 active catalog SKUs plus `unknown`), and count Coca-Cola Freestyle ingredient cartridges from photographs captured by an in-door fisheye cabinet camera (`docs/freestyle/ARCHITECTURE.md:3-6`).

<!-- GT-PENDING: Insert the final selected architecture (localizer + identifier + pair-rule configuration) and headline human-GT test metrics (AP50, ID Top-1, Total MAE, Class MAE, Exact Inventory Rate, all with 95% group-bootstrap CIs) from docs/freestyle/RESULTS_GT.md once human quiz answers are complete and freestyle.experiments.score_all has run. -->

**Current evidence so far (pre-human-GT):**

1. **Data reality:** Out of 1,383 declared server frames, only **54** survived retention (~96.1% retention loss), joined by **15** local workstation frames (**69 images** across **56 independent groups**). Removing 29 `no_cabinet`/`occluded` negatives leaves **40 usable cabinet frames** (`29` `cabinet_full`, `11` `cabinet_partial`), all from a single physical cabinet and stocking arrangement (`docs/freestyle/DATA_AUDIT.md:31-56, 305-318`).
2. **Bootstrap & fine-print limits:** Vision Intern's `label_dataset` + `judge_labels` + a new two-stage VLM identifier (`david-birks/inkling-2-mini` labeller/identifier, `qwen3-8-flash-next` judge) produced **563 proposals** across 29 unique cabinet scenes (`runs/bootstrap_v2/summary.md:8`). Two-stage identification reduced `unknown` labels from **77.8%** (438/563) in `bootstrap_v1` to **45.5%** (256/563) in `bootstrap_v2`, while revealing that printed Part 1 / Part 2 numerals are legible in only **2 of 123** (1.6%) two-part brand crops (`runs/bootstrap_v2/summary.md:49, 70-74`).
3. **Trained tiled RF-DETR vs zero-shot detectors (vs model labels, NOT GT):** In GPU Window #1 on an RTX 5090, class-agnostic tiled RF-DETR trained on 10 frames of model labels (`S2_c2_tile1280` at **0.710 AP50**, `C2_rfdetr_medium_agnostic_siglip2` at **0.697 AP50**, `C1_rfdetr_small_agnostic_dinov2` at **0.694 AP50**) outperformed zero-shot open-vocabulary detectors (`B3_gdino_dinov2` at **0.232 AP50**, `B1_owlv2_dinov2` at **0.181 AP50**) by ~3x, because zero-shot models merge adjacent cartridges and stacked NNS tiers (`devlog/content/2026-11-26-rfdetr-beats-zero-shot.md:12-23`, `runs/*/metrics.json`).
4. **Two label spaces, one decision left open:** A second detector family was trained on the cartridges' printed **front faces** instead of the whole leaning box (`faces_v1` → cleaned `faces_v2` → GPU runs `G1`/`G2`). Faces overlap less (mean neighbour max-IoU **0.119** vs **0.201** for whole boxes) and carry readable Part numerals (59 numerals on 597 faces vs 2 of 123 whole-frame crops), and the face detectors land in the same metric range as the whole-cartridge detector (`G2` total-count MAE **3.333** vs `S2` **3.583**, `G1` AP50 **0.663** vs `S2` **0.710**) — but they are scored against *their own* model labels, so no GPU variant can settle the choice. Only the human counts can (`docs/freestyle/FAILURES.md`, §8.9 here).
5. **GT quiz & scoring readiness:** The phone-first GT quiz queue is generated (`data/gt/quiz_queue.json`: **333** questions over **18** images — 188/`test`, 145/`val`; 18 `count` + 315 `crop`), `freestyle/quiz/server.py` is implemented, `freestyle/experiments/score_all.py` has a dry-run harness verified in `runs/score_all_demo/RESULTS_GT.md` (the historical test snapshot is noted in Section 13), and `scripts/finalize_after_gt.sh` now holds the exact post-GT sequence. Human answers so far: **6 of 333** questions (one `not_cabinet` negative plus five `skip`s on a single near-blank test frame), i.e. the quiz has been smoke-tested, not answered.

---

## 2. Problem & Data

### 2.1 Corpus & Archive Contents

Per `docs/freestyle/DATA_AUDIT.md:31-46`, the audited corpus (`freestyle/audit.py`) comprises **69 photographs** partitioned into **56 independent groups**:

| Metric | Count | Source |
|---|---:|---|
| Total audited photographs | 69 | `docs/freestyle/DATA_AUDIT.md:35` |
| Server frames (`images/island-cabinet-01/...`) | 54 | `docs/freestyle/DATA_AUDIT.md:36` |
| Local workstation frames (`images/local/...`) | 15 | `docs/freestyle/DATA_AUDIT.md:37` |
| Independent groups (split unit) | 56 | `docs/freestyle/DATA_AUDIT.md:38` |
| Native `1600x1200` (UXGA) | 53 | `docs/freestyle/DATA_AUDIT.md:39` |
| Native `2048x1536` (QXGA) | 15 | `docs/freestyle/DATA_AUDIT.md:40` |
| Native `640x480` (VGA) | 1 | `docs/freestyle/DATA_AUDIT.md:41` |

`data/local_images.csv` contained swapped/incorrect metadata (`width=769` for every local row and true width stored in `height`); `freestyle/data.py` therefore inspects image dimensions directly from disk and preserves CSV fields as `csv_width`/`csv_height` (`docs/freestyle/DATA_AUDIT.md:43-46`).

The target label space comes from `data/raw/freestyle-vision-data/data/catalog_items.json` (`active: true` rows only): **18 active catalog classes** (`nns-sweetener`, `ingredient-aa`, `ingredient-cc`, `fanta-orange`, `hi-c`, `sprite`, `flavor-orange`, `flavor-vanilla`, `flavor-lemon`, `flavor-cherry`, and paired Part 1 / Part 2 classes for `coca-cola`, `diet-coke`, `dr-pepper`, and `coke-zero`), plus `cartridge` (class-agnostic detection super-class) and `unknown` (`docs/freestyle/ARCHITECTURE.md:25-31`, `freestyle/eval/failures.py:28-49`).

### 2.2 Audit: Scene Triage, Sharpness, Rotation, Near-Duplicates

`freestyle/audit.py` computed full-resolution and 512 px Laplacian variance, luma, pHash (`hash_size=8`), and Collective VLM scene triage (`qwen3-8-flash-next`, temperature 0, 1024 px longest side), with 4 images (6 fields) manually corrected in `freestyle/audit_overrides.json` (`docs/freestyle/DATA_AUDIT.md:25-29, 279-301`):

- **Scene triage:** `cabinet_full` = **29**, `cabinet_partial` = **11**, `no_cabinet` = **26**, `occluded` = **3**, `unknown` = **0** (`docs/freestyle/DATA_AUDIT.md:50-56`).
- **People & rotation:** `person_visible` in **38 / 69** frames (`cabinet_full` 16, `cabinet_partial` 9, `no_cabinet` 12, `occluded` 1). Rotations: **3** `upside_down` (`ab8c7fc6.../043`, `de19c989.../032`, `d7c48eea.../027`) and **1** `rotated_90` (`c40dc230.../019`). Motion blur (`none` / `mild` / `severe`) = **6 / 48 / 15** (`docs/freestyle/DATA_AUDIT.md:58-63`).
- **Sharpness & exposure:** Full-resolution Laplacian variance quantiles: `p0 = 0.86`, `p25 = 7.47`, `p50 = 14.5`, `p75 = 20.64`, `p100 = 75.38` (`docs/freestyle/DATA_AUDIT.md:73-79`). Using corpus buckets (`sharp >= 20`, `soft 8-20`, `blurry < 8`), only **21 of 40** usable cabinet frames reach `sharp >= 20` (highest candidate frame = **26.1**; `cabinet_partial` is 0 sharp / 4 soft / 7 blurry). Luma ranges from **5.0 to 146.3** (mean **86.0**): a dim interior fighting bright shelf LEDs (`docs/freestyle/DATA_AUDIT.md:85-95`).
- **Near-duplicates:** At pHash Hamming distance `<= 8`, **32 of 69** images fall into **9 clusters** (`c00`–`c08`), led by `c00` (14 `cabinet_full` images: the 12-frame `local/blur` burst plus 2 server frames at a matching door angle) (`docs/freestyle/DATA_AUDIT.md:97-118`).

### 2.3 Frozen Grouped Split & Negatives

`freestyle/make_splits.py` assigns splits by **group** (server event UUID, `local/blur` for the 12-frame burst, or one group per remaining local image) so no opening straddles a boundary (`docs/freestyle/DATA_AUDIT.md:138-168`):

| Split | Groups | Images | Cabinet Frames | Notes |
|---|---:|---:|---:|---|
| `train` | 11 | 22 | 22 | 10 server frames + forced `local/blur` 12-frame burst (`20` full, `2` partial) |
| `val` | 6 | 6 | 6 | Includes forced `local/downloads/frame.jpg` (best full view, `sharp_full=24.9`) (`4` full, `2` partial) |
| `test` | 11 | 12 | 12 | Stratified 40% hard slice: `5` full, `7` partial; `6` mild / `6` severe blur; `3` `>=20` sharp; `1` 90°-rotated |
| `exclude` | 28 | 29 | 0 | `26` `no_cabinet` + `3` `occluded` negatives (incl. `bb2c251e.../002.jpg` inside a test group) |

- **Leakage checks:** `freestyle/make_splits.py --check` verifies **0 near-duplicates touching `test`** (`<= 8` bits) (`docs/freestyle/DATA_AUDIT.md:124-127`). Two cross-opening pairs cross `train`/`val` (`94562f2f.../001` ~ `d622b414.../055`, Hamming 2; `ab8c7fc6.../043` ~ `de19c989.../032`, Hamming 3 — both 180° upside-down).
- **Freeze discipline:** `data/splits.frozen.json` is authoritative and copied verbatim into `data/splits.json`. Subsequent human rotation overrides in `audit_overrides.json` caused the un-frozen generator to drift on 4 groups (`frozen_drift`), but the frozen split did not move (`docs/freestyle/DATA_AUDIT.md:154-168`).

### 2.4 Data Limitations

1. **96.1% retention loss & tiny sample:** Only **54 of 1,383** declared server frames survived retention (`docs/freestyle/DATA_AUDIT.md:315`), mostly deployment-selected keyframes, leaving **40 usable cabinet frames** (`54 + 15 = 69` total). Twelve test images across 11 groups yield wide group-bootstrap CIs.
2. **Single cabinet, single arrangement:** Every frame depicts the same physical cabinet and cartridge layout (`docs/freestyle/DATA_AUDIT.md:305-314`). Models can score well by memorizing slot positions; cross-cabinet generalization and stocking-change sensitivity are completely unmeasured.
3. **Fisheye distortion & dim lighting:** Under the ~160° in-door fisheye lens, bottom-shelf cartridges lean outward and partially occlude neighbours (`docs/freestyle/GT_PROTOCOL.md:12-14`, `docs/freestyle/llm_collective.md:43`).
4. **Blur & resolution floor:** No candidate frame exceeds `sharp_full = 26.1`; most `1600x1200` frames are too soft for small text (`docs/freestyle/DATA_AUDIT.md:328-330`).
5. **Hard test slice & 180° blind spot:** `test` over-represents partial (`7/12`) and severely blurred (`6/12`) frames by design (`docs/freestyle/DATA_AUDIT.md:201-206, 340-351`). The two upside-down (`180°`) frames sit in `train` and `val`, while `test` covers `90°` rotation only.
6. **Missing capture metadata:** Raw `GAPS.md` notes no timestamps, firmware attribution, or door-angle offsets (`docs/freestyle/DATA_AUDIT.md:331-336`).

---

## 3. Ground Truth Protocol

Model-generated labels (`data/labels/*.json`) are **never** treated as ground truth (`docs/freestyle/ARCHITECTURE.md:13-14`, `.claude/agents/freestyle-cartridge-runner.md:56-58`). Two local web tools produce human-verified GT under `data/gt/`:

1. **Full-resolution Review App (`freestyle/review/server.py`, `docs/freestyle/REVIEW_APP.md`):** Displays raw full-resolution images with pan/zoom, pre-filled from proposals, plus a `?mode=grid` crop-audit sheet. Humans create/edit/delete boxes, set `truncated`/`occluded`/`identity_visible` flags, inspect live per-class counts, and press `v` (`verified: true`) with atomic `fsync` + `rename` saves. Unchanged accepted boxes record `source = "human-verified:<orig>"`; edited/drawn boxes record `source = "human"`.
2. **Phone-First GT Quiz (`freestyle/quiz/`, `docs/freestyle/QUIZ_APP.md`):** Designed to annotate `test` and `val` from a phone in ~20 minutes (`build_queue.py` -> `server.py` -> `to_coco.py`). Queue order is deterministic: `test` then `val`, and within each image a `count` question (top/bottom shelf counts on a numbered `<=1280 px` overlay) followed by `crop` questions left-to-right, top shelf before bottom shelf (using `freestyle.count.shelves.assign_shelves`). Each crop is rendered in memory from **original pixels** (+15% padding, up to 6x upscale) with big tap targets for catalog classes, two-part `Part 1`/`Part 2`/`Can't tell` shortcuts, `Can't tell (unknown)`, `Not a cartridge`, `Box covers 2+ cartridges`, `Box cuts a cartridge (partial)`, and `Not a usable cabinet view` (converting the image to a `no_cabinet` negative).

**What is human vs model in the quiz path (`freestyle/quiz/to_coco.py`, `docs/freestyle/QUIZ_APP.md:91-110`):**
- **Human:** Per-image `human_counts = {top, bottom, total}`, usability (`no_cabinet` triage), and per-crop identities (`attributes.identity_source = "human"`, `source = "quiz:human"`), plus multi-box / partial-box flags.
- **Model:** Bounding-box geometry originates from `data/labels/bootstrap_v2.json` (`attributes.box_source = "model"`). Consequently, `to_coco` emits `<split>_counts.json` flagging `counts_mismatch` wherever human total counts disagree with answered crop counts—identifying frames with missed/merged model boxes that need box correction in `review.server`.
- **Protocol rules (`docs/freestyle/GT_PROTOCOL.md:7-38`):** One box per visible physical cartridge (two-high NNS stacks get two boxes); never guess `Part 1`/`Part 2` or `AA`/`CC` from shelf order when pixels do not support it (mark `unknown`); `score_all.py:31-72` (`build_human_gt`) requires `>= 80%` answered questions and a non-null `human_counts` per image, and strips any annotation lacking `identity_source == "human"`.

**Current GT status:** `data/gt/quiz_queue.json` is built (**333** questions across **18** images: 12 test images / 11 groups first, then 6 val images; 18 `count` + 315 `crop`). `data/gt/quiz_answers.jsonl` holds **6** answers — the 2026-11-27 smoke pass over one near-blank test frame (`10455345…/013.jpg`: `not_cabinet` on the count question, 5 `skip`s), which exercised the skip-cascade path added in `b29ada6`. No `data/gt/{val,test}.json` export exists yet, so no human GT number appears anywhere in this report. `scripts/finalize_after_gt.sh` refuses to score anything until ≥80 % of each split's questions are answered (`<!-- GT-PENDING: Verify quiz completion counts, mismatch list, and human GT dataset statistics in data/gt/{val,test}.json and docs/freestyle/RESULTS_GT.md. -->`).

---

## 4. System Overview

```mermaid
flowchart TD
    RAW["Raw Corpus (69 imgs, 56 groups)"] --> AUDIT["Audit & Triage (freestyle/audit.py)"]
    AUDIT --> SPLITS["Frozen Group Splits (freestyle/make_splits.py)"]

    subgraph BOOT["VLM Bootstrap (Collective API, no GPU)"]
        LBL["Vision Intern label_dataset (Inkling full + 1280 tiles)"] --> WBF["WBF Fusion (freestyle/geometry.py)"]
        WBF --> JDG["Vision Intern judge_labels (Flash Next crop-mode)"]
        JDG --> IDVLM["2-Stage VLM Crop ID (freestyle/vlm_identify.py)"]
    end

    SPLITS --> BOOT
    IDVLM --> PROPOSALS["data/labels/bootstrap_v2.json (563 proposals)"]

    PROPOSALS --> QUIZ["Phone GT Quiz / Review App (freestyle/quiz, review)"]
    QUIZ --> GT["Human GT: data/gt/val.json & test.json"]

    subgraph K8S["k8s GPU Experiments (freestyle/experiments/entry.py)"]
        STAGE["k8s/stage-data.sh -> /data-ro PVC"] --> TRAIN["RF-DETR / YOLO Tiled Train OR Zero-Shot Proposals"]
        TRAIN --> EMB["Crop Embeddings (DINOv2 / SigLIP2)"]
        EMB --> INF["Tiled Inference -> val & test predictions.json"]
    end

    PROPOSALS --> K8S
    SPLITS --> K8S

    subgraph IDOPT["Identification Options (score_all / pipeline)"]
        OPT1["own: D1 multiclass head"]
        OPT2["vlm: 2-stage Collective crop ID"]
        OPT3["knn / proto / linear: val or trainvlm embeddings"]
    end

    INF --> IDOPT
    IDOPT --> COUNT["Shelf Fit + Aggregate + Pair Rule (shelves.py, aggregate.py, pairs.py)"]
    COUNT --> EVAL["Eval & Failures (evaluate.py, failures.py, score_all.py, report.py)"]
    GT --> EVAL
    INF --> BUNDLE["Frozen Bundle & CLI (artifacts.py -> cli.py count/fuse)"]
```

**Pipeline stages in code:**
1. **Bootstrap (`freestyle/bootstrap_labels.py:142-248`):** Runs `workflows.vlm_label.label_dataset` (`david-birks/inkling-2-mini`, full frame + 1280 px tiles, 0.2 overlap, PNG, `max_dimension=None`), merges via `weighted_box_fusion` (`fusion_iou=0.55`), judges via `workflows.vlm_judge.judge_labels` (`qwen3-8-flash-next`, `judge_mode="crops"`, `judge_batch_size=8`, `max_tokens=4096`), and identifies crops via `VLMCropIdentifier`.
2. **k8s Detector & Embeddings (`freestyle/experiments/entry.py:401-445`):** Stage pipeline (`train` -> `gallery` -> `embeddings` -> `val` -> `test`) exporting 0.25-overlap masked shelf tiles (`freestyle/train/export_tiles.py`), training `RFDETR{Small,Medium,Large}` or YOLO (`freestyle/train/train_detector.py`), extracting `facebook/dinov2-base` or `google/siglip2-base-patch16-224` crop embeddings (`embeddings.npz`), and emitting COCO `predictions.json` via `TiledDetector` (`freestyle/infer/detector.py`) or zero-shot `ProposalRunner` (`freestyle/propose/backends.py`).
3. **Identification & Counting (`freestyle/pipeline.py`, `freestyle/count/`):** Supports `none`, `own` (D1 multiclass), `vlm` (two-stage Collective VLM), or `knn`/`proto`/`linear` over train-only or val-only galleries (`freestyle/identify/classify.py`). `assign_shelves` (`freestyle/count/shelves.py:37-109`) clusters two-high stacks and fits up to 4 quadratic row curves selected by BIC; `aggregate_counts` (`freestyle/count/aggregate.py:8-43`) downgrades `id_score < identity_thr` (`0.5`) to `unknown` and supports optional `neighbour_vote` (`off`/`report`/`apply`); `infer_pairs` (`freestyle/count/pairs.py:33-112`) optionally resolves two-part brands; `fuse_counts` (`freestyle/count/aggregate.py:46-75`) combines multi-frame bursts via `max`, `median`, or `weighted_mode`.
4. **Evaluation & Packaging (`freestyle/eval/`, `freestyle/artifacts.py`, `freestyle/cli.py`):** Val-only best-F1 threshold selection (`evaluate.py`), 1,000-replicate grouped bootstrap 95% CIs (`metrics.py`), FN/FP/ID/LOC diagnostics (`failures.py`), human-GT matrix scoring (`score_all.py`), and SHA-256-verified frozen bundle export (`artifacts.py` -> `freestyle.cli count|fuse|bundles`).

---

## 5. Vision Intern Usage & Extensions

### 5.1 Core Dense-Scene Commits in Vision Intern

Three commits added opt-in dense-scene support to upstream Vision Intern (`tools/`, `workflows/`) while keeping default behaviour intact (`docs/freestyle/VISION_INTERN_MAP.md:7-12, 406-417`):

| Commit | Summary | Concrete Core Changes |
|---|---|---|
| `fb8c805` | Opt-in dense-scene support | Added `max_dimension`, `image_format`, `jpeg_quality` (`tools/vlm_client.py`); `tile`, `tile_overlap`, `max_tokens`, edge-box rejection, `finish_reason == "length"` depth-1 quadrant auto-split (`tools/vlm_detect.py`); `judge_batch_size` and `judge_mode="crops"` 15%-padded ≤256 px sheets (`workflows/vlm_judge.py`); `label_aliases`, `keep_unmatched`, `recursive`, `image_paths` (`workflows/vlm_label.py`); `examples/dense-scenes/README.md`, `tests/unit/test_dense_scenes.py`. |
| `289cc0e` | Sampling, extra_body & logging | Added real `temperature`, `top_p`, `seed`, `extra_body` (`tools/vlm_client.py`); `VLMResponse` (`str` subclass with `finish_reason` and `usage`); JSONL `raw_response_log` auditing across `tools/vlm_detect.py` and `workflows/vlm_label.py`. |
| `4b96e5f` | Review #3 fixes | Tile merge prefers full boxes over clipped edge fragments (`tools/bbox_utils.py`); label normalization runs before `n_samples` consensus and `nms` (`tools/vlm_detect.py`); crop-mode sheets draw proposed boxes/labels and guard OOB/inverted boxes (`workflows/vlm_judge.py`); `transformers` backend derives `finish_reason` for truncation detection and `raw_response_log` uses a thread lock (`tools/vlm_client.py`); tolerant `raw_decode` + per-object salvage in `_parse_verdicts` (`workflows/vlm_judge.py`); symlink-safe `_label_local` (`workflows/vlm_label.py`); `tests/unit/test_dense_scenes_review3.py`. |

### 5.2 Upstream Gaps Bridged by `freestyle/`

From `docs/freestyle/VISION_INTERN_MAP.md` (§3, §5, §6a, §8):

- **Resolution & judge crop ceiling (`L1`–`L2`):** Core `vlm_client` defaults still resize to `1280 px` JPEG q85 (`tools/vlm_client.py:53-81`), and `judge_labels` crop sheets thumbnail crops to `≤256 px` (`workflows/vlm_judge.py:380`). Every Freestyle caller passes `max_dimension=None, image_format="png"`, and crop identification (`freestyle/vlm_identify.py`, `freestyle/identify/`) operates on full-resolution original crops.
- **Judge token-budget trap & single-shot scores (`L3`–`L4`):** `workflows/vlm_judge.py:502` defaults `max_tokens` to `1024` unless `"lfm3-vl"` is in `model_id`; `freestyle/bootstrap_labels.py:213` passes `max_tokens=4096` explicitly. Single-shot `vlm_detect` emits no `score`, so `freestyle/bootstrap_labels.py:188` generates `support` via multi-run WBF (`fuse_runs`).
- **Same-label-only `nms` & missing labels (`L5`, `L7`):** Core `nms` (`tools/bbox_utils.py:223-241`) only suppresses same-label overlaps; `freestyle/geometry.py` adds class-agnostic `nms`, containment suppression, and WBF across tiles/runs, while `freestyle/catalog.py` pins the 18 catalog IDs and aliases.
- **Training resolution, negatives, and destructive augmentations (`L9`–`L10`):** `workflows/train_rfdetr.py:252, 334-352, 522` hardcodes `rf-detr-large` at `704x704`, drops zero-box negative frames, and enables `HorizontalFlip` + `HueSaturationValue` by default (breaking shelf-side and brand-colour identity). `freestyle/train/` invokes `rfdetr` / `ultralytics` directly with explicit `resolution` (`512` / `576`), masked shelf tiles, and non-flip/non-hue augmentations.
- **HF Hub coupling, split leakage, and missing metrics (`L12`–`L13`, §5):** Upstream `jobs/*` clone from GitHub and push to Hugging Face Hub, and `workflows/train_rfdetr.py` only logs torchmetrics COCO mAP (`max_det=100`, no prediction export). `freestyle/` replaces this with local `data/splits.json`, `k8s/` jobs, `TiledDetector.predict_coco`, `freestyle/eval/` (ID Top-1, Macro-F1, Count MAE, Exact Inventory, grouped bootstrap CIs), and `freestyle/count/`.
- **Agent orchestration (`.claude/agents/freestyle-cartridge-runner.md`):** Defines a two-phase autonomous runner replacing upstream's HF-coupled `.claude/agents/vision-e2e-runner.md`. **Phase 1** runs `audit` -> `make_splits` -> Collective `bootstrap_labels` and **halts** at the human GT gate (`review.server` / quiz). **Phase 2** runs only once human-verified `data/gt/` exists: `k8s/stage-data.sh` -> `freestyle.experiments.launch` -> `eval` / `failures` / `report` -> `freestyle.artifacts export` -> `freestyle.cli count|fuse`, enforcing clean-room, private-data, k8s-only-GPU, Collective-only-VLM, val-only-threshold, and train-only-gallery invariants.

---

## 6. LLM Collective Findings

`freestyle/collective.py` reads `~/.pi/agent/models.json` (`provider: collective`) and `${LLMC_API_KEY}` at runtime without printing or persisting secrets, retrying 429/5xx/timeouts with exponential backoff + jitter up to `retry_budget_s=600` (`docs/freestyle/llm_collective.md:8-10, 93`). Probing `https://llm-collective.e-gineering.com/v1` on 3 real frames (`local/downloads/frame.jpg` `2048x1536`, `50828e4b.../008.jpg` `2048x1536`, `2ee373c6.../006.jpg` `1600x1200`) established (`docs/freestyle/llm_collective.md:5-106`, `runs/probe-collective/results.json`):

| Model (`/v1/models`: 13 IDs total) | Vision | Latency / Call | Token Usage (`frame.jpg` full -> 1024) | Reliability & Caveats |
|---|---|---|---|---|
| `david-birks/inkling-2-mini` | Yes | 0.4–5.4 s | Count 3.1k -> 1.3k; Boxes ~4.1k -> 2.6k | Fast, reliable default; follows prompted coordinate convention. |
| `qwen3-8-flash-next` | Yes | 13–107 s | Count 4.0k -> 2.8k; Boxes ~5.1k -> 2.7k | Thinking model; 429-throttled heavily under load (502.9 s tiny probe; >14 min hang; 530 s bootstrap call). |
| `david-birks/qwen3-8-flash-next` | Yes | 1.7–46 s | Count ~4.0k -> 2.1k; Boxes ~5.1k -> 2.7k | Thinking model; more stable than unprefixed `qwen3-8-flash-next`. |
| `inkling-2-mini`, `justin-dickey/inkling-2-mini`, `auto` | Yes | Fast | — | Fallbacks / router. |
| `qwen3-8-27b` | **No** | — | — | **HTTP 404** (`model "qwen3-8-27b" is not currently available`). |

**Key empirical findings (`docs/freestyle/llm_collective.md:26-105`):**
- **Coordinate trap:** `qwen3-8-flash-next` ignores pixel-coordinate instructions and **always** returns `0–1000` normalized coordinates, whereas `inkling-2-mini` obeys whichever convention is requested. Furthermore, `max_side` is a request-size knob only—neither model rescales coordinates when `max_side=1024` is used. Therefore all VLM grounding/bootstrap code prompts and parses `0–1000` normalized coordinates (`freestyle/vlm_ground.py`).
- **Thinking-model budget trap:** At `max_tokens=64`, `flash-next` spends its budget on `<think>` tokens and returns empty text (`finish_reason=length`). It requires `max_tokens >= 256` for short answers and `4096–8192` for box lists.
- **Max image size & quality:** Synthetic images up to `4096x3072` (`263 KB`, `~5.6 s`) were accepted by `inkling-2-mini`. On `frame.jpg` (~27 expected cartridges), top-shelf boxes were tight and accurate (17 top / 9–10 bottom), while bottom-shelf fanned/occluded cartridges produced overlapping or merged boxes. All 3 vision models scored 2/2 on two hand-picked clear crops (`ingredient-aa`, `coca-cola-part1`).
- **Role split:** Because `qwen3-8-flash-next` hit 429 capacity during labelling (`~530 s/call` vs `~7 s` for Inkling), `configs/bootstrap/default.yaml` uses `david-birks/inkling-2-mini` as primary labeller + identifier and reserves `qwen3-8-flash-next` for light crop-mode judging (`~30 s/image`).

---

## 7. Infrastructure

### 7.1 Kubernetes GPU Cluster, Modes & Benchmarks

All heavy GPU jobs run via `k8s/submit.sh` (`k8s/README.md:1-148`) against context `admin@home`, namespace `default`:

| Node | GPU | Driver | Total VRAM | Resident Serving Workload | HAMi Free | Physically Free (baseline) |
|---|---|---|---|---|---|---|
| `talos-8me-tcy` (`home-k8s/egpu=rtx5090`) | NVIDIA GeForce RTX 5090 | `580.178.04` | 32,607 MiB | `inkling-2-mini-basalt` (`gpumem-percentage: 100`) | 0 MiB | ~909–975 MiB |
| `talos-210-73x` | NVIDIA RTX PRO 6000 Blackwell WS | `580.167.08` | 97,887 MiB | `pennyroyal-flashnext` (SGLang, 97,800 MiB, 300 W cap) | 87 MiB | 1,317 MiB |
| `dave-droid` (tainted `dedicated=pascal:NoSchedule`) | NVIDIA GeForce GTX 1070 | — | 8,192 MiB | `qwen3-asr` (3,000 MiB) | ~5,192 MiB | Excluded (`sm_61` unsupported by CUDA 13) |

- **GPU smoke benchmarks (`runs/smoke-5090/smoke.json`, `runs/smoke-6000/smoke.json`):** Executed in `--mode share` alongside resident workloads:
  - **RTX 5090 (`sm_120`, 170 SMs):** `207.75 TFLOPS` float16 (`4096x4096`), `198.15 TFLOPS` bfloat16-inplace; peak PyTorch allocated `104 MiB` (`120 MiB` reserved) under `192 MiB` allocator cap (`877 MiB` physically free at start).
  - **RTX PRO 6000 (`sm_120`, 188 SMs, 300 W):** `112.80 TFLOPS` float16-inplace (`101.60 TFLOPS` float16), `105.15 TFLOPS` bfloat16; peak PyTorch allocated `200 MiB` (`240 MiB` reserved) under `512 MiB` allocator cap (`1,317 MiB` physically free at start).
- **HAMi vs `share` vs `none` (`k8s/README.md:96-118`):**
  - `hami` (default for real experiments): reserves `nvidia.com/gpu: 1` + `nvidia.com/gpumem: <MiB>` via `hami-scheduler`. During GPU Window #1 (`18:15`, `devlog/content/2026-11-26-web-references-and-gpu-window.md:35`), `inkling-2-mini` was paused on `talos-8me-tcy`, freeing all `32,607 MiB` HAMi / `32,109 MiB` physical VRAM (`runs/driver-S2.log`).
  - `share` (explicit opt-in only): requires a valid `gpu-vram-exporter` reading (`phys_free >= --mem + 512 MiB`) and injects `/_vi_site/sitecustomize.py` calling `torch.cuda.set_per_process_memory_fraction` plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`.
  - `none` (CPU-only): proved full cluster plumbing before spending GPU time via `runs/chain-check/Z0_cpu_chain_check-87d5893f` (`val` 98.4 s, `test` 27.1 s, `peak_gpu_bytes=0`).

### 7.2 Reproducibility, Provenance & Storage Hardening

Following Review #4 (`docs/freestyle/reviews/codex-review-4-k8s.md`, commit `84042db`, `k8s/README.md:49-139`):
- **Pinned container digest:** `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0`.
- **Immutable run identity & hashes:** `freestyle/experiments/launch.py` hashes tracked + non-ignored untracked files (`git_sha`, `git_dirty`, `git_diff_hash`) into an 8-char suffix (`runs/<id>-<hash>/`). `submit.sh` verifies SHA-256 of the uploaded repo tarball on the workspace pod and inside the Job, writing `provenance.json` (`run_id`, `repo_tar_sha256`, `image`, `gpu`, `mode`, `vram_mb`) and `env.txt` (`pip freeze`, `pod_image_id`).
- **Storage & secret isolation:** Separate node-local PVCs mount private data read-only at `/data-ro` (`vision-intern-data-<gpu>`) and outputs read-write at `/outputs` (`vision-intern-out-<gpu>`). Ownership labels/annotations (`app=vision-intern`, `vision-intern/owner`) gate `--replace` and PVC/pod reuse; `--dry-run` mutates nothing; `HF_TOKEN` is mounted only when `needs_hf_token: true`; `collect.sh` verifies file-by-file SHA-256 atomically.
- **Checkpoint pruning (`17222a2`):** After local disk filled from ~0.5 GB PyTorch Lightning checkpoints per run (`devlog/content/2026-11-26-rfdetr-beats-zero-shot.md:36-38`), `entry.py:207-231` prunes intermediate checkpoints locally while keeping best weights.

### 7.3 Resource Usage per Experiment

The full ledger — every run plus the bundle CPU demos, Collective call counts and the
field-by-field provenance — lives in [`RESOURCES.md`](RESOURCES.md). Condensed below.

All figures come from `runs/*/meta.json` (`timings`, `inference`, `peak_gpu_bytes`, `job_wall`) and `runs/*/provenance.json` on `RTX 5090` (`k8s:5090`, `mode: hami`, `vram_mb: 12000` unless noted):

| Run | Train Time (s) | Peak PyTorch VRAM (`peak_gpu_bytes`) | Test Inference (`s/img`) | Val Inference (`s/img`) | Job Wall Total (s) |
|---|---:|---|---:|---:|---:|
| `B1_owlv2_dinov2-4c99011d` | — | `1.97 GiB` (`2,112,137,216 B`) | `0.945` proposal (`0.059` pipe) | `30.226` proposal (`0.101` pipe) | `330` (`47` pip + `283` cmd) |
| `B2_sam3_siglip2-babcbfe8` | — | `0.01 GiB` (`9,568,256 B`, emb only) | Blocked (403 gated HF) | Blocked | Unfinished (`83` pip, failed) |
| `B3_gdino_dinov2-be668339` | — | `1.68 GiB` (`1,798,717,952 B`) | `0.363` proposal (`0.026` pipe) | `17.769` proposal (`0.017` pipe) | `150` (`25` pip + `125` cmd) |
| `C1_rfdetr_small_agnostic_dinov2-55ef8ae7` | `247.4` (`238.6` trainer) | `6.15 GiB` (`6,603,773,952 B`) | `0.432` | `0.599` | `615` (`338` pip + `277` cmd) |
| `C2_rfdetr_medium_agnostic_siglip2-89fe7c2b` | `289.0` (`281.0` trainer) | `4.09 GiB` (`4,394,649,088 B`) | `0.328` | `0.421` | `409` (`100` pip + `309` cmd) |
| `D1_rfdetr_small_multiclass-6f3c344b` | `195.7` (`181.7` trainer) | `6.16 GiB` (`6,616,291,328 B`) | `0.352` | `0.513` | `301` (`80` pip + `221` cmd) |
| `S1_c2_tile768-d52f6d7c` | `223.2` (`215.6` trainer) | `4.09 GiB` (`4,394,649,088 B`) | `0.363` | `0.519` | `293` (`49` pip + `244` cmd) |
| `S2_c2_tile1280-db269936` | `145.2` (`137.8` trainer) | `4.09 GiB` (`4,394,649,088 B`) | `0.223` | `0.309` | `213` (`50` pip + `163` cmd) |
| `S3_c2_copypaste-205b1c7c` | `427.3` | `4.09 GiB` (`4,394,649,088 B`) | `0.337` | `0.429` | `499` (`51` pip + `448` cmd) |
| `S5_rfdetr_large_agnostic-7f6457d4` | `176.3` | `4.09 GiB` (`4,394,649,088 B`) | `0.359` | `0.445` | `247` (`50` pip + `197` cmd; `vram_mb: 16384`) |
| `S6_owlv2_containment-cbc70383` | — | `1.97 GiB` (`2,112,137,216 B`) | `0.965` proposal (`0.016` pipe) | `1.109` proposal (`0.050` pipe) | `52` (`19` pip + `33` cmd) |
| `S7_c2_tile1280_copypaste-b6fc182e` | `327.6` | `4.09 GiB` (`4,394,649,088 B`) | `0.221` | `0.351` | `396` (`50` pip + `346` cmd) |
| `G1_rfdetr_medium_faces-cc431fcd` | `75.3` | `4.09 GiB` (`4,394,649,088 B`) | `0.168` | `0.279` | `139` (`49` pip + `90` cmd) |
| `G2_rfdetr_small_faces-7c7b4587` | `191.4` | `6.89 GiB` (`7,398,841,344 B`) | `0.242` | `0.365` | `262` (`52` pip + `210` cmd) |

*Notes:* For `B1`, `B3` and `S6`, `proposal_seconds_per_image` on `val` (the first split evaluated) includes initial HF checkpoint load, whereas `test` reflects warm tiled inference; `pipe` is post-hoc cached-proposal lookup. `S3_c2_copypaste-e6ce578e` failed wait (`jobs.batch ... not found`, `runs/driver-S.log`) and `S3_c2_copypaste-205b1c7c` was subsequently collected with metrics and metadata; `C2_...-27f2e3d6` did not produce those files. Peak VRAM does not rank by detector size: `C1`/`D1` (Small @ 512) peak at ~6.15 GiB and `G2` (Small, 1024 px tiles) at 6.89 GiB, while every Medium @ 576 run sits at 4.09 GiB — resolution and batch size drive the mark. Card time actually burned: ~61 min of job wall across 12 jobs in GPU window #1 (of which ~25 min was `unpack+pip`), and 401 s across `G1`+`G2` in window #2. CPU bundle cost: v0 = 67.18 s / 3 frames, 1.73 GB peak RSS (`runs/bundle_demo/count.err`); v1 = 38.8 s identify on the densest frame because it makes one Collective call per crop (`runs/bundle_demo_v1/counts.json`). VLM budget: bootstrap v1 266 calls / 1,896.7 s, bootstrap v2 686 calls / 2,276.0 s, face labelling 197 calls / ~28 min, all with 0 failed calls (`runs/bootstrap_v1/timing.json`, `runs/bootstrap_v2/summary.md`, `runs/faces_v1/timing.json`).

---

## 8. Experiments & Comparison

### 8.1 Results vs Model Labels (`bootstrap_v2.json`, NOT Human GT)

The seven runs in the original table below (excluding `runs/chain-check/`) are evaluated on the **12 test images (11 groups)** against `data/labels/bootstrap_v2.json` (`verdicts: [correct]`, 176 model boxes) using val-selected best-F1 operating thresholds and 1,000-sample group-bootstrap 95% CIs (`freestyle/eval/evaluate.py`). Class-agnostic runs (`B1`, `B3`, `C1`, `C2`, `S1`, `S2`) have `identifier.type: none`, so every predicted box is `unknown` (`ID Top-1 = 0.000`).

| Experiment (`vs model labels, NOT GT`) | Thr | AP50 (95% CI) | AP50:95 (95% CI) | P / R / F1 | ID Top-1 (95% CI) | ID Macro-F1 | Total MAE (95% CI) | Class MAE | Exact Tot / Exact Inv | Imgs (Grps) |
|---|---:|---|---|---|---|---|---:|---:|---:|---:|
| `S2_c2_tile1280-db269936` | 0.308 | **0.710** (0.504–0.841) | **0.495** (0.332–0.616) | **0.772** / 0.750 / **0.761** | 0.000 (0.000–0.000) | 0.000 | **3.583** (1.333–6.730) | 0.961 | **0.083** / 0.000 | 12 (11) |
| `C2_rfdetr_medium_agnostic_siglip2-89fe7c2b` | 0.538 | 0.697 (0.520–0.814) | 0.469 (0.312–0.587) | 0.728 / 0.699 / 0.713 | 0.000 (0.000–0.000) | 0.000 | 4.417 (2.333–7.365) | 1.013 | 0.000 / 0.000 | 12 (11) |
| `C1_rfdetr_small_agnostic_dinov2-55ef8ae7` | 0.315 | 0.694 (0.522–0.823) | 0.474 (0.316–0.595) | 0.638 / **0.761** / 0.694 | 0.000 (0.000–0.000) | 0.000 | 5.833 (3.182–8.548) | 1.114 | 0.000 / 0.000 | 12 (11) |
| `D1_rfdetr_small_multiclass-6f3c344b` | 0.576 | 0.653 (0.455–0.803) | 0.409 (0.252–0.528) | 0.762 / 0.693 / 0.726 | **0.570** (0.371–0.725) | **0.414** | 5.167 (2.070–8.818) | **0.500** | **0.083** / 0.000 | 12 (11) |
| `S1_c2_tile768-d52f6d7c` | 0.813 | 0.614 (0.413–0.767) | 0.362 (0.227–0.495) | 0.717 / 0.562 / 0.631 | 0.000 (0.000–0.000) | 0.000 | 6.000 (3.333–9.365) | 0.895 | 0.000 / 0.000 | 12 (11) |
| `B3_gdino_dinov2-be668339` | 0.252 | 0.232 (0.119–0.360) | 0.152 (0.073–0.260) | 0.683 / 0.318 / 0.434 | 0.000 (0.000–0.000) | 0.000 | 9.500 (5.500–13.639) | 0.675 | 0.000 / 0.000 | 12 (11) |
| `B1_owlv2_dinov2-4c99011d` | 0.101 | 0.181 (0.079–0.294) | 0.107 (0.048–0.196) | 0.376 / 0.381 / 0.379 | 0.000 (0.000–0.000) | 0.000 | 8.833 (4.833–13.274) | 1.070 | 0.000 / 0.000 | 12 (11) |

**Observations (strictly vs `bootstrap_v2` model labels):**
- **Trained RF-DETR vs zero-shot:** All trained RF-DETR configurations (`0.614–0.710 AP50`) beat zero-shot `B3` (`0.232`) and `B1` (`0.181`) by ~2.6x–3.9x AP50.
- **Tile size sweep (`S1` 768 vs `C2` 1024 vs `S2` 1280, all `RFDETRMedium` at `576` px):** Larger tiles improved AP50 (`0.614` -> `0.697` -> `0.710`), reduced count MAE (`6.000` -> `4.417` -> `3.583`), while train times were `223.2 s` -> `289.0 s` -> `145.2 s` and test latency (`0.363` -> `0.328` -> `0.223 s/img`), likely because 1280 px tiles preserve multi-cartridge shelf context and create fewer boundary cuts on `1600x1200` / `2048x1536` frames.
- **Class-agnostic vs multi-class (`C1` vs `D1`):** Keeping 18+1 classes inside `RFDETRSmall` slightly lowered localization AP50 (`0.694` -> `0.653`) and recall (`0.761` -> `0.693`), while achieving `0.570` ID Top-1 (`0.414` Macro-F1) over `79` matched known-GT boxes (`43` matched unknown-GT, `15` unmatched known-GT) and halving per-class count MAE (`1.114` -> `0.500`) against model labels.
- **Uncompleted / unrun experiments:** `B2_sam3_siglip2` failed with HF 403 gated access (`facebook/sam3`); `S4_c2_long`, `A1`–`A3`, `E1`, and `F1` have configs without completed `runs/` outputs. S3, S5, S6 and S7 were subsequently collected (see the sweep table below).

### 8.2 Human-GT Comparison Table (`score_all`)

`freestyle/experiments/score_all.py` will score all runs that have `val/predictions.json` and `test/predictions.json` across identifier modes (`none`, `own`, `vlm`, `knn`, `proto`, `knn_trainvlm`, `knn_trainvlm+val`, `linear_trainvlm`, `linear_trainvlm+val`) × pair rule (`off`, `on`) × subsets (`all`, `sharp-full`, `partial/blurry`), using val-selected thresholds (`score_thr`, `identity_thr`, and `train_vlm_confidence_thr`).

A dry-run verification with synthetic model-accepted answers exists at `runs/score_all_demo/RESULTS_GT.md` (42 rows across `B1`, `B3`, `C1`, `C2`, `D1`), which is **not human GT** and predates `f1e4c6f` (so its `sharp-full` subset has 0 images).

<!-- GT-PENDING: Replace this block with the human-GT comparison table (Localizer × Identifier × Pair Rule × Subset -> Total MAE [95% CI], Top MAE, Bottom MAE, Exact Inventory, ID Accuracy, AP50, plus Per-class count MAE) from docs/freestyle/RESULTS_GT.md once human quiz answers are complete and score_all has run. -->

### 8.3 Sweep update (vs model labels, NOT human GT)

The collected sweep uses 12 test images in 11 groups; thresholds were selected on validation. Values are from `docs/freestyle/RESULTS.md` and the corresponding `runs/S*/metrics.json`; configuration and run provenance are in each matching `meta.json`. S4 has no collected metrics. These are comparisons to bootstrap labels, not estimates of human-ground-truth accuracy.

| Run | Change | AP50 | Precision / recall | Total count MAE |
|---|---|---:|---|---:|
| S1 (`d52f6d7c`) | Medium, 768 px tiles | 0.614 | 0.717 / 0.562 | 6.000 |
| S2 (`db269936`) | Medium, 1280 px tiles | 0.710 | 0.772 / 0.750 | 3.583 |
| S3 (`205b1c7c`) | Medium, 1024 px tiles + copy-paste | 0.718 | 0.720 / 0.773 | 4.417 |
| S4 | Longer training | — | — | Not collected |
| S5 (`7f6457d4`) | Large, 1024 px tiles | 0.689 | 0.804 / 0.676 | 4.167 |
| S6 (`cbc70383`) | OWLv2 + containment filter | 0.120 | 0.564 / 0.176 | 11.583 |
| S7 (`b6fc182e`) | Medium, 1280 px tiles + copy-paste | 0.675 | 0.794 / 0.722 | 4.500 |

S3 has the highest observed AP50, while S2 has the lowest observed total MAE; the small test set and overlapping intervals do not establish a winner. The sweep detectors used `identifier: none`, so their identity metrics do not measure the later kNN bundle (`runs/S*/meta.json`; `devlog/content/2026-11-26-sweep-and-inkling-restored.md`).

### 8.4 Bundle v0 and class-index investigation

`rfdetr-m-1280-knn-v0` combines S2's RF-DETR Medium detector, a SigLIP2 embedding kNN identifier, and shelf-aware counting. Its train-only gallery contains 139 VLM-labelled known crops across 10 classes, including 60 NNS crops; the original S2 evaluation had no identifier. Bundle export and operation are documented in `docs/freestyle/HOW_TO_RUN.md`. On CPU, the recorded `frame.jpg` output is 8 NNS, 4 Hi-C, 3 Fanta, 2 Vanilla Flavor and 9 unknown; localization took 5,516 ms, identification 9,364 ms and counting 636 ms. The recorded `008.jpg` output includes 22 NNS and 3 unknown, with gap and neighbour-vote flags (`runs/bundle_demo/frames/counts.json`). The demo measured approximately 67 s for three frames including approximately 15 s model load and 1.7 GB peak RAM; S2 GPU detector inference was 0.223 s/test image (`docs/freestyle/HOW_TO_RUN.md`; `runs/S2_c2_tile1280-db269936/meta.json`). These timings and counts are observations, not GT validation.

The demo's identity is weak: the NNS-heavy gallery biases kNN toward NNS, including CC as NNS and AA as vanilla in the qualitative review. Overlapping tiles leave duplicate boxes on the leaning bottom shelf, and the two-high NNS stack can be mistaken for a third shelf. These are observed failure modes to address, not verified accuracy figures (`devlog/content/2026-11-26-end-to-end-on-cpu.md`). The same entry records plans to compare identity approaches and suppress duplicates using validation photos; final selection awaits the quiz.

An interim modulo class-index workaround risked silently shifting multiclass identities. Investigation of RF-DETR 1.12.1 found zero-based head indices over sorted training categories. Training now saves an explicit head-index-to-category mapping, existing checkpoints had mappings recovered, and unknown indices fail loudly. The audit found D1 was not shifted (`devlog/content/2026-11-26-end-to-end-on-cpu.md`; `freestyle/train/recover_mapping.py`; `freestyle/infer/detector.py`).

### 8.5 Bundle v1: duplicate suppression, 2-shelf counting, hybrid identifier

`artifacts/rfdetr-m-1280-v1` (frozen from `runs/S2_c2_tile1280-db269936`, config
`configs/pipeline/v1_cartridges.yaml`, all knobs tuned on the **6 val frames only**) changes three
things relative to v0 (`artifacts/rfdetr-m-1280-knn-v0`):

1. **Duplicate suppression after tile merge** — class-agnostic, so it no longer needs a class
   agreement it does not have: IoU 0.45, containment 0.7, and a looser same-shelf-row IoU 0.3 when
   the two boxes share a column (x-overlap ≥ 0.5 of the narrower box). On val against bootstrap
   labels this took overlapping box pairs from **13 → 0** (`artifacts/rfdetr-m-1280-v1/manifest.json`
   `provenance.notes`).
2. **Shelf cap `max_rows: 2`** — the two-high NNS stack is now a *tier* of the top shelf instead of
   a phantom third shelf; `shelf_hint: true` also feeds top/bottom context to the identifier.
3. **Hybrid identifier `hybrid_vlm_knn`** — `david-birks/inkling-2-mini` (two-stage, 2× upscale)
   names the **brand family** from logo/colour; embedding kNN (`google/siglip2-base-patch16-224`,
   `k=5`) may only refine *within* that family, over a **class-balanced** gallery (≤10 crops per
   class, vs v0's 139 crops of which 60 were NNS). Family confidence < 0.45 or identity score < 0.45
   stays `unknown`. `pair_rule: report` (never `apply`) pending human GT
   (`artifacts/rfdetr-m-1280-v1/pipeline.yaml`).

Measured effects (all **unverified by a human**; `runs/bundle_demo_v1/counts.json`,
`devlog/content/2026-11-27-bundle-v1.md`):

| Frame | v0 counts | v1 counts |
|---|---|---|
| `local/downloads/frame.jpg` | NNS 8, Hi-C 4, Fanta 3, Vanilla 2 + 9 unknown | NNS 7, Lemon 2, Fanta 2, CC 1, AA 1, Sprite 1, Orange 1, Vanilla 1 + 7 unknown |
| `50828e4b…/008.jpg` | **NNS 22**, Fanta 3, Hi-C 3, Vanilla 1 + 3 unknown | NNS 5, Fanta 2, CC 2, AA 1 + 10 unknown |
| `10455345…/013.jpg` | NNS 3 | Sprite 1 + 2 unknown |

The 7 NNS on `frame.jpg` is what is physically there (a 3-stack plus 4) on visual inspection, which
is the *only* evidence for it; the 10455345 frame has since been answered `not_cabinet` in a quiz
smoke pass, so v0's 3 NNS on it were three hallucinations (`data/gt/quiz_answers.jsonl`).

Val proxy vs bootstrap labels: v1 predicts 103 boxes / 139 bootstrap, AP50 0.591 with **0**
overlapping box pairs; v0 predicts 113 boxes, AP50 0.624, 13 overlapping pairs — i.e. v1 buys
clean geometry and fewer duplicates at a small localisation cost on 6 images
(`artifacts/rfdetr-m-1280-v1/manifest.json`).

**Cost:** identify time rises from 9.4 s to 38.8 s on the densest frame because the hybrid
identifier makes one Collective call per crop; v1 needs `LLMC_API_KEY` at inference time
(`runs/bundle_demo_v1/counts.json` vs `runs/bundle_demo/frames/counts.json`).

**Known gaps in v1 (stated plainly, not hidden):** the bottom shelf (Coke family) is mostly
`unknown` because the auto-labelled gallery has almost no confident Part 1/Part 2 examples; and
`score_thr` is pinned at the v0 value 0.308 even though val prefers 0.10–0.15 once suppression is
on, because below ~0.25 the class-agnostic head emits a **phantom second class index** and
`RFDetrPredictor.predict` raised `Unknown RF-DETR detector class index: 1`
(`artifacts/rfdetr-m-1280-v1/pipeline.yaml` comment, `configs/pipeline/v1_cartridges.yaml`).

### 8.6 RF-DETR class-index handling and the phantom no-object slot

An interim modulo class-index workaround risked silently shifting multiclass identities.
Investigation of RF-DETR 1.12.1 found zero-based head indices over sorted training categories.
Training now saves an explicit head-index-to-category mapping (`freestyle/train/recover_mapping.py`
recovers it for existing checkpoints), unknown indices fail loudly, and the audit confirmed `D1`
was **not** shifted (`devlog/content/2026-11-26-end-to-end-on-cpu.md`; `freestyle/infer/detector.py`).
A second, related bug surfaced while tuning v1: for a **single-category** checkpoint the classifier
head is `(2, 256)` — one foreground plus one background slot — and treating that background slot as
a detectable class index is what produced the `Unknown RF-DETR detector class index: 1` error below
`score_thr ≈ 0.25`. Fixed by dropping the no-object slot for single-category checkpoints (`80c1a44`),
but the v1 bundle still ships the conservative threshold until the fix is re-exported and re-tuned
against human GT.

### 8.7 Experiment family G — detecting the printed front face

**Idea and origin.** On the bottom shelf the cartridges lean outward, so an axis-aligned box around
a whole leaning cartridge covers parts of its neighbours and is hard to count
(`devlog/content/2026-11-27-front-faces-idea.md`). Family **G** detects only the **front face** — the
narrow printed end carrying the logo and the Part 1/2 numeral — which is small, upright and
non-overlapping. A full 3D cartridge model was considered and rejected: it needs 3D annotations or a
calibrated camera, which ~25 photos cannot support.

**Label construction (`runs/faces_v1/`, `runs/faces_v2/`).** Inkling 2 Mini labelled all 29 cabinet
frames (full frame + 2×2 tiles) through Vision Intern's label → judge path, judged by
`qwen3-8-flash-next` in crop mode: **715** face proposals in ~28 min from **197** fresh Collective
calls with 0 errors (`runs/faces_v1/timing.json`, `runs/faces_v1/summary.md`). Mean neighbour
max-IoU fell from **0.201** (whole-cartridge `bootstrap_v2`) to **0.146**, and to **0.098** where both
labeller runs agreed. **326 / 715** faces carried an identity. Critically, asked about the *face*
the VLM read **59** Part numerals (45 `1`, 14 `2`) where whole-cartridge crops yielded **2 of 123**
(`runs/faces_v2/summary.md` vs `runs/bootstrap_v2/summary.md`).

Cleanup (`faces_v2`) kept **597 / 715**: dropped 95 judge-rejected, 15 weak-evidence, 6
leaning-column duplicates, 1 flat-aspect and 1 thin-strip. Neighbour overlap improved to **0.119**.
One rule needed care: "one face per leaning column" also requires a vertical-gap guard, otherwise a
fanned-out bottom row chains 17 real faces into a single "column". **53 residual duplicate pairs**
were flagged for human review and deliberately *not* auto-deleted, because the same geometry also
describes a genuine back-row face behind a front-row one.

**Training and results (GPU window #2, RTX 5090):**

| Run | Target | Train s | Peak VRAM | Test s/img | AP50 (95 % CI) | P / R / F1 | Total count MAE | ID Top-1 |
|---|---|---:|---:|---:|---|---|---:|---:|
| `G1_rfdetr_medium_faces-cc431fcd` | front faces | 75.3 | 4.09 GiB | 0.168 | **0.663** (0.453–0.800) | 0.573 / 0.718 / 0.637 | 5.750 (3.081–8.919) | 0.000 (`none`) |
| `G2_rfdetr_small_faces-7c7b4587` | front faces | 191.4 | 6.89 GiB | 0.242 | **0.631** (0.402–0.793) | 0.697 / 0.624 / 0.658 | **3.333** (1.500–5.636) | 0.000 (`none`) |
| `S2_c2_tile1280-db269936` | whole cartridges | 145.2 | 4.09 GiB | 0.223 | **0.710** (0.504–0.841) | 0.772 / 0.750 / 0.761 | 3.583 (1.333–6.730) | 0.000 (`none`) |

(`runs/*/metrics.json`; `docs/freestyle/RESULTS.md`; trained on 9 train frames of `faces_v2` vs 10 of
`bootstrap_v2`. G1/G2 have `identifier: none`, so their identity rows are structural, not measured.)

**The rows are not comparable and that is the finding.** `G1`/`G2` are scored against face labels,
`S2` against whole-cartridge labels, and both "answer keys" are model-made. Faces and whole boxes now
sit in the same range, so no further GPU variant can separate them — which is why GPU window #2 was
closed after ~12 minutes of the ~60 offered and Inkling was restored
(`devlog/content/2026-11-27-face-detectors-trained.md`). Both detectors predicted the same 12 test
images, so a single pass of "N top / M bottom: right?" answers scores faces vs whole cartridges, all
identity methods and the pair rule at once with confidence intervals. The face failure signature is
also different — `G1`'s dominant error is `FP:background = 56` (faces invented on shelf slats and
frame edges) versus `S2`'s `FP:fragment = 17` — see [`FAILURES.md`](FAILURES.md) §1, §2.6.

### 8.8 Review rounds and the bug classes they found

Five adversarial reviews were run against the code that produces every number in this report; each
finding was reproduced on CPU with synthetic data before being fixed. Findings and fixes:

| Review | Target | Bug classes found | Disposition |
|---|---|---|---|
| #1 (`docs/freestyle/reviews/codex-review-1.md`) | identification + eval | **Gallery leakage** (test set and same-event burst frames could train the identifier); pipeline fitted every gallery row with no split check | Fixed: validated group/split provenance, train-only galleries, group-disjointness enforced |
| #2 (`docs/freestyle/reviews/agy-review-2.md`) | `freestyle/` runtime | `identity_thr` **defeated** by normalized `predict_proba`; OWLv2 broken on `transformers 5.24.1` (scores rescaled to 1.0); SAM3 `bfloat16` dtype crash; copy-paste `--photometric` warped pixels without moving boxes; string `image_id`s made `eval` silently drop 100 % of predictions; unused `gallery_groups`; `Siglip2` ignoring `do_resize=False`; cross-class duplicates at tile seams | Fixed; the cross-class tile-seam duplicate class is what later became the v1 `suppress_*` work |
| #3 (`docs/freestyle/reviews/agy-review-3.md`) | Vision Intern **core** dense-scene path | Cross-tile `nms` preferred clipped fragments over full boxes; consensus/NMS ran **before** alias normalisation; crop-mode sheets omitted the boxes the judge was asked to verify and crashed on OOB/inverted boxes; `transformers` backend never derived `finish_reason` (silently disabling truncation auto-split); greedy `\[.*\]` verdict parsing dropped all verdicts on truncated output; unlocked `raw_response_log` appends; symlinked images rejected | Fixed upstream in `fb8c805` / `289cc0e` / `4b96e5f` (see §5.1) |
| #4 (`docs/freestyle/reviews/codex-review-4-k8s.md`) | `k8s/` launch + provenance | **`--dry-run` could delete an unrelated Job**; auto `share` mode ran training with no enforced VRAM cap; failed jobs waited to timeout and lost collection; existing workspace pod/PVC trusted (foreign pod deletable); gated HF models got no token; artifact/snapshot reuse could corrupt results; recorded git state did not identify the executed snapshot; private data mounted writable | Fixed in `84042db` + `17222a2`: pinned image digest, tarball SHA-256 verified twice, `provenance.json` + `env.txt`, ownership labels gate `--replace`, HF token opt-in, read-only `/data-ro`, `sitecustomize` memory cap in share mode |
| #5 (`docs/freestyle/reviews/agy-review-5-score-all.md`) | `score_all.py` + `quiz/to_coco.py` — the code that will consume the human answers | `pair=on` was a **100 % no-op** for all 7 non-VLM identifiers and *bypassed* `identity_thr` on `vlm`; human-verified `not_cabinet` negatives silently dropped by the 80 % rule; partially answered / `counts_mismatch` images corrupted `class:{c}` MAE, AP50 and threshold selection; predicted-only classes (incl. `unknown`) omitted from per-class MAE and generic `cartridge` not mapped to `unknown`; `to_coco` promoted a wrong model guess to `identity_source="human"` on `"partial"`; threshold tie-breaks collapsed to `0.0`; `match_boxes` called without scores so low-confidence duplicates stole GT matches | Fixed in `5ff55c2` (pair rule for every identifier, negatives kept, partial-answer handling, generic→unknown, tie-breaks, score-aware matching) |

Review #5 matters most for the pending work: it is the only review whose subject is the human-GT
scoring path, and it found the pair rule — the hypothesis §10.2 says the quiz will test — was
not running at all.

### 8.9 Provisional architecture selection (pending human GT)

> [!IMPORTANT]
> **This selection is unverified.** It is the candidate the evidence *available today* points at,
> recorded so the final step is a scoring pass rather than a redesign. Nothing below has been
> measured against human ground truth; the deciding numbers are pending.
>
> `<!-- GT-PENDING: fill the table in §8.2 and the face-vs-whole-box comparison in this section from docs/freestyle/RESULTS_GT.md (produced by scripts/finalize_after_gt.sh). -->`

**Candidate (working default): bundle `rfdetr-m-1280-v1`** — RF-DETR **Medium**, 1280 px tiles
(`S2`), plus:
* class-agnostic duplicate suppression after tile merge (IoU 0.45 / containment 0.7 / same-row IoU 0.3),
* **2-shelf** counting (`max_rows: 2`, a 2-high NNS stack is a tier), `shelf_hint: true`,
* **hybrid** identifier: Inkling 2 Mini brand family + class-balanced SigLIP2 kNN refining inside
  that family, thresholds 0.45/0.45, `unknown` below either,
* pair rule in `report` mode only.

**Alternative to be decided by GT: family G** — the same pipeline shape driven by a **front-face**
detector (`G2` RF-DETR Small @ 1024 px tiles, or `G1` Medium @ 1280 px) instead of a
whole-cartridge detector.

**Justification from evidence that exists today (and only that):**

1. *Model-label agreement.* Among the **class-agnostic** whole-cartridge runs, `S2` has the lowest
   total-count MAE (**3.583**, CI 1.333–6.730) of any run ever collected, the joint-best AP50
   (0.710, CI 0.504–0.841) behind `S3`'s 0.718, the lowest `FN:stacked_tier` (1) and the
   second-lowest FP count (`runs/S2_c2_tile1280-db269936/metrics.json`,
   `runs/S2_c2_tile1280-db269936/failures/failures.json`).
   It is also the cheapest run in the sweep: 145.2 s train, 213 s job wall, 0.223 s/img.
2. *Visual review (unverified).* `S2` + suppression + 2-shelf counting produced **0** overlapping
   box pairs on val and the physically correct 7 NNS on the best frame, where v0's plain kNN produced
   8 NNS / 4 Hi-C and **22 NNS** on another frame (`runs/bundle_demo*/…counts.json`). The identity
   review over 69 val crops judged 29 / 32 confidently-readable crops correct for the hybrid vs
   26 / 32 for plain kNN — my own read, not a human's
   (`artifacts/rfdetr-m-1280-v1/manifest.json` `provenance.notes`, `runs/v1iter/eyeball.json`).
3. *Resources.* `S2` is the cheapest detector in the family; faces are cheaper still (`G1` 139 s job
   wall, 0.168 s/img) but the v1 hybrid's per-crop VLM call dominates end-to-end cost either way
   ([`RESOURCES.md`](RESOURCES.md) §2). Choosing `S2` costs no extra card time.
4. *Why the alternative is genuinely still open.* `G2` reports the lowest total-count MAE of any
   class-agnostic run (**3.333**, CI 1.500–5.636) *in its own label space*, faces reduce neighbour
   overlap (0.119 vs 0.201) and unlock numerals the whole-box crops cannot read (59 vs 2). Two
   label spaces that cannot be compared is exactly the situation the quiz breaks — the human counts
   are the same object for both families.

**What would flip this selection:** human counts that show the face detector counts the leaning
bottom row better (G's whole point), or an ID accuracy on human-answered crops where the
whole-cartridge hybrid loses; also a `counts_mismatch` list dominated by merged bottom-row boxes
would argue that whole-cartridge boxes are the wrong target regardless of AP50.

**Explicitly not decided here:** `identity_thr` / `family_threshold` (0.45/0.45), `score_thr`
(0.308, pinned by the phantom-class-index bug rather than by data), the pair rule's `apply` mode, and
whether `S3` (copy-paste, AP50 0.718) is a better base than `S2` — all four must be re-tuned against
`data/gt/val.json` before the bundle is called final.

### 8.10 GPU window follow-up

SAM 3 remained blocked by a Hugging Face 403: the token was not approved for the gated model (`devlog/content/2026-11-26-gpu-window-first-results.md`). During collection the workstation disk reached 100%; Lightning checkpoints were approximately 0.5 GB per run. Intermediate local checkpoints are pruned while best weights are retained; the full set remains on the cluster volume (`devlog/content/2026-11-26-rfdetr-beats-zero-shot.md`).

Inkling's scheduled restoration reminder did not fire while the session was waiting. It was restored around 22:45, roughly two hours beyond the time box, although the GPU had been idle since 20:05. The recorded operational fix is a detached host-side restoration timer independent of the session (`devlog/content/2026-11-26-sweep-and-inkling-restored.md`).

**Window #2 (2026-11-27).** Inkling was paused at 09:11 with explicit permission to run past 60
minutes; `G1` and `G2` consumed 401 s of job wall and the window was closed after ~12 minutes because
the model-made answer key could not separate the two families any further
(`devlog/content/2026-11-27-gpu-window-2.md`, `runs/G*/meta.json`). The restore is now driven by a
detached timer in its own tmux window on the host (set for 11:15), so it cannot be skipped the way
the window-#1 restore was. Two operational lessons are recorded: the restore timer must be independent
of the agent session, and a GPU window is only worth holding while it can change a decision.

---

## 9. Visual Failure Analysis

`freestyle/eval/failures.py` decomposes test-set errors (vs `bootstrap_v2` model labels — or vs
`faces_v2` face labels for the G family, `n_gt = 170` — at IoU `0.5` and val-selected `score_thr`;
`n_gt = 176` across `12` images) into missed detections (**FN**: `adjacent_merge`, `edge_truncated`, `stacked_tier`, `plain`), false positives (**FP**: `duplicate`, `fragment`, `background`), identity errors (**ID**), and localization errors (**LOC**, `0.3 <= IoU < 0.5`). Note: in `failures.json`, `n_tp` counts boxes correct in **both** localization and class, so class-agnostic runs (`B1`, `B3`, `C1`, `C2`, `S1`, `S2`, `S3`, `S5`, `S6`, `S7`, `G1`, `G2`) move correctly localized known-class boxes into `ID` (`<class>->UNK`). [`FAILURES.md`](FAILURES.md) carries the same table for every collected run, a curated six-image gallery of the recurring modes, and the follow-ups:

| Run | Thr | Pred Kept | TP (Loc+Cls) | Total Failures | FN (Tot) | FN: adj_merge / edge / plain / stack | FP (Tot) | FP: bg / dup / frag | ID | LOC |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| `B1_owlv2_dinov2-4c99011d` | 0.101 | 178 | 26 | 247 | 95 | 51 / 10 / 29 / 5 | 97 | 54 / 10 / 33 | 41 | 14 |
| `B3_gdino_dinov2-be668339` | 0.252 | 82 | 17 | 180 | 115 | 45 / 11 / 48 / 11 | 21 | 18 / 0 / 3 | 39 | 5 |
| `C1_rfdetr_small_agnostic_dinov2-55ef8ae7` | 0.315 | 210 | 50 | 199 | 39 | 19 / 2 / 16 / 2 | 73 | 26 / 12 / 35 | 84 | 3 |
| `C2_rfdetr_medium_agnostic_siglip2-89fe7c2b` | 0.538 | 169 | 42 | 175 | 48 | 16 / 4 / 24 / 4 | 41 | 10 / 7 / 24 | 81 | 5 |
| `D1_rfdetr_small_multiclass-6f3c344b` | 0.576 | 160 | 88 | 124 | 52 | 15 / 3 / 26 / 8 | 36 | 4 / 15 / 17 | 34 | 2 |
| `S1_c2_tile768-d52f6d7c` | 0.813 | 138 | 32 | 180 | 74 | 19 / 6 / 40 / 9 | 36 | 4 / 9 / 23 | 67 | 3 |
| `S2_c2_tile1280-db269936` | 0.308 | 171 | 46 | 166 | 41 | 16 / 4 / 20 / 1 | 36 | 12 / 7 / 17 | 86 | 3 |
| `S3_c2_copypaste-205b1c7c` | 0.373 | 189 | 52 | 173 | 36 | 17 / 3 / 14 / 2 | 49 | 14 / **16** / 19 | 84 | 4 |
| `S5_rfdetr_large_agnostic-7f6457d4` | 0.657 | 148 | 40 | 163 | 55 | 20 / 4 / 26 / 5 | 27 | 6 / 7 / 14 | 79 | 2 |
| `S6_owlv2_containment-cbc70383` | 0.104 | 55 | 13 | 180 | 138 | 32 / 13 / 74 / **19** | 17 | 16 / 1 / 0 | 18 | 7 |
| `S7_c2_tile1280_copypaste-b6fc182e` | 0.443 | 160 | 45 | 163 | 48 | 18 / 2 / 23 / 5 | 32 | 12 / 4 / 16 | 82 | 1 |
| `G1_rfdetr_medium_faces-cc431fcd` *(faces)* | 0.374 | 213 | 48 | 201 | 36 | 12 / 4 / 18 / 2 | 79 | **56** / 2 / 21 | 74 | 12 |
| `G2_rfdetr_small_faces-7c7b4587` *(faces)* | 0.559 | 152 | 36 | 171 | 55 | 10 / 10 / 30 / 5 | 37 | 18 / 4 / 15 | 70 | 9 |

**Typical failure patterns documented across runs (`devlog/content/2026-11-26-gpu-window-first-results.md:23-30`, `devlog/content/2026-11-26-rfdetr-beats-zero-shot.md:21-26`, `runs/bootstrap_v1/summary.md:53-60`):**
- **Zero-shot adjacent & stacked merging (`B1`, `B3`):** `B1` (`FN:adjacent_merge = 51`) and `B3` (`FN:adjacent_merge = 45`, `FN:stacked_tier = 11`) merge the two-high NNS sweetener stack into one tall box, span multiple neighbouring cartridges with a single wide box, and miss most of the dark, leaning bottom shelf. `B1` also emits `54` background false positives and `33` fragments.
- **RF-DETR localization gains & residual fragments (`C1`, `C2`, `S2`):** Supervised tiled training slashes `FN:adjacent_merge` (`51` -> `15–19`), `FN:stacked_tier` (`11` in `B3` -> `1` in `S2`), and `LOC` errors (`14` in `B1` -> `2–5`). Residual localization errors shift toward `FP:fragment` (`17–35`, partial sub-boxes on cartridges or tile borders) and `FP:duplicate` (`7–15`, including cross-class duplicates at tile seams in `D1` where `FP:duplicate = 15`).
- **`D1` multi-class confusions:** Out of `34` ID errors in `D1`, `23` are known-to-`UNK` (`FL-VAN->UNK`: 6, `NNS->UNK`: 4, `FANTA->UNK`: 3, `FL-LEM->UNK`: 3, `HI-C->UNK`: 3, `SPRITE->UNK`: 3, `FL-ORG->UNK`: 1), and `11` are inter-class confusions, predominantly yellow/orange/cream flavours and `HI-C`/`ING-AA` collapsing toward `FL-VAN` or `FANTA` (`FANTA->FL-VAN`: 2, `ING-AA->FL-VAN`: 2, `FANTA->FL-LEM`: 1, `FL-LEM->FANTA`: 1, `FL-LEM->FL-VAN`: 1, `FL-VAN->FANTA`: 1, `HI-C->FL-VAN`: 1, `SPRITE->FL-VAN`: 1, `SPRITE->NNS`: 1). No two-part brand pair appears anywhere in the matrix — the labels almost never resolve a Part numeral, so Part 1/Part 2 confusion is unmeasurable, not absent.
- **Face-family signature (`G1`, `G2`):** faces halve `FN:adjacent_merge` (16→12) and nearly eliminate `FP:duplicate` (7→2/4), but `FP:background` triples (12→56 in `G1`): a face detector fires on shelf slats, door edges and frame corners, and the face labels themselves carry 53 unresolved duplicate pairs (`runs/faces_v2/summary.md`). `G1` also has the highest `LOC` (12) because a face is small, so a few pixels move IoU a lot (`runs/G1_rfdetr_medium_faces-cc431fcd/failures/failures.json`, [`FAILURES.md`](FAILURES.md) §1).

<!-- GT-PENDING: Re-run freestyle.eval.failures against data/gt/test.json (once human GT exists) and update the FN/FP/ID/LOC breakdown and true confusion pairs against human ground truth from docs/freestyle/RESULTS_GT.md. -->

---

## 10. Identity: Two-Stage VLM, Numeral Readability, Placement Maps & Pair Rule

### 10.1 Two-Stage VLM Identification (`v1` -> `v2`)

In `bootstrap_v1` (one-shot 18-way contact sheet), `inkling-2-mini` left **438 of 563** proposals (`77.8%`) as `unknown`, including almost every bottom-shelf cartridge (`runs/bootstrap_v1/summary.md:49, 56`, `devlog/content/2026-11-26-first-bootstrap-labels.md:21-25`). To separate brand recognition from fine-print numeral reading, `freestyle/vlm_identify.py:158-200` (`mode='two_stage'`) was introduced:

1. **Stage 1 (Family):** Upscale full-resolution crop (`2x` Lanczos, `pad=0.08`) and classify brand family using `FAMILY_CUES` + optional caller-supplied shelf hint (`top` vs `bottom`, split via `freestyle/count/shelves.py` or largest vertical center gap). Adding shelf context eliminated widespread misclassification of red/silver/maroon bottom-shelf cartridges as top-shelf `hi-c` (`devlog/content/2026-11-26-identity-and-ground-truth-plan.md:18-20`).
2. **Stage 2 (Part numeral):** Only for two-part families (`Coca-Cola`, `Diet Coke`, `Dr Pepper`, `Coca-Cola Zero`), crop `part_region=(0.0, 0.5, 1.0, 1.0)` and upscale `3x`, asking strictly for `1`, `2`, or `"unreadable"` (`Do not guess`).

Running `reidentify` over all 563 `bootstrap_v1` boxes (`runs/bootstrap_v2/summary.md:3-16, 49, 70-74`, `686` Inkling calls: `563` family + `123` part stage, `2276.0 s`, `31` cache hits):
- Overall `unknown` dropped from **438 (77.8%)** to **256 (45.5%)** (`307` known-class boxes).
- **Critical finding (2/123):** Out of **123** part-capable two-part family crops, only **2 (1.6%)** yielded a legible numeral (**1** `Part 1`, **1** `Part 2`); **121 of 123** were unreadable at this camera's resolution and blur (`runs/bootstrap_v2/summary.md:70-74`).

### 10.2 Public Reference Imagery, Placement Maps & the Pair Rule Hypothesis

To determine whether external references or layout geometry can resolve Part 1 vs Part 2 (`docs/freestyle/research/reference_imagery.md`, `docs/freestyle/research/placement_maps.md`):
- **Reference imagery (~70 sources):** Public close-ups of individual Freestyle cartridges are scarce (an eBay `AA` listing, an eBay `Cherry` listing, and one 2011 Wikimedia front cabinet photo). No verified public photograph shows a physical `Part 1` next to a `Part 2` end face-on, so physical end-face differences (numeral print, colour band, dimensions) remain unverified from public sources (`docs/freestyle/research/reference_imagery.md:30-35, 49-59`).
- **Double cartridges & placement maps:** Coca-Cola's *Cartridge Management* guide confirms two-part brands ship as glued **double cartridges** (`docs/freestyle/research/placement_maps.md:63`). Across all transcribed public maps—**UK FS9000** (Spring 2018), **UK FS3100** (Spring 2018, vertical left column), and **Canadian 9000/9100 & 8000**—every explicitly labeled two-part brand places `CART 1` / `Partie 1` to the **left of** (or above) `CART 2` / `Partie 2` (`docs/freestyle/research/placement_maps.md:5-59`).
- **Pair rule hypothesis (`freestyle/count/pairs.py:33-112`):** `infer_pairs` groups consecutive same-family columns along a shelf/tier (reversing order when `upside_down=True`), anchors alternation on any confident legible numeral, alternates `Part 1, Part 2` from left to right on even runs, flags `odd_run` (resolving only pairs containing an anchor) and `conflicting_anchors`, and supports `off`, `report`, and `apply` modes. Because public maps show diagram order rather than guaranteed physical installation across all machines, the pair rule remains an explicit **hypothesis** pending human quiz answers.

<!-- GT-PENDING: Record empirical human-GT validation of the pair rule (score_all pair=off vs pair=on accuracy and per-class MAE for coca-cola/diet-coke/dr-pepper/coke-zero part1/part2) from docs/freestyle/RESULTS_GT.md. -->

---

## 11. Limitations & Risks

1. **No human ground truth yet:** All `runs/*/metrics.json` and `failures.json` numbers measure agreement with `bootstrap_v2.json` VLM proposals (or `faces_v2.json` for the G family), not verified human GT. The quiz has been smoke-tested only: **6 of 333** queued questions answered (`data/gt/quiz_answers.jsonl`, one `not_cabinet` negative plus five `skip`s), and no `data/gt/{val,test}.json` export exists.
2. **Quiz GT box dependency — and the face family has no quiz path:** `freestyle.quiz` labels identities and counts on `bootstrap_v2` **whole-cartridge** boxes (`box_source = "model"`), so quiz-derived AP50 measures agreement with human-filtered proposals rather than independent box-drawing GT (`runs/score_all_demo/RESULTS_GT.md:4`). Frames flagged `counts_mismatch` need `freestyle.review.server` box edits. Consequence for §8.9: the quiz can decide faces vs whole boxes on **counts** (a cartridge is a cartridge), but if faces win, face-level box GT does not exist yet and would have to be drawn in the review app over `data/labels/faces_v2.json` before any face-family AP/ID number can be called human-verified.
3. **Single cabinet & slot-memorization hazard:** All 69 frames come from one physical cabinet with no observed stocking changes (`docs/freestyle/DATA_AUDIT.md:305-314`).
4. **Tiny dataset, 96.1% retention loss, & hard test slice:** Only 40 usable cabinet frames (`12` test frames in `11` groups, `7/12` partial, `6/12` severe blur, `0` upside-down in `test`), yielding wide bootstrap CIs (`docs/freestyle/DATA_AUDIT.md:315-351`).
5. **Fine-print numeral ceiling (partly lifted, unverified):** Whole-cartridge crops yielded **2 / 123** legible Part numerals (`runs/bootstrap_v2/summary.md:70-74`); the front-face pass read **59** numerals off 597 cleaned faces (45 `1`, 14 `2`, `runs/faces_v2/summary.md`) — an order-of-magnitude improvement that no human has checked, and the pair rule is still unvalidated (§10.2).
6. **Gated/incomplete experiments:** `B2_sam3_siglip2` is blocked by HF gated access (`facebook/sam3`); `S4`, `A1`–`A3`, `E1`, and `F1` have configs without completed runs (`configs/experiments/`). `S3`, `S5`, `S6`, `S7`, `G1`, `G2` have since been collected.
7. **Privacy & PII:** People appear in **38 / 69** frames; `data/`, `runs/`, and `artifacts/` must stay gitignored and on private hosts/cluster PVCs (`docs/freestyle/DATA_AUDIT.md:325-327`). The galleries referenced from `FAILURES.md` are paths into those gitignored trees, never committed assets.
8. **Bundle `v1` is provisional by construction:** its `score_thr` (0.308) is pinned by the RF-DETR phantom no-object class index rather than chosen on data (§8.6), and its identity thresholds (0.45/0.45) were picked from an unverified visual crop review — all four counting/identity knobs must be re-tuned on `data/gt/val.json` before any bundle is called final (§8.9).
9. **Current working-tree state:** the review-#5 fixes are committed (`5ff55c2`) and `tests/unit/freestyle/test_score_all.py::test_full_scoring_with_fake_runs` passes again — the full unit suite is **398 passed, 9 skipped** (`UV_PROJECT_ENVIRONMENT=/var/tmp/vision-intern-test-venv uv run pytest tests/unit -q`). The only uncommitted changes at the time of writing are the documents and `scripts/finalize_after_gt.sh` produced by this finalization pass.

---

## 12. How to Run

Every command below was verified with `--help` via `uv run python -m ... --help` (except `k8s/*.sh` shell scripts, documented in `k8s/README.md`).

### 12.1 Data Audit & Frozen Splits

```bash
# Audit corpus (69 images -> data/derived/audit/audit.{json,csv} + contact sheets)
uv run python -m freestyle.audit [--no-vlm] [--limit N] [--out-dir DIR]

# Regenerate & verify frozen group-safe splits (data/splits.json)
uv run python -m freestyle.make_splits
uv run python -m freestyle.make_splits --check
```

### 12.2 Collective VLM Bootstrap Proposals

```bash
# Requires LLMC_API_KEY (or ~/.pi/agent/models.json collective key); never write --out under data/gt/
uv run python -m freestyle.bootstrap_labels \
  --config configs/bootstrap/default.yaml \
  --images val \
  --out data/labels/bootstrap_val.json
```

### 12.3 Ground Truth: Phone-First Quiz & Full Review App

```bash
# 1. Build deterministic quiz queue (test first, then val)
uv run python -m freestyle.quiz.build_queue \
  --proposals data/labels/bootstrap_v2.json \
  --splits data/splits.json --splits-order test,val \
  --out data/gt/quiz_queue.json

# 2. Serve quiz (bind to localhost or tailnet IP; never public internet)
uv run python -m freestyle.quiz.server \
  --queue data/gt/quiz_queue.json \
  --answers data/gt/quiz_answers.jsonl \
  --host 127.0.0.1 --port 8766

# 3. Export human-verified COCO + <split>_counts.json mismatch reports
uv run python -m freestyle.quiz.to_coco \
  --queue data/gt/quiz_queue.json --answers data/gt/quiz_answers.jsonl \
  --out data/gt/test.json --split test
uv run python -m freestyle.quiz.to_coco \
  --queue data/gt/quiz_queue.json --answers data/gt/quiz_answers.jsonl \
  --out data/gt/val.json --split val

# Alternative / box-fix path: full-res drag-box review app
uv run python -m freestyle.review.server \
  --proposals data/labels/bootstrap_v2.json \
  --out data/gt/test.json --images test --host 127.0.0.1 --port 8765
```

### 12.4 Cluster Staging & k8s Experiments

```bash
# Stage private data + baseline repo to cluster PVCs
k8s/stage-data.sh [--gpu 5090|6000]
k8s/stage-data.sh --list [--gpu 5090|6000]

# Dry-run, launch, or check status of experiment configs
uv run python -m freestyle.experiments.launch configs/experiments/S2_c2_tile1280.yaml --dry-run
uv run python -m freestyle.experiments.launch configs/experiments/S2_c2_tile1280.yaml
uv run python -m freestyle.experiments.launch --status
```

### 12.5 Evaluation, Cross-Run Scoring & Reports

```bash
# Score all runs against human quiz answers (or --demo dry run)
uv run python -m freestyle.experiments.score_all \
  --queue data/gt/quiz_queue.json \
  --answers data/gt/quiz_answers.jsonl \
  --runs "runs/*" \
  --out docs/freestyle/RESULTS_GT.md
# Dry-run check:
uv run python -m freestyle.experiments.score_all \
  --queue data/gt/quiz_queue.json --runs "runs/*" --demo

# Single-run evaluation, failure analysis, and legacy metrics table
uv run python -m freestyle.eval.evaluate \
  --gt data/gt/test.json --pred runs/<run>/test/predictions.json \
  --select-thr-on data/gt/val.json runs/<run>/val/predictions.json \
  --split-map data/splits.json --out runs/<run>/metrics.json --experiment <exp_id>

uv run python -m freestyle.eval.failures \
  --gt data/gt/test.json --pred runs/<run>/test/predictions.json \
  --images-root data/raw/freestyle-vision-data/images \
  --score-thr <val_thr> --out runs/<run>/failures

uv run python -m freestyle.eval.report --runs-dir runs --out docs/freestyle/RESULTS.md
```

### 12.6 Frozen Bundle Export & CLI Inference

```bash
# Export & verify SHA-256 bundle under artifacts/<name>/
uv run python -m freestyle.artifacts export --run runs/<winning_run> --name <bundle_name>
uv run python -m freestyle.artifacts verify <bundle_name>

# List bundles, count single/multiple frames, or fuse one door-opening burst
uv run python -m freestyle.cli bundles
uv run python -m freestyle.cli count <image...> --bundle <bundle_name> --out runs/cli_preview --json --device cpu
uv run python -m freestyle.cli fuse <frame1.jpg> <frame2.jpg> --bundle <bundle_name> --method max --json --device cpu
```

### 12.7 One command once the human answers exist

Everything GT-dependent is sequenced in `scripts/finalize_after_gt.sh` (see its header for the
flags). It answers the queue first (`freestyle.quiz.server`, bound to the tailnet address), then
checks sufficiency (≥ 80 % of each split's questions answered — it aborts loudly below that),
exports `data/gt/{test,val}.json`, scores every run with `score_all` (1,000 group-bootstrap reps),
reports which `<!-- GT-PENDING -->` placeholders in this report still need numbers, exports the
winning bundle (best **test total-count MAE**, ties broken by identity accuracy) and finishes with a
`freestyle.cli count` demo over three frames.

```bash
scripts/finalize_after_gt.sh --dry-run     # plan + validate paths, mutate nothing, skip the quiz
scripts/finalize_after_gt.sh               # full sequence once data/gt/quiz_answers.jsonl is filled in
```

---

## 13. Appendix

### 13.1 Experiment Configs (`configs/experiments/*.yaml`)

| Config ID | Family | Target (`vram_mb`) | Localizer / Training | Identifier / Embeddings | Status in `runs/` |
|---|---|---:|---|---|---|
| `A1_vlm_flash_full` | `A_vlm` | `local` | `vlm` (`gemini-2.5-flash`, `tile: 0`, `with_identity: true`) | `none` | Not run |
| `A2_vlm_inkling_full` | `A_vlm` | `local` | `vlm` (`inkling`, `tile: 0`, `with_identity: true`) | `none` | Not run |
| `A3_vlm_inkling_tiles` | `A_vlm` | `local` | `vlm` (`inkling`, `tile: 1024`, `with_identity: true`) | `none` | Not run |
| `B1_owlv2_dinov2` | `B_zeroshot_plus_embed` | `k8s:5090` (12000) | `owlv2-large-patch14-ensemble`, `tile: 1024`, `overlap: 0.25`, `score_thr: 0.1` | `none` / `dinov2-base` | Completed (`4c99011d`) |
| `B2_sam3_siglip2` | `B_zeroshot_plus_embed` | `k8s:5090` (12000) | `sam3` (`needs_hf_token: true`), `tile: 1024`, `score_thr: 0.2` | `none` / `siglip2-base-patch16-224` | Blocked (403 HF gated) |
| `B3_gdino_dinov2` | `B_zeroshot_plus_embed` | `k8s:5090` (12000) | `IDEA-Research/grounding-dino-base`, `tile: 1024`, `score_thr: 0.25` | `none` / `dinov2-base` | Completed (`be668339`) |
| `C1_rfdetr_small_agnostic_dinov2` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRSmall` (`res: 512`, `ep: 100`, `bs: 8`, `tile: 1024`) | `none` / `dinov2-base` | Completed (`55ef8ae7`) |
| `C2_rfdetr_medium_agnostic_siglip2` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRMedium` (`res: 576`, `ep: 100`, `bs: 4`, `tile: 1024`) | `none` / `siglip2-base-patch16-224` | Completed (`89fe7c2b`) |
| `D1_rfdetr_small_multiclass` | `D_det_multiclass` | `k8s:5090` (12000) | `RFDETRSmall` (`res: 512`, `ep: 100`, `bs: 8`, `tile: 1024`, 18+1 cls) | `none` / `dinov2-base` | Completed (`6f3c344b`) |
| `E1_rfdetr_agnostic_shelf` | `E_shelf` | `k8s:5090` (12000) | `RFDETRSmall` (`res: 512`, `ep: 50`, `bs: 4`) | `embedding_knn` (`dinov2-base`), `neighbour_vote: report` | Not run |
| `F1_rfdetr_agnostic_vlm_id` | `F_det_plus_vlm_id` | `k8s:5090` (12000) | `RFDETRSmall` (`res: 512`, `ep: 50`, `bs: 4`) | `vlm_crop` (`inkling`) | Not run |
| `S1_c2_tile768` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRMedium` (`res: 576`, `ep: 100`, `bs: 4`, `tile: 768`) | `none` / `siglip2-base-patch16-224` | Completed (`d52f6d7c`) |
| `S2_c2_tile1280` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRMedium` (`res: 576`, `ep: 100`, `bs: 4`, `tile: 1280`) | `none` / `siglip2-base-patch16-224` | Completed (`db269936`) |
| `S3_c2_copypaste` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRMedium` (`res: 576`, `ep: 100`, `bs: 4`, `tile: 1024`) + copy-paste (`n=3`, `slot_shuffle_prob=0.5`, `hybrid`, `persp=0.03`) | `none` / `siglip2-base-patch16-224` | Completed (`205b1c7c`); earlier attempt `e6ce578e` failed |
| `S4_c2_long` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRMedium` (`res: 576`, `ep: 250`, `bs: 4`, `tile: 1024`) | `none` / `siglip2-base-patch16-224` | Not run |
| `S5_rfdetr_large_agnostic` | `C_det_agnostic_plus_embed` | `k8s:5090` (16384) | `RFDETRLarge` (`res: 576`, `ep: 100`, `bs: 4`, `tile: 1024`) | `none` / `siglip2-base-patch16-224` | Completed (`7f6457d4`) |
| `S6_owlv2_containment` | `B_zeroshot_plus_embed` | `k8s:5090` (12000) | `owlv2-large-patch14-ensemble`, `containment_threshold: 0.8`, `max_box_area_frac: 0.08` | `none` / `dinov2-base` | Completed (`cbc70383`) |
| `S7_c2_tile1280_copypaste` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRMedium`, 1280 px tiles + copy-paste | `none` / `siglip2-base-patch16-224` | Completed (`b6fc182e`) |
| `G1_rfdetr_medium_faces` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRMedium` (`res: 576`, `ep: 100`, `bs: 4`, `tile: 1280`), target **`cartridge_face`** from `labels/faces_v2.json` (single variable vs `S2`) | `none` / `siglip2-base-patch16-224` | Completed (`cc431fcd`) |
| `G2_rfdetr_small_faces` | `C_det_agnostic_plus_embed` | `k8s:5090` (12000) | `RFDETRSmall` (`res: 512`, `ep: 100`, `bs: 8`, `tile: 1024`), target **`cartridge_face`** from `labels/faces_v2.json` | `none` / `siglip2-base-patch16-224` | Completed (`7c7b4587`) |
| `Z0_cpu_chain_check` | `B_zeroshot_plus_embed` | `k8s:5090` (`mode: none`) | `owlv2-base-patch16-ensemble`, `device: cpu`, `chain_check: true` | `none` | Passed (`runs/chain-check/Z0_cpu_chain_check-87d5893f`) |

### 13.2 Unit Test Summary

Current status on this branch (`UV_PROJECT_ENVIRONMENT=/var/tmp/vision-intern-test-venv uv run pytest tests/unit -q`, 2026-11-27, after the review-#5 fixes):

```text
398 passed, 9 skipped in 30.74s
```

Earlier historical draft snapshot (superseded; kept so the record of the breakage is not lost):

```text
uv run pytest tests/unit -q
FAILED tests/unit/freestyle/test_score_all.py::test_full_scoring_with_fake_runs
1 failed, 332 passed, 9 skipped in 26.83s
```

- **Passing:** `332` unit tests across core Vision Intern dense-scene extensions (`test_dense_scenes.py`, `test_dense_scenes_review3.py`, `test_bbox_utils.py`, etc.) and `freestyle/` modules (`audit`, `splits`, `bootstrap`, `eval`, `failures`, `identify`, `propose`, `train`, `count`, `augment`, `pipeline`, `artifacts`, `cli`, `quiz`, `review`, `k8s` scripts).
- **Skipped (9):** Optional/GPU-dependency tests (`pycocotools`-gated eval test and GPU/HF-job tests not installed in the CPU `uv` environment).
- **Failing (1):** `tests/unit/freestyle/test_score_all.py::test_full_scoring_with_fake_runs` (`ValueError: too many values to unpack (expected 2)` at `freestyle/experiments/score_all.py:457` in that historical snapshot).
