# Vision Intern → Freestyle cartridge recognition: a code-grounded map

What Vision Intern actually provides, its real limits for **dense, small, many-instance,
fine-grained** cartridge recognition, and where our code hooks in. Every claim cites
`path:line` in the working tree. Contract under test: [`ARCHITECTURE.md`](ARCHITECTURE.md).

> **Tree state.** Line numbers are for core @ `4b96e5f` ("review #3 fixes"), which absorbed
> three dense-scene commits (`fb8c805` opt-in dense support, `289cc0e` sampling params +
> `finish_reason`/usage logging, `4b96e5f` tile-merge/alias/crop-outline fixes) while this
> map was being written. `workflows/train_rfdetr.py` and `jobs/*` are untouched since
> `e328fde`. Re-check `git log --oneline -5 -- tools workflows` before trusting a line
> number; §6/§8 are the sections most sensitive to that churn.

Light path needs no torch — verified: `uv run python -c "import workflows, tools"` (§7).
Training/local-GPU tools need `uv sync --extra train` (`pyproject.toml:25-41`);
`tools/__init__.py:52-118` and `workflows/__init__.py:15-30` import torch lazily.

---

## 1. Workflows (the pipeline stages)

Four public entry points, exported lazily from `workflows/__init__.py:16-21`:
`label_dataset`, `judge_labels`, `gen_descriptions`, `train`; each is also a CLI
(`python -m workflows.<mod>`). **The CLI does not expose the dense-scene knobs**
(`workflows/vlm_label.py:388-439`, `workflows/vlm_judge.py:1000-1083` register only the
pre-dense flags) — dense runs must go through the Python API or the agent registry.

### 1.1 `label_dataset` — VLM pseudo-labelling

```python
# workflows/vlm_label.py:284-307 (abridged)
def label_dataset(source, classes, output, model_id=DEFAULT_VLM,
                  backend="transformers", base_url=None, api_key=None, *,
                  image_column="image", split="train", max_samples=None,
                  push_to_hub=False, hf_token=None, dataset_config=None,
                  dedupe=False, dedupe_key_columns=None,
                  max_dimension=1280, image_format=None, jpeg_quality=85,
                  max_tokens=4096, tile=None, tile_overlap=0.2, n_samples=1,
                  temperature=0, top_p=1, seed=None, extra_body=None,
                  raw_response_log=None, label_aliases=None,
                  keep_unmatched=False, recursive=False, image_paths=None) -> dict | Any
```
Default labeller `Qwen/Qwen3.5-9B` (`workflows/vlm_label.py:34`); the dense knobs
(`max_dimension…image_paths`) are at `:301-307` and are packed into `detect_options` at
`:361-366`, then passed straight to `vlm_detect` (`:143-146` local, `:229-232` hub).

| mode | trigger | input | output |
|---|---|---|---|
| local | `source` is an existing dir (`:38-39`, dispatch `:368-372`) | image dir, optionally **recursive** or an explicit **`image_paths`** list of source-relative ids (`_collect_local_images` `:61-112`, traversal-guarded) | **COCO dict + file**; `file_name` keeps the source-relative posix path (`:151`), annotations are **xywh px** (`:160-167`), `score` is carried when present (`:167`), and a per-image `"truncated": true` flag marks a cut-off response (`:147`, `:152`) |
| hub | otherwise (`:373-380`) | HF dataset (`_label_hub` `:182`, `load_dataset` `:201-203`) | HF dataset / `save_to_disk` with a `detections` column (`:262`) plus a `truncated` bool column when any row was cut (`:263-264`) |

- **Coordinates.** The model answers on the **Qwen 0-1000 grid of the region it was
  shown** (`tools/vlm_detect.py:265`), denormalised by `denormalize_bbox`
  (`tools/vlm_detect.py:110-124`) against the **tile/region size**, then offset into
  **original-image pixels** (`tools/vlm_detect.py:331-335`). Local mode emits xywh
  (`workflows/vlm_label.py:160-167`), hub mode keeps xyxy dicts.
- **Label normalisation is now tolerant but not fuzzy**: `_normalize_detections`
  (`workflows/vlm_label.py:42-60`) → `build_label_resolver` (`tools/bbox_utils.py:146-160`)
  matches exact-casefold and punctuation/underscore-stripped keys, plus an explicit
  `label_aliases` map; unmatched labels are dropped, or kept as `unmatched:<norm>` under
  `keep_unmatched=True` (`workflows/vlm_label.py:52-58`) — in local mode that **appends a
  new category on the fly** (`:156-158`), so class ids are not stable across runs.
- `n_samples > 1` requires `temperature > 0` (`workflows/vlm_label.py:358-359`,
  `tools/vlm_detect.py:226-227`).
- GPU: **no** on the `openai` path. Throughput: hub fans out 16 threads (`:240-245`,
  `LABEL_MAX_WORKERS`); local mode is sequential (`:138-171`).

### 1.2 `gen_descriptions` — the one human-approval gate

```python
# workflows/gen_descriptions.py:104-111
def gen_descriptions(labels, model_id, output, *, backend="openai",
                     base_url=None, api_key=None) -> dict[str, str]
# workflows/gen_descriptions.py:46-53
def collect_labels(source, *, dataset_config=None, split="train",
                   detections_column="detections", drop=(), scan_detections=False,
                   max_scan=None) -> list[str]
```
Writes `{label: definition}` with **every** requested label as a key (empty = undefined,
flagged loudly: `workflows/gen_descriptions.py:130-138`), after `generate_object_specs`
(`workflows/vlm_judge.py:101-159`) runs a meta-prompt demanding one positive,
physical-object definition per label (`workflows/vlm_judge.py:56-88`), re-asking up to 6
times over a 64×64 blank image (`:125`, `:127`, `:136`). No GPU; no images of ours leave
the machine (the blank image is generated, `:125`). `collect_labels` is HF-only
(`:78-101`) → we pass `--labels` (`:155`). This design fits our 18 look-alike catalog ids
(identity = spine text + brand colour,
`data/raw/freestyle-vision-data/hardware/cabinet_and_scene.md:30-34`, `:47`).

### 1.3 `judge_labels` — VLM-as-judge (+ 2-judge ensemble)

```python
# workflows/vlm_judge.py:877-901 (abridged)
def judge_labels(source, output, annotations=None, model_id=DEFAULT_VLM,
                 judges=None, threshold=0.0, min_agree=1, backend="transformers",
                 base_url=None, api_key=None, labels=None,
                 class_descriptions=None, *, image_column="image",
                 detections_column="detections", split="train", max_samples=None,
                 push_to_hub=False, hf_token=None, dataset_config=None,
                 max_dimension=1280, image_format=None, jpeg_quality=85,
                 max_tokens=None, judge_batch_size=None, overlay=False,
                 judge_mode="overlay") -> dict | Any
```
Reusable pieces (what `freestyle/bootstrap_labels.py` imports): `score_detections`
(`workflows/vlm_judge.py:426-511`), `score_detections_batch` (`:522-630`), `ensemble_row`
(`:632-670`), `generate_object_specs` (`:101`). Options are packed at `:955-957` and
forwarded to every judge call (`:836`); local/hub dispatch `:976-993`; descriptions are
auto-generated only when neither `class_descriptions` nor a d1 judge is supplied
(`:964-972`). Detail in [§2](#2-how-the-judge-really-works).

### 1.4 `train` — RF-DETR fine-tune

```python
# workflows/train_rfdetr.py:439-463
def train(source, output_dir="checkpoints/rfdetr-finetuned",
          model_id="Roboflow/rf-detr-large", epochs=20, batch_size=8, lr=5e-5,
          weight_decay=1e-4, gradient_accumulation_steps=1, *, augment=True,
          image_column="image", detections_column="detections",
          annotation_source="auto", train_split="train", val_split="test",
          val_size=0.0, seed=1337, num_workers=4, eval_threshold=0.0,
          push_to_hub=False, hub_model_id=None, report_to="none",
          run_name=None) -> Path
```
GPU: **yes**. Detail in [§3](#3-training-what-it-does-and-what-k8s-needs).

---

## 2. Tools relevant to detection

Registry: `tools/registry.py:69-129` (`_ENTRIES`), `get_tools` (`:310-322`, torch-free by
default), `as_json_schema` (`:334`), `call` (`:339-367`). Endpoint/credential params
(`backend/base_url/api_key/model_id`) are hidden from the schema (`tools/registry.py:66`)
and injected per role (`tools/registry.py:360-366`) from `tools/config.py:45-52`; env
defaults `VISION_AGENT_BACKEND/MODEL/BASE_URL` (`tools/config.py:54-63`). Because the
schema is introspected from the signature, **the new dense params surface to the agent
automatically** (`tools/registry.py:249-283`) — no registry edit needed to use them.

| tool | signature | model loaded (HF id) | GPU | I/O + coordinates |
|---|---|---|---|---|
| `vlm_detect` | `tools/vlm_detect.py:168-183` | caller's `model_id` (default `Qwen/Qwen2.5-VL-7B-Instruct`, `:36`); d1 requires `transformers` (`:231-239`) | no via `openai` | out `[{"bbox":[x1,y1,x2,y2] original px,"label","sub_label",("score")}]`; `score` appears **only** with `n_samples>1` (`:370-384`); the returned list carries `.truncated` and `.did_split` (`:292-294`) |
| `run_vlm` / `run_vlm_batch` | `tools/vlm_client.py:323-335` / `:394-409` | n/a (transport) | no via `openai` | returns a `VLMResponse` (`str` subclass) carrying `finish_reason` + `usage` (`:31-42`, `:161`); transformers-side truncation is *derived* (`_derive_finish_reason` `:212-228`); `raw_response_log` writes one JSON object per call incl. usage (`:305-311`, `:389-390`, `:448-450`); `run_vlm_batch` on `openai` is **sequential** (`:436-442`) |
| `detect` | `tools/detect.py:25-28` | `Roboflow/rf-detr-large` (`:11`) | **yes** | closed-set COCO-80 → `{"label","score","box" xyxy px}` (`:50-56`), `threshold=0.3` |
| `grounded_detect` | `tools/grounded_detect.py:31-37` | `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det` / `…_large_all` (`:13-16`), **float32** (`:23`) | **yes** | open-vocab → `{"label","score","box" xyxy px}` (`:72-80`); `threshold=0.3`, `text_threshold=0.25` (`:34-35`); **`num_select` not exposed** (`:66-71`) |
| `segment_from_text` | `tools/segment_from_text.py:26-29` | `tiiuae/Falcon-Perception` (`:11`) | **yes** | text→instances; normalized `center`/`size` + COCO RLE at original res (`:36-56`) |
| `instance_segment` | `tools/instance_segment.py:25-28` | `Roboflow/rf-detr-seg-large` (`:11`) | **yes** | closed-set COCO-80; `(H,W)` int32 seg + `segments_info` (`:51-57`) |
| `segment_from_bbox` (SAM) | `tools/segment_from_bbox.py:29-32` | `facebook/sam3` tracker (`:15`) | **yes** | in xyxy px → `{"mask":(H,W) bool at original res,"iou_score"}` (`:60-68`) |
| `fast_segment` (SAM-lite) | `tools/fast_segment.py:25-28` | `yonigozlan/EdgeTAM-hf` (`:11`) | **yes** | same contract, cheaper (`:29-40`) |
| `ocr` | `tools/ocr.py:151-156` | `lightonai/LightOnOCR-3-0.8B`, `datalab-to/chandra-ocr-2`, `rednote-hilab/dots.mocr`, `datalab-to/surya-ocr-2`, `PaddlePaddle/PaddleOCR-VL-1.6` (`:23-29`) | **yes** | `{"text","model"}`; `size="lighton", prompt="grounding"` puts 0-1000 layout boxes in the *text* (`README.md:262-272`) |
| `ocr_judge` | `tools/ocr_judge.py:86-94` | any VLM (`:32`) | no via `openai` | pairwise A/B/tie, `max_tokens=512` (`:125`) |
| `d1_detect/decide/review/judge` | `tools/d1.py:76`, `:37`, `:169`, `:126` | `LiquidAI/d1-Omni-2B` (`:16`) | **yes** | System One API, `transformers` only (`is_d1` `:19`) |
| CPU helpers | `convert_bbox` `tools/bbox_utils.py:42`; **`bbox_iou` `:115`, `containment` `:127`, `nms` `:223`**; `normalize_label_key` `:141`, `build_label_resolver` `:146`; `validate_annotations` `:260`; `compute_stats` `:346`; `dedupe_by_image` `tools/dataset_utils.py:108`; `grouped_train_val_split` `:147`; `draw_detections` `tools/bbox_viz.py:44`; `annotate` `tools/sv_viz.py:109`; `push_dataset_with_viz` `tools/hub_viz.py:60`; `run_store` `tools/run_store.py:40`,`:63` | — | no | 6 bbox formats (`tools/bbox_utils.py:15-24`); `nms` is **same-label only**, prefers `(score, area)` (`tools/bbox_utils.py:226-241`) |

Irrelevant to us: `estimate_depth` (`tools/depth.py:27`), Sapiens2 pose/body/normal/matting
tools (`tools/pose.py:25` etc.), `track_video`.

### Pointing the VLM paths at the LLM Collective

One transport, four knobs (`tools/vlm_client.py:112-169`; default base_url is the HF
router `:46`, `:127`):

```bash
uv run python -m workflows.gen_descriptions --labels "$CLASSES" \
  --model qwen3-8-flash-next --backend openai \
  --base-url https://llm-collective.e-gineering.com/v1 --api-key "$COLLECTIVE_API_KEY" \
  --output descriptions.json
```
```python
import vision_agent as va
from freestyle.collective import read_provider_config     # key read at runtime, never stored
base_url, key = read_provider_config()                    # freestyle/collective.py:73-88
va.configure(labeller=va.ToolConfig(backend="openai", base_url=base_url,
                                    model_id="qwen3-8-flash-next", api_key=key),
             judge=va.ToolConfig(backend="openai", base_url=base_url,
                                 model_id="david-birks/inkling-2-mini", api_key=key))
```
Caveats, all in `tools/vlm_client.py`: `api_key` silently falls back to
`HF_TOKEN`/`OPENAI_API_KEY`/`"no-key"` (`:103-110`) — always pass it; retries are 3 attempts
with exponential backoff and no per-attempt timeout (`:141-168`); `temperature=0`,
`top_p=1`, `seed` are now real params (`:122-123`, `:153-155`) — pass `seed` for
reproducibility; `extra_body.chat_template_kwargs` is only auto-added for two hard-coded
Qwen ids (`tools/model_options.py:10-19`, `tools/vlm_client.py:132-136`), so our model names
go out as plain chat requests (no 400s) and `enable_thinking` control is **ours** to send via
`extra_body`; `extra_body` is rejected on the transformers backend (`:383-384`, `:430-431`).
`tools/config.py` never holds the key in env-derived config (`tools/config.py:59-62`).
`freestyle/collective.py:73-88` (`read_provider_config`) + `chat_image(..., max_side=None)`
(`freestyle/collective.py:167`) are our key-free-of-disk, no-downscale client;
`configs/bootstrap/default.yaml:1-2` points at it (`base_url`, `api_key_env: LLMC_API_KEY`).

---

## 3. Training: what it does, and what k8s needs

**Detector.** Any `AutoModelForObjectDetection` (`workflows/train_rfdetr.py:535`); default
`Roboflow/rf-detr-large` (`:50`); classification head re-inited via
`id2label`/`label2id` + `ignore_mismatched_sizes=True` (`:536-540`). No segmentation
(`README.md:13`).

**Input resolution / query budget.** Taken from the checkpoint's own processor
(`workflows/train_rfdetr.py:522`, applied in `_transform_batch` `:305-316`): for
`rf-detr-large` that is `size={704,704}, do_resize/do_pad` (published
`preprocessor_config.json`), so a 2048×1536 frame is squeezed into 704². The decoder's
`num_queries` (300 for `rf-detr-large`, published `config.json`) is **never overridden**
(`:536-540`). No image-size knob is exposed (`:439-463`) — which is why our k8s training
path uses the `rfdetr` package directly with an explicit `resolution`
(`freestyle/train/train_detector.py:31-43`) and an Ultralytics YOLO alternative
(`:45-58`) instead of `workflows.train_rfdetr.train`.

**Augmentation** (`workflows/train_rfdetr.py:334-352`, on by default `:449`):
`A.Perspective(p=0.1)`, `A.HorizontalFlip(p=0.5)`, `A.RandomBrightnessContrast(p=0.5)`,
`A.HueSaturationValue(p=0.1)`, `BboxParams(format="coco", clip=True, min_area=25)`.

**How it consumes COCO / detections.** Three formats auto-detected
(`workflows/train_rfdetr.py:5-22`): local `<dir>/train/images/` + `<dir>/train/labels.json`
(+ optional `val/`) via `_load_local_coco` (`:163-206`); HF `objects` (`:69-97`); HF
`detections` xyxy→xywh (`:100-124`), selected by `annotation_source`
(`tools/dataset_utils.py:29-62`, used at `workflows/train_rfdetr.py:233-236`).
Only `bbox`/`category_id`/`area`/`image_id` are read (`:176-190`) — `attributes`, `quad`,
`score` are ignored; **images with zero boxes are filtered out** (`:252`, val `:267`,`:273`).
Class ids are re-derived (local COCO re-map `:171-173`; hub sorted-alphabetically
`_categories_from_detections` `:148-160`), so **pin your own catalog→id map** —
`freestyle/catalog.py:68-88` (`category_map`/`category_id`) is ours.

**Outputs.** Checkpoints per epoch (`save_strategy="epoch"`, `save_total_limit=2`,
`:556-558`), then `output_dir/final` (`:588-590`), optional `push_to_hub` (`:591-592`).
Hyperparams: cosine LR, `max_grad_norm=0.01`, `warmup_ratio=0.05`, bf16 on CUDA,
`load_best_model_at_end` when a val set exists (`:543-570`). Eval per epoch = torchmetrics
`MeanAveragePrecision(box_format="xyxy", class_metrics=True)` (`:389-436`, `:412`) —
`map`, `map_50`, `mar_100`, per-class `map_<name>` (`:424-434`). **No prediction export.**

**How `jobs/` runs it today = HF Jobs, not k8s.** `jobs/train_rfdetr_job.py` is a PEP-723
script (`:1-20`) that clones this repo from GitHub at `REPO_REF` (`:41-48`) and shells out to
`python -m workflows.train_rfdetr` **always with `--push-to-hub` + required
`--hub-model-id`** (`:80-97`); launch pattern `hf jobs uv run --flavor l4x1 …`
(`jobs/README.md:185-190`). Same coupling in `jobs/judge_one.py:53-60`,
`jobs/merge_judges.py:41-48`, `jobs/label_qwen.py:36-42`.

**To run training in a plain k8s Job** (our rules: GPU only on the home cluster, nothing
pushed to HF):
1. bake the repo into the image / initContainer from our own registry → delete the clone
   bootstrap (`jobs/train_rfdetr_job.py:41-48`);
2. `--source` a **local COCO dir on the PVC** (`workflows/train_rfdetr.py:163-206`), pre-cut
   by *group* ourselves ([§8](#8-ranked-extension-points-with-hook-sites) row 6);
3. drop `--push-to-hub`/`--hub-model-id` (`jobs/train_rfdetr_job.py:93-95`), write
   `--output-dir` to the PVC, `report_to="none"` (`workflows/train_rfdetr.py:461`, `:565`);
4. put the `train` extra deps in the image (`pyproject.toml:25-41`) instead of PEP-723;
5. add the missing export step → `runs/<exp>/predictions.json` + `meta.json`
   (`docs/freestyle/ARCHITECTURE.md` §Prediction format; reference inference loop
   `jobs/eval_vs_gt.py:100-137`, `jobs/eval_overlay.py:57-75`).
`REPO_DIR=$(pwd)` is the "run against my working tree" knob (`jobs/README.md:110-118`), and
`jobs/*.py` are compile/PEP-723-guarded by `tests/unit/test_job_scripts.py`. Our rendered
k8s templates live in `k8s/job-template.yaml` (+ `k8s/submit.sh`, `k8s/lib.sh`,
`k8s/stage-data.sh`, `k8s/collect.sh`).

---

## 4. How the judge really works

**What the judge sees.** Either a **numbered overlay** (each box drawn 3px thick, tagged
`#i label`: `tools/bbox_viz.py:66-88`, `:73-76`, `show_index` `:49`) or — new — a
**crop sheet** of padded, numbered crops cut from the **original pixels**
(`_crop_sheet` `workflows/vlm_judge.py:356-401`; 15 % padding `:365-369`, each crop
thumbnailed to ≤256 px `:380`, tiles laid out `:382-389`), chosen by `judge_mode="crops"`
(`:457-473`). Raw pixel coordinates are never shown to the model in these modes; the
rationale is in code (`workflows/vlm_judge.py:440-448`). Local mode without
`overlay`/`judge_mode`/`judge_batch_size` still uses the **raw-coordinate COCO prompt**
(`workflows/vlm_judge.py:744-758`, prompt `:201-240`).

**Prompt shape** (`_build_judge_prompt_overlay` `workflows/vlm_judge.py:161-199`): the
approved definitions for the labels present on *that* image (`:178-183`), the numbered
proposals, two questions (present & matches definition; box reasonably tight), and a
strict JSON-array contract
(`workflows/vlm_judge.py:192-197`):
`[{"id","verdict":"correct"|"incorrect"|"imprecise","score":0-1,"reason"}]`.

**Parsing / failure semantics.** Tolerant parse — strip `thinking` text, salvage arrays,
coerce `"#0"`/`"box 3"` ids (`_parse_verdicts` `:252-316`); a dead judge returns `"[]"`
rather than aborting (`:506-507`); truncation is warned (`:508-509`). **A box the response
never reached is now `{verdict:"unjudged", score:0.0, reason:"no verdict"}`**
(`:318-320`, `:513-519`) instead of a silent `incorrect` — auditable, but `ensemble_keep`
still requires `verdict == "correct"` votes (`:659-667`), so unjudged boxes are still
dropped unless `min_agree=0`. Degenerate/empty boxes are pre-rejected in crop mode
(`_sanitize_bbox` `:328-353`, `_INVALID_BOX_VERDICT` `:321-326`).

**Chunking.** `judge_batch_size` splits one image's boxes into chunks — crop-sheet mode
defaults to 16 crops per sheet (`:457-459`, `_prepare_crop_chunks` `:403-424`), overlay mode
recursively judges slices and remaps indices (`:474-486`); the GPU batch path mirrors it
(`:522-585`). Token budget per call is `max_tokens or (16384 if "lfm3-vl" in model_id else
1024)` (`:502`) — i.e. **1024 unless you pass `max_tokens` or name the model lfm3**, so the
per-name special case is a trap for new models (`:502`, and the same rule in local mode
`:751`).

**Ensemble merge.** `ensemble_row(per_judge_row, n_dets, min_agree)`
(`workflows/vlm_judge.py:632-670`) → `{"detection_idx","per_judge",
"n_correct","mean_score","ensemble_keep": n_correct >= min_agree}` (`:661-668`). Judge count
is capped at 1-2 (`tools/model_options.py:22-30`, enforced `workflows/vlm_judge.py:644`,
`:684`, `jobs/merge_judges.py:78-81`). Keep rule: `ensemble_keep AND mean_score >= threshold`
(`workflows/vlm_judge.py:843-847`; local variant `:759-764`).

**Geometric checks.** Not in `judge_labels` at all — only the merge job's non-VLM
page-spanning guard: `area_frac`, `geom_keep = area_frac <= --max-area-frac` (default 0.9),
recorded per detection (`jobs/merge_judges.py:111-120`, `:134-140`, arg `:68-71`).
**The judge can only delete, never add** — recall is fixed by labelling.

**Plumbing.** Jobs path writes one parquet per judge `{row_idx, model, verdicts}` into a
shared bucket (`jobs/judge_one.py:136-156`; root resolution `tools/run_store.py:40-79`,
default bucket `tools/run_store.py:32`); `jobs/merge_judges.py` re-joins by `row_idx`
(`:97-103`, `:129-130`).

---

## 5. Evaluation code that exists

| script | reference | metrics | notes for us |
|---|---|---|---|
| `jobs/eval_vs_gt.py` | `:52-164` | `map, map_50, map_75, map_small, map_medium, map_large, mar_1, mar_10, mar_100` (`:146-149`) + per-class mAP/mAR100 (`:151-160`) | Requires an **HF dataset with a human `objects` ClassLabel column** (`:82-88`, `:118-133`); classes bridged **by name**, out-of-vocab GT boxes dropped and counted (`:74-75`, `:122-125`, `:137-140`); `MeanAveragePrecision(...)` with torchmetrics default `max_det=100` (`:90`) |
| `jobs/eval_pr_roc.py` | `:117-275`, CLI `:277-300` | per-prediction TP/FP with greedy per-image/per-class match at `--iou` (default 0.5) (`:83-114`), PR-curve AP (`:212-224`), ROC/AUC over predictions = **confidence separation, not specificity** (`:226-233`, docstring `:31-38`), P/R/F1 at `--op-threshold` (`:235-244`), per-class P/R (`:245-260`), result dict `:262-273` | Same HF-`objects` GT dependency (`:181-198`); `--class-aliases`/`--exclude-classes` (`:288-296`); uploads plots + `metrics.json` **to a Hub repo** (`:351-412`) |
| `jobs/eval_overlay.py` | `:24-45` | mAP/mAR against the dataset's `detections` **pseudo**-labels + GT-vs-pred PNG pairs | Measures self-agreement, not truth |
| `workflows/train_rfdetr.py:389-436` | — | per-epoch mAP/mAR | same caveat |
| `jobs/job_ledger.py` | `:1-30` | HF job wall-time/cost | HF-specific |

**Nothing in the repo computes** identification top-1/macro-F1 over matched boxes,
per-image count MAE, exact-inventory rate, or group-bootstrapped CIs — all of
[ARCHITECTURE.md §Evaluation] is ours (`freestyle/eval/matching.py`, `metrics.py`).

---

## 6. Known limitations relevant to us

Upstream now ships opt-in dense-scene knobs (`agents.md:222-229`,
`examples/dense-scenes/README.md`, tested by `tests/unit/test_dense_scenes.py`). The honest
statement of our position: **the knobs exist, the defaults are still the document-era
defaults**, and several of our failure modes are only half-covered.

### 6a Still-open, in rough order of harm

- **L1 — Defaults still downscale + re-encode.** `max_dimension` defaults to 1280
  (`tools/vlm_client.py:53`, `:56`, `:70-74`) and JPEG q85 (`:77-81`); our frames are
  1600×1200 / 2048×1536 (`data/raw/freestyle-vision-data/data/images.csv`) → 0.80/0.625
  scale unless every caller passes `max_dimension=None, image_format="png"` (as
  `freestyle/bootstrap_labels.py:172-177` does). Miss it once and ~60 px faces become 37 px.
- **L2 — Judging identity detail is capped at 256 px per crop** in crop mode
  (`workflows/vlm_judge.py:380`), and the crop sheet packs 16 crops into one image
  (`:459`, `:382-389`) which is then itself resized to ≤1280 by default (`:499-504`).
  Spine text is exactly what we ask the judge to verify
  (`data/raw/freestyle-vision-data/hardware/cabinet_and_scene.md:47`) → per-crop calls at
  full resolution remain ours (`freestyle/identify/`, `freestyle/vlm_identify.py`).
- **L3 — Per-call answer budget is still the ceiling.** Detection default
  `max_tokens=4096` (`tools/vlm_detect.py:178`), judge default 1024 for any model whose
  name doesn't contain `lfm3-vl` (`workflows/vlm_judge.py:502`, `:751`). Now *detectable*
  (`finish_reason == "length"` `tools/vlm_client.py:161`, derived for transformers
  `:212-228`) and *self-healing* — a truncated tile call recursively splits into quadrants
  once (`tools/vlm_detect.py:315-323`, `_quadrants` `:146-166`) and sets
  `.truncated`/`.did_split` (`:292-294`, warned `:326-327`) — but a truncated **judge**
  response is only warned about (`:508-509`) and the un-reached boxes become `unjudged`
  (`:318-320`) → still deleted by the ensemble rule (`:659-667`). Pass `max_tokens` ≥ 4096
  and chunk, always.
- **L4 — No confidence without sampling.** `vlm_detect` emits `score` only in the
  `n_samples>1` self-consistency path, where score = fraction of samples that agreed
  (`tools/vlm_detect.py:370-384`) and temperature must be > 0
  (`tools/vlm_detect.py:226-227`). Single-shot labelling still yields nothing to threshold
  or rank (`:337-342`), so our `predictions.json` `score`/`id_score` are still ours to
  produce (multi-run fusion: `freestyle/bootstrap_labels.py:188`).
- **L5 — Dedupe is conditional and label-locked.** `nms` runs only when tiling or after a
  truncation split (`tools/vlm_detect.py:367`) and only suppresses **same-label** overlap
  (`tools/bbox_utils.py:223-252`, `:234`); a plain full-frame call keeps every duplicate.
  Cross-label duplicates (our 18 catalog ids are look-alikes) survive everywhere.
- **L6 — 0-1000 quantisation per region.** Boxes arrive on a 1000-step grid
  (`tools/vlm_detect.py:265`, `:110-124`) → ≈1-2 px steps on a 1024–1536 px tile, on objects
  of ~60-150 px, then `round()` (`:117-122`). Tiling actually *improves* this (grid is
  per-region, `:331`), full-frame does not.
- **L7 — No fuzzy label recovery.** The resolver handles case/punctuation/aliases only
  (`tools/bbox_utils.py:141-160`); `coke zero part1` will **not** resolve to
  `coca-cola-part1`/`coke-zero-part1` without an explicit alias entry — and a missed label
  in local mode silently creates a new COCO category (`workflows/vlm_label.py:156-158`).
  Feed `label_aliases` from `freestyle/catalog.py` (`aliases` in
  `data/raw/freestyle-vision-data/data/catalog_items.json`).
- **L8 — Two judge slots, one usable vision family here.** Cap at 2
  (`tools/model_options.py:26-30`); on the Collective only `qwen3-8-flash-next` and
  `david-birks/inkling-2-mini` declare image inputs (`~/.pi/agent/models.json`
  `providers.collective.models[].input`); `qwen3-8-27b` is text-only → orchestrator/
  description writer, not a vision judge.
- **L9 — Trainer drops zero-box images** (`workflows/train_rfdetr.py:252`, `:267`, `:273`)
  → no negatives → hallucinated cartridges in `no_cabinet` frames
  (`docs/freestyle/ARCHITECTURE.md` §Annotation format).
- **L10 — Augmentation is label-breaking for us.** `HorizontalFlip` mirrors shelf side,
  `HueSaturationValue` scrambles brand colour (`workflows/train_rfdetr.py:344-347`),
  `min_area=25` drops sub-5×5 px boxes after aug (`:349-350`). Upstream precedent: no-aug
  0.685 vs aug 0.30 (`examples/roadsign-detection/README.md:95-103`, `agents.md:183-188`).
- **L11 — `label_dataset` never writes `detections_overlay`** (only
  `jobs/push_labeled.py:89-97`, `jobs/build_combined_labeled.py:105-110`), so the hub judge
  re-renders overlays (`workflows/vlm_judge.py:487-490`) and expects a column it won't find
  (`:830`) — and `agents.md:91-93` still claims the labeller produces it.
- **L12 — Hub coupling in every non-local path**: `push_to_hub`/`push_dataset_with_viz`
  (`workflows/vlm_label.py:267-272`; `workflows/vlm_judge.py:860-865`), the `jobs/*` clone
  bootstrap, `run_store` default bucket `merve/vision-agent-runs`
  (`tools/run_store.py:32`), and the `vision-e2e-runner` instruction (HF router `:41-45`,
  HF Jobs `:84-104`, Hub pushes `:106-140`). Forbidden for us → local mode + our k8s.
- **L13 — Splitting is Hub + image-hash/group-column** (`jobs/make_splits.py:23-56`,
  `tools/dataset_utils.py:147-153`); our independent unit is the door-event uuid, which is
  not a column → `freestyle/make_splits.py:101` + `freestyle/data.py:87-95`,`:224`.
- **L14 — mAP eval caps at 100 detections/image** (torchmetrics default `max_det`; never
  set: `workflows/train_rfdetr.py:412`, `jobs/eval_vs_gt.py:90`) — irrelevant at ≤20
  boxes/frame, relevant the moment we evaluate tiles as separate images.
- **L15 — No quad/polygon training** (`README.md:13`; objects schema is bbox-only,
  `workflows/train_rfdetr.py:57-64`) while our format wants `quad` corners.

### 6b What the dense-scene commits already fixed (so we don't re-build it)

No-downscale/PNG/quality params (`tools/vlm_client.py:56-83`), tiling with overlap and
edge-truncation rejection (`tools/vlm_detect.py:273-276`, `:346-350`, `:126-145`, `:335`),
token-budget param + truncation detection/quad-split (`:315-323`), same-label NMS helper
(`tools/bbox_utils.py:223-252`), self-consistency `score` (`tools/vlm_detect.py:370-384`),
tolerant label resolution + `keep_unmatched` audit trail
(`workflows/vlm_label.py:42-60`), recursive/`image_paths` local dirs preserving relative
paths (`:61-112`, `:151`), judge crop-sheet + chunked judging
(`workflows/vlm_judge.py:356-401`, `:403-424`, `:457-486`), `unjudged` instead of fake
`incorrect` (`:318-320`), and `finish_reason`/`usage`/`raw_response_log` audit trail
(`tools/vlm_client.py:31-42`, `:305-311`).

---

## 7. Runbook: light path

```bash
uv sync                                              # no torch (pyproject.toml:13-20)
uv run python -c "import workflows, tools"           # ✓ no ImportError
uv run python -c "import tools; print(tools.list_tools(include_train=False))"
#   ['compute_stats','convert_bbox','judge_labels','label_dataset','ocr_judge',
#    'validate_annotations','vlm_detect']            # tools/registry.py:305-322
uv run pytest tests/unit -q                          # incl. tests/unit/test_dense_scenes.py
uv run --extra train python -c "import workflows.train_rfdetr"   # GPU path; host-forbidden
```
No GPU code run on this host (project rule). RF-DETR facts in §3 came from the published
`config.json` / `preprocessor_config.json` (read-only HTTPS GETs), not execution.

---

## 8. Ranked extension points (with hook sites)

"What's there" = already in the tree (upstream core or our `freestyle/` package) — don't
rebuild. Ranked by metric gain per unit of work.

| # | extension | hook site(s) | status |
|---|---|---|---|
| 1 | **Full-resolution tiled localisation** (`cartridge`, class-agnostic) | `workflows/vlm_label.py:284-307` (`tile`, `tile_overlap`, `max_dimension=None`, `image_format="png"`, `max_tokens`), consumed at `:143-146`; per-region denorm+offset in `tools/vlm_detect.py:331-335` | **there**; driven per-image with a fusion step in `freestyle/bootstrap_labels.py:172-188`; tile geometry helper `freestyle/geometry.py:46` |
| 2 | **Cross-tile / cross-run merge with a real `score`** | `tools/bbox_utils.py:223` (`nms`) — but it is same-label only (`:233-236`), so cross-label vote fusion lives above it | **ours**: IoU fusion across labeller runs `freestyle/bootstrap_labels.py:188` (`fusion_iou`, `configs/bootstrap/default.yaml:3`); proposal runner `freestyle/propose/run.py:204`,`:287`; tiled inference `freestyle/infer/detector.py:43` |
| 3 | **Detect-then-identify: per-crop 18-way identification at full res** | crops cut from the ORIGINAL file + box (`tools/bbox_utils.py:42`); model call `freestyle/collective.py:167` `chat_image(..., max_side=None)`; must not go through `tools/vlm_client.py:56` defaults (L1/L2) | **ours**: `freestyle/vlm_identify.py`, `freestyle/identify/{classify,embed,cli}.py`; wired in `freestyle/bootstrap_labels.py:209-215`; end-to-end `freestyle/pipeline.py:151` |
| 4 | **Judge chunking + explicit token budget (+ per-crop judging for text-heavy IDs)** | `workflows/vlm_judge.py:457-486` (`judge_batch_size`, `judge_mode="crops"`), token default at `:502` | chunking/sheets **there** (used `freestyle/bootstrap_labels.py:196-204`); per-crop identity review **ours** (`freestyle/review/server.py` for the human pass) |
| 5 | **Catalog-driven alias + class-id pinning** | `workflows/vlm_label.py:42-60` (`label_aliases`, `keep_unmatched`), `tools/bbox_utils.py:141-160`; class space: `workflows/train_rfdetr.py:148-160` vs our pinned map | resolver **there**; alias table + `catalog→id` **ours** (`freestyle/catalog.py:62-88`, aliases from `data/raw/freestyle-vision-data/data/catalog_items.json`) |
| 6 | **Group-safe splits → local COCO dirs the trainer reads** | `jobs/make_splits.py:23-56` pattern, `tools/dataset_utils.py:147-153`, consumed by `workflows/train_rfdetr.py:163-206` | **ours**: `freestyle/make_splits.py:101`, `freestyle/data.py:87-95`,`:197-223` |
| 7 | **Prediction export (`runs/<exp>/predictions.json` + `meta.json`)** | nothing in `workflows/train_rfdetr.py` writes it; inference reference `jobs/eval_vs_gt.py:100-137`, `jobs/eval_vs_gt.py:108` (`post_process_object_detection`) | **ours**: `freestyle/infer/detector.py:74` (`predict_coco`, `class_agnostic`, `split_map`) |
| 8 | **Our evaluation (AP50/95, top-1 ID, count MAE, exact-inventory, group bootstrap)** | replaces the HF-`objects` assumptions of `jobs/eval_vs_gt.py:82-88` / `jobs/eval_pr_roc.py:181-198` | **ours**: `freestyle/eval/matching.py`, `metrics.py:430`/`:566`/`:674`/`:698`/`:745`, CLIs `freestyle/eval/evaluate.py`, `failures.py`, `report.py`; counting `freestyle/count/{aggregate,shelves}.py` |
| 9 | **k8s GPU jobs instead of HF Jobs** | strip clone+push from `jobs/train_rfdetr_job.py:41-48`,`:93-95`; keep the flow (label → judge → merge → train) | **ours**: `k8s/job-template.yaml`, `k8s/submit.sh`, `k8s/lib.sh`, `k8s/stage-data.sh`, `k8s/collect.sh`; training with an explicit `resolution` (bypasses the 704² input cap, §3) `freestyle/train/train_detector.py:31-43`, tiles `freestyle/train/export_tiles.py`; orchestration `freestyle/experiments/{schema,launch,entry}.py` + `configs/experiments/*.yaml` |
| 10 | **Keep zero-box frames in training (negatives)** | `workflows/train_rfdetr.py:252` — moot on the `rfdetr`/YOLO package path, needed if we ever use the HF trainer | todo upstream flag `keep_empty`; our path: include `no_cabinet`/empty images in the tile dataset |
| 11 | **Augmentation policy for colour/position-coded classes** | `workflows/train_rfdetr.py:334-352`; our configs `aug_config`/`yolo_augment` (`freestyle/train/train_detector.py:38-40`,`:45-58`) | ours: `freestyle/augment/{copy_paste,photometric}.py`; default = no flip, no hue jitter |
| 12 | **Open-vocab / specialist prior as stage A (real scores + NMS)** | `tools/grounded_detect.py:31-37` (needs `num_select` at `:66-71` + tiling) | ours: backends `freestyle/propose/backends.py`, experiments `B1_owlv2_dinov2`, `B2_sam3_siglip2` |
| 13 | **SAM-class mask → tight boxes / `quad` corners** | `tools/segment_from_bbox.py:29-32` (`facebook/sam3`), `tools/fast_segment.py:25-28` (EdgeTAM) | later; masks→quad helper belongs in `freestyle/geometry.py` |
| 14 | **OCR-assisted identification (spine text)** | `tools/ocr.py:151-156` (GPU, HF weights) vs a Collective crop call via `freestyle/collective.py:167` | later; prefer the Collective path (single provider rule) |
| 15 | **Second/third ensemble member** | relax `tools/model_options.py:26-30`; keep per-judge `base_url`/`model_id` (already supported: `workflows/vlm_judge.py:672-702`) | TBD: needs a second vision family on the Collective |
| 16 | **`max_side`/`extra_body` policy for Collective thinking models** | `tools/vlm_client.py:132-136`, `tools/model_options.py:10-19` (only 2 ids get `enable_thinking` today) | ours: pass `extra_body` explicitly, or extend the id set upstream |
| 17 | **Our own agent-facing tool layer** | `tools/registry.py:69-129` `_ENTRIES` (add `freestyle` tools), role config `tools/config.py:69-86` | later — the in-process registry already works against the Collective |
| 18 | **Do not follow the upstream e2e agent instruction** | `.claude/agents/vision-e2e-runner.md:84-140` (push the branch, judge on HF Jobs, merge+push both policies) | keep its *discipline* (descriptions gate, smoke-20-then-scale, emit both agree-policies); its plumbing is HF-only and forbidden |

**Upstream defects found while mapping** (reported, not patched here):
`agents.md:91-93` and `jobs/README.md:90-96` state that labelling produces
`detections_overlay`, which `label_dataset` never does (L11); the judge token budget is keyed
on the model-name substring `"lfm3-vl"` (`workflows/vlm_judge.py:502`, `:751`) so every new
model silently inherits 1024 tokens (L3); `nms` silently no-ops outside tiled/split
labelling (`tools/vlm_detect.py:367`, L5); and the judge CLIs expose none of the dense
options (`workflows/vlm_judge.py:1000-1083`), so the documented `python -m workflows.*`
quickstart cannot reach the behaviour `examples/dense-scenes/README.md` advertises.
