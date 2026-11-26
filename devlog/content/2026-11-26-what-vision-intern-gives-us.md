---
title: What Vision Intern gives us
date: 2026-11-26T12:30:00-05:00
tags: [vision-intern, tooling, dense-scenes, architecture]
---

Before writing custom training and evaluation code under `freestyle/`, we mapped the upstream `vision-intern` codebase line by line (`docs/freestyle/VISION_INTERN_MAP.md`). Upstream gives us a clean three-stage agent toolkit—**label -> judge -> train**—plus composable vision tools, while leaving several dense-scene and domain-specific gaps that `freestyle/` bridges.

## What comes out of the box

The core repository splits cleanly into a **torch-free light path** (for VLM orchestration, bbox math, and dataset management) and an optional `--extra train` GPU path:

- **Workflows (`workflows/`):** `label_dataset` (VLM pseudo-labelling to COCO JSON or HF Dataset), `judge_labels` (1–2 judge verification via numbered overlays or 16-up crop sheets), `gen_descriptions` (human-approved visual definitions), and `train` (`AutoModelForObjectDetection` fine-tuning with torchmetrics mAP).
- **Dense-scene extensions (already merged into core):** Opt-in `max_dimension=None`, `image_format="png"`, overlapping `tile` + `tile_overlap`, explicit `max_tokens`, `n_samples` self-consistency scoring, `finish_reason == "length"` truncation detection with automatic quadrant splitting, `label_aliases`, and `unjudged` tracking when a judge response truncates.
- **Tools (`tools/`):** `vlm_detect`, `run_vlm` / `run_vlm_batch`, `grounded_detect` (MM-Grounding-DINO), `detect` / `instance_segment` (RF-DETR), `segment_from_bbox` (SAM 3), `fast_segment` (EdgeTAM), `ocr` (LightOnOCR, Chandra, PaddleOCR-VL, etc.), plus pure-CPU `bbox_utils.py` (`convert_bbox`, `bbox_iou`, `containment`, `nms`, `build_label_resolver`).
- **Agent registry (`tools/registry.py`):** Introspects function signatures into JSON Schema and injects per-role `ToolConfig` credentials (`labeller`, `judge`, `default`) without exposing API keys in schemas.

## Key gaps we bridged in `freestyle/`

Our dense, small-object, look-alike cabinet problem violates several upstream document-era assumptions:

1. **Resolution & crop detail (`L1`–`L2`):** Core `vlm_client` defaults still resize to `1280px` JPEG q85, and `judge_labels` crop sheets cap crops at `256px` inside a 16-up grid. Every Freestyle caller passes `max_dimension=None, image_format="png"`, and `freestyle/vlm_identify.py` + `freestyle/identify/` inspect full-resolution original-pixel crops.
2. **Confidence & cross-class duplicates (`L4`–`L5`):** Single-shot `vlm_detect` emits no confidence score, and core `nms` only suppresses same-label overlaps. `freestyle/geometry.py` (`merge_tiled`) and `freestyle/bootstrap_labels.py` (`fuse_runs`) add WBF fusion across models/tiles, containment fragment suppression, and class-agnostic deduplication.
3. **Training resolution, negatives, and colour/flip augs (`L9`–`L10`):** `workflows/train_rfdetr.py` locks `rf-detr-large` to `704x704`, drops zero-box negative frames, and enables `HorizontalFlip` + `HueSaturationValue` by default—both of which break shelf-side and brand-colour identity. `freestyle/train/` exports masked shelf tiles directly and trains `rfdetr` / `ultralytics` with flip and hue jitter off by default.
4. **Local k8s & group splits instead of HF Hub (`L12`–`L13`):** Upstream `jobs/` scripts clone from GitHub and push datasets/models to HF Hub. Our data is private, so `freestyle/make_splits.py`, `freestyle/experiments/`, and `k8s/` run everything against local COCO files and our home cluster.
5. **Counting & group-bootstrap evaluation:** Upstream evaluates COCO mAP only (`max_det=100`). `freestyle/eval/` adds identification top-1/macro-F1, per-class count MAE, exact-inventory rate, failure galleries, and deterministic cluster bootstrapping over door-opening groups.
