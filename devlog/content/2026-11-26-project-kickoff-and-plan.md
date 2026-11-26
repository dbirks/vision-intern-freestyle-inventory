---
title: Project kickoff & plan
date: 2026-11-26T10:00:00-05:00
tags: [kickoff, architecture, data-audit, evaluation, experiments]
---

We are building a computer-vision pipeline for **Coca-Cola Freestyle cartridge recognition**: given a photograph from the in-door fisheye camera of a Freestyle cartridge cabinet, locate every visible cartridge face, identify its SKU among 18 active catalog classes (or `unknown` when unreadable), and produce an accurate per-SKU inventory count.

Rather than assuming a single architecture up front, the project sets up a strict evaluation contract, a human-verified ground-truth loop, and six competing experiment families (`A` through `F`) tested on identical group-safe splits.

## What the data audit told us

Running `freestyle/audit.py` and `freestyle/make_splits.py` over the initial corpus revealed several hard physical constraints:

- **Small, single-cabinet corpus:** 69 total photographs (53 at `1600x1200`, 15 at `2048x1536`, 1 at `640x480`), grouped into **56 independent units** (by door-opening event UUID, keeping the 12-frame `local/blur` motion burst together).
- **Usable vs. negative frames:** 29 frames show no cabinet or an occluded lens (`exclude` / negatives), leaving **40 usable cabinet frames** (`29` full cabinet, `11` partial).
- **Sharpness & motion blur:** Only 21 of the 40 usable frames reach `sharp_full >= 20` (Laplacian variance). Because the camera sits on a swinging door under dim interior lighting, fine spine text (`Part 1` vs `Part 2`, `AA`/`BB`/`CC`) is frequently soft or unreadable at `1600x1200`.
- **Single physical arrangement:** Near-duplicate pHash analysis found 9 clusters (32 of 69 frames). Cross-opening frames often show the same cartridge layout, which creates a slot-memorisation hazard: a detector can appear accurate by memorising slot positions rather than reading labels.

To keep evaluation honest, `data/splits.json` freezes splits strictly by **group** (`11` train groups / `22` images, `6` val groups / `6` images, `11` test groups / `12` images, `28` excluded negative groups), with `test` stratified across sharp, partial, blurry, rotated, person-visible, and both high resolutions.

## Ground truth before optimization

Model proposals never count as ground truth. We built a local review web app (`freestyle/review/server.py`) that displays full-resolution original images, pre-fills boxes from our multi-VLM bootstrap (`freestyle/bootstrap_labels.py`), and writes human-verified COCO JSON atomically to `data/gt/`. Key annotation rules from `docs/freestyle/GT_PROTOCOL.md`:

- Every visible cartridge face gets one box in original pixels (two-high NNS sweetener stacks get two boxes).
- When `Part 1` vs `Part 2` or `AA`/`CC` cannot be determined from the pixels, label it `unknown` rather than guessing from shelf order.
- Negative (`no_cabinet`) frames are verified with zero boxes to catch hallucinated detections.

## Evaluation contract & experiment families

Every method writes standard COCO predictions (`runs/<exp>/predictions.json`) evaluated by `freestyle/eval/evaluate.py` across three axes with **1,000-sample group-bootstrap 95% confidence intervals**:

1. **Class-agnostic localization:** AP50, AP50:95, and P/R/F1 at the validation-selected operating threshold.
2. **Identification:** Top-1 accuracy and macro-F1 over matched non-unknown GT boxes (IoU >= 0.5).
3. **Counting:** Per-image total count MAE, per-class count MAE (18 SKUs + `unknown`), and exact-inventory rate.

| Family | Approach | Core Hypothesis |
|---|---|---|
| **A** | Direct VLM grounding (`A1`–`A3`) | Zero-training baseline using tiled/full-frame VLMs on the Collective. |
| **B** | Zero-shot proposals + crop kNN (`B1`–`B3`) | Promptable detectors (OWLv2, SAM 3, Grounding DINO) + DINOv2/SigLIP 2 gallery. |
| **C** | Class-agnostic detector + crop kNN (`C1`–`C2`) | Trained tiled RF-DETR (`cartridge` only) + frozen crop embeddings. |
| **D** | End-to-end 18-class detector (`D1`) | Direct supervised multi-class RF-DETR / YOLO comparison. |
| **E** | Shelf-structure-aware counting (`E1`) | Adds row clustering, gap/overlap flags, and neighbour voting atop C. |
| **F** | Trained detector + VLM crop ID (`F1`) | Combines supervised box recall with VLM fine-print reading on crops. |

## Cluster compute reality

We run local VLM endpoints on the LLM Collective (`qwen3-8-flash-next` on an RTX PRO 6000 96 GB via Pennyroyal/SGLang; `inkling-2-mini` on an RTX 5090 32 GB eGPU via Basalt) and submit GPU training/eval jobs via `k8s/submit.sh`. Because both Blackwell cards are nearly full in HAMi (`~900–1,300 MiB` physically free while serving), GPU experiments schedule explicitly with `--mode share` / `--mem` coordination or dedicated maintenance windows, while CPU and Collective VLM workloads run directly from the workstation.
