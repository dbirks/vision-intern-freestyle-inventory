---
title: "GPU cluster ready (but full), frozen data split, and reviews catching real bugs"
date: 2026-11-26T15:55:00-05:00
tags: [k8s, data, reviews, status]
---

**Kubernetes jobs work.** `k8s/submit.sh <name> 5090|6000 -- <command>` renders a Job from
`k8s/job-template.yaml`, ships the repo as a tarball, mounts our private data from a per-node
PVC, and `k8s/collect.sh` pulls `/out` back into `runs/`. Smoke jobs ran on both cards:
**RTX 5090 ≈ 219 TFLOPS bf16**, **RTX PRO 6000 ≈ 102 TFLOPS fp16** (it's power-capped at 300 W).
The image is `pytorch/pytorch:2.11.0-cuda13.0` (Blackwell needs CUDA ≥ 12.8): a cold pull takes
about 6 minutes, a warm one about 1.

**But the cards are full.** Flash Next holds about 97.8 of 97.9 GB on the 6000, and Inkling
holds all of the 5090. The smokes squeezed into ~0.4 GB. Per your call, the plan is to pause
Inkling during one batch window of real GPU jobs. Inkling is also our fastest labeller (about
3 s per frame), so the window opens after bootstrap labelling finishes, and I'll ask right
before I scale it down.

**Data split is frozen** (`data/splits.json`), grouped by door-opening event so near-identical
frames never land on both sides:

| split | images | groups | notes |
|---|---|---|---|
| test | 12 | 11 | 5 sharp full views, 7 partial or severely blurred (deliberately realistic: the camera swings with the door) |
| val | 6 | 6 | includes the best full view `frame.jpg`, used for thresholds |
| train | 22 | 11 | 10 server frames + the 12-frame blur burst (one group) |
| exclude | 29 | | ceiling, people, laptop, occluded lens: kept as negative tests |

**Reviews pay off.** Four independent review passes so far: Codex and Gemini Hydrogen checking
each other's code. They found about 30 real defects, each fixed with a regression test. Examples:
- test-set leakage in the identifier gallery
- a VLM coordinate "auto" guess that silently shrank boxes
- unknown-identity thresholds that never rejected anything
- an OWLv2 exemplar path that would have crashed in transformers 5.24
- copy-paste augmentation warping pixels without moving the boxes

The latest review (Kubernetes scripts) found a dry run that could delete someone else's Job;
fixes are queued. Current state: 245 unit tests, plus an end-to-end synthetic integration test.

**Running now:** label bootstrap (Vision Intern `label_dataset` → `judge_labels` → crop
identifier) on all cabinet frames, finishing the data audit, and a CPU-only end-to-end
experiment through the real cluster to shake out failures before we spend GPU time.
