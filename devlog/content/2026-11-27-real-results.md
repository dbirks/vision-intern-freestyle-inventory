---
title: "Real results on your answers: RF-DETR Medium wins, about 2 cartridges off per photo"
date: 2026-11-27T20:10:00-05:00
tags: [results, ground-truth, selection]
---

The final scoring ran on your quiz answers: 10 fully answered test photos for counting, plus your
6 validation counts for tuning each model's threshold. 95% ranges come from bootstrapping over
door openings.

| detector | avg. count error / photo | top | bottom | naming accuracy (VLM reader) |
|---|---|---|---|---|
| **RF-DETR Medium, 1280 px tiles (S2)**, selected | **2.1** (0–5.1) | 0.7 | 1.4 | **59%** (50–67%) |
| RF-DETR Small (C1) | 2.3 | 1.1 | 1.2 | 50% |
| front-face detector (G2, your idea) | 2.7 | 1.1 | 1.6 | (few crops) |
| RF-DETR all-class (D1) | 4.3 | 2.8 | 1.5 | (few crops) |
| copy-paste / smaller tiles (S3, S7, S1) | 4.4–4.6 | ~3 | ~1.5 | |
| OWLv2 zero-shot | 5.7 | 3.3 | 2.4 | |

**What this says:**
- **A small detector trained on 10 auto-labelled frames is the right call.** It's about 3× better
  than off-the-shelf open-vocabulary detectors at counting, on mostly blurry, partial door-swing photos.
- **Naming:** the VLM reader (59%) beats embedding matching trained on model labels (~52%).
  Brand is mostly right; Part 1 vs Part 2 is the remaining gap.
- **Caveat:** 10 test photos give wide ranges. S2, C1 and the face detector are close; S2 leads
  on the point estimate and on the top shelf.

**Shipped:** bundle `rfdetr-final-v2`, run with `freestyle.cli count`.

**Next:** score Flash Next as the reader, fill the final report, and optionally learn
Part 1/2 from your training crops.
