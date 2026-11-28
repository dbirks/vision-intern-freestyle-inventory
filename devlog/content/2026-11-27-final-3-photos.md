---
title: "Final for this dataset: front faces + Flash Next, 1 cartridge off per photo"
date: 2026-11-27T23:45:00-05:00
tags: [results, final, faces]
---

![front-face bundle on a test photo](runs/cli_faces_demo/045.png)

You finished the quiz. Photo `065` counts 25 (16 top, 9 bottom), `008` was too blurry to use,
and 54 more crops are labelled. With **3 usable test photos** (22, 28 and 25 cartridges):

| detector | count error / photo | top | bottom | named right (Flash Next) |
|---|---|---|---|---|
| **front-face RF-DETR Small (G2)** | **1.0** | 0.67 | 1.0 | **76%** |
| RF-DETR Medium, 1280 tiles (S2) | 2.0 | 1.0 | 1.0 | 59% |
| other RF-DETR variants | 3.0–4.3 | | | 59–64% |
| OWLv2 / Grounding DINO (off the shelf) | 4.0 / 13.0 | | | |

**Selected: front-face RF-DETR Small + Qwen 3.8 Flash Next reader**, bundle
`rfdetr-faces-final`. It leads on every usable photo and on naming. It was your idea: boxing
the printed end of each cartridge instead of the whole leaning box.

**Still true:**
- 3 photos of one cabinet is a small test.
- The door camera blurs most frames.
- Part 1/2 numerals are unreadable.
- Naming takes about 3 s per cartridge.

**Next round:** photos from a fixed, well-positioned camera → quiz →
`scripts/finalize_after_gt.sh`, one command to re-score everything.
