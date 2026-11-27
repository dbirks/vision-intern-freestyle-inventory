---
title: "RF-DETR trained on 10 frames beats zero-shot detectors 3×; quiz is live"
date: 2026-11-26T19:30:00-05:00
tags: [gpu, experiments, rf-detr, quiz, findings]
---

![RF-DETR Medium on a test frame (colours: blue = match, red = extra, orange dashed = missed)](runs/C2_rfdetr_medium_agnostic_siglip2-89fe7c2b/failures/overlays/island-cabinet-01_50828e4b-1e18-43a5-9c51-a53226bcb264_008_preview.png)

GPU window 1 on the RTX 5090: every number below is on the 12 **test** frames, **scored against
the VLM-made labels, not human ground truth**.

| run | AP50 | precision / recall | count error per photo | train time | peak GPU | speed |
|---|---|---|---|---|---|---|
| OWLv2 large (zero-shot) | 0.18 | 0.38 / 0.38 | 8.8 | — | 2.1 GB | 0.94 s/img |
| Grounding DINO base (zero-shot) | 0.23 | 0.68 / 0.32 | 9.5 | — | 1.7 GB | 0.36 s/img |
| RF-DETR Small, "cartridge" class | 0.69 | 0.64 / 0.76 | 5.8 | 4 min | 6.3 GB | 0.43 s/img |
| **RF-DETR Medium, "cartridge" class** | **0.70** | **0.73 / 0.70** | **4.4** | 5 min | 4.2 GB | 0.33 s/img |
| RF-DETR Small, all 18 classes + unknown | 0.65 | 0.76 / 0.69 | 5.2 | 3 min | 6.3 GB | — |
| SAM 3 | — | — | — | — | — | blocked: gated model, our HF token isn't approved |

**Take-away:** a small detector trained on *our* cartridges, from only about 10 training frames
of auto-labels, cut into 1024 px tiles at full resolution, is far better at "where is each box"
than general open-vocabulary detectors. Those merge neighbours and miss the stacked NNS. The
multi-class variant gets the right product on 57% of matched boxes (again vs model labels), so
identity is still the weak link.

**Running now:** a short sweep (tile 768 vs 1280, copy-paste augmentation, longer training,
RF-DETR Large) until 20:35, then Inkling goes back up.

**The phone quiz is live** at `:8766` (link on the home page card). Once your answers are in,
`score_all` re-scores every run against *your* counts and identities. It also compares four
ways to name the cartridges (the detector's own classes, embedding nearest-neighbour, the VLM,
and the Part 1-left-of-Part 2 pair rule) with 95% confidence intervals, plus sharp vs blurry
photos separately.

**Housekeeping:** this box's disk hit 100% while collecting checkpoints (Lightning saves about
0.5 GB per run). We now keep only the best weights locally; the full set stays on the cluster
volume.
