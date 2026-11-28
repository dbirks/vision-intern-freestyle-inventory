---
title: "Corrected results: your front-face detector takes the lead (on 2 photos)"
date: 2026-11-27T22:05:00-05:00
tags: [results, faces, ground-truth]
---

With unusable photos excluded (not scored as zeros), only **2 test photos** show full shelves:
22 and 28 cartridges. On those:

| detector | count error / photo | naming (Flash Next reader) |
|---|---|---|
| **front-face RF-DETR Small (G2), your idea** | **0.5** | **75%** |
| RF-DETR Medium, 1280 tiles (S2) | 2.0 | 58% |
| RF-DETR Small (C1) / copy-paste (S3) | 2.5 | ~58–61% |
| front-face RF-DETR Medium (G1) | 4.0 | 74% |
| OWLv2 / Grounding DINO zero-shot | 4.5 / 10 | |

**Plainly:** two photos can't prove a winner; one lucky photo moves these numbers a lot. But
the direction matches the reasoning. Front faces don't balloon on the leaning bottom row, and the
printed end is where the logo and Part number are, so naming improves too.

**Now:**
- The front-face detector is packaged as bundle `rfdetr-faces-final` with the Flash Next reader,
  and the report is being rewritten around the corrected numbers, including a short correction history.
- Two pending test-photo counts in the quiz would double the evidence.
- The new photos from a fixed, well-positioned camera tonight are the real fix for thin evidence.

(The reboot cleared the SSH key again, so commits wait for `ssh-add`.)
