---
title: "Sweep results: bigger tiles help; Inkling is back (later than planned)"
date: 2026-11-26T22:55:00-05:00
tags: [gpu, experiments, sweep, ops]
---

**Sweep on the RTX 5090** (each run changes one thing vs the RF-DETR Medium baseline, all trained
on model labels and scored on the 12 test photos **against model labels, not human truth**):

| run | change | AP50 | precision / recall | count error per photo |
|---|---|---|---|---|
| C2 | baseline: Medium, 1024 px tiles | 0.70 | 0.73 / 0.70 | 4.4 |
| S1 | 768 px tiles | 0.61 | 0.72 / 0.56 | 6.0 |
| **S2** | **1280 px tiles** | **0.71** | **0.77 / 0.75** | **3.6** |
| S3 | + copy-paste / slot-shuffle synthetic training data | 0.72 | 0.72 / 0.77 | 4.4 |
| S5 | RF-DETR Large | 0.69 | 0.80 / 0.68 | 4.2 |
| S7 | 1280 tiles + copy-paste | 0.68 | 0.79 / 0.72 | 4.5 |
| S6 | OWLv2 + containment filter (zero-shot) | 0.12 | 0.56 / 0.18 | 11.6 |

Reading it honestly: with 12 test photos, differences of a few points are within noise. The
clear signals:
- **smaller tiles hurt** (S1). The model resizes each tile to its input size, so 768 px tiles
  don't buy more detail, and they cut more cartridges in half.
- **zero-shot detectors aren't competitive** here.
- copy-paste augmentation and the Large model don't clearly help yet.

Your quiz answers decide the rest.

**Bug caught in review:** the sweep worker patched a class-index crash with a `% len(classes)`
trick. That's fine for one-class detectors, but it could silently shift every identity by one
in the 18-class model. Codex is checking RF-DETR's source for the real class-id convention,
replacing the trick with an explicit saved mapping, and auditing whether the earlier
all-class run was affected.

**Ops note, my mistake:** the reminder that should have restored Inkling at 20:43 didn't fire
while I was waiting on a question to you, so Inkling stayed down until about 22:45, about 2 h
longer than the time box, even though the GPU was idle from 20:05. It's back and verified
(the Collective answers in 0.8 s). Next time the restore runs as a detached timer on the host,
independent of this session.

**Now:** packaging the best candidate (RF-DETR Medium, 1280 tiles, plus an embedding identifier
trained on auto-labels) as a model bundle and running the full `count` command on real photos
on CPU, to prove the end-to-end system while we wait for quiz answers.
