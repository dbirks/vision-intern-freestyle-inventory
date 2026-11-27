---
title: "Face detectors trained; window closed early; the quiz decides"
date: 2026-11-27T09:30:00-05:00
tags: [gpu, faces, results]
---

![G1 face detector on a test frame](runs/G1_rfdetr_medium_faces-cc431fcd/failures/overlays/island-cabinet-01_50828e4b-1e18-43a5-9c51-a53226bcb264_008_preview.png)

The front-face detectors, trained on only 9 training frames of cleaned face labels:

| run | target | AP50* | precision / recall* | count error per photo* | peak GPU |
|---|---|---|---|---|---|
| G1 RF-DETR Medium, 1280 tiles | front faces | 0.66 | 0.57 / 0.72 | 5.8 | 4.4 GB |
| G2 RF-DETR Small, 1024 tiles | front faces | 0.63 | 0.70 / 0.62 | 3.3 | 7.4 GB |
| S2 RF-DETR Medium, 1280 tiles | whole cartridges | 0.71 | 0.77 / 0.75 | 3.6 | 4.2 GB |

\* Each row is scored against its *own* kind of model labels (faces vs whole boxes), so the rows
aren't directly comparable, and none of it is human truth.

That's the honest state: faces and whole-cartridge boxes are now in the same range, and more GPU
variants can't separate them, because the "answer key" is itself model-made. So I closed GPU
window 2 after 12 minutes instead of using the extra time you offered, and Inkling is back.

**What decides it:** your quiz counts. Both detectors predicted the same 12 test photos, so a
single pass of "N top / M bottom: right?" answers scores faces vs whole cartridges, all the
identity methods and the pair rule in one go, with confidence intervals.
