---
title: "Front faces, first look: the Part 1/2 numerals become readable"
date: 2026-11-27T02:45:00-05:00
tags: [faces, identity, findings]
---

![front-face labels on the best validation frame (Inkling 2 Mini, 2×2 tiles)](runs/faces_v1/viz-tune/local__downloads__frame.jpg-v1-tiles.png)

First result from your front-face idea, on one validation photo (prompt tuning only):

- **Bottom shelf:** instead of big overlapping rectangles around leaning cartridges, the VLM
  now boxes the **small printed end faces** at the bottom of each one. Asked about the face,
  it **reads the Part numerals**: labels like `21 1`, `22 2`, `25 1`, `27 2`. With
  whole-cartridge crops it managed 2 of 123. That could resolve Part 1 vs Part 2 directly,
  with no guessing from positions.
- **Top shelf:** faces are tight and complete, including the 2-high NNS stack.
- **One flaw to fix:** for three red Coke cartridges it *also* boxed the top end of the same
  cartridge, which would double count. Fix: within one leaning column, keep only the lowest
  face.

The worker is now labelling all 29 frames this way. Next: a duplicate rule for faces, then
train RF-DETR on faces in the next GPU window and compare against whole-cartridge detection
on your quiz answers.
