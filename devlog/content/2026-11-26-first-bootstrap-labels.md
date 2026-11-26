---
title: "First auto-labels: the top shelf is easy, the bottom shelf is the real problem"
date: 2026-11-26T16:25:00-05:00
tags: [bootstrap, vision-intern, findings]
---

![bootstrap overlay on the best frame](../runs/bootstrap_v1/viz/local__downloads__frame.jpg.png)

The label bootstrap now runs Vision Intern end to end:
1. `label_dataset` (dense mode, Inkling 2 Mini, full frame + 2×2 tiles)
2. weighted box fusion across those runs
3. `judge_labels` in crop mode (Flash Next)
4. a VLM crop identifier

It costs about 2 minutes per frame. All 29 cabinet frames are being labelled now.

**What the first frame shows (I checked it by eye):**
- **Top shelf: mostly right.** The NNS stack (7), AA, Fanta BB ×2, CC ×2, Sprite, the orange and
  vanilla flavours. One wrong identity: the Hi-C cartridge (red/yellow Hi-C logo) came out as
  *lemon flavour*. One vanilla came out `unknown`.
- **Bottom shelf: every identity is `unknown`,** and the boxes are loose on the leaning
  cartridges. The bottom shelf holds the two-part brands (Coke, Diet Coke, Dr Pepper, Coke Zero:
  Part 1 and Part 2 each). The part is only a small printed **1** or **2** near the box end. You
  can see it at full resolution, but a one-shot "which of 18 products is this?" question gives up.

**Change made:** a new *two-stage* identifier. First the brand family from colour and logo,
using an explicit cue table. Then, only for two-part brands, a 3× zoom on the end region of the
box asking just "is the numeral 1 or 2?". When the numeral isn't legible it keeps the family and
reports the part as unknown, so we don't fake confidence.

**Why Flash Next is now only the judge:** one Flash Next labelling call took 530 s with
repeated 429s, against about 7 s for Inkling. Two different models labelling would be better
science (Vision Intern's own rule wants the labeller and judges from different families), so the
judge stays on Flash Next.

**Next:** finish labelling, then the one human step, verifying the 12 test frames in the
review app, then the GPU window for the detector experiments.
