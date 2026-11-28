---
title: "Correction: the 2.1 'off per photo' was mostly unusable photos"
date: 2026-11-27T21:30:00-05:00
tags: [results, correction, ground-truth]
---

I had to retract part of the earlier result.

**What went wrong:** in the quiz, "Not a usable cabinet view" meant *too blurry, partial or
sideways to count* (you confirmed). The scorer treated it as *"this photo has zero
cartridges"*. 8 of the 10 test photos got that answer. So the scoring rewarded predicting
nothing: it chose an extreme 0.90 confidence threshold and produced a flattering average
error. On the two photos that really show the full shelves (22 and 28 cartridges), that tuning
misses badly.

**Fix (running):** unusable photos are now **excluded** from count scoring, not counted as
zeros. Crop names you gave on them still count for identity accuracy. Thresholds get re-tuned
on usable validation photos only, then everything is re-scored.

**What that means for the evidence:** the honest test set is just **2 photos** right now,
possibly 4 once you answer the two skipped count questions. That's thin, so the next real
step is **better photos**. The door-swing camera produced mostly blurred, partial or sideways
frames. With a well-positioned fixed camera, which is the intended setup, and the new images
you're bringing tonight, we can build a proper test set.

The reader result stands, since it's crop-level and unaffected: Flash Next names cartridges
better than Inkling, with the brand right about 70% and Part 1/2 nearly unreadable.
