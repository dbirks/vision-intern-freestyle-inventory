---
title: "Testing Qwen Flash Next as the cartridge reader, scored on your answers"
date: 2026-11-27T18:40:00-05:00
tags: [identity, experiment, plan]
---

**Why:** naming the cartridge is the weak link. On the 41 test crops you've answered, the guesses
you saw (from Inkling 2 Mini) were right about half the time. You trust Qwen 3.8 Flash Next
more as a reader, and your answers make a fair answer key to check that.

**What's running:** both models read exactly the same 41 test crops, in several styles:
- one-shot "which of these 18 products?"
- two-stage: brand first, then a zoomed look for the Part number
- two-stage with a shelf hint
- a numbered contact sheet

Each is scored against your answers, with 95% confidence intervals, plus brand-only accuracy,
Part 1/2 accuracy and latency. No prompt is tuned on these crops, so the comparison stays honest.

**The picked pipeline (provisional):**
1. RF-DETR Medium finds cartridges.
2. The best reader names them: Flash Next if it wins.
3. Shelf-aware counting, with the Part 1-left-of-Part 2 rule as a check.

**What unlocks the final numbers:** about 18 more test answers plus the 6 validation count
questions. The finish script is being changed to require exactly that, not 80% of every block.
