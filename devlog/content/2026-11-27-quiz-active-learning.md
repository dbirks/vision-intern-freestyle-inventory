---
title: "Quiz is live: fewer, smarter questions, and why test answers never train the model"
date: 2026-11-27T12:00:00-05:00
tags: [quiz, ground-truth, active-learning]
---

You've started the quiz: answers are coming in, and your CC correction on photo `e194690f/002` was
recorded. Changes in progress:

**Answer flow:**
- ✓ Correct flashes a checkmark for half a second, then advances.
- Picking a different cartridge now only *selects* it, so you can tap around. A **Next →**
  button saves it.

**Fewer questions (about 333 → 140), each one doing useful work:**
- **Test photos = the exam.** All 12 count questions plus a fixed random sample of about 30
  crops. Random keeps the identity score honest. **Test answers are never used for training**;
  otherwise the score would just measure memorisation.
- **Training photos = where your taps teach the model.** It asks only about the crops the
  identifier is unsure of, mostly bottom-shelf Coke, Diet Coke, Dr Pepper and Zero, and their
  Part 1/Part 2, shown as front-face crops where the printed number is most visible. Those
  answers go straight into the identifier's reference gallery: a CPU-only refresh, no GPU window.
- **Validation photos:** counts plus the most uncertain crops, used to tune thresholds.

**Model choice (provisional):**
- RF-DETR Medium on 1280 px tiles locates the cartridges.
- A hybrid identifies them: the VLM picks the brand, embeddings refine it.
- Your front-face detector is the runner-up.

Your test-photo counts make the final call.
