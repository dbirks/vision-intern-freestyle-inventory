---
title: "First honest signal: identity is about a coin flip"
date: 2026-11-27T14:05:00-05:00
tags: [ground-truth, identity, findings]
---

Your first 46 quiz answers are all on **test** photos, the exam set, so they can't be used for
training. They give the first *independent* measurement of the system:

- On the 41 test crops you answered, the model's guess was **right 20 times**. You corrected it
  **21 times**: 18 to a different cartridge, 3 to a different Part number.
- So identity is at **roughly 50%** on this early sample. That matches the visual review:
  *finding* cartridges works well, *naming* them is the weak link, especially the bottom-shelf
  Coke family and Part 1 vs Part 2.

This is a small sample (no confidence interval yet), but it's the first number in this project
that didn't come from a model grading itself.

**Set up to improve it:** a background loop watches the quiz. As soon as you answer the
**training** crops (the next block, 60 crops the model is unsure about), it rebuilds the
identifier's reference gallery from your answers (bundle v2, CPU, seconds). Then I re-score v2
vs v1 on your test answers to see whether teaching it with about 60 crops moves that 50%.
