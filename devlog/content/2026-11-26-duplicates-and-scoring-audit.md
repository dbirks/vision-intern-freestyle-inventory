---
title: "Fewer duplicate boxes, and the scoring code audited before it scores anything"
date: 2026-11-26T23:55:00-05:00
tags: [pipeline, review, quiz, status]
---

![validation frame: before (left) vs after (right) duplicate suppression](runs/v1iter/ov_supcell_8.png)

**Duplicates:** overlapping 1280 px tiles were leaving stacked boxes on the leaning bottom row.
A new final pass removes a box when another box mostly contains it or sits in the same
shelf "cell". It's tuned on validation photos only. The top-shelf faces and the stacked NNS are
untouched.

**Auditing the scorer before it runs:** `score_all` will produce the project's final accuracy
numbers from your quiz answers, so Gemini Hydrogen reviewed it adversarially, with synthetic
quiz data. It found 9 real problems, all now fixed with regression tests. The main ones:
- the Part 1/Part 2 pair rule was silently a no-op for 7 of 8 identity methods
- photos you mark "not a cabinet" were being dropped instead of counted as zero-cartridge
  negatives
- half-answered photos could leak into per-class metrics
- marking a crop "partial" could promote the model's unconfirmed guess to "human verified"

That's the fifth review-and-fix round; together they've caught 40+ bugs that would have
quietly skewed results.

**Now:** finishing the validation-only tuning (cap shelves at 2, pick the identity strategy
among embedding kNN, class-balanced prototypes, VLM two-stage and a hybrid), then exporting
bundle v1.

**Waiting on:** the phone quiz (`:8766`, 333 quick questions). It's the only thing between us
and real accuracy numbers.
