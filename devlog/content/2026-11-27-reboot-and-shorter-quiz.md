---
title: "Back after a reboot; the quiz is now 180 questions, not 665"
date: 2026-11-27T13:20:00-05:00
tags: [quiz, ops, active-learning]
---

**Reboot:** this box restarted around midday. The quiz, this site and the quiz-builder worker
were all tmux processes, so they died. All three are back, and your answers were never at risk
(they're appended to a file on disk with every tap).

**The quiz is now about active learning:**

| block | purpose | questions |
|---|---|---|
| test counts + a fixed random sample of test crops | honest grading (never trains anything) | 74 |
| training crops the identifier is unsure about | teaches the model (bottom-shelf Coke, Diet Coke, Dr Pepper and Zero end faces: Part 1 or 2?) | 60 |
| validation counts + uncertain crops | tunes thresholds | 46 |

That's 180 in total, with 46 already answered. A leakage test guarantees test and validation photos
can never enter the identifier's reference gallery.

Your training-crop answers feed `refresh_gallery`. It rebuilds the identifier's reference set on
CPU in seconds: bundle `rfdetr-m-1280-v2`, identical to v1 until your training answers arrive.

**Small snag:** commits are paused until the SSH agent has your key again after the reboot.
