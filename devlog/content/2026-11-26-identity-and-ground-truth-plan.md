---
title: "Two-stage identity, unreadable part numerals, and a phone quiz for ground truth"
date: 2026-11-26T17:40:00-05:00
tags: [identity, ground-truth, findings, plan]
---

![identity grid for the best frame, crops sorted by predicted class](runs/bootstrap_v2/viz/local__downloads__frame.jpg_grid.jpg)

**Two-stage identifier.** We re-ran only the identity step on the 563 bootstrap boxes: brand
family first (colour and logo cues), then a zoomed look for the Part 1/Part 2 numeral.
The `unknown` rate dropped from **78% to 46%** (686 Inkling calls, about 38 minutes).

**Finding: the part numerals are basically unreadable at this camera's resolution.** Only
2 of 123 bottom-shelf crops gave a legible 1 or 2. So "Coke Part 1 vs Part 2" probably can't
be decided by reading text. It has to come from other visual differences (box size or
graphics), learned from labelled examples, or be reported as a limit of the camera.

**Fisheye trap the worker found.** Without knowing which shelf a box is on, the model called
most red, silver and maroon bottom-shelf boxes "Hi-C" (Hi-C is a top-shelf red box). Splitting
shelves by the largest vertical gap between box centres fixed most of that.

**Before asking you anything:** Codex is searching the web for public reference photos:
- Freestyle cartridge product shots, loading guides and service manuals
- marketplace listings and patents
- what Part 1 vs Part 2 physically look like

If those exist, they become a reference gallery for the identifier and for embedding matching,
and they shrink what you need to check.

**Then a short phone quiz** (this site's style): one zoomed cartridge at a time with my guess.
Tap ✓, or pick the right one. Plus one quick "I count N top / M bottom: right?" per photo.
Test photos first. That gives honest identity and count accuracy; box outlines stay
model-drawn (noted as a limitation).

**GPU window next:** a CPU-only run of a real experiment through the cluster passed
(code upload → data mount → model download → predictions → metrics → failure gallery). Once
the Kubernetes fixes land, I'll ask you before pausing Inkling to train detectors.
