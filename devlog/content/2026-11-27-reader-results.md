---
title: "Flash Next vs Inkling as the reader: brand ~70%, the Part number is the wall"
date: 2026-11-27T19:45:00-05:00
tags: [identity, results, ground-truth]
---

First head-to-head scored on **your** answers: 39 test crops with a definite human label.

| reader (best mode) | exact cartridge | brand only | Part 1/2 | says unknown | s/crop |
|---|---|---|---|---|---|
| **Qwen 3.8 Flash Next**, brand → zoomed part, with shelf hint | **56%** (41–71%) | **69%** | 9% | 18% | 9.1 |
| Inkling 2 Mini, same mode | 49% (34–64%) | 67% | 0% | 44% | 3.5 |
| one-shot "which of 18?" (either model) | ~20% | | | ~75% | |

**What it means:**
- Flash Next commits more and is right a bit more often. The ranges overlap at n = 39, so
  it's a lean, not a proof. It becomes the brand reader.
- **The prompting style matters more than the model:** brand first, then a zoomed look, beats
  one-shot by about 2.5×.
- **Brands are about 70%, Part 1 vs Part 2 is about 0–9%.** At this camera resolution the
  printed numeral basically can't be read. That's exactly why we built the pair rule (Part 1 sits left
  of Part 2 in a glued double cartridge) and the gallery that learns from your training answers.
- Specific confusion: CC read as NNS (3×).

**Next:** the 6 validation count questions (next in the quiz), then the final scoring of
detector × reader × pair rule with confidence intervals, and the final pick.
