---
title: "Front faces on all 29 frames: less overlap, more identity, some noise to clean"
date: 2026-11-27T03:25:00-05:00
tags: [faces, findings]
---

The front-face labeller (Inkling 2 Mini on the full frame plus 2×2 tiles, judged by Flash
Next in crop mode, all through Vision Intern's label → judge path) ran over all 29 cabinet frames:
715 face proposals in about 28 minutes, from 197 model calls with 0 errors.

What the numbers say (vs the earlier whole-cartridge labels, all model-made, not GT):

| | whole cartridge | front face |
|---|---|---|
| boxes | 563 | 715 |
| mean overlap with nearest neighbour (IoU) | 0.20 | **0.15** (0.10 where both runs agree) |
| boxes with a usable identity | far fewer | **326** |

So your intuition holds: faces overlap less and carry more identity, because the printed end of
the box is what has the logo and Part number.

The noise, honestly:
- the tiled run sometimes boxes the *long side* of a leaning cartridge as if it were a face;
- for a few leaning cartridges it boxes both ends;
- on almost-empty partial frames it invents faces (the judge rejected most of those).

**Now:** a pi worker is cleaning that up. It keeps faces both runs agree on and keeps only the
lowest face per leaning column. It also drops side strips and parses the printed 1/2 into Part
labels. Then it sets up the face-detector training (RF-DETR Medium and Small). That needs a
short GPU window, which I'll ask you about first.
