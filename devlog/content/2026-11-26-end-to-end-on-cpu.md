---
title: "End to end on a real photo: counting is close, identity isn't yet"
date: 2026-11-26T23:15:00-05:00
tags: [pipeline, cli, findings, identity]
---

![bundle v0 on a test photo: boxes + predicted names](runs/bundle_demo/frames/008.png)

The whole system now runs from one command on this box's CPU, with no GPU needed:

```bash
uv run --extra train python -m freestyle.cli count photo.jpg --bundle rfdetr-m-1280-knn-v0 --device cpu --out out/
```

Bundle v0 has three parts:
- the best sweep detector (RF-DETR Medium on 1280 px tiles)
- an embedding nearest-neighbour identifier, whose gallery is training crops named by the VLM
- shelf-aware counting

**Cost:** about 16 s per photo on CPU (6 s locate, 9 s identify) and 1.7 GB RAM.

**What I see on the photos:**
- **Counting is in the right range.** 26 cartridges on the best frame, where about 27 are there.
- **Identity is the weak link.** The embedding identifier labels most things "NNS" (22 of about
  30 on one photo), because NNS dominates the auto-labelled gallery. CC cartridges become NNS
  and AA becomes vanilla, even though the VLM two-stage identifier got those right.
- **Duplicates:** overlapping tiles leave stacked duplicate boxes on the leaning bottom row.
- **Shelf logic** sometimes counts the 2-high NNS stack as a third shelf.

**A bug caught before it mattered:** a class-id hack (`% number_of_classes`) would have
silently shifted identities in the all-class detector. After reading RF-DETR 1.12.1's source
(it predicts zero-based head indices over sorted training categories), we now save an explicit
mapping at train time and recovered it for all 14 existing checkpoints. Unknown ids fail loudly.
An audit showed the earlier multi-class run was *not* shifted.

**Now:** tuning on validation photos only:
1. suppress duplicates
2. cap shelves at two
3. compare four identity strategies: embedding kNN, class-balanced prototypes, VLM two-stage,
   and a hybrid (VLM family plus embedding part), with and without the Part 1/Part 2 pair rule

The final choice waits on your quiz answers.
