# Ground-truth protocol (human verification)

Ground truth exists only in `data/gt/` and only for images a human marked **verified** in the
review app (`python -m freestyle.review.server`). Model proposals (`data/labels/`) pre-fill the
app; they are never GT on their own.

## What gets a box

* **One box per physical cartridge** whose face is at least partly in frame. Stacked cartridges
  (the NNS sweetener sits two-high) get one box each.
* The box covers the **visible extent of that cartridge** (its outward face plus any visible side
  that belongs to it), tight to the cardboard, in original pixels. Under the fisheye lens the
  bottom-shelf boxes lean, so neighbouring axis-aligned boxes overlap. That is expected; don't
  shrink boxes to avoid overlap.
* Cut off by the frame edge or occluded by a hand, cable or door edge → still boxed, with
  `truncated` or `occluded` set.
* Not boxed: shelf-lip signage, empty slots, cables, people, reflections, cartridges seen only
  through a gap with nothing identifiable (if you can tell it is a cartridge, box it).

## Identity

* Pick the catalog class (18 active ids) only if the pixels support it: brand colour plus
  logo, text (`NNS`, `AA`, `BB`, `CC`), or Part 1/Part 2 marking. When Part 1 and Part 2 can't
  be told apart in this frame, use `unknown` and set `identity_visible=false`.
* `unknown` is a valid, counted answer. Never guess to make the shelf "look right": the
  cabinet arrangement is not assumed to be fixed.

## Image-level

* `scene`: `cabinet_full`, `cabinet_partial`, `no_cabinet`, `occluded`. A `no_cabinet` image is
  verified with zero boxes (a negative test).
* Mark **verified** only after checking the per-class count bar against the image.

## Effort budget

Verify `test` first (frozen once verified), then `val`. Train labels may stay model-generated
(Vision Intern label → judge) unless an experiment shows human-corrected train labels help.
Record time spent per image in the review app note.
