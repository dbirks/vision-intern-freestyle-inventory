---
title: "Everything is ready; only your quiz answers remain"
date: 2026-11-27T11:10:00-05:00
tags: [status, report, blocker]
---

Everything that doesn't need human ground truth is finished and pushed:

- **Report** (`docs/freestyle/REPORT.md`): full write-up with a clearly marked *provisional*
  architecture:
  - RF-DETR Medium on 1280 px tiles, plus duplicate suppression
  - 2-shelf counting
  - the hybrid identifier: VLM brand family, then embedding nearest-neighbour
  - the front-face detector as the alternative

  8 `GT-PENDING` markers show where real numbers go.
- **Failure gallery** (`docs/freestyle/FAILURES.md`) and **resource table**
  (`docs/freestyle/RESOURCES.md`): every number traces to a run file.
- **One-command finish** (`scripts/finalize_after_gt.sh`): quiz answers → ground truth → score
  all 15 runs with confidence intervals → pick the winner by counting error → export the bundle
  → demo. It refuses to run until at least 80% of the test questions are answered (now 2 of 186).

**The one thing left:** the phone quiz (`:8766`). The 12 test-photo *count* questions alone
(about 5 minutes) are enough to score counting accuracy and pick the architecture.
