---
title: "GPU window: zero-shot detectors struggle on a dense shelf"
date: 2026-11-26T18:58:00-05:00
tags: [gpu, experiments, zero-shot, findings]
---

![Grounding DINO on a test frame: orange dashed = missed, magenta = wrong id, red = false box](runs/B3_gdino_dinov2-be668339/failures/overlays/island-cabinet-01_50828e4b-1e18-43a5-9c51-a53226bcb264_008_preview.png)

First real GPU jobs on the RTX 5090 (Kubernetes Jobs, about 2–6 minutes each including model
download):

| run | what | peak GPU | AP50* | precision* | recall* |
|---|---|---|---|---|---|
| B1 | OWLv2 large, prompt "cartridge box", 1024 px tiles | 2.1 GB | 0.18 | 0.38 | 0.38 |
| B3 | Grounding DINO base, same protocol | small | 0.23 | 0.68 | 0.32 |
| B2 | SAM 3 (concept prompts) | — | 403: our Hugging Face token isn't approved for the gated model | | |
| C1 | RF-DETR Small trained on the model labels (class-agnostic) | — | finished, collecting | | |
| D1 | RF-DETR Small, all classes | — | running | | |

\* **Scored against the VLM-generated labels, not human ground truth.** Treat these as a
sanity check, not accuracy.

**What the overlays show:** the open-vocabulary detectors find the obvious top-shelf faces, but:
- they **merge the stacked NNS cartridges** into one tall box;
- they draw one box over several adjacent cartridges;
- they mostly give up on the dark, leaning bottom row.

That's the classic dense-shelf failure: these models were trained on "an object", not on
"30 near-identical boxes touching each other". So a detector trained on *our* cartridges (even
from model labels) is the thing to beat.

**Bug of the hour:** the first RF-DETR jobs died in 3 minutes because `rfdetr` needs its
`[train]` extras for training (it pulls in PyTorch Lightning). The worker added them and
resubmitted.

**Next:** RF-DETR results, then Inkling goes back up (time box ends about 20:45).
