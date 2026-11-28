---
title: "Phone photos arrive, as a reference only"
date: 2026-11-28T12:00:00-05:00
tags: [data, phone, reference]
---

You sent two sharp phone photos of the open cabinet. Both shelves are fully in view and the labels
are readable.

**How we use them:** as a reference, so the agent knows which cartridge is which. We won't train on
them, won't add them to the naming gallery, and won't put them in the device test scores, so they
can't poison the model. They live outside the frozen split in a git-ignored folder.

**Planned check:** run both bundles (front-face G2 and Medium S2) on them as a separate
"phone, best case" result. If counting and naming jump on sharp photos, the camera is the
bottleneck, not the model.

**Blocked for now:** the Collective answers "backend at capacity" for Qwen 3.8 Flash Next, and
Inkling 2 Mini isn't available. A retry loop reruns the check once Flash Next has room.
