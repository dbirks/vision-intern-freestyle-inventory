---
title: "Web research: Part 1 sits left of Part 2, and the first GPU window opens"
date: 2026-11-26T18:20:00-05:00
tags: [research, gpu, k8s, plan]
---

**Before asking you to label anything, we searched the web** (Codex, about 70 sources). What we
learned:

- Public close-ups of individual cartridges are rare: an eBay AA cartridge, an eBay cherry
  cartridge, and one 2011 Wikimedia photo of a loaded cabinet. No public photo shows a Part 1
  next to a Part 2 face-on.
- **The two-part brands ship as "double cartridges":** Part 1 and Part 2 glued together, per
  Coca-Cola's own cartridge-management guide.
- **Every public placement map puts CART 1 to the left of (or above) CART 2**: the UK FS9000
  and FS3100 maps and the Canadian 9000/9100 and 8000 maps. Details are in
  `docs/freestyle/research/placement_maps.md`.

So even when the printed 1/2 is unreadable (true for 121 of 123 crops), we can infer it: find a
run of same-brand boxes on a shelf and alternate P1, P2 from the left. A confidently read numeral
anchors the alternation, and odd-length runs get flagged. It's coded as an optional, testable
rule (`freestyle/count/pairs.py`). It stays a hypothesis until your quiz answers say whether it
holds in this cabinet.

**The Kubernetes path got hardened** after a review found a dry run that could delete an
unrelated Job:
- Jobs are now ownership-checked before any delete.
- Waiting returns on failure and collects the logs.
- Image digests and the exact code tarball hash are recorded.
- Data is mounted read-only.
- The Hugging Face token is mounted only for gated models.

308 unit tests pass.

**GPU window #1 is open (18:15, time-boxed to 2.5 h).** With your OK, Inkling 2 Mini is paused,
so the RTX 5090 is ours. The queue:
1. RF-DETR Small trained to find "a cartridge" (class-agnostic) on the model labels
2. RF-DETR Small trained on all classes
3. Zero-shot OWLv2, Grounding DINO and SAM 3
4. RF-DETR Medium, if time allows

Each one predicts on val and test so it can be scored the moment your answers exist. Inkling
comes back at the end of the window either way.

**Phone quiz:** being built in this site's style. One zoomed cartridge at a time with my
guess, plus "I count N top / M bottom: right?" per photo. Test photos first.
