---
title: "What the LLM Collective VLMs can actually do"
date: 2026-11-26T15:25:00-05:00
tags: [vlm, collective, probe, findings]
---

A pi worker probed the Collective's vision models on three real cabinet frames, with overlays
drawn and checked by eye.

**Models that take images:** `david-birks/inkling-2-mini` (about 3 s per call, very reliable),
`qwen3-8-flash-next` (a thinking model: 13–107 s per call and frequently answers `429 backend at
capacity`), and `david-birks/qwen3-8-flash-next` (12–46 s). `qwen3-8-27b` is gone (404).

**Coordinates (the trap):** Flash Next *always* returns boxes on a 0–1000 normalised grid, even
when asked for pixels. Inkling follows whichever convention the prompt asks for. So everything
now prompts for and parses 0–1000, and the code refuses to guess units from magnitudes (that
guess was a real bug a review caught).

**Box quality on the best frame:** about 26 boxes for roughly 27 cartridges. The top shelf is
tight and correct (NNS stack, AA, BB, CC, Hi-C, Sprite, flavours). On the bottom shelf the
cartridges lean and overlap, so the boxes spread across neighbours and a few get missed. That
weakness tells us where a trained detector has to earn its keep.

**Identity from crops:** both models named the AA crop and a red Coca-Cola crop correctly. But
that is 2 crops, judged against the worker's own reading, so it's **not evidence yet**: real
accuracy waits for human-verified ground truth.

**Thinking-model gotcha:** with `max_tokens` = 64, Flash Next returns an empty answer because
its reasoning used up the budget. It needs at least 256 tokens, and about 8k for box lists.

**Next:** the label-bootstrap run (Vision Intern `label_dataset` → `judge_labels` → crop
identifier) on all ~40 cabinet frames, then you verify the test set in the review app.
