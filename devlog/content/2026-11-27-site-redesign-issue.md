---
title: "From experiment to product: a redesign issue for the inventory site"
date: 2026-11-27T22:55:00-05:00
tags: [product, design, issue]
---

The vision project's lessons are now written up as a redesign proposal for the inventory site:
**freestyle-inventory #30** (private repo).

**Core idea:** the page answers *"what does the cabinet look like right now?"* It shows the
best current photo, large, with detected counts beside it and a **Fix / correct me** button.
The AI is an annotation on the photo, not a dashboard competing with it.

Three pages:
1. **Now:** photo first, counts and "counted but not named" to the side, why this frame was picked.
2. **Retraining:** the model's shaky items, one zoomed crop at a time, modelled on the phone quiz
   that produced our ground truth.
3. **Trends:** past runs vs the current one, plus reorder rates. It says up front "still gathering
   data to make recommendations" until there's real history.

The design language is this site's: the study guide's pastel cards, serif headings and mono
numbers, with Coke red for the main action.

The lessons it carries:
- photo quality is the ceiling (fixed camera, sharpest frame)
- naming is the weak link
- "not usable" is never zero
- the correction loop is cheap and valuable
