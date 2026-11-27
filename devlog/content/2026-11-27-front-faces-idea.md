---
title: "Your idea: detect the front face, not the whole leaning cartridge"
date: 2026-11-27T01:50:00-05:00
tags: [idea, experiment, quiz]
---

**Problem you spotted in the quiz:** on the bottom shelf the cartridges lean. An axis-aligned
box around a whole leaning cartridge becomes a big rectangle that covers parts of its
neighbours, which makes it hard to say "yes, that's one cartridge".

**Your fix, now experiment family G:** detect only each cartridge's **front face**, the narrow
end facing the camera where the logo and the Part 1/Part 2 number are printed. On the top shelf
that's the face we already see. On the bottom shelf it's the small end face near the bottom of
each box. Faces are small, upright and don't overlap, so they should be easier to count and to
identify.

Plan:
1. A pi worker is auto-labelling front faces with the Collective VLM through Vision Intern's
   label → judge path (running now), and measuring how much less the boxes overlap.
2. Train RF-DETR on faces in the next GPU window (I'll ask before pausing Inkling).
3. Score faces vs whole-cartridge boxes on your quiz answers.

A full 3D cartridge model was considered: it would need 3D annotations or a calibrated camera,
which ~25 photos can't support. Front faces get most of the benefit for a fraction of the effort.

**Quiz changes from your feedback:**
- "Not usable" or "Skip" on a photo now skips its crops.
- Sharp photos come first.
- The crop view is being fitted to one screen, no scrolling, with tap-to-zoom.
- "Can't tell" never spoils the data; only a confident wrong answer would.
