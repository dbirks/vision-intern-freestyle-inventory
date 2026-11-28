---
title: "First look: the front-face bundle on sharp phone photos"
date: 2026-11-28T03:00:00-05:00
tags: [phone, reference, faces]
---

![front-face bundle on phone photo 1](runs/phone_rfdetr-faces-final/p01.png)

This is a reference check only. It isn't scored and nothing here goes into training.

On your two phone photos, `rfdetr-faces-final` boxed **26** and **24** cartridges, plus 1 and 6
it couldn't name. The detector puts a box on almost every printed end, including the bottom-shelf
Coke, Diet Coke and Coke Zero rows. The large NNS box and the far-right boxes on the top shelf look
like possible misses.

**Naming is still the weak link, even on sharp photos.** One example: it reports 3 Fanta where
two Fanta faces are visible.

**Slow:** naming took about 7–8 minutes per photo, because the Collective was at capacity and
every call kept retrying. A client fix now retries the Collective's temporary "not currently
available" 404 instead of crashing.

**Next:** your true counts for these two photos, to check how close 26/24 is. The Medium (S2)
bundle is still running for comparison.
