---
title: "Final pick: RF-DETR Medium + Qwen Flash Next reader"
date: 2026-11-27T21:00:00-05:00
tags: [results, selection, final]
---

![final bundle on the best validation frame](runs/cli_final_demo_flash/frame.png)

With Flash Next added as a reader, the official scoring on your answers picks:

**RF-DETR Medium on 1280 px tiles + Qwen 3.8 Flash Next (brand first, then zoomed part, with a
shelf hint) + shelf-aware counting.** Bundle: `rfdetr-final-v2`.

| | value (95% range) |
|---|---|
| avg. count error per photo | **2.1** (0–5.1), top 0.7, bottom 1.4 |
| cartridges named exactly right | **65.5%** (Flash), vs 58.6% with Inkling |
| finding cartridges | ~2 s per photo on CPU |
| naming them | 45–125 s per photo (Flash reads every crop) |

Flash Next beat Inkling as the reader with every detector we tried, which is consistent with
the earlier crop-only test.

**Known weak spots:**
- CC read as NNS, and red bottom-shelf Coke boxes read as Hi-C. Both are brand confusions on
  dark, blurry crops.
- The Part 1/2 number is almost never readable at this camera resolution.
- 10 test photos means wide error bars.

**Next:** the final report is being written now. After that, the main way to improve is your
training crops, which give the system real Part 1/2 and Coke-family examples.
