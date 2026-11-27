# Freestyle cartridge recognition — failure catalogue

Failure decomposition for every collected experiment run, plus a curated gallery of the
recurring visual failure modes. Read [`REPORT.md`](REPORT.md) §9 for the metric context and
[`RESOURCES.md`](RESOURCES.md) for the cost of each run.

> [!WARNING]
> **Nothing here is measured against human ground truth.** `freestyle/eval/failures.py` compares
> each run's `test/predictions.json` against the model-generated bootstrap proposals
> (`data/labels/bootstrap_v2.json`, `verdicts: [correct]`; the face runs G1/G2 against
> `data/labels/faces_v2.json`), at IoU 0.5 and each run's **val-selected** `score_thr`.
> An `FN` means *the detector missed something the VLM labeller claimed*; an `FP` means *the
> detector fired where the VLM labeller claimed nothing*. Both can be wrong. The face family
> (G1/G2) is scored against face labels, so its rows are **not comparable** to the
> whole-cartridge rows. `<!-- GT-PENDING: re-run `freestyle.eval.failures` against
> `data/gt/test.json` (and the face-GT variant if the quiz decides for faces) and replace every
> count and every confusion pair in this file. -->`

Countable taxonomy (`freestyle/eval/failures.py`):
* **FN** (missed proposal): `adjacent_merge` (the model box spans ≥2 cartridges / one box covers a
  neighbour too), `stacked_tier` (a tier of a 2-high stack is missing), `edge_truncated` (box cut
  by the frame edge), `plain`.
* **FP** (prediction with no proposal match): `background` (no cartridge there at all),
  `duplicate` (a second box on an already-covered cartridge), `fragment` (a sub-box / tile-seam
  sliver of one cartridge).
* **ID**: correctly localised, wrong class. For class-agnostic runs every correctly localised
  known-class box lands here as `<CLASS>→UNK`, which is why their ID column is large by
  construction and not evidence of a bad identifier.
* **LOC**: right box within `0.3 ≤ IoU < 0.5`, i.e. localised but sloppily.

`n_tp` counts boxes correct in **both** localisation and class.

## 1. Counts per run, type and sub-type

Whole-cartridge localizers (`n_gt` = 176 proposals across 12 test images / 11 groups):

| Run (label space) | thr | kept | TP | failures | FN | adj_merge / edge / plain / stack | FP | bg / dup / frag | ID | LOC |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| `B1_owlv2_dinov2-4c99011d` | 0.101 | 178 | 26 | 247 | 95 | 51 / 10 / 29 / 5 | 97 | 54 / 10 / 33 | 41 | 14 |
| `B3_gdino_dinov2-be668339` | 0.252 | 82 | 17 | 180 | 115 | 45 / 11 / 48 / 11 | 21 | 18 / 0 / 3 | 39 | 5 |
| `S6_owlv2_containment-cbc70383` | 0.104 | 55 | 13 | 180 | 138 | 32 / 13 / 74 / 19 | 17 | 16 / 1 / 0 | 18 | 7 |
| `C1_rfdetr_small_agnostic_dinov2-55ef8ae7` | 0.315 | 210 | 50 | 199 | 39 | 19 / 2 / 16 / 2 | 73 | 26 / 12 / 35 | 84 | 3 |
| `C2_rfdetr_medium_agnostic_siglip2-89fe7c2b` | 0.538 | 169 | 42 | 175 | 48 | 16 / 4 / 24 / 4 | 41 | 10 / 7 / 24 | 81 | 5 |
| `D1_rfdetr_small_multiclass-6f3c344b` | 0.576 | 160 | 88 | 124 | 52 | 15 / 3 / 26 / 8 | 36 | 4 / 15 / 17 | 34 | 2 |
| `S1_c2_tile768-d52f6d7c` | 0.813 | 138 | 32 | 180 | 74 | 19 / 6 / 40 / 9 | 36 | 4 / 9 / 23 | 67 | 3 |
| `S2_c2_tile1280-db269936` | 0.308 | 171 | 46 | 166 | 41 | 16 / 4 / 20 / 1 | 36 | 12 / 7 / 17 | 86 | 3 |
| `S3_c2_copypaste-205b1c7c` | 0.373 | 189 | 52 | 173 | 36 | 17 / 3 / 14 / 2 | 49 | 14 / 16 / 19 | 84 | 4 |
| `S5_rfdetr_large_agnostic-7f6457d4` | 0.657 | 148 | 40 | 163 | 55 | 20 / 4 / 26 / 5 | 27 | 6 / 7 / 14 | 79 | 2 |
| `S7_c2_tile1280_copypaste-b6fc182e` | 0.443 | 160 | 45 | 163 | 48 | 18 / 2 / 23 / 5 | 32 | 12 / 4 / 16 | 82 | 1 |

Front-face localizers (G family; `n_gt` = 170 **faces** from `data/labels/faces_v2.json`, same 12
test images / 11 groups — different label space, do not read across the two tables):

| Run (label space) | thr | kept | TP | failures | FN | adj_merge / edge / plain / stack | FP | bg / dup / frag | ID | LOC |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| `G1_rfdetr_medium_faces-cc431fcd` | 0.374 | 213 | 48 | 201 | 36 | 12 / 4 / 18 / 2 | 79 | **56** / 2 / 21 | 74 | 12 |
| `G2_rfdetr_small_faces-7c7b4587` | 0.559 | 152 | 36 | 171 | 55 | 10 / 10 / 30 / 5 | 37 | 18 / 4 / 15 | 70 | 9 |

Source: `runs/<run>/failures/failures.json` (`summary.counts_by_type`,
`summary.counts_by_subtype`, `summary.score_thr`, `summary.n_*`). Contact-sheet galleries of the
individual FN/FP/ID/LOC crops are at `runs/<run>/failures/galleries/{FN,FP,ID,LOC}.png` (per-image
full-size overlays and previews in `runs/<run>/failures/overlays/`, browsable via
`runs/<run>/failures/index.html`). All of those are private and uncommitted.

What the counts say, structurally:
* Supervised tiled RF-DETR removes the zero-shot merge pathology: `adjacent_merge` falls from
  51 (`B1`) / 45 (`B3`) to 15–20, and `stacked_tier` from 11 (`B3`) to 1–9. In exchange the residual
  error moves to `FP:fragment` (14–35 partial sub-boxes, mostly tile-seam slivers) and ID.
* Copy-paste augmentation trades recall for over-prediction: `S3`/`S7` have the fewest FN (36/48)
  but the most `FP:duplicate` (16/4) — S3 is the duplicate-suppression case that motivated the v1
  `suppress_*` knobs.
* The containment filter on OWLv2 (`S6`) mostly deletes boxes rather than merging them: FP drops
  to 17 but FN climbs to 138 with 19 `stacked_tier`, i.e. the whole 2-high NNS stack disappears.
* Faces vs whole boxes: G1 halves `adjacent_merge` (16→12) but triples `FP:background` (12→56):
  a face detector fires on shelf-edge fragments and background slats that are not cartridges, and
  the face label space itself is noisier (53 residual duplicate face pairs were flagged, not
  deleted, in `data/labels/faces_v2.json` — `runs/faces_v2/summary.md`).
* `LOC` is uniformly small for trained detectors (1–5) and large only for zero-shot (14 in `B1`)
  and for G1 (12) where faces are small so a few pixels move IoU a lot.

## 2. Gallery — six representative failures

Paths only: the images show the cabinet and the person stocking it, so they stay under `runs/`
(gitignored, private) and are never committed or uploaded outside the Collective/cluster. All
six were looked at directly, downscaled, for this catalogue.

### 2.1 Merged neighbours — zero-shot spans a whole shelf run

`runs/B1_owlv2_dinov2-4c99011d/failures/overlays/island-cabinet-01_98c44b08-c6c9-42b3-ab50-89bb71ec9232_006.png`
(preview: same path + `_preview.png`)

OWLv2 large on `98c44b08…/006.jpg`: a handful of enormous boxes swallow the top-shelf NNS row and
the leaning bottom row (19 `FN:adjacent_merge` on this one image) while three more boxes are drawn
on empty door/ceiling (`FP:background 0.12–0.18`). This is the failure the trained detectors exist
to fix — same image contributes 3 merges and 4 background FPs in `S2`.

### 2.2 Stacked NNS tiers counted as one object

`runs/B1_owlv2_dinov2-4c99011d/failures/overlays/island-cabinet-01_b5c9000a-8ede-4aba-aa52-edbd897ab0d6_012.png`
(preview: same path + `_preview.png`)

The two-high `nns-sweetener` stack near the left of the top shelf. `B1` takes 5 `FN:stacked_tier`
on this frame and 18 `FN:plain` (the dark, leaning bottom row is simply not seen). `B3` shows the
same shape with 11 stacked-tier misses corpus-wide; `S2` reduces it to a single stacked-tier miss
across all 12 test images, and `S6`'s containment filter *increases* it to 19 because it removes
the tall stack box entirely.

### 2.3 Leaning bottom row — boxes slide across neighbours and duplicate

`runs/bundle_demo/frames/frame.png` (bundle v0 overlay; frame `local/downloads/frame.jpg`)

The best-lit frame in the corpus, run through the v0 bundle. Boxes on the leaning bottom shelf are
drawn as tall axis-aligned rectangles that cut across the cartridge behind them, so several
cartridges carry two overlapping boxes and the red Coca-Cola family boxes are labelled `Hi-C S1`;
v0 reports 8 NNS / 4 Hi-C / 3 Fanta / 2 vanilla + 9 unknown on this frame
(`runs/bundle_demo/frames/counts.json`). The v1 duplicate suppression (containment 0.7, same-row
IoU 0.3) plus the `max_rows: 2` shelf cap exist precisely for this frame; v1 gives 7 NNS + 1 each
of AA/CC/Sprite/Orange/Vanilla + 2 Fanta + 2 Lemon + 7 unknown
(`runs/bundle_demo_v1/counts.json`), which matches the physical 3-stack + 4 NNS on visual
inspection — **unverified by a human**.

### 2.4 Blur / dark frames — nothing is localizable

`runs/S2_c2_tile1280-db269936/failures/overlays/island-cabinet-01_b5c9000a-8ede-4aba-aa52-edbd897ab0d6_012_preview.png`

A `cabinet_partial` frame with `motion_blur = severe`, `sharp_full = 5.68`, a hand and a stylus
across most of the field (`data/derived/audit/audit.json`). The detector keeps three boxes on the
visible NNS column and drops the rest: 16 `FN:plain` + 2 `FN:adjacent_merge` + 1 `LOC` on this one
image, the single largest FN contributor to `S2`'s 41. The same frame is the worst in `G1` (12
`FN:plain`) and `D1` (17 `FN:plain`, 3 `FN:stacked_tier`). Six of the 12 test images are severe
blur by construction (`docs/freestyle/DATA_AUDIT.md` §splits), so this is the expected shape of the
test number, not an outlier.

### 2.5 Identity confusion — gallery bias toward NNS (bundle v0)

`runs/bundle_demo/008_view.jpg` and `runs/bundle_demo/frames/counts.json` (frame `50828e4b…/008.jpg`)

Plain SigLIP2 kNN over the train-only gallery: **22 NNS** on a frame that physically holds ~7, plus
`gap` and `neighbour_vote` flags (`runs/bundle_demo/frames/counts.json`). The v0 gallery held 139
crops of which 60 were NNS (`docs/freestyle/HOW_TO_RUN.md` §5), so everything blue-ish — and
several things that were not — votes NNS. On the best frame the same mechanism turned CC into NNS
and AA into vanilla during visual review (`devlog/content/2026-11-26-end-to-end-on-cpu.md`). v1's
fix is a class-balanced gallery (≤10 crops/class) that may only refine *within* the VLM's family:
the same frame drops to 5 NNS + 2 CC + 2 AA + 2 Fanta + 10 unknown (`runs/bundle_demo_v1/counts.json`).

### 2.6 Identity confusion — Part 1 / Part 2 and yellow-family collapse (trained multiclass)

`runs/D1_rfdetr_small_multiclass-6f3c344b/failures/galleries/ID.png`
(per-image overlays in `runs/D1_rfdetr_small_multiclass-6f3c344b/failures/overlays/`)

`D1` is the only trained run with a real identity head, and its 34 ID errors split into 23
known→`UNK` (`FL-VAN→UNK` 6, `NNS→UNK` 4, `FANTA→UNK`/`FL-LEM→UNK`/`HI-C→UNK`/`SPRITE→UNK` 3 each,
`FL-ORG→UNK` 1) and 11 inter-class confusions that are all pale-yellow/cream flavours
(`FANTA→FL-VAN` 2, `ING-AA→FL-VAN` 2, `FL-LEM→FANTA`, `FL-LEM→FL-VAN`, `FL-VAN→FANTA`,
`HI-C→FL-VAN`, `SPRITE→FL-VAN`, `SPRITE→NNS`, `FANTA→FL-LEM` —
`runs/D1_rfdetr_small_multiclass-6f3c344b/failures/failures.json`, `summary.confusion_matrix.pairs`).
**No two-part brand confusion appears at all**, because the bootstrap labels almost never resolve a
Part numeral: 2 of 123 two-part crops were legible whole-frame
(`runs/bootstrap_v2/summary.md`), and even the face labeller only read 59 numerals off 597 faces
(45 `1`, 14 `2`, `runs/faces_v2/summary.md`). Part 1 vs Part 2 therefore remains an open question
that only the pair rule + human GT can answer, not a measured confusion.

## 3. Follow-ups these failures point at

1. Duplicate suppression on the leaning bottom row is a **counting** fix, not a detector fix:
   `S3` has the most `FP:duplicate` (16) *and* the best AP50 (0.718), so the boxes are good and the
   merge step is what double-counts. Verify against human counts.
2. G1's `FP:background = 56` needs either a negative-aware face training set (only 9 train frames
   carry faces today) or a minimum-face-size/aspect gate; `faces_v2` already drops `thin-strip` and
   `flat-aspect` at label time but keeps 53 flagged duplicate pairs for human review.
3. `FP:fragment` errors are tile-seam artefacts: raise tile overlap or suppress fragments below
   `0.35×` the row's median face width (the rule already used in `freestyle/` face cleanup).
4. The `unknown` bucket is the honest answer in most of these frames; every metric change that
   lowers `unknown` must be checked against human answers, not against bootstrap labels.
