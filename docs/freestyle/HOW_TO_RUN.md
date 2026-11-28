# How to run the Freestyle counting CLI

End-to-end: a cabinet photo goes in, cartridges come out **located → identified →
counted**, plus an annotated overlay. The counting model is a frozen *bundle* under
`artifacts/<name>/` (gitignored).

This document is written against **`rfdetr-faces-final`** — the bundle selected by the **corrected**
human-GT scoring in [`RESULTS_GT.md`](RESULTS_GT.md) (128 answered quiz questions; the front-face
detector `G2_rfdetr_small_faces` + the Qwen 3.8 Flash Next reader, count MAE **0.500** and identity
**0.750** on the **2** usable test photos). **`rfdetr-final-v2`** (whole-cartridge `S2_c2_tile1280`,
identical reader and counting knobs, count MAE 2.000 but AP50 0.950 vs 0.527) stays documented as the
alternative: use it when you need cartridge *boxes* rather than a shelf count. Both picks rest on two
photographs of one cabinet — read §8.9/§11 of [`REPORT.md`](REPORT.md) first. Every number here is either
read from a run artifact or from a command actually executed on this host.

> Everything here runs on CPU. A trained bundle's detector is happiest on a GPU, but the
> full pipeline works on CPU too (see *GPU vs CPU* at the bottom for measured timings).
>
> **Both bundles need `LLMC_API_KEY` in the environment at inference time.** Their identifier is a
> vision language model (`qwen3-8-flash-next`, two-stage), called **once per detection** — there is no
> local gallery any more. Without the key the run identifies nothing and every box lands in `unknown`.

## 1. Install

The CLI needs the optional `train` extra (torch + transformers + RF-DETR):

```bash
uv sync --extra train            # base + torch/torchvision/transformers/timm/...
# RF-DETR is not pinned in pyproject (the k8s training images pin it separately):
uv pip install "rfdetr==1.12.1"
```

The `--json`/overlay path needs no GPU libraries beyond torch. Neither bundle pulls **any** Hugging Face
weights: the detector checkpoint is inside the bundle and the identifier is an API call. (The
superseded kNN bundles `rfdetr-m-1280-knn-v0` / `-v1` / `-v2` do pull `google/siglip2-base-patch16-224`
into the local HF cache on first use.)

## 2. List bundles

```bash
uv run python -m freestyle.cli bundles
```

```
Bundle                   Experiment                     Embedder                   Thresholds
------------------------ ------------------------------ -------------------------- --------------------
rfdetr-faces-final       G2_rfdetr_small_faces          qwen3-8-flash-next         score=0.65, id=0.45
rfdetr-final-v2          S2_c2_tile1280                 qwen3-8-flash-next         score=0.3080451190471649, id=0.45
rfdetr-m-1280-knn-v0     S2_c2_tile1280                 google/siglip2-base-patch16-224 score=0.3080451190471649, id=0.5
rfdetr-m-1280-v1         S2_c2_tile1280                 google/siglip2-base-patch16-224 score=0.3080451190471649, id=0.45
rfdetr-m-1280-v2         S2_c2_tile1280                 google/siglip2-base-patch16-224 score=0.3080451190471649, id=0.45
```

`bundles --json` gives the same rows as JSON, including integrity status and the metrics summary — which
is always labelled **"vs model labels, NOT GT"**, because `freestyle.artifacts export` copies the
*run's* model-label metrics block into the manifest. The human-GT numbers for this bundle are in
[`RESULTS_GT.md`](RESULTS_GT.md) and [`REPORT.md`](REPORT.md) §8.2/§8.9, not in the manifest.

`rfdetr-faces-final` is the provisional pick on the metric the product cares about (how many cartridges
are on the shelves: 0.500 vs 2.000 count MAE on the 2 scored photos) and `rfdetr-final-v2` is the
alternative when box quality matters (AP50 0.950 vs 0.527 — a face box is a *fragment* of a cartridge box,
so the face detector is penalised by the yardstick as much as by the misses, REPORT §8.2.4/§9).
`-knn-v0`, `-v1` and `-v2` are the pre-GT candidates, kept so the identifier comparison in §8.2.2 of the
report is reproducible; against a human their embedding-kNN identifier scores 0.421 (16/38) on `S2` where
the shipped reader scores 0.579 (22/38), and it collapses to 0.036 (1/28) on the face bundle.

## 3. Count cartridges in one or more frames

```bash
uv run --extra train python -m freestyle.cli count \
    data/raw/freestyle-vision-data/images/local/downloads/frame.jpg \
    data/raw/freestyle-vision-data/images/island-cabinet-01/e194690f-0b7e-4c37-ad88-0fd27481212f/002.jpg \
    data/raw/freestyle-vision-data/images/island-cabinet-01/581c0d5e-848f-4292-9dcc-43d0cc6990fd/045.jpg \
    --bundle rfdetr-faces-final \
    --device cpu \
    --out runs/cli_faces_demo \
    --json
```

Each frame gets an overlay PNG (boxes + `class S<shelf>` + unknowns) under `--out`, and
`--out/counts.json` collects the machine-readable result. Verbatim (keys/values as recorded) output from
the real CPU run — `runs/cli_faces_demo/counts.json`; wrapper `runs/cli_faces_demo/_run_demo.py` logged
`WALL_SECONDS=230.54 EXIT=0 PEAK_RSS_GB=1.26`, no 429/5xx retries in `runs/cli_faces_demo/stderr.log`:

```json
{
  "bundle": "rfdetr-faces-final",
  "device": "cpu",
  "images": [
    { "image": ".../local/downloads/frame.jpg",
      "counts": { "NNS": 6, "Hi-C": 5, "Fanta": 2, "Ingredient Solution CC": 2,
                  "Ingredient Solution AA": 1, "Dr Pepper Part 2": 1, "Lemon Flavor": 1, "Sprite": 1 },
      "unknown": 7, "flags": [],
      "timings_ms": { "localize": 1783.9, "deduplicate": 616.3, "identify": 64901.3, "count": 5.2 },
      "overlay": "runs/cli_faces_demo/frame.png" },
    { "image": ".../e194690f-…/002.jpg",
      "counts": { "Hi-C": 1, "Ingredient Solution AA": 1, "Fanta": 2, "NNS": 6, "Sprite": 1 },
      "unknown": 10, "flags": [],
      "timings_ms": { "localize": 1712.7, "deduplicate": 490.7, "identify": 71154.9, "count": 4.8 },
      "overlay": "runs/cli_faces_demo/002.png" },
    { "image": ".../581c0d5e-…/045.jpg",
      "counts": { "NNS": 7, "Vanilla Flavor": 1, "Hi-C": 3, "Ingredient Solution AA": 1,
                  "Sprite": 1, "Fanta": 2 },
      "unknown": 10, "flags": [],
      "timings_ms": { "localize": 1740.5, "deduplicate": 608.5, "identify": 80072.4, "count": 2.9 },
      "overlay": "runs/cli_faces_demo/045.png" }
  ]
}
```

Counts use catalog **display names** (resolved from `data/raw/freestyle-vision-data/data/catalog_items.json`, override with
`--catalog`). Anything the identifier could not place above its `identity_thr` (0.45), or that the
detector only called generically `cartridge`, is reported in `unknown`, **not** guessed into a class.
Drop `--json` for a human-readable table. `--group <id>` adds a gallery-leakage check that only applies
to the kNN bundles.

**Read those three outputs against the human answers before believing them.** These are the two test
photos the whole selection rests on, plus the one usable validation photo:

| Frame | What the human said | What the bundle did (boxes = named + `unknown`) |
|---|---|---|
| `local/downloads/frame.jpg` | usable cabinet, **26** cartridges (17 top / 9 bottom), flagged `counts_mismatch` in `data/gt/val_counts.json`; this single frame is what selected `score_thr 0.65` | **26** (19 + 7) — total exact, 7 cartridges left `unknown` |
| `e194690f…/002.jpg` | usable, **22** (14 top / 8 bottom), fully answered, no mismatch | **21** (11 + 10) — 1 low, **10 unnamed** |
| `581c0d5e…/045.jpg` | usable, **28** (17 top / 11 bottom), `counts_mismatch: true` (only 22 human-verified boxes exist) | **25** (15 + 10) — 3 low, 10 unnamed |

Two honest wrinkles in that table. **(a)** The scored row says `G2` predicts 22 and 27 on these two photos
(error 0 and 1, MAE 0.500) while the *live bundle* gives 21 and 25: `score_all` counts the run's saved rows
above `score_thr`, the CLI additionally applies the bundle's duplicate suppression and 2-shelf refit, which
costs 1–2 real boxes here (REPORT §7.4 "suppression gap", §8.9 deviation 2). **(b)** The bundle's
`identity_thr` is **0.45** where the scored identity accuracy (0.750 = 21/28) used **0.25**, so the
named/`unknown` split above is more pessimistic than the scored row — re-export with
`"identity_thr":0.25` to reproduce it. The named counts are also NNS-skewed (6 of 11, 7 of 15): the
auto-labelled gallery and the reader both see NNS often and the red/silver bottom shelf rarely.

**Not scored, not evidence:** 8 more test photos were answered *"not a usable cabinet view"* and 2 have no
count answer at all (`3270fb41…/065.jpg`, `50828e4b…/008.jpg`). Those 10 frames are **excluded** from every
metric — uncountable photos, not zero-cartridge photos — so running the CLI on one of them proves nothing
either way. (The old demo in `runs/cli_final_demo/` used two such frames; its `3 boxes on
10455345…/013.jpg` line was never a measured false positive, it was a box count on a frame nobody
scored.)

## 4. Fuse several frames of one door opening

A single door opening is captured as a short burst of near-duplicate frames. Counting
each frame and **summing** would multiply the same cartridge by the frame count, so use
`fuse`, which combines the per-frame count vectors with a cross-frame rule
(`max` (default), `median`, `weighted_mode`):

```bash
uv run python -m freestyle.cli fuse \
    data/raw/freestyle-vision-data/images/island-cabinet-01/bb2c251e-38ec-4952-a1d2-250436c7dbe5/000.jpg \
    data/raw/freestyle-vision-data/images/island-cabinet-01/bb2c251e-38ec-4952-a1d2-250436c7dbe5/001.jpg \
    --bundle rfdetr-final-v2 --device cpu --method max --json
```

The recorded `max` output on these two frames with the *superseded* kNN bundle was
`{"method":"max","fused_counts":{"Fanta":1,"NNS":7},"fused_unknown":0,...}`
(`runs/bundle_demo/`; frame `000` located 8 faces, `001` was a near-blank moved frame that found none,
so `max` keeps the good frame's inventory instead of averaging it down). Both frames were answered *"not a usable cabinet view"*
by the human, so under the corrected scoring they are **excluded** (`reason = not_usable`) and that
`Fanta 1 / NNS 7` is *unverified* output rather than a measured hallucination — but `fuse`'s `max` rule
loudly amplifies whichever frame the detector liked best, which is exactly the behaviour to be wary of. `weighted_mode`
picks the most-supported observed count across frames; `median` under-counts when most frames show only
a partial view; `--weights` (one float per frame) tunes `weighted_mode`. Treat burst fusion as untested
against human answers: `score_all` scores single frames only.

## 5. Rebuilding the bundle from an experiment run

A bundle freezes one training run. `rfdetr-faces-final` was exported **by hand** from
`runs/G2_rfdetr_small_faces-7c7b4587/` — the run that wins on corrected human-GT test counting — with the
reader chosen by the same scoring pass (`vlm_flash`). The exact commands, as executed:

```bash
# (only if a checkpoint lacks its head-index map: it fails loudly with
#  "RF-DETR checkpoint lacks its training category mapping")
uv run --extra train python -m freestyle.train.recover_mapping runs/G2_rfdetr_small_faces-7c7b4587

uv run --extra train python -m freestyle.artifacts export \
    --run runs/G2_rfdetr_small_faces-7c7b4587 \
    --name rfdetr-faces-final --overwrite \
    --identifier '{"type":"vlm_crop","model":"qwen3-8-flash-next","mode":"two_stage","upscale":2}' \
    --counting   '{"identity_thr":0.45,"neighbour_vote":"report","max_rows":2,"score_thr":0.65,"suppress_iou":0.45,"suppress_containment":0.7,"suppress_row_iou":0.3,"suppress_row_x_overlap":0.5,"shelf_hint":true,"pair_rule":"report"}'

uv run python -m freestyle.artifacts verify rfdetr-faces-final
# OK: rfdetr-faces-final (3 files verified)
```

`recover_mapping` was **not** needed: the checkpoint already carries
`freestyle_category_names = {1: "cartridge"}` / `freestyle_label_to_category = {0: 1}` (checked with
`torch.load(...)`), which is what `RFDetrPredictor` insists on.

What lands in `artifacts/rfdetr-faces-final/` (**3 files**, no `gallery.npz`, because the identifier is an
API call and not embedding search):

- `pipeline.yaml` — tiled RF-DETR (`tile: 1024`, `overlap: 0.25`, front-face weights), `vlm_crop`
  `qwen3-8-flash-next` `two_stage` upscale 2, `counting: {score_thr: 0.65, identity_thr: 0.45, max_rows: 2,
  neighbour_vote: report, pair_rule: report, shelf_hint: true, suppress_*}`. `score_thr 0.65` **is** the
  val-selected value from `docs/freestyle/results_gt.json` — but only because it was passed explicitly:
  `freestyle.artifacts export` otherwise copies the run's own `metrics.json` threshold (`0.5587` for `G2`),
  which is how `rfdetr-final-v2` ended up shipping `0.308`.
- `detector/checkpoint_best_total.pth` — 127,904,367 bytes, sha256 `5f7f0bc9…`, `detector_classes: ["cartridge"]`.
- `classes.json` — the 19 catalog ids + `cartridge`/`unknown` map (1,431 bytes).
- `manifest.json` — provenance: run id `G2_rfdetr_small_faces-7c7b4587`, git sha `e9adc98…` (dirty),
  thresholds `{score_thr: 0.65, identity_thr: 0.45}`, reader id, the run's **model-label** metrics block
  (explicitly caveated `NOT human GT`), the selection note, and a SHA-256 per file.

To rebuild the alternative instead, run the same two commands with `--run
runs/S2_c2_tile1280-db269936 --name rfdetr-final-v2` (drop `"score_thr":0.65` from `--counting` to
reproduce exactly what is on disk today; `"score_thr":0.20` is what its own human-GT validation selection
says). `scripts/finalize_after_gt.sh` step 6 picks the winner the same way but hardcodes
`--bundle-name rfdetr-final-v2` and sends no `score_thr`, so after a re-score it will happily export a
bundle that does not reproduce its own headline (REPORT §12.7).

Verify integrity any time (also runs automatically on load):

```bash
uv run python -m freestyle.artifacts verify rfdetr-faces-final
# OK: rfdetr-faces-final (3 files verified)
```

The pre-GT kNN bundles worked differently: `freestyle.artifacts export` copied a run's `gallery.npz` or
rebuilt one from the run's train-split, known-class crop embeddings (`embeddings.npz`) using the same
embedder, and rejected an `--identifier` model that disagreed with the cached vectors (the vectors would
not be comparable). None of that applies to either current bundle — passing `--identifier
'{"type":"embedding_knn",...}'` today would just re-create the weaker reader that scores 0.421 (16/38) on
`S2` and 0.036 (1/28) on the faces (§8.2.2 of the report).

## 6. GPU vs CPU

| | detector (faces: RF-DETR-S, tile 1024) | reader | notes |
|---|---|---|---|
| **GPU** (RTX 5090, k8s Job) | 0.242 s/img test, 0.365 s/img val, 191.4 s train, 6.89 GiB peak (`runs/G2_rfdetr_small_faces-7c7b4587/meta.json`) | n/a (API) | only place GPU work may run |
| **GPU** (alternative: RF-DETR-M, tile 1280) | 0.223 s/img test, 0.309 s/img val, 145.2 s train, 4.09 GiB peak (`runs/S2_c2_tile1280-db269936/meta.json`) | n/a (API) | |
| **CPU** (this host, measured) | localize **1.71–1.78 s/img** + dedup 0.49–0.62 s | **64.9–80.1 s per photo** (2.5–3.4 s/crop) | `runs/cli_faces_demo/counts.json`, `wall.json` |

Measured on this host (CPU only, no GPU touched):

- **Wall time splits overwhelmingly into identification.** Across the three demo frames: localize 5.24 s,
  dedup 1.72 s, count 0.013 s, **identify 216.13 s** of a 230.54 s wall — ~94 % of the pipeline's stage
  time. 72 crops at **3.0 s/crop average**.
- **Plan for the slow version of that number.** The same reader measured **9.1 s mean / 6.3 s median per
  crop** in `runs/reader_compare/results.json` under a different endpoint load, so a stocked photo
  (~25 cartridges) is **75–225 s** of API time; today's run needed 65–80 s. `freestyle/collective.py`
  retries 429/5xx with exponential backoff + jitter up to 600 s per call, and the demo log shows no
  retries today.
- **Process memory**: **1.26 GB** peak RSS for the whole three-frame run
  (`runs/cli_faces_demo/wall.json`, `resource.getrusage(RUSAGE_CHILDREN)`) — RF-DETR Small weights, no
  embedder loaded. The older kNN v0 demo peaked at 1.73 GB (`runs/bundle_demo/count.err`).
- On a GPU the localize stage collapses to ~0.24 s and the reader stays an API call — the reader is the
  bottleneck either way. Per-stage `timings_ms` are emitted per image in the JSON (`localize`,
  `deduplicate`, `identify`, `count`), so you can watch it.

GPU experiments run only as Kubernetes Jobs on the home cluster (see `k8s/README.md`);
this CPU path exists so the *full* product loop — locate, identify, count, overlay —
can be demonstrated and smoke-tested without holding a cluster card. Use `--device cuda`
when a CUDA device is present and `--device cpu` (or leave `auto`) otherwise.

## 7. Scoring the pipeline against human ground truth

The CLI is how you run the bundle; the numbers quoted above come from the scoring pass, not from the
CLI. To reproduce them:

```bash
uv run python -m freestyle.experiments.score_all \
  --queue data/gt/quiz_queue.json --answers data/gt/quiz_answers.jsonl \
  --proposals data/labels/bootstrap_v2.json \
  --proposals-flash data/labels/bootstrap_v3_flash.json \
  --runs "runs/*" --bootstrap 1000 --out docs/freestyle/RESULTS_GT.md

uv run python -m freestyle.experiments.reader_compare          # per-crop reader bake-off
```

or, end to end (quiz gate → `to_coco` → `score_all` → report placeholders → bundle export → CLI demo):

```bash
scripts/finalize_after_gt.sh --skip-quiz        # re-scores; usable-frame gate: >=2 test, >=1 val photos
scripts/finalize_after_gt.sh --dry-run          # prints everything, mutates nothing
```

A frame answered *"not a usable cabinet view"* never satisfies that gate — the gate counts photos a human
actually **counted**, which is currently 2 test + 1 val. Read
[`RESULTS_GT.md`](RESULTS_GT.md)'s header and REPORT §8.9 correction history before trusting a row.

## What the overlays actually show (honest read, against a human)

On the three photos above the face detector finds the stocked shelves and shelves them into two rows, and
the overlays (`runs/cli_faces_demo/*.png`, private) show what that means: 26/21/25 boxes where a human
counted 26/22/28, i.e. it is **slightly low on both stocked photos** after suppression rather than
doubling cartridges. Scored without the CLI's suppression (`score_all`'s rule) the same weights give 22
and 27 — total-count MAE **0.500**, top **1.000**, bottom **0.500**, and the alternative whole-cartridge
bundle's scored top/bottom are 1.000/1.000 at MAE 2.000. The face detector's remaining misses are the
leaning bottom row and the tile edges; its false positives against whole-cartridge boxes are mostly
**fragments** (16 of 18 on the two scored frames), which is a face box being what a face box is.

Identification is good on the brands the auto-labelled gallery and the reader both see often — NNS, Fanta,
Sprite, Hi-C, AA — and falls off on the red/silver bottom shelf, where the reader answers `unknown` (10 of
21 crops on `002.jpg`, 10 of 25 on `045.jpg` at the bundle's `identity_thr 0.45`) or, worse with Flash
Next, calls `ingredient-cc` a NNS cartridge (3 crops in the crop-level bake-off). The Part 1 / Part 2 split
is effectively unavailable: the best reader gets the printed numeral right on **1 of 11** crops a human
named, because the numerals are not legible at this camera's resolution and blur, not because the model is
weak. `unknown` is the single biggest per-class error term (7 on `002.jpg`).

The operational warning is unchanged but re-justified: **10 of the 12 test photos could not be counted by
a human at all** (door swing, person, blur, angle) and are excluded from every metric, so nothing in this
document claims the bundle stays quiet on such a frame — that question is simply unmeasured, and the two
`missing_count` frames are the cheapest way to start measuring it. `neighbour_vote` and `gap` flags fire
on suspicious rows so a human reviews them rather than trusting the count — keep that habit. Every number
here was chosen on **two** photographs of **one** cabinet with **one** stocking; `coca-cola-part1` and both
`coke-zero` SKUs never appear at all. Treat per-class numbers as indicative and re-read
[`REPORT.md`](REPORT.md) §8.9 and §11 before acting on one.
