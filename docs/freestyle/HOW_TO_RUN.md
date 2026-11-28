# How to run the Freestyle counting CLI

End-to-end: a cabinet photo goes in, cartridges come out **located → identified →
counted**, plus an annotated overlay. The counting model is a frozen *bundle* under
`artifacts/<name>/` (gitignored).

This document is written against **`rfdetr-final-v2`** — the bundle selected and shipped by
`scripts/finalize_after_gt.sh` on **human ground truth** (128 answered quiz questions, scored in
[`RESULTS_GT.md`](RESULTS_GT.md)). It was produced from real CPU runs on this host; every number here is
either read from a run artifact or from a command actually executed.

> Everything here runs on CPU. A trained bundle's detector is happiest on a GPU, but the
> full pipeline works on CPU too (see *GPU vs CPU* at the bottom for measured timings).
>
> **`rfdetr-final-v2` needs `LLMC_API_KEY` in the environment at inference time.** Its identifier is a
> vision language model (`qwen3-8-flash-next`, two-stage), called **once per detection** — there is no
> local gallery any more. Without the key the run identifies nothing and every box lands in `unknown`.

## 1. Install

The CLI needs the optional `train` extra (torch + transformers + RF-DETR):

```bash
uv sync --extra train            # base + torch/torchvision/transformers/timm/...
# RF-DETR is not pinned in pyproject (the k8s training images pin it separately):
uv pip install "rfdetr==1.12.1"
```

The `--json`/overlay path needs no GPU libraries beyond torch. The final bundle pulls **no** Hugging Face
weights at all: the detector checkpoint is inside the bundle and the identifier is an API call. (The
superseded kNN bundles `rfdetr-m-1280-knn-v0` / `-v1` / `-v2` do pull `google/siglip2-base-patch16-224`
into the local HF cache on first use.)

## 2. List bundles

```bash
uv run python -m freestyle.cli bundles
```

```
Bundle                   Experiment                     Embedder                   Thresholds
------------------------ ------------------------------ -------------------------- --------------------
rfdetr-final-v2          S2_c2_tile1280                 qwen3-8-flash-next         score=0.3080451190471649, id=0.45
rfdetr-m-1280-knn-v0     S2_c2_tile1280                 google/siglip2-base-patch16-224 score=0.3080451190471649, id=0.5
rfdetr-m-1280-v1         S2_c2_tile1280                 google/siglip2-base-patch16-224 score=0.3080451190471649, id=0.45
rfdetr-m-1280-v2         S2_c2_tile1280                 google/siglip2-base-patch16-224 score=0.3080451190471649, id=0.45
```

`bundles --json` gives the same rows as JSON, including integrity status and the metrics summary — which
is always labelled **"vs model labels, NOT GT"**, because `freestyle.artifacts export` copies the
*run's* model-label metrics block into the manifest. The human-GT numbers for this bundle are in
[`RESULTS_GT.md`](RESULTS_GT.md) and [`REPORT.md`](REPORT.md) §8.2/§8.9, not in the manifest.

`rfdetr-final-v2` is the one to use. `-knn-v0`, `-v1` and `-v2` are the pre-GT candidates, kept so the
identifier comparison in §8.2.2 of the report is reproducible; against a human their embedding-kNN
identifier scores 0.517 where the shipped reader scores 0.655.

## 3. Count cartridges in one or more frames

```bash
uv run python -m freestyle.cli count \
    data/raw/freestyle-vision-data/images/local/downloads/frame.jpg \
    data/raw/freestyle-vision-data/images/island-cabinet-01/50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg \
    data/raw/freestyle-vision-data/images/island-cabinet-01/10455345-10b3-4baf-bf2e-00bfefc562c7/013.jpg \
    --bundle rfdetr-final-v2 \
    --device cpu \
    --out runs/cli_final_demo \
    --json
```

Each frame gets an overlay PNG (boxes + `class S<shelf>` + unknowns) under `--out`, and
`--out/counts.json` collects the machine-readable result. Verbatim `--json` output from the real run
(`runs/cli_final_demo/counts.json`, log `~/.local/share/vision-intern/final-demo.log`, `DEMO_EXIT=0`):

```json
{
  "bundle": "rfdetr-final-v2",
  "device": "cpu",
  "images": [
    {
      "image": "data/raw/freestyle-vision-data/images/local/downloads/frame.jpg",
      "counts": { "NNS": 7, "Lemon Flavor": 2, "Fanta": 2, "Ingredient Solution CC": 1,
                  "Sprite": 1, "Orange Flavor": 1, "Vanilla Flavor": 1, "Ingredient Solution AA": 1 },
      "unknown": 7,
      "flags": [],
      "timings_ms": { "localize": 1520.8, "deduplicate": 621.9, "identify": 82651.1, "count": 6.9 },
      "overlay": "runs/cli_final_demo/frame.png"
    },
    {
      "image": ".../50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg",
      "counts": { "Fanta": 2, "Ingredient Solution CC": 2, "Ingredient Solution AA": 1, "NNS": 5 },
      "unknown": 10,
      "flags": [ { "type": "gap", "detections": [9, 3] } ],
      "timings_ms": { "localize": 1284.8, "deduplicate": 500.0, "identify": 79169.7, "count": 2.5 },
      "overlay": "runs/cli_final_demo/008.png"
    },
    {
      "image": ".../10455345-10b3-4baf-bf2e-00bfefc562c7/013.jpg",
      "counts": { "Sprite": 1 }, "unknown": 2, "flags": [],
      "timings_ms": { "localize": 3330.5, "deduplicate": 5.3, "identify": 16477.0, "count": 0.3 },
      "overlay": "runs/cli_final_demo/013.png"
    }
  ]
}
```

Counts use catalog **display names** (resolved from `data/raw/freestyle-vision-data/data/catalog_items.json`, override with
`--catalog`). Anything the identifier could not place above its `identity_thr` (0.45), or that the
detector only called generically `cartridge`, is reported in `unknown`, **not** guessed into a class.
Drop `--json` for a human-readable table. `--group <id>` adds a gallery-leakage check that only applies
to the kNN bundles.

**Read those three outputs against the human answers before believing them:**

| Frame | What the human said | What the bundle did |
|---|---|---|
| `local/downloads/frame.jpg` | usable cabinet, **26** cartridges (17 top / 9 bottom) — flagged `counts_mismatch` in `data/gt/val_counts.json` | 23 boxes, 16 named (`7 NNS` is the physically right NNS count), 7 `unknown` |
| `50828e4b…/008.jpg` | **no human count at all** — the count question was skipped, so this image is excluded from every scored metric | 20 boxes, 10 named — unverified |
| `10455345…/013.jpg` | `not_cabinet` — a human saw no usable cabinet, so the right answer is **0** | 3 boxes, 1 named → **3 false positives** |

Those false positives are a threshold artefact, not a model failure: this bundle ships
`score_thr 0.308`, while the configuration that won on human validation counts used **0.90**, at which
`S2` emits exactly **1** box across all eight frames a human called empty. See §8.9 (deviation 1) —
re-export with the val-selected threshold before using this on a real cabinet.

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
so `max` keeps the good frame's inventory instead of averaging it down). **Both frames are now known
to be `not_cabinet` negatives in the human GT** — a human answered "no usable cabinet view" on them, so
that `Fanta 1 / NNS 7` was pure hallucination and `fuse`'s `max` rule amplified it. `weighted_mode`
picks the most-supported observed count across frames; `median` under-counts when most frames show only
a partial view; `--weights` (one float per frame) tunes `weighted_mode`. Treat burst fusion as untested
against human answers: `score_all` scores single frames only.

## 5. Rebuilding the bundle from an experiment run

A bundle freezes one training run. `rfdetr-final-v2` was exported by `scripts/finalize_after_gt.sh`
(step 6) from `runs/S2_c2_tile1280-db269936/` — the run that won on human-GT test counting — with the
reader chosen by the same pass. The exact commands:

```bash
uv run python -m freestyle.artifacts export \
    --run runs/S2_c2_tile1280-db269936 \
    --name rfdetr-final-v2 --overwrite \
    --identifier '{"type":"vlm_crop","model":"qwen3-8-flash-next","mode":"two_stage","upscale":2}' \
    --counting   '{"identity_thr":0.45,"neighbour_vote":"report","max_rows":2,"suppress_iou":0.45,"suppress_containment":0.7,"suppress_row_iou":0.3,"suppress_row_x_overlap":0.5,"shelf_hint":true,"pair_rule":"report"}'

uv run python -m freestyle.artifacts verify rfdetr-final-v2
# OK: rfdetr-final-v2 (3 files verified)
```

What lands in `artifacts/rfdetr-final-v2/` (128 MB, **3 files** — no `gallery.npz`, because the
identifier is an API call and not embedding search):

- `pipeline.yaml` — frozen config: tiled RF-DETR (`tile: 1280`, `overlap: 0.25`), `vlm_crop`
  `qwen3-8-flash-next` `two_stage` upscale 2, `counting: {score_thr: 0.3080451190471649, identity_thr:
  0.45, max_rows: 2, neighbour_vote: report, pair_rule: report, shelf_hint: true, suppress_*}`. The
  `score_thr` here is the **run's** val-selected operating point from `val_manifest.json`, *not* the
  0.90 that won on human validation counts — the known deviation in §8.9.
- `detector/checkpoint_best_total.pth` — the run's best weights (class-agnostic), 133,802,771 bytes,
  sha256 `342b964c…`.
- `classes.json` — the 19 catalog ids + `cartridge`/`unknown` head-index map (1,431 bytes).
- `manifest.json` — provenance: run id, git sha (`7296a98…`, dirty), thresholds, reader id, the run's
  **model-label** metrics block (explicitly caveated `NOT human GT`), and a SHA-256 for every file.

Verify integrity any time (also runs automatically on load):

```bash
uv run python -m freestyle.artifacts verify rfdetr-final-v2
# OK: rfdetr-final-v2 (3 files verified)
```

The pre-GT kNN bundles worked differently: `freestyle.artifacts export` copied a run's `gallery.npz` or
rebuilt one from the run's train-split, known-class crop embeddings (`embeddings.npz`) using the same
embedder, and rejected an `--identifier` model that disagreed with the cached vectors (the vectors would
not be comparable). None of that applies to the shipped bundle — passing `--identifier
'{"type":"embedding_knn",...}'` today would just re-create the weaker 0.517 reader.

## 6. GPU vs CPU

| | detector (RF-DETR-M, tile 1280) | reader | notes |
|---|---|---|---|
| **GPU** (RTX 5090, k8s Job) | 0.223 s/img test, 0.309 s/img val (`runs/S2_c2_tile1280-db269936/meta.json`) | n/a (API) | only place GPU work may run |
| **CPU** (this host, measured) | localize **1.28–3.33 s/img** + dedup 0.005–0.62 s | **79–83 s per stocked photo** (Inkling, one call/crop) | `runs/cli_final_demo/counts.json` |

Measured on this host (CPU only, no GPU touched):

- **Wall time splits overwhelmingly into identification.** Across the three demo frames: localize 6.1 s,
  dedup 1.1 s, count 0.01 s, **identify 178.3 s** — about 94 % of the runtime. Per crop: 3.6 s
  (`frame.jpg`, 23 crops) to 5.5 s (sparse frame). A dense photo costs ~80 s.
- **The shipped reader is slower than that demo.** `rfdetr-final-v2` names Flash Next, measured at
  **9.1 s mean / 6.3 s median per crop** vs Inkling's 3.5 s / 2.8 s
  (`runs/reader_compare/results.json`), so expect roughly **3 min per stocked photo**. The trade bought
  0.586 → 0.655 identity accuracy against human answers (§8.9).
- **Process memory** was measured only for the kNN v0 demo: **1.73 GB** peak RSS, 67.18 s for three
  frames including ~15 s model load (`runs/bundle_demo/count.err`). The final bundle loads the same
  RF-DETR Medium weights in float32 but *drops* the SigLIP2 embedder, so its local floor is lower and
  the remaining cost is network. No peak-RSS figure exists for it.
- On a GPU the localize stage collapses to ~0.22 s and the reader stays an API call — the reader is the
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

## What the overlays actually show (honest read, against a human)

On a well-lit, well-framed, **full** cabinet photo the detector finds cartridges on both shelves and
shelves them into two rows; on this data the top shelf is the reliable one (human-GT top-shelf count MAE
**0.70** vs bottom **1.40**, and at the scored threshold it under-counts the two stocked frames rather
than doubling them: 14 boxes where a human counted 22, 16 where a human counted 28). Identification is
good on the brands the auto-labelled gallery and the reader both see often — NNS, Fanta, Sprite, Hi-C,
AA — and falls off on the red/silver bottom shelf, where the reader answers `unknown` or, worse with
Flash Next, calls `ingredient-cc` a NNS cartridge (3 crops). The Part 1 / Part 2 split is effectively
unavailable: the best reader gets the printed numeral right on **1 of 11** crops a human named, because
the numerals are not legible at this camera's resolution and blur, not because the model is weak.

The failure that matters operationally is the **door-swing frame**: 8 of the 10 test photos a human
answered showed no usable cabinet at all (person, blur, angle), and at the shipped `score_thr 0.308` the
bundle still put 62 boxes on such frames. `neighbour_vote` and `gap` flags fire on suspicious rows so a
human reviews them rather than trusting the count — keep that habit. Every count you see from this CLI
was chosen on **two** photographs of **one** cabinet with **one** stocking; `coca-cola-part1` and both
`coke-zero` SKUs never appear at all. Treat per-class numbers as indicative, and re-read
[`REPORT.md`](REPORT.md) §11 before acting on one.
