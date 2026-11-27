# How to run the Freestyle counting CLI

End-to-end: a cabinet photo goes in, cartridges come out **located → identified →
counted**, plus an annotated overlay. The counting model is a frozen *bundle* under
`artifacts/<name>/` (gitignored). This document is written against the
`rfdetr-m-1280-knn-v0` bundle and was produced from a real CPU run on this host.

> Everything here runs on CPU. A trained bundle's detector is happiest on a GPU, but the
> full pipeline works on CPU too (see *GPU vs CPU* at the bottom for measured timings).

## 1. Install

The CLI needs the optional `train` extra (torch + transformers + RF-DETR):

```bash
uv sync --extra train            # base + torch/torchvision/transformers/timm/...
# RF-DETR is not pinned in pyproject (the k8s training images pin it separately):
uv pip install "rfdetr==1.12.1"
```

The `--json`/overlay path needs no GPU libraries beyond torch. The embedder weights
(`google/siglip2-base-patch16-224`) are pulled from Hugging Face on first use into the
local HF cache.

## 2. List bundles

```bash
uv run python -m freestyle.cli bundles
```

```
Bundle                   Experiment                     Embedder                   Thresholds
------------------------ ------------------------------ -------------------------- --------------------
rfdetr-m-1280-knn-v0     S2_c2_tile1280                 google/siglip2-base-patch16-224 score=0.3080451190471649, id=0.5
```

`bundles --json` gives the same rows as JSON, including integrity status and the
metrics summary (always labelled **"vs model labels, NOT GT"** — see §6).

## 3. Count cartridges in one or more frames

```bash
uv run python -m freestyle.cli count \
    data/raw/freestyle-vision-data/images/local/downloads/frame.jpg \
    data/raw/freestyle-vision-data/images/island-cabinet-01/50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg \
    --bundle rfdetr-m-1280-knn-v0 \
    --device cpu \
    --out runs/bundle_demo/frames/ \
    --json
```

Each frame gets an overlay PNG (boxes + `class S<shelf>` + unknowns) under `--out`,
and `--out/counts.json` collects the machine-readable result. Real `--json` output for
`frame.jpg` (CPU):

```json
{
  "bundle": "rfdetr-m-1280-knn-v0",
  "device": "cpu",
  "images": [
    {
      "image": "data/raw/freestyle-vision-data/images/local/downloads/frame.jpg",
      "counts": { "NNS": 8, "Hi-C": 4, "Fanta": 3, "Vanilla Flavor": 2 },
      "unknown": 9,
      "flags": [],
      "timings_ms": { "localize": 5516.4, "identify": 9363.8, "count": 635.7 },
      "overlay": "runs/bundle_demo/frames/frame.png"
    }
  ]
}
```

Counts use catalog **display names** (resolved from `data/raw/.../catalog_items.json`,
override with `--catalog`). Anything the identifier could not place above its
`identity_thr` (or that the detector only called generically `cartridge`) is reported in
`unknown`, **not** guessed into a class. Drop `--json` for a human-readable table.

## 4. Fuse several frames of one door opening

A single door opening is captured as a short burst of near-duplicate frames. Counting
each frame and **summing** would multiply the same cartridge by the frame count, so use
`fuse`, which combines the per-frame count vectors with a cross-frame rule
(`max` (default), `median`, `weighted_mode`):

```bash
uv run python -m freestyle.cli fuse \
    data/raw/freestyle-vision-data/images/island-cabinet-01/bb2c251e-38ec-4952-a1d2-250436c7dbe5/000.jpg \
    data/raw/freestyle-vision-data/images/island-cabinet-01/bb2c251e-38ec-4952-a1d2-250436c7dbe5/001.jpg \
    --bundle rfdetr-m-1280-knn-v0 --device cpu --method max --json
```

Real output (`max`):

```json
{
  "method": "max",
  "fused_counts": { "Fanta": 1, "NNS": 7 },
  "fused_unknown": 0,
  "frames": [ ...per-frame records identical in shape to `count`... ]
}
```

Here frame `000` located 8 faces and frame `001` was a near-blank/moved frame that found
none — `max` keeps the good frame's inventory instead of averaging it down. `weighted_mode`
picks the most-supported observed count across frames; `median` under-counts when most
frames show only a partial view. `--weights` (one float per frame) tunes `weighted_mode`.

## 5. Rebuilding the bundle from an experiment run

A bundle freezes one training run. `rfdetr-m-1280-knn-v0` was exported from the sweep run
`runs/S2_c2_tile1280-db269936/` — the best **class-agnostic tile-1280** detector
(RF-DETR-Medium). That run evaluated with `identifier: none` (so its own metrics show 0
identification), but it cached every labelled crop embedding in `embeddings.npz`. To turn
it into an identifying bundle, export it and *attach* an `embedding_knn` identifier; the
gallery is rebuilt from the run's train-split, known-class crops (VLM `bootstrap_v2`
identities; `unknown`/`cartridge` rows dropped) using the **same** embedder that produced
`embeddings.npz`:

```bash
uv run python -m freestyle.artifacts export \
    --run runs/S2_c2_tile1280-db269936 \
    --name rfdetr-m-1280-knn-v0 \
    --overwrite \
    --identifier '{"type":"embedding_knn","model":"google/siglip2-base-patch16-224","k":5}' \
    --counting   '{"neighbour_vote":"report"}'
```

What lands in `artifacts/rfdetr-m-1280-knn-v0/`:

- `pipeline.yaml` — frozen config: tiled RF-DETR detector (tile 1280, overlap 0.25),
  `embedding_knn` identifier, `counting: {score_thr: 0.308, identity_thr: 0.5,
  neighbour_vote: report, max_rows: 4}`. `score_thr` is the run's **val-selected**
  operating point; `identity_thr`/shelf `max_rows` and the pair/neighbour rule ride along.
- `detector/checkpoint_best_total.pth` — the run's best weights (class-agnostic).
- `gallery.npz` — 139 train-only crop embeddings, 10 catalog classes
  (nns-sweetener 60, flavor-vanilla 19, fanta-orange 17, hi-c 13, flavor-lemon 11,
  sprite 10, flavor-orange 5, ingredient-cc 2, ingredient-aa 1, coca-cola-part2 1).
- `manifest.json` — provenance: run id, git sha (`7296a98…`, dirty), thresholds,
  embedder id and a SHA-256 checksum for every file.

Verify integrity any time (also run automatically on load):

```bash
uv run python -m freestyle.artifacts verify rfdetr-m-1280-knn-v0
# OK: rfdetr-m-1280-knn-v0 (4 files verified)
```

If the source run already shipped a `gallery.npz`, export copies it; the `embeddings.npz`
rebuild only kicks in when no gallery is present. Passing an `--identifier` model that
disagrees with the run's cached embeddings is rejected outright (the vectors would not be
comparable — you would have to re-embed, not relabel).

## 6. GPU vs CPU

| | detector (RF-DETR-M, tile 1280) | embedder (siglip2-224, CPU) | notes |
|---|---|---|---|
| **GPU** (RTX 5090, k8s Job) | 0.22–0.31 s/image (run `meta.json`) | n/a (gallery built on GPU) | only place GPU work may run |
| **CPU** (this host, this demo) | 2.0–5.5 s/image | ~0.3 s/crop | whole pipeline, incl. KNN |

Measured on this host (CPU only, no GPU touched):

- **Process peak RAM**: ~**1.7 GB** for the whole `count` invocation (RF-DETR-Medium +
  SigLIP2 on CPU, float32).
- **Wall time**: ~**67 s for 3 frames** including ~15 s of one-time model load → roughly
  **10–15 s per well-filled frame**, dominated by per-crop identification (the identifier
  runs the embedder once per detection). Sparse frames are much cheaper (a 3-detection
  frame identified in 0.8 s).
- Per-stage `timings_ms` are emitted per image in the JSON, so you can see localize vs
  identify vs count split. On a GPU the localize stage collapses to ~0.2 s and the
  identifier is a few ms per crop.

GPU experiments run only as Kubernetes Jobs on the home cluster (see `k8s/README.md`);
this CPU path exists so the *full* product loop — locate, identify, count, overlay —
can be demonstrated and smoke-tested without holding a cluster card. Use `--device cuda`
when a CUDA device is present and `--device cpu` (or leave `auto`) otherwise.

## What the overlays actually show (honest read)

On well-lit, well-framed cabinet photos the **detector** finds cartridge faces on both
shelves and shelves them into rows; on the **blue NNS** faces the identifier is reliable.
It is **weaker** on red/orange faces and rare classes (the gallery is dominated by
nns-sweetener), and it degrades in dim light and heavy occlusion (a person standing in
front of the cabinet → very low recall, only a few faces located). `neighbour_vote` and
`gap` flags fire on the suspicious rows so a human reviews them rather than trusting the
count. Counts are **against VLM bootstrap labels, not human ground truth**, so treat the
per-class numbers as indicative, not verified.
