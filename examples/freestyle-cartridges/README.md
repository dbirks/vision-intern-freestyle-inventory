# Freestyle cartridge recognition

> Detection + identification + counting of Coca-Cola Freestyle syrup cartridges from the
> in-door fisheye camera, built on Vision Intern's label → judge → train path and extended
> where the core can't reach.
>
> **Status (2026-11-27): everything that does not need human ground truth is done.**
> 14 GPU runs collected (whole-cartridge + front-face detector families), two frozen CPU
> bundles, a working counting CLI, a phone GT quiz, and a scoring pass that will rank every
> run the moment human answers exist. **Human GT is still pending** — 6 of 333 queued quiz
> questions answered — so *no number in this repo is verified against a human*. Contract:
> [`docs/freestyle/ARCHITECTURE.md`](../../docs/freestyle/ARCHITECTURE.md).
> Narrative: [`docs/freestyle/REPORT.md`](../../docs/freestyle/REPORT.md). Costs:
> [`docs/freestyle/RESOURCES.md`](../../docs/freestyle/RESOURCES.md). Failure modes:
> [`docs/freestyle/FAILURES.md`](../../docs/freestyle/FAILURES.md). What the core provides and
> where it stops: [`docs/freestyle/VISION_INTERN_MAP.md`](../../docs/freestyle/VISION_INTERN_MAP.md).

## Goal
Locate **every cartridge face**, identify it against the 18 active catalog ids, and count
per id — from one fisheye photograph per door-opening. Two shelves of spine-out boxes,
~10-20 instances per frame, faces ~60-150 px tall in a 1600×1200 / 2048×1536 frame, strong
wide-angle distortion, identity carried by small printed text + brand colour. This is the
dense/small/many-instance shape the toolkit still labels experimental, so the pipeline is
stage-decomposed rather than asked of one model: class-agnostic **localise** → full-res
**identify** → **count**, judged in between.

## Where it stands (measured, vs model labels — NOT human GT)

| Question | Answer so far | Evidence |
|---|---|---|
| Trained detector or zero-shot open-vocab? | **Trained.** RF-DETR 0.614–0.718 AP50 vs OWLv2 0.181 / GroundingDINO 0.232; zero-shot merges adjacent cartridges and the 2-high NNS stack | `docs/freestyle/RESULTS.md`, `runs/*/metrics.json` |
| Tile size? | **1280 px** (S2) beat 768 and 1024 on MAE (3.583) at the lowest cost (145 s train, 0.223 s/img) | `runs/S2_c2_tile1280-db269936/` |
| Whole cartridge or just the printed front face? | **Undecided — the quiz decides.** Faces overlap less (0.119 vs 0.201 neighbour IoU) and unlock Part numerals (59 vs 2 read), and G2 posts the lowest count MAE (3.333) *in its own label space*, so the rows are not comparable | `runs/faces_v2/summary.md`, `runs/G*/metrics.json` |
| How is identity done? | **Hybrid:** Inkling 2 Mini names the brand family, class-balanced SigLIP2 kNN refines inside it. Plain kNN said "22 NNS" on a frame that holds ~7 | `artifacts/rfdetr-m-1280-v1/pipeline.yaml`, `runs/bundle_demo*/counts.json` |
| Can Part 1 / Part 2 be read? | Barely: 2 of 123 whole crops, 59 of 597 faces (unverified). The pair rule (CART 1 left of CART 2, from public placement maps) is a **hypothesis** pending GT | `runs/bootstrap_v2/summary.md`, `docs/freestyle/research/placement_maps.md` |
| What is the working architecture? | Bundle `rfdetr-m-1280-v1` (S2 + duplicate suppression + 2-shelf counting + hybrid identifier), **provisional** | `docs/freestyle/REPORT.md` §8.9 |

## Data
- **Source:** `data/raw/freestyle-vision-data/` — read-only, **private**, original encoders
  (54 server-verified photographs + 15 workstation copies; 96.1 % of the declared server frames
  were lost to retention). 1600×1200 UXGA / 2048×1536 QXGA.
- **Preprocessing:** no re-encode of raw; derived artefacts only under `data/derived/`
  (tiles via `freestyle/train/export_tiles.py`, crops on demand).
- **Splits:** `data/splits.json` built by `freestyle/make_splits.py` — the independent unit is the
  **door-event group**, which Vision Intern's splitter cannot express. 11 train / 6 val / 11 test
  groups (22 / 6 / 12 images), 29 `no_cabinet`+`occluded` negatives excluded. Frozen in
  `data/splits.frozen.json`; `--check` verifies 0 near-duplicates touch `test`.
- **Model labels vs GT:** `data/labels/*.json` (bootstrap_v1/v2, faces_v1/v2) are model output and
  **never** GT. Only `data/gt/`, produced by a human in the quiz or the review app, is GT.
- **Boundary:** images go only to the LLM Collective and our k8s cluster; nothing under `data/`,
  `runs/`, `artifacts/` is committed; **no Hugging Face pushes at all**.

## Classes
18 `active: true` ids from `data/raw/freestyle-vision-data/data/catalog_items.json`, plus
`cartridge` (class-agnostic super-class; `cartridge_face` for the G family, which the shared
export path also names `cartridge`) and `unknown` (identity not determinable). Single source of
truth is `freestyle/catalog.py` — do not paste the list into a labeller call. Feed
`label_aliases` from the catalog's `aliases` field: the resolver is tolerant to case/punctuation
but **not** fuzzy, so `coke zero part1` will not reach `coke-zero-part1` without an alias.

## Models per role
Collective only (`https://llm-collective.e-gineering.com/v1`); the key is `LLMC_API_KEY`, read at
runtime by `freestyle/collective.py` and never written to a file. Retry policy: 429/5xx/timeouts
with exponential backoff + jitter, `retry_budget_s=600`, ≤2 in-flight requests per process.

| Role | Model | Where | Notes |
|---|---|---|---|
| Labeller (stage A + front faces) | `david-birks/inkling-2-mini` | Collective | 0.4–5.4 s/call, reliable; obeys whichever coordinate convention you prompt |
| Judge (crop sheets) | `qwen3-8-flash-next` | Collective | thinking model: needs `max_tokens=4096`, intermittently 429-throttled (use the `david-birks/` prefix for stability) |
| Identifier (family) | `david-birks/inkling-2-mini` (`two_stage`) | Collective | family from logo/colour; part numeral only off a 3× end-crop |
| Embedder (identifier kNN) | `google/siglip2-base-patch16-224` | local / k8s | also `facebook/dinov2-base` in the B/C/D runs |
| Trained detector | `RFDETR{Small,Medium,Large}` @ 512/576 px, tiled | **k8s Job only** | never `workflows.train_rfdetr` (hardcodes large @ 704², drops negatives, flips/hues the image) |
| Text-only drafting | `qwen3-8-27b` | — | **not** a vision judge: the endpoint returns 404 for it |

**Coordinate trap:** `qwen3-8-flash-next` ignores pixel-coordinate instructions and always returns
0–1000 normalized values; `max_side` is a request-size knob, not a coordinate-frame knob. All
grounding code prompts and parses normalized (`freestyle/vlm_ground.py`).

## Commands
Each block is a stage of the real pipeline; all flags below were checked with `--help`.

```bash
# 0 — collective probe (confirms ids/params before betting a run on them)
uv run python -m freestyle.probe_collective list-models|tiny|detect|crop|identify

# 1 — corpus audit (pixel stats, pHash near-duplicates, VLM scene triage) -> data/derived/audit/
uv run python -m freestyle.audit [--no-vlm] [--limit N] [--only IMAGE_OR_GROUP ...]

# 2 — group-safe frozen splits
uv run python -m freestyle.make_splits --test-frac 0.4 --val-frac 0.2 --seed 0 --out data/splits.json
uv run python -m freestyle.make_splits --check

# 3 — VLM bootstrap proposals (whole cartridges; never written under data/gt/)
uv run python -m freestyle.bootstrap_labels --config configs/bootstrap/default.yaml \
    --images val --out data/labels/bootstrap_val.json
#     two-stage re-identification (family, then part numeral) over an existing label file:
uv run python -m freestyle.reidentify --in data/labels/bootstrap_v1.json \
    --out data/labels/bootstrap_v2.json --mode two_stage

# 3b — front-face labels (experiment family G): tune a prompt, run it, clean the result
uv run python -m freestyle.faces tune  --image local/downloads/frame.jpg --tile 1024
uv run python -m freestyle.faces run   --config configs/bootstrap/faces.yaml \
    --images runs/bootstrap_v1/images.txt --out data/labels/faces_v1.json \
    --timing runs/faces_v1/timing.json          # 29 cabinet frames, 197 calls, ~28 min
uv run python -m freestyle.faces clean            # faces_v1 -> faces_v2 (no new VLM calls)
uv run python -m freestyle.faces summary

# 4 — human GT (the only gate): phone quiz, or full-res drag-box review app
uv run python -m freestyle.quiz.build_queue --proposals data/labels/bootstrap_v2.json \
    --splits data/splits.json --splits-order test,val --out data/gt/quiz_queue.json
uv run python -m freestyle.quiz.server --queue data/gt/quiz_queue.json \
    --answers data/gt/quiz_answers.jsonl --host <tailnet-ip> --port 8766
uv run python -m freestyle.quiz.to_coco --queue data/gt/quiz_queue.json \
    --answers data/gt/quiz_answers.jsonl --out data/gt/test.json --split test
uv run python -m freestyle.review.server --proposals data/labels/bootstrap_v2.json \
    --images test --out data/gt/test.json --host 127.0.0.1 --port 8765   # box fixes

# 5 — GPU experiments as k8s Jobs on the home cluster (never on this host)
k8s/stage-data.sh --list
uv run python -m freestyle.experiments.launch configs/experiments/S2_c2_tile1280.yaml --dry-run
uv run python -m freestyle.experiments.launch configs/experiments/S2_c2_tile1280.yaml
uv run python -m freestyle.experiments.launch --status
uv run python -m freestyle.train.recover_mapping runs/<run>   # head-index -> category map

# 6 — score one run, decompose its failures, rebuild the generated table
uv run python -m freestyle.eval.evaluate --gt <labels-or-gt.json> --pred runs/<run>/test/predictions.json \
    --select-thr-on <val-gt> runs/<run>/val/predictions.json --split-map data/splits.json \
    --out runs/<run>/metrics.json --experiment <exp_id>
uv run python -m freestyle.eval.failures --gt <gt.json> --pred runs/<run>/test/predictions.json \
    --images-root data/raw/freestyle-vision-data/images --score-thr <val_thr> --out runs/<run>/failures
uv run python -m freestyle.eval.report --runs-dir runs --out docs/freestyle/RESULTS.md

# 7 — frozen bundle export + CPU inference (needs the `train` extra; see HOW_TO_RUN.md)
uv sync --extra train && uv pip install "rfdetr==1.12.1"
uv run python -m freestyle.cli bundles
uv run python -m freestyle.cli count <frame.jpg> --bundle rfdetr-m-1280-v1 --device cpu \
    --out runs/cli_preview --json                 # v1 also needs LLMC_API_KEY at inference
uv run python -m freestyle.cli fuse <f1.jpg> <f2.jpg> --bundle rfdetr-m-1280-v1 --method max --json

# 8 — the whole post-GT sequence, in order, with a hard sufficiency gate
scripts/finalize_after_gt.sh --dry-run     # validate + print everything it would run
scripts/finalize_after_gt.sh               # quiz → to_coco → score_all → report → export → demo
```

GPU work runs **only** as Kubernetes Jobs (context `admin@home`, `k8s/README.md`); CPU/VLM-API
stages may run on this host. `runs/`, `data/` and `artifacts/` stay gitignored.

## Outputs
All local — no Hub repos, no public registry.
- Proposals: `data/labels/{bootstrap_v1,bootstrap_v2,faces_v1,faces_v2}.json` (model-generated, never GT)
- GT: `data/gt/<split>.json` + `<split>_counts.json` (human; quiz or review app)
- Runs: `runs/<exp_id>-<hash>/{resolved.yaml,meta.json,provenance.json,env.txt,val/,test/,metrics.json,failures/}`
  (`failures/{failures.json,index.html,galleries/,overlays/}` are private images — paths only in docs)
- Models: `runs/<exp>/training/` (best weights kept, intermediates pruned; full set on the PVC)
- Bundles: `artifacts/rfdetr-m-1280-knn-v0` (plain kNN, superseded) and
  `artifacts/rfdetr-m-1280-v1` (hybrid, the working candidate), each with `pipeline.yaml`,
  `manifest.json` (SHA-256 per file) and `gallery.npz`
- Tables: `docs/freestyle/RESULTS.md` (generated by `freestyle/eval/report.py`, vs model labels)
  and `docs/freestyle/RESULTS_GT.md` (generated by `score_all`, **not yet produced**)

## Notes / gotchas (the ones that actually bit)
- **Nothing the VLM sees is full-resolution unless you say so.** `max_dimension=1280` and JPEG q85
  are the core defaults; pass `max_dimension=None, image_format="png"`. Judge crop sheets thumbnail
  to 256 px — fine for "is that a cartridge", useless for reading a spine.
- **The judge can only delete.** Stage-A recall bounds every counting number; measure it first.
- **Token ceilings.** `judge_labels` defaults to `max_tokens=1024` unless the model name contains
  `lfm3-vl` (≈20-25 verdicts/image, silently truncating); thinking models return **empty text** at
  `max_tokens≤64` because `<think>` eats the budget. Always pass `max_tokens` + `judge_batch_size`.
- **RF-DETR class indices are zero-based over sorted training categories.** Training now writes an
  explicit head-index→category mapping (`freestyle/train/recover_mapping.py`) and unknown indices
  fail loudly. For a **single-category** checkpoint the head is `(2,256)` — 1 foreground + 1
  background slot — and treating that background slot as a class produced
  `Unknown RF-DETR detector class index: 1`, which is why bundle v1 is stuck at `score_thr=0.308`
  although validation prefers 0.10–0.15 once duplicate suppression is on.
- **Copy-paste augmentation doubles train time** for no clear test win (S3 427 s vs C2 289 s;
  S7 328 s vs S2 145 s) and buys the most `FP:duplicate` of any run.
- **Peak VRAM does not track model size:** Small @ 512 px peaks at 6.15 GiB, every Medium @ 576 at
  4.09 GiB, `G2` (Small, 1024 px tiles) at 6.89 GiB. Resolution and batch size drive it.
- **The GT quiz answers whole-cartridge boxes.** If the front-face family wins on counts,
  face-level box GT does not exist yet and must be drawn in the review app over
  `data/labels/faces_v2.json` before any face AP/ID number can be called human-verified.
- **Review #5 found `pair=on` was a 100 % no-op** for every non-VLM identifier before `5ff55c2`.
  Anything that claims to have measured the pair rule before that commit is measuring nothing.

## Open / next
1. Answer the quiz (20 min on a phone) — it decides whole-box vs face, every identity method, the
   pair rule, and all four bundle thresholds in one scoring pass: `scripts/finalize_after_gt.sh`.
2. Re-tune `score_thr`/`identity_thr`/`family_threshold` on `data/gt/val.json`, then re-export the
   winning bundle at a lower threshold once the phantom-index fix is in the exported checkpoint.
3. Face family needs negative-aware training (G1's dominant error is 56 `FP:background`) or a
   minimum-face-size gate; 53 residual duplicate face pairs await a human.
4. Unrun configs: `S4_c2_long`, `A1`–`A3` (pure VLM localizers), `E1` (shelf + kNN),
   `F1` (shelf + VLM id). `B2_sam3_siglip2` is blocked by HF gated access (403).
5. Cross-cabinet and stocking-change generalization is completely unmeasured: every frame is the
   same cabinet, so slot memorization cannot be ruled out.

**Index row:** add `freestyle-cartridges` to the `examples/README.md` index once human-GT numbers
exist (that file is outside this task's ownership).
