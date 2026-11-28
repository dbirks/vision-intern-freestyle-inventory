# Freestyle cartridge recognition

> Detection + identification + counting of Coca-Cola Freestyle syrup cartridges from the
> in-door fisheye camera, built on Vision Intern's label → judge → train path and extended
> where the core can't reach.
>
> **Status (2026-11-27): finished and scored against human ground truth.** 13 GPU runs were collected
> (whole-cartridge + front-face detector families), the phone quiz got 128 answers, and every run was
> scored on them ([`docs/freestyle/RESULTS_GT.md`](../../docs/freestyle/RESULTS_GT.md)). One architecture
> won: **`S2_c2_tile1280` (RF-DETR Medium, 1280 px tiles) + Qwen 3.8 Flash Next as the two-stage reader**,
> shipped as bundle **`rfdetr-final-v2`**. On the 10 human-scored test photos: **2.1 cartridges off per
> photo** (95 % CI 0.0–5.1), top shelf 0.70 / bottom 1.40, AP50 0.875, and the right product named on
> 19 of the 29 cartridges a human could also name (0.655). Read
> [`docs/freestyle/REPORT.md`](../../docs/freestyle/REPORT.md) §11 before repeating any of it: 8 of those
> 10 photos are frames on which a human could not see a usable cabinet at all, so the count rests on
> **two** stocked photographs. Contract:
> [`docs/freestyle/ARCHITECTURE.md`](../../docs/freestyle/ARCHITECTURE.md). Costs:
> [`docs/freestyle/RESOURCES.md`](../../docs/freestyle/RESOURCES.md). Failure modes (vs model labels):
> [`docs/freestyle/FAILURES.md`](../../docs/freestyle/FAILURES.md). Crop readers scored on human answers:
> [`docs/freestyle/READER_COMPARE.md`](../../docs/freestyle/READER_COMPARE.md). What the core provides and
> where it stops: [`docs/freestyle/VISION_INTERN_MAP.md`](../../docs/freestyle/VISION_INTERN_MAP.md).

## Goal
Locate **every cartridge face**, identify it against the 18 active catalog ids, and count
per id — from one fisheye photograph per door-opening. Two shelves of spine-out boxes,
~10-20 instances per frame, faces ~60-150 px tall in a 1600×1200 / 2048×1536 frame, strong
wide-angle distortion, identity carried by small printed text + brand colour. This is the
dense/small/many-instance shape the toolkit still labels experimental, so the pipeline is
stage-decomposed rather than asked of one model: class-agnostic **localise** → full-res
**identify** → **count**, judged in between.

## Where it stands — scored against human answers, then vs model labels

Every row below is answered on the quiz GT (`docs/freestyle/RESULTS_GT.md`, 10 test images / 9
door-openings, 1,000-replicate group bootstrap) with the older model-label number kept for contrast.

| Question | Answer on human GT | Answer vs model labels | Evidence |
|---|---|---|---|
| Trained detector or zero-shot open-vocab? | **Trained, decisively.** Count MAE 2.10 (`S2`) vs 5.60–5.70 for `S6`/`B1`/`B3`; AP50 0.875 vs 0.147–0.159. `B1`'s best prediction score on the whole test set is 0.427, so zero-shot cannot even be thresholded into usefulness | RF-DETR 0.614–0.718 AP50 vs OWLv2 0.181 / GroundingDINO 0.232 | `docs/freestyle/RESULTS_GT.md`, `docs/freestyle/RESULTS.md` |
| Tile size? | **1280 px** — `S2` 2.10 vs `S1` (768 px) 4.60 vs `C2` (1024 px) 4.90 count MAE. The tile sweep holds on both answer keys | Same ordering (MAE 3.583 / 6.000 / 4.417) | `docs/freestyle/RESULTS_GT.md` rows `S1/S2/C2`, `runs/S2_c2_tile1280-db269936/` |
| Copy-paste augmentation? | **No — it hurt.** `S3` 4.50 and `S7` 4.40 vs `S2` 2.10, at 2–3× the training time | It *helped*: `S3` had the best model-label AP50 (0.718) | `docs/freestyle/RESULTS_GT.md`, `runs/S3_*`, `runs/S7_*` |
| Whole cartridge or just the printed front face? | **Whole cartridge.** `S2` 2.10 [0.00, 5.11] vs face detector `G2` 2.70 [0.40, 5.89]; and against whole-box GT the face runs score AP50 0.379 / 0.318 vs 0.875. The intervals overlap, so it is a lean — but the face family's own promise (a better count on the leaning bottom row) did not happen: bottom MAE 1.60 vs 1.40 | Undecidable — each family was scored against its own model-made answer key | `docs/freestyle/RESULTS_GT.md` rows `G1/G2/S2`, `docs/freestyle/REPORT.md` §8.2.4 |
| How is identity done? | **A VLM reading the crop, not embedding matching.** Same 29 crops on the winner: Flash Next **0.655** > Inkling 0.586 > SigLIP2 kNN / linear probe on the 108-crop model-labelled gallery **0.517**. On the face runs kNN collapses to 0.048. Adding human val crops to the gallery changes nothing — no usable val crops were answered | Hybrid (Inkling family + class-balanced kNN refinement), picked by an unverified visual read | `docs/freestyle/RESULTS_GT.md` §8.2.2 of REPORT, `docs/freestyle/READER_COMPARE.md` |
| Which VLM reads the crops? | **Qwen 3.8 Flash Next, two-stage + shelf hint.** 39 human-labelled test crops: top-1 0.56 (CI 0.41–0.71) vs Inkling 0.49 (0.34–0.64); brand-family 0.69 vs 0.67; abstains 18 % vs 44 % — at 9.1 s vs 3.5 s per crop. Overlapping intervals, so a lean, taken because it is consistent across 9 localizers | Inkling 2 Mini (faster, and it was the labeller) | `docs/freestyle/READER_COMPARE.md` |
| Can Part 1 / Part 2 be read? | **No.** Against a human the best reader gets the numeral on **1 of 11** crops (0.09), Inkling on **0 of 11**; the full Flash Next re-pass read a digit on **3 of 47** part-capable crops (6.4 %) vs 2 of 123 for Inkling on whole crops. The 59 numerals claimed off faces were never human-checked | 2 of 123 whole crops (Inkling), 59 of 597 faces (unverified) | `docs/freestyle/READER_COMPARE.md`, `runs/bootstrap_v3_flash/summary.md`, `runs/bootstrap_v2/summary.md` |
| Does the pair rule work? | **Direction confirmed, sample too small to prove it.** On 92 pair-comparable rows it never lowers identity, raises it on 4 (winner 0.655 → 0.690 with `vlm_flash`) and changes no count anywhere. Ships in `report` mode | Untestable — review #5 found `pair=on` was a 100 % no-op before `5ff55c2` | `docs/freestyle/REPORT.md` §10.2, `docs/freestyle/results_gt.json` |
| What is the architecture? | **Bundle `rfdetr-final-v2`** = `S2` detector + duplicate suppression + 2-shelf counting + `shelf_hint` + Flash Next two-stage reader, `pair_rule: report`. **Deviation to fix:** it ships `score_thr 0.308` while the scored winner used the val-selected `0.90` — at 0.308 the same weights score 6.6 count MAE | Bundle `rfdetr-m-1280-v1` (provisional) | `artifacts/rfdetr-final-v2/pipeline.yaml`, `docs/freestyle/REPORT.md` §8.9 |
| Is it good enough? | **Not for a real inventory.** Human-filtered model boxes mean AP50 is not independent GT; exact inventory (0.778) is 7 empty frames out of 9; the per-photo error on the only two photos that really show the cabinet is **10.0 [8.0, 12.0]** cartridges; and identification costs one API call per cartridge (~80 s/photo, ~3 min with Flash Next) | — | `docs/freestyle/REPORT.md` §11, `docs/freestyle/results_gt.json` `sharp-full` subset |

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
| **Reader / identifier (shipped)** | **`qwen3-8-flash-next`** (`two_stage` + shelf hint) | Collective | **Chosen on human answers**: 0.655 identity accuracy on the winner's 29 matched crops vs 0.586 for Inkling; 0.56 vs 0.49 top-1 on 39 human-labelled test crops; 9.1 s mean / 6.3 s median per crop; thinking model — needs `max_tokens=8192` |
| Labeller (stage A + front faces) | `david-birks/inkling-2-mini` | Collective | 0.4–5.4 s/call, reliable; obeys whichever coordinate convention you prompt; 2.8 s median per crop as a reader — the cheap fallback |
| Judge (crop sheets) | `qwen3-8-flash-next` | Collective | needs `max_tokens=4096`, intermittently 429-throttled (use the `david-birks/` prefix for stability) |
| Embedder (superseded kNN identifier) | `google/siglip2-base-patch16-224` | local / k8s | also `facebook/dinov2-base` in the B/C/D runs. Scored **0.517** (kNN) / **0.048** (face gallery) against a human — the reason the shipped bundle carries no `gallery.npz` |
| Trained detector | `RFDETR{Small,Medium,Large}` @ 512/576 px, tiled | **k8s Job only** | never `workflows.train_rfdetr` (hardcodes large @ 704², drops negatives, flips/hues the image) |
| Text-only drafting | `qwen3-8-27b` | — | **not** a vision judge: the endpoint returns 404 for it |

**Contact sheets are not a reader.** On human answers, sheet-mode top-1 is 0.21 (Flash Next) / **0.00**
(Inkling) with 59 % / 100 % abstention — one crop at a time, upscaled, is the only thing that works
(`docs/freestyle/READER_COMPARE.md`).

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
#     the active-learning queue actually used (180 questions / 28 images: 74 test eval,
#     46 val tune, 60 train gallery picks — the crops the identifier is unsure about)
uv run python -m freestyle.quiz.active --proposals data/labels/bootstrap_v2.json \
    --faces data/labels/faces_v2.json --splits data/splits.json \
    --answers data/gt/quiz_answers.jsonl --bundle rfdetr-m-1280-v1 \
    --out data/gt/quiz_queue.json          # --dry-run prints the counts, writes nothing
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

# 6 — score: one run, every run against human GT, the reader bake-off, and the generated tables
uv run python -m freestyle.eval.evaluate --gt <labels-or-gt.json> --pred runs/<run>/test/predictions.json \
    --select-thr-on <val-gt> runs/<run>/val/predictions.json --split-map data/splits.json \
    --out runs/<run>/metrics.json --experiment <exp_id>
uv run python -m freestyle.eval.failures --gt <gt.json> --pred runs/<run>/test/predictions.json \
    --images-root data/raw/freestyle-vision-data/images --score-thr <val_thr> --out runs/<run>/failures
uv run python -m freestyle.eval.report --runs-dir runs --out docs/freestyle/RESULTS.md
uv run python -m freestyle.experiments.score_all --queue data/gt/quiz_queue.json \
    --answers data/gt/quiz_answers.jsonl --proposals data/labels/bootstrap_v2.json \
    --proposals-flash data/labels/bootstrap_v3_flash.json --runs "runs/*" --bootstrap 1000 \
    --out docs/freestyle/RESULTS_GT.md
uv run python -m freestyle.experiments.reader_compare            # Flash Next vs Inkling on human crops
uv run python -m freestyle.reidentify --in data/labels/bootstrap_v1.json \
    --out data/labels/bootstrap_v2.json --mode two_stage          # Inkling identities (the labeller)
uv run python -m freestyle.reidentify --in data/labels/bootstrap_v2.json \
    --out data/labels/bootstrap_v3_flash.json --mode two_stage_shelf --profile reader_compare

# 7 — frozen bundle export + CPU inference (needs the `train` extra; see HOW_TO_RUN.md)
uv sync --extra train && uv pip install "rfdetr==1.12.1"
uv run python -m freestyle.cli bundles
uv run python -m freestyle.artifacts verify rfdetr-final-v2      # OK: rfdetr-final-v2 (3 files verified)
uv run python -m freestyle.cli count <frame.jpg> --bundle rfdetr-final-v2 --device cpu \
    --out runs/cli_final_demo --json     # needs LLMC_API_KEY: the identifier is one Collective call per crop
uv run python -m freestyle.cli fuse <f1.jpg> <f2.jpg> --bundle rfdetr-final-v2 --method max --json

# 8 — the whole post-GT sequence, in order, with a hard sufficiency gate
scripts/finalize_after_gt.sh --dry-run     # validate + print everything it would run
scripts/finalize_after_gt.sh --skip-quiz   # what produced RESULTS_GT.md + rfdetr-final-v2
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
- **Bundles:** `artifacts/rfdetr-final-v2` — **the human-GT-selected bundle** (`S2` weights + `vlm_crop`
  `qwen3-8-flash-next` `two_stage` + 2-shelf counting/suppression, `pair_rule: report`; 128 MB, 3 files,
  **no `gallery.npz`** because the identifier is a VLM call). Superseded pre-GT candidates kept for
  comparison: `rfdetr-m-1280-knn-v0` (plain kNN), `rfdetr-m-1280-v1` (hybrid), `rfdetr-m-1280-v2`
  (v1 knobs + a gallery refreshed from the 6 human train crops). Each has `pipeline.yaml`,
  `manifest.json` (SHA-256 per file) and, where applicable, `gallery.npz`.
- Tables: `docs/freestyle/RESULTS.md` (generated by `freestyle/eval/report.py`, **vs model labels**),
  `docs/freestyle/RESULTS_GT.md` + `results_gt.json` (generated by `freestyle.experiments.score_all`,
  **human GT** — 184 rows), and `docs/freestyle/READER_COMPARE.md` (generated by
  `freestyle.experiments.reader_compare`, human GT, per-crop)

## Notes / gotchas (the ones that actually bit)
- **Nothing the VLM sees is full-resolution unless you say so.** `max_dimension=1280` and JPEG q85
  are the core defaults; pass `max_dimension=None, image_format="png"`. Judge crop sheets thumbnail
  to 256 px — fine for "is that a cartridge", useless for reading a spine.
- **The judge can only delete, and the quiz inherits that.** Stage-A recall bounds every counting
  number, and the human GT proves it: `581c0d5e…/045.jpg` came back `counts_mismatch` (a human counted
  28 cartridges, only 22 answered crops exist) because the labeller merged boxes the quiz can only
  reject, never redraw. `S2` under-counts both stocked frames at its scored threshold (14 boxes on 22
  cartridges, 16 on 28). Measure stage-A recall before believing any count number.
- **Token ceilings.** `judge_labels` defaults to `max_tokens=1024` unless the model name contains
  `lfm3-vl` (≈20-25 verdicts/image, silently truncating); thinking models return **empty text** at
  `max_tokens≤64` because the reasoning tokens eat the budget. Always pass `max_tokens` +
  `judge_batch_size` (and 8192 for Flash Next as a reader).
- **RF-DETR class indices are zero-based over sorted training categories.** Training now writes an
  explicit head-index→category mapping (`freestyle/train/recover_mapping.py`) and unknown indices
  fail loudly. For a **single-category** checkpoint the head is `(2,256)` — 1 foreground + 1
  background slot — and treating that background slot as a class produced
  `Unknown RF-DETR detector class index: 1`, which is why bundle v1 was stuck at `score_thr=0.308`.
  **Still live in `rfdetr-final-v2`:** the human-val-selected threshold is `0.90`, the bundle ships
  `0.308`, and at the shipped value the same weights score **6.6** count MAE instead of 2.10, with 62
  boxes on the frames a human called empty (`docs/freestyle/REPORT.md` §8.9 deviation 1, §11.8).
- **Copy-paste augmentation doubles train time and loses on human answers** (`S3` 427.3 s / `S7`
  327.6 s vs `S2` 145.2 s; count MAE 4.50 / 4.40 vs 2.10) — it had been tuned against an answer key
  that merged cartridges, and inherited the merging.
- **Threshold selection is only as good as the validation split.** `score_all.select_thresholds`
  minimises validation *counting* error, and 5 of the 6 val frames are human-empty — so the grid
  happily lands on 0.90–0.95, and every run that landed on **0.95** (`D1`, `S7`, `S3`, `S1`, `C2`,
  `S5`) emits **zero** boxes on both stocked frames and posts 4.3–5.0 MAE while looking
  "perfect" on the empty ones (exact inventory 0.889!). Tune on frames that actually show a cabinet.
- **Peak VRAM does not track model size:** Small @ 512 px peaks at 6.15 GiB, every Medium @ 576 at
  4.09 GiB, `G2` (Small, 1024 px tiles) at 6.89 GiB. Resolution and batch size drive it.
- **The GT quiz answers whole-cartridge boxes.** Family G lost on counts anyway, but had it won,
  face-level box GT still would not exist: it would have to be drawn in the review app over
  `data/labels/faces_v2.json` before any face AP/ID number could be called human-verified. Its AP50
  (0.379 / 0.318 vs whole-box 0.875) is partly that mismatch, not only worse detection.
- **AP50 here is not independent GT.** Every human-GT annotation carries `box_source = "model"`; the
  human kept, corrected and rejected model proposals. 0.875 is agreement with what the labeller chose
  to draw.
- **Review #5 found `pair=on` was a 100 % no-op** for every non-VLM identifier before `5ff55c2`.
  Anything that claims to have measured the pair rule before that commit is measuring nothing.

## Open / next
1. ~~Answer the quiz~~ **done** — 128 questions answered, `data/gt/{test,val}.json` exported,
   `scripts/finalize_after_gt.sh` scored all 13 localizers and shipped `rfdetr-final-v2`.
2. **Re-export the winning bundle at the thresholds that actually won** (`score_thr 0.90`,
   `identity_thr 0.25` per `docs/freestyle/results_gt.json`), and confirm the phantom-class-index fix
   lets the threshold go below ~0.25 so it is a choice rather than a workaround. Until then the shipped
   bundle does not reproduce §8.2's numbers.
3. **Go get the frames that make the numbers mean something:** the 2 skipped test counts
   (`3270fb41…/065.jpg`, `50828e4b…/008.jpg`), the 4 unanswered test crops, the 47 unanswered train
   gallery questions, and — most of all — a **second cabinet and one re-stocked shelf**. Eight of ten
   test photos carry no usable cabinet, so a second stocked cabinet is worth more than another detector
   sweep. Cross-cabinet and stocking-change generalisation is completely unmeasured, and slot
   memorisation cannot be ruled out.
4. **Teach Part 1 vs Part 2 to be learned instead of read.** The numeral is unreadable (1 of 11 crops
   for the best reader, 3 of 47 digits found in a 315-crop re-pass); the 6 human train crops, 12 human
   `part` answers and the pair-rule geometry are the only plausible signal, and no run here uses them.
   The pair rule itself remains a hypothesis that never hurts (§REPORT §10.2).
5. **Batch the reader.** Identification is one Collective call per cartridge — 79–83 s per stocked photo
   with Inkling (`runs/cli_final_demo/counts.json`), ~2.2× that with the shipped Flash Next reader, and
   `LLMC_API_KEY` is required at inference time. Batched/cached reading is the biggest latency win left;
   contact sheets are already disproven (top-1 0.21 / 0.00).
6. Face family, if ever revived, needs negative-aware training (`G1`'s dominant model-label error is 56
   `FP:background`) plus a face-level box-GT pass; 53 residual duplicate face pairs await a human.
7. Unrun configs: `S4_c2_long`, `A1`–`A3` (pure VLM localizers), `E1` (shelf + kNN),
   `F1` (shelf + VLM id). `B2_sam3_siglip2` is blocked by HF gated access (403).

**Index row:** human-GT numbers now exist, so `freestyle-cartridges` is due in the `examples/README.md`
index — that file is outside this task's ownership, so it is left for the orchestrator.
