# Freestyle cartridge recognition

> Detection + identification + counting of Coca-Cola Freestyle syrup cartridges from the
> in-door fisheye camera, built on Vision Intern's label → judge → train path and extended
> where the core can't reach.
>
> **Status (2026-11-27): scored against human ground truth, then corrected — and now provisional.**
> 13 GPU runs were collected (whole-cartridge + front-face detector families), the phone quiz got 128
> answers, and every run was scored on them
> ([`docs/freestyle/RESULTS_GT.md`](../../docs/freestyle/RESULTS_GT.md)). The first scoring wrongly treated
> a photo answered *"not a usable cabinet view"* as a zero-cartridge frame; with that fixed those frames are
> **excluded**, and the remaining evidence is **2 test photographs** (`002.jpg` 22 cartridges, `045.jpg`
> 28) and **1 validation photograph** (`frame.jpg` 26). On them the **front-face detector wins the count**:
> **`G2_rfdetr_small_faces` (RF-DETR Small on printed faces, `score_thr 0.65`) + Qwen 3.8 Flash Next as the
> two-stage reader**, count MAE **0.500** [0.000, 1.000] (errors 0 and 1), identity **0.750** (21/28
> matched crops), shipped as bundle **`rfdetr-faces-final`**. The whole-cartridge
> **`S2_c2_tile1280` + the same reader** is the close alternative (2.000 count MAE, 0.579 identity, but
> AP50 **0.950** vs 0.527 — it is the better *box* detector) and stays shipped as **`rfdetr-final-v2`**.
> Read [`docs/freestyle/REPORT.md`](../../docs/freestyle/REPORT.md) §8.9 (selection + correction history)
> and §11 before repeating any of it: **at n = 2 a 95 % CI is arithmetic, not evidence.** Costs:
> [`docs/freestyle/RESOURCES.md`](../../docs/freestyle/RESOURCES.md). Failure modes (vs model labels):
> [`docs/freestyle/FAILURES.md`](../../docs/freestyle/FAILURES.md). Crop readers scored on human answers:
> [`docs/freestyle/READER_COMPARE.md`](../../docs/freestyle/READER_COMPARE.md). What the core provides and
> where it stops: [`docs/freestyle/VISION_INTERN_MAP.md`](../../docs/freestyle/VISION_INTERN_MAP.md).
> Contract: [`docs/freestyle/ARCHITECTURE.md`](../../docs/freestyle/ARCHITECTURE.md).

## Goal
Locate **every cartridge face**, identify it against the 18 active catalog ids, and count
per id — from one fisheye photograph per door-opening. Two shelves of spine-out boxes,
~10-20 instances per frame, faces ~60-150 px tall in a 1600×1200 / 2048×1536 frame, strong
wide-angle distortion, identity carried by small printed text + brand colour. This is the
dense/small/many-instance shape the toolkit still labels experimental, so the pipeline is
stage-decomposed rather than asked of one model: class-agnostic **localise** → full-res
**identify** → **count**, judged in between.

## Where it stands — corrected human GT, with the older model-label number kept for contrast

Every human-GT row below is answered on the quiz GT ([`docs/freestyle/RESULTS_GT.md`](../../docs/freestyle/RESULTS_GT.md),
**2 usable test images** / 2 door-openings, 1,000-replicate group bootstrap, thresholds selected on the 1
usable val image). Read the n = 2 warning above before using any ordering as a claim.

| Question | Answer on corrected human GT | Answer vs model labels | Evidence |
|---|---|---|---|
| Trained detector or zero-shot open-vocab? | **Trained, still decisive.** `G2` 0.500 / `S2` 2.000 count MAE vs `B1` 4.500, `B3` 10.000, `S6` 14.500; zero-shot AP50 0.507/0.522/0.357 vs 0.950–0.983 for the trained whole-box runs. `B1`'s best prediction score on the whole test set is 0.427, so zero-shot cannot be thresholded into usefulness | RF-DETR 0.614–0.718 AP50 vs OWLv2 0.181 / GroundingDINO 0.232 | `RESULTS_GT.md`, `docs/freestyle/RESULTS.md` |
| Whole cartridge or just the printed front face? | **Faces lead the count, whole boxes lead the box.** `G2` **0.500** [0.000, 1.000] vs `S2` 2.000 [1.000, 3.000], and the best identity of any row with >12 matched crops (0.750 = 21/28 vs 0.579 = 22/38; only the 12-crop `S6` rows rival it) — but AP50 0.527 vs 0.950 against whole-cartridge human boxes, 16 of whose 18 `FP`s for `G2` are `FP:fragment` because a face *is* a fragment of a cartridge box. `G1` (same faces, Medium, thr 0.45) is 4.000, so "faces count better" is **not** a family property — it is this run on 2 photos | Undecidable — each family was scored against its own model-made answer key | `RESULTS_GT.md` rows `G1/G2/S2`, `REPORT.md` §8.2.4, §9, §11.4 |
| Tile size? | **1280 px** within the whole-cartridge family: `S2` 2.000 vs `S1` (768 px) 4.000 vs `C2` (1024 px) 5.000 | Same ordering (MAE 3.583 / 6.000 / 4.417) | `RESULTS_GT.md` rows `S1/S2/C2` |
| Copy-paste augmentation? | **No.** `S3` 2.500 and `S7` 3.000 vs `S2` 2.000 at 2–3× the training time (427.3 s / 327.6 s vs 145.2 s) — no longer the catastrophe it looked like under the broken scoring, still no gain | It *helped*: `S3` had the best model-label AP50 (0.718) | `RESULTS_GT.md`, `runs/S3_*`, `runs/S7_*` |
| How is identity done? | **A VLM reading the crop, not embedding matching.** On the pick: Flash Next 0.750 (21/28) > Inkling 0.643 (18/28) > `linear_trainvlm` 0.179 > `knn_trainvlm` 0.036 (1/28) — face embeddings do not transfer to whole-box crops. On `S2` the same comparison is 0.579 / 0.474 / 0.421 / 0.421. Adding human val crops to any gallery changes nothing (0 usable val crops answered) | Hybrid (Inkling family + class-balanced kNN refinement), picked by an unverified visual read | `RESULTS_GT.md` §8.2.2 of REPORT, `READER_COMPARE.md` |
| Which VLM reads the crops? | **Qwen 3.8 Flash Next, two-stage + shelf hint** — better than Inkling on **all 13** localizers now that identity is defined everywhere. Standalone bake-off on 39 human-labelled crops: top-1 0.56 (CI 0.41–0.71) vs 0.49 (0.34–0.64), family 0.69 vs 0.67, abstention 18 % vs 44 %, at 9.1 s vs 3.5 s per crop | Inkling 2 Mini (faster, and it was the labeller) | `READER_COMPARE.md`, `RESULTS_GT.md` |
| Can Part 1 / Part 2 be read? | **No — unchanged by the correction.** Best reader: numeral right on **1 of 11** crops (0.09), Inkling **0 of 11**; the Flash re-pass found a digit on **3 of 47** part-capable crops vs 2 of 123 on whole crops. `unknown` (7 on the one class-scored frame) is the largest per-class error | 2 of 123 whole crops (Inkling), 59 of 597 faces (never human-checked) | `READER_COMPARE.md`, `runs/bootstrap_v3_flash/summary.md`, `runs/bootstrap_v2/summary.md` |
| Does the pair rule work? | **Direction confirmed, sample too small.** 92 pair-comparable rows, identity defined on 79: **0 lowered, 12 raised, 67 unchanged**, every count/AP number bit-identical. It helps the abstaining Inkling rows (10 of the 12) and moves **0 crops on the shipped configuration** | Untestable — review #5 found `pair=on` was a 100 % no-op before `5ff55c2` | `REPORT.md` §10.2, `results_gt.json` |
| What is the architecture? | **Bundle `rfdetr-faces-final`** = `G2` front-face weights + duplicate suppression + 2-shelf counting + `shelf_hint` + Flash Next two-stage reader, `score_thr 0.65` (the val-selected value), `pair_rule: report`. **Live deviations:** `identity_thr 0.45` vs the scored 0.25, and the CLI's suppression drops 1–2 real boxes the scored row kept (21/25 live vs 22/27 scored) | Bundle `rfdetr-m-1280-v1` (provisional) | `artifacts/rfdetr-faces-final/pipeline.yaml`, `REPORT.md` §7.4, §8.9 |
| Is it good enough? | **No.** Two photos, one cabinet, one stocking; AP50 is not independent GT (every human box came from a model proposal); exact inventory is computable on **1** frame and is 0.000 for every run; identification costs one API call per cartridge (3–9 s each) | — | `REPORT.md` §11, `results_gt.json` |

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
| **Reader / identifier (in both shipped bundles)** | **`qwen3-8-flash-next`** (`two_stage` + shelf hint) | Collective | **Chosen on human answers**: better than Inkling on **all 13** localizers — 0.750 (21/28) vs 0.643 on the pick, 0.579 vs 0.474 on `S2`; standalone bake-off 0.56 vs 0.49 top-1 on 39 human-labelled test crops; 9.1 s mean / 6.3 s median per crop measured there, **3.0 s/crop** in today's live CLI demo (budget the slow one); thinking model — needs `max_tokens=8192` |
| Labeller (stage A + front faces) | `david-birks/inkling-2-mini` | Collective | 0.4–5.4 s/call, reliable; obeys whichever coordinate convention you prompt; 2.8 s median per crop as a reader — the cheap fallback |
| Judge (crop sheets) | `qwen3-8-flash-next` | Collective | needs `max_tokens=4096`, intermittently 429-throttled (use the `david-birks/` prefix for stability) |
| Embedder (superseded kNN identifier) | `google/siglip2-base-patch16-224` | local / k8s | also `facebook/dinov2-base` in the B/C/D runs. Scored **0.421** (16/38, `S2` gallery) / **0.036** (1/28, face gallery) against a human where the VLM reader scores 0.579 / 0.750 — the reason neither shipped bundle carries a `gallery.npz` |
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
uv run python -m freestyle.artifacts verify rfdetr-faces-final   # OK: rfdetr-faces-final (3 files verified)
uv run --extra train python -m freestyle.cli count <frame.jpg> --bundle rfdetr-faces-final --device cpu \
    --out runs/cli_faces_demo --json     # needs LLMC_API_KEY: one Collective call per crop
uv run --extra train python -m freestyle.cli fuse <f1.jpg> <f2.jpg> --bundle rfdetr-faces-final --method max --json
uv run --extra train python -m freestyle.artifacts export --run runs/G2_rfdetr_small_faces-7c7b4587 \
    --name rfdetr-faces-final --overwrite \
    --identifier '{"type":"vlm_crop","model":"qwen3-8-flash-next","mode":"two_stage","upscale":2}' \
    --counting   '{"identity_thr":0.45,"neighbour_vote":"report","max_rows":2,"score_thr":0.65,"suppress_iou":0.45,"suppress_containment":0.7,"suppress_row_iou":0.3,"suppress_row_x_overlap":0.5,"shelf_hint":true,"pair_rule":"report"}'
#  note: export copies the *run's* metrics.json score_thr unless --counting overrides it

# 8 — the whole post-GT sequence, in order, with a hard sufficiency gate
scripts/finalize_after_gt.sh --dry-run     # validate + print everything it would run
scripts/finalize_after_gt.sh --skip-quiz   # re-scores everything (gate: >=2 usable test, >=1 usable val)
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
- **Bundles:** `artifacts/rfdetr-faces-final` — **the provisional pick of the corrected human-GT scoring**
  (`G2` front-face weights, detector class `cartridge` again because the family is class-agnostic,
  `vlm_crop qwen3-8-flash-next two_stage`, 2-shelf counting/suppression, `score_thr 0.65`,
  `pair_rule: report`; 3 files, 127,904,367-byte checkpoint `5f7f0bc9…`, **no `gallery.npz`**).
  `artifacts/rfdetr-final-v2` is the whole-cartridge **alternative** (`S2` weights, same reader,
  `score_thr 0.308` — its own val-selected value is 0.20). Superseded pre-GT candidates kept for
  comparison: `rfdetr-m-1280-knn-v0` (plain kNN), `rfdetr-m-1280-v1` (hybrid), `rfdetr-m-1280-v2`
  (v1 knobs + a gallery refreshed from the 6 human train crops). Each has `pipeline.yaml`,
  `manifest.json` (SHA-256 per file) and, where applicable, `gallery.npz`. All gitignored — never
  committed, never pushed to a registry.
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
  `Unknown RF-DETR detector class index: 1`, which is why bundle v1 was stuck at `score_thr=0.308`
  (fixed: the no-object slot is dropped for single-category checkpoints). **What is still open:**
  `freestyle.artifacts export` copies the **run's** `metrics.json` threshold unless `--counting` overrides
  it, so `rfdetr-final-v2` ships `0.308` while its val-selected value is `0.20`; `rfdetr-faces-final` only
  got the right `0.65` because it was passed explicitly. `scripts/finalize_after_gt.sh` step 6 sends no
  `score_thr` at all and hardcodes the bundle name — wire it to `results_gt.json` before trusting a
  re-exported bundle (`docs/freestyle/REPORT.md` §8.9 deviations, §12.6/§12.7).
- **Copy-paste augmentation doubles train time and did not buy counting** (`S3` 427.3 s / `S7`
  327.6 s vs `S2` 145.2 s; corrected count MAE 2.50 / 3.00 vs 2.00, count AP50 0.964 / 0.964 vs 0.950)
  — it had been tuned against an answer key that merged cartridges, and inherited the merging. Under the
  *broken* scoring it looked catastrophic (4.50 / 4.40 vs 2.10); the correction moved everyone.
- **Threshold selection is only as good as the validation split — and it is now down to one frame.**
  `score_all.select_thresholds` minimises validation *counting* error on **usable** val frames, and
  exactly one of the 6 val frames is usable (`frame.jpg`, 26 cartridges), so every `score_thr` in
  `results_gt.json` is fitted to a single number: `G2` 0.65, `S2` 0.20, `C1` 0.85, `C2` 0.90, `S6` 0.05.
  The pre-correction version of this pass fitted six numbers, five of them fake zeros, which is how
  `D1`/`S7`/`S3`/`S1`/`C2`/`S5` landed on 0.90–0.95 and silenced themselves. Tune on frames that actually
  show a cabinet, and get more of them.
- **Peak VRAM does not track model size:** Small @ 512 px peaks at 6.15 GiB, every Medium @ 576 at
  4.09 GiB, `G2` (Small, 1024 px tiles) at 6.89 GiB. Resolution and batch size drive it.
- **The GT quiz answers whole-cartridge boxes, and the pick is now a face detector.** `G2` wins the count
  on 2 photos but is scored against boxes that describe whole cartridges: its AP50 is 0.527 vs `S2` 0.950 /
  `C1` 0.983, and §9 of REPORT shows 16 of its 18 false positives on the scored frames are
  `FP:fragment` — a face box inside a cartridge box. Face-level human box GT would have to be drawn in the
  review app over `data/labels/faces_v2.json` before that gap could be read as anything but the mismatch.
  This is the strongest reason not to treat the pick as settled.
- **AP50 here is not independent GT.** Every human-GT annotation carries `box_source = "model"`; the
  human kept, corrected and rejected model proposals. 0.983 is agreement with what the labeller chose
  to draw.
- **Review #5 found `pair=on` was a 100 % no-op** for every non-VLM identifier before `5ff55c2`.
  Anything that claims to have measured the pair rule before that commit is measuring nothing.

## Open / next
1. ~~Answer the quiz~~ **done** — 128 questions answered, `data/gt/{test,val}.json` exported,
   `scripts/finalize_after_gt.sh` scored all 13 localizers; the scoring was then **corrected** (uncountable
   photos excluded, not zeroed) and the winner moved from `S2` to `G2`. See REPORT §8.9.1.
2. **Make the shipped bundle the scored configuration.** `rfdetr-faces-final` already ships the val-selected
   `score_thr 0.65`, but it ships `identity_thr 0.45` where scoring used **0.25**, and the CLI's duplicate
   suppression costs 1–2 real boxes per photo relative to `score_all`'s counting (21/25 live vs 22/27
   scored, REPORT §7.4). Re-export with `identity_thr 0.25` and either re-tune `suppress_*` on the usable
   photos or teach `score_all` the same suppression — until then nobody should quote 0.500 as "what the
   bundle does".
3. **Go get the frames that make the numbers mean something — this is the whole ballgame now.** The 2
   skipped test counts (`3270fb41…/065.jpg`, `50828e4b…/008.jpg`) turn n = 2 into n = 4 and
   `scripts/finalize_after_gt.sh --skip-quiz` re-scores every row; the 4 unanswered test crops, the 47
   unanswered train gallery questions, and — most of all — a **fixed, well-positioned camera** and a
   **second cabinet / re-stocked shelf**. 10 of 12 test photos carry no usable cabinet at all, so more
   usable photos are worth more than another detector sweep. Cross-cabinet and stocking-change
   generalisation is completely unmeasured, and slot memorisation cannot be ruled out.
4. **Teach Part 1 vs Part 2 to be learned instead of read.** The numeral is unreadable (1 of 11 crops
   for the best reader, 3 of 47 digits found in a 315-crop re-pass); the 6 human train crops, 12 human
   `part` answers and the pair-rule geometry are the only plausible signal, and no run here uses them.
   The pair rule itself remains a hypothesis that never hurts (REPORT §10.2: 12 of 79 defined rows raised,
   0 lowered, and 0 crops moved on the shipped configuration).
5. **Batch the reader.** Identification is one Collective call per cartridge — 65–80 s per stocked photo
   today (3.0 s/crop, `runs/cli_faces_demo/counts.json`), 9.1 s/crop mean in the bake-off, so plan on
   75–225 s per photo — and `LLMC_API_KEY` is required at inference time. Batched/cached reading is the
   biggest latency win left; contact sheets are already disproven (top-1 0.21 / 0.00).
6. **Settle the face question properly.** Family G now leads on counts, so its two open debts matter:
   a **face-level box-GT pass** in the review app over `data/labels/faces_v2.json` (53 residual duplicate
   face pairs await a human), and negative-aware training — `G1`'s dominant model-label error is 56
   `FP:background`, i.e. faces invented on slats and door edges, and `G2` still shows 2 `FP:background`
   plus 13 missed cartridges against whole-box GT on the two scored frames (REPORT §9).
7. Unrun configs: `S4_c2_long`, `A1`–`A3` (pure VLM localizers), `E1` (shelf + kNN),
   `F1` (shelf + VLM id). `B2_sam3_siglip2` is blocked by HF gated access (403).

**Index row:** human-GT numbers (and a corrected rescoring) now exist, so `freestyle-cartridges` is due in
the `examples/README.md` index — that file is outside this task's ownership, so it is left for the
orchestrator.
