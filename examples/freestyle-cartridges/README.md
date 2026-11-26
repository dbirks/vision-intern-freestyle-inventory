# Freestyle cartridge recognition

> Detection + identification + counting of Coca-Cola Freestyle syrup cartridges from the
> in-door fisheye camera, built on Vision Intern's label → judge → train path and extended
> where the core can't reach. **Status: wiring complete, first numbers pending** — every
> `TBD` below is answered by the experiment matrix in `configs/experiments/`, not assumed
> (contract: [`docs/freestyle/ARCHITECTURE.md`](../../docs/freestyle/ARCHITECTURE.md)).
> What the core provides and where it stops:
> [`docs/freestyle/VISION_INTERN_MAP.md`](../../docs/freestyle/VISION_INTERN_MAP.md).

## Goal
Locate **every cartridge face**, identify it against the 18 active catalog ids, and count
per id — from one fisheye photograph per door-opening. Two shelves of spine-out boxes,
~10-20 instances per frame, faces ~60-150 px tall in a 1600×1200 / 2048×1536 frame, strong
wide-angle distortion, identity carried by small spine text + brand colour
(`data/raw/freestyle-vision-data/hardware/cabinet_and_scene.md:30-34`, `:47`). This is the
dense/small/many-instance shape the toolkit still labels experimental
(`README.md:14`), so the pipeline is stage-decomposed rather than asked of
one model: class-agnostic **localise** → full-res **identify** → **count**, judged in
between.

## Data
- **Source:** `data/raw/freestyle-vision-data/` — read-only, **private**, original encoders
  (54 server-verified photographs + 15 workstation copies,
  `data/raw/freestyle-vision-data/README.md:18-19`); 1600×1200 UXGA / 2048×1536 QXGA at
  JPEG quality 11 (`data/raw/freestyle-vision-data/data/images.csv`;
  `data/raw/freestyle-vision-data/hardware/capture_configuration.md:15`).
- **Preprocessing:** no re-encode of raw; derived artefacts only under `data/derived/`
  (tiles via `freestyle/train/export_tiles.py`, crops on demand); image/group ids from
  `freestyle/data.py:96`, `:224`.
- **Splits:** `data/splits.json` built by `freestyle/make_splits.py:101` — the independent
  unit is the **door-event group**, which Vision Intern's splitter cannot express
  (`jobs/make_splits.py:23-56` groups by a dataset column or image hash; map §6 L13).
  Test groups freeze once human GT exists.
- **GT:** only `data/gt/`, produced by a human in `freestyle/review/server.py` on top of
  model proposals (`freestyle/bootstrap_labels.py`, which refuses `--out` under `data/gt/`:
  `freestyle/bootstrap_labels.py:242-243`). Model output is never GT.
- **Boundary:** images go only to the LLM Collective and our k8s cluster; nothing under
  `data/`, `runs/`, `artifacts/` is committed; **no Hugging Face pushes at all** (so the
  whole `jobs/` HF-Jobs + `push_to_hub` layer is unused — map §6 L12).

## Classes
18 `active: true` ids from `data/raw/freestyle-vision-data/data/catalog_items.json`, plus
`cartridge` (class-agnostic super-class) and `unknown`
([`ARCHITECTURE.md`](../../docs/freestyle/ARCHITECTURE.md) §Classes). Single source of
truth is `freestyle/catalog.py:62-88` (`class_names`, `category_map`, `category_id`) — do
not paste the list into a labeller call.
```
coca-cola-part1,coca-cola-part2,coke-zero-part1,coke-zero-part2,diet-coke-part1,
diet-coke-part2,dr-pepper-part1,dr-pepper-part2,fanta-orange,flavor-cherry,flavor-lemon,
flavor-orange,flavor-vanilla,hi-c,ingredient-aa,ingredient-cc,nns-sweetener,sprite
+ cartridge (stage A)  + unknown (identity not determinable)
```
Stage A localises **`cartridge` only**; the 18-way choice happens on full-res crops (see
Commands). Reason: an 18-way closed-set detector at 704² input
(`workflows/train_rfdetr.py:522`, map §3) turns ~60 px faces into ~25 px, and free-form VLM
names must be normalised before any class filter (`tools/bbox_utils.py:141-160`).
Feed `label_aliases` from the catalog's `aliases` field — the resolver is tolerant to
case/punctuation but **not** fuzzy, so `coke zero part1` will not reach `coke-zero-part1`
without an alias (map §6 L7).

## Models per role
Collective only (`https://llm-collective.e-gineering.com/v1`); the key is read at runtime
(`freestyle/collective.py:73-88`, `configs/bootstrap/default.yaml:1-2`) and never written
to a file. Vision-capable ids were confirmed from `~/.pi/agent/models.json`
`providers.collective.models[].input`; `qwen3-8-27b` declares **text-only** input, so it is
the description-writer/orchestrator, not a vision judge.

| Role | Model | Family | Size | Where |
|---|---|---|---|---|
| Orchestrator | the driving agent (+ `qwen3-8-27b` for text-only drafting) | — | — | issues calls, never labels/judges (`agents.md:30-38`) |
| Labeller (stage A, `cartridge`) | `qwen3-8-flash-next` | Qwen | TBD params | Collective, `backend="openai"` |
| Labeller (fusion member) | `david-birks/inkling-2-mini` | david-birks | TBD | Collective; boxes fused by IoU (`configs/bootstrap/default.yaml:3`) |
| Judge | `qwen3-8-flash-next` (`configs/bootstrap/default.yaml:14-17`) | Qwen | TBD | Collective, crop-sheet mode + chunked (`workflows/vlm_judge.py:457-488`) |
| Second judge | **TBD** | — | — | only 2 image-capable families exist on the Collective and one is the labeller's; ensemble is hard-capped at 2 (`tools/model_options.py:22-30`) → first runs record single-judge + geometric-guard behaviour honestly |
| Identifier (per crop) | `david-birks/inkling-2-mini` (`configs/bootstrap/default.yaml:18-22`) | david-birks | TBD | Collective, **full-res crops** via `freestyle/collective.py:167` `chat_image(max_side=None)` |
| Detector (trained) | `RFDETRSmall` @ `resolution: 512` in `F1` (TBD: 512 vs 704 vs 1024, nano→large) | — | — | **k8s Job**, home cluster, via `freestyle/train/train_detector.py:31-43` (not `workflows.train_rfdetr`, which exposes no image-size knob — map §3) |
| Open-vocab prior (B family) | OWLv2/DINOv2, SAM3+SGLIP (`configs/experiments/B*.yaml`) | — | — | k8s Job; alternative stage A when the VLM's box quality is the bottleneck |

## Commands
Plan; `--dry-run` first, and the model-matrix entry point is
`freestyle/experiments/launch.py` (it resolves config → run dir → local or k8s → eval).
```bash
# 0 — collective probe (confirms the ids/params we are about to bet the run on)
uv run python -m freestyle.probe_collective

# 1 — category definitions (the one human gate, workflows/gen_descriptions.py:104-147)
uv run python -m workflows.gen_descriptions --labels "<freestyle.catalog.class_names()>" \
  --model qwen3-8-27b --backend openai --base-url "$COLLECTIVE_BASE_URL" \
  --api-key "$COLLECTIVE_API_KEY" --output descriptions.json      # ← HUMAN REVIEW

# 2 — group-safe splits (never the HF splitter)
uv run python -m freestyle.make_splits --test-frac 0.4 --val-frac 0.2 --seed 0 \
  --out data/splits.json

# 3 — resumable VLM label proposals (never written to data/gt/)
uv run python -m freestyle.bootstrap_labels --config configs/bootstrap/default.yaml \
  --images train --out data/labels/proposals-train.json
#     ↑ calls workflows.label_dataset(classes=['cartridge'], tile=…, max_dimension=None,
#       image_format='png', max_tokens=8192, image_paths=[id])
#       (freestyle/bootstrap_labels.py:172-182) and workflows.judge_labels(judge_mode='crops',
#       judge_batch_size=8, max_tokens=4096) (freestyle/bootstrap_labels.py:196-204)

# 4 — human verification → the only GT (local server; --out must land under data/gt/)
uv run python -m freestyle.review.server --proposals data/labels/proposals-train.json \
  --images train --out data/gt/train.json                              # resume-safe

# 5 — detector training (class-agnostic, tiled) as a GPU Job on the home cluster
uv run python -m freestyle.experiments.launch configs/experiments/F1_rfdetr_agnostic_vlm_id.yaml
#   (submit path: k8s/submit.sh + k8s/job-template.yaml; stage code:
#    freestyle/experiments/entry.py:61-87 → export_tiles + train_detector;
#    augmentation off by default — HorizontalFlip/HueSaturationValue break shelf side and
#    brand colour, workflows/train_rfdetr.py:344-347, map §6 L10)

# 6 — predict (contract format) and evaluate against HUMAN GT
uv run python -m freestyle.infer.detector --checkpoint runs/<exp>/training --backend rfdetr \
  --images <split>.coco.json --images-root data/raw/freestyle-vision-data/images \
  --out runs/<exp>/predictions.json
uv run python -m freestyle.eval.evaluate --gt data/gt/val.json \
  --pred runs/<exp>/predictions.json --split-map data/splits.json
```
Why not the documented quickstart: `python -m workflows.vlm_label …` cannot reach the
dense-scene options at all — its CLI registers none of them (`workflows/vlm_label.py:388-439`,
`workflows/vlm_judge.py:1000-1083`), while `examples/dense-scenes/README.md` advertises them
through the Python API. Use the API (or `freestyle/` wrappers).

## Outputs
All local — no Hub repos, no public registry.
- Proposals: `data/labels/proposals-<split>.json` (model-generated, never GT)
- GT: `data/gt/<split>.json` (human-verified, `freestyle/review/`)
- Runs: `runs/<exp_id>/{resolved.yaml,predictions.json,meta.json,metrics.json,failures/}`
  (`freestyle/experiments/schema.py`, `freestyle/eval/{evaluate,failures,report}.py`)
- Models: `runs/<exp>/training/` (RF-DETR / YOLO checkpoints, PVC-backed)
- Table: `docs/freestyle/RESULTS.md` (generated by `freestyle/eval/report.py`, not
  hand-edited). Best mAP / top-1 ID / count MAE: **TBD**.

## Notes / gotchas
- **The many-instance ceiling is in the tokens, and the defaults are still the old ones.**
  Detection defaults to `max_tokens=4096` (`tools/vlm_detect.py:178`) and judging to
  **1024 for any model whose name does not contain `lfm3-vl`**
  (`workflows/vlm_judge.py:501`, `:751`) ≈ 20-25 verdicts/image; on a full two-shelf frame
  that truncates. Truncation is now *detectable* (`finish_reason == "length"`,
  `tools/vlm_client.py:159-161`; derived for the transformers backend `:212-228`), a
  truncated **tile** self-heals by one quad-split
  (`tools/vlm_detect.py:315-323`), and un-reached verdicts are `unjudged`
  (`workflows/vlm_judge.py:318-320`) — but `unjudged` still fails `ensemble_keep`
  (`workflows/vlm_judge.py:659-667`), so a truncated judge response still **deletes correct
  boxes**. Always pass `max_tokens` and `judge_batch_size`.
- **Nothing the VLM sees is full-resolution unless you say so.** `max_dimension=1280` and
  JPEG q85 are the defaults (`tools/vlm_client.py:53`, `:70-74`, `:80-81`); pass
  `max_dimension=None, image_format="png"` on every call, and judge/identify crops with
  `freestyle/collective.py:167` (`max_side=None`). Crop-sheet judging thumbnails each crop
  to 256 px (`workflows/vlm_judge.py:380`) — fine for "is that a cartridge", **not** fine
  for reading a spine.
- **The judge can only delete.** It never adds a missed cartridge
  (`jobs/merge_judges.py:111-140`, `workflows/vlm_judge.py:843-847`), so stage-A recall
  bounds every counting number; measure A's recall before trusting identification.
- **No confidence, no dedupe for free.** `vlm_detect` returns a `score` only in the
  `n_samples>1` self-consistency path (`tools/vlm_detect.py:370-384`, and that requires
  `temperature>0`, `tools/vlm_detect.py:226-227`), and `nms` is applied only when tiling or
  after a truncation split (`tools/vlm_detect.py:367`) and only within the same label
  (`tools/bbox_utils.py:223-252`). Our `predictions.json` `score`/`id_score` come from the
  fusion/identification stages, not from the VLM.
- **Local mode never produces the numbered overlay** the judge was redesigned around
  (`label_dataset` writes `detections` only, `workflows/vlm_label.py:262`; the column comes
  from `jobs/push_labeled.py:89-97`), so local judging without `overlay`/`judge_mode`
  falls back to the raw-coordinate COCO prompt (`workflows/vlm_judge.py:742-758`) — the
  path the code itself calls the hallucination source (`:440-448`). `agents.md:91-93` and
  `jobs/README.md:90-96` still claim otherwise.
- **Trainer drops zero-box images** (`workflows/train_rfdetr.py:252`), so `no_cabinet` and
  empty frames never teach negatives. Our tile exporter keeps them; if we switch to the HF
  trainer, that filter has to go (map §8 row 10).
- **`jobs/` is HF-Jobs-shaped and unusable here**: clone-from-GitHub bootstrap
  (`jobs/train_rfdetr_job.py:41-48`), forced `--push-to-hub` (`:93-95`), and the
  `vision-e2e-runner` instruction built on it (`.claude/agents/vision-e2e-runner.md:84-140`).
  Keep its discipline (descriptions gate, smoke ~5 images then scale, record both
  agree-policies), replace the plumbing with `k8s/submit.sh`.
- **Upstream eval can't be reused wholesale**: `jobs/eval_vs_gt.py` and
  `jobs/eval_pr_roc.py` require an HF dataset with a human `objects` ClassLabel column and
  bridge classes by name (`jobs/eval_vs_gt.py:82-88`, `jobs/eval_pr_roc.py:181-198`);
  neither computes top-1 identification, count MAE, exact-inventory rate, or group
  bootstrap → `freestyle/eval/`.
- **TBD / open questions:** (a) can 18-way identity be read from a ~100-150 px crop at
  JPEG q11 at all, or does it need the `shelf_hint` + `par` priors
  (`data/raw/freestyle-vision-data/data/catalog_items.json`) as a language-side constraint;
  (b) does the Collective honour `chat_template_kwargs`/`enable_thinking` for our ids
  (only two hard-coded Qwen ids get it today, `tools/model_options.py:10-19`) — pass
  `extra_body` explicitly rather than relying on it; (c) second vision family for a real
  two-judge ensemble; (d) detector `resolution` and class-agnostic-vs-18-way, answered by
  the `F*` vs `D*` experiments; (e) fisheye undistortion before tiling (derived copies only).
- **Index row:** add `freestyle-cartridges` to the `examples/README.md` index when this run
  produces numbers (not done here — that file is outside this task's ownership).
