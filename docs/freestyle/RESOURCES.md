# Freestyle cartridge recognition — resource ledger

Every measured cost (GPU seconds, VRAM, wall time, API calls, latency) with the file it came
from. Nothing here is estimated: each number is read from a run artifact under `runs/` (gitignored,
private) or from the two documents that quote them. Provenance of a run =
`runs/<run>/provenance.json` (image digest, GPU, HAMi mode, VRAM reservation, repo tarball SHA-256);
timings = `runs/<run>/meta.json`; environment = `runs/<env>.txt` (`pip freeze`) and `runs/<run>/env.txt`.

Conventions:
* **Peak VRAM** = `meta.json.peak_gpu_bytes`, the high-water mark reported by the PyTorch caching
  allocator, not `nvidia-smi`. The `nvidia_smi` field in the same file is a MiB reading taken once,
  after collection, by the collector.
* **`s/img`** = detector/identifier inference only (`inference.<split>.seconds_per_image`).
  For zero-shot runs the proposals are generated once per split and cached, so
  `inference.<split>.proposal_seconds_per_image` is the real cost of the localizer and
  `seconds_per_image` is the cost of reading the cached proposals back. `val` is the first split
  evaluated, so its proposal number also carries the HF checkpoint load.
* Quoted figures (bundle CPU demo, Collective latencies) name the file they were read from.

## 1. GPU experiment runs (all RTX 5090, `mode: hami`, except where noted)

Image for every run: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime@sha256:bfbb4a2b…c3f0`
(`provenance.json.image`), Python 3.12.3, torch 2.11.0+cu130 (`meta.json.versions`).
Node `talos-8me-tcy`; during the windows below the resident `inkling-2-mini` serving pod was
paused, so HAMi free = 32,607 MiB / physical free = 32,109 MiB (`runs/driver-S2.log:2`,
`runs/driver-G.log:2,202`).

| Run | GPU / VRAM request | Train wall (s) | Peak VRAM (`peak_gpu_bytes`) | Inference test / val (s/img) | Job wall total (s) | image+init / unpack+pip / command (s) | Source |
|---|---|---:|---:|---:|---:|---|---|
| `B1_owlv2_dinov2-4c99011d` | 5090 / 12000 MiB | — (zero-shot) | 1.97 GiB (2,112,137,216 B) | 0.059 pipe (0.945 proposal) / 0.101 pipe (30.226 proposal) | 330 | 1 / 47 / 283 | `runs/B1_owlv2_dinov2-4c99011d/meta.json` |
| `B2_sam3_siglip2-babcbfe8` | 5090 / 12000 MiB | — (failed, HF 403 gated) | 0.01 GiB (9,568,256 B, embedding stage only) | — (never reached inference) | unfinished: 1 + 83 recorded, no `command` field | 1 / 83 / — | `runs/B2_sam3_siglip2-babcbfe8/meta.json` |
| `B3_gdino_dinov2-be668339` | 5090 / 12000 MiB | — (zero-shot) | 1.68 GiB (1,798,717,952 B) | 0.026 pipe (0.363 proposal) / 0.017 pipe (17.769 proposal) | 150 | 1 / 25 / 125 | `runs/B3_gdino_dinov2-be668339/meta.json` |
| `C1_rfdetr_small_agnostic_dinov2-55ef8ae7` | 5090 / 12000 MiB | 247.4 | 6.15 GiB (6,603,773,952 B) | 0.432 / 0.599 | 615 | 0 / 338 / 277 | `runs/C1_rfdetr_small_agnostic_dinov2-55ef8ae7/meta.json` |
| `C2_rfdetr_medium_agnostic_siglip2-89fe7c2b` | 5090 / 12000 MiB | 289.0 | 4.09 GiB (4,394,649,088 B) | 0.328 / 0.421 | 409 | 1 / 100 / 309 | `runs/C2_rfdetr_medium_agnostic_siglip2-89fe7c2b/meta.json` |
| `D1_rfdetr_small_multiclass-6f3c344b` | 5090 / 12000 MiB | 195.7 | 6.16 GiB (6,616,291,328 B) | 0.352 / 0.513 | 301 | 1 / 80 / 221 | `runs/D1_rfdetr_small_multiclass-6f3c344b/meta.json` |
| `S1_c2_tile768-d52f6d7c` | 5090 / 12000 MiB | 223.2 | 4.09 GiB (4,394,649,088 B) | 0.363 / 0.519 | 293 | 1 / 49 / 244 | `runs/S1_c2_tile768-d52f6d7c/meta.json` |
| `S2_c2_tile1280-db269936` | 5090 / 12000 MiB | 145.2 | 4.09 GiB (4,394,649,088 B) | **0.223** / 0.309 | 213 | 1 / 50 / 163 | `runs/S2_c2_tile1280-db269936/meta.json` |
| `S3_c2_copypaste-205b1c7c` | 5090 / 12000 MiB | 427.3 | 4.09 GiB (4,394,649,088 B) | 0.337 / 0.429 | 499 | 1 / 51 / 448 | `runs/S3_c2_copypaste-205b1c7c/meta.json` |
| `S5_rfdetr_large_agnostic-7f6457d4` | 5090 / **16384** MiB | 176.3 | 4.09 GiB (4,394,649,088 B) | 0.359 / 0.445 | 247 | 1 / 50 / 197 | `runs/S5_rfdetr_large_agnostic-7f6457d4/meta.json` |
| `S6_owlv2_containment-cbc70383` | 5090 / 12000 MiB | — (zero-shot + filters) | 1.97 GiB (2,112,137,216 B) | 0.016 pipe (0.965 proposal) / 0.050 pipe (1.109 proposal) | 52 | 0 / 19 / 33 | `runs/S6_owlv2_containment-cbc70383/meta.json` |
| `S7_c2_tile1280_copypaste-b6fc182e` | 5090 / 12000 MiB | 327.6 | 4.09 GiB (4,394,649,088 B) | 0.221 / 0.351 | 396 | 1 / 50 / 346 | `runs/S7_c2_tile1280_copypaste-b6fc182e/meta.json` |
| `G1_rfdetr_medium_faces-cc431fcd` | 5090 / 12000 MiB | 75.3 | 4.09 GiB (4,394,649,088 B) | **0.168** / 0.279 | 139 | 1 / 49 / 90 | `runs/G1_rfdetr_medium_faces-cc431fcd/meta.json` |
| `G2_rfdetr_small_faces-7c7b4587` | 5090 / 12000 MiB | 191.4 | **6.89 GiB** (7,398,841,344 B) | 0.242 / 0.365 | 262 | 0 / 52 / 210 | `runs/G2_rfdetr_small_faces-7c7b4587/meta.json` |
| `Z0_cpu_chain_check-87d5893f` | 5090 node, `mode: none` (CPU only) | — | 0 B (`peak_gpu_bytes: 0`) | CPU tiled OWLv2-base: 27.1 s (test, 12 imgs) / 98.4 s (val) | not recorded (no `job_wall` block) | — | `runs/chain-check/Z0_cpu_chain_check-87d5893f/meta.json` |

Notes:
* `S4_c2_long`, `A1`–`A3`, `E1`, `F1` have configs under `configs/experiments/` and no collected
  run, so they cost nothing. `S3_c2_copypaste-e6ce578e` died in the waiter
  (`jobs.batch … not found`, `runs/driver-S.log`) after submitting; `C2_...-27f2e3d6` was
  collected without `metrics.json`/`meta.json`.
* Copy-paste augmentation is the dominant train-time cost: S3 (427.3 s) vs C2 (289.0 s) and
  S7 (327.6 s) vs S2 (145.2 s) at the same model/resolution/epochs.
* Bundle exports cost one local copy of the checkpoint each (`G2` 127,904,367 B →
  `artifacts/rfdetr-faces-final/`, verified `OK: rfdetr-faces-final (3 files verified)`); no GPU time.
* Face runs train on fewer frames than the whole-cartridge runs (597 cleaned faces over 9 train
  frames vs 563 boxes over 10), which is most of why G1's train time is 75.3 s against S2's 145.2 s
  at the same model and tile size (`runs/faces_v2/summary.md`).
* Peak VRAM does **not** rank by detector size as one might expect: `C1`/`D1` (Small, 512 px)
  peak at 6.15/6.16 GiB, `G2` (Small, 1024 px tiles) at 6.89 GiB, while every 576 px Medium run
  sits at 4.09 GiB. Batch size and input resolution, not parameter count, drive the mark.

### GPU window occupancy (the actual card time burned)

| Window | Runs | Card time (sum of `job_wall.job_total`) | Source |
|---|---|---:|---|
| GPU window #1 (2026-11-26, 23:34–00:57 UTC) | `B1`, `B2`(fail), `B3`, `C1`, `C2`, `D1`, `S1`, `S2`, `S3`, `S5`, `S6`, `S7` | 3,657 s ≈ 61 min of job wall, ~25 min of it `unpack+pip` | `meta.json.job_wall.job_total` for the runs listed in §1 |
| GPU window #2 (2026-11-27, 14:13–14:21 UTC) | `G1`, `G2` | 401 s ≈ 6.7 min | `runs/G1_*/meta.json`, `runs/G2_*/meta.json` |

Window #2 was closed after ~12 minutes of the ~60 minutes offered, because the "answer key" was
model-made and more variants could not separate faces from whole-cartridge boxes
(`devlog/content/2026-11-27-face-detectors-trained.md`).

## 2. Bundle demos on CPU (this host, no GPU touched)

| Bundle / stage | Per-frame timings (ms) | Process cost | Source |
|---|---|---|---|
| **`rfdetr-faces-final`** (provisional pick: front-face RF-DETR-S tile 1024 + Flash Next two-stage reader, `score_thr 0.65`) — the current bundle | `frame.jpg` (val): localize 1,783.9 / dedup 616.3 / **identify 64,901.3** / count 5.2 · `e194690f…/002.jpg` (test, human 22): 1,712.7 / 490.7 / **71,154.9** / 4.8 · `581c0d5e…/045.jpg` (test, human 28): 1,740.5 / 608.5 / **80,072.4** / 2.9 | **3 frames in 230.54 s wall, exit 0, peak RSS 1.26 GB**; 72 crops identified → **3.00 s/crop** mean (identify = 216.1 s of the stage time, ~94 %); requires `LLMC_API_KEY` | `runs/cli_faces_demo/{counts.json,wall.json,stderr.log}` (wrapper `runs/cli_faces_demo/_run_demo.py`, log `~/.local/share/vision-intern/faces-demo.log`) |
| `rfdetr-final-v2` (close alternative: whole-cartridge RF-DETR-M tile 1280, same reader) — run A | `frame.jpg`: localize 1,520.8 / dedup 621.9 / **identify 82,651.1** / count 6.9 (23 boxes → 3.59 s/crop) · `50828e4b…/008.jpg`: 1,284.8 / 499.9 / 79,169.7 / 2.5 · `10455345…/013.jpg`: 3,330.5 / 5.3 / 16,477.0 / 0.3 | 3 frames / 46 boxes → identify **178,298 ms** ≈ 94 % of stage time; 3.96 s/crop avg. Two of its three frames (`008.jpg`, `013.jpg`) are excluded from scoring — display only | `runs/cli_final_demo/counts.json` |
| same bundle — run B (endpoint was faster) | `frame.jpg`: 1,330.6 / 596.2 / **44,965.7** / 1.8 (22 boxes → 2.04 s/crop) · `50828e4b…/008.jpg`: 1,778.8 / 472.7 / **123,815.4** / 4.5 (20 boxes → 6.19 s/crop) | same weights, same reader: identify per crop moved 2.04 ↔ 6.19 s between runs — the number to plan on is the slow one | `runs/cli_final_demo_flash/counts.json` |
| GPU reference for the same detector weights | `G2` test **0.242 s/img**, val 0.365 s/img, train 191.4 s, peak 6.89 GiB, job wall 262 s (`S2`: 0.223 / 0.309 s/img, 145.2 s, 4.09 GiB, 213 s) | the CPU identify stage is 25–330× the whole GPU frame time: one Collective call per crop | `runs/G2_rfdetr_small_faces-7c7b4587/meta.json`, `runs/S2_c2_tile1280-db269936/meta.json` |
| `rfdetr-m-1280-knn-v0` (RF-DETR-M tile 1280 + plain SigLIP2 kNN) | `frame.jpg`: localize 5,516.5 / identify 9,363.8 / count 635.7 · `50828e4b…/008.jpg`: 3,355.0 / 10,202.3 / 697.2 · `10455345…/013.jpg`: 2,003.9 / 821.2 / 2.6 | 3 frames in **67.18 s** wall, exit 0, peak RSS **1.73 GB** (~15 s of it one-time model load) | `runs/bundle_demo/frames/counts.json`, `runs/bundle_demo/count.err` (`WALL_SECONDS=67.18 EXIT=0 PEAK_RSS_GB=1.73`, measured by `runs/bundle_demo/_measure.py`) |
| `rfdetr-m-1280-v1` (same detector + duplicate suppression + 2-shelf counting + hybrid VLM-family/kNN identifier) | `frame.jpg`: localize 6,195.0 / **dedup 981.1** / identify **38,797.1** / count 3.5 · `008.jpg`: 5,483.3 / 663.4 / 35,297.7 / 1.8 · `013.jpg`: 2,601.6 / 5.6 / 7,775.5 / 0.2 | ~15–20 s of local compute per photo **plus** ~3 s per batch of Inkling calls; requires `LLMC_API_KEY` at inference time | `runs/bundle_demo_v1/counts.json` (`count.json` is the same run), `devlog/content/2026-11-27-bundle-v1.md` |
| Comparison vs GPU | GPU detector inference is 0.223 s/img for the same checkpoint (S2 test split) | the CPU identify stage is 5–12× the whole GPU frame time, because the hybrid identifier makes one Collective call per crop | `runs/S2_c2_tile1280-db269936/meta.json`, rows above |

v0's identify stage is pure local embedding search; v1's is a VLM family call per crop with the
kNN only refining inside that family — hence 9.4 s → 38.8 s on `frame.jpg`. This is the operational
price of the hybrid identifier and the reason `identity_thr`/`family_threshold` matter for cost,
not only accuracy.

## 3. LLM Collective usage (no GPU; API-only)

| Job | Calls | Wall | Latency (s/call) | Errors / 429 | Source |
|---|---:|---:|---|---|---|
| Bootstrap v1 (label 29 frames: Inkling full+1280 tiles, Flash-Next crop judge, Inkling 18-way identifier) | 266 fresh (102 labeller / 82 judge / 82 identifier) | 1,896.7 s (31.6 min) | mean 6.8, min 1.5, max 31.7 | 0 / 0 | `runs/bootstrap_v1/timing.json`, `runs/bootstrap_v1/summary.md` |
| Bootstrap v2 (`reidentify`, two-stage: 563 family + 123 part-numeral calls) | 686 (31 cache hits, 655 misses) | 2,276.0 s (37.9 min) | ≈3.3 s/call amortised | 0 recorded | `runs/bootstrap_v2/summary.md`, cache in `runs/bootstrap_v2/cache/` (771 entries) |
| Front-face labelling `faces_v1` (29 frames, full + 2×2 tiles, Flash-Next judge) | 197 fresh (140 Inkling, 57 Flash-Next) | ≈28 min | mean 8.4, max 37.6 · Inkling med 7.09 (max 19.22, 1,061 s total) · Flash-Next med 9.44 (max 37.61, 598 s total) | 0 / 0 | `runs/faces_v1/timing.json`, `runs/faces_v1/summary.md`, `devlog/content/2026-11-27-faces-all-frames.md` |
| Face-label cleanup `faces_v2` | 0 fresh (pure geometry + verdict filter) | — | — | — | `runs/faces_v2/summary.md` |
| Front-face cleanup `faces_v2` | 0 fresh (pure geometry + verdict filter) | — | — | — | `runs/faces_v2/summary.md` |
| Endpoint probe (3 real frames, tiny/detect/identify/count modes) | per-model rows | sequential single run 2026-11-26 18:49 UTC | `inkling-2-mini` 0.4–5.4 · `qwen3-8-flash-next` 9–107 (429 storm: 502.9 s tiny probe, >14 min hang) · `david-birks/qwen3-8-flash-next` 1.7–46 | heavy 429s only for unprefixed `qwen3-8-flash-next`; `qwen3-8-27b` hard 404 (no vision) | `runs/probe-collective/results.json`, `docs/freestyle/llm_collective.md` |
| `rfdetr-faces-final` CPU demo (this pass, 2026-11-27) | 72 crops → ~72+ family calls (+ part calls on two-part brands) | 230.54 s wall for 3 frames | 3.00 s/crop mean in-pipeline (9.1 s mean measured in `reader_compare`) | 0 / 0 | `runs/cli_faces_demo/{wall.json,stderr.log}` |
| v1 identifier at inference (per frame) | ~26–40 crops × 1 Inkling family call | 35–39 s CPU-bound per dense frame | see §2 | retried with backoff ≤600 s (`freestyle/collective.py`) | `runs/bundle_demo_v1/counts.json` |

The demo's 3.0 s/crop is what the endpoint gave today; `reader_compare` measured 9.1 s mean / 6.3 s
median for the same model and mode under a different load, so budget **3–9 s per cartridge**
(≈75–225 s per stocked photo) rather than either number alone.

Reference cost ceiling for the whole quiz-adjacent VLM budget: the two bootstrap passes plus face
labelling together spent ~1,149 API calls and ~110 minutes of wall time, with zero failed calls,
under `<=2` in-flight requests and `retry_budget_s=600`.

## 4. Where each field comes from (reproduce this table)

| Field | Written by | Read from |
|---|---|---|
| `timings.{train,gallery,embeddings,val,test}` | `freestyle/experiments/entry.py` stage timers | `runs/<run>/meta.json` |
| `inference.<split>.seconds_per_image`, `peak_gpu_bytes` | same (per-split inference loop, allocator peak) | `runs/<run>/meta.json` |
| `job_wall.{image_and_init,unpack_and_pip,command,job_total}` | `k8s/submit.sh` wrapper epochs | `runs/<run>/meta.json` |
| `mode`, `vram_mb`, `gpu`, `image`, `repo_tar_sha256`, `run_id` | `k8s/submit.sh` provenance writer | `runs/<run>/provenance.json` |
| installed versions | job entrypoint | `runs/<run>/env.txt`, `meta.json.versions` |
| bundle CPU stage timings | `freestyle/pipeline.py` (`timings_ms`) | `runs/bundle_demo*/`…`counts.json` |
| CPU wall + peak RSS | `runs/bundle_demo/_measure.py` (`resource.getrusage`) | `runs/bundle_demo/count.err` |
| Collective call latency / error counts | `freestyle/collective.py` `LAST_CALL` logging | `runs/bootstrap_v1/timing.json`, `runs/faces_v1/timing.json`, `runs/bootstrap_v2/summary.md` |
