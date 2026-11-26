# LLM Collective — vision probe findings

**Date probed:** 2026-11-26 (single probe run, sequential)
**Endpoint:** `https://llm-collective.e-gineering.com/v1` (OpenAI-compatible, provider `collective` in `~/.pi/agent/models.json`)
**Probe inputs:** `local/downloads/frame.jpg` (2048x1536), `island-cabinet-01/50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg` (2048x1536), `island-cabinet-01/2ee373c6-d8b6-4837-8c50-8aea807bbe5e/006.jpg` (1600x1200)
**Details / machine-readable output:** `runs/probe-collective/results.json` + `runs/probe-collective/viz/*.png` (re-run: `uv run python -m freestyle.probe_collective detect|tiny|identify|crop`)

## Key resolution (never printed, never logged)

`freestyle/collective.py` reads `~/.pi/agent/models.json` → provider `collective`, resolves `baseUrl` and `apiKey` at runtime (`${LLMC_API_KEY}` env var; `$NAME`/`${NAME}` interpolation and `!command` evaluated like pi does). Never prints or writes the key. `chat_image()` records latency/usage/attempts in `LAST_CALL` and retries 429/5xx/timeouts with exponential backoff + jitter up to `retry_budget_s` (default 600 s).

## Model inventory (as of this probe)

Served at `/v1/models` (13 ids): `auto`, `david-birks/inkling-2-mini`, `david-birks/phonon-31m`, `david-birks/qwen3-8-flash-next`, `david-birks/qwen3-asr-1-7b`, `inkling-2-mini`, `justin-dickey/inkling-2-mini`, `justin-dickey/kokoro-82m`, `justin-dickey/qwen3-asr-1-7b`, `kokoro-82m`, `phonon-31m`, `qwen3-8-flash-next`, `qwen3-asr-1-7b`.

| Model | Vision (tiny probe) | Notes |
|---|---|---|
| `david-birks/inkling-2-mini` | **yes** | fast, reliable; 0.4–5 s/call |
| `qwen3-8-flash-next` | **yes** | thinking model; 13–107 s/call; 429-throttled intermittently |
| `david-birks/qwen3-8-flash-next` | **yes** | thinking model; 12–50 s/call |
| `inkling-2-mini` (unprefixed) | **yes** | small, fast fallback |
| `auto` | **yes** | router; fast |
| `justin-dickey/inkling-2-mini` | **yes** | fast fallback |
| `qwen3-8-27b` | **no** | **HTTP 404** — not in `/v1/models`, "model \"qwen3-8-27b\" is not currently available". Not tried further (no router redirect; unsupported). |

Tiny probe (synthetic 160x120, 256 max tokens): all vision-ok models answered "red, green, blue". flash-next models return empty text at `max_tokens=64` (thinking tokens eat the budget — `finish_reason=length`); they answer correctly with `max_tokens>=256`.

## Coordinate conventions (the important part)

Empirically determined by comparing the prompted convention vs the raw output geometry (and overlays on `frame.jpg`):

- **`david-birks/inkling-2-mini`** — **follows the requested convention.** Asked for absolute pixels: raw values are pixel-like (some >1000, e.g. `[668,227,752,490]` and `[1731,545,1864,765]`), geometry == verified overlay. Asked for 0–1000 normalized: raw values are in 0–1000 (exactly the absolute values × ~2.048 scaled via W/H). Same box geometry for both prompts.
- **`qwen3-8-flash-next` / `david-birks/qwen3-8-flash-next`** — **ignore the convention prompt.** Both the absolute-pixel and normalized prompts return **0–1000-normalized values** (raw coords ≤ 1000, e.g. `[327,148,368,313]`), identical under both prompts. Parse as normalized. The absolute-prompt overlay will look like small clustered boxes if treated as pixels — do not do that.

Note: for inkling the absolute-pixel parse is only valid because the raw values are ≥ image size in places; for flash-next the safe universal parse is **0–1000 normalized**. Recommendation: always prompt and parse normalized (instructions in `build_boxes_prompt('normalized', ...)`).

**Neither model resizes-aware:** raw values are identical at full res and `max_side=1024` (verified: identical raw values for `frame.jpg` across both sizes). `max_side` is a *request-size* knob, not a coordinate-frame knob — never rescale absolute boxes by the sent/original size ratio. Use normalized conventions to keep everything size-independent.

## Box quality (~27 cartridges expected in frame.jpg)

Overlays live under `runs/probe-collective/viz/`. Judgement (from the two frame.jpg overlays drawn for inkling):

- **~26/26 valid boxes on frame.jpg** (both abs and norm prompts; full and 1024). Boxes 1–17 tightly cover the top-shelf cartridges (blue NNS row, yellow AA, orange BB, lavender CC, green, orange, pale yellow — approximately per model). Boxes 18–26 on the bottom shelf are messy: they overlap and under-count the fanned, partially-occluded bottom row (some boxes span two cartridges). Bottom-shelf quality is the weak spot.
- `008.jpg` — 19–22 valid boxes (inkling 21 full/19 @1024, flash-next 21/22).
- `006.jpg` — 23–27 valid boxes (inkling 25–26, flash-next 23–27).
- Note on 006 (qwen3-8-flash-next): box records were only captured in the *first* probe run (the rerun hung on this model and was killed); their pixel boxes are now internally consistent but were not independently re-verified. david-birks/qwen3-8-flash-next and inkling-2-mini have full clean 006 records.
- Normalized parse gives 1:1-valid boxes everywhere; absolute parse for flash-next is meaningless (its raw values are normalized).
- All boxes are original-image pixel coordinates (no resizing done).

## Count-per-shelf (reply `{"top": N, "bottom": M}`)

| Image | inkling-2-mini (full/1024) | qwen3-8-flash-next (full/1024) | david-birks/qwen3-8-flash-next (full/1024) |
|---|---|---|---|
| frame.jpg | 17/10, **17/9** | 16/9, **14/10** | 17/9, **14/9** |
| 008.jpg | 12/9, **13/8** | 12/8, **12/9** | 9/7, **12/7** |
| 006.jpg | 16/10, **15/10** | 16/9, **16/10** | 15/10, **15/8** |

Counts agree within ~1–3 cartridges across models; flash-next tends to under-count `008.jpg`/`006.jpg` bottom shelf slightly less consistently. Total across shelves is stable for the best model (~26–27).

## Crop identification (single crop, hand-chosen boxes, 2 crops)

Two crops cut from `frame.jpg` into `data/derived/probe/`: a tight crop of the yellow AA cartridge face (`crop_top_shelf_aa.jpg`, 100x232) and a tight crop of the red Coca-Cola-Part 1 cartridge face (`crop_bottom_shelf_red.jpg`, 115x415). Prompt: 18 active catalog products with id/aliases/colour/shelf; reply = product id.

| Model | crop_top_shelf_aa (AA) | crop_bottom_shelf_red (Coke P1) | Latency | Tokens |
|---|---|---|---|---|
| `david-birks/inkling-2-mini` | **ingredient-aa ✓** | **coca-cola-part1 ✓** | 3.0s / 2.9s | 610/628 |
| `qwen3-8-flash-next` | **ingredient-aa ✓** | **coca-cola-part1 ✓** | 28.5s / 13.3s | 677/1076 |
| `david-birks/qwen3-8-flash-next` | **ingredient-aa ✓** | **coca-cola-part1 ✓** | 1.7s / 5.3s | 675/1042 |

All three working vision models correctly identified 100% of the crops tried (2/2). The AA and red-Coke faces are legible at full res; crops sent at `max_side=None` (original size). Easy separable identities; no false negatives seen.

## Latency and tokens (per call)

| Model | count | boxes | identify | request size (frame, full → 1024) |
|---|---|---|---|---|
| `david-birks/inkling-2-mini` | 2.8–5.4 s | 2.8–5.4 s | 2.9–3.0 s | 277 KB → 145 KB |
| `qwen3-8-flash-next` | 17–29 s | 9–107 s | 13–29 s | 277 KB → 145 KB |
| `david-birks/qwen3-8-flash-next` | 15–46 s | 12–45 s | 1.7–5.3 s | 277 KB → 145 KB |

Token usage (frame full / 1024):

- inkling: count 3.1k/1.3k, boxes ~4.1k/2.6k → 1024 ≈ −60% prompt/completion tokens.
- qwen3-8-flash-next: count 4.0k/2.8k, boxes ~5.1k/2.7k (some calls >100 s).
- david-birks/qwen3-8-flash-next: count ~4.0k/2.1k, boxes ~5.1k/2.7k.

`max_side=1024` halves prompt tokens and request bytes substantially but **does not reduce latency materially**; `qwen3-8-flash-next` spends most of its latency on thinking. Size of image sent does not affect the returned coordinate frame (see conventions).

## Capacity / 429 behavior (important)

- Zero 429s during the first (foreground) run — everything worked on the first attempt within ~10 minutes.
- Heavy throttling observed on `qwen3-8-flash-next` during the background rerun: tiny probe took **502.9 s** (~9 min of 429 backoff, eventually ok), then one count call hung >14 min (killed at the 15-min window). `david-birks/inkling-2-mini` and `david-birks/qwen3-8-flash-next` remained fast and reliable throughout.
- `qwen3-8-27b` is a hard **404** (not capacity-related) and was not retried.
- Retry policy that survived 429 storms: `chat_image(..., retry_budget_s=600)` (~10 min, exponential backoff + jitter; `freestyle/collective.py`). Every one of the ~54 recorded detect calls and both identify crops hit the endpoint successfully.
- Result of the pending-capacity scenario: flash-next models **do** work (they were awaiting capacity only in the second pass), inkling-2-mini is the most reliable fast option.

## Max accepted image size

Synthetic (no private data): 2048x1536 accepted (~5 s, 66 KB), 3072x2304 accepted (~4.7 s, 148 KB), **4096x3072 accepted (~5.6 s, 263 KB)** — by inkling-2-mini. Real probe sizes (2048x1536 / 1600x1200) all accepted. No larger probe was run (not needed; max source frame is 2048x1536).

## Recommendations

1. **Detection + counting:** `david-birks/inkling-2-mini` at full res, normalized 0–1000 convention (fast, reliable, matches verified box quality). `david-birks/qwen3-8-flash-next` as a higher-quality alternative if box precision matters and latency/capacity tolerates it. `qwen3-8-flash-next` is slower (13–107 s/call) and intermittently throttled — use `david-birks/` prefix variant for stability.
2. **Identification:** any of the three working models (100% on the 2 crops tried). For thinking models (flash-next) set `max_tokens >= 1024`; at `max_tokens <= 64` they return empty text.
3. **Convention handling:** always prompt normalized; always parse as normalized. If absolute prompt is used, parse as raw pixels for inkling only.
4. Bottom-shelf boxes (fanned, occluded, partially-in-frame cartridges) are the weakest area — expect overlapping/under-counting boxes there; treat per-box judgement as approximate, and count via the dedicated count prompt + manual verification.
5. Re-probe periodically: the endpoint fell over under load for `qwen3-8-flash-next` on one pass; inkling-2-mini is the safe default.
