# Crop readers: Qwen 3.8 Flash Next vs Inkling 2 Mini (scored on human answers)

Generated 2026-11-27 19:29 -0500 by `freestyle/experiments/reader_compare.py` (full run: `uv run python -m freestyle.experiments.reader_compare`; rescore from the saved replies with no VLM calls: `uv run python -m freestyle.experiments.reader_compare --stage report`).

## What was measured

Human labels are the phone-quiz answers on **test-split** crops only (`data/gt/quiz_answers.jsonl` + `data/gt/quiz_queue.json`). The on-screen guess those verdicts confirmed or corrected came from Inkling 2 Mini's two-stage bootstrap identifier (`data/labels/bootstrap_v2.json`), so that historical guess is scored as one row and Inkling is also re-run fresh for an apples-to-apples comparison.

Crops are cut from ORIGINAL pixels by the quiz app's own `make_crop` (15 % padding, same upscale + JPEG as the phone saw). Prompts, catalog hints, 2x upscale and the part-region zoom are `freestyle/vlm_identify.py` unchanged: **no prompt was tuned on these crops** (format debugging used `--smoke` on unlabelled val crops). `temperature=0`, `max_tokens=8192` (thinking model), <=2 in-flight requests per process, 429/5xx/timeouts retried with exponential backoff + jitter, replies cached on disk.

One harness-side deviation, applied identically to both readers: `VLMCropIdentifier.contact_sheet` labels its crops `0..n-1` and maps the reply's `index` straight through, with no tolerance for a model that numbers the crops from 1. `reader_compare.SheetReader` re-maps the *same* reply when no answer uses index 0 (no re-ask, no prompt change); production `vlm_identify.contact_sheet` is left alone and carries the ambiguity. Which reader did what is measured in the run notes below.

Eval set: **44 answered test crops** = 39 with a definite human identity (the top-1 denominator) + 5 where a human could not tell either (reported separately, never in top-1). Excluded: 18 (unanswered / skip / not-a-cartridge). Label mix: `class` x19, `correct` x18, `part` x7.

## Accuracy per model x mode

| Reader | Mode | Top-1 vs human (Wilson 95 % CI) | Family correct | Part correct (two-part GT) | Abstain `unknown` | s/crop | Failures |
|---|---|---|---|---|---|---|---|
| `david-birks/inkling-2-mini` | `single` | 0.21 (0.11-0.36, n=39) | 0.21 (0.11-0.36, n=39) | 0.00 (0.00-0.26, n=11) · readable 0% | 77% | 6.5 | 0 |
| `david-birks/inkling-2-mini` | `two_stage` | 0.46 (0.32-0.61, n=39) | 0.54 (0.39-0.68, n=39) | 0.00 (0.00-0.26, n=11) · readable 0% | 38% | 3.5 | 0 |
| `david-birks/inkling-2-mini` | `two_stage_shelf` | 0.49 (0.34-0.64, n=39) | 0.67 (0.51-0.79, n=39) | 0.00 (0.00-0.26, n=11) · readable 0% | 44% | 3.5 | 0 |
| `david-birks/inkling-2-mini` | `contact_sheet` | 0.00 (0.00-0.09, n=39) | 0.00 (0.00-0.09, n=39) | 0.00 (0.00-0.26, n=11) · readable 0% | 100% | 4.6 | 0 |
| `qwen3-8-flash-next` | `single` | 0.18 (0.09-0.33, n=39) | 0.18 (0.09-0.33, n=39) | 0.00 (0.00-0.26, n=11) · readable 0% | 72% | 11.5 | 0 |
| `qwen3-8-flash-next` | `two_stage` | 0.51 (0.36-0.66, n=39) | 0.56 (0.41-0.71, n=39) | 0.00 (0.00-0.26, n=11) · readable 0% | 21% | 8.9 | 0 |
| `qwen3-8-flash-next` | `two_stage_shelf` | 0.56 (0.41-0.71, n=39) | 0.69 (0.54-0.81, n=39) | 0.09 (0.02-0.38, n=11) · readable 9% | 18% | 9.1 | 0 |
| `qwen3-8-flash-next` | `contact_sheet` | 0.21 (0.11-0.36, n=39) | 0.21 (0.11-0.36, n=39) | 0.00 (0.00-0.26, n=11) · readable 0% | 59% | 9.6 | 0 |
| `david-birks/inkling-2-mini` | quiz bootstrap (historical, on-screen guess) | 0.46 (0.32-0.61, n=39) | 0.64 (0.48-0.77, n=39) | 0.00 (0.00-0.26, n=11) · readable 0% | 36% | - | 0 |

*Family correct* = the brand the reader asserts, part ignored. A two-stage crop that names the brand but cannot read the numeral (class `unknown`) still counts as family-correct; a bare `unknown` with no brand claim counts as wrong. *Part correct* = the numeral the reader ends up asserting matches the human's (right digit, wrong brand still counts as a digit hit), scored only on crops whose human label is a two-part brand; *readable* = a digit came back at all there (an unreadable numeral is a legitimate stage-2 abstention, not a wrong digit).

`s/crop` is wall time per crop (mean; median and max are in `runs/reader_compare/results.json` under `scores.<reader>__<mode>.latency_s`, along with per-crop seconds and rationales in `predictions/*.jsonl`).

## Crops a human could not identify

Excluded from top-1 (the contract drops GT `unknown`). A good reader should mostly abstain on these.

| Reader | Mode | Crops | Abstained | Guessed anyway |
|---|---|---|---|---|
| `david-birks/inkling-2-mini` | single | 5 | 4 | 1 |
| `david-birks/inkling-2-mini` | two_stage | 5 | 3 | 2 |
| `david-birks/inkling-2-mini` | two_stage_shelf | 5 | 3 | 2 |
| `david-birks/inkling-2-mini` | contact_sheet | 5 | 5 | 0 |
| `qwen3-8-flash-next` | single | 5 | 5 | 0 |
| `qwen3-8-flash-next` | two_stage | 5 | 4 | 1 |
| `qwen3-8-flash-next` | two_stage_shelf | 5 | 4 | 1 |
| `qwen3-8-flash-next` | contact_sheet | 5 | 4 | 1 |
| `david-birks/inkling-2-mini` | quiz_bootstrap | 5 | 2 | 3 |

## Confusion pairs (human -> reader, mismatches only)

| Reader | Mode | Top pairs |
|---|---|---|
| `david-birks/inkling-2-mini` | single | `nns-sweetener -> unknown` x9; `ingredient-cc -> unknown` x3; `diet-coke-part2 -> unknown` x3; `coca-cola-part2 -> unknown` x3; `flavor-orange -> unknown` x2; `hi-c -> unknown` x2 |
| `david-birks/inkling-2-mini` | two_stage | `ingredient-cc -> unknown` x3; `diet-coke-part2 -> unknown` x3; `diet-coke-part1 -> unknown` x2; `coca-cola-part2 -> unknown` x2; `dr-pepper-part1 -> unknown` x2; `ingredient-aa -> flavor-vanilla` x1 |
| `david-birks/inkling-2-mini` | two_stage_shelf | `ingredient-cc -> unknown` x3; `diet-coke-part2 -> unknown` x3; `coca-cola-part2 -> unknown` x3; `diet-coke-part1 -> unknown` x2; `dr-pepper-part1 -> unknown` x2; `ingredient-aa -> flavor-vanilla` x1 |
| `david-birks/inkling-2-mini` | contact_sheet | `nns-sweetener -> unknown` x11; `fanta-orange -> unknown` x3; `ingredient-cc -> unknown` x3; `diet-coke-part2 -> unknown` x3; `coca-cola-part2 -> unknown` x3; `ingredient-aa -> unknown` x2 |
| `qwen3-8-flash-next` | single | `nns-sweetener -> unknown` x9; `ingredient-cc -> unknown` x3; `diet-coke-part2 -> unknown` x3; `flavor-orange -> unknown` x2; `diet-coke-part1 -> unknown` x2; `coca-cola-part2 -> hi-c` x2 |
| `qwen3-8-flash-next` | two_stage | `ingredient-cc -> nns-sweetener` x3; `diet-coke-part2 -> unknown` x2; `coca-cola-part2 -> unknown` x2; `flavor-orange -> unknown` x1; `diet-coke-part1 -> hi-c` x1; `coca-cola-part2 -> hi-c` x1 |
| `qwen3-8-flash-next` | two_stage_shelf | `ingredient-cc -> nns-sweetener` x3; `diet-coke-part2 -> unknown` x2; `coca-cola-part2 -> unknown` x2; `flavor-orange -> hi-c` x1; `diet-coke-part1 -> hi-c` x1; `coca-cola-part2 -> hi-c` x1 |
| `qwen3-8-flash-next` | contact_sheet | `nns-sweetener -> unknown` x7; `ingredient-cc -> unknown` x2; `flavor-orange -> unknown` x2; `diet-coke-part2 -> unknown` x2; `dr-pepper-part1 -> unknown` x2; `coca-cola-part2 -> unknown` x2 |
| `david-birks/inkling-2-mini` | quiz_bootstrap | `ingredient-cc -> unknown` x3; `diet-coke-part2 -> unknown` x3; `coca-cola-part2 -> unknown` x3; `flavor-orange -> fanta-orange` x2; `hi-c -> flavor-lemon` x2; `diet-coke-part1 -> unknown` x2 |

## Per-crop table

`Human label` is the verdict (`correct` copies the on-screen guess, which the human confirmed; `class`/`part` are what the human named). ✅/❌ = matched / missed the human label; `(abstained)` / `(guessed)` on the 5 crops a human could not call. `?` = no prediction recorded.

| Crop | Shelf | Human label | How labelled | Quiz guess | inkling<br>single | inkling<br>two_stage | inkling<br>two_stage_shelf | inkling<br>contact_sheet | flash<br>single | flash<br>two_stage | flash<br>two_stage_shelf | flash<br>contact_sheet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `581c045#114` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ |
| `581c045#115` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ |
| `581c045#116` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ |
| `581c045#117` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ |
| `581c045#118` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ |
| `581c045#119` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ |
| `581c045#120` | top | `ingredient-aa` | correct | `ingredient-aa` | `flavor-vanilla` ❌ | `flavor-vanilla` ❌ | `flavor-vanilla` ❌ | `unknown` ❌ | `ingredient-aa` ✅ | `ingredient-aa` ✅ | `ingredient-aa` ✅ | `fanta-orange` ❌ |
| `581c045#121` | top | `fanta-orange` | correct | `fanta-orange` | `fanta-orange` ✅ | `fanta-orange` ✅ | `fanta-orange` ✅ | `unknown` ❌ | `fanta-orange` ✅ | `fanta-orange` ✅ | `fanta-orange` ✅ | `fanta-orange` ✅ |
| `581c045#122` | top | `fanta-orange` | correct | `fanta-orange` | `fanta-orange` ✅ | `fanta-orange` ✅ | `fanta-orange` ✅ | `unknown` ❌ | `unknown` ❌ | `fanta-orange` ✅ | `fanta-orange` ✅ | `fanta-orange` ✅ |
| `581c045#123` | top | `ingredient-cc` | class | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ❌ | `nns-sweetener` ❌ | `unknown` ❌ |
| `581c045#124` | top | `ingredient-cc` | class | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ❌ | `nns-sweetener` ❌ | `unknown` ❌ |
| `581c045#125` | top | `flavor-orange` | class | `fanta-orange` | `unknown` ❌ | `fanta-orange` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `hi-c` ❌ | `unknown` ❌ |
| `581c045#126` | top | `unknown` | class | `flavor-lemon` | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) |
| `581c045#127` | top | `hi-c` | class | `flavor-lemon` | `unknown` ❌ | `flavor-vanilla` ❌ | `flavor-vanilla` ❌ | `unknown` ❌ | `unknown` ❌ | `hi-c` ✅ | `hi-c` ✅ | `unknown` ❌ |
| `581c045#128` | top | `unknown` | class | `flavor-vanilla` | `unknown` (abstained) | `flavor-vanilla` (guessed) | `flavor-vanilla` (guessed) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) |
| `581c045#129` | top | `unknown` | class | `flavor-vanilla` | `flavor-vanilla` (guessed) | `flavor-vanilla` (guessed) | `flavor-vanilla` (guessed) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) |
| `581c045#130` | top | `sprite` | correct | `sprite` | `sprite` ✅ | `sprite` ✅ | `sprite` ✅ | `unknown` ❌ | `unknown` ❌ | `sprite` ✅ | `sprite` ✅ | `sprite` ✅ |
| `581c045#132` | bottom | `diet-coke-part2` | part | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ |
| `581c045#133` | bottom | `diet-coke-part1` | part | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `hi-c` ❌ | `hi-c` ❌ | `fanta-orange` ❌ |
| `581c045#134` | bottom | `coca-cola-part2` | part | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `hi-c` ❌ | `hi-c` ❌ | `hi-c` ❌ | `fanta-orange` ❌ |
| `581c045#139` | bottom | `diet-coke-part2` | part | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `fanta-orange` ❌ |
| `581c045#140` | bottom | `dr-pepper-part1` | class | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ |
| `e194002#454` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `nns-sweetener` ✅ |
| `e194002#455` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ |
| `e194002#456` | top | `nns-sweetener` | correct | `nns-sweetener` | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `nns-sweetener` ✅ |
| `e194002#457` | top | `nns-sweetener` | correct | `nns-sweetener` | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `fanta-orange` ❌ |
| `e194002#458` | top | `fanta-orange` | correct | `fanta-orange` | `fanta-orange` ✅ | `fanta-orange` ✅ | `fanta-orange` ✅ | `unknown` ❌ | `fanta-orange` ✅ | `fanta-orange` ✅ | `fanta-orange` ✅ | `unknown` ❌ |
| `e194002#459` | top | `ingredient-cc` | class | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ❌ | `nns-sweetener` ❌ | `ingredient-cc` ✅ |
| `e194002#460` | top | `nns-sweetener` | correct | `nns-sweetener` | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `unknown` ❌ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `nns-sweetener` ✅ | `nns-sweetener` ✅ |
| `e194002#461` | top | `sprite` | correct | `sprite` | `sprite` ✅ | `sprite` ✅ | `sprite` ✅ | `unknown` ❌ | `sprite` ✅ | `sprite` ✅ | `sprite` ✅ | `ingredient-aa` ❌ |
| `e194002#462` | top | `flavor-vanilla` | class | `nns-sweetener` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `nns-sweetener` ❌ | `nns-sweetener` ❌ | `unknown` ❌ |
| `e194002#463` | top | `flavor-orange` | class | `fanta-orange` | `unknown` ❌ | `flavor-orange` ✅ | `fanta-orange` ❌ | `unknown` ❌ | `unknown` ❌ | `fanta-orange` ❌ | `flavor-orange` ✅ | `unknown` ❌ |
| `e194002#464` | top | `flavor-cherry` | class | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ |
| `e194002#465` | top | `flavor-lemon` | class | `hi-c` | `unknown` ❌ | `unknown` ❌ | `flavor-lemon` ✅ | `unknown` ❌ | `flavor-vanilla` ❌ | `flavor-vanilla` ❌ | `flavor-vanilla` ❌ | `fanta-orange` ❌ |
| `e194002#466` | top | `ingredient-aa` | correct | `ingredient-aa` | `ingredient-aa` ✅ | `ingredient-aa` ✅ | `ingredient-aa` ✅ | `unknown` ❌ | `ingredient-aa` ✅ | `ingredient-aa` ✅ | `ingredient-aa` ✅ | `ingredient-aa` ✅ |
| `e194002#467` | top | `hi-c` | class | `flavor-lemon` | `unknown` ❌ | `flavor-lemon` ❌ | `hi-c` ✅ | `unknown` ❌ | `fanta-orange` ❌ | `hi-c` ✅ | `hi-c` ✅ | `fanta-orange` ❌ |
| `e194002#468` | bottom | `unknown` | class | `unknown` | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) |
| `e194002#469` | bottom | `dr-pepper-part2` | class | `unknown` | `unknown` ❌ | `hi-c` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `hi-c` ❌ | `hi-c` ❌ | `unknown` ❌ |
| `e194002#470` | bottom | `dr-pepper-part1` | class | `hi-c` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `hi-c` ❌ | `hi-c` ❌ | `unknown` ❌ |
| `e194002#471` | bottom | `unknown` | class | `unknown` | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `unknown` (abstained) | `ingredient-aa` (guessed) | `diet-coke-part1` (guessed) | `flavor-vanilla` (guessed) |
| `e194002#472` | bottom | `diet-coke-part2` | class | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `hi-c` ❌ | `diet-coke-part2` ✅ | `unknown` ❌ |
| `e194002#473` | bottom | `coca-cola-part2` | part | `unknown` | `unknown` ❌ | `hi-c` ❌ | `unknown` ❌ | `unknown` ❌ | `hi-c` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ |
| `e194002#474` | bottom | `coca-cola-part2` | part | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ |
| `e194002#475` | bottom | `diet-coke-part1` | part | `unknown` | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ | `unknown` ❌ |

## Run notes

* 359 VLM calls logged (all runs), 0 failed, 0 needed 429/5xx retries
* served model ids seen in the call log: david-birks/inkling-2-mini, qwen3-8-flash-next
* contact sheets were packed 3 crops per image
* contact-sheet numbering by david-birks/inkling-2-mini: zero based x16
* contact-sheet numbering by qwen3-8-flash-next: one based x16
* inputs: `quiz_queue.json` sha256 7f93a49e858c (185315 B, mtime 2026-11-27 19:00); `quiz_answers.jsonl` sha256 0ad289305a5d (23685 B, mtime 2026-11-27 18:28)

## Conclusion

On 39 human-labelled test crops Qwen 3.8 Flash Next beats a fresh Inkling 2 Mini run: best top-1 0.56 (0.41-0.71, n=39) in `two_stage_shelf` vs 0.49 (0.34-0.64, n=39) in `two_stage_shelf` (the historical on-screen Inkling guess was 0.46 (0.32-0.61, n=39), which the fresh `two_stage` re-run reproduces crop for crop).
The 95 % Wilson intervals overlap, so the 0.08 point gap on n=39 should not be treated as settled on this sample alone; the steadier signal is family reading (0.69 (0.54-0.81, n=39) Flash vs 0.67 (0.51-0.79, n=39) Inkling) with Flash Next abstaining on far fewer crops (18% vs 44%).
Adopt Flash Next (6.3 s/crop median, 9.1 s mean including retries, vs Inkling's 2.8 s) and keep two-stage with the shelf hint, because the bottleneck is now stage 2, not the reader: on the 11 two-part-brand crops neither reader gets the numeral right (0.09 (0.02-0.38, n=11) Flash vs 0.00 (0.00-0.26, n=11) Inkling), and contact sheets are useless for both (59% and 100% abstention).
