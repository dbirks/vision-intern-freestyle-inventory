# Phone-first ground-truth quiz

A one-question-at-a-time web app for building human-verified ground truth from a
phone in ~20 minutes. It is the fast alternative to the drag-boxes
[review app](REVIEW_APP.md): you never draw or move a box — you confirm the
model's count, then confirm or correct the identity of one zoomed crop at a time.
Styled with the dev log's tokens (Young Serif / Manrope / DM Mono, Coke-red
accent, light + dark).

Ground-truth rules come from [GT_PROTOCOL.md](GT_PROTOCOL.md); the file formats
from [ARCHITECTURE.md](ARCHITECTURE.md). Model proposals are never GT — only the
COCO that `to_coco` writes under `data/gt/` is.

## Run

```bash
# 1. turn proposals into a question queue (CPU, ~10 s for 300+ boxes)
uv run python -m freestyle.quiz.build_queue \
    --proposals data/labels/bootstrap_v2.json \
    --splits data/splits.json --splits-order test,val \
    --out data/gt/quiz_queue.json

# 2. serve the quiz, bind to the phone-reachable address (tailscale IP here)
uv run python -m freestyle.quiz.server \
    --queue data/gt/quiz_queue.json \
    --answers data/gt/quiz_answers.jsonl \
    --host 100.89.241.57 --port 8766

# 3. export human-verified COCO + per-split count checks
uv run python -m freestyle.quiz.to_coco --queue data/gt/quiz_queue.json \
    --answers data/gt/quiz_answers.jsonl --out data/gt/test.json --split test
uv run python -m freestyle.quiz.to_coco --queue data/gt/quiz_queue.json \
    --answers data/gt/quiz_answers.jsonl --out data/gt/val.json --split val
```

`--host` defaults to `127.0.0.1`; keep the port closed to the public internet.
Everything under `data/` is gitignored and private — no image ever leaves the
LAN. Crops are rendered from ORIGINAL pixels in memory and never written back;
nothing in `data/raw/` is opened for writing or re-encoded on disk.

## Question types and queue order

`build_queue` emits one `count` question per image plus one `crop` question per
proposed box, ordered deterministically so two sessions see the same numbering:
all images of the first requested split (`test`), then the next (`val`), and
inside an image the count question first, then its crops **left→right with the
top shelf before the bottom shelf**. Shelf bands come from
`freestyle.count.shelves.assign_shelves` (the same fisheye row fit the counting
pipeline uses), falling back to a position split for frames where it finds only
one band. Images outside the requested splits, unreadable files and degenerate
boxes are dropped.

**count** — the whole frame downscaled to ≤1280 px with the model's boxes drawn
thin, haloed and numbered (1..N in crop order, so the number is also the answer
order). Asks: *Model counts N top / M bottom cartridges (T total). Correct?*
Buttons: `✓ Yes`, `✎ Fix the numbers` (two steppers, top/bottom),
`Not a usable cabinet view`, `Skip`.

**crop** — one box at a time: the crop from ORIGINAL pixels with 15 % padding,
upscaled to fill the phone width (capped at 6× so tiny boxes do not turn into
soup, and scrollable/zoomable because tall crops keep their aspect), a context
thumbnail of the frame with that box highlighted, and the guess in big serif
type with its confidence. Buttons: `✓ Correct`, the 18 active catalog classes as
tap targets with their catalog `color_hint` swatch, `Can't tell (unknown)`,
`Not a cartridge`, `Box covers 2+ cartridges`, `Box cuts a cartridge (partial)`,
`Skip`. When the model guessed a two-part family without the part (`unknown` +
`id_family` of Coca-Cola / Diet Coke / Dr Pepper / Coca-Cola Zero), three big
buttons come first: `Part 1`, `Part 2`, `Can't tell`.

## Answers, resume, undo

Every answer is one JSON line appended to `--answers` (`data/gt/quiz_answers.jsonl`)
with `qid`, `image_id`, `ann_id`, `split`, the model `guess` + its confidence and
the `answer` payload (including an ISO timestamp). The append is a single
buffered write + `fsync` under a lock, so a killed server loses nothing.

* **Resume** — the server replays the log at startup and returns
  `next_index`; opening the app jumps there. Re-answering a question appends a
  new line and the last one wins.
* **Undo** — `POST /api/undo` drops the last log line and atomically rewrites
  the file (temp + `fsync` + `os.replace`), which walks back through re-answers.
* **Prefetch** — the next three crops and context thumbnails are warmed
  server-side and in the browser cache, so tapping forward feels instant.
* Keyboard (desktop): `y` correct, `1`–`9` class, `f` fix panel, `x`/`n` not a
  cabinet view / not a cartridge, `s` skip, `u` undo, `Esc` close the zoom.
* `GET /img/frame?i=N&full=1`, `/img/crop?i=N`, `/img/context?i=N` take
  *indices into the queue*, never paths; image ids are resolved under
  `--images-root` with the review app's `safe_join`, so `..` traversal is refused
  (400) and an id outside the queue is never opened.

## What `to_coco` writes

Boxes always come from the proposal (`attributes.box_source = "model"`, original
pixels); identity comes from the human (`attributes.identity_source = "human"`,
and `source = "quiz:human"`). Category is the human answer: `Correct` keeps the
model's guess, a class/part tap replaces it. `Not a cartridge` drops the box;
`Box covers 2+ cartridges` keeps it as `unknown` with `box_covers_2_plus` and
`identity_visible = false`; `Box cuts a cartridge (partial)` keeps the identity
and sets `box_partial` + `truncated`. Image attributes carry
`human_counts = {top, bottom, total}` (from `Yes` or the steppers), the model's
counts for comparison, `usable`, and `verified` — true only when the count
question and **every** crop of that image carry a human identity, because a
skipped box would otherwise smuggle a model label into GT.
`Not a usable cabinet view` sets `scene = "no_cabinet"` and drops the frame's
boxes (a GT_PROTOCOL negative).

Alongside the COCO it writes `<split>_counts.json`: per image the human total,
the per-class counts derived from human-answered crops, and `counts_mismatch`
when the two disagree — that is the signal that boxes were missed or merged and
the frame needs the review app. Unverified images are listed too.

## Test / development

```bash
uv run pytest tests/unit/freestyle/test_quiz.py -q   # queue order, append/undo/resume, every button, traversal
uv run ruff check freestyle/quiz
node --check freestyle/quiz/static/quiz.js
```
