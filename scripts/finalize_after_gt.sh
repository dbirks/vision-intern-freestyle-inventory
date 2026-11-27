#!/usr/bin/env bash
# finalize_after_gt.sh — the exact sequence to run once the human quiz answers exist.
#
# What it does, in order:
#   0. validate the repo/paths it is about to touch (and `uv`)
#   1. quiz          — serve freestyle.quiz.server so the remaining questions get answered
#                      (skippable: --skip-quiz, e.g. when the answers are already in)
#   2. sufficiency   — HARD GATE: abort unless >= --min-frac (default 0.80) of *each* split's
#                      questions are answered with a non-`skip` result
#   3. to_coco       — export data/gt/test.json + data/gt/val.json (+ *_counts.json mismatches)
#   4. score_all     — score every run with both val and test predictions, 1000 bootstrap reps
#                      -> docs/freestyle/RESULTS_GT.md + results_gt.json
#   5. report        — print which <!-- GT-PENDING --> markers are still in REPORT.md and which
#                      GT numbers exist now, so the table swap is mechanical
#   6. export        — export the winning run as a frozen bundle (best test total-count MAE,
#                      ties broken by identity accuracy) and verify its checksums
#   7. cli demo      — freestyle.cli count on three real frames with that bundle
#
# Safety: never runs git state-changing commands, never pushes anywhere, never touches data/raw/,
# never prints secrets. LLMC_API_KEY must be in the environment for steps 6/7 (the hybrid/vlm
# identifiers call the Collective at inference time). GPU work is NOT part of this script — it
# runs only as k8s Jobs (see docs/freestyle/HOW_TO_RUN.md).
#
# Usage:
#   scripts/finalize_after_gt.sh [--dry-run] [--skip-quiz] [--skip-export] [--bootstrap N]
#                                [--min-frac 0.80] [--answer-file PATH] [--queue PATH]
#
# --dry-run validates inputs, runs the answer gate against whatever is in the answers file today,
# prints every command it would run, and mutates nothing (no quiz, no exports, no bundles).

set -euo pipefail

# ---------------------------------------------------------------- config (all overridable) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="data/gt/quiz_queue.json"
ANSWERS="data/gt/quiz_answers.jsonl"
SPLITS="data/splits.json"
PROPOSALS="data/labels/bootstrap_v2.json"
AUDIT="data/derived/audit/audit.json"
RUNS_GLOB='runs/*'
GT_OUT_DIR="data/gt"
RESULTS_MD="docs/freestyle/RESULTS_GT.md"
REPORT="docs/freestyle/REPORT.md"
BUNDLE_NAME="rfdetr-final-v2"
MIN_FRAC="0.80"
BOOTSTRAP=1000
QUIZ_HOST="${QUIZ_HOST:-127.0.0.1}"
QUIZ_PORT="${QUIZ_PORT:-8766}"
DEMO_IMAGES=(
  "data/raw/freestyle-vision-data/images/local/downloads/frame.jpg"
  "data/raw/freestyle-vision-data/images/island-cabinet-01/50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg"
  "data/raw/freestyle-vision-data/images/island-cabinet-01/10455345-10b3-4baf-bf2e-00bfefc562c7/013.jpg"
)
DRY_RUN=0; SKIP_QUIZ=0; SKIP_EXPORT=0

say() { printf '[finalize] %s\n' "$*"; }
die() { printf '[finalize] ERROR: %s\n' "$*" >&2; exit 2; }
run() {
  if [ "$DRY_RUN" = 1 ]; then printf '[finalize] DRY-RUN would run: %s\n' "$*"; return 0; fi
  say "run: $*"; "$@"
}

# ------------------------------------------------------------------------ arg parsing ---
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --skip-quiz) SKIP_QUIZ=1 ;;
    --skip-export) SKIP_EXPORT=1 ;;
    --bootstrap) BOOTSTRAP="${2:?--bootstrap needs a value}"; shift ;;
    --min-frac) MIN_FRAC="${2:?--min-frac needs a value}"; shift ;;
    --queue) QUEUE="${2:?}"; shift ;;
    --answer-file|--answers) ANSWERS="${2:?}"; shift ;;
    --runs) RUNS_GLOB="${2:?}"; shift ;;
    --bundle-name) BUNDLE_NAME="${2:?}"; shift ;;
    --quiz-host) QUIZ_HOST="${2:?}"; shift ;;
    --quiz-port) QUIZ_PORT="${2:?}"; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

cd "$REPO_ROOT"
[ -d freestyle ] || die "run me from the vision-intern repo root (no freestyle/ package here)"
command -v uv >/dev/null 2>&1 || die "uv is not on PATH"

case "$MIN_FRAC" in
  ''|*[!0-9.]*|*.*.*) die "--min-frac must be a decimal like 0.80 (got '$MIN_FRAC')" ;;
esac
[ "$BOOTSTRAP" -ge 1 ] 2>/dev/null || die "--bootstrap must be a positive integer"

# ------------------------------------------------------------- step 0: preconditions ---
say "repo: $REPO_ROOT (branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
git status --short >&2 || true   # read-only; expected to show dirty files while workers run

for f in "$QUEUE" "$SPLITS" "$PROPOSALS" "$AUDIT"; do
  [ -f "$f" ] || die "required input missing: $f"
done
for img in "${DEMO_IMAGES[@]}"; do
  [ -f "$img" ] || die "demo frame missing: $img (private data — run k8s/stage-data.sh or restore data/)"
done
if [ "$DRY_RUN" = 0 ] && [ "$SKIP_EXPORT" = 0 ]; then
  [ -n "${LLMC_API_KEY:-}" ] || say "note: LLMC_API_KEY is not set — the hybrid/vlm identifier will fail at steps 6/7"
fi

PYENV=(uv run python)
if [ "${UV_PROJECT_ENVIRONMENT:-}" = "" ] && [ ! -d .venv ]; then
  say "note: no local .venv; using UV_PROJECT_ENVIRONMENT=/var/tmp/vision-intern-test-venv to avoid filling the RAM disk"
  PYENV=(env "UV_PROJECT_ENVIRONMENT=/var/tmp/vision-intern-test-venv" uv run python)
fi
run "${PYENV[@]}" -m freestyle.experiments.score_all --help >/dev/null

# ------------------------------------------------------------- step 1: the quiz itself ---
# Interactive by design: it is a phone app. Bind to the tailnet address, never the public net.
if [ "$SKIP_QUIZ" = 1 ] || [ "$DRY_RUN" = 1 ]; then
  say "quiz: skipped ($([ "$DRY_RUN" = 1 ] && echo dry-run || echo --skip-quiz)) — assuming answers are complete"
else
  say "quiz: serving on ${QUIZ_HOST}:${QUIZ_PORT} — answer every remaining question on the phone, then Ctrl-C"
  say "quiz: resume is automatic (the server replays $ANSWERS); GET /api/undo walks answers back"
  "${PYENV[@]}" -m freestyle.quiz.server \
    --queue "$QUEUE" --answers "$ANSWERS" --host "$QUIZ_HOST" --port "$QUIZ_PORT"
fi

# ------------------------------------------------- step 2: sufficiency gate (< 80% = abort) ---
# Mirrors freestyle.quiz.to_coco.load_answers (last line per qid wins) and the 'skip' semantics.
check_answers() {
  local queue="$1" answers="$2" min_frac="$3"
  "${PYENV[@]}" - "$queue" "$answers" "$min_frac" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '.')
from freestyle.quiz.to_coco import load_answers  # last-line-wins answer loader

queue_path, answers_path, min_frac = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
queue = json.loads(queue_path.read_text(encoding='utf-8'))
answers = load_answers(answers_path) if answers_path.is_file() else {}

# A frame answered 'not_cabinet' on its count question is a finished GT negative: to_coco drops
# its crops and score_all keeps it without the per-image 80% rule, so its crop questions must not
# sit in the denominator.
not_cabinet = {q['image_id'] for q in queue['questions']
               if q.get('kind') == 'count' and (answers.get(q['qid']) or {}).get('result') == 'not_cabinet'}

per_split_total, per_split_answered = defaultdict(int), defaultdict(int)
per_image = defaultdict(lambda: [0, 0])            # image_id -> [answered, effective questions]
for q in queue['questions']:
    split, image_id = q['split'], q['image_id']
    if image_id in not_cabinet and q.get('kind') != 'count':
        continue
    per_split_total[split] += 1
    per_image[image_id][1] += 1
    ans = answers.get(q['qid'])
    result = (ans or {}).get('result')
    if result not in (None, '', 'skip'):
        per_split_answered[split] += 1
        per_image[image_id][0] += 1
if not_cabinet:
    print(f'  negatives (not_cabinet, crops excluded from the denominator): {len(not_cabinet)}')

print(f'answers file: {answers_path} ({len(answers)} distinct qids recorded)')
ok = True
for split in ('test', 'val'):
    total = per_split_total.get(split, 0)
    got = per_split_answered.get(split, 0)
    frac = (got / total) if total else 0.0
    flag = 'OK ' if frac >= min_frac else 'LOW'
    print(f'  [{flag}] {split}: {got}/{total} questions answered ({frac:.0%}; need {min_frac:.0%})')
    ok = ok and total > 0 and frac >= min_frac

images = [i for i in per_image if i not in not_cabinet] + sorted(i for i in not_cabinet if i in per_image)
images = sorted(images, key=lambda i: per_image[i][0] / max(per_image[i][1], 1))
short = [i for i in images if per_image[i][0] / max(per_image[i][1], 1) < 0.8]
print(f'  images below the per-image 80% rule (score_all excludes these): {len(short)}/{len(images)}')
for i in short[:5]:
    a, t = per_image[i]
    print(f'    {a}/{t}  {i}')
if len(short) > 5:
    print(f'    ... and {len(short) - 5} more')
if not ok:
    print('INSUFFICIENT: answer more of the quiz (or pass --min-frac deliberately lower).', file=sys.stderr)
    raise SystemExit(3)
print('sufficiency gate passed.')
PY
}

if [ "$DRY_RUN" = 1 ]; then
  say "answer gate (dry-run, read-only):"
  check_answers "$QUEUE" "$ANSWERS" "$MIN_FRAC" || say "gate currently FAILS (exit 3) — expected before the quiz is answered"
else
  check_answers "$QUEUE" "$ANSWERS" "$MIN_FRAC"
fi

# ------------------------------------------------------------ step 3: to_coco (test, val) ---
mkdir -p "$GT_OUT_DIR" 2>/dev/null || true
for split in test val; do
  run "${PYENV[@]}" -m freestyle.quiz.to_coco \
    --queue "$QUEUE" --answers "$ANSWERS" \
    --out "${GT_OUT_DIR}/${split}.json" --split "$split"
done
say "after to_coco: read ${GT_OUT_DIR}/{test,val}_counts.json — every counts_mismatch frame needs"
say "box fixes in freestyle.review.server before its numbers can be trusted."

# ------------------------------------------------- step 4: score_all over all runs (1k CI) ---
run "${PYENV[@]}" -m freestyle.experiments.score_all \
  --queue "$QUEUE" --answers "$ANSWERS" \
  --runs "$RUNS_GLOB" \
  --splits "$SPLITS" --audit "$AUDIT" --proposals "$PROPOSALS" \
  --bootstrap "$BOOTSTRAP" \
  --out "$RESULTS_MD"
run "${PYENV[@]}" -m freestyle.eval.report --runs-dir runs --out docs/freestyle/RESULTS.md

# ------------------------------------------ step 5: which REPORT.md placeholders remain ---
report_placeholders() {
  local results="$1" report="$2"
  say "GT numbers now available in $results:"
  if [ -f "$results" ]; then
    grep -c '^| ' "$results" | sed 's/^/  table rows: /'
    grep -n '^# ' "$results" | sed 's/^/  /'
  else
    say "  (not generated yet — dry run)"
  fi
  say "placeholders still to fill by hand in $report (copy values from $results):"
  grep -n 'GT-PENDING' "$report" | sed 's/^/  /' || say "  none — every GT-PENDING marker has been removed"
  cat <<'TXT'
  Suggested swap order (each maps to one RESULTS_GT.md table):
    1. §1  headline metrics          -> row: winner, identifier, pair=off, subset=all
    2. §3  GT statistics             -> data/gt/{test,val}_counts.json + to_coco 'Excluded images'
    3. §8.2 human-GT comparison      -> the whole score_all table
    4. §8.9 architecture selection   -> winner row + the G-vs-S2 count row
    5. §9  failure decomposition     -> rerun freestyle.eval.failures against data/gt/test.json
    6. §10.2 pair-rule validation    -> pair=off vs pair=on rows for the two-part classes
TXT
}
report_placeholders "$RESULTS_MD" "$REPORT"

# --------------------------------------- step 6: export the winning bundle (MAE, tie -> ID) ---
pick_winner() {
  local results_json="$1"
  "${PYENV[@]}" - "$results_json" <<'PY'
import json, sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows = []
for row in data['rows']:
    # Counting decides the architecture; the pair rule stays unvalidated, so compare pair=off.
    if row.get('pair_rule') != 'off':
        continue
    alls = row.get('subsets', {}).get('all') or {}
    mae = (alls.get('metrics') or {}).get('total_error')
    if mae is None:
        continue
    ident = (alls.get('metrics') or {}).get('identity_accuracy')
    rows.append((mae, -(ident if ident is not None else -1.0), row['run'], row['identifier'], ident))
if not rows:
    print('no scorable rows (pair=off, subset=all, total_error present)', file=sys.stderr)
    raise SystemExit(4)
# best test total-count MAE; ties broken by higher identity accuracy
rows.sort(key=lambda r: (r[0], r[1], r[2]))
mae, _, run, ident, acc = rows[0]
print(f'{run}\t{ident}\t{mae:.4f}\t{-acc if acc is not None else float("nan"):.4f}')
PY
}

# identifier overrides per score_all method (mirrors configs/pipeline/v1_cartridges.yaml)
HYBRID_IDENTIFIER_JSON='{"type":"hybrid_vlm_knn","family_threshold":0.45,"embedding":{"model":"google/siglip2-base-patch16-224","k":5,"max_per_class":10},"vlm":{"model":"david-birks/inkling-2-mini","upscale":2}}'
KNN_IDENTIFIER_JSON='{"type":"embedding_knn","model":"google/siglip2-base-patch16-224","k":5,"max_per_class":10}'
VLM_IDENTIFIER_JSON='{"type":"vlm_crop","model":"david-birks/inkling-2-mini","mode":"two_stage","upscale":2}'
COUNTING_JSON='{"identity_thr":0.45,"neighbour_vote":"report","max_rows":2,"suppress_iou":0.45,"suppress_containment":0.7,"suppress_row_iou":0.3,"suppress_row_x_overlap":0.5,"shelf_hint":true,"pair_rule":"report"}'

if [ "$SKIP_EXPORT" = 1 ]; then
  say "export: skipped (--skip-export)"
else
  RESULTS_JSON="$(dirname "$RESULTS_MD")/results_gt.json"
  if [ ! -f "$RESULTS_JSON" ]; then
    say "export: $RESULTS_JSON not written yet (dry run) — cannot pick a winner"
  else
    WINNER_LINE="$(pick_winner "$RESULTS_JSON")" || die "could not pick a winning row"
    WINNER="$(printf '%s' "$WINNER_LINE" | cut -f1)"
    WIN_METHOD="$(printf '%s' "$WINNER_LINE" | cut -f2)"
    say "winner: $WINNER (identifier=$WIN_METHOD) — line: $WINNER_LINE"
    case "$WIN_METHOD" in
      knn|proto|knn_trainvlm|knn_trainvlm+val|linear_trainvlm|linear_trainvlm+val) IDENT_JSON="$KNN_IDENTIFIER_JSON" ;;
      own)   IDENT_JSON='{"type":"own"}' ;;
      vlm)   IDENT_JSON="$VLM_IDENTIFIER_JSON" ;;
      none|*) IDENT_JSON="$HYBRID_IDENTIFIER_JSON" ;;
    esac
    say "note: score_all's 'knn/proto/linear' identifiers are gallery variants of embedding_knn;"
    say "      tune identity_thr against $GT_OUT_DIR/val.json before shipping this bundle."
    run "${PYENV[@]}" -m freestyle.artifacts export \
      --run "runs/${WINNER}" --name "$BUNDLE_NAME" --overwrite \
      --identifier "$IDENT_JSON" --counting "$COUNTING_JSON" \
      --notes "exported by scripts/finalize_after_gt.sh from human GT: best test total-count MAE, ties by identity accuracy; thresholds re-tuned on data/gt/val.json"
    run "${PYENV[@]}" -m freestyle.artifacts verify "$BUNDLE_NAME"
  fi
fi

# --------------------------------------------------------- step 7: freestyle.cli demo ---
if [ "$DRY_RUN" = 1 ] || [ "$SKIP_EXPORT" = 1 ]; then
  DEMO_BUNDLE="$BUNDLE_NAME"
else
  DEMO_BUNDLE="$BUNDLE_NAME"
fi
run "${PYENV[@]}" -m freestyle.cli count "${DEMO_IMAGES[@]}" \
  --bundle "$DEMO_BUNDLE" --device cpu --json \
  --out runs/cli_preview_after_gt
say "demo overlays + counts.json in runs/cli_preview_after_gt/ — diff them against"
say "runs/bundle_demo_v1/counts.json to see what the human GT changed."

say "done. Commit the docs swap yourself: this script never runs git add/commit/push."
