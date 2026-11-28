#!/usr/bin/env bash
# finalize_after_gt.sh — the exact sequence to run once the human quiz answers exist.
#
# What it does, in order:
#   0. validate the repo/paths it is about to touch (and `uv`)
#   1. quiz          — serve freestyle.quiz.server so the remaining questions get answered
#                      (skippable: --skip-quiz, e.g. when the answers are already in)
#   2. sufficiency   — HARD GATE on USABLE images: abort unless at least MIN_USABLE_TEST (2)
#                      test images and MIN_USABLE_VAL (1) val images are a frame the human
#                      actually counted ("not a usable cabinet view" excludes a frame, it is
#                      never a zero count). Prints a loud `thin evidence` warning below
#                      THIN_EVIDENCE_TEST (5) usable test images; the --min-frac question
#                      coverage line is informational only.
#   3. to_coco       — export data/gt/test.json + data/gt/val.json (+ *_counts.json mismatches)
#   4. score_all     — score every run with both val and test predictions, 1000 bootstrap reps,
#                      identifier variants: none/vlm (Inkling identities) and vlm_flash (Flash
#                      Next identities, when --proposals-flash exists), each with the pair rule
#                      off and on -> docs/freestyle/RESULTS_GT.md + results_gt.json
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
#                                [--min-frac 0.80] [--min-usable-test N] [--min-usable-val N]
#                                [--thin-evidence N] [--answer-file PATH] [--queue PATH]
#                                [--runs GLOB] [--proposals PATH] [--proposals-flash PATH]
#                                [--bundle-name NAME] [--quiz-host H] [--quiz-port P]
#
# --dry-run validates inputs, runs the answer gate against whatever is in the answers file today,
# prints every command it would run, and mutates nothing (no quiz, no exports, no bundles).
# A failing gate still aborts the dry run with exit 3, exactly like the real run would.

set -euo pipefail

# ---------------------------------------------------------------- config (all overridable) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="data/gt/quiz_queue.json"
ANSWERS="data/gt/quiz_answers.jsonl"
SPLITS="data/splits.json"
PROPOSALS="data/labels/bootstrap_v2.json"
# Flash Next re-identification of the same boxes (freestyle.reidentify --mode two_stage_shelf
# --profile reader_compare). Optional: score_all only adds the vlm_flash identifier rows when
# the file is there, so the script still finalizes without it.
PROPOSALS_FLASH="data/labels/bootstrap_v3_flash.json"
AUDIT="data/derived/audit/audit.json"
RUNS_GLOB='runs/*'
GT_OUT_DIR="data/gt"
RESULTS_MD="docs/freestyle/RESULTS_GT.md"
REPORT="docs/freestyle/REPORT.md"
BUNDLE_NAME="rfdetr-final-v2"
MIN_FRAC="0.80"          # informational question-coverage line
MIN_USABLE_TEST=2      # hard: test images with a human count
MIN_USABLE_VAL=1         # hard: val images with a human count (threshold selection)
THIN_EVIDENCE_TEST=5     # below this many usable test images -> loud warning
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
    --min-usable-test) MIN_USABLE_TEST="${2:?--min-usable-test needs a value}"; shift ;;
    --min-usable-val) MIN_USABLE_VAL="${2:?--min-usable-val needs a value}"; shift ;;
    --thin-evidence) THIN_EVIDENCE_TEST="${2:?--thin-evidence needs a value}"; shift ;;
    --queue) QUEUE="${2:?}"; shift ;;
    --answer-file|--answers) ANSWERS="${2:?}"; shift ;;
    --proposals) PROPOSALS="${2:?}"; shift ;;
    --proposals-flash) PROPOSALS_FLASH="${2:?}"; shift ;;
    --runs) RUNS_GLOB="${2:?}"; shift ;;
    --bundle-name) BUNDLE_NAME="${2:?}"; shift ;;
    --quiz-host) QUIZ_HOST="${2:?}"; shift ;;
    --quiz-port) QUIZ_PORT="${2:?}"; shift ;;
    -h|--help) sed -n '2,39p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
for n in "$MIN_USABLE_TEST" "$MIN_USABLE_VAL" "$THIN_EVIDENCE_TEST"; do
  case "$n" in ''|*[!0-9]*) die "usable-image thresholds must be non-negative integers (got '$n')" ;; esac
done

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

# The Inkling identities are always scored; the Flash Next ones join as 'vlm_flash' when
# data/labels/bootstrap_v3_flash.json exists (score_all fails loudly if it is named but absent).
SCORE_ARGS=(--proposals "$PROPOSALS")
if [ -f "$PROPOSALS_FLASH" ]; then
  say "identifier readers: vlm ($PROPOSALS) + vlm_flash ($PROPOSALS_FLASH)"
  SCORE_ARGS+=(--proposals-flash "$PROPOSALS_FLASH")
else
  say "note: $PROPOSALS_FLASH not found — scoring the vlm (Inkling) reader only"
  say "      make it with: ${PYENV[*]} -m freestyle.reidentify --in $PROPOSALS --out $PROPOSALS_FLASH \\"
  say "        --model qwen3-8-flash-next --mode two_stage_shelf --profile reader_compare --only-split val --only-split test"
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

# ------------- step 2: sufficiency gate (usable test/val images with a human count) ---
# What scoring actually needs (see freestyle/experiments/score_all.py + quiz/to_coco.py):
#   a USABLE image with a human COUNT  — the only frames that exist in the metrics at all.
#                          'not_cabinet' (and the implied_not_usable crop answers it writes)
#                          means "too blurry / partial to count", so such a frame is EXCLUDED
#                          (reason not_usable); it is never a verified zero-cartridge frame.
#                          Need >= MIN_USABLE_TEST test and >= MIN_USABLE_VAL val of them.
#   test / purpose=eval    — >= --min-frac of these answered is printed for information;
#                          score_all still drops a single image below its own 80% rule.
#   val  / kind=count      — only usable frames drive select_thresholds(); val crops and all
#                          train questions stay optional (they only feed identity galleries).
# Mirrors freestyle.quiz.to_coco.load_answers (last line per qid wins) and the 'skip' semantics.
check_answers() {
  local queue="$1" answers="$2" min_frac="$3"
  "${PYENV[@]}" - "$queue" "$answers" "$min_frac" "$MIN_USABLE_TEST" "$MIN_USABLE_VAL" \
      "$THIN_EVIDENCE_TEST" <<'PY'
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '.')
from freestyle.quiz.to_coco import has_human_count, is_not_usable, load_answers  # answer semantics

queue_path, answers_path, min_frac = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
min_test, min_val, thin = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
queue = json.loads(queue_path.read_text(encoding='utf-8'))
answers = load_answers(answers_path) if answers_path.is_file() else {}

questions = queue['questions']


def answer_result(question):
    return (answers.get(question['qid']) or {}).get('result')


def answered(question):
    return answer_result(question) not in (None, '', 'skip')


def purpose(question):
    return str(question.get('purpose') or 'eval')


def kind(question):
    return str(question.get('kind') or 'crop')


counts = {split: [q for q in questions if q['split'] == split and kind(q) == 'count']
          for split in ('train', 'val', 'test')}
# usable == the human stated a total; a refused frame is excluded from every metric
counted = {split: {q['image_id'] for q in qs if has_human_count(answer_result(q))}
           for split, qs in counts.items()}
unusable = {split: {q['image_id'] for q in qs if is_not_usable(answer_result(q))}
            for split, qs in counts.items()}

eval_test = [q for q in questions if q['split'] == 'test' and purpose(q) == 'eval']
val_crops = [q for q in questions if q['split'] == 'val' and kind(q) != 'count']
train_qs = [q for q in questions if q['split'] == 'train' or purpose(q) == 'train']

print(f'answers file: {answers_path} ({len(answers)} distinct qids recorded)')
print(f'usable test images with counts: {len(counted["test"])} (need >= {min_test})')
print(f'usable val images with counts: {len(counted["val"])} (need >= {min_val})')
print(f'excluded as not usable: {len(unusable["test"])} test, {len(unusable["val"])} val '
      f'(uncountable frames — never zero-cartridge frames)')


def flag(enough):
    return 'OK ' if enough else 'LOW'


print(f'  [{flag(len(counted["test"]) >= min_test)}] test: {len(counted["test"])} usable '
      f'frames with a human count (need >= {min_test})')
print(f'  [{flag(len(counted["val"]) >= min_val)}] val: {len(counted["val"])} usable frames '
      f'with a human count for threshold selection (need >= {min_val})')
if len(counted['test']) < thin:
    print(f'  *** WARNING: THIN EVIDENCE — only {len(counted["test"])} usable test images '
          f'with counts (< {thin}); every headline number hangs off them ***')

# informational: question coverage (the 80% rule score_all applies per image)
eval_total = len(eval_test)
eval_got = sum(1 for q in eval_test if answered(q))
eval_need = min(eval_total, max(0, math.ceil(eval_total * min_frac - 1e-9))) if eval_total else 1
if eval_total:
    print(f'  [inf] test eval questions: {eval_got}/{eval_total} answered '
          f'(informational, --min-frac {min_frac:.0%} would need {eval_need})')
else:
    print('  [inf] test eval: the queue has no purpose=eval test questions')
val_count_got = sum(1 for q in counts['val'] if answered(q))
print(f'  [inf] val count questions answered: {val_count_got}/{len(counts["val"])} '
      f'({len(counted["val"])} usable with a count)')
val_crop_got = sum(1 for q in val_crops if answered(q))
print(f'  [opt] val crop answers (identity gallery): {val_crop_got}/{len(val_crops)}'
      f' — {"complete" if val_crop_got >= len(val_crops) else "insufficient val crops"}')
train_got = sum(1 for q in train_qs if answered(q))
print(f'  [opt] train questions: {train_got}/{len(train_qs)} — not required by scoring')

# Informational: score_all excludes individual usable test images below its per-image 80% rule,
# so a frame can hold a count and still drop out of the metrics.
per_image = defaultdict(lambda: [0, 0])            # image_id -> [answered, effective questions]
for q in eval_test:
    if q['image_id'] in unusable['test'] or q['image_id'] not in counted['test']:
        continue                                   # excluded frames are not scored at all
    per_image[q['image_id']][1] += 1
    if answered(q):
        per_image[q['image_id']][0] += 1
images = sorted(per_image, key=lambda i: per_image[i][0] / max(per_image[i][1], 1))
short = [i for i in images if per_image[i][0] / max(per_image[i][1], 1) < 0.8]
print(f'  usable test images with a count below score_all\'s per-image 80% rule '
      f'(these get excluded too): {len(short)}/{len(images)}')
for i in short[:5]:
    a, t = per_image[i]
    print(f'    {a}/{t}  {i}')
if len(short) > 5:
    print(f'    ... and {len(short) - 5} more')
scored_test = len(counted['test']) - len(short)
if scored_test < min_test:
    print(f'  *** WARNING: only {scored_test} usable test images are expected to survive the '
          f'per-image 80% rule (< {min_test}) ***')

ok = len(counted['test']) >= min_test and len(counted['val']) >= min_val
if not ok:
    print(f'INSUFFICIENT: need >= {min_test} USABLE test images and >= {min_val} usable val '
          'image with a human count (a "not a usable cabinet view" tap excludes the frame, it '
          'does not make it a zero). Answer the count questions on real cabinet views.',
          file=sys.stderr)
    raise SystemExit(3)
print('sufficiency gate passed.')
PY
}

if [ "$DRY_RUN" = 1 ]; then
  say "answer gate (dry-run, read-only):"
  if ! check_answers "$QUEUE" "$ANSWERS" "$MIN_FRAC"; then
    say "gate currently FAILS — the real run would abort here (exit 3); expected before the quiz is answered"
    exit 3
  fi
  say "gate would PASS — continuing the dry run"
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
  --splits "$SPLITS" --audit "$AUDIT" "${SCORE_ARGS[@]}" \
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
# the reader comparison's winner (docs/freestyle/READER_COMPARE.md); the pipeline supplies the
# per-box shelf hint, which is what score_all's vlm_flash identities were read with
VLM_FLASH_IDENTIFIER_JSON='{"type":"vlm_crop","model":"qwen3-8-flash-next","mode":"two_stage","upscale":2}'
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
      vlm_flash) IDENT_JSON="$VLM_FLASH_IDENTIFIER_JSON" ;;
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
