"""Prepare human-only quiz ground truth for the scoring pass.

Score human quiz counts and identities against experiment predictions.

Identity variants that read model-labelled proposals (``data/labels/*.json``, never GT):
    ``vlm``       Inkling 2 Mini two-stage identities from ``--proposals``
    ``vlm_flash`` Qwen 3.8 Flash Next two-stage + shelf-hint identities from
                  ``--proposals-flash`` (``data/labels/bootstrap_v3_flash.json``, made by
                  ``python -m freestyle.reidentify --mode two_stage_shelf --profile
                  reader_compare``). Same IoU >= 0.5 box matching, only the reader differs,
                  so the two rows are a direct A/B of the readers on human answers.
Both are also scored with the pair rule off and on.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from freestyle.count.pairs import infer_pairs
from freestyle.count.shelves import assign_shelves
from freestyle.eval.matching import box_iou, match_boxes
from freestyle.eval.metrics import (
    _compute_coco_ap_from_records, _per_image_localization_records, _prepare_dataset,
    compute_group_bootstrap_ci, load_groups_map,
)
from freestyle.identify.classify import KNNClassifier, LinearProbe, PrototypeClassifier
from freestyle.quiz.build_queue import shelf_labels
from freestyle.quiz.to_coco import build_coco, load_answers


def build_human_gt(
    queue: dict[str, Any], answers: dict[str, dict[str, Any]], split: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build quiz COCO, excluding incomplete images and model identities.

    ``test`` is the eval split: skips and missing answers do not count towards the
    80% per-image cutoff and a human count is required independently.

    ``val`` is the tuning split and only ever drives threshold selection, which reads
    the per-image human total (see select_thresholds). So a val image only needs its
    human COUNT; unanswered val crops never exclude it. Answered val crops are kept and
    used exactly as before (identity gallery / tuning), they are just optional.
    """
    coco, counts = build_coco(queue, answers, split)
    requires_crop_coverage = split != 'val'
    questions: dict[str, list[dict[str, Any]]] = {}
    for question in queue['questions']:
        questions.setdefault(question['image_id'], []).append(question)
    excluded = []
    kept = []
    for image in coco['images']:
        image_id = image['file_name']
        qs = questions.get(image_id, [])
        answered = sum(
            bool(answers.get(q['qid']))
            and answers[q['qid']].get('result') not in (None, '', 'skip')
            for q in qs
        )
        fraction = answered / len(qs) if qs else 0.0
        short_of_crops = (fraction < 0.8 and image['attributes'].get('usable', True)
                          and requires_crop_coverage)
        if short_of_crops or image['attributes']['human_counts'] is None:
            excluded.append({'image_id': image_id, 'answered': answered,
                             'questions': len(qs), 'fraction': fraction,
                             'reason': 'missing_count' if image['attributes']['human_counts'] is None
                                       else 'below_80_percent'})
        else:
            kept.append(image)
    ids = {image['id'] for image in kept}
    coco['images'] = kept
    coco['annotations'] = [ann for ann in coco['annotations']
                           if ann['image_id'] in ids
                           and ann.get('attributes', {}).get('identity_source') == 'human']
    counts['images'] = {image['file_name']: counts['images'][image['file_name']] for image in kept}
    counts['mismatches'] = [name for name in counts['mismatches'] if name in counts['images']]
    counts['unverified'] = [name for name in counts['unverified'] if name in counts['images']]
    coco['info']['total_images'] = len(kept)
    coco['info']['human_verified_images'] = sum(
        bool(image['attributes']['verified']) for image in kept
    )
    return coco, counts, excluded



def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def demo_answers(queue, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'qid': q['qid'], 'answer': {'result': 'correct'}}) + '\n'
                            for q in queue['questions']), encoding='utf-8')


def grouped_predictions(predictions):
    result = defaultdict(list)
    for pred in predictions:
        box = pred.get('bbox', [])
        if len(box) != 4 or not all(np.isfinite(box)) or box[2] <= 0 or box[3] <= 0:
            continue
        key = pred['image_id']
        if isinstance(key, str) and key.isdecimal():
            key = int(key)
        result[key].append(dict(pred))
    return result


def xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def proposal_data(coco):
    names = {c['id']: c['name'] for c in coco['categories']}
    images = {i['id']: i['file_name'] for i in coco['images']}
    result = defaultdict(list)
    for ann in coco['annotations']:
        if ann['image_id'] in images:
            result[images[ann['image_id']]].append({**ann, 'class': names.get(ann['category_id'], 'unknown')})
    return result


def load_embeddings(path):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def gallery(emb, val_gt, held_out_group=None):
    """Human VAL labels keyed by proposal id; never use test labels."""
    names = {c['id']: c['name'] for c in val_gt['categories']}
    images = {i['id']: i for i in val_gt['images']}
    labels = {}
    for ann in val_gt['annotations']:
        image = images[ann['image_id']]
        if image['group_id'] == held_out_group:
            continue
        name = names.get(ann['category_id'], 'unknown')
        if name != 'unknown':
            labels[(image['file_name'], str(ann['attributes']['proposal_ann_id']))] = name
    indexes, targets = [], []
    for idx, (file_name, ann_id) in enumerate(zip(emb['file_names'], emb['annotation_ids'], strict=True)):
        key = (str(file_name), str(ann_id))
        if key in labels:
            indexes.append(idx)
            targets.append(labels[key])
    return indexes, targets


def train_gallery(emb, proposals, splits, confidence):
    """Select VLM-labelled training proposals using the authoritative split map."""
    labels = {}
    for name, anns in proposals.items():
        if splits.get('images', {}).get(name) != 'train':
            continue
        for ann in anns:
            attrs = ann.get('attributes', {})
            source = str(attrs.get('source') or '')
            score = attrs.get('id_confidence')
            if (source.startswith('vlm:') and score is not None
                    and 0 <= float(score) <= 1 and float(score) >= confidence
                    and ann['class'] not in ('unknown', 'cartridge')):
                labels[(name, str(ann['id']))] = ann['class']
    indexes, targets = [], []
    for idx, (name, ann_id) in enumerate(zip(emb['file_names'], emb['annotation_ids'], strict=True)):
        key = (str(name), str(ann_id))
        if key in labels:
            indexes.append(idx)
            targets.append(labels[key])
    return indexes, targets


def apply_proposal_identities(rows, image, proposals, modes=('two_stage',)):
    """Copy a VLM two-stage identity onto predicted boxes that match a proposal box.

    ``proposals`` is ``proposal_data(...)`` of a model-labelled COCO keyed by image file
    name; only proposals whose ``id_mode`` is in *modes* are used (the plain ``vlm``
    identifier keeps the strict ``two_stage`` match it was validated with, while
    ``vlm_flash`` also accepts the ``two_stage_shelf`` replies of the flash reader).
    A predicted box with no proposal at IoU >= 0.5 keeps no identity.
    """
    candidates = [p for p in proposals.get(image['file_name'], [])
                  if str(p.get('attributes', {}).get('id_mode') or '') in modes]
    if not candidates or not rows:
        return rows
    overlaps = box_iou([r['bbox'] for r in rows], [p['bbox'] for p in candidates])
    for row, ious in zip(rows, overlaps, strict=True):
        idx = int(np.argmax(ious))
        if ious[idx] >= .5:
            candidate = candidates[idx]
            attrs = candidate.get('attributes', {})
            row.update({'class': candidate['class'],
                        'identity_score': float(attrs.get('id_confidence') or 0),
                        'family': attrs.get('id_family'), 'part': attrs.get('id_part'),
                        'family_confidence': float(attrs.get('id_confidence') or 0),
                        'part_confidence': float(attrs.get('id_confidence') or 0)})
    return rows


def identify(preds, image, method, proposals, emb, val_gt, validation=False,
             train_items=None, fitted=None, flash_proposals=None):
    rows = [dict(p, **{'class': 'unknown', 'identity_score': 0.0}) for p in preds]
    if method == 'none':
        return rows
    if method == 'own':
        for row in rows:
            row['class'] = str(row.get('proposed_class') or 'unknown')
            row['identity_score'] = float(row.get('id_score', row.get('score', 0)))
        return rows
    if method == 'vlm':
        return apply_proposal_identities(rows, image, proposals)
    if method == 'vlm_flash':
        # Same IoU>=0.5 box matching as 'vlm', identities from the Flash Next pass.
        return apply_proposal_identities(rows, image, flash_proposals or {},
                                         modes=('two_stage', 'two_stage_shelf'))
    if emb is None or not rows:
        return rows
    uses_train = 'trainvlm' in method
    uses_val = not uses_train or method.endswith('+val')
    indexes, targets = (list(train_items[0]), list(train_items[1])) if uses_train and train_items else ([], [])
    if uses_val:
        val_indexes, val_targets = gallery(emb, val_gt, image['group_id'] if validation else None)
        indexes.extend(val_indexes)
        targets.extend(val_targets)
    if not indexes:
        return rows
    query = np.flatnonzero(emb['file_names'].astype(str) == image['file_name'])
    if not len(query):
        return rows
    key = (method, tuple(indexes))
    if fitted is None:
        fitted = {}
    if key not in fitted:
        if method.startswith('knn'):
            clf = KNNClassifier(k=5)
        elif method.startswith('linear'):
            clf = LinearProbe()
        else:
            clf = PrototypeClassifier()
        fitted[key] = clf.fit(emb['embeddings'][indexes], targets)
    clf = fitted[key]
    overlaps = box_iou([r['bbox'] for r in rows], emb['bboxes'][query])
    chosen = np.argmax(overlaps, axis=1)
    valid = np.flatnonzero(overlaps[np.arange(len(rows)), chosen] >= .5)
    if len(valid):
        labels, scores = clf.predict_with_scores(emb['embeddings'][query[chosen[valid]]])
        for idx, label, score in zip(valid, labels, scores, strict=True):
            rows[idx].update({'class': str(label), 'identity_score': float(score)})
    return rows


@lru_cache(maxsize=2048)
def geometry(boxes, height):
    raw = [{'box': list(box)} for box in boxes]
    rows = assign_shelves(raw) if raw else []
    shelves = shelf_labels([list(box) for box in boxes], height) if raw else []
    return [(r['shelf_idx'], r['column_idx'], r['tier'], shelf)
            for r, shelf in zip(rows, shelves, strict=True)]


def prepare_rows(rows, image, score_thr, identity_thr, pair, with_shelves=True):
    rows = [dict(r) for r in rows if float(r['score']) >= score_thr
            and len(r['bbox']) == 4 and all(np.isfinite(r['bbox']))
            and r['bbox'][2] > 0 and r['bbox'][3] > 0]
    for row in rows:
        if row['identity_score'] < identity_thr or row['class'] == 'cartridge':
            row['class'] = 'unknown'
            row['family'] = None
            row['part'] = None
        row['box'] = xyxy(row['bbox'])
        match = re.match(r'(.+)-part([12])$', row['class'])
        if match and not row.get('family'):
            row.update(family=match.group(1), part=None,
                       family_confidence=row['identity_score'], part_confidence=0.)
    if rows and (with_shelves or pair and any(r.get('family') for r in rows)):
        boxes = tuple(tuple(r['box']) for r in rows)
        for row, (shelf, column, tier, label) in zip(rows, geometry(boxes, image['height']), strict=True):
            row.update(shelf_idx=shelf, column_idx=column, tier=tier, shelf_name=label)
        if pair:
            rows = infer_pairs(rows, mode='apply', family_threshold=identity_thr,
                               part_threshold=identity_thr)
    return rows


def per_image(image, gt_anns, rows, counts, category_names, none):
    human = image['attributes']['human_counts']
    actual = counts['classes']
    predicted = Counter(r['class'] for r in rows)
    errors = {c: abs(predicted.get(c, 0) - actual.get(c, 0))
              for c in set(actual) | set(predicted)}
    gt_boxes = [a['bbox'] for a in gt_anns]
    matches = match_boxes([r['bbox'] for r in rows], gt_boxes, iou_thr=.5,
                          pred_scores=[float(r['score']) for r in rows]).matches
    correct = sum(rows[p]['class'] == category_names.get(gt_anns[g]['category_id'])
                  for p, g, _ in matches if category_names.get(gt_anns[g]['category_id']) != 'unknown')
    eligible = sum(category_names.get(gt_anns[g]['category_id']) != 'unknown' for _, g, _ in matches)
    return {'total_error': abs(len(rows) - human['total']),
            'top_error': abs(sum(r['shelf_name'] == 'top' for r in rows) - human['top']),
            'bottom_error': abs(sum(r['shelf_name'] == 'bottom' for r in rows) - human['bottom']),
            'class_errors': None if none or not counts['verified'] or counts['counts_mismatch'] else errors,
            'exact_inventory': None if none or not counts['verified'] or counts['counts_mismatch'] else int(not any(errors.values())),
            'id_correct': None if none else correct, 'id_n': None if none else eligible}


def aggregate(records, indices, classes, none):
    if not indices:
        return {}
    sample = [records[i] for i in indices]
    result = {key: float(np.mean([r[key] for r in sample]))
              for key in ('total_error', 'top_error', 'bottom_error')}
    if not none:
        exact = [r['exact_inventory'] for r in sample if r['exact_inventory'] is not None]
        result['exact_inventory'] = float(np.mean(exact)) if exact else None
        n = sum(r['id_n'] for r in sample)
        result['identity_accuracy'] = sum(r['id_correct'] for r in sample) / n if n else None
        eligible = [r['class_errors'] for r in sample if r['class_errors'] is not None]
        result.update({f'class:{c}': float(np.mean([r.get(c, 0) for r in eligible])) if eligible else None
                       for c in classes})
    return result


def score_subset(images, gt, counts, rows_by_image, score_thr, identity_thr, pair, none, n_bootstrap):
    annotations = defaultdict(list)
    for ann in gt['annotations']:
        annotations[ann['image_id']].append(ann)
    names = {c['id']: c['name'] for c in gt['categories']}
    records = []
    for image in images:
        rows = prepare_rows(rows_by_image[image['id']], image, score_thr, identity_thr, pair)
        records.append(per_image(image, annotations[image['id']], rows,
                                 counts['images'][image['file_name']], names, none))
    classes = sorted({c for r in records for c in (r['class_errors'] or {})})
    eligible_images = [im for im in images if counts['images'][im['file_name']]['verified']
                       and not counts['images'][im['file_name']]['counts_mismatch']]
    loc_gt = {**gt, 'images': eligible_images,
              'annotations': [a for im in eligible_images for a in annotations[im['id']]]}
    # AP uses all scores; bootstrap resamples whole groups including their boxes.
    loc_all = [p for im in eligible_images for p in rows_by_image[im['id']]]
    prepared = _prepare_dataset(loc_gt, loc_all)
    loc_records, n_gt = _per_image_localization_records(prepared, iou_thresholds=(0.5,))
    loc_index = {im['id']: i for i, im in enumerate(eligible_images)}
    def summary(ix):
        loc_ix = [loc_index[images[i]['id']] for i in ix if images[i]['id'] in loc_index]
        values = aggregate(records, ix, classes, none)
        scores = np.concatenate([loc_records[i][0.5][0] for i in loc_ix]) if loc_ix else np.array([])
        matches = np.concatenate([loc_records[i][0.5][1] for i in loc_ix]) if loc_ix else np.array([])
        values['ap50'] = _compute_coco_ap_from_records(scores, matches, sum(n_gt[i] for i in loc_ix))
        return values
    metrics = summary(list(range(len(images)))) if images else {}
    ci = compute_group_bootstrap_ci([im['group_id'] for im in images], summary,
                                    n_bootstrap=n_bootstrap)
    return {'n_images': len(images), 'images': [i['file_name'] for i in images],
            'metrics': metrics, 'bootstrap_ci95': ci, 'per_image': records}


def select_thresholds(images, gt, counts, rows_by_image, pair, none):
    # Deterministic grid chosen solely by validation counting error.
    if not images:
        raise ValueError('no eligible validation images for threshold selection')
    candidates = tuple(i / 20 for i in range(20))
    totals = {thr: sum(abs(sum(float(r['score']) >= thr for r in rows_by_image[im['id']])
                           - im['attributes']['human_counts']['total']) for im in images)
              for thr in candidates}
    best_total = min(totals.values())
    options = []
    for score_thr in (t for t in candidates if totals[t] == best_total):
        for identity_thr in ((0.,) if none else (0., .25, .5, .75)):
            class_error = 0
            if not none:
                for im in images:
                    if not counts['images'][im['file_name']].get('verified', True) or counts['images'][im['file_name']].get('counts_mismatch', False):
                        continue
                    rows = prepare_rows(rows_by_image[im['id']], im, score_thr, identity_thr, pair, False)
                    actual = counts['images'][im['file_name']]['classes']
                    predicted = Counter(r['class'] for r in rows)
                    class_error += sum(abs(predicted.get(c, 0) - actual.get(c, 0))
                                       for c in set(actual) | set(predicted))
            options.append((best_total, class_error, score_thr, identity_thr))
    best_error = min(option[:2] for option in options)
    tied = [option for option in options if option[:2] == best_error]
    scores = sorted({option[2] for option in tied})
    score = scores[(len(scores) - 1) // 2]
    identities = sorted(option[3] for option in tied if option[2] == score)
    identity = identities[(len(identities) - 1) // 2]
    return (*best_error, score, identity)


def validate_groups(gt, split, splits):
    groups = load_groups_map(gt, splits)
    for image in gt['images']:
        name = image['file_name']
        group = groups[name]
        if splits.get('images', {}).get(name) != split or splits.get('groups', {}).get(group) != split:
            raise ValueError(f'missing or inconsistent {split} split/group for {name}')
        image['group_id'] = group


def markdown(result):
    title = '# DEMO — synthetic model-accepted answers' if result['demo'] else '# Human ground-truth results'
    lines = [title, '', 'Counting is scored against quiz counts; per-class counts come from answered crops.',
             'AP50 compares with human-filtered boxes that originate from model proposals; it is not independent localization GT.',
             'Identity accuracy is conditional on a matched human-answered, known-identity crop.',
             'Class inventory and AP exclude incomplete images and count mismatches.',
             'Thresholds minimize validation total count error, then class count error; embedding validation uses held-out groups.',
             'The validation split needs human count answers only; unanswered validation crops just shrink the identity gallery.',
             'Training-gallery confidence thresholds are selected on validation; +val galleries hold out the query group during validation and use all validation groups at test time. Train and validation may contain near duplicates.',
             'Sharp-full means cabinet_full with no or mild motion blur; partial/blurry means cabinet_partial or severe blur. Other audit cases are omitted from these subsets.',
             '', 'Excluded images:']
    for split in ('val', 'test'):
        lines.append(f"- {split}: " + (', '.join(x['image_id'] + ' (' + x['reason'] + ')'
                                                for x in result['excluded'][split]) or 'none'))
    lines += ['', '| Localiser | Identifier | Pair | Train confidence | Train crops | Subset | Images | Total MAE (95% CI) | Top MAE | Bottom MAE | Exact inventory | ID accuracy | AP50 |',
              '|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|']
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    def estimate(value, key):
        ci = value['bootstrap_ci95']['metrics'].get(key)
        interval = '' if ci is None else f' [{fmt(ci[0])}, {fmt(ci[1])}]'
        return fmt(value['metrics'].get(key)) + interval
    for row in result['rows']:
        for subset, value in row['subsets'].items():
            lines.append(f"| {row['run']} | {row['identifier']} | {row['pair_rule']} | {fmt(row['train_vlm_confidence_thr'])} | "
                         f"{row['train_gallery_size']} | {subset} | {value['n_images']} | "
                         f"{estimate(value, 'total_error')} | {estimate(value, 'top_error')} | "
                         f"{estimate(value, 'bottom_error')} | {estimate(value, 'exact_inventory')} | "
                         f"{estimate(value, 'identity_accuracy')} | {estimate(value, 'ap50')} |")
    lines += ['', 'Per-class count MAE (95% group-bootstrap CI):', '',
              '| Localiser | Identifier | Pair | Subset | Class | MAE (95% CI) |',
              '|---|---|---|---|---|---|']
    for row in result['rows']:
        for subset, value in row['subsets'].items():
            for key in sorted(k for k in value['metrics'] if k.startswith('class:')):
                lines.append(f"| {row['run']} | {row['identifier']} | {row['pair_rule']} | {subset} | "
                             f"{key[6:]} | {estimate(value, key)} |")
    valid = [r for r in result['rows'] if r['subsets']['all']['metrics'].get('total_error') is not None]
    if valid:
        best = min(valid, key=lambda r: r['subsets']['all']['metrics']['total_error'])
        lines += ['', f"Best by total count MAE: **{best['run']} / {best['identifier']} / pair {best['pair_rule']}** "
                  f"({best['subsets']['all']['metrics']['total_error']:.3f})."]
    lines += ['', 'Thresholds, exclusions and image lists are in results_gt.json.', '']
    return '\n'.join(lines)


def audit_subsets(images, audit):
    usable = [im for im in images if im['attributes'].get('usable', True)]
    sharp = [im for im in usable if audit.get(im['file_name'], {}).get('scene_label') == 'cabinet_full'
             and audit.get(im['file_name'], {}).get('motion_blur') in ('none', 'mild')]
    partial = [im for im in usable if audit.get(im['file_name'], {}).get('scene_label') == 'cabinet_partial'
               or audit.get(im['file_name'], {}).get('motion_blur') == 'severe']
    return sharp, partial


def run(args):
    queue = read_json(args.queue)
    if args.demo:
        demo_dir = Path('runs/score_all_demo')
        answers_path = demo_dir / 'quiz_answers.jsonl'
        demo_answers(queue, answers_path)
        out = demo_dir / 'RESULTS_GT.md'
    else:
        answers_path, out = args.answers, args.out
    if answers_path is None or not answers_path.is_file():
        raise ValueError(f'answers file does not exist: {answers_path}')
    answers = load_answers(answers_path)
    if not answers:
        raise ValueError('answers file contains no valid answers')
    splits = read_json(args.splits)
    audit = {i['image_id']: i for i in read_json(args.audit)['images']}
    proposal_coco = read_json(args.proposals)
    proposals = proposal_data(proposal_coco)
    flash_proposals = None
    if getattr(args, 'proposals_flash', None):
        flash_path = Path(args.proposals_flash)
        if not flash_path.is_file():
            raise ValueError(f'flash proposal file does not exist: {flash_path}')
        flash_proposals = proposal_data(read_json(flash_path))
    gt, counts, excluded = {}, {}, {}
    for split in ('val', 'test'):
        gt[split], counts[split], excluded[split] = build_human_gt(queue, answers, split)
        validate_groups(gt[split], split, splits)
    if not gt['val']['images']:
        raise ValueError('no eligible validation images (val needs its per-image count answers; '
                         'val crop answers are optional)')
    if not gt['test']['images']:
        raise ValueError('no eligible test images')
    result = {'version': 1, 'demo': args.demo, 'excluded': excluded, 'rows': [], 'skipped_runs': [],
              'limitations': ['All localization reference boxes originate from model proposals.',
                              'Per-class and AP metrics use fully answered images without count mismatches.',
                              'Training and validation splits may include near-duplicate images.',
                              'Validation embeddings use leave-one-group-out gallery; test uses all eligible validation crops.',
                              'Training VLM gallery confidence is selected on validation; test labels are never used.']}
    if flash_proposals is not None:
        result['limitations'].append(
            f'vlm_flash identities come from {args.proposals_flash}: model-generated, '
            'matched to predicted boxes at IoU>=0.5, never human labels.')
    # Val crops are optional: with none answered, the val-only embedding galleries are empty
    # and the identifier variants that depend on them cannot produce meaningful numbers.
    val_gallery_total = len(gt['val']['annotations'])
    if not val_gallery_total:
        result['limitations'].append(
            'No answered val crops: val-only identity galleries are reported as '
            '"insufficient val crops"; train-VLM galleries still score normally.')
    for path in sorted(map(Path, glob.glob(args.runs))):
        val_path, test_path = path / 'val/predictions.json', path / 'test/predictions.json'
        if not val_path.is_file() or not test_path.is_file():
            result['skipped_runs'].append({'run': path.name, 'reason': 'missing val or test predictions'})
            continue
        meta = read_json(path / 'meta.json') if (path / 'meta.json').is_file() else {}
        family = meta.get('config', {}).get('family', '')
        category_names = {c['id']: c['name'] for c in proposal_coco['categories']}
        predictions = {}
        for split, pred_path in (('val', val_path), ('test', test_path)):
            raw = read_json(pred_path)
            for pred in raw:
                if not pred.get('proposed_class'):
                    pred['proposed_class'] = category_names.get(pred.get('category_id'), 'unknown')
            predictions[split] = grouped_predictions(raw)
        emb = load_embeddings(path / 'embeddings.npz')
        val_gallery_size = len(gallery(emb, gt['val'])[1]) if emb is not None else 0
        methods = ['none', 'vlm']
        if flash_proposals is not None:
            methods.append('vlm_flash')
        if family.startswith('D_'):
            methods.append('own')
        if emb is not None:
            methods += ['knn', 'proto', 'knn_trainvlm', 'knn_trainvlm+val',
                        'linear_trainvlm', 'linear_trainvlm+val']
        for method in methods:
            if method in ('knn', 'proto') and not val_gallery_size:
                # These galleries are human val crops only; without them the identifiers would
                # return 'unknown' for every crop and report a huge, meaningless class error.
                result['skipped_runs'].append({'run': path.name, 'identifier': method,
                                               'reason': 'insufficient val crops'})
                continue
            train_variant = 'trainvlm' in method
            candidates = (0., .5, .75, .9) if train_variant else (None,)
            fitted = {}
            variants = {}
            for confidence in candidates:
                train_items = train_gallery(emb, proposals, splits, confidence) if train_variant else None
                if train_variant and not train_items[0]:
                    continue
                val_rows = {im['id']: identify((predictions['val'].get(im['id']) or predictions['val'].get(im['file_name'], [])), im, method, proposals, emb, gt['val'], validation=True, train_items=train_items, fitted=fitted, flash_proposals=flash_proposals)
                            for im in gt['val']['images']}
                variants[confidence] = (train_items, val_rows)
            if not variants:
                result['skipped_runs'].append({'run': path.name, 'identifier': method,
                                               'reason': 'no VLM-labelled training crop embeddings'})
                continue
            for pair in (False, True):
                selections = []
                for confidence, (train_items, val_rows) in variants.items():
                    key = select_thresholds(gt['val']['images'], gt['val'], counts['val'],
                                            val_rows, pair, method == 'none')
                    selections.append((key, confidence, train_items))
                best_error = min(item[0][:2] for item in selections)
                tied = [item for item in selections if item[0][:2] == best_error]
                tied.sort(key=lambda item: item[1] if item[1] is not None else -1.)
                key, confidence, train_items = tied[(len(tied) - 1) // 2]
                score_thr, id_thr = key[2:4]
                test_rows = {im['id']: identify((predictions['test'].get(im['id']) or predictions['test'].get(im['file_name'], [])), im, method, proposals, emb, gt['val'], train_items=train_items, fitted=fitted, flash_proposals=flash_proposals)
                             for im in gt['test']['images']}
                images = gt['test']['images']
                sharp, partial = audit_subsets(images, audit)
                subsets = {name: score_subset(ims, gt['test'], counts['test'], test_rows, score_thr, id_thr,
                                              pair, method == 'none', args.bootstrap)
                           for name, ims in [('all', images), ('sharp-full', sharp), ('partial/blurry', partial)]}
                result['rows'].append({'run': path.name, 'identifier': method, 'pair_rule': 'on' if pair else 'off',
                                       'score_thr': score_thr, 'identity_thr': id_thr,
                                       'train_vlm_confidence_thr': confidence,
                                       'train_gallery_size': len(train_items[0]) if train_items else 0,
                                       'val_gallery_size': val_gallery_size,
                                       'threshold_source': 'val', 'subsets': subsets})

    if not result['rows']:
        raise ValueError('no runs with both validation and test predictions')
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out.parent / 'results_gt.json', result)
    out.write_text(markdown(result), encoding='utf-8')
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Score experiments against quiz counts and human crop labels')
    parser.add_argument('--answers', type=Path)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', type=Path, default=Path('docs/freestyle/RESULTS_GT.md'))
    parser.add_argument('--splits', type=Path, default=Path('data/splits.json'))
    parser.add_argument('--audit', type=Path, default=Path('data/derived/audit/audit.json'))
    parser.add_argument('--proposals', type=Path, default=Path('data/labels/bootstrap_v2.json'))
    parser.add_argument('--proposals-flash', type=Path, default=None,
                        help='second reader proposal COCO (e.g. data/labels/bootstrap_v3_flash.json); '
                             'adds the vlm_flash identifier variant when given')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args(argv)
    if args.bootstrap <= 0:
        parser.error('--bootstrap must be positive')
    try:
        run(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
