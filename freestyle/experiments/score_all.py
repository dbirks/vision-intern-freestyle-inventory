"""Prepare human-only quiz ground truth for the scoring pass.

Score human quiz counts and identities against experiment predictions.
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
from freestyle.identify.classify import KNNClassifier, PrototypeClassifier
from freestyle.quiz.build_queue import shelf_labels
from freestyle.quiz.to_coco import build_coco, load_answers


def build_human_gt(
    queue: dict[str, Any], answers: dict[str, dict[str, Any]], split: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build quiz COCO, excluding incomplete images and model identities.

    Skips and missing answers do not count towards the 80% cutoff. A human
    count is required independently, since it is the primary count target.
    """
    coco, counts = build_coco(queue, answers, split)
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
        if fraction < 0.8 or image['attributes']['human_counts'] is None:
            excluded.append({'image_id': image_id, 'answered': answered,
                             'questions': len(qs), 'fraction': fraction,
                             'reason': 'below_80_percent' if fraction < 0.8 else 'missing_count'})
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
        result[int(pred['image_id'])].append(dict(pred))
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


def identify(preds, image, method, proposals, emb, val_gt, validation=False):
    rows = [dict(p, **{'class': 'unknown', 'identity_score': 0.0}) for p in preds]
    if method == 'none':
        return rows
    if method == 'own':
        for row in rows:
            row['class'] = str(row.get('proposed_class') or 'unknown')
            row['identity_score'] = float(row.get('id_score', row.get('score', 0)))
        return rows
    if method == 'vlm':
        candidates = [p for p in proposals.get(image['file_name'], [])
                      if p.get('attributes', {}).get('id_mode') == 'two_stage']
        if not candidates or not rows:
            return rows
        overlaps = box_iou([r['bbox'] for r in rows], [p['bbox'] for p in candidates])
        for row, ious in zip(rows, overlaps, strict=True):
            idx = int(np.argmax(ious))
            if ious[idx] >= .5:
                candidate = candidates[idx]
                attrs = candidate.get('attributes', {})
                row.update({'class': candidate['class'], 'identity_score': float(attrs.get('id_confidence') or 0),
                            'family': attrs.get('id_family'), 'part': attrs.get('id_part'),
                            'family_confidence': float(attrs.get('id_confidence') or 0),
                            'part_confidence': float(attrs.get('id_confidence') or 0)})
        return rows
    if emb is None or not rows:
        return rows
    indexes, targets = gallery(emb, val_gt, image['group_id'] if validation else None)
    if not indexes:
        return rows
    query = np.flatnonzero(emb['file_names'].astype(str) == image['file_name'])
    if not len(query):
        return rows
    clf = (KNNClassifier(k=5) if method == 'knn' else PrototypeClassifier())
    clf.fit(emb['embeddings'][indexes], targets)
    overlaps = box_iou([r['bbox'] for r in rows], emb['bboxes'][query])
    for row, ious in zip(rows, overlaps, strict=True):
        idx = int(np.argmax(ious))
        if ious[idx] >= .5:
            labels, scores = clf.predict_with_scores(emb['embeddings'][query[idx:idx + 1]])
            row.update({'class': str(labels[0]), 'identity_score': float(scores[0])})
    return rows


@lru_cache(maxsize=2048)
def geometry(boxes, height):
    raw = [{'box': list(box)} for box in boxes]
    rows = assign_shelves(raw) if raw else []
    shelves = shelf_labels([list(box) for box in boxes], height) if raw else []
    return [(r['shelf_idx'], r['column_idx'], r['tier'], shelf)
            for r, shelf in zip(rows, shelves, strict=True)]


def prepare_rows(rows, image, score_thr, identity_thr, pair, with_shelves=True):
    rows = [dict(r) for r in rows if float(r['score']) >= score_thr]
    for row in rows:
        if row['identity_score'] < identity_thr:
            row['class'] = 'unknown'
        row['box'] = xyxy(row['bbox'])
        match = re.match(r'(.+)-part([12])$', row['class'])
        if match and not row.get('family'):
            row.update(family=match.group(1), part=int(match.group(2)),
                       family_confidence=row['identity_score'], part_confidence=row['identity_score'])
    if rows and (with_shelves or pair and any(r.get('family') for r in rows)):
        boxes = tuple(tuple(r['box']) for r in rows)
        for row, (shelf, column, tier, label) in zip(rows, geometry(boxes, image['height']), strict=True):
            row.update(shelf_idx=shelf, column_idx=column, tier=tier, shelf_name=label)
        if pair:
            rows = infer_pairs(rows, mode='apply')
    return rows


def per_image(image, gt_anns, rows, counts, category_names, none):
    human = image['attributes']['human_counts']
    actual = counts['classes']
    predicted = Counter(r['class'] for r in rows)
    errors = {c: abs(predicted.get(c, 0) - actual.get(c, 0))
              for c in set(actual) | set(predicted)}
    gt_boxes = [a['bbox'] for a in gt_anns]
    matches = match_boxes([r['bbox'] for r in rows], gt_boxes, iou_thr=.5).matches
    correct = sum(rows[p]['class'] == category_names.get(gt_anns[g]['category_id'])
                  for p, g, _ in matches if category_names.get(gt_anns[g]['category_id']) != 'unknown')
    eligible = sum(category_names.get(gt_anns[g]['category_id']) != 'unknown' for _, g, _ in matches)
    return {'total_error': abs(len(rows) - human['total']),
            'top_error': abs(sum(r['shelf_name'] == 'top' for r in rows) - human['top']),
            'bottom_error': abs(sum(r['shelf_name'] == 'bottom' for r in rows) - human['bottom']),
            'class_errors': None if none else errors,
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
        result.update({f'class:{c}': float(np.mean([r['class_errors'].get(c, 0) for r in sample]))
                       for c in classes})
    return result


def score_subset(images, gt, counts, rows_by_image, score_thr, identity_thr, pair, none, n_bootstrap):
    annotations = defaultdict(list)
    for ann in gt['annotations']:
        annotations[ann['image_id']].append(ann)
    names = {c['id']: c['name'] for c in gt['categories']}
    classes = sorted({c for image in images for c in counts['images'][image['file_name']]['classes']})
    records = []
    loc_preds = []
    for image in images:
        rows = prepare_rows(rows_by_image[image['id']], image, score_thr, identity_thr, pair)
        records.append(per_image(image, annotations[image['id']], rows,
                                 counts['images'][image['file_name']], names, none))
        loc_preds.extend(rows)
    metrics = aggregate(records, range(len(records)), classes, none)
    loc_gt = {**gt, 'images': images,
              'annotations': [a for im in images for a in annotations[im['id']]]}
    # AP uses all scores; bootstrap resamples whole groups including their boxes.
    loc_all = [p for im in images for p in rows_by_image[im['id']]]
    prepared = _prepare_dataset(loc_gt, loc_all)
    loc_records, n_gt = _per_image_localization_records(prepared, iou_thresholds=(0.5,))
    def summary(ix):
        values = aggregate(records, ix, classes, none)
        scores = np.concatenate([loc_records[i][0.5][0] for i in ix]) if ix else np.array([])
        matches = np.concatenate([loc_records[i][0.5][1] for i in ix]) if ix else np.array([])
        values['ap50'] = _compute_coco_ap_from_records(scores, matches, sum(n_gt[i] for i in ix))
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
    candidates = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9)
    totals = {thr: sum(abs(sum(float(r['score']) >= thr for r in rows_by_image[im['id']])
                           - im['attributes']['human_counts']['total']) for im in images)
              for thr in candidates}
    best_total = min(totals.values())
    best = None
    for score_thr in (t for t in candidates if totals[t] == best_total):
        for identity_thr in ((0.,) if none else (0., .25, .5, .75)):
            class_error = 0
            if not none:
                for im in images:
                    rows = prepare_rows(rows_by_image[im['id']], im, score_thr, identity_thr, pair, False)
                    actual = counts['images'][im['file_name']]['classes']
                    predicted = Counter(r['class'] for r in rows)
                    class_error += sum(abs(predicted.get(c, 0) - actual.get(c, 0))
                                       for c in set(actual) | set(predicted))
            key = (best_total, class_error, score_thr, identity_thr)
            if best is None or key < best:
                best = key
    return best[2], best[3]


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
             'Class inventory may be incomplete where questions are unanswered or human totals disagree with crop counts; exact inventory excludes those images.',
             'Thresholds minimize validation total count error, then class count error; embedding validation uses held-out groups.',
             'Sharp-full means cabinet_full with no or mild motion blur; partial/blurry means cabinet_partial or severe blur. Other audit cases are omitted from these subsets.',
             '', 'Excluded images:']
    for split in ('val', 'test'):
        lines.append(f"- {split}: " + (', '.join(x['image_id'] + ' (' + x['reason'] + ')'
                                                for x in result['excluded'][split]) or 'none'))
    lines += ['', '| Localiser | Identifier | Pair | Subset | Images | Total MAE (95% CI) | Top MAE | Bottom MAE | Exact inventory | ID accuracy | AP50 |',
              '|---|---|---|---|---:|---|---:|---:|---:|---:|---:|']
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    def estimate(value, key):
        ci = value['bootstrap_ci95']['metrics'].get(key)
        interval = '' if ci is None else f' [{fmt(ci[0])}, {fmt(ci[1])}]'
        return fmt(value['metrics'].get(key)) + interval
    for row in result['rows']:
        for subset, value in row['subsets'].items():
            lines.append(f"| {row['run']} | {row['identifier']} | {row['pair_rule']} | {subset} | {value['n_images']} | "
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
    proposals = proposal_data(read_json(args.proposals))
    gt, counts, excluded = {}, {}, {}
    for split in ('val', 'test'):
        gt[split], counts[split], excluded[split] = build_human_gt(queue, answers, split)
        validate_groups(gt[split], split, splits)
    if not gt['val']['images']:
        raise ValueError('no eligible validation images')
    if not gt['test']['images']:
        raise ValueError('no eligible test images')
    result = {'version': 1, 'demo': args.demo, 'excluded': excluded, 'rows': [], 'skipped_runs': [],
              'limitations': ['All localization reference boxes originate from model proposals.',
                              'Per-class human counts include only answered crops.',
                              'Validation embeddings use leave-one-group-out gallery; test uses all eligible validation crops.']}
    for path in sorted(map(Path, glob.glob(args.runs))):
        val_path, test_path = path / 'val/predictions.json', path / 'test/predictions.json'
        if not val_path.is_file() or not test_path.is_file():
            result['skipped_runs'].append({'run': path.name, 'reason': 'missing val or test predictions'})
            continue
        meta = read_json(path / 'meta.json') if (path / 'meta.json').is_file() else {}
        family = meta.get('config', {}).get('family', '')
        category_names = {c['id']: c['name'] for c in gt['val']['categories']}
        predictions = {}
        for split, pred_path in (('val', val_path), ('test', test_path)):
            raw = read_json(pred_path)
            for pred in raw:
                if not pred.get('proposed_class'):
                    pred['proposed_class'] = category_names.get(pred.get('category_id'), 'unknown')
            predictions[split] = grouped_predictions(raw)
        emb = load_embeddings(path / 'embeddings.npz')
        methods = ['none', 'vlm']
        if family.startswith('D_'):
            methods.append('own')
        if emb is not None:
            methods += ['knn', 'proto']
        for method in methods:
            rows = {split: {im['id']: identify(predictions[split][im['id']], im, method, proposals, emb,
                                               gt['val'], validation=split == 'val')
                            for im in gt[split]['images']} for split in ('val', 'test')}
            for pair in (False, True):
                score_thr, id_thr = select_thresholds(gt['val']['images'], gt['val'], counts['val'],
                                                      rows['val'], pair, method == 'none')
                images = gt['test']['images']
                sharp = [im for im in images if audit.get(im['file_name'], {}).get('scene_label') == 'cabinet_full'
                         and audit.get(im['file_name'], {}).get('motion_blur') in ('none', 'mild')]
                partial = [im for im in images if audit.get(im['file_name'], {}).get('scene_label') == 'cabinet_partial'
                           or audit.get(im['file_name'], {}).get('motion_blur') == 'severe']
                subsets = {name: score_subset(ims, gt['test'], counts['test'], rows['test'], score_thr, id_thr,
                                              pair, method == 'none', args.bootstrap)
                           for name, ims in [('all', images), ('sharp-full', sharp), ('partial/blurry', partial)]}
                result['rows'].append({'run': path.name, 'identifier': method, 'pair_rule': 'on' if pair else 'off',
                                       'score_thr': score_thr, 'identity_thr': id_thr,
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
