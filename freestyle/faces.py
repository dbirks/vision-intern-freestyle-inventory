"""Experiment family G: bootstrap FRONT-FACE labels instead of whole-cartridge boxes.

Bottom-shelf cartridges lean back, so an axis-aligned box around a whole
cartridge is huge and overlaps its neighbours (see ``runs/bootstrap_v1``).  The
narrower target is the cartridge **front face**: the printed end panel that
points at the camera (brand logo + Part number / letter code).  Top shelf: the
visible upright end face.  Bottom shelf: the outward end face near the bottom of
each leaning box.  Faces are small, roughly upright and non-overlapping, so
counting and identity should get easier.

Same Vision Intern shape as :mod:`freestyle.bootstrap_labels` (label runs ->
weighted fusion -> VLM judge); only the prompt/target and the identity step
change: identity is *copied* from the whole-cartridge proposals
(``data/labels/bootstrap_v2.json``) when a face box lies inside a
whole-cartridge box (containment >= 0.8), otherwise ``unknown``.  Everything
written here is model-generated — never ground truth.

CLI::

    # prompt design on one frame (overlay + raw detections)
    uv run python -m freestyle.faces tune --image local/downloads/frame.jpg \
        --variant v1 --out-dir runs/faces_v1/viz-tune
    uv run python -m freestyle.faces tune --image local/downloads/frame.jpg \
        --variant v1 --tile 1024          # 1024 side = 2x2 tiles on 2048x1536

    uv run python -m freestyle.faces run --config configs/bootstrap/faces.yaml \
        --images runs/bootstrap_v1/images.txt --out data/labels/faces_v1.json

    uv run python -m freestyle.faces summary --labels data/labels/faces_v1.json \
        --out runs/faces_v1/summary.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy
import yaml
from PIL import Image, ImageDraw

from freestyle import bootstrap_labels as boot
from freestyle import catalog, data
from freestyle.geometry import iou_matrix, xyxy_to_xywh
from workflows.vlm_judge import judge_labels

ROOT = boot.ROOT
FACE_CLASS = "cartridge front face"
REFERENCE_LABELS = ROOT / "data/labels/bootstrap_v2.json"
DEFAULT_VARIANT = "v3"
DEFAULT_BASE_URL = "https://llm-collective.e-gineering.com/v1"
_ORIGINAL_RUN_VLM = boot.run_vlm

# ---------------------------------------------------------------- prompt design

_RULES = """\
Detect the FRONT FACE of every cartridge in this Coca-Cola Freestyle cabinet photo.

A front face is the narrow printed END panel of a cartridge that points at the \
camera: the flat panel carrying the brand logo and the Part number or letter code \
(NNS, AA, BB, CC, a circled 1 or 2). It is much narrower than the cartridge body \
and normally taller than it is wide.
- Top shelf: cartridges stand upright, so the front face is the upright printed \
end you can read (the "NNS", "AA", "BB", "CC" ends).
- Bottom shelf: cartridges lean back, so the front face is only the small outward \
end face near the BOTTOM of each leaning box (for example the Coca-Cola script \
plus a circled 1 or 2). Box that end face only, never the long side of the \
leaning box.
- Stacked or second-row cartridges count separately: one box per visible face.
- Faces must not overlap. Do not box the whole cartridge, the long body side, \
empty slots, shelf edges, or shelf signage. If two faces touch, still return two \
separate boxes; a box wider than tall usually means two faces were merged, so \
split it."""

_FORMAT = """\
For each face return a JSON object with "bbox_2d" ([x1, y1, x2, y2] normalised \
0-1000 relative to the image you were sent, not pixels), "label" \
("cartridge front face"), and "sub_label" (the short printed code you can read, \
e.g. "NNS", "AA", "BB", "CC", "1", "2", or ""). Return a JSON array of all faces, \
or [] if there are none."""

PROMPT_VARIANTS = {
    # v1: shape description only.
    "v1": f"{_RULES}\n{_FORMAT}",
    # v2: count-then-box + don't skip edge/occluded faces.
    "v2": (f"{_RULES}\nFirst count the visible front faces (top shelf, then bottom "
           "shelf), then return exactly that many boxes, ordered left to right, top "
           "shelf before bottom shelf. Every cartridge with a visible end face gets "
           "its own box, including faces at the frame edge or partly behind a "
           f"cable.\n{_FORMAT}"),
    # v3: v2 + per-shelf count prior for the fanned bottom row.
    "v3": (f"{_RULES}\nWork shelf by shelf, left to right, and box every face even "
           "where the fanned bottom row is partly hidden by the cartridge in front. "
           "A full cabinet has roughly 13-17 faces on the upper shelf and 6-13 on "
           "the lower shelf; if you return far fewer, look again at the bottom row, "
           "where the faces are small and close together.\n"
           f"{_FORMAT}"),
}

FACE_DESCRIPTION = (
    "cartridge front face (the narrow printed end panel of a cartridge pointing at "
    "the camera: brand logo + Part number/letter code; NOT the whole cartridge and "
    "NOT the long side of a leaning box)"
)


def build_prompt(variant=DEFAULT_VARIANT):
    """Detection prompt for *variant* (normalized 0-1000 coordinate convention)."""
    try:
        return PROMPT_VARIANTS[variant]
    except KeyError:
        raise ValueError(f"unknown prompt variant: {variant} "
                         f"(have {sorted(PROMPT_VARIANTS)})") from None


def endpoint(config=None, base_url=None):
    """Return ``(base_url, api_key)``; the key is read from the env, never stored."""
    config = config or {}
    return (base_url or config.get('base_url') or DEFAULT_BASE_URL,
            os.environ.get(config.get('api_key_env', 'LLMC_API_KEY')))


# ------------------------------------------------------------------- geometry


def face_containment(face, box):
    """Fraction of *face* (XYXY) lying inside *box*; 0 for a degenerate face."""
    ax1, ay1, ax2, ay2 = face
    bx1, by1, bx2, by2 = box
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    return inter / area if area else 0.0


def copy_identity(face_box, whole_boxes, names, threshold=0.8):
    """Identity of the whole-cartridge box that best contains *face_box*.

    Returns ``(class_name, containment)``; ``("unknown", best)`` when no box
    contains the face by at least *threshold*.
    """
    best_name, best_score = "unknown", 0.0
    for box, name in zip(whole_boxes, names):
        score = face_containment(face_box, box)
        if score > best_score:
            best_name, best_score = name, score
    return (best_name, best_score) if best_score >= threshold else ("unknown", best_score)


def neighbour_iou_stats(boxes):
    """Mean per-box max IoU against the other boxes of the same image (None if <2)."""
    if len(boxes) < 2:
        return None
    matrix = iou_matrix(boxes, boxes)
    numpy.fill_diagonal(matrix, 0.0)
    return float(matrix.max(axis=1).mean())


# --------------------------------------------------------------- label import


def load_reference(path=REFERENCE_LABELS):
    """Whole-cartridge proposals keyed by image_id: ``{'boxes': XYXY, 'names': [...]}``."""
    coco = json.loads(Path(path).read_text())
    names = {c['id']: c['name'] for c in coco['categories']}
    files = {im['id']: im['file_name'] for im in coco['images']}
    reference = {file_name: {'boxes': [], 'names': []} for file_name in files.values()}
    for ann in coco['annotations']:
        file_name = files.get(ann['image_id'])
        if file_name not in reference:
            continue
        x, y, w, h = ann['bbox']
        reference[file_name]['boxes'].append([x, y, x + w, y + h])
        reference[file_name]['names'].append(names.get(ann['category_id'], 'unknown'))
    return reference


# ------------------------------------------------------------------- detection


def detect_faces(image, *, prompt, model_id, base_url=None, api_key=None, tile=None,
                 tile_overlap=0.0, max_tokens=8192, max_dimension=None):
    """Detect front faces in *image*: ``[{'box', 'label', 'sub_label', 'score'}]``."""
    from tools.vlm_detect import vlm_detect
    detections = vlm_detect(
        image, prompt=prompt, classes=[FACE_CLASS], model_id=model_id,
        backend='openai', base_url=base_url, api_key=api_key,
        max_dimension=max_dimension, image_format='png', max_tokens=max_tokens,
        tile=tile, tile_overlap=tile_overlap,
    )
    out = []
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det['bbox']]
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        out.append({'box': [x1, y1, x2, y2], 'label': FACE_CLASS,
                    'sub_label': str(det.get('sub_label') or ''),
                    'score': float(det.get('score', 1.0))})
    return out


def judge_faces(image_root, image_id, boxes, *, judge, descriptions, base_url, api_key, tmp):
    """Judge fused face boxes with the Vision Intern judge (crops mode).

    Returns ``(verdicts, model_used, fell_back)`` where *verdicts* maps annotation
    id (int) -> verdict dict.  If the primary judge answers with nothing usable
    (429 budget blown, or an unparseable reply) the fallback model is tried and
    ``fell_back`` is True.
    """
    anns = [{'id': index + 1, 'image_id': 1, 'category_id': 1,
             'bbox': xyxy_to_xywh(box).tolist()} for index, box in enumerate(boxes)]
    coco_input = {'images': [{'id': 1, 'file_name': image_id}], 'annotations': anns,
                  'categories': [{'id': 1, 'name': FACE_CLASS}]}
    models = [judge['model']] + ([judge['fallback']] if judge.get('fallback') else [])
    verdicts, model_used, fell_back = [], models[0], False
    for index, model in enumerate(models):
        options = {'model_id': model, 'backend': 'openai', 'base_url': base_url,
                   'judge_mode': judge.get('mode', 'crops'),
                   'judge_batch_size': judge.get('batch_size', 8),
                   'max_dimension': None, 'image_format': 'png',
                   'max_tokens': judge.get('max_tokens', 4096), 'threshold': 0.0,
                   'class_descriptions': {FACE_CLASS.lower(): descriptions}}
        (tmp / 'input.json').write_text(json.dumps(coco_input))

        def ask():
            judge_labels(source=image_root, output=tmp / 'judged.json',
                         annotations=tmp / 'input.json', api_key=api_key, **options)
            return json.loads((tmp / 'judged_verdicts.json').read_text())
        verdicts = ask()
        model_used, fell_back = model, index > 0
        if any(v.get('verdict', 'unjudged') != 'unjudged' for v in verdicts):
            break
    return ({int(v['annotation_id']): v for v in verdicts}, model_used, fell_back)


# --------------------------------------------------------------------- overlay


def render_overlay(image, boxes, out, *, max_width=1400, title='', sub_labels=None,
                   whole_boxes=None):
    """Draw numbered face boxes (blue) on *image*, downscaled to at most *max_width*."""
    scale = min(1.0, max_width / image.width)
    canvas = (image.resize((round(image.width * scale), round(image.height * scale)),
                           Image.LANCZOS) if scale < 1 else image.copy())
    draw = ImageDraw.Draw(canvas)
    for box in whole_boxes or []:
        draw.rectangle([v * scale for v in box], outline=(235, 235, 235), width=1)
    for index, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = [v * scale for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(30, 120, 255), width=max(2, round(3 * scale)))
        text = str(index) + (f" {sub_labels[index - 1][:6]}"
                            if sub_labels and index - 1 < len(sub_labels) and sub_labels[index - 1]
                            else '')
        top = max(0.0, y1 - 15 * scale - 3)
        draw.rectangle([x1, top, x1 + draw.textlength(text) + 5, top + 14 * scale + 3],
                       fill=(30, 120, 255))
        draw.text((x1 + 3, top + 1), text, fill='white')
    if title:
        draw.rectangle([0, 0, canvas.width, 22], fill=(20, 20, 20))
        draw.text((6, 5), title[:200], fill='white')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def overlay_name(image_id, tag):
    """File name for one overlay/JSON pair (image ids contain '/')."""
    return f"{image_id.replace('/', '__')}-{tag}"


# ------------------------------------------------------------------- pipeline

TIMING: list[dict] = []


def _instrumented_run_vlm(image, prompt, model_id, **kwargs):
    """Cache-miss counter/timer for fresh Collective calls (no key, no payload)."""
    started = time.monotonic()
    record = {'ts': time.time(), 'model': model_id, 'prompt_head': prompt[:80]}
    try:
        return _ORIGINAL_RUN_VLM(image, prompt, model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        record['error'] = str(exc)[:200]
        record['status_code'] = getattr(exc, 'status_code', None)
        raise
    finally:
        record['duration_s'] = round(time.monotonic() - started, 2)
        TIMING.append(record)


def bootstrap_faces(config, images, out, *, reference=None):
    """Run the face bootstrap over *images*; write model-generated COCO to *out*."""
    out = Path(out)
    if out.resolve().is_relative_to((ROOT / 'data/gt').resolve()):
        raise ValueError('proposals must not be written under data/gt/')
    cache_dir = Path(config.get('cache_dir', ROOT / 'data/labels/cache'))
    root = Path(config.get('images_root', data.IMAGES_ROOT))
    names = catalog.class_names(config.get('catalog'))
    ids = {name: index + 1 for index, name in enumerate(names)}
    base_url, api_key = endpoint(config)
    threshold = config.get('identity_containment', 0.8)
    if reference is None:
        reference = load_reference(config.get('reference_labels', REFERENCE_LABELS))
    description = config.get('face_description', FACE_DESCRIPTION)
    variant = config.get('prompt_variant', DEFAULT_VARIANT)
    prompt = build_prompt(variant)
    cached_vlm = boot._cached_vlm(cache_dir)
    result = {'info': {
        'description': 'Model-generated front-face proposals; not ground truth',
        'target': FACE_CLASS, 'prompt_variant': variant,
        'identity_source': Path(config.get('reference_labels', REFERENCE_LABELS)).name,
        'identity_containment': threshold},
        'images': [], 'annotations': [],
        'categories': [{'id': index + 1, 'name': name} for index, name in enumerate(names)]}
    fallbacks = []
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        with patch('tools.vlm_detect.run_vlm', cached_vlm), \
                patch('workflows.vlm_judge.run_vlm', cached_vlm), \
                patch.object(boot, 'run_vlm', _instrumented_run_vlm):
            for image_index, record in enumerate(images, 1):
                image_id = record['image_id']
                path = root / image_id
                image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as opened:
                    image = opened.convert('RGB')
                info = {'id': image_index, 'file_name': image_id, 'width': image.width,
                        'height': image.height, 'group_id': record.get('group_id'),
                        'attributes': {'verified': False, 'target': FACE_CLASS, 'image_area':
                                   image.width * image.height}}
                result['images'].append(info)
                runs = []
                for run_index, run in enumerate(config['labellers'], 1):
                    options = {'model': run['model'], 'prompt': prompt, 'tile': run.get('tile'),
                               'tile_overlap': run.get('tile_overlap', 0.0),
                               'max_tokens': run.get('max_tokens', 8192), 'base_url': base_url}

                    def detect():
                        cached_vlm.reset()
                        return detect_faces(
                            image, prompt=options['prompt'], model_id=options['model'],
                            base_url=base_url, api_key=api_key, tile=options['tile'],
                            tile_overlap=options['tile_overlap'],
                            max_tokens=options['max_tokens'])
                    detections = boot._cache(cache_dir, ['faces-detect-v1', image_hash,
                                                         image_id, options], detect)
                    runs.append({'run': run.get('name', str(run_index)),
                                 'annotations': [{'bbox': xyxy_to_xywh(det['box']).tolist(),
                                                  'score': det.get('score', 1.0)}
                                                 for det in detections]})
                fused = boot.fuse_runs(runs, config.get('fusion_iou', 0.55))
                boxes = [det['box'] for det in fused]

                def judged():
                    cached_vlm.reset()
                    verdicts, model_used, fell_back = judge_faces(
                        root, image_id, boxes, judge=config['judge'], descriptions=description,
                        base_url=base_url, api_key=api_key, tmp=tmp)
                    return [{str(k): v for k, v in verdicts.items()}, model_used, fell_back]
                verdicts, judge_model, fell_back = boot._cache(
                    cache_dir, ['faces-judge-v1', image_hash, boxes, config.get('judge')],
                    judged) if boxes else ({}, config['judge']['model'], False)
                verdicts = {int(k): v for k, v in verdicts.items()}
                if fell_back:
                    fallbacks.append(image_id)
                for det_index, det in enumerate(fused, 1):
                    identity, containment = copy_identity(det['box'], reference.get(image_id, {})
                                                          .get('boxes', []),
                                                          reference.get(image_id, {})
                                                          .get('names', []), threshold)
                    if identity not in ids:
                        identity = 'unknown'
                    x1, y1, x2, y2 = det['box']
                    verdict = verdicts.get(det_index, {})
                    wide = (x2 - x1) > (y2 - y1)      # real faces are taller than wide
                    if config.get('drop_wide_boxes') and wide:
                        continue
                    result['annotations'].append({
                        'id': len(result['annotations']) + 1, 'image_id': image_index,
                        'category_id': ids[identity],
                        'bbox': [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                        'area': round((x2 - x1) * (y2 - y1), 1), 'iscrowd': 0,
                        'attributes': {
                            'source': 'vlm:faces', 'face_class': FACE_CLASS,
                            'wider_than_tall': wide,
                            'sub_label': det.get('sub_label', ''),
                            'truncated': False, 'occluded': False,
                            'identity_visible': identity != 'unknown',
                            'identity_containment': round(containment, 3),
                            'support': det['support'], 'provenance': det['provenance'],
                            'judge_verdict': verdict.get('verdict', 'unjudged'),
                            'judge_score': verdict.get('score', 0.0),
                            'judge_model': judge_model}})
    result['info']['judge_fallbacks'] = fallbacks
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    return result


# -------------------------------------------------------------------- summary


def class_counts(anns, names):
    counts: dict[str, int] = {}
    for ann in anns:
        name = names.get(ann['category_id'], '?')
        if name == 'unknown':
            continue
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _mean(values):
    """Mean of *values*, or None when there is nothing to average."""
    return statistics.mean(values) if values else None


def _fmt(value, digits=3):
    """Stat string; em dash when too few boxes existed to score it."""
    return '—' if value is None else f'{value:.{digits}f}'


def _subset_iou(by_image, keep):
    """Mean per-image neighbour max-IoU over the subset of faces passing *keep*."""
    values = []
    for pairs in by_image.values():
        boxes = [box for box, ann in pairs if keep(ann)]
        iou = neighbour_iou_stats(boxes)
        if iou is not None:
            values.append(iou)
    return len(values), _mean(values)


def summarize(labels, reference=None, *, timings=None, overlays=(), images_read=None,
              identity_threshold=0.8, judgement=None):
    """Markdown summary: face vs whole-cartridge counts and neighbour overlap."""
    coco = json.loads(Path(labels).read_text())
    reference = reference if reference is not None else load_reference()
    names = {c['id']: c['name'] for c in coco['categories']}
    by_image: dict[int, list] = {}
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        by_image.setdefault(ann['image_id'], []).append(([x, y, x + w, y + h], ann))
    files = {im['id']: im['file_name'] for im in coco['images']}

    rows, face_ious, whole_ious = [], [], []
    for image_id, image_file in sorted(files.items(), key=lambda kv: kv[1]):
        pairs = by_image.get(image_id, [])
        face_boxes = [box for box, _ in pairs]
        whole_boxes = reference.get(image_file, {}).get('boxes', [])
        face_iou, whole_iou = neighbour_iou_stats(face_boxes), neighbour_iou_stats(whole_boxes)
        if face_iou is not None:
            face_ious.append(face_iou)
        if whole_iou is not None:
            whole_ious.append(whole_iou)
        rows.append((image_file, len(face_boxes), len(whole_boxes), face_iou, whole_iou))

    anns = coco['annotations']
    attributes = [a.get('attributes', {}) for a in anns]
    identified = sum(1 for a in attributes if a.get('identity_visible'))
    verdicts: dict[str, int] = {}
    for a in attributes:
        key = a.get('judge_verdict', 'unjudged')
        verdicts[key] = verdicts.get(key, 0) + 1
    supports = [a.get('support', 0) for a in attributes]
    wide = sum(1 for ann in anns if ann['bbox'][2] > ann['bbox'][3])
    image_area = {im['id']: im.get('attributes', {}).get('image_area') or im['width'] * im['height']
                  for im in coco['images']}
    median_area_frac = statistics.median(
        [ann['area'] / image_area[ann['image_id']] for ann in anns]) if anns else 0.0

    lines = ['# Faces v1 — front-face proposals (experiment family G; NOT ground truth)', '']
    lines.append(f"- target: `{FACE_CLASS}`; prompt variant "
                 f"`{coco['info'].get('prompt_variant')}`; {len(coco['images'])} images, "
                 f"{len(anns)} face boxes")
    lines.append(f"- identity copied from `{coco['info'].get('identity_source', 'bootstrap_v2.json')}` "
                 f"at containment >= {identity_threshold}: {identified}/{len(anns)} identified, "
                 f"{len(anns) - identified} unknown")
    lines.append(f"- **neighbour overlap (mean max IoU)**: faces "
                 f"{_fmt(_mean(face_ious))} vs whole cartridges "
                 f"{_fmt(_mean(whole_ious))} — lower is the point of the idea")
    lines.append(f"- mean fusion support {_fmt(_mean(supports), 2)}; median face area "
                 f"{median_area_frac * 100:.2f}% of frame; {wide} boxes wider than tall "
                 f"(merged-face smell); judge verdicts {verdicts}")
    lines += ['', '## Overlap by quality tier (mean neighbour max-IoU; whole cartridges are the baseline)', '']
    lines.append('| subset | images scored | mean neighbour max-IoU |')
    lines.append('|---|---|---|')
    for label, keep in (('all faces', lambda ann: True),
                        ('both labeller runs agree (support 1.0)',
                         lambda ann: ann.get('attributes', {}).get('support', 0) >= 0.99),
                        ('upright only (taller than wide)',
                         lambda ann: not ann.get('attributes', {}).get('wider_than_tall', False)),
                        ('agreed **and** upright',
                         lambda ann: ann.get('attributes', {}).get('support', 0) >= 0.99
                         and not ann.get('attributes', {}).get('wider_than_tall', False))):
        scored, mean = _subset_iou(by_image, keep)
        n = sum(1 for pairs in by_image.values() for _, ann in pairs if keep(ann))
        lines.append(f'| {label} ({n} boxes) | {scored} | {_fmt(mean)} |')
    lines.append(f'| whole cartridges, bootstrap_v2 ({sum(len(r["boxes"]) for r in reference.values())} '
                 f'boxes) | {len(whole_ious)} | {_fmt(_mean(whole_ious))} |')
    lines.append('')
    fallbacks = coco['info'].get('judge_fallbacks') or []
    lines.append(f"- judge models: {sorted({a.get('judge_model', '?') for a in attributes})}"
                 + (f", fell back on {len(fallbacks)} image(s): {fallbacks}" if fallbacks else
                    " (no fallback needed)"))
    if timings:
        lat = [t['duration_s'] for t in timings]
        lines.append(f"- fresh Collective calls: {len(timings)}, errors "
                     f"{sum(1 for t in timings if t.get('error'))}, mean latency "
                     f"{_fmt(_mean(lat), 1)}s, max {max(lat):.1f}s")
    if images_read is not None:
        lines.append(f"- images read into the worker context (downscaled): {images_read}")
    if overlays:
        lines.append('- overlays reviewed by the worker: '
                     + ', '.join(f'`{o}`' for o in overlays))
    lines += ['', '## Per image: face boxes vs whole-cartridge boxes (bootstrap_v2)', '',
              '| image | faces | whole | mean max-IoU faces | mean max-IoU whole |',
              '|---|---|---|---|---|']
    for image_file, n_faces, n_whole, face_iou, whole_iou in rows:
        lines.append(f'| `{image_file}` | {n_faces} | {n_whole} | '
                     + (f'{face_iou:.3f}' if face_iou is not None else '—') + ' | '
                     + (f'{whole_iou:.3f}' if whole_iou is not None else '—') + ' |')
    lines += ['', f'- totals: {sum(r[1] for r in rows)} faces vs {sum(r[2] for r in rows)} '
              'whole-cartridge boxes across {0} images'.format(len(rows)),
              f"- identified classes: {json.dumps(class_counts(anns, names))}", '']
    if judgement:
        lines += ['## Honest visual judgement (worker looked at these overlays)', '',
                  Path(judgement).read_text().rstrip(), '']
    else:
        lines += ['## Honest visual judgement', '', '(filled in by the worker after looking at overlays)', '']
    return '\n'.join(lines)


# ------------------------------------------------------------------------ CLI


def _tune(args):
    record = {r['image_id']: r for r in data.list_images()}
    if args.image not in record:
        raise SystemExit(f'unknown image id: {args.image}')
    base_url, api_key = endpoint()
    path = data.IMAGES_ROOT / args.image
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    started = time.monotonic()
    detections = detect_faces(image, prompt=build_prompt(args.variant), model_id=args.model,
                             base_url=base_url, api_key=api_key, tile=args.tile,
                             tile_overlap=args.tile_overlap)
    tag = f"{args.variant}-{'tiles' if args.tile else 'full'}"
    out = overlay_name(args.image, tag)
    whole = load_reference().get(args.image, {}).get('boxes', []) if args.show_whole else None
    render_overlay(image, [d['box'] for d in detections], args.out_dir / f'{out}.png',
                   title=f'{args.image} [{tag}] {len(detections)} faces '
                         f'({time.monotonic() - started:.1f}s, {args.model})',
                   sub_labels=[d['sub_label'] for d in detections], whole_boxes=whole)
    (args.out_dir / f'{out}.json').write_text(json.dumps(detections, indent=2))
    print(f'{tag}: {len(detections)} faces in {time.monotonic() - started:.1f}s')
    for det in detections:
        print('  ', [round(v) for v in det['box']], repr(det['sub_label']))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='cmd', required=True)

    tune = sub.add_parser('tune', help='one prompt variant on one image, with overlay')
    tune.add_argument('--image', required=True)
    tune.add_argument('--variant', default=DEFAULT_VARIANT, choices=sorted(PROMPT_VARIANTS))
    tune.add_argument('--model', default='david-birks/inkling-2-mini')
    tune.add_argument('--tile', type=int, default=None,
                      help='tile side in px (1024 -> 2x2 tiles on a 2048x1536 frame)')
    tune.add_argument('--tile-overlap', type=float, default=0.0)
    tune.add_argument('--out-dir', type=Path, default=ROOT / 'runs/faces_v1/viz-tune')
    tune.add_argument('--show-whole', action='store_true',
                      help='draw the whole-cartridge boxes faintly for comparison')

    run = sub.add_parser('run', help='full bootstrap over an image list')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--images', required=True, help='split name or file of image ids')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--timing', type=Path, default=None)

    summ = sub.add_parser('summary')
    summ.add_argument('--labels', type=Path, default=ROOT / 'data/labels/faces_v1.json')
    summ.add_argument('--out', type=Path, default=ROOT / 'runs/faces_v1/summary.md')
    summ.add_argument('--timing', type=Path, default=ROOT / 'runs/faces_v1/timing.json')
    summ.add_argument('--overlay', action='append', default=[])
    summ.add_argument('--images-read', type=int, default=None)
    summ.add_argument('--judgement', type=Path, default=None,
                      help='markdown file with the worker\'s visual judgement to splice in')

    args = parser.parse_args(argv)
    if args.cmd == 'tune':
        return _tune(args)
    if args.cmd == 'run':
        config = yaml.safe_load(args.config.read_text())
        result = bootstrap_faces(config, boot._images(args.images), args.out)
        if args.timing and TIMING:
            args.timing.parent.mkdir(parents=True, exist_ok=True)
            args.timing.write_text(json.dumps(TIMING, indent=2))
        print(f'wrote {args.out}: {len(result["images"])} images, '
              f'{len(result["annotations"])} faces, {len(TIMING)} fresh calls')
        return 0
    timings = json.loads(args.timing.read_text()) if args.timing.is_file() else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summarize(args.labels, timings=timings, overlays=args.overlay,
                                 images_read=args.images_read, judgement=args.judgement))
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
