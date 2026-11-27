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

    # clean the v1 proposals into the faces_v2 training labels (no new VLM calls)
    uv run python -m freestyle.faces clean --labels data/labels/faces_v1.json \
        --out data/labels/faces_v2.json --out-dir runs/faces_v2
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy
import yaml
from PIL import Image, ImageDraw

from freestyle import bootstrap_labels as boot
from freestyle import catalog, data
from freestyle.count.shelves import assign_shelves
from freestyle.geometry import iou_matrix, xyxy_to_xywh
from workflows.vlm_judge import judge_labels

ROOT = boot.ROOT
FACE_CLASS = "cartridge front face"
REFERENCE_LABELS = ROOT / "data/labels/bootstrap_v2.json"
FACES_V1_LABELS = ROOT / "data/labels/faces_v1.json"
FACES_V2_LABELS = ROOT / "data/labels/faces_v2.json"
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
    """Whole-cartridge proposals keyed by image_id: ``{'boxes': XYXY, 'names': [...],
    'families': [...], 'parts': [...]}`` (the last two parallel to *boxes*, from the
    reference annotations' own ``id_family`` / ``id_part`` attributes, ``None`` when absent)."""
    coco = json.loads(Path(path).read_text())
    names = {c['id']: c['name'] for c in coco['categories']}
    files = {im['id']: im['file_name'] for im in coco['images']}
    reference = {file_name: {'boxes': [], 'names': [], 'families': [], 'parts': []}
                 for file_name in files.values()}
    for ann in coco['annotations']:
        file_name = files.get(ann['image_id'])
        if file_name not in reference:
            continue
        x, y, w, h = ann['bbox']
        attributes = ann.get('attributes', {})
        reference[file_name]['boxes'].append([x, y, x + w, y + h])
        reference[file_name]['names'].append(names.get(ann['category_id'], 'unknown'))
        reference[file_name]['families'].append(attributes.get('id_family'))
        reference[file_name]['parts'].append(attributes.get('id_part'))
    return reference


def best_containing(face_box, boxes, threshold=0.8):
    """``(index, containment)`` of the most containing box, or ``(None, best)`` below *threshold*."""
    scores = [face_containment(face_box, box) for box in boxes]
    if not scores:
        return None, 0.0
    index = max(range(len(scores)), key=lambda i: scores[i])
    return ((index, scores[index]) if scores[index] >= threshold else (None, scores[index]))


def reference_identity(file_name, face_box, reference, threshold=0.8):
    """Identity inherited from the whole-cartridge box that contains *face_box*.

    Returns ``{'class', 'family', 'part', 'containment'}`` (``class`` ``'unknown'`` and the
    optional fields ``None`` when no reference box contains the face).
    """
    entry = (reference or {}).get(file_name) or {}
    boxes = entry.get('boxes', [])
    index, containment = best_containing(face_box, boxes, threshold)
    if index is None:
        return {'class': 'unknown', 'family': None, 'part': None, 'containment': containment}
    def field(key):
        values = entry.get(key) or []
        return values[index] if index < len(values) else None
    return {'class': field('names') or 'unknown', 'family': field('families'),
            'part': field('parts'), 'containment': containment}


# ------------------------------------------------------------------- detection


def detect_options(run, prompt, base_url=None):
    """The cache-key options of one labeller *run* (shared by :func:`detect_faces` and
    :func:`recover_sub_labels`, so the cache keys of a finished run stay reproducible)."""
    return {'model': run['model'], 'prompt': prompt, 'tile': run.get('tile'),
            'tile_overlap': run.get('tile_overlap', 0.0),
            'max_tokens': run.get('max_tokens', 8192), 'base_url': base_url}


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
                   whole_boxes=None, drop_boxes=None):
    """Draw numbered face boxes (blue) on *image*, downscaled to at most *max_width*.

    *drop_boxes* (rejected candidates) are drawn in red without numbers, *whole_boxes* in
    faint white, so one image shows the whole decision.
    """
    scale = min(1.0, max_width / image.width)
    canvas = (image.resize((round(image.width * scale), round(image.height * scale)),
                           Image.LANCZOS) if scale < 1 else image.copy())
    draw = ImageDraw.Draw(canvas)
    for box in whole_boxes or []:
        draw.rectangle([v * scale for v in box], outline=(235, 235, 235), width=1)
    for box in drop_boxes or []:
        draw.rectangle([v * scale for v in box], outline=(235, 60, 60),
                       width=max(2, round(3 * scale)))
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


# ---------------------------------------------------------------- v2: cleaning

#: Class-agnostic detection target of experiment family G (configs/experiments/G*.yaml).
FACE_TARGET_SLUG = 'cartridge_face'
LABEL_VERSION = 'faces_v2'

# Circled / full-width digits the labellers actually print for the Part number.
_PART_TRANSLATE = {0x2460: '1', 0x2461: '2', 0x2776: '1', 0x2777: '2',
                   0xFF11: '1', 0xFF12: '2'}

#: Thresholds of :func:`clean_faces`; every one of them is justified in ``runs/faces_v1``.
CLEAN_RULES = {
    # Both labeller runs found the face: keep it even when the judge was lukewarm.
    # (runs/faces_v1: neighbour max-IoU 0.098 on these vs 0.146 on all faces.)
    'support_agreed': 1.0,
    # A single run is only evidence when the judge agreed with it (60 of the 260
    # single-run faces were judged incorrect, 50 of those wider than tall).
    'support_judged': 0.5,
    'judge_keep_verdicts': ('correct',),
    'drop_verdicts': ('incorrect',),
    # An end face is square-to-tall.  A bottom-shelf box wider than 1.6x its height is
    # the long side of a leaning cartridge, not its outward end.
    'max_aspect_bottom': 1.6,
    # A face much narrower than the median face of its own shelf band is the printed
    # side strip the 2x2-tile labeller boxes along a leaning box.
    'min_width_frac': 0.35,
    # Faces sharing >40% of their x-interval *and* separated vertically by at least one face
    # height sit on one leaning cartridge (both its ends were boxed in faces_v1); only the
    # lowest one, the outward end, is kept.  The vertical guard matters: the fanned bottom row
    # overlaps itself in x, so without it one chain swallowed 17 real faces of a val frame.
    'column_overlap': 0.4,
    'column_min_y_gap': 1.0,
    'max_rows': 2,
}


def _as_labels(labels):
    """Accept a COCO dict or a path to one; return the label dict."""
    if isinstance(labels, (str, Path)):
        return json.loads(Path(labels).read_text())
    return labels


def parse_part(text):
    """Part numeral (1 or 2) readable in a face's printed code, or ``None``.

    Handles the forms the labellers emit: ``"1"``, ``"Part 2"``, ``"\u2461````, and refuses
    anything ambiguous (two numbers) or outside the two parts a family has.
    """
    if text is None:
        return None
    normalized = str(text).translate(_PART_TRANSLATE)
    numbers = re.findall(r'\d+', normalized)
    if len(numbers) != 1:
        return None
    return int(numbers[0]) if numbers[0] in ('1', '2') else None


def box_key(box):
    """Hashable key for a box, stable across the JSON round trip of the label cache."""
    return tuple(round(float(v), 3) for v in box)


def _xyxy(ann):
    x, y, w, h = ann['bbox']
    return [float(x), float(y), float(x + w), float(y + h)]


def x_overlap_ratio(a, b):
    """Overlap of two boxes' x-intervals relative to the narrower box (0..1)."""
    narrower = min(a[2] - a[0], b[2] - b[0])
    if narrower <= 0:
        return 0.0
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / narrower


def x_columns(boxes, overlap=0.4, min_y_gap=None):
    """Group box indices by transitive x-interval overlap > *overlap*.

    When *min_y_gap* (pixels) is given, two boxes join a group only if they are also
    separated vertically by at least that much, so a row that overlaps itself in x (the
    fanned bottom shelf) stays a row of separate faces instead of chaining into one column.
    """
    def same_column(a, b):
        if x_overlap_ratio(a, b) <= overlap:
            return False
        if min_y_gap is None:
            return True
        return max(a[1], b[1]) - min(a[3], b[3]) >= min_y_gap

    parent = list(range(len(boxes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(boxes)), 2):
        if same_column(boxes[i], boxes[j]):
            parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(root(i), []).append(i)
    return [sorted(members) for _, members in sorted(groups.items(), key=lambda kv: kv[1][0])]


def shelf_bands(boxes, max_rows=2):
    """``(band index per box, distinct bands top-to-bottom)`` from the heuristic row fits.

    Falls back to a single band when the fit is impossible (too few boxes, degenerate
    boxes), which keeps :func:`bottom_band` from acting on a shelf it cannot see.
    """
    if len(boxes) < 2:
        return [0] * len(boxes), [0]
    try:
        assigned = assign_shelves([{'box': [float(v) for v in box]} for box in boxes],
                                  max_rows=max_rows)
    except ValueError:
        return [0] * len(boxes), [0]
    bands = [int(item['shelf_idx']) for item in assigned]
    return bands, sorted(set(bands))


def bottom_band(boxes, bands, distinct):
    """Index of the leaning bottom shelf, or ``None`` when the frame shows only one band.

    The lowest fitted band must also be lower in the frame than every other band, so a
    top-only frame (or a nonsense fit) is never collapsed as if it were the fanned row.
    """
    if len(distinct) < 2:
        return None
    lowest = max(distinct)
    def centre_y(band):
        rows = [boxes[i] for i, b in enumerate(bands) if b == band]
        return statistics.mean([(b[1] + b[3]) / 2 for b in rows])
    others = [band for band in distinct if band != lowest]
    return lowest if centre_y(lowest) > max(centre_y(band) for band in others) else None


def face_shape_flags(boxes, bands, bottom, *, max_aspect_bottom=1.6, min_width_frac=0.35):
    """``{box index: reason}`` for shapes that cannot be an end face, judged per shelf band.

    ``'flat-aspect'``: on the bottom shelf, wider than *max_aspect_bottom* x its height.
    ``'thin-strip'``: narrower than *min_width_frac* x the median face width of its band.
    """
    medians = {}
    for band in set(bands):
        widths = sorted(boxes[i][2] - boxes[i][0] for i, b in enumerate(bands) if b == band)
        medians[band] = statistics.median(widths)
    flags = {}
    for index, (box, band) in enumerate(zip(boxes, bands)):
        width, height = box[2] - box[0], box[3] - box[1]
        if width <= 0 or height <= 0:
            flags[index] = 'degenerate'
        elif band == bottom and width / height > max_aspect_bottom:
            flags[index] = 'flat-aspect'
        elif width < min_width_frac * medians[band]:
            flags[index] = 'thin-strip'
    return flags


def lowest_per_column(boxes, anns, overlap=0.4, min_y_gap=None):
    """Keep the outward end of every leaning column: the LOWEST face of the group wins.

    A leaning cartridge presents one readable end face near the bottom of its whole-box;
    faces_v1 boxed the far end (and split duplicates) as well.  Ties are broken towards
    the stronger evidence (support, then judge score, then area).

    Returns ``(kept_indices, {dropped index: reason})``.
    """
    def rank(index):
        attributes = anns[index].get('attributes', {}) if index < len(anns) else {}
        box = boxes[index]
        return ((box[1] + box[3]) / 2.0,                        # bottom-most centre first
                float(attributes.get('support') or 0.0),
                float(attributes.get('judge_score') or 0.0),
                (box[2] - box[0]) * (box[3] - box[1]))

    kept, dropped = list(range(len(boxes))), {}
    for group in x_columns(boxes, overlap, min_y_gap):
        if len(group) < 2:
            continue
        winner = max(group, key=rank)
        for index in group:
            if index != winner:
                dropped[index] = 'leaning-column-duplicate'
    return [i for i in kept if i not in dropped], dropped


def y_overlap_containment(a, b):
    """Overlap of two boxes' y-intervals relative to the shorter box (0..1)."""
    shorter = min(a[3] - a[1], b[3] - b[1])
    if shorter <= 0:
        return 0.0
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / shorter


def residual_duplicates(boxes, x_ratio=0.8, y_contain=0.75):
    """Index pairs that are almost certainly one end boxed twice (loose + tight box).

    Reported, never deleted: unlike a leaning column's two ends, these two boxes overlap
    vertically, so no geometric rule separates them from a real back-row face behind a
    front-row one — that distinction needs a human (or a per-crop judge re-run).
    """
    pairs = []
    for i, j in itertools.combinations(range(len(boxes)), 2):
        if (x_overlap_ratio(boxes[i], boxes[j]) >= x_ratio
                and y_overlap_containment(boxes[i], boxes[j]) >= y_contain):
            pairs.append((i, j))
    return pairs


def face_sub_label(lookup, file_name, provenance):
    """Printed code of one fused face: majority vote of the runs that saw it (first run wins)."""
    if not lookup:
        return ''
    texts = []
    for entry in provenance or []:
        for box in entry.get('boxes', []):
            text = lookup.get((file_name, entry.get('run'), box_key(box)))
            if text:
                texts.append(str(text).strip())
                break
    if not texts:
        return ''
    counts = Counter(texts)
    best = max(counts.values())
    return next(text for text in texts if counts[text] == best)


def recover_sub_labels(labels, config, *, cache_dir=None, images_root=None):
    """Recover the printed codes (``NNS``/``AA``/``1``/``2``) of a finished faces run.

    :func:`freestyle.bootstrap_labels.fuse_runs` keeps only boxes, so the ``sub_label``
    text the labellers returned never reaches ``faces_v1.json``.  The per-run detections
    are still in the label cache (key: tag + image hash + image id + detection options)
    and the fused provenance records the exact source boxes, so the codes are recovered
    offline, without touching the Collective again.
    """
    coco = _as_labels(labels)
    cache_dir = Path(cache_dir or config.get('cache_dir', ROOT / 'data/labels/cache'))
    root = Path(images_root or config.get('images_root', data.IMAGES_ROOT))
    prompt = build_prompt(config.get('prompt_variant', DEFAULT_VARIANT))
    base_url = config.get('base_url') or DEFAULT_BASE_URL
    lookup = {}
    for run_index, run in enumerate(config.get('labellers', []), 1):
        name = run.get('name', str(run_index))
        options = detect_options(run, prompt, base_url)
        for image in coco['images']:
            file_name = image['file_name']
            path = root / file_name
            if not path.is_file():
                continue
            key = ['faces-detect-v1', hashlib.sha256(path.read_bytes()).hexdigest(),
                   file_name, options]
            cached = cache_dir / (boot._hash(key) + '.json')
            if not cached.is_file():
                continue
            for detection in json.loads(cached.read_text()):
                lookup[(file_name, name, box_key(detection['box']))] = detection.get('sub_label', '')
    return lookup


def _drop(file_name, ann, reason):
    attributes = ann.get('attributes', {})
    return {'file_name': file_name, 'bbox': list(ann['bbox']), 'reason': reason,
            'support': attributes.get('support', 0.0),
            'judge_verdict': attributes.get('judge_verdict', 'unjudged'),
            'judge_score': attributes.get('judge_score', 0.0), 'area': ann.get('area', 0)}


def clean_faces(labels, *, reference=None, sub_labels=None, rules=None, threshold=0.8):
    """Turn faces_v1 proposals into the faces_v2 labels: keep only defensible faces.

    * support 1.0 (both labeller runs) survives, and so does a single-run face the judge
      called correct; every judge-rejected face goes.
    * on the leaning bottom shelf one face per column survives — the LOWEST, i.e. the
      outward end a face reader would use (x-interval overlap > 0.4 groups a column, but
      only when the two faces are ≥ one face height apart vertically; see CLEAN_RULES).
    * shapes that cannot be end faces go too (flat bottom boxes, thin side strips).

    ``reference`` (see :func:`load_reference`) supplies the family / Part fallback, exactly
    as the identity class is copied by containment in :func:`bootstrap_faces`; *sub_labels*
    comes from :func:`recover_sub_labels`.  Model-generated labels, never ground truth.

    Returns ``(coco, drops)``: a faces_v1-shaped COCO dict and one record per dropped face.
    """
    coco = copy.deepcopy(_as_labels(labels))
    if rules:
        rules = {**CLEAN_RULES, **rules}
    else:
        rules = dict(CLEAN_RULES)
    files = {image['id']: image['file_name'] for image in coco['images']}
    by_image: dict[int, list] = {}
    for ann in coco['annotations']:
        by_image.setdefault(ann['image_id'], []).append(ann)

    kept, drops, tally, kept_by = [], [], Counter(), Counter()
    for image in coco['images']:
        file_name = files[image['id']]
        survivors = []
        for ann in by_image.get(image['id'], []):
            attributes = ann.get('attributes', {})
            verdict = str(attributes.get('judge_verdict', 'unjudged'))
            support = float(attributes.get('support') or 0.0)
            if verdict in rules['drop_verdicts']:
                reason = 'judge-incorrect'
            elif support >= rules['support_agreed'] - 1e-9:
                survivors.append({'ann': ann, 'keep': 'agreed-support'})
                continue
            elif (verdict in rules['judge_keep_verdicts']
                  and support >= rules['support_judged'] - 1e-9):
                survivors.append({'ann': ann, 'keep': 'judge-correct'})
                continue
            else:
                reason = 'weak-evidence'
            tally[reason] += 1
            drops.append(_drop(file_name, ann, reason))

        for entry in survivors:
            entry['box'] = _xyxy(entry['ann'])
        boxes = [entry['box'] for entry in survivors]
        bands, distinct = shelf_bands(boxes, rules['max_rows'])
        for entry, band in zip(survivors, bands):
            entry['band'] = band
        bottom = bottom_band(boxes, bands, distinct)
        flags = face_shape_flags(boxes, bands, bottom, max_aspect_bottom=rules['max_aspect_bottom'],
                                 min_width_frac=rules['min_width_frac'])
        for index, reason in sorted(flags.items()):
            survivors[index]['drop'] = reason
            tally[reason] += 1
            drops.append(_drop(file_name, survivors[index]['ann'], reason))
        survivors = [entry for entry in survivors if 'drop' not in entry]
        if bottom is not None:
            group = [entry for entry in survivors if entry['band'] == bottom]
            heights = [entry['box'][3] - entry['box'][1] for entry in group] or [1.0]
            _kept_column, column_drops = lowest_per_column(
                [entry['box'] for entry in group], [entry['ann'] for entry in group],
                rules['column_overlap'],
                rules['column_min_y_gap'] * statistics.median(heights))
            for index, reason in sorted(column_drops.items()):
                group[index]['drop'] = reason
                tally[reason] += 1
                drops.append(_drop(file_name, group[index]['ann'], reason))
            survivors = [entry for entry in survivors if 'drop' not in entry]

        for entry in survivors:
            ann = entry['ann']
            attributes = dict(ann.get('attributes', {}))
            box = entry['box']
            text = face_sub_label(sub_labels, file_name, attributes.get('provenance')) \
                or str(attributes.get('sub_label') or '')
            part = parse_part(text)
            source = 'face_text' if part is not None else None
            family = None
            if reference is not None:
                inherited = reference_identity(file_name, box, reference, threshold)
                family = inherited['family']
                if part is None and inherited['part'] is not None:
                    part, source = int(inherited['part']), 'whole_box'
            width, height = box[2] - box[0], box[3] - box[1]
            attributes.update({
                'label_version': LABEL_VERSION, 'clean_keep': entry['keep'],
                'sub_label': text, 'id_part': part, 'id_part_source': source,
                'id_family': family, 'shelf_band': entry['band'],
                'bottom_shelf': bottom is not None and entry['band'] == bottom,
                'aspect_ratio': round(width / height, 3)})
            kept_by[entry['keep']] += 1
            kept.append({**ann, 'id': len(kept) + 1, 'attributes': attributes})

    coco['annotations'] = kept
    coco['info'] = {**coco.get('info', {}),
                    'description': 'Model-generated front-face labels, cleaned (faces_v2); '
                                   'not ground truth',
                    'label_version': LABEL_VERSION, 'target': FACE_TARGET_SLUG,
                    'target_description': FACE_DESCRIPTION,
                    'cleaned_from': FACES_V1_LABELS.name,
                    'clean_rules': {key: list(value) if isinstance(value, tuple) else value
                                    for key, value in rules.items()},
                    'clean_kept': dict(kept_by), 'clean_dropped': dict(tally)}
    return coco, drops


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
                    options = detect_options(run, prompt, base_url)

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


# ------------------------------------------------------------------ v2: reporting


def _per_image_counts(labels):
    counts: dict[str, int] = {}
    files = {image['id']: image['file_name'] for image in labels['images']}
    for ann in labels['annotations']:
        name = files.get(ann['image_id'])
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _per_image_iou(labels):
    ious, files = {}, {image['id']: image['file_name'] for image in labels['images']}
    boxes: dict[str, list] = {}
    for ann in labels['annotations']:
        name = files.get(ann['image_id'])
        if name is not None:
            boxes.setdefault(name, []).append(_xyxy(ann))
    for name, image_boxes in boxes.items():
        value = neighbour_iou_stats(image_boxes)
        if value is not None:
            ious[name] = value
    return ious


def select_overlay_images(labels, drops, splits, *, per_split=2):
    """Deterministic overlay picks: the *per_split* most-cleaned train and val images.

    Test images are never rendered or inspected here, so nothing about the cleaning can be
    tuned on them (ARCHITECTURE.md).  Ordering is by dropped faces, then file name.
    """
    splits = _as_splits(splits)
    dropped: dict[str, int] = Counter(drop['file_name'] for drop in drops)
    picks = []
    for split in ('train', 'val'):
        candidates = sorted((image['file_name'] for image in _as_labels(labels)['images']
                             if splits.get(image['file_name']) == split),
                            key=lambda name: (-dropped.get(name, 0), name))
        picks.extend(candidates[:per_split])
    return picks


def _as_splits(splits):
    """Split map (image id -> split) from a dict, a splits.json path, or None."""
    if splits is None:
        return {}
    loaded = _as_labels(splits)
    return loaded.get('images', loaded) if isinstance(loaded, dict) else {}


def render_clean_overlays(labels, drops, image_ids, out_dir, *, images_root=None,
                          reference=None, max_width=1400):
    """Draw kept faces (blue, numbered) against the rejected candidates (red) per image.

    Returns the paths written (one PNG + one JSON sidecar per image); the sidecar is the
    worker's evidence trail, so the overlay can be re-read without re-running the cleaning.
    """
    coco = _as_labels(labels)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(images_root or data.IMAGES_ROOT)
    files = {image['id']: image['file_name'] for image in coco['images']}
    kept: dict[str, list] = {}
    for ann in coco['annotations']:
        name = files.get(ann['image_id'])
        if name is not None:
            kept.setdefault(name, []).append(ann)
    rejected: dict[str, list] = {}
    for drop in drops:
        rejected.setdefault(drop['file_name'], []).append(drop)
    paths = []
    for image_id in image_ids:
        path = root / image_id
        if not path.is_file():
            continue
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        anns = sorted(kept.get(image_id, []), key=lambda a: (a['bbox'][1], a['bbox'][0]))
        boxes = [_xyxy(a) for a in anns]
        sub_labels = [str(a['attributes'].get('sub_label') or '') for a in anns]
        parts = {a['attributes'].get('id_part') for a in anns if a['attributes'].get('id_part')}
        bad = rejected.get(image_id, [])
        reasons = dict(Counter(d['reason'] for d in bad).most_common()) or 'none'
        title = (f'{image_id} [faces_v2] {len(boxes)} kept / {len(bad)} dropped {reasons}')
        name = overlay_name(image_id, 'faces-v2')
        paths.append(render_overlay(image, boxes, out_dir / f'{name}.png', title=title,
                                   sub_labels=sub_labels, max_width=max_width,
                                   whole_boxes=(reference or {}).get(image_id, {}).get('boxes'),
                                   drop_boxes=[[d['bbox'][0], d['bbox'][1],
                                                d['bbox'][0] + d['bbox'][2],
                                                d['bbox'][1] + d['bbox'][3]]
                                               for d in rejected.get(image_id, [])]))
        sidecar = {'image_id': image_id, 'parts_readable': sorted(parts),
                   'kept': [{'bbox': a['bbox'], **{k: a['attributes'].get(k) for k in
                                                   ('sub_label', 'id_part', 'id_family',
                                                    'support', 'judge_verdict', 'clean_keep')}}
                            for a in anns],
                   'dropped': rejected.get(image_id, [])}
        paths.append(out_dir / f'{name}.json')
        (out_dir / f'{name}.json').write_text(json.dumps(sidecar, indent=2) + '\n')
    return paths


def summarize_clean(v1, v2, drops, *, reference=None, splits=None, overlays=(),
                    images_read=None, judgement=None):
    """Markdown report for faces_v2: rule tallies and per-image counts vs v1 and vs whole boxes."""
    v1, v2 = _as_labels(v1), _as_labels(v2)
    splits = _as_splits(splits)
    reference = reference if reference is not None else load_reference()
    names = {c['id']: c['name'] for c in v2['categories']}
    v1_counts, v2_counts = _per_image_counts(v1), _per_image_counts(v2)
    v1_ious, v2_ious = _per_image_iou(v1), _per_image_iou(v2)
    reason_tally = Counter(drop['reason'] for drop in drops)
    per_image_drops = Counter(drop['file_name'] for drop in drops)
    attributes = [a.get('attributes', {}) for a in v2['annotations']]
    readable = Counter(a.get('sub_label') or '-' for a in attributes)
    parts = Counter(a.get('id_part_source') or 'none' for a in attributes)
    part_values = Counter(a.get('id_part') for a in attributes if a.get('id_part') is not None)
    bottom = sum(1 for a in attributes if a.get('bottom_shelf'))
    rules = v2['info'].get('clean_rules', {})
    whole_ious = [value for value in (neighbour_iou_stats(entry['boxes'])
                                      for entry in reference.values()) if value is not None]
    v2_boxes: dict[str, list] = {}
    v2_files = {image['id']: image['file_name'] for image in v2['images']}
    for ann in v2['annotations']:
        name = v2_files.get(ann['image_id'])
        if name is not None:
            v2_boxes.setdefault(name, []).append(_xyxy(ann))
    residual = {name: residual_duplicates(boxes) for name, boxes in v2_boxes.items()}
    residual_pairs = sum(len(pairs) for pairs in residual.values())

    dropped_rules = json.dumps(dict(reason_tally.most_common()), indent=None) or 'null'
    lines = [f'# {LABEL_VERSION} — cleaned front-face labels (experiment family G; '
             f'NOT ground truth)', '',
             f'- target `{FACE_TARGET_SLUG}` (the cartridge front face), cleaned from '
             f'`{v2["info"].get("cleaned_from", "faces_v1.json")}` (prompt variant '
             f'`{v1["info"].get("prompt_variant", "?")}`): {len(v1["annotations"])} proposals -> '
             f'{len(v2["annotations"])} faces over {len(v2["images"])} images',
             f'- kept by rule: {json.dumps(v2["info"].get("clean_kept", {}))}',
             f'- dropped by rule: {dropped_rules} '
             f'({sum(reason_tally.values())} of {len(v1["annotations"])})',
             f'- rules: support agreed >= {rules.get("support_agreed", CLEAN_RULES["support_agreed"])}, '
             f'single run kept only when judged correct at support >= '
             f'{rules.get("support_judged", CLEAN_RULES["support_judged"])}, '
             f'bottom-shelf aspect <= {rules.get("max_aspect_bottom", CLEAN_RULES["max_aspect_bottom"])}, '
             f'width >= {rules.get("min_width_frac", CLEAN_RULES["min_width_frac"])} x row median, '
             f'one face per leaning column (x-overlap > '
             f'{rules.get("column_overlap", CLEAN_RULES["column_overlap"])}, y-gap >= '
             f'{rules.get("column_min_y_gap", CLEAN_RULES["column_min_y_gap"])} x face height)',
             f'- neighbour overlap (mean per-image max-IoU): {LABEL_VERSION} '
             f'{_fmt(_mean(list(v2_ious.values())))} vs faces_v1 '
             f'{_fmt(_mean(list(v1_ious.values())))} vs whole cartridges {_fmt(_mean(whole_ious))} (baseline)',
             f'- bottom-shelf faces kept: {bottom}; printed codes readable on kept faces: '
             f'{sum(1 for a in attributes if a.get("sub_label"))} '
             f'({json.dumps(dict(readable.most_common(8)))})',
             f'- id_part filled from face text {parts.get("face_text", 0)}x, from the containing '
             f'whole-cartridge box {parts.get("whole_box", 0)}x, else None; '
             f'part values {json.dumps({str(k): v for k, v in part_values.items()})}',
             f'- residual duplicates NOT covered by the rules above: {residual_pairs} pairs '
             f'(x-overlap >= 0.8 *and* y-containment >= 0.75, i.e. a loose and a tight box of '
             f'the same end) in {sum(1 for pairs in residual.values() if pairs)} image(s) — '
             f'flagged for the human review pass, never auto-deleted, because the same geometry '
             f'also describes a real back-row face behind a front-row one',
             f'- mean faces per image: {LABEL_VERSION} '
             f'{len(v2["annotations"]) / max(1, len(v2["images"])):.1f} vs faces_v1 '
             f'{len(v1["annotations"]) / max(1, len(v1["images"])):.1f} vs whole cartridges '
             f'{sum(len(r["boxes"]) for r in reference.values()) / max(1, len(reference)):.1f}']
    if v2['info'].get('judge_fallbacks'):
        lines.append(f'- judge fallbacks inherited from v1: {len(v2["info"]["judge_fallbacks"])} image(s)')
    lines += ['', '## Per image: faces_v1 -> faces_v2 vs whole-cartridge boxes', '',
              '| image | split | v1 | v2 | dropped | max-IoU v1 | max-IoU v2 | whole | dup |',
              '|---|---|---|---|---|---|---|---|---|']
    total_v1 = total_v2 = total_whole = 0
    for name in sorted(set(v1_counts) | set(v2_counts) | set(reference)):
        whole = len(reference.get(name, {}).get('boxes', []))
        v1_iou, v2_iou = v1_ious.get(name), v2_ious.get(name)
        total_v1 += v1_counts.get(name, 0)
        total_v2 += v2_counts.get(name, 0)
        total_whole += whole
        lines.append(f'| `{name}` | {splits.get(name, "?")} | {v1_counts.get(name, 0)} | '
                     f'{v2_counts.get(name, 0)} | {per_image_drops.get(name, 0)} | '
                     + ('—' if v1_iou is None else f'{v1_iou:.3f}') + ' | '
                     + ('—' if v2_iou is None else f'{v2_iou:.3f}')
                     + f' | {whole} | {len(residual.get(name, []))} |')
    lines.append(f'| **total** |  | {total_v1} | {total_v2} | {sum(per_image_drops.values())} | '
                 f' |  | {total_whole} | {residual_pairs} |')
    total = sum(1 for pairs in residual.values() if pairs)
    lines += ['', f'- `dup` = surviving pairs of one end boxed twice (x-overlap >= 0.8, '
              f'y-containment >= 0.75) in {total} of {len(residual)} images; they need the '
              f'human review pass, not another geometric rule', '']
    lines += ['', f'- identified classes on kept faces: '
              f'{json.dumps(class_counts(v2["annotations"], names))}', '']
    if images_read is not None:
        lines.append(f'- images read into the worker context (downscaled): {images_read}')
    if overlays:
        lines.append('- overlays reviewed by the worker (2 train + 2 val, never test): '
                     + ', '.join(f'`{o}`' for o in overlays))
    lines += ['', '## Honest visual judgement (worker looked at these overlays)', '']
    lines.append(Path(judgement).read_text().rstrip() if judgement
                 else '(filled in by the worker after looking at overlays)')
    lines.append('')
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


def _clean(args):
    """Write the cleaned labels, the dropped candidates, overlays, and the v2 report."""
    v1 = json.loads(Path(args.labels).read_text())
    config = yaml.safe_load(args.config.read_text()) if args.config.is_file() else {}
    splits = json.loads(Path(args.splits).read_text())['images']
    reference = load_reference(args.reference)
    sub_labels = None if args.no_cache else recover_sub_labels(v1, config)
    coco, drops = clean_faces(v1, reference=reference, sub_labels=sub_labels)
    out = Path(args.out)
    if out.resolve().is_relative_to((ROOT / 'data/gt').resolve()):
        raise ValueError('cleaned labels must not be written under data/gt/')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, indent=2) + '\n')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'dropped.json').write_text(json.dumps(drops, indent=2) + '\n')
    image_ids = args.overlay_image or select_overlay_images(coco, drops, splits,
                                                           per_split=max(0, args.overlays))
    paths = render_clean_overlays(coco, drops, image_ids, out_dir / 'viz',
                                 reference=reference) if image_ids else []
    summary = args.summary or out_dir / 'summary.md'
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(summarize_clean(
        v1, coco, drops, reference=reference, splits=splits,
        overlays=[p.name for p in paths if p.suffix == '.png'],
        images_read=args.images_read, judgement=args.judgement))
    print(f'wrote {out}: {len(v1["annotations"])} faces -> {len(coco["annotations"])} kept, '
          f'{len(drops)} dropped {dict(Counter(d["reason"] for d in drops).most_common())}')
    print(f'wrote {summary} (+ {len(paths) // 2 if paths else 0} overlay pairs in '
          f'{out_dir / "viz"})')
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

    clean = sub.add_parser('clean', help='faces_v1 -> faces_v2: drop the noise, keep one face '
                                         'per leaning column (no new VLM calls)')
    clean.add_argument('--labels', type=Path, default=FACES_V1_LABELS)
    clean.add_argument('--config', type=Path, default=ROOT / 'configs/bootstrap/faces.yaml',
                       help='the faces bootstrap config, to recover printed codes from the cache')
    clean.add_argument('--reference', type=Path, default=REFERENCE_LABELS,
                       help='whole-cartridge labels: identity/family/Part fallback by containment')
    clean.add_argument('--splits', type=Path, default=data.SPLITS_PATH)
    clean.add_argument('--out', type=Path, default=FACES_V2_LABELS)
    clean.add_argument('--out-dir', type=Path, default=ROOT / 'runs/faces_v2')
    clean.add_argument('--summary', type=Path, default=None)
    clean.add_argument('--overlays', type=int, default=2,
                       help='images per split (train and val) to render overlays for; '
                            '0 disables (default 2 + 2 = 4 images)')
    clean.add_argument('--overlay-image', action='append', default=[],
                       help='render this image id instead of the automatic picks')
    clean.add_argument('--no-cache', action='store_true',
                       help='skip reading the detection cache (leaves sub_label empty)')
    clean.add_argument('--judgement', type=Path, default=None)
    clean.add_argument('--images-read', type=int, default=None)

    args = parser.parse_args(argv)
    if args.cmd == 'tune':
        return _tune(args)
    if args.cmd == 'clean':
        return _clean(args)
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
