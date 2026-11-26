"""Re-identify bootstrap proposals with the two-stage VLM identifier.

Reads a v1 proposal COCO (``bootstrap_v1.json``), recomputes every annotation's
identity with ``freestyle.vlm_identify.VLMCropIdentifier`` (``mode="two_stage"``),
and writes a fresh COCO with the same boxes and attributes. The ``category_id``
is replaced and identity attributes are refreshed/added (``id_family``,
``id_part``, ``id_confidence``, ``id_model``, ``id_mode``). Everything else —
boxes, provenance, judge verdicts, source — is kept unchanged. Model-generated
labels are never ground truth (see ``docs/freestyle/GT_PROTOCOL.md``).

Every VLM call is cached under ``--cache-dir`` (keyed on the encoded image,
prompt, model and pipeline settings), so re-running the CLI is cheap and
resumable. Calls run through a 2-worker pool with the shared 429/5xx/timeout
backoff (``freestyle.bootstrap_labels._retry``).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw

from freestyle import collective
from freestyle.geometry import crop
from freestyle.vlm_identify import VLMCropIdentifier
from freestyle.bootstrap_labels import _cache, _retry, _hash

ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = ROOT / 'data/raw/freestyle-vision-data/images'

DEFAULT_MODEL = 'david-birks/inkling-2-mini'
DEFAULT_MODE = 'two_stage'
DEFAULT_UPSCALE = 2.0
DEFAULT_PAD = 0.08
DEFAULT_PART_REGION = (0.0, 0.5, 1.0, 1.0)
MAX_SIDE = 1600  # viz downscale cap


class Stats:
    """Thread-safe counters for one run."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls_family = 0
        self.calls_part = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start = time.monotonic()

    def _bump(self, kind):
        if kind == 'part':
            self.calls_part += 1
        else:
            self.calls_family += 1

    def hit(self, kind):
        with self._lock:
            self.cache_hits += 1
            self._bump(kind)

    def miss(self, kind):
        with self._lock:
            self.cache_misses += 1
            self._bump(kind)

    def elapsed(self):
        return time.monotonic() - self.start


def _nc(value):
    return value not in (None, 'unknown', 'unknown-part')


def _parse_region(text):
    try:
        values = tuple(float(part) for part in str(text).split(','))
    except ValueError:
        raise SystemExit('--part-region must be "left,top,right,bottom"')
    if len(values) != 4 or not all(0.0 <= v <= 1.0 for v in values):
        raise SystemExit('--part-region must be four normalized bounds 0..1')
    return values


def _encoded_hash(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _cached_ask(cache_dir, model, stats, settings):
    """Build the ``_ask(image, prompt)`` used by the identifier.

    Wraps ``collective.chat_image`` with the shared cache helper + patient
    retry so every VLM call is cached under ``cache_dir`` and resumable.
    """
    def ask(image, prompt):
        kind = 'part' if 'part numeral' in prompt else 'family'
        key = ['reidentify-v1', _encoded_hash(image), prompt, model, settings]
        cache_path = Path(cache_dir) / (_hash(key) + '.json')
        if cache_path.exists():
            stats.hit(kind)
            return json.loads(cache_path.read_text())['text']

        def request():
            return collective.chat_image(model, image, prompt, max_tokens=4096)

        result = _cache(cache_dir, key, lambda: {'text': _retry(request)})
        stats.miss(kind)
        return result['text']

    return ask


def _overlay(image, annotations, cats, out_path, unknown_id):
    """Draw per-annotation boxes coloured by v2 verdict (green = known, red = unknown)."""
    im = image.convert('RGB')
    if max(im.size) > MAX_SIDE:
        s = MAX_SIDE / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
        scale = s
    else:
        scale = 1.0
    draw = ImageDraw.Draw(im)
    for ann in annotations:
        x, y, w, h = ann['bbox']
        x0, y0, x1, y1 = x * scale, y * scale, (x + w) * scale, (y + h) * scale
        is_known = ann['category_id'] != unknown_id
        color = '#2e7d32' if is_known else '#c62828'
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        name = cats.get(ann['category_id'], '?')
        label = f'{name} {ann["attributes"].get("id_confidence", 0.0):.2f}'
        bbox = draw.textbbox((0, 0), label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = y0 + 2 if y0 + th + 6 < y1 else y1 - th - 6
        draw.rectangle([x0, ty, x0 + tw + 6, ty + th + 4], fill='white')
        draw.text((x0 + 3, ty + 2), label, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=88)


def _grid(crops, results, out_path, max_side=MAX_SIDE):
    """Grid of crops sorted by predicted class, with v2 identity labels."""
    entries = [(im, res) for im, res in zip(crops, results) if im is not None]
    if not entries:
        return
    entries.sort(key=lambda t: (t[1].get('class', '?'), t[1].get('family') or ''))
    cell = 260
    cols = min(5, len(entries))
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell, rows * (cell + 22)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (im, res) in enumerate(entries):
        im = im.convert('RGB')
        if im.size[0] <= 0 or im.size[1] <= 0:
            continue
        s = min(cell / im.size[0], cell / im.size[1])
        im2 = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
        x = (i % cols) * cell
        y = (i // cols) * (cell + 22)
        label = f'{i}: {res.get("class", "?")} ({res.get("part")})'
        draw.text((x + 2, y + 2), label, fill='black')
        sheet.paste(im2, (x + (cell - im2.size[0]) // 2, y + 18 + (cell - im2.size[1]) // 2))
    if max(sheet.size) > max_side:
        s = max_side / max(sheet.size)
        sheet = sheet.resize((round(sheet.size[0] * s), round(sheet.size[1] * s)), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)


def _match_image(file_name, wanted):
    return any(file_name == w or file_name.endswith('/' + w) for w in wanted)


def _shelf_hints(annotations, min_gap=200.0):
    """Per-annotation shelf hint ('top' | 'bottom' | None) from box geometry.

    The two cabinet shelves are separated by a clear vertical gap in box
    centroids; the split is derived from the v1 BOXES only (positions, not
    identities). If no gap is large enough the boxes are ambiguous and every
    hint is None (the identifier is then shelf-agnostic).
    """
    n = len(annotations)
    if n < 2:
        return [None] * n
    centroids = [(a['bbox'][1] + a['bbox'][3] / 2.0, i) for i, a in enumerate(annotations)]
    centroids.sort()
    gaps = [(centroids[k + 1][0] - centroids[k][0], k) for k in range(n - 1)]
    biggest_gap, idx = max(gaps, key=lambda t: t[0])
    if biggest_gap < min_gap:
        return [None] * n
    line = (centroids[idx][0] + centroids[idx + 1][0]) / 2.0
    return ['top' if (a['bbox'][1] + a['bbox'][3] / 2.0) < line else 'bottom'
            for a in annotations]


def run(args):
    label_path = Path(args.in_path)
    label = json.loads(label_path.read_text())
    if Path(args.out).resolve().is_relative_to((ROOT / 'data/gt').resolve()):
        raise SystemExit('--out must not be under data/gt/')

    settings = {'mode': args.mode, 'upscale': args.upscale, 'pad': args.pad,
                'part_region': list(args.part_region), 'model': args.model}
    stats = Stats()
    cache_dir = Path(args.cache_dir)

    identifier = VLMCropIdentifier(args.model, catalog=None, upscale=args.upscale,
                                   mode=args.mode, part_region=args.part_region)
    identifier._ask = _cached_ask(cache_dir, args.model, stats, settings)

    cats_inv = {c['id']: c['name'] for c in label['categories']}
    ids_by_name = {c['name']: c['id'] for c in label['categories']}
    unknown_id = ids_by_name.get('unknown')

    wanted = args.images if args.images else None
    if wanted:
        unmatched = {w for w in wanted
                     if not any(_match_image(img['file_name'], [w]) for img in label['images'])}
        if unmatched:
            raise SystemExit(f'--images entries not present in {label_path}: {sorted(unmatched)}')
        all_images = [img for img in label['images'] if _match_image(img['file_name'], wanted)]
    else:
        all_images = list(label['images'])

    out_images = []
    out_annotations = []
    per_image = []
    for image_info in all_images:
        path = IMAGES_ROOT / image_info['file_name']
        image = Image.open(path).convert('RGB')
        annotations = [a for a in label['annotations'] if a['image_id'] == image_info['id']]
        shelf_hints = _shelf_hints(annotations)

        jobs = list(zip(annotations, shelf_hints))

        def one(job):
            ann, shelf = job
            x, y, w, h = ann['bbox']
            box = [x, y, x + w, y + h]
            try:
                im = crop(image, box, pad=args.pad)
            except ValueError:
                return {'crop': None, 'result': {'class': 'unknown', 'family': 'unknown',
                                                  'part': None, 'confidence': 0.0,
                                                  'error': 'crop-invalid'}}
            try:
                result = identifier.identify(im, shelf=shelf)
            except Exception as exc:
                result = {'class': 'unknown', 'family': 'unknown', 'part': None,
                          'confidence': 0.0, 'error': str(exc)[:200]}
            return {'crop': im, 'result': result, 'shelf': shelf}

        with ThreadPoolExecutor(max_workers=2) as pool:
            identified = list(pool.map(one, jobs))

        image_anns = []
        for ann, ident in zip(annotations, identified):
            result = ident['result']
            name = result.get('class', 'unknown')
            if name not in ids_by_name:
                name = 'unknown'
            new_attrs = dict(ann.get('attributes', {}))
            new_attrs['id_family'] = result.get('family') if _nc(result.get('family')) else None
            new_attrs['id_part'] = result.get('part') if result.get('part') in (1, 2) else None
            new_attrs['id_confidence'] = float(result.get('confidence', 0.0))
            new_attrs['id_model'] = args.model
            new_attrs['id_mode'] = args.mode
            if ident.get('shelf'):
                new_attrs['id_shelf_hint'] = ident['shelf']
            new_attrs['identity_visible'] = name != 'unknown'
            if result.get('error'):
                new_attrs['id_error'] = result['error']
            new_ann = dict(ann)
            new_ann['category_id'] = ids_by_name[name]
            new_ann['attributes'] = new_attrs
            image_anns.append(new_ann)
            out_annotations.append(new_ann)

        v1_unknown = sum(1 for a in annotations if a['category_id'] == unknown_id)
        v2_unknown = sum(1 for a in image_anns if a['category_id'] == unknown_id)
        per_image.append({'image_id': image_info['id'], 'file_name': image_info['file_name'],
                          'n_annotations': len(annotations), 'v1_unknown': v1_unknown,
                          'v2_unknown': v2_unknown, 'v2_known': len(annotations) - v2_unknown})
        out_images.append(image_info)

        if not args.no_viz:
            crops = [ident['crop'] for ident in identified]
            results = [ident['result'] for ident in identified]
            _overlay(image, image_anns, cats_inv,
                     Path(args.viz_dir) / (image_info['file_name'].replace('/', '__') + '_overlay.jpg'),
                     unknown_id)
            _grid(crops, results,
                  Path(args.viz_dir) / (image_info['file_name'].replace('/', '__') + '_grid.jpg'))

        print(f'[{image_info["file_name"]}] {len(annotations)} boxes -> '
              f'{v2_known_count(image_anns, unknown_id)} known / {v2_unknown} unknown', flush=True)

    result = {'info': dict(label.get('info', {})),
              'images': out_images,
              'annotations': out_annotations,
              'categories': label['categories']}
    result['info']['reidentification'] = {
        'source': str(label_path),
        'model': args.model,
        'mode': args.mode,
        'upscale': args.upscale,
        'pad': args.pad,
        'part_region': list(args.part_region),
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'shelf_inference': 'box-centroid gap (min_gap=200px); None when no two-shelf split is clear',
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = tempfile.NamedTemporaryFile('w', dir=out_path.parent, delete=False, suffix='.tmp')
    try:
        json.dump(result, tmp_out, indent=2)
        tmp_out.write('\n')
        tmp_out.close()
        os.replace(tmp_out.name, out_path)
    except BaseException:
        Path(tmp_out.name).unlink(missing_ok=True)
        raise

    if not args.no_summary:
        write_summary(args.viz_dir, result, per_image, stats, Path(args.out))

    print(f'\nwrote {out_path}: {len(out_images)} images, {len(out_annotations)} annotations '
          f'({stats.calls_family + stats.calls_part} VLM calls, {stats.cache_hits} cache hits, '
          f'{stats.elapsed():.1f}s)')
    return 0


def v2_known_count(image_anns, unknown_id):
    return sum(1 for a in image_anns if a['category_id'] != unknown_id)


def write_summary(viz_dir, result, per_image, stats, out_path):
    all_anns = result['annotations']
    families = {}
    parts = {}
    part_capable_families = {'coca cola', 'diet coke', 'dr pepper', 'coca cola zero'}
    for ann in all_anns:
        fam = ann['attributes'].get('id_family') or 'unknown'
        families[fam] = families.get(fam, 0) + 1
        part = ann['attributes'].get('id_part')
        key = part if part in (1, 2) else None
        parts[key] = parts.get(key, 0) + 1
    n_part_capable = sum(1 for ann in all_anns
                         if ann['attributes'].get('id_family') in part_capable_families)
    n_part_readable = sum(1 for ann in all_anns
                          if ann['attributes'].get('id_family') in part_capable_families
                          and ann['attributes'].get('id_part') in (1, 2))

    lines = ['# bootstrap_v2 summary', '',
             '- Source: `data/labels/bootstrap_v1.json`',
             f'- Output: `{out_path}`',
             f'- Model: `{result["info"]["reidentification"]["model"]}`, mode: `{result["info"]["reidentification"]["mode"]}`',
             f'- Settings: upscale={result["info"]["reidentification"]["upscale"]}, pad={result["info"]["reidentification"]["pad"]}, part_region={result["info"]["reidentification"]["part_region"]}',
             f'- Wall time: {stats.elapsed():.1f}s', f'- Annotations: {len(all_anns)} across {len(per_image)} images', '']
    lines.append('## VLM calls')
    total_calls = stats.calls_family + stats.calls_part
    hit_rate = stats.cache_hits / (stats.cache_hits + stats.cache_misses) if (stats.cache_hits + stats.cache_misses) else 0.0
    lines += [f'- family stage: {stats.calls_family}', f'- part stage: {stats.calls_part}',
              f'- total calls: {total_calls}',
              f'- cache hits: {stats.cache_hits} ({100 * hit_rate:.1f}%)',
              f'- cache misses: {stats.cache_misses}', '']
    lines.append('## Per-image unknown rate')
    lines.append('| image | annotations | v1 unknown | v2 unknown | v2 known |')
    lines.append('| --- | --- | --- | --- | --- |')
    for row in per_image:
        lines.append(f'| {row["file_name"]} | {row["n_annotations"]} | '
                     f'{row["v1_unknown"]} ({_pct(row["v1_unknown"], row["n_annotations"])}) | '
                     f'{row["v2_unknown"]} ({_pct(row["v2_unknown"], row["n_annotations"])}) | {row["v2_known"]} |')
    tot_v1 = sum(r['v1_unknown'] for r in per_image)
    tot_v2 = sum(r['v2_unknown'] for r in per_image)
    tot_n = sum(r['n_annotations'] for r in per_image)
    lines.append(f'| **total** | **{tot_n}** | **{tot_v1}** | **{tot_v2}** | **{tot_n - tot_v2}** |')
    lines.append('')
    lines.append('## Family distribution (v2)')
    lines.append('| family | count |')
    lines.append('| --- | --- |')
    for fam, count in sorted(families.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f'| {fam} | {count} |')
    lines.append('')
    lines.append('## Part readability')
    lines += [f'- Part-capable families: {n_part_capable} annotations',
              f'- Part readable: {n_part_readable} ({_pct(n_part_readable, n_part_capable)} of part-capable)',
              f'- Part 1: {parts.get(1, 0)}',
              f'- Part 2: {parts.get(2, 0)}',
              '']
    lines.append('Note: model-generated labels, never GT.')
    summary = Path(viz_dir).parent / 'summary.md'
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text('\n'.join(lines) + '\n')
    print(f'wrote {summary}')


def _pct(n, total):
    return f'{100 * n / total:.1f}%' if total else '-'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--in', dest='in_path', required=True, type=Path,
                        help='input v1 proposal COCO (never data/gt)')
    parser.add_argument('--out', required=True, type=Path, help='output COCO (never data/gt)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'model id (default {DEFAULT_MODEL})')
    parser.add_argument('--mode', default=DEFAULT_MODE, choices=['two_stage', 'existing'])
    parser.add_argument('--upscale', type=float, default=DEFAULT_UPSCALE,
                        help='family-stage upscale factor (default 2.0)')
    parser.add_argument('--pad', type=float, default=DEFAULT_PAD,
                        help='crop padding fraction (default 0.08)')
    parser.add_argument('--part-region', type=_parse_region, default=DEFAULT_PART_REGION,
                        metavar='L,T,R,B',
                        help='normalized part-region window (default 0.0,0.5,1.0,1.0)')
    parser.add_argument('--images', nargs='*', metavar='IMAGE_ID',
                        help='subset of image file_names to process (default: all)')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'runs/bootstrap_v2/cache')
    parser.add_argument('--viz-dir', type=Path, default=ROOT / 'runs/bootstrap_v2/viz')
    parser.add_argument('--no-viz', action='store_true', help='skip overlay/grid rendering')
    parser.add_argument('--no-summary', action='store_true', help='skip summary.md')
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
