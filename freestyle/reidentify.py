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

``--profile`` decides *how a box becomes a crop*, and has to match whatever
produced the replies you want to reuse:

``bootstrap`` (default)
    ``freestyle.geometry.crop`` at ``--pad`` (8 %) and the shelf hint derived
    from the box-centroid gap (``_shelf_hints``). This is what
    ``data/labels/bootstrap_v2.json`` was made with; replies land in the
    ``reidentify-v1`` cache layout.
``reader_compare``
    The crop and shelf logic of ``freestyle/experiments/reader_compare.py``
    verbatim: the phone quiz's own ``make_crop`` (15 % padding, quiz JPEG) and
    ``freestyle.quiz.build_queue.shelf_labels`` (the repo's two-band shelf fit,
    position split as fallback). Transport is that experiment's
    ``CallCache`` — the same content-addressed keys, ``max_tokens=8192`` for
    thinking models, <=2 in-flight requests, patient 429 retries and a
    calls.jsonl log — so pointing ``--cache-dir`` at ``runs/reader_compare/cache``
    reuses every reply whose crop is identical instead of re-asking.

``--mode two_stage_shelf`` runs the two-stage identifier with a shelf hint in
the prompt for every box (the reader the reader comparison picked: Qwen 3.8
Flash Next, 0.56 top-1 vs Inkling's 0.49 on human test answers, see
``docs/freestyle/READER_COMPARE.md``). ``--only-split`` restricts the pass to
the splits you actually need (e.g. ``val`` + ``test``).
"""
from __future__ import annotations

import argparse
import contextlib
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
from freestyle.experiments.reader_compare import CallCache
from freestyle.geometry import crop
from freestyle.quiz.build_queue import shelf_labels
from freestyle.quiz.server import CROP_PAD, make_crop
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

#: Crop + shelf + cache recipe (see the module docstring).
PROFILES = ('bootstrap', 'reader_compare')
DEFAULT_PROFILE = 'bootstrap'
#: ``--mode`` values. ``two_stage_shelf`` is ``two_stage`` + a per-box shelf hint.
MODES = ('two_stage', 'two_stage_shelf', 'existing')
#: Default reply caches, per profile. ``reader_compare`` shares the experiment's
#: store so identical crops are served from disk instead of re-asked.
DEFAULT_CACHE_DIRS = {'bootstrap': ROOT / 'runs/bootstrap_v2/cache',
                      'reader_compare': ROOT / 'runs/reader_compare/cache'}
#: max_tokens per profile: flash-next is a thinking model and gets truncated
#: below ~8k, but the bootstrap cache keys stay at the 4096 they were built with.
DEFAULT_MAX_TOKENS = {'bootstrap': 4096, 'reader_compare': 8192}
#: Project rule: the Collective is often at capacity, so keep in-flight low.
MAX_INFLIGHT = 2


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

    def attempt(self, kind):
        """Count one family/part stage request, served from cache or from the network."""
        with self._lock:
            self._bump(kind)

    def set_cache(self, hits, misses):
        with self._lock:
            self.cache_hits, self.cache_misses = int(hits), int(misses)

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


def _xyxy(bbox):
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w, y + h]


def _crop_fn(profile, pad):
    """Box -> crop image, per profile (see the module docstring)."""
    if profile == 'reader_compare':
        def quiz_crop(image, bbox):
            data = make_crop(image, _xyxy(bbox), pad=pad)
            return Image.open(io.BytesIO(data)).convert('RGB')
        return quiz_crop

    def bootstrap_crop(image, bbox):
        return crop(image, _xyxy(bbox), pad=pad)
    return bootstrap_crop


def _shelf_fn(profile):
    """Annotations -> per-box shelf hint ('top' | 'bottom' | None), per profile."""
    if profile == 'reader_compare':
        def quiz_shelves(annotations, height):
            return shelf_labels([_xyxy(a['bbox']) for a in annotations], int(height or 0))
        return quiz_shelves
    return lambda annotations, height: _shelf_hints(annotations)


def _uses_shelf(mode, profile):
    """Whether this mode+profile pair puts a shelf hint in the prompt.

    ``bootstrap`` always hints when a two-shelf split is clear (that is what
    ``bootstrap_v2.json`` was built with, and ``existing`` ignores it anyway);
    ``reader_compare`` only hints in ``two_stage_shelf``, exactly like the
    experiment whose replies we reuse.
    """
    if not mode.startswith('two_stage'):
        return False
    return True if profile == 'bootstrap' else mode == 'two_stage_shelf'


@contextlib.contextmanager
def _noop_tag(tag):
    """Tag context for the legacy transport (nothing to attribute calls to)."""
    yield


def _cache_tag(cache):
    """Tag factory that attributes every call inside a box to that box's crop id."""
    def tag(name):
        return cache.tag(name, [name])
    return tag


def _profile_ask(cache, model, stats):
    """``_ask(image, prompt)`` on top of a :class:`CallCache` (patient, capped, logged).

    ``CallCache.chat`` has the same positional shape as ``collective.chat_image``, so it
    drops straight into the identifier; the wrapper only adds the per-stage (family vs
    part) counters the summary reports.
    """
    def ask(image, prompt):
        stats.attempt('part' if 'part numeral' in prompt else 'family')
        return cache.chat(model, image, prompt)
    return ask


def _call_log_stats(rows):
    """Summarize the calls this process appended to a ``CallCache`` log."""
    hits = sum(1 for row in rows if row.get('cache') == 'hit')
    misses = sum(1 for row in rows if row.get('cache') == 'miss')
    return {'calls': len(rows), 'cache_hits': hits, 'cache_misses': misses,
            'retried': sum(1 for row in rows if (row.get('attempts') or 0) > 1),
            'failed': sum(1 for row in rows if not row.get('ok')),
            'network_s': round(sum(float(row.get('latency_s') or 0.0) for row in rows), 1),
            'served_models': sorted({str(row.get('model')) for row in rows if row.get('model')})}


def _split_wanted(splits, file_name, only):
    if not only:
        return True
    return splits.get('images', {}).get(file_name) in only


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


def _resolve(args):
    """Normalize the CLI/parsed args into the recipe this run actually uses."""
    profile = getattr(args, 'profile', DEFAULT_PROFILE) or DEFAULT_PROFILE
    if profile not in PROFILES:
        raise SystemExit(f'--profile must be one of {PROFILES}')
    if args.mode not in MODES:
        raise SystemExit(f'--mode must be one of {MODES}')
    workers = int(getattr(args, 'max_workers', MAX_INFLIGHT) or MAX_INFLIGHT)
    if workers < 1:
        raise SystemExit('--max-workers must be >= 1')
    if workers > MAX_INFLIGHT:
        raise SystemExit(f'--max-workers must be <= {MAX_INFLIGHT} '
                         '(project rule: at most 2 in-flight requests per process)')
    pad = args.pad
    if pad is None:
        pad = CROP_PAD if profile == 'reader_compare' else DEFAULT_PAD
    max_tokens = int(getattr(args, 'max_tokens', None) or DEFAULT_MAX_TOKENS[profile])
    cache_dir = Path(getattr(args, 'cache_dir', None) or DEFAULT_CACHE_DIRS[profile])
    viz_dir = Path(getattr(args, 'viz_dir', None) or ROOT / f'runs/{Path(args.out).stem}/viz')
    calls_log = getattr(args, 'calls_log', None)
    calls_log = Path(calls_log) if calls_log else viz_dir.parent / 'calls.jsonl'
    only = tuple(getattr(args, 'only_split', None) or ())
    return {'profile': profile, 'workers': workers, 'pad': float(pad), 'max_tokens': max_tokens,
            'cache_dir': cache_dir, 'viz_dir': viz_dir, 'calls_log': calls_log, 'only': only}


def run(args):
    label_path = Path(args.in_path)
    label = json.loads(label_path.read_text())
    if Path(args.out).resolve().is_relative_to((ROOT / 'data/gt').resolve()):
        raise SystemExit('--out must not be under data/gt/')

    plan = _resolve(args)
    if plan['profile'] == 'bootstrap' and getattr(args, 'max_tokens', None):
        print('note: --max-tokens only applies to --profile reader_compare; the bootstrap '
              'cache keys were built at max_tokens=4096', flush=True)
    stats = Stats()
    identifier = VLMCropIdentifier(args.model, catalog=None, upscale=args.upscale,
                                   mode='two_stage' if args.mode.startswith('two_stage') else 'existing',
                                   part_region=args.part_region)
    cache = None
    start_rows = 0
    if plan['profile'] == 'reader_compare':
        # Same cache layout, transport and generation settings as the reader comparison,
        # so a reply already stored under --cache-dir is reused instead of re-asked.
        cache = CallCache(plan['cache_dir'], plan['calls_log'], max_inflight=plan['workers'],
                          max_tokens=plan['max_tokens'])
        identifier._ask = _profile_ask(cache, args.model, stats)
        tag_for = _cache_tag(cache)
        start_rows = len(cache.call_rows())
    else:
        settings = {'mode': args.mode, 'upscale': args.upscale, 'pad': plan['pad'],
                    'part_region': list(args.part_region), 'model': args.model}
        identifier._ask = _cached_ask(plan['cache_dir'], args.model, stats, settings)
        tag_for = _noop_tag

    cats_inv = {c['id']: c['name'] for c in label['categories']}
    ids_by_name = {c['name']: c['id'] for c in label['categories']}
    unknown_id = ids_by_name.get('unknown')
    cut = _crop_fn(plan['profile'], plan['pad'])
    shelves_for = _shelf_fn(plan['profile'])
    use_shelf = _uses_shelf(args.mode, plan['profile'])
    stem = Path(args.out).stem

    wanted = args.images if args.images else None
    if wanted:
        unmatched = {w for w in wanted
                     if not any(_match_image(img['file_name'], [w]) for img in label['images'])}
        if unmatched:
            raise SystemExit(f'--images entries not present in {label_path}: {sorted(unmatched)}')
        all_images = [img for img in label['images'] if _match_image(img['file_name'], wanted)]
    else:
        all_images = list(label['images'])

    if plan['only']:
        splits_path = Path(getattr(args, 'splits', None) or ROOT / 'data/splits.json')
        if not splits_path.is_file():
            raise SystemExit(f'--only-split needs a splits file: {splits_path} not found')
        splits = json.loads(splits_path.read_text())
        skipped = [img['file_name'] for img in all_images
                   if not _split_wanted(splits, img['file_name'], plan['only'])]
        print(f'--only-split {",".join(plan["only"])}: keeping '
              f'{len(all_images) - len(skipped)} of {len(all_images)} images', flush=True)
        all_images = [img for img in all_images
                      if _split_wanted(splits, img['file_name'], plan['only'])]
        if not all_images:
            raise SystemExit(f'no images of {label_path} are in split(s) {plan["only"]}')

    out_images = []
    out_annotations = []
    per_image = []
    for image_info in all_images:
        path = IMAGES_ROOT / image_info['file_name']
        image = Image.open(path).convert('RGB')
        annotations = [a for a in label['annotations'] if a['image_id'] == image_info['id']]
        shelf_hints = (shelves_for(annotations, image_info.get('height')) if use_shelf
                       else [None] * len(annotations))

        jobs = [(ann, shelf, f'{stem}.{image_info["id"]}.{ann["id"]}')
                for ann, shelf in zip(annotations, shelf_hints)]

        def one(job):
            ann, shelf, tag = job
            try:
                im = cut(image, ann['bbox'])
            except ValueError:
                return {'crop': None, 'result': {'class': 'unknown', 'family': 'unknown',
                                                  'part': None, 'confidence': 0.0,
                                                  'error': 'crop-invalid'}}
            try:
                with tag_for(tag):
                    result = identifier.identify(im, shelf=shelf)
            except Exception as exc:
                result = {'class': 'unknown', 'family': 'unknown', 'part': None,
                          'confidence': 0.0, 'error': str(exc)[:200]}
            return {'crop': im, 'result': result, 'shelf': shelf}

        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
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
            new_attrs['id_profile'] = plan['profile']
            if ident.get('shelf'):
                new_attrs['id_shelf_hint'] = ident['shelf']
            else:
                new_attrs.pop('id_shelf_hint', None)   # never keep a stale hint
            new_attrs['identity_visible'] = name != 'unknown'
            new_attrs.pop('id_error', None)
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
                     plan['viz_dir'] / (image_info['file_name'].replace('/', '__') + '_overlay.jpg'),
                     unknown_id)
            _grid(crops, results,
                  plan['viz_dir'] / (image_info['file_name'].replace('/', '__') + '_grid.jpg'))

        print(f'[{image_info["file_name"]}] {len(annotations)} boxes -> '
              f'{v2_known_count(image_anns, unknown_id)} known / {v2_unknown} unknown', flush=True)

    calls = _call_log_stats(cache.call_rows()[start_rows:]) if cache is not None else None
    if calls is not None:
        stats.set_cache(calls['cache_hits'], calls['cache_misses'])

    result = {'info': dict(label.get('info', {})),
              'images': out_images,
              'annotations': out_annotations,
              'categories': label['categories']}
    result['info']['reidentification'] = {
        'source': str(label_path),
        'model': args.model,
        'mode': args.mode,
        'profile': plan['profile'],
        'upscale': args.upscale,
        'pad': plan['pad'],
        'max_tokens': plan['max_tokens'],
        'cache_dir': str(plan['cache_dir']),
        'calls_log': str(plan['calls_log']) if calls is not None else None,
        'splits': list(plan['only']) or None,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'shelf_inference': (
            'freestyle.quiz.build_queue.shelf_labels (two-band fit, position split fallback)'
            if plan['profile'] == 'reader_compare'
            else 'box-centroid gap (min_gap=200px); None when no two-shelf split is clear'),
    }
    if calls is not None:
        result['info']['reidentification']['calls'] = calls
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
        write_summary(plan['viz_dir'], result, per_image, stats, Path(args.out))

    total_calls = stats.calls_family + stats.calls_part
    extra = ''
    if calls is not None:
        served = ', '.join(calls['served_models']) or '-'
        extra = (f", {calls['retried']} retried, {calls['failed']} failed, "
                 f"{calls['network_s']:.0f}s in requests (served by {served})")
    print(f'\nwrote {out_path}: {len(out_images)} images, {len(out_annotations)} annotations '
          f'({total_calls} stage requests, {stats.cache_hits} cache hits, '
          f'{stats.cache_misses} sent{extra}, {stats.elapsed():.1f}s)')
    return 0


def v2_known_count(image_anns, unknown_id):
    return sum(1 for a in image_anns if a['category_id'] != unknown_id)


def write_summary(viz_dir, result, per_image, stats, out_path):
    all_anns = result['annotations']
    reid = result['info']['reidentification']
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

    lines = [f'# {Path(out_path).stem} summary', '',
             f'- Source: `{reid["source"]}`',
             f'- Output: `{out_path}`',
             f'- Model: `{reid["model"]}`, mode: `{reid["mode"]}`, profile: `{reid["profile"]}`',
             f'- Settings: upscale={reid["upscale"]}, pad={reid["pad"]}, max_tokens={reid["max_tokens"]}, '
             f'splits={reid["splits"] or "all"}',
             f'- Shelf hints: {reid["shelf_inference"]}',
             f'- Reply cache: `{reid["cache_dir"]}`',
             f'- Wall time: {stats.elapsed():.1f}s', f'- Annotations: {len(all_anns)} across {len(per_image)} images', '']
    lines.append('## VLM calls')
    total_calls = stats.calls_family + stats.calls_part
    hit_rate = stats.cache_hits / (stats.cache_hits + stats.cache_misses) if (stats.cache_hits + stats.cache_misses) else 0.0
    lines += [f'- family stage: {stats.calls_family}', f'- part stage: {stats.calls_part}',
              f'- total calls: {total_calls}',
              f'- cache hits: {stats.cache_hits} ({100 * hit_rate:.1f}%)',
              f'- cache misses: {stats.cache_misses}', '']
    if reid.get('calls'):
        calls = reid['calls']
        lines += [f'- retried after 429/5xx: {calls["retried"]}',
                  f'- failed: {calls["failed"]}',
                  f'- seconds in network requests: {calls["network_s"]}',
                  '- served model ids: ' + (', '.join(calls['served_models']) or '-'), '']
    lines.append('## Per-image unknown rate')
    lines.append('| image | annotations | input unknown | output unknown | output known |')
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
    lines.append(f'## Family distribution ({Path(out_path).stem})')
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
                        help='input proposal COCO (never data/gt)')
    parser.add_argument('--out', required=True, type=Path, help='output COCO (never data/gt)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'model id (default {DEFAULT_MODEL})')
    parser.add_argument('--mode', default=DEFAULT_MODE, choices=list(MODES),
                        help='two_stage_shelf = two_stage + a per-box shelf hint '
                             '(the reader_compare-winning recipe)')
    parser.add_argument('--profile', default=DEFAULT_PROFILE, choices=list(PROFILES),
                        help='crop + shelf-hint + cache recipe; reader_compare reuses '
                             'runs/reader_compare/cache replies')
    parser.add_argument('--upscale', type=float, default=DEFAULT_UPSCALE,
                        help='family-stage upscale factor (default 2.0)')
    parser.add_argument('--pad', type=float, default=None,
                        help=f'crop padding fraction (default {DEFAULT_PAD} for the bootstrap '
                             f'profile, {CROP_PAD} = the quiz crop for reader_compare)')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help=f'per-request max_tokens, reader_compare profile only '
                             f'(default {DEFAULT_MAX_TOKENS["bootstrap"]} bootstrap / '
                             f'{DEFAULT_MAX_TOKENS["reader_compare"]} reader_compare)')
    parser.add_argument('--max-workers', type=int, default=MAX_INFLIGHT,
                        help=f'concurrent crops, capped at {MAX_INFLIGHT} in-flight requests')
    parser.add_argument('--part-region', type=_parse_region, default=DEFAULT_PART_REGION,
                        metavar='L,T,R,B',
                        help='normalized part-region window (default 0.0,0.5,1.0,1.0)')
    parser.add_argument('--images', nargs='*', metavar='IMAGE_ID',
                        help='subset of image file_names to process (default: all)')
    parser.add_argument('--only-split', action='append', choices=['train', 'val', 'test', 'exclude'],
                        help='process only images of this split (repeatable, e.g. '
                             '--only-split val --only-split test)')
    parser.add_argument('--splits', type=Path, default=ROOT / 'data/splits.json',
                        help='split map used by --only-split')
    parser.add_argument('--cache-dir', type=Path, default=None,
                        help='reply cache (default runs/bootstrap_v2/cache, or '
                             'runs/reader_compare/cache with --profile reader_compare)')
    parser.add_argument('--calls-log', type=Path, default=None,
                        help='CallCache call log (default <run dir>/calls.jsonl, reader_compare profile)')
    parser.add_argument('--viz-dir', type=Path, default=None,
                        help='overlay/grid dir (default runs/<output stem>/viz)')
    parser.add_argument('--no-viz', action='store_true', help='skip overlay/grid rendering')
    parser.add_argument('--no-summary', action='store_true', help='skip summary.md')
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
