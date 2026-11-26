"""Create resumable, model-generated COCO proposals for human review."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from freestyle import catalog, data
from freestyle.geometry import crop, iou_matrix, weighted_box_fusion
from freestyle.vlm_identify import VLMCropIdentifier
from tools.vlm_client import VLMResponse, run_vlm
from workflows.vlm_judge import judge_labels
from workflows.vlm_label import label_dataset

ROOT = Path(__file__).resolve().parent.parent


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _cache(directory, key, call):
    path = directory / ( _hash(key) + '.json')
    if path.exists():
        return json.loads(path.read_text())
    result = call()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=directory, delete=False) as f:
        try:
            json.dump(result, f)
            name = f.name
        except BaseException:
            Path(f.name).unlink(missing_ok=True)
            raise
    os.replace(name, path)
    return result


def _retry(call):
    """Retry *call* on 429/5xx/timeout/connection errors (exp. backoff + jitter).

    Bounded to ~10 min total per request, matching the project rule that the
    Collective is often at capacity (HTTP 429 "backend at capacity").
    """
    import random

    import openai

    deadline = time.monotonic() + 600
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            retryable = (
                isinstance(exc, (openai.RateLimitError, openai.APITimeoutError,
                                openai.APIConnectionError))
                or (isinstance(status, int) and (status == 429 or status >= 500))
            )
            if not retryable:
                raise
            delay = min(60, 2 ** min(attempt + 1, 6)) * (1 + random.random() * 0.3)
            if time.monotonic() + delay > deadline:
                raise
            time.sleep(delay)
            attempt += 1


def _cached_vlm(cache_dir):
    counts = {}

    def call(image, prompt, model_id, **kwargs):
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        options = {k: v for k, v in kwargs.items() if k not in ('api_key', 'raw_response_log')}
        key = ['response-v1', hashlib.sha256(buf.getvalue()).hexdigest(), prompt, model_id, options]

        request_id = _hash(key)
        occurrence = counts.get(request_id, 0)
        counts[request_id] = occurrence + 1
        key.append(occurrence)

        def request():
            response = _retry(lambda: run_vlm(image, prompt, model_id, **kwargs))
            return {'text': str(response), 'finish_reason': getattr(response, 'finish_reason', None)}

        result = _cache(cache_dir, key, request)
        return VLMResponse(result['text'], result['finish_reason'])
    call.reset = counts.clear
    return call


def fuse_runs(runs, threshold=0.55):
    """Fuse COCO annotations and record which runs overlap each fused box."""
    groups = []
    for run in runs:
        group = []
        for ann in run['annotations']:
            x, y, w, h = ann['bbox']
            group.append({'box': [x, y, x + w, y + h], 'score': ann.get('score', 1.0),
                          'category_id': 'cartridge'})
        groups.append(group)
    fused = weighted_box_fusion(groups, iou=threshold)
    for det in fused:
        provenance = []
        for index, group in enumerate(groups):
            matches = [d for d in group if iou_matrix([det['box']], [d['box']])[0, 0] > threshold]
            if matches:
                provenance.append({'run': runs[index]['run'], 'boxes': [d['box'] for d in matches]})
        det['provenance'] = provenance
        det['support'] = len(provenance) / len(runs) if runs else 0
    return sorted(fused, key=lambda d: (d['box'][1], d['box'][0]))


def _images(spec):
    records = {r['image_id']: r for r in data.list_images()}
    if Path(spec).is_file():
        ids = [line.strip() for line in Path(spec).read_text().splitlines()
               if line.strip() and not line.lstrip().startswith('#')]
    else:
        splits = data.load_splits()
        if spec not in (*data.VALID_SPLITS, 'negatives'):
            raise ValueError(f'unknown split: {spec}')
        ids = (splits.get('negatives', []) if spec == 'negatives' else
               [k for k, v in splits['images'].items() if v == spec])
    missing = sorted(set(ids) - records.keys())
    if missing:
        raise FileNotFoundError(f'images missing from inventory: {missing[:3]}')
    return [records[i] for i in sorted(set(ids))]


def bootstrap(config, images, out):
    out = Path(out)
    if out.resolve().is_relative_to((ROOT / 'data/gt').resolve()):
        raise ValueError('proposals must not be written under data/gt/')
    cache_dir = Path(config.get('cache_dir', ROOT / 'data/labels/cache'))
    root = Path(config.get('images_root', data.IMAGES_ROOT))
    catalog_path = config.get('catalog')
    names = catalog.class_names(catalog_path)
    descriptions = {n: catalog.describe(n, catalog_path) for n in names}
    identifier = VLMCropIdentifier(config['identifier']['model'], catalog=descriptions,
                                   upscale=config['identifier'].get('upscale', 2))
    api_key = os.environ.get(config.get('api_key_env', 'LLMC_API_KEY'))
    base = config.get('base_url')
    cached_vlm = _cached_vlm(cache_dir)
    identifier._ask = lambda image, prompt: cached_vlm(
        image, prompt, identifier.model, backend='openai', base_url=base, api_key=api_key,
        max_tokens=config['identifier'].get('max_tokens', 4096), max_dimension=None,
        image_format='png')
    audit_path = Path(config.get('audit', ROOT / 'data/derived/audit/audit.json'))
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    scenes = {r['image_id']: r.get('scene_label') for r in audit.get('images', [])}
    categories = [{'id': i + 1, 'name': n} for i, n in enumerate(names)]
    ids = {c['name']: c['id'] for c in categories}
    result = {'info': {'description': 'Model-generated proposals; not ground truth'},
              'images': [], 'annotations': [], 'categories': categories}
    runs_config = config['labellers']
    judge = config['judge']
    if judge['model'] == config['primary_labeller']:
        raise ValueError('judge model must differ from primary labeller')
    with tempfile.TemporaryDirectory() as tmp, patch('tools.vlm_detect.run_vlm', cached_vlm), \
            patch('workflows.vlm_judge.run_vlm', cached_vlm):
        tmp = Path(tmp)
        for image_index, record in enumerate(images, 1):
            image_id = record['image_id']
            path = root / image_id
            image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as opened:
                image = opened.convert('RGB')
            info = {'id': image_index, 'file_name': image_id, 'width': image.width,
                    'height': image.height, 'group_id': record.get('group_id'),
                    'attributes': {'verified': False, 'scene': scenes.get(image_id)}}
            result['images'].append(info)
            runs = []
            for index, run in enumerate(runs_config):
                options = {'model_id': run['model'], 'backend': 'openai', 'base_url': base,
                           'max_dimension': None, 'image_format': 'png',
                           'max_tokens': run.get('max_tokens', 8192), 'tile': run.get('tile'),
                           'tile_overlap': run.get('tile_overlap', 0.2),
                           'n_samples': run.get('n_samples', 1),
                           'temperature': run.get('temperature', 0)}
                key = ['label-v1', image_hash, image_id, options]
                def label():
                    cached_vlm.reset()
                    return label_dataset(source=root, classes=['cartridge'], output=tmp / 'label.json',
                                         api_key=api_key, image_paths=[image_id], **options)
                coco = _cache(cache_dir, key, label)
                runs.append({'run': run.get('name', str(index)), 'annotations': coco['annotations']})
                if any(im.get('truncated') for im in coco['images']):
                    info['attributes']['truncated_detection'] = True
            fused = fuse_runs(runs, config.get('fusion_iou', 0.55))
            anns = []
            for index, det in enumerate(fused, 1):
                x1, y1, x2, y2 = det['box']
                anns.append({'id': index, 'image_id': image_index, 'category_id': 1,
                             'bbox': [x1, y1, x2-x1, y2-y1]})
            judge_input = {'images': [info], 'annotations': anns,
                           'categories': [{'id': 1, 'name': 'cartridge'}]}
            (tmp / 'input.json').write_text(json.dumps(judge_input))
            judge_options = {'model_id': judge['model'], 'backend': 'openai', 'base_url': base,
                             'judge_mode': 'crops', 'judge_batch_size': judge.get('batch_size', 8),
                             'max_dimension': None, 'image_format': 'png',
                             'max_tokens': judge.get('max_tokens', 4096), 'threshold': 0.0,
                             'class_descriptions': {'cartridge': catalog.describe('cartridge')}}
            def judge_call():
                cached_vlm.reset()
                judge_labels(source=root, output=tmp / 'judged.json', annotations=tmp / 'input.json',
                             api_key=api_key, **judge_options)
                return json.loads((tmp / 'judged_verdicts.json').read_text())
            verdicts = _cache(cache_dir, ['judge-v1', image_hash, judge_input, judge_options], judge_call) if anns else []
            verdict_map = {v['annotation_id']: v for v in verdicts}
            batch_size = config['identifier'].get('batch_size', 8)
            identities = []
            for start in range(0, len(fused), batch_size):
                boxes = [d['box'] for d in fused[start:start + batch_size]]
                key = ['identify-v1', image_hash, boxes, config['identifier'], descriptions, base]
                def identify():
                    cached_vlm.reset()
                    return identifier.contact_sheet([crop(image, box) for box in boxes])
                identities.extend(_cache(cache_dir, key, identify))
            for ann, det, identity in zip(anns, fused, identities):
                name = identity.get('class', 'unknown')
                confidence = identity.get('confidence', 0)
                if name not in ids or confidence < config['identifier'].get('min_confidence', 0):
                    name = 'unknown'
                verdict = verdict_map.get(ann['id'], {})
                x, y, w, h = ann['bbox']
                result['annotations'].append({**ann, 'id': len(result['annotations']) + 1,
                    'category_id': ids[name], 'area': w*h, 'iscrowd': 0,
                    'attributes': {'source': 'vlm:bootstrap', 'truncated': False, 'occluded': False,
                        'identity_visible': name != 'unknown', 'support': det['support'],
                        'provenance': det['provenance'], 'judge_verdict': verdict.get('verdict', 'unjudged'),
                        'judge_score': verdict.get('score', 0.0), 'id_model': identifier.model,
                        'id_confidence': confidence}})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--images', required=True, help='split name or text file of image IDs')
    parser.add_argument('--out', required=True, type=Path, help='proposal COCO JSON; never data/gt')
    args = parser.parse_args(argv)
    if args.out.resolve().is_relative_to((ROOT / 'data/gt').resolve()):
        parser.error('--out must not be under data/gt/')
    config = yaml.safe_load(args.config.read_text())
    bootstrap(config, _images(args.images), args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
