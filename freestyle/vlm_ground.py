"""Direct VLM grounding in original pixels.

Uses the bbox parser and 0–1000 convention from :mod:`tools.vlm_detect`.
Coordinate units must be specified; the bundled prompt uses norm1000.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image

from freestyle import catalog, collective
from freestyle.geometry import iou_matrix, make_tiles, merge_tiled, weighted_box_fusion
from tools.vlm_detect import denormalize_bbox, parse_bboxes_from_response


def parse_boxes(text):
    """Reuse Vision Intern's tolerant parser, adding Qwen token boxes."""
    result = parse_bboxes_from_response(text)
    if '<think>' in text and '</think>' not in text:
        return result
    text = text.rsplit('</think>', 1)[-1]
    token = r'<\|box_start\|>\s*\(?\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*\)?\s*[,;]?\s*\(?\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*\)?\s*<\|box_end\|>'
    result.extend({'bbox_2d': [float(v) for v in m], 'label': 'cartridge'}
                  for m in re.findall(token, text))
    return result


def _convert(raw, size, mode):
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        return None
    w, h = size
    if mode not in ('norm1000', 'pixel', 'norm1'):
        raise ValueError('Pass the coordinate mode measured by the probe')
    if mode == 'norm1000':
        box = denormalize_bbox(values, w, h)
    else:
        scale = (w, h, w, h) if mode == 'norm1' else (1, 1, 1, 1)
        box = [max(0, min(limit, round(v * s))) for v, s, limit in
               zip(values, scale, (w, h, w, h))]
    return box if box[2] > box[0] and box[3] > box[1] else None


class VLMGrounder:
    def __init__(self, model, coord_mode='norm1000', tile=0, overlap=0.25, max_side=None,
                 prompt_variant='per_shelf', with_identity=False, temperature=0,
                 n_samples=1, raw_jsonl=None):
        if coord_mode not in ('norm1000', 'pixel', 'norm1'):
            raise ValueError('Pass the coordinate mode measured by the probe')
        if prompt_variant not in ('per_shelf', 'all') or tile < 0 or not 0 <= overlap < 1 or n_samples < 1:
            raise ValueError('invalid grounding options')
        self.model, self.coord_mode, self.tile, self.overlap = model, coord_mode, tile, overlap
        self.max_side, self.prompt_variant, self.with_identity = max_side, prompt_variant, with_identity
        self.temperature, self.n_samples, self.raw_jsonl = temperature, n_samples, raw_jsonl

    def _prompt(self):
        labels = ', '.join(catalog.class_names()) if self.with_identity else 'cartridge'
        shelf = 'Inspect each shelf in turn; do not omit partially visible cartridges. ' if self.prompt_variant == 'per_shelf' else ''
        return (shelf + 'Detect EVERY cartridge face. Return a JSON array of objects with '
                '"bbox_2d": [x1,y1,x2,y2] normalized to 0-1000 and "label": '
                f'one of {labels}. Return [] if none.')

    def detect(self, pil, image_id=None):
        rects = make_tiles(*pil.size, tile=self.tile, overlap=self.overlap) if self.tile else [(0, 0, *pil.size)]
        samples = []
        for sample in range(self.n_samples):
            detections = []
            for rect in rects:
                image = pil.crop(rect)
                response = collective.chat_image(self.model, image, self._prompt(), max_side=self.max_side,
                                                 temperature=self.temperature)
                if self.raw_jsonl:
                    with Path(self.raw_jsonl).open('a', encoding='utf-8') as f:
                        f.write(json.dumps({'image_id': image_id, 'tile': rect, 'sample': sample,
                                            'response': response}) + '\n')
                size = collective.resized_dims(image.size, self.max_side)
                for obj in parse_boxes(response):
                    box = _convert(obj.get('bbox_2d', []), size, self.coord_mode)
                    if box is None:
                        continue
                    box = [box[0] * image.width / size[0] + rect[0], box[1] * image.height / size[1] + rect[1],
                           box[2] * image.width / size[0] + rect[0], box[3] * image.height / size[1] + rect[1]]
                    label = str(obj.get('label', 'unknown')) if self.with_identity else 'cartridge'
                    detections.append({'box': box, 'score': 1.0, 'category_id': label})
            samples.append(merge_tiled(detections))
        fused = weighted_box_fusion(samples) if self.n_samples > 1 else samples[0]
        result = []
        for det in fused:
            agreement = sum(any(d['category_id'] == det['category_id'] and
                                iou_matrix([d['box']], [det['box']])[0, 0] > 0.55
                                for d in group) for group in samples)
            result.append({'xyxy': det['box'], 'score': agreement / self.n_samples,
                           'label': det['category_id']})
        return result

    def label_coco(self, images, out_coco):
        """Write COCO; images is an iterable of (image id, path) pairs.

        Paths become file_name values, relative to the judge's source directory
        when callers pass relative paths. Use an absolute path if appropriate.
        """
        names = catalog.class_names()
        if not self.with_identity:
            names = names + ['cartridge']
        categories = [{'id': i + 1, 'name': name} for i, name in enumerate(names)]
        ids = {c['name']: c['id'] for c in categories}
        coco = {'images': [], 'annotations': [], 'categories': categories}
        for image_id, path in images:
            with Image.open(path) as source:
                image = source.convert('RGB')
            coco['images'].append({'id': image_id, 'file_name': str(path), 'width': image.width, 'height': image.height})
            for det in self.detect(image, image_id):
                x1, y1, x2, y2 = det['xyxy']
                label = det['label'] if det['label'] in ids else 'unknown'
                coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': image_id,
                    'category_id': ids[label], 'bbox': [x1, y1, x2-x1, y2-y1],
                    'area': (x2-x1)*(y2-y1), 'iscrowd': 0,
                    'attributes': {'truncated': False, 'occluded': False,
                                   'identity_visible': label not in ('unknown', 'cartridge'),
                                   'source': f'vlm:{self.model}'}})
        path = Path(out_coco)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(coco, indent=2), encoding='utf-8')
        return coco


def label_coco(images, out_coco, model, **kwargs):
    """Ground images and write a COCO file using the supplied model options."""
    return VLMGrounder(model, **kwargs).label_coco(images, out_coco)


class Localizer(VLMGrounder):
    def predict(self, image):
        return [{'box': d['xyxy'], 'score': d['score'],
                 'class': (d['label'] if d['label'] in catalog.class_names() else 'unknown') if self.with_identity else None,
                 'id_score': d['score'] if self.with_identity and d['label'] in catalog.class_names() else 0.0}
                for d in self.detect(image)]


from freestyle.vlm_identify import CropIdentifier  # noqa: E402,F401
