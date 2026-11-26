"""Identify original-image crops with a Collective vision model."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from freestyle import catalog as catalog_module, collective
from freestyle.geometry import crop


def _json(text):
    if '<think>' in text and '</think>' not in text:
        return None
    text = text.rsplit('</think>', 1)[-1]
    decoder = json.JSONDecoder()
    for match in re.finditer(r'[\[{]', text):
        try:
            return decoder.raw_decode(text[match.start():])[0]
        except ValueError:
            continue
    return None


# Visual cues supplement catalog names, aliases, colors, and shelf hints.
FAMILY_CUES = {
    'coca cola': 'red box, Coca-Cola logo',
    'diet coke': 'silver or white box, Diet Coke logo',
    'dr pepper': 'maroon box, Dr Pepper logo',
    'coca cola zero': 'black box, Coca-Cola Zero logo',
    'cherry': 'dark red box, Cherry text',
    'nns': 'blue box, NNS text, often stacked two high',
    'aa': 'yellow box, AA text',
    'fanta bb': 'orange box, Fanta logo and BB text',
    'cc': 'lavender box, CC text',
    'hi c': 'red and yellow Hi-C logo',
    'sprite': 'green box, Sprite logo',
    'orange': 'orange box, Orange text',
    'vanilla': 'tan box, Vanilla text',
    'lemon': 'yellow box, Lemon text',
}


def _normalized(value):
    return re.sub(r'[^a-z0-9]+', ' ', str(value).lower()).strip()


def _family_and_part(value):
    """Derive a family label and part from a catalog name or id."""
    text = _normalized(value)
    match = re.search(r'\bpart\s*([12])\b', text)
    if not match:
        return text, None
    return ' '.join((text[:match.start()] + text[match.end():]).split()), int(match.group(1))


def _confidence(value):
    try:
        value = float(value)
        return value if 0 <= value <= 1 else 0.0
    except (ValueError, TypeError):
        return 0.0


class VLMCropIdentifier:
    def __init__(self, model, catalog=None, reference_gallery=None, max_refs_per_class=1, upscale=2,
                 mode='existing', part_region=(0.0, 0.5, 1.0, 1.0)):
        if upscale <= 0 or max_refs_per_class < 0:
            raise ValueError('invalid upscale or reference limit')
        if mode not in ('existing', 'two_stage'):
            raise ValueError('invalid identification mode')
        if (len(part_region) != 4 or not all(0 <= v <= 1 for v in part_region)
                or part_region[0] >= part_region[2] or part_region[1] >= part_region[3]):
            raise ValueError('part_region must be normalized (left, top, right, bottom)')
        self.mode, self.part_region = mode, tuple(part_region)
        self.model, self.catalog = model, catalog
        self.reference_gallery, self.max_refs_per_class, self.upscale = reference_gallery or {}, max_refs_per_class, upscale

    def _descriptions(self):
        if self.catalog is None:
            return {name: catalog_module.describe(name) for name in catalog_module.class_names()}
        if isinstance(self.catalog, (str, Path)):
            return {name: catalog_module.describe(name, str(self.catalog)) for name in catalog_module.class_names(str(self.catalog))}
        if isinstance(self.catalog, dict):
            return {str(k): str(v) for k, v in self.catalog.items()}
        return {str(row['id']): (f"{row.get('name', row['id'])} [{row['id']}] aliases={row.get('aliases', [])} "
                                 f"colour={row.get('color_hint', '')} shelf={row.get('shelf_hint', '')}")
                if isinstance(row, dict) else str(row) for row in self.catalog}

    def _ask(self, image, prompt):
        if not self.reference_gallery:
            return collective.chat_image(self.model, image, prompt)
        content = [{'type': 'text', 'text': prompt},
                   {'type': 'image_url', 'image_url': {'url': collective.image_to_data_url(image)}}]
        for name, refs in self.reference_gallery.items():
            if isinstance(refs, (str, Path, Image.Image)):
                refs = [refs]
            for ref in list(refs)[:self.max_refs_per_class]:
                content.extend([{'type': 'text', 'text': f'Reference: {name}'},
                    {'type': 'image_url', 'image_url': {'url': collective.image_to_data_url(collective.load_image_source(ref))}}])
        client = collective.get_client()
        for attempt in range(5):
            try:
                response = client.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': content}],
                                                          max_tokens=4096, temperature=0)
                return response.choices[0].message.content or ''
            except Exception as exc:
                if attempt == 4 or not collective._is_retryable(exc):
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))

    def _prompt(self):
        return ('Identify the cartridge using these catalog choices:\n' + '\n'.join(self._descriptions().values()) +
                '\nChoose an exact catalog id or unknown when uncertain. Return JSON with class, confidence (0 to 1), and rationale.')

    def _result(self, obj):
        if not isinstance(obj, dict):
            obj = {}
        name = str(obj.get('class', 'unknown'))
        invalid = name not in self._descriptions()
        if invalid:
            name = 'unknown'
        try:
            confidence = float(obj.get('confidence', 0))
            if not 0 <= confidence <= 1:
                confidence = 0.0
        except (ValueError, TypeError):
            confidence = 0.0
        return {'class': name, 'confidence': 0.0 if invalid else confidence, 'rationale': str(obj.get('rationale', ''))}

    def _families(self):
        if self.catalog is None or isinstance(self.catalog, (str, Path)):
            rows = catalog_module.load_catalog(path=str(self.catalog) if self.catalog is not None else None)
        elif isinstance(self.catalog, dict):
            rows = [{'id': key, 'name': str(value).split(' [', 1)[0]}
                    for key, value in self.catalog.items()]
        else:
            rows = self.catalog
        families = {}
        for row in rows:
            if not isinstance(row, dict) or str(row.get('id', 'unknown')) == 'unknown':
                continue
            class_id = str(row['id'])
            family, part = _family_and_part(row.get('name', class_id))
            if part is None:
                id_family, id_part = _family_and_part(class_id)
                if id_part is not None:
                    family, part = id_family, id_part
            entry = families.setdefault(family, {'classes': {}, 'descriptions': []})
            entry['classes'][part] = class_id
            entry['descriptions'].append(self._descriptions().get(class_id, class_id))
        return families

    def _identify_two_stage(self, image, shelf=None):
        if shelf is not None and shelf not in ('top', 'bottom'):
            raise ValueError('shelf must be top, bottom, or None')
        families = self._families()
        choices = []
        for family, entry in families.items():
            cue = FAMILY_CUES.get(family, '')
            choices.append(f"{family}: {'; '.join(entry['descriptions'])}; visual cues={cue}")
        prompt = ('Identify the cartridge family from its logo, text and colour. '
                  'Choose a family exactly as listed, or unknown. Return JSON with family and '
                  'confidence (0 to 1). Do not guess the part number.\n' + '\n'.join(choices))
        if shelf is not None:
            prompt += f'\nCaller-provided shelf: {shelf}. Use this as a hint.'
        enlarged = image.resize(tuple(max(1, round(x * self.upscale)) for x in image.size),
                                Image.Resampling.LANCZOS)
        response = _json(self._ask(enlarged, prompt))
        response = response if isinstance(response, dict) else {}
        family = _normalized(response.get('family', 'unknown'))
        confidence = _confidence(response.get('confidence', 0))
        if family not in families:
            return {'class': 'unknown', 'family': 'unknown', 'part': None, 'confidence': 0.0}
        classes = families[family]['classes']
        if 1 not in classes and 2 not in classes:
            return {'class': classes.get(None, 'unknown'), 'family': family, 'part': None,
                    'confidence': confidence}
        left, top, right, bottom = self.part_region
        region = image.crop((int(left * image.width), int(top * image.height),
                             max(int(left * image.width) + 1, round(right * image.width)),
                             max(int(top * image.height) + 1, round(bottom * image.height))))
        region = region.resize((region.width * 3, region.height * 3), Image.Resampling.LANCZOS)
        response = _json(self._ask(region, 'Read the small printed part numeral near the box end face, '
                                   'possibly inside a circle. Return JSON with part: 1, 2, or '
                                   '"unreadable", and confidence (0 to 1). Do not guess.'))
        response = response if isinstance(response, dict) else {}
        value = response.get('part')
        part = int(value) if not isinstance(value, bool) and str(value) in ('1', '2') else None
        if part not in classes:
            part = None
        return {'class': classes.get(part, 'unknown') if part is not None else 'unknown',
                'family': family, 'part': part,
                'confidence': min(confidence, _confidence(response.get('confidence', 0)))
                if part is not None else confidence}

    def identify(self, image, shelf=None):
        if self.mode == 'two_stage':
            return self._identify_two_stage(image, shelf)
        image = image.resize(tuple(max(1, round(x * self.upscale)) for x in image.size), Image.Resampling.LANCZOS)
        return self._result(_json(self._ask(image, self._prompt())))

    def identify_batch(self, crops, shelves=None):
        crops = list(crops)
        if shelves is None:
            shelves = [None] * len(crops)
        else:
            shelves = list(shelves)
            if len(shelves) != len(crops):
                raise ValueError('one shelf is required per crop')
        with ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(self.identify, crops, shelves))

    def contact_sheet(self, crops, shelves=None):
        crops = list(crops)
        if self.mode == 'two_stage':
            return self.identify_batch(crops, shelves)
        if not crops:
            return []
        width = max(im.width for im in crops) * self.upscale
        height = max(im.height for im in crops) * self.upscale
        width, height = max(1, round(width)), max(1, round(height))
        sheet = Image.new('RGB', (width * len(crops), height + 24), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, im in enumerate(crops):
            draw.text((i * width, 0), str(i), fill='black')
            im = im.resize((max(1, round(im.width * self.upscale)),
                            max(1, round(im.height * self.upscale))), Image.Resampling.LANCZOS)
            sheet.paste(im.convert('RGB'), (i * width, 24))
        response = _json(self._ask(sheet, self._prompt() + '\nFor each numbered crop return a JSON array of objects with index, class, and confidence (0 to 1).'))
        results = [self._result(None) for _ in crops]
        if isinstance(response, list):
            for obj in response:
                if isinstance(obj, dict) and isinstance(obj.get('index'), int) and 0 <= obj['index'] < len(results):
                    results[obj['index']] = self._result(obj)
        return results


class CropIdentifier(VLMCropIdentifier):
    def predict(self, image, detections):
        results = self.identify_batch([crop(image, d['box']) for d in detections],
                                      [d.get('shelf') for d in detections])
        if self.mode == 'two_stage':
            return [{**r, 'id_score': r['confidence']} for r in results]
        return [{'class': r['class'], 'id_score': r['confidence']} for r in results]
