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


class VLMCropIdentifier:
    def __init__(self, model, catalog=None, reference_gallery=None, max_refs_per_class=1, upscale=2):
        if upscale <= 0 or max_refs_per_class < 0:
            raise ValueError('invalid upscale or reference limit')
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

    def identify(self, image):
        image = image.resize(tuple(max(1, round(x * self.upscale)) for x in image.size), Image.Resampling.LANCZOS)
        return self._result(_json(self._ask(image, self._prompt())))

    def identify_batch(self, crops):
        with ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(self.identify, crops))

    def contact_sheet(self, crops):
        crops = list(crops)
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
        response = _json(self._ask(sheet, self._prompt() + '\nFor each numbered crop return a JSON array of objects with index and class.'))
        results = [self._result(None) for _ in crops]
        if isinstance(response, list):
            for obj in response:
                if isinstance(obj, dict) and isinstance(obj.get('index'), int) and 0 <= obj['index'] < len(results):
                    results[obj['index']] = self._result(obj)
        return results


class CropIdentifier(VLMCropIdentifier):
    def predict(self, image, detections):
        results = self.identify_batch([crop(image, d['box']) for d in detections])
        return [{'class': r['class'], 'id_score': r['confidence']} for r in results]
