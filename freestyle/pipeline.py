"""Composable original-pixel localization, identification and counting pipeline."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from freestyle.count import aggregate_counts, assign_shelves, suspicious_gaps, suspicious_overlaps
from freestyle.geometry import crop, xywh_to_xyxy, xyxy_to_xywh


class ProposalLocalizer:
    def __init__(self, path, categories=None):
        self.records = json.loads(Path(path).read_text())
        self.categories = {str(v): k for k, v in (categories or {}).items()}

    def predict(self, image, image_id):
        return [{'box': xywh_to_xyxy(d['bbox']).tolist(), 'score': d['score'],
                 'class': self.categories.get(str(d['category_id'])), 'id_score': d.get('id_score', 0)}
                for d in self.records if str(d['image_id']) == str(image_id)]


class EmbeddingIdentifier:
    def __init__(self, gallery, model='facebook/dinov2-base', k=5, **kwargs):
        from freestyle.identify.classify import KNNClassifier
        from freestyle.identify.embed import Embedder

        with np.load(gallery, allow_pickle=False) as data:
            self.classifier = KNNClassifier(k=k).fit(data['embeddings'], data['labels'])
        self.embedder = Embedder(name=model, **kwargs)

    def predict(self, image, detections):
        embeddings = self.embedder.embed([crop(image, d['box']) for d in detections])
        if not len(detections):
            return []
        # The maximum class probability is the confidence used for rejection.
        probs = self.classifier.predict_proba(embeddings)
        indices = probs.argmax(axis=1)
        return [{'class': str(self.classifier.classes_[i]), 'id_score': float(p[i])}
                for i, p in zip(indices, probs)]


class Pipeline:
    def __init__(self, localizer, identifier=None, score_thr=0.0, identity_thr=0.5,
                 neighbour_vote='off', max_rows=4):
        self.localizer, self.identifier = localizer, identifier
        self.score_thr, self.identity_thr = score_thr, identity_thr
        self.neighbour_vote, self.max_rows = neighbour_vote, max_rows

    @classmethod
    def from_config(cls, yaml_path):
        import yaml

        config = yaml.safe_load(Path(yaml_path).read_text())
        loc = dict(config['localizer'])
        kind = loc.pop('type')
        if kind == 'tiled_detector':
            from freestyle.infer.detector import TiledDetector
            localizer = TiledDetector(**loc)
        elif kind == 'proposals':
            localizer = ProposalLocalizer(categories=config.get('categories'), **loc)
        elif kind == 'vlm':
            from freestyle.vlm_ground import Localizer
            localizer = Localizer(**loc)
        else:
            raise ValueError(f'unknown localizer: {kind}')
        ident = dict(config.get('identifier', {'type': 'none'}))
        kind = ident.pop('type')
        if kind == 'embedding_knn':
            identifier = EmbeddingIdentifier(**ident)
        elif kind == 'none':
            identifier = None
        elif kind == 'vlm_crop':
            from freestyle.vlm_ground import CropIdentifier
            identifier = CropIdentifier(**ident)
        else:
            raise ValueError(f'unknown identifier: {kind}')
        return cls(localizer, identifier, **config.get('counting', {}))

    def run(self, pil_image, image_id):
        timings = {}
        start = time.perf_counter()
        if isinstance(self.localizer, ProposalLocalizer):
            detections = self.localizer.predict(pil_image, image_id)
        else:
            detections = self.localizer.predict(pil_image)
        detections = [dict(d) for d in detections if d['score'] >= self.score_thr]
        timings['localize'] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        if self.identifier is not None:
            identities = self.identifier.predict(pil_image, detections)
            if len(identities) != len(detections):
                raise ValueError('identifier must return one identity per detection')
            for d, identity in zip(detections, identities):
                d.update(identity)
        for d in detections:
            d['class'] = d.get('class') or 'unknown'
            d['id_score'] = float(d.get('id_score', 0))
        timings['identify'] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        detections = assign_shelves(detections, self.max_rows)
        flags = []
        for shelf in sorted({d['shelf_idx'] for d in detections}):
            indices = [i for i, d in enumerate(detections) if d['shelf_idx'] == shelf]
            row = [detections[i] for i in indices]
            for name, pairs in [('gap', suspicious_gaps(row)), ('overlap', suspicious_overlaps(row))]:
                flags.extend({'type': name, 'detections': [indices[a], indices[b]]} for a, b in pairs)
        counts, unknown, votes = aggregate_counts(detections, self.score_thr, self.identity_thr,
                                                   self.neighbour_vote)
        flags.extend(votes)
        if self.neighbour_vote == 'apply':
            for flag in votes:
                detections[flag['detection']]['class'] = flag['suggested_class']
        timings['count'] = (time.perf_counter() - start) * 1000
        for d in detections:
            d['shelf'] = d.pop('shelf_idx')
            d['column'] = d.pop('column_idx')
        return {'detections': detections, 'counts': counts, 'unknown': unknown,
                'flags': flags, 'timings_ms': timings}


def run_images(pipeline, images, out, categories):
    """Write outputs for iterable (COCO image ID, image path) pairs."""
    out = Path(out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'overlays').mkdir(parents=True, exist_ok=True)
    predictions, counts = [], {}
    for image_id, path in images:
        with Image.open(path) as source:
            image = source.convert('RGB')
        result = pipeline.run(image, image_id)
        name = str(image_id).replace('/', '_').replace('\\', '_')
        (out / 'images' / f'{name}.json').write_text(json.dumps(result, indent=2))
        counts[str(image_id)] = {'counts': result['counts'], 'unknown': result['unknown']}
        draw = ImageDraw.Draw(image)
        for d in result['detections']:
            if d['class'] not in categories:
                raise ValueError(f"Missing COCO category ID for {d['class']!r}")
            predictions.append({'image_id': image_id, 'bbox': xyxy_to_xywh(d['box']).tolist(),
                                'score': float(d['score']), 'category_id': categories[d['class']],
                                'id_score': d['id_score']})
            draw.rectangle(d['box'], outline='lime', width=2)
            draw.text((d['box'][0], d['box'][1]), f"{d['class'][:18]} S{d['shelf']}", fill='yellow')
        image.save(out / 'overlays' / f'{name}.png')
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2))
    (out / 'counts.json').write_text(json.dumps(counts, indent=2))


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--images', nargs='+', required=True,
                        help='Image paths, or a split name when split_map is configured')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if len(args.images) == 1 and args.images[0] in ('train', 'val', 'test', 'exclude'):
        mapping = json.loads(Path(config['split_map']).read_text())['images']
        paths = [p for p, split in mapping.items() if split == args.images[0]]
    else:
        paths = args.images
    root = Path(config.get('images_root', '.'))
    manifest = json.loads(Path(config['manifest']).read_text()) if config.get('manifest') else None
    ids = {item['file_name']: item['id'] for item in manifest['images']} if manifest else {}
    if manifest and any(p not in ids for p in paths):
        raise ValueError('every image must be present in manifest')
    if not manifest:
        raise ValueError('manifest is required for COCO image IDs')
    run_images(Pipeline.from_config(args.config), [(ids[p], root / p) for p in paths],
               args.out, config['categories'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
