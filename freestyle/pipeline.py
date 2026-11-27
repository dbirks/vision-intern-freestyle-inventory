"""Composable original-pixel localization, identification and counting pipeline."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from freestyle.count import aggregate_counts, assign_shelves, suspicious_gaps, suspicious_overlaps
from freestyle.count.pairs import infer_pairs
from freestyle.geometry import crop, suppress_duplicates, xywh_to_xyxy, xyxy_to_xywh


class ProposalLocalizer:
    def __init__(self, path, categories=None):
        self.records = json.loads(Path(path).read_text())
        self.categories = {str(v): k for k, v in (categories or {}).items()}

    def predict(self, image, image_id):
        return [{'box': xywh_to_xyxy(d['bbox']).tolist(), 'score': d['score'],
                 'class': self.categories.get(str(d['category_id'])), 'id_score': d.get('id_score', 0)}
                for d in self.records if str(d['image_id']) == str(image_id)]


class EmbeddingIdentifier:
    """Identify crops from a train-only embedding gallery.

    ``classifier='knn'`` votes over the top-k gallery rows, ``'prototype'`` compares
    against one mean vector per class. ``max_per_class`` caps how many rows a class
    may contribute and ``balance`` (knn) weights each class inversely to its gallery
    size; both stop a 60-crop class from deciding every vote.
    """

    def __init__(self, gallery, model='facebook/dinov2-base', k=5, classifier='knn',
                 max_per_class=None, balance=False, temperature=None, **kwargs):
        from freestyle.identify.embed import Embedder

        if classifier not in ('knn', 'prototype'):
            raise ValueError(f'unknown classifier: {classifier}')
        with np.load(gallery, allow_pickle=False) as data:
            for key in ('image_ids', 'splits', 'groups'):
                if key not in data or len(data[key]) != len(data['labels']) or any(not str(v) for v in data[key]):
                    raise ValueError(f'Gallery requires complete {key} provenance')
            if any(str(v) != 'train' for v in data['splits']):
                raise ValueError('Gallery may contain train rows only')
            self.gallery_groups = set(map(str, data['groups']))
            self.embeddings = np.asarray(data['embeddings'], dtype=np.float32)
            self.labels = np.asarray(data['labels'], dtype=str)
        self.classifier_name = classifier
        self.options = {'k': k, 'max_per_class': max_per_class, 'balance': balance,
                        'temperature': temperature}
        self.classifier = self._fit(self.embeddings, self.labels)
        self.embedder = Embedder(name=model, **kwargs)

    def _fit(self, embeddings, labels):
        from freestyle.identify.classify import KNNClassifier, PrototypeClassifier

        if self.classifier_name == 'prototype':
            options = {'max_per_class': self.options['max_per_class']}
            if self.options['temperature'] is not None:
                options['temperature'] = self.options['temperature']
            clf = PrototypeClassifier(**options)
        else:
            options = {'k': self.options['k'], 'balance': self.options['balance'],
                       'max_per_class': self.options['max_per_class']}
            if self.options['temperature'] is not None:
                options['temperature'] = self.options['temperature']
            clf = KNNClassifier(**options)
        return clf.fit(embeddings, labels)

    def restricted(self, classes):
        """Return a classifier over gallery rows of ``classes`` only (or None)."""
        keep = np.isin(self.labels, list(classes))
        if not keep.any():
            return None
        return self._fit(self.embeddings[keep], self.labels[keep])

    def embed(self, image, detections):
        return self.embedder.embed([crop(image, d['box']) for d in detections])

    def predict(self, image, detections):
        embeddings = self.embed(image, detections)
        return self.predict_embeddings(embeddings)

    def predict_embeddings(self, embeddings):
        if len(embeddings) == 0:
            return []
        labels, scores = self.classifier.predict_with_scores(embeddings, unknown_thr=None)
        return [{'class': str(label), 'id_score': float(score)}
                for label, score in zip(labels, scores)]


class HybridIdentifier:
    """VLM family first, then embedding kNN inside that family to pick the part.

    The VLM two-stage pass is asked for the family only; the part numeral is decided
    by nearest gallery crops *restricted to that family's classes*, so a misread
    numeral cannot move the crop to another product. Unknown families stay unknown, and
    so does any family whose gallery is missing one of its parts (``part_source``
    ``no_gallery`` / ``gallery_gap``) rather than guessing the part the gallery knows.
    """

    def __init__(self, embedding, vlm, family_threshold=0.5, **kwargs):
        """``embedding`` is an :class:`EmbeddingIdentifier` (or its config), ``vlm`` a VLM identifier."""
        from freestyle.vlm_identify import VLMCropIdentifier

        self.embedding = embedding if hasattr(embedding, 'restricted') \
            else EmbeddingIdentifier(**embedding)
        self.vlm = vlm if hasattr(vlm, 'identify_batch') and hasattr(vlm, '_families') else \
            VLMCropIdentifier(mode='two_stage', **vlm)
        self.family_threshold = float(family_threshold)
        self.gallery_groups = self.embedding.gallery_groups
        self.families = {family: [cls for cls in entry['classes'].values() if cls != 'unknown']
                        for family, entry in self.vlm._families().items()}

    def predict(self, image, detections):
        crops = [crop(image, d['box']) for d in detections]
        if not crops:
            return []
        shelves = [d.get('shelf') if d.get('shelf') in ('top', 'bottom') else None
                   for d in detections]
        families = self.vlm.identify_batch(crops, shelves)
        embeddings = self.embedding.embed(image, detections)
        results = []
        for crop_index, (family_hit, embedding) in enumerate(zip(families, embeddings)):
            family = str(family_hit.get('family', 'unknown'))
            confidence = float(family_hit.get('confidence', 0.0))
            base = {'family': family, 'part': family_hit.get('part'),
                    'family_confidence': confidence, 'rationale': family_hit.get('rationale', '')}
            classes = self.families.get(family, [])
            if not classes or confidence < self.family_threshold:
                results.append({**base, 'class': 'unknown', 'id_score': 0.0, 'part_source': 'vlm'})
                continue
            if len(classes) == 1:
                results.append({**base, 'class': classes[0], 'id_score': confidence,
                                'part_source': 'single'})
                continue
            vlm_class = str(family_hit.get('class') or 'unknown')
            if vlm_class in classes:      # the VLM read the part numeral itself
                results.append({**base, 'class': vlm_class, 'id_score': confidence,
                                'part_source': 'vlm'})
                continue
            # Part unreadable: let the gallery decide, but only when it actually holds
            # crops for every part of this family. Otherwise kNN would silently collapse
            # the choice onto the one part it knows (e.g. coca-cola-part2 alone), which is
            # the same confident-guessing failure the kNN-only identifier had.
            classifier = self.embedding.restricted(classes)
            covered = set(map(str, getattr(classifier, 'classes_', []))) if classifier is not None else set()
            if classifier is None or not set(classes) <= covered:
                results.append({**base, 'class': 'unknown', 'id_score': 0.0,
                                'part_source': 'no_gallery' if classifier is None else 'gallery_gap'})
                continue
            label, score = classifier.predict_with_scores(embedding[None], unknown_thr=None)
            results.append({**base, 'class': str(label[0]), 'part': None,
                            'id_score': min(confidence, float(score[0])),
                            'part_confidence': float(score[0]), 'part_source': 'knn'})
        return results


class Pipeline:
    """Locate, identify, deduplicate and count cartridges in one original-pixel image.

    Counting knobs (all config-settable): ``max_rows`` caps shelf rows so a two-high
    stack stays a tier, ``suppress_*`` controls the final class-agnostic duplicate
    suppression (including the looser same-row / same-column threshold), ``shelf_hint``
    lets the
    identifier see a top/bottom hint before the final deduplication, and ``pair_rule``
    reports or applies the shelf-order part rule.
    """

    def __init__(self, localizer, identifier=None, score_thr=0.0, identity_thr=0.5,
                 neighbour_vote='off', max_rows=2, suppress_iou=None, suppress_containment=None,
                 suppress_row_iou=None, suppress_row_x_overlap=0.5, shelf_hint=False, pair_rule='off',
                 pair_family_threshold=0.5, pair_part_threshold=0.5):
        # YAML 1.1 readers turn an unquoted ``off`` into False; accept that spelling.
        neighbour_vote = 'off' if neighbour_vote is False else neighbour_vote
        pair_rule = 'off' if pair_rule is False else pair_rule
        if neighbour_vote not in ('off', 'report', 'apply') or pair_rule not in ('off', 'report', 'apply'):
            raise ValueError('neighbour_vote and pair_rule must be off, report, or apply')
        self.localizer, self.identifier = localizer, identifier
        self.score_thr, self.identity_thr = score_thr, identity_thr
        self.neighbour_vote, self.max_rows = neighbour_vote, max_rows
        self.suppression = {'iou': suppress_iou, 'containment': suppress_containment,
                            'row_iou': suppress_row_iou, 'row_x_overlap': suppress_row_x_overlap}
        self.shelf_hint = bool(shelf_hint)
        self.pair_rule = pair_rule
        self.pair_thresholds = {'family_threshold': pair_family_threshold,
                                'part_threshold': pair_part_threshold}

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
        identifier = build_identifier(config.get('identifier', {'type': 'none'}))
        return cls(localizer, identifier, **config.get('counting', {}))

    def _shelf_labels(self, detections):
        """Attach a catalog top/bottom shelf hint to each detection, or None.

        Identifiers read the hint from ``shelf`` (the same key the finished result
        uses for the numeric row), which is overwritten by the final row assignment.
        """
        rows = {d['shelf_idx'] for d in detections}
        highest, lowest = (min(rows), max(rows)) if rows else (None, None)
        for d in detections:
            label = None if not self.shelf_hint or len(rows) < 2 else \
                ('top' if d['shelf_idx'] == highest else 'bottom' if d['shelf_idx'] == lowest else None)
            d['shelf_hint'] = label
            if label is not None:
                d['shelf'] = label

    def run(self, pil_image, image_id, group=None):
        gallery = getattr(self.identifier, "gallery_groups", None)
        if gallery is not None:
            if group is None:
                raise ValueError("Query group provenance is required")
            if str(group) in gallery:
                raise ValueError("Query group overlaps gallery groups")
        timings = {}
        start = time.perf_counter()
        if isinstance(self.localizer, ProposalLocalizer):
            detections = self.localizer.predict(pil_image, image_id)
        else:
            detections = self.localizer.predict(pil_image)
        detections = [dict(d) for d in detections if d['score'] >= self.score_thr]
        timings['localize'] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        duplicates = 0
        if any(self.suppression[key] is not None
               for key in ('iou', 'containment', 'row_iou')) and detections:
            # Row context first: the class-agnostic deduplication uses it, then the
            # survivors are re-assigned so columns and tiers describe the final set.
            rows = [d['shelf_idx'] for d in assign_shelves(detections, self.max_rows)]
            kept = suppress_duplicates(detections, rows=rows, **self.suppression)
            duplicates = len(detections) - len(kept)
            detections = kept
        if detections:
            detections = assign_shelves(detections, self.max_rows)
            self._shelf_labels(detections)
        timings['deduplicate'] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        if self.identifier is not None:
            identities = self.identifier.predict(pil_image, detections)
            if len(identities) != len(detections):
                raise ValueError('identifier must return one identity per detection')
            for d, identity in zip(detections, identities):
                d.update(identity)
        for d in detections:
            if self.identifier is None and not d.get('class') and 'category_id' in d:
                names = getattr(getattr(self.localizer, 'backend', self.localizer), 'class_names', None)
                idx = d['category_id']
                if names is None:
                    raise ValueError('Detector class names are required')
                try:
                    d['class'] = names[idx] if not isinstance(names, dict) else names.get(idx, names.get(str(idx)))
                except (IndexError, KeyError):
                    raise ValueError(f'Unknown detector class index: {idx}') from None
                if not d['class']:
                    raise ValueError(f'Unknown detector class index: {idx}')
                d['id_score'] = d['score']
            d['class'] = d.get('class') or 'unknown'
            d['id_score'] = float(d.get('id_score', 0))
            d['proposed_class'], d['proposed_score'] = d['class'], d['id_score']
            if d['id_score'] < self.identity_thr or d['class'] == 'cartridge':
                d['class'] = 'unknown'
        timings['identify'] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
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
                detections[flag['detection']]['id_score'] = flag['suggested_score']
        if self.pair_rule != 'off' and detections:
            annotated = infer_pairs(detections, mode=self.pair_rule, **self.pair_thresholds)
            for index, (before, after) in enumerate(zip(detections, annotated)):
                if after['run_id'] is None:
                    continue
                flags.append({'type': 'pair_rule', 'detection': index, 'run_id': after['run_id'],
                              'reason': after['rule_reason'], 'flags': after['flags'],
                              **({'suggested_class': after['class']} if after['class'] != before['class'] else {})})
                if self.pair_rule == 'apply':
                    detections[index].update({k: v for k, v in after.items() if k in
                                              ('class', 'part_inferred', 'rule_reason', 'run_id')})
            counts, unknown, _ = aggregate_counts(detections, self.score_thr, self.identity_thr, 'off')
        timings['count'] = (time.perf_counter() - start) * 1000
        for d in detections:
            d['shelf'] = d.pop('shelf_idx')
            d['column'] = d.pop('column_idx')
        return {'detections': detections, 'counts': counts, 'unknown': unknown,
                'duplicates_removed': duplicates, 'rows': len({d['shelf'] for d in detections}),
                'flags': flags, 'timings_ms': timings}


def build_identifier(spec):
    """Instantiate an identifier from a config block (``type`` plus its options)."""
    ident = dict(spec or {'type': 'none'})
    kind = ident.pop('type')
    if kind in ('embedding_knn', 'embedding_prototype'):
        if kind == 'embedding_prototype':
            ident.setdefault('classifier', 'prototype')
        return EmbeddingIdentifier(**ident)
    if kind == 'none':
        return None
    if kind == 'vlm_crop':
        from freestyle.vlm_ground import CropIdentifier
        return CropIdentifier(**ident)
    if kind == 'hybrid_vlm_knn':
        return HybridIdentifier(**ident)
    raise ValueError(f'unknown identifier: {kind}')


def run_images(pipeline, images, out, categories):
    """Write outputs for iterable (COCO image ID, image path) pairs."""
    out = Path(out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'overlays').mkdir(parents=True, exist_ok=True)
    predictions, counts = [], {}
    for entry in images:
        image_id, path = entry[:2]
        group = entry[2] if len(entry) > 2 else None
        with Image.open(path) as source:
            image = source.convert('RGB')
        result = pipeline.run(image, image_id, group=group) if group is not None else pipeline.run(image, image_id)
        name = str(image_id).replace('/', '_').replace('\\', '_')
        (out / 'images' / f'{name}.json').write_text(json.dumps(result, indent=2))
        counts[str(image_id)] = {'counts': result['counts'], 'unknown': result['unknown']}
        draw = ImageDraw.Draw(image)
        for d in result['detections']:
            if d['class'] not in categories:
                raise ValueError(f"Missing COCO category ID for {d['class']!r}")
            predictions.append({'image_id': image_id, 'bbox': xyxy_to_xywh(d['box']).tolist(),
                                'score': float(d['score']), 'category_id': categories[d['class']],
                                'id_score': d['id_score'], 'proposed_class': d['proposed_class'],
                                'proposed_score': d['proposed_score'],
                                **({'alt_class': d['alt_class']} if 'alt_class' in d else {})})
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
    pipeline = Pipeline.from_config(args.config)
    if getattr(pipeline.identifier, 'gallery_groups', None) is not None:
        from freestyle.identify.cli import _resolve_groups_and_splits
        split_data = json.loads(Path(config['split_map']).read_text())
        groups, _ = _resolve_groups_and_splits(paths, [str(ids[p]) for p in paths], split_data)
        entries = [(ids[p], root / p, group) for p, group in zip(paths, groups)]
    else:
        entries = [(ids[p], root / p) for p in paths]
    run_images(pipeline, entries,
               args.out, config['categories'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
