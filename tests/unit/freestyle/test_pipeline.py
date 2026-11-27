import json

import numpy as np
from PIL import Image

from freestyle.pipeline import Pipeline, run_images


class Localizer:
    def predict(self, image):
        return [{'box': [5, 5, 20, 25], 'score': .9}]


class Identifier:
    def predict(self, image, detections):
        return [{'class': 'cola', 'id_score': .8} for _ in detections]


def test_pipeline_outputs(tmp_path):
    path = tmp_path / 'input.png'
    Image.new('RGB', (40, 40)).save(path)
    pipeline = Pipeline(Localizer(), Identifier())
    run_images(pipeline, [(7, path)], tmp_path / 'run', {'cola': 4, 'unknown': 5})
    out = tmp_path / 'run'
    result = json.loads((out / 'images/7.json').read_text())
    assert result['counts'] == {'cola': 1}
    assert result['detections'][0]['shelf'] == 0
    predictions = json.loads((out / 'predictions.json').read_text())
    assert predictions[0]['bbox'] == [5, 5, 15, 20]
    assert predictions[0]['image_id'] == 7 and predictions[0]['category_id'] == 4
    assert (out / 'counts.json').exists() and (out / 'overlays/7.png').exists()


def test_configured_proposals_and_cli(tmp_path):
    from freestyle.pipeline import main

    Image.new('RGB', (40, 40)).save(tmp_path / 'input.png')
    (tmp_path / 'manifest.json').write_text(json.dumps({'images': [{'id': 7, 'file_name': 'input.png'}]}))
    (tmp_path / 'proposals.json').write_text(json.dumps([
        {'image_id': 7, 'bbox': [5, 5, 15, 20], 'score': .9, 'category_id': 4, 'id_score': .8}
    ]))
    config = tmp_path / 'config.yaml'
    config.write_text(f'''images_root: {tmp_path}
manifest: {tmp_path / "manifest.json"}
categories: {{cola: 4, unknown: 5}}
localizer:
  type: proposals
  path: {tmp_path / "proposals.json"}
identifier: {{type: none}}
''')
    assert main(['--config', str(config), '--images', 'input.png', '--out', str(tmp_path / 'out')]) == 0
    assert json.loads((tmp_path / 'out/counts.json').read_text())['7']['counts'] == {'cola': 1}


def test_detector_identity_and_rejection():
    class Detector:
        class_names = {1: 'cola', 2: 'cartridge'}
        def predict(self, image):
            return [{'box': [0, 0, 10, 10], 'score': .9, 'category_id': 1},
                    {'box': [15, 0, 25, 10], 'score': .3, 'category_id': 1},
                    {'box': [30, 0, 40, 10], 'score': .9, 'category_id': 2}]
    result = Pipeline(Detector()).run(Image.new('RGB', (50, 20)), 1)
    assert result['counts'] == {'cola': 1}
    assert result['unknown'] == 2
    low = result['detections'][1]
    assert low['class'] == 'unknown' and low['proposed_class'] == 'cola'
    assert low['proposed_score'] == .3


def test_gallery_rejects_non_train_rows(tmp_path):
    import numpy as np
    import pytest
    from freestyle.pipeline import EmbeddingIdentifier
    path = tmp_path / 'gallery.npz'
    np.savez(path, embeddings=[[1., 0.]], labels=['cola'], image_ids=['1'],
             groups=['event'], splits=['test'])
    with pytest.raises(ValueError, match='train'):
        EmbeddingIdentifier(path, model='stub')


class ShelfLocalizer:
    """Curved two-shelf layout; optionally with a shifted cross-tile duplicate."""

    class_names = {1: 'cartridge'}

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, image):
        return [{'box': b, 'score': s, 'category_id': 1} for b, s in self.boxes]


def _shelves(duplicate=False):
    boxes = []
    for base in (180, 380):
        for x in (0, 40, 80, 120):
            bottom = base + .003 * (x - 60) ** 2
            boxes.append(([x, bottom - 80, x + 30, bottom], .9))
    if duplicate:
        boxes.append(([44, 302 + .003 * (40 - 60) ** 2, 74, 382 + .003 * (40 - 60) ** 2], .5))
    return boxes


def test_pipeline_suppresses_cross_tile_duplicates():
    image = Image.new('RGB', (200, 450))
    plain = Pipeline(ShelfLocalizer(_shelves(True)), score_thr=0.0).run(image, 1)
    assert plain['rows'] == 2 and plain['duplicates_removed'] == 0 and len(plain['detections']) == 9
    deduped = Pipeline(ShelfLocalizer(_shelves(True)), score_thr=0.0,
                       suppress_iou=0.4, suppress_containment=0.7).run(image, 1)
    assert deduped['duplicates_removed'] == 1 and len(deduped['detections']) == 8


def test_pipeline_caps_rows_and_keeps_tiers():
    image = Image.new('RGB', (200, 450))
    boxes = _shelves() + [([40, 215, 70, 295], .8)]     # cartridge stacked above a bottom one
    result = Pipeline(ShelfLocalizer(boxes), score_thr=0.0, max_rows=2).run(image, 1)
    assert result['rows'] == 2 and len(result['detections']) == 9
    assert {d['tier'] for d in result['detections']} == {0, 1}
    assert Pipeline(ShelfLocalizer(boxes), score_thr=0.0, max_rows=4).run(image, 1)['rows'] >= 2


def test_shelf_hint_reaches_identifier():
    seen = []

    class HintIdentifier:
        def predict(self, image, detections):
            seen.extend(d.get('shelf') for d in detections)
            return [{'class': 'cola', 'id_score': .9} for _ in detections]

    Pipeline(ShelfLocalizer(_shelves()), HintIdentifier(), score_thr=0.0, max_rows=2,
             shelf_hint=True).run(Image.new('RGB', (200, 450)), 1)
    assert set(seen) == {'top', 'bottom'}


def test_pair_rule_reports_and_applies_runs():
    class PairIdentifier:
        def predict(self, image, detections):
            return [{'class': 'coca-cola-part1', 'id_score': .9, 'family': 'coca cola',
                     'family_confidence': .9, 'part': 1 if i in (0, 4) else None,
                     'part_confidence': .9}
                    for i, _ in enumerate(detections)]

    image = Image.new('RGB', (200, 450))
    reported = Pipeline(ShelfLocalizer(_shelves()), PairIdentifier(), score_thr=0.0, max_rows=2,
                        pair_rule='report').run(image, 1)
    pair_flags = [f for f in reported['flags'] if f['type'] == 'pair_rule']
    assert len(pair_flags) == 8 and all(f['run_id'] for f in pair_flags)
    assert all('suggested_class' not in f for f in pair_flags)
    assert reported['counts'] == {'coca-cola-part1': 8}
    applied = Pipeline(ShelfLocalizer(_shelves()), PairIdentifier(), score_thr=0.0, max_rows=2,
                       pair_rule='apply').run(image, 1)
    assert applied['counts'] == {'coca-cola-part1': 4, 'coca-cola-part2': 4}


def test_prototype_and_hybrid_identifier_configuration(tmp_path):
    import numpy as np
    from freestyle.pipeline import build_identifier
    from freestyle.identify.embed import Embedder

    embeddings = Embedder(name='stub').embed([Image.new('RGB', (40, 80), c)
                                              for c in ((200, 30, 30), (30, 30, 200), (30, 200, 30))])
    path = tmp_path / 'gallery.npz'
    np.savez(path, embeddings=np.stack([embeddings[0], embeddings[1], embeddings[2]] * 3),
             labels=['cola', 'sprite', 'fanta'] * 3, image_ids=[str(i) for i in range(9)],
             groups=[f'g{i % 3}' for i in range(9)], splits=['train'] * 9)
    identifier = build_identifier({'type': 'embedding_prototype', 'gallery': str(path),
                                   'model': 'stub', 'max_per_class': 2})
    assert identifier.classifier_name == 'prototype' and identifier.options['max_per_class'] == 2
    assert sorted(identifier.classifier.classes_) == ['cola', 'fanta', 'sprite']
    balanced = build_identifier({'type': 'embedding_knn', 'gallery': str(path), 'model': 'stub',
                                 'k': 3, 'max_per_class': 2, 'balance': True})
    assert balanced.classifier.class_weights_ is not None
    assert balanced.predict(Image.new('RGB', (40, 80), (30, 30, 200)), [{'box': [4, 8, 36, 72]}])[0]['class'] == 'sprite'


def test_hybrid_identifier_picks_part_inside_vlm_family():
    from freestyle.pipeline import HybridIdentifier

    class StubVLM:
        mode = 'two_stage'

        def __init__(self, family):
            self.family = family

        def _families(self):
            return {'coca cola': {'classes': {1: 'coca-cola-part1', 2: 'coca-cola-part2'},
                                  'descriptions': ['coca-cola-part1', 'coca-cola-part2']},
                    'nns': {'classes': {None: 'nns-sweetener'}, 'descriptions': ['nns-sweetener']}}

        def identify_batch(self, crops, shelves=None):
            return [{'family': self.family, 'confidence': .9, 'part': None, 'class': 'unknown'}
                    for _ in crops]

    class StubEmbedding:
        gallery_groups = {'other-event'}

        def embed(self, image, detections):
            # First crop is closest to the part2 gallery row, second to part1.
            return np.array([[0., 1.], [1., 0.]], dtype=np.float32)

        def restricted(self, classes):
            import numpy as np
            from freestyle.identify.classify import KNNClassifier
            return KNNClassifier(k=1).fit(np.array([[1., 0.], [0., 1.]], dtype=np.float32),
                                         ['coca-cola-part1', 'coca-cola-part2'])

    image = Image.new('RGB', (60, 60))
    dets = [{'box': [0, 0, 20, 40], 'score': .9}, {'box': [20, 0, 40, 40], 'score': .9}]
    result = HybridIdentifier(StubEmbedding(), StubVLM('coca cola')).predict(image, dets)
    assert [r['class'] for r in result] == ['coca-cola-part2', 'coca-cola-part1']
    assert all(r['part_source'] == 'knn' for r in result)
    single = HybridIdentifier(StubEmbedding(), StubVLM('nns')).predict(image, dets)
    assert [r['class'] for r in single] == ['nns-sweetener', 'nns-sweetener']
    weak = HybridIdentifier(StubEmbedding(), StubVLM('unknown')).predict(image, dets)
    assert [r['class'] for r in weak] == ['unknown', 'unknown']


def test_hybrid_identifier_refuses_parts_the_gallery_cannot_choose():
    from freestyle.pipeline import HybridIdentifier
    from freestyle.identify.classify import KNNClassifier

    class StubVLM:
        def __init__(self, part=None):
            self.part = part

        def _families(self):
            return {'coca cola': {'classes': {1: 'coca-cola-part1', 2: 'coca-cola-part2'},
                                  'descriptions': ['coca-cola-part1', 'coca-cola-part2']}}

        def identify_batch(self, crops, shelves=None):
            return [{'family': 'coca cola', 'confidence': .9, 'part': self.part,
                     'class': f'coca-cola-part{self.part}' if self.part else 'unknown'}
                    for _ in crops]

    class Gallery:
        """Embedding identifier whose gallery only knows coca-cola-part2."""

        gallery_groups = frozenset()

        def embed(self, image, detections):
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(detections), 1))

        def restricted(self, classes):
            return KNNClassifier(k=1).fit(np.array([[1.0, 0.0]], dtype=np.float32),
                                          ['coca-cola-part2'])

    image = Image.new('RGB', (40, 40))
    dets = [{'box': [0, 0, 20, 40], 'score': .9}]
    gap = HybridIdentifier(Gallery(), StubVLM()).predict(image, dets)
    assert gap[0]['class'] == 'unknown' and gap[0]['part_source'] == 'gallery_gap'
    read = HybridIdentifier(Gallery(), StubVLM(part=1)).predict(image, dets)
    assert read[0]['class'] == 'coca-cola-part1' and read[0]['part_source'] == 'vlm'


def test_hybrid_bundle_export_and_load_roundtrip(tmp_path, monkeypatch):
    """A hybrid identifier freezes its gallery into the bundle and reloads on CPU."""
    import numpy as np
    from freestyle import artifacts
    from freestyle.identify.embed import Embedder
    from freestyle.pipeline import HybridIdentifier

    run_dir = tmp_path / 'runs' / 'X1'
    run_dir.mkdir(parents=True)
    (run_dir / 'pipeline_test.yaml').write_text(yaml_dump({
        'localizer': {'type': 'proposals', 'path': 'predictions.json'},
        'identifier': {'type': 'none'},
        'counting': {'identity_thr': 0.5},
        'categories': {'coca-cola-part1': 1, 'unknown': 2},
    }))
    (run_dir / 'predictions.json').write_text('[]')
    embeddings = Embedder(name='stub').embed([Image.new('RGB', (30, 60), c)
                                              for c in ((200, 30, 30), (30, 30, 200))])
    np.savez(run_dir / 'gallery.npz', embeddings=np.stack([embeddings[0], embeddings[1]]),
             labels=['coca-cola-part1', 'coca-cola-part2'], image_ids=['1', '2'],
             groups=['event-a', 'event-b'], splits=['train', 'train'])
    (run_dir / 'git_sha').write_text('abc123\n')

    bundle = artifacts.export_bundle(
        run_dir, 'hybrid-v1', artifacts_dir=tmp_path / 'artifacts',
        identifier={'type': 'hybrid_vlm_knn', 'family_threshold': 0.45,
                    'embedding': {'gallery': 'gallery.npz', 'model': 'stub', 'k': 1},
                    'vlm': {'model': 'inkling-test'}},
        counting_override={'max_rows': 2, 'suppress_iou': 0.45},
        notes={'tuned_on': 'val', 'labels': 'bootstrap (not GT)'})
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['provenance']['notes']['tuned_on'] == 'val'
    assert manifest['embedder_id'] == 'stub'
    assert 'gallery.npz' in manifest['files']
    frozen = yaml_load((bundle / 'pipeline.yaml').read_text())
    assert frozen['identifier']['embedding']['gallery'] == 'gallery.npz'
    assert frozen['counting']['max_rows'] == 2

    pipeline = artifacts.load_bundle('hybrid-v1', device='cpu',
                                     artifacts_dir=tmp_path / 'artifacts')
    assert isinstance(pipeline.identifier, HybridIdentifier)
    assert str(pipeline.identifier.embedding.classifier.X_train_.shape[1]) == '168'
    assert pipeline.max_rows == 2 and pipeline.suppression['iou'] == 0.45


def yaml_dump(payload):
    import yaml
    return yaml.safe_dump(payload, sort_keys=False)


def yaml_load(text):
    import yaml
    return yaml.safe_load(text)



def test_yaml_unquoted_off_is_accepted_as_off(tmp_path):
    """YAML 1.1 reads an unquoted ``off`` as False; the pipeline must still run."""
    from freestyle.pipeline import main as pipeline_main

    Image.new('RGB', (40, 40)).save(tmp_path / 'input.png')
    (tmp_path / 'manifest.json').write_text(json.dumps({'images': [{'id': 3, 'file_name': 'input.png'}]}))
    (tmp_path / 'proposals.json').write_text(json.dumps(
        [{'image_id': 3, 'bbox': [5, 5, 15, 20], 'score': .9, 'category_id': 4, 'id_score': .8}]))
    config = tmp_path / 'config.yaml'
    config.write_text(f'''images_root: {tmp_path}
manifest: {tmp_path / "manifest.json"}
categories: {{cola: 4, unknown: 5}}
localizer: {{type: proposals, path: {tmp_path / "proposals.json"}}}
identifier: {{type: none}}
counting:
  neighbour_vote: off
  pair_rule: off
''')
    assert pipeline_main(['--config', str(config), '--images', 'input.png',
                          '--out', str(tmp_path / 'out')]) == 0
    assert json.loads((tmp_path / 'out/counts.json').read_text())['3']['counts'] == {'cola': 1}
