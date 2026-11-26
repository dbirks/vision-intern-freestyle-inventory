import json

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
