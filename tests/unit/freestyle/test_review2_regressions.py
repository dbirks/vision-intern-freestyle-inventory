import numpy as np
import pytest
from PIL import Image

from freestyle.eval.metrics import compute_localization_metrics
from freestyle.geometry import merge_tiled
from freestyle.identify.classify import KNNClassifier
from freestyle.identify.cli import _resolve_groups_and_splits
from freestyle.pipeline import EmbeddingIdentifier, Pipeline
from freestyle.propose.backends import _move_inputs


def test_identity_score_is_similarity():
    ident = EmbeddingIdentifier.__new__(EmbeddingIdentifier)
    ident.classifier = KNNClassifier(k=1).fit(np.array([[1., 0.]]), np.array(['cola']))
    ident.embedder = type('Embedder', (), {'embed': lambda self, crops: np.array([[0.05, 0.998749]])})()
    result = ident.predict(Image.new('RGB', (10, 10)), [{'box': [0, 0, 10, 10]}])
    assert result[0]['id_score'] == pytest.approx(0.05, abs=1e-4)


def test_gallery_group_rejected_before_localization():
    ident = type('Identifier', (), {'gallery_groups': {'g'}})()
    pipeline = Pipeline(None, ident)
    with pytest.raises(ValueError, match='overlaps'):
        pipeline.run(Image.new('RGB', (1, 1)), 1, group='g')


def test_negative_frame_preserves_group():
    groups, splits = _resolve_groups_and_splits(['g/negative.jpg'], ['1'],
        {'groups': {'g': 'train'}, 'images': {'g/negative.jpg': 'exclude'}})
    assert (groups, splits) == (['g'], ['exclude'])


def test_path_predictions_and_unknown_images():
    gt = {'images': [{'id': 1, 'file_name': 'g/a.jpg'}], 'categories': [{'id': 1, 'name': 'a'}],
          'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10]}]}
    pred = {'image_id': 'g/a.jpg', 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': .9}
    assert compute_localization_metrics(gt, [pred])['ap50'] == pytest.approx(1)
    with pytest.raises(ValueError, match='unknown images'):
        compute_localization_metrics(gt, [{**pred, 'image_id': 'missing'}])


def test_cross_class_nms_preserves_alternative():
    detections = [{'box': [0, 0, 10, 10], 'score': .9, 'category_id': 'a'},
                  {'box': [0, 0, 10, 10], 'score': .8, 'category_id': 'b'}]
    assert merge_tiled(detections, class_agnostic_iou=.7)[0]['alt_class'] == 'b'


def test_move_inputs_casts_geometry():
    class Tensor:
        def to(self, **kwargs):
            return kwargs
    result = _move_inputs({'input_boxes': Tensor(), 'input_ids': Tensor()}, 'cuda', 'bf16')
    assert result['input_boxes']['dtype'] == 'bf16'
    assert 'dtype' not in result['input_ids']
