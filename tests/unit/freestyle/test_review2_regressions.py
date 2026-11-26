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


def test_owlv2_exemplar_uses_raw_sigmoid_and_handles_empty(monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace
    from freestyle.propose.backends import Owlv2Backend

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
        def __getitem__(self, index):
            return Tensor(self.value[index])
        def max(self, dim):
            return SimpleNamespace(values=Tensor(self.value.max(axis=dim)))
        def sigmoid(self):
            return Tensor(1 / (1 + np.exp(-self.value)))
        def float(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.value

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=contextlib.nullcontext))
    outputs = SimpleNamespace(logits=Tensor([[[-2.0], [-3.0]]]),
                              target_pred_boxes=Tensor([[[.5, .5, .4, .4], [.5, .5, .4, .4]]]))
    model = SimpleNamespace(device='cpu', image_guided_detection=lambda **kw: outputs)
    backend = Owlv2Backend(score_thr=.1)
    monkeypatch.setattr(backend, '_ensure_loaded', lambda: (model, lambda **kw: {}))
    found = backend.detect(Image.new('RGB', (100, 100)), [], [Image.new('RGB', (5, 5))])
    assert len(found) == 1 and found[0][1] == pytest.approx(1 / (1 + np.exp(2)))
    backend.score_thr = .9
    assert backend.detect(Image.new('RGB', (100, 100)), [], [Image.new('RGB', (5, 5))]) == []


def test_sam3_boxes_cast_and_object_postprocessing(monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace
    from freestyle.propose.backends import Sam3Backend
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=contextlib.nullcontext))
    class Tensor:
        def to(self, **kw):
            return kw
    class Processor:
        def __call__(self, **kwargs):
            return {'input_boxes': Tensor()}
        def post_process_object_detection(self, outputs, threshold, target_sizes):
            return [{'boxes': [], 'scores': []}]
    def predict(**kw):
        assert kw['input_boxes'] == {'device': 'cuda', 'dtype': 'bf16'}
        return object()
    backend = Sam3Backend()
    monkeypatch.setattr(backend, '_ensure_loaded', lambda: (SimpleNamespace(
        device='cuda', dtype='bf16', __call__=predict), Processor()))
    class Model:
        device, dtype = 'cuda', 'bf16'
        __call__ = staticmethod(predict)
    monkeypatch.setattr(backend, '_ensure_loaded', lambda: (Model(), Processor()))
    assert backend.detect(Image.new('RGB', (10, 10)), [], box_exemplars=[[1, 1, 3, 3]]) == []


def test_siglip2_patch_budget():
    from freestyle.identify.embed import _processor_options
    class Siglip2ImageProcessorPil:
        patch_size = 16
        def preprocess(self, **kwargs):
            pass
    options = _processor_options(Siglip2ImageProcessorPil(), (512, 512))
    assert options['max_num_patches'] == 1024
    with pytest.raises(ValueError, match='divisible'):
        _processor_options(Siglip2ImageProcessorPil(), (510, 512))


def test_vlm_label_and_contact_sheet_confidence(monkeypatch):
    from freestyle import vlm_ground
    from freestyle.vlm_identify import VLMCropIdentifier
    monkeypatch.setattr(vlm_ground.catalog, 'class_names', lambda: ['cola', 'unknown'])
    loc = vlm_ground.Localizer('mock', with_identity=True)
    monkeypatch.setattr(loc, 'detect', lambda image: [{'xyxy': [0, 0, 2, 2], 'score': 1, 'label': 'hallucination'}])
    result = loc.predict(Image.new('RGB', (2, 2)))[0]
    assert (result['class'], result['id_score']) == ('unknown', 0)
    ident = VLMCropIdentifier('mock', {'cola': 'Cola'})
    def ask(image, prompt):
        assert 'index, class, and confidence' in prompt
        return '[{"index":0,"class":"cola","confidence":0.8}]'
    monkeypatch.setattr(ident, '_ask', ask)
    assert ident.contact_sheet([Image.new('RGB', (2, 2))])[0]['confidence'] == .8


def test_vlm_tile_cross_class_suppression(monkeypatch):
    from freestyle import vlm_ground
    responses = iter(['[{"bbox_2d":[80,10,100,40],"label":"a"}]',
                      '[{"bbox_2d":[0,10,20,40],"label":"b"}]'])
    monkeypatch.setattr(vlm_ground.collective, 'chat_image', lambda *a, **kw: next(responses))
    loc = vlm_ground.Localizer('mock', coord_mode='pixel', tile=100, overlap=.2,
                               with_identity=True, class_agnostic_iou=.7)
    monkeypatch.setattr(vlm_ground.catalog, 'class_names', lambda: ['a', 'b', 'unknown'])
    found = loc.predict(Image.new('RGB', (180, 100)))
    assert len(found) == 1 and found[0]['alt_class'] == 'b'


def test_unknown_image_opt_out_all_evaluation_paths():
    from freestyle.eval.metrics import compute_counting_metrics, compute_identification_metrics, evaluate_predictions
    from freestyle.eval.failures import analyze
    gt = {'images': [{'id': 1, 'file_name': 'g/a.jpg', 'group_id': 'g'}],
          'categories': [{'id': 1, 'name': 'a'}], 'annotations': []}
    pred = [{'image_id': 'missing', 'category_id': 1, 'bbox': [0, 0, 1, 1], 'score': .01}]
    for fn in (compute_localization_metrics, compute_identification_metrics, compute_counting_metrics):
        with pytest.raises(ValueError, match='unknown image'):
            fn(gt, pred)
        fn(gt, pred, allow_unknown_images=True)
    with pytest.raises(ValueError, match='unknown image'):
        analyze(gt, pred, .5)
    assert analyze(gt, pred, .5, allow_unknown_images=True) == []
    result = evaluate_predictions(gt, pred, n_bootstrap=0, allow_unknown_images=True)
    assert result['n_pred_ignored_unknown_image'] == 1


def test_sliver_polygon_masks_only_unprotected_pixels_for_l_and_rgba(tmp_path):
    import json
    from freestyle.train.export_tiles import export_tiles
    for mode in ('L', 'RGBA'):
        image = Image.new(mode, (20, 10), 100 if mode == 'L' else (100, 100, 100, 255))
        image.save(tmp_path / f'{mode}.png')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        color = 240 if mode == 'L' else (240, 240, 240, 255)
        draw.rectangle((6, 2, 9, 7), fill=color)
        image.save(tmp_path / f'{mode}.png')
        coco = {'images': [{'id': 1, 'file_name': f'{mode}.png', 'width': 20, 'height': 10}],
                'categories': [{'id': 1, 'name': 'x'}], 'annotations': [
                    {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [6, 2, 3, 6]},
                    {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [6, 1, 12, 8],
                     'segmentation': [[6, 2, 17, 2, 17, 4, 6, 4]]}]}
        (tmp_path / 'gt.json').write_text(json.dumps(coco))
        (tmp_path / 'split.json').write_text(json.dumps({'images': {f'{mode}.png': 'val'}}))
        out = tmp_path / f'out-{mode}'
        export_tiles(tmp_path / 'gt.json', tmp_path, out, tmp_path / 'split.json', tile=10, overlap=0)
        tile = np.asarray(Image.open(out / 'valid/00000001.png'))
        assert tile[3, 7, 0] == 240  # protected overlap
        assert tile[3, 9, 0] != 240  # exposed sliver
        assert tile[6, 9, 0] == 240  # outside the sliver polygon


def test_threshold_selection_rejects_unmapped_eval_image(tmp_path):
    import json
    from freestyle.eval.evaluate import main
    val = {'images': [{'id': 1, 'file_name': 'g/val.jpg', 'group_id': 'g'}], 'annotations': [], 'categories': []}
    gt = {'images': [{'id': 2, 'file_name': 'renamed.jpg', 'group_id': 'g'}], 'annotations': [], 'categories': []}
    mapping = {'images': {'g/val.jpg': 'val'}, 'groups': {'g': 'val'}}
    for name, value in [('val', val), ('gt', gt), ('split', mapping), ('pred', [])]:
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    with pytest.raises(SystemExit):
        main(['--gt', str(tmp_path / 'gt.json'), '--pred', str(tmp_path / 'pred.json'),
              '--select-thr-on', str(tmp_path / 'val.json'), str(tmp_path / 'pred.json'),
              '--split-map', str(tmp_path / 'split.json')])


def test_cartridge_gt_excluded_from_identification():
    from freestyle.eval.metrics import compute_identification_metrics
    gt = {'images': [{'id': 1}], 'categories': [{'id': 1, 'name': 'cartridge'}, {'id': 2, 'name': 'cola'}],
          'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10]}]}
    pred = [{'image_id': 1, 'category_id': 2, 'bbox': [0, 0, 10, 10], 'score': 1}]
    result = compute_identification_metrics(gt, pred)
    assert result['n_matched_known_gt'] == 0
