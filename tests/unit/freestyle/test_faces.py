"""Unit tests for freestyle.faces (offline; no Collective calls)."""
import json

import pytest
from PIL import Image

from freestyle import faces


def test_prompt_variants_use_normalised_boxes_and_face_label():
    assert sorted(faces.PROMPT_VARIANTS) == ['v1', 'v2', 'v3']
    for variant in faces.PROMPT_VARIANTS:
        prompt = faces.build_prompt(variant)
        assert 'bbox_2d' in prompt
        assert '0-1000' in prompt
        assert faces.FACE_CLASS in prompt
        assert 'end' in prompt.lower()          # what to box is described
    assert faces.build_prompt() == faces.build_prompt(faces.DEFAULT_VARIANT)
    with pytest.raises(ValueError, match='unknown prompt variant'):
        faces.build_prompt('v9')


def test_endpoint_reads_key_from_env_without_storing_it(monkeypatch):
    monkeypatch.setenv('LLMC_API_KEY', 'secret-not-returned')
    base, key = faces.endpoint({})
    assert base == faces.DEFAULT_BASE_URL
    assert key == 'secret-not-returned'
    assert 'secret' not in base
    base, key = faces.endpoint({'api_key_env': 'MISSING_VAR', 'base_url': 'http://x/v1'})
    assert (base, key) == ('http://x/v1', None)


def test_face_containment_and_identity_copy():
    face = [100, 100, 140, 180]
    inside = [80, 80, 300, 400]
    assert faces.face_containment(face, inside) == 1.0
    assert faces.face_containment(face, [100, 100, 110, 180]) == 0.25
    assert faces.face_containment([5, 5, 5, 5], inside) == 0.0     # degenerate face
    assert faces.face_containment(face, [900, 900, 1000, 1000]) == 0.0

    name, score = faces.copy_identity(face, [inside], ['sprite'])
    assert (name, score) == ('sprite', 1.0)
    # half-in -> below the 0.8 threshold -> unknown, but the score is reported
    name, score = faces.copy_identity(face, [[100, 100, 120, 180]], ['sprite'])
    assert name == 'unknown' and score == pytest.approx(0.5)
    # best-matching box wins, even if an earlier one is unknown
    name, _ = faces.copy_identity(face, [[0, 0, 101, 101], inside], ['unknown', 'hi-c'], 0.8)
    assert name == 'hi-c'


def test_neighbour_iou_stats():
    assert faces.neighbour_iou_stats([[0, 0, 10, 10]]) is None
    disjoint = faces.neighbour_iou_stats([[0, 0, 10, 10], [20, 20, 30, 30]])
    assert disjoint == 0.0
    overlapping = faces.neighbour_iou_stats([[0, 0, 10, 10], [5, 0, 15, 10]])
    assert overlapping == pytest.approx(1 / 3)


def test_load_reference(tmp_path):
    coco = {'images': [{'id': 1, 'file_name': 'a/b.jpg'}],
            'categories': [{'id': 1, 'name': 'sprite'}, {'id': 2, 'name': 'unknown'}],
            'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 30, 40]},
                            {'image_id': 1, 'category_id': 2, 'bbox': [0, 0, 5, 5]},
                            {'image_id': 99, 'category_id': 1, 'bbox': [0, 0, 1, 1]}]}
    path = tmp_path / 'ref.json'
    path.write_text(json.dumps(coco))
    reference = faces.load_reference(path)
    assert reference['a/b.jpg']['names'] == ['sprite', 'unknown']
    assert reference['a/b.jpg']['boxes'][0] == [10, 20, 40, 60]


def test_detect_faces_parses_and_filters_degenerate(monkeypatch):
    import tools.vlm_detect as vlm_detect

    reply = json.dumps([
        {'bbox_2d': [0, 0, 250, 500], 'label': 'cartridge front face', 'sub_label': 'NNS'},
        {'bbox_2d': [500, 500, 501, 501], 'label': 'cartridge front face', 'sub_label': ''},
    ])
    monkeypatch.setattr(vlm_detect, 'run_vlm', lambda image, prompt, model_id, **kw: reply)
    detections = faces.detect_faces(Image.new('RGB', (2000, 1000)), prompt='p',
                                   model_id='m', base_url='http://x/v1', api_key='k')
    assert len(detections) == 1                      # degenerate box dropped
    assert detections[0]['box'] == pytest.approx([0.0, 0.0, 500.0, 500.0])
    assert detections[0]['label'] == faces.FACE_CLASS
    assert detections[0]['sub_label'] == 'NNS'


def _fake_judge(store, verdicts_per_call):
    def judge(source, output, annotations, api_key=None, **options):
        coco = json.loads(annotations.read_text())
        call = len(store)
        store.append({'options': options, 'coco': coco})
        (output.with_name('judged_verdicts.json')).write_text(
            json.dumps(verdicts_per_call(call, len(coco['annotations']))))
    return judge


def test_judge_faces_uses_crops_mode_and_falls_back(tmp_path):
    boxes = [[10, 10, 60, 90], [70, 20, 120, 100]]
    calls = []

    def verdicts(call, n):
        if call == 0:                                  # primary judge answers nothing
            return []
        return [{'annotation_id': i + 1, 'verdict': 'correct', 'score': 0.9} for i in range(n)]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(faces, 'judge_labels', _fake_judge(calls, verdicts))
    result = faces.judge_faces(tmp_path, 'x.jpg', boxes,
                               judge={'model': 'flash', 'fallback': 'inkling'},
                               descriptions='d', base_url='http://x/v1', api_key='k',
                               tmp=tmp_path)
    monkey.undo()
    verdicts_map, model, fell_back = result
    assert fell_back and model == 'inkling'
    assert sorted(verdicts_map) == [1, 2]
    assert calls[0]['options']['judge_mode'] == 'crops'
    assert list(calls[0]['options']['class_descriptions']) == [faces.FACE_CLASS.lower()]
    assert calls[0]['coco']['categories'][0]['name'] == faces.FACE_CLASS


def test_judge_faces_keeps_primary_when_it_answers(tmp_path):
    calls = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(faces, 'judge_labels', _fake_judge(
        calls, lambda call, n: [{'annotation_id': i + 1, 'verdict': 'imprecise', 'score': 0.4}
                               for i in range(n)]))
    verdicts_map, model, fell_back = faces.judge_faces(
        tmp_path, 'x.jpg', [[1, 2, 30, 40]], judge={'model': 'flash', 'fallback': 'inkling'},
        descriptions='d', base_url='http://x/v1', api_key='k', tmp=tmp_path)
    monkey.undo()
    assert len(calls) == 1 and not fell_back and model == 'flash'
    assert verdicts_map[1]['verdict'] == 'imprecise'


def _setup_run(tmp_path, monkeypatch, whole_box, detection_boxes):
    Image.new('RGB', (200, 200)).save(tmp_path / 'image.jpg')
    monkeypatch.setattr(faces.catalog, 'class_names', lambda path: ['sprite', 'hi-c', 'unknown'])
    reference = {'image.jpg': {'boxes': [whole_box], 'names': ['sprite']}}

    def detect(image, **kwargs):
        return [{'box': list(b), 'label': faces.FACE_CLASS, 'sub_label': '', 'score': 1.0}
                for b in detection_boxes]
    monkeypatch.setattr(faces, 'detect_faces', detect)
    monkeypatch.setattr(faces, 'judge_labels', _fake_judge(
        [], lambda call, n: [{'annotation_id': i + 1, 'verdict': 'correct', 'score': 0.8}
                             for i in range(n)]))
    cfg = {'images_root': str(tmp_path), 'cache_dir': str(tmp_path / 'cache'),
           'prompt_variant': 'v2', 'fusion_iou': 0.5, 'identity_containment': 0.8,
           'labellers': [{'name': 'inkling-full', 'model': 'a'},
                         {'name': 'inkling-tiles', 'model': 'b', 'tile': 100}],
           'judge': {'model': 'flash', 'fallback': 'inkling'}}
    return cfg, [{'image_id': 'image.jpg', 'group_id': 'g'}], reference


def test_bootstrap_faces_schema_identity_and_resume(tmp_path, monkeypatch):
    cfg, records, reference = _setup_run(
        tmp_path, monkeypatch, whole_box=[10, 10, 120, 190],
        detection_boxes=[[20, 20, 50, 100], [150, 150, 190, 190]])
    out = tmp_path / 'faces.json'
    first = faces.bootstrap_faces(cfg, records, out, reference=reference)
    second = faces.bootstrap_faces(cfg, records, out, reference=reference)
    assert first == second == json.loads(out.read_text())

    assert first['info']['target'] == faces.FACE_CLASS
    assert first['info']['prompt_variant'] == 'v2'
    assert [im['attributes']['target'] for im in first['images']] == [faces.FACE_CLASS]
    assert first['images'][0]['attributes']['verified'] is False
    assert [c['name'] for c in first['categories']] == ['sprite', 'hi-c', 'unknown']

    anns = first['annotations']
    assert [a['bbox'] for a in anns] == [[20, 20, 30, 80], [150, 150, 40, 40]]
    names = {c['id']: c['name'] for c in first['categories']}
    # face 1 lies inside the whole-cartridge box -> identity copied; face 2 does not
    assert [names[a['category_id']] for a in anns] == ['sprite', 'unknown']
    assert anns[0]['attributes']['identity_containment'] == 1.0
    assert anns[1]['attributes']['identity_containment'] == 0.0
    for ann in anns:
        attributes = ann['attributes']
        assert attributes['source'] == 'vlm:faces'
        assert attributes['face_class'] == faces.FACE_CLASS
        assert attributes['judge_verdict'] == 'correct'
        assert attributes['support'] == 1.0            # both labeller runs agree
        assert [p['run'] for p in attributes['provenance']] == ['inkling-full', 'inkling-tiles']
        assert attributes['identity_visible'] == (names[ann['category_id']] != 'unknown')
        assert ann['area'] == pytest.approx(ann['bbox'][2] * ann['bbox'][3])
        assert ann['iscrowd'] == 0


def test_bootstrap_faces_drops_reference_class_missing_from_catalog(tmp_path, monkeypatch):
    cfg, records, reference = _setup_run(
        tmp_path, monkeypatch, whole_box=[10, 10, 120, 190], detection_boxes=[[20, 20, 50, 100]])
    reference['image.jpg']['names'] = ['dr-pepper-part1']       # not in the patched catalog
    result = faces.bootstrap_faces(cfg, records, tmp_path / 'f.json', reference=reference)
    names = {c['id']: c['name'] for c in result['categories']}
    assert names[result['annotations'][0]['category_id']] == 'unknown'


def test_drop_wide_boxes_is_opt_in(tmp_path, monkeypatch):
    cfg, records, reference = _setup_run(
        tmp_path, monkeypatch, whole_box=[10, 10, 190, 190],
        detection_boxes=[[20, 20, 50, 100], [60, 60, 150, 100]])      # 2nd is wider than tall
    kept = faces.bootstrap_faces(cfg, records, tmp_path / 'a.json', reference=reference)
    assert len(kept['annotations']) == 2
    assert [a['attributes']['wider_than_tall'] for a in kept['annotations']] == [False, True]
    cfg['drop_wide_boxes'] = True
    dropped = faces.bootstrap_faces(cfg, records, tmp_path / 'b.json', reference=reference)
    assert [a['bbox'] for a in dropped['annotations']] == [[20, 20, 30, 80]]


def test_bootstrap_faces_refuses_gt():
    with pytest.raises(ValueError, match='data/gt'):
        faces.bootstrap_faces({}, [], faces.ROOT / 'data/gt/nested/faces.json')


def test_render_overlay_downscales(tmp_path):
    image = Image.new('RGB', (3000, 2000), 'white')
    out = faces.render_overlay(image, [[0, 0, 100, 200]], tmp_path / 'ov.png',
                              title='t', sub_labels=['NNS'], whole_boxes=[[5, 5, 400, 400]])
    with Image.open(out) as rendered:
        assert rendered.width <= 1400
    assert out.name == 'ov.png'
    assert faces.overlay_name('a/b/c.jpg', 'v3-full') == 'a__b__c.jpg-v3-full'


def test_summarize_reports_counts_and_overlap(tmp_path):
    labels = {'info': {'prompt_variant': 'v3', 'identity_source': 'ref.json',
                       'identity_containment': 0.8, 'judge_fallbacks': []},
              'images': [{'id': 1, 'file_name': 'image.jpg', 'width': 200, 'height': 200,
                          'attributes': {'image_area': 40000}},
                         {'id': 2, 'file_name': 'empty.jpg', 'width': 200, 'height': 200,
                          'attributes': {'image_area': 40000}}],
              'categories': [{'id': 1, 'name': 'sprite'}, {'id': 2, 'name': 'unknown'}],
              'annotations': [
                  {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [20, 20, 30, 80],
                   'area': 2400, 'attributes': {'identity_visible': True, 'support': 1.0,
                                                'judge_verdict': 'correct',
                                                'judge_model': 'flash'}},
                  {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [150, 150, 40, 20],
                   'area': 800, 'attributes': {'identity_visible': False, 'support': 0.5,
                                               'judge_verdict': 'imprecise',
                                               'judge_model': 'flash'}}]}
    path = tmp_path / 'faces.json'
    path.write_text(json.dumps(labels))
    reference = {'image.jpg': {'boxes': [[10, 10, 120, 190], [100, 100, 200, 200]],
                              'names': ['sprite', 'unknown']},
                 'empty.jpg': {'boxes': [], 'names': []}}
    markdown = faces.summarize(path, reference,
                              timings=[{'duration_s': 2.0}, {'duration_s': 4.0, 'error': '429'}],
                              overlays=['a.png'], images_read=4)
    assert 'NOT ground truth' in markdown
    assert 'faces 0.000 vs whole cartridges' in markdown          # overlap comparison present
    assert '| `image.jpg` | 2 | 2 |' in markdown
    assert '| `empty.jpg` | 0 | 0 |' in markdown
    assert '1/2 identified' in markdown and '1 unknown' in markdown
    assert 'images read into the worker context' in markdown
    assert '`a.png`' in markdown
    assert 'errors 1' in markdown
    assert 'both labeller runs agree (support 1.0) (1 boxes)' in markdown
    assert '| whole cartridges, bootstrap_v2 (2 boxes)' in markdown


def test_summarize_splices_judgement(tmp_path):
    labels = {'info': {'prompt_variant': 'v3', 'identity_source': 'ref.json',
                       'judge_fallbacks': []},
              'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 10, 'height': 10,
                          'attributes': {'image_area': 100}}],
              'categories': [{'id': 1, 'name': 'unknown'}],
              'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 5, 5],
                               'area': 25, 'attributes': {'identity_visible': False,
                                                          'support': 1.0,
                                                          'judge_verdict': 'correct'}}]}
    (tmp_path / 'faces.json').write_text(json.dumps(labels))
    (tmp_path / 'judge.md').write_text('Looked at the overlays. v3 wins.\n')
    markdown = faces.summarize(tmp_path / 'faces.json', {'a.jpg': {'boxes': [], 'names': []}},
                              judgement=tmp_path / 'judge.md')
    assert 'v3 wins' in markdown and '(filled in by the worker' not in markdown
