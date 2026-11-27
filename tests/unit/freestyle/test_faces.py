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


# --------------------------------------------------------------- faces_v2 cleaning


def _face(index, box, *, support=1.0, verdict='correct', score=0.9, provenance=None,
          category_id=1, sub_label=''):
    """One faces_v1-style annotation from an XYXY box."""
    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    return {'id': index, 'image_id': 1, 'category_id': category_id,
            'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': 0,
            'attributes': {'source': 'vlm:faces', 'support': support,
                           'judge_verdict': verdict, 'judge_score': score,
                           'sub_label': sub_label, 'identity_visible': category_id != 1,
                           'identity_containment': 1.0 if category_id != 1 else 0.0,
                           'provenance': provenance if provenance is not None else
                           [{'run': 'inkling-full', 'boxes': [list(box)]}]}}


def _labels(anns, file_name='cab.jpg', width=2048, height=1536):
    return {'info': {'prompt_variant': 'v3', 'target': faces.FACE_CLASS},
            'images': [{'id': 1, 'file_name': file_name, 'width': width, 'height': height,
                        'attributes': {'image_area': width * height, 'verified': False}}],
            'categories': [{'id': 1, 'name': 'unknown'}, {'id': 2, 'name': 'coca-cola-part1'}],
            'annotations': list(anns)}


def test_parse_part_reads_only_the_two_printed_numerals():
    assert faces.parse_part('1') == 1 and faces.parse_part('2') == 2
    assert faces.parse_part('Part 2') == 2 and faces.parse_part('part1') == 1
    assert faces.parse_part('\u2461') == 2                       # circled two
    for text in ('', None, 'NNS', 'AA', '4', '0', '1/2', 'part 1 and 2'):
        assert faces.parse_part(text) is None, text


def test_x_columns_groups_by_interval_overlap_and_y_gap():
    left, right = [100, 800, 145, 865], [300, 800, 345, 865]
    assert faces.x_columns([left, right], 0.4) == [[0], [1]]     # side by side
    assert faces.x_overlap_ratio(left, right) == 0.0
    near = [120, 835, 165, 900]                                  # fanned neighbour
    assert faces.x_overlap_ratio(left, near) > 0.4
    assert faces.x_columns([left, near], 0.4) == [[0, 1]]        # chained without a guard
    assert faces.x_columns([left, near], 0.4, 60.0) == [[0], [1]]   # y-gap guard
    far_top = [100, 600, 145, 665]                               # the other end of the lean
    assert faces.x_columns([left, far_top], 0.4, 60.0) == [[0, 1]]


def test_lowest_per_column_keeps_the_outward_end():
    top_end = [500, 600, 545, 665]
    out_end = [505, 820, 550, 890]
    anns = [_face(1, top_end), _face(2, out_end)]
    kept, dropped = faces.lowest_per_column([top_end, out_end], anns, 0.4, 60.0)
    assert kept == [1] and dropped == {0: 'leaning-column-duplicate'}
    # the lower face loses only if the evidence for it is weaker
    weak = _face(2, out_end, support=0.5, score=0.2)
    kept, _ = faces.lowest_per_column([top_end, out_end], [_face(1, top_end), weak], 0.4, 60.0)
    assert kept == [1]
    # no y-gap guard given -> a fanned row would chain into one column
    kept, dropped = faces.lowest_per_column([top_end, out_end], anns, 0.4)
    assert len(kept) == 1 and len(dropped) == 1


def test_shelf_bands_and_bottom_band():
    top = [[100 + 100 * i, 150, 140 + 100 * i, 210] for i in range(5)]
    bottom = [[300 + 60 * i, 800 + 20 * i, 345 + 60 * i, 865 + 20 * i] for i in range(4)]
    boxes = top + bottom
    bands, distinct = faces.shelf_bands(boxes)
    assert distinct == [0, 1] and bands[:5] == [0] * 5 and set(bands[5:]) == {1}
    assert faces.bottom_band(boxes, bands, distinct) == 1
    assert faces.bottom_band(top, [0] * 5, [0]) is None          # one band: nothing to collapse
    inverted = [[100, 100, 140, 160], [100, 800, 140, 860]]      # band 1 is the higher one
    assert faces.bottom_band(inverted, [1, 0], [0, 1]) is None


def test_face_shape_flags_aspect_and_thin_strip():
    boxes = [[300, 800, 345, 865],            # normal end face
             [400, 800, 560, 860],            # flat: the long side of a leaning box
             [600, 800, 612, 865],            # thin strip
             [100, 150, 260, 210]]            # wide but on the TOP band -> not flat-flagged
    bands = [1, 1, 1, 0]
    flags = faces.face_shape_flags(boxes, bands, bottom=1)
    assert flags == {1: 'flat-aspect', 2: 'thin-strip'}
    assert faces.face_shape_flags(boxes, bands, bottom=None) == {2: 'thin-strip'}


def test_load_reference_carries_family_and_part(tmp_path):
    coco = {'images': [{'id': 1, 'file_name': 'a/b.jpg'}],
            'categories': [{'id': 1, 'name': 'coca-cola-part1'}],
            'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 30, 40],
                             'attributes': {'id_family': 'coca cola', 'id_part': 1}}]}
    path = tmp_path / 'ref.json'
    path.write_text(json.dumps(coco))
    reference = faces.load_reference(path)
    entry = reference['a/b.jpg']
    assert entry['boxes'] == [[10, 20, 40, 60]] and entry['names'] == ['coca-cola-part1']
    assert entry['families'] == ['coca cola'] and entry['parts'] == [1]
    assert faces.best_containing([12, 22, 38, 58], entry['boxes']) == (0, 1.0)
    assert faces.best_containing([500, 500, 520, 520], entry['boxes'])[0] is None
    assert faces.reference_identity('a/b.jpg', [12, 22, 38, 58], reference) == {
        'class': 'coca-cola-part1', 'family': 'coca cola', 'part': 1, 'containment': 1.0}
    assert faces.reference_identity('a/b.jpg', [900, 900, 940, 960], reference)['class'] == 'unknown'
    assert faces.reference_identity('missing.jpg', [0, 0, 10, 10], reference)['family'] is None


def test_residual_duplicates_reports_pairs_it_will_not_delete():
    tight = [458, 124, 499, 194]
    loose = [457, 120, 515, 303]                               # same end, boxy up
    assert faces.residual_duplicates([tight, loose]) == [(0, 1)]
    assert faces.residual_duplicates([tight, [600, 120, 650, 200]]) == []


def test_face_sub_label_votes_across_runs():
    box = [10, 20, 40, 80]
    provenance = [{'run': 'inkling-full', 'boxes': [box]}, {'run': 'inkling-tiles', 'boxes': [box]}]
    lookup = {('img.jpg', 'inkling-full', faces.box_key(box)): '1',
              ('img.jpg', 'inkling-tiles', faces.box_key(box)): '1'}
    assert faces.face_sub_label(lookup, 'img.jpg', provenance) == '1'
    lookup[('img.jpg', 'inkling-tiles', faces.box_key(box))] = '2'
    assert faces.face_sub_label(lookup, 'img.jpg', provenance) == '1'      # first run wins ties
    assert faces.face_sub_label({}, 'img.jpg', provenance) == ''
    assert faces.face_sub_label(lookup, 'other.jpg', provenance) == ''
    assert faces.face_sub_label(lookup, 'img.jpg', None) == ''


def test_clean_faces_keeps_by_support_or_judge_and_renumbers():
    anns = [_face(1, [0, 100, 40, 160], support=1.0, verdict='correct'),
            _face(2, [100, 100, 140, 160], support=1.0, verdict='incorrect'),
            _face(3, [200, 100, 240, 160], support=0.5, verdict='correct'),
            _face(4, [300, 100, 340, 160], support=0.5, verdict='imprecise'),
            _face(5, [400, 100, 440, 160], support=0.5, verdict='incorrect'),
            _face(6, [500, 100, 540, 160], support=1.0, verdict='imprecise')]
    coco, drops = faces.clean_faces(_labels(anns))
    assert [a['bbox'][:2] for a in coco['annotations']] == [[0, 100], [200, 100], [500, 100]]
    assert [a['id'] for a in coco['annotations']] == [1, 2, 3]
    assert [a['attributes']['clean_keep'] for a in coco['annotations']] == \
        ['agreed-support', 'judge-correct', 'agreed-support']
    assert sorted((drop['reason'], drop['bbox'][:2]) for drop in drops) == \
        [('judge-incorrect', [100, 100]), ('judge-incorrect', [400, 100]),
         ('weak-evidence', [300, 100])]
    assert coco['info']['clean_dropped'] == {'judge-incorrect': 2, 'weak-evidence': 1}
    assert coco['info']['clean_kept'] == {'agreed-support': 2, 'judge-correct': 1}
    assert coco['info']['target'] == faces.FACE_TARGET_SLUG
    assert coco['info']['label_version'] == faces.LABEL_VERSION
    assert coco['images'][0]['file_name'] == 'cab.jpg'           # image list untouched
    assert coco['categories'] == _labels([])['categories']


def test_clean_faces_collapses_leaning_columns_not_the_fanned_row():
    top = [_face(i + 1, [100 + 100 * i, 150, 140 + 100 * i, 210]) for i in range(5)]
    bottom = [_face(6, [300, 800, 345, 865]),                    # two outward ends...
              _face(7, [320, 835, 365, 900]),
              _face(8, [500, 600, 545, 665], support=1.0, sub_label='1'),   # ...and both ends
              _face(9, [505, 820, 550, 890], category_id=2, sub_label='2')]
    reference = {'cab.jpg': {'boxes': [[500, 590, 560, 900]],
                            'names': ['coca-cola-part1'],
                            'families': ['coca cola'], 'parts': [None]}}
    coco, drops = faces.clean_faces(_labels(top + bottom), reference=reference)
    boxes = [a['attributes'] for a in coco['annotations']]
    assert len(coco['annotations']) == 8                          # one face dropped
    assert [drop['reason'] for drop in drops] == ['leaning-column-duplicate']
    assert drops[0]['bbox'] == [500, 600, 45, 65]                 # the far end went, not the near
    kept = coco['annotations'][-1]
    assert kept['attributes']['bottom_shelf'] is True
    assert kept['attributes']['id_part'] == 2                     # read from the face text
    assert kept['attributes']['id_part_source'] == 'face_text'
    assert kept['attributes']['id_family'] == 'coca cola'         # inherited by containment
    assert boxes[-1]['shelf_band'] == 1 and boxes[0]['shelf_band'] == 0
    assert all(not a['bottom_shelf'] for a in boxes[:5])          # top shelf untouched


def test_clean_faces_part_falls_back_to_the_whole_cartridge_box():
    face = _face(1, [505, 820, 550, 890], sub_label='')
    reference = {'cab.jpg': {'boxes': [[500, 590, 560, 900]], 'names': ['coca-cola-part1'],
                            'families': ['coca cola'], 'parts': [1]}}
    coco, _ = faces.clean_faces(_labels([face, _face(2, [0, 800, 60, 900])]),
                               reference=reference)
    attributes = coco['annotations'][0]['attributes']
    assert (attributes['id_part'], attributes['id_part_source']) == (1, 'whole_box')
    assert attributes['id_family'] == 'coca cola'
    other = coco['annotations'][1]['attributes']
    assert (other['id_part'], other['id_family']) == (None, None)   # outside every reference box


def test_clean_faces_rules_are_overridable():
    anns = [_face(1, [0, 100, 40, 160], support=0.5, verdict='correct'),
            _face(2, [100, 100, 140, 160], support=0.5, verdict='imprecise')]
    strict, _ = faces.clean_faces(_labels(anns), rules={'support_judged': 1.0})
    assert [a['id'] for a in strict['annotations']] == []
    loose, _ = faces.clean_faces(_labels(anns), rules={'support_judged': 0.5,
                                                      'judge_keep_verdicts': ('correct',
                                                                              'imprecise')})
    assert [a['attributes']['clean_keep'] for a in loose['annotations']] == \
        ['judge-correct', 'judge-correct']


def _detect_with_sub_labels(image, **kwargs):
    """Stand-in labeller: both runs see the same faces and read 'NNS' / '2' on them."""
    return [{'box': [20, 20, 60, 100], 'label': faces.FACE_CLASS, 'sub_label': 'NNS', 'score': 1.0},
            {'box': [100, 100, 145, 165], 'label': faces.FACE_CLASS, 'sub_label': '2', 'score': 1.0}]


def test_recover_sub_labels_reads_back_what_the_run_cached(tmp_path, monkeypatch):
    cfg, records, reference = _setup_run_with_codes(tmp_path, monkeypatch)
    out = tmp_path / 'faces.json'
    labelled = faces.bootstrap_faces(cfg, records, out, reference=reference)
    assert all(a['attributes']['sub_label'] == '' for a in labelled['annotations'])  # lost in fusion
    lookup = faces.recover_sub_labels(labelled, cfg)
    assert sorted(code for code in lookup.values()) == ['2', '2', 'NNS', 'NNS']    # 2 runs x 2 faces
    coco, _ = faces.clean_faces(labelled, sub_labels=lookup)
    codes = [a['attributes']['sub_label'] for a in coco['annotations']]
    assert codes == ['NNS', '2'] and coco['annotations'][1]['attributes']['id_part'] == 2


def _setup_run_with_codes(tmp_path, monkeypatch):
    Image.new('RGB', (200, 200)).save(tmp_path / 'image.jpg')
    monkeypatch.setattr(faces.catalog, 'class_names', lambda path: ['sprite', 'unknown'])
    monkeypatch.setattr(faces, 'detect_faces', _detect_with_sub_labels)
    monkeypatch.setattr(faces, 'judge_labels', _fake_judge(
        [], lambda call, n: [{'annotation_id': i + 1, 'verdict': 'correct', 'score': 0.9}
                            for i in range(n)]))
    cfg = {'images_root': str(tmp_path), 'cache_dir': str(tmp_path / 'cache'),
           'base_url': 'http://x/v1', 'prompt_variant': 'v3', 'fusion_iou': 0.35,
           'labellers': [{'name': 'inkling-full', 'model': 'a'},
                         {'name': 'inkling-tiles', 'model': 'b', 'tile': 100}],
           'judge': {'model': 'flash'}}
    return cfg, [{'image_id': 'image.jpg', 'group_id': 'g'}], {'image.jpg':
                                                                {'boxes': [], 'names': [],
                                                                 'families': [], 'parts': []}}


def test_select_overlay_images_never_touches_test():
    labels = _labels([])
    labels['images'] += [{'id': index + 2, 'file_name': f't{index}.jpg'} for index in range(3)]
    labels['images'] += [{'id': 6, 'file_name': 'v0.jpg'}, {'id': 7, 'file_name': 'v1.jpg'}]
    splits = {'cab.jpg': 'test', 't0.jpg': 'train', 't1.jpg': 'train', 't2.jpg': 'train',
              'v0.jpg': 'val', 'v1.jpg': 'val'}
    drops = [{'file_name': 't2.jpg', 'reason': 'judge-incorrect'}] * 3
    drops += [{'file_name': 't0.jpg', 'reason': 'judge-incorrect'}]
    drops += [{'file_name': 'v1.jpg', 'reason': 'judge-incorrect'}] * 5
    assert faces.select_overlay_images(labels, drops, splits, per_split=2) == \
        ['t2.jpg', 't0.jpg', 'v1.jpg', 'v0.jpg']
    assert faces.select_overlay_images(labels, drops, splits, per_split=1) == ['t2.jpg', 'v1.jpg']


def test_render_clean_overlays_writes_png_and_sidecar(tmp_path):
    Image.new('RGB', (400, 300), 'white').save(tmp_path / 'cab.jpg')
    face = _face(1, [50, 60, 90, 130], sub_label='1')
    face['attributes']['id_part'] = 1                             # as clean_faces would set it
    dropped = {'file_name': 'cab.jpg', 'bbox': [100.0, 60.0, 40.0, 70.0],
               'reason': 'judge-incorrect', 'support': 1.0, 'judge_verdict': 'incorrect',
               'judge_score': 0.1, 'area': 2800.0}
    paths = faces.render_clean_overlays(_labels([face]), [dropped], ['cab.jpg'],
                                       tmp_path / 'viz', images_root=tmp_path,
                                       reference={'cab.jpg': {'boxes': [[40, 50, 150, 150]],
                                                             'names': ['coca-cola-part1']}})
    png = next(path for path in paths if path.suffix == '.png')
    sidecar = json.loads(png.with_suffix('.json').read_text())
    assert png.parent == tmp_path / 'viz' and png.stat().st_size > 0
    assert sidecar['parts_readable'] == [1]
    assert sidecar['kept'][0]['sub_label'] == '1' and sidecar['kept'][0]['bbox'][:2] == [50, 60]
    assert sidecar['dropped'][0]['reason'] == 'judge-incorrect'
    assert faces.render_clean_overlays(_labels([face]), [], ['missing.jpg'], tmp_path / 'viz2',
                                      images_root=tmp_path) == []


def test_summarize_clean_reports_rules_counts_and_residuals(tmp_path):
    anns = [_face(1, [0, 100, 40, 160], support=1.0, verdict='correct'),
            _face(2, [100, 100, 140, 160], support=1.0, verdict='incorrect'),
            _face(3, [104, 104, 136, 156], support=0.5, verdict='correct'),   # loose duplicate
            _face(4, [100, 100, 140, 160], support=0.5, verdict='imprecise')]
    v1 = _labels(anns)
    coco, drops = faces.clean_faces(v1)
    markdown = faces.summarize_clean(v1, coco, drops,
                                   reference={'cab.jpg': {'boxes': [[0, 100, 140, 160]],
                                                         'names': ['unknown']}},
                                   splits={'cab.jpg': 'train'},
                                   overlays=['cab.png'], images_read=4,
                                   judgement=None)
    assert 'NOT ground truth' in markdown
    assert '4 proposals -> 2 faces over 1 images' in markdown
    assert 'judge-incorrect' in markdown and 'weak-evidence' in markdown
    assert 'one face per leaning column' in markdown
    assert '| `cab.jpg` | train | 4 | 2 | 2 |' in markdown
    assert 'residual duplicates NOT covered by the rules above: 0 pairs' in markdown
    assert 'images read into the worker context (downscaled): 4' in markdown
    assert '`cab.png`' in markdown and '(filled in by the worker' in markdown
    (tmp_path / 'judge.md').write_text('Looked at 4 overlays; the fanned row survives.\n')
    markdown = faces.summarize_clean(v1, coco, drops, reference={}, splits={},
                                    judgement=tmp_path / 'judge.md')
    assert 'fanned row survives' in markdown


def test_clean_cli_writes_labels_drops_and_report(tmp_path, monkeypatch, capsys):
    v1 = _labels([_face(1, [0, 100, 40, 160], support=1.0, verdict='correct'),
                  _face(2, [100, 100, 140, 160], support=1.0, verdict='incorrect')])
    labels = tmp_path / 'faces_v1.json'
    labels.write_text(json.dumps(v1))
    reference = tmp_path / 'bootstrap.json'
    reference.write_text(json.dumps({
        'images': [{'id': 1, 'file_name': 'cab.jpg'}],
        'categories': [{'id': 2, 'name': 'coca-cola-part1'}],
        'annotations': [{'image_id': 1, 'category_id': 2, 'bbox': [0, 100, 140, 60],
                         'attributes': {'id_family': 'coca cola', 'id_part': None}}]}))
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'images': {'cab.jpg': 'train'}}))
    config = tmp_path / 'faces.yaml'
    config.write_text('prompt_variant: v3\ncache_dir: %s\nlabellers: []\n' % (tmp_path / 'cache'))
    argv = ['clean', '--labels', str(labels), '--config', str(config), '--reference',
            str(reference), '--splits', str(splits), '--out', str(tmp_path / 'faces_v2.json'),
            '--out-dir', str(tmp_path / 'run'), '--overlays', '0']
    assert faces.main(argv) == 0
    out = json.loads((tmp_path / 'faces_v2.json').read_text())
    report = (tmp_path / 'run' / 'summary.md').read_text()
    assert [a['bbox'] for a in out['annotations']] == [[0.0, 100.0, 40.0, 60.0]]
    assert out['annotations'][0]['attributes']['id_family'] == 'coca cola'
    assert json.loads((tmp_path / 'run' / 'dropped.json').read_text())[0]['reason'] == \
        'judge-incorrect'
    assert 'faces_v2' in report and 'judge-incorrect' in report
    assert 'wrote' in capsys.readouterr().out
    with pytest.raises(ValueError, match='data/gt'):
        faces.main(argv[:-2] + ['--out', str(faces.ROOT / 'data/gt/faces_v2.json'),
                               '--out-dir', str(tmp_path / 'run2')])
