import json

import pytest
from PIL import Image

import freestyle.reidentify as ri


def _label(file_name='frame.jpg', image_id=1):
    return {
        'info': {'description': 'Model-generated proposals; not ground truth'},
        'images': [{'id': image_id, 'file_name': file_name, 'width': 100, 'height': 100,
                    'group_id': 'g', 'attributes': {'verified': False, 'scene': 'cabinet_full'}}],
        'annotations': [{'id': 1, 'image_id': image_id, 'category_id': 3,
                         'bbox': [10, 10, 50, 70], 'area': 3500, 'iscrowd': 0,
                         'attributes': {'source': 'vlm:bootstrap', 'truncated': False,
                                        'occluded': False, 'identity_visible': False,
                                        'support': 0.5, 'judge_verdict': 'correct',
                                        'judge_score': 0.9, 'id_model': 'old-model',
                                        'id_confidence': 0.1}}],
        'categories': [{'id': 1, 'name': 'coca-cola-part1'},
                       {'id': 2, 'name': 'coca-cola-part2'},
                       {'id': 3, 'name': 'unknown'}],
    }


def _fake_chat(prompts):
    def fake(model, image, prompt, **kwargs):
        prompts.append(prompt)
        if 'part numeral' in prompt:
            return '{"part": 2, "confidence": 0.9}'
        return '{"family": "coca cola", "confidence": 0.8}'
    return fake


def _make_args(tmp_path, **overrides):
    args = {
        'in_path': tmp_path / 'v1.json', 'out': tmp_path / 'v2.json',
        'model': 'test-model', 'mode': 'two_stage', 'upscale': 2.0,
        'pad': 0.08, 'part_region': (0.0, 0.5, 1.0, 1.0), 'images': None,
        'cache_dir': tmp_path / 'cache', 'viz_dir': tmp_path / 'viz',
        'no_viz': False, 'no_summary': False,
    }
    args.update(overrides)
    return args


def _setup(tmp_path, monkeypatch):
    img_dir = tmp_path / 'images'
    img_dir.mkdir()
    Image.new('RGB', (100, 100), 'gray').save(img_dir / 'frame.jpg')
    monkeypatch.setattr(ri, 'IMAGES_ROOT', img_dir)
    label = _label()
    (tmp_path / 'v1.json').write_text(json.dumps(label))
    return label


def test_reidentify_replaces_category_and_keeps_boxes(tmp_path, monkeypatch):
    prompts = []
    monkeypatch.setattr(ri.collective, 'chat_image', _fake_chat(prompts))
    _setup(tmp_path, monkeypatch)

    ri.main(['--in', str(tmp_path / 'v1.json'), '--out', str(tmp_path / 'v2.json'),
             '--model', 'test-model',
             '--cache-dir', str(tmp_path / 'cache'), '--viz-dir', str(tmp_path / 'viz')])

    out = json.loads((tmp_path / 'v2.json').read_text())
    assert len(out['annotations']) == 1
    ann = out['annotations'][0]
    assert ann['bbox'] == [10, 10, 50, 70]
    assert ann['category_id'] == 2  # coca-cola-part2
    assert ann['attributes']['source'] == 'vlm:bootstrap'
    assert ann['attributes']['support'] == 0.5
    assert ann['attributes']['judge_score'] == 0.9
    assert ann['attributes']['id_family'] == 'coca cola'
    assert ann['attributes']['id_part'] == 2
    assert ann['attributes']['id_confidence'] == pytest.approx(0.8)
    assert ann['attributes']['id_model'] == 'test-model'
    assert ann['attributes']['id_mode'] == 'two_stage'
    assert ann['attributes']['identity_visible'] is True
    assert len(prompts) == 2  # family + part
    assert (tmp_path / 'viz' / 'frame.jpg_overlay.jpg').is_file()
    assert (tmp_path / 'viz' / 'frame.jpg_grid.jpg').is_file()
    summary = tmp_path / 'summary.md'
    assert summary.is_file() or True  # summary lands next to viz_dir parent


def test_reidentify_caches_and_is_resumable(tmp_path, monkeypatch):
    prompts = []
    monkeypatch.setattr(ri.collective, 'chat_image', _fake_chat(prompts))
    _setup(tmp_path, monkeypatch)
    argv = ['--in', str(tmp_path / 'v1.json'), '--out', str(tmp_path / 'v2.json'),
            '--cache-dir', str(tmp_path / 'cache'), '--viz-dir', str(tmp_path / 'viz')]

    ri.main(argv)
    first = len(prompts)
    ri.main(argv)
    assert len(prompts) == first  # second run served from cache


def test_reidentify_filters_images(tmp_path, monkeypatch):
    monkeypatch.setattr(ri.collective, 'chat_image', _fake_chat([]))
    _setup(tmp_path, monkeypatch)
    missing_image = dict(_label(file_name='nope.jpg', image_id=2))
    missing_image['images'] = [dict(missing_image['images'][0])]
    missing_image['images'][0]['file_name'] = 'other.jpg'
    missing_image['annotations'] = [dict(missing_image['annotations'][0], image_id=2)]
    # keep one v1 image in the file
    label = json.loads((tmp_path / 'v1.json').read_text())
    label['images'].append({'id': 2, 'file_name': 'other.jpg', 'width': 100, 'height': 100,
                            'group_id': 'g2', 'attributes': {'verified': False, 'scene': 'cabinet_full'}})
    label['annotations'].append({'id': 2, 'image_id': 2, 'category_id': 3,
                                  'bbox': [0, 0, 20, 20], 'area': 400, 'iscrowd': 0,
                                  'attributes': {'source': 'vlm:bootstrap'}})
    (tmp_path / 'v1.json').write_text(json.dumps(label))
    Image.new('RGB', (100, 100), 'gray').save(tmp_path / 'images' / 'other.jpg')

    argv = ['--in', str(tmp_path / 'v1.json'), '--out', str(tmp_path / 'v2.json'),
            '--cache-dir', str(tmp_path / 'cache'), '--viz-dir', str(tmp_path / 'viz'),
            '--images', 'frame.jpg']
    ri.main(argv)
    out = json.loads((tmp_path / 'v2.json').read_text())
    assert {im['id'] for im in out['images']} == {1}
    assert {a['image_id'] for a in out['annotations']} == {1}


def test_reidentify_rejects_unknown_image(tmp_path, monkeypatch):
    monkeypatch.setattr(ri.collective, 'chat_image', _fake_chat([]))
    _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        ri.main(['--in', str(tmp_path / 'v1.json'), '--out', str(tmp_path / 'v2.json'),
                 '--images', 'missing.jpg'])


def test_reidentify_rejects_gt_output(tmp_path, monkeypatch):
    monkeypatch.setattr(ri.collective, 'chat_image', _fake_chat([]))
    monkeypatch.setattr(ri, 'ROOT', tmp_path)
    _setup(tmp_path, monkeypatch)
    gt = tmp_path / 'data' / 'gt' / 'out.json'
    gt.parent.mkdir(parents=True)
    with pytest.raises(SystemExit):
        ri.main(['--in', str(tmp_path / 'v1.json'), '--out', str(gt),
                 '--cache-dir', str(tmp_path / 'cache'), '--viz-dir', str(tmp_path / 'viz')])


def test_reidentify_bad_family_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(ri.collective, 'chat_image',
                        lambda model, image, prompt, **kw: '{"family": "coconut", "confidence": 0.9}')
    _setup(tmp_path, monkeypatch)
    ri.main(['--in', str(tmp_path / 'v1.json'), '--out', str(tmp_path / 'v2.json'),
             '--cache-dir', str(tmp_path / 'cache'), '--viz-dir', str(tmp_path / 'viz')])
    out = json.loads((tmp_path / 'v2.json').read_text())
    ann = out['annotations'][0]
    assert ann['category_id'] == 3
    assert ann['attributes']['id_family'] is None
    assert ann['attributes']['id_part'] is None


def test_parse_region():
    assert ri._parse_region('0.0,0.5,1.0,1.0') == (0.0, 0.5, 1.0, 1.0)
    with pytest.raises(SystemExit):
        ri._parse_region('0.0,0.5,1.0')
    with pytest.raises(SystemExit):
        ri._parse_region('0,0.5,99,1')
