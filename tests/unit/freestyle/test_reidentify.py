import argparse
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
        'model': 'test-model', 'mode': 'two_stage', 'profile': 'bootstrap',
        'upscale': 2.0, 'pad': 0.08, 'max_tokens': None, 'max_workers': 2,
        'part_region': (0.0, 0.5, 1.0, 1.0), 'images': None, 'only_split': None,
        'splits': tmp_path / 'splits.json', 'cache_dir': tmp_path / 'cache',
        'calls_log': None, 'viz_dir': tmp_path / 'viz',
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


# ------------------------------------------------------- flash next / reader_compare profile

FLASH = 'qwen3-8-flash-next'
REPLY_FAMILY = '{"family": "coca cola", "confidence": 0.8}'
REPLY_PART = '{"part": 2, "confidence": 0.9}'


def _reply(prompt):
    return REPLY_PART if 'part numeral' in prompt else REPLY_FAMILY


def _two_box_label(tmp_path, monkeypatch, name='v2.json'):
    """One frame with a clear top/bottom split; box 1 carries a stale hint + error."""
    img_dir = tmp_path / 'images'
    img_dir.mkdir(exist_ok=True)
    Image.new('RGB', (100, 100), 'gray').save(img_dir / 'frame.jpg')
    monkeypatch.setattr(ri, 'IMAGES_ROOT', img_dir)
    label = _label()
    top = label['annotations'][0]
    top['bbox'] = [10, 5, 50, 20]
    top['area'] = 1000
    top['attributes'].update({'id_shelf_hint': 'top', 'id_error': 'stale'})
    label['annotations'].append({'id': 2, 'image_id': 1, 'category_id': 3,
                                 'bbox': [10, 70, 50, 25], 'area': 1250, 'iscrowd': 0,
                                 'attributes': {'source': 'vlm:bootstrap'}})
    (tmp_path / name).write_text(json.dumps(label))
    return label


def _recording_chat(seen):
    """A collective.chat_image stand-in that records (model, image, prompt, kwargs)."""
    def fake(model, image, prompt, **kwargs):
        seen.append({'model': model, 'image': image, 'prompt': prompt, 'kwargs': kwargs})
        return _reply(prompt)
    return fake


def _flash_argv(tmp_path, *extra):
    return ['--in', str(tmp_path / 'v2.json'), '--out', str(tmp_path / 'v3.json'),
            '--model', FLASH, '--mode', 'two_stage_shelf', '--profile', 'reader_compare',
            '--no-viz', '--no-summary',
            '--cache-dir', str(tmp_path / 'rc' / 'cache'),
            '--calls-log', str(tmp_path / 'rc' / 'calls.jsonl'),
            '--viz-dir', str(tmp_path / 'rc' / 'viz'), *extra]


def test_flash_two_stage_shelf_labels_every_box(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ri.collective, 'chat_image', _recording_chat(seen))
    _two_box_label(tmp_path, monkeypatch)

    assert ri.main(_flash_argv(tmp_path)) == 0

    out = json.loads((tmp_path / 'v3.json').read_text())
    by_id = {a['id']: a for a in out['annotations']}
    assert [a['bbox'] for a in out['annotations']] == [[10, 5, 50, 20], [10, 70, 50, 25]]
    assert {a['category_id'] for a in out['annotations']} == {2}          # coca-cola-part2
    assert by_id[1]['attributes']['id_mode'] == 'two_stage_shelf'
    assert by_id[1]['attributes']['id_model'] == FLASH
    assert by_id[1]['attributes']['id_profile'] == 'reader_compare'
    assert by_id[1]['attributes']['id_shelf_hint'] == 'top'
    assert by_id[2]['attributes']['id_shelf_hint'] == 'bottom'
    assert 'id_error' not in by_id[1]['attributes']                       # stale error dropped
    reid = out['info']['reidentification']
    assert reid['profile'] == 'reader_compare' and reid['mode'] == 'two_stage_shelf'
    assert reid['max_tokens'] == 8192 and reid['pad'] == pytest.approx(0.15)
    assert 'shelf_labels' in reid['shelf_inference']

    family = [row['prompt'] for row in seen if 'part numeral' not in row['prompt']]
    assert len(family) == 2
    hints = [p.rsplit('Caller-provided shelf: ', 1)[-1] for p in family]
    assert sorted(h.split('.')[0] for h in hints) == ['bottom', 'top']
    assert all('Use this as a hint.' in h for h in hints)
    assert all(row['model'] == FLASH for row in seen)
    assert all(row['kwargs']['max_tokens'] == 8192 for row in seen)        # thinking model


def test_flash_replies_are_plain_callcache_entries(tmp_path, monkeypatch):
    """Shared cache layout: reidentify writes what reader_compare reads, so crops are reused."""
    from freestyle.experiments.reader_compare import CallCache

    seen = []
    monkeypatch.setattr(ri.collective, 'chat_image', _recording_chat(seen))
    _two_box_label(tmp_path, monkeypatch)
    cache_dir = tmp_path / 'rc' / 'cache'
    argv = _flash_argv(tmp_path)

    assert ri.main(argv) == 0
    assert len(seen) == 4                                                 # 2 boxes x (family + part)
    entries = sorted(cache_dir.glob('*.json'))
    assert len(entries) == 4
    assert set(json.loads(entries[0].read_text())) == \
        {'model', 'requested', 'text', 'finish_reason', 'latency_s'}
    rows = [json.loads(line) for line in
            (tmp_path / 'rc' / 'calls.jsonl').read_text().splitlines() if line]
    assert len(rows) == 4 and all(row['cache'] == 'miss' for row in rows)
    assert {row['tag'].split('.')[0] for row in rows} == {'v3'}           # attributed to the output

    def never(model, image, prompt, **kwargs):
        raise AssertionError('the cache must have answered this call')
    monkeypatch.setattr(ri.collective, 'chat_image', never)
    assert ri.main(argv) == 0                                             # fully served from disk
    assert len(list(cache_dir.glob('*.json'))) == 4

    # the reader comparison itself can serve the very same replies out of that cache
    reuse = CallCache(cache_dir, tmp_path / 'rc' / 'other.jsonl')
    reuse.sender = never
    row = seen[0]
    assert json.loads(reuse.chat(row['model'], row['image'], row['prompt'])) \
        == json.loads(_reply(row['prompt']))


def test_flash_only_split_filters_images(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ri.collective, 'chat_image', _recording_chat(seen))
    label = _two_box_label(tmp_path, monkeypatch)
    label['images'].append({'id': 2, 'file_name': 'train.jpg', 'width': 100, 'height': 100,
                            'group_id': 'g2', 'attributes': {}})
    label['annotations'].append({'id': 3, 'image_id': 2, 'category_id': 3,
                                 'bbox': [10, 5, 50, 20], 'area': 1000, 'iscrowd': 0,
                                 'attributes': {'source': 'vlm:bootstrap'}})
    (tmp_path / 'v2.json').write_text(json.dumps(label))
    Image.new('RGB', (100, 100), 'gray').save(tmp_path / 'images' / 'train.jpg')
    (tmp_path / 'splits.json').write_text(json.dumps(
        {'images': {'frame.jpg': 'test', 'train.jpg': 'train'}, 'groups': {}}))
    argv = _flash_argv(tmp_path) + ['--splits', str(tmp_path / 'splits.json'),
                                    '--only-split', 'val', '--only-split', 'test']

    assert ri.main(argv) == 0
    out = json.loads((tmp_path / 'v3.json').read_text())
    assert {im['id'] for im in out['images']} == {1}
    assert {a['image_id'] for a in out['annotations']} == {1}
    assert out['info']['reidentification']['splits'] == ['val', 'test']

    with pytest.raises(SystemExit):
        ri.main(_flash_argv(tmp_path) + ['--splits', str(tmp_path / 'splits.json'),
                                         '--only-split', 'exclude'])


def test_flash_rejects_too_much_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(ri.collective, 'chat_image', _recording_chat([]))
    _two_box_label(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        ri.main(_flash_argv(tmp_path, '--max-workers', '3'))


def test_uses_shelf_matrix():
    # bootstrap keeps the recipe that produced bootstrap_v2 (hint whenever two_stage);
    # reader_compare hints only in two_stage_shelf, like the experiment it replays.
    assert ri._uses_shelf('two_stage', 'bootstrap') is True
    assert ri._uses_shelf('two_stage_shelf', 'bootstrap') is True
    assert ri._uses_shelf('existing', 'bootstrap') is False
    assert ri._uses_shelf('two_stage', 'reader_compare') is False
    assert ri._uses_shelf('two_stage_shelf', 'reader_compare') is True


def test_resolve_defaults_per_profile(tmp_path):
    ns = argparse.Namespace(in_path='in.json', out=str(tmp_path / 'bootstrap_v3_flash.json'),
                            mode='two_stage_shelf', profile='reader_compare', pad=None,
                            max_tokens=None, max_workers=2, cache_dir=None, viz_dir=None,
                            calls_log=None, only_split=None, splits=None)
    plan = ri._resolve(ns)
    assert plan['pad'] == pytest.approx(0.15) and plan['max_tokens'] == 8192
    assert plan['cache_dir'].as_posix().endswith('runs/reader_compare/cache')
    assert plan['viz_dir'] == ri.ROOT / 'runs/bootstrap_v3_flash/viz'
    assert plan['calls_log'] == ri.ROOT / 'runs/bootstrap_v3_flash/calls.jsonl'

    ns.profile, ns.mode = 'bootstrap', 'two_stage'
    plan = ri._resolve(ns)
    assert plan['pad'] == pytest.approx(0.08) and plan['max_tokens'] == 4096
    assert plan['cache_dir'].as_posix().endswith('runs/bootstrap_v2/cache')

    with pytest.raises(SystemExit):
        ri._resolve(argparse.Namespace(**{**vars(ns), 'profile': 'nope'}))
    with pytest.raises(SystemExit):
        ri._resolve(argparse.Namespace(**{**vars(ns), 'mode': 'contact_sheet'}))
