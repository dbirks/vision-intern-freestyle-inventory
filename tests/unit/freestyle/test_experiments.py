import pytest

from freestyle.experiments import entry, launch, schema


def config():
    return {'id': 'example', 'family': 'C_det_agnostic_plus_embed',
            'compute': {'target': 'k8s:5090'}, 'localizer': {'type': 'tiled_detector'}}


def test_defaults_and_hash():
    c = schema.resolve(config())
    assert c['data']['label_source'] == 'gt'
    assert c['eval']['score_thr'] == 'val'
    assert schema.content_hash(c, 'abc') == schema.content_hash(config(), 'abc')
    assert schema.content_hash(c, 'abc') != schema.content_hash(c, 'def')
    with pytest.raises(ValueError):
        schema.resolve({**config(), 'family': 'wrong'})


def test_immutable(tmp_path):
    c = schema.resolve(config())
    schema.create_run_dir(c, 'sha', tmp_path)
    with pytest.raises(FileExistsError):
        schema.create_run_dir(c, 'sha', tmp_path)


def test_submit():
    cmd = launch.submit_argv(schema.resolve(config()), 'configs/x.yaml', 'example-abcd', True)
    assert cmd[:3] == ['k8s/submit.sh', 'example-abcd', '5090']
    assert '--dry-run' in cmd and 'rfdetr==1.12.1' in cmd[cmd.index('--pip') + 1]
    assert cmd[cmd.index('--data-root') + 1] == '/work/data'


def test_resume(tmp_path):
    called = []
    stages = [('first', lambda: called.append('first')), ('second', lambda: called.append('second'))]
    entry.run(schema.resolve(config()), tmp_path, tmp_path / 'out', stages)
    entry.run(schema.resolve(config()), tmp_path, tmp_path / 'out', stages)
    assert called == ['first', 'second']


def test_bootstrap_filter_and_train_only(tmp_path):
    import json

    def write(name, obj):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    cats = [{'id': 1, 'name': 'drink'}]
    images = [{'id': n, 'file_name': f'{name}.jpg'} for n, name in enumerate(('train', 'val', 'test'), 1)]
    gt = {'categories': cats, 'images': images, 'annotations': [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 2, 2]}]}
    pseudo = {'categories': cats, 'images': [*images, {'id': 4, 'file_name': 'train2.jpg'}],
              'annotations': [{'id': n, 'image_id': n, 'category_id': 1, 'bbox': [0, 0, 2, 2],
                               'attributes': {'judge_verdict': 'correct', 'judge_score': score,
                                              'support': .8, 'id_confidence': .9}}
                              for n, score in [(1, .9), (2, .9), (3, .9), (4, .2)]]}
    write('gt/train.json', gt)
    write('labels/pseudo.json', pseudo)
    write('splits.json', {'images': {'train.jpg': 'train', 'train2.jpg': 'train',
                                     'val.jpg': 'val', 'test.jpg': 'test'}})
    base = config()
    base['data'] = {'label_source': 'bootstrap', 'bootstrap': 'labels/pseudo.json',
                    'bootstrap_filter': {'min_judge_score': .5}}
    c = schema.resolve(base)
    result = json.loads(entry.prepare_training(c, tmp_path, tmp_path).read_text())
    assert {i['file_name'] for i in result['images']} == {'train.jpg', 'train2.jpg'}
    assert len(result['annotations']) == 1
    c['data']['label_source'] = 'bootstrap+gt'
    result = json.loads(entry.prepare_training(c, tmp_path, tmp_path).read_text())
    assert len(result['annotations']) == 1
    assert result['annotations'][0]['id'] == 1
    assert {i['file_name'] for i in result['images']} == {'train.jpg', 'train2.jpg'}


def test_dirty_hash_changes():
    c = config()
    clean = {'dirty': False, 'diff_hash': None}
    dirty = {'dirty': True, 'diff_hash': 'a' * 64}
    assert schema.content_hash(c, 'sha', clean) != schema.content_hash(c, 'sha', dirty)
    assert schema.content_hash(c, 'sha', dirty) != schema.content_hash(
        c, 'sha', {'dirty': True, 'diff_hash': 'b' * 64})


def test_entry_family_c_end_to_end(tmp_path, monkeypatch):
    import json

    from PIL import Image
    import freestyle.identify.cli as identify_cli
    import freestyle.infer.detector as detector
    import freestyle.pipeline as pipeline
    import freestyle.train.export_tiles as exporter
    import freestyle.train.train_detector as trainer

    root = tmp_path / 'fixture'
    out = tmp_path / 'out'
    root.mkdir()
    images_root = root / 'images'
    categories = [{'id': 1, 'name': 'drink'}, {'id': 2, 'name': 'unknown'}]
    splits = {'images': {}, 'groups': {}}
    for split, groups in [('train', ['g1', 'g2']), ('val', ['g3']), ('test', ['g4'])]:
        records, anns = [], []
        for idx, group in enumerate(groups, 1):
            name = f'{group}/image.png'
            path = images_root / name
            path.parent.mkdir(parents=True)
            Image.new('RGB', (30, 30)).save(path)
            splits['groups'][group] = split
            splits['images'][name] = split
            records.append({'id': idx, 'file_name': name, 'width': 30, 'height': 30})
            anns.append({'id': idx, 'image_id': idx, 'category_id': 1, 'bbox': [2, 2, 10, 10]})
        (root / f'{split}.json').write_text(json.dumps({'images': records, 'annotations': anns,
                                                        'categories': categories}))
    (root / 'splits.json').write_text(json.dumps(splits))
    events = []

    def fake_export(coco, images_root, out, split_map, **kwargs):
        exported = json.loads(open(coco).read())
        assert {i['file_name'] for i in exported['images']} == {'g1/image.png', 'g2/image.png'}
        assert set(json.loads(open(split_map).read())['images'].values()) == {'train', 'val'}
        events.append('export')

    def fake_train(c, out):
        events.append('train')

    def fake_gallery(args):
        assert len(json.loads(open(args[args.index('--coco') + 1]).read())['images']) == 2
        events.append('gallery')

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def predict(self, image):
            events.append('infer')
            return [{'box': [2, 2, 12, 12], 'score': .9}]

    class FakeIdentifier:
        def __init__(self, **kwargs):
            pass

        def predict(self, image, detections):
            events.append('identify')
            return [{'class': 'drink', 'id_score': .9} for d in detections]

    monkeypatch.setattr(exporter, 'export_tiles', fake_export)
    monkeypatch.setattr(trainer, 'train', fake_train)
    monkeypatch.setattr(identify_cli, 'main', fake_gallery)
    monkeypatch.setattr(detector, 'TiledDetector', FakeDetector)
    monkeypatch.setattr(pipeline, 'EmbeddingIdentifier', FakeIdentifier)
    c = schema.resolve({**config(), 'data': {'train': 'train.json', 'val': 'val.json',
                                             'test': 'test.json', 'images_root': 'images'},
                        'localizer': {'type': 'tiled_detector', 'training': {}},
                        'identifier': {'type': 'embedding_knn', 'model': 'fake'}})
    state = {'dirty': True, 'diff_hash': 'abc'}
    entry.run(c, root, out, git_sha='sha', git_state=state)
    assert events == ['export', 'train', 'gallery', 'infer', 'identify', 'infer', 'identify']
    for split in ('val', 'test'):
        pred = json.loads((out / split / 'predictions.json').read_text())
        counts = json.loads((out / split / 'counts.json').read_text())
        assert pred[0]['category_id'] == 1 and pred[0]['bbox'] == [2, 2, 10, 10]
        assert next(iter(counts.values()))['counts'] == {'drink': 1}
    meta = json.loads((out / 'meta.json').read_text())
    assert meta['git_dirty'] is True and meta['git_diff_hash'] == 'abc'


def test_shelf_reports_gaps_and_votes():
    from PIL import Image
    from freestyle.pipeline import Pipeline

    class Localizer:
        def predict(self, image):
            return [{'box': [x, 0, x + 10, 10], 'score': .9} for x in (0, 30, 70)]

    class Identifier:
        def predict(self, image, detections):
            return [{'class': 'drink', 'id_score': score} for score in (.9, .1, .9)]

    c = schema.resolve({**config(), 'family': 'E_shelf'})
    result = Pipeline(Localizer(), Identifier(), **c['count']).run(Image.new('RGB', (100, 30)), 1)
    assert result['counts'] == {'drink': 2} and result['unknown'] == 1
    assert {'gap', 'neighbour_vote'} <= {flag['type'] for flag in result['flags']}
    with pytest.raises(ValueError):
        schema.resolve({**config(), 'family': 'E_shelf', 'count': {'neighbour_vote': 'apply'}})
    assert 'rfdetr==1.12.1' in launch.submit_argv(c, 'x.yaml', 'e')[8]


def test_git_state_hashes_untracked_and_diffs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'new.txt').write_text('first')

    def fake_git(args):
        if 'ls-files' in args:
            return b'new.txt\0'
        return b'staged' if '--cached' in args else b''

    monkeypatch.setattr(schema.subprocess, 'check_output', fake_git)
    first = schema.git_state()
    (tmp_path / 'new.txt').write_text('second')
    second = schema.git_state()
    assert first['dirty'] and second['dirty']
    assert first['diff_hash'] != second['diff_hash']
