import json

import pytest
from PIL import Image

from freestyle import bootstrap_labels as boot


def test_fusion_support_and_provenance():
    runs = [
        {'run': 'a', 'annotations': [{'bbox': [10, 10, 20, 20]}]},
        {'run': 'b', 'annotations': [{'bbox': [11, 10, 20, 20]}, {'bbox': [70, 70, 10, 10]}]},
        {'run': 'c', 'annotations': []},
    ]
    fused = boot.fuse_runs(runs)
    assert len(fused) == 2
    assert fused[0]['support'] == pytest.approx(2 / 3)
    assert [p['run'] for p in fused[0]['provenance']] == ['a', 'b']
    assert fused[1]['support'] == pytest.approx(1 / 3)


def test_pipeline_resume_and_schema(tmp_path, monkeypatch):
    Image.new('RGB', (100, 100)).save(tmp_path / 'image.png')
    monkeypatch.setattr(boot.catalog, 'class_names', lambda path: ['cola', 'unknown'])
    monkeypatch.setattr(boot.catalog, 'describe', lambda name, *args: name)
    calls = {'label': 0, 'judge': 0, 'id': 0}

    def label(**kwargs):
        calls['label'] += 1
        assert kwargs['classes'] == ['cartridge']
        assert kwargs['image_format'] == 'png' and kwargs['max_dimension'] is None
        return {'images': [{'id': 0}], 'annotations': [{'bbox': [10, 20, 20, 30]}]}

    def judge(**kwargs):
        calls['judge'] += 1
        assert kwargs['judge_mode'] == 'crops'
        path = kwargs['output'].with_name('judged_verdicts.json')
        path.write_text(json.dumps([{'annotation_id': 1, 'verdict': 'correct', 'score': 0.9}]))

    def identify(self, crops):
        calls['id'] += 1
        return [{'class': 'cola', 'confidence': 0.8} for _ in crops]

    monkeypatch.setattr(boot, 'label_dataset', label)
    monkeypatch.setattr(boot, 'judge_labels', judge)
    monkeypatch.setattr(boot.VLMCropIdentifier, 'contact_sheet', identify)
    cfg = {'images_root': str(tmp_path), 'cache_dir': str(tmp_path / 'cache'),
           'audit': str(tmp_path / 'audit.json'), 'primary_labeller': 'a',
           'labellers': [{'name': 'first', 'model': 'a'}, {'name': 'second', 'model': 'b'}],
           'judge': {'model': 'b'}, 'identifier': {'model': 'b'}}
    (tmp_path / 'audit.json').write_text(json.dumps({'images': [
        {'image_id': 'image.png', 'scene_label': 'cabinet_full'}]}))
    records = [{'image_id': 'image.png', 'group_id': 'group'}]
    out = tmp_path / 'out.json'
    first = boot.bootstrap(cfg, records, out)
    second = boot.bootstrap(cfg, records, out)
    assert first == second == json.loads(out.read_text())
    assert calls == {'label': 2, 'judge': 1, 'id': 1}
    assert first['images'][0]['attributes']['scene'] == 'cabinet_full'
    ann = first['annotations'][0]
    assert ann['category_id'] == 1 and ann['bbox'] == [10, 20, 20, 30]
    assert ann['attributes']['source'] == 'vlm:bootstrap'
    assert ann['attributes']['support'] == 1
    assert ann['attributes']['judge_verdict'] == 'correct'
    assert ann['attributes']['id_confidence'] == 0.8


def test_refuses_gt():
    with pytest.raises(ValueError, match='data/gt'):
        boot.bootstrap({}, [], boot.ROOT / 'data/gt/nested/proposals.json')
