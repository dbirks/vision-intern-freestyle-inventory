import json

from PIL import Image

from freestyle import vlm_ground as ground


def test_messy_and_coordinates():
    text = '```json\n[{"bbox_2d":[10,20,100,200],"label":"x"}, {"bbox_2d":[1,2'
    assert len(ground.parse_boxes(text)) == 1
    assert ground._convert([100, 200, 900, 800], (200, 100), 'norm1000') == [20, 20, 180, 80]
    assert ground._convert([10, 20, 100, 80], (200, 100), 'pixel') == [10, 20, 100, 80]


def test_tiles_and_agreement(monkeypatch):
    responses = iter(['[{"bbox_2d":[0,0,50,50]}]', '[]', '[{"bbox_2d":[0,0,50,50]}]', '[]'])
    monkeypatch.setattr(ground.collective, 'chat_image', lambda *a, **k: next(responses))
    g = ground.VLMGrounder('mock', coord_mode='pixel', tile=100, overlap=0, n_samples=2)
    result = g.detect(Image.new('RGB', (200, 100)), 'a')
    assert result == [{'xyxy': [0.0, 0.0, 50.0, 50.0], 'score': 1.0, 'label': 'cartridge'}]


def test_coco(tmp_path, monkeypatch):
    monkeypatch.setattr(ground.catalog, 'class_names', lambda: ['unknown'])
    g = ground.VLMGrounder('mock')
    monkeypatch.setattr(g, 'detect', lambda *a: [{'xyxy': [1, 2, 5, 8], 'label': 'cartridge', 'score': 1}])
    image = tmp_path / 'im.png'
    Image.new('RGB', (10, 10)).save(image)
    out = tmp_path / 'labels.json'
    g.label_coco([(1, image)], out)
    coco = json.loads(out.read_text())
    assert coco['annotations'][0]['bbox'] == [1, 2, 4, 6]
    assert coco['annotations'][0]['attributes']['source'] == 'vlm:mock'
    assert coco['categories'][1]['name'] == 'cartridge'


def test_ambiguous_coordinates_require_explicit_mode():
    import pytest
    assert ground._convert([100, 100, 500, 500], (800, 800), 'norm1000') == [80, 80, 400, 400]
    with pytest.raises(ValueError, match='probe'):
        ground.VLMGrounder('mock', coord_mode='auto')
