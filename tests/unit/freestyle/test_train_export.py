import json

import numpy as np
from PIL import Image

from freestyle.train.export_tiles import export_tiles


def test_export_tiles(tmp_path):
    pixels = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    Image.fromarray(pixels).save(tmp_path / "frame.png")
    coco = {"images": [{"id": 42, "file_name": "frame.png", "width": 12, "height": 8}],
            "categories": [{"id": 3, "name": "foo"}],
            "annotations": [{"id": 1, "image_id": 42, "category_id": 3, "bbox": [3, 1, 4, 4]}]}
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    (tmp_path / "splits.json").write_text(json.dumps({"images": {"frame.png": "train"}}))
    out = tmp_path / "export"
    manifest = export_tiles(tmp_path / "coco.json", tmp_path, out, tmp_path / "splits.json",
                            tile=8, overlap=0, min_visible=0.6, class_agnostic=True)
    assert len(manifest) == 2
    assert all(m["source_image_id"] == 42 for m in manifest)
    assert json.loads((out / "manifest.json").read_text()) == manifest
    ds = json.loads((out / "train/_annotations.coco.json").read_text())
    assert ds["categories"] == [{"id": 1, "name": "cartridge"}]
    assert [a["bbox"] for a in ds["annotations"]] == [[3, 1, 4, 4], [0, 1, 3, 4]]
    assert np.array_equal(np.asarray(Image.open(out / "train" / manifest[1]["file_name"])),
                          pixels[:, 4:12])
    export_tiles(tmp_path / "coco.json", tmp_path, tmp_path / "strict", tmp_path / "splits.json",
                 tile=8, overlap=0, min_visible=0.8)
    strict = json.loads((tmp_path / "strict/train/_annotations.coco.json").read_text())
    assert len(strict["annotations"]) == 1


def test_small_visible_piece_masked_in_validation(tmp_path):
    pixels = np.full((10, 20, 3), 100, dtype=np.uint8)
    pixels[2:8, 8:18] = 200
    Image.fromarray(pixels).save(tmp_path / 'frame.png')
    coco = {'images': [{'id': 1, 'file_name': 'frame.png', 'width': 20, 'height': 10}],
            'categories': [{'id': 1, 'name': 'cola'}],
            'annotations': [{'id': 5, 'image_id': 1, 'category_id': 1, 'bbox': [8, 2, 10, 6]}]}
    (tmp_path / 'coco.json').write_text(json.dumps(coco))
    (tmp_path / 'splits.json').write_text(json.dumps({'images': {'frame.png': 'val'}}))
    out = tmp_path / 'out'
    manifest = export_tiles(tmp_path / 'coco.json', tmp_path, out, tmp_path / 'splits.json',
                            tile=10, overlap=0, min_visible=.6)
    assert manifest[0]['masked_instances'][0]['annotation_id'] == 5
    tile = np.asarray(Image.open(out / 'valid' / manifest[0]['file_name']))
    assert not np.any(tile[2:8, 8:10] == 200)
    ds = json.loads((out / 'valid/_annotations.coco.json').read_text())
    assert len(ds['annotations']) == 1 and ds['annotations'][0]['bbox'] == [0, 2, 8, 6]
