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
