import json

from PIL import Image

from freestyle.infer.detector import TiledDetector, predict_coco


class Fake:
    def __init__(self):
        self.calls = 0

    def predict(self, image, threshold):
        self.calls += 1
        box = [4, 1, 7, 5] if self.calls % 2 else [0, 1, 3, 5]
        return [{"box": box, "score": 0.9, "category_id": 0}]


def test_merge_and_json(tmp_path):
    Image.new("RGB", (12, 8)).save(tmp_path / "image.png")
    data = {"images": [{"id": 7, "file_name": "image.png", "width": 12, "height": 8}],
            "categories": [{"id": 5, "name": "cartridge"}]}
    (tmp_path / "images.json").write_text(json.dumps(data))
    for mode in ("nms", "wbf"):
        detector = TiledDetector(None, Fake(), tile=8, overlap=0, merge=mode)
        result = predict_coco(detector, tmp_path / "images.json", tmp_path,
                              tmp_path / mode / "predictions.json", class_agnostic=True)
        assert len(result) == 1
        assert result[0] == {"image_id": 7, "bbox": [4, 1, 3, 4], "score": 0.9,
                             "category_id": 5, "id_score": 0.9}
        assert json.loads((tmp_path / mode / "predictions.json").read_text()) == result
        assert json.loads((tmp_path / mode / "meta.json").read_text())["image_count"] == 1
