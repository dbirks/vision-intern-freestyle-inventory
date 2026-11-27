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


def test_merge_and_json(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), __version__="test",
        version=SimpleNamespace(cuda=None)))
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


def test_rfdetr_maps_training_label_indices_to_category_ids(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import numpy as np
    import pytest

    from freestyle.infer.detector import RFDetrPredictor

    class Model:
        def predict(self, image, threshold):
            return SimpleNamespace(xyxy=np.array([[0, 0, 2, 2]]),
                                   confidence=np.array([0.9]), class_id=np.array([self.label]))

    model = Model()
    state = {}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **kw: state))
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(
        RFDETR=SimpleNamespace(from_checkpoint=lambda path: model)))
    for categories in ({1: "cartridge"}, {0: "cartridge"},
                       {1: "first", 2: "second", 3: "third"}):
        state.update(freestyle_category_names=categories,
                     freestyle_label_to_category=dict(enumerate(categories)))
        predictor = RFDetrPredictor(tmp_path / "checkpoint.pth")
        for label, category in enumerate(categories):
            model.label = label
            detection = predictor.predict(None, 0.3)[0]
            assert detection["category_id"] == category
            assert predictor.class_names[category] == categories[category]
        model.label = len(categories)
        if len(categories) == 1:
            assert predictor.predict(None, 0.3) == []
            model.label = 2
        with pytest.raises(ValueError, match="Unknown RF-DETR detector class index"):
            predictor.predict(None, 0.3)
    state.clear()
    with pytest.raises(ValueError, match="lacks its training category mapping"):
        RFDetrPredictor(tmp_path / "checkpoint.pth")


def test_annotation_only_export_matches_tile_annotations(tmp_path, monkeypatch):
    from freestyle.train.export_tiles import export_tiles

    Image.new("RGB", (12, 8)).save(tmp_path / "image.png")
    coco = {"images": [{"id": 1, "file_name": "image.png", "width": 12, "height": 8}],
            "categories": [{"id": 3, "name": "third"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3,
                             "bbox": [4, 1, 3, 4]}]}
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    (tmp_path / "splits.json").write_text(json.dumps({"images": {"image.png": "train"}}))
    export_tiles(tmp_path / "coco.json", tmp_path, tmp_path / "out", tmp_path / "splits.json",
                 tile=8, overlap=0, include_full_frame_downscaled=True)
    expected = json.loads((tmp_path / "out/train/_annotations.coco.json").read_text())
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("annotation-only export opened an image")))
    actual = export_tiles(tmp_path / "coco.json", None, None, tmp_path / "splits.json",
                          tile=8, overlap=0, include_full_frame_downscaled=True,
                          annotation_only=True)
    assert actual["train"] == expected


def test_recover_mapping_writes_checkpoint_and_rejects_conflicts(tmp_path, monkeypatch):
    import pickle
    import sys
    from types import SimpleNamespace

    import pytest

    def save(value, path):
        with open(path, "wb") as stream:
            pickle.dump(value, stream)

    def load(path, **kwargs):
        with open(path, "rb") as stream:
            return pickle.load(stream)

    torch = SimpleNamespace(save=save, load=load)
    monkeypatch.setitem(sys.modules, "torch", torch)
    from freestyle.train.recover_mapping import recover

    dataset = tmp_path / "dataset/train"
    training = tmp_path / "training"
    dataset.mkdir(parents=True)
    training.mkdir()
    (dataset / "_annotations.coco.json").write_text(json.dumps({
        "categories": [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}],
        "annotations": [{"category_id": 1}, {"category_id": 2}]}))
    checkpoint = training / "checkpoint.pth"
    torch.save({"rfdetr_version": "1.12.1", "args": {"class_names": ["first", "second"]}},
               checkpoint)
    recovered = recover(tmp_path)
    assert recovered["label_to_category"] == {0: 1, 1: 2}
    state = torch.load(checkpoint, weights_only=False)
    assert state["freestyle_category_names"] == {1: "first", 2: "second"}
    state["freestyle_label_to_category"] = {0: 2, 1: 1}
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="existing mapping conflicts"):
        recover(tmp_path)
