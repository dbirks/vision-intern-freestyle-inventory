"""Tiled detector and COCO-results inference CLI."""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from freestyle.geometry import make_tiles, merge_tiled, tile_boxes_to_image
from freestyle.train.train_detector import metadata


class RFDetrPredictor:
    def __init__(self, checkpoint):
        from rfdetr import RFDETR

        self.model = RFDETR.from_checkpoint(str(checkpoint))
        self.class_names = self.model.class_names

    def predict(self, image, threshold):
        result = self.model.predict(image, threshold=threshold)
        return [{"box": box.tolist(), "score": float(score), "category_id": int(cls)}
                for box, score, cls in zip(result.xyxy, result.confidence, result.class_id, strict=True)]


class YoloPredictor:
    def __init__(self, checkpoint):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise NotImplementedError("Install ultralytics to use the YOLO backend") from exc
        self.model = YOLO(str(checkpoint))
        self.class_names = self.model.names

    def predict(self, image, threshold):
        result = self.model.predict(image, conf=threshold, verbose=False)[0].boxes
        return [{"box": box.tolist(), "score": float(score), "category_id": int(cls)}
                for box, score, cls in zip(result.xyxy.cpu(), result.conf.cpu(),
                                           result.cls.cpu(), strict=True)]


class TiledDetector:
    def __init__(self, checkpoint, backend="rfdetr", tile=1024, overlap=0.25,
                 score_thr=0.3, merge="nms", containment_threshold=None, iou=0.5):
        if merge not in ("nms", "wbf") or not 0 <= score_thr <= 1 or not 0 <= iou <= 1:
            raise ValueError("Invalid merge mode, score threshold, or IoU")
        self.checkpoint, self.backend_name = str(checkpoint), (backend if isinstance(backend, str) else type(backend).__name__)
        if isinstance(backend, str):
            factories = {"rfdetr": RFDetrPredictor, "yolo": YoloPredictor}
            if backend not in factories:
                raise ValueError(f"Unknown backend: {backend}")
            backend = factories[backend](checkpoint)
        self.backend, self.tile, self.overlap = backend, tile, overlap
        self.score_thr, self.merge, self.containment_threshold, self.iou = (
            score_thr, merge, containment_threshold, iou)

    def predict(self, image):
        detections = []
        for rect in make_tiles(image.width, image.height, self.tile, self.overlap):
            for det in self.backend.predict(image.crop(rect), self.score_thr):
                if det["score"] < self.score_thr:
                    continue
                box = tile_boxes_to_image([det["box"]], rect)[0].tolist()
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                detections.append({**det, "box": box})
        return merge_tiled(detections, iou=self.iou, mode=self.merge,
                           containment_threshold=self.containment_threshold)


def predict_coco(detector, coco, images_root, out, class_agnostic=False, split_map=None,
                 split=None, category_map=None):
    """Run on COCO images; category_map maps backend indices to COCO category IDs."""
    try:
        import torch
    except ImportError:
        torch = None

    data = json.loads(Path(coco).read_text())
    selected = json.loads(Path(split_map).read_text())["images"] if split_map else None
    if bool(split_map) != bool(split):
        raise ValueError("split-map and split must be supplied together")
    categories = {c["name"]: c["id"] for c in data["categories"]}
    if class_agnostic and "cartridge" not in categories:
        raise ValueError("COCO categories must include cartridge for class-agnostic inference")
    if not class_agnostic and category_map is None:
        names = getattr(detector.backend, "class_names", None)
        if names is None:
            raise ValueError("Provide category_map mapping model indices to COCO category IDs")
        names = names.items() if isinstance(names, dict) else enumerate(names)
        try:
            category_map = {str(i): categories[name] for i, name in names}
        except KeyError as exc:
            raise ValueError(f"Model class absent from COCO categories: {exc}") from exc
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start, count, predictions = time.perf_counter(), 0, []
    for item in data["images"]:
        if selected is not None:
            if item["file_name"] not in selected:
                raise ValueError(f"Missing split for {item['file_name']}")
            if selected[item["file_name"]] != split:
                continue
        with Image.open(Path(images_root) / item["file_name"]) as image:
            if image.size != (item["width"], item["height"]):
                raise ValueError(f"Image dimensions differ from COCO: {item['file_name']}")
            detections = detector.predict(image.convert("RGB"))
        count += 1
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            category = categories["cartridge"] if class_agnostic else det["category_id"]
            if not class_agnostic and category_map is not None:
                category = category_map[str(category)]
            predictions.append({"image_id": item["id"], "bbox": [x1, y1, x2-x1, y2-y1],
                                "score": float(det["score"]), "category_id": int(category),
                                "id_score": float(det.get("id_score", det["score"]))})
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, indent=2))
    meta = metadata()
    meta.update(experiment=out.parent.name, config={"checkpoint": detector.checkpoint,
                "backend": detector.backend_name, "class_agnostic": class_agnostic,
                "tile": detector.tile,
                "overlap": detector.overlap, "score_thr": detector.score_thr,
                "merge": detector.merge, "iou": detector.iou,
                "containment_threshold": detector.containment_threshold},
                seconds_per_image=elapsed / count if count else None, image_count=count)
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backend", choices=("rfdetr", "yolo"), default="rfdetr")
    p.add_argument("--images", required=True, help="COCO JSON containing image IDs and categories")
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split-map")
    p.add_argument("--split", choices=("train", "val", "test", "exclude"))
    p.add_argument("--category-map", help="JSON mapping model class indices to COCO category IDs")
    p.add_argument("--class-agnostic", action="store_true")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--merge", choices=("nms", "wbf"), default="nms")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--containment-threshold", type=float)
    a = p.parse_args()
    detector = TiledDetector(a.checkpoint, a.backend, a.tile, a.overlap, a.score_thr,
                             a.merge, a.containment_threshold, a.iou)
    mapping = json.loads(Path(a.category_map).read_text()) if a.category_map else None
    predict_coco(detector, a.images, a.images_root, a.out, a.class_agnostic,
                 a.split_map, a.split, mapping)


if __name__ == "__main__":
    main()
