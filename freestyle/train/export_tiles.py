"""Export original-pixel COCO annotations to a tiled RF-DETR dataset."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

from freestyle.geometry import make_tiles


def export_tiles(coco_file, images_root, out, split_map, tile=1024, overlap=0.25,
                 min_visible=0.6, class_agnostic=False,
                 include_full_frame_downscaled=False, resolution=512):
    if not 0 <= min_visible <= 1 or resolution <= 0:
        raise ValueError("min_visible must be in [0, 1] and resolution positive")
    coco = json.loads(Path(coco_file).read_text())
    splits = json.loads(Path(split_map).read_text())["images"]
    out, root = Path(out), Path(images_root)
    categories = ([{"id": 1, "name": "cartridge"}] if class_agnostic
                  else coco["categories"])
    datasets = {s: {"info": {}, "licenses": [], "images": [], "annotations": [],
                    "categories": categories} for s in ("train", "valid", "test")}
    for split in datasets:
        (out / split).mkdir(parents=True, exist_ok=True)
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    manifest = []
    image_id = ann_id = 0
    for source in coco["images"]:
        split = splits.get(source["file_name"])
        if split is None:
            raise ValueError(f"Missing split for {source['file_name']}")
        if split == "exclude":
            continue
        split = {"val": "valid", "train": "train", "test": "test"}.get(split)
        if split is None:
            raise ValueError(f"Invalid split for {source['file_name']}")
        with Image.open(root / source["file_name"]) as img:
            width, height = img.size
            if (width, height) != (source["width"], source["height"]):
                raise ValueError(f"Image dimensions differ from COCO: {source['file_name']}")
            samples = [(rect, False) for rect in make_tiles(width, height, tile, overlap)]
            if include_full_frame_downscaled:
                samples.append(((0, 0, width, height), True))
            for rect, full in samples:
                image_id += 1
                x0, y0, x1, y1 = rect
                sample = img.crop(rect).convert("RGB")
                if full:
                    sample = sample.resize((resolution, resolution), Image.Resampling.LANCZOS)
                sx, sy = sample.width / (x1 - x0), sample.height / (y1 - y0)
                name = f"{image_id:08d}.png"
                mean = tuple(round(v) for v in ImageStat.Stat(sample.convert("RGB")).mean)
                masked = []
                original = sample.copy()
                mask = Image.new("L", sample.size, 0)
                protected = Image.new("L", sample.size, 0)
                ds = datasets[split]
                ds["images"].append({"id": image_id, "file_name": name,
                                     "width": sample.width, "height": sample.height})
                manifest.append({"image_id": image_id, "source_image_id": source["id"],
                                 "source_file_name": source["file_name"], "split": split,
                                 "file_name": name, "tile": list(rect), "full_frame": full,
                                 "masked_instances": masked})
                for ann in annotations[source["id"]]:
                    x, y, w, h = ann["bbox"]
                    if w <= 0 or h <= 0:
                        continue
                    left, top = max(x, x0), max(y, y0)
                    right, bottom = min(x + w, x1), min(y + h, y1)
                    cw, ch = max(0, right - left), max(0, bottom - top)
                    if not cw or not ch:
                        continue
                    if cw * ch / (w * h) < min_visible:
                        import math
                        box = [(left - x0) * sx, (top - y0) * sy,
                               (right - x0) * sx, (bottom - y0) * sy]
                        ImageDraw.Draw(mask).rectangle(
                            (math.floor(box[0]), math.floor(box[1]),
                             math.ceil(box[2]) - 1, math.ceil(box[3]) - 1), fill=255)
                        masked.append({"annotation_id": ann.get("id"), "box": box})
                        continue
                    box = [(left - x0) * sx, (top - y0) * sy, cw * sx, ch * sy]
                    import math
                    ImageDraw.Draw(protected).rectangle(
                        (math.floor(box[0]), math.floor(box[1]),
                         math.ceil(box[0] + box[2]) - 1, math.ceil(box[1] + box[3]) - 1), fill=255)
                    ann_id += 1
                    ds["annotations"].append({"id": ann_id, "image_id": image_id,
                        "category_id": 1 if class_agnostic else ann["category_id"],
                        "bbox": box, "area": box[2] * box[3],
                        "iscrowd": ann.get("iscrowd", 0)})
                from PIL import ImageChops
                mask = ImageChops.subtract(mask, protected)
                sample = Image.composite(Image.new("RGB", sample.size, mean), original, mask)
                sample.save(out / split / name, format="PNG")
    for split, dataset in datasets.items():
        (out / split / "_annotations.coco.json").write_text(json.dumps(dataset, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split-map", required=True)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--min-visible", type=float, default=0.6)
    p.add_argument("--class-agnostic", action="store_true")
    p.add_argument("--include-full-frame-downscaled", action="store_true")
    p.add_argument("--resolution", type=int, default=512)
    a = p.parse_args()
    export_tiles(a.coco, a.images_root, a.out, a.split_map, a.tile, a.overlap,
                 a.min_visible, a.class_agnostic, a.include_full_frame_downscaled, a.resolution)


if __name__ == "__main__":
    main()
