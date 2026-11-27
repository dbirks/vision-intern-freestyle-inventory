"""Build the phone-first ground-truth quiz queue from model proposals.

The queue is a flat, ordered list of two question kinds derived from a COCO
proposals file (``data/labels/*.json``, never ground truth):

``count``
    One per image. The human confirms/fixes the model's top-shelf and
    bottom-shelf cartridge totals, or marks the frame unusable.
``crop``
    One per proposed box. The human confirms or corrects that box's identity.

Order (fixed, so two sessions resume identically): every image of the first
requested split, then the next split; within an image the ``count`` question
first, then its crops left→right with the top shelf before the bottom shelf.

Run:
    uv run python -m freestyle.quiz.build_queue \
        --proposals data/labels/bootstrap_v2.json --splits data/splits.json \
        --splits-order test,val --out data/gt/quiz_queue.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from freestyle.count.shelves import assign_shelves
from freestyle.geometry import xywh_to_xyxy
from freestyle.review.server import load_classes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_ROOT = REPO_ROOT / "data" / "raw" / "freestyle-vision-data" / "images"
DEFAULT_CATALOG = REPO_ROOT / "data" / "raw" / "freestyle-vision-data" / "data" / "catalog_items.json"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits.json"

COUNT = "count"
CROP = "crop"
UNKNOWN = "unknown"
#: Question kinds in the order they appear inside one image.
KINDS = (COUNT, CROP)
SHELF_RANK = {"top": 0, "bottom": 1}
QUEUE_VERSION = 1


# --------------------------------------------------------------------------- ids


def count_qid(image_id: str) -> str:
    """Stable question id for an image's count question."""
    return f"count:{image_id}"


def crop_qid(image_id: str, ann_id: int) -> str:
    """Stable question id for one proposed box."""
    return f"crop:{image_id}:{int(ann_id)}"


def qid_image_id(qid: str) -> str:
    """Image id embedded in a question id ('' when malformed)."""
    parts = str(qid).split(":")
    if len(parts) >= 2 and parts[0] in KINDS:
        joined = ":".join(parts[1:-1]) if parts[0] == CROP else ":".join(parts[1:])
        return joined
    return ""


def qid_ann_id(qid: str) -> int | None:
    """Annotation id embedded in a crop question id (None for count questions)."""
    parts = str(qid).split(":")
    if len(parts) >= 3 and parts[0] == CROP:
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- text


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def family_key(text: Any) -> str:
    """Normalized brand family of a catalog name/id: 'Diet Coke Part 1' -> 'diet coke'."""
    return _norm(re.sub(r"\bpart\s*[12]\b", " ", str(text or ""), flags=re.I))


def family_classes(classes: list[dict[str, Any]]) -> dict[str, dict[int, str]]:
    """``family key -> {part number: class name}`` for two-part catalog brands."""
    out: dict[str, dict[int, str]] = {}
    for entry in classes:
        name = str(entry.get("name") or "")
        if name == UNKNOWN:
            continue
        match = re.search(r"\bpart\s*([12])\b", _norm(entry.get("display") or name) + " " + _norm(name))
        key = family_key(entry.get("display") or name) or family_key(name)
        if match:
            out.setdefault(key, {})[int(match.group(1))] = name
    return {k: v for k, v in out.items() if len(v) == 2}


def quiz_classes(catalog_path: Path | None = None) -> list[dict[str, Any]]:
    """Tap-target classes: the 18 active catalog ids plus ``unknown`` (no super-class)."""
    return [
        {
            "name": c["name"],
            "display": c["display"],
            "color": c["color"],
            "id": int(c["id"]),
            "generic": bool(c.get("generic")),
        }
        for c in load_classes(Path(catalog_path) if catalog_path else DEFAULT_CATALOG)
        if c["name"] != "cartridge"
    ]


# --------------------------------------------------------------------------- shelves


def shelf_labels(boxes_xyxy: list[list[float]], height: int) -> list[str]:
    """'top'/'bottom' per box: the repo's fisheye shelf fit, position split as fallback.

    ``assign_shelves`` fits up to two quadratic bands and is the same heuristic the
    counting pipeline uses. When it only finds one band (a partial view, or too few
    boxes to fit two), each box is placed by where it sits in the frame instead.
    """
    n = len(boxes_xyxy)
    if n == 0:
        return []
    centers_y = [(float(b[1]) + float(b[3])) / 2.0 for b in boxes_xyxy]
    mid = (float(height) / 2.0) if height and height > 0 else _median(centers_y)

    def by_position() -> list[str]:
        return ["top" if cy < mid else "bottom" for cy in centers_y]

    try:
        rows = assign_shelves(
            [{"box": [float(v) for v in b], "score": 1.0} for b in boxes_xyxy], max_rows=2
        )
    except Exception:  # degenerate boxes, too few points, numeric trouble
        return by_position()
    labels = [row.get("shelf_idx") for row in rows]
    if any(not isinstance(v, int) for v in labels) or len({int(v) for v in labels}) < 2:
        return by_position()
    # rank bands by their median y so band order (not the fit's order) decides top/bottom
    bands = sorted({int(v) for v in labels})
    rank = {
        band: i
        for i, band in enumerate(
            sorted(
                bands,
                key=lambda band: _median(
                    [cy for cy, lab in zip(centers_y, labels, strict=True) if int(lab) == band]
                ),
            )
        )
    }
    return ["top" if rank[int(v)] == 0 else "bottom" for v in labels]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


# --------------------------------------------------------------------------- queue


def _clip(box: list[float], width: int, height: int) -> list[float] | None:
    """Clip a box to the frame; None when it has no area left."""
    x0, y0 = max(0.0, float(box[0])), max(0.0, float(box[1]))
    x1, y1 = min(float(width), float(box[2])), min(float(height), float(box[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _guess_fields(
    cls: str, attrs: dict[str, Any], classes: list[dict[str, Any]], parts: dict[str, dict[int, str]]
) -> dict[str, Any]:
    """Guess display text, family, part and part-question options for one annotation."""
    attrs = attrs if isinstance(attrs, dict) else {}
    display = {c["name"]: c["display"] for c in classes}.get(cls, cls)
    family_raw = attrs.get("id_family")
    family = _norm(family_raw) if family_raw else ""
    part = attrs.get("id_part")
    part = int(part) if part in (1, 2) or (isinstance(part, str) and part.strip() in ("1", "2")) else None
    # a family-only guess (bootstrap emits 'unknown' + id_family) still tells us the brand
    two_part = cls == UNKNOWN and bool(family) and family in parts and part is None
    if two_part:
        guess_text = f"{_FAMILY_DISPLAY.get(family, family)} (part unknown)"
    else:
        guess_text = display
    try:
        confidence = float(attrs.get("id_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    options: list[dict[str, Any]] = []
    ask_part = two_part
    if ask_part:
        by_name = {c["name"]: c for c in classes}
        for number in (1, 2):
            name = parts[family].get(number)
            if name in by_name:
                options.append(
                    {
                        "part": number,
                        "class": name,
                        "display": by_name[name]["display"],
                        "color": by_name[name]["color"],
                    }
                )
        if len(options) != 2:
            ask_part = False
            guess_text = display
    return {
        "guess": cls,
        "guess_text": guess_text,
        "confidence": round(confidence, 2),
        "family": family or None,
        "part": part,
        "part_options": options,
        "ask_part": ask_part,
    }


#: ``id_family`` strings used by the bootstrap pipeline -> human-readable brand.
_FAMILY_DISPLAY = {
    "coca cola": "Coca-Cola",
    "coca cola zero": "Coca-Cola Zero",
    "diet coke": "Diet Coke",
    "dr pepper": "Dr Pepper",
    "cherry": "Cherry",
    "nns": "NNS",
    "sprite": "Sprite",
    "fanta": "Fanta",
    "hi c": "Hi-C",
    "fanta bb": "Fanta BB",
    "aa": "Ingredient AA",
    "cc": "Ingredient CC",
    "vanilla flavor": "Vanilla Flavor",
    "lemon flavor": "Lemon Flavor",
    "orange flavor": "Orange Flavor",
    "orange": "Orange Flavor",
    "unknown": "unknown",
}


def build_queue(
    proposals: Path | dict[str, Any],
    splits: Path | dict[str, Any] | None = None,
    splits_order: list[str] | tuple[str, ...] = ("test", "val"),
    classes: list[dict[str, Any]] | None = None,
    images_root: Path = DEFAULT_IMAGES_ROOT,
    audit: Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ordered quiz queue from a COCO proposals mapping."""
    data = proposals if isinstance(proposals, dict) else json.loads(Path(proposals).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("proposals must be a COCO JSON object")
    split_map: dict[str, str] = {}
    if splits is not None:
        sdata = splits if isinstance(splits, dict) else json.loads(Path(splits).read_text(encoding="utf-8"))
        raw_images = (sdata or {}).get("images") or {}
        split_map = {str(k): str(v) for k, v in raw_images.items()}
    order = [str(s).strip() for s in splits_order if str(s).strip()]
    if not order:
        raise ValueError("splits_order must name at least one split")
    rank = {name: i for i, name in enumerate(order)}

    classes = classes if classes is not None else quiz_classes()
    parts = family_classes(classes)
    cat_names = {int(c["id"]): str(c.get("name") or UNKNOWN) for c in data.get("categories") or []}

    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations") or []:
        try:
            by_image.setdefault(int(ann["image_id"]), []).append(ann)
        except (KeyError, TypeError, ValueError):
            continue

    images_root = Path(images_root)
    out_images: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []

    for img in data.get("images") or []:
        image_id = PurePosixPath(str(img.get("file_name") or "")).as_posix()
        if not image_id:
            continue
        split = split_map.get(image_id) or split_map.get(str(img.get("id")))
        if split is None or split not in rank:
            continue
        width, height = int(img.get("width") or 0), int(img.get("height") or 0)
        path = images_root / PurePosixPath(image_id)
        if width <= 0 or height <= 0 or not path.is_file():
            print(f"warning: skipping {image_id} (no readable file/size)", file=sys.stderr)
            continue

        anns: list[dict[str, Any]] = []
        raw_anns = sorted(by_image.get(int(img.get("id") or 0), []), key=lambda a: int(a.get("id") or 0))
        for ann in raw_anns:
            try:
                bbox = [float(v) for v in ann.get("bbox") or []]
            except (TypeError, ValueError):
                continue
            if len(bbox) != 4:
                continue
            x0, y0, w, h = xywh_to_xyxy([bbox]).reshape(-1, 4)[0].tolist()
            box = _clip([x0, y0, w, h], width, height)
            if box is None:
                continue
            attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
            entry = {
                "ann_id": int(ann.get("id") or 0) or None,
                "class": cat_names.get(int(ann.get("category_id") or -1), UNKNOWN),
                "bbox": box,
                "raw_attributes": attrs,
                "attributes": {
                    "truncated": bool(attrs.get("truncated")),
                    "occluded": bool(attrs.get("occluded")),
                    "identity_visible": bool(attrs.get("identity_visible", True)),
                    "source": str(attrs.get("source") or ""),
                },
            }
            if entry["ann_id"] is None:
                continue
            anns.append(entry)

        for ann, shelf in zip(anns, shelf_labels([a["bbox"] for a in anns], height), strict=True):
            ann["shelf"] = shelf

        top = sum(1 for a in anns if a["shelf"] == "top")
        bottom = sum(1 for a in anns if a["shelf"] == "bottom")
        image_questions = [
            {
                "qid": count_qid(image_id),
                "kind": COUNT,
                "image_id": image_id,
                "split": split,
                "model_top": top,
                "model_bottom": bottom,
                "model_total": len(anns),
            }
        ]
        # left→right within a shelf, top shelf first
        anns.sort(key=lambda a: (SHELF_RANK.get(a["shelf"], 2), (a["bbox"][0] + a["bbox"][2]) / 2.0))
        for number, ann in enumerate(anns, start=1):
            guess = _guess_fields(ann["class"], ann["raw_attributes"], classes, parts)
            image_questions.append(
                {
                    "qid": crop_qid(image_id, ann["ann_id"]),
                    "kind": CROP,
                    "image_id": image_id,
                    "split": split,
                    "ann_id": ann["ann_id"],
                    "number": number,
                    "shelf": ann["shelf"],
                    "bbox": [round(v, 2) for v in ann["bbox"]],
                    "attributes": ann["attributes"],
                    **guess,
                }
            )

        qids = [q["qid"] for q in image_questions]
        out_images.append(
            {
                "image_id": image_id,
                "coco_image_id": int(img.get("id") or 0) or None,
                "split": split,
                "scene": (img.get("attributes") or {}).get("scene") or img.get("scene"),
                "width": width,
                "height": height,
                "model_top": top,
                "model_bottom": bottom,
                "model_total": len(anns),
                "qids": qids,
            }
        )
        questions.extend(image_questions)

    audit_data = audit if isinstance(audit, dict) else (json.loads(Path(audit).read_text()) if audit else {})
    audit_images = {im["image_id"]: im for im in audit_data.get("images", [])}

    def image_key(image_id: str) -> tuple:
        item = audit_images.get(image_id, {})
        scene, blur = item.get("scene_label"), item.get("motion_blur")
        tier = 3 if blur == "severe" else (0 if scene == "cabinet_full" and blur in ("none", "mild")
                                            else 1 if scene == "cabinet_partial" else 2)
        return (tier, -float(item.get("sharp_full") or 0) if tier == 0 else 0, image_id)

    # Keep each image's count and crops together in audit quality order.
    questions.sort(
        key=lambda q: (rank.get(q["split"], len(rank)), image_key(q["image_id"]), q["kind"] != COUNT, q.get("number", 0))
    )
    index = {q["qid"]: i for i, q in enumerate(questions)}
    for image in out_images:
        image["first_index"] = min(index[q] for q in image["qids"])
    out_images.sort(key=lambda im: (rank.get(im["split"], len(rank)), image_key(im["image_id"])))

    return {
        "version": QUEUE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "splits_order": order,
        "classes": classes,
        "images": out_images,
        "questions": questions,
    }


def validate_queue(queue: dict[str, Any]) -> dict[str, Any]:
    """Sanity-check a queue document; raises ValueError on a bad file."""
    if not isinstance(queue, dict) or not isinstance(queue.get("questions"), list):
        raise ValueError("not a quiz queue (no 'questions' list)")
    if int(queue.get("version") or 0) != QUEUE_VERSION:
        raise ValueError(f"unsupported queue version {queue.get('version')!r}")
    seen: set[str] = set()
    for question in queue["questions"]:
        qid = str(question.get("qid") or "")
        if not qid or qid in seen:
            raise ValueError(f"duplicate/empty question id {qid!r}")
        seen.add(qid)
        if question.get("kind") not in KINDS:
            raise ValueError(f"unknown question kind {question.get('kind')!r}")
    return queue


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the quiz question queue from model proposals")
    p.add_argument("--proposals", type=Path, required=True, help="COCO proposals json (data/labels/*.json)")
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS, help="data/splits.json (image_id -> split)")
    p.add_argument("--splits-order", default="test,val", help="comma-separated splits to quiz, in order")
    p.add_argument("--out", type=Path, required=True, help="queue json to write (typically data/gt/quiz_queue.json)")
    p.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    p.add_argument("--audit", type=Path, default=Path("data/derived/audit/audit.json"))
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    classes = quiz_classes(args.catalog)
    queue = build_queue(
        args.proposals,
        args.splits,
        [s for s in args.splits_order.split(",") if s.strip()],
        classes,
        args.images_root,
        args.audit,
    )
    validate_queue(queue)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    counts = {split: sum(1 for im in queue["images"] if im["split"] == split) for split in queue["splits_order"]}
    print(
        f"queue: {len(queue['questions'])} questions over {len(queue['images'])} images "
        f"({', '.join(f'{s}={n}' for s, n in counts.items())}) -> {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
