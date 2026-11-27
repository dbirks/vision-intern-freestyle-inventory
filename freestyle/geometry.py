"""Geometry for original-pixel detections and overlapping image tiles.

Boxes use half-open XYXY coordinates. Detection dictionaries use ``box`` (XYXY),
``score``, and optionally ``category_id``. Quad corners are top-left, top-right,
bottom-right, bottom-left. Functions return new arrays/dictionaries.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from tools.bbox_utils import convert_bbox


def _boxes(boxes):
    arr = np.asarray(boxes, dtype=float)
    if arr.size == 0:
        return np.empty((0, 4), dtype=float)
    return arr.reshape(-1, 4)


def _convert(boxes, source, target):
    arr = np.asarray(boxes, dtype=float)
    if arr.size == 0:
        return np.empty(arr.shape if arr.ndim > 1 else (0, 4), dtype=float)
    # Derive the linear conversion from the shared scalar helper, then apply it
    # to all boxes at once.
    basis = np.eye(4)
    matrix = np.asarray([convert_bbox(row.tolist(), source, target) for row in basis])
    return arr @ matrix


def xywh_to_xyxy(boxes):
    """Convert COCO XYWH boxes to XYXY, preserving the input shape."""
    return _convert(boxes, "coco_xywh", "xyxy")


def xyxy_to_xywh(boxes):
    """Convert XYXY boxes to COCO XYWH, preserving the input shape."""
    return _convert(boxes, "xyxy", "coco_xywh")


def make_tiles(width, height, tile=1024, overlap=0.25, min_tile=None):
    """Return row-major, unpadded tiles; ``min_tile`` is a minimum tile side.

    A dimension shorter than the minimum is retained whole. Otherwise edge tiles
    shift inward to retain the full tile size.
    """
    if width <= 0 or height <= 0 or tile <= 0 or not 0 <= overlap < 1:
        raise ValueError("positive dimensions and tile, and 0 <= overlap < 1 required")
    if min_tile is not None and (min_tile <= 0 or min_tile > tile):
        raise ValueError("min_tile must be positive and no larger than tile")

    def starts(length):
        size = min(tile, length)
        if length <= size:
            return [0]
        step = max(1, int(tile * (1 - overlap)))
        positions = list(range(0, length - size + 1, step))
        if positions[-1] != length - size:
            positions.append(length - size)
        return positions

    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in starts(height) for x in starts(width)]


def tile_boxes_to_image(boxes_xyxy, tile_origin):
    """Map tile-local boxes to image coordinates; a four-value tile clips them."""
    boxes = _boxes(boxes_xyxy).copy()
    origin = np.asarray(tile_origin, dtype=float)
    if origin.size == 4:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, origin[2] - origin[0])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, origin[3] - origin[1])
    elif origin.size != 2:
        raise ValueError("tile_origin must be (x, y) or a tile rectangle")
    return boxes + np.tile(origin[:2], 2)


def image_boxes_to_tile(boxes_xyxy, tile_origin):
    """Map image boxes to tile coordinates; a four-value tile clips them."""
    origin = np.asarray(tile_origin, dtype=float)
    if origin.size not in (2, 4):
        raise ValueError("tile_origin must be (x, y) or a tile rectangle")
    boxes = _boxes(boxes_xyxy) - np.tile(origin[:2], 2)
    if origin.size == 4:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, origin[2] - origin[0])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, origin[3] - origin[1])
    return boxes


def drop_edge_truncated(boxes, tile_rect, img_rect, margin=4):
    """Return image-coordinate boxes away from internal tile edges."""
    b = _boxes(boxes)
    t, im = np.asarray(tile_rect), np.asarray(img_rect)
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    drop = np.zeros(len(b), dtype=bool)
    for edge in range(4):
        if (edge < 2 and t[edge] > im[edge]) or (edge >= 2 and t[edge] < im[edge]):
            drop |= b[:, edge] <= t[edge] + margin if edge < 2 else b[:, edge] >= t[edge] - margin
    return b[~drop]


def iou_matrix(a, b):
    """Pairwise IoU; degenerate boxes have zero overlap."""
    a, b = _boxes(a), _boxes(b)
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh[..., 0] * wh[..., 1]
    aa = np.prod(np.maximum(0, a[:, 2:] - a[:, :2]), axis=1)
    bb = np.prod(np.maximum(0, b[:, 2:] - b[:, :2]), axis=1)
    union = aa[:, None] + bb[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def containment(a, b):
    """Intersection divided by smaller area, or zero for degenerate boxes."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    inter = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    area = min(np.prod(np.maximum(0, a[2:] - a[:2])), np.prod(np.maximum(0, b[2:] - b[:2])))
    return float(inter / area) if area else 0.0


def nms(boxes, scores, iou):
    """Return indices kept in descending score order, stable for ties."""
    boxes, scores = _boxes(boxes), np.asarray(scores, dtype=float)
    if len(boxes) != len(scores) or not 0 <= iou <= 1:
        raise ValueError("scores must match boxes and IoU must be in [0, 1]")
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        idx = order[0]
        keep.append(int(idx))
        order = order[1:][iou_matrix(boxes[idx:idx + 1], boxes[order[1:]])[0] <= iou]
    return np.asarray(keep, dtype=int)


def class_aware_nms(boxes, scores, classes, iou):
    """Return NMS indices, suppressing only within the same class."""
    boxes, scores, classes = _boxes(boxes), np.asarray(scores), np.asarray(classes)
    if len(boxes) != len(scores) or len(boxes) != len(classes):
        raise ValueError("boxes, scores and classes must have equal lengths")
    keep = []
    for cls in dict.fromkeys(classes.tolist()):
        indices = np.flatnonzero(classes == cls)
        keep.extend(indices[nms(boxes[indices], scores[indices], iou)].tolist())
    return np.asarray(sorted(keep, key=lambda i: (-scores[i], i)), dtype=int)


def weighted_box_fusion(list_of_box_sets, iou=0.55):
    """Fuse detection dictionaries by class and IoU using score-weighted boxes.

    Each box set is an iterable of dictionaries with ``box``, ``score``, and
    optional ``category_id``. Fused score is the mean of contributing scores.
    """
    if not 0 <= iou <= 1:
        raise ValueError("IoU must be in [0, 1]")
    detections = [dict(d) for group in list_of_box_sets for d in group]
    detections.sort(key=lambda d: -d["score"])
    clusters = []
    for det in detections:
        box, score = np.asarray(det["box"], dtype=float), float(det["score"])
        if score < 0:
            raise ValueError("scores must be nonnegative")
        matches = [(iou_matrix([box], [c["box"]])[0, 0], c) for c in clusters
                   if c["category_id"] == det.get("category_id")]
        best = max(matches, key=lambda pair: pair[0], default=(0, None))
        if best[1] is None or best[0] <= iou:
            clusters.append({"box": box.copy(), "category_id": det.get("category_id"),
                             "members": [det], "weights": [score]})
        else:
            c = best[1]
            c["members"].append(det)
            c["weights"].append(score)
            weights = np.asarray(c["weights"])
            if not weights.any():
                weights = np.ones_like(weights)
            c["box"] = np.average([d["box"] for d in c["members"]], axis=0, weights=weights)
    result = []
    for c in clusters:
        det = dict(c["members"][0])
        det["box"] = c["box"].tolist()
        det["score"] = float(np.mean(c["weights"]))
        result.append(det)
    return result


def merge_tiled(dets, iou=0.5, mode="nms", containment_threshold=None, class_agnostic_iou=None):
    """Merge image-coordinate detection dictionaries, optionally suppress fragments.

    Fragment suppression removes a lower-scored, no-larger box mostly contained
    in a higher-scored box of the same class. It is disabled by default.
    """
    dets = [dict(d) for d in dets]
    if containment_threshold is not None and not 0 <= containment_threshold <= 1:
        raise ValueError("containment threshold must be in [0, 1]")
    if containment_threshold is not None:
        kept = []
        for det in sorted(dets, key=lambda d: (
            -d["score"],
            -np.prod(np.maximum(0, np.asarray(d["box"])[2:] - np.asarray(d["box"])[:2])),
        )):
            area = np.prod(np.maximum(0, np.asarray(det["box"])[2:] - np.asarray(det["box"])[:2]))
            if not any(d.get("category_id") == det.get("category_id")
                       and np.prod(np.maximum(0, np.asarray(d["box"])[2:] - np.asarray(d["box"])[:2])) >= area
                       and containment(d["box"], det["box"]) > 0
                       and containment(d["box"], det["box"]) >= containment_threshold for d in kept):
                kept.append(det)
        dets = kept
    if mode == "wbf":
        result = weighted_box_fusion([dets], iou)
    elif mode == "nms":
        indices = class_aware_nms([d["box"] for d in dets], [d["score"] for d in dets],
                                  [d.get("category_id") for d in dets], iou)
        result = [dets[i] for i in indices]
    else:
        raise ValueError("mode must be 'nms' or 'wbf'")
    if class_agnostic_iou is not None:
        if not 0 <= class_agnostic_iou <= 1:
            raise ValueError("class-agnostic IoU must be in [0, 1]")
        kept = []
        for det in result:
            duplicate = next((d for d in kept if iou_matrix([d["box"]], [det["box"]])[0, 0] >= class_agnostic_iou), None)
            if duplicate is None:
                kept.append(det)
            elif duplicate.get("category_id") != det.get("category_id"):
                duplicate.setdefault("alt_class", det.get("category_id"))
        result = kept
    return result


def _containment_ratio(a, b):
    """Intersection over the smaller box area (``containment`` under another name)."""
    return containment(a, b)


def suppress_duplicates(dets, iou=None, containment=None, row_iou=None, rows=None,
                        row_x_overlap=0.5, score="score"):
    """Class-agnostic greedy duplicate suppression, keeping the higher score.

    Detections are dictionaries with ``box`` (XYXY) and ``score``. A candidate is
    dropped when, against any kept box, it overlaps by at least ``iou``, or one box
    lies at least ``containment`` inside the other (intersection over the smaller
    area, so a tile-edge fragment of a leaning cartridge never survives next to the
    full box). Identities are ignored: two boxes are duplicates when they sit on the
    same physical face, whatever class each side claims.

    ``rows`` optionally gives the shelf-row key of every detection (``shelf_idx`` from
    :func:`freestyle.count.shelves.assign_shelves`). A leaning cartridge commonly gets
    one box per overlapping tile with slightly shifted columns, so its IoU stays below
    the global threshold; ``row_iou`` is the looser threshold used for two boxes that
    share a row *and* overlap in x by at least ``row_x_overlap`` of the narrower box.
    Cartridges stacked in the same column keep their own box because a real tier sits
    above or below its partner, which makes both IoU and y-overlap near zero.
    """
    dets = [dict(d) for d in dets]
    if rows is not None:
        rows = list(rows)
        if len(rows) != len(dets):
            raise ValueError("rows must have one key per detection")
    for name, value in (("iou", iou), ("containment", containment), ("row_iou", row_iou),
                        ("row_x_overlap", row_x_overlap if row_iou is not None else None)):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if row_iou is not None and rows is None:
        raise ValueError("row_iou requires rows")
    if iou is None and containment is None and row_iou is None:
        return dets

    boxes = _boxes([d["box"] for d in dets])
    if len(boxes) and (not np.isfinite(boxes).all() or np.any(boxes[:, 2] <= boxes[:, 0])
                       or np.any(boxes[:, 3] <= boxes[:, 1])):
        raise ValueError("duplicate suppression needs finite, nondegenerate boxes")

    def x_overlap(a, b):
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        narrowest = min(a[2] - a[0], b[2] - b[0])
        return overlap / narrowest if narrowest > 0 else 0.0

    order = np.argsort(-np.asarray([d[score] for d in dets], dtype=float), kind="stable")
    kept = []
    for index in order:
        box = boxes[index]
        duplicate = False
        for other in kept:
            j = other["index"]
            overlap = iou_matrix(box[None], boxes[j][None])[0, 0]
            same_row = row_iou is not None and rows[index] == rows[j] \
                and x_overlap(box, boxes[j]) >= row_x_overlap
            if (iou is not None and overlap >= iou) or \
               (same_row and overlap >= row_iou) or \
               (containment is not None and _containment_ratio(box, boxes[j]) >= containment):
                duplicate = True
                break
        if not duplicate:
            kept.append({"index": int(index), "det": dets[index]})
    return [k["det"] for k in sorted(kept, key=lambda k: k["index"])]


def crop(image_PIL, box_xyxy, pad=0.08, min_size=None):
    """Crop with proportional padding; min_size optionally upsizes the result.

    ``min_size`` is a minimum shorter side in pixels; aspect ratio is preserved.
    """
    x0, y0, x1, y1 = map(float, box_xyxy)
    if x1 <= x0 or y1 <= y0 or pad < 0 or (min_size is not None and min_size <= 0):
        raise ValueError("box, padding or min_size is invalid")
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    rect = (max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)),
            min(image_PIL.width, math.ceil(x1 + dx)), min(image_PIL.height, math.ceil(y1 + dy)))
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        raise ValueError("box does not intersect image")
    result = image_PIL.crop(rect)
    if min_size is not None and min(result.size) < min_size:
        scale = min_size / min(result.size)
        result = result.resize(tuple(round(s * scale) for s in result.size), Image.Resampling.LANCZOS)
    return result


def crop_quad(image_PIL, quad_4x2, out_h=None):
    """Rectify ordered TL, TR, BR, BL corners using a Pillow homography."""
    q = np.asarray(quad_4x2, dtype=float)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("quad must contain four finite 2D corners")
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    if width <= 0 or height <= 0 or (out_h is not None and out_h <= 0):
        raise ValueError("quad and output height must be nondegenerate")
    h = max(2, round(height if out_h is None else out_h))
    w = max(2, round(width * h / height))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
    matrix, rhs = [], []
    for (x, y), (u, v) in zip(dst, q, strict=True):
        matrix.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        rhs.extend([u, v])
    try:
        coeffs = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("quad must be nondegenerate") from exc
    return image_PIL.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
