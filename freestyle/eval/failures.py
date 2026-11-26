"""Failure analysis and visual diagnostic reporting for Freestyle recognition."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from freestyle.eval.matching import MatchMethod, match_boxes
from freestyle.geometry import crop, iou_matrix, xywh_to_xyxy
from tools.bbox_viz import _font

FailureType = Literal["FN", "FP", "ID", "LOC"]

FAILURE_TYPES: tuple[FailureType, ...] = ("FN", "FP", "ID", "LOC")
FN_SUBTYPES = ("adjacent_merge", "edge_truncated", "stacked_tier", "plain")
FP_SUBTYPES = ("duplicate", "fragment", "background")

#: Short display abbreviations for all active catalog classes plus special classes.
CLASS_ABBREV: dict[str, str] = {
    "nns-sweetener": "NNS",
    "ingredient-aa": "ING-AA",
    "ingredient-cc": "ING-CC",
    "fanta-orange": "FANTA",
    "hi-c": "HI-C",
    "sprite": "SPRITE",
    "flavor-orange": "FL-ORG",
    "flavor-vanilla": "FL-VAN",
    "flavor-lemon": "FL-LEM",
    "flavor-cherry": "FL-CHR",
    "coca-cola-part1": "CC-P1",
    "coca-cola-part2": "CC-P2",
    "diet-coke-part1": "DC-P1",
    "diet-coke-part2": "DC-P2",
    "dr-pepper-part1": "DP-P1",
    "dr-pepper-part2": "DP-P2",
    "coke-zero-part1": "CZ-P1",
    "coke-zero-part2": "CZ-P2",
    "unknown": "UNK",
    "cartridge": "CART",
}

# Overlay palette (RGB / hex).
COLOR_GT = "#22C55E"
COLOR_TP = "#2563EB"
COLOR_FP = "#EF4444"
COLOR_FN = "#F97316"
COLOR_ID = "#D946EF"
COLOR_LOC = "#06B6D4"

PREVIEW_WIDTH = 1600
EDGE_MARGIN_PX = 4.0
LOC_DEFAULT_IOU = 0.3
GALLERY_PAD_CONTEXT = 0.5  # +0.5 box on each side => 2x context box around target


@dataclass
class FailureCase:
    """Single localization or identification failure detected during evaluation."""

    image_id: int | str
    file_name: str | None
    type: FailureType
    subtype: str | None
    box: list[float]
    gt_box: list[float] | None
    pred_box: list[float] | None
    score: float | None
    id_score: float | None
    iou: float | None
    gt_category_id: int | None
    gt_class: str | None
    gt_short: str | None
    pred_category_id: int | None
    pred_class: str | None
    pred_short: str | None
    confusion_pair: str | None
    sharpness: float | None
    norm_x: float
    norm_y: float
    shelf: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return asdict(self)


def short_label(name: str | None) -> str:
    """Return the short catalog abbreviation for *name*, with a clean fallback."""
    if not name:
        return "?"
    key = str(name).strip().lower()
    if key in CLASS_ABBREV:
        return CLASS_ABBREV[key]
    parts = [p for p in re.split(r"[-_\s]+", key) if p]
    if len(parts) >= 2:
        return "-".join(p[:3].upper() for p in parts[:2])
    return key[:6].upper()


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the discrete 4-neighbor Laplacian on a 2D grayscale array."""
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    kernel = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(np.var(kernel))


def _crop_sharpness(image: Image.Image | None, box_xywh: Sequence[float]) -> float | None:
    """Compute Laplacian variance on the original-pixel crop of *box_xywh*."""
    if image is None:
        return None
    x, y, w, h = (float(v) for v in box_xywh)
    if w <= 0 or h <= 0:
        return 0.0
    try:
        patch = crop(image, [x, y, x + w, y + h], pad=0.0)
    except ValueError:
        return 0.0
    gray = np.asarray(patch.convert("L"), dtype=np.float64)
    return round(_laplacian_variance(gray), 3)


def _extract_shelf(pred: Mapping[str, Any] | None) -> int | str | None:
    """Extract optional shelf hint from prediction extras."""
    if pred is None:
        return None
    for key in ("shelf", "shelf_idx"):
        if pred.get(key) is not None:
            return pred[key]
    extras = pred.get("extras")
    if isinstance(extras, Mapping):
        for key in ("shelf", "shelf_idx"):
            if extras.get(key) is not None:
                return extras[key]
    return None


def _box_areas_xyxy(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    wh = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2])
    return wh[:, 0] * wh[:, 1]


def _intersection_matrix_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    wh = np.maximum(
        0.0,
        np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]),
    )
    return wh[..., 0] * wh[..., 1]


def _resolve_image_path(
    img_rec: Mapping[str, Any],
    images_root: str | Path | None,
) -> Path | None:
    file_name = img_rec.get("file_name")
    if not file_name:
        return None
    if images_root is not None:
        cand = Path(images_root) / str(file_name)
        if cand.is_file():
            return cand
    p = Path(str(file_name))
    if p.is_file():
        return p
    return None


def _is_edge_truncated(
    gt_ann: Mapping[str, Any],
    gt_xyxy: np.ndarray,
    img_w: float,
    img_h: float,
) -> bool:
    attrs = gt_ann.get("attributes")
    if isinstance(attrs, Mapping) and attrs.get("truncated") is True:
        return True
    x1, y1, x2, y2 = gt_xyxy.tolist()
    return (
        x1 <= EDGE_MARGIN_PX
        or y1 <= EDGE_MARGIN_PX
        or x2 >= img_w - EDGE_MARGIN_PX
        or y2 >= img_h - EDGE_MARGIN_PX
    )


def _is_stacked_tier(
    gj: int,
    gt_xyxy: np.ndarray,
    gt_anns: Sequence[Mapping[str, Any]],
) -> bool:
    """Detect vertical column stacking (same heuristic as ``freestyle.count.shelves``)."""
    attrs = gt_anns[gj].get("attributes")
    if isinstance(attrs, Mapping):
        tier = attrs.get("tier")
        if tier is not None and int(tier) > 0:
            return True

    a = gt_xyxy[gj]
    ha = max(0.0, float(a[3] - a[1]))
    wa = max(0.0, float(a[2] - a[0]))
    if ha <= 0 or wa <= 0:
        return False

    for other_idx in range(len(gt_xyxy)):
        if other_idx == gj:
            continue
        b = gt_xyxy[other_idx]
        hb = max(0.0, float(b[3] - b[1]))
        wb = max(0.0, float(b[2] - b[0]))
        if hb <= 0 or wb <= 0:
            continue
        overlap = max(0.0, float(min(a[2], b[2]) - max(a[0], b[0])))
        ratio = overlap / min(wa, wb)
        vertical_gap = float(max(a[1], b[1]) - min(a[3], b[3]))
        if ratio > 0.5 and -0.25 * min(ha, hb) <= vertical_gap <= 0.35 * max(ha, hb):
            return True
    return False


def _classify_fn_subtype(
    gj: int,
    gt_xyxy: np.ndarray,
    pred_xyxy: np.ndarray,
    pred_gt_inter: np.ndarray,
    gt_areas: np.ndarray,
    gt_anns: Sequence[Mapping[str, Any]],
    img_w: float,
    img_h: float,
) -> str:
    # 1. adjacent_merge: a kept prediction covers this GT (>=0.5) plus a neighbour GT (>=0.5).
    if len(pred_xyxy) > 0 and gt_areas[gj] > 0:
        gt_cov = np.where(gt_areas[None, :] > 0, pred_gt_inter / gt_areas[None, :], 0.0)
        for pi in range(len(pred_xyxy)):
            if gt_cov[pi, gj] >= 0.5:
                other_covered = np.any((np.arange(len(gt_xyxy)) != gj) & (gt_cov[pi] >= 0.5))
                if other_covered:
                    return "adjacent_merge"

    # 2. edge_truncated: annotation attribute or touching image edge.
    if _is_edge_truncated(gt_anns[gj], gt_xyxy[gj], img_w, img_h):
        return "edge_truncated"

    # 3. stacked_tier: stacked vertically with another cartridge in the same column.
    if _is_stacked_tier(gj, gt_xyxy, gt_anns):
        return "stacked_tier"

    return "plain"


def _classify_fp_subtype(
    pi: int,
    pred_xyxy: np.ndarray,
    pred_areas: np.ndarray,
    matched_pred_indices: Sequence[int],
    gt_xyxy: np.ndarray,
    pred_gt_inter: np.ndarray,
) -> str:
    # 1. duplicate: overlaps a matched prediction significantly.
    if len(matched_pred_indices) > 0 and pred_areas[pi] > 0:
        m_idx = np.asarray(matched_pred_indices, dtype=np.int64)
        p_iou = iou_matrix(pred_xyxy[pi : pi + 1], pred_xyxy[m_idx])[0]
        m_areas = pred_areas[m_idx]
        area_ratio = np.minimum(pred_areas[pi], m_areas) / np.maximum(
            np.maximum(pred_areas[pi], m_areas), 1e-9
        )
        if np.any((p_iou >= 0.5) | ((p_iou >= 0.3) & (area_ratio >= 0.5))):
            return "duplicate"

    # 2. fragment: mostly inside a GT box.
    if len(gt_xyxy) > 0 and pred_areas[pi] > 0:
        frag_frac = pred_gt_inter[pi] / pred_areas[pi]
        if np.any(frag_frac >= 0.6):
            return "fragment"

    return "background"


def _normalized_center(box_xywh: Sequence[float], img_w: float, img_h: float) -> tuple[float, float]:
    x, y, w, h = (float(v) for v in box_xywh)
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    nx = min(max(cx / max(img_w, 1.0), 0.0), 1.0)
    ny = min(max(cy / max(img_h, 1.0), 0.0), 1.0)
    return round(nx, 4), round(ny, 4)


def _analyze_with_context(
    gt_coco: Mapping[str, Any],
    preds: Sequence[Mapping[str, Any]],
    score_thr: float,
    iou: float = 0.5,
    *,
    images_root: str | Path | None = None,
    matching: MatchMethod = "greedy",
    loc_iou: float = LOC_DEFAULT_IOU,
) -> tuple[list[FailureCase], list[dict[str, Any]]]:
    """Run failure classification and return ``(failure_cases, per_image_contexts)``."""
    images = gt_coco.get("images", [])
    categories = gt_coco.get("categories", [])
    annotations = gt_coco.get("annotations", [])

    cat_id_to_name: dict[int, str] = {int(c["id"]): str(c["name"]) for c in categories}
    unknown_cat_ids = {cid for cid, name in cat_id_to_name.items() if name.lower() == "unknown"}

    id_to_idx: dict[Any, int] = {}
    str_id_to_idx: dict[str, int] = {}
    for idx, img in enumerate(images):
        raw_id = img["id"]
        id_to_idx[raw_id] = idx
        str_id_to_idx.setdefault(str(raw_id), idx)
        if img.get("file_name"):
            str_id_to_idx.setdefault(str(img["file_name"]), idx)

    gt_by_img: list[list[Mapping[str, Any]]] = [[] for _ in images]
    for ann in annotations:
        idx = id_to_idx.get(ann["image_id"])
        if idx is None:
            idx = str_id_to_idx.get(str(ann["image_id"]))
        if idx is not None:
            gt_by_img[idx].append(ann)
            cid = int(ann["category_id"])
            cat_id_to_name.setdefault(cid, str(cid))

    preds_by_img: list[list[Mapping[str, Any]]] = [[] for _ in images]
    for pred in preds:
        idx = id_to_idx.get(pred["image_id"])
        if idx is None:
            idx = str_id_to_idx.get(str(pred["image_id"]))
        if idx is None:
            raise ValueError(f"Prediction references unknown image: {pred['image_id']}")
        if float(pred["score"]) < score_thr:
            continue
        if idx is not None:
            preds_by_img[idx].append(pred)
            cid = int(pred.get("category_id", -1))
            if cid != -1:
                cat_id_to_name.setdefault(cid, str(cid))

    all_cases: list[FailureCase] = []
    image_contexts: list[dict[str, Any]] = []

    for img_idx, img_rec in enumerate(images):
        img_id = img_rec["id"]
        file_name = str(img_rec["file_name"]) if img_rec.get("file_name") else None
        img_path = _resolve_image_path(img_rec, images_root)
        gt_anns = gt_by_img[img_idx]
        kept_preds = preds_by_img[img_idx]

        pil_img: Image.Image | None = None
        try:
            if img_path is not None:
                pil_img = Image.open(img_path).convert("RGB")

            img_w = float(
                img_rec.get("width")
                or (pil_img.width if pil_img is not None else 0)
                or max((a["bbox"][0] + a["bbox"][2] for a in gt_anns), default=1.0)
            )
            img_h = float(
                img_rec.get("height")
                or (pil_img.height if pil_img is not None else 0)
                or max((a["bbox"][1] + a["bbox"][3] for a in gt_anns), default=1.0)
            )

            gt_boxes = (
                np.asarray([[float(v) for v in a["bbox"]] for a in gt_anns], dtype=np.float64)
                if gt_anns
                else np.zeros((0, 4), dtype=np.float64)
            )
            pred_boxes = (
                np.asarray([[float(v) for v in p["bbox"]] for p in kept_preds], dtype=np.float64)
                if kept_preds
                else np.zeros((0, 4), dtype=np.float64)
            )
            pred_scores = np.asarray([float(p["score"]) for p in kept_preds], dtype=np.float64)

            gt_xyxy = xywh_to_xyxy(gt_boxes)
            pred_xyxy = xywh_to_xyxy(pred_boxes)
            gt_areas = _box_areas_xyxy(gt_xyxy)
            pred_areas = _box_areas_xyxy(pred_xyxy)
            pred_gt_iou = iou_matrix(pred_xyxy, gt_xyxy)
            pred_gt_inter = _intersection_matrix_xyxy(pred_xyxy, gt_xyxy)

            primary = match_boxes(
                pred_boxes,
                gt_boxes,
                iou_thr=iou,
                method=matching,
                pred_scores=pred_scores,
                box_format="xywh",
            )

            tp_records: list[dict[str, Any]] = []
            img_cases: list[FailureCase] = []
            matched_pred_indices: list[int] = []

            for pi, gj, match_iou in primary.matches:
                matched_pred_indices.append(pi)
                gt_cid = int(gt_anns[gj]["category_id"])
                gt_cls = cat_id_to_name.get(gt_cid, str(gt_cid))
                pred_cid = int(kept_preds[pi].get("category_id", -1))
                pred_cls = cat_id_to_name.get(pred_cid, str(pred_cid)) if pred_cid != -1 else None
                gt_b = gt_boxes[gj].tolist()
                pred_b = pred_boxes[pi].tolist()

                if gt_cid in unknown_cat_ids or gt_cid == pred_cid:
                    tp_records.append(
                        {
                            "gt_box": gt_b,
                            "pred_box": pred_b,
                            "score": float(pred_scores[pi]),
                            "label": short_label(pred_cls or gt_cls),
                            "iou": round(float(match_iou), 4),
                        }
                    )
                    continue

                gt_s = short_label(gt_cls)
                pred_s = short_label(pred_cls)
                nx, ny = _normalized_center(pred_b, img_w, img_h)
                img_cases.append(
                    FailureCase(
                        image_id=img_id,
                        file_name=file_name,
                        type="ID",
                        subtype=None,
                        box=pred_b,
                        gt_box=gt_b,
                        pred_box=pred_b,
                        score=float(pred_scores[pi]),
                        id_score=(
                            float(kept_preds[pi]["id_score"])
                            if kept_preds[pi].get("id_score") is not None
                            else None
                        ),
                        iou=round(float(match_iou), 4),
                        gt_category_id=gt_cid,
                        gt_class=gt_cls,
                        gt_short=gt_s,
                        pred_category_id=pred_cid,
                        pred_class=pred_cls,
                        pred_short=pred_s,
                        confusion_pair=f"{gt_s}→{pred_s}",
                        sharpness=_crop_sharpness(pil_img, pred_b),
                        norm_x=nx,
                        norm_y=ny,
                        shelf=_extract_shelf(kept_preds[pi]),
                    )
                )

            # Secondary matching on remaining unmatched boxes for LOC (IoU in [loc_iou, iou)).
            rem_preds = list(primary.unmatched_preds)
            rem_gts = list(primary.unmatched_gts)
            loc_matched_p: set[int] = set()
            loc_matched_g: set[int] = set()

            if rem_preds and rem_gts and loc_iou < iou:
                sec = match_boxes(
                    pred_boxes[rem_preds],
                    gt_boxes[rem_gts],
                    iou_thr=loc_iou,
                    method=matching,
                    pred_scores=pred_scores[rem_preds],
                    box_format="xywh",
                )
                for local_pi, local_gj, sec_iou in sec.matches:
                    pi = rem_preds[local_pi]
                    gj = rem_gts[local_gj]
                    loc_matched_p.add(pi)
                    loc_matched_g.add(gj)
                    matched_pred_indices.append(pi)

                    gt_cid = int(gt_anns[gj]["category_id"])
                    gt_cls = cat_id_to_name.get(gt_cid, str(gt_cid))
                    pred_cid = int(kept_preds[pi].get("category_id", -1))
                    pred_cls = (
                        cat_id_to_name.get(pred_cid, str(pred_cid)) if pred_cid != -1 else None
                    )
                    gt_b = gt_boxes[gj].tolist()
                    pred_b = pred_boxes[pi].tolist()
                    nx, ny = _normalized_center(pred_b, img_w, img_h)
                    img_cases.append(
                        FailureCase(
                            image_id=img_id,
                            file_name=file_name,
                            type="LOC",
                            subtype=None,
                            box=pred_b,
                            gt_box=gt_b,
                            pred_box=pred_b,
                            score=float(pred_scores[pi]),
                            id_score=(
                                float(kept_preds[pi]["id_score"])
                                if kept_preds[pi].get("id_score") is not None
                                else None
                            ),
                            iou=round(float(sec_iou), 4),
                            gt_category_id=gt_cid,
                            gt_class=gt_cls,
                            gt_short=short_label(gt_cls),
                            pred_category_id=pred_cid,
                            pred_class=pred_cls,
                            pred_short=short_label(pred_cls),
                            confusion_pair=None,
                            sharpness=_crop_sharpness(pil_img, pred_b),
                            norm_x=nx,
                            norm_y=ny,
                            shelf=_extract_shelf(kept_preds[pi]),
                        )
                    )

            # Still-unmatched GTs -> FN.
            for gj in rem_gts:
                if gj in loc_matched_g:
                    continue
                subtype = _classify_fn_subtype(
                    gj,
                    gt_xyxy,
                    pred_xyxy,
                    pred_gt_inter,
                    gt_areas,
                    gt_anns,
                    img_w,
                    img_h,
                )
                gt_cid = int(gt_anns[gj]["category_id"])
                gt_cls = cat_id_to_name.get(gt_cid, str(gt_cid))
                gt_b = gt_boxes[gj].tolist()
                max_iou = float(np.max(pred_gt_iou[:, gj])) if len(pred_xyxy) > 0 else 0.0
                nx, ny = _normalized_center(gt_b, img_w, img_h)
                img_cases.append(
                    FailureCase(
                        image_id=img_id,
                        file_name=file_name,
                        type="FN",
                        subtype=subtype,
                        box=gt_b,
                        gt_box=gt_b,
                        pred_box=None,
                        score=None,
                        id_score=None,
                        iou=round(max_iou, 4),
                        gt_category_id=gt_cid,
                        gt_class=gt_cls,
                        gt_short=short_label(gt_cls),
                        pred_category_id=None,
                        pred_class=None,
                        pred_short=None,
                        confusion_pair=None,
                        sharpness=_crop_sharpness(pil_img, gt_b),
                        norm_x=nx,
                        norm_y=ny,
                        shelf=None,
                    )
                )

            # Still-unmatched predictions -> FP.
            for pi in rem_preds:
                if pi in loc_matched_p:
                    continue
                subtype = _classify_fp_subtype(
                    pi,
                    pred_xyxy,
                    pred_areas,
                    matched_pred_indices,
                    gt_xyxy,
                    pred_gt_inter,
                )
                pred_cid = int(kept_preds[pi].get("category_id", -1))
                pred_cls = cat_id_to_name.get(pred_cid, str(pred_cid)) if pred_cid != -1 else None
                pred_b = pred_boxes[pi].tolist()
                max_iou = float(np.max(pred_gt_iou[pi, :])) if len(gt_xyxy) > 0 else 0.0
                nx, ny = _normalized_center(pred_b, img_w, img_h)
                img_cases.append(
                    FailureCase(
                        image_id=img_id,
                        file_name=file_name,
                        type="FP",
                        subtype=subtype,
                        box=pred_b,
                        gt_box=None,
                        pred_box=pred_b,
                        score=float(pred_scores[pi]),
                        id_score=(
                            float(kept_preds[pi]["id_score"])
                            if kept_preds[pi].get("id_score") is not None
                            else None
                        ),
                        iou=round(max_iou, 4),
                        gt_category_id=None,
                        gt_class=None,
                        gt_short=None,
                        pred_category_id=pred_cid,
                        pred_class=pred_cls,
                        pred_short=short_label(pred_cls),
                        confusion_pair=None,
                        sharpness=_crop_sharpness(pil_img, pred_b),
                        norm_x=nx,
                        norm_y=ny,
                        shelf=_extract_shelf(kept_preds[pi]),
                    )
                )

            all_cases.extend(img_cases)
            gt_records = [
                {
                    "bbox": [float(v) for v in a["bbox"]],
                    "label": short_label(cat_id_to_name.get(int(a["category_id"]))),
                }
                for a in gt_anns
            ]
            image_contexts.append(
                {
                    "image_id": img_id,
                    "file_name": file_name,
                    "path": img_path,
                    "width": int(round(img_w)),
                    "height": int(round(img_h)),
                    "n_pred_kept": len(kept_preds),
                    "gt_records": gt_records,
                    "tp_records": tp_records,
                    "cases": img_cases,
                }
            )
        finally:
            if pil_img is not None:
                pil_img.close()

    return all_cases, image_contexts


def analyze(
    gt_coco: Mapping[str, Any],
    preds: Sequence[Mapping[str, Any]],
    score_thr: float,
    iou: float = 0.5,
    *,
    images_root: str | Path | None = None,
    matching: MatchMethod = "greedy",
    loc_iou: float = LOC_DEFAULT_IOU,
) -> list[FailureCase]:
    """Classify every detection/identification failure on *gt_coco* vs *preds*."""
    cases, _ = _analyze_with_context(
        gt_coco,
        preds,
        score_thr,
        iou=iou,
        images_root=images_root,
        matching=matching,
        loc_iou=loc_iou,
    )
    return cases


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    xyxy: Sequence[float],
    color: str,
    width: int = 2,
) -> None:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str,
    width: int = 3,
    dash: int = 10,
    gap: int = 6,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line(
            [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)],
            fill=color,
            width=width,
        )
        pos += dash + gap


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    xyxy: Sequence[float],
    color: str,
    width: int = 3,
    dash: int = 10,
    gap: int = 6,
) -> None:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    _draw_dashed_line(draw, (x1, y1), (x2, y1), color, width=width, dash=dash, gap=gap)
    _draw_dashed_line(draw, (x2, y1), (x2, y2), color, width=width, dash=dash, gap=gap)
    _draw_dashed_line(draw, (x2, y2), (x1, y2), color, width=width, dash=dash, gap=gap)
    _draw_dashed_line(draw, (x1, y2), (x1, y1), color, width=width, dash=dash, gap=gap)


def _draw_label_pill(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    bg_color: str,
    font: Any,
    img_w: int,
    img_h: int,
) -> None:
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad_x, pad_y = 4, 3
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y
    px = max(0.0, min(float(x), float(max(0, img_w - box_w))))
    py = float(y) - box_h
    if py < 0.0:
        py = min(float(y), float(max(0, img_h - box_h)))
    draw.rectangle([px, py, px + box_w, py + box_h], fill=bg_color)
    draw.text((px + pad_x, py + pad_y - tb[1]), text, fill="white", font=font)


def draw_image_overlay(
    image: Image.Image,
    gt_records: Sequence[Mapping[str, Any]],
    tp_records: Sequence[Mapping[str, Any]],
    cases: Sequence[FailureCase],
) -> Image.Image:
    """Render GT, TP, and failure boxes onto a copy of *image* at original resolution."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    font_size = max(12, min(24, round(min(w, h) * 0.022)))
    font = _font(font_size)
    thin_w = 1 if max(w, h) < 900 else 2
    thick_w = 2 if max(w, h) < 900 else 3

    # 1. GT boxes in thin green.
    for gt in gt_records:
        x, y, bw, bh = gt["bbox"]
        _draw_rect(draw, [x, y, x + bw, y + bh], COLOR_GT, width=thin_w)

    # 2. TP boxes in blue.
    for tp in tp_records:
        x, y, bw, bh = tp["pred_box"]
        _draw_rect(draw, [x, y, x + bw, y + bh], COLOR_TP, width=thin_w)
        _draw_label_pill(
            draw, x, y, f"TP:{tp['label']} {tp['score']:.2f}", COLOR_TP, font, w, h
        )

    # 3. Failure boxes on top.
    for case in cases:
        x, y, bw, bh = case.box
        xyxy = [x, y, x + bw, y + bh]
        if case.type == "FN":
            _draw_dashed_rect(draw, xyxy, COLOR_FN, width=thick_w)
            lbl = f"FN:{case.gt_short or '?'} ({case.subtype})"
            _draw_label_pill(draw, x, y + bh, lbl, COLOR_FN, font, w, h)
        elif case.type == "FP":
            _draw_rect(draw, xyxy, COLOR_FP, width=thick_w)
            lbl = f"FP:{case.pred_short or '?'} {case.score:.2f} ({case.subtype})"
            _draw_label_pill(draw, x, y, lbl, COLOR_FP, font, w, h)
        elif case.type == "ID":
            _draw_rect(draw, xyxy, COLOR_ID, width=thick_w)
            lbl = case.confusion_pair or f"{case.gt_short}→{case.pred_short}"
            if case.score is not None:
                lbl = f"{lbl} {case.score:.2f}"
            _draw_label_pill(draw, x, y, lbl, COLOR_ID, font, w, h)
        elif case.type == "LOC":
            _draw_rect(draw, xyxy, COLOR_LOC, width=thick_w)
            iou_str = f" iou={case.iou:.2f}" if case.iou is not None else ""
            lbl = f"LOC:{case.pred_short or '?'}{iou_str}"
            _draw_label_pill(draw, x, y, lbl, COLOR_LOC, font, w, h)

    return canvas


def _case_caption_lines(case: FailureCase) -> tuple[str, str]:
    if case.type == "ID":
        head = f"{case.confusion_pair or f'{case.gt_short}→{case.pred_short}'}"
        if case.score is not None:
            head += f"  s={case.score:.2f}"
    elif case.type == "FN":
        head = f"FN:{case.gt_short or '?'} [{case.subtype}]"
    elif case.type == "FP":
        s_str = f" s={case.score:.2f}" if case.score is not None else ""
        head = f"FP:{case.pred_short or '?'}{s_str} [{case.subtype}]"
    else:
        iou_str = f" iou={case.iou:.2f}" if case.iou is not None else ""
        head = f"LOC:{case.pred_short or '?'}{iou_str}"

    sharp_str = f"lap={case.sharpness:.0f}" if case.sharpness is not None else "lap=-"
    shelf_str = f" S{case.shelf}" if case.shelf is not None else ""
    meta = f"img={case.image_id} {sharp_str} ({case.norm_x:.2f},{case.norm_y:.2f}){shelf_str}"
    return head, meta


def _render_cropped_tile(
    orig_img: Image.Image,
    case: FailureCase,
    thumb_size: int = 240,
) -> Image.Image:
    """Crop a 2x-context patch from original pixels and outline the failure box."""
    x, y, w, h = case.box
    x0, y0, x1, y1 = x, y, x + max(w, 1.0), y + max(h, 1.0)
    dx, dy = (x1 - x0) * GALLERY_PAD_CONTEXT, (y1 - y0) * GALLERY_PAD_CONTEXT
    rect = (
        max(0, math.floor(x0 - dx)),
        max(0, math.floor(y0 - dy)),
        min(orig_img.width, math.ceil(x1 + dx)),
        min(orig_img.height, math.ceil(y1 + dy)),
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        patch = orig_img.copy()
        rect = (0, 0, orig_img.width, orig_img.height)
    else:
        patch = crop(orig_img, [x0, y0, x1, y1], pad=GALLERY_PAD_CONTEXT)

    draw = ImageDraw.Draw(patch)
    color_map = {
        "FN": COLOR_FN,
        "FP": COLOR_FP,
        "ID": COLOR_ID,
        "LOC": COLOR_LOC,
    }
    box_color = color_map.get(case.type, COLOR_FP)
    local_box = [x0 - rect[0], y0 - rect[1], x1 - rect[0], y1 - rect[1]]
    if case.type == "FN":
        _draw_dashed_rect(draw, local_box, box_color, width=2)
    else:
        _draw_rect(draw, local_box, box_color, width=2)

    if case.gt_box is not None and case.type in ("ID", "LOC"):
        gx, gy, gw, gh = case.gt_box
        _draw_rect(
            draw,
            [gx - rect[0], gy - rect[1], gx + gw - rect[0], gy + gh - rect[1]],
            COLOR_GT,
            width=1,
        )

    patch.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    return patch


def build_crop_gallery(
    failure_type: FailureType,
    cases: Sequence[FailureCase],
    image_paths: Mapping[Any, Path | None],
    out_path: Path,
    *,
    thumb_size: int = 240,
    cols: int = 4,
    max_items: int = 64,
) -> Path:
    """Write a captioned crop-gallery PNG for one failure type from original pixels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset = list(cases)[:max_items]
    title_font = _font(16)
    cap_font = _font(12)

    if not subset:
        sheet = Image.new("RGB", (640, 96), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 36), f"{failure_type} Gallery — 0 failure cases", fill="#334155", font=title_font)
        sheet.save(out_path)
        return out_path

    pad = 10
    header_h = 36
    cap_h = 38
    n_cols = min(cols, max(1, len(subset)))
    n_rows = math.ceil(len(subset) / n_cols)
    cell_w = thumb_size
    cell_h = thumb_size + cap_h
    sheet_w = pad + n_cols * (cell_w + pad)
    sheet_h = header_h + pad + n_rows * (cell_h + pad)

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    suffix = f" (showing first {len(subset)} of {len(cases)})" if len(cases) > len(subset) else ""
    draw.text(
        (pad, 10),
        f"{failure_type} Failure Crops ({len(cases)} total){suffix}",
        fill="#0F172A",
        font=title_font,
    )

    img_cache: dict[str, Image.Image] = {}
    try:
        for idx, case in enumerate(subset):
            col = idx % n_cols
            row = idx // n_cols
            x = pad + col * (cell_w + pad)
            y = header_h + pad + row * (cell_h + pad)

            img_key = str(case.image_id)
            if img_key not in img_cache:
                p = image_paths.get(case.image_id) or image_paths.get(img_key)
                if p is not None and p.is_file():
                    img_cache[img_key] = Image.open(p).convert("RGB")
            src_img = img_cache.get(img_key)

            draw.rectangle([x, y, x + cell_w, y + thumb_size], fill="#F1F5F9", outline="#CBD5E1")
            if src_img is not None:
                tile = _render_cropped_tile(src_img, case, thumb_size=thumb_size)
                sheet.paste(
                    tile,
                    (x + (cell_w - tile.width) // 2, y + (thumb_size - tile.height) // 2),
                )
            else:
                draw.text((x + 12, y + thumb_size // 2 - 8), "no image", fill="#64748B", font=cap_font)

            line1, line2 = _case_caption_lines(case)
            draw.text((x + 2, y + thumb_size + 4), line1[:36], fill="#0F172A", font=cap_font)
            draw.text((x + 2, y + thumb_size + 20), line2[:36], fill="#475569", font=cap_font)
    finally:
        for im in img_cache.values():
            im.close()

    sheet.save(out_path)
    return out_path


def build_confusion_matrix(cases: Sequence[FailureCase]) -> dict[str, Any]:
    """Compute ID error confusion matrix over short labels."""
    id_cases = [c for c in cases if c.type == "ID"]
    labels = sorted(
        {c.gt_short for c in id_cases if c.gt_short}
        | {c.pred_short for c in id_cases if c.pred_short}
    )
    idx_map = {lbl: i for i, lbl in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    pairs: dict[str, int] = {}

    for c in id_cases:
        g = c.gt_short or "?"
        p = c.pred_short or "?"
        pair_key = f"{g}→{p}"
        pairs[pair_key] = pairs.get(pair_key, 0) + 1
        if g in idx_map and p in idx_map:
            matrix[idx_map[g]][idx_map[p]] += 1

    return {"labels": labels, "matrix": matrix, "pairs": dict(sorted(pairs.items()))}


def _safe_stem(file_name: str | None, image_id: Any, used: set[str]) -> str:
    raw = str(file_name or image_id)
    stem = str(Path(raw).with_suffix(""))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or f"img_{image_id}"
    if safe in used:
        safe = f"{safe}_{image_id}"
    used.add(safe)
    return safe


def render_html_report(report: Mapping[str, Any]) -> str:
    """Render a self-contained local HTML report (no external assets or CDNs)."""
    summary = report["summary"]
    counts = summary["counts_by_type"]
    subtypes = summary["counts_by_subtype"]
    cm = summary["confusion_matrix"]
    images = report["images"]
    galleries = report["galleries"]
    failures = report["failures"]

    # Confusion matrix HTML table.
    cm_labels = cm["labels"]
    cm_matrix = cm["matrix"]
    if cm_labels:
        header_cells = "".join(f"<th>{html.escape(lbl)}</th>" for lbl in cm_labels)
        body_rows = []
        for r_idx, gt_lbl in enumerate(cm_labels):
            row_cells = []
            for c_idx, val in enumerate(cm_matrix[r_idx]):
                cls = "cm-hit" if val > 0 else "cm-zero"
                row_cells.append(f'<td class="{cls}">{val}</td>')
            body_rows.append(f"<tr><th>{html.escape(gt_lbl)}</th>{''.join(row_cells)}</tr>")
        cm_html = (
            "<table><thead><tr><th>GT \\ Pred</th>"
            f"{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        )
    else:
        cm_html = "<p class='muted'>No ID confusion errors at this threshold.</p>"

    # Subtype summary rows.
    subtype_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(subtypes.items())
    ) or "<tr><td colspan='2' class='muted'>No failures</td></tr>"

    # Gallery cards.
    gallery_cards = []
    for ftype in FAILURE_TYPES:
        rel_path = galleries.get(ftype)
        if not rel_path:
            continue
        cnt = counts.get(ftype, 0)
        gallery_cards.append(
            f"<div class='card'><h3>{ftype} ({cnt})</h3>"
            f"<a href='{html.escape(rel_path)}'>"
            f"<img src='{html.escape(rel_path)}' alt='{ftype} gallery'></a></div>"
        )

    # Overlay cards.
    overlay_cards = []
    for img in images:
        prev = img.get("preview_path")
        full = img.get("overlay_path")
        c = img.get("counts", {})
        stats = (
            f"TP={img.get('n_tp', 0)} · FN={c.get('FN', 0)} · "
            f"FP={c.get('FP', 0)} · ID={c.get('ID', 0)} · LOC={c.get('LOC', 0)}"
        )
        img_tag = (
            f"<a href='{html.escape(full)}'><img src='{html.escape(prev)}' "
            f"alt='{html.escape(str(img.get('file_name') or img['image_id']))}'></a>"
            if prev and full
            else "<p class='muted'>Image file not found on disk.</p>"
        )
        link_tag = (
            f" · <a href='{html.escape(full)}'>Full resolution ({img['width']}×{img['height']})</a>"
            if full
            else ""
        )
        overlay_cards.append(
            f"<div class='card'><h3>{html.escape(str(img.get('file_name') or img['image_id']))}"
            f"</h3><p class='muted'>{stats}{link_tag}</p>{img_tag}</div>"
        )

    # Case rows (capped at 500 in HTML to keep local page snappy; full data in failures.json).
    case_rows = []
    for f in failures[:500]:
        pair_or_sub = f.get("confusion_pair") or f.get("subtype") or "-"
        lbl = f.get("gt_short") or f.get("pred_short") or "-"
        score_s = f"{f['score']:.2f}" if f.get("score") is not None else "-"
        iou_s = f"{f['iou']:.2f}" if f.get("iou") is not None else "-"
        sharp_s = f"{f['sharpness']:.1f}" if f.get("sharpness") is not None else "-"
        pos_s = f"({f['norm_x']:.2f}, {f['norm_y']:.2f})"
        shelf_s = str(f["shelf"]) if f.get("shelf") is not None else "-"
        case_rows.append(
            "<tr>"
            f"<td>{html.escape(str(f['image_id']))}</td>"
            f"<td><span class='badge b-{html.escape(f['type'])}'>{html.escape(f['type'])}</span></td>"
            f"<td>{html.escape(str(pair_or_sub))}</td>"
            f"<td>{html.escape(str(lbl))}</td>"
            f"<td>{score_s}</td>"
            f"<td>{iou_s}</td>"
            f"<td>{sharp_s}</td>"
            f"<td>{pos_s}</td>"
            f"<td>{html.escape(shelf_s)}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Freestyle Failure Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #0f172a; background: #f8fafc; }}
  h1, h2, h3 {{ margin-top: 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .stat {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; }}
  .stat .num {{ font-size: 24px; font-weight: 700; }}
  .stat .lbl {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .card img {{ max-width: 100%; height: auto; border: 1px solid #cbd5e1; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; margin-bottom: 20px; font-size: 13px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }}
  th {{ background: #f1f5f9; }}
  .cm-hit {{ background: #fae8ff; font-weight: 700; color: #86198f; }}
  .cm-zero {{ color: #94a3b8; }}
  .muted {{ color: #64748b; font-size: 13px; }}
  .badge {{ padding: 2px 6px; border-radius: 4px; font-weight: 600; color: #fff; font-size: 11px; }}
  .b-FN {{ background: {COLOR_FN}; }}
  .b-FP {{ background: {COLOR_FP}; }}
  .b-ID {{ background: {COLOR_ID}; }}
  .b-LOC {{ background: {COLOR_LOC}; }}
  .legend span {{ display: inline-block; margin-right: 12px; font-size: 13px; }}
</style>
</head>
<body>
<h1>Freestyle Cartridge Failure Analysis</h1>
<p class="muted">
  score_thr={summary['score_thr']} · iou_thr={summary['iou_thr']} ·
  images={summary['n_images']} · GT={summary['n_gt']} · kept_preds={summary['n_pred_kept']} ·
  TP={summary['n_tp']} · <a href="failures.json">failures.json</a>
</p>
<div class="legend">
  <span style="color:{COLOR_GT}">■ GT (thin green)</span>
  <span style="color:{COLOR_TP}">■ TP (blue)</span>
  <span style="color:{COLOR_FP}">■ FP (red)</span>
  <span style="color:{COLOR_FN}">■ FN (orange dashed)</span>
  <span style="color:{COLOR_ID}">■ ID (magenta gt→pred)</span>
  <span style="color:{COLOR_LOC}">■ LOC (cyan IoU 0.3–0.5)</span>
</div>

<div class="grid" style="margin-top: 16px;">
  <div class="stat"><div class="num">{summary['n_failures']}</div><div class="lbl">Total Errors</div></div>
  <div class="stat"><div class="num">{counts.get('FN', 0)}</div><div class="lbl">FN (Missed GT)</div></div>
  <div class="stat"><div class="num">{counts.get('FP', 0)}</div><div class="lbl">FP (False Pos)</div></div>
  <div class="stat"><div class="num">{counts.get('ID', 0)}</div><div class="lbl">ID (Wrong Class)</div></div>
  <div class="stat"><div class="num">{counts.get('LOC', 0)}</div><div class="lbl">LOC (IoU 0.3–0.5)</div></div>
</div>

<h2>Sub-type Breakdown</h2>
<table><thead><tr><th>Type:Subtype</th><th>Count</th></tr></thead><tbody>{subtype_rows}</tbody></table>

<h2>ID Confusion Matrix (GT row → Pred col)</h2>
{cm_html}

<h2>Crop Galleries (2× Context from Original Pixels)</h2>
{''.join(gallery_cards)}

<h2>Per-Image Overlays</h2>
{''.join(overlay_cards)}

<h2>Failure Cases ({len(failures)})</h2>
<table>
  <thead>
    <tr><th>Image</th><th>Type</th><th>Sub-type / Pair</th><th>Label</th><th>Score</th><th>IoU</th><th>Sharpness</th><th>Pos (x,y)</th><th>Shelf</th></tr>
  </thead>
  <tbody>{''.join(case_rows)}</tbody>
</table>
</body>
</html>
"""


def run_failure_analysis(
    gt_coco: Mapping[str, Any],
    preds: Sequence[Mapping[str, Any]],
    *,
    images_root: str | Path | None = None,
    score_thr: float = 0.5,
    iou: float = 0.5,
    out_dir: str | Path,
    matching: MatchMethod = "greedy",
    max_gallery_crops: int = 64,
) -> dict[str, Any]:
    """Run full failure analysis and write overlays, galleries, JSON, and HTML to *out_dir*."""
    out = Path(out_dir)
    overlays_dir = out / "overlays"
    galleries_dir = out / "galleries"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    galleries_dir.mkdir(parents=True, exist_ok=True)

    cases, contexts = _analyze_with_context(
        gt_coco,
        preds,
        score_thr,
        iou=iou,
        images_root=images_root,
        matching=matching,
    )

    used_stems: set[str] = set()
    image_paths: dict[Any, Path | None] = {}
    image_summaries: list[dict[str, Any]] = []
    n_gt = 0
    n_tp = 0

    for ctx in contexts:
        img_id = ctx["image_id"]
        file_name = ctx["file_name"]
        img_path = ctx["path"]
        image_paths[img_id] = img_path
        image_paths[str(img_id)] = img_path

        n_gt += len(ctx["gt_records"])
        n_tp += len(ctx["tp_records"])
        per_type_counts = {t: sum(1 for c in ctx["cases"] if c.type == t) for t in FAILURE_TYPES}

        overlay_rel: str | None = None
        preview_rel: str | None = None
        if img_path is not None and img_path.is_file():
            stem = _safe_stem(file_name, img_id, used_stems)
            overlay_path = overlays_dir / f"{stem}.png"
            preview_path = overlays_dir / f"{stem}_preview.png"
            with Image.open(img_path) as raw:
                overlay = draw_image_overlay(
                    raw, ctx["gt_records"], ctx["tp_records"], ctx["cases"]
                )
                overlay.save(overlay_path)
                prev_h = max(1, round(overlay.height * (PREVIEW_WIDTH / max(overlay.width, 1))))
                preview = overlay.resize((PREVIEW_WIDTH, prev_h), Image.Resampling.LANCZOS)
                preview.save(preview_path)
            overlay_rel = f"overlays/{overlay_path.name}"
            preview_rel = f"overlays/{preview_path.name}"

        image_summaries.append(
            {
                "image_id": img_id,
                "file_name": file_name,
                "width": ctx["width"],
                "height": ctx["height"],
                "n_gt": len(ctx["gt_records"]),
                "n_tp": len(ctx["tp_records"]),
                "counts": per_type_counts,
                "overlay_path": overlay_rel,
                "preview_path": preview_rel,
            }
        )

    gallery_paths: dict[str, str] = {}
    for ftype in FAILURE_TYPES:
        subset = [c for c in cases if c.type == ftype]
        g_path = galleries_dir / f"{ftype}.png"
        build_crop_gallery(ftype, subset, image_paths, g_path, max_items=max_gallery_crops)
        gallery_paths[ftype] = f"galleries/{g_path.name}"

    counts_by_type = {t: sum(1 for c in cases if c.type == t) for t in FAILURE_TYPES}
    counts_by_subtype: dict[str, int] = {}
    for c in cases:
        key = f"{c.type}:{c.subtype}" if c.subtype else c.type
        counts_by_subtype[key] = counts_by_subtype.get(key, 0) + 1

    n_pred_kept = sum(int(ctx["n_pred_kept"]) for ctx in contexts)
    report: dict[str, Any] = {
        "summary": {
            "score_thr": float(score_thr),
            "iou_thr": float(iou),
            "n_images": len(contexts),
            "n_gt": n_gt,
            "n_pred_kept": n_pred_kept,
            "n_tp": n_tp,
            "n_failures": len(cases),
            "counts_by_type": counts_by_type,
            "counts_by_subtype": dict(sorted(counts_by_subtype.items())),
            "confusion_matrix": build_confusion_matrix(cases),
        },
        "galleries": gallery_paths,
        "images": image_summaries,
        "failures": [c.to_dict() for c in cases],
    }

    (out / "failures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "index.html").write_text(render_html_report(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze Freestyle cartridge failure modes and render visual diagnostics."
    )
    p.add_argument("--gt", required=True, help="Ground-truth COCO JSON file.")
    p.add_argument("--pred", required=True, help="Predictions COCO results JSON list.")
    p.add_argument(
        "--images-root",
        default=None,
        help="Root directory prepended to image file_name values.",
    )
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.5,
        help="Prediction confidence threshold (default 0.5).",
    )
    p.add_argument("--iou", type=float, default=0.5, help="Primary IoU threshold (default 0.5).")
    p.add_argument("--out", required=True, help="Output directory for overlays, galleries, and HTML.")
    p.add_argument(
        "--matching",
        choices=("greedy", "hungarian"),
        default="greedy",
        help="1-to-1 matching method (default greedy).",
    )
    p.add_argument(
        "--max-gallery-crops",
        type=int,
        default=64,
        help="Maximum crops per failure gallery PNG (default 64).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gt_coco = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    preds = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    report = run_failure_analysis(
        gt_coco,
        preds,
        images_root=args.images_root,
        score_thr=args.score_thr,
        iou=args.iou,
        out_dir=args.out,
        matching=args.matching,
        max_gallery_crops=args.max_gallery_crops,
    )
    s = report["summary"]
    print(
        f"Failure analysis -> {args.out}: "
        f"TP={s['n_tp']} FN={s['counts_by_type']['FN']} "
        f"FP={s['counts_by_type']['FP']} ID={s['counts_by_type']['ID']} "
        f"LOC={s['counts_by_type']['LOC']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
