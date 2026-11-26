"""Evaluation metrics and group bootstrap CIs.

Inventory vectors contain catalog classes and unknown; generic cartridge counts
as unknown. Exact inventory requires every entry to match; total exact compares totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from freestyle.eval.matching import MatchMethod, match_boxes

COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
SPECIAL_CLASS_NAMES = frozenset({"unknown", "cartridge"})


@dataclass
class _PreparedDataset:
    """Pre-indexed GT and prediction records grouped by COCO image."""

    image_keys: list[str]
    image_groups: dict[str, str]
    cat_id_to_name: dict[int, str]
    unknown_cat_ids: set[int]
    cartridge_cat_ids: set[int]
    catalog_cat_ids: list[int]
    gt_boxes_by_img: list[np.ndarray]
    gt_cats_by_img: list[list[int]]
    pred_boxes_by_img: list[np.ndarray]
    pred_scores_by_img: list[np.ndarray]
    pred_cats_by_img: list[list[int]]
    n_pred_ignored_unknown_image: int


def _resolve_image_key(img: Mapping[str, Any]) -> str:
    if "file_name" in img and img["file_name"]:
        return str(img["file_name"])
    return str(img["id"])


def _match_split_group_from_name(image_name: str, split_groups: Mapping[str, Any]) -> str | None:
    """Resolve group id for a file_name against ``splits.json['groups']``."""
    if image_name in split_groups:
        return image_name
    parts = [p for p in image_name.split("/") if p]
    for i in range(len(parts), 0, -1):
        prefix = "/".join(parts[:i])
        if prefix in split_groups:
            return prefix
    for seg in parts:
        if seg in split_groups:
            return seg
    return None


def load_groups_map(
    gt_coco: Mapping[str, Any],
    groups_data: Mapping[str, Any] | None = None,
    allow_image_groups: bool = False,
) -> dict[str, str]:
    """Build an ``image_key -> group_id`` mapping for all images in ``gt_coco``.

    Supports:
    - ``None``: uses embedded group IDs; missing IDs require explicit opt-in.
    - ``data/splits.json`` schema (``{"version": 1, "groups": {...}, "images": {...}}``):
      matches each image's ``file_name`` (or ``id``) against ``groups`` keys.
    - Explicit mapping JSON: ``{"image_to_group": {...}}`` or flat ``{image_id_or_file_name: group_id}``.
    """
    images = gt_coco.get("images", [])
    explicit_map: Mapping[str, Any] | None = None
    split_groups: Mapping[str, Any] | None = None

    if groups_data is not None:
        if "image_to_group" in groups_data and isinstance(groups_data["image_to_group"], Mapping):
            explicit_map = groups_data["image_to_group"]
        elif "image_groups" in groups_data and isinstance(groups_data["image_groups"], Mapping):
            explicit_map = groups_data["image_groups"]
        elif "groups" in groups_data and "images" in groups_data:
            split_groups = groups_data["groups"]
        else:
            explicit_map = groups_data

    out: dict[str, str] = {}
    for img in images:
        key = _resolve_image_key(img)
        raw_id = str(img["id"])
        file_name = str(img.get("file_name") or "")

        group_id: str | None = None
        if explicit_map is not None:
            for candidate in (key, file_name, raw_id):
                if candidate and candidate in explicit_map:
                    group_id = str(explicit_map[candidate])
                    break
        elif split_groups is not None:
            for candidate in (file_name, key, raw_id):
                if candidate:
                    matched = _match_split_group_from_name(candidate, split_groups)
                    if matched is not None:
                        group_id = matched
                        break

        if not group_id:
            attrs = img.get("attributes") if isinstance(img.get("attributes"), Mapping) else {}
            if "group_id" in img and img["group_id"] is not None:
                group_id = str(img["group_id"])
            elif "group_id" in attrs and attrs["group_id"] is not None:
                group_id = str(attrs["group_id"])
            else:
                if not allow_image_groups:
                    raise ValueError(f"Missing group for image {key}")
                group_id = key
        if not group_id:
            raise ValueError(f"Missing group for image {key}")
        out[key] = group_id
    return out


def _prepare_dataset(
    gt_coco: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    groups_data: Mapping[str, Any] | None = None,
    allow_unknown_images: bool = False,
) -> _PreparedDataset:
    images = gt_coco.get("images", [])
    categories = gt_coco.get("categories", [])
    annotations = gt_coco.get("annotations", [])

    cat_id_to_name: dict[int, str] = {}
    for cat in categories:
        cat_id_to_name[int(cat["id"])] = str(cat["name"])

    unknown_cat_ids = {cid for cid, name in cat_id_to_name.items() if name.lower() == "unknown"}
    cartridge_cat_ids = {cid for cid, name in cat_id_to_name.items() if name.lower() == "cartridge"}

    id_to_idx: dict[Any, int] = {}
    str_id_to_idx: dict[str, int] = {}
    image_keys: list[str] = []
    for idx, img in enumerate(images):
        raw_id = img["id"]
        id_to_idx[raw_id] = idx
        str_id_to_idx.setdefault(str(raw_id), idx)
        if img.get("file_name"):
            str_id_to_idx.setdefault(str(img["file_name"]), idx)
        image_keys.append(_resolve_image_key(img))

    n_images = len(images)
    gt_boxes_raw: list[list[list[float]]] = [[] for _ in range(n_images)]
    gt_cats_raw: list[list[int]] = [[] for _ in range(n_images)]
    for ann in annotations:
        img_idx = id_to_idx.get(ann["image_id"])
        if img_idx is None:
            img_idx = str_id_to_idx.get(str(ann["image_id"]))
        if img_idx is None:
            continue
        gt_boxes_raw[img_idx].append([float(v) for v in ann["bbox"]])
        cid = int(ann["category_id"])
        gt_cats_raw[img_idx].append(cid)
        cat_id_to_name.setdefault(cid, str(cid))

    pred_boxes_raw: list[list[list[float]]] = [[] for _ in range(n_images)]
    pred_scores_raw: list[list[float]] = [[] for _ in range(n_images)]
    pred_cats_raw: list[list[int]] = [[] for _ in range(n_images)]
    n_ignored = 0
    for pred in predictions:
        img_idx = id_to_idx.get(pred["image_id"])
        if img_idx is None:
            img_idx = str_id_to_idx.get(str(pred["image_id"]))
        if img_idx is None:
            n_ignored += 1
            continue
        pred_boxes_raw[img_idx].append([float(v) for v in pred["bbox"]])
        pred_scores_raw[img_idx].append(float(pred["score"]))
        cid = int(pred.get("category_id", -1))
        pred_cats_raw[img_idx].append(cid)
        if cid != -1:
            cat_id_to_name.setdefault(cid, str(cid))

    if n_ignored and not allow_unknown_images:
        raise ValueError(f"{n_ignored} predictions reference unknown images")

    # Catalog classes exclude special classes ('unknown', 'cartridge') unless none remain.
    non_special = sorted(
        cid
        for cid, name in cat_id_to_name.items()
        if name.lower() not in SPECIAL_CLASS_NAMES and cid != -1
    )
    catalog_cat_ids = non_special

    gt_boxes_by_img = [
        np.asarray(b, dtype=np.float64).reshape(-1, 4) if b else np.zeros((0, 4), dtype=np.float64)
        for b in gt_boxes_raw
    ]
    pred_boxes_by_img = [
        np.asarray(b, dtype=np.float64).reshape(-1, 4) if b else np.zeros((0, 4), dtype=np.float64)
        for b in pred_boxes_raw
    ]
    pred_scores_by_img = [
        np.asarray(s, dtype=np.float64) if s else np.zeros((0,), dtype=np.float64)
        for s in pred_scores_raw
    ]

    image_groups = load_groups_map(gt_coco, groups_data, allow_image_groups=True)
    return _PreparedDataset(
        image_keys=image_keys,
        image_groups=image_groups,
        cat_id_to_name=cat_id_to_name,
        unknown_cat_ids=unknown_cat_ids,
        cartridge_cat_ids=cartridge_cat_ids,
        catalog_cat_ids=catalog_cat_ids,
        gt_boxes_by_img=gt_boxes_by_img,
        gt_cats_by_img=gt_cats_raw,
        pred_boxes_by_img=pred_boxes_by_img,
        pred_scores_by_img=pred_scores_by_img,
        pred_cats_by_img=pred_cats_raw,
        n_pred_ignored_unknown_image=n_ignored,
    )


def _compute_coco_ap_from_records(
    scores: np.ndarray,
    is_tp: np.ndarray,
    n_gt: int,
) -> float | None:
    """Compute 101-point COCO interpolated AP from (scores, is_tp) records."""
    if n_gt == 0:
        return None if len(scores) == 0 else 0.0
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    tp = is_tp[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / float(n_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
    # Monotonic non-increasing precision envelope from right to left.
    prec_env = np.maximum.accumulate(precisions[::-1])[::-1]

    inds = np.searchsorted(recalls, COCO_RECALL_THRESHOLDS, side="left")
    q = np.zeros(len(COCO_RECALL_THRESHOLDS), dtype=np.float64)
    valid = inds < len(prec_env)
    q[valid] = prec_env[inds[valid]]
    return float(np.mean(q))


def find_best_f1_threshold(
    scores: np.ndarray,
    is_tp: np.ndarray,
    n_gt: int,
) -> dict[str, float | None]:
    """Select the score threshold maximizing F1 at IoU=0.5."""
    if n_gt == 0 and len(scores) == 0:
        return {"threshold": 0.5, "f1": None, "precision": None, "recall": None}
    if len(scores) == 0:
        return {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    if n_gt == 0:
        return {"threshold": float(np.max(scores)) + 1e-6, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_tp = is_tp[order].astype(np.int64)
    tp_cum = np.cumsum(sorted_tp)
    fp_cum = np.cumsum(1 - sorted_tp)

    # Evaluate at the last index of each distinct score so tied scores move together.
    distinct_mask = np.empty(len(sorted_scores), dtype=bool)
    distinct_mask[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    distinct_mask[-1] = True
    idxs = np.flatnonzero(distinct_mask)

    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = float(sorted_scores[0])

    for idx in idxs.tolist():
        tp = int(tp_cum[idx])
        fp = int(fp_cum[idx])
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(n_gt)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        thr = float(sorted_scores[idx])
        if f1 > best_f1 or (
            abs(f1 - best_f1) <= 1e-12 and (rec > best_rec or (rec == best_rec and thr > best_thr))
        ):
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_thr = thr

    return {
        "threshold": float(best_thr),
        "f1": float(best_f1),
        "precision": float(best_prec),
        "recall": float(best_rec),
    }


def _operating_point_from_records(
    scores: np.ndarray,
    is_tp: np.ndarray,
    n_gt: int,
    score_thr: float,
) -> dict[str, Any]:
    keep = scores >= score_thr
    tp = int(np.sum(is_tp[keep]))
    fp = int(np.sum(1 - is_tp[keep]))
    fn = int(n_gt - tp)
    n_pred = tp + fp
    if n_gt == 0 and n_pred == 0:
        precision: float | None = 1.0
        recall: float | None = 1.0
        f1: float | None = 1.0
    else:
        precision = float(tp / n_pred) if n_pred > 0 else 0.0
        recall = float(tp / n_gt) if n_gt > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": int(n_gt),
        "n_pred_at_thr": int(n_pred),
    }


def _per_image_localization_records(
    prepared: _PreparedDataset,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
    *,
    max_dets: int | None = None,
) -> tuple[list[dict[float, tuple[np.ndarray, np.ndarray]]], list[int]]:
    """Compute class-agnostic per-image ``(scores, is_tp)`` at each IoU threshold."""
    records_by_img: list[dict[float, tuple[np.ndarray, np.ndarray]]] = []
    n_gt_by_img: list[int] = []

    for img_idx in range(len(prepared.image_keys)):
        gt_boxes = prepared.gt_boxes_by_img[img_idx]
        pred_boxes = prepared.pred_boxes_by_img[img_idx]
        pred_scores = prepared.pred_scores_by_img[img_idx]
        n_gt = len(gt_boxes)
        n_gt_by_img.append(n_gt)

        if len(pred_scores) == 0:
            empty_s = np.zeros((0,), dtype=np.float64)
            empty_tp = np.zeros((0,), dtype=np.int64)
            records_by_img.append({thr: (empty_s, empty_tp) for thr in iou_thresholds})
            continue

        order = np.argsort(-pred_scores, kind="mergesort")
        if max_dets is not None and len(order) > max_dets:
            order = order[:max_dets]
        p_boxes = pred_boxes[order]
        p_scores = pred_scores[order]

        per_thr: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for iou_thr in iou_thresholds:
            res = match_boxes(
                p_boxes,
                gt_boxes,
                iou_thr=iou_thr,
                method="greedy",
                pred_scores=p_scores,
                box_format="xywh",
            )
            is_tp = (res.pred_to_gt >= 0).astype(np.int64)
            per_thr[iou_thr] = (p_scores, is_tp)
        records_by_img.append(per_thr)

    return records_by_img, n_gt_by_img


def _aggregate_localization(
    sample_indices: Sequence[int],
    records_by_img: Sequence[dict[float, tuple[np.ndarray, np.ndarray]]],
    n_gt_by_img: Sequence[int],
    score_thr: float | None,
    *,
    include_best_f1: bool = True,
) -> dict[str, Any]:
    n_gt = int(sum(n_gt_by_img[i] for i in sample_indices))

    aps: dict[float, float | None] = {}
    for iou_thr in COCO_IOU_THRESHOLDS:
        s_list = [records_by_img[i][iou_thr][0] for i in sample_indices]
        tp_list = [records_by_img[i][iou_thr][1] for i in sample_indices]
        scores = np.concatenate(s_list) if s_list else np.zeros((0,), dtype=np.float64)
        is_tp = np.concatenate(tp_list) if tp_list else np.zeros((0,), dtype=np.int64)
        aps[iou_thr] = _compute_coco_ap_from_records(scores, is_tp, n_gt)

    ap50 = aps[0.5]
    valid_aps = [v for v in aps.values() if v is not None]
    ap50_95 = float(np.mean(valid_aps)) if len(valid_aps) == len(COCO_IOU_THRESHOLDS) else None

    s50 = np.concatenate([records_by_img[i][0.5][0] for i in sample_indices]) if sample_indices else np.zeros((0,), dtype=np.float64)
    tp50 = np.concatenate([records_by_img[i][0.5][1] for i in sample_indices]) if sample_indices else np.zeros((0,), dtype=np.int64)

    best_info = find_best_f1_threshold(s50, tp50, n_gt) if include_best_f1 else None
    applied_thr = float(score_thr) if score_thr is not None else float(best_info["threshold"])  # type: ignore[index]
    op = _operating_point_from_records(s50, tp50, n_gt, applied_thr)

    out: dict[str, Any] = {
        "ap50": ap50,
        "ap50_95": ap50_95,
        "score_thr": applied_thr,
        **op,
    }
    if best_info is not None:
        out["best_f1_threshold"] = best_info["threshold"]
        out["best_f1"] = best_info["f1"]
        out["best_f1_precision"] = best_info["precision"]
        out["best_f1_recall"] = best_info["recall"]
    return out


def compute_localization_metrics(
    gt_coco: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    score_thr: float | None = None,
    max_dets: int | None = None,
    include_best_f1: bool = True,
    allow_unknown_images: bool = False,
) -> dict[str, Any]:
    """Compute class-agnostic localization AP50, AP50:95, and operating-point P/R/F1."""
    prepared = _prepare_dataset(gt_coco, predictions, allow_unknown_images=allow_unknown_images)
    records_by_img, n_gt_by_img = _per_image_localization_records(prepared, max_dets=max_dets)
    return _aggregate_localization(
        list(range(len(prepared.image_keys))),
        records_by_img,
        n_gt_by_img,
        score_thr=score_thr,
        include_best_f1=include_best_f1,
    )


def _per_image_identification_records(
    prepared: _PreparedDataset,
    score_thr: float,
    *,
    iou_thr: float = 0.5,
    matching: MatchMethod = "greedy",
) -> list[tuple[list[int], list[int], int, int]]:
    """Per image: ``(y_true, y_pred, n_matched_unknown, n_unmatched_known)``."""
    per_img: list[tuple[list[int], list[int], int, int]] = []
    for img_idx in range(len(prepared.image_keys)):
        gt_boxes = prepared.gt_boxes_by_img[img_idx]
        gt_cats = prepared.gt_cats_by_img[img_idx]
        pred_boxes = prepared.pred_boxes_by_img[img_idx]
        pred_scores = prepared.pred_scores_by_img[img_idx]
        pred_cats = prepared.pred_cats_by_img[img_idx]

        keep = np.flatnonzero(pred_scores >= score_thr)
        kept_boxes = pred_boxes[keep] if len(keep) else np.zeros((0, 4), dtype=np.float64)
        kept_scores = pred_scores[keep] if len(keep) else np.zeros((0,), dtype=np.float64)
        kept_cats = [pred_cats[i] for i in keep.tolist()]

        res = match_boxes(
            kept_boxes,
            gt_boxes,
            iou_thr=iou_thr,
            method=matching,
            pred_scores=kept_scores,
            box_format="xywh",
        )
        y_true: list[int] = []
        y_pred: list[int] = []
        n_matched_unknown = 0
        for pi, gj, _ in res.matches:
            gt_cid = gt_cats[gj]
            if gt_cid in prepared.unknown_cat_ids | prepared.cartridge_cat_ids:
                n_matched_unknown += 1
                continue
            y_true.append(gt_cid)
            y_pred.append(kept_cats[pi])
        n_unmatched_known = sum(
            1 for gj in res.unmatched_gts if gt_cats[gj] not in prepared.unknown_cat_ids | prepared.cartridge_cat_ids
        )
        per_img.append((y_true, y_pred, n_matched_unknown, n_unmatched_known))
    return per_img


def _aggregate_identification(
    sample_indices: Sequence[int],
    per_img_id: Sequence[tuple[list[int], list[int], int, int]],
    prepared: _PreparedDataset,
    *,
    include_per_class: bool = True,
) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    n_matched_unknown = 0
    n_unmatched_known = 0
    for idx in sample_indices:
        yt, yp, mu, uk = per_img_id[idx]
        y_true.extend(yt)
        y_pred.extend(yp)
        n_matched_unknown += mu
        n_unmatched_known += uk

    n_matched_known = len(y_true)
    if n_matched_known == 0:
        out: dict[str, Any] = {
            "top1_acc": None,
            "macro_f1": None,
            "n_matched_known_gt": 0,
            "n_matched_unknown_gt": int(n_matched_unknown),
            "n_unmatched_known_gt": int(n_unmatched_known),
        }
        if include_per_class:
            out["per_class"] = {}
        return out

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    top1_acc = float(correct / n_matched_known)

    special_ids = prepared.unknown_cat_ids | prepared.cartridge_cat_ids | {-1}
    eval_classes = sorted(set(y_true) | (set(y_pred) - special_ids))

    per_class: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for cid in eval_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cid and p == cid)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cid and p == cid)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cid and p != cid)
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        if include_per_class:
            name = prepared.cat_id_to_name.get(cid, str(cid))
            per_class[name] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": int(tp + fn),
            }

    macro_f1 = float(np.mean(f1s)) if f1s else None
    out = {
        "top1_acc": top1_acc,
        "macro_f1": macro_f1,
        "n_matched_known_gt": int(n_matched_known),
        "n_matched_unknown_gt": int(n_matched_unknown),
        "n_unmatched_known_gt": int(n_unmatched_known),
    }
    if include_per_class:
        out["per_class"] = per_class
    return out


def compute_identification_metrics(
    gt_coco: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    score_thr: float = 0.0,
    iou_thr: float = 0.5,
    matching: MatchMethod = "greedy",
    allow_unknown_images: bool = False,
) -> dict[str, Any]:
    """Compute top-1 accuracy and macro-F1 over matched non-unknown GT boxes."""
    prepared = _prepare_dataset(gt_coco, predictions, allow_unknown_images=allow_unknown_images)
    per_img = _per_image_identification_records(
        prepared, score_thr=score_thr, iou_thr=iou_thr, matching=matching
    )
    return _aggregate_identification(
        list(range(len(prepared.image_keys))), per_img, prepared, include_per_class=True
    )


def _per_image_counting_records(
    prepared: _PreparedDataset,
    score_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Compute per-image total counts and per-catalog-class count vectors."""
    class_ids = [*prepared.catalog_cat_ids, -1]
    cid_to_col = {cid: idx for idx, cid in enumerate(class_ids)}
    n_images = len(prepared.image_keys)
    n_classes = len(class_ids)

    gt_totals = np.zeros(n_images, dtype=np.int64)
    pred_totals = np.zeros(n_images, dtype=np.int64)
    gt_counts = np.zeros((n_images, n_classes), dtype=np.int64)
    pred_counts = np.zeros((n_images, n_classes), dtype=np.int64)

    for img_idx in range(n_images):
        gt_cats = prepared.gt_cats_by_img[img_idx]
        gt_totals[img_idx] = len(gt_cats)
        for cid in gt_cats:
            col = cid_to_col.get(cid, len(class_ids) - 1)
            if col is not None:
                gt_counts[img_idx, col] += 1

        scores = prepared.pred_scores_by_img[img_idx]
        keep = np.flatnonzero(scores >= score_thr)
        pred_totals[img_idx] = len(keep)
        pred_cats = prepared.pred_cats_by_img[img_idx]
        for pi in keep.tolist():
            col = cid_to_col.get(pred_cats[pi], len(class_ids) - 1)
            if col is not None:
                pred_counts[img_idx, col] += 1

    return gt_totals, pred_totals, gt_counts, pred_counts, class_ids


def _aggregate_counting(
    sample_indices: Sequence[int],
    gt_totals: np.ndarray,
    pred_totals: np.ndarray,
    gt_counts: np.ndarray,
    pred_counts: np.ndarray,
    class_ids: Sequence[int],
    prepared: _PreparedDataset,
    *,
    include_per_class: bool = True,
) -> dict[str, Any]:
    if len(sample_indices) == 0:
        out: dict[str, Any] = {
            "total_count_mae": None,
            "total_count_exact_rate": None,
            "per_class_count_mae": None,
            "exact_inventory_rate": None,
        }
        if include_per_class:
            out["per_class_mae"] = {}
        return out

    idx = np.asarray(sample_indices, dtype=np.int64)
    total_abs_err = np.abs(pred_totals[idx] - gt_totals[idx])
    total_count_mae = float(np.mean(total_abs_err))

    if len(class_ids) == 0:
        per_class_count_mae: float | None = total_count_mae
        exact_inventory_rate = float(np.mean(total_abs_err == 0))
        per_class_mae: dict[str, float] = {}
    else:
        abs_err = np.abs(pred_counts[idx] - gt_counts[idx])  # (N, C)
        per_class_count_mae = float(np.mean(abs_err))
        exact_inventory_rate = float(np.mean(np.all(abs_err == 0, axis=1)))
        per_class_mae = (
            {
                ("unknown" if cid == -1 else prepared.cat_id_to_name.get(cid, str(cid))): float(val)
                for cid, val in zip(class_ids, np.mean(abs_err, axis=0).tolist())
            }
            if include_per_class
            else {}
        )

    out = {
        "total_count_mae": total_count_mae,
        "total_count_exact_rate": float(np.mean(total_abs_err == 0)),
        "per_class_count_mae": per_class_count_mae,
        "exact_inventory_rate": exact_inventory_rate,
    }
    if include_per_class:
        out["per_class_mae"] = per_class_mae
    return out


def compute_counting_metrics(
    gt_coco: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    score_thr: float = 0.0,
    allow_unknown_images: bool = False,
) -> dict[str, Any]:
    """Compute total-count MAE, per-class count MAE, and exact-inventory rate."""
    prepared = _prepare_dataset(gt_coco, predictions, allow_unknown_images=allow_unknown_images)
    gt_totals, pred_totals, gt_counts, pred_counts, class_ids = _per_image_counting_records(
        prepared, score_thr=score_thr
    )
    return _aggregate_counting(
        list(range(len(prepared.image_keys))),
        gt_totals,
        pred_totals,
        gt_counts,
        pred_counts,
        class_ids,
        prepared,
        include_per_class=True,
    )


def compute_group_bootstrap_ci(
    image_groups: Sequence[str],
    metric_fn: Any,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run deterministic cluster bootstrap over groups and return 95% percentile CIs."""
    group_to_indices: dict[str, list[int]] = {}
    for idx, gid in enumerate(image_groups):
        group_to_indices.setdefault(gid, []).append(idx)

    unique_groups = sorted(group_to_indices)
    n_groups = len(unique_groups)
    if n_groups == 0 or n_bootstrap <= 0:
        return {"n_bootstrap": int(n_bootstrap), "seed": int(seed), "n_groups": n_groups, "metrics": {}}

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        drawn = rng.integers(0, n_groups, size=n_groups)
        sample_indices: list[int] = []
        for g_idx in drawn.tolist():
            sample_indices.extend(group_to_indices[unique_groups[g_idx]])
        res = metric_fn(sample_indices)
        for key, val in res.items():
            samples.setdefault(key, []).append(float("nan") if val is None else float(val))

    ci_metrics: dict[str, list[float] | None] = {}
    for key, vals in samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            ci_metrics[key] = None
        else:
            lo, hi = np.percentile(valid, [2.5, 97.5])
            ci_metrics[key] = [float(lo), float(hi)]

    return {
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "n_groups": n_groups,
        "metrics": ci_metrics,
    }


def evaluate_predictions(
    gt_coco: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    score_thr: float | None = None,
    groups_data: Mapping[str, Any] | None = None,
    matching: MatchMethod = "greedy",
    n_bootstrap: int = 1000,
    seed: int = 42,
    max_dets: int | None = None,
    include_best_f1: bool = True,
    allow_image_groups: bool = False,
    allow_unknown_images: bool = False,
) -> dict[str, Any]:
    """Run the full Freestyle evaluation suite (localization, id, counting, group bootstrap)."""
    groups = load_groups_map(gt_coco, groups_data, allow_image_groups=allow_image_groups)
    prepared = _prepare_dataset(gt_coco, predictions, groups_data=groups_data,
                                allow_unknown_images=allow_unknown_images)
    prepared.image_groups = groups
    all_indices = list(range(len(prepared.image_keys)))

    loc_records, n_gt_by_img = _per_image_localization_records(prepared, max_dets=max_dets)
    loc = _aggregate_localization(
        all_indices,
        loc_records,
        n_gt_by_img,
        score_thr=score_thr,
        include_best_f1=include_best_f1,
    )
    applied_thr = float(loc["score_thr"])

    id_records = _per_image_identification_records(
        prepared, score_thr=applied_thr, iou_thr=0.5, matching=matching
    )
    ident = _aggregate_identification(all_indices, id_records, prepared, include_per_class=True)

    gt_totals, pred_totals, gt_counts, pred_counts, class_ids = _per_image_counting_records(
        prepared, score_thr=applied_thr
    )
    counting = _aggregate_counting(
        all_indices,
        gt_totals,
        pred_totals,
        gt_counts,
        pred_counts,
        class_ids,
        prepared,
        include_per_class=True,
    )

    ordered_groups = [prepared.image_groups[k] for k in prepared.image_keys]

    def _boot_step(sample_indices: Sequence[int]) -> dict[str, float | None]:
        b_loc = _aggregate_localization(
            sample_indices,
            loc_records,
            n_gt_by_img,
            score_thr=applied_thr,
            include_best_f1=False,
        )
        b_id = _aggregate_identification(
            sample_indices, id_records, prepared, include_per_class=False
        )
        b_cnt = _aggregate_counting(
            sample_indices,
            gt_totals,
            pred_totals,
            gt_counts,
            pred_counts,
            class_ids,
            prepared,
            include_per_class=False,
        )
        return {
            "localization.ap50": b_loc["ap50"],
            "localization.ap50_95": b_loc["ap50_95"],
            "localization.precision": b_loc["precision"],
            "localization.recall": b_loc["recall"],
            "localization.f1": b_loc["f1"],
            "identification.top1_acc": b_id["top1_acc"],
            "identification.macro_f1": b_id["macro_f1"],
            "counting.total_count_mae": b_cnt["total_count_mae"],
            "counting.total_count_exact_rate": b_cnt["total_count_exact_rate"],
            "counting.per_class_count_mae": b_cnt["per_class_count_mae"],
            "counting.exact_inventory_rate": b_cnt["exact_inventory_rate"],
        }

    bootstrap = compute_group_bootstrap_ci(
        ordered_groups, _boot_step, n_bootstrap=n_bootstrap, seed=seed
    )

    bootstrap["grouping"] = "ungrouped" if allow_image_groups else "grouped"
    return {
        "score_thr": applied_thr,
        "matching": matching,
        "n_images": len(prepared.image_keys),
        "n_groups": bootstrap["n_groups"],
        "n_pred_ignored_unknown_image": prepared.n_pred_ignored_unknown_image,
        "images": prepared.image_keys,
        "image_groups": prepared.image_groups,
        "localization": loc,
        "identification": ident,
        "counting": counting,
        "bootstrap_ci95": bootstrap,
    }
