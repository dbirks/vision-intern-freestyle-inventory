"""Box IoU and 1-to-1 matching utilities (greedy and Hungarian)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError:  # pragma: no cover - optional in light env
    _linear_sum_assignment = None

BoxFormat = Literal["xywh", "xyxy"]
MatchMethod = Literal["greedy", "hungarian"]


@dataclass(frozen=True)
class MatchResult:
    """Result of 1-to-1 matching between predictions and ground-truth boxes."""

    matches: list[tuple[int, int, float]]
    unmatched_preds: list[int]
    unmatched_gts: list[int]
    pred_to_gt: np.ndarray
    gt_to_pred: np.ndarray


def coco_xywh_to_xyxy(boxes: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Convert ``[x, y, w, h]`` boxes to ``[x1, y1, x2, y2]``."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    arr = arr.reshape(-1, 4)
    xyxy = np.empty_like(arr)
    xyxy[:, 0] = arr[:, 0]
    xyxy[:, 1] = arr[:, 1]
    xyxy[:, 2] = arr[:, 0] + np.maximum(arr[:, 2], 0.0)
    xyxy[:, 3] = arr[:, 1] + np.maximum(arr[:, 3], 0.0)
    return xyxy


def _to_xyxy(boxes: Sequence[Sequence[float]] | np.ndarray, box_format: BoxFormat) -> np.ndarray:
    if box_format == "xywh":
        return coco_xywh_to_xyxy(boxes)
    if box_format == "xyxy":
        arr = np.asarray(boxes, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, 4), dtype=np.float64)
        return arr.reshape(-1, 4)
    raise ValueError(f"Unsupported box_format={box_format!r}; expected 'xywh' or 'xyxy'.")


def box_iou(
    boxes1: Sequence[Sequence[float]] | np.ndarray,
    boxes2: Sequence[Sequence[float]] | np.ndarray,
    *,
    box_format: BoxFormat = "xywh",
) -> np.ndarray:
    """Compute pairwise IoU matrix of shape ``(len(boxes1), len(boxes2))``."""
    b1 = _to_xyxy(boxes1, box_format)
    b2 = _to_xyxy(boxes2, box_format)
    n1, n2 = len(b1), len(b2)
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float64)

    p = b1[:, None, :]
    g = b2[None, :, :]
    x1 = np.maximum(p[..., 0], g[..., 0])
    y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2])
    y2 = np.minimum(p[..., 3], g[..., 3])

    inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    area1 = np.maximum(b1[:, 2] - b1[:, 0], 0.0) * np.maximum(b1[:, 3] - b1[:, 1], 0.0)
    area2 = np.maximum(b2[:, 2] - b2[:, 0], 0.0) * np.maximum(b2[:, 3] - b2[:, 1], 0.0)
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _build_match_result(
    n_pred: int,
    n_gt: int,
    matches: list[tuple[int, int, float]],
) -> MatchResult:
    pred_to_gt = np.full(n_pred, -1, dtype=np.int64)
    gt_to_pred = np.full(n_gt, -1, dtype=np.int64)
    for pi, gj, _ in matches:
        pred_to_gt[pi] = gj
        gt_to_pred[gj] = pi
    unmatched_preds = [i for i in range(n_pred) if pred_to_gt[i] == -1]
    unmatched_gts = [j for j in range(n_gt) if gt_to_pred[j] == -1]
    return MatchResult(
        matches=matches,
        unmatched_preds=unmatched_preds,
        unmatched_gts=unmatched_gts,
        pred_to_gt=pred_to_gt,
        gt_to_pred=gt_to_pred,
    )


def match_greedy(
    pred_boxes: Sequence[Sequence[float]] | np.ndarray,
    gt_boxes: Sequence[Sequence[float]] | np.ndarray,
    iou_thr: float = 0.5,
    *,
    pred_scores: Sequence[float] | np.ndarray | None = None,
    box_format: BoxFormat = "xywh",
    pred_classes: Sequence[int] | None = None,
    gt_classes: Sequence[int] | None = None,
) -> MatchResult:
    """Greedily match predictions to GT boxes at ``iou >= iou_thr``.

    When ``pred_scores`` is given, predictions are processed in descending score
    order (stable tie-breaking by original index), matching COCO evaluation.
    Otherwise predictions are matched greedily in descending IoU order over
    eligible pairs.
    """
    ious = box_iou(pred_boxes, gt_boxes, box_format=box_format)
    n_pred, n_gt = ious.shape
    if n_pred == 0 or n_gt == 0:
        return _build_match_result(n_pred, n_gt, [])

    class_filter = pred_classes is not None and gt_classes is not None
    gt_used = np.zeros(n_gt, dtype=bool)
    matches: list[tuple[int, int, float]] = []

    if pred_scores is not None:
        scores = np.asarray(pred_scores, dtype=np.float64)
        if len(scores) != n_pred:
            raise ValueError(f"pred_scores length {len(scores)} != n_pred {n_pred}")
        order = np.argsort(-scores, kind="mergesort")
        for pi in order.tolist():
            best_j = -1
            best_iou = iou_thr
            for gj in range(n_gt):
                if gt_used[gj]:
                    continue
                if class_filter and pred_classes[pi] != gt_classes[gj]:
                    continue
                val = float(ious[pi, gj])
                if val >= best_iou:
                    if val > best_iou or best_j == -1:
                        best_iou = val
                        best_j = gj
            if best_j >= 0:
                gt_used[best_j] = True
                matches.append((pi, best_j, best_iou))
    else:
        # Score-free greedy: sort all eligible pairs by descending IoU.
        candidates: list[tuple[float, int, int]] = []
        for pi in range(n_pred):
            for gj in range(n_gt):
                if class_filter and pred_classes[pi] != gt_classes[gj]:
                    continue
                val = float(ious[pi, gj])
                if val >= iou_thr:
                    candidates.append((val, pi, gj))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        pred_used = np.zeros(n_pred, dtype=bool)
        for val, pi, gj in candidates:
            if pred_used[pi] or gt_used[gj]:
                continue
            pred_used[pi] = True
            gt_used[gj] = True
            matches.append((pi, gj, val))

    return _build_match_result(n_pred, n_gt, matches)


def match_hungarian(
    pred_boxes: Sequence[Sequence[float]] | np.ndarray,
    gt_boxes: Sequence[Sequence[float]] | np.ndarray,
    iou_thr: float = 0.5,
    *,
    pred_scores: Sequence[float] | np.ndarray | None = None,
    box_format: BoxFormat = "xywh",
    pred_classes: Sequence[int] | None = None,
    gt_classes: Sequence[int] | None = None,
) -> MatchResult:
    """Maximum-cardinality, maximum-IoU bipartite match at ``iou >= iou_thr``.

    Uses ``scipy.optimize.linear_sum_assignment`` when available, and falls back
    to ``match_greedy`` in environments without SciPy.
    """
    if _linear_sum_assignment is None:
        return match_greedy(
            pred_boxes,
            gt_boxes,
            iou_thr=iou_thr,
            pred_scores=pred_scores,
            box_format=box_format,
            pred_classes=pred_classes,
            gt_classes=gt_classes,
        )

    ious = box_iou(pred_boxes, gt_boxes, box_format=box_format)
    n_pred, n_gt = ious.shape
    if n_pred == 0 or n_gt == 0:
        return _build_match_result(n_pred, n_gt, [])

    class_filter = pred_classes is not None and gt_classes is not None
    # Pad with n_pred dummy "unmatched" columns of cost 0.0 so invalid edges
    # (cost +1e6) are never forced, while every valid edge has cost -(1 + iou).
    large_cost = 1e6
    cost = np.full((n_pred, n_gt + n_pred), large_cost, dtype=np.float64)
    for pi in range(n_pred):
        cost[pi, n_gt + pi] = 0.0
        for gj in range(n_gt):
            if class_filter and pred_classes[pi] != gt_classes[gj]:
                continue
            val = float(ious[pi, gj])
            if val >= iou_thr:
                cost[pi, gj] = -(1.0 + val)

    row_ind, col_ind = _linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    for pi, gj in zip(row_ind.tolist(), col_ind.tolist()):
        if gj < n_gt and cost[pi, gj] < 0.0:
            matches.append((pi, gj, float(ious[pi, gj])))
    matches.sort(key=lambda item: (item[0], item[1]))
    return _build_match_result(n_pred, n_gt, matches)


def match_boxes(
    pred_boxes: Sequence[Sequence[float]] | np.ndarray,
    gt_boxes: Sequence[Sequence[float]] | np.ndarray,
    iou_thr: float = 0.5,
    *,
    method: MatchMethod = "greedy",
    pred_scores: Sequence[float] | np.ndarray | None = None,
    box_format: BoxFormat = "xywh",
    pred_classes: Sequence[int] | None = None,
    gt_classes: Sequence[int] | None = None,
) -> MatchResult:
    """Match predictions to GT boxes using ``'greedy'`` or ``'hungarian'``."""
    if method == "greedy":
        return match_greedy(
            pred_boxes,
            gt_boxes,
            iou_thr=iou_thr,
            pred_scores=pred_scores,
            box_format=box_format,
            pred_classes=pred_classes,
            gt_classes=gt_classes,
        )
    if method == "hungarian":
        return match_hungarian(
            pred_boxes,
            gt_boxes,
            iou_thr=iou_thr,
            pred_scores=pred_scores,
            box_format=box_format,
            pred_classes=pred_classes,
            gt_classes=gt_classes,
        )
    raise ValueError(f"Unsupported matching method={method!r}; expected 'greedy' or 'hungarian'.")
