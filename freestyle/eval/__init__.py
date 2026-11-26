"""Evaluation harness for Coca-Cola Freestyle cartridge recognition."""

from freestyle.eval.matching import (
    MatchResult,
    box_iou,
    coco_xywh_to_xyxy,
    match_boxes,
    match_greedy,
    match_hungarian,
)
from freestyle.eval.metrics import (
    compute_counting_metrics,
    compute_group_bootstrap_ci,
    compute_identification_metrics,
    compute_localization_metrics,
    evaluate_predictions,
    find_best_f1_threshold,
    load_groups_map,
)
from freestyle.eval.report import build_results_markdown, render_markdown_table

__all__ = [
    "MatchResult",
    "box_iou",
    "build_results_markdown",
    "coco_xywh_to_xyxy",
    "compute_counting_metrics",
    "compute_group_bootstrap_ci",
    "compute_identification_metrics",
    "compute_localization_metrics",
    "evaluate_predictions",
    "find_best_f1_threshold",
    "load_groups_map",
    "match_boxes",
    "match_greedy",
    "match_hungarian",
    "render_markdown_table",
]
