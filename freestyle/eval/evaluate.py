"""CLI evaluator for Freestyle cartridge recognition predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from freestyle.eval.metrics import compute_localization_metrics, evaluate_predictions


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_test_split(
    gt_path: Path,
    gt_coco: Mapping[str, Any],
    groups_data: Mapping[str, Any] | None,
) -> bool:
    if "test" in gt_path.stem.lower().split("_") or "test" in gt_path.stem.lower().split("-"):
        return True
    if groups_data and isinstance(groups_data.get("images"), Mapping):
        split_images = groups_data["images"]
        for img in gt_coco.get("images", []):
            for cand in (img.get("file_name"), str(img.get("id"))):
                if cand and split_images.get(cand) == "test":
                    return True
    return False


def _infer_experiment_name(pred_path: Path, out_path: Path | None, meta: Mapping[str, Any] | None) -> str:
    if meta and meta.get("experiment"):
        return str(meta["experiment"])
    if pred_path.parent.name and pred_path.parent.name not in (".", ""):
        return pred_path.parent.name
    if out_path is not None and out_path.parent.name not in (".", ""):
        return out_path.parent.name
    return pred_path.stem


def _fmt_metric(val: float | None, ci: Sequence[float] | None = None) -> str:
    if val is None:
        return "-"
    if ci is not None and len(ci) == 2:
        return f"{val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{val:.3f}"


def format_summary_table(metrics: Mapping[str, Any]) -> str:
    """Format a compact plain-text summary table for stdout."""
    loc = metrics["localization"]
    ident = metrics["identification"]
    cnt = metrics["counting"]
    cis = metrics.get("bootstrap_ci95", {}).get("metrics", {})

    exp = metrics.get("experiment", "run")
    lines = [
        f"=== Freestyle Eval: {exp} ===",
        f"images={metrics['n_images']}  groups={metrics['n_groups']}  "
        f"score_thr={metrics['score_thr']:.4f} ({metrics.get('score_thr_source', 'explicit')})",
        "",
        f"{'Metric':<24} {'Value (95% CI)':<26}",
        f"{'-' * 24} {'-' * 26}",
        f"{'Loc AP50':<24} {_fmt_metric(loc.get('ap50'), cis.get('localization.ap50'))}",
        f"{'Loc AP50:95':<24} {_fmt_metric(loc.get('ap50_95'), cis.get('localization.ap50_95'))}",
        f"{'Loc Precision@thr':<24} {_fmt_metric(loc.get('precision'), cis.get('localization.precision'))}",
        f"{'Loc Recall@thr':<24} {_fmt_metric(loc.get('recall'), cis.get('localization.recall'))}",
        f"{'Loc F1@thr':<24} {_fmt_metric(loc.get('f1'), cis.get('localization.f1'))}",
        f"{'ID Top-1 Acc':<24} {_fmt_metric(ident.get('top1_acc'), cis.get('identification.top1_acc'))}",
        f"{'ID Macro-F1':<24} {_fmt_metric(ident.get('macro_f1'), cis.get('identification.macro_f1'))}",
        f"{'Count Total MAE':<24} {_fmt_metric(cnt.get('total_count_mae'), cis.get('counting.total_count_mae'))}",
        f"{'Count Class MAE':<24} {_fmt_metric(cnt.get('per_class_count_mae'), cis.get('counting.per_class_count_mae'))}",
        f"{'Exact Inventory Rate':<24} {_fmt_metric(cnt.get('exact_inventory_rate'), cis.get('counting.exact_inventory_rate'))}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate Freestyle cartridge localization, identification, and counting."
    )
    p.add_argument("--gt", required=True, help="Ground-truth COCO JSON file.")
    p.add_argument("--pred", required=True, help="Predictions COCO results JSON list.")
    p.add_argument(
        "--groups",
        default=None,
        help="Optional data/splits.json or image->group JSON mapping for group bootstrap.",
    )
    thr_group = p.add_mutually_exclusive_group(required=True)
    thr_group.add_argument(
        "--score-thr",
        type=float,
        default=None,
        help="Explicit operating confidence threshold.",
    )
    thr_group.add_argument(
        "--select-thr-on",
        nargs=2,
        metavar=("VAL_GT", "VAL_PRED"),
        default=None,
        help="Select best-F1 operating threshold on a validation GT+pred pair.",
    )
    p.add_argument("--out", default=None, help="Optional path to write metrics.json.")
    p.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap replicates (default 1000).")
    p.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed (default 42).")
    p.add_argument(
        "--matching",
        choices=("greedy", "hungarian"),
        default="greedy",
        help="1-to-1 matching method for identification at IoU>=0.5 (default greedy).",
    )
    p.add_argument("--experiment", default=None, help="Optional experiment label.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    out_path = Path(args.out) if args.out else None

    gt_coco = _load_json(gt_path)
    predictions = _load_json(pred_path)
    groups_data = _load_json(args.groups) if args.groups else None

    if args.score_thr is not None:
        score_thr = float(args.score_thr)
        score_thr_source = "explicit"
    else:
        val_gt_path, val_pred_path = (Path(p) for p in args.select_thr_on)
        if val_gt_path.resolve() == gt_path.resolve() and _looks_like_test_split(
            gt_path, gt_coco, groups_data
        ):
            parser.error(
                "--select-thr-on cannot use the test GT file; pass --score-thr or a val split."
            )
        val_gt = _load_json(val_gt_path)
        val_pred = _load_json(val_pred_path)
        val_loc = compute_localization_metrics(val_gt, val_pred, score_thr=None, include_best_f1=True)
        score_thr = float(val_loc["best_f1_threshold"])
        score_thr_source = f"selected_on:{val_gt_path}"

    meta_path = pred_path.parent / "meta.json"
    meta = _load_json(meta_path) if meta_path.is_file() else None

    metrics = evaluate_predictions(
        gt_coco,
        predictions,
        score_thr=score_thr,
        groups_data=groups_data,
        matching=args.matching,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        include_best_f1=False,
    )
    metrics["experiment"] = args.experiment or _infer_experiment_name(pred_path, out_path, meta)
    metrics["score_thr_source"] = score_thr_source
    metrics["gt_path"] = str(gt_path)
    metrics["pred_path"] = str(pred_path)
    if meta is not None:
        metrics["meta"] = meta

    print(format_summary_table(metrics))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
