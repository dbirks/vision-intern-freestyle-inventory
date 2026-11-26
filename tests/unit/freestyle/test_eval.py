"""Unit tests for the Freestyle evaluation harness (synthetic COCO data, no private images)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# If tests/unit/freestyle/__init__.py causes pytest to bind `freestyle` to the
# test directory, extend its __path__ with the real package directory so both
# test modules and `freestyle.eval` resolve cleanly.
_REPO_FREESTYLE = str(Path(__file__).resolve().parents[3] / "freestyle")
if "freestyle" in sys.modules and hasattr(sys.modules["freestyle"], "__path__"):
    if _REPO_FREESTYLE not in sys.modules["freestyle"].__path__:
        sys.modules["freestyle"].__path__.append(_REPO_FREESTYLE)

from freestyle.eval import (  # noqa: E402
    box_iou,
    build_results_markdown,
    coco_xywh_to_xyxy,
    compute_counting_metrics,
    compute_identification_metrics,
    compute_localization_metrics,
    evaluate_predictions,
    load_groups_map,
    match_boxes,
    match_greedy,
    match_hungarian,
)
from freestyle.eval.evaluate import main as evaluate_main  # noqa: E402
from freestyle.eval.report import main as report_main  # noqa: E402


def _make_coco(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict] | None = None,
) -> dict:
    if categories is None:
        categories = [
            {"id": 1, "name": "coca-cola-part1"},
            {"id": 2, "name": "nns-sweetener"},
            {"id": 3, "name": "unknown"},
        ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _baseline_gt() -> dict:
    images = [
        {"id": 1, "width": 400, "height": 400, "file_name": "srv/evt-a/001.jpg"},
        {"id": 2, "width": 400, "height": 400, "file_name": "srv/evt-b/002.jpg"},
    ]
    annotations = [
        {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100]},
        {"id": 11, "image_id": 1, "category_id": 2, "bbox": [150, 10, 100, 100]},
        {"id": 12, "image_id": 2, "category_id": 1, "bbox": [20, 20, 100, 100]},
    ]
    return _make_coco(images, annotations)


def test_matching_iou_and_greedy_vs_hungarian(monkeypatch):
    xyxy = coco_xywh_to_xyxy([[10, 20, 30, 40]])
    assert np.allclose(xyxy, [[10, 20, 40, 60]])

    iou = box_iou([[0, 0, 10, 10]], [[5, 0, 10, 10]], box_format="xywh")
    # intersection 5x10=50, union 100+100-50=150 -> 1/3
    assert math.isclose(iou[0, 0], 1.0 / 3.0, rel_tol=1e-6)
    assert box_iou([], [[0, 0, 10, 10]]).shape == (0, 1)

    preds = [[0, 0, 10, 10], [8, 0, 10, 10]]
    gts = [[1, 0, 10, 10], [10, 0, 10, 10]]
    # pred0 overlaps gt0 (IoU=9/11=0.818) and gt1 (0); pred1 overlaps gt0 (2/18=0.11) and gt1 (8/12=0.667)
    res_greedy = match_greedy(preds, gts, iou_thr=0.5, pred_scores=[0.9, 0.8])
    assert res_greedy.pred_to_gt.tolist() == [0, 1]

    # Force fallback path when scipy is absent
    import freestyle.eval.matching as matching_mod

    monkeypatch.setattr(matching_mod, "_linear_sum_assignment", None)
    res_fallback = match_hungarian(preds, gts, iou_thr=0.5, pred_scores=[0.9, 0.8])
    assert res_fallback.pred_to_gt.tolist() == [0, 1]

    with pytest.raises(ValueError):
        match_boxes(preds, gts, method="invalid")  # type: ignore[arg-type]


def test_perfect_prediction_metrics():
    gt = _baseline_gt()
    preds = [
        {"image_id": 1, "bbox": [10, 10, 100, 100], "score": 0.95, "category_id": 1, "id_score": 0.99},
        {"image_id": 1, "bbox": [150, 10, 100, 100], "score": 0.90, "category_id": 2, "id_score": 0.98},
        {"image_id": 2, "bbox": [20, 20, 100, 100], "score": 0.85, "category_id": 1, "id_score": 0.97},
    ]
    res = evaluate_predictions(gt, preds, score_thr=0.5, n_bootstrap=50, seed=7)
    loc = res["localization"]
    ident = res["identification"]
    cnt = res["counting"]

    assert loc["ap50"] == pytest.approx(1.0)
    assert loc["ap50_95"] == pytest.approx(1.0)
    assert loc["precision"] == pytest.approx(1.0)
    assert loc["recall"] == pytest.approx(1.0)
    assert loc["f1"] == pytest.approx(1.0)
    assert ident["top1_acc"] == pytest.approx(1.0)
    assert ident["macro_f1"] == pytest.approx(1.0)
    assert cnt["total_count_mae"] == pytest.approx(0.0)
    assert cnt["per_class_count_mae"] == pytest.approx(0.0)
    assert cnt["exact_inventory_rate"] == pytest.approx(1.0)


def test_shifted_boxes_partial_and_miss():
    # Box is 100x100 at [0,0]. Shift by x=15 -> inter=85*100=8500, union=11500 -> IoU=85/115=0.7391
    gt = _make_coco(
        images=[{"id": 1, "width": 200, "height": 200}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100]}],
    )
    moderate_shift = [{"image_id": 1, "bbox": [15, 0, 100, 100], "score": 0.9, "category_id": 1}]
    res_mod = evaluate_predictions(gt, moderate_shift, score_thr=0.5, n_bootstrap=20, seed=1)
    assert res_mod["localization"]["ap50"] == pytest.approx(1.0)
    # IoU 0.7391 passes 0.50, 0.55, 0.60, 0.65, 0.70 (5 of 10 thresholds) -> AP50:95 == 0.5
    assert res_mod["localization"]["ap50_95"] == pytest.approx(0.5)
    assert res_mod["identification"]["top1_acc"] == pytest.approx(1.0)

    # Shift by x=40 -> inter=60*100=6000, union=14000 -> IoU=0.4286 < 0.5
    large_shift = [{"image_id": 1, "bbox": [40, 0, 100, 100], "score": 0.9, "category_id": 1}]
    res_bad = evaluate_predictions(gt, large_shift, score_thr=0.5, n_bootstrap=20, seed=1)
    assert res_bad["localization"]["ap50"] == pytest.approx(0.0)
    assert res_bad["localization"]["recall"] == pytest.approx(0.0)
    assert res_bad["identification"]["n_matched_known_gt"] == 0
    assert res_bad["identification"]["top1_acc"] is None
    # Counting does not depend on spatial IoU
    assert res_bad["counting"]["total_count_mae"] == pytest.approx(0.0)
    assert res_bad["counting"]["exact_inventory_rate"] == pytest.approx(1.0)


def test_missing_and_extra_boxes_and_best_f1():
    gt = _baseline_gt()
    # 2 TPs (high score), 1 missing (image 2), 1 low-score FP in image 1
    preds = [
        {"image_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9, "category_id": 1},
        {"image_id": 1, "bbox": [150, 10, 100, 100], "score": 0.8, "category_id": 2},
        {"image_id": 1, "bbox": [300, 300, 50, 50], "score": 0.3, "category_id": 1},
    ]
    loc = compute_localization_metrics(gt, preds, score_thr=None, include_best_f1=True)
    # Best-F1 should cut off the 0.3 FP -> threshold 0.8, TP=2, FP=0, FN=1 -> P=1, R=2/3, F1=0.8
    assert loc["best_f1_threshold"] == pytest.approx(0.8)
    assert loc["best_f1_precision"] == pytest.approx(1.0)
    assert loc["best_f1_recall"] == pytest.approx(2.0 / 3.0)
    assert loc["best_f1"] == pytest.approx(0.8)
    assert loc["ap50"] == pytest.approx(2.0 / 3.0, abs=0.02)

    # Evaluating at score_thr=0.2 keeps the FP -> image 1 has 3 preds vs 2 GT, image 2 has 0 vs 1 GT
    cnt = compute_counting_metrics(gt, preds, score_thr=0.2)
    assert cnt["total_count_mae"] == pytest.approx((1 + 1) / 2.0)
    # Per-class counts: img1 cat1=2 vs 1 (err 1), cat2=1 vs 1 (err 0); img2 cat1=0 vs 1 (err 1), cat2=0 vs 0 (err 0)
    assert cnt["per_class_count_mae"] == pytest.approx(2.0 / 4.0)
    assert cnt["exact_inventory_rate"] == pytest.approx(0.0)


def test_wrong_identity_keeps_localization_perfect():
    gt = _baseline_gt()
    # Swap category_ids on image 1; keep image 2 correct
    preds = [
        {"image_id": 1, "bbox": [10, 10, 100, 100], "score": 0.95, "category_id": 2},
        {"image_id": 1, "bbox": [150, 10, 100, 100], "score": 0.90, "category_id": 1},
        {"image_id": 2, "bbox": [20, 20, 100, 100], "score": 0.85, "category_id": 1},
    ]
    res = evaluate_predictions(gt, preds, score_thr=0.5, n_bootstrap=20, seed=1)
    assert res["localization"]["ap50"] == pytest.approx(1.0)
    assert res["identification"]["top1_acc"] == pytest.approx(1.0 / 3.0)
    # cat1: tp=1, fp=1, fn=1 -> f1=0.5; cat2: tp=0, fp=1, fn=1 -> f1=0.0 => macro_f1=0.25
    assert res["identification"]["macro_f1"] == pytest.approx(0.25)
    # Image 1 swapped counts cancel out per-class, image 2 matches -> exact inventory rate is 1.0!
    assert res["counting"]["total_count_mae"] == pytest.approx(0.0)
    assert res["counting"]["exact_inventory_rate"] == pytest.approx(1.0)

    # Now predict cat 2 on image 2 as well -> class counts are wrong on image 2
    preds[2]["category_id"] = 2
    res2 = evaluate_predictions(gt, preds, score_thr=0.5, n_bootstrap=20, seed=1)
    assert res2["counting"]["total_count_mae"] == pytest.approx(0.0)
    assert res2["counting"]["per_class_count_mae"] > 0.0
    assert res2["counting"]["exact_inventory_rate"] == pytest.approx(0.5)


def test_unknown_class_gt_handling():
    gt = _make_coco(
        images=[{"id": 1, "width": 400, "height": 400}],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 80, 80]},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [120, 10, 80, 80]},  # unknown
        ],
    )
    # Predict both boxes; the unknown box gets predicted as category 2
    preds = [
        {"image_id": 1, "bbox": [10, 10, 80, 80], "score": 0.9, "category_id": 1},
        {"image_id": 1, "bbox": [120, 10, 80, 80], "score": 0.8, "category_id": 2},
    ]
    res = evaluate_predictions(gt, preds, score_thr=0.5, n_bootstrap=10, seed=0)
    assert res["localization"]["ap50"] == pytest.approx(1.0)
    assert res["identification"]["n_matched_known_gt"] == 1
    assert res["identification"]["n_matched_unknown_gt"] == 1
    assert res["identification"]["top1_acc"] == pytest.approx(1.0)
    assert res["identification"]["macro_f1"] == pytest.approx(1.0)
    # Total count includes unknown GT (2 vs 2)
    assert res["counting"]["total_count_mae"] == pytest.approx(0.0)

    # Missing the unknown GT box hurts localization & total count, not identification accuracy
    preds_miss_unk = [preds[0]]
    res_miss = evaluate_predictions(gt, preds_miss_unk, score_thr=0.5, n_bootstrap=10, seed=0)
    assert res_miss["localization"]["recall"] == pytest.approx(0.5)
    assert res_miss["identification"]["top1_acc"] == pytest.approx(1.0)
    assert res_miss["counting"]["total_count_mae"] == pytest.approx(1.0)

    ident_direct = compute_identification_metrics(gt, preds, score_thr=0.5, matching="hungarian")
    assert ident_direct["top1_acc"] == pytest.approx(1.0)
    assert ident_direct["macro_f1"] == pytest.approx(1.0)


def test_empty_images_and_all_empty_dataset():
    gt = _make_coco(
        images=[
            {"id": 1, "width": 400, "height": 400, "file_name": "img1.jpg"},
            {"id": 2, "width": 400, "height": 400, "file_name": "empty.jpg", "attributes": {"scene": "no_cabinet"}},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 80, 80]},
        ],
    )
    preds_clean = [
        {"image_id": 1, "bbox": [10, 10, 80, 80], "score": 0.95, "category_id": 1},
    ]
    res_clean = evaluate_predictions(gt, preds_clean, score_thr=0.5, n_bootstrap=20, seed=3)
    assert res_clean["localization"]["ap50"] == pytest.approx(1.0)
    assert res_clean["counting"]["exact_inventory_rate"] == pytest.approx(1.0)
    assert res_clean["counting"]["total_count_mae"] == pytest.approx(0.0)

    # False positive on the empty image
    preds_fp = preds_clean + [
        {"image_id": 2, "bbox": [20, 20, 50, 50], "score": 0.99, "category_id": 1},
    ]
    res_fp = evaluate_predictions(gt, preds_fp, score_thr=0.5, n_bootstrap=20, seed=3)
    assert res_fp["localization"]["ap50"] < 0.6
    assert res_fp["localization"]["precision"] == pytest.approx(0.5)
    assert res_fp["counting"]["total_count_mae"] == pytest.approx(0.5)
    assert res_fp["counting"]["exact_inventory_rate"] == pytest.approx(0.5)

    # All-empty dataset should not crash
    gt_empty = _make_coco(images=[{"id": 1, "width": 100, "height": 100}], annotations=[])
    res_empty = evaluate_predictions(gt_empty, [], score_thr=0.5, n_bootstrap=10, seed=0)
    assert res_empty["localization"]["ap50"] is None
    assert res_empty["counting"]["exact_inventory_rate"] == pytest.approx(1.0)


def test_group_bootstrap_determinism_and_splits_mapping():
    images = [
        {"id": 1, "width": 200, "height": 200, "file_name": "island-cabinet-01/uuid-1/001.jpg"},
        {"id": 2, "width": 200, "height": 200, "file_name": "island-cabinet-01/uuid-1/002.jpg"},
        {"id": 3, "width": 200, "height": 200, "file_name": "local/blur/001.jpg"},
        {"id": 4, "width": 200, "height": 200, "file_name": "local/downloads/frame.jpg"},
        {"id": 5, "width": 200, "height": 200, "file_name": "island-cabinet-01/uuid-2/001.jpg"},
        {"id": 6, "width": 200, "height": 200, "file_name": "island-cabinet-01/uuid-3/001.jpg"},
        {"id": 7, "width": 200, "height": 200, "file_name": "island-cabinet-01/uuid-4/001.jpg"},
        {"id": 8, "width": 200, "height": 200, "file_name": "island-cabinet-01/uuid-5/001.jpg"},
    ]
    annotations = [
        {"id": i, "image_id": i, "category_id": 1 if i != 3 else 2, "bbox": [10, 10, 50, 50]}
        for i in range(1, 9)
    ]
    gt = _make_coco(images, annotations)
    splits_json = {
        "version": 1,
        "groups": {
            "uuid-1": "test",
            "local/blur": "test",
            "local/downloads/frame.jpg": "test",
            "uuid-2": "test",
            "uuid-3": "test",
            "uuid-4": "test",
            "uuid-5": "test",
        },
        "images": {img["file_name"]: "test" for img in images},
    }
    group_map = load_groups_map(gt, splits_json)
    assert group_map["island-cabinet-01/uuid-1/001.jpg"] == "uuid-1"
    assert group_map["island-cabinet-01/uuid-1/002.jpg"] == "uuid-1"
    assert group_map["local/blur/001.jpg"] == "local/blur"
    assert group_map["local/downloads/frame.jpg"] == "local/downloads/frame.jpg"

    preds = [
        {"image_id": 1, "bbox": [10, 10, 50, 50], "score": 0.9, "category_id": 1},
        {"image_id": 2, "bbox": [10, 10, 50, 50], "score": 0.9, "category_id": 1},
        {"image_id": 3, "bbox": [10, 10, 50, 50], "score": 0.8, "category_id": 1},  # wrong class
        {"image_id": 4, "bbox": [10, 10, 50, 50], "score": 0.4, "category_id": 1},  # below thr=0.5
        {"image_id": 5, "bbox": [10, 10, 50, 50], "score": 0.9, "category_id": 1},
        {"image_id": 6, "bbox": [10, 10, 50, 50], "score": 0.9, "category_id": 2},  # wrong class
        {"image_id": 7, "bbox": [10, 10, 50, 50], "score": 0.9, "category_id": 1},
        {"image_id": 8, "bbox": [10, 10, 50, 50], "score": 0.9, "category_id": 2},  # wrong class
    ]

    r1 = evaluate_predictions(gt, preds, score_thr=0.5, groups_data=splits_json, n_bootstrap=40, seed=123)
    r2 = evaluate_predictions(gt, preds, score_thr=0.5, groups_data=splits_json, n_bootstrap=40, seed=123)
    r3 = evaluate_predictions(gt, preds, score_thr=0.5, groups_data=splits_json, n_bootstrap=40, seed=999)

    assert r1["n_groups"] == 7
    assert r1["bootstrap_ci95"] == r2["bootstrap_ci95"]
    lo, hi = r1["bootstrap_ci95"]["metrics"]["identification.top1_acc"]
    assert 0.0 <= lo <= r1["identification"]["top1_acc"] <= hi <= 1.0
    assert lo < hi
    assert r1["bootstrap_ci95"]["metrics"] != r3["bootstrap_ci95"]["metrics"]


def test_pycocotools_ap_crosscheck():
    pycocotools = pytest.importorskip("pycocotools")
    assert pycocotools is not None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(7)
    images = [{"id": i + 1, "width": 500, "height": 500} for i in range(6)]
    categories = [{"id": 1, "name": "cartridge"}]
    annotations = []
    preds = []
    ann_id = 1
    for img_id in range(1, 7):
        for _ in range(4):
            x, y = rng.uniform(10, 350, size=2).tolist()
            w, h = rng.uniform(40, 80, size=2).tolist()
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            if rng.random() > 0.2:
                dx, dy = rng.normal(0, 8, size=2).tolist()
                preds.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x + dx, y + dy, w, h],
                        "score": float(rng.uniform(0.1, 0.99)),
                    }
                )
        # Extra FP
        preds.append(
            {
                "image_id": img_id,
                "category_id": 1,
                "bbox": [5.0, 5.0, 30.0, 30.0],
                "score": float(rng.uniform(0.05, 0.7)),
            }
        )

    gt_dict = {"images": images, "annotations": annotations, "categories": categories}
    ours = compute_localization_metrics(gt_dict, preds, max_dets=100)

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.useCats = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    assert ours["ap50_95"] == pytest.approx(float(coco_eval.stats[0]), abs=1e-4)
    assert ours["ap50"] == pytest.approx(float(coco_eval.stats[1]), abs=1e-4)


def test_cli_evaluate_and_report(tmp_path: Path, capsys):
    val_gt = _baseline_gt()
    val_preds = [
        {"image_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9, "category_id": 1},
        {"image_id": 1, "bbox": [150, 10, 100, 100], "score": 0.75, "category_id": 2},
        {"image_id": 2, "bbox": [20, 20, 100, 100], "score": 0.70, "category_id": 1},
        {"image_id": 2, "bbox": [250, 250, 50, 50], "score": 0.25, "category_id": 1},
    ]
    test_gt = _baseline_gt()
    test_preds = [
        {"image_id": 1, "bbox": [10, 10, 100, 100], "score": 0.92, "category_id": 1},
        {"image_id": 1, "bbox": [150, 10, 100, 100], "score": 0.80, "category_id": 2},
        {"image_id": 2, "bbox": [20, 20, 100, 100], "score": 0.72, "category_id": 1},
    ]

    val_gt_p = tmp_path / "val.json"
    val_pred_p = tmp_path / "val_preds.json"
    test_gt_p = tmp_path / "test.json"
    run_dir = tmp_path / "runs" / "exp_a"
    run_dir.mkdir(parents=True)
    test_pred_p = run_dir / "predictions.json"
    out_p = run_dir / "metrics.json"
    meta_p = run_dir / "meta.json"

    val_gt_p.write_text(json.dumps(val_gt), encoding="utf-8")
    val_pred_p.write_text(json.dumps(val_preds), encoding="utf-8")
    test_gt_p.write_text(json.dumps(test_gt), encoding="utf-8")
    test_pred_p.write_text(json.dumps(test_preds), encoding="utf-8")
    meta_p.write_text(json.dumps({"experiment": "exp_a", "seconds_per_image": 0.42}), encoding="utf-8")

    # Omitting both --score-thr and --select-thr-on must fail
    with pytest.raises(SystemExit):
        evaluate_main(["--gt", str(test_gt_p), "--pred", str(test_pred_p)])

    # Selecting threshold on the test GT itself must fail
    with pytest.raises(SystemExit):
        evaluate_main(
            [
                "--gt",
                str(test_gt_p),
                "--pred",
                str(test_pred_p),
                "--select-thr-on",
                str(test_gt_p),
                str(test_pred_p),
            ]
        )

    # Selecting threshold on val and evaluating on test succeeds
    rc = evaluate_main(
        [
            "--gt",
            str(test_gt_p),
            "--pred",
            str(test_pred_p),
            "--select-thr-on",
            str(val_gt_p),
            str(val_pred_p),
            "--n-bootstrap",
            "25",
            "--seed",
            "11",
            "--out",
            str(out_p),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Loc AP50" in captured
    saved = json.loads(out_p.read_text(encoding="utf-8"))
    assert saved["score_thr"] == pytest.approx(0.70)
    assert saved["localization"]["ap50"] == pytest.approx(1.0)
    assert saved["meta"]["seconds_per_image"] == pytest.approx(0.42)

    # Report merges metrics.json into Markdown
    report_p = tmp_path / "RESULTS.md"
    assert report_main([str(out_p), "--out", str(report_p)]) == 0
    md = report_p.read_text(encoding="utf-8")
    assert "`exp_a`" in md
    assert "1.000" in md
    assert "0.42" in md
    assert "no runs" not in md.lower()

    empty_md = build_results_markdown([])
    assert "No experiment" in empty_md
