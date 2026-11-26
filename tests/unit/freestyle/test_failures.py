"""Unit tests for Freestyle failure analysis and visualization (synthetic data only)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from freestyle.catalog import class_names
from freestyle.eval.failures import (
    CLASS_ABBREV,
    FAILURE_TYPES,
    analyze,
    build_confusion_matrix,
    main as failures_main,
    short_label,
)


def _categories() -> list[dict]:
    return [
        {"id": 1, "name": "coca-cola-part1"},
        {"id": 2, "name": "diet-coke-part2"},
        {"id": 3, "name": "nns-sweetener"},
        {"id": 4, "name": "sprite"},
        {"id": 5, "name": "unknown"},
    ]


def _make_textured_image(width: int, height: int, sharp_box: tuple[int, int, int, int]) -> Image.Image:
    """Create a synthetic image with a flat background and high-frequency pattern in *sharp_box*."""
    arr = np.full((height, width, 3), 120, dtype=np.uint8)
    x, y, w, h = sharp_box
    yy, xx = np.mgrid[y : y + h, x : x + w]
    checker = ((xx + yy) % 2) * 255
    arr[y : y + h, x : x + w, 0] = checker.astype(np.uint8)
    arr[y : y + h, x : x + w, 1] = (255 - checker).astype(np.uint8)
    arr[y : y + h, x : x + w, 2] = 64
    return Image.fromarray(arr, mode="RGB")


def test_short_labels_cover_catalog_uniquely():
    names = class_names()
    shorts = [short_label(n) for n in names]
    assert len(shorts) == len(set(shorts)), f"Duplicate abbreviations: {shorts}"
    assert short_label("coca-cola-part1") == "CC-P1"
    assert short_label("diet-coke-part2") == "DC-P2"
    assert short_label("nns-sweetener") == "NNS"
    assert short_label("custom-flavor-x") == "CUS-FLA"
    for name in names:
        assert name in CLASS_ABBREV


def test_analyze_classifies_all_types_subtypes_sharpness_and_position(tmp_path: Path):
    w, h = 1000, 800
    img = _make_textured_image(w, h, sharp_box=(400, 100, 80, 100))
    img_path = tmp_path / "cam" / "frame_001.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(img_path)

    gt_coco = {
        "images": [{"id": 1, "width": w, "height": h, "file_name": "cam/frame_001.png"}],
        "categories": _categories(),
        "annotations": [
            # 0: TP (matched with same class)
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 100, 80, 100]},
            # 1: ID error (GT CC-P1 -> Pred DC-P2)
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [200, 100, 80, 100]},
            # 2: LOC error (IoU ~0.38 with pred at [435, 100, 80, 100])
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [400, 100, 80, 100]},
            # 3 & 4: adjacent_merge (pred covers GT 3 + GT 4 completely, IoU=0.5 with #3)
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [100, 300, 80, 100]},
            {"id": 5, "image_id": 1, "category_id": 2, "bbox": [180, 300, 80, 100]},
            # 5: edge_truncated FN (touches left border x=0)
            {"id": 6, "image_id": 1, "category_id": 4, "bbox": [0, 500, 60, 100]},
            # 6 & 7: stacked_tier (6 is TP bottom, 7 is FN stacked directly above 6)
            {"id": 7, "image_id": 1, "category_id": 1, "bbox": [600, 500, 80, 100]},
            {"id": 8, "image_id": 1, "category_id": 3, "bbox": [602, 390, 78, 100]},
            # 8: plain FN (isolated inside image)
            {"id": 9, "image_id": 1, "category_id": 2, "bbox": [800, 300, 80, 100]},
            # 9: unknown GT matched at IoU=1.0 -> should NOT produce an ID error
            {"id": 10, "image_id": 1, "category_id": 5, "bbox": [800, 500, 80, 100]},
        ],
    }

    preds = [
        # TP for GT 0
        {"image_id": 1, "category_id": 1, "bbox": [50, 100, 80, 100], "score": 0.95},
        # ID error for GT 1 (CC-P1 -> DC-P2)
        {
            "image_id": 1,
            "category_id": 2,
            "bbox": [200, 100, 80, 100],
            "score": 0.90,
            "id_score": 0.85,
            "shelf": 0,
        },
        # LOC error for GT 2 (overlap 45x100=4500, union 8000+8000-4500=11500 => IoU=0.3913)
        {
            "image_id": 1,
            "category_id": 3,
            "bbox": [435, 100, 80, 100],
            "score": 0.88,
            "extras": {"shelf": 0},
        },
        # Merged pred covering GT 3 & GT 4 (IoU=0.5 with GT 3 => matches GT 3 as TP, GT 4 is FN adjacent_merge)
        {"image_id": 1, "category_id": 1, "bbox": [100, 300, 160, 100], "score": 0.85},
        # TP for GT 6 (bottom of stack)
        {"image_id": 1, "category_id": 1, "bbox": [600, 500, 80, 100], "score": 0.84},
        # TP for unknown GT 9 (predicted as sprite)
        {"image_id": 1, "category_id": 4, "bbox": [800, 500, 80, 100], "score": 0.80},
        # FP duplicate overlapping GT 0's matched pred
        {"image_id": 1, "category_id": 1, "bbox": [54, 102, 80, 100], "score": 0.75, "shelf": 0},
        # FP fragment mostly inside GT 6
        {"image_id": 1, "category_id": 1, "bbox": [610, 510, 30, 40], "score": 0.70, "shelf": 1},
        # FP background away from all GTs
        {"image_id": 1, "category_id": 4, "bbox": [700, 50, 60, 80], "score": 0.65, "shelf": 0},
        # Below score_thr (0.5): must be ignored
        {"image_id": 1, "category_id": 4, "bbox": [700, 700, 60, 80], "score": 0.20},
    ]

    cases = analyze(gt_coco, preds, score_thr=0.5, iou=0.5, images_root=tmp_path)

    by_type: dict[str, list] = {t: [] for t in FAILURE_TYPES}
    for c in cases:
        by_type[c.type].append(c)

    assert len(by_type["ID"]) == 1
    id_case = by_type["ID"][0]
    assert id_case.confusion_pair == "CC-P1→DC-P2"
    assert id_case.gt_short == "CC-P1"
    assert id_case.pred_short == "DC-P2"
    assert id_case.shelf == 0
    assert math.isclose(id_case.norm_x, 240 / w, abs_tol=1e-3)
    assert math.isclose(id_case.norm_y, 150 / h, abs_tol=1e-3)

    assert len(by_type["LOC"]) == 1
    loc_case = by_type["LOC"][0]
    assert loc_case.iou is not None and 0.3 <= loc_case.iou < 0.5
    assert loc_case.shelf == 0
    # Checkerboard inside [400..480, 100..200] overlaps pred [435..515] -> very high sharpness.
    assert loc_case.sharpness is not None and loc_case.sharpness > 1000.0
    # Flat ID crop [200..280, 100..200] -> zero sharpness.
    assert id_case.sharpness == 0.0

    fn_subtypes = sorted(c.subtype for c in by_type["FN"])
    assert fn_subtypes == ["adjacent_merge", "edge_truncated", "plain", "stacked_tier"]

    fp_subtypes = sorted(c.subtype for c in by_type["FP"])
    assert fp_subtypes == ["background", "duplicate", "fragment"]


def test_confusion_counts_and_cli_outputs_sizes(tmp_path: Path):
    w, h = 800, 600
    img1 = _make_textured_image(w, h, sharp_box=(50, 50, 80, 100))
    img2 = _make_textured_image(w, h, sharp_box=(200, 50, 80, 100))
    img_root = tmp_path / "images"
    img_root.mkdir()
    img1.save(img_root / "a.png")
    img2.save(img_root / "b.png")

    gt_coco = {
        "images": [
            {"id": 10, "width": w, "height": h, "file_name": "a.png"},
            {"id": 20, "width": w, "height": h, "file_name": "b.png"},
        ],
        "categories": _categories(),
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [50, 50, 80, 100]},
            {"id": 2, "image_id": 10, "category_id": 1, "bbox": [200, 50, 80, 100]},
            {"id": 3, "image_id": 20, "category_id": 2, "bbox": [200, 50, 80, 100]},
            {"id": 4, "image_id": 20, "category_id": 3, "bbox": [400, 200, 80, 100]},
        ],
    }
    preds = [
        # Two CC-P1 -> DC-P2 errors on image 10
        {"image_id": 10, "category_id": 2, "bbox": [50, 50, 80, 100], "score": 0.9},
        {"image_id": 10, "category_id": 2, "bbox": [200, 50, 80, 100], "score": 0.85},
        # One DC-P2 -> CC-P1 error on image 20 + one background FP; GT 4 missed (plain FN)
        {"image_id": 20, "category_id": 1, "bbox": [200, 50, 80, 100], "score": 0.92},
        {"image_id": 20, "category_id": 4, "bbox": [600, 400, 60, 60], "score": 0.70},
    ]

    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "preds.json"
    out_dir = tmp_path / "out"
    gt_path.write_text(json.dumps(gt_coco), encoding="utf-8")
    pred_path.write_text(json.dumps(preds), encoding="utf-8")

    rc = failures_main(
        [
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--images-root",
            str(img_root),
            "--score-thr",
            "0.5",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0

    failures_json = out_dir / "failures.json"
    index_html = out_dir / "index.html"
    assert failures_json.is_file()
    assert index_html.is_file()

    report = json.loads(failures_json.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["counts_by_type"] == {"FN": 1, "FP": 1, "ID": 3, "LOC": 0}
    cm = summary["confusion_matrix"]
    assert cm["pairs"] == {"CC-P1→DC-P2": 2, "DC-P2→CC-P1": 1}
    assert cm["labels"] == ["CC-P1", "DC-P2"]
    assert cm["matrix"] == [[0, 2], [1, 0]]

    # Also verify build_confusion_matrix matches directly.
    cases = analyze(gt_coco, preds, score_thr=0.5, images_root=img_root)
    assert build_confusion_matrix(cases) == cm

    # Verify overlay PNGs exist at full original resolution and 1600px preview width.
    assert len(report["images"]) == 2
    for img_info in report["images"]:
        full_p = out_dir / img_info["overlay_path"]
        prev_p = out_dir / img_info["preview_path"]
        assert full_p.is_file()
        assert prev_p.is_file()
        with Image.open(full_p) as im:
            assert im.size == (w, h)
        with Image.open(prev_p) as prev_im:
            assert prev_im.size == (1600, round(h * 1600 / w))

    # Verify crop gallery PNGs exist for every failure type.
    for ftype in FAILURE_TYPES:
        g_p = out_dir / report["galleries"][ftype]
        assert g_p.is_file()
        with Image.open(g_p) as gim:
            assert gim.width > 0 and gim.height > 0

    # Verify HTML links everything locally with no CDN references.
    html_text = index_html.read_text(encoding="utf-8")
    assert "http://" not in html_text
    assert "https://" not in html_text
    assert "failures.json" in html_text
    assert "galleries/ID.png" in html_text
    assert "overlays/a.png" in html_text
    assert "overlays/a_preview.png" in html_text
    assert "CC-P1" in html_text
    assert "DC-P2" in html_text
