"""Unit tests for freestyle.augment (synthetic images/boxes only)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from freestyle.augment import (
    build_bank,
    color_cast,
    compose,
    fisheye_warp,
    jpeg_compress,
    low_light,
    motion_blur,
    slot_shuffle,
    synthesize,
    vignette,
)
from freestyle.train.export_tiles import export_tiles


def _make_shelf_fixture(tmp_path: Path):
    """Create synthetic 2-row shelf images (train + val + test) and COCO + splits."""
    w, h = 240, 180
    # Train image: uniform gray background with distinct colored cartridges per slot.
    train_img = Image.new("RGB", (w, h), (120, 120, 120))
    draw = ImageDraw.Draw(train_img)
    # Row 0 (y=15..75): 4 slots of size 36x60, categories [1, 1, 1, 2] (cat 2 is rare)
    row0_boxes = [[20, 15, 36, 60], [66, 15, 36, 60], [112, 15, 36, 60], [158, 15, 36, 60]]
    row0_cats = [1, 1, 1, 2]
    colors = {1: (220, 30, 30), 2: (30, 210, 40), 3: (30, 60, 230)}
    for box, cid in zip(row0_boxes, row0_cats, strict=True):
        x, y, bw, bh = box
        draw.rectangle([x, y, x + bw - 1, y + bh - 1], fill=colors[cid])
        draw.rectangle([x + 6, y + 10, x + bw - 7, y + bh - 11], fill=(250, 250, 250))

    # Row 1 (y=100..160): 3 slots of size 40x60, categories [1, 1, 1]
    row1_boxes = [[25, 100, 40, 60], [80, 100, 40, 60], [135, 100, 40, 60]]
    row1_cats = [1, 1, 1]
    for box, cid in zip(row1_boxes, row1_cats, strict=True):
        x, y, bw, bh = box
        draw.rectangle([x, y, x + bw - 1, y + bh - 1], fill=colors[cid])

    train_path = tmp_path / "train_01.png"
    train_img.save(train_path)

    # Val and Test images with unique category 3 that must NEVER leak into train bank.
    for name in ("val_01.png", "test_01.png"):
        im = Image.new("RGB", (w, h), (50, 50, 50))
        ImageDraw.Draw(im).rectangle([20, 20, 60, 80], fill=colors[3])
        im.save(tmp_path / name)

    annotations = []
    ann_id = 0
    for idx, (box, cid) in enumerate(
        zip(row0_boxes + row1_boxes, row0_cats + row1_cats, strict=True)
    ):
        ann_id += 1
        x, y, bw, bh = box
        ann = {
            "id": ann_id,
            "image_id": 1,
            "category_id": cid,
            "bbox": box,
            "area": bw * bh,
            "iscrowd": 0,
            "attributes": {
                "truncated": False,
                "occluded": False,
                "identity_visible": True,
                "source": "human",
            },
        }
        if idx == 0:
            ann["quad"] = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
        annotations.append(ann)

    # Add a synthetic annotation on train image to ensure build_bank excludes synthetic ones,
    # plus val/test annotations for category 3.
    annotations.append(
        {
            "id": 99,
            "image_id": 1,
            "category_id": 3,
            "bbox": [190, 100, 30, 50],
            "area": 1500,
            "iscrowd": 0,
            "attributes": {"source": "synthetic:copy_paste"},
        }
    )
    annotations.append(
        {
            "id": 100,
            "image_id": 2,
            "category_id": 3,
            "bbox": [20, 20, 40, 60],
            "area": 2400,
            "iscrowd": 0,
            "attributes": {"source": "human"},
        }
    )
    annotations.append(
        {
            "id": 101,
            "image_id": 3,
            "category_id": 3,
            "bbox": [20, 20, 40, 60],
            "area": 2400,
            "iscrowd": 0,
            "attributes": {"source": "human"},
        }
    )

    coco = {
        "images": [
            {"id": 1, "file_name": "train_01.png", "width": w, "height": h},
            {"id": 2, "file_name": "val_01.png", "width": w, "height": h},
            {"id": 3, "file_name": "test_01.png", "width": w, "height": h},
        ],
        "categories": [
            {"id": 1, "name": "coca-cola-part1"},
            {"id": 2, "name": "sprite"},
            {"id": 3, "name": "fanta-orange"},
        ],
        "annotations": annotations,
    }
    splits = {
        "version": 1,
        "groups": {"g1": "train", "g2": "val", "g3": "test"},
        "images": {"train_01.png": "train", "val_01.png": "val", "test_01.png": "test"},
    }
    return coco, splits, row0_boxes, row1_boxes


def test_build_bank_split_guard_and_quad(tmp_path):
    coco, splits, _, _ = _make_shelf_fixture(tmp_path)

    # Must refuse without a split filter.
    with pytest.raises(ValueError, match="split filter"):
        build_bank(coco, tmp_path)

    # Must refuse non-train splits.
    for bad_split in ("val", "test", None):
        with pytest.raises(ValueError, match="train"):
            build_bank(coco, tmp_path, split_map=splits, split=bad_split)

    bank = build_bank(coco, tmp_path, split_map=splits, split="train")
    assert len(bank) == 7
    assert {e["category_id"] for e in bank} == {1, 2}  # cat 3 from val/test/synthetic excluded
    assert all(e["source_file_name"] == "train_01.png" for e in bank)
    assert bank[0]["quad"] is not None
    assert bank[0]["quad_image"] is not None
    assert bank[1]["quad"] is None


def test_synthesize_coords_blending_balance_and_determinism(tmp_path):
    coco, splits, row0_boxes, row1_boxes = _make_shelf_fixture(tmp_path)
    bank = build_bank(coco, tmp_path, split_map=splits, split="train")
    bg = Image.open(tmp_path / "train_01.png")

    row_lines = [
        [{"bbox": b, "category_id": 1} for b in row0_boxes],
        [{"bbox": b, "category_id": 1} for b in row1_boxes],
    ]

    for mode in ("hybrid", "feather", "poisson"):
        out_a = synthesize(
            bg,
            row_lines,
            bank,
            n=2,
            rng=np.random.default_rng(123),
            class_balance=True,
            perspective_jitter=0.03,
            blend_mode=mode,
        )
        out_b = synthesize(
            bg,
            row_lines,
            bank,
            n=2,
            rng=np.random.default_rng(123),
            class_balance=True,
            perspective_jitter=0.03,
            blend_mode=mode,
        )
        assert len(out_a) == 2
        for (img_a, anns_a), (img_b, anns_b) in zip(out_a, out_b, strict=True):
            assert img_a.size == bg.size
            assert img_a.mode == bg.mode
            assert np.array_equal(np.asarray(img_a), np.asarray(img_b))
            assert anns_a == anns_b
            assert len(anns_a) == len(row0_boxes) + len(row1_boxes)
            expected_slots = row0_boxes + row1_boxes
            for ann, slot in zip(anns_a, expected_slots, strict=True):
                assert ann["attributes"]["source"] == "synthetic:copy_paste"
                bx, by, bw, bh = ann["bbox"]
                assert 0 <= bx < bg.width and 0 <= by < bg.height
                assert bx + bw <= bg.width + 1e-6 and by + bh <= bg.height + 1e-6
                assert bw > 0 and bh > 0
                assert ann["area"] == pytest.approx(bw * bh, abs=0.1)
                # Pasted box center remains close to original slot center.
                assert abs((bx + 0.5 * bw) - (slot[0] + 0.5 * slot[2])) <= 4.0
                assert abs((by + 0.5 * bh) - (slot[1] + 0.5 * slot[3])) <= 4.0

    # Verify class_balance=True oversamples rare class 2 relative to class_balance=False.
    balanced = synthesize(
        bg, row_lines, bank, n=25, rng=np.random.default_rng(7), class_balance=True, perspective_jitter=0.0
    )
    unbalanced = synthesize(
        bg, row_lines, bank, n=25, rng=np.random.default_rng(7), class_balance=False, perspective_jitter=0.0
    )
    bal_counts = Counter(a["category_id"] for _, anns in balanced for a in anns)
    unbal_counts = Counter(a["category_id"] for _, anns in unbalanced for a in anns)
    assert bal_counts[2] > unbal_counts[2]


def test_slot_shuffle_preserves_counts_per_row_and_determinism():
    w, h = 260, 180
    img = Image.new("RGB", (w, h), (100, 100, 100))
    draw = ImageDraw.Draw(img)
    # Row 0: 4 same-size slots with distinct categories [10, 20, 30, 40] + 1 different-size slot [50]
    row0 = [
        {"id": 1, "category_id": 10, "bbox": [10, 15, 30, 50], "attributes": {"source": "human"}},
        {"id": 2, "category_id": 20, "bbox": [50, 15, 30, 50], "attributes": {"source": "human"}},
        {"id": 3, "category_id": 30, "bbox": [90, 15, 30, 50], "attributes": {"source": "human"}},
        {"id": 4, "category_id": 40, "bbox": [130, 15, 30, 50], "attributes": {"source": "human"}},
        {"id": 5, "category_id": 50, "bbox": [175, 25, 65, 30], "attributes": {"source": "human"}},
    ]
    # Row 1: 3 same-size slots with categories [60, 70, 80]
    row1 = [
        {"id": 6, "category_id": 60, "bbox": [20, 105, 34, 52], "attributes": {"source": "human"}},
        {"id": 7, "category_id": 70, "bbox": [70, 105, 34, 52], "attributes": {"source": "human"}},
        {"id": 8, "category_id": 80, "bbox": [120, 105, 34, 52], "attributes": {"source": "human"}},
    ]
    for i, ann in enumerate(row0 + row1):
        x, y, bw, bh = ann["bbox"]
        draw.rectangle([x, y, x + bw - 1, y + bh - 1], fill=(20 + 25 * i, 180 - 15 * i, 60 + 20 * i))

    anns = row0 + row1
    shuf_a, anns_a = slot_shuffle(img, anns, rng=np.random.default_rng(42))
    shuf_b, anns_b = slot_shuffle(img, anns, rng=np.random.default_rng(42))

    assert shuf_a.size == img.size
    assert shuf_a.mode == img.mode
    assert np.array_equal(np.asarray(shuf_a), np.asarray(shuf_b))
    assert anns_a == anns_b

    # Exact counts preserved per row.
    assert Counter(a["category_id"] for a in anns_a[:5]) == Counter(a["category_id"] for a in row0)
    assert Counter(a["category_id"] for a in anns_a[5:]) == Counter(a["category_id"] for a in row1)
    # Different-sized slot in row 0 was NOT swapped with the narrow slots.
    assert anns_a[4]["category_id"] == 50
    # Positions unchanged.
    assert [a["bbox"] for a in anns_a] == [a["bbox"] for a in anns]
    # At least some same-size slots actually permuted.
    assert [a["category_id"] for a in anns_a[:4]] != [10, 20, 30, 40]


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_photometric_ops_preserve_size_mode_and_determinism(mode):
    arr = np.zeros((64, 80, 3), dtype=np.uint8)
    arr[:, ::8, :] = 220
    arr[::8, :, 1] = 200
    base = Image.fromarray(arr, mode="RGB").convert(mode)
    before = np.asarray(base).copy()

    ops = [
        lambda im, r: motion_blur(im, r, length_range=(3, 15)),
        lambda im, r: low_light(im, r, target_luma=70.0),
        lambda im, r: jpeg_compress(im, r, quality_range=(60, 90)),
        lambda im, r: color_cast(im, r),
        lambda im, r: vignette(im, r),
        lambda im, r: fisheye_warp(im, r, k1_range=(-0.3, -0.1)),
        compose(
            {
                "fisheye_warp": {"p": 1.0, "k1_range": (-0.25, -0.1)},
                "motion_blur": {"p": 1.0, "length_range": (3, 9)},
                "color_cast": {"p": 1.0},
                "low_light": {"p": 1.0, "target_luma": 70.0},
                "vignette": {"p": 1.0},
                "jpeg_compress": {"p": 1.0, "quality_range": (65, 85)},
            }
        ),
    ]

    for op in ops:
        out1 = op(base, np.random.default_rng(99))
        out2 = op(base, np.random.default_rng(99))
        assert out1.size == base.size
        assert out1.mode == base.mode
        assert np.array_equal(np.asarray(out1), np.asarray(out2))
        # Input image must not be mutated.
        assert np.array_equal(np.asarray(base), before)


def test_photometric_semantic_effects():
    bright = Image.new("RGB", (64, 64), (210, 210, 210))
    dim = low_light(bright, np.random.default_rng(1), target_luma=70.0, luma_jitter=5.0)
    mean_luma = np.asarray(dim, dtype=float).mean()
    assert 50.0 <= mean_luma <= 90.0

    vig = vignette(bright, np.random.default_rng(2), strength_range=(0.5, 0.5), center_jitter=0.0)
    v_arr = np.asarray(vig, dtype=float)
    assert v_arr[32, 32].mean() > v_arr[0, 0].mean() + 30.0


def test_cli_generates_exportable_dataset(tmp_path):
    coco, splits, _, _ = _make_shelf_fixture(tmp_path)
    coco_path = tmp_path / "coco.json"
    split_path = tmp_path / "splits.json"
    coco_path.write_text(json.dumps(coco))
    split_path.write_text(json.dumps(splits))

    out_dir = tmp_path / "synth_out"
    cmd = [
        sys.executable,
        "-m",
        "freestyle.augment.copy_paste",
        "--coco",
        str(coco_path),
        "--split-map",
        str(split_path),
        "--split",
        "train",
        "--images-root",
        str(tmp_path),
        "--n-per-image",
        "3",
        "--out",
        str(out_dir),
        "--seed",
        "4",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert proc.returncode == 0

    assert (out_dir / "annotations.json").exists()
    assert (out_dir / "splits.json").exists()
    assert (out_dir / "contact_sheet.png").exists()

    synth_coco = json.loads((out_dir / "annotations.json").read_text())
    assert len(synth_coco["images"]) == 3
    assert len(synth_coco["annotations"]) == 3 * 7
    assert all(
        ann["attributes"]["source"] == "synthetic:copy_paste" for ann in synth_coco["annotations"]
    )
    for img_rec in synth_coco["images"]:
        assert (out_dir / img_rec["file_name"]).exists()

    # Verify direct consumption by freestyle.train.export_tiles.
    tiles_dir = tmp_path / "tiles"
    manifest = export_tiles(
        out_dir / "annotations.json",
        out_dir,
        tiles_dir,
        out_dir / "splits.json",
        tile=128,
        overlap=0.25,
    )
    assert len(manifest) > 0
    exported = json.loads((tiles_dir / "train/_annotations.coco.json").read_text())
    assert len(exported["annotations"]) > 0

    # CLI must refuse non-train split.
    bad_cmd = [
        sys.executable,
        "-m",
        "freestyle.augment.copy_paste",
        "--coco",
        str(coco_path),
        "--split-map",
        str(split_path),
        "--split",
        "test",
        "--images-root",
        str(tmp_path),
        "--n-per-image",
        "1",
        "--out",
        str(tmp_path / "bad_out"),
    ]
    bad_proc = subprocess.run(bad_cmd, capture_output=True, text=True)
    assert bad_proc.returncode != 0
