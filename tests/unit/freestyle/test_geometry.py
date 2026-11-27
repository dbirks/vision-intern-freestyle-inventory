import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

_spec = importlib.util.spec_from_file_location(
    "freestyle_geometry", Path(__file__).resolve().parents[3] / "freestyle" / "geometry.py"
)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


def test_tiles_cover_and_shift_edges():
    tiles = g.make_tiles(2100, 1300, 1024, 0.25)
    assert tiles[-1] == (1076, 276, 2100, 1300)
    coverage = np.zeros((1300, 2100), dtype=bool)
    for x0, y0, x1, y1 in tiles:
        coverage[y0:y1, x0:x1] = True
    assert coverage.all()
    assert g.make_tiles(20, 30, 100, min_tile=50) == [(0, 0, 20, 30)]


def test_coordinate_mapping_and_edges():
    tile = (100, 50, 200, 150)
    boxes = [[10, 20, 40, 80], [-5, 0, 110, 100]]
    mapped = g.tile_boxes_to_image(boxes, tile)
    np.testing.assert_array_equal(mapped, [[110, 70, 140, 130], [100, 50, 200, 150]])
    np.testing.assert_array_equal(g.image_boxes_to_tile(mapped, tile), [[10, 20, 40, 80], [0, 0, 100, 100]])
    np.testing.assert_array_equal(g.drop_edge_truncated(mapped, tile, (0, 0, 200, 150)), [mapped[0]])
    np.testing.assert_array_equal(g.drop_edge_truncated([[0, 0, 20, 20]], (0, 0, 100, 100), (0, 0, 200, 200)), [[0, 0, 20, 20]])


def test_conversions_and_iou():
    boxes = np.array([[1, 2, 3, 4], [5, 6, 0, 0]])
    np.testing.assert_array_equal(g.xyxy_to_xywh(g.xywh_to_xyxy(boxes)), boxes)
    np.testing.assert_allclose(g.iou_matrix([[0, 0, 10, 10], [0, 0, 0, 0]], [[5, 0, 15, 10]]), [[1/3], [0]])
    assert g.containment([0, 0, 10, 10], [2, 2, 4, 4]) == 1


def test_nms_and_fusion():
    boxes = [[0, 0, 10, 10], [1, 0, 11, 10], [0, 0, 10, 10]]
    assert g.nms(boxes, [0.9, 0.8, 0.7], 0.5).tolist() == [0]
    assert g.class_aware_nms(boxes, [0.9, 0.8, 0.7], [1, 1, 2], 0.5).tolist() == [0, 2]
    dets = [[{"box": boxes[0], "score": 0.9, "category_id": 1}],
            [{"box": boxes[1], "score": 0.6, "category_id": 1}]]
    fused = g.weighted_box_fusion(dets)
    assert len(fused) == 1
    np.testing.assert_allclose(fused[0]["box"], [0.4, 0, 10.4, 10])
    assert fused[0]["score"] == pytest.approx(0.75)


def test_dense_and_fragments():
    dets = [{"box": [42*i, 0, 42*i+40, 200], "score": 0.9, "category_id": 1} for i in range(10)]
    dets.append({"box": [5, 20, 30, 100], "score": 0.5, "category_id": 1})
    for mode in ("nms", "wbf"):
        merged = g.merge_tiled(dets, mode=mode, containment_threshold=0.9)
        assert len(merged) == 10


def test_crop_padding_and_resize():
    image = Image.new("RGB", (100, 100))
    assert g.crop(image, [0, 0, 20, 10], pad=0.1).size == (22, 11)
    assert g.crop(image, [0, 0, 20, 10], pad=0, min_size=20).size == (40, 20)


def test_quad_rectifies_pattern():
    image = Image.new("RGB", (160, 150), "black")
    draw = ImageDraw.Draw(image)
    draw.polygon([(30, 20), (110, 20), (130, 120), (50, 120)], fill="white")
    draw.polygon([(30, 20), (70, 20), (90, 120), (50, 120)], fill="red")
    result = np.asarray(g.crop_quad(image, [[30, 20], [110, 20], [130, 120], [50, 120]], out_h=100))
    assert (result[20, 10] == [255, 0, 0]).all()
    assert (result[80, 10] == [255, 0, 0]).all()
    assert (result[20, -10] == [255, 255, 255]).all()
    assert (result[80, -10] == [255, 255, 255]).all()


def test_equal_score_fragment_is_suppressed_before_merge():
    fragment = {"box": [0, 0, 5, 10], "score": 0.8, "category_id": 1}
    full = {"box": [0, 0, 10, 10], "score": 0.8, "category_id": 1}
    for mode in ("nms", "wbf"):
        assert g.merge_tiled([fragment, full], mode=mode, containment_threshold=0.9) == [full]


def test_suppress_duplicates_keeps_higher_score_and_is_class_agnostic():
    keeper = {'box': [0, 0, 20, 40], 'score': .9, 'category_id': 1}
    dup = {'box': [3, 1, 23, 41], 'score': .5, 'category_id': 2}
    other = {'box': [40, 0, 60, 40], 'score': .7, 'category_id': 3}
    kept = g.suppress_duplicates([dup, other, keeper], iou=0.4, containment=0.7)
    assert [d['category_id'] for d in kept] == [3, 1]   # input order preserved


def test_suppress_duplicates_containment_keeps_higher_score_fragment():
    big = {'box': [0, 0, 20, 40], 'score': .4}
    fragment = {'box': [2, 2, 12, 22], 'score': .9}
    assert g.suppress_duplicates([big, fragment], containment=0.7) == [fragment]
    assert g.suppress_duplicates([big, fragment]) == [big, fragment]


def test_row_rule_catches_shifted_leaning_duplicate_but_keeps_tiers():
    full = {'box': [0, 0, 20, 40], 'score': .9}
    leaning = {'box': [10, 0, 30, 40], 'score': .6}   # IoU 1/3, half-width x-shift
    tier = {'box': [0, 45, 20, 85], 'score': .8}      # stacked below, no y-overlap
    far = {'box': [60, 0, 80, 40], 'score': .7}
    dets = [full, leaning, tier, far]
    rows = [0, 0, 0, 0]
    assert g.suppress_duplicates(dets, iou=0.4, rows=rows, row_iou=0.3) == [full, tier, far]
    # without row context the leaning duplicate survives the global threshold
    assert len(g.suppress_duplicates(dets, iou=0.4)) == 4
    # a box in another row is judged only by the global threshold
    assert len(g.suppress_duplicates(dets, iou=0.4, rows=[0, 1, 0, 0], row_iou=0.3)) == 4
    with pytest.raises(ValueError):
        g.suppress_duplicates(dets, row_iou=0.3)
    with pytest.raises(ValueError):
        g.suppress_duplicates(dets, iou=0.4, rows=[0, 1])
