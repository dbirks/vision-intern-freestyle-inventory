"""Unit tests for freestyle.identify (stub embedder, classifiers, LOGO eval, CLI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from freestyle.identify import (
    MODEL_SPECS,
    Embedder,
    KNNClassifier,
    LinearProbe,
    PrototypeClassifier,
    calibrate_unknown_threshold,
    leave_one_group_out_eval,
    letterbox,
    select_unknown_threshold,
)
from freestyle.identify import cli as identify_cli


def _solid_crop(color: tuple[int, int, int], size: tuple[int, int] = (40, 120)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_lazy_imports_avoid_heavy_deps():
    # Importing the module must work in the light env without loading torch/transformers.
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_model_registry_includes_expected_backbones():
    required = {
        "stub",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "google/siglip2-base-patch16-512",
        "google/siglip2-so400m-patch16-512",
    }
    assert required.issubset(MODEL_SPECS.keys())
    assert MODEL_SPECS["facebook/dinov3-vitb16-pretrain-lvd1689m"]["gated"] is True
    assert MODEL_SPECS["facebook/dinov2-base"]["gated"] is False
    assert MODEL_SPECS["google/siglip2-base-patch16-512"]["default_size"] == (512, 512)


def test_letterbox_preserves_aspect_ratio_without_stretching():
    # Narrow tall crop (40x200 -> 1:5 aspect) letterboxed to 224x224.
    tall = Image.new("RGB", (40, 200), (255, 0, 0))
    lb_tall = letterbox(tall, 224)
    assert lb_tall.size == (224, 224)
    arr = np.asarray(lb_tall)
    # Left & right borders must be neutral grey (128, 128, 128), not stretched red.
    assert np.array_equal(arr[112, 0], [128, 128, 128])
    assert np.array_equal(arr[112, -1], [128, 128, 128])
    # Non-grey columns should have width ~ round(40 * 224 / 200) = 45.
    non_grey_cols = np.where(np.any(arr[112] != 128, axis=1))[0]
    assert len(non_grey_cols) == pytest.approx(45, abs=1)
    # Full height is occupied by red.
    assert np.array_equal(arr[0, 112], [255, 0, 0])
    assert np.array_equal(arr[-1, 112], [255, 0, 0])

    # Wide short crop (200x40 -> 5:1 aspect) letterboxed to 512x512.
    wide = Image.new("RGB", (200, 40), (0, 0, 255))
    lb_wide = letterbox(wide, 512)
    assert lb_wide.size == (512, 512)
    arr_w = np.asarray(lb_wide)
    assert np.array_equal(arr_w[0, 256], [128, 128, 128])
    assert np.array_equal(arr_w[-1, 256], [128, 128, 128])
    non_grey_rows = np.where(np.any(arr_w[:, 256] != 128, axis=1))[0]
    assert len(non_grey_rows) == pytest.approx(round(40 * 512 / 200), abs=1)


def test_stub_embedder_deterministic_normalized_batched_and_tta():
    emb = Embedder("stub", batch_size=2)
    assert emb.tta_hflip is False

    crops = [
        _solid_crop((220, 20, 20)),
        _solid_crop((20, 220, 20)),
        _solid_crop((20, 20, 220)),
    ]
    z1 = emb.embed(crops)
    z2 = emb.embed(crops, batch_size=1)
    assert z1.shape == (3, emb.dim)
    assert z1.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(z1, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(z1, z2, atol=1e-6)

    # Empty input preserves (0, dim).
    empty = emb.embed([])
    assert empty.shape == (0, emb.dim)
    assert empty.dtype == np.float32

    # Asymmetric crop: left red, right blue.
    asym = Image.new("RGB", (80, 80), (220, 20, 20))
    ImageDraw.Draw(asym).rectangle([40, 0, 80, 80], fill=(20, 20, 220))
    asym_flip = asym.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    e_no_tta = emb.embed([asym, asym_flip], tta_hflip=False)
    assert float(e_no_tta[0] @ e_no_tta[1]) < 0.95

    e_tta = emb.embed([asym, asym_flip], tta_hflip=True)
    np.testing.assert_allclose(e_tta[0], e_tta[1], atol=1e-5)


def test_classifiers_and_unknown_thresholding():
    emb = Embedder("stub")
    X_train = emb.embed(
        [
            _solid_crop((230, 15, 15)),
            _solid_crop((210, 25, 25)),
            _solid_crop((15, 230, 15)),
            _solid_crop((25, 210, 25)),
            _solid_crop((15, 15, 230)),
            _solid_crop((25, 25, 210)),
        ]
    )
    y_train = np.array(["cola", "cola", "sprite", "sprite", "fanta", "fanta"])

    X_test = emb.embed([_solid_crop((220, 20, 20)), _solid_crop((20, 220, 20))])

    classifiers = [
        PrototypeClassifier(),
        KNNClassifier(k=2),
        LinearProbe(backend="numpy"),
        LinearProbe(backend="auto"),
    ]
    for clf in classifiers:
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        assert probs.shape == (2, 3)
        assert np.all(probs >= 0.0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        preds = clf.predict(X_test)
        assert preds.tolist() == ["cola", "sprite"]

    # Explicit orthogonal / OOD unknown rejection on Prototype, KNN(k=1), and LinearProbe.
    X_simple = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    y_simple = np.array(["a", "a", "b", "b"])
    X_ood = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    for clf in (PrototypeClassifier(), KNNClassifier(k=1), LinearProbe(backend="numpy")):
        clf.fit(X_simple, y_simple)
        preds = clf.predict(X_ood, unknown_thr=0.8)
        assert preds[0] == "a"
        assert preds[1] == "unknown"

    # Integer class labels mixed with "unknown" must keep int equality for accepted items.
    proto_int = PrototypeClassifier().fit(X_simple, np.array([10, 10, 20, 20]))
    preds_int = proto_int.predict(X_ood, unknown_thr=0.8)
    assert preds_int[0] == 10
    assert preds_int[1] == "unknown"


def test_leave_one_group_out_eval_never_leaks_group():
    # Embed sample index in an orthogonal block to inspect exact training/query split per fold.
    emb = Embedder("stub")
    colors = {
        "coca-cola": (220, 20, 20),
        "sprite": (20, 220, 20),
        "fanta": (240, 140, 10),
    }
    imgs, labels, groups = [], [], []
    for grp in ("g1", "g2", "g3"):
        for cls_name, rgb in colors.items():
            # Slightly vary brightness across groups while keeping color identity clear.
            delta = {"g1": 0, "g2": -10, "g3": 10}[grp]
            c = tuple(int(np.clip(v + delta, 0, 255)) for v in rgb)
            imgs.append(_solid_crop(c))
            labels.append(cls_name)
            groups.append(grp)

    X = emb.embed(imgs)
    y = np.array(labels)
    groups_arr = np.array(groups)

    # Track exact vectors seen in fit() vs predict() to guarantee zero group leakage.
    seen_folds: list[tuple[set[int], set[int]]] = []
    row_to_idx = {tuple(np.round(row, 6).tolist()): i for i, row in enumerate(X)}

    class SpyClassifier:
        def __init__(self) -> None:
            self.inner = PrototypeClassifier()

        def fit(self, X_tr, y_tr):
            tr_idx = {row_to_idx[tuple(np.round(r, 6).tolist())] for r in X_tr}
            self._tr_idx = tr_idx
            self.inner.fit(X_tr, y_tr)
            return self

        def predict_with_scores(self, X_va, unknown_thr=None, unknown_label="unknown"):
            va_idx = {row_to_idx[tuple(np.round(r, 6).tolist())] for r in X_va}
            seen_folds.append((self._tr_idx, va_idx))
            return self.inner.predict_with_scores(
                X_va, unknown_thr=unknown_thr, unknown_label=unknown_label
            )

    res = leave_one_group_out_eval(X, y, groups_arr, SpyClassifier)
    assert res["n_groups"] == 3
    assert res["n_samples"] == 9
    assert len(seen_folds) == 3

    all_val_indices: set[int] = set()
    for tr_idx, va_idx in seen_folds:
        assert tr_idx.isdisjoint(va_idx)
        tr_groups = {groups[i] for i in tr_idx}
        va_groups = {groups[i] for i in va_idx}
        assert tr_groups.isdisjoint(va_groups)
        assert len(va_groups) == 1
        all_val_indices |= va_idx
    assert all_val_indices == set(range(9))

    assert res["accuracy"] == pytest.approx(1.0)
    assert res["macro_f1"] == pytest.approx(1.0)
    assert res["confusion_matrix"] == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    assert all(v == pytest.approx(1.0) for v in res["per_class_recall"].values())

    # Unknown threshold high enough to reject everything yields 0 coverage & unknown column.
    res_rej = leave_one_group_out_eval(X, y, groups_arr, PrototypeClassifier, unknown_thr=1.1)
    assert res_rej["coverage"] == pytest.approx(0.0)
    assert res_rej["accuracy"] == pytest.approx(0.0)
    assert res_rej["pred_labels"][-1] == "unknown"


def test_select_unknown_threshold_maximizes_accuracy_with_coverage():
    scores = np.array([0.95, 0.90, 0.80, 0.40, 0.30])
    y_pred = np.array(["a", "b", "a", "a", "b"])
    y_true = np.array(["a", "b", "a", "b", "a"])  # first 3 right, last 2 wrong

    cal = select_unknown_threshold(scores, y_pred, y_true, target_coverage=0.6)
    assert cal["target_met"] is True
    assert cal["coverage"] == pytest.approx(0.6)
    assert cal["covered_accuracy"] == pytest.approx(1.0)
    assert cal["threshold"] == pytest.approx(0.6)  # midpoint between 0.80 and 0.40

    # End-to-end calibration helper with GT unknown samples in validation set.
    X_gal = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    y_gal = np.array(["a", "b"])
    clf = PrototypeClassifier().fit(X_gal, y_gal)

    X_val = np.array(
        [[0.98, 0.1, 0.0], [0.1, 0.98, 0.0], [0.0, 0.0, 1.0], [0.1, 0.1, 0.98]],
        dtype=np.float32,
    )
    y_val = np.array(["a", "b", "unknown", "unknown"])
    cal_clf = calibrate_unknown_threshold(clf, X_val, y_val, target_coverage=0.5)
    assert cal_clf["overall_accuracy"] == pytest.approx(1.0)
    preds = clf.predict(X_val, unknown_thr=cal_clf["threshold"])
    assert preds.tolist() == ["a", "b", "unknown", "unknown"]


def test_cli_embed_and_eval_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    images_root = tmp_path / "images"
    (images_root / "sess1").mkdir(parents=True)
    (images_root / "sess2").mkdir(parents=True)

    # Image 1 (sess1): red at [10, 10, 40, 100], blue at [80, 10, 40, 100]
    im1 = Image.new("RGB", (200, 150), (30, 30, 30))
    d1 = ImageDraw.Draw(im1)
    d1.rectangle([10, 10, 50, 110], fill=(230, 15, 15))
    d1.rectangle([80, 10, 120, 110], fill=(15, 15, 230))
    im1.save(images_root / "sess1" / "frame1.png")

    # Image 2 (sess2): red slanted quad + green box
    im2 = Image.new("RGB", (200, 150), (30, 30, 30))
    d2 = ImageDraw.Draw(im2)
    d2.polygon([(20, 20), (60, 20), (70, 120), (30, 120)], fill=(230, 15, 15))
    d2.rectangle([100, 15, 140, 115], fill=(15, 230, 15))
    im2.save(images_root / "sess2" / "frame1.png")

    # Image 3 (sess3): green and blue so all 3 classes appear in at least 2 groups
    (images_root / "sess3").mkdir(parents=True)
    im3 = Image.new("RGB", (200, 150), (30, 30, 30))
    d3 = ImageDraw.Draw(im3)
    d3.rectangle([15, 15, 55, 115], fill=(15, 230, 15))
    d3.rectangle([90, 15, 130, 115], fill=(15, 15, 230))
    im3.save(images_root / "sess3" / "frame1.png")

    coco = {
        "images": [
            {"id": 1, "file_name": "sess1/frame1.png", "width": 200, "height": 150},
            {"id": 2, "file_name": "sess2/frame1.png", "width": 200, "height": 150},
            {"id": 3, "file_name": "sess3/frame1.png", "width": 200, "height": 150},
        ],
        "categories": [
            {"id": 1, "name": "red-cartridge"},
            {"id": 2, "name": "blue-cartridge"},
            {"id": 3, "name": "green-cartridge"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 100]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [80, 10, 40, 100]},
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [20, 20, 50, 100],
                "quad": [[20, 20], [60, 20], [70, 120], [30, 120]],
            },
            {"id": 4, "image_id": 2, "category_id": 3, "bbox": [100, 15, 40, 100]},
            {"id": 5, "image_id": 3, "category_id": 3, "bbox": [15, 15, 40, 100]},
            {"id": 6, "image_id": 3, "category_id": 2, "bbox": [90, 15, 40, 100]},
        ],
    }
    coco_path = tmp_path / "gt.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    splits = {
        "version": 1,
        "groups": {"sess1": "train", "sess2": "val", "sess3": "val"},
        "images": {
            "sess1/frame1.png": "train",
            "sess2/frame1.png": "val",
            "sess3/frame1.png": "val",
        },
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits), encoding="utf-8")

    npz_path = tmp_path / "crops.npz"
    rc = identify_cli.main(
        [
            "embed",
            "--split-map", str(splits_path),
            "--coco",
            str(coco_path),
            "--images-root",
            str(images_root),
            "--out",
            str(npz_path),
            "--model",
            "stub",
            "--pad",
            "0.05",
            "--quad",
        ]
    )
    assert rc == 0
    assert npz_path.is_file()

    with np.load(npz_path, allow_pickle=False) as d:
        assert d["embeddings"].shape[0] == 6
        assert d["used_quad"].tolist() == [False, False, True, False, False, False]
        np.testing.assert_allclose(np.linalg.norm(d["embeddings"], axis=1), 1.0, atol=1e-5)

    for clf_name in ("proto", "knn", "linear"):
        out_json = tmp_path / f"metrics_{clf_name}.json"
        eval_args = [
            "eval",
            "--npz",
            str(npz_path),
            "--split-map",
            str(splits_path),
            "--include-val",
            "--clf",
            clf_name,
            "--out",
            str(out_json),
        ]
        if clf_name == "knn":
            eval_args.extend(["--k", "1"])
        rc_eval = identify_cli.main(eval_args)
        assert rc_eval == 0
        metrics = json.loads(out_json.read_text(encoding="utf-8"))
        assert metrics["n_samples"] == 6
        assert metrics["n_groups"] == 3
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)

    # Also verify fallback crop path works when freestyle.geometry is unavailable.
    monkeypatch.setitem(sys.modules, "freestyle.geometry", None)
    crop_fn, quad_fn = identify_cli._get_crop_fns()
    assert crop_fn is identify_cli._fallback_crop
    assert quad_fn is identify_cli._fallback_crop_quad
    c_img = crop_fn(im1, [10, 10, 50, 110], pad=0.05)
    assert c_img.size[0] >= 40 and c_img.size[1] >= 100


def test_cli_eval_stdout_unknown_exclusion_and_calibration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    npz_path = tmp_path / "data.npz"
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.98, 0.1, 0.0],
            [0.1, 0.98, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savez(
        npz_path,
        embeddings=embeddings,
        labels=np.array(["a", "b", "a", "b", "unknown"], dtype=str),
        file_names=np.array(["g1/1.png", "g1/2.png", "g2/1.png", "g2/2.png", "g2/3.png"], dtype=str),
        image_ids=np.array(["1", "2", "3", "4", "5"], dtype=str),
        model=np.array(["stub"], dtype=str),
    )

    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({"groups": {"g1": "train", "g2": "train"},
        "images": {f"g{g}/{i}.png": "train" for g, indices in [(1, (1, 2)), (2, (1, 2, 3))] for i in indices}}))
    rc = identify_cli.main(
        [
            "eval",
            "--npz",
            str(npz_path),
            "--split-map", str(split_path),
            "--clf",
            "proto",
            "--calibrate-unknown-coverage",
            "0.75",
        ]
    )
    assert rc == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["n_samples"] == 4
    assert out_json["n_excluded_unknown"] == 1
    assert out_json["n_groups"] == 2
    assert out_json["accuracy"] == pytest.approx(1.0)
    assert out_json["calibrated_unknown"]["target_met"] is True


def test_hf_embedder_raises_helpful_error_without_train_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="train"):
        Embedder("facebook/dinov2-base")



def test_identification_rejects_missing_split_provenance():
    with pytest.raises(ValueError, match='provenance'):
        identify_cli._resolve_groups_and_splits(['g/a'], ['1'], None)


def test_holdout_uses_train_gallery_only(tmp_path):
    path = tmp_path / 'embeddings.npz'
    np.savez(path, embeddings=np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
             labels=['a', 'b', 'a', 'b'],
             file_names=['train/a', 'train/b', 'test/a', 'exclude/a'],
             image_ids=['1', '2', '3', '4'])
    split = tmp_path / 'splits.json'
    split.write_text(json.dumps({'groups': {'train': 'train', 'test': 'test', 'exclude': 'exclude'},
        'images': {'train/a': 'train', 'train/b': 'train', 'test/a': 'test', 'exclude/a': 'exclude'}}))
    args = identify_cli.build_parser().parse_args(['eval', '--npz', str(path), '--split-map', str(split),
                                                  '--mode', 'holdout', '--split', 'test'])
    result = identify_cli.run_eval(args)
    assert result['n_samples'] == 1 and result['accuracy'] == 1
