"""End-to-end CPU integration test of the Freestyle stack on synthetic data.

Exercises in order:
  export_tiles -> TiledDetector (fake jittered-GT backend) -> identify (stub) ->
  pipeline (YAML config) -> eval.evaluate (val threshold) -> eval.failures ->
  eval.report -> bootstrap_labels (monkeypatched VLM) -> review.server.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import socket
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from freestyle import bootstrap_labels
from freestyle.eval import evaluate as eval_cli
from freestyle.eval import failures as failures_cli
from freestyle.eval import report as report_cli
from freestyle.geometry import image_boxes_to_tile, make_tiles, xywh_to_xyxy
from freestyle.identify import cli as identify_cli
from freestyle.infer.detector import TiledDetector, predict_coco
from freestyle.pipeline import EmbeddingIdentifier, Pipeline
from freestyle.pipeline import main as pipeline_main
from freestyle.review import server as review_server
from freestyle.train.export_tiles import export_tiles
from tests.integration.freestyle import synth_shelf
from tools.vlm_client import VLMResponse


class FakeTileDetectorBackend:
    """Fake per-tile detector returning slightly jittered GT boxes visible in each tile."""

    class_names = ["cartridge"]

    def __init__(
        self,
        gt_coco: dict[str, Any],
        images_root: Path,
        *,
        tile: int = 1024,
        overlap: float = 0.25,
        min_tile_vis: float = 0.35,
        add_low_score_fp_ids: set[int] | None = None,
    ) -> None:
        self.tile = tile
        self.overlap = overlap
        self.min_tile_vis = min_tile_vis
        self.add_low_score_fp_ids = add_low_score_fp_ids or set()
        self._tile_index: dict[bytes, tuple[int, tuple[int, int, int, int]]] = {}
        self._anns_by_img: dict[int, list[dict[str, Any]]] = {}

        for ann in gt_coco["annotations"]:
            self._anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_rec in gt_coco["images"]:
            img_id = img_rec["id"]
            with Image.open(images_root / img_rec["file_name"]) as opened:
                rgb = opened.convert("RGB")
                for rect in make_tiles(rgb.width, rgb.height, tile, overlap):
                    sig = hashlib.sha256(rgb.crop(rect).tobytes()).digest()
                    self._tile_index[sig] = (img_id, rect)

    def predict(self, tile_image: Image.Image, threshold: float) -> list[dict[str, Any]]:
        sig = hashlib.sha256(tile_image.tobytes()).digest()
        img_id, rect = self._tile_index[sig]
        tx0, ty0, tx1, ty1 = rect

        dets: list[dict[str, Any]] = []
        for idx, ann in enumerate(self._anns_by_img.get(img_id, [])):
            x, y, w, h = ann["bbox"]
            inter_w = max(0.0, min(x + w, tx1) - max(x, tx0))
            inter_h = max(0.0, min(y + h, ty1) - max(y, ty0))
            vis_frac = (inter_w * inter_h) / max(w * h, 1e-6)
            if vis_frac < self.min_tile_vis:
                continue

            local = image_boxes_to_tile([[x, y, x + w, y + h]], rect)[0]
            dx = ((idx % 3) - 1) * 2.0
            dy = (((idx + 1) % 3) - 1) * 2.0
            bx0 = max(0.0, float(local[0] + dx))
            by0 = max(0.0, float(local[1] + dy))
            bx1 = min(float(tx1 - tx0), float(local[2] + dx))
            by1 = min(float(ty1 - ty0), float(local[3] + dy))
            if bx1 <= bx0 or by1 <= by0:
                continue
            score = (0.88 + (idx % 5) * 0.02) * (0.9 + 0.1 * min(1.0, vis_frac))
            if score >= threshold:
                dets.append({"box": [bx0, by0, bx1, by1], "score": score, "category_id": 0})

        if img_id in self.add_low_score_fp_ids and rect == (0, 0, self.tile, self.tile):
            fp_score = 0.25
            if fp_score >= threshold:
                dets.append({"box": [40.0, 40.0, 110.0, 210.0], "score": fp_score, "category_id": 0})
        return dets


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _http_json(port: int, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(
        method,
        path,
        body=json.dumps(body) if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    ctype = resp.getheader("Content-Type") or ""
    parsed = json.loads(raw) if "application/json" in ctype else None
    return resp.status, parsed, raw


def test_freestyle_e2e_synthetic_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 0. Generate synthetic 2048x1536 barrel-distorted cabinet dataset.
    ds = synth_shelf.generate_dataset(tmp_path / "synth", seed=42)
    images_root: Path = ds["images_root"]
    splits_path: Path = ds["splits_path"]
    gt_all_path: Path = ds["gt_all_path"]
    train_coco_path: Path = ds["train_coco_path"]
    val_coco_path: Path = ds["val_coco_path"]
    test_coco_path: Path = ds["test_coco_path"]
    catalog_path: Path = ds["catalog_path"]
    full_coco: dict[str, Any] = ds["full_coco"]
    cat_map: dict[str, int] = ds["category_map"]

    assert len(full_coco["images"]) == 7
    assert len(ds["splits"]["negatives"]) == 2
    for img_rec in full_coco["images"]:
        p = images_root / img_rec["file_name"]
        assert p.is_file()
        with Image.open(p) as im:
            assert im.size == (2048, 1536)

    # 1. freestyle.train.export_tiles on train split.
    tiles_dir = tmp_path / "tiles"
    manifest = export_tiles(
        train_coco_path,
        images_root,
        tiles_dir,
        splits_path,
        tile=1024,
        overlap=0.25,
        min_visible=0.6,
        class_agnostic=True,
    )
    assert len(manifest) == 12  # 2 train images * 6 tiles
    train_tile_coco = json.loads((tiles_dir / "train/_annotations.coco.json").read_text())
    assert train_tile_coco["categories"] == [{"id": 1, "name": "cartridge"}]
    assert len(train_tile_coco["images"]) == 12
    assert len(train_tile_coco["annotations"]) >= 34

    # 2. freestyle.infer.detector.TiledDetector with fake backend returning jittered GT per tile.
    val_ids = {img["id"] for img in json.loads(val_coco_path.read_text())["images"]}
    test_ids = {img["id"] for img in json.loads(test_coco_path.read_text())["images"]}
    fp_ids = {min(val_ids), min(test_ids)}
    fake_backend = FakeTileDetectorBackend(
        full_coco,
        images_root,
        tile=1024,
        overlap=0.25,
        add_low_score_fp_ids=fp_ids,
    )
    detector = TiledDetector(
        checkpoint="fake-ckpt",
        backend=fake_backend,
        tile=1024,
        overlap=0.25,
        score_thr=0.1,
        merge="nms",
        containment_threshold=0.75,
        iou=0.5,
    )

    det_coco = dict(full_coco)
    det_coco["categories"] = [*full_coco["categories"], {"id": len(cat_map) + 1, "name": "cartridge"}]
    det_coco_path = tmp_path / "det_input.json"
    det_coco_path.write_text(json.dumps(det_coco))

    det_preds_path = tmp_path / "det_run/predictions.json"
    det_preds = predict_coco(
        detector,
        det_coco_path,
        images_root,
        det_preds_path,
        class_agnostic=True,
    )
    det_meta = json.loads((tmp_path / "det_run/meta.json").read_text())
    assert det_meta["image_count"] == 7
    assert all(p["image_id"] not in {img["id"] for img in full_coco["images"] if img["attributes"]["scene"] == "no_cabinet"} for p in det_preds)

    # 3. freestyle.identify (stub embedder, gallery from train crops, holdout on test).
    all_crops_npz = tmp_path / "identify/all_crops.npz"
    train_gallery_npz = tmp_path / "identify/train_gallery.npz"
    id_eval_json = tmp_path / "identify/holdout_test.json"

    assert (
        identify_cli.main(
            [
                "embed",
                "--split-map",
                str(splits_path),
                "--coco",
                str(gt_all_path),
                "--images-root",
                str(images_root),
                "--out",
                str(all_crops_npz),
                "--model",
                "stub",
                "--quad",
            ]
        )
        == 0
    )
    assert (
        identify_cli.main(
            [
                "eval",
                "--npz",
                str(all_crops_npz),
                "--split-map",
                str(splits_path),
                "--mode",
                "holdout",
                "--split",
                "test",
                "--clf",
                "knn",
                "--k",
                "1",
                "--out",
                str(id_eval_json),
            ]
        )
        == 0
    )
    id_metrics = json.loads(id_eval_json.read_text())
    assert id_metrics["n_samples"] == 36
    assert id_metrics["accuracy"] == pytest.approx(1.0)
    assert id_metrics["macro_f1"] == pytest.approx(1.0)

    # Train-only gallery (using axis-aligned crops to match Pipeline's EmbeddingIdentifier).
    assert (
        identify_cli.main(
            [
                "embed",
                "--split-map",
                str(splits_path),
                "--coco",
                str(train_coco_path),
                "--images-root",
                str(images_root),
                "--out",
                str(train_gallery_npz),
                "--model",
                "stub",
            ]
        )
        == 0
    )

    # 4. freestyle.pipeline from YAML config written to tmp.
    pipe_cat_map = {**cat_map, "cartridge": len(cat_map) + 1}
    pipeline_yaml = tmp_path / "pipeline.yaml"
    cat_yaml = "\n".join(f"  {k}: {v}" for k, v in pipe_cat_map.items())
    pipeline_yaml.write_text(
        f"""images_root: {images_root}
manifest: {gt_all_path}
split_map: {splits_path}
categories:
{cat_yaml}
localizer:
  type: proposals
  path: {det_preds_path}
identifier:
  type: embedding_knn
  gallery: {train_gallery_npz}
  model: stub
  k: 1
counting:
  score_thr: 0.0
  identity_thr: 0.5
  neighbour_vote: "off"
  max_rows: 4
"""
    )

    val_run_dir = tmp_path / "runs/val_run"
    test_run_dir = tmp_path / "runs/test_run"
    assert pipeline_main(["--config", str(pipeline_yaml), "--images", "val", "--out", str(val_run_dir)]) == 0
    assert pipeline_main(["--config", str(pipeline_yaml), "--images", "test", "--out", str(test_run_dir)]) == 0

    for run_dir in (val_run_dir, test_run_dir):
        assert (run_dir / "predictions.json").is_file()
        assert (run_dir / "counts.json").is_file()
        assert list((run_dir / "overlays").glob("*.png"))

    # Verify shelf assignment (2 shelves + 2-high stacked column) on the clean test image output.
    clean_test_id = sorted(test_ids)[1]
    clean_test_res = json.loads((test_run_dir / f"images/{clean_test_id}.json").read_text())
    shelves = {d["shelf"] for d in clean_test_res["detections"] if d["score"] >= 0.5}
    assert shelves == {0, 1}
    tiers = [d["tier"] for d in clean_test_res["detections"] if d["score"] >= 0.5]
    assert 1 in tiers

    # Also verify direct Pipeline(TiledDetector, EmbeddingIdentifier) on a negative image.
    neg_img_rec = next(img for img in full_coco["images"] if img["attributes"]["scene"] == "no_cabinet")
    direct_pipe = Pipeline(
        detector,
        EmbeddingIdentifier(train_gallery_npz, model="stub", k=1),
        score_thr=0.5,
        identity_thr=0.5,
    )
    with Image.open(images_root / neg_img_rec["file_name"]) as neg_im:
        neg_res = direct_pipe.run(neg_im.convert("RGB"), neg_img_rec["id"], group=neg_img_rec["group_id"])
    assert neg_res["detections"] == []
    assert neg_res["counts"] == {}
    assert neg_res["unknown"] == 0

    # 5. freestyle.eval.evaluate (threshold selected on val).
    metrics_json = test_run_dir / "metrics.json"
    assert (
        eval_cli.main(
            [
                "--gt",
                str(test_coco_path),
                "--pred",
                str(test_run_dir / "predictions.json"),
                "--split-map",
                str(splits_path),
                "--select-thr-on",
                str(val_coco_path),
                str(val_run_dir / "predictions.json"),
                "--n-bootstrap",
                "40",
                "--experiment",
                "synth-e2e",
                "--out",
                str(metrics_json),
            ]
        )
        == 0
    )
    metrics = json.loads(metrics_json.read_text())
    assert metrics["score_thr_source"].startswith("selected_on:")
    assert 0.25 < metrics["score_thr"] <= 0.95
    assert metrics["n_images"] == 2
    assert metrics["n_groups"] == 2
    assert metrics["bootstrap_ci95"]["grouping"] == "grouped"
    assert metrics["localization"]["ap50"] > 0.8
    assert metrics["localization"]["recall"] == pytest.approx(1.0)
    assert metrics["localization"]["precision"] == pytest.approx(1.0)
    assert metrics["identification"]["top1_acc"] == pytest.approx(1.0)
    assert metrics["identification"]["macro_f1"] == pytest.approx(1.0)
    assert metrics["counting"]["exact_inventory_rate"] == pytest.approx(1.0)
    assert metrics["counting"]["total_count_mae"] == pytest.approx(0.0)
    assert metrics["counting"]["per_class_count_mae"] == pytest.approx(0.0)

    # 6. freestyle.eval.failures.
    failures_dir = tmp_path / "failures"
    assert (
        failures_cli.main(
            [
                "--gt",
                str(test_coco_path),
                "--pred",
                str(test_run_dir / "predictions.json"),
                "--images-root",
                str(images_root),
                "--score-thr",
                str(metrics["score_thr"]),
                "--out",
                str(failures_dir),
            ]
        )
        == 0
    )
    failures_data = json.loads((failures_dir / "failures.json").read_text())
    assert failures_data["summary"]["n_images"] == 2
    assert failures_data["summary"]["n_tp"] == 36
    assert failures_data["summary"]["n_failures"] == 0
    assert (failures_dir / "index.html").is_file()
    for ftype in ("FN", "FP", "ID", "LOC"):
        assert (failures_dir / "galleries" / f"{ftype}.png").is_file()
    assert len(list((failures_dir / "overlays").glob("*_preview.png"))) == 2

    # 7. freestyle.eval.report.
    report_md = tmp_path / "RESULTS.md"
    assert report_cli.main([str(metrics_json), "--out", str(report_md)]) == 0
    md_text = report_md.read_text()
    assert "| Experiment |" in md_text
    assert "`synth-e2e`" in md_text

    # 8. freestyle.bootstrap_labels with monkeypatched VLM calls.
    val_coco = json.loads(val_coco_path.read_text())
    boot_img_rec = val_coco["images"][0]
    boot_anns = [a for a in val_coco["annotations"] if a["image_id"] == boot_img_rec["id"]]
    id_to_slug = {c["id"]: c["name"] for c in val_coco["categories"]}

    # Build expected contact-sheet crop signatures -> true class slug for the stubbed VLM identifier.
    with Image.open(images_root / boot_img_rec["file_name"]) as opened:
        boot_pil = opened.convert("RGB")

    def fake_run_vlm(image: Image.Image, prompt: str, model_id: str, **kwargs: Any) -> VLMResponse:
        if "bbox_2d" in prompt:
            payload = []
            for ann in boot_anns:
                x, y, w, h = ann["bbox"]
                payload.append(
                    {
                        "bbox_2d": [
                            round(x * 1000.0 / boot_pil.width, 1),
                            round(y * 1000.0 / boot_pil.height, 1),
                            round((x + w) * 1000.0 / boot_pil.width, 1),
                            round((y + h) * 1000.0 / boot_pil.height, 1),
                        ],
                        "label": "cartridge",
                        "sub_label": "",
                    }
                )
            return VLMResponse(json.dumps(payload), finish_reason="stop")

        if "quality-control judge" in prompt:
            indices = [int(m) for m in re.findall(r"#(\d+):", prompt)]
            verdicts = [
                {"id": idx, "verdict": "correct", "score": 0.95, "reason": "ok"}
                for idx in indices
            ]
            return VLMResponse(json.dumps(verdicts), finish_reason="stop")

        if "index, class, and confidence" in prompt:
            # Match each numbered cell on the contact sheet to the nearest GT crop by RGB mean.
            n_cells = len(boot_anns)
            cell_w = max(1, image.width // n_cells)
            # Precompute GT crop means for this image.
            gt_means = []
            for ann in boot_anns:
                x0, y0, x1, y1 = xywh_to_xyxy(ann["bbox"]).tolist()
                crop_arr = np.asarray(
                    boot_pil.crop((int(x0), int(y0), int(x1), int(y1))), dtype=np.float32
                )
                gt_means.append((crop_arr.mean(axis=(0, 1)), id_to_slug[ann["category_id"]]))
            out_items = []
            for cell_idx in range(n_cells):
                cell = image.crop((cell_idx * cell_w, 24, (cell_idx + 1) * cell_w, image.height))
                # Exclude white contact-sheet padding when averaging.
                arr = np.asarray(cell, dtype=np.float32)
                fg = arr[np.any(arr < 245, axis=2)]
                mean_rgb = fg.mean(axis=0) if len(fg) else arr.mean(axis=(0, 1))
                best_slug = min(gt_means, key=lambda pair: float(np.linalg.norm(pair[0] - mean_rgb)))[1]
                out_items.append({"index": cell_idx, "class": best_slug, "confidence": 0.92})
            return VLMResponse(json.dumps(out_items), finish_reason="stop")

        raise AssertionError(f"Unexpected VLM prompt: {prompt[:120]}")

    monkeypatch.setattr(bootstrap_labels, "run_vlm", fake_run_vlm)

    boot_cfg = {
        "images_root": str(images_root),
        "cache_dir": str(tmp_path / "boot_cache"),
        "catalog": str(catalog_path),
        "audit": str(tmp_path / "missing_audit.json"),
        "primary_labeller": "mock-labeller-a",
        "labellers": [
            {"name": "run-a", "model": "mock-labeller-a"},
            {"name": "run-b", "model": "mock-labeller-b"},
        ],
        "judge": {"model": "mock-judge-b", "batch_size": 8},
        "identifier": {"model": "mock-id-b", "batch_size": 32, "upscale": 1, "min_confidence": 0.5},
    }
    boot_out_path = tmp_path / "labels/bootstrap_val.json"
    boot_coco = bootstrap_labels.bootstrap(
        boot_cfg,
        [{"image_id": boot_img_rec["file_name"], "group_id": boot_img_rec["group_id"]}],
        boot_out_path,
    )
    assert boot_out_path.is_file()
    assert len(boot_coco["images"]) == 1
    assert len(boot_coco["annotations"]) == len(boot_anns)
    for ann in boot_coco["annotations"]:
        assert ann["attributes"]["source"] == "vlm:bootstrap"
        assert ann["attributes"]["support"] == pytest.approx(1.0)
        assert ann["attributes"]["judge_verdict"] == "correct"
        assert ann["attributes"]["id_confidence"] == pytest.approx(0.92)

    # 9. freestyle.review.server loading the bootstrap output.
    review_gt_out = tmp_path / "review_gt/val_reviewed.json"
    store = review_server.build_store(
        images_spec="val",
        proposals=boot_out_path,
        out_path=review_gt_out,
        images_root=images_root,
        catalog=catalog_path,
        splits=splits_path,
    )
    port = _free_port()
    httpd = review_server.ReviewServer("127.0.0.1", port, store)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, cls_payload, _ = _http_json(port, "GET", "/api/classes")
        assert status == 200
        assert len(cls_payload["classes"]) == 20  # 18 active + unknown + cartridge

        status, imgs_payload, _ = _http_json(port, "GET", "/api/images")
        assert status == 200
        by_id = {im["image_id"]: im for im in imgs_payload["images"]}
        assert by_id[boot_img_rec["file_name"]]["n_boxes"] == len(boot_anns)

        boot_idx = by_id[boot_img_rec["file_name"]]["idx"]
        status, ann_payload, _ = _http_json(port, "GET", f"/api/ann/{boot_idx}")
        assert status == 200
        assert len(ann_payload["annotations"]) == len(boot_anns)
        assert ann_payload["annotations"][0]["attributes"]["source"] == "vlm:bootstrap"

        # Accept proposals unchanged -> human-verified:vlm:bootstrap.
        status, put_payload, _ = _http_json(
            port,
            "PUT",
            f"/api/ann/{boot_idx}",
            {
                "annotations": ann_payload["annotations"],
                "verified": True,
                "scene": "cabinet_full",
            },
        )
        assert status == 200
        assert put_payload["verified"] is True
        assert put_payload["annotations"][0]["attributes"]["source"] == "human-verified:vlm:bootstrap"
        saved_gt = json.loads(review_gt_out.read_text())
        assert saved_gt["info"]["human_verified_images"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.xfail(
    strict=True,
    reason="freestyle/review/server.py:531 calls _as_floats(quad, 4, 'quad') on 4x2 nested points, rejecting valid quad payloads with HTTP 400",
)
def test_review_server_quad_roundtrip(tmp_path: Path) -> None:
    """Loading contract COCO with ``quad`` into review.server and PUTting back should succeed."""
    images_root = tmp_path / "images"
    (images_root / "grp").mkdir(parents=True)
    Image.new("RGB", (64, 48), (30, 30, 30)).save(images_root / "grp/a.jpg", format="JPEG")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(synth_shelf.build_catalog_items(["hi-c"])))
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"groups": {"grp": "val"}, "images": {"grp/a.jpg": "val"}}))
    proposals_path = tmp_path / "proposals.json"
    proposals_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "grp/a.jpg", "width": 64, "height": 48}],
                "categories": [{"id": 1, "name": "hi-c"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "quad": [[10, 10], [30, 10], [30, 30], [10, 30]],
                        "attributes": {"source": "human"},
                    }
                ],
            }
        )
    )
    store = review_server.build_store(
        images_spec="val",
        proposals=proposals_path,
        out_path=tmp_path / "reviewed.json",
        images_root=images_root,
        catalog=catalog_path,
        splits=splits_path,
    )
    port = _free_port()
    httpd = review_server.ReviewServer("127.0.0.1", port, store)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, ann_payload, _ = _http_json(port, "GET", "/api/ann/0")
        assert status == 200
        assert ann_payload["annotations"][0]["quad"] is not None

        status, put_payload, _ = _http_json(
            port,
            "PUT",
            "/api/ann/0",
            {"annotations": ann_payload["annotations"], "verified": True},
        )
        assert status == 200, put_payload
    finally:
        httpd.shutdown()
        httpd.server_close()
