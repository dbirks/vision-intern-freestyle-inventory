import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from freestyle import artifacts, cli


def _make_fake_run(tmp_path: Path, *, with_gallery: bool = True) -> Path:
    run_dir = tmp_path / "runs" / "C1_test-12345678"
    (run_dir / "training").mkdir(parents=True)
    ckpt = run_dir / "training" / "checkpoint_best_total.pth"
    ckpt.write_bytes(b"fake-detector-weights")

    gallery = run_dir / "gallery.npz"
    if with_gallery:
        from freestyle.identify.embed import Embedder

        embs = Embedder("stub").embed(
            [
                Image.new("RGB", (24, 24), color=(200, 10, 10)),
                Image.new("RGB", (24, 24), color=(10, 200, 10)),
            ]
        )
        np.savez(
            gallery,
            embeddings=embs,
            labels=np.array(["coca-cola-part1", "sprite"]),
            image_ids=np.array(["1", "2"]),
            groups=np.array(["train-group-1", "train-group-2"]),
            splits=np.array(["train", "train"]),
        )

    resolved = {
        "id": "C1_test",
        "family": "C_det_agnostic_plus_embed",
        "localizer": {"type": "tiled_detector", "tile": 512, "score_thr": 0.25},
        "identifier": {
            "type": "embedding_knn" if with_gallery else "none",
            **({"model": "stub", "k": 1} if with_gallery else {}),
        },
        "count": {"identity_thr": 0.4, "neighbour_vote": "off", "max_rows": 4},
    }
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")

    # Simulate a k8s-collected pipeline_test.yaml using /out/... container paths.
    pipeline_cfg = {
        "localizer": {
            "type": "tiled_detector",
            "checkpoint": "/out/training/checkpoint_best_total.pth",
            "tile": 512,
            "score_thr": 0.25,
        },
        "identifier": {
            "type": "embedding_knn" if with_gallery else "none",
            **({"gallery": "/out/gallery.npz", "model": "stub", "k": 1} if with_gallery else {}),
        },
        "counting": {"identity_thr": 0.4, "neighbour_vote": "off", "max_rows": 4},
        "categories": {"coca-cola-part1": 1, "sprite": 2, "unknown": 3},
    }
    (run_dir / "pipeline_test.yaml").write_text(yaml.safe_dump(pipeline_cfg), encoding="utf-8")
    (run_dir / "git_sha").write_text("deadbeef1234\n", encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps({"experiment": "C1_test", "git_sha": "deadbeef1234", "git_dirty": False}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "C1_test",
                "n_images": 4,
                "n_groups": 2,
                "score_thr": 0.42,
                "score_thr_source": "selected_on:data/gt/val.json",
                "localization": {"ap50": 0.85, "ap50_95": 0.61, "f1": 0.88},
                "identification": {"top1_acc": 0.91, "macro_f1": 0.89},
                "counting": {"total_count_mae": 0.25, "exact_inventory_rate": 0.75},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_bundle_export_load_roundtrip_and_hash_failure(tmp_path, monkeypatch):
    import freestyle.infer.detector as detector_mod

    class FakeTiledDetector:
        def __init__(self, checkpoint, **kwargs):
            self.checkpoint = checkpoint
            self.kwargs = kwargs
            self.backend = type("Backend", (), {"class_names": {0: "cartridge"}})()

        def predict(self, image):
            return [{"box": [2, 2, 22, 22], "score": 0.9}]

    monkeypatch.setattr(detector_mod, "TiledDetector", FakeTiledDetector)

    run_dir = _make_fake_run(tmp_path, with_gallery=True)
    artifacts_dir = tmp_path / "artifacts"
    bundle_dir = artifacts.export_bundle(run_dir, "prod-v1", artifacts_dir=artifacts_dir)

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "prod-v1"
    assert manifest["embedder_id"] == "stub"
    assert manifest["detector_classes"] == ["cartridge"]
    assert manifest["thresholds"] == {"score_thr": 0.42, "identity_thr": 0.4}
    assert manifest["provenance"]["run_id"] == run_dir.name
    assert manifest["provenance"]["git_sha"] == "deadbeef1234"
    assert manifest["provenance"]["metrics_summary"]["localization"]["ap50"] == 0.85
    assert "pipeline.yaml" in manifest["files"]
    assert "detector/checkpoint_best_total.pth" in manifest["files"]
    assert "gallery.npz" in manifest["files"]
    assert "classes.json" in manifest["files"]

    listed = artifacts.list_bundles(artifacts_dir=artifacts_dir)
    assert [b["name"] for b in listed] == ["prod-v1"]

    pipeline = artifacts.load_bundle("prod-v1", device="cpu", artifacts_dir=artifacts_dir)
    assert pipeline.score_thr == 0.42
    assert pipeline.identity_thr == 0.4
    assert Path(pipeline.localizer.checkpoint) == bundle_dir / "detector" / "checkpoint_best_total.pth"

    img = Image.new("RGB", (32, 32), color=(200, 10, 10))
    res = pipeline.run(img, "img1", group="query-group-1")
    assert sum(res["counts"].values()) + res["unknown"] == 1

    # Tamper with a bundled file -> load_bundle must fail SHA-256 verification.
    ckpt_path = bundle_dir / "detector" / "checkpoint_best_total.pth"
    ckpt_path.write_bytes(b"corrupted-detector-weights")
    with pytest.raises(artifacts.BundleIntegrityError, match="SHA-256 mismatch"):
        artifacts.load_bundle("prod-v1", artifacts_dir=artifacts_dir)


def test_cli_count_json_and_overlay(tmp_path, monkeypatch, capsys):
    img_path = tmp_path / "door.jpg"
    Image.new("RGB", (64, 64), color=(20, 20, 20)).save(img_path)

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "coca-cola-part1", "name": "Coca-Cola Part 1", "active": True, "sort_order": 1},
                    {"id": "sprite", "name": "Sprite", "active": True, "sort_order": 2},
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakePipeline:
        def __init__(self):
            self.calls = []

        def run(self, image, image_id):
            self.calls.append(image_id)
            return {
                "detections": [
                    {"box": [4, 4, 24, 24], "score": 0.95, "class": "coca-cola-part1", "shelf": 0},
                    {"box": [30, 4, 50, 24], "score": 0.90, "class": "sprite", "shelf": 0},
                    {"box": [4, 30, 24, 50], "score": 0.80, "class": "unknown", "shelf": 1},
                ],
                "counts": {"coca-cola-part1": 1, "sprite": 1},
                "unknown": 1,
                "flags": [{"type": "gap", "detections": [0, 1]}],
                "timings_ms": {"localize": 12.5, "identify": 3.2, "count": 0.4},
            }

    fake = FakePipeline()
    seen_calls = []

    def fake_load_bundle(name, device="auto"):
        seen_calls.append((name, device))
        return fake

    monkeypatch.setattr(cli, "load_bundle", fake_load_bundle)

    out_dir = tmp_path / "out"
    rc = cli.main(
        [
            "count",
            str(img_path),
            "--bundle",
            "prod-v1",
            "--out",
            str(out_dir),
            "--json",
            "--device",
            "cpu",
            "--catalog",
            str(catalog_path),
        ]
    )
    assert rc == 0
    assert seen_calls == [("prod-v1", "cpu")]
    payload = json.loads(capsys.readouterr().out)
    assert payload["bundle"] == "prod-v1"
    assert payload["device"] == "cpu"
    assert len(payload["images"]) == 1
    entry = payload["images"][0]
    assert entry["counts"] == {"Coca-Cola Part 1": 1, "Sprite": 1}
    assert entry["unknown"] == 1
    assert entry["flags"] == [{"type": "gap", "detections": [0, 1]}]
    assert entry["timings_ms"] == {"localize": 12.5, "identify": 3.2, "count": 0.4}
    assert (out_dir / "door.png").is_file()
    assert Path(entry["overlay"]) == out_dir / "door.png"
    assert (out_dir / "counts.json").is_file()


def test_cli_fuse_and_bundles(tmp_path, monkeypatch, capsys):
    f1 = tmp_path / "f1.jpg"
    f2 = tmp_path / "f2.jpg"
    Image.new("RGB", (32, 32)).save(f1)
    Image.new("RGB", (32, 32)).save(f2)

    class FakePipeline:
        def __init__(self):
            self.idx = 0

        def run(self, image, image_id):
            self.idx += 1
            if self.idx == 1:
                return {
                    "detections": [],
                    "counts": {"sprite": 1},
                    "unknown": 2,
                    "flags": [],
                    "timings_ms": {"localize": 5.0, "identify": 1.0, "count": 0.1},
                }
            return {
                "detections": [],
                "counts": {"sprite": 3, "hi-c": 1},
                "unknown": 1,
                "flags": [],
                "timings_ms": {"localize": 5.0, "identify": 1.0, "count": 0.1},
            }

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "sprite", "name": "Sprite", "active": True, "sort_order": 1},
                    {"id": "hi-c", "name": "Hi-C", "active": True, "sort_order": 2},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "load_bundle", lambda name, device="auto": FakePipeline())
    rc = cli.main(
        [
            "fuse",
            str(f1),
            str(f2),
            "--bundle",
            "prod-v1",
            "--method",
            "max",
            "--json",
            "--device",
            "cpu",
            "--catalog",
            str(catalog_path),
        ]
    )
    assert rc == 0
    fused = json.loads(capsys.readouterr().out)
    assert fused["fused_counts"] == {"Sprite": 3, "Hi-C": 1}
    assert fused["fused_unknown"] == 2
    assert len(fused["frames"]) == 2

    monkeypatch.setattr(
        cli,
        "list_bundles",
        lambda: [{"name": "prod-v1", "experiment": "C1_test", "embedder_id": "stub", "valid": True}],
    )
    assert cli.main(["bundles", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [b["name"] for b in listed["bundles"]] == ["prod-v1"]


def _stub_embeddings(tmp_path: Path, run_dir: Path) -> Path:
    """Write a run ``embeddings.npz`` with train/val/test rows, incl. an unknown row.

    Train rows use solid colours so a matching query crop retrieves them via the stub
    embedder. Val/test/unknown rows exist to prove the gallery builder filters them out.
    """
    from freestyle.identify.embed import Embedder

    red = Image.new("RGB", (24, 24), color=(200, 20, 20))
    green = Image.new("RGB", (24, 24), color=(20, 200, 20))
    blue = Image.new("RGB", (24, 24), color=(20, 20, 200))
    emb = Embedder("stub")
    vectors = emb.embed([red, red, green, blue])
    np.savez(
        run_dir / "embeddings.npz",
        embeddings=np.concatenate([vectors, vectors[:1]], axis=0),
        labels=np.array(["coca-cola-part1", "coca-cola-part1", "sprite", "hi-c", "unknown"]),
        image_ids=np.array(["10", "10", "11", "12", "10"]),
        groups=np.array(["tr-a", "tr-a", "tr-b", "va-c", "tr-a"]),
        splits=np.array(["train", "train", "train", "val", "train"]),
        bboxes=np.zeros((5, 4), dtype=np.float32),
        annotation_ids=np.array(["1", "2", "3", "4", "5"]),
        file_names=np.array(["a.jpg", "a.jpg", "b.jpg", "c.jpg", "a.jpg"]),
        model=np.array(["stub"]),
        pad=np.array([0.08], dtype=np.float32),
    )
    return run_dir / "embeddings.npz"


def _knn_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "S2_test-abcdefgh"
    (run_dir / "training").mkdir(parents=True)
    (run_dir / "training" / "checkpoint_best_total.pth").write_bytes(b"fake-detector-weights")
    _stub_embeddings(tmp_path, run_dir)
    # The run evaluated identifier=none (family-C detector-only run); the gallery is attached later.
    resolved = {
        "id": "S2_test",
        "family": "C_det_agnostic_plus_embed",
        "localizer": {"type": "tiled_detector", "tile": 1280, "overlap": 0.25},
        "identifier": {"type": "none"},
        "embeddings": {"model": "stub"},
        "count": {"identity_thr": 0.4},
    }
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    (run_dir / "git_sha").write_text("abc123\n", encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps({"experiment": "S2_test", "git_sha": "abc123", "git_dirty": False}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "S2_test (vs-model-labels, NOT GT)",
                "n_images": 12,
                "n_groups": 11,
                "score_thr": 0.31,
                "score_thr_source": "selected_on:val",
                "localization": {"ap50": 0.71, "ap50_95": 0.49, "f1": 0.76},
                "identification": {"top1_acc": 0.0, "macro_f1": 0.0},
                "counting": {"total_count_mae": 3.58, "exact_inventory_rate": 0.0},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_export_attaches_knn_gallery_from_embeddings(tmp_path):
    run_dir = _knn_run(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    bundle = artifacts.export_bundle(
        run_dir,
        "rfdetr-m-1280-knn-v0",
        artifacts_dir=artifacts_dir,
        identifier={"type": "embedding_knn", "model": "stub", "k": 1},
        counting_override={"neighbour_vote": "report"},
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedder_id"] == "stub"
    assert manifest["detector_classes"] == ["cartridge"]
    assert manifest["thresholds"] == {"score_thr": 0.31, "identity_thr": 0.4}
    assert manifest["provenance"]["metrics_caveat"].upper().find("NOT") >= 0
    assert manifest["provenance"]["metrics_source"].endswith("NOT GT)")
    assert "gallery.npz" in manifest["files"]

    cfg = yaml.safe_load((bundle / "pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["identifier"] == {"type": "embedding_knn", "model": "stub", "k": 1, "gallery": "gallery.npz"}
    assert cfg["counting"]["neighbour_vote"] == "report"

    with np.load(bundle / "gallery.npz", allow_pickle=False) as g:
        # Train-split, known-class rows only: the 3 train rows minus the 1 dropped unknown.
        assert list(g["labels"]) == ["coca-cola-part1", "coca-cola-part1", "sprite"]
        assert list(g["groups"]) == ["tr-a", "tr-a", "tr-b"]
        # Regression: splits must persist as full 'train' strings, not truncated to 't'.
        assert all(str(s) == "train" for s in g["splits"])


def test_export_knn_gallery_rejects_embedder_mismatch(tmp_path):
    run_dir = _knn_run(tmp_path)
    with pytest.raises(ValueError, match="disagrees"):
        artifacts.export_bundle(
            run_dir,
            "bad-embedder",
            artifacts_dir=tmp_path / "artifacts",
            identifier={"type": "embedding_knn", "model": "facebook/dinov2-base", "k": 5},
        )


def test_knn_bundle_loads_and_identifies(tmp_path, monkeypatch):
    import freestyle.infer.detector as detector_mod

    class FakeTiledDetector:
        def __init__(self, checkpoint, **kwargs):
            self.checkpoint = checkpoint
            self.backend = type("Backend", (), {"class_names": {0: "cartridge"}})()

        def predict(self, image):
            return [{"box": [2, 2, 22, 22], "score": 0.9}]

    monkeypatch.setattr(detector_mod, "TiledDetector", FakeTiledDetector)

    run_dir = _knn_run(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    artifacts.export_bundle(
        run_dir,
        "rfdetr-m-1280-knn-v0",
        artifacts_dir=artifacts_dir,
        identifier={"type": "embedding_knn", "model": "stub", "k": 1},
        counting_override={"neighbour_vote": "report"},
    )

    # load_bundle rebuilds EmbeddingIdentifier from the train-only gallery; the truncated
    # 't' split bug would raise 'Gallery may contain train rows only' here.
    pipeline = artifacts.load_bundle("rfdetr-m-1280-knn-v0", device="cpu", artifacts_dir=artifacts_dir)
    assert pipeline.identity_thr == 0.4
    assert pipeline.neighbour_vote == "report"

    query = Image.new("RGB", (32, 32), color=(200, 20, 20))
    result = pipeline.run(query, "img1", group="query-group-9")
    assert result["counts"] == {"coca-cola-part1": 1}
    assert result["unknown"] == 0
