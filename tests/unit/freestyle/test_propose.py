"""CPU-only unit tests for freestyle.propose using fake backends."""

from __future__ import annotations

import json
import pytest
from PIL import Image

from freestyle.propose.backends import Sam3Backend, build_backend
from freestyle.propose.run import main as propose_main
from freestyle.propose.run import propose_image, resolve_images, run_proposals


class RecordingFakeBackend:
    """Deterministic fake backend recording tile sizes, prompts, and exemplars."""

    name = "fake"
    model_id = "fake/proposal-model"
    revision = "rev-test-123"
    resolved_revision = "rev-test-123"

    def __init__(self, detections_fn=None):
        self.detections_fn = detections_fn
        self.calls: list[dict] = []
        self.processed_dims_seen: set[tuple[int, int]] = set()

    def detect(
        self,
        pil_tile: Image.Image,
        text_prompts: list[str],
        exemplars: list[Image.Image] | None = None,
    ):
        self.calls.append(
            {
                "tile_size": pil_tile.size,
                "prompts": list(text_prompts),
                "exemplar_sizes": [ex.size for ex in exemplars] if exemplars else [],
                "n_exemplars": len(exemplars) if exemplars else 0,
            }
        )
        self.processed_dims_seen.add((pil_tile.height, pil_tile.width))
        if self.detections_fn is not None:
            return self.detections_fn(len(self.calls) - 1, pil_tile, text_prompts, exemplars)
        return []


def test_tiling_remap_edge_drop_and_merge():
    # Image 16x10 with tile=10, overlap=0.4 -> step=6 -> tiles [0..10] and [6..16] in x.
    img = Image.new("RGB", (16, 10))

    def fake_dets(call_idx, pil_tile, prompts, exemplars):
        if call_idx == 0:
            # Box A: interior to tile 0 at [2, 2, 5, 8] -> image [2, 2, 5, 8]
            # Box B: hits internal right edge (x2=10) of tile 0 -> should be dropped!
            return [
                ([2.0, 2.0, 5.0, 8.0], 0.90, prompts[0]),
                ([7.0, 2.0, 10.0, 8.0], 0.99, prompts[0]),
            ]
        # Tile 1 (x0=6):
        # Duplicate of Box A in tile-local coords would be out-of-range, so put Box C at
        # tile-local [1, 2, 4, 8] -> image [7, 2, 10, 8] (interior to tile 1? Wait: x0=1<=4!).
        # Put Box C away from left internal edge (x0=6 of image, tile-local x0=5 -> image [11, 2, 14, 8])
        # and also emit a duplicate of [11, 2, 14, 8] with lower score to test NMS merging.
        return [
            ([5.0, 2.0, 8.0, 8.0], 0.85, prompts[0]),
            ([5.0, 2.0, 8.0, 8.0], 0.60, "alt"),
        ]

    backend = RecordingFakeBackend(fake_dets)
    merged = propose_image(
        backend,
        img,
        ["cartridge box"],
        tile=10,
        overlap=0.4,
        score_thr=0.05,
        merge="nms",
        merge_iou=0.5,
        edge_margin=2,
        cartridge_category_id=7,
    )
    assert len(backend.calls) == 2
    boxes = [d["box"] for d in merged]
    assert boxes == [[2.0, 2.0, 5.0, 8.0], [11.0, 2.0, 14.0, 8.0]]
    assert all(d["category_id"] == 7 for d in merged)
    assert [d["score"] for d in merged] == [0.9, 0.85]


def test_full_frame_mode_tile_zero():
    img = Image.new("RGB", (24, 18))

    def fake_dets(call_idx, pil_tile, prompts, exemplars):
        # Even boxes touching image borders must be kept in full-frame mode.
        return [([0.0, 1.0, 4.0, 10.0], 0.7, prompts[0])]

    backend = RecordingFakeBackend(fake_dets)
    merged = propose_image(backend, img, ["cartridge"], tile=0, score_thr=0.05)
    assert len(backend.calls) == 1
    assert backend.calls[0]["tile_size"] == (24, 18)
    assert len(merged) == 1
    assert merged[0]["box"] == [0.0, 1.0, 4.0, 10.0]


def test_exemplar_plumbing_and_output_schema_list_file(tmp_path):
    # Create target image and exemplar image.
    img_dir = tmp_path / "images"
    (img_dir / "burst").mkdir(parents=True)
    Image.new("RGB", (20, 12)).save(img_dir / "burst" / "target.png")
    Image.new("RGB", (30, 20)).save(img_dir / "burst" / "ref.png")

    list_file = tmp_path / "frames.txt"
    list_file.write_text("# comment\nburst/target.png\n", encoding="utf-8")

    exemplars_coco = {
        "images": [{"id": 1, "file_name": "burst/ref.png", "width": 30, "height": 20}],
        "categories": [{"id": 1, "name": "cartridge"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 6, 10], "attributes": {"truncated": False}},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 2, 5, 12], "attributes": {"truncated": True}},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [18, 4, 4, 8]},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [24, 1, 3, 5]},
        ],
    }
    (tmp_path / "exemplars.json").write_text(json.dumps(exemplars_coco), encoding="utf-8")

    backend = RecordingFakeBackend(
        lambda idx, tile, prompts, exemplars: [([1.0, 2.0, 5.0, 9.0], 0.42, "exemplar:0")]
    )
    out_dir = tmp_path / "runs" / "exp1"
    preds, meta = run_proposals(
        backend,
        list_file,
        img_dir,
        out_dir,
        prompts=["a narrow upright cardboard cartridge box"],
        exemplars_coco=tmp_path / "exemplars.json",
        n_exemplars=2,
        tile=0,
        score_thr=0.05,
        merge="wbf",
    )

    # Truncated annotation id=2 skipped; capped at n_exemplars=2 -> annotations 1 and 3.
    assert backend.calls[0]["n_exemplars"] == 2
    assert backend.calls[0]["exemplar_sizes"] == [(6, 10), (4, 8)]

    assert len(preds) == 1
    assert preds[0] == {
        "image_id": "burst/target.png",
        "bbox": [1.0, 2.0, 4.0, 7.0],
        "score": 0.42,
        "category_id": 1,
        "id_score": 0.42,
    }
    loaded_preds = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
    assert loaded_preds == preds

    loaded_meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert loaded_meta["experiment"] == "exp1"
    assert loaded_meta["model_id"] == "fake/proposal-model"
    assert loaded_meta["model_revision"] == "rev-test-123"
    assert loaded_meta["processed_input_dims"] == [[12, 20]]
    assert loaded_meta["image_count"] == 1
    assert loaded_meta["seconds_per_image"] >= 0.0
    assert loaded_meta["gpu_mem_gb"] == 0.0


def test_resolve_images_split_name_and_coco(tmp_path):
    splits = {
        "version": 1,
        "groups": {"g1": "val", "g2": "test", "g3": "exclude"},
        "images": {
            "g1/001.jpg": "val",
            "g2/002.jpg": "test",
            "g3/003.jpg": "exclude",
        },
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits), encoding="utf-8")

    val_entries, cat_id = resolve_images("val", splits_path=splits_path)
    assert cat_id == 1
    assert [e.file_name for e in val_entries] == ["g1/001.jpg"]

    all_entries, _ = resolve_images("all", splits_path=splits_path)
    assert [e.file_name for e in all_entries] == ["g1/001.jpg", "g2/002.jpg"]

    coco_path = tmp_path / "val.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [{"id": 99, "file_name": "g1/001.jpg", "width": 10, "height": 10}],
                "categories": [{"id": 3, "name": "coke"}, {"id": 19, "name": "cartridge"}],
            }
        ),
        encoding="utf-8",
    )
    coco_entries, coco_cat_id = resolve_images(coco_path)
    assert coco_cat_id == 19
    assert coco_entries[0].image_id == 99


def test_cli_main_with_patched_backend(tmp_path, monkeypatch):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (16, 12)).save(img_dir / "a.png")
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [{"id": 42, "file_name": "a.png", "width": 16, "height": 12}],
                "categories": [{"id": 5, "name": "cartridge"}],
            }
        ),
        encoding="utf-8",
    )
    fake = RecordingFakeBackend(
        lambda idx, tile, prompts, ex: [([2.0, 2.0, 8.0, 10.0], 0.88, prompts[0])]
    )
    monkeypatch.setattr("freestyle.propose.run.build_backend", lambda name, **kw: fake)

    out_dir = tmp_path / "runs" / "cli_run"
    rc = propose_main(
        [
            "--backend",
            "owlv2",
            "--images",
            str(coco_path),
            "--images-root",
            str(img_dir),
            "--prompts",
            "cartridge box",
            "--tile",
            "0",
            "--score-thr",
            "0.05",
            "--merge",
            "nms",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    preds = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
    assert preds == [
        {
            "image_id": 42,
            "bbox": [2.0, 2.0, 6.0, 8.0],
            "score": 0.88,
            "category_id": 5,
            "id_score": 0.88,
        }
    ]


def test_sam3_requires_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    backend = build_backend("sam3")
    assert isinstance(backend, Sam3Backend)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        backend._resolve_hf_token()


def test_backend_factory_variants():
    assert build_backend("owlv2").model_id == "google/owlv2-large-patch14-ensemble"
    assert build_backend("owlv2", variant="base").model_id == "google/owlv2-base-patch16-ensemble"
    assert build_backend("grounding_dino").model_id == "IDEA-Research/grounding-dino-base"
    assert build_backend("florence2").model_id == "florence-community/Florence-2-large"
    assert build_backend("florence2", model_id="ms-large").model_id == "microsoft/Florence-2-large"
    assert build_backend("sam3").model_id == "facebook/sam3"
    with pytest.raises(ValueError):
        build_backend("florence2", task="invalid")
    with pytest.raises(ValueError):
        build_backend("unknown_backend")


def test_predictions_compatible_with_eval(tmp_path):
    from freestyle.eval.metrics import compute_localization_metrics

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (20, 20)).save(img_dir / "f.png")
    gt_coco = {
        "images": [{"id": 1, "file_name": "f.png", "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "nns-sweetener"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2.0, 4.0, 6.0, 10.0]}
        ],
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt_coco), encoding="utf-8")

    fake = RecordingFakeBackend(
        lambda idx, tile, prompts, ex: [([2.0, 4.0, 8.0, 14.0], 0.95, prompts[0])]
    )
    preds, _ = run_proposals(fake, gt_path, img_dir, tmp_path / "out", tile=0)
    metrics = compute_localization_metrics(gt_coco, preds, score_thr=0.05)
    assert metrics["ap50"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)

