"""Regression tests for all 11 findings in docs/freestyle/reviews/agy-review-3.md."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import vlm_client
from tools.bbox_utils import nms
from tools.vlm_client import VLMResponse, run_vlm, run_vlm_batch
from tools.vlm_detect import vlm_detect
from workflows.vlm_judge import (
    _crop_sheet,
    _parse_verdicts,
    score_detections,
    score_detections_batch,
)
from workflows.vlm_label import _normalize_detections, label_dataset


def test_finding1_symlinks_local_and_recursive(tmp_path, monkeypatch):
    ext_dir = tmp_path / "external"
    ext_dir.mkdir()
    Image.new("RGB", (20, 20)).save(ext_dir / "raw1.png")
    ext_sub = ext_dir / "nested"
    ext_sub.mkdir()
    Image.new("RGB", (20, 20)).save(ext_sub / "raw2.png")

    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "img1.png").symlink_to(ext_dir / "raw1.png")

    monkeypatch.setattr("tools.vlm_detect.run_vlm", lambda *a, **k: "[]")

    # Default non-recursive mode accepts symlinked image in source dir.
    coco = label_dataset(src_dir, ["box"], tmp_path / "out1.json")
    assert [img["file_name"] for img in coco["images"]] == ["img1.png"]

    # Explicit image_paths accepts symlink inside source dir, rejects traversal.
    coco_explicit = label_dataset(
        src_dir, ["box"], tmp_path / "out2.json", image_paths=["img1.png"],
    )
    assert [img["file_name"] for img in coco_explicit["images"]] == ["img1.png"]
    with pytest.raises(ValueError, match="image_paths must stay within source"):
        label_dataset(src_dir, ["box"], tmp_path / "out_bad.json", image_paths=["../external/raw1.png"])

    # Recursive mode traverses symlinked subdirectories without looping on cycles.
    (src_dir / "sub_link").symlink_to(ext_sub)
    (src_dir / "cycle").symlink_to(src_dir)
    coco_rec = label_dataset(src_dir, ["box"], tmp_path / "out3.json", recursive=True)
    assert [img["file_name"] for img in coco_rec["images"]] == ["img1.png", "sub_link/raw2.png"]


def test_finding2_cross_tile_merge_prefers_full_box(monkeypatch):
    # Direct NMS check: full box beats clipped fragment regardless of input order.
    clipped = {"bbox": [950, 100, 1000, 200], "label": "cartridge", "sub_label": ""}
    full = {"bbox": [950, 100, 1050, 200], "label": "cartridge", "sub_label": ""}
    assert nms([clipped, full]) == [full]

    # Cross-tile vlm_detect drops internal-edge-truncated boxes and keeps full box.
    seen_crops: list[tuple[int, int]] = []

    def worker(crop, *args, **kwargs):
        seen_crops.append(crop.size)
        if len(seen_crops) == 1:
            # Tile 0 is (0, 0, 1000, 600): clipped box [950, 100, 1000, 200]
            return json.dumps([{"label": "cartridge", "bbox_2d": [950, 167, 1000, 333]}])
        # Tile 1 is (600, 0, 1600, 600): full box [950, 100, 1050, 200]
        return json.dumps([{"label": "cartridge", "bbox_2d": [350, 167, 450, 333]}])

    monkeypatch.setattr("tools.vlm_detect.run_vlm", worker)
    dets = vlm_detect(Image.new("RGB", (1600, 600)), tile=1000, tile_overlap=0.2)
    assert len(dets) == 1
    assert dets[0]["bbox"][2] > 1000


def test_finding3_normalize_labels_before_consensus_and_nms(monkeypatch):
    calls = 0

    def worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        label = "cola" if calls % 2 == 1 else "Coke Zero!"
        return json.dumps([{"label": label, "bbox_2d": [100, 100, 200, 200]}])

    monkeypatch.setattr("tools.vlm_detect.run_vlm", worker)
    dets = vlm_detect(
        Image.new("RGB", (1000, 1000)),
        classes=["coke-zero"],
        label_aliases={"cola": "coke-zero"},
        n_samples=2,
        temperature=0.7,
    )
    assert len(dets) == 1
    assert dets[0]["label"] == "coke-zero"
    assert dets[0]["score"] == 1.0


def test_finding4_crop_sheet_draws_box_and_sanitizes_oob_inverted(monkeypatch):
    img = Image.new("RGB", (300, 300), "white")
    # _crop_sheet draws a visible box outline inside the crop and does not crash on OOB/inverted.
    sheet = _crop_sheet(img, [
        {"label": "table", "bbox": [50, 50, 150, 150]},
        {"label": "table", "bbox": [450, 100, 500, 150]},
        {"label": "table", "bbox": [200, 200, 100, 100]},
    ])
    # First cell starts at (10, 20) in sheet; crop should contain non-white outline pixels.
    cell0 = sheet.crop((10, 20, 266, 276))
    assert cell0.tobytes() != Image.new("RGB", cell0.size, "white").tobytes()

    # score_detections in crops mode skips invalid boxes with "invalid_box" and clamps partial OOB.
    seen_prompts: list[str] = []

    def judge_worker(image, prompt, *args, **kwargs):
        seen_prompts.append(prompt)
        return '[{"id": 0, "verdict": "correct", "score": 0.9}, {"id": 1, "verdict": "imprecise", "score": 0.5}]'

    monkeypatch.setattr("workflows.vlm_judge.run_vlm", judge_worker)
    dets = [
        {"label": "table", "bbox": [450, 100, 500, 150]},   # 0: fully OOB -> invalid_box
        {"label": "table", "bbox": [40, 40, 140, 140]},     # 1: valid -> #0 to judge
        {"label": "table", "bbox": [200, 200, 100, 100]},   # 2: inverted -> invalid_box
        {"label": "table", "bbox": [-20, 40, 80, 140]},     # 3: partial OOB -> clamped -> #1 to judge
    ]
    verdicts = score_detections(
        img, dets, "model", backend="openai", base_url=None, api_key=None, judge_mode="crops",
    )
    assert len(seen_prompts) == 1
    assert [v["detection_idx"] for v in verdicts] == [0, 1, 2, 3]
    assert verdicts[0]["verdict"] == "invalid_box"
    assert verdicts[1]["verdict"] == "correct"
    assert verdicts[2]["verdict"] == "invalid_box"
    assert verdicts[3]["verdict"] == "imprecise"

    # Default crop batching caps large dense sheets (e.g. 20 boxes -> 2 sheets).
    seen_prompts.clear()
    many = [{"label": "table", "bbox": [10, 10, 30, 30]} for _ in range(20)]
    score_detections(img, many, "model", backend="openai", base_url=None, api_key=None, judge_mode="crops")
    assert len(seen_prompts) == 2


def test_finding5_transformers_backend_returns_finish_reason(monkeypatch):
    class _FakeTensor:
        def __init__(self, rows):
            self._rows = rows

        @property
        def shape(self):
            return (len(self._rows), len(self._rows[0]) if self._rows else 0)

        def __getitem__(self, item):
            if isinstance(item, tuple) and len(item) == 2:
                r_sel, c_sel = item
                rows = self._rows[r_sel] if isinstance(r_sel, slice) else [self._rows[r_sel]]
                return _FakeTensor([r[c_sel] for r in rows])
            return self._rows[item]

        def __iter__(self):
            return iter(self._rows)

        def tolist(self):
            return self._rows

    class _FakeInputs(dict):
        def to(self, device):
            return self

    class _FakeModel:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=99, pad_token_id=0)

        def generate(self, **kwargs):
            # prompt len = 2; row 0 hits max_new_tokens=4 without EOS; row 1 hits EOS (99) + pad (0)
            n = kwargs["input_ids"].shape[0]
            rows = [
                [10, 11, 20, 21, 22, 23],
                [10, 11, 30, 99, 0, 0],
            ]
            return _FakeTensor(rows[:n])

    class _FakeProcessor:
        tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)

        def apply_chat_template(self, conversations, **kwargs):
            n = len(conversations)
            return _FakeInputs({"input_ids": _FakeTensor([[10, 11]] * n)})

        def batch_decode(self, gen, skip_special_tokens=True):
            return [f"decoded-{i}" for i in range(gen.shape[0])]

    import sys
    fake_torch = SimpleNamespace(
        inference_mode=nullcontext,
        manual_seed=lambda s: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("tools.utils.get_or_load", lambda *a, **k: (_FakeModel(), _FakeProcessor()))

    img = Image.new("RGB", (16, 16))
    batch_res = run_vlm_batch([img, img], ["p1", "p2"], "m", backend="transformers", max_tokens=4)
    assert isinstance(batch_res[0], VLMResponse) and batch_res[0].finish_reason == "length"
    assert isinstance(batch_res[1], VLMResponse) and batch_res[1].finish_reason == "stop"

    # Single run_vlm also carries finish_reason="length" when generated tokens == max_tokens.
    single = run_vlm(img, "p1", "m", backend="transformers", max_tokens=4)
    assert isinstance(single, VLMResponse) and single.finish_reason == "length"


def test_finding6_parse_verdicts_salvages_truncated_and_trailing_brackets():
    trailing = '[{"id": 0, "verdict": "correct", "score": 0.95}] Note: [box 0] is valid.'
    assert _parse_verdicts(trailing)[0]["verdict"] == "correct"

    truncated = (
        '[{"id": 0, "verdict": "correct", "score": 0.9, "reason": "ok [1]"}, '
        '{"id": 1, "verdict": "incorrect", "score": null, "reason": "wrong"}, '
        '{"id": 2, "verdict": "impre'
    )
    salvaged = _parse_verdicts(truncated)
    assert set(salvaged) == {0, 1}
    assert salvaged[0]["score"] == 0.9
    assert salvaged[1]["score"] == 0.0


def test_finding7_normalize_detections_edge_cases(tmp_path, monkeypatch):
    # label: None does not crash.
    assert _normalize_detections([{"label": None, "bbox": [0, 0, 1, 1]}], ["coke"], None, False) == []

    # Class key collisions preserve exact class names instead of silent overwrite.
    collided = _normalize_detections(
        [{"label": "a_b"}, {"label": "a-b"}], ["a_b", "a-b"], None, False,
    )
    assert [d["label"] for d in collided] == ["a_b", "a-b"]

    # Order-independent chained aliases and cycle detection.
    chained = _normalize_detections(
        [{"label": "b"}], ["coke"], {"b": "a", "a": "coke"}, False,
    )
    assert [d["label"] for d in chained] == ["coke"]
    with pytest.raises(ValueError, match="Cyclic alias"):
        _normalize_detections([{"label": "a"}], ["coke"], {"a": "b", "b": "a"}, False)

    # Deduplicated classes and consolidated unmatched categories in label_dataset.
    Image.new("RGB", (20, 20)).save(tmp_path / "a.png")
    monkeypatch.setattr("tools.vlm_detect.run_vlm", lambda *a, **k: json.dumps([
        {"label": "Pepsi", "bbox_2d": [0, 0, 100, 100]},
        {"label": "pepsi", "bbox_2d": [100, 100, 200, 200]},
        {"label": "Pepsi!", "bbox_2d": [200, 200, 300, 300]},
    ]))
    coco = label_dataset(
        tmp_path, ["Cat", "cat"], tmp_path / "out.json", keep_unmatched=True,
    )
    assert [c["name"] for c in coco["categories"]] == ["cat", "unmatched:pepsi"]


def test_finding8_autosplit_tile_none_quadrant_overlap_and_iou_consensus(monkeypatch):
    seen_regions: list[tuple[int, int]] = []

    def split_worker(crop, *args, **kwargs):
        seen_regions.append(crop.size)
        if len(seen_regions) == 1:
            return VLMResponse("[]", "length")
        # Quadrant 0 is (0, 0, 120, 120): box [90, 40, 110, 80] crosses mx=100!
        if len(seen_regions) == 2:
            return VLMResponse(json.dumps([
                {"label": "item", "bbox_2d": [750, 333, 917, 667]},
            ]), "stop")
        return VLMResponse("[]", "stop")

    monkeypatch.setattr("tools.vlm_detect.run_vlm", split_worker)
    dets = vlm_detect(Image.new("RGB", (200, 200)), tile=None, tile_overlap=0.2)
    assert len(seen_regions) == 5  # 1 full + 4 overlapping quadrants
    assert seen_regions[1:] == [(120, 120), (120, 120), (120, 120), (120, 120)]
    assert len(dets) == 1
    assert dets[0]["bbox"][0] < 100 < dets[0]["bbox"][2]

    # Consensus clustering uses IoU instead of asymmetric containment.
    sample_idx = 0

    def consensus_worker(*args, **kwargs):
        nonlocal sample_idx
        sample_idx += 1
        if sample_idx == 1:
            return json.dumps([{"label": "item", "bbox_2d": [100, 100, 120, 120]}])
        return json.dumps([{"label": "item", "bbox_2d": [0, 0, 1000, 1000]}])

    monkeypatch.setattr("tools.vlm_detect.run_vlm", consensus_worker)
    res = vlm_detect(Image.new("RGB", (1000, 1000)), n_samples=2, temperature=0.7)
    assert len(res) == 2
    assert [d["score"] for d in res] == [0.5, 0.5]


def test_finding9_batch_paths_forward_dense_scene_controls(tmp_path, monkeypatch):
    seen_openai: list[dict] = []

    def fake_openai(image, prompt, model_id, base_url, api_key, max_tokens,
                    max_dimension=1280, image_format=None, jpeg_quality=85,
                    temperature=0, top_p=1, seed=None, extra_body=None):
        seen_openai.append({
            "max_tokens": max_tokens, "max_dimension": max_dimension,
            "image_format": image_format, "jpeg_quality": jpeg_quality,
            "temperature": temperature, "top_p": top_p, "seed": seed,
            "extra_body": extra_body,
        })
        return VLMResponse('[{"id": 0, "verdict": "correct", "score": 0.8}]', "length", {"total": 7})

    monkeypatch.setattr(vlm_client, "_run_openai", fake_openai)
    log_path = tmp_path / "batch.jsonl"
    img = Image.new("RGB", (100, 100))
    responses = run_vlm_batch(
        [img], ["p"], "m", backend="openai", max_tokens=2048,
        max_dimension=640, image_format="png", jpeg_quality=90,
        temperature=0.5, top_p=0.8, seed=11, extra_body={"k": "v"},
        raw_response_log=log_path,
    )
    assert len(responses) == 1
    assert seen_openai[0]["max_dimension"] == 640
    assert seen_openai[0]["extra_body"] == {"k": "v"}
    assert json.loads(log_path.read_text().splitlines()[0])["finish_reason"] == "length"

    # score_detections_batch forwards crops/batching/max_tokens/warnings and invalid_box.
    with pytest.warns(RuntimeWarning, match="truncated"):
        scored = score_detections_batch(
            [(img, [
                {"label": "cat", "bbox": [10, 10, 50, 50]},
                {"label": "cat", "bbox": [200, 200, 220, 220]},
            ], None)],
            "m", backend="openai", base_url=None, api_key=None,
            judge_mode="crops", judge_batch_size=1, max_tokens=3333,
            max_dimension=512, image_format="png",
        )
    assert seen_openai[-1]["max_tokens"] == 3333
    assert seen_openai[-1]["max_dimension"] == 512
    assert [v["verdict"] for v in scored[0]] == ["correct", "invalid_box"]


def test_finding10_raw_response_log_thread_safe(tmp_path, monkeypatch):
    payload = "x" * 20000
    monkeypatch.setattr(
        vlm_client, "_run_openai",
        lambda *a, **k: VLMResponse(payload, "stop", {"n": 1}),
    )
    img = Image.new("RGB", (10, 10))
    log = tmp_path / "concurrent.jsonl"

    def call_one(i: int):
        run_vlm(img, f"p{i}", "m", backend="openai", raw_response_log=log)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(call_one, range(32)))

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    parsed = [json.loads(line) for line in lines]
    assert all(len(r["response"]) == 20000 and r["finish_reason"] == "stop" for r in parsed)


def test_finding11_dedupes_image_paths_in_coco(tmp_path, monkeypatch):
    Image.new("RGB", (20, 20)).save(tmp_path / "a.png")
    (tmp_path / "sub").mkdir()
    monkeypatch.setattr(
        "tools.vlm_detect.run_vlm",
        lambda *a, **k: json.dumps([{"label": "box", "bbox_2d": [0, 0, 500, 500]}]),
    )
    coco = label_dataset(
        tmp_path, ["box"], tmp_path / "out.json",
        image_paths=["a.png", "./a.png", "sub/../a.png"],
    )
    assert [img["file_name"] for img in coco["images"]] == ["a.png"]
    assert len(coco["annotations"]) == 1
