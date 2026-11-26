import base64
import io
import json

import pytest
from PIL import Image

from tools.bbox_utils import nms
from tools.vlm_client import VLMResponse, _pil_to_data_url
from tools.vlm_detect import vlm_detect
from workflows.vlm_judge import score_detections
from workflows.vlm_label import label_dataset


def test_encoding_options_and_no_upscale():
    image = Image.new("RGB", (40, 20))
    url = _pil_to_data_url(image, None, "png")
    assert url.startswith("data:image/png")
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))).size == (40, 20)
    url = _pil_to_data_url(image, 100, "jpeg")
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))).size == (40, 20)


def test_adjacent_boxes_survive_nms():
    boxes = [{"bbox": [i * 40, 0, i * 40 + 40, 200], "label": "box"} for i in range(10)]
    assert len(nms(boxes + boxes[:1])) == 10


def test_tiles_retry_truncation_and_consensus(monkeypatch):
    calls = []
    def worker(image, *args, **kwargs):
        calls.append(image.size)
        if len(calls) == 1:
            return VLMResponse("[]", "length")
        return VLMResponse('[{"label":"box","bbox_2d":[0,0,500,500]}]', "stop")
    monkeypatch.setattr("tools.vlm_detect.run_vlm", worker)
    result = vlm_detect(Image.new("RGB", (100, 100)), tile=100, n_samples=2, temperature=0.7)
    assert len(calls) == 6
    assert all("score" in d for d in result)
    assert not result.truncated


def test_truncated_empty_and_label_paths(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (20, 20)).save(tmp_path / "sub" / "a.png")
    monkeypatch.setattr("tools.vlm_detect.run_vlm", lambda *a, **k: VLMResponse(
        '[{"label":"Coke Zero!","bbox_2d":[0,0,500,500]}]', "length"))
    with pytest.warns(RuntimeWarning):
        coco = label_dataset(tmp_path, ["coke-zero"], tmp_path / "out.json",
                             recursive=True, n_samples=2, temperature=0.7)
    assert coco["images"][0]["file_name"] == "sub/a.png"
    assert coco["images"][0]["truncated"]
    assert coco["annotations"][0]["score"] == 1


def test_unmatched_alias_and_explicit_paths(tmp_path, monkeypatch):
    Image.new("RGB", (20, 20)).save(tmp_path / "a.png")
    monkeypatch.setattr("tools.vlm_detect.run_vlm", lambda *a, **k: json.dumps([
        {"label": "cola", "bbox_2d": [0, 0, 100, 100]},
        {"label": "mystery", "bbox_2d": [300, 0, 500, 100]},
    ]))
    coco = label_dataset(tmp_path, ["coke"], tmp_path / "out.json", image_paths=["a.png"],
                         label_aliases={"cola": "coke"}, keep_unmatched=True)
    assert [c["name"] for c in coco["categories"]] == ["coke", "unmatched:mystery"]


def test_judge_chunks_crops_and_unjudged(monkeypatch):
    seen = []
    def worker(image, prompt, *args, **kwargs):
        seen.append((image.size, kwargs))
        return '[{"id":0,"verdict":"correct","score":0.9}]'
    monkeypatch.setattr("workflows.vlm_judge.run_vlm", worker)
    dets = [{"label": "box", "bbox": [i * 10, 0, i * 10 + 8, 20]} for i in range(5)]
    verdicts = score_detections(Image.new("RGB", (100, 50)), dets, "model",
        backend="openai", base_url=None, api_key=None, judge_batch_size=2,
        judge_mode="crops", max_tokens=3000, image_format="png")
    assert len(seen) == 3 and seen[0][1]["max_tokens"] == 3000
    assert [v["detection_idx"] for v in verdicts] == list(range(5))
    assert verdicts[1]["verdict"] == "unjudged"


def test_sampling_options_and_raw_log(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools import vlm_client

    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"), finish_reason="stop")],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 42}),
        )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(vlm_client, "_get_client", lambda *args: client)
    Image.new("RGB", (10, 10)).save(tmp_path / "a.png")
    log = tmp_path / "responses.jsonl"
    label_dataset(tmp_path, ["box"], tmp_path / "out.json", backend="openai",
                  n_samples=2, temperature=0.8, top_p=0.9, seed=17,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                  raw_response_log=log)
    assert [c["seed"] for c in calls] == [17, 18]
    assert all(c["temperature"] == 0.8 and c["top_p"] == 0.9 for c in calls)
    assert all(c["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False for c in calls)
    assert [json.loads(line) for line in log.read_text().splitlines()] == [
        {"response": "[]", "finish_reason": "stop", "usage": {"total_tokens": 42}},
    ] * 2


def test_consensus_requires_sampling(tmp_path):
    with pytest.raises(ValueError, match="nonzero temperature"):
        vlm_detect(Image.new("RGB", (10, 10)), n_samples=2)
    with pytest.raises(ValueError, match="nonzero temperature"):
        label_dataset(tmp_path, ["box"], tmp_path / "out.json", n_samples=2)
