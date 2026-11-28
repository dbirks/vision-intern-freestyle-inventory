"""Unit tests for the crop-reader comparison (synthetic answers, no private data, no network)."""

from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from freestyle.experiments import reader_compare as rc
from freestyle.quiz.server import CROP_PAD

CATALOG = {
    "nns-sweetener": "NNS [nns-sweetener]",
    "sprite": "Sprite [sprite]",
    "coca-cola-part1": "Coca-Cola Part 1 [coca-cola-part1]",
    "coca-cola-part2": "Coca-Cola Part 2 [coca-cola-part2]",
}


def _question(ann_id: int, qid: str, *, guess="sprite", split="test", shelf="top",
              bbox=(100.0, 100.0, 200.0, 200.0), two_part=False, purpose=None, image_id=None):
    options = ([{"part": 1, "class": "coca-cola-part1", "display": "Coca-Cola Part 1"},
                {"part": 2, "class": "coca-cola-part2", "display": "Coca-Cola Part 2"}]
               if two_part else [])
    return {"qid": qid, "kind": "crop",
            "image_id": image_id or f"evt-{ann_id}/frame.jpg", "split": split,
            "ann_id": ann_id, "number": ann_id, "shelf": shelf, "bbox": list(bbox),
            "attributes": {"identity_visible": True}, "guess": guess, "confidence": 0.7,
            "family": None, "part_options": options, "ask_part": bool(options),
            "purpose": purpose or {"test": "eval", "val": "tune", "train": "train"}[split]}


def _queue(questions):
    return {"version": 1, "classes": [{"name": name} for name in CATALOG],
            "questions": questions}


ANSWERS = {
    "crop:a:1": {"result": "correct"},                        # confirms the on-screen guess
    "crop:a:2": {"result": "class", "class": "sprite"},       # human names the right id
    "crop:a:3": {"result": "part", "part": 2},                # family + part
    "crop:a:4": {"result": "class", "class": "unknown"},      # the human cannot tell either
    "crop:a:5": {"result": "two_plus"},                       # another cannot-tell verdict
    "crop:a:6": {"result": "skip"},                           # no identity verdict
    "crop:a:7": {"result": "not_cartridge"},                  # not a cartridge at all
}


def _eval_set():
    questions = [_question(1, "crop:a:1", guess="sprite"),
                 _question(2, "crop:a:2", guess="unknown"),
                 _question(3, "crop:a:3", guess="unknown", two_part=True),
                 _question(4, "crop:a:4", guess="unknown"),
                 _question(5, "crop:a:5", guess="sprite"),
                 _question(6, "crop:a:6", guess="sprite"),
                 _question(7, "crop:a:7", guess="sprite"),
                 _question(8, "crop:a:8", guess="sprite"),              # unanswered
                 _question(9, "crop:v:9", guess="sprite", split="val")]
    return rc.build_eval_set(_queue(questions), ANSWERS, split="test")


CROP_IDS = [f"evt{ann}frame#{ann}" for ann in range(1, 6)]


# --------------------------------------------------------------------- eval set


def test_eval_set_keeps_definite_labels_and_drops_no_verdict_answers():
    data = _eval_set()
    by_ann = {item["ann_id"]: item for item in data["items"]}
    assert set(by_ann) == {1, 2, 3, 4, 5}, "skip / not_cartridge / unanswered are not eval crops"
    assert by_ann[1]["human_class"] == "sprite" and by_ann[1]["definite"]
    assert by_ann[1]["label_source"] == "correct", "correct copies the confirmed on-screen guess"
    assert by_ann[2]["human_class"] == "sprite"
    assert by_ann[3]["human_class"] == "coca-cola-part2", "a bare part tap resolves to family+part"
    assert by_ann[3]["human_part"] == 2 and by_ann[3]["human_family"] == "coca cola"
    assert not by_ann[4]["definite"] and by_ann[4]["human_class"] == "unknown"
    assert not by_ann[5]["definite"], "two_plus is a human cannot-tell verdict, kept separately"
    counts = data["counts"]
    assert counts["definite"] == 3 and counts["human_cannot_tell"] == 2
    assert counts["excluded"] == 3, "skip + not_cartridge + unanswered"
    assert counts["excluded_unanswered"] == 1 and counts["excluded_no_identity_verdict"] == 2
    assert counts["result_correct"] == 1 and counts["result_class"] == 2
    assert [item["crop_id"] for item in data["items"]] == CROP_IDS


def test_eval_set_only_reads_the_requested_split():
    assert all("evt-9" not in item["image_id"] for item in _eval_set()["items"])
    val = rc.build_eval_set(_queue([_question(9, "crop:v:9", split="val")]),
                           {"crop:v:9": {"result": "class", "class": "sprite"}}, split="val")
    assert [it["human_class"] for it in val["items"]] == ["sprite"]


def test_crop_label_is_short_unique_and_disambiguates():
    assert rc.crop_label("island-cabinet-01/abcd1234-uuid/045.jpg", 114) == "abcd045#114"
    dup = rc.build_eval_set(_queue([
        _question(114, "crop:x:1", image_id="abcd-1/045.jpg"),
        _question(114, "crop:y:2", image_id="abcd-2/045.jpg")]),
        {"crop:x:1": {"result": "correct"}, "crop:y:2": {"result": "correct"}})
    ids = [it["crop_id"] for it in dup["items"]]
    assert len(ids) == 2 and len(set(ids)) == 2, f"collision-free even on equal ids: {ids}"


def test_identity_from_answer_handles_partial_and_bare_part():
    q = _question(3, "crop:a:3", two_part=True)
    out = rc.identity_from_answer(q, {"result": "part", "part": 1})
    assert out["human_class"] == "coca-cola-part1" and out["definite"]
    out = rc.identity_from_answer(q, {"result": "partial", "class": "sprite"})
    assert out["human_class"] == "sprite" and out["definite"]
    out = rc.identity_from_answer(q, {"result": "partial"})
    assert out["human_class"] == "unknown" and not out["definite"], "a human cannot tell"
    for result in ("skip", "not_cartridge", "implied_not_usable", "nonsense", ""):
        assert rc.identity_from_answer(q, {"result": result})["excluded"], result


# --------------------------------------------------------------------- metrics


def test_wilson_ci_known_values():
    assert rc.wilson_ci(0, 10)["hi"] == pytest.approx(0.2776, abs=1e-3)
    assert rc.wilson_ci(10, 10)["lo"] == pytest.approx(0.7224, abs=1e-3)
    row = rc.wilson_ci(20, 41)
    assert row["p"] == pytest.approx(20 / 41, abs=1e-4)
    assert 0.30 < row["lo"] < row["hi"] < 0.70
    assert rc.wilson_ci(0, 0) == {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}


def test_family_part_split():
    assert rc.family_part("coca-cola-part2") == ("coca cola", 2)
    assert rc.family_part("sprite") == ("sprite", None)
    assert rc.family_part("unknown") == rc.family_part(None) == ("unknown", None)


# --------------------------------------------------------------------- scoring


def _scored(preds):
    return rc.score_mode(_eval_set()["items"], preds)


def test_score_mode_reports_accuracy_family_part_abstention():
    preds = {CROP_IDS[0]: {"class": "sprite", "seconds": 1.0},
             CROP_IDS[1]: {"class": "coca-cola-part1", "seconds": 2.0},
             CROP_IDS[2]: {"class": "coca-cola-part2", "seconds": 3.0},
             CROP_IDS[3]: {"class": "unknown"},
             CROP_IDS[4]: {"class": "sprite"}}
    row = _scored(preds)
    assert row["n_definite"] == 3 and row["n_missing"] == 0
    assert (row["top1"]["k"], row["top1"]["n"]) == (2, 3)
    assert (row["family"]["k"], row["family"]["n"]) == (2, 3), "part1 vs part2 is still the right brand"
    assert (row["part"]["k"], row["part"]["n"]) == (1, 1), "only the two-part GT crop is scored"
    assert row["part"]["lo"] == pytest.approx(0.2065, abs=1e-3)
    assert row["part_readable"]["k"] == 1
    assert row["abstention"]["k"] == 0
    assert row["confusions"] == [("sprite -> coca-cola-part1", 1)]
    assert row["latency_s"] == {"mean": 2.0, "median": 2.0, "max": 3.0, "n": 3}
    assert row["indeterminate"] == {"n": 2, "abstained": 1, "guessed": 1, "n_missing": 0,
                                   "crop_ids": CROP_IDS[3:]}
    assert row["by_crop"][CROP_IDS[0]]["pred"] == "sprite"


def test_score_mode_counts_abstention_and_failures():
    preds = {CROP_IDS[0]: {"class": "unknown", "error": "RateLimitError: capacity"},
             CROP_IDS[1]: {"class": "sprite"},
             CROP_IDS[2]: {"class": "unknown"}}
    row = _scored(preds)
    assert row["top1"]["k"] == 1 and row["abstention"]["k"] == 2
    assert row["family"]["k"] == 1, "unknown names no brand, so it is a family miss"
    assert row["part"]["k"] == 0 and row["part_readable"]["k"] == 0
    assert row["failures"] == [{"crop_id": CROP_IDS[0], "error": "RateLimitError: capacity"}]
    assert row["n_missing"] == 0


def test_score_mode_with_no_predictions():
    row = _scored({})
    assert row["n_missing"] == 3 and row["top1"]["n"] == 0
    assert row["latency_s"] == {"mean": None, "median": None, "max": None, "n": 0}
    assert row["indeterminate"] == {"n": 2, "abstained": 0, "guessed": 0, "n_missing": 2,
                                   "crop_ids": CROP_IDS[3:]}


def test_catalog_family_map_matches_what_two_stage_claims():
    rows = [{"id": "nns-sweetener", "name": "NNS"},
            {"id": "coca-cola-part2", "name": "Coca-Cola Part 2"},
            {"id": "flavor-orange", "name": "Orange Flavor"},
            {"id": "unknown", "name": "unknown"}]
    assert rc.catalog_family_map(rows) == {"nns-sweetener": "nns", "coca-cola-part2": "coca cola",
                                           "flavor-orange": "orange flavor"}


def test_family_accuracy_credits_a_brand_claim_without_a_part():
    """Stage 2 may abstain on the numeral while still naming the right brand."""
    families = rc.catalog_family_map([{"id": "coca-cola-part2", "name": "Coca-Cola Part 2"},
                                      {"id": "sprite", "name": "Sprite"}])
    items = [dict(item, human_family=families.get(item["human_class"], item["human_family"]))
             for item in _eval_set()["items"] if item["human_class"] == "coca-cola-part2"]
    assert len(items) == 1
    claimed = rc.score_mode(items, {items[0]["crop_id"]: {"class": "unknown", "family": "coca cola"}},
                           families)
    assert (claimed["top1"]["k"], claimed["family"]["k"]) == (0, 1), "brand read, numeral not"
    assert claimed["part"]["k"] == 0 and claimed["part_readable"]["k"] == 0
    silent = rc.score_mode(items, {items[0]["crop_id"]: {"class": "unknown", "family": "unknown"}},
                          families)
    assert (silent["top1"]["k"], silent["family"]["k"]) == (0, 0), "no brand claim is not a hit"
    wrong = rc.score_mode(items, {items[0]["crop_id"]: {"class": "unknown", "family": "dr pepper"}},
                         families)
    assert wrong["family"]["k"] == 0


def test_quiz_baseline_row_is_scored_like_a_run():
    base = rc.score_all(_eval_set(), {})[rc.BASELINE_KEY]
    assert (base["top1"]["k"], base["top1"]["n"]) == (1, 3), "only the confirmed-guess crop matches"
    assert {pair for pair, _ in base["confusions"]} == {"sprite -> unknown",
                                                       "coca-cola-part2 -> unknown"}
    assert base["indeterminate"]["abstained"] == 1 and base["indeterminate"]["guessed"] == 1


# --------------------------------------------------------------------- crops


def _gradient(size=(400, 300)) -> Image.Image:
    base = Image.linear_gradient("L").resize(size)
    return Image.merge("RGB", (base, base.point(lambda v: (v * 2) % 256), base)).convert("RGB")


def test_crop_image_reuses_the_quiz_crop_and_padding():
    from freestyle.quiz.server import make_crop

    image = _gradient()
    bbox = [100.0, 100.0, 200.0, 200.0]
    assert rc.CROP_PAD == pytest.approx(0.15)
    crop = rc.crop_image(image, bbox)
    # 100 px box + 15 % padding all round = 130 px, upscaled x5.54 to fill a phone
    assert crop.size == (720, 720)
    assert crop.tobytes() == Image.open(
        io.BytesIO(make_crop(image, bbox, pad=CROP_PAD))).convert("RGB").tobytes()
    assert crop.size != image.size, "not the original frame"


def test_build_crop_images_writes_the_quiz_bytes(tmp_path: Path):
    root = tmp_path / "images"
    (root / "evt-1").mkdir(parents=True)
    _gradient().save(root / "evt-1" / "frame.jpg", "JPEG")
    item = {"crop_id": "evt1frame#1", "image_id": "evt-1/frame.jpg", "bbox": [10.0, 10.0, 60.0, 60.0]}
    images = rc.build_crop_images([item], root, tmp_path / "crops")
    assert set(images) == {"evt1frame#1"}
    written = tmp_path / "crops" / "evt1frame_1.jpg"
    assert written.is_file()
    # 50 px box + 15 % padding = 66 px, and the quiz caps the upscale at x6
    assert Image.open(written).size == images["evt1frame#1"].size == (396, 396)


def test_tune_crops_only_uses_unlabelled_val_or_train():
    items = rc.tune_crops(_queue([_question(9, "crop:v:9", split="val"),
                                 _question(10, "crop:t:10", split="train"),
                                 _question(11, "crop:e:11", split="test")]), 3)
    assert [it["image_id"] for it in items] == ["evt-9/frame.jpg", "evt-10/frame.jpg"]
    assert all(it["human_class"] is None for it in items), "no labels are read here"


# --------------------------------------------------------------------- VLM plumbing


class FakeChat:
    """Stand-in for ``collective.chat_image``; the first *fail* calls raise."""

    def __init__(self, text='{"class":"sprite","confidence":0.9}', fail=0, delay=0.0):
        self.text, self.fail, self.delay = text, fail, delay
        self.calls: list[dict] = []
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, model, image, prompt, **kwargs):
        with self._lock:
            self.calls.append({"model": model, "prompt": prompt, "kwargs": kwargs})
            index = len(self.calls)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if index <= self.fail:
                raise RuntimeError("429 backend at capacity")
            if self.delay:
                threading.Event().wait(self.delay)
        finally:
            with self._lock:
                self._active -= 1
        return self.text


def test_call_cache_stores_replies_and_keys_on_prompt_and_image(tmp_path, monkeypatch):
    fake = FakeChat()
    monkeypatch.setattr(rc.collective, "chat_image", fake)
    cache = rc.CallCache(tmp_path / "cache", tmp_path / "calls.jsonl")
    image = Image.new("RGB", (8, 8), "red")
    assert cache.chat("m", image, "ask") == fake.text
    assert cache.chat("m", image, "ask") == fake.text
    assert len(fake.calls) == 1, "the identical second call is a cache hit"
    cache.chat("m", image, "different prompt")
    cache.chat("other-model", image, "ask")
    cache.chat("m", Image.new("RGB", (8, 8), "blue"), "ask")
    assert len(fake.calls) == 4, "prompt, model id or pixels changed => a fresh question"
    first = fake.calls[0]["kwargs"]
    assert first["max_tokens"] >= 2048 and first["temperature"] == 0.0
    assert [row["cache"] for row in cache.call_rows()] == ["miss", "hit", "miss", "miss", "miss"]


def test_call_cache_requires_a_thinking_model_sized_token_budget(tmp_path):
    with pytest.raises(ValueError, match="2048"):
        rc.CallCache(tmp_path, None, max_tokens=512)


def test_call_cache_never_caches_failures(tmp_path, monkeypatch):
    fake = FakeChat(fail=1)
    monkeypatch.setattr(rc.collective, "chat_image", fake)
    cache = rc.CallCache(tmp_path / "cache", tmp_path / "calls.jsonl", outage_s=1e9, fallbacks={})
    image = Image.new("RGB", (4, 4))
    with pytest.raises(RuntimeError):
        cache.chat("m", image, "ask")
    assert cache.chat("m", image, "ask") == fake.text, "a failure is not a cached reply"
    assert len(fake.calls) == 2 and not cache.events


def test_call_cache_caps_inflight_requests(tmp_path, monkeypatch):
    fake = FakeChat(delay=0.05)
    monkeypatch.setattr(rc.collective, "chat_image", fake)
    cache = rc.CallCache(tmp_path / "cache", None, max_inflight=2, outage_s=1e9, fallbacks={})
    image = Image.new("RGB", (4, 4))
    with pytest.raises(ValueError, match="max_inflight"):
        rc.CallCache(tmp_path / "cache", None, max_inflight=0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: cache.chat("m", image, f"prompt {i}"), range(8)))
    assert len(fake.calls) == 8
    assert fake.max_active <= 2, f"{fake.max_active} requests were in flight"


def test_call_cache_switches_model_id_after_a_long_outage(tmp_path, monkeypatch):
    now = {"t": 0.0}
    fake = FakeChat(fail=2)
    monkeypatch.setattr(rc.collective, "chat_image", fake)
    cache = rc.CallCache(tmp_path / "cache", tmp_path / "calls.jsonl", outage_s=1200.0,
                        fallbacks={"primary": "secondary"}, clock=lambda: now["t"])
    image = Image.new("RGB", (4, 4))
    with pytest.raises(RuntimeError):
        cache.chat("primary", image, "a")
    assert cache.effective_model("primary") == "primary", "one failure is not yet a 20 min outage"

    now["t"] = 1300.0
    assert cache.chat("primary", image, "a") == fake.text, "the same crop is retried on the alias"
    assert [call["model"] for call in fake.calls] == ["primary", "primary", "secondary"]
    assert cache.events == [{"event": "model_fallback", "requested": "primary",
                            "using": "secondary", "down_s": 1200.0}]
    assert cache.effective_model("primary") == "secondary", "the switch sticks"
    assert cache.chat("primary", image, "b") == fake.text
    assert fake.calls[-1]["model"] == "secondary"
    assert [row["model"] for row in cache.call_rows()] == ["primary", "primary", "secondary",
                                                           "secondary"]
    assert [row["ok"] for row in cache.call_rows()] == [False, False, True, True]
    assert cache.call_rows()[0]["error"].startswith("RuntimeError")


def test_sheet_reader_accepts_one_based_crop_indexes(monkeypatch):
    """The production prompt draws 0-based labels; thinking models answer 1-based."""
    from freestyle.vlm_identify import VLMCropIdentifier

    def ask(self, image, prompt):
        return ask.text

    monkeypatch.setattr(VLMCropIdentifier, "_ask", ask)
    reader = rc.SheetReader("model-q", CATALOG)
    ask.text = '[{"index":1,"class":"sprite","confidence":0.9},' \
               ' {"index":2,"class":"nns-sweetener","confidence":0.8}]'
    assert [row["class"] for row in reader.contact_sheet([Image.new("RGB", (8, 8))] * 2)] \
        == ["sprite", "nns-sweetener"]
    ask.text = '[{"index":0,"class":"sprite","confidence":0.9},' \
               ' {"index":1,"class":"nns-sweetener","confidence":0.8}]'
    assert [row["class"] for row in reader.contact_sheet([Image.new("RGB", (8, 8))] * 2)] \
        == ["sprite", "nns-sweetener"], "0-based replies keep working"
    ask.text = '[{"bbox_2d":[0,0,1000,1000],"label_text":"unreadable"}]'
    assert [row["class"] for row in reader.contact_sheet([Image.new("RGB", (8, 8))] * 2)] \
        == ["unknown", "unknown"], "a hallucinated reply abstains, it does not shift"


def test_contact_sheet_uses_the_sheet_reader():
    assert isinstance(rc.identifier_for("m", "contact_sheet"), rc.SheetReader)
    assert not isinstance(rc.identifier_for("m", "single"), rc.SheetReader)
    assert rc.identifier_for("m", "single").mode == "existing"


def test_patched_chat_routes_vlm_identify_through_the_cache(tmp_path, monkeypatch):
    fake = FakeChat('{"class":"sprite","confidence":0.9}')
    monkeypatch.setattr(rc.collective, "chat_image", fake)
    cache = rc.CallCache(tmp_path, tmp_path / "calls.jsonl")
    assert rc.identifier_for("model-x", "single").mode == "existing"
    assert rc.identifier_for("model-x", "contact_sheet").mode == "existing"
    assert rc.identifier_for("model-x", "two_stage").mode == "two_stage"
    assert rc.identifier_for("model-x", "two_stage_shelf").mode == "two_stage"

    from freestyle.vlm_identify import VLMCropIdentifier

    ident = VLMCropIdentifier("model-x", CATALOG)
    with rc.patched_chat(cache):
        result = ident.identify(Image.new("RGB", (8, 8)))
    assert result["class"] == "sprite"
    kwargs = fake.calls[0]["kwargs"]
    assert kwargs["max_tokens"] == 8192 and kwargs["retry_budget_s"] == 600.0
    assert rc.collective.chat_image is fake, "the patch is undone"


def test_reader_runner_rejects_a_pool_bigger_than_the_inflight_cap(tmp_path):
    cache = rc.CallCache(tmp_path, None, max_inflight=2)
    with pytest.raises(ValueError, match="in-flight"):
        rc.ReaderRunner(tmp_path, {"flash": "m"}, ["single"], cache=cache, max_workers=4)


def _one_crop_eval(tmp_path: Path):
    root = tmp_path / "images"
    (root / "evt-1").mkdir(parents=True, exist_ok=True)
    _gradient().save(root / "evt-1" / "frame.jpg", "JPEG")
    eval_set = rc.build_eval_set(_queue([_question(1, "crop:a:1", guess="unknown")]),
                                {"crop:a:1": {"result": "class", "class": "sprite"}})
    images = rc.build_crop_images(eval_set["items"], root, tmp_path / "crops")
    return eval_set, images


def test_run_records_predictions_is_resumable_and_keeps_errors(tmp_path, monkeypatch):
    eval_set, images = _one_crop_eval(tmp_path)
    fake = FakeChat('{"class":"sprite","confidence":0.8}')
    monkeypatch.setattr(rc.collective, "chat_image", fake)
    cache = rc.CallCache(tmp_path / "cache", tmp_path / "calls.jsonl")
    runner = rc.ReaderRunner(tmp_path, {"flash": "model-y"}, ["single"], cache=cache)

    runs = runner.run(eval_set["items"], images)
    row = runs["flash__single"]["evt1frame#1"]
    assert row["class"] == "sprite" and row["error"] is None and row["seconds"] >= 0
    assert row["mode"] == "single" and row["model"] == "flash__single"
    assert runner.load_done("flash", "single")["evt1frame#1"]["class"] == "sprite"

    assert runner.run(eval_set["items"], images)["flash__single"] == runs["flash__single"]
    assert len(fake.calls) == 1, "saved crops are not re-read"

    # force=True ignores saved rows, but the reply itself is cached; use a cold cache to
    # exercise a transport failure
    monkeypatch.setattr(rc.collective, "chat_image", FakeChat(fail=99))
    cold = rc.ReaderRunner(tmp_path, {"flash": "model-y"}, ["single"],
                          cache=rc.CallCache(tmp_path / "cold", tmp_path / "cold.jsonl",
                                             outage_s=1e9, fallbacks={}))
    forced = cold.run(eval_set["items"], images, force=True)["flash__single"]["evt1frame#1"]
    assert forced["class"] == "unknown" and forced["error"].startswith("RuntimeError")
    assert runner.load_done("flash", "single")["evt1frame#1"]["error"].startswith("RuntimeError"), \
        "the newest row for a crop wins"


def test_contact_sheet_mode_sends_one_image_and_maps_indexes(tmp_path, monkeypatch):
    root = tmp_path / "images"
    (root / "evt-1").mkdir(parents=True)
    _gradient().save(root / "evt-1" / "frame.jpg", "JPEG")
    questions = [_question(1, "crop:s:1", guess="unknown", image_id="evt-1/frame.jpg"),
                 _question(2, "crop:s:2", guess="unknown", image_id="evt-1/frame.jpg")]
    answers = {"crop:s:1": {"result": "class", "class": "sprite"},
               "crop:s:2": {"result": "class", "class": "sprite"}}
    eval_set = rc.build_eval_set(_queue(questions), answers)
    images = rc.build_crop_images(eval_set["items"], root, tmp_path / "crops")

    fake = FakeChat('[{"index":0,"class":"sprite","confidence":0.9}]')
    monkeypatch.setattr(rc.collective, "chat_image", fake)
    cache = rc.CallCache(tmp_path / "cache", tmp_path / "calls.jsonl")
    runner = rc.ReaderRunner(tmp_path, {"flash": "model-z"}, ["contact_sheet"], cache=cache,
                            sheet_size=2)
    rows = runner.run(eval_set["items"], images)["flash__contact_sheet"]
    assert len(fake.calls) == 1, "both crops travelled in one sheet image"
    assert [rows[cid]["class"] for cid in sorted(rows)] == ["sprite", "unknown"]
    assert all(row["sheet_size"] == 2 for row in rows.values())
    scores = rc.score_all(eval_set, {"flash__contact_sheet": rows})["flash__contact_sheet"]
    assert (scores["top1"]["k"], scores["top1"]["n"]) == (1, 2)


def test_smoke_reads_only_unlabelled_crops_and_flags_empty_replies(tmp_path, monkeypatch, capsys):
    root = tmp_path / "images"
    (root / "evt-9").mkdir(parents=True)
    _gradient().save(root / "evt-9" / "frame.jpg", "JPEG")
    items = rc.tune_crops(_queue([_question(9, "crop:v:9", split="val")]), 1)
    assert all(it["human_class"] is None for it in items), "no human labels are read here"
    images = rc.build_crop_images(items, root, tmp_path / "crops_smoke")
    monkeypatch.setattr(rc.collective, "chat_image", FakeChat('{"class":"sprite","confidence":0.5}'))
    cache = rc.CallCache(tmp_path / "cache", tmp_path / "calls.jsonl")
    runner = rc.ReaderRunner(tmp_path, {"flash": "m"}, ["single"], cache=cache)
    assert runner.smoke(items, images) is True
    assert "SMOKE flash__single evt9frame#9 class=sprite" in capsys.readouterr().out

    empty = rc.ReaderRunner(tmp_path / "s2", {"flash": "m"}, ["single"],
                           cache=rc.CallCache(tmp_path / "s2" / "cache", tmp_path / "s2" / "c.jsonl"))
    monkeypatch.setattr(rc.collective, "chat_image", FakeChat("   "))
    assert empty.smoke(items, images) is False, "a thinking model that spent its budget is a failure"
    assert not (tmp_path / "results.json").exists(), "smoke writes no report"


# --------------------------------------------------------------------- report


def test_sheet_index_stats_counts_numbering_styles(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    cases = {"a.json": {"model": "m1", "text": '[{"index":0,"class":"sprite"}]'},
             "b.json": {"model": "m2", "text": '[{"index":1,"class":"sprite"},'
                                                '{"index":2,"class":"sprite"}]'},
             "c.json": {"model": "m2", "text": 'no json and no key here'},
             "d.json": {"model": "m2", "text": '{"class":"sprite","rationale":"the '
                                               '\"index\" label is blurry"}'},
             "e.json": {"model": "m3", "text": '{"class":"sprite"}'}}
    for name, body in cases.items():
        (cache / name).write_text(json.dumps(body))
    assert rc.sheet_index_stats(tmp_path / "missing") == {}, "no cache dir is not an error"
    stats = rc.sheet_index_stats(cache)
    assert stats["m1"] == {"zero_based": 1, "one_based": 0, "no_index": 0, "unparsed": 0}
    assert stats["m2"] == {"zero_based": 0, "one_based": 1, "no_index": 0, "unparsed": 1}
    assert "m3" not in stats, "single-crop replies are not sheets"


def _report_scores():
    data = _eval_set()
    preds = {"flash__single": {CROP_IDS[0]: {"class": "sprite"},
                              CROP_IDS[1]: {"class": "sprite"},
                              CROP_IDS[2]: {"class": "coca-cola-part1", "seconds": 30.0},
                              CROP_IDS[3]: {"class": "unknown"}},
             "inkling__single": {CROP_IDS[0]: {"class": "coca-cola-part1"},
                                CROP_IDS[1]: {"class": "sprite"},
                                CROP_IDS[2]: {"class": "coca-cola-part2"},
                                CROP_IDS[3]: {"class": "sprite"}}}
    return data, rc.score_all(data, preds)


def test_render_report_has_every_table_and_three_conclusion_sentences():
    data, scores = _report_scores()
    text = rc.render_report(scores, data, {"flash": "qwen3-8-flash-next",
                                          "inkling": "david-birks/inkling-2-mini"},
                           ["single"], generated="TEST", notes=["42 calls, 2 failed"])
    for section in ("## Accuracy per model x mode", "## Crops a human could not identify",
                   "## Confusion pairs", "## Per-crop table", "## Run notes", "## Conclusion"):
        assert section in text, section
    assert "`qwen3-8-flash-next` | `single`" in text
    assert "quiz bootstrap (historical" in text
    assert "Wilson" in text and "n=3)" in text
    assert "`sprite -> coca-cola-part1` x1" in text
    assert f"`{CROP_IDS[0]}`" in text and "42 calls, 2 failed" in text
    assert "no prompt was tuned on these crops" in text
    body = [line for line in text.split("## Conclusion", 1)[1].splitlines() if line.strip()]
    assert len(body) == 3, body
    assert "Flash Next" in body[0]


def test_conclusion_says_so_when_a_reader_is_missing():
    data = _eval_set()
    scores = rc.score_all(data, {"flash__single": {CROP_IDS[0]: {"class": "sprite"}}})
    assert rc.conclusion(scores, {"flash": "qwen3-8-flash-next"}) == [
        "Only one reader ran — no comparison yet. Re-run with `--models inkling,flash`."]


def test_conclusion_flags_baseline_reproducibility():
    data = _eval_set()
    as_bootstrap = {CROP_IDS[0]: {"class": "sprite"}, CROP_IDS[1]: {"class": "unknown"},
                    CROP_IDS[2]: {"class": "unknown"}}
    better = {CROP_IDS[0]: {"class": "sprite"}, CROP_IDS[1]: {"class": "sprite"},
              CROP_IDS[2]: {"class": "coca-cola-part2"}}
    scores = rc.score_all(data, {"inkling__two_stage": as_bootstrap,
                                "inkling__single": as_bootstrap,
                                "flash__two_stage_shelf": better})
    line = rc.conclusion(scores, {"inkling": "ink", "flash": "flash"})[0]
    assert "reproduces crop for crop" in line, line
    drift = {CROP_IDS[0]: {"class": "sprite"}, CROP_IDS[1]: {"class": "sprite"},
             CROP_IDS[2]: {"class": "unknown"}}
    scores = rc.score_all(data, {"inkling__two_stage": drift, "inkling__single": drift,
                                "flash__two_stage_shelf": better})
    line = rc.conclusion(scores, {"inkling": "ink", "flash": "flash"})[0]
    assert "re-run scores 0.67" in line, line


def test_summary_table_handles_a_zero_denominator():
    data = _eval_set()
    scores = rc.score_all(data, {"flash__single": {}})
    lines = rc.summary_table(scores, {"flash": "qwen3-8-flash-next"}, ["single"],
                            rc.ranked_keys(scores, {"flash": "x"}, ["single"]))
    text = "\n".join(lines)
    assert "- (n=0)" in text, "an empty metric is shown, not crashed on"


def test_parse_selection_accepts_names_and_literal_ids():
    assert rc.parse_selection("inkling,flash", rc.MODELS, (), "model") == {
        "inkling": "david-birks/inkling-2-mini", "flash": "qwen3-8-flash-next"}
    assert rc.parse_selection("me/other", rc.MODELS, (), "model") == {"me/other": "me/other"}
    assert rc.parse_selection("single,two_stage", {m: m for m in rc.MODES}, rc.MODES, "mode") == {
        "single": "single", "two_stage": "two_stage"}
    with pytest.raises(SystemExit):
        rc.parse_selection("nope", {m: m for m in rc.MODES}, rc.MODES, "mode")
