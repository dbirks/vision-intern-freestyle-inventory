"""Unit tests for the phone-first ground-truth quiz (no private data needed).

Synthetic COCO proposals + generated JPEG frames under a temp images root, so
these tests run anywhere and never read ``data/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from freestyle.identify.embed import Embedder
from freestyle.identify import refresh_gallery as rg
from freestyle.quiz import active as act
from freestyle.quiz import build_queue as bq
from freestyle.quiz import server as quiz_server
from freestyle.quiz import to_coco
from freestyle.quiz.server import QuizError, QuizStore

TOP_CLASS = "sprite"
CLASSES = [
    {"name": "sprite", "display": "Sprite", "color": "#2f9e44", "id": 6, "generic": False},
    {"name": "coca-cola-part1", "display": "Coca-Cola Part 1", "color": "#e3342f", "id": 10,
     "generic": False},
    {"name": "coca-cola-part2", "display": "Coca-Cola Part 2", "color": "#e3342f", "id": 11,
     "generic": False},
    {"name": "unknown", "display": "unknown", "color": "#868e96", "id": 91, "generic": True},
]

IMAGE_A = "gA/001.jpg"  # split test
IMAGE_B = "gB/002.jpg"  # split val


def _write_frame(root: Path, image_id: str, size=(400, 300)) -> None:
    path = root / image_id
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 180, 160)).save(path, "JPEG")


@pytest.fixture
def proposals(tmp_path: Path) -> tuple[dict, dict, Path]:
    root = tmp_path / "images"
    _write_frame(root, IMAGE_A)
    _write_frame(root, IMAGE_B)
    data = {
        "images": [
            {"id": 1, "file_name": IMAGE_A, "width": 400, "height": 300,
             "attributes": {"scene": "cabinet_full"}},
            {"id": 2, "file_name": IMAGE_B, "width": 400, "height": 300,
             "attributes": {"scene": "cabinet_partial"}},
        ],
        "categories": [{"id": i + 1, "name": c["name"]} for i, c in enumerate(CLASSES)],
        "annotations": [
            # image 1: two on the top row, one on the bottom row (right to left ids)
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [200.0, 20.0, 60.0, 90.0],
             "attributes": {"source": "vlm:bootstrap", "id_confidence": 0.62}},
            {"id": 11, "image_id": 1, "category_id": 4, "bbox": [20.0, 25.0, 60.0, 90.0],
             "attributes": {"source": "vlm:bootstrap", "id_family": "coca cola", "id_part": None,
                            "id_confidence": 0.45}},
            {"id": 12, "image_id": 1, "category_id": 4, "bbox": [300.0, 180.0, 70.0, 100.0],
             "attributes": {"source": "vlm:bootstrap", "id_family": "coca cola", "id_confidence": 0.4}},
            # image 2 (val): one box
            {"id": 13, "image_id": 2, "category_id": 1, "bbox": [40.0, 30.0, 60.0, 80.0],
             "attributes": {"source": "vlm:bootstrap", "id_confidence": 0.7}},
        ],
    }
    splits = {
        "version": 1,
        "groups": {},
        "images": {IMAGE_A: "test", IMAGE_B: "val", "gC/003.jpg": "train"},
    }
    return data, splits, root


def _queue(proposals, order=("test", "val")) -> dict:
    data, splits, root = proposals
    return bq.build_queue(data, splits, list(order), CLASSES, root)


# ----------------------------------------------------------------------- queue


def test_queue_orders_test_then_val_count_then_crops(proposals):
    queue = _queue(proposals)
    kinds = [(q["kind"], q["image_id"]) for q in queue["questions"]]
    assert kinds[0] == ("count", IMAGE_A), "the test image comes first"
    assert [img["split"] for img in queue["images"]] == ["test", "val"]
    # count first, then every crop of that image, before the next image starts
    a_questions = [q for q in queue["questions"] if q["image_id"] == IMAGE_A]
    assert [q["kind"] for q in a_questions] == ["count", "crop", "crop", "crop"]
    assert queue["questions"][-1]["image_id"] == IMAGE_B


def test_queue_orders_top_shelf_left_to_right_then_bottom(proposals):
    queue = _queue(proposals)
    crops = [q for q in queue["questions"] if q["image_id"] == IMAGE_A and q["kind"] == "crop"]
    assert [q["shelf"] for q in crops] == ["top", "top", "bottom"]
    assert [q["number"] for q in crops] == [1, 2, 3]
    left_to_right = [(q["bbox"][0] + q["bbox"][2]) / 2 for q in crops]
    assert left_to_right[0] < left_to_right[1] < left_to_right[2]
    count = next(q for q in queue["questions"] if q["kind"] == "count")
    assert (count["model_top"], count["model_bottom"], count["model_total"]) == (2, 1, 3)


def test_queue_marks_two_part_family_guess_as_part_question(proposals):
    queue = _queue(proposals)
    ann11 = next(q for q in queue["questions"] if q.get("ann_id") == 11)
    assert ann11["ask_part"] is True
    assert ann11["family"] == "coca cola"
    assert [o["class"] for o in ann11["part_options"]] == ["coca-cola-part1", "coca-cola-part2"]
    assert "part unknown" in ann11["guess_text"]
    ann10 = next(q for q in queue["questions"] if q.get("ann_id") == 10)
    assert ann10["ask_part"] is False and ann10["guess_text"] == "Sprite"
    assert ann10["confidence"] == pytest.approx(0.62)


def test_queue_split_order_is_respected(proposals):
    queue = _queue(proposals, order=("val", "test"))
    assert [img["split"] for img in queue["images"]] == ["val", "test"]
    assert queue["questions"][0]["image_id"] == IMAGE_B


def test_queue_only_keeps_requested_splits(proposals):
    queue = _queue(proposals, order=("train",))
    assert queue["questions"] == [] and queue["images"] == []


def test_validate_queue_rejects_bad_documents():
    with pytest.raises(ValueError):
        bq.validate_queue({"questions": "nope"})
    with pytest.raises(ValueError, match="version"):
        bq.validate_queue({"version": 99, "questions": []})
    with pytest.raises(ValueError, match="duplicate"):
        bq.validate_queue({"version": 1, "questions": [{"qid": "x", "kind": "count"}, {"qid": "x", "kind": "count"}]})


def test_qid_helpers():
    assert bq.count_qid("a/b.jpg") == "count:a/b.jpg"
    assert bq.crop_qid("a/b.jpg", 7) == "crop:a/b.jpg:7"
    assert bq.qid_image_id("crop:a/b.jpg:7") == "a/b.jpg"
    assert bq.qid_image_id("count:a/b.jpg") == "a/b.jpg"
    assert bq.qid_ann_id("crop:a/b.jpg:7") == 7
    assert bq.qid_ann_id("count:a/b.jpg") is None


def test_family_classes_only_two_part_brands():
    families = bq.family_classes(CLASSES)
    assert families == {"coca cola": {1: "coca-cola-part1", 2: "coca-cola-part2"}}


# --------------------------------------------------------------------- answers


@pytest.fixture
def store(proposals, tmp_path) -> QuizStore:
    data, splits, root = proposals
    queue = bq.build_queue(data, splits, ["test", "val"], CLASSES, root)
    return QuizStore(queue, tmp_path / "gt" / "quiz_answers.jsonl", root)


def test_answer_appends_one_json_line_each(store):
    first = store.questions[0]["qid"]
    second = store.questions[1]["qid"]
    store.answer(first, {"result": "correct"})
    store.answer(second, {"result": "class", "class": TOP_CLASS})
    lines = store.answers_path.read_text().strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["qid"] == first
    assert record["image_id"] == IMAGE_A
    assert record["answer"] == {"result": "correct", "at": record["answer"]["at"]}
    assert record["guess"] is None or isinstance(record["guess"], str)
    assert store.state_payload()["answered"] == 2


def test_answer_records_the_model_guess(store):
    crop = next(q for q in store.questions if q["kind"] == "crop")
    record = store.answer(crop["qid"], {"result": "correct"})
    assert record["guess"] == crop["guess"]
    assert record["ann_id"] == crop["ann_id"]
    assert record["guess_confidence"] == crop["confidence"]


def test_fix_answer_stores_counts_and_total(store):
    count_q = store.questions[0]["qid"]
    record = store.answer(count_q, {"result": "fix", "top": 3, "bottom": 2})
    assert record["answer"]["total"] == 5
    assert (record["answer"]["top"], record["answer"]["bottom"]) == (3, 2)


def test_undo_drops_the_last_line(store):
    count_q = store.questions[0]["qid"]
    crop_q = store.questions[1]["qid"]
    store.answer(count_q, {"result": "correct"})
    store.answer(crop_q, {"result": "skip"})
    removed = store.undo()
    assert removed["qid"] == crop_q
    assert store.answers_path.read_text().strip().splitlines().__len__() == 1
    assert store.state_payload()["answered"] == 1
    assert store.undo()["qid"] == count_q
    assert store.undo() is None, "undo on an empty log is a no-op"


def test_resume_continues_after_existing_answers(store):
    store.answer(store.questions[0]["qid"], {"result": "correct"})
    store.answer(store.questions[1]["qid"], {"result": "correct"})
    reloaded = QuizStore(store.queue, store.answers_path, store.images_root)
    assert reloaded.next_index() == 2
    assert reloaded.state_payload()["answered"] == 2
    assert reloaded.state_payload()["undo_available"] is True


def test_re_answer_wins_and_undoes_backwards(store):
    qid = store.questions[0]["qid"]
    store.answer(qid, {"result": "correct"})
    store.answer(qid, {"result": "fix", "top": 1, "bottom": 1})
    assert store.state_payload()["answers"][qid]["result"] == "fix"
    store.undo()
    assert store.state_payload()["answers"][qid]["result"] == "correct"


def test_answer_validation(store):
    crop = next(q for q in store.questions if q["kind"] == "crop")
    with pytest.raises(QuizError):
        store.answer("nope", {"result": "correct"})
    with pytest.raises(QuizError):
        store.answer(crop["qid"], {"result": "not_cabinet"})  # count-only answer
    with pytest.raises(QuizError):
        store.answer(crop["qid"], {"result": "class", "class": "not-a-class"})
    with pytest.raises(QuizError):
        store.answer(crop["qid"], {"result": "class"})
    with pytest.raises(QuizError):
        store.answer(crop["qid"], {"result": "fix", "top": 1})  # crop question
    with pytest.raises(QuizError):
        store.answer(store.questions[0]["qid"], {"result": "fix", "top": -1, "bottom": 0})
    with pytest.raises(QuizError):
        store.answer(store.questions[0]["qid"], "correct")  # answer must be an object
    part_q = next(q for q in store.questions if q.get("ask_part"))
    store.answer(part_q["qid"], {"result": "part", "part": 2})
    assert store.state_payload()["answers"][part_q["qid"]]["class"] == "coca-cola-part2"
    with pytest.raises(QuizError):
        store.answer(part_q["qid"], {"result": "part", "part": 3})


def test_path_traversal_is_refused(store, tmp_path):
    from freestyle.review.server import safe_join

    assert safe_join(store.images_root, "../../etc/passwd") is None
    assert safe_join(store.images_root, "/etc/passwd") is None
    with pytest.raises(QuizError):
        store.image("../../etc/passwd")
    with pytest.raises(QuizError):
        store.image("gA/../../etc/passwd")
    # a queue that names an image outside the images root is rejected up front
    bad = json.loads(json.dumps(store.queue))
    bad["questions"][0]["image_id"] = "../secrets.txt"
    for question in bad["questions"]:
        if question["image_id"] == store.questions[0]["image_id"]:
            question["image_id"] = "../secrets.txt"
    with pytest.raises(QuizError):
        QuizStore(bad, tmp_path / "a.jsonl", store.images_root)


def test_frame_index_out_of_range(store):
    with pytest.raises(QuizError):
        store.frame_png(99)
    with pytest.raises(QuizError):
        store.crop_jpeg(0)  # a count question has no crop


def test_images_render(store):
    frame = store.frame_png(0)
    crop_q_index = next(i for i, q in enumerate(store.questions) if q["kind"] == "crop")
    crop = store.crop_jpeg(crop_q_index)
    context = store.context_jpeg(crop_q_index)
    for blob in (frame, crop, context):
        assert blob[:2] == b"\xff\xd8", "JPEG"
    assert store.crop_jpeg(crop_q_index) is crop, "crops are cached"


def test_static_assets_exist():
    static = Path(quiz_server.STATIC_DIR)
    for name in ("index.html", "quiz.js", "quiz.css"):
        assert (static / name).is_file(), name


# ---------------------------------------------------------------------- to_coco


def _answers_from(store: QuizStore, mapping: dict[str, dict]) -> dict[str, dict]:
    """Resolve a friendly {label: answer} map to real question ids."""
    out = {}
    for key, answer in mapping.items():
        if key == "count_a":
            qid = bq.count_qid(IMAGE_A)
        elif key == "count_b":
            qid = bq.count_qid(IMAGE_B)
        else:
            qid = bq.crop_qid(IMAGE_A, int(key))
        out[qid] = {"result": answer.get("result", "correct"), **{
            k: v for k, v in answer.items() if k != "result"
        }}
    return out


def counts_of(counts_doc: dict, image_id: str) -> dict:
    return counts_doc["images"][image_id]


def test_to_coco_correct_keeps_the_guess(store):
    crop = next(q for q in store.questions if q.get("ann_id") == 10)
    answers = _answers_from(store, {"count_a": {"result": "correct"}, "10": {"result": "correct"}})
    doc, counts = to_coco.build_coco(store.queue, answers, "test")
    by_proposal = {a["attributes"]["proposal_ann_id"]: a for a in doc["annotations"]}
    assert sorted(by_proposal) == [10, 11, 12], "proposal ann ids survive; unanswered boxes stay"
    ann = by_proposal[crop["ann_id"]]
    assert ann["category_id"] == 6  # sprite
    assert ann["bbox"] == [200.0, 20.0, 60.0, 90.0], "original pixels, xywh"
    assert ann["attributes"]["box_source"] == "model"
    assert ann["attributes"]["identity_source"] == "human"
    assert by_proposal[11]["attributes"]["identity_source"] == "model", "never answered"
    assert counts_of(counts, IMAGE_A)["human_total"] == 3
    assert counts_of(counts, IMAGE_A)["classes"] == {"sprite": 1}
    assert counts_of(counts, IMAGE_A)["counts_mismatch"] is True, "only 1 of 3 crops answered by a human"


def test_to_coco_class_and_part_answers(store):
    answers = _answers_from(store, {
        "count_a": {"result": "fix", "top": 2, "bottom": 1},
        "10": {"result": "class", "class": "coca-cola-part1"},
        "11": {"result": "part", "part": 2},
        "12": {"result": "correct"},
    })
    doc, counts = to_coco.build_coco(store.queue, answers, "test")
    by_proposal = {a["attributes"]["proposal_ann_id"]: a for a in doc["annotations"]}
    assert by_proposal[10]["category_id"] == 10  # coca-cola-part1
    assert by_proposal[11]["category_id"] == 11  # coca-cola-part2 (from the part tap)
    assert by_proposal[12]["category_id"] == 91  # unknown guess kept as the human's "correct"
    assert counts_of(counts, IMAGE_A)["human_total"] == 3
    assert counts_of(counts, IMAGE_A)["counts_mismatch"] is False, "crop total agrees with the human"
    image = doc["images"][0]
    assert image["attributes"]["human_counts"] == {"top": 2, "bottom": 1, "total": 3}
    assert image["attributes"]["verified"] is True


def test_to_coco_drops_not_a_cartridge_and_flags_two_plus_partial(store):
    answers = _answers_from(store, {
        "count_a": {"result": "fix", "top": 1, "bottom": 0},
        "10": {"result": "not_cartridge"},
        "11": {"result": "two_plus"},
        "12": {"result": "partial"},
    })
    doc, counts = to_coco.build_coco(store.queue, answers, "test")
    by_proposal = {a["attributes"]["proposal_ann_id"]: a for a in doc["annotations"]}
    assert 10 not in by_proposal, "not a cartridge -> no box"
    assert by_proposal[11]["category_id"] == 91, "unknown"
    assert by_proposal[11]["attributes"]["box_covers_2_plus"] is True
    assert by_proposal[11]["attributes"]["identity_visible"] is False
    assert by_proposal[12]["attributes"]["box_partial"] is True
    assert by_proposal[12]["attributes"]["truncated"] is True
    assert counts_of(counts, IMAGE_A)["crop_total"] == 1


def test_to_coco_not_cabinet_clears_boxes_and_marks_unusable(store):
    answers = _answers_from(store, {"count_a": {"result": "not_cabinet"}})
    for question in store.questions:
        if question["image_id"] == IMAGE_A and question["kind"] == "crop":
            answers[question["qid"]] = {"result": "skip"}
    doc, counts = to_coco.build_coco(store.queue, answers, "test")
    image = doc["images"][0]
    assert image["attributes"]["usable"] is False
    assert image["attributes"]["scene"] == "no_cabinet"
    assert doc["annotations"] == []
    assert counts_of(counts, IMAGE_A)["human_total"] == 0
    assert image["attributes"]["verified"] is True


def test_to_coco_skip_keeps_model_identity_and_unverifies(store):
    answers = _answers_from(store, {
        "count_a": {"result": "correct"},
        "10": {"result": "skip"},
        "11": {"result": "correct"},
        "12": {"result": "correct"},
    })
    doc, counts = to_coco.build_coco(store.queue, answers, "test")
    by_proposal = {a["attributes"]["proposal_ann_id"]: a for a in doc["annotations"]}
    assert by_proposal[10]["attributes"]["identity_source"] == "model"
    assert by_proposal[10]["attributes"]["source"] == "quiz:model"
    assert doc["images"][0]["attributes"]["verified"] is False
    assert counts_of(counts, IMAGE_A)["classes"] == {"unknown": 2}, "skipped crops are not human counts"


def test_to_coco_export_only_writes_the_requested_split(store):
    answers = _answers_from(store, {"count_b": {"result": "correct"}})
    doc, _ = to_coco.build_coco(store.queue, answers, "val")
    assert [i["file_name"] for i in doc["images"]] == [IMAGE_B]


def test_to_coco_categories_come_from_the_queue(store):
    doc, _ = to_coco.build_coco(store.queue, {}, "test")
    assert [c["name"] for c in doc["categories"]] == [c["name"] for c in store.classes]
    assert [c["id"] for c in doc["categories"]] == [c["id"] for c in store.classes]


def test_to_coco_cli_writes_both_files(store, tmp_path, capsys):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(store.queue))
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text(
        json.dumps({"qid": bq.count_qid(IMAGE_A), "image_id": IMAGE_A, "answer": {"result": "correct"}}) + "\n"
    )
    out = tmp_path / "gt" / "test.json"
    rc = to_coco.main(["--queue", str(queue_path), "--answers", str(answers_path), "--out", str(out),
                       "--split", "test"])
    assert rc == 0
    assert out.is_file() and (tmp_path / "gt" / "test_counts.json").is_file()
    assert json.loads(out.read_text())["info"]["total_images"] == 1


def test_partial_identity_requires_explicit_class():
    from freestyle.quiz.to_coco import _identity

    question = {'guess': 'sprite'}
    assert _identity(question, {'result': 'partial'}) == ('unknown', 'model', 'partial')
    assert _identity(question, {'result': 'partial', 'class': 'sprite'}) == ('sprite', 'human', 'partial')
    with pytest.raises(ValueError):
        _identity(question, {'result': 'unexpected'})


def test_unusable_implies_crops_and_undo_restores_them(store):
    count = store.questions[0]
    crops = [q for q in store.by_image[count["image_id"]] if q["kind"] == "crop"]
    store.answer(count["qid"], {"result": "not_cabinet"})
    answers = store.state_payload()["answers"]
    assert all(answers[q["qid"]]["result"] == "implied_not_usable" for q in crops)
    doc, counts = to_coco.build_coco(store.queue, answers, count["split"])
    image = next(im for im in doc["images"] if im["file_name"] == count["image_id"])
    assert image["attributes"]["verified"]
    assert not [ann for ann in doc["annotations"] if ann["image_id"] == image["id"]]
    assert counts["images"][count["image_id"]]["human_total"] == 0
    assert store.undo()["qid"] == count["qid"]
    assert not store.state_payload()["answers"]


def test_count_skip_defers_image_once_and_undo(store):
    count = store.questions[0]
    store.answer(count["qid"], {"result": "skip"})
    assert store.questions[store.next_index()]["image_id"] != count["image_id"]
    assert all(q["qid"] not in store.state_payload()["answers"] for q in store.by_image[count["image_id"]][1:])
    reloaded = QuizStore(store.queue, store.answers_path, store.images_root)
    assert reloaded.next_index() == store.next_index()
    assert reloaded.undo()["qid"] == count["qid"]
    assert reloaded.next_index() == 0
    store = reloaded
    store.answer(count["qid"], {"result": "skip"})
    for q in store.questions:
        if q["image_id"] != count["image_id"]:
            store.answer(q["qid"], {"result": "correct"})
    assert store.next_index() == 0
    store.answer(count["qid"], {"result": "skip"})
    assert store.next_index() == len(store.questions)


def test_audit_quality_orders_within_split(proposals):
    data, splits, root = proposals
    audit = {"images": [
        {"image_id": IMAGE_A, "scene_label": "cabinet_partial", "motion_blur": "mild", "sharp_full": 500},
        {"image_id": IMAGE_B, "scene_label": "cabinet_full", "motion_blur": "none", "sharp_full": 10},
    ]}
    same_split = {"images": {IMAGE_A: "test", IMAGE_B: "test"}}
    queue = bq.build_queue(data, same_split, ["test"], CLASSES, root, audit)
    assert [im["image_id"] for im in queue["images"]] == [IMAGE_B, IMAGE_A]


# ============================================================ active-learning queue

IMAGE_T = "gT/010.jpg"  # split train
SPRITE, COLA1, COLA2, BLANK = 1, 2, 3, 4  # category ids in the synthetic proposals


def _ann(ann_id, image_id, category_id, x, y, *, conf=0.9, family=None, part=None):
    attrs = {"source": "vlm:bootstrap", "id_confidence": conf}
    if family:
        attrs["id_family"] = family
    if part:
        attrs["id_part"] = part
    return {"id": ann_id, "image_id": image_id, "category_id": category_id,
            "bbox": [float(x), float(y), 40.0, 80.0], "attributes": attrs}


@pytest.fixture
def active_doc(tmp_path):
    """Proposals for one test, one val and one train frame, plus their split map."""
    root = tmp_path / "images"
    for name in (IMAGE_A, IMAGE_B, IMAGE_T):
        _write_frame(root, name)
    anns = [
        # test frame: three boxes per shelf, all read confidently
        *[_ann(i, 1, SPRITE, x, 20.0, conf=0.9) for i, x in enumerate((20.0, 120.0, 220.0), start=1)],
        *[_ann(i, 1, SPRITE, x, 200.0, conf=0.9) for i, x in enumerate((20.0, 120.0, 220.0), start=4)],
        # val frame: one confident match, one shy of the threshold, one disagreement
        _ann(10, 2, SPRITE, 20.0, 20.0, conf=0.9),
        _ann(11, 2, SPRITE, 120.0, 25.0, conf=0.55),
        _ann(12, 2, BLANK, 220.0, 30.0, conf=0.9, family="coca cola"),
        _ann(13, 2, SPRITE, 20.0, 200.0, conf=0.95),
        # train frame: a well-fed class, a two-part family with no part read, a thin class
        _ann(20, 3, SPRITE, 20.0, 20.0, conf=0.8),
        _ann(21, 3, BLANK, 120.0, 25.0, conf=0.5, family="coca cola"),
        _ann(22, 3, BLANK, 220.0, 30.0, conf=0.4, family="coca cola zero"),
        _ann(23, 3, COLA2, 20.0, 200.0, conf=0.5),
        _ann(24, 3, BLANK, 120.0, 205.0, conf=0.95, family="sprite"),
    ]
    data = {
        "images": [
            {"id": 1, "file_name": IMAGE_A, "width": 400, "height": 300,
             "attributes": {"scene": "cabinet_full"}},
            {"id": 2, "file_name": IMAGE_B, "width": 400, "height": 300,
             "attributes": {"scene": "cabinet_full"}},
            {"id": 3, "file_name": IMAGE_T, "width": 400, "height": 300,
             "attributes": {"scene": "cabinet_full"}},
        ],
        "categories": [{"id": i + 1, "name": c["name"]} for i, c in enumerate(CLASSES)],
        "annotations": anns,
    }
    splits = {"version": 1, "groups": {}, "images": {IMAGE_A: "test", IMAGE_B: "val", IMAGE_T: "train"}}
    queue = bq.build_queue(data, splits, ["test", "val", "train"], CLASSES, root)
    return queue, splits, root


def _crops(queue, split):
    return [q for q in queue["questions"] if q["kind"] == bq.CROP and q["split"] == split]


def _preds(queue, split, default=("sprite", 0.9), overrides=None):
    """k-NN predictions for a split's crops; ``overrides`` maps ann_id -> (class, score)."""
    overrides = overrides or {}
    out = {}
    for question in _crops(queue, split):
        cls, score = overrides.get(question["ann_id"], default)
        out[question["qid"]] = {"class": cls, "score": score}
    return out


def test_active_queue_selects_test_sample_then_train_then_val(active_doc):
    queue, _, _ = active_doc
    chosen = act.select_questions(queue, (), _all_preds(active_doc), {}, test_crops=2, val_cap=10,
                                 train_cap=10)
    purposes = [q["purpose"] for q in chosen]
    assert purposes == sorted(purposes, key=act.BLOCK_ORDER.index), "test, then train, then val"
    assert [q["split"] for q in chosen if q["kind"] == bq.COUNT] == ["test", "val"], \
        "every count question stays, the train frame gets no count question"
    assert {q["purpose"] for q in chosen if q["split"] == "test"} == {"eval"}
    assert {q["purpose"] for q in chosen if q["split"] == "val"} == {"tune"}
    assert {q["purpose"] for q in chosen if q["split"] == "train"} == {"train"}
    eval_crops = [q for q in chosen if q["purpose"] == "eval" and q["kind"] == bq.CROP]
    assert len(eval_crops) == 2, "exactly the sampled number of evaluation crops"
    assert {q["shelf"] for q in eval_crops} == {"top", "bottom"}


def _all_preds(doc):
    queue, _, _ = doc
    preds = {}
    preds.update(_preds(queue, "test"))
    preds.update(_preds(queue, "val", overrides={12: ("sprite", 0.8)}))
    preds.update(_preds(queue, "train", overrides={21: ("sprite", 0.8), 22: ("coca-cola-part1", 0.7),
                                                   23: ("coca-cola-part2", 0.7)}))
    return preds


def test_active_eval_sample_is_fixed_and_stratified(active_doc):
    queue, _, _ = active_doc
    crops = _crops(queue, "test")
    first = act.select_crop_sample(crops, 2, seed=0)
    assert act.select_crop_sample(crops, 2, seed=0) == first, "the same seed asks the same questions"
    assert len(first) == 2
    shelves = {next(q["shelf"] for q in crops if q["qid"] == qid) for qid in first}
    assert shelves == {"top", "bottom"}, "3 top / 3 bottom of 2 -> one per shelf"
    assert act.select_crop_sample(crops, 99, seed=0) == {q["qid"] for q in crops}


def test_active_val_block_only_takes_uncertain_crops(active_doc):
    queue, _, _ = active_doc
    preds = _all_preds(active_doc)
    chosen = act.select_questions(queue, (), preds, {}, test_crops=0, val_cap=10, train_cap=0)
    val = {q["ann_id"] for q in chosen if q["split"] == "val" and q["kind"] == bq.CROP}
    assert val == {11, 12}, "0.55 confidence and the family disagreement, not the confident ones"
    assert all(q["selected_by"] == "uncertain" for q in chosen
               if q["split"] == "val" and q["kind"] == bq.CROP)
    capped = act.select_questions(queue, (), preds, {}, test_crops=0, val_cap=1, train_cap=0)
    assert sum(1 for q in capped if q["split"] == "val" and q["kind"] == bq.CROP) == 1


def test_active_train_block_prioritises_two_part_and_thin_classes(active_doc):
    queue, _, _ = active_doc
    preds = _all_preds(active_doc)
    counts = {"sprite": 40, "nns-sweetener": 60, "coca-cola-part1": 0, "coca-cola-part2": 1}
    chosen = act.select_questions(queue, (), preds, counts, test_crops=0, val_cap=0, train_cap=2)
    train = [q for q in chosen if q["split"] == "train"]
    assert [q["ann_id"] for q in train] == [21, 22], "unread parts first, then the thinnest classes"
    assert train[0]["ask_part"] is True
    wider = act.select_questions(queue, (), preds, counts, test_crops=0, val_cap=0, train_cap=99)
    order = {q["ann_id"] for q in wider if q["split"] == "train"}
    assert order == {21, 22, 23}
    assert 20 not in order and 24 not in order, "certain crops are not worth asking"
    by_priority = sorted((q for q in wider if q["split"] == "train"),
                         key=lambda q: act.train_priority(q, bq.family_classes(CLASSES), counts,
                                                          prediction=preds[q["qid"]]))
    assert [q["ann_id"] for q in by_priority] == [21, 22, 23], "unread part, then thin, then fed"


def test_active_train_block_never_asks_test_or_val_boxes(active_doc):
    queue, _, _ = active_doc
    chosen = act.select_questions(queue, (), _all_preds(active_doc), {}, test_crops=0, val_cap=0,
                                 train_cap=99)
    assert {q["image_id"] for q in chosen if q["purpose"] == "train"} == {IMAGE_T}


def test_active_keeps_every_answered_question(active_doc):
    queue, _, _ = active_doc
    crops = _crops(queue, "val")
    answered = {crops[0]["qid"]: {"result": "correct"}, _crops(queue, "train")[0]["qid"]: {"result": "skip"}}
    chosen = act.select_questions(queue, answered, _all_preds(active_doc), {}, test_crops=0,
                                 val_cap=0, train_cap=0)
    qids = {q["qid"] for q in chosen}
    assert set(answered) <= qids, "an answer is never thrown away"
    assert all(q["selected_by"] == "answered" for q in chosen if q["qid"] in answered)


def test_active_build_writes_purposes_counts_and_images(active_doc):
    queue, _, _ = active_doc
    answered = {_crops(queue, "test")[0]["qid"]: {"result": "correct"}}
    rebuilt = act.build_active_queue(queue, answered, _all_preds(active_doc), {}, test_crops=2,
                                    val_cap=2, train_cap=2)
    bq.validate_queue(rebuilt)
    assert rebuilt["selection"]["mode"] == "active"
    assert rebuilt["selection"]["before"]["total"] == len(queue["questions"])
    assert rebuilt["selection"]["after"]["total"] == len(rebuilt["questions"])
    assert rebuilt["splits_order"] == ["test", "train", "val"]
    assert [im["image_id"] for im in rebuilt["images"]] == [IMAGE_A, IMAGE_T, IMAGE_B]
    assert all(im["qids"] for im in rebuilt["images"])
    firsts = [im["first_index"] for im in rebuilt["images"]]
    assert firsts == sorted(firsts)
    index = {q["qid"]: i for i, q in enumerate(rebuilt["questions"])}
    assert all(index[qid] >= im["first_index"] for im in rebuilt["images"] for qid in im["qids"])


def test_active_uncertainty_rules():
    base = {"qid": "crop:x/1.jpg:1", "confidence": 0.9, "family": "sprite", "guess": "sprite"}
    assert act.is_uncertain(base, {"class": "sprite", "score": 0.9}) is False
    assert act.is_uncertain(base, {"class": "hi-c", "score": 0.9}) is True, "family disagreement"
    assert act.is_uncertain(base, {"class": "unknown", "score": 0.0}) is True
    assert act.is_uncertain({**base, "confidence": 0.59}, {"class": "sprite"}) is True
    assert act.is_uncertain({**base, "confidence": 0.59}, None, use_knn=False) is True
    assert act.is_uncertain({**base, "guess": "unknown", "family": None}, None, use_knn=False) is False
    assert act.crop_family({"family": "Coca-Cola Zero", "guess": "unknown"}) == "coca cola zero"
    assert act.gallery_shortfall({"part_options": [{"class": "coca-cola-part1"},
                                                  {"class": "coca-cola-part2"}]},
                                 {"coca-cola-part1": 4, "coca-cola-part2": 40}) == 1, \
        "the scarcest part of the family sets the need"


def test_active_gallery_counts_accepts_lists_and_arrays():
    assert act.gallery_counts(["sprite", "sprite", "hi-c"]) == {"sprite": 2, "hi-c": 1}
    assert act.gallery_counts(np.asarray(["sprite", "hi-c", "hi-c"])) == {"sprite": 1, "hi-c": 2}


def test_active_face_merge_offsets_ids_and_keeps_splits_separate(active_doc):
    queue, splits, root = active_doc
    bootstrap = {
        "images": [{"id": 1, "file_name": IMAGE_A, "width": 400, "height": 300}],
        "categories": [{"id": 1, "name": "sprite"}],
        "annotations": [_ann(1, 1, SPRITE, 20.0, 200.0)],
    }
    faces = {
        "images": [{"id": 1, "file_name": IMAGE_A, "width": 400, "height": 300}],
        "categories": [{"id": 1, "name": "sprite"}],
        "annotations": [
            {"id": 5, "image_id": 1, "category_id": BLANK, "bbox": [20.0, 200.0, 40.0, 80.0],
             "attributes": {"source": "vlm:faces", "bottom_shelf": True, "identity_visible": True,
                            "truncated": False}},
            {"id": 6, "image_id": 1, "category_id": BLANK, "bbox": [300.0, 205.0, 40.0, 80.0],
             "attributes": {"source": "vlm:faces", "bottom_shelf": True, "identity_visible": True,
                            "truncated": False}},
            {"id": 7, "image_id": 1, "category_id": BLANK, "bbox": [300.0, 20.0, 40.0, 80.0],
             "attributes": {"source": "vlm:faces", "bottom_shelf": False, "identity_visible": True}},
        ],
    }
    merged = act.merge_face_proposals(bootstrap, faces, {IMAGE_A: "test"}, want_split="test")
    ids = sorted(a["id"] for a in merged["annotations"])
    assert ids == [5 + act.FACE_ANN_OFFSET, 6 + act.FACE_ANN_OFFSET], "a clean face box replaces its twin"
    assert merged["n_bootstrap_replaced"] == 1
    assert all(int(a["id"]) >= act.FACE_ANN_OFFSET for a in merged["annotations"] if int(a["id"]) != 1)
    untouched = act.merge_face_proposals(bootstrap, faces, {IMAGE_A: "test"}, want_split="train")
    assert [a["id"] for a in untouched["annotations"]] == [1], "other splits' proposals are left alone"


def test_active_queue_still_serves_the_quiz_app(active_doc, tmp_path):
    queue, _, root = active_doc
    rebuilt = act.build_active_queue(queue, {}, _all_preds(active_doc), {}, test_crops=2, val_cap=2,
                                    train_cap=2)
    store = QuizStore(rebuilt, tmp_path / "gt" / "quiz_answers.jsonl", root)
    assert store.state_payload()["answered"] == 0
    crop = next(q for q in store.questions if q["kind"] == bq.CROP and q["purpose"] == "train")
    store.answer(crop["qid"], {"result": "class", "class": TOP_CLASS})
    assert store.state_payload()["answered"] == 1
    assert store.crop_jpeg(store.questions.index(crop))[:2] == b"\xff\xd8"


# ================================================================ gallery refresh


STUB_DIM = 168


def _gallery(rows, *, model="stub"):
    """rows: ``(file_name, ann_id, bbox_xywh, label)`` -> gallery.npz arrays."""
    rng = np.random.default_rng(0)
    embeddings = np.stack([rng.normal(size=STUB_DIM).astype(np.float32) for _ in rows]) if rows \
        else np.zeros((0, STUB_DIM), dtype=np.float32)
    return {
        "embeddings": embeddings,
        "labels": np.asarray([r[3] for r in rows], dtype=str),
        "image_ids": np.asarray([str(100 + i) for i in range(len(rows))], dtype=str),
        "groups": np.asarray([rg.group_id_for(r[0]) for r in rows], dtype=str),
        "splits": np.asarray(["train"] * len(rows), dtype=str),
        "bboxes": np.asarray([r[2] for r in rows], dtype=np.float32),
        "annotation_ids": np.asarray([str(r[1]) for r in rows], dtype=str),
        "file_names": np.asarray([r[0] for r in rows], dtype=str),
        "model": np.asarray([model], dtype=str),
    }


def _bundle(dir_path: Path, rows, *, name="base-v1", with_files=False):
    dir_path.mkdir(parents=True, exist_ok=True)
    np.savez(dir_path / "gallery.npz", **_gallery(rows))
    (dir_path / "pipeline.yaml").write_text(
        "localizer:\n  type: tiled_detector\n  checkpoint: detector/c.pth\n"
        "identifier:\n  type: embedding_knn\n  gallery: gallery.npz\n  model: stub\n"
        "  k: 5\n  max_per_class: 10\ncounting:\n  identity_thr: 0.45\n", encoding="utf-8")
    (dir_path / "classes.json").write_text(json.dumps({"detector_classes": ["cartridge"]}), encoding="utf-8")
    (dir_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "name": name, "embedder_id": "stub",
        "detector_classes": ["cartridge"], "thresholds": {"score_thr": 0.3, "identity_thr": 0.45},
        "provenance": {"run_id": "R", "notes": "base notes"},
        "files": {} if not with_files else {"pipeline.yaml": {"sha256": "x"}},
    }), encoding="utf-8")
    return dir_path


def _active_fixture(tmp_path, base_rows=None):
    """A tiny base bundle plus a queue with train, val and test crop questions."""
    root = tmp_path / "images"
    for name in (IMAGE_T, "gU/011.jpg", IMAGE_A, IMAGE_B):
        _write_frame(root, name)
    base_rows = base_rows if base_rows is not None else [
        (IMAGE_T, 20, [20.0, 20.0, 40.0, 80.0], "sprite"),
        ("gU/011.jpg", 21, [120.0, 25.0, 40.0, 80.0], "hi-c"),
    ]
    base = _bundle(tmp_path / "artifacts" / "base-v1", base_rows)
    questions = [
        {"qid": bq.crop_qid(IMAGE_T, 20), "kind": "crop", "split": "train", "purpose": "train",
         "image_id": IMAGE_T, "ann_id": 20, "bbox": [20.0, 20.0, 60.0, 100.0], "guess": "sprite",
         "confidence": 0.5, "attributes": {}},
        {"qid": bq.crop_qid(IMAGE_T, 100021), "kind": "crop", "split": "train", "purpose": "train",
         "image_id": IMAGE_T, "ann_id": 100021, "bbox": [120.0, 205.0, 160.0, 285.0],
         "guess": "unknown", "confidence": 0.4, "attributes": {},
         "part_options": [{"part": 1, "class": "coca-cola-part1"}, {"part": 2, "class": "coca-cola-part2"}]},
        {"qid": bq.crop_qid(IMAGE_A, 30), "kind": "crop", "split": "test", "purpose": "eval",
         "image_id": IMAGE_A, "ann_id": 30, "bbox": [20.0, 20.0, 60.0, 100.0], "guess": "sprite",
         "confidence": 0.5, "attributes": {}},
        {"qid": bq.crop_qid(IMAGE_B, 31), "kind": "crop", "split": "val", "purpose": "tune",
         "image_id": IMAGE_B, "ann_id": 31, "bbox": [20.0, 20.0, 60.0, 100.0], "guess": "sprite",
         "confidence": 0.5, "attributes": {}},
    ]
    queue = {"version": 1, "classes": CLASSES, "questions": questions,
             "images": [{"image_id": q["image_id"], "split": q["split"], "qids": [q["qid"]],
                         "coco_image_id": 1, "width": 400, "height": 300} for q in questions]}
    splits = {IMAGE_T: "train", "gU/011.jpg": "train", IMAGE_A: "test", IMAGE_B: "val"}
    return base, root, queue, splits


def test_refresh_reuses_model_rows_and_applies_human_labels(tmp_path):
    base, root, queue, splits = _active_fixture(tmp_path)
    answers = {
        queue["questions"][0]["qid"]: {"result": "class", "class": "coca-cola-part1"},
        queue["questions"][1]["qid"]: {"result": "part", "part": 2,
                                       "part_options": [{"part": 1, "class": "coca-cola-part1"},
                                                          {"part": 2, "class": "coca-cola-part2"}]},
        queue["questions"][2]["qid"]: {"result": "class", "class": "coca-cola-part2"},
        queue["questions"][3]["qid"]: {"result": "class", "class": "coca-cola-part2"},
    }
    labels = rg.collect_train_labels(queue, answers, splits, gallery=load_base(base))
    by_key = {(row["image_id"], row["ann_id"]): row for row in labels["rows"]}
    assert by_key[(IMAGE_T, 20)]["class"] == "coca-cola-part1", "the human overrides the model"
    assert by_key[(IMAGE_T, 20)]["identity_source"] == "human"
    assert by_key[(IMAGE_T, 100021)]["class"] == "coca-cola-part2", "a face crop joins the gallery"
    assert by_key[("gU/011.jpg", 21)]["identity_source"] == "vlm", "unanswered keeps the model label"
    assert IMAGE_A not in {row["image_id"] for row in labels["rows"]}
    assert IMAGE_B not in {row["image_id"] for row in labels["rows"]}


def load_base(base: Path):
    return rg.load_gallery(base / "gallery.npz")


def test_refresh_gallery_never_holds_test_or_val_images(tmp_path):
    """The leakage assertion: no test or val image id in a refreshed gallery."""
    base, root, queue, splits = _active_fixture(tmp_path)
    answers = {q["qid"]: {"result": "class", "class": "coca-cola-part2"} for q in queue["questions"]}
    result = rg.refresh_bundle(base=base, out="out-v2", queue=queue, answers=answers, splits=splits,
                              images_root=root, artifacts_dir=tmp_path / "artifacts",
                              embedder=Embedder("stub"), overwrite=True)
    gallery = load_base(Path(result["bundle"]))
    leaked = [name for name in map(str, gallery["file_names"]) if splits.get(name) != "train"]
    assert leaked == []
    assert {str(s) for s in gallery["splits"]} == {"train"}
    assert set(gallery["file_names"].tolist()) <= {IMAGE_T, "gU/011.jpg"}


def test_refresh_gallery_reproduces_the_base_with_no_answers(tmp_path):
    base, root, queue, splits = _active_fixture(tmp_path)
    base_gallery = load_base(base)
    result = rg.refresh_bundle(base=base, out="out-v2", queue=queue, answers={}, splits=splits,
                              images_root=root, artifacts_dir=tmp_path / "artifacts",
                              embedder=Embedder("stub"), overwrite=True)
    fresh = load_base(Path(result["bundle"]))
    assert [str(x) for x in fresh["labels"]] == [str(x) for x in base_gallery["labels"]]
    assert np.array_equal(fresh["embeddings"], base_gallery["embeddings"])
    assert [str(x) for x in fresh["file_names"]] == [str(x) for x in base_gallery["file_names"]]
    assert result["n_human"] == 0


def test_refresh_gallery_writes_bundle_manifest_and_verifies(tmp_path):
    base, root, queue, splits = _active_fixture(tmp_path)
    answers = {queue["questions"][0]["qid"]: {"result": "class", "class": "dr-pepper-part1"},
               queue["questions"][1]["qid"]: {"result": "not_cartridge"}}
    result = rg.refresh_bundle(base=base, out="out-v2", queue=queue, answers=answers, splits=splits,
                              images_root=root, artifacts_dir=tmp_path / "artifacts",
                              embedder=Embedder("stub"), overwrite=True,
                              answers_path="data/gt/quiz_answers.jsonl")
    from freestyle.artifacts import verify_bundle

    manifest = verify_bundle(Path(result["bundle"]))
    refresh = manifest["provenance"]["gallery_refresh"]
    assert refresh["n_human_train_labels"] == 1
    assert refresh["n_human_corrections"] == 1
    assert refresh["answers"] == "data/gt/quiz_answers.jsonl"
    assert manifest["embedder_id"] == "stub"
    assert manifest["provenance"]["base_bundle"] == "base-v1"
    assert manifest["thresholds"] == {"score_thr": 0.3, "identity_thr": 0.45}
    assert "human-verified TRAIN crop labels" in manifest["provenance"]["notes"]
    assert (Path(result["bundle"]) / "pipeline.yaml").is_file()
    assert [str(x) for x in load_base(Path(result["bundle"]))["labels"]] == ["dr-pepper-part1", "hi-c"]


def test_refresh_gallery_refuses_a_test_row(tmp_path):
    base, root, queue, splits = _active_fixture(tmp_path)
    queue["questions"][0]["split"] = "test"  # a mislabelled question must not leak
    answers = {queue["questions"][0]["qid"]: {"result": "class", "class": "coca-cola-part1"}}
    labels = rg.collect_train_labels(queue, answers, splits, gallery=load_base(base))
    row = next(row for row in labels["rows"] if row["image_id"] == IMAGE_T)
    assert row["class"] == "sprite" and row["identity_source"] == "vlm", "the test answer is ignored"
    assert labels["n_human"] == 0
    with pytest.raises(ValueError, match="leak"):
        rg._assert_train_only({"file_names": np.asarray([IMAGE_T, IMAGE_A])}, splits)


def test_refresh_cli_writes_labels_and_bundle(tmp_path, capsys, active_doc):
    queue, splits, root = active_doc
    (tmp_path / "queue.json").write_text(json.dumps(queue), encoding="utf-8")
    (tmp_path / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    base = _bundle(tmp_path / "artifacts" / "base-v1", [(IMAGE_T, 20, [20.0, 20.0, 40.0, 80.0], "sprite")])
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"qid": bq.crop_qid(IMAGE_T, 21), "kind": "crop",
                                  "image_id": IMAGE_T, "split": "train", "ann_id": 21,
                                  "answer": {"result": "class", "class": "dr-pepper-part2"}}) + "\n",
                      encoding="utf-8")
    code = rg.main(["--base", str(base), "--out", "cli-v2",
                    "--queue", str(tmp_path / "queue.json"), "--answers", str(answers),
                    "--splits", str(tmp_path / "splits.json"), "--images-root", str(root),
                    "--labels-out", str(tmp_path / "labels.json"),
                    "--artifacts-dir", str(tmp_path / "artifacts"), "--dry-run"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "train crop questions: 5" in printed
    labels = json.loads((tmp_path / "labels.json").read_text())
    assert labels["n_rows"] == 2 and labels["n_human"] == 1
    assert labels["rows"][0]["identity_source"] == "vlm", "the unanswered box keeps the model label"
    assert {row["image_id"] for row in labels["rows"]} == {IMAGE_T}
