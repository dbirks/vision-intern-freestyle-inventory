"""Unit tests for the phone-first ground-truth quiz (no private data needed).

Synthetic COCO proposals + generated JPEG frames under a temp images root, so
these tests run anywhere and never read ``data/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

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
    assert counts_of(counts, IMAGE_A)["crop_total"] == 2


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
