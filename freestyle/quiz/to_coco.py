"""Turn quiz answers into COCO ground truth in the contract format.

Only answered questions are human; unanswered proposals survive as
``identity_source = "model"`` and keep their image ``verified = false``.

Run:
    uv run python -m freestyle.quiz.to_coco --queue data/gt/quiz_queue.json \
        --answers data/gt/quiz_answers.jsonl --out data/gt/test.json --split test

Box geometry always comes from the proposal (``box_source = "model"``); the
human only ever sets identity, count and drop flags. Alongside ``--out`` a
``<split>_counts.json`` is written: per image the human-stated total, the
per-class counts derived from the answered crops, and a flag when the two
disagree (boxes missed or merged).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from freestyle.catalog import UNKNOWN
from freestyle.quiz.build_queue import COUNT, CROP, validate_queue
from freestyle.review.server import group_id_for

CARTRIDGE_SUPER = "cartridge"
HUMAN = "human"
MODEL = "model"


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    """Question id -> answer payload; the last line for a question wins."""
    out: dict[str, dict[str, Any]] = {}
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("qid") and isinstance(record.get("answer"), dict):
            out[str(record["qid"])] = record["answer"]
    return out


def _identity(question: dict[str, Any], answer: dict[str, Any] | None) -> tuple[str | None, str, str]:
    """(class or None-to-drop, identity_source, answer result) for one crop question."""
    guess = str(question.get("guess") or UNKNOWN)
    if answer is None or str(answer.get("result") or "") == "skip":
        return guess, MODEL, ""
    result = str(answer.get("result") or "")
    if result in ("not_cartridge", "implied_not_usable"):
        return None, HUMAN, result
    if result in ("class", "part"):
        return _answer_class(question, answer), HUMAN, result
    if result == "two_plus":
        return UNKNOWN, HUMAN, result
    if result == "correct":
        return guess, HUMAN, result
    if result == "partial":
        return (str(answer["class"]), HUMAN, result) if answer.get("class") else (UNKNOWN, MODEL, result)
    raise ValueError(f"unknown crop answer result: {result!r}")


def _answer_class(question: dict[str, Any], answer: dict[str, Any]) -> str:
    """Class a class/part answer names, resolving a bare part tap through the question."""
    named = str(answer.get("class") or "")
    if named:
        return named
    if answer.get("part") is not None:
        wanted = int(answer["part"])
        for option in question.get("part_options") or []:
            if int(option.get("part") or 0) == wanted:
                return str(option.get("class") or UNKNOWN)
    return UNKNOWN


def _image_counts(image: dict[str, Any], answer: dict[str, Any] | None) -> dict[str, Any]:
    """human_counts + usability for one image from its count question's answer."""
    model = {"top": int(image.get("model_top") or 0), "bottom": int(image.get("model_bottom") or 0)}
    model["total"] = int(image.get("model_total") or 0)
    result = str((answer or {}).get("result") or "")
    if result == "fix":
        top, bottom = int(answer.get("top") or 0), int(answer.get("bottom") or 0)
        return {"human_counts": {"top": top, "bottom": bottom, "total": top + bottom},
                "model_counts": model, "usable": True, "count_answered": True}
    if result == "not_cabinet":
        zero = {"top": 0, "bottom": 0, "total": 0}
        return {"human_counts": zero, "model_counts": model, "usable": False, "count_answered": True}
    if result == "correct":
        return {"human_counts": model, "model_counts": model, "usable": True, "count_answered": True}
    return {"human_counts": None, "model_counts": model, "usable": True, "count_answered": False}


def build_coco(
    queue: dict[str, Any], answers: dict[str, dict[str, Any]], split: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(COCO GT document, counts document) for one split of the queue."""
    validate_queue(queue)
    questions = {q["qid"]: q for q in queue["questions"]}
    by_image: dict[str, list[str]] = {}
    for question in queue["questions"]:
        by_image.setdefault(question["image_id"], []).append(question["qid"])

    names = [str(c["name"]) for c in queue.get("classes") or []] or [UNKNOWN]
    ids = {str(c["name"]): int(c["id"]) for c in queue.get("classes") or []}
    categories = [
        {"id": ids[name], "name": name, "supercategory": CARTRIDGE_SUPER}
        for name in names
        if name != "cartridge"
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    ann_id = 0

    for image in queue.get("images") or []:
        if image.get("split") != split:
            continue
        image_id = str(image["image_id"])
        qids = by_image.get(image_id, [])
        count_answer = answers.get(f"{COUNT}:{image_id}")
        meta = _image_counts(image, count_answer)
        crop_qids = [qid for qid in qids if questions[qid]["kind"] == CROP]
        answered = sum(1 for qid in qids if answers.get(qid, {}).get("result") not in (None, "", "skip"))
        # every crop identity must be human 'verified' is on: a skipped box keeps
        # model identity, which is never ground truth (ARCHITECTURE.md)
        human_crops = all(
            _identity(questions[qid], answers.get(qid))[1] == HUMAN for qid in crop_qids
        )
        count_ok = bool(count_answer) and str(count_answer.get("result") or "") != "skip"
        verified = bool(qids) and count_ok and (human_crops or not meta["usable"])

        coco_image = int(image.get("coco_image_id") or 0) or (len(images) + 1)
        per_class: dict[str, int] = {}
        crop_total = 0
        for qid in qids:
            question = questions[qid]
            if question["kind"] != CROP:
                continue
            name, source, result = _identity(question, answers.get(qid))
            if name is None or not meta["usable"]:
                continue  # 'not a cartridge', or a frame the human called unusable
            if source == HUMAN:  # only human-verified boxes count toward the totals
                crop_total += 1
                per_class[name] = per_class.get(name, 0) + 1
            ann_id += 1
            x0, y0, x1, y1 = [float(v) for v in question["bbox"]]
            base = dict(question.get("attributes") or {})
            attributes = {
                "truncated": bool(base.get("truncated")) or "partial" == result,
                "occluded": bool(base.get("occluded")),
                "identity_visible": bool(base.get("identity_visible", True)) and result != "two_plus" and name != UNKNOWN,
                "source": f"quiz:{source}",
                "box_source": MODEL,
                "identity_source": source,
                "proposal_ann_id": question.get("ann_id"),
            }
            if result == "two_plus":
                attributes["box_covers_2_plus"] = True
            if result == "partial":
                attributes["box_partial"] = True
            annotations.append(
                {
                    "id": int(question.get("ann_id") or ann_id),
                    "image_id": coco_image,
                    "category_id": ids.get(name, ids.get(UNKNOWN)),
                    "bbox": [round(x0, 3), round(y0, 3), round(x1 - x0, 3), round(y1 - y0, 3)],
                    "area": round((x1 - x0) * (y1 - y0), 3),
                    "iscrowd": 0,
                    "attributes": attributes,
                }
            )

        human_total = (meta["human_counts"] or {}).get("total")
        mismatch = meta["count_answered"] and meta["usable"] and human_total is not None and human_total != crop_total
        images.append(
            {
                "id": coco_image,
                "file_name": image_id,
                "coco_url": "",
                "width": int(image.get("width") or 0),
                "height": int(image.get("height") or 0),
                "group_id": group_id_for(image_id),
                "attributes": {
                    "verified": verified,
                    "scene": "no_cabinet" if not meta["usable"] else image.get("scene"),
                    "usable": bool(meta["usable"]),
                    "human_counts": meta["human_counts"],
                    "model_counts": meta["model_counts"],
                    "quiz": {
                        "questions": len(qids),
                        "answered": answered,
                        "crop_total": crop_total,
                        "counts_mismatch": bool(mismatch),
                    },
                },
            }
        )
        counts[image_id] = {
            "human_total": human_total,
            "model_total": meta["model_counts"]["total"],
            "crop_total": crop_total,
            "classes": dict(sorted(per_class.items())),
            "answered": answered,
            "questions": len(qids),
            "verified": verified,
            "usable": bool(meta["usable"]),
            "counts_mismatch": bool(mismatch),
        }

    verified_images = sum(1 for i in images if i["attributes"]["verified"])
    document = {
        "info": {
            "description": f"Freestyle cartridge ground truth — quiz-built, split '{split}'",
            "human_verified_images": verified_images,
            "total_images": len(images),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "annotation_contract": "docs/freestyle/ARCHITECTURE.md",
            "source": "freestyle.quiz",
        },
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    counts_doc = {
        "version": 1,
        "split": split,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "images": counts,
        "mismatches": sorted(k for k, v in counts.items() if v["counts_mismatch"]),
        "unverified": sorted(k for k, v in counts.items() if not v["verified"]),
    }
    return document, counts_doc


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quiz answers -> COCO ground truth")
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="COCO json to write (e.g. data/gt/test.json)")
    p.add_argument("--split", required=True, choices=["train", "val", "test"], help="which split to export")
    p.add_argument(
        "--counts",
        type=Path,
        default=None,
        help="counts json (default: <out dir>/<split>_counts.json)",
    )
    return p


def _write(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    answers = load_answers(args.answers)
    document, counts = build_coco(queue, answers, args.split)
    _write(args.out, document)
    counts_path = args.counts or Path(args.out).parent / f"{args.split}_counts.json"
    _write(counts_path, counts)
    info = document["info"]
    print(
        f"{args.split}: {info['human_verified_images']}/{info['total_images']} images verified, "
        f"{len(document['annotations'])} boxes -> {args.out}; counts -> {counts_path}"
        f"{' (mismatch: ' + ', '.join(counts['mismatches'][:3]) + ')' if counts['mismatches'] else ''}",
        file=sys.stderr,
    )
    if not answers:
        print("warning: no answers found; everything is identity_source=model", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
