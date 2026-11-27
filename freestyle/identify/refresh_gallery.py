"""Refresh the identifier gallery with human-verified TRAIN crops and re-export a bundle.

The deployed bundle's gallery is the identifier's memory. It starts as VLM-labelled
train crops (model labels, never ground truth); every train crop a human answers in
the phone quiz either confirms one of those rows or adds a better one, and the human
label always wins over the model's.

Leakage rule, enforced here and asserted in the tests: only images whose
``data/splits.json`` split is ``train`` may contribute a gallery row. Test and val
crops are measured with, never trained on — a row from them would quietly inflate
every metric the project reports.

Run:
    uv run python -m freestyle.identify.refresh_gallery \
        --base rfdetr-m-1280-v1 --out rfdetr-m-1280-v2 \
        --queue data/gt/quiz_queue.json --answers data/gt/quiz_answers.jsonl \
        --labels-out data/gt/train_crop_labels.json

Re-runnable: point ``--answers`` at the live quiz log again whenever more answers
arrive. With no train answers at all it still runs and reproduces the base bundle's
gallery row for row (untouched rows keep their stored vectors, so the reproduction
is exact, not approximate).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from freestyle.artifacts import MANIFEST_VERSION, ARTIFACTS_DIR, _sha256_file, verify_bundle
from freestyle.catalog import CARTRIDGE, UNKNOWN
from freestyle.quiz.build_queue import CROP, PURPOSE_TRAIN, qid_ann_id
from freestyle.quiz.to_coco import _identity, load_answers
from freestyle.review.server import group_id_for

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE = REPO_ROOT / "data" / "gt" / "quiz_queue.json"
DEFAULT_ANSWERS = REPO_ROOT / "data" / "gt" / "quiz_answers.jsonl"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits.json"
DEFAULT_IMAGES_ROOT = REPO_ROOT / "data" / "raw" / "freestyle-vision-data" / "images"
DEFAULT_BASE = "rfdetr-m-1280-v1"
DEFAULT_OUT = "rfdetr-m-1280-v2"
DEFAULT_LABELS_OUT = REPO_ROOT / "data" / "gt" / "train_crop_labels.json"
#: Crop padding the frozen galleries were embedded with (``embeddings.npz['pad']``).
CROP_PAD = 0.08
GALLERY_VERSION = 1

#: Row sources, recorded per gallery row so provenance survives the refresh.
SOURCE_VLM = "vlm"
SOURCE_HUMAN = "human"
#: Answers that make a box useless as a gallery example (wrong/ambiguous geometry).
_BAD_BOX_ANSWERS = frozenset({"not_cartridge", "two_plus", "partial", "implied_not_usable"})


# ------------------------------------------------------------------- train answers


def train_crop_questions(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Crop questions the human may feed to the identifier: ``train`` split only."""
    out = []
    for question in queue.get("questions") or []:
        if question.get("kind") != CROP:
            continue
        split = str(question.get("split") or "")
        purpose = str(question.get("purpose") or "")
        if split == PURPOSE_TRAIN or (not split and purpose == PURPOSE_TRAIN):
            out.append(dict(question))
    return out


def answer_label(question: Mapping[str, Any], answer: Mapping[str, Any] | None) -> dict[str, Any]:
    """What one crop answer says about the crop's identity.

    Returns ``{"class": str | None, "usable": bool, "source": 'human'|'model'|None}``
    where ``class`` ``None`` means 'take this box out of the gallery' and
    ``source`` ``None`` means 'the human did not answer it'.
    """
    if not answer:
        return {"class": None, "usable": True, "source": None}
    result = str(answer.get("result") or "")
    name, identity_source, _ = _identity(question, answer)
    if result in _BAD_BOX_ANSWERS or result == "skip":
        return {"class": None, "usable": False,
                "source": SOURCE_HUMAN if result != "skip" else None}
    if result == "correct":  # human confirmed the model's guess
        return {"class": str(question.get("guess") or UNKNOWN), "usable": True, "source": SOURCE_HUMAN}
    if name in (None, UNKNOWN) or (identity_source or "") == "model":
        return {"class": None, "usable": True, "source": SOURCE_HUMAN}
    return {"class": str(name), "usable": True, "source": SOURCE_HUMAN}


def collect_train_labels(
    queue: Mapping[str, Any],
    answers: Mapping[str, Mapping[str, Any]],
    splits: Mapping[str, str],
    *,
    gallery: Mapping[str, np.ndarray] | None = None,
    allow_unverified_vlm: bool = False,
) -> dict[str, Any]:
    """Merge the base gallery's train rows with the human's train-crop answers.

    ``gallery`` is the base bundle's ``gallery.npz`` (already train-only). Each base
    row keeps its model label unless a human answered that same crop; answered crops
    the gallery does not hold (the tighter face boxes, classes the VLM skipped) are
    added. ``rows`` are the records written to ``train_crop_labels.json``.
    """
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if gallery is not None:
        n = len(gallery["labels"])
        for i in range(n):
            file_name = str(gallery["file_names"][i]) if "file_names" in gallery else ""
            image_id = str(gallery["image_ids"][i])
            ann_id = int(str(gallery["annotation_ids"][i])) if "annotation_ids" in gallery else i
            label = str(gallery["labels"][i])
            bbox = [float(v) for v in gallery["bboxes"][i]] if "bboxes" in gallery else [0, 0, 0, 0]
            group = str(gallery["groups"][i]) if "groups" in gallery else group_id_for(file_name or image_id)
            rows[(file_name or image_id, ann_id)] = {
                "image_id": file_name or image_id,
                "coco_image_id": image_id,
                "group_id": group,
                "ann_id": ann_id,
                "bbox_xywh": [round(v, 3) for v in bbox],
                "class": label,
                "identity_source": SOURCE_VLM,
                "verified": False,
            }

    rejected: list[dict[str, Any]] = []
    for question in train_crop_questions(queue):
        image_id = str(question["image_id"])
        split = splits.get(image_id)
        if split != PURPOSE_TRAIN:
            # a train-labelled question on a non-train image would leak eval data
            # into the identifier's memory: drop it loudly, never silently.
            rejected.append({"qid": question.get("qid"), "image_id": image_id,
                             "split": split, "reason": "not_train_split"})
            continue
        ann_id = int(question.get("ann_id") if question.get("ann_id") is not None
                     else (qid_ann_id(str(question.get("qid") or "")) or 0))
        outcome = answer_label(question, answers.get(str(question["qid"])))
        key = (image_id, ann_id)
        existing = rows.get(key)
        if outcome["source"] is None:  # unanswered: the model's label stands
            if existing is None and allow_unverified_vlm and str(question.get("guess") or "") not in ("", UNKNOWN):
                rows[key] = {
                    "image_id": image_id, "ann_id": ann_id,
                    "bbox_xywh": _bbox_xywh(question),
                    "group_id": group_id_for(image_id),
                    "coco_image_id": question.get("coco_image_id"),
                    "class": str(question["guess"]), "identity_source": SOURCE_VLM,
                    "verified": False,
                }
            continue
        if existing is not None:
            existing["verified"] = True
            existing["identity_source"] = SOURCE_HUMAN
            existing["qid"] = question["qid"]
            if outcome["class"] is None or not outcome["usable"]:
                existing["class"] = None
            else:
                existing["class"] = outcome["class"]
            continue
        if outcome["class"] is None:
            continue  # the human rejected a box the gallery never held
        rows[key] = {
            "image_id": image_id, "ann_id": ann_id, "qid": question["qid"],
            "bbox_xywh": _bbox_xywh(question), "group_id": group_id_for(image_id),
            "coco_image_id": question.get("coco_image_id"),
            "class": outcome["class"], "identity_source": SOURCE_HUMAN, "verified": True,
        }

    kept = {key: row for key, row in rows.items() if row.get("class") not in (None, UNKNOWN, CARTRIDGE)}
    dropped = {key: row for key, row in rows.items() if row.get("class") in (None, UNKNOWN, CARTRIDGE)}
    ordered = [kept[key] for key in sorted(kept)]
    human = sum(1 for row in ordered if row["identity_source"] == SOURCE_HUMAN)
    return {
        "version": GALLERY_VERSION,
        "split": PURPOSE_TRAIN,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": ("identity rows; only identity_source='human' rows are human-verified "
                 "ground truth (docs/freestyle/ARCHITECTURE.md). Test and val crops are "
                 "excluded here by construction and asserted in the unit tests."),
        "n_rows": len(ordered),
        "n_human": human,
        "n_model": len(ordered) - human,
        "counts_by_class": dict(sorted(Counter(row["class"] for row in ordered).items())),
        "rows": ordered,
        "rejected": rejected,
        "dropped": [dropped[key] for key in sorted(dropped)],
    }


def _bbox_xywh(question: Mapping[str, Any]) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in question["bbox"]]
    return [round(x0, 3), round(y0, 3), round(x1 - x0, 3), round(y1 - y0, 3)]


# ------------------------------------------------------------------------ gallery


def _load_base_bundle(name_or_path: str | Path, artifacts_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [Path(name_or_path), Path(artifacts_dir) / str(name_or_path),
                  ARTIFACTS_DIR / str(name_or_path)]
    path = next((c for c in candidates if c.is_dir()), None)
    if path is None:
        raise FileNotFoundError(f"base bundle not found: {name_or_path}")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if manifest.get("files"):
        verify_bundle(path)  # production bundles must verify, fixture ones may not
    return path, manifest


def load_gallery(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def embed_crops(
    crops: Sequence[tuple[str, Sequence[float]]],
    *,
    embedder: Any = None,
    model: str | None = None,
    images_root: Path = DEFAULT_IMAGES_ROOT,
    device: str = "cpu",
    batch_size: int = 32,
    pad: float = CROP_PAD,
) -> np.ndarray:
    """Embed ``(image_id, bbox_xywh)`` crops with the bundle's own embedder (CPU)."""
    if not crops:
        return np.empty((0, 0), dtype=np.float32)
    from PIL import Image

    from freestyle.geometry import crop as crop_box

    if embedder is None:
        from freestyle.identify.embed import Embedder

        if not model:
            raise ValueError("either an embedder or a model name is required")
        embedder = Embedder(name=model, device=device, batch_size=batch_size)
    images: dict[str, Image.Image] = {}
    pil: list[Image.Image] = []
    for image_id, box in crops:
        if image_id not in images:
            with Image.open(Path(images_root) / image_id) as handle:
                images[image_id] = handle.convert("RGB")
        image = images[image_id]
        x, y, w, h = [float(v) for v in box]
        pil.append(crop_box(image, [x, y, x + w, y + h], pad=pad))
    vectors = embedder.embed(pil)
    images.clear()
    return np.asarray(vectors, dtype=np.float32)


def build_gallery_arrays(
    labels: Mapping[str, Any],
    base: Mapping[str, np.ndarray],
    *,
    model: str,
    vectors: np.ndarray | None = None,
    reembed: str = "auto",
    embedder: Any = None,
    images_root: Path = DEFAULT_IMAGES_ROOT,
    device: str = "cpu",
    batch_size: int = 32,
    pad: float = CROP_PAD,
) -> dict[str, np.ndarray]:
    """``gallery.npz`` arrays for the refreshed rows.

    ``auto`` keeps the base gallery's vectors for rows it already holds (so a run
    with no new answers reproduces the base gallery exactly) and embeds only the
    added crops; ``all`` re-embeds every row with ``model`` on ``device``.
    """
    rows = list(labels.get("rows") or [])
    base_index = _base_index(base)

    need: list[tuple[str, Sequence[float]]] = []
    needed_at: dict[int, int] = {}
    for i, row in enumerate(rows):
        key = (str(row["image_id"]), int(row["ann_id"]))
        if reembed == "all" or key not in base_index:
            needed_at[i] = len(need)
            need.append((str(row["image_id"]), row["bbox_xywh"]))
    if vectors is None:
        vectors = embed_crops(need, embedder=embedder, model=model, images_root=images_root,
                             device=device, batch_size=batch_size, pad=pad)
    vectors = np.asarray(vectors, dtype=np.float32)
    by_row = {row_index: vectors[position] for row_index, position in needed_at.items()}

    base_embeddings = np.asarray(base.get("embeddings") if base.get("labels") is not None else np.empty((0, 0)),
                                 dtype=np.float32)
    dim = int(vectors.shape[1]) if vectors.ndim == 2 and len(vectors) else int(base_embeddings.shape[-1] or 0)
    embeddings = np.zeros((len(rows), dim or 1), dtype=np.float32)
    out = {
        "labels": np.asarray([str(r["class"]) for r in rows], dtype=str),
        "image_ids": np.asarray([str(r.get("coco_image_id") or r["image_id"]) for r in rows], dtype=str),
        "groups": np.asarray([str(r.get("group_id") or group_id_for(str(r["image_id"]))) for r in rows], dtype=str),
        "splits": np.asarray([PURPOSE_TRAIN] * len(rows), dtype=str),
        "bboxes": np.asarray([[float(v) for v in r["bbox_xywh"]] for r in rows], dtype=np.float32)
        if rows else np.empty((0, 4), dtype=np.float32),
        "annotation_ids": np.asarray([str(int(r["ann_id"])) for r in rows], dtype=str),
        "file_names": np.asarray([str(r["image_id"]) for r in rows], dtype=str),
        "sources": np.asarray([str(r.get("identity_source") or SOURCE_VLM) for r in rows], dtype=str),
        "model": np.asarray([model], dtype=str),
    }
    for i, row in enumerate(rows):
        key = (str(row["image_id"]), int(row["ann_id"]))
        if i in by_row:
            embeddings[i] = by_row[i]
        elif key in base_index:
            embeddings[i] = base_embeddings[base_index[key]]
    out["embeddings"] = embeddings if rows else np.empty((0, dim or 1), dtype=np.float32)
    return out


def _base_index(base: Mapping[str, np.ndarray]) -> dict[tuple[str, int], int]:
    """``(file_name, annotation id)`` -> row number in the base gallery."""
    labels = base.get("labels")
    if labels is None or len(labels) == 0:
        return {}
    files = base.get("file_names")
    image_ids = base.get("image_ids")
    anns = base.get("annotation_ids")
    index: dict[tuple[str, int], int] = {}
    for i in range(len(labels)):
        file_name = str(files[i]) if files is not None else ""
        if not file_name:
            file_name = str(image_ids[i]) if image_ids is not None else ""
        ann_id = int(str(anns[i])) if anns is not None else i
        index[(file_name, ann_id)] = i
    return index


# -------------------------------------------------------------------------- export


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _base_labels(base: Mapping[str, np.ndarray]) -> list[str]:
    labels = base.get("labels")
    return [] if labels is None else [str(v) for v in labels]


def refresh_bundle(
    *,
    base: str | Path = DEFAULT_BASE,
    out: str | Path = DEFAULT_OUT,
    queue: Mapping[str, Any],
    answers: Mapping[str, Mapping[str, Any]],
    splits: Mapping[str, str],
    artifacts_dir: str | Path | None = None,
    images_root: Path = DEFAULT_IMAGES_ROOT,
    labels_doc: Mapping[str, Any] | None = None,
    answers_path: str | Path | None = None,
    embedder: Any = None,
    device: str = "cpu",
    reembed: str = "auto",
    overwrite: bool = False,
    include_unanswered_vlm: bool = False,
    batch_size: int = 32,
    pad: float = CROP_PAD,
) -> dict[str, Any]:
    """Write ``artifacts/<out>`` = ``base`` with a gallery refreshed from train answers."""
    if reembed not in ("auto", "all"):
        raise ValueError(f"reembed must be 'auto' or 'all', got {reembed!r}")
    root = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    base_dir, base_manifest = _load_base_bundle(base, root)
    base_gallery = load_gallery(base_dir / "gallery.npz")
    model = str((base_gallery.get("model") or [b""])[0]) or str(base_manifest.get("embedder_id") or "")
    if embedder is not None:
        model = str(getattr(embedder, "name", model))

    labels = labels_doc or collect_train_labels(
        queue, answers, splits, gallery=base_gallery, allow_unverified_vlm=include_unanswered_vlm)
    arrays = build_gallery_arrays(
        labels, base_gallery, model=model, reembed=reembed, images_root=images_root,
        device=device, batch_size=batch_size, pad=pad, embedder=embedder)

    n_rows = int(len(arrays["labels"]))
    if n_rows == 0:
        raise ValueError("refreshed gallery is empty; refusing to export a bundle with no gallery")
    _assert_train_only(arrays, splits)

    out_name = Path(str(out)).name
    out_dir = root / out_name
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"bundle already exists: {out_dir} (use --overwrite)")
    stage = Path(tempfile.mkdtemp(prefix=f".{out_name}.", dir=str(root)))
    try:
        for src in sorted(p for p in base_dir.rglob("*") if p.is_file()):
            rel = src.relative_to(base_dir).as_posix()
            if rel in ("gallery.npz", "manifest.json"):
                continue
            _link_or_copy(src, stage / rel)
        np.savez(stage / "gallery.npz", **arrays)

        if (base_dir / "pipeline.yaml").is_file():
            (stage / "pipeline.yaml").write_text(
                (base_dir / "pipeline.yaml").read_text(encoding="utf-8"), encoding="utf-8")

        base_index = _base_index(base_gallery)
        base_labels = _base_labels(base_gallery)
        human_rows = [row for row in labels.get("rows") or []
                      if row.get("identity_source") == SOURCE_HUMAN]
        n_human = len(human_rows)
        n_corrected = n_confirmed = n_added = 0
        for row in human_rows:
            position = base_index.get((str(row["image_id"]), int(row["ann_id"])))
            if position is None:
                n_added += 1
            elif base_labels[position] != str(row["class"]):
                n_corrected += 1
            else:
                n_confirmed += 1
        base_provenance = dict(base_manifest.get("provenance") or {})
        note = str(base_provenance.get("notes") or "").strip()
        base_provenance["notes"] = " | ".join(
            part for part in (note,
                              f"gallery refreshed from {n_human} human-verified TRAIN crop labels "
                              f"({n_corrected} corrections, {n_confirmed} confirmations, "
                              f"{n_added} new crops); no test or val crop in the gallery")
            if part)
        files = {
            p.relative_to(stage).as_posix(): dict(zip(("sha256", "bytes"), _sha256_file(p)))
            for p in sorted(p for p in stage.rglob("*") if p.is_file())
        }
        thresholds = dict(base_manifest.get("thresholds") or {})
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "name": out_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedder_id": model,
            "detector_classes": base_manifest.get("detector_classes") or [],
            "thresholds": thresholds,
            "provenance": {
                **base_provenance,
                "base_bundle": base_manifest.get("name") or base_dir.name,
                "gallery_refresh": {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "split": PURPOSE_TRAIN,
                    "n_human_train_labels": n_human,
                    "n_human_corrections": n_corrected,
                    "n_human_confirmations": n_confirmed,
                    "n_human_added": n_added,
                    "n_gallery_rows": n_rows,
                    "n_base_rows": len(_base_labels(base_gallery)),
                    "n_dropped_rows": len(labels.get("dropped") or []),
                    "answers": str(answers_path) if answers_path else None,
                    "embedder": model,
                    "reembed": reembed,
                    "crop_pad": pad,
                    "counts_by_class": labels.get("counts_by_class"),
                    "leakage_check": "no test/val image id in any gallery row",
                },
            },
            "files": files,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        stage.rename(out_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"bundle": str(out_dir), "n_rows": n_rows, "n_human": n_human,
            "n_added": n_added, "n_corrected": n_corrected, "labels": labels, "manifest": manifest}


def _assert_train_only(arrays: Mapping[str, np.ndarray], splits: Mapping[str, str]) -> None:
    """Refuse a gallery that holds a test or val image."""
    files = arrays.get("file_names")
    if files is None:
        return
    bad = sorted({str(name) for name in files if splits.get(str(name)) in ("test", "val", "exclude")})
    if bad:
        raise ValueError(f"gallery would leak non-train images into training: {bad[:5]}")


# ------------------------------------------------------------------------------ cli


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refresh the identifier gallery from human train labels")
    p.add_argument("--base", default=DEFAULT_BASE, help="base bundle name or directory")
    p.add_argument("--out", default=DEFAULT_OUT, help="bundle name to write under artifacts/")
    p.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    p.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS,
                   help="live quiz answer log (re-run this as answers arrive)")
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    p.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--device", default="cpu", help="embedder device for added crops (cpu)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pad", type=float, default=CROP_PAD)
    p.add_argument("--reembed", choices=("auto", "all"), default="auto",
                   help="re-embed added crops only (auto) or every row (all)")
    p.add_argument("--include-unanswered-vlm", action="store_true",
                   help="also add unanswered train crops under their model label (off by default)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print what the refresh would do")
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    answers = load_answers(args.answers)
    split_data = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    splits = {str(k): str(v) for k, v in (split_data.get("images") or {}).items()}

    base_dir, base_manifest = _load_base_bundle(args.base, args.artifacts_dir or ARTIFACTS_DIR)
    base_gallery = load_gallery(base_dir / "gallery.npz")
    labels = collect_train_labels(queue, answers, splits, gallery=base_gallery,
                                 allow_unverified_vlm=args.include_unanswered_vlm)
    n_train_q = len(train_crop_questions(queue))
    n_answered = sum(1 for q in train_crop_questions(queue) if str(q["qid"]) in answers)
    print(f"train crop questions: {n_train_q} ({n_answered} answered, {labels['n_human']} usable human labels)")
    print(f"gallery rows: {labels['n_rows']} "
          f"({labels['n_human']} human / {labels['n_model']} model) from base {base_dir.name}")
    if labels["rejected"]:
        print(f"REFUSED {len(labels['rejected'])} train questions on non-train images: "
              f"{labels['rejected'][:3]}", file=sys.stderr)
    if args.labels_out:
        path = Path(args.labels_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        print(f"train crop labels -> {path}")
    if args.dry_run:
        return 0

    result = refresh_bundle(
        base=args.base, out=args.out, queue=queue, answers=answers, splits=splits,
        artifacts_dir=args.artifacts_dir, images_root=args.images_root, labels_doc=labels,
        answers_path=args.answers, device=args.device, reembed=args.reembed, overwrite=True,
        include_unanswered_vlm=args.include_unanswered_vlm, batch_size=args.batch_size, pad=args.pad)
    gallery = load_gallery(Path(result["bundle"]) / "gallery.npz")
    base_labels = _base_labels(base_gallery)
    same = (len(gallery["labels"]) == len(base_labels)
            and [str(v) for v in gallery["labels"]] == base_labels
            and bool(np.array_equal(gallery["embeddings"], base_gallery["embeddings"])))
    print(f"{result['bundle']}: {result['n_rows']} rows, {result['n_human']} human labels"
          + ("; gallery identical to base" if same else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
