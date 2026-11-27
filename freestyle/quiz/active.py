"""Active-learning selection: keep only the quiz questions that teach the system something.

The queue a human answers is not a sample of the proposals — it is a budget of
human attention. Every question must earn its place by one of three purposes
(``purpose`` field, see :mod:`freestyle.quiz.build_queue`):

``eval`` (test split)
    Honest measurement only. A fixed random sample of test crops (seeded, shelf
    stratified) plus every test crop already answered, and all test count
    questions. **Never read by the gallery builder or used for training** —
    test answers measure the system, they do not train it.
``tune`` (val split)
    Crops where the current bundle's identifier is not sure of itself
    (``confidence < CONF_THR`` or the VLM's family disagrees with the gallery
    k-NN's family). That is where a threshold is worth tuning; capped.
``train`` (train split)
    Crops from train-split images (bootstrap boxes plus the tighter bottom-shelf
    face boxes, which show the printed part numeral best) chosen where the
    identifier is uncertain, ordered so the most useful come first: two-part
    brand families whose part the VLM could not read, then classes whose gallery
    holds fewer than ``MIN_GALLERY`` examples.

A question the human has already answered is always kept (dropping it would
throw their answer away); removed questions are only ever unanswered ones.

Nothing here is ground truth, and nothing here may put a test or val image in
front of the gallery builder — see :mod:`freestyle.identify.refresh_gallery`.

Run:
    uv run python -m freestyle.quiz.active \
        --proposals data/labels/bootstrap_v2.json --faces data/labels/faces_v2.json \
        --splits data/splits.json --answers data/gt/quiz_answers.jsonl \
        --bundle rfdetr-m-1280-v1 --out data/gt/quiz_queue.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from freestyle.quiz.build_queue import (
    COUNT,
    CROP,
    PURPOSE_BY_SPLIT,
    PURPOSE_EVAL,
    PURPOSE_TRAIN,
    PURPOSE_TUNE,
    DEFAULT_CATALOG,
    DEFAULT_IMAGES_ROOT,
    DEFAULT_SPLITS,
    UNKNOWN,
    family_classes,
    family_key,
    quiz_classes,
    validate_queue,
)
from freestyle.quiz.to_coco import load_answers

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOOTSTRAP = REPO_ROOT / "data" / "labels" / "bootstrap_v2.json"
DEFAULT_FACES = REPO_ROOT / "data" / "labels" / "faces_v2.json"
DEFAULT_QUEUE = REPO_ROOT / "data" / "gt" / "quiz_queue.json"
DEFAULT_ANSWERS = REPO_ROOT / "data" / "gt" / "quiz_answers.jsonl"
DEFAULT_KNN_CACHE = REPO_ROOT / "data" / "derived" / "quiz" / "knn_predictions.json"

#: A crop question this confident is treated as settled unless the gallery disagrees.
CONF_THR = 0.6
#: How many test crops to sample for evaluation, and the caps on the other blocks.
TEST_CROPS = 30
VAL_CAP = 40
TRAIN_CAP = 60
#: Seed of the fixed evaluation sample: same proposals -> same 30 test crops.
SAMPLE_SEED = 0
#: A class with fewer than this many gallery rows is under-fed (train block priority).
MIN_GALLERY = 5
#: Face annotations are offset into this id range so their qids cannot collide with
#: bootstrap annotation ids of the same image; ``ann_id >= FACE_ANN_OFFSET`` means face box.
FACE_ANN_OFFSET = 100_000
#: Boxes this overlapped are the same physical cartridge (bootstrap vs face proposal).
DEDUPE_IOU = 0.5
#: Question blocks in the order the human answers them.
BLOCK_ORDER = (PURPOSE_EVAL, PURPOSE_TRAIN, PURPOSE_TUNE)


# --------------------------------------------------------------------------- helpers


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _confidence(question: Mapping[str, Any]) -> float:
    try:
        return float(question.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def crop_family(question: Mapping[str, Any]) -> str:
    """Brand family the VLM proposed for a crop question ('' when it proposed none)."""
    family = family_key(question.get("family") or "")
    if family and family != UNKNOWN:
        return family
    guess = str(question.get("guess") or "")
    if guess and guess != UNKNOWN:
        return family_key(guess)
    return ""


def crop_classes(question: Mapping[str, Any], prediction: Mapping[str, Any] | None = None) -> list[str]:
    """Catalog classes this question is really about (both parts for a part question)."""
    options = [str(o["class"]) for o in question.get("part_options") or [] if o.get("class")]
    if options:
        return options
    guess = str(question.get("guess") or "")
    if guess and guess != UNKNOWN:
        return [guess]
    knn = str((prediction or {}).get("class") or "")
    return [knn] if knn and knn != UNKNOWN else []


# ---------------------------------------------------------------------- uncertainty


def is_uncertain(
    question: Mapping[str, Any],
    prediction: Mapping[str, Any] | None = None,
    *,
    conf_thr: float = CONF_THR,
    use_knn: bool | None = None,
) -> bool:
    """True when the bundle's identifier is not confident about this crop.

    ``prediction`` is the gallery k-NN's answer for the crop
    (``{"class": ..., "score": ...}``); ``use_knn`` defaults to whether any
    predictions were supplied at all. Uncertain means the VLM confidence is
    below ``conf_thr``, or there is no usable k-NN class, or the k-NN's family
    disagrees with the family the VLM named.
    """
    if use_knn is None:
        use_knn = prediction is not None
    if _confidence(question) < float(conf_thr):
        return True
    if not use_knn:
        return False
    knn_class = str((prediction or {}).get("class") or "") or UNKNOWN
    if knn_class == UNKNOWN:
        return True
    family = crop_family(question)
    return not family or family_key(knn_class) != family


def gallery_counts(labels: Iterable[str] | Path | str | Any) -> dict[str, int]:
    """Rows per class in an identifier gallery (from a label list or a ``gallery.npz``)."""
    if isinstance(labels, (str, Path)) and str(labels).endswith(".npz") and Path(labels).is_file():
        import numpy as np

        with np.load(labels, allow_pickle=False) as data:
            values = [str(v) for v in data["labels"]]
    else:
        values = [str(v) for v in labels] if not _is_array(labels) else [str(v) for v in labels.tolist()]
    return dict(Counter(values))


def _is_array(value: Any) -> bool:
    return hasattr(value, "dtype") or (hasattr(value, "shape") and not isinstance(value, (str, list, tuple)))


def gallery_shortfall(
    question: Mapping[str, Any],
    counts: Mapping[str, int],
    *,
    prediction: Mapping[str, Any] | None = None,
    min_gallery: int = MIN_GALLERY,
) -> int:
    """How many rows the scarcest class this question is about lacks (0 = well fed)."""
    classes = crop_classes(question, prediction)
    if not classes:
        return int(min_gallery)
    return max(0, max(int(min_gallery) - int(counts.get(cls, 0)) for cls in classes))


def train_priority(
    question: Mapping[str, Any],
    parts: Mapping[str, Mapping[int, str]],
    counts: Mapping[str, int],
    *,
    prediction: Mapping[str, Any] | None = None,
    min_gallery: int = MIN_GALLERY,
) -> tuple[int, int, float, str]:
    """Sort key (lower = more useful) for one train crop question.

    1. a two-part brand (Coke, Diet Coke, Dr Pepper, Zero) whose part is unread;
    2. classes whose gallery is thinnest (an absent class beats a thin one);
    3. the least confident proposal first; qid last, so the order is reproducible.
    """
    two_part_unread = bool(question.get("ask_part")) and str(question.get("part") or "") in ("", "None")
    shortfall = gallery_shortfall(question, counts, prediction=prediction, min_gallery=min_gallery)
    return (0 if two_part_unread else 1, -shortfall, _confidence(question), str(question["qid"]))


def val_priority(
    question: Mapping[str, Any], prediction: Mapping[str, Any] | None = None
) -> tuple[int, float, str]:
    """Sort key (lower = more useful) for one val crop question: disagreement first."""
    knn_class = str((prediction or {}).get("class") or "") or UNKNOWN
    disagree = 0 if (knn_class == UNKNOWN or not crop_family(question)
                     or family_key(knn_class) != crop_family(question)) else 1
    return (disagree, _confidence(question), str(question["qid"]))


def select_crop_sample(
    crops: Sequence[Mapping[str, Any]], n: int, *, seed: int = SAMPLE_SEED
) -> set[str]:
    """Deterministic ``n``-crop sample stratified by shelf (top/bottom kept proportional).

    Sorting by qid before sampling makes the result independent of the order the
    proposals were walked in, so the same seed always asks the same questions.
    """
    if n <= 0 or not crops:
        return set()
    strata: dict[str, list[str]] = {}
    for question in crops:
        strata.setdefault(str(question.get("shelf") or "unknown"), []).append(str(question["qid"]))
    qids = {key: sorted(values) for key, values in sorted(strata.items())}
    total = sum(len(values) for values in qids.values())
    if n >= total:
        return {qid for values in qids.values() for qid in values}
    # largest-remainder allocation, ties broken by the larger stratum then the name
    sizes = {key: len(values) for key, values in qids.items()}
    shares = {key: n * size / total for key, size in sizes.items()}
    alloc = {key: int(value) for key, value in shares.items()}
    left = n - sum(alloc.values())
    for key in sorted(qids, key=lambda k: (-(shares[k] - alloc[k]), -sizes[k], k))[:left]:
        alloc[key] += 1
    rng = random.Random(int(seed))
    chosen: set[str] = set()
    for key in sorted(qids):
        pool = [qid for qid in qids[key] if qid not in chosen]
        want = min(alloc[key], len(pool))
        chosen.update(rng.sample(pool, want))
    if len(chosen) < n:  # a stratum came up short: top up from the rest, stably
        rest = sorted(qid for values in qids.values() for qid in values if qid not in chosen)
        chosen.update(rng.sample(rest, min(n - len(chosen), len(rest))))
    return chosen


# --------------------------------------------------------------------------- queue


def _question_index(questions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(q["qid"]): i for i, q in enumerate(questions)}


def select_questions(
    queue: Mapping[str, Any],
    answers: Mapping[str, Any] | Iterable[str] = (),
    predictions: Mapping[str, Mapping[str, Any]] | None = None,
    counts: Mapping[str, int] | None = None,
    *,
    test_crops: int = TEST_CROPS,
    val_cap: int = VAL_CAP,
    train_cap: int = TRAIN_CAP,
    seed: int = SAMPLE_SEED,
    conf_thr: float = CONF_THR,
    min_gallery: int = MIN_GALLERY,
) -> list[dict[str, Any]]:
    """Pick the questions worth asking and label each with its ``purpose``.

    ``answers`` may be the ``load_answers`` mapping or any iterable of qids;
    ``predictions`` maps qid -> gallery k-NN result (omit entirely to skip the
    k-NN disagreement test). Returns the questions in block order (test, train,
    val), each a shallow copy with ``purpose`` and ``selected_by`` set.
    """
    validate_queue(queue)
    answered: set[str] = set(answers) if not hasattr(answers, "keys") else set(answers.keys())
    questions = [dict(q) for q in queue.get("questions") or []]
    order = _question_index(questions)
    classes = queue.get("classes") or []
    parts = family_classes(classes)
    use_knn = predictions is not None
    prediction_map: Mapping[str, Mapping[str, Any]] = predictions or {}
    counts = counts if counts is not None else {}

    picked: dict[str, str] = {}  # qid -> why

    def take(question: Mapping[str, Any], why: str) -> None:
        picked.setdefault(str(question["qid"]), why)

    by_split_kind = {(s, k): [q for q in questions if q.get("split") == s and q.get("kind") == k]
                     for s in ("test", "val", "train") for k in (COUNT, CROP)}

    # 1. eval: every test count, a fixed stratified sample of test crops.
    for question in by_split_kind[("test", COUNT)]:
        take(question, "count_all")
    sampled = select_crop_sample(by_split_kind[("test", CROP)], test_crops, seed=seed)
    for question in by_split_kind[("test", CROP)]:
        if str(question["qid"]) in sampled:
            take(question, "eval_sample")

    # 2. tune: val counts, plus the most uncertain val crops up to the cap.
    for question in by_split_kind[("val", COUNT)]:
        take(question, "count_all")
    uncertain_val = [
        q for q in by_split_kind[("val", CROP)]
        if is_uncertain(q, prediction_map.get(str(q["qid"])) if use_knn else None,
                        conf_thr=conf_thr, use_knn=use_knn)
    ]
    uncertain_val.sort(key=lambda q: val_priority(q, prediction_map.get(str(q["qid"]))))
    for question in uncertain_val[: max(0, val_cap)]:
        take(question, "uncertain")

    # 3. train: train-split crops only, most useful first, capped.
    uncertain_train = [
        q for q in by_split_kind[("train", CROP)]
        if is_uncertain(q, prediction_map.get(str(q["qid"])) if use_knn else None,
                        conf_thr=conf_thr, use_knn=use_knn)
    ]
    uncertain_train.sort(
        key=lambda q: train_priority(q, parts, counts,
                                     prediction=prediction_map.get(str(q["qid"])),
                                     min_gallery=min_gallery)
    )
    for question in uncertain_train[: max(0, train_cap)]:
        take(question, "uncertain")

    # 4. never discard an answer: any question the human touched stays in.
    for question in questions:
        if str(question["qid"]) in answered:
            take(question, "answered")

    out: list[dict[str, Any]] = []
    for question in questions:
        qid = str(question["qid"])
        if qid not in picked:
            continue
        purpose = PURPOSE_BY_SPLIT.get(str(question.get("split")), PURPOSE_TRAIN)
        question["purpose"] = purpose
        question["selected_by"] = picked[qid]
        out.append(question)
    out.sort(key=lambda q: (BLOCK_ORDER.index(q["purpose"]), order[str(q["qid"])]))
    return out


def _tally(questions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {"total": len(questions)}
    for question in questions:
        split, kind = str(question.get("split")), str(question.get("kind"))
        purpose = str(question.get("purpose") or "")
        out[f"{split}/{kind}"] = out.get(f"{split}/{kind}", 0) + 1
        if purpose:
            out[f"{purpose}"] = out.get(f"{purpose}", 0) + 1
    return out


def build_active_queue(
    base_queue: Mapping[str, Any],
    answers: Mapping[str, Any] | Iterable[str] = (),
    predictions: Mapping[str, Mapping[str, Any]] | None = None,
    counts: Mapping[str, int] | None = None,
    *,
    test_crops: int = TEST_CROPS,
    val_cap: int = VAL_CAP,
    train_cap: int = TRAIN_CAP,
    seed: int = SAMPLE_SEED,
    conf_thr: float = CONF_THR,
    min_gallery: int = MIN_GALLERY,
    gallery: Mapping[str, Any] | None = None,
    knn: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild ``base_queue`` as an active-learning queue (same schema, purposes added)."""
    chosen = select_questions(
        base_queue, answers, predictions, counts,
        test_crops=test_crops, val_cap=val_cap, train_cap=train_cap,
        seed=seed, conf_thr=conf_thr, min_gallery=min_gallery,
    )
    base_questions = list(base_queue.get("questions") or [])
    before, after = _tally(base_questions), _tally(chosen)

    index = {str(q["qid"]): i for i, q in enumerate(chosen)}
    kept_by_image: dict[str, list[str]] = {}
    for question in chosen:
        kept_by_image.setdefault(str(question["image_id"]), []).append(str(question["qid"]))
    images: list[dict[str, Any]] = []
    for image in base_queue.get("images") or []:
        qids = kept_by_image.get(str(image["image_id"]))
        if not qids:
            continue
        entry = dict(image)
        entry["qids"] = qids
        entry["first_index"] = min(index[qid] for qid in qids)
        entry["purposes"] = sorted({str(q.get("purpose")) for q in chosen
                                    if str(q["image_id"]) == str(image["image_id"])})
        images.append(entry)
    images.sort(key=lambda im: (BLOCK_ORDER.index(str(im["purposes"][0])), im["first_index"]))
    purposes = [str(q["purpose"]) for q in chosen]

    return {
        "version": base_queue.get("version"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "splits_order": [split for split in ("test", "train", "val")
                         if any(q.get("split") == split for q in chosen)],
        "classes": base_queue.get("classes") or [],
        "selection": {
            "mode": "active",
            "seed": int(seed),
            "test_crops": int(test_crops),
            "val_cap": int(val_cap),
            "train_cap": int(train_cap),
            "conf_thr": float(conf_thr),
            "min_gallery": int(min_gallery),
            "gallery": dict(gallery or {}),
            "knn": dict(knn or {}),
            "before": before,
            "after": after,
            "purposes": {p: int(sum(1 for x in purposes if x == p)) for p in BLOCK_ORDER},
        },
        "images": images,
        "questions": chosen,
    }


# ---------------------------------------------------------------- face proposals


def merge_face_proposals(
    bootstrap: Mapping[str, Any],
    faces: Mapping[str, Any],
    split_map: Mapping[str, str],
    *,
    want_split: str = "train",
    shelf: str = "bottom",
    offset: int = FACE_ANN_OFFSET,
    iou_thr: float = DEDUPE_IOU,
) -> dict[str, Any]:
    """One proposals document: ``bootstrap`` plus tighter face boxes for ``want_split``.

    Face annotations only enter for the given shelf (the bottom shelf's boxes show
    the printed part numeral best) and only where they add something: a face box
    that overlaps a bootstrap box replaces it when it is an untruncated,
    identity-visible bottom-shelf box, otherwise it is dropped as a duplicate.
    Kept face annotations get ``ann_id = offset + id`` so their question ids cannot
    collide with bootstrap ids in the same image.
    """
    boot_images = {int(im["id"]): str(im.get("file_name") or "")
                   for im in bootstrap.get("images") or [] if im.get("id") is not None}
    file_to_coco = {name: coco for coco, name in boot_images.items() if name}
    face_images = {int(im["id"]): str(im.get("file_name") or "")
                   for im in faces.get("images") or [] if im.get("id") is not None}
    wanted = {name for name, split in split_map.items()
              if split == want_split and name in file_to_coco}

    boot_anns = [dict(ann) for ann in bootstrap.get("annotations") or [] if ann.get("id") is not None]
    boot_by_file: dict[str, list[dict[str, Any]]] = {}
    for ann in boot_anns:
        boot_by_file.setdefault(boot_images.get(int(ann["image_id"]), ""), []).append(ann)

    faces_here: dict[str, list[dict[str, Any]]] = {}
    for ann in faces.get("annotations") or []:
        file_name = face_images.get(int(ann.get("image_id") or 0), "")
        if file_name not in wanted or ann.get("id") is None:
            continue
        attrs = ann.get("attributes") or {}
        if shelf == "bottom" and not attrs.get("bottom_shelf"):
            continue
        faces_here.setdefault(file_name, []).append(
            {"ann": ann, "box": _xywh(ann), "attrs": attrs, "file_name": file_name}
        )

    replaced: set[int] = set()
    kept: list[dict[str, Any]] = []
    for file_name in sorted(faces_here):
        boxes = boot_by_file.get(file_name, [])
        for entry in faces_here[file_name]:
            match = next((ann for ann in boxes
                          if int(ann["id"]) not in replaced
                          and _iou(entry["box"], _xywh(ann)) >= iou_thr), None)
            if match is None:
                kept.append(entry)  # a face the bootstrap pass never proposed
            elif _clean_face(entry["attrs"]):
                replaced.add(int(match["id"]))  # tighter view of the same cartridge
                kept.append(entry)

    annotations = [ann for ann in boot_anns if int(ann["id"]) not in replaced]
    for entry in kept:
        ann = dict(entry["ann"])
        ann["id"] = int(ann["id"]) + offset
        ann["image_id"] = file_to_coco[entry["file_name"]]
        attrs = dict(ann.get("attributes") or {})
        attrs["proposal_source"] = "faces"
        ann["attributes"] = attrs
        annotations.append(ann)

    merged = dict(bootstrap)
    merged["annotations"] = annotations
    merged["categories"] = bootstrap.get("categories") or faces.get("categories") or []
    merged["images"] = [dict(im) for im in bootstrap.get("images") or []]
    return {**merged, "n_face_added": len(kept), "n_bootstrap_replaced": len(replaced)}


def _xywh(ann: Mapping[str, Any]) -> list[float]:
    x, y, w, h = [float(v) for v in ann.get("bbox") or [0.0, 0.0, 0.0, 0.0]]
    return [x, y, x + w, y + h]


def _clean_face(attrs: Mapping[str, Any]) -> bool:
    """A face box wins its slot over a bootstrap box when the front is fully readable."""
    return bool(attrs.get("identity_visible")) and not bool(attrs.get("truncated"))


# ------------------------------------------------------------------------ k-NN pass


def _gallery_fingerprint(bundle_dir: Path) -> dict[str, str]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = (manifest.get("files") or {}).get("gallery.npz") or {}
    return {"bundle": str(manifest.get("name") or bundle_dir.name),
            "gallery_sha256": str(entry.get("sha256") or ""),
            "embedder": str(manifest.get("embedder_id") or "")}


def knn_predictions(
    questions: Sequence[Mapping[str, Any]],
    *,
    bundle: str | Path,
    images_root: Path = DEFAULT_IMAGES_ROOT,
    cache: Path | None = DEFAULT_KNN_CACHE,
    device: str = "cpu",
    batch_size: int = 32,
    pad: float = 0.08,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Gallery k-NN class/score per crop question, from the bundle's own embedder.

    CPU only (this host has no GPU work), and cached under ``cache`` keyed by the
    bundle's gallery checksum and each crop's box, so re-building the queue costs
    no re-embedding. Returns ``(qid -> prediction, provenance)``.
    """
    from PIL import Image

    from freestyle.geometry import crop as crop_box
    from freestyle.pipeline import EmbeddingIdentifier

    bundle_dir = Path(bundle)
    if not bundle_dir.is_dir():
        from freestyle import artifacts

        bundle_dir = artifacts.ARTIFACTS_DIR / str(bundle)
    fingerprint = _gallery_fingerprint(bundle_dir)
    gallery = bundle_dir / "gallery.npz"

    import yaml

    pipeline = yaml.safe_load((bundle_dir / "pipeline.yaml").read_text(encoding="utf-8")) or {}
    block = (pipeline.get("identifier") or {}).get("embedding") or (pipeline.get("identifier") or {})
    model = str(block.get("model") or fingerprint["embedder"] or "google/siglip2-base-patch16-224")

    wanted = [q for q in questions if q.get("kind") == CROP]
    store: dict[str, Any] = {}
    if cache and Path(cache).is_file():
        try:
            store = json.loads(Path(cache).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            store = {}
    cached = store.get("predictions") if isinstance(store.get("predictions"), dict) else {}
    if store.get("gallery_sha256") != fingerprint["gallery_sha256"] or store.get("model") != model:
        cached, store = {}, {}

    predictions: dict[str, dict[str, Any]] = {}
    todo: list[tuple[dict[str, Any], str]] = []
    for question in wanted:
        qid = str(question["qid"])
        box = [round(float(v), 2) for v in question.get("bbox") or []]
        hit = cached.get(qid)
        if isinstance(hit, dict) and hit.get("box") == box:
            predictions[qid] = {"class": hit.get("class"), "score": hit.get("score")}
        else:
            todo.append((question, qid))

    if todo:
        identifier = EmbeddingIdentifier(gallery, model=model,
                                        k=int(block.get("k") or 5),
                                        max_per_class=block.get("max_per_class"),
                                        classifier=str(block.get("classifier") or "knn"),
                                        device=device)
        by_image: dict[str, list[tuple[dict[str, Any], str]]] = {}
        for question, qid in todo:
            by_image.setdefault(str(question["image_id"]), []).append((question, qid))
        for image_id, items in by_image.items():
            path = Path(images_root) / image_id
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                crops = [crop_box(image, [float(v) for v in q["bbox"]], pad=pad) for q, _ in items]
            vectors = identifier.embedder.embed(crops, batch_size=batch_size)
            for (question, qid), result in zip(items, identifier.predict_embeddings(vectors)):
                predictions[qid] = {"class": str(result["class"]), "score": round(float(result["id_score"]), 4)}
                cached[qid] = {"class": str(result["class"]), "score": round(float(result["id_score"]), 4),
                               "image_id": image_id,
                               "box": [round(float(v), 2) for v in question["bbox"]]}
        if cache:
            cache = Path(cache)
            cache.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       **fingerprint, "model": model, "pad": pad, "predictions": cached}
            tmp = cache.with_suffix(cache.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            tmp.replace(cache)

    counts = gallery_counts(gallery)
    provenance = {**fingerprint, "model": model, "n_crops": len(wanted), "n_embedded": len(todo),
                  "gallery_rows": sum(counts.values()), "rows_per_class": counts}
    return predictions, provenance


# ----------------------------------------------------------------------------- cli


def _split_map(path: Path | None) -> dict[str, str]:
    if not path or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (data.get("images") or {}).items()}


def report(before: Mapping[str, int], after: Mapping[str, int], answered: int) -> str:
    """Printable before/after summary, one line per split/purpose bucket."""
    rows: dict[str, dict[str, int]] = {}
    for table, tag in ((before, "before"), (after, "after")):
        for key, value in table.items():
            rows.setdefault(key, {})[tag] = value
    lines = [f"{'bucket':<24}{'before':>8}{'after':>8}"]
    for key in sorted(rows):
        row = rows[key]
        lines.append(f"{key:<24}{row.get('before', 0):>8}{row.get('after', 0):>8}")
    lines.append(f"answered kept: {answered}")
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rebuild the quiz queue as an active-learning selection")
    p.add_argument("--proposals", type=Path, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--faces", type=Path, default=DEFAULT_FACES, help="face proposals merged in for train")
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    p.add_argument("--out", type=Path, default=DEFAULT_QUEUE)
    p.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--audit", type=Path, default=REPO_ROOT / "data" / "derived" / "audit" / "audit.json")
    p.add_argument("--bundle", default="rfdetr-m-1280-v1",
                   help="bundle whose gallery scores crop uncertainty ('' to skip the k-NN test)")
    p.add_argument("--knn-cache", type=Path, default=DEFAULT_KNN_CACHE)
    p.add_argument("--device", default="cpu", help="embedder device for the k-NN pass (cpu)")
    p.add_argument("--test-crops", type=int, default=TEST_CROPS)
    p.add_argument("--val-cap", type=int, default=VAL_CAP)
    p.add_argument("--train-cap", type=int, default=TRAIN_CAP)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--conf-thr", type=float, default=CONF_THR)
    p.add_argument("--min-gallery", type=int, default=MIN_GALLERY)
    p.add_argument("--dry-run", action="store_true", help="print the counts, write nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    from freestyle.quiz import build_queue as bq

    args = make_parser().parse_args(argv)
    splits = _split_map(args.splits)
    bootstrap = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    proposals = bootstrap
    if args.faces and Path(args.faces).is_file():
        faces = json.loads(Path(args.faces).read_text(encoding="utf-8"))
        proposals = merge_face_proposals(bootstrap, faces, splits)
    classes = quiz_classes(args.catalog)
    base = bq.build_queue(proposals, args.splits, ["test", "val", "train"], classes,
                          args.images_root, args.audit)
    crop_questions = [q for q in base["questions"] if q.get("kind") == CROP
                      and q.get("split") in ("val", "train")]

    answers = load_answers(args.answers) if args.answers else {}
    predictions: Mapping[str, Mapping[str, Any]] | None = None
    knn_provenance: dict[str, Any] = {}
    counts: dict[str, int] | None = None
    gallery_provenance: dict[str, Any] = {}
    if args.bundle:
        predictions, knn_provenance = knn_predictions(
            crop_questions, bundle=args.bundle, images_root=args.images_root,
            cache=args.knn_cache, device=args.device)
        counts = knn_provenance.get("rows_per_class")
        gallery_provenance = {"bundle": knn_provenance.get("bundle"),
                             "gallery_sha256": knn_provenance.get("gallery_sha256"),
                             "rows": knn_provenance.get("gallery_rows")}
    queue = build_active_queue(
        base, answers, predictions, counts,
        test_crops=args.test_crops, val_cap=args.val_cap, train_cap=args.train_cap,
        seed=args.seed, conf_thr=args.conf_thr, min_gallery=args.min_gallery,
        gallery=gallery_provenance, knn=knn_provenance,
    )
    validate_queue(queue)
    before = queue["selection"]["before"]
    after = queue["selection"]["after"]
    kept = sum(1 for q in queue["questions"] if str(q["qid"]) in answers)
    print(report(before, after, kept), file=sys.stderr)
    print(f"unanswered left: {len(queue['questions']) - kept}", file=sys.stderr)
    if args.dry_run:
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    return 0


__all__ = [
    "BLOCK_ORDER",
    "CONF_THR",
    "FACE_ANN_OFFSET",
    "MIN_GALLERY",
    "SAMPLE_SEED",
    "TEST_CROPS",
    "TRAIN_CAP",
    "VAL_CAP",
    "build_active_queue",
    "crop_classes",
    "crop_family",
    "gallery_counts",
    "gallery_shortfall",
    "is_uncertain",
    "knn_predictions",
    "merge_face_proposals",
    "report",
    "select_crop_sample",
    "select_questions",
    "train_priority",
    "val_priority",
]


if __name__ == "__main__":
    raise SystemExit(main())
