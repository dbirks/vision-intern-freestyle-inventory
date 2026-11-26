"""CLI for crop embedding extraction and Leave-One-Group-Out evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from freestyle.identify.classify import (
    KNNClassifier,
    _compute_classification_metrics,
    LinearProbe,
    PrototypeClassifier,
    leave_one_group_out_eval,
    select_unknown_threshold,
)
from freestyle.identify.embed import Embedder

EXCLUDED_GT_LABELS = frozenset({"unknown", "cartridge"})


def _fallback_crop(
    image_pil: Image.Image,
    box_xyxy: Sequence[float],
    pad: float = 0.08,
) -> Image.Image:
    x0, y0, x1, y1 = map(float, box_xyxy)
    if x1 <= x0 or y1 <= y0 or pad < 0:
        raise ValueError("box or padding is invalid")
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    rect = (
        max(0, math.floor(x0 - dx)),
        max(0, math.floor(y0 - dy)),
        min(image_pil.width, math.ceil(x1 + dx)),
        min(image_pil.height, math.ceil(y1 + dy)),
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        raise ValueError("box does not intersect image")
    return image_pil.crop(rect)


def _fallback_crop_quad(image_pil: Image.Image, quad_4x2: Any, out_h: int | None = None) -> Image.Image:
    q = np.asarray(quad_4x2, dtype=float)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("quad must contain four finite 2D corners")
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
    if width <= 0 or height <= 0 or (out_h is not None and out_h <= 0):
        raise ValueError("quad and output height must be nondegenerate")
    h = max(2, round(height if out_h is None else out_h))
    w = max(2, round(width * h / height))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
    matrix, rhs = [], []
    for (x, y), (u, v) in zip(dst, q, strict=True):
        matrix.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
        rhs.extend([u, v])
    coeffs = np.linalg.solve(matrix, rhs)
    return image_pil.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def _get_crop_fns():
    try:
        from freestyle.geometry import crop as geom_crop
        from freestyle.geometry import crop_quad as geom_crop_quad

        return geom_crop, geom_crop_quad
    except ImportError:
        return _fallback_crop, _fallback_crop_quad


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _match_split_group(name: str, split_groups: Mapping[str, Any]) -> str | None:
    if name in split_groups:
        return name
    parts = [p for p in name.split("/") if p]
    for i in range(len(parts), 0, -1):
        prefix = "/".join(parts[:i])
        if prefix in split_groups:
            return prefix
    for seg in parts:
        if seg in split_groups:
            return seg
    return None


def _resolve_groups_and_splits(
    file_names: Sequence[str],
    image_ids: Sequence[str],
    groups_data: Mapping[str, Any] | None,
    npz_groups: Sequence[str] | None = None,
) -> tuple[list[str], list[str | None]]:
    explicit_map: Mapping[str, Any] | None = None
    split_groups: Mapping[str, Any] | None = None
    split_images: Mapping[str, Any] | None = None

    if groups_data is not None:
        if "image_to_group" in groups_data and isinstance(groups_data["image_to_group"], Mapping):
            explicit_map = groups_data["image_to_group"]
        elif "image_groups" in groups_data and isinstance(groups_data["image_groups"], Mapping):
            explicit_map = groups_data["image_groups"]
        elif "groups" in groups_data and "images" in groups_data:
            split_groups = groups_data["groups"]
            split_images = groups_data["images"]
        else:
            explicit_map = groups_data

    resolved_groups: list[str] = []
    resolved_splits: list[str | None] = []
    for idx, (fname, img_id) in enumerate(zip(file_names, image_ids, strict=True)):
        grp: str | None = None
        split: str | None = None

        if explicit_map is not None:
            for cand in (fname, img_id):
                if cand and cand in explicit_map:
                    grp = str(explicit_map[cand])
                    break
        elif split_groups is not None:
            for cand in (fname, img_id):
                if cand:
                    matched = _match_split_group(cand, split_groups)
                    if matched is not None:
                        grp = matched
                        split = str(split_groups[matched])
                        break
            if split is None and split_images is not None:
                for cand in (fname, img_id):
                    if cand and cand in split_images:
                        split = str(split_images[cand])
                        break

        if split_images is not None:
            image_split = split_images.get(fname, split_images.get(img_id))
            if image_split == "exclude" and split is not None:
                split = "exclude"
            if image_split is None or split is None or image_split != split:
                raise ValueError(f"Missing or inconsistent split/group for {fname}")
        if not grp or split not in ("train", "val", "test", "exclude"):
            raise ValueError(f"Missing split/group provenance for {fname}")
        if npz_groups is not None and str(npz_groups[idx]) != grp:
            raise ValueError(f"Inconsistent stored group for {fname}")
        resolved_groups.append(grp)
        resolved_splits.append(split)

    return resolved_groups, resolved_splits


def run_embed(args: argparse.Namespace) -> dict[str, Any]:
    coco = _load_json(args.coco)
    images_root = Path(args.images_root)
    out_path = Path(args.out)

    crop_fn, crop_quad_fn = _get_crop_fns()

    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    cats_by_id = {cat["id"]: str(cat["name"]) for cat in coco.get("categories", [])}

    anns_by_image: dict[Any, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    crops: list[Image.Image] = []
    labels: list[str] = []
    category_ids: list[int] = []
    image_ids: list[str] = []
    file_names: list[str] = []
    annotation_ids: list[str] = []
    bboxes: list[list[float]] = []
    used_quad: list[bool] = []
    skipped = 0

    for img_id, img_rec in images_by_id.items():
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue
        file_name = str(img_rec.get("file_name") or img_id)
        img_path = images_root / file_name
        with Image.open(img_path) as im:
            orig_img = im.convert("RGB")

        for ann in anns:
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            quad = ann.get("quad")
            if args.quad and quad is not None:
                crop_img = crop_quad_fn(orig_img, quad)
                is_quad = True
            else:
                crop_img = crop_fn(orig_img, [x, y, x + w, y + h], pad=args.pad)
                is_quad = False

            cat_id = int(ann["category_id"])
            label = cats_by_id.get(cat_id, str(cat_id))
            crops.append(crop_img)
            labels.append(label)
            category_ids.append(cat_id)
            image_ids.append(str(img_id))
            file_names.append(file_name)
            annotation_ids.append(str(ann.get("id", len(annotation_ids))))
            bboxes.append([x, y, w, h])
            used_quad.append(is_quad)

    groups, splits = _resolve_groups_and_splits(file_names, image_ids, _load_json(args.split_map))
    embedder = Embedder(
        args.model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        tta_hflip=args.tta_hflip,
    )
    embeddings = embedder.embed(crops)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=embeddings,
        labels=np.asarray(labels, dtype=str),
        category_ids=np.asarray(category_ids, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=str),
        groups=np.asarray(groups, dtype=str),
        splits=np.asarray(splits, dtype=str),
        file_names=np.asarray(file_names, dtype=str),
        annotation_ids=np.asarray(annotation_ids, dtype=str),
        bboxes=np.asarray(bboxes, dtype=np.float32) if bboxes else np.empty((0, 4), dtype=np.float32),
        used_quad=np.asarray(used_quad, dtype=bool),
        model=np.asarray([embedder.name], dtype=str),
        pad=np.asarray([float(args.pad)], dtype=np.float32),
    )

    return {
        "out": str(out_path),
        "model": embedder.name,
        "n_crops": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "skipped": skipped,
        "pad": float(args.pad),
        "quad": bool(args.quad),
    }


def _make_clf_factory(clf_name: str, k: int) -> Any:
    if clf_name == "proto":
        return lambda: PrototypeClassifier()
    if clf_name == "knn":
        return lambda: KNNClassifier(k=k)
    if clf_name == "linear":
        return lambda: LinearProbe()
    raise ValueError(f"Unsupported classifier: {clf_name!r}")


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.npz, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=str)
        n = embeddings.shape[0]
        file_names = (
            data["file_names"].tolist()
            if "file_names" in data
            else [str(i) for i in range(n)]
        )
        image_ids = data["image_ids"].tolist() if "image_ids" in data else list(file_names)
        npz_groups = data["groups"].tolist() if "groups" in data else None
        model_name = str(data["model"][0]) if "model" in data and len(data["model"]) else None

    groups_data = _load_json(args.split_map)
    if not isinstance(groups_data.get("images"), Mapping) or not isinstance(groups_data.get("groups"), Mapping):
        raise ValueError("split map must contain images and groups")
    groups_list, splits_list = _resolve_groups_and_splits(
        file_names=file_names,
        image_ids=image_ids,
        groups_data=groups_data,
        npz_groups=npz_groups,
    )

    splits = np.asarray(splits_list)
    group_array = np.asarray(groups_list, dtype=str)
    valid = ~np.isin(labels, list(EXCLUDED_GT_LABELS))
    selected_splits = (["train", "val"] if args.include_val else ["train"]) if args.mode == "logo" else [args.split]
    n_excluded_unknown = int(np.sum(~valid & np.isin(splits, selected_splits)))
    clf_factory = _make_clf_factory(args.clf, k=args.k)
    if args.mode == "logo":
        allowed = ["train", "val"] if args.include_val else ["train"]
        mask = np.isin(splits, allowed) & valid
        logo = leave_one_group_out_eval(embeddings[mask], labels[mask], group_array[mask],
                                       clf_factory, unknown_thr=args.unknown_thr)
    else:
        if args.split not in ("val", "test"):
            raise ValueError("holdout requires --split val or test")
        train = (splits == "train") & valid
        query = (splits == args.split) & valid
        if not train.any() or not query.any():
            raise ValueError("holdout requires nonempty train and query sets")
        if not set(group_array[train]).isdisjoint(set(group_array[query])):
            raise ValueError("Gallery and query groups overlap")
        clf = clf_factory().fit(embeddings[train], labels[train])
        pred, scores = clf.predict_with_scores(embeddings[query], unknown_thr=args.unknown_thr)
        logo = _compute_classification_metrics(labels[query], pred, unknown_label="unknown")
        logo.update(n_groups=len(set(group_array[query])), y_true=labels[query].tolist(),
                    y_pred=pred.tolist(), scores=scores.tolist(), per_group_accuracy={})

    result: dict[str, Any] = {
        "npz": str(args.npz),
        "model": model_name,
        "clf": args.clf,
        "k": int(args.k) if args.clf == "knn" else None,
        "n_samples": logo["n_samples"],
        "n_groups": logo["n_groups"],
        "n_excluded_unknown": n_excluded_unknown,
        "accuracy": logo["accuracy"],
        "macro_f1": logo["macro_f1"],
        "coverage": logo["coverage"],
        "covered_accuracy": logo["covered_accuracy"],
        "per_class_recall": logo["per_class_recall"],
        "confusion_matrix": logo["confusion_matrix"],
        "classes": logo["classes"],
        "pred_labels": logo["pred_labels"],
        "per_group_accuracy": logo["per_group_accuracy"],
        "unknown_thr": args.unknown_thr,
    }

    if args.calibrate_unknown_coverage is not None:
        cal = select_unknown_threshold(
            scores=logo["scores"],
            y_pred=logo["y_pred"],
            y_true=logo["y_true"],
            target_coverage=args.calibrate_unknown_coverage,
        )
        result["calibrated_unknown"] = cal

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m freestyle.identify.cli",
        description="Freestyle cartridge crop embedding and identification evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_embed = sub.add_parser("embed", help="Crop COCO annotations and extract embeddings to .npz.")
    p_embed.add_argument("--split-map", required=True)
    p_embed.add_argument("--coco", required=True, help="Input COCO JSON file.")
    p_embed.add_argument("--images-root", required=True, help="Root directory for image files.")
    p_embed.add_argument("--out", required=True, help="Output .npz archive path.")
    p_embed.add_argument("--model", required=True, help="HF model ID or 'stub'.")
    p_embed.add_argument("--pad", type=float, default=0.08, help="Proportional bbox padding (default 0.08).")
    p_embed.add_argument("--quad", action="store_true", help="Use quad rectification when annotation has 'quad'.")
    p_embed.add_argument("--device", default="auto", help="Device ('auto', 'cpu', 'cuda').")
    p_embed.add_argument("--dtype", default="auto", help="Dtype ('auto', 'float32', 'float16', 'bfloat16').")
    p_embed.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default 32).")
    p_embed.add_argument("--tta-hflip", action="store_true", help="Enable horizontal flip TTA (off by default).")

    p_eval = sub.add_parser("eval", help="Run Leave-One-Group-Out evaluation on crop embeddings.")
    p_eval.add_argument("--npz", required=True, help="Input .npz archive from embed subcommand.")
    p_eval.add_argument("--split-map", required=True)
    p_eval.add_argument("--mode", choices=("logo", "holdout"), default="logo")
    p_eval.add_argument("--include-val", action="store_true")
    p_eval.add_argument(
        "--clf",
        choices=("proto", "knn", "linear"),
        default="proto",
        help="Classifier family (default proto).",
    )
    p_eval.add_argument("--k", type=int, default=5, help="Neighbors for knn (default 5).")
    p_eval.add_argument("--unknown-thr", type=float, default=None, help="Optional unknown rejection threshold.")
    p_eval.add_argument(
        "--calibrate-unknown-coverage",
        type=float,
        default=None,
        help="Optional target coverage in (0, 1] to calibrate unknown threshold on OOF scores.",
    )
    p_eval.add_argument("--split", choices=("val", "test"), default=None)
    p_eval.add_argument("--out", default=None, help="Optional output JSON metrics file.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "embed":
        summary = run_embed(args)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "eval":
        metrics = run_eval(args)
        print(json.dumps(metrics, indent=2))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
