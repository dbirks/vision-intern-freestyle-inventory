"""Tiled / full-frame class-agnostic cartridge proposal CLI (pipeline B)."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from freestyle.geometry import (
    crop,
    drop_edge_truncated,
    image_boxes_to_tile,
    make_tiles,
    merge_tiled,
    tile_boxes_to_image,
    xywh_to_xyxy,
    xyxy_to_xywh,
)
from freestyle.propose.backends import BACKENDS, TRANSFORMERS_VERSION, ProposalBackend, build_backend

DEFAULT_PROMPTS = ["a narrow upright cardboard cartridge box"]
SPLIT_NAMES = frozenset({"train", "val", "test", "exclude", "all"})


@dataclass(frozen=True)
class ImageEntry:
    """Input image entry resolved from COCO JSON, list file, or split name."""

    image_id: int | str
    file_name: str
    width: int | None = None
    height: int | None = None


def _collect_hardware_meta() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        torch = None
    try:
        import transformers

        tf_ver = transformers.__version__
    except ImportError:
        tf_ver = None

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None

    cuda = torch is not None and torch.cuda.is_available()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if cuda else 0.0
    return {
        "git_sha": sha,
        "transformers_version": tf_ver,
        "pinned_transformers_version": TRANSFORMERS_VERSION,
        "hardware": {
            "torch": torch.__version__ if torch else None,
            "cuda": torch.version.cuda if torch else None,
            "gpu": torch.cuda.get_device_name() if cuda else None,
        },
        "gpu_mem_gb": peak_gb,
        "peak_gpu_mem_gb": peak_gb,
    }


def resolve_images(
    images_arg: str | Path,
    splits_path: str | Path = "data/splits.json",
) -> tuple[list[ImageEntry], int]:
    """Resolve ``--images`` from a COCO JSON, text list file, or split name.

    Returns ``(entries, cartridge_category_id)``.
    """
    arg_str = str(images_arg)
    path = Path(arg_str)
    cartridge_cat_id = 1

    if path.is_file():
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "images" not in data:
                raise ValueError(f"Expected COCO JSON object with 'images' key in {path}")
            for cat in data.get("categories", []):
                if str(cat.get("name", "")).lower() == "cartridge":
                    cartridge_cat_id = int(cat["id"])
                    break
            entries = [
                ImageEntry(
                    image_id=img["id"],
                    file_name=str(img["file_name"]),
                    width=int(img["width"]) if "width" in img else None,
                    height=int(img["height"]) if "height" in img else None,
                )
                for img in data["images"]
            ]
            return entries, cartridge_cat_id

        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return [ImageEntry(image_id=line, file_name=line) for line in lines], cartridge_cat_id

    if arg_str in SPLIT_NAMES:
        splits_file = Path(splits_path)
        if not splits_file.is_file():
            raise FileNotFoundError(
                f"Split name {arg_str!r} requested, but splits file not found: {splits_file}"
            )
        splits_data = json.loads(splits_file.read_text(encoding="utf-8"))
        image_splits = splits_data.get("images", {})
        selected = sorted(
            img_name
            for img_name, sp in image_splits.items()
            if (sp != "exclude" if arg_str == "all" else sp == arg_str)
        )
        return [ImageEntry(image_id=name, file_name=name) for name in selected], cartridge_cat_id

    raise FileNotFoundError(
        f"--images target {arg_str!r} is neither an existing file nor a split in {sorted(SPLIT_NAMES)}"
    )


def load_exemplars_from_coco(
    exemplars_coco: str | Path,
    images_root: str | Path,
    n_exemplars: int = 3,
) -> tuple[list[Image.Image], dict[str, list[list[float]]]]:
    """Load up to *n_exemplars* cropped PIL images and per-file XYXY boxes from COCO JSON."""
    if n_exemplars < 0:
        raise ValueError("n_exemplars must be non-negative")
    coco_path = Path(exemplars_coco)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    root = Path(images_root)

    crops: list[Image.Image] = []
    boxes_by_file: dict[str, list[list[float]]] = {}
    opened_images: dict[str, Image.Image] = {}

    try:
        for ann in coco.get("annotations", []):
            if n_exemplars and len(crops) >= n_exemplars:
                break
            img_meta = images_by_id.get(ann["image_id"])
            if img_meta is None:
                continue
            attrs = ann.get("attributes") or {}
            if attrs.get("truncated") is True:
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            file_name = str(img_meta["file_name"])
            if file_name not in opened_images:
                candidate = root / file_name
                if not candidate.is_file():
                    candidate = coco_path.parent / file_name
                with Image.open(candidate) as im:
                    opened_images[file_name] = im.convert("RGB")
            src_img = opened_images[file_name]
            xyxy = xywh_to_xyxy([[x, y, w, h]])[0].tolist()
            ex_crop = crop(src_img, xyxy, pad=0.0)
            crops.append(ex_crop)
            boxes_by_file.setdefault(file_name, []).append(xyxy)
    finally:
        for im in opened_images.values():
            im.close()

    return crops, boxes_by_file


def _supports_box_exemplars(backend: ProposalBackend) -> bool:
    sig = inspect.signature(backend.detect)
    return "box_exemplars" in sig.parameters


def _detect_tile(
    backend: ProposalBackend,
    tile_img: Image.Image,
    prompts: list[str],
    exemplars: list[Image.Image] | None,
    tile_box_exemplars: list[list[float]] | None,
) -> list[tuple[list[float], float, str]]:
    if tile_box_exemplars and _supports_box_exemplars(backend):
        return backend.detect(  # type: ignore[call-arg]
            tile_img, prompts, exemplars, box_exemplars=tile_box_exemplars
        )
    return backend.detect(tile_img, prompts, exemplars)


def propose_image(
    backend: ProposalBackend,
    image: Image.Image,
    prompts: list[str],
    exemplars: list[Image.Image] | None = None,
    *,
    tile: int = 1024,
    overlap: float = 0.25,
    score_thr: float = 0.05,
    merge: str = "nms",
    merge_iou: float = 0.5,
    edge_margin: int = 4,
    containment_threshold: float | None = None,
    cartridge_category_id: int = 1,
    image_box_exemplars: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Run tiled or full-frame (``tile <= 0``) detection and merge into image-space boxes."""
    if merge not in ("nms", "wbf"):
        raise ValueError(f"merge must be 'nms' or 'wbf', got {merge!r}")
    if not 0 <= score_thr <= 1 or not 0 <= merge_iou <= 1:
        raise ValueError("score_thr and merge_iou must be in [0, 1]")
    if edge_margin < 0:
        raise ValueError("edge_margin must be non-negative")

    rgb = image.convert("RGB")
    img_rect = (0, 0, rgb.width, rgb.height)
    if tile <= 0:
        rects = [img_rect]
        use_edge_drop = False
    else:
        rects = make_tiles(rgb.width, rgb.height, tile=tile, overlap=overlap)
        use_edge_drop = overlap > 0 and edge_margin > 0

    raw_dets: list[dict[str, Any]] = []
    for rect in rects:
        tile_img = rgb if rect == img_rect else rgb.crop(rect)
        tile_boxes: list[list[float]] | None = None
        if image_box_exemplars:
            mapped_ex = image_boxes_to_tile(image_box_exemplars, rect)
            tile_boxes = [
                b.tolist() for b in mapped_ex if b[2] > b[0] and b[3] > b[1]
            ] or None

        for xyxy, score, label in _detect_tile(backend, tile_img, prompts, exemplars, tile_boxes):
            score_f = float(score)
            if score_f < score_thr:
                continue
            mapped = tile_boxes_to_image([xyxy], rect)
            if use_edge_drop:
                mapped = drop_edge_truncated(mapped, rect, img_rect, margin=edge_margin)
                if len(mapped) == 0:
                    continue
            box = mapped[0].tolist()
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            raw_dets.append(
                {
                    "box": box,
                    "score": score_f,
                    "category_id": int(cartridge_category_id),
                    "label": str(label),
                }
            )

    return merge_tiled(
        raw_dets,
        iou=merge_iou,
        mode=merge,
        containment_threshold=containment_threshold,
    )


def _resolve_output_paths(out: str | Path) -> tuple[Path, Path, Path]:
    out_path = Path(out)
    if out_path.suffix.lower() == ".json":
        out_dir = out_path.parent
        pred_path = out_path
    else:
        out_dir = out_path
        pred_path = out_dir / "predictions.json"
    return out_dir, pred_path, out_dir / "meta.json"


def run_proposals(
    backend: ProposalBackend,
    images: str | Path,
    images_root: str | Path,
    out: str | Path,
    *,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    exemplars_coco: str | Path | None = None,
    n_exemplars: int = 3,
    tile: int = 1024,
    overlap: float = 0.25,
    score_thr: float = 0.05,
    merge: str = "nms",
    merge_iou: float = 0.5,
    edge_margin: int = 4,
    containment_threshold: float | None = None,
    splits: str | Path = "data/splits.json",
    experiment: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run proposal backend over resolved images and write ``predictions.json`` + ``meta.json``."""
    entries, cartridge_cat_id = resolve_images(images, splits_path=splits)
    root = Path(images_root)
    prompt_list = list(prompts)

    exemplar_crops: list[Image.Image] | None = None
    exemplar_boxes_by_file: dict[str, list[list[float]]] = {}
    if exemplars_coco is not None:
        exemplar_crops, exemplar_boxes_by_file = load_exemplars_from_coco(
            exemplars_coco, root, n_exemplars=n_exemplars
        )

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    for entry in entries:
        img_path = root / entry.file_name
        with Image.open(img_path) as im:
            if entry.width is not None and entry.height is not None:
                if im.size != (entry.width, entry.height):
                    raise ValueError(
                        f"Image dimensions differ from COCO metadata for {entry.file_name}: "
                        f"{im.size} != {(entry.width, entry.height)}"
                    )
            dets = propose_image(
                backend,
                im,
                prompt_list,
                exemplar_crops,
                tile=tile,
                overlap=overlap,
                score_thr=score_thr,
                merge=merge,
                merge_iou=merge_iou,
                edge_margin=edge_margin,
                containment_threshold=containment_threshold,
                cartridge_category_id=cartridge_cat_id,
                image_box_exemplars=exemplar_boxes_by_file.get(entry.file_name),
            )
        for det in dets:
            xywh = xyxy_to_xywh([det["box"]])[0].tolist()
            predictions.append(
                {
                    "image_id": entry.image_id,
                    "bbox": [float(v) for v in xywh],
                    "score": float(det["score"]),
                    "category_id": int(cartridge_cat_id),
                    "id_score": float(det["score"]),
                }
            )

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out_dir, pred_path, meta_path = _resolve_output_paths(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")

    hw_meta = _collect_hardware_meta()
    dims_seen = sorted(getattr(backend, "processed_dims_seen", set()))
    processed_dims = [[int(h), int(w)] for h, w in dims_seen]
    model_id = getattr(backend, "model_id", None)
    model_revision = getattr(backend, "resolved_revision", None) or getattr(
        backend, "revision", None
    )
    n_images = len(entries)
    meta: dict[str, Any] = {
        **hw_meta,
        "experiment": experiment or out_dir.name,
        "backend": getattr(backend, "name", type(backend).__name__),
        "model_id": model_id,
        "model_revision": model_revision,
        "processed_input_dims": processed_dims,
        "image_count": n_images,
        "prediction_count": len(predictions),
        "total_seconds": elapsed,
        "seconds_per_image": elapsed / n_images if n_images else None,
        "config": {
            "backend": getattr(backend, "name", type(backend).__name__),
            "model_id": model_id,
            "model_revision": model_revision,
            "images": str(images),
            "images_root": str(images_root),
            "splits": str(splits),
            "prompts": prompt_list,
            "exemplars_coco": str(exemplars_coco) if exemplars_coco is not None else None,
            "n_exemplars": n_exemplars if exemplars_coco is not None else 0,
            "tile": tile,
            "overlap": overlap,
            "score_thr": score_thr,
            "merge": merge,
            "merge_iou": merge_iou,
            "edge_margin": edge_margin,
            "containment_threshold": containment_threshold,
            "cartridge_category_id": cartridge_cat_id,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return predictions, meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    p.add_argument("--model-id", default=None, help="Optional Hugging Face model ID override.")
    p.add_argument("--revision", default=None, help="Optional Hugging Face revision / commit SHA.")
    p.add_argument(
        "--images",
        required=True,
        help="COCO JSON path, text list file (one relative path per line), or split name.",
    )
    p.add_argument("--images-root", required=True, help="Directory containing image files.")
    p.add_argument("--splits", default="data/splits.json", help="Splits JSON for split-name mode.")
    p.add_argument(
        "--prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Text concept prompts (pass empty --prompts for exemplar-only or <OD>).",
    )
    p.add_argument("--exemplars-coco", default=None, help="Optional COCO JSON for exemplar crops.")
    p.add_argument("--n-exemplars", type=int, default=3, help="Max exemplar crops to load.")
    p.add_argument("--tile", type=int, default=1024, help="Tile size in px (<=0 for full-frame).")
    p.add_argument("--overlap", type=float, default=0.25, help="Tile overlap fraction in [0, 1).")
    p.add_argument("--score-thr", type=float, default=0.05, help="Minimum proposal score.")
    p.add_argument("--merge", choices=("nms", "wbf"), default="nms", help="Tile merge algorithm.")
    p.add_argument("--merge-iou", type=float, default=0.5, help="IoU threshold for tile merging.")
    p.add_argument("--edge-margin", type=int, default=4, help="Internal tile-edge drop margin.")
    p.add_argument("--containment-threshold", type=float, default=None)
    p.add_argument(
        "--florence-task",
        choices=("auto", "<OD>", "<CAPTION_TO_PHRASE_GROUNDING>", "both"),
        default="auto",
        help="Task prompt mode when --backend florence2.",
    )
    p.add_argument("--device", default=None, help="Device override (e.g. cuda, cpu, auto).")
    p.add_argument("--experiment", default=None, help="Optional experiment name in meta.json.")
    p.add_argument("--out", required=True, help="Output directory (or predictions.json path).")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend_kwargs: dict[str, Any] = {
        "model_id": args.model_id,
        "revision": args.revision,
        "score_thr": args.score_thr,
        "device": args.device,
    }
    if args.backend == "florence2":
        backend_kwargs["task"] = args.florence_task
    backend = build_backend(args.backend, **backend_kwargs)
    run_proposals(
        backend,
        args.images,
        args.images_root,
        args.out,
        prompts=args.prompts,
        exemplars_coco=args.exemplars_coco,
        n_exemplars=args.n_exemplars,
        tile=args.tile,
        overlap=args.overlap,
        score_thr=args.score_thr,
        merge=args.merge,
        merge_iou=args.merge_iou,
        edge_margin=args.edge_margin,
        containment_threshold=args.containment_threshold,
        splits=args.splits,
        experiment=args.experiment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
