"""Auto-label images for object detection using a Vision Language Model.

Accepts **local image directories** or **Hugging Face datasets** as input.

Local mode::

    python -m workflows.vlm_label \\
        --source data/images --classes "cat,dog" --output annotations.json

Hub mode::

    python -m workflows.vlm_label \\
        --source merve/my-images --classes "cat,dog" \\
        --output merve/my-images-labeled --push-to-hub \\
        --backend openai --api-key hf_...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from tools.utils import load_image
from tools.vlm_detect import vlm_detect

DEFAULT_VLM = "Qwen/Qwen3.5-9B"
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _is_local_dir(source: str | Path) -> bool:
    return Path(source).is_dir()


def _normalize_detections(detections: list[dict], classes: list[str],
                          aliases: dict[str, str] | None, keep_unmatched: bool) -> list[dict]:
    def key(value: str) -> str:
        return re.sub(r"[^\w]+|_", "", value.casefold())
    canonical = {key(c): c for c in classes}
    for alias, target in (aliases or {}).items():
        if key(target) not in canonical:
            raise ValueError(f"Alias target {target!r} is not a class")
        canonical[key(alias)] = canonical[key(target)]
    out = []
    for det in detections:
        raw = det.get("label", "")
        label = canonical.get(key(raw))
        if label is None:
            if not keep_unmatched:
                continue
            label = "unmatched:" + raw
        out.append({**det, "label": label})
    return out


# ------------------------------------------------------------------
# Local mode  (image_dir → COCO JSON)
# ------------------------------------------------------------------

def _label_local(
    image_dir: Path,
    classes: list[str],
    output_path: Path,
    model_id: str,
    backend: str,
    base_url: str | None,
    api_key: str | None,
    detect_options: dict, aliases: dict | None, keep_unmatched: bool,
    recursive: bool, image_paths: list[str] | None, max_samples: int | None,
) -> dict:
    image_files = ([image_dir / p for p in image_paths] if image_paths is not None else
                   sorted(p for p in (image_dir.rglob("*") if recursive else image_dir.iterdir())
                          if p.is_file() and p.suffix.lower() in _IMAGE_EXT))
    if any(not p.resolve().is_relative_to(image_dir.resolve()) for p in image_files):
        raise ValueError("image_paths must stay within source")
    if max_samples is not None:
        image_files = image_files[:max_samples]
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    categories = [{"id": i, "name": n} for i, n in enumerate(classes)]
    name2id = {c["name"]: c["id"] for c in categories}

    coco: dict[str, list] = {
        "images": [], "annotations": [], "categories": categories,
    }
    ann_id = 0

    for img_id, path in enumerate(tqdm(image_files, desc="Labeling")):
        image = load_image(path)
        w, h = image.size

        detections = vlm_detect(
            image, classes=classes, model_id=model_id,
            backend=backend, base_url=base_url, api_key=api_key, **detect_options,
        )
        truncated = getattr(detections, "truncated", False)
        detections = _normalize_detections(detections, classes, aliases, keep_unmatched)

        coco["images"].append(
            {"id": img_id, "file_name": path.relative_to(image_dir).as_posix(), "width": w, "height": h,
             **({"truncated": True} if truncated else {})},
        )
        for det in detections:
            label = det["label"]
            if label not in name2id:
                name2id[label] = len(categories)
                categories.append({"id": name2id[label], "name": label})
            x1, y1, x2, y2 = det["bbox"]
            bw, bh = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": name2id[label],
                "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1),
                "iscrowd": 0,
                **({"score": det["score"]} if "score" in det else {}),
            })
            ann_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))
    print(f"Labeled {len(coco['images'])} images → {ann_id} annotations → {output_path}")
    return coco


# ------------------------------------------------------------------
# Hub mode  (HF dataset → HF dataset with detections column)
# ------------------------------------------------------------------

def _label_hub(
    dataset_id: str,
    classes: list[str],
    output_id: str,
    model_id: str,
    backend: str,
    base_url: str | None,
    api_key: str | None,
    image_column: str,
    split: str,
    max_samples: int | None,
    push_to_hub: bool,
    hf_token: str | None,
    dataset_config: str | None = None,
    dedupe: bool = False,
    dedupe_key_columns: list[str] | None = None,
    detect_options: dict | None = None, aliases: dict | None = None,
    keep_unmatched: bool = False,
) -> Any:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, name=dataset_config, split=split)

    if image_column not in ds.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {ds.column_names}"
        )

    # Collapse repeated images (e.g. VQA datasets with many rows per page) so we
    # don't label the same image several times.
    if dedupe:
        from tools.dataset_utils import dedupe_by_image
        before = len(ds)
        ds = dedupe_by_image(
            ds, image_column=image_column, key_columns=dedupe_key_columns,
        )
        print(f"Deduped {before} → {len(ds)} unique images "
              f"(key={dedupe_key_columns or 'image-hash'})")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))


    def _label_row(img: Any) -> list[dict]:
        if not isinstance(img, Image.Image):
            img = load_image(img)
        try:
            dets = vlm_detect(
                img, classes=classes, model_id=model_id,
                backend=backend, base_url=base_url, api_key=api_key, **(detect_options or {}),
            )
        except Exception as e:
            print(f"\n  [warn] vlm_detect failed, skipping: {e}")
            dets = []
        return _normalize_detections(dets, classes, aliases, keep_unmatched), getattr(dets, "truncated", False)

    # Router/openai calls are I/O-bound and the per-image router latency is the
    # long pole for large splits, so fan them out across threads (order
    # preserved by index). Local `transformers` inference stays sequential to
    # avoid GPU contention/OOM. Tune with LABEL_MAX_WORKERS (default 16 for the
    # router, 1 otherwise).
    import os as _os
    default_workers = 16 if backend == "openai" else 1
    max_workers = int(_os.environ.get("LABEL_MAX_WORKERS", default_workers))

    images = [ds[i][image_column] for i in range(len(ds))]
    if max_workers <= 1:
        all_detections = [
            _label_row(img) for img in tqdm(images, desc="Labeling", total=len(ds))
        ]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        all_detections = [None] * len(images)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_label_row, img): i for i, img in enumerate(images)}
            for fut in tqdm(
                as_completed(futures), desc="Labeling", total=len(images),
            ):
                all_detections[futures[fut]] = fut.result()

    ds = ds.add_column("detections", [row[0] for row in all_detections])
    if any(row[1] for row in all_detections):
        ds = ds.add_column("truncated", [row[1] for row in all_detections])

    if push_to_hub:
        from tools.hub_viz import push_dataset_with_viz
        push_dataset_with_viz(
            ds, output_id, token=hf_token, image_column=image_column,
            detections_column="detections",
        )
        print(f"Labeled {len(ds)} rows → https://huggingface.co/datasets/{output_id}")
    else:
        ds.save_to_disk(output_id)
        print(f"Labeled {len(ds)} rows → {output_id}")

    return ds


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def label_dataset(
    source: str | Path,
    classes: list[str],
    output: str | Path,
    model_id: str = DEFAULT_VLM,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    image_column: str = "image",
    split: str = "train",
    max_samples: int | None = None,
    push_to_hub: bool = False,
    hf_token: str | None = None,
    dataset_config: str | None = None,
    dedupe: bool = False,
    dedupe_key_columns: list[str] | None = None,
    max_dimension: int | None = 1280, image_format: str | None = None,
    jpeg_quality: int = 85, max_tokens: int = 4096, tile: int | None = None,
    tile_overlap: float = 0.2, n_samples: int = 1,
    temperature: float = 0, top_p: float = 1, seed: int | None = None,
    extra_body: dict | None = None, raw_response_log: str | Path | None = None,
    label_aliases: dict[str, str] | None = None, keep_unmatched: bool = False,
    recursive: bool = False, image_paths: list[str] | None = None,
) -> dict | Any:
    """Auto-label images and write detection annotations.

    Parameters
    ----------
    source : str or Path
        **Local directory** of images, or a **Hugging Face dataset ID**
        (e.g. ``"merve/my-images"``).  Detected automatically.
    classes : list[str]
        Object categories the VLM should look for.
    output : str or Path
        Local COCO JSON path (local mode) or HF dataset ID (Hub mode).
    model_id : str
        Model identifier.
    backend / base_url / api_key
        Inference backend (see :mod:`tools.vlm_client`).
    image_column : str
        Image column name in the HF dataset (Hub mode only).
    split : str
        Dataset split (Hub mode only).
    max_samples : int, optional
        Cap the number of images to process.
    push_to_hub : bool
        If ``True``, push the labeled dataset to the Hub.
    hf_token : str, optional
        Hugging Face token for pushing.
    dataset_config : str, optional
        Dataset configuration name (Hub mode only).  Required for
        multi-config datasets like ``lmms-lab/DocVQA``.
    dedupe : bool
        Collapse repeated images to one row before labelling (Hub mode only).
        Useful for VQA-style datasets with many rows per image.
    dedupe_key_columns : list[str], optional
        Column(s) identifying the same image for *dedupe* (e.g. ``["docId"]``).
        When omitted, dedupes by image content hash.

    temperature, top_p, seed
        Sampling controls; multiple samples require a nonzero temperature.
        When supplied, seed is incremented for each sample.
    extra_body : dict, optional
        Additional OpenAI-compatible request fields.
    raw_response_log : str or Path, optional
        Append each response with finish reason and usage as JSONL.

    Returns
    -------
    dict or datasets.Dataset
        COCO dict (local mode) or HF Dataset with ``detections``
        column (Hub mode).
    """
    if n_samples > 1 and temperature == 0:
        raise ValueError("n_samples > 1 requires nonzero temperature for meaningful self-consistency")
    classes_lower = [c.lower() for c in classes]
    detect_options = dict(max_dimension=max_dimension, image_format=image_format,
                          jpeg_quality=jpeg_quality, max_tokens=max_tokens, tile=tile,
                          tile_overlap=tile_overlap, n_samples=n_samples,
                          temperature=temperature, top_p=top_p, seed=seed,
                          extra_body=extra_body, raw_response_log=raw_response_log)

    if _is_local_dir(source):
        return _label_local(
            Path(source), classes_lower, Path(output),
            model_id, backend, base_url, api_key, detect_options, label_aliases,
            keep_unmatched, recursive, image_paths, max_samples,
        )
    return _label_hub(
        str(source), classes_lower, str(output),
        model_id, backend, base_url, api_key,
        image_column, split, max_samples, push_to_hub, hf_token,
        dataset_config=dataset_config,
        dedupe=dedupe, dedupe_key_columns=dedupe_key_columns,
        detect_options=detect_options, aliases=label_aliases, keep_unmatched=keep_unmatched,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Auto-label images with a VLM")
    parser.add_argument("--source", required=True,
                        help="Image directory or HF dataset ID")
    parser.add_argument("--classes", required=True,
                        help="Comma-separated object categories")
    parser.add_argument("--output", required=True,
                        help="COCO JSON path or HF dataset ID")
    parser.add_argument("--model", default=DEFAULT_VLM)
    parser.add_argument("--backend", default="transformers",
                        choices=["openai", "transformers"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dataset-config", default=None,
                        help="Dataset config name (multi-config datasets)")
    parser.add_argument("--dedupe", action="store_true",
                        help="Collapse repeated images to one row before "
                             "labelling (Hub mode).")
    parser.add_argument("--dedupe-key-columns", default=None,
                        help="Comma-separated column(s) identifying the same "
                             "image for --dedupe (default: image-content hash).")
    args = parser.parse_args()

    label_dataset(
        source=args.source,
        classes=[c.strip() for c in args.classes.split(",")],
        output=args.output,
        model_id=args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        image_column=args.image_column,
        split=args.split,
        max_samples=args.max_samples,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
        dataset_config=args.dataset_config,
        dedupe=args.dedupe,
        dedupe_key_columns=(
            [c.strip() for c in args.dedupe_key_columns.split(",") if c.strip()]
            if args.dedupe_key_columns else None
        ),
    )


if __name__ == "__main__":
    _cli()
