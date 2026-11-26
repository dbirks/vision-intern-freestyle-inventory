"""Instruction-prompted object detection using Vision Language Models.

Adapted from `uv-scripts/vlm-object-detection
<https://huggingface.co/datasets/uv-scripts/vlm-object-detection>`_.
Sends a free-form detection prompt to a VLM and parses ``bbox_2d`` JSON
from the response.  Coordinates are denormalised from the Qwen-VL 0-1000
scale to original-image pixel coordinates.

Supports two backends (see :mod:`tools.vlm_client`):

- ``"openai"``  — HF Inference Providers, vLLM server, llama-server
- ``"transformers"`` — local GPU with ``transformers``

CLI usage::

    python -m tools.vlm_detect image.jpg --classes "cat,dog"

    python -m tools.vlm_detect image.jpg --classes "cat,dog" \\
        --backend openai --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from typing import Any

from PIL import Image

from .model_options import QWEN_LABEL_MODELS
from .utils import load_image
from .vlm_client import run_vlm

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

DEFAULT_PROMPT = (
    "Detect every distinct object in the image. For each object, output a JSON "
    'object with keys: "bbox_2d" (an array [x1, y1, x2, y2] normalised to '
    '0-1000), "label" (category), and "sub_label" (short attribute or ""). '
    "Return a JSON array."
)


# ------------------------------------------------------------------
# Parsing  (battle-tested logic from qwen3vl-detect.py)
# ------------------------------------------------------------------

def parse_bboxes_from_response(text: str) -> list[dict[str, Any]]:
    """Extract bbox objects from VLM text.

    Tolerates fenced code blocks, trailing commas, comments, and
    malformed JSON.  Falls back to regex extraction when no parseable
    objects are found.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return []  # No final answer: do not label from reasoning examples.
    results: list[dict[str, Any]] = []

    pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[\d\s.,\-]+\][^{}]*\}'
    for match in re.findall(pattern, text, re.DOTALL):
        try:
            obj = json.loads(match)
            if "bbox_2d" in obj:
                results.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        try:
            cleaned = re.sub(r"#.*$", "", match, flags=re.MULTILINE).strip()
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*\]", "]", cleaned)
            obj = json.loads(cleaned)
            if "bbox_2d" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    if not results:
        bbox_pat = (
            r'"bbox_2d"\s*:\s*\[\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,'
            r"\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*\]"
        )
        for i, (x1, y1, x2, y2) in enumerate(re.findall(bbox_pat, text)):
            results.append({
                "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
                "label": f"object_{i + 1}",
                "sub_label": "",
            })

    if not results:
        generic = r'\{[^{}]*"(?:box|bbox)"\s*:\s*\[[\d\s.,\-]+\][^{}]*\}'
        for match in re.findall(generic, text, re.DOTALL):
            try:
                obj = json.loads(match)
                key = "box" if "box" in obj else "bbox"
                obj["bbox_2d"] = obj.pop(key)
                if "label" not in obj:
                    obj["label"] = "object"
                results.append(obj)
            except (json.JSONDecodeError, KeyError):
                continue

    return results


def denormalize_bbox(
    bbox: list[float], width: int, height: int,
) -> list[int]:
    """Convert a 0-1000 normalised bbox to original-image pixel coords."""
    if len(bbox) != 4:
        return []
    sx, sy = width / 1000.0, height / 1000.0
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(width, round(x1 * sx))),
        max(0, min(height, round(y1 * sy))),
        max(0, min(width, round(x2 * sx))),
        max(0, min(height, round(y2 * sy))),
    ]


def _is_edge_truncated(
    bbox: list[int],
    region: tuple[int, int, int, int],
    img_rect: tuple[int, int, int, int],
    margin: int = 4,
) -> bool:
    """Return True if *bbox* touches an internal edge of *region*."""
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = region
    ix1, iy1, ix2, iy2 = img_rect
    mx = min(margin, max(0, (rx2 - rx1) // 4))
    my = min(margin, max(0, (ry2 - ry1) // 4))
    return (
        (rx1 > ix1 and x1 <= rx1 + mx)
        or (ry1 > iy1 and y1 <= ry1 + my)
        or (rx2 < ix2 and x2 >= rx2 - mx)
        or (ry2 < iy2 and y2 >= ry2 - my)
    )


def _quadrants(
    region: tuple[int, int, int, int],
    tile_overlap: float,
) -> tuple[tuple[int, int, int, int], ...]:
    """Split *region* into four quadrants overlapping by *tile_overlap*."""
    x, y, right, bottom = region
    mx, my = (x + right) // 2, (y + bottom) // 2
    rw, rh = right - x, bottom - y
    pad_x = min(round(rw * tile_overlap / 2), max(0, right - mx - 1), max(0, mx - x - 1))
    pad_y = min(round(rh * tile_overlap / 2), max(0, bottom - my - 1), max(0, my - y - 1))
    return (
        (x, y, mx + pad_x, my + pad_y),
        (mx - pad_x, y, right, my + pad_y),
        (x, my - pad_y, mx + pad_x, bottom),
        (mx - pad_x, my - pad_y, right, bottom),
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def vlm_detect(
    image: str | Image.Image,
    prompt: str | None = None,
    classes: list[str] | None = None,
    model_id: str = DEFAULT_MODEL,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
    class_descriptions: dict[str, str] | None = None,
    max_dimension: int | None = 1280, image_format: str | None = None,
    jpeg_quality: int = 85, max_tokens: int = 4096,
    tile: int | None = None, tile_overlap: float = 0.2, n_samples: int = 1,
    temperature: float = 0, top_p: float = 1, seed: int | None = None,
    extra_body: dict | None = None, raw_response_log: str | None = None,
    label_aliases: dict[str, str] | None = None,
) -> list[dict]:
    """Detect objects in *image* using a VLM with a free-form prompt.

    Parameters
    ----------
    image : str or PIL.Image
        Path, URL, or PIL Image.
    prompt : str, optional
        Free-form detection instruction.  When *classes* is given the
        prompt is built automatically.
    classes : list[str], optional
        Object categories to look for.
    model_id : str
        Model identifier (HF repo id or server model name).
    backend : ``"openai"`` | ``"transformers"``
        Inference backend.  ``"openai"`` works with HF Inference
        Providers, vLLM, and llama-server.
    base_url : str, optional
        API endpoint for the ``openai`` backend.
    api_key : str, optional
        API key / HF token for the ``openai`` backend.
    class_descriptions : dict[str, str], optional
        Per-class definitions injected into the auto-built prompt to
        disambiguate look-alike categories (e.g. chart vs table) and curb
        common failure modes.  Ignored when an explicit *prompt* is given.

    temperature, top_p, seed
        Sampling controls. Sample i uses seed + i when a seed is supplied.
        Multiple samples require a nonzero temperature.
    extra_body : dict, optional
        Additional OpenAI-compatible request fields.
    raw_response_log : str or Path, optional
        Append each response and its finish reason and usage as JSONL.
    label_aliases : dict[str, str], optional
        Optional alias-to-class mapping normalized prior to cross-tile NMS
        and multi-sample consensus.

    Returns
    -------
    list[dict]
        Each dict contains ``bbox`` ([x1,y1,x2,y2] in pixels),
        ``label`` (str), and ``sub_label`` (str).
    """
    if n_samples > 1 and temperature == 0:
        raise ValueError("n_samples > 1 requires nonzero temperature for meaningful self-consistency")

    from .d1 import d1_detect, is_d1

    if is_d1(model_id):
        if n_samples != 1 or temperature != 0 or top_p != 1 or seed is not None or extra_body or raw_response_log:
            raise ValueError("d1 does not support VLM sampling options or raw response logging")
        if backend != "transformers":
            raise ValueError("d1 requires backend='transformers' and its System One API")
        if prompt is not None:
            raise ValueError("Use classes/class_descriptions for d1, or d1_decide for typed questions")
        return d1_detect(image, classes, model_id=model_id,
                         class_descriptions=class_descriptions)

    image = load_image(image)
    w, h = image.size

    if prompt is None:
        if classes:
            class_list = ", ".join(classes)
            if class_descriptions:
                defs = "\n".join(
                    f"- {c}: {class_descriptions[c]}"
                    for c in classes if c in class_descriptions
                )
                category_block = (
                    f"Detect every instance of these categories:\n{defs}\n"
                )
            else:
                category_block = (
                    f"Detect every instance of these categories: {class_list}. "
                )
            prompt = (
                f"{category_block}"
                "Draw each box tightly around the single element it refers to. "
                "Do NOT return a box covering the whole page or most of it, and "
                "do NOT box plain text, paragraphs, or headers. "
                "For each instance, return a JSON object with "
                '"bbox_2d" ([x1, y1, x2, y2] normalised 0-1000), '
                '"label" (one of the listed categories), '
                'and "sub_label" (short attribute or ""). '
                "Return a JSON array, or [] if none are present."
            )
        else:
            prompt = DEFAULT_PROMPT

    if tile is not None and tile <= 0:
        raise ValueError("tile must be positive")
    if not 0 <= tile_overlap < 1 or n_samples < 1:
        raise ValueError("tile_overlap must be in [0, 1) and n_samples positive")
    from .bbox_utils import (
        bbox_iou,
        build_label_resolver,
        nms,
        normalize_label_key,
    )

    label_resolver = (
        build_label_resolver(classes, label_aliases)
        if (classes is not None or label_aliases is not None)
        else None
    )
    img_rect = (0, 0, w, h)
    seen_surface: dict[str, str] = {}

    class DetectionList(list):
        truncated = False
        did_split = False

    def _canonicalize_label(raw_label: str, unify_unmatched: bool) -> str:
        if label_resolver is not None:
            resolved = label_resolver(raw_label)
            if resolved is not None:
                return resolved
        if unify_unmatched:
            return seen_surface.setdefault(normalize_label_key(raw_label), raw_label)
        return raw_label

    def detect_region(region: tuple[int, int, int, int], retry: bool,
                      sample_seed: int | None) -> DetectionList:
        x, y, right, bottom = region
        crop = image.crop(region)
        response = run_vlm(
            crop, prompt, model_id, backend=backend, base_url=base_url, api_key=api_key,
            max_tokens=max_tokens, max_dimension=max_dimension, image_format=image_format,
            jpeg_quality=jpeg_quality, temperature=temperature, top_p=top_p,
            seed=sample_seed, extra_body=extra_body, raw_response_log=raw_response_log,
        )
        truncated = getattr(response, "finish_reason", None) == "length"
        if truncated and retry and right - x > 1 and bottom - y > 1:
            result = DetectionList()
            result.did_split = True
            for box in _quadrants(region, tile_overlap):
                part = detect_region(box, False, sample_seed)
                result.extend(part)
                result.truncated |= part.truncated
            return result
        result = DetectionList()
        result.truncated = truncated
        if truncated:
            warnings.warn("VLM detection response was truncated", RuntimeWarning, stacklevel=2)
        for obj in parse_bboxes_from_response(response):
            raw = obj.get("bbox_2d", [])
            raw = [float(v) for v in raw] if isinstance(raw, list) and len(raw) == 4 else []
            bbox = denormalize_bbox(raw, right - x, bottom - y)
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            mapped = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
            if tile_overlap > 0 and _is_edge_truncated(mapped, region, img_rect):
                continue
            raw_label = str(obj.get("label") or "")
            unify = tile is not None or n_samples > 1 or not retry
            result.append({
                "bbox": mapped,
                "label": _canonicalize_label(raw_label, unify_unmatched=unify),
                "sub_label": str(obj.get("sub_label") or ""),
            })
        return result

    def starts(length: int) -> list[int]:
        if tile is None or length <= tile:
            return [0]
        step = max(1, round(tile * (1 - tile_overlap)))
        return list(range(0, length - tile, step)) + [length - tile]

    samples = []
    truncated = False
    for sample_index in range(n_samples):
        detections = []
        did_split = False
        for y in starts(h):
            for x in starts(w):
                part = detect_region(
                    (x, y, min(w, x + (tile or w)), min(h, y + (tile or h))),
                    True,
                    None if seed is None else seed + sample_index,
                )
                truncated |= part.truncated
                did_split |= part.did_split
                detections.extend(part)
        samples.append(nms(detections) if (tile is not None or did_split) else detections)
    result = DetectionList()
    result.truncated = truncated
    if n_samples == 1:
        result.extend(samples[0])
        return result
    clusters: list[dict] = []
    for sample_index, detections in enumerate(samples):
        for det in detections:
            match = next((c for c in clusters if sample_index not in c["samples"]
                          and c["det"]["label"].casefold() == det["label"].casefold()
                          and bbox_iou(c["det"]["bbox"], det["bbox"]) >= 0.5), None)
            if match is None:
                clusters.append({"det": det, "samples": {sample_index}})
            else:
                match["samples"].add(sample_index)
    result.extend({**c["det"], "score": len(c["samples"]) / n_samples} for c in clusters)
    return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM instruction-prompted object detection",
    )
    parser.add_argument("image", help="Image path or URL")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--classes", default=None, help="Comma-separated classes")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model ID; Qwen labeller options: " + ", ".join(QWEN_LABEL_MODELS))
    parser.add_argument("--backend", default="transformers",
                        choices=["openai", "transformers"])
    parser.add_argument("--base-url", default=None,
                        help="API base URL (openai backend)")
    parser.add_argument("--api-key", default=None,
                        help="API key / HF token (openai backend)")
    args = parser.parse_args()

    cls = [c.strip() for c in args.classes.split(",")] if args.classes else None
    dets = vlm_detect(
        args.image, prompt=args.prompt, classes=cls, model_id=args.model,
        backend=args.backend, base_url=args.base_url, api_key=args.api_key,
    )
    print(json.dumps(dets, indent=2))
