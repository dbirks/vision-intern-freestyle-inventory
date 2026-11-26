"""Label-efficient copy-paste & slot-shuffle synthesis for shelf cartridges."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from freestyle.augment.photometric import DEFAULT_PIPELINE, compose as compose_photometric
from freestyle.count.shelves import assign_shelves
from freestyle.geometry import crop_quad

RngLike = np.random.Generator | random.Random | int | None


def _coerce_rng(rng: RngLike) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, random.Random):
        return np.random.default_rng(rng.getrandbits(64))
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError(f"Unsupported rng type: {type(rng)!r}")


def _load_json(obj: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return json.loads(Path(obj).read_text())


def _resolve_split_lookup(split_map: dict[str, Any] | str | Path) -> dict[str, str]:
    data = _load_json(split_map)
    if "images" in data and isinstance(data["images"], dict):
        return {str(k): str(v) for k, v in data["images"].items()}
    return {str(k): str(v) for k, v in data.items()}


def build_bank(
    coco: dict[str, Any] | str | Path,
    images_root: str | Path,
    *,
    split_map: dict[str, Any] | str | Path | None = None,
    split: str | None = "train",
    allowed_image_ids: set[str | int] | list[str | int] | tuple[str | int, ...] | None = None,
    exclude_unknown: bool = False,
    exclude_synthetic: bool = True,
    min_size: int = 1,
) -> list[dict[str, Any]]:
    """Extract per-instance crops (+ optional quad) from TRAIN-split real annotations only.

    Refuses to run without an explicit ``split_map`` or ``allowed_image_ids``
    filter, and refuses any split other than ``'train'`` to prevent test leakage.
    """
    if split_map is None and allowed_image_ids is None:
        raise ValueError(
            "build_bank requires a split filter (split_map or allowed_image_ids) "
            "to prevent test leakage."
        )
    if split != "train":
        raise ValueError(
            f"build_bank only extracts from split='train' real annotations, got {split!r}."
        )

    coco_data = _load_json(coco)
    root = Path(images_root)
    split_lookup = _resolve_split_lookup(split_map) if split_map is not None else None
    allowed_set = set(allowed_image_ids) if allowed_image_ids is not None else None

    cat_names = {int(c["id"]): c.get("name", str(c["id"])) for c in coco_data.get("categories", [])}
    unknown_ids = {
        int(c["id"]) for c in coco_data.get("categories", []) if c.get("name") == "unknown"
    }

    anns_by_img: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    bank: list[dict[str, Any]] = []
    for img_rec in coco_data.get("images", []):
        file_name = img_rec["file_name"]
        img_id = img_rec["id"]

        if split_lookup is not None and split_lookup.get(file_name) != "train":
            continue
        if allowed_set is not None and file_name not in allowed_set and img_id not in allowed_set:
            continue

        img_anns = anns_by_img.get(img_id, [])
        if not img_anns:
            continue

        with Image.open(root / file_name) as im:
            rgb = im.convert("RGB")
            iw, ih = rgb.size
            for ann in img_anns:
                attrs = ann.get("attributes") or {}
                source = str(attrs.get("source", ""))
                if exclude_synthetic and source.startswith("synthetic"):
                    continue
                cat_id = int(ann["category_id"])
                if exclude_unknown and cat_id in unknown_ids:
                    continue

                x, y, w, h = map(float, ann["bbox"])
                if w < min_size or h < min_size:
                    continue
                x0 = max(0, math.floor(x))
                y0 = max(0, math.floor(y))
                x1 = min(iw, math.ceil(x + w))
                y1 = min(ih, math.ceil(y + h))
                if x1 - x0 < min_size or y1 - y0 < min_size:
                    continue

                patch = rgb.crop((x0, y0, x1, y1))
                quad = ann.get("quad")
                quad_image = None
                if quad is not None:
                    try:
                        quad_image = crop_quad(rgb, quad)
                    except ValueError:
                        quad_image = None

                bank.append(
                    {
                        "image": patch,
                        "quad_image": quad_image,
                        "quad": [list(pt) for pt in quad] if quad is not None else None,
                        "bbox": [x, y, w, h],
                        "width": float(x1 - x0),
                        "height": float(y1 - y0),
                        "aspect": float(x1 - x0) / max(float(y1 - y0), 1.0),
                        "category_id": cat_id,
                        "category_name": cat_names.get(cat_id, str(cat_id)),
                        "source_image_id": img_id,
                        "source_file_name": file_name,
                        "source_ann_id": ann.get("id"),
                        "attributes": dict(attrs),
                    }
                )
    return bank


# ---------------------------------------------------------------------------
# Pure-numpy Poisson & feathered seam blending
# ---------------------------------------------------------------------------


def _dst1(x: np.ndarray, axis: int) -> np.ndarray:
    """1D Type-I Discrete Sine Transform via ``np.fft.rfft`` (no scipy needed)."""
    n = x.shape[axis]
    if n <= 0:
        return x
    x_moved = np.moveaxis(x, axis, -1)
    pad_shape = (*x_moved.shape[:-1], 2 * (n + 1))
    buf = np.zeros(pad_shape, dtype=np.float64)
    buf[..., 1 : n + 1] = x_moved
    buf[..., n + 2 :] = -x_moved[..., ::-1]
    spec = np.fft.rfft(buf, axis=-1)
    out = -0.5 * spec[..., 1 : n + 1].imag
    return np.moveaxis(out, -1, axis)


def _idst1(x: np.ndarray, axis: int) -> np.ndarray:
    n = x.shape[axis]
    if n <= 0:
        return x
    return _dst1(x, axis) * (2.0 / (n + 1))


def poisson_blend(bg_patch: np.ndarray, fg_patch: np.ndarray) -> np.ndarray:
    """Exact 2D Dirichlet Poisson seamless cloning on a rectangular patch (pure numpy)."""
    h, w = bg_patch.shape[:2]
    if h < 3 or w < 3:
        return fg_patch.copy()

    bg = bg_patch.astype(np.float64)
    fg = fg_patch.astype(np.float64)

    # Interior discrete Laplacian of the foreground patch.
    lap = (
        fg[:-2, 1:-1]
        + fg[2:, 1:-1]
        + fg[1:-1, :-2]
        + fg[1:-1, 2:]
        - 4.0 * fg[1:-1, 1:-1]
    )

    rhs = lap.copy()
    rhs[0, :] -= bg[0, 1:-1]
    rhs[-1, :] -= bg[-1, 1:-1]
    rhs[:, 0] -= bg[1:-1, 0]
    rhs[:, -1] -= bg[1:-1, -1]

    ih, iw = h - 2, w - 2
    eig_y = 2.0 * np.cos(math.pi * np.arange(1, ih + 1) / (ih + 1)) - 2.0
    eig_x = 2.0 * np.cos(math.pi * np.arange(1, iw + 1) / (iw + 1)) - 2.0
    denom = eig_y[:, None] + eig_x[None, :]
    if rhs.ndim == 3:
        denom = denom[..., None]

    freq = _dst1(_dst1(rhs, axis=0), axis=1)
    solved = _idst1(_idst1(freq / denom, axis=0), axis=1)

    out = bg.copy()
    out[1:-1, 1:-1] = np.clip(solved, 0.0, 255.0)
    return out.astype(np.float32)


def _feather_alpha(mask: np.ndarray, feather_px: int = 3) -> np.ndarray:
    """Smooth interior seam alpha in [0, 1] without leaking outside ``mask``."""
    h, w = mask.shape
    if feather_px <= 0 or h < 3 or w < 3:
        return mask.astype(np.float32)

    mask_u8 = np.clip(mask * 255.0, 0.0, 255.0).round().astype(np.uint8)
    pil_m = Image.fromarray(mask_u8, mode="L")
    # Erode by 1px before blur so boundary stays strictly inside valid pixels.
    eroded = pil_m.filter(ImageFilter.MinFilter(3))
    blurred = eroded.filter(ImageFilter.GaussianBlur(radius=max(0.8, feather_px * 0.6)))
    alpha = np.asarray(blurred, dtype=np.float32) / 255.0
    return np.minimum(alpha, mask.astype(np.float32))


def blend_patch(
    canvas: np.ndarray,
    patch_rgb: np.ndarray,
    mask: np.ndarray,
    x0: int,
    y0: int,
    *,
    mode: str = "hybrid",
    feather_px: int = 3,
) -> None:
    """Blend ``patch_rgb`` into ``canvas`` in-place at ``(x0, y0)``."""
    ph, pw = patch_rgb.shape[:2]
    bg = canvas[y0 : y0 + ph, x0 : x0 + pw].astype(np.float32)
    fg = patch_rgb.astype(np.float32)

    if mode == "feather":
        composed = fg
    elif mode == "poisson":
        composed = poisson_blend(bg, fg)
    elif mode == "hybrid":
        # Keep true cartridge colours in interior while adapting boundary lighting.
        poiss = poisson_blend(bg, fg)
        composed = 0.30 * poiss + 0.70 * fg
    else:
        raise ValueError(f"blend_mode must be 'hybrid', 'poisson', or 'feather', got {mode!r}")

    alpha = _feather_alpha(mask, feather_px=feather_px)[..., None]
    canvas[y0 : y0 + ph, x0 : x0 + pw] = np.clip(
        alpha * composed + (1.0 - alpha) * bg, 0.0, 255.0
    ).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometry & slot helpers
# ---------------------------------------------------------------------------


def _slot_from_item(item: Any) -> dict[str, Any]:
    """Normalize a slot dict or box into ``{bbox: [x,y,w,h], ...}``."""
    if isinstance(item, dict):
        if "bbox" in item:
            x, y, w, h = map(float, item["bbox"])
        elif "box" in item:
            x0, y0, x1, y1 = map(float, item["box"])
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
        else:
            raise ValueError(f"Slot dict missing 'bbox' or 'box': {item!r}")
        slot = dict(item)
        slot["bbox"] = [x, y, w, h]
        return slot
    vals = [float(v) for v in item]
    if len(vals) != 4:
        raise ValueError(f"Slot box must have 4 elements, got {item!r}")
    return {"bbox": vals}


def _group_flat_slots(slots: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group a flat list of slots into shelf rows ordered top-to-bottom, left-to-right."""
    if not slots:
        return []
    if all("shelf_idx" in s for s in slots):
        by_row: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for s in slots:
            key = (s["shelf_idx"], s.get("tier", 0))
            by_row[key].append(s)
        return [
            sorted(by_row[k], key=lambda s: s["bbox"][0])
            for k in sorted(by_row)
        ]
    try:
        xyxy = [
            {"box": [s["bbox"][0], s["bbox"][1], s["bbox"][0] + s["bbox"][2], s["bbox"][1] + s["bbox"][3]]}
            for s in slots
        ]
        assigned = assign_shelves(xyxy)
        by_row = defaultdict(list)
        for s, a in zip(slots, assigned, strict=True):
            by_row[(a["shelf_idx"], a["tier"])].append(s)
        return [sorted(by_row[k], key=lambda s: s["bbox"][0]) for k in sorted(by_row)]
    except Exception:
        # Fallback: simple vertical band clustering by y-center.
        med_h = float(np.median([s["bbox"][3] for s in slots])) if slots else 10.0
        order = sorted(slots, key=lambda s: s["bbox"][1] + 0.5 * s["bbox"][3])
        rows: list[list[dict[str, Any]]] = []
        row_y: list[float] = []
        for s in order:
            cy = s["bbox"][1] + 0.5 * s["bbox"][3]
            if not rows or abs(cy - row_y[-1]) > 0.6 * med_h:
                rows.append([s])
                row_y.append(cy)
            else:
                rows[-1].append(s)
                row_y[-1] = float(np.mean([r["bbox"][1] + 0.5 * r["bbox"][3] for r in rows[-1]]))
        return [sorted(r, key=lambda s: s["bbox"][0]) for r in rows]


def _normalize_rows(row_lines: Any) -> list[list[dict[str, Any]]]:
    """Accept grouped rows, flat annotation lists, or row dicts with 'slots'."""
    if not row_lines:
        return []
    if isinstance(row_lines, dict) and "rows" in row_lines:
        row_lines = row_lines["rows"]

    first = row_lines[0]
    if isinstance(first, list):
        return [[_slot_from_item(it) for it in row] for row in row_lines if row]
    if isinstance(first, dict) and "slots" in first:
        return [[_slot_from_item(it) for it in row["slots"]] for row in row_lines if row.get("slots")]
    return _group_flat_slots([_slot_from_item(it) for it in row_lines])


def _homography_coeffs(dst_pts: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    """Solve 8-DOF perspective coefficients mapping ``dst_pts`` -> ``src_pts``."""
    matrix, rhs = [], []
    for (x, y), (u, v) in zip(dst_pts, src_pts, strict=True):
        matrix.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
        rhs.extend([u, v])
    return np.linalg.solve(matrix, rhs)


def _warp_crop_to_slot(
    crop_img: Image.Image,
    slot_bbox: list[float],
    img_size: tuple[int, int],
    gen: np.random.Generator,
    *,
    perspective_jitter: float = 0.0,
    target_quad: list[list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[float], list[list[float]] | None]:
    """Scale and optionally perspective-jitter a crop to fit a target slot."""
    iw, ih = img_size
    tx, ty, tw, th = slot_bbox

    if target_quad is not None and len(target_quad) == 4:
        corners = np.asarray(target_quad, dtype=np.float64)
    else:
        corners = np.array(
            [[tx, ty], [tx + tw, ty], [tx + tw, ty + th], [tx, ty + th]],
            dtype=np.float64,
        )

    if perspective_jitter > 0.0:
        max_shift = min(tw, th) * perspective_jitter
        jitter = gen.uniform(-max_shift, max_shift, size=(4, 2))
        corners = corners + jitter
        corners[:, 0] = np.clip(corners[:, 0], 0.0, float(iw))
        corners[:, 1] = np.clip(corners[:, 1], 0.0, float(ih))

    min_x = float(np.clip(corners[:, 0].min(), 0.0, float(iw)))
    min_y = float(np.clip(corners[:, 1].min(), 0.0, float(ih)))
    max_x = float(np.clip(corners[:, 0].max(), 0.0, float(iw)))
    max_y = float(np.clip(corners[:, 1].max(), 0.0, float(ih)))

    x0 = max(0, math.floor(min_x))
    y0 = max(0, math.floor(min_y))
    x1 = min(iw, math.ceil(max_x))
    y1 = min(ih, math.ceil(max_y))
    pw = max(1, x1 - x0)
    ph = max(1, y1 - y0)

    cw, ch = crop_img.size
    if perspective_jitter <= 0.0 and target_quad is None:
        resized = crop_img.resize((pw, ph), Image.Resampling.BICUBIC)
        patch_rgb = np.asarray(resized, dtype=np.uint8)
        mask = np.ones((ph, pw), dtype=np.float32)
        out_bbox = [min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)]
        return patch_rgb, mask, (x0, y0), out_bbox, None

    src_pts = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float64)
    dst_pts = corners - np.array([x0, y0], dtype=np.float64)
    # Keep corners cleanly ordered & non-degenerate inside patch.
    try:
        coeffs = _homography_coeffs(dst_pts, src_pts)
        warped_rgb = crop_img.transform(
            (pw, ph), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC
        )
        white = Image.new("L", (cw, ch), 255)
        warped_mask = white.transform(
            (pw, ph), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR
        )
        patch_rgb = np.asarray(warped_rgb, dtype=np.uint8)
        mask = (np.asarray(warped_mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    except np.linalg.LinAlgError:
        resized = crop_img.resize((pw, ph), Image.Resampling.BICUBIC)
        patch_rgb = np.asarray(resized, dtype=np.uint8)
        mask = np.ones((ph, pw), dtype=np.float32)

    out_bbox = [min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)]
    return patch_rgb, mask, (x0, y0), out_bbox, corners.tolist()


# ---------------------------------------------------------------------------
# Synthesis & slot shuffle
# ---------------------------------------------------------------------------


def _choose_bank_entry(
    bank: list[dict[str, Any]],
    by_cat: dict[int, list[dict[str, Any]]],
    categories: list[int],
    tw: float,
    th: float,
    gen: np.random.Generator,
    *,
    class_balance: bool,
) -> dict[str, Any]:
    """Sample a bank crop matching slot dimensions, oversampling rare classes if enabled."""
    if class_balance and categories:
        cat_id = categories[int(gen.integers(0, len(categories)))]
        pool = by_cat[cat_id]
    else:
        pool = bank

    target_aspect = tw / max(th, 1e-6)
    target_area = max(tw * th, 1.0)
    scores = np.empty(len(pool), dtype=np.float64)
    for i, entry in enumerate(pool):
        ew, eh = entry["width"], entry["height"]
        aspect_err = abs(math.log((ew / max(eh, 1e-6)) / max(target_aspect, 1e-6)))
        area_err = abs(math.log(max(ew * eh, 1.0) / target_area))
        scores[i] = math.exp(-(2.5 * aspect_err + 1.0 * area_err))

    total = float(scores.sum())
    if total <= 0.0 or not math.isfinite(total):
        return pool[int(gen.integers(0, len(pool)))]
    probs = scores / total
    return pool[int(gen.choice(len(pool), p=probs))]


def synthesize(
    background_img: Image.Image,
    row_lines: Any,
    bank: list[dict[str, Any]],
    n: int,
    rng: RngLike = None,
    class_balance: bool = True,
    *,
    replace_prob: float = 1.0,
    perspective_jitter: float = 0.03,
    blend_mode: str = "hybrid",
    feather_px: int = 3,
) -> list[tuple[Image.Image, list[dict[str, Any]]]]:
    """Compose ``n`` synthetic shelf images by replacing cartridges along ``row_lines``.

    Returns a list of ``n`` ``(image, annotations)`` tuples. Pasted instances
    emit COCO annotations in original pixel coordinates with
    ``attributes["source"] = "synthetic:copy_paste"``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not bank:
        raise ValueError("bank must be non-empty")

    gen = _coerce_rng(rng)
    rows = _normalize_rows(row_lines)
    orig_mode = background_img.mode
    base_rgb = np.asarray(background_img.convert("RGB"), dtype=np.uint8)
    iw, ih = background_img.size

    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in bank:
        by_cat[int(entry["category_id"])].append(entry)
    categories = sorted(by_cat)

    results: list[tuple[Image.Image, list[dict[str, Any]]]] = []
    for _ in range(n):
        canvas = base_rgb.copy()
        anns: list[dict[str, Any]] = []
        ann_idx = 0

        for row in rows:
            for slot in row:
                sx, sy, sw, sh = slot["bbox"]
                if sw <= 0 or sh <= 0:
                    continue
                if replace_prob < 1.0 and float(gen.random()) > replace_prob and "category_id" in slot:
                    ann_idx += 1
                    attrs = dict(slot.get("attributes") or {})
                    attrs.setdefault("truncated", False)
                    attrs.setdefault("occluded", False)
                    attrs.setdefault("identity_visible", True)
                    attrs.setdefault("source", "human")
                    kept = {
                        "id": ann_idx,
                        "category_id": int(slot["category_id"]),
                        "bbox": [float(sx), float(sy), float(sw), float(sh)],
                        "area": float(sw * sh),
                        "iscrowd": int(slot.get("iscrowd", 0)),
                        "attributes": attrs,
                    }
                    if slot.get("quad") is not None:
                        kept["quad"] = slot["quad"]
                    anns.append(kept)
                    continue

                entry = _choose_bank_entry(
                    bank, by_cat, categories, sw, sh, gen, class_balance=class_balance
                )
                crop_src = (
                    entry["quad_image"]
                    if slot.get("quad") is not None and entry.get("quad_image") is not None
                    else entry["image"]
                )
                patch_rgb, mask, (x0, y0), out_bbox, out_quad = _warp_crop_to_slot(
                    crop_src,
                    [sx, sy, sw, sh],
                    (iw, ih),
                    gen,
                    perspective_jitter=perspective_jitter,
                    target_quad=slot.get("quad"),
                )
                if out_bbox[2] <= 0 or out_bbox[3] <= 0:
                    continue
                blend_patch(
                    canvas,
                    patch_rgb,
                    mask,
                    x0,
                    y0,
                    mode=blend_mode,
                    feather_px=feather_px,
                )

                ann_idx += 1
                entry_attrs = entry.get("attributes") or {}
                attrs = {
                    "truncated": bool(entry_attrs.get("truncated", False)),
                    "occluded": bool(entry_attrs.get("occluded", False)),
                    "identity_visible": bool(entry_attrs.get("identity_visible", True)),
                    "source": "synthetic:copy_paste",
                }
                rounded_bbox = [round(v, 2) for v in out_bbox]
                ann: dict[str, Any] = {
                    "id": ann_idx,
                    "category_id": int(entry["category_id"]),
                    "bbox": rounded_bbox,
                    "area": round(rounded_bbox[2] * rounded_bbox[3], 2),
                    "iscrowd": 0,
                    "attributes": attrs,
                }
                if out_quad is not None:
                    ann["quad"] = [[round(c[0], 2), round(c[1], 2)] for c in out_quad]
                anns.append(ann)

        out_img = Image.fromarray(canvas, mode="RGB")
        if orig_mode != "RGB":
            out_img = out_img.convert(orig_mode)
        results.append((out_img, anns))

    return results


def _same_size_clusters(slots: list[dict[str, Any]], size_tol: float) -> list[list[int]]:
    """Cluster slot indices within a row into same-size compatibility buckets."""
    clusters: list[list[int]] = []
    reps: list[tuple[float, float]] = []
    for idx, slot in enumerate(slots):
        w, h = slot["bbox"][2], slot["bbox"][3]
        if w <= 0 or h <= 0:
            continue
        placed = False
        for c_idx, (rw, rh) in enumerate(reps):
            w_ratio = max(w / rw, rw / w)
            h_ratio = max(h / rh, rh / h)
            if w_ratio <= 1.0 + size_tol and h_ratio <= 1.0 + size_tol:
                clusters[c_idx].append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
            reps.append((w, h))
    return clusters


def slot_shuffle(
    img: Image.Image,
    anns: list[dict[str, Any]],
    rng: RngLike = None,
    *,
    row_lines: Any = None,
    size_tol: float = 0.25,
    blend_mode: str = "feather",
    feather_px: int = 2,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Permute identities and pixels of same-size slots within each shelf row.

    Breaks memorization of slot position -> cartridge identity while preserving
    exact per-row category counts and spatial geometry.
    """
    if not anns:
        return img.copy(), []

    gen = _coerce_rng(rng)
    orig_mode = img.mode
    rgb = img.convert("RGB")
    iw, ih = rgb.size
    canvas = np.asarray(rgb, dtype=np.uint8).copy()

    indexed_anns = []
    for idx, ann in enumerate(anns):
        copy = dict(ann)
        copy["_orig_idx"] = idx
        indexed_anns.append(copy)

    rows = _normalize_rows(row_lines if row_lines is not None else indexed_anns)

    # Pre-extract all source crops from the original image before overwriting canvas.
    extracted: dict[int, Image.Image] = {}
    for row in rows:
        for slot in row:
            x, y, w, h = slot["bbox"]
            x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
            x1, y1 = min(iw, math.ceil(x + w)), min(ih, math.ceil(y + h))
            if x1 > x0 and y1 > y0:
                extracted[slot["_orig_idx"]] = rgb.crop((x0, y0, x1, y1))

    out_anns: list[dict[str, Any] | None] = [None] * len(anns)
    for row in rows:
        for group in _same_size_clusters(row, size_tol=size_tol):
            if len(group) == 1:
                slot = row[group[0]]
                orig_idx = slot["_orig_idx"]
                clean = {k: v for k, v in slot.items() if k != "_orig_idx"}
                out_anns[orig_idx] = clean
                continue

            perm = gen.permutation(len(group))
            for pos, src_pos in enumerate(perm.tolist()):
                dst_slot = row[group[pos]]
                src_slot = row[group[src_pos]]
                dst_idx = dst_slot["_orig_idx"]
                src_idx = src_slot["_orig_idx"]

                if dst_idx != src_idx and src_idx in extracted:
                    dx, dy, dw, dh = dst_slot["bbox"]
                    x0, y0 = max(0, math.floor(dx)), max(0, math.floor(dy))
                    x1, y1 = min(iw, math.ceil(dx + dw)), min(ih, math.ceil(dy + dh))
                    pw, ph = x1 - x0, y1 - y0
                    if pw > 0 and ph > 0:
                        resized = extracted[src_idx].resize((pw, ph), Image.Resampling.BICUBIC)
                        patch_rgb = np.asarray(resized, dtype=np.uint8)
                        mask = np.ones((ph, pw), dtype=np.float32)
                        blend_patch(
                            canvas, patch_rgb, mask, x0, y0, mode=blend_mode, feather_px=feather_px
                        )

                new_ann = {k: v for k, v in dst_slot.items() if k != "_orig_idx"}
                new_ann["category_id"] = src_slot["category_id"]
                src_attrs = dict(src_slot.get("attributes") or {})
                src_attrs["source"] = (
                    "synthetic:slot_shuffle" if dst_idx != src_idx else src_attrs.get("source", "human")
                )
                new_ann["attributes"] = src_attrs
                out_anns[dst_idx] = new_ann

    # Preserve any annotation not assigned to a row untouched.
    final_anns: list[dict[str, Any]] = []
    for idx, item in enumerate(out_anns):
        if item is not None:
            final_anns.append(item)
        else:
            final_anns.append(dict(anns[idx]))

    out_img = Image.fromarray(canvas, mode="RGB")
    if orig_mode != "RGB":
        out_img = out_img.convert(orig_mode)
    return out_img, final_anns


# ---------------------------------------------------------------------------
# Contact sheet + CLI
# ---------------------------------------------------------------------------


def make_contact_sheet(
    records: list[tuple[Path, list[dict[str, Any]]]],
    out_path: Path,
    *,
    max_items: int = 16,
    thumb: int = 320,
) -> Path:
    """Render a grid preview of synthetic frames with overlaid boxes."""
    sample = records[:max_items]
    if not sample:
        sheet = Image.new("RGB", (thumb, 64), "white")
        ImageDraw.Draw(sheet).text((8, 24), "No synthetic images generated", fill="black")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path)
        return out_path

    cols = min(4, len(sample))
    rows_n = math.ceil(len(sample) / cols)
    pad, cap_h = 8, 22
    sheet = Image.new(
        "RGB", (cols * (thumb + pad) + pad, rows_n * (thumb + cap_h + pad) + 28), "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 6), f"Synthetic copy-paste ({len(records)} total, showing {len(sample)})", fill="black")

    for i, (img_path, anns) in enumerate(sample):
        with Image.open(img_path) as im:
            vis = im.convert("RGB")
            iw, ih = vis.size
            vis.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            sx, sy = vis.width / max(iw, 1), vis.height / max(ih, 1)
            vd = ImageDraw.Draw(vis)
            for ann in anns:
                bx, by, bw, bh = ann["bbox"]
                vd.rectangle(
                    [bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy],
                    outline=(0, 220, 80),
                    width=1,
                )
            x = pad + (i % cols) * (thumb + pad)
            y = 28 + (i // cols) * (thumb + cap_h + pad)
            sheet.paste(vis, (x + (thumb - vis.width) // 2, y))
            draw.text((x, y + thumb + 3), f"{img_path.name} ({len(anns)} boxes)"[:42], fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def run_cli(
    *,
    coco_path: str | Path,
    split_map_path: str | Path,
    split: str,
    images_root: str | Path,
    out_dir: str | Path,
    n_per_image: int = 20,
    seed: int = 0,
    class_balance: bool = True,
    blend_mode: str = "hybrid",
    perspective_jitter: float = 0.03,
    slot_shuffle_prob: float = 0.0,
    apply_photometric: bool = False,
) -> dict[str, Any]:
    """Generate synthetic COCO dataset (PNGs + annotations.json + contact sheet)."""
    if split != "train":
        raise ValueError(f"CLI refuses non-train split {split!r} to avoid test leakage.")

    coco = _load_json(coco_path)
    splits = _resolve_split_lookup(split_map_path)
    root = Path(images_root)
    out = Path(out_dir)
    img_out_dir = out / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    bank = build_bank(coco, root, split_map=splits, split=split)
    if not bank:
        raise ValueError("No train annotations found to build copy-paste bank.")

    anns_by_img: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        source = str((ann.get("attributes") or {}).get("source", ""))
        if source.startswith("synthetic"):
            continue
        anns_by_img[ann["image_id"]].append(ann)

    gen = np.random.default_rng(seed)
    photo_fn = compose_photometric({k: v for k, v in DEFAULT_PIPELINE.items() if k != "fisheye_warp"}) if apply_photometric else None
    from freestyle.eval.metrics import load_groups_map
    split_data = _load_json(split_map_path)
    source_images = [im for im in coco.get("images", []) if splits.get(im["file_name"]) == "train"]
    parent_groups = load_groups_map({"images": source_images}, split_data)
    group_by_image = {im["id"]: parent_groups[im["file_name"]] for im in source_images}
    synth_groups = {}

    out_images: list[dict[str, Any]] = []
    out_annotations: list[dict[str, Any]] = []
    synth_splits: dict[str, str] = {}
    sheet_records: list[tuple[Path, list[dict[str, Any]]]] = []
    next_img_id = 0
    next_ann_id = 0

    for src_img in coco.get("images", []):
        if splits.get(src_img["file_name"]) != "train":
            continue
        src_anns = anns_by_img.get(src_img["id"], [])
        if not src_anns:
            continue

        with Image.open(root / src_img["file_name"]) as bg:
            bg_rgb = bg.convert("RGB")
            generated = synthesize(
                bg_rgb,
                src_anns,
                bank,
                n_per_image,
                gen,
                class_balance=class_balance,
                perspective_jitter=perspective_jitter,
                blend_mode=blend_mode,
            )

        stem = Path(src_img["file_name"]).stem
        for k, (synth_img, synth_anns) in enumerate(generated):
            if slot_shuffle_prob > 0.0 and float(gen.random()) < slot_shuffle_prob:
                synth_img, synth_anns = slot_shuffle(synth_img, synth_anns, gen)
            if photo_fn is not None:
                synth_img = photo_fn(synth_img, gen)

            next_img_id += 1
            rel_name = f"images/{stem}_synth_{next_img_id:06d}_{k:03d}.png"
            dst_path = out / rel_name
            synth_img.save(dst_path, format="PNG")
            synth_splits[rel_name] = "train"
            group = "synthetic:" + group_by_image[src_img["id"]]
            synth_groups[group] = "train"
            out_images.append(
                {
                    "id": next_img_id,
                    "file_name": rel_name,
                    "width": synth_img.width,
                    "height": synth_img.height,
                    "attributes": {"source": "synthetic:copy_paste", "parent": src_img["file_name"], "group_id": group},
                }
            )
            frame_anns: list[dict[str, Any]] = []
            for ann in synth_anns:
                next_ann_id += 1
                rec = dict(ann)
                rec["id"] = next_ann_id
                rec["image_id"] = next_img_id
                out_annotations.append(rec)
                frame_anns.append(rec)
            sheet_records.append((dst_path, frame_anns))

    out_coco = {
        "info": {"description": "Synthetic copy-paste Freestyle shelf dataset"},
        "licenses": [],
        "categories": coco.get("categories", []),
        "images": out_images,
        "annotations": out_annotations,
    }
    (out / "annotations.json").write_text(json.dumps(out_coco, indent=2))
    (out / "splits.json").write_text(
        json.dumps({"version": 1, "groups": synth_groups, "images": synth_splits}, indent=2)
    )
    make_contact_sheet(sheet_records, out / "contact_sheet.png")
    return out_coco


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", required=True)
    p.add_argument("--split-map", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--images-root", required=True)
    p.add_argument("--n-per-image", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-class-balance", action="store_true")
    p.add_argument("--blend-mode", choices=("hybrid", "feather", "poisson"), default="hybrid")
    p.add_argument("--perspective-jitter", type=float, default=0.03)
    p.add_argument("--slot-shuffle-prob", type=float, default=0.0)
    p.add_argument("--photometric", action="store_true")
    a = p.parse_args()
    run_cli(
        coco_path=a.coco,
        split_map_path=a.split_map,
        split=a.split,
        images_root=a.images_root,
        out_dir=a.out,
        n_per_image=a.n_per_image,
        seed=a.seed,
        class_balance=not a.no_class_balance,
        blend_mode=a.blend_mode,
        perspective_jitter=a.perspective_jitter,
        slot_shuffle_prob=a.slot_shuffle_prob,
        apply_photometric=a.photometric,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "blend_patch",
    "build_bank",
    "make_contact_sheet",
    "poisson_blend",
    "run_cli",
    "slot_shuffle",
    "synthesize",
]
