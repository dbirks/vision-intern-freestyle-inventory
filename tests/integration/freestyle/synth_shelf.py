"""Deterministic synthetic Freestyle cabinet renderer and COCO GT generator.

Generates 2048x1536 barrel-distorted two-shelf cartridge cabinet images with:
- 18 visually distinct cartridge classes (colour + quadrant pattern + text),
- a 2-high stacked column on the top shelf,
- analytical barrel distortion applied to image pixels, quad corners, and bboxes,
- COCO GT in contract format, group-safe ``splits.json``, and no-cabinet negatives.
"""

from __future__ import annotations

import functools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536
BARREL_K = 0.35

# Canonical 18 active catalog slugs (used as fallback if private catalog JSON is absent).
FALLBACK_SLUGS: tuple[str, ...] = (
    "nns-sweetener",
    "ingredient-aa",
    "ingredient-cc",
    "fanta-orange",
    "hi-c",
    "sprite",
    "flavor-orange",
    "flavor-vanilla",
    "flavor-lemon",
    "flavor-cherry",
    "coca-cola-part1",
    "coca-cola-part2",
    "diet-coke-part1",
    "diet-coke-part2",
    "dr-pepper-part1",
    "dr-pepper-part2",
    "coke-zero-part1",
    "coke-zero-part2",
)

# High-contrast (top_color, bottom_color, accent_color, pattern) specs for 18 classes
# so stub colour-histogram + quadrant features remain well separated.
_CLASS_VISUALS: tuple[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], str], ...] = (
    ((235, 45, 45), (235, 220, 45), (255, 255, 255), "top_band"),
    ((35, 195, 70), (20, 90, 210), (245, 245, 245), "bottom_band"),
    ((30, 110, 240), (235, 90, 20), (255, 255, 255), "split_lr"),
    ((245, 145, 20), (245, 145, 20), (25, 45, 175), "center_stripe"),
    ((215, 25, 75), (25, 205, 195), (255, 240, 120), "checker"),
    ((35, 215, 110), (235, 235, 45), (20, 60, 30), "top_band"),
    ((250, 115, 15), (40, 40, 45), (255, 235, 180), "bottom_band"),
    ((240, 225, 175), (125, 65, 25), (30, 20, 10), "center_stripe"),
    ((245, 235, 30), (45, 175, 45), (20, 20, 20), "split_lr"),
    ((185, 20, 65), (245, 190, 210), (70, 10, 25), "checker"),
    ((210, 20, 20), (245, 245, 245), (20, 20, 20), "top_band"),
    ((145, 15, 25), (25, 25, 30), (245, 215, 215), "bottom_band"),
    ((215, 220, 228), (195, 25, 35), (25, 25, 30), "split_lr"),
    ((155, 165, 175), (25, 115, 215), (250, 250, 250), "center_stripe"),
    ((115, 25, 45), (240, 185, 75), (255, 245, 220), "checker"),
    ((85, 15, 30), (155, 95, 205), (245, 235, 255), "top_band"),
    ((28, 30, 36), (225, 30, 35), (245, 245, 245), "bottom_band"),
    ((20, 24, 30), (35, 185, 235), (245, 245, 245), "split_lr"),
)


def get_class_names(*, include_unknown: bool = True) -> list[str]:
    """Return active catalog slugs from ``freestyle.catalog`` when available, else fallback."""
    try:
        from freestyle import catalog

        if catalog.CATALOG_PATH.is_file():
            names = [str(n) for n in catalog.class_names()]
            if not include_unknown:
                return [n for n in names if n != catalog.UNKNOWN]
            return names
    except Exception:
        pass
    names = list(FALLBACK_SLUGS)
    if include_unknown:
        names.append("unknown")
    return names


def build_catalog_items(active_slugs: list[str]) -> dict[str, Any]:
    """Build a synthetic ``catalog_items.json`` structure matching the active slugs."""
    color_hints = [
        "yellow", "green", "blue", "orange", "red", "green",
        "orange", "tan", "yellow", "dark red", "red", "maroon",
        "silver", "grey", "maroon", "dark red", "black", "black",
    ]
    items = []
    for idx, slug in enumerate(active_slugs):
        items.append(
            {
                "id": slug,
                "name": slug.replace("-", " ").title(),
                "aliases": [slug.split("-")[0]],
                "color_hint": color_hints[idx % len(color_hints)],
                "shelf_hint": "top" if idx < 10 else "bottom",
                "kind": "cartridge",
                "active": True,
                "sort_order": idx + 1,
            }
        )
    return {"items": items}


def distort_point(
    x: float,
    y: float,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    k: float = BARREL_K,
) -> tuple[float, float]:
    """Analytically apply radial barrel distortion to an undistorted pixel coordinate."""
    cx = width * 0.5
    cy = height * 0.5
    radius = math.hypot(cx, cy)
    dx = (x - cx) / radius
    dy = (y - cy) / radius
    r2 = dx * dx + dy * dy
    scale = 1.0 / (1.0 + k * r2)
    return cx + (x - cx) * scale, cy + (y - cy) * scale


def distort_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    k: float = BARREL_K,
    edge_samples: int = 11,
) -> tuple[list[float], list[list[float]]]:
    """Return ``(bbox_xywh, quad_4x2)`` after analytical barrel distortion."""
    corners_u = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    quad = [
        [round(px, 2), round(py, 2)]
        for px, py in (distort_point(ux, uy, width, height, k) for ux, uy in corners_u)
    ]

    xs: list[float] = []
    ys: list[float] = []
    ts = np.linspace(0.0, 1.0, edge_samples)
    for (ax, ay), (bx, by) in zip(corners_u, corners_u[1:] + corners_u[:1]):
        for t in ts:
            dx, dy = distort_point(ax + (bx - ax) * float(t), ay + (by - ay) * float(t), width, height, k)
            xs.append(dx)
            ys.append(dy)

    bx0 = max(0.0, min(xs))
    by0 = max(0.0, min(ys))
    bx1 = min(float(width), max(xs))
    by1 = min(float(height), max(ys))
    bbox = [round(bx0, 2), round(by0, 2), round(bx1 - bx0, 2), round(by1 - by0, 2)]
    return bbox, quad


@functools.lru_cache(maxsize=4)
def _barrel_remap_grid(
    width: int,
    height: int,
    k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute inverse barrel-distortion lookup indices and bilinear weights."""
    cx = np.float32(width * 0.5)
    cy = np.float32(height * 0.5)
    radius = np.float32(math.hypot(float(cx), float(cy)))

    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    dx_d = (xs - cx) / radius
    dy_d = (ys - cy) / radius
    rd2 = dx_d * dx_d + dy_d * dy_d
    disc = np.float32(1.0) - np.float32(4.0 * k) * rd2
    valid_rad = disc > 0.0
    sqrt_disc = np.sqrt(np.maximum(disc, np.float32(0.0)))
    s_inv = np.float32(2.0) / (np.float32(1.0) + sqrt_disc)

    xu = cx + (xs - cx) * s_inv
    yu = cy + (ys - cy) * s_inv

    in_bounds = valid_rad & (xu >= 0.0) & (xu < width - 1) & (yu >= 0.0) & (yu < height - 1)
    x0 = np.clip(np.floor(xu), 0, width - 2).astype(np.int32)
    y0 = np.clip(np.floor(yu), 0, height - 2).astype(np.int32)
    wx = np.clip(xu - x0, 0.0, 1.0)[..., None]
    wy = np.clip(yu - y0, 0.0, 1.0)[..., None]
    return in_bounds, x0, y0, wx, wy


def apply_barrel_distortion(image: Image.Image, k: float = BARREL_K) -> Image.Image:
    """Warp a full RGB image with the exact inverse of :func:`distort_point`."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    in_bounds, x0, y0, wx, wy = _barrel_remap_grid(w, h, float(k))

    c00 = arr[y0, x0].astype(np.float32)
    c10 = arr[y0, x0 + 1].astype(np.float32)
    c01 = arr[y0 + 1, x0].astype(np.float32)
    c11 = arr[y0 + 1, x0 + 1].astype(np.float32)

    interp = (
        c00 * (1.0 - wx) * (1.0 - wy)
        + c10 * wx * (1.0 - wy)
        + c01 * (1.0 - wx) * wy
        + c11 * wx * wy
    )
    out = np.full_like(arr, (18, 20, 24), dtype=np.uint8)
    out[in_bounds] = np.clip(np.rint(interp[in_bounds]), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _draw_cartridge(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    class_idx: int,
    slug: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    x0, y0, x1, y1 = rect
    top_c, bot_c, acc_c, pattern = _CLASS_VISUALS[class_idx % len(_CLASS_VISUALS)]
    xm = (x0 + x1) // 2
    ym = (y0 + y1) // 2

    draw.rectangle([x0, y0, x1, y1], fill=top_c, outline=(235, 235, 235), width=2)
    if pattern == "top_band":
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y0 + (y1 - y0) // 3], fill=bot_c)
        draw.rectangle([x0 + 6, ym - 12, x1 - 6, ym + 12], fill=acc_c)
    elif pattern == "bottom_band":
        draw.rectangle([x0 + 2, ym, x1 - 2, y1 - 2], fill=bot_c)
        draw.rectangle([x0 + 8, y0 + 12, x1 - 8, y0 + 30], fill=acc_c)
    elif pattern == "split_lr":
        draw.rectangle([xm, y0 + 2, x1 - 2, y1 - 2], fill=bot_c)
        draw.rectangle([x0 + 6, ym - 14, x1 - 6, ym + 14], fill=acc_c)
    elif pattern == "center_stripe":
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=top_c)
        draw.rectangle([x0 + 12, y0 + 8, x1 - 12, y1 - 8], fill=acc_c)
        draw.rectangle([x0 + 2, ym - 18, x1 - 2, ym + 18], fill=bot_c)
    else:  # checker
        draw.rectangle([xm, y0 + 2, x1 - 2, ym], fill=bot_c)
        draw.rectangle([x0 + 2, ym, xm, y1 - 2], fill=bot_c)
        draw.ellipse([xm - 14, ym - 14, xm + 14, ym + 14], fill=acc_c)

    tag = "".join(part[:2].upper() for part in slug.split("-")[:2])[:5]
    draw.rectangle([x0 + 6, ym - 10, x1 - 6, ym + 10], fill=(18, 18, 22))
    draw.text((x0 + 10, ym - 6), tag, fill=(250, 250, 250), font=font)


def render_cabinet_image(
    active_slugs: list[str],
    category_map: dict[str, int],
    *,
    seed: int = 0,
    stacked_col: int = 4,
    brightness_offset: int = 0,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Render one 2048x1536 barrel-distorted cabinet image and return GT annotations."""
    rng = random.Random(seed)
    order = list(range(len(active_slugs)))
    rng.shuffle(order)

    x_jitter = rng.randint(-18, 18)
    y_jitter = rng.randint(-12, 12)

    bg_val = max(20, min(65, 38 + brightness_offset))
    canvas = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (bg_val, bg_val + 2, bg_val + 6))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Cabinet frame & shelf rails.
    draw.rectangle([140, 220 + y_jitter, 1908, 1340 + y_jitter], outline=(85, 90, 98), width=6)
    draw.rectangle([160, 654 + y_jitter, 1888, 672 + y_jitter], fill=(95, 100, 108))
    draw.rectangle([160, 1214 + y_jitter, 1888, 1232 + y_jitter], fill=(95, 100, 108))

    # Build 18 cartridge slots across two shelves (including 1 2-high stacked column).
    slots_u: list[tuple[int, int, int, int, dict[str, Any]]] = []
    cart_w = 82
    top_x0 = 240 + x_jitter
    top_pitch = 185
    top_bottom = 652 + y_jitter

    for col in range(9):
        cx0 = top_x0 + col * top_pitch
        cx1 = cx0 + cart_w
        if col == stacked_col:
            # 2-high stacked column: upper tier + lower tier with 12px gap.
            slots_u.append((cx0, top_bottom - 322, cx1, top_bottom - 167, {"tier": 1, "shelf": 0}))
            slots_u.append((cx0, top_bottom - 155, cx1, top_bottom, {"tier": 0, "shelf": 0}))
        else:
            slots_u.append((cx0, top_bottom - 240, cx1, top_bottom, {"tier": 0, "shelf": 0}))

    bot_x0 = 310 + x_jitter
    bot_pitch = 190
    bot_bottom = 1212 + y_jitter
    for col in range(8):
        cx0 = bot_x0 + col * bot_pitch
        cx1 = cx0 + cart_w
        slots_u.append((cx0, bot_bottom - 240, cx1, bot_bottom, {"tier": 0, "shelf": 1}))

    annotations: list[dict[str, Any]] = []
    for slot_idx, (ux0, uy0, ux1, uy1, meta) in enumerate(slots_u):
        cls_idx = order[slot_idx % len(order)]
        slug = active_slugs[cls_idx]
        _draw_cartridge(draw, (ux0, uy0, ux1, uy1), cls_idx, slug, font)
        bbox, quad = distort_box(ux0, uy0, ux1, uy1)
        annotations.append(
            {
                "category_id": category_map[slug],
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0,
                "quad": quad,
                "attributes": {
                    "truncated": False,
                    "occluded": False,
                    "identity_visible": True,
                    "source": "human",
                    "tier": meta["tier"],
                },
            }
        )

    distorted = apply_barrel_distortion(canvas)
    return distorted, annotations


def render_negative_image(seed: int = 0) -> Image.Image:
    """Render a 2048x1536 no-cabinet negative frame with barrel distortion."""
    rng = random.Random(seed)
    canvas = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (26, 28, 34))
    draw = ImageDraw.Draw(canvas)
    for i in range(6):
        x0 = rng.randint(100, 1400)
        y0 = rng.randint(100, 1000)
        c = rng.randint(38, 68)
        draw.ellipse([x0, y0, x0 + 350, y0 + 220], fill=(c, c + 3, c + 6))
    return apply_barrel_distortion(canvas)


def _subset_coco(full_coco: dict[str, Any], split_name: str, split_images: dict[str, str]) -> dict[str, Any]:
    imgs = [img for img in full_coco["images"] if split_images[img["file_name"]] == split_name]
    img_ids = {img["id"] for img in imgs}
    anns = [ann for ann in full_coco["annotations"] if ann["image_id"] in img_ids]
    return {
        "info": full_coco["info"],
        "licenses": [],
        "images": imgs,
        "categories": full_coco["categories"],
        "annotations": anns,
    }


def generate_dataset(out_dir: Path | str, *, seed: int = 42) -> dict[str, Any]:
    """Write synthetic images, COCO GT files, ``splits.json``, and ``catalog_items.json``."""
    out = Path(out_dir)
    images_root = out / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    all_names = get_class_names(include_unknown=True)
    active_slugs = [n for n in all_names if n != "unknown"]
    cat_map = {name: i + 1 for i, name in enumerate(all_names)}
    categories = [{"id": cid, "name": name} for name, cid in cat_map.items()]

    catalog_path = out / "catalog_items.json"
    catalog_path.write_text(json.dumps(build_catalog_items(active_slugs), indent=2))

    # 2 train groups, 1 val group, 2 test groups, and 2 no-cabinet negative groups.
    plan = [
        ("synth_train_a", "train", True, 0),
        ("synth_train_b", "train", True, 1),
        ("synth_val_a", "val", True, 2),
        ("synth_test_a", "test", True, 3),
        ("synth_test_b", "test", True, 4),
        ("synth_neg_a", "exclude", False, 5),
        ("synth_neg_b", "exclude", False, 6),
    ]

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    splits_groups: dict[str, str] = {}
    splits_images: dict[str, str] = {}
    negatives: list[str] = []

    ann_id = 1
    for img_idx, (group_id, split, has_cabinet, offset) in enumerate(plan, start=1):
        file_name = f"{group_id}/frame_001.jpg"
        img_path = images_root / file_name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        splits_groups[group_id] = split
        splits_images[file_name] = split

        if has_cabinet:
            img, anns = render_cabinet_image(
                active_slugs,
                cat_map,
                seed=seed + offset * 17,
                stacked_col=3 + (offset % 3),
                brightness_offset=(offset % 3 - 1) * 4,
            )
            scene = "cabinet_full"
            for ann in anns:
                ann["id"] = ann_id
                ann["image_id"] = img_idx
                coco_annotations.append(ann)
                ann_id += 1
        else:
            img = render_negative_image(seed=seed + offset * 31)
            scene = "no_cabinet"
            negatives.append(file_name)

        img.save(img_path, format="JPEG", quality=95)
        coco_images.append(
            {
                "id": img_idx,
                "file_name": file_name,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "group_id": group_id,
                "attributes": {"verified": True, "scene": scene},
            }
        )

    full_coco: dict[str, Any] = {
        "info": {"description": "Synthetic Freestyle cabinet GT", "version": 1},
        "licenses": [],
        "images": coco_images,
        "categories": categories,
        "annotations": coco_annotations,
    }
    gt_path = out / "gt_all.json"
    gt_path.write_text(json.dumps(full_coco, indent=2))

    train_coco_path = out / "gt_train.json"
    val_coco_path = out / "gt_val.json"
    test_coco_path = out / "gt_test.json"
    train_coco_path.write_text(json.dumps(_subset_coco(full_coco, "train", splits_images), indent=2))
    val_coco_path.write_text(json.dumps(_subset_coco(full_coco, "val", splits_images), indent=2))
    test_coco_path.write_text(json.dumps(_subset_coco(full_coco, "test", splits_images), indent=2))

    splits_data = {
        "version": 1,
        "seed": seed,
        "unit": "group",
        "groups": splits_groups,
        "images": splits_images,
        "negatives": sorted(negatives),
    }
    splits_path = out / "splits.json"
    splits_path.write_text(json.dumps(splits_data, indent=2))

    return {
        "root": out,
        "images_root": images_root,
        "gt_all_path": gt_path,
        "train_coco_path": train_coco_path,
        "val_coco_path": val_coco_path,
        "test_coco_path": test_coco_path,
        "splits_path": splits_path,
        "catalog_path": catalog_path,
        "full_coco": full_coco,
        "splits": splits_data,
        "category_map": cat_map,
        "active_slugs": active_slugs,
    }
