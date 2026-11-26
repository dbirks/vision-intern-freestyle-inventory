#!/usr/bin/env python
"""Probe the LLM Collective vision models for cartridge grounding.

Subcommands:
    list-models   print the ids served at /v1/models (no image calls)
    detect        per-model vision smoke test, then cartridge-box detection
                  on the 3 probe images, at full res and max_side=1024, with
                  both absolute-pixel and 0-1000-normalized prompts
    crop          cut a crop from an image (hand-chosen box) for identify
    identify      identify a cartridge crop against the 18-product catalog

Everything accumulates into ``runs/probe-collective/results.json``;
boxes overlays go to ``runs/probe-collective/viz/``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = REPO_ROOT / "data/raw/freestyle-vision-data/images"
DEFAULT_OUTDIR = REPO_ROOT / "runs/probe-collective"
DEFAULT_MODELS = (
    "qwen3-8-flash-next",
    "david-birks/qwen3-8-flash-next",
    "david-birks/inkling-2-mini",
    "qwen3-8-27b",
)

DEFAULT_IMAGES = (
    "local/downloads/frame.jpg",
    "island-cabinet-01/50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg",
    "island-cabinet-01/2ee373c6-d8b6-4837-8c50-8aea807bbe5e/006.jpg",
)

# ----------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------

def load_active_products(catalog: Any | None = None) -> list[dict]:
    """Active catalog products with id, name, aliases, colour hint, shelf."""
    if catalog is None:
        from freestyle.catalog import load_catalog

        catalog = load_catalog(active_only=True)
    products = []
    for item in catalog:
        products.append({
            "id": str(item["id"]),
            "name": str(item.get("name", item["id"])),
            "aliases": [str(a) for a in (item.get("aliases") or [])],
            "color_hint": str(item.get("color_hint") or ""),
            "shelf": str(item.get("shelf_hint") or ""),
        })
    return products


def build_identify_prompt(products: list[dict]) -> str:
    """Prompt asking for the catalog product id of a cartridge crop."""
    lines = []
    for p in products:
        aliases = ", ".join(p["aliases"]) if p["aliases"] else "-"
        color = p["color_hint"] if p["color_hint"] else "-"
        lines.append(
            f"- {p['name']} [{p['id']}] — aliases: {aliases} — colour: {color} — {p['shelf']} shelf"
        )
    catalog = "\n".join(lines)
    return (
        "Here is a close-up photo of a beverage cartridge box from a Coca-Cola "
        "Freestyle cabinet. Identify which catalog product this cartridge is.\n"
        "Reply with ONLY the product id (the lowercase slug in square brackets), "
        "nothing else.\n\n"
        "Products:\n" + catalog
    )


def build_count_prompt() -> str:
    return (
        "Count the cartridge boxes visible in this photo of a drinks cartridge "
        "cabinet (two shelves). Reply with ONLY a JSON object of the form "
        '{"top": <int>, "bottom": <int>}. No other text.'
    )


def build_boxes_prompt(convention: str, width: int, height: int) -> str:
    """Prompt for cartridge boxes; *convention* is 'absolute' or 'normalized'."""
    if convention == "absolute":
        coord_desc = (
            f"ABSOLUTE PIXEL coordinates within the image: x in [0, {width}], "
            f"y in [0, {height}]"
        )
    elif convention == "normalized":
        coord_desc = "normalised coordinates in the range [0, 1000]"
    else:
        raise ValueError(f"unknown convention: {convention!r}")
    return (
        "Detect every cartridge box in this photo of a drinks cartridge cabinet "
        "(two shelves of cartridge packs, spines out).\n"
        "For every cartridge box you can see, return a JSON array of objects "
        "with keys:\n"
        f'  "bbox_2d": [x1, y1, x2, y2] in {coord_desc}\n'
        '  "label": "cartridge"\n'
        "Return ONLY the JSON array, e.g. "
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "cartridge"}].'
    )


def parse_response_json(text: str) -> Any | None:
    """Extract a JSON array/object from VLM text (fences/whitespace tolerant)."""
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def detect_coordinate_convention(boxes: list[list[float]], width: int, height: int) -> str:
    """Heuristic: absolute pixels if any coordinate exceeds 1000, else normalized.

    Not a substitute for eyeballing overlays — the operator must verify visually.
    """
    all_coords = [c for box in boxes for c in box]
    if not all_coords:
        return "unknown"
    if max(all_coords) > 1000:
        return "absolute"
    # Purely informational fallback: 1000-scale values are plausible for both
    # conventions when the image is small; flag it as ambiguous.
    max_short = max(width, height)
    if max_short <= 1000:
        return "ambiguous"
    return "normalized"


def to_pixel_boxes(
    boxes: list[list[float]], convention: str, width: int, height: int
) -> tuple[list[list[int]], str]:
    """Scale raw boxes to original-image pixel coordinates.

    Returns ``(pixel_boxes, convention_used)`` where *convention_used* is the
    convention written in the prompt or the heuristic fallback.
    """
    used = convention
    if convention not in ("absolute", "normalized"):
        used = detect_coordinate_convention(boxes, width, height)
    if used == "ambiguous":
        used = "normalized"
    out = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if used == "normalized":
            x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
            y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
        out.append([
            max(0, min(width, round(x1))),
            max(0, min(height, round(y1))),
            max(0, min(width, round(x2))),
            max(0, min(height, round(y2))),
        ])
    return [b for b in out if b[2] > b[0] and b[3] > b[1]], used


def extract_boxes(text: str) -> list[dict]:
    """Parse bbox objects out of a VLM reply (reuses tools.vlm_detect parsing)."""
    from tools.vlm_detect import parse_bboxes_from_response

    objects = parse_bboxes_from_response(text)
    if objects:
        return objects
    parsed = parse_response_json(text)
    if isinstance(parsed, list):
        return [o for o in parsed if isinstance(o, dict) and "bbox_2d" in o]
    if isinstance(parsed, dict) and "bbox_2d" in parsed:
        return [parsed]
    return []


def parse_count_json(text: str) -> dict[str, Any] | None:
    """Parse the count prompt's JSON reply, tolerating junk around it."""
    parsed = parse_response_json(text)
    out: dict[str, Any] = {}
    if isinstance(parsed, dict):
        for shelf in ("top", "bottom"):
            val = parsed.get(shelf)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                out[shelf] = int(val)
            elif isinstance(val, str):
                m = re.search(r"\d+", val)
                if m:
                    out[shelf] = int(m.group())
        if out:
            return out
    # Tolerate bare "top: 8, bottom: 9" style replies
    for shelf in ("top", "bottom"):
        m = re.search(rf"{shelf}\s*[:=]\s*(\d+)", text, re.IGNORECASE)
        if m:
            out[shelf] = int(m.group(1))
    return out or None


def match_product(text: str, products: list[dict]) -> str | None:
    """Match a VLM reply to a product id (id, name, or alias, word-bounded)."""
    if not text:
        return None
    lower = text.lower()
    best: tuple[int, str] | None = None
    for p in products:
        score = 0
        for token in [p["id"], p["name"], *p["aliases"]]:
            tok = token.lower().strip()
            if not tok:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", lower):
                score += 3 if tok == p["id"] else 2
        if score and (best is None or score > best[0]):
            best = (score, p["id"])
    return best[1] if best else None


def draw_overlay(
    image: Image.Image,
    boxes: list[list[int]],
    out_path: Path,
    labels: list[str] | None = None,
    title: str | None = None,
) -> Path:
    """Draw boxes (green) with numbered labels onto the image; save as PNG."""
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    W, H = im.size
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="#00c851", width=3)
        label = f"{i + 1} {(labels[i] if labels and i < len(labels) else 'cartridge')}"
        ty = max(0, y1 - 18)
        draw.rectangle([x1, ty, x1 + 4 + 8 * len(label), ty + 18], fill="#00c851")
        draw.text((x1 + 3, ty + 2), label, fill="black")
    if title:
        draw.rectangle([0, 0, W, 30], fill="#222222")
        draw.text((8, 7), title, fill="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return out_path


def synthetic_tiny_image() -> Image.Image:
    """160x120 synthetic RGB image with three solid blocks (no private data)."""
    im = Image.new("RGB", (160, 120), "white")
    for x1, x2, color in ((0, 54, (220, 30, 30)), (54, 107, (30, 180, 30)), (107, 160, (40, 40, 220))):
        for x in range(x1, x2):
            for y in range(0, 120):
                im.putpixel((x, y), color)
    return im


def image_id_for(path: Path) -> str:
    """Image id = path relative to the images root (matches ARCHITECTURE.md)."""
    try:
        return str(path.relative_to(IMAGES_ROOT))
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------
# Results bookkeeping
# ----------------------------------------------------------------------

def load_results(outdir: Path) -> dict:
    path = outdir / "results.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models_list": [],
        "tiny_probes": {},
        "detect": [],
        "identify": [],
        "errors": [],
    }


def save_results(outdir: Path, results: dict) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    results["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = outdir / "results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    return path


def _upsert_call(results: dict, section: str, result: dict) -> None:
    """Insert or replace a call record keyed by (model, image/crop, max_side, stage)."""
    key = (result.get("model"), result.get("image_id") or result.get("crop"),
           result.get("max_side"), result.get("stage"))
    section_list = results.setdefault(section, [])
    for i, existing in enumerate(section_list):
        ek = (existing.get("model"), existing.get("image_id") or existing.get("crop"),
              existing.get("max_side"), existing.get("stage"))
        if ek == key:
            section_list[i] = result
            return
    section_list.append(result)


# ----------------------------------------------------------------------
# Call helpers
# ----------------------------------------------------------------------

def _snapshot(last_call: dict, extra: dict | None = None) -> dict:
    """Copy the module-level LAST_CALL snapshot into a plain dict."""
    out = {k: v for k, v in last_call.items() if k != "original_image_size"}
    out["original_image_size"] = list(last_call.get("original_image_size", []) or [])
    if extra:
        out.update(extra)
    return out


def _call(model: str, image: Image.Image, prompt: str, max_side: int | None,
          *, max_tokens: int = 12000, retry_budget_s: float = 600.0) -> tuple[str, dict, str | None]:
    """One chat_image call; never raises — returns (text, meta, error)."""
    from freestyle.collective import chat_image, LAST_CALL

    try:
        text = chat_image(
            model, image, prompt, max_side=max_side,
            max_tokens=max_tokens, retry_budget_s=retry_budget_s,
        )
        meta = _snapshot(dict(LAST_CALL))
        return text, meta, None
    except Exception as exc:  # noqa: BLE001 - record and continue
        meta = _snapshot(dict(LAST_CALL))
        meta["error"] = str(exc)[:400]
        return "", meta, str(exc)[:300]


# ----------------------------------------------------------------------
# Subcommand: list-models
# ----------------------------------------------------------------------

def cmd_list_models(args: argparse.Namespace) -> int:
    from freestyle.collective import list_models

    outdir = Path(args.outdir)
    results = load_results(outdir)
    try:
        ids = list_models()
        results["models_list"] = ids
        save_results(outdir, results)
        print("\n".join(ids))
        return 0
    except Exception as exc:  # noqa: BLE001
        results["errors"].append({"stage": "list-models", "error": str(exc)[:400]})
        save_results(outdir, results)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


# ----------------------------------------------------------------------
# Subcommand: detect
# ----------------------------------------------------------------------

def _tiny_probe(model: str, outdir: Path, max_tokens: int = 256) -> dict:
    t0 = time.monotonic()
    text, meta, err = _call(model, synthetic_tiny_image(),
                            "Describe the colors of the solid blocks in this "
                            "image, one word per block, comma separated.",
                            None, max_tokens=max_tokens)
    ok = err is None
    return {
        "model": model,
        "ok": ok,
        "latency_s": round(time.monotonic() - t0, 2),
        "meta": meta,
        "response_text": text.strip()[:300] if text else "",
        "error": err,
    }


def _detect_one(model: str, image_id: str, image_path: Path, max_side: int | None,
                outdir: Path, retry_budget_s: float = 600.0) -> list[dict]:
    """count + absolute/normalized boxes calls for one image at one size."""
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    viz_dir = outdir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    def _stage(name: str, prompt: str, post_process=None) -> dict:
        text, meta, err = _call(model, image, prompt, max_side,
                                max_tokens=16000, retry_budget_s=retry_budget_s)
        rec: dict[str, Any] = {
            "model": model,
            "image_id": image_id,
            "image_path": str(image_path),
            "image_size": [W, H],
            "max_side": max_side,
            "stage": name,
            "error": err,
            "response_text": text.strip()[:1500] if text else "",
            **meta,
        }
        if callable(post_process):
            post_process(rec, text, image, viz_dir)
        results.append(rec)
        return rec

    # (a) count
    def _count_post(rec, text, image, viz_dir):
        rec["count"] = parse_count_json(text)

    # (b) boxes, both conventions
    def _boxes_post(convention: str):
        def post(rec, text, image, viz_dir):
            objects = extract_boxes(text)
            raw_boxes = []
            labels = []
            for obj in objects:
                try:
                    box = [float(v) for v in obj["bbox_2d"]]
                except (KeyError, TypeError, ValueError):
                    continue
                if len(box) != 4:
                    continue
                raw_boxes.append(box)
                labels.append(str(obj.get("label", "cartridge")))
            rec["raw_box_count"] = len(raw_boxes)
            rec["raw_boxes"] = raw_boxes
            # Empirically (inkling-2-mini, flash-next) the model answers in
            # ORIGINAL image pixel coordinates even when the image was resized:
            # raw absolute values are identical at full res and max_side=1024.
            # Normalised (0-1000) values are simply scaled by /1000 * W,H.
            # Neither needs the sent-size ratio; store both original sizes.
            sent = rec.get("sent_image_size") or [W, H]
            rec["sent_image_size"] = list(sent)
            # Trust the prompt only when the data can't tell us otherwise;
            # all-<=1000 values on a big image mean the model ignored the
            # absolute convention and returned normalized coordinates.
            detected = detect_coordinate_convention(raw_boxes, W, H)
            used = detected if detected in ("absolute", "normalized") else convention
            pixel, _ = to_pixel_boxes(raw_boxes, used, W, H)
            rec["convention"] = used
            rec["convention_prompt"] = convention
            rec["pixel_boxes"] = pixel
            rec["valid_box_count"] = len(pixel)
            # Alternative-convention parse (informational)
            alt = "normalized" if used == "absolute" else "absolute"
            alt_boxes, alt_used = to_pixel_boxes(raw_boxes, alt, W, H)
            rec["alt_convention"] = alt_used
            rec["alt_valid_box_count"] = len(alt_boxes)
            if raw_boxes and not rec.get("error"):
                slug = image_id.replace("/", "_").replace(".jpg", "")
                size_tag = "full" if max_side is None else f"max{max_side}"
                ov_path = viz_dir / f"{model.replace('/', '_')}__{slug}__{size_tag}__{convention}.png"
                draw_overlay(image, pixel, ov_path,
                             labels=labels[:len(pixel)],
                             title=f"{model} | {image_id} | {convention} | {len(pixel)} boxes")
                rec["overlay_path"] = str(ov_path)
        return post

    _stage("count", build_count_prompt(), _count_post)
    _stage("boxes_abs", build_boxes_prompt("absolute", W, H), _boxes_post("absolute"))
    _stage("boxes_norm", build_boxes_prompt("normalized", W, H), _boxes_post("normalized"))
    return results


def cmd_tiny(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    results = load_results(outdir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        probe = _tiny_probe(model, outdir, max_tokens=args.max_tokens)
        results["tiny_probes"][model] = probe
        save_results(outdir, results)
        ok = "ok" if probe["ok"] else f"FAILED ({probe['error']})"
        print(f"  {model:48s} {ok}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = load_results(outdir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    images = [i.strip() for i in args.images.split(",") if i.strip()]
    sizes = [None if s.lower() == "full" else int(s) for s in args.max_sides.split(",")]
    only_ok = set()

    print("== tiny vision smoke test (1 call per model)")
    for model in models:
        probe = _tiny_probe(model, outdir)
        results["tiny_probes"][model] = probe
        save_results(outdir, results)
        status = "ok" if probe["ok"] else f"FAILED ({probe['error']})"
        print(f"  {model:44s} {status}")
        if probe["ok"]:
            only_ok.add(model)
    save_results(outdir, results)

    for model in sorted(only_ok):
        print(f"== detect: {model}")
        for image_id in images:
            image_path = IMAGES_ROOT / image_id
            if not image_path.exists():
                results["errors"].append({"stage": "detect", "model": model,
                                          "image_id": image_id, "error": "image not found"})
                continue
            for max_side in sizes:
                tag = "full" if max_side is None else f"max_side={max_side}"
                for rec in _detect_one(model, image_id, image_path, max_side, outdir,
                                       retry_budget_s=args.retry_budget_s):
                    _upsert_call(results, "detect", rec)
                    save_results(outdir, results)
                    nboxes = rec.get("valid_box_count", "-")
                    print(f"  {tag:12s} {image_id:60s} {rec['stage']:11s} "
                          f"boxes={nboxes} err={rec.get('error') or ''}")
    skipped = [m for m in models if m not in only_ok]
    for model in skipped:
        print(f"== skipping {model}: vision unavailable this run "
              f"({results['tiny_probes'][model].get('error', 'unknown')})")
    save_results(outdir, results)
    return 0 if only_ok else 2


# ----------------------------------------------------------------------
# Subcommand: crop
# ----------------------------------------------------------------------

def cmd_crop(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = [float(v) for v in args.box.split(",")]
    W, H = image.size
    x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
    y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        print("ERROR: box must be ordered and non-empty", file=sys.stderr)
        return 1
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, "JPEG", quality=95)
    print(f"saved crop {out_path} ({crop.size[0]}x{crop.size[1]}) from {image_path}")
    return 0


# ----------------------------------------------------------------------
# Subcommand: identify
# ----------------------------------------------------------------------

def _identify_models(args, results: dict) -> list[str]:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    ok = [m for m, p in results.get("tiny_probes", {}).items() if p.get("ok")]
    return ok or [m for m in DEFAULT_MODELS]


def cmd_identify(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = load_results(outdir)
    products = load_active_products()
    prompt = build_identify_prompt(products)
    models = _identify_models(args, results)

    crop_path = Path(args.crop)
    crop = Image.open(crop_path).convert("RGB")
    print(f"== identify {crop_path} ({crop.size[0]}x{crop.size[1]})")

    for model in models:
        text, meta, err = _call(model, crop, prompt, args.max_side,
                                max_tokens=args.max_tokens, retry_budget_s=args.retry_budget_s)
        parsed = match_product(text, products) if text else None
        rec = {
            "model": model,
            "crop": str(crop_path),
            "max_side": args.max_side,
            "stage": "identify",
            "response_text": text.strip()[:400] if text else "",
            "parsed_product": parsed,
            "error": err,
            **meta,
        }
        _upsert_call(results, "identify", rec)
        save_results(outdir, results)
        print(f"  {model:44s} -> {parsed or '(none)'} ({rec.get('latency_s')}s)")
    save_results(outdir, results)
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                        help="output directory (default %(default)s)")
    parser.add_argument("--retry-budget-s", type=float, default=600.0,
                        help="per-call retry budget in seconds (default %(default)s)")
    sub = parser.add_subparsers(dest="cmd")

    p_models = sub.add_parser("list-models", help="list served model ids")
    p_models.set_defaults(func=cmd_list_models)

    p_tiny = sub.add_parser("tiny", help="vision smoke test (1 call per model)")
    p_tiny.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p_tiny.add_argument("--max-tokens", type=int, default=256)
    p_tiny.set_defaults(func=cmd_tiny)

    p_detect = sub.add_parser("detect", help="run the full detection matrix")
    p_detect.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p_detect.add_argument("--images", default=",".join(DEFAULT_IMAGES))
    p_detect.add_argument("--max-sides", default="full,1024",
                          help="comma list, 'full' = original size (default %(default)s)")
    p_detect.set_defaults(func=cmd_detect)

    p_crop = sub.add_parser("crop", help="cut a crop from an image")
    p_crop.add_argument("--image", required=True)
    p_crop.add_argument("--box", required=True, help="x1,y1,x2,y2 in original pixels")
    p_crop.add_argument("--out", required=True)
    p_crop.set_defaults(func=cmd_crop)

    p_identify = sub.add_parser("identify", help="identify a cartridge crop")
    p_identify.add_argument("--crop", required=True)
    p_identify.add_argument("--models", default="")
    p_identify.add_argument("--max-side", type=int, default=None,
                            help="optional max image side (pixels)")
    p_identify.add_argument("--max-tokens", type=int, default=2048,
                            help="max completion tokens per call (default %(default)s)")
    p_identify.set_defaults(func=cmd_identify)

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
