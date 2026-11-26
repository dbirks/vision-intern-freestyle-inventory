"""Per-image data audit: pixel stats, near-duplicate clusters, VLM scene triage.

    uv run python -m freestyle.audit                 # full audit (stats + VLM)
    uv run python -m freestyle.audit --no-vlm        # pixel stats only (offline)
    uv run python -m freestyle.audit --limit 5       # smoke run

Writes (all derived, never touches ``data/raw``):

    data/derived/audit/audit.csv          flat table, one row per photograph
    data/derived/audit/audit.json         same data + summary + duplicates
    data/derived/audit/vlm_cache.json     raw VLM answers (cache; skip re-calls)
    data/derived/audit/sheet_<label>.jpg  contact sheet per scene label

Scene labels come from the Collective VLM (``qwen3-8-flash-next``) and are
**model opinions, never ground truth** (ARCHITECTURE.md).  Obvious errors are
corrected in the committed override table ``freestyle/audit_overrides.json``
(``image_id -> {field: value}``); overrides win over both VLM and cache and are
tracked per row in ``overrides`` / ``label_source``.

Stats implemented with numpy + Pillow only (no opencv/imagehash in this env):
luma mean/std, Laplacian-variance sharpness at full res and on a 512 px
downscale, mean HSV saturation, and a 64-bit DCT perceptual hash.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):  # allow `python freestyle/audit.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freestyle.data import REPO_ROOT, list_images  # noqa: E402

OUT_DIR = REPO_ROOT / "data/derived/audit"
OVERRIDES_PATH = REPO_ROOT / "freestyle/audit_overrides.json"
VLM_MODEL = "qwen3-8-flash-next"

SCENE_LABELS = ("cabinet_full", "cabinet_partial", "no_cabinet", "occluded", "unknown")
MOTION_LEVELS = ("none", "mild", "severe")
DUPLICATE_MAX_DISTANCE = 8  # Hamming distance on the 64-bit pHash
SHEET_THUMB = 320


# ---------------------------------------------------------------------------
# pixel stats
# ---------------------------------------------------------------------------

def _lumacity(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the discrete Laplacian — the classic sharpness proxy."""
    kernel = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] \
        - 4.0 * gray[1:-1, 1:-1]
    return float(np.var(kernel))


def mean_saturation(img: Image.Image) -> float:
    """Mean HSV saturation (0..255) — how colourful the frame is."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    return float(hsv[..., 1].mean())


def _dct_matrix(n: int) -> np.ndarray:
    i = np.arange(n)
    m = np.cos(math.pi * (2.0 * i[None, :] + 1.0) * i[:, None] / (2.0 * n))
    m *= np.where(i == 0, math.sqrt(1.0 / n), math.sqrt(2.0 / n))[:, None]
    return m


_DCT32 = _dct_matrix(32)


def phash(img: Image.Image, hash_size: int = 8) -> str:
    """64-bit DCT perceptual hash as a hex string (imagehash ``phash`` method)."""
    size = hash_size * 4
    gray = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float64)
    coeffs = _DCT32 @ gray @ _DCT32.T
    low = coeffs[:hash_size, :hash_size]
    med = np.median(low[1:, 1:].ravel())  # ignore the DC term
    bits = (low > med).ravel()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def image_stats(path: str | Path, short_side: int = 512) -> dict[str, Any]:
    with Image.open(str(path)) as im:
        img = im.convert("RGB")
        w, h = img.size
        full = _lumacity(img)
        scale = short_side / max(w, h)
        if scale < 1.0:
            small = img.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
            )
        else:
            small = img
        return {
            "width": w,
            "height": h,
            "luma_mean": round(float(full.mean()), 2),
            "luma_std": round(float(full.std()), 2),
            "sharp_full": round(laplacian_variance(full), 2),
            "sharp_512": round(laplacian_variance(_lumacity(small)), 2),
            "saturation": round(mean_saturation(img), 2),
            "phash": phash(img),
        }


def duplicate_clusters(rows: list[dict[str, Any]], max_distance: int = DUPLICATE_MAX_DISTANCE) -> list[list[str]]:
    """Union-find clusters of images whose pHash differs by <= ``max_distance``."""
    ids = [r["image_id"] for r in rows if r.get("phash")]
    hashes = {r["image_id"]: r["phash"] for r in rows if r.get("phash")}
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if hamming(hashes[ids[a]], hashes[ids[b]]) <= max_distance:
                ra, rb = find(ids[a]), find(ids[b])
                if ra != rb:
                    parent[rb] = ra
    clusters: dict[str, list[str]] = {}
    for i in ids:
        clusters.setdefault(find(i), []).append(i)
    return sorted((sorted(v) for v in clusters.values() if len(v) > 1), key=lambda c: (-len(c), c[0]))


# ---------------------------------------------------------------------------
# VLM triage
# ---------------------------------------------------------------------------

PROMPT = """You are triaging one photograph from a camera mounted inside the door of a \
Coca-Cola Freestyle cartridge cabinet (160-degree wide-angle lens, so the shelf line is \
curved and the edges are distorted). The cabinet has two shelves of cartridge boxes.

Classify THIS image and answer with a single JSON object, no prose, exactly these keys:

{"scene_label": "cabinet_full" | "cabinet_partial" | "no_cabinet" | "occluded",
 "person_visible": true | false,
 "motion_blur": "none" | "mild" | "severe",
 "upside_down": true | false,
 "description": "<max 20 words>"}

Definitions:
- cabinet_full: both shelves are mostly visible in frame.
- cabinet_partial: some cartridge boxes or shelf are visible, but not both shelves.
- no_cabinet: no cabinet at all (ceiling, desk, room, a laptop, a bench).
- occluded: the lens is covered or pressed against something, so essentially nothing is recognizable.
- person_visible: any part of a person (head, hand, arm) is in frame.
- motion_blur: none = readable labels, mild = readable with effort, severe = smeared, labels unreadable.
- upside_down: the scene is rotated roughly 180 degrees (shelves/cartridge tops pointing down).
"""


def parse_vlm_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a VLM reply, normalising enum fields."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:160]!r}")
    obj = json.loads(text[start : end + 1])

    label = str(obj.get("scene_label", "")).strip().lower().replace("-", "_")
    obj["scene_label"] = label if label in SCENE_LABELS else "unknown"
    blur = str(obj.get("motion_blur", "")).strip().lower()
    obj["motion_blur"] = blur if blur in MOTION_LEVELS else "none"
    for key in ("person_visible", "upside_down"):
        obj[key] = bool(obj.get(key))
    obj["description"] = " ".join(str(obj.get("description", "")).split())[:200]
    return obj


def ask_vlm(image_path: str | Path, model: str = VLM_MODEL, max_side: int = 1024) -> dict[str, Any]:
    from freestyle.collective import chat_image

    text = chat_image(model, image_path, PROMPT, max_side=max_side, temperature=0.0)
    out = parse_vlm_json(text)
    out["vlm_raw"] = text.strip()[:600]
    out["vlm_model"] = model
    return out


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    # keys starting with '_' are comments in the override table, not entries
    data = {k: v for k, v in data.items() if not str(k).startswith("_")}
    bad = [k for k, v in data.items() if not isinstance(v, dict)]
    if bad:
        raise ValueError(f"{path}: values must be objects, got {bad[:3]}")
    return {k: {f: v for f, v in fields.items() if not f.startswith("_")}
            for k, fields in data.items()}


# ---------------------------------------------------------------------------
# contact sheets
# ---------------------------------------------------------------------------

def contact_sheet(rows: list[dict[str, Any]], out_path: Path, title: str, thumb: int = SHEET_THUMB) -> Path:
    """Grid of thumbnails with id/label captions; None when there are no rows."""
    if not rows:
        return out_path
    cols = 4
    pad, caption_h = 6, 34
    rows_n = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows_n * (thumb + caption_h + pad) + 24), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 6), f"{title}  ({len(rows)} images)", fill="black")
    for i, r in enumerate(rows):
        try:
            with Image.open(r["path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb), Image.LANCZOS)
                x = pad + (i % cols) * (thumb + pad)
                y = 24 + (i // cols) * (thumb + caption_h + pad)
                sheet.paste(im, (x + (thumb - im.width) // 2, y))
                d = ImageDraw.Draw(sheet)
                cap = f"{r['scene_label']} b={r.get('motion_blur')} p={int(bool(r.get('person_visible')))}"
                d.text((x, y + thumb + 2), r["image_id"][:44], fill="black")
                d.text((x, y + thumb + 16), cap[:44], fill=(90, 90, 90))
        except Exception as exc:  # noqa: BLE001 - a bad thumb must not kill the audit
            d = ImageDraw.Draw(sheet)
            x = pad + (i % cols) * (thumb + pad)
            y = 24 + (i // cols) * (thumb + caption_h + pad)
            d.rectangle([x, y, x + thumb, y + thumb], outline="red")
            d.text((x, y + 4), f"ERR {exc}"[:40], fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=85)
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _final_row(rec: dict[str, Any], cache: dict[str, Any], overrides: dict[str, dict[str, Any]],
               *, use_vlm: bool, model: str, max_side: int, retries: int = 3) -> dict[str, Any]:
    image_id = rec["image_id"]
    row: dict[str, Any] = {
        "image_id": image_id,
        "path": rec["path"],
        "source": rec["source"],
        "group_id": rec["group_id"],
        "event_id": rec.get("event_id"),
        "frame_index": rec.get("frame_index"),
        "server_received_at_utc_minute": rec.get("server_received_at_utc_minute"),
        "door_state_reported_by_device": rec.get("door_state_reported_by_device"),
        "frame_size_name": rec.get("frame_size_name"),
        "bytes": rec.get("bytes"),
    }
    row.update(image_stats(rec["path"]))

    if use_vlm:
        if image_id in cache and "error" not in cache[image_id]:
            vlm = dict(cache[image_id])
        else:
            vlm, err = None, None
            for attempt in range(retries):
                try:
                    vlm = ask_vlm(rec["path"], model=model, max_side=max_side)
                    break
                except Exception as exc:  # noqa: BLE001 - report, keep going
                    err = f"{type(exc).__name__}: {exc}"[:200]
                    time.sleep(1.5 * (attempt + 1))
            if vlm is None:
                vlm = {"scene_label": "unknown", "person_visible": False, "motion_blur": "none",
                       "upside_down": False, "description": "", "error": err, "vlm_model": model}
            cache[image_id] = vlm
    else:
        vlm = {"scene_label": "unknown", "person_visible": False, "motion_blur": "none",
               "upside_down": False, "description": "", "vlm_model": None}

    for key in ("scene_label", "person_visible", "motion_blur", "upside_down",
                "description", "vlm_model", "vlm_raw", "error"):
        row[key] = vlm.get(key)

    ov = overrides.get(image_id, {})
    row["overrides"] = sorted(ov)
    row["label_source"] = "vlm+override" if ov else ("vlm" if use_vlm else "none")
    row.update(ov)
    return row


def run_audit(out_dir: Path = OUT_DIR, *, use_vlm: bool = True, model: str = VLM_MODEL,
              max_side: int = 1024, limit: int | None = None, only: list[str] | None = None,
              overrides_path: Path = OVERRIDES_PATH) -> dict[str, Any]:
    recs = list_images()
    if only:
        wanted = set(only)
        recs = [r for r in recs if r["image_id"] in wanted or r["group_id"] in wanted]
    if limit:
        recs = recs[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "vlm_cache.json"
    cache = load_cache(cache_path)
    overrides = load_overrides(overrides_path)

    rows: list[dict[str, Any]] = []
    for n, rec in enumerate(recs, 1):
        rows.append(_final_row(rec, cache, overrides, use_vlm=use_vlm, model=model, max_side=max_side))
        if use_vlm:
            cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
        print(f"[{n}/{len(recs)}] {rec['image_id']} -> {rows[-1]['scene_label']} "
              f"sharp512={rows[-1]['sharp_512']:.1f}", flush=True)

    clusters = duplicate_clusters(rows)
    dup_ids = {i for c in clusters for i in c}
    for row in rows:
        row["duplicate_cluster"] = next(
            (f"c{k:02d}" for k, c in enumerate(clusters) if row["image_id"] in c), "")

    summary = {
        "n_images": len(rows),
        "scene_labels": {lab: sum(1 for r in rows if r["scene_label"] == lab) for lab in SCENE_LABELS},
        "person_visible": sum(1 for r in rows if r["person_visible"]),
        "upside_down": sum(1 for r in rows if r["upside_down"]),
        "rotated_90": sum(1 for r in rows if r.get("rotated_90")),
        "motion_blur": {lvl: sum(1 for r in rows if r["motion_blur"] == lvl) for lvl in MOTION_LEVELS},
        "overrides_applied": sum(len(r["overrides"]) for r in rows),
        "n_overridden_images": sum(1 for r in rows if r["overrides"]),
        "vlm_model": model if use_vlm else None,
        "duplicate_clusters": clusters,
        "n_duplicate_images": len(dup_ids),
        "duplicate_max_hamming": DUPLICATE_MAX_DISTANCE,
        "sharp_512_quantiles": _quantiles([r["sharp_512"] for r in rows]),
        "sharp_full_quantiles": _quantiles([r["sharp_full"] for r in rows]),
    }

    fields: list[str] = []
    for row in rows:                      # union of keys, in first-seen order
        for k in row:
            if k not in fields:
                fields.append(k)
    with (out_dir / "audit.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()})
    (out_dir / "audit.json").write_text(json.dumps({"summary": summary, "images": rows}, indent=1))

    for label in SCENE_LABELS:
        subset = [r for r in rows if r["scene_label"] == label]
        if subset:
            contact_sheet(subset, out_dir / f"sheet_{label}.jpg", label)

    print(json.dumps(summary, indent=1))
    return {"summary": summary, "images": rows, "out_dir": out_dir}


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {f"p{int(q * 100)}": round(float(np.quantile(a, q)), 2)
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--no-vlm", action="store_true", help="pixel stats only, no Collective call")
    p.add_argument("--model", default=VLM_MODEL)
    p.add_argument("--max-side", type=int, default=1024, help="image side sent to the VLM")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None, help="subset of image_ids or group_ids")
    p.add_argument("--overrides", default=str(OVERRIDES_PATH))
    args = p.parse_args(argv)

    run_audit(Path(args.out_dir), use_vlm=not args.no_vlm, model=args.model,
              max_side=args.max_side, limit=args.limit, only=args.only,
              overrides_path=Path(args.overrides))


if __name__ == "__main__":
    main()
