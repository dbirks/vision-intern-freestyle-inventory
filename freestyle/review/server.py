"""Local ground-truth review/annotation server (stdlib only, no build step).

A human verifies/corrects model-proposed cartridge boxes and identities; the result
is HUMAN-VERIFIED ground truth in COCO contract format under ``data/gt/``.

Run:
    uv run python -m freestyle.review.server \
        --proposals data/labels/<file>.json --out data/gt/<name>.json \
        --images <image-id-list-file | split name | all> --port 8765 --host 127.0.0.1

Endpoints (JSON unless noted):
    GET  /                      UI (static/index.html)
    GET  /static/<file>         UI assets
    GET  /api/classes           class list (canonical order, colour, hotkey)
    GET  /api/images            image list + status + box counts
    GET  /api/image/<idx>/meta  one image's metadata
    GET  /api/ann/<idx>         one image's annotations + image attributes
    PUT  /api/ann/<idx>         replace annotations/attrs for one image (autosave)
    GET  /img/<idx>             raw original image bytes (also accepts the image id)

Only files under the images root (default ``data/raw/freestyle-vision-data/images``)
are ever served, and only for images in the current list.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import threading
import urllib.parse
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_ROOT = REPO_ROOT / "data" / "raw" / "freestyle-vision-data" / "images"
DEFAULT_CATALOG = REPO_ROOT / "data" / "raw" / "freestyle-vision-data" / "data" / "catalog_items.json"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

CARTRIDGE = "cartridge"
UNKNOWN = "unknown"
CARTRIDGE_ID = 90
UNKNOWN_ID = 91
HUMAN_SOURCE = "human"
VERIFIED_PREFIX = "human-verified:"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BODY_BYTES = 8 * 1024 * 1024

# Keys owned by the UI (navigation/verify/flags); never handed out as class hotkeys.
RESERVED_KEYS = {"v", "n", "x", "c", "z", "g", "t", "b", "a", "s", "m"}
HOTKEY_LETTERS = "qwertyuiopasdfghjklzxcvbnm"

COLOR_HEX = {
    "orange": "#ff8c1a",
    "red": "#e3342f",
    "green": "#2f9e44",
    "tan": "#c8a26b",
    "yellow": "#f2c200",
    "silver": "#b8bec7",
    "grey": "#868e96",
    "gray": "#868e96",
    "black": "#232629",
    "maroon": "#7a1f2b",
    "dark red": "#5c1017",
    "white": "#f1f3f5",
    "blue": "#2b6cb0",
    "brown": "#8b5e34",
    "clear": "#a5d8ff",
}
PALETTE = [
    "#e3342f", "#ff8c1a", "#f2c200", "#2f9e44", "#1098ad", "#2b6cb0",
    "#6b46c1", "#d6336c", "#7a1f2b", "#8b5e34", "#232629", "#b8bec7",
]


def _color_for(name: str, hint: str) -> str:
    h = (hint or "").strip().lower()
    if h.startswith("#"):
        return h
    if h in COLOR_HEX:
        return COLOR_HEX[h]
    return PALETTE[zlib.crc32(name.encode("utf-8")) % len(PALETTE)]


def load_classes(catalog_path: Path = DEFAULT_CATALOG) -> list[dict[str, Any]]:
    """Canonical class list: 18 active catalog ids, then ``unknown``, then ``cartridge``."""
    raw = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    items = raw.get("items") if isinstance(raw, dict) else raw
    items = [it for it in (items or []) if isinstance(it, dict)]
    active = [it for it in items if it.get("active")]
    active.sort(key=lambda it: (it.get("sort_order") or 0, str(it.get("id"))))
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, it in enumerate(active, start=1):
        slug = str(it.get("id"))
        if not slug or slug in seen:
            continue
        seen.add(slug)
        entries.append(
            {
                "name": slug,
                "display": str(it.get("name") or slug),
                "id": i,
                "color": _color_for(slug, str(it.get("color_hint") or "")),
                "shelf": it.get("shelf_hint"),
                "kind": it.get("kind"),
                "generic": False,
            }
        )
    entries.append(
        {
            "name": UNKNOWN,
            "display": "unknown",
            "id": UNKNOWN_ID,
            "color": "#868e96",
            "shelf": None,
            "kind": None,
            "generic": True,
        }
    )
    entries.append(
        {
            "name": CARTRIDGE,
            "display": "cartridge (generic)",
            "id": CARTRIDGE_ID,
            "color": "#f1f3f5",
            "shelf": None,
            "kind": None,
            "generic": True,
        }
    )
    pool = [str(d) for d in [1,2,3,4,5,6,7,8,9,0]] + [c for c in HOTKEY_LETTERS if c not in RESERVED_KEYS]
    for i, entry in enumerate(entries):
        entry["hotkey"] = pool[i] if i < len(pool) else ""
        entry["index"] = i
    return entries


# --------------------------------------------------------------------------- images


def _read_id_list(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("images", "image_ids", "ids", "file_names"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:  # {"images": {image_id: split}} style mapping
                data = list(data.get("images", []))
        return [str(v) for v in data if str(v).strip()]
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def resolve_images(
    spec: str | None,
    images_root: Path = DEFAULT_IMAGES_ROOT,
    splits_path: Path = DEFAULT_SPLITS,
) -> list[str]:
    """Resolve ``--images`` (list file / split name / 'all') to sorted image ids."""
    images_root = Path(images_root)
    if not images_root.is_dir():
        raise FileNotFoundError(f"images root not found: {images_root}")

    def scan() -> list[str]:
        return sorted(
            p.relative_to(images_root).as_posix()
            for p in images_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

    if spec is None or spec.strip().lower() in {"", "all", "*"}:
        return scan()

    candidate = Path(spec)
    if not candidate.is_absolute():
        candidate = candidate  # relative to cwd (repo root when run via uv)
    if candidate.is_file():
        ids = _read_id_list(candidate)
    else:
        if not Path(splits_path).is_file():
            raise FileNotFoundError(
                f"--images {spec!r} is not a file and {splits_path} does not exist "
                "(cannot resolve a split name without splits.json)"
            )
        splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
        mapping = splits.get("images") if isinstance(splits, dict) else None
        if not isinstance(mapping, dict):
            raise ValueError(f"{splits_path} has no 'images' mapping")
        ids = sorted(str(k) for k, v in mapping.items() if str(v) == spec)

    known = set(scan())
    missing = [i for i in ids if i not in known]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image id(s) not found under {images_root}, e.g. {missing[:3]}"
        )
    return ids


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the file header without re-encoding (PIL if available)."""
    try:
        from PIL import Image

        with Image.open(path) as im:  # lazy header read only
            return int(im.width), int(im.height)
    except Exception:
        return _size_from_header(path)


def _size_from_header(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        i, n = 2, len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seglen = int.from_bytes(data[i + 2 : i + 4], "big")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return int.from_bytes(data[i + 7 : i + 9], "big"), int.from_bytes(data[i + 5 : i + 7], "big")
            i += 2 + seglen
    return 0, 0


def group_id_for(image_id: str) -> str:
    parent = PurePosixPath(image_id).parent.as_posix()
    return parent if parent and parent != "." else image_id


# --------------------------------------------------------------------------- store


class ValidationError(ValueError):
    """Raised for a malformed PUT payload (-> HTTP 400)."""


def _as_floats(value: Any, count: int, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValidationError(f"{what} must be a list of {count} numbers")
    out: list[float] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValidationError(f"{what} must be numeric")
        f = float(v)
        if not math.isfinite(f):
            raise ValidationError(f"{what} must be finite")
        out.append(f)
    return out


def _as_bool(value: Any, default: bool = False) -> bool:
    return bool(value) if value is not None else default


def _bbox_equal(a: list[float], b: list[float]) -> bool:
    return all(abs(x - y) <= 0.01 for x, y in zip(a, b))


def _quad_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    fa = [float(v) for pt in a for v in pt]
    fb = [float(v) for pt in b for v in pt]
    return len(fa) == len(fb) and all(abs(x - y) <= 0.01 for x, y in zip(fa, fb))


def normalize_source(original: Any) -> str:
    base = str(original or HUMAN_SOURCE)
    if base.startswith(VERIFIED_PREFIX):
        base = base[len(VERIFIED_PREFIX) :]
    return base or HUMAN_SOURCE


def resolve_source(prev_source: Any, changed: bool) -> str:
    """Human-edited boxes are 'human'; accepted-unchanged proposals are 'human-verified:<src>'."""
    base = normalize_source(prev_source)
    if base.startswith(HUMAN_SOURCE):
        return HUMAN_SOURCE
    return HUMAN_SOURCE if changed else f"{VERIFIED_PREFIX}{base}"


class ReviewStore:
    """In-memory review state for one image list, autosaved to ``--out`` as COCO."""

    def __init__(
        self,
        image_ids: list[str],
        classes: list[dict[str, Any]],
        out_path: Path,
        images_root: Path,
    ) -> None:
        self.images_root = Path(images_root).resolve()
        self.out_path = Path(out_path)
        self.classes = classes
        self.by_name = {c["name"]: c for c in classes}
        self.id_to_name = {c["id"]: c["name"] for c in classes}
        self._lock = threading.RLock()
        self.images: list[dict[str, Any]] = []
        for idx, image_id in enumerate(image_ids):
            path = self.images_root / PurePosixPath(image_id)
            width, height = image_size(path) if path.is_file() else (0, 0)
            self.images.append(
                {
                    "idx": idx,
                    "id": idx + 1,  # COCO image id
                    "image_id": image_id,
                    "path": path,
                    "width": width,
                    "height": height,
                    "group_id": group_id_for(image_id),
                    "verified": False,
                    "scene": None,
                    "reviewer_note": "",
                    "annotations": [],
                }
            )
        self._by_coco_id = {img["id"]: img for img in self.images}
        self._by_image_id = {img["image_id"]: img for img in self.images}
        self._next_ann_id = 1

    # -- loading ------------------------------------------------------------

    def _fuzzy_image(self, key: str) -> dict[str, Any] | None:
        """Match a proposals file_name that is a suffix/basename of one of our image ids."""
        key = key.strip().lstrip("/")
        if not key:
            return None
        needle = key if "/" in key else f"/{key}"
        hits = [im for im in self.images if im["image_id"].endswith(needle)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            print(f"warning: {key!r} matches {len(hits)} images; ignoring", file=sys.stderr)
        return None

    def load_coco(self, path: Path, *, as_proposals: bool, human_verified: bool = False) -> int:
        """Load COCO: proposals (source kept) or a resumed ``--out`` (human state)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValidationError("COCO file must be a JSON object")
        cats = data.get("categories") or []
        cat_names: dict[int, str] = {}
        for cat in cats:
            if isinstance(cat, dict) and cat.get("id") is not None:
                cat_names[int(cat["id"])] = str(cat.get("name") or "")
        unknown_categories: set[str] = set()
        skipped = 0
        n_applied = 0
        with self._lock:
            for img in data.get("images") or []:
                if not isinstance(img, dict):
                    continue
                key = str(img.get("file_name") or img.get("coco_url") or img.get("id") or "")
                target = self._by_image_id.get(key) or self._by_coco_id.get(_safe_int(img.get("id")))
                if target is None:
                    target = self._fuzzy_image(key)
                if target is None:
                    continue
                if human_verified:
                    attrs = img.get("attributes")
                    if isinstance(attrs, dict):
                        target["verified"] = _as_bool(attrs.get("verified"))
                        target["scene"] = attrs.get("scene")
                        target["reviewer_note"] = str(attrs.get("reviewer_note") or "")
                elif target["annotations"]:
                    continue  # proposals never overwrite existing human/resumed boxes
                anns: list[dict[str, Any]] = []
                target["annotations"] = anns
                for ann in data.get("annotations") or []:
                    if not isinstance(ann, dict) or _safe_int(ann.get("image_id")) != _safe_int(img.get("id")):
                        continue
                    name = cat_names.get(_safe_int(ann.get("category_id")) or -1)
                    if name is None:
                        name = str(ann.get("category_name") or UNKNOWN)
                    if name not in self.by_name:
                        unknown_categories.add(name)
                        name = UNKNOWN
                    try:
                        a = self._internalize(ann, name, human=human_verified)
                    except ValidationError:
                        skipped += 1
                        continue
                    anns.append(a)
                    n_applied += 1
            if human_verified:
                used = [a["id"] for im in self.images for a in im["annotations"] if a.get("id")]
                self._next_ann_id = max([self._next_ann_id, *used, 0]) + 1
        if unknown_categories and as_proposals:
            print(
                f"warning: {len(unknown_categories)} category name(s) not in catalog "
                f"(mapped to {UNKNOWN}): {sorted(unknown_categories)[:5]}",
                file=sys.stderr,
            )
        if skipped:
            print(f"warning: skipped {skipped} malformed annotation(s)", file=sys.stderr)
        return n_applied

    def _internalize(self, ann: dict[str, Any], name: str, *, human: bool) -> dict[str, Any]:
        bbox = _as_floats(ann.get("bbox"), 4, "bbox")
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        source = str(attrs.get("source") or ("vlm:unknown" if not human else HUMAN_SOURCE))
        if human and not source.startswith(HUMAN_SOURCE) and not source.startswith(VERIFIED_PREFIX):
            source = f"{VERIFIED_PREFIX}{normalize_source(source)}"
        quad = ann.get("quad")
        if quad is not None:
            quad = [[float(v) for v in pt] for pt in quad]
        return {
            "id": int(ann.get("id") or 0) or None,
            "category": name,
            "bbox": bbox,
            "quad": quad,
            "attributes": {
                "truncated": _as_bool(attrs.get("truncated")),
                "occluded": _as_bool(attrs.get("occluded")),
                "identity_visible": _as_bool(attrs.get("identity_visible"), default=True),
                "source": source,
            },
        }

    # -- views --------------------------------------------------------------

    def _status(self, img: dict[str, Any]) -> str:
        if img["verified"]:
            return "verified"
        if img["annotations"] or (img["reviewer_note"] or img["scene"]):
            return "in_progress"
        return "todo"

    def classes_payload(self) -> dict[str, Any]:
        return {"classes": self.classes}

    def images_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "out": str(self.out_path),
                "images": [
                    {
                        "idx": img["idx"],
                        "image_id": img["image_id"],
                        "group_id": img["group_id"],
                        "width": img["width"],
                        "height": img["height"],
                        "n_boxes": len(img["annotations"]),
                        "n_human": sum(
                            1 for a in img["annotations"] if a["attributes"]["source"].startswith(HUMAN_SOURCE)
                        ),
                        "verified": bool(img["verified"]),
                        "scene": img["scene"],
                        "status": self._status(img),
                    }
                    for img in self.images
                ],
            }

    def meta_payload(self, idx: int) -> dict[str, Any]:
        img = self.images[idx]
        return {
            "idx": idx,
            "image_id": img["image_id"],
            "group_id": img["group_id"],
            "width": img["width"],
            "height": img["height"],
            "verified": bool(img["verified"]),
            "scene": img["scene"],
            "reviewer_note": img["reviewer_note"],
            "n_boxes": len(img["annotations"]),
            "src": f"/img/{idx}",
        }

    def ann_payload(self, idx: int) -> dict[str, Any]:
        img = self.images[idx]
        with self._lock:
            return {
                "idx": idx,
                "image_id": img["image_id"],
                "width": img["width"],
                "height": img["height"],
                "verified": bool(img["verified"]),
                "scene": img["scene"],
                "reviewer_note": img["reviewer_note"],
                "annotations": [json.loads(json.dumps(a)) for a in img["annotations"]],
            }

    # -- mutation -----------------------------------------------------------

    def apply_ann_update(self, idx: int, payload: dict[str, Any]) -> dict[str, Any]:
        """Replace one image's annotations + attributes, recompute sources, autosave."""
        if not isinstance(payload, dict):
            raise ValidationError("payload must be a JSON object")
        with self._lock:
            img = self.images[idx]
            prev = {a["id"]: a for a in img["annotations"] if a.get("id")}
            raw_anns = payload.get("annotations")
            if raw_anns is None:
                raw_anns = []
            if not isinstance(raw_anns, list):
                raise ValidationError("'annotations' must be a list")
            anns: list[dict[str, Any]] = []
            for item in raw_anns:
                if not isinstance(item, dict):
                    raise ValidationError("annotation must be an object")
                category = str(item.get("category") or "")
                if category not in self.by_name:
                    raise ValidationError(f"unknown category {category!r}")
                x, y, w, h = _as_floats(item.get("bbox"), 4, "bbox")
                if w < 0 or h < 0:
                    raise ValidationError("bbox width/height must be >= 0")
                attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
                quad = item.get("quad")
                if quad is not None:
                    if not isinstance(quad, list) or len(quad) != 4:
                        raise ValidationError("quad must contain 4 points")
                    quad = [_as_floats(pt, 2, "quad point") for pt in quad]
                aid = item.get("id")
                old = prev.get(aid) if isinstance(aid, int) else None
                if old is None:
                    source = HUMAN_SOURCE
                else:
                    changed = (
                        old["category"] != category
                        or not _bbox_equal(old["bbox"], [x, y, w, h])
                        or not _quad_equal(old.get("quad"), quad)
                        or any(
                            old["attributes"][k] != _as_bool(attrs.get(k), default=v)
                            for k, v in (
                                ("truncated", False),
                                ("occluded", False),
                                ("identity_visible", True),
                            )
                        )
                    )
                    source = resolve_source(old["attributes"].get("source"), changed)
                if not old:
                    # keep a client-assigned id when it's positive; otherwise allocate one
                    if isinstance(aid, int) and not isinstance(aid, bool) and aid > 0:
                        pass
                    else:
                        aid = self._next_ann_id
                        self._next_ann_id += 1
                anns.append(
                    {
                        "id": int(aid),
                        "category": category,
                        "bbox": [x, y, w, h],
                        "quad": quad,
                        "attributes": {
                            "truncated": _as_bool(attrs.get("truncated")),
                            "occluded": _as_bool(attrs.get("occluded")),
                            "identity_visible": _as_bool(attrs.get("identity_visible"), default=True),
                            "source": source,
                        },
                    }
                )
            seen: set[int] = set()
            for a in anns:
                if a["id"] in seen:
                    a["id"] = self._next_ann_id
                    self._next_ann_id += 1
                seen.add(a["id"])
            used_now = [a["id"] for im in self.images for a in im["annotations"]]
            self._next_ann_id = max(self._next_ann_id, max(used_now, default=0) + 1)
            img["annotations"] = anns

            if "verified" in payload:
                img["verified"] = _as_bool(payload.get("verified"))
            if "scene" in payload:
                scene = payload.get("scene")
                img["scene"] = None if scene in ("", None) else str(scene)
            if "reviewer_note" in payload:
                img["reviewer_note"] = str(payload.get("reviewer_note") or "")
            self.save()
            return self.ann_payload(idx)

    # -- output -------------------------------------------------------------

    def to_coco(self) -> dict[str, Any]:
        with self._lock:
            categories = [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "supercategory": "cartridge" if c["generic"] else "flavor",
                    "attributes": {"display": c["display"], "color": c["color"], "shelf_hint": c.get("shelf")},
                }
                for c in self.classes
            ]
            images, annotations = [], []
            for img in self.images:
                images.append(
                    {
                        "id": img["id"],
                        "file_name": img["image_id"],
                        "coco_url": "",
                        "width": img["width"],
                        "height": img["height"],
                        "group_id": img["group_id"],
                        "attributes": {
                            "verified": bool(img["verified"]),
                            "scene": img["scene"],
                            "reviewer_note": img["reviewer_note"],
                            "n_boxes": len(img["annotations"]),
                        },
                    }
                )
                for a in img["annotations"]:
                    x, y, w, h = a["bbox"]
                    ann = {
                        "id": a["id"],
                        "image_id": img["id"],
                        "category_id": self.by_name[a["category"]]["id"],
                        "bbox": [round(v, 3) for v in a["bbox"]],
                        "area": round(w * h, 3),
                        "iscrowd": 0,
                        "attributes": dict(a["attributes"]),
                    }
                    if a.get("quad") is not None:
                        ann["quad"] = a["quad"]
                    annotations.append(ann)
            verified = sum(1 for i in self.images if i["verified"])
            return {
                "info": {
                    "description": "Freestyle cartridge ground truth (human-verified)",
                    "human_verified_images": verified,
                    "total_images": len(self.images),
                    "annotation_contract": "docs/freestyle/ARCHITECTURE.md",
                },
                "licenses": [],
                "images": images,
                "categories": categories,
                "annotations": annotations,
            }

    def save(self) -> None:
        """Atomic write: temp file in the same directory + rename."""
        data = self.to_coco()
        out = self.out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=f".{out.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, out)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- http


def safe_join(root: Path, rel: str) -> Path | None:
    """Join ``rel`` under ``root`` refusing traversal; None when it escapes or is missing."""
    rel = urllib.parse.unquote(rel).replace("\\", "/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    root_r = Path(root).resolve()
    candidate = (root_r / Path(*parts)).resolve()
    try:
        candidate.relative_to(root_r)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore  # set on the class by serve()
    protocol_version = "HTTP/1.1"
    server_version = "FreestyleReview/1.0"

    # -- helpers ------------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _image_path(self, rest: str) -> Path | None:
        store = self.store
        rest = urllib.parse.unquote(rest)
        head = rest.strip("/")
        if head.isdigit():
            idx = int(head)
            if not 0 <= idx < len(store.images):
                return None
            return Path(store.images[idx]["path"])
        candidate = safe_join(store.images_root, rest)
        if candidate is None:
            return None
        try:
            image_id = candidate.relative_to(store.images_root).as_posix()
        except ValueError:
            return None
        if image_id not in store._by_image_id:
            return None
        return candidate

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parts = urllib.parse.urlsplit(self.path)
        path = parts.path
        store = self.store
        try:
            if path in ("/", "/index.html"):
                self._static("index.html")
            elif path.startswith("/static/"):
                self._static(path[len("/static/") :])
            elif path == "/api/classes":
                self._json(200, store.classes_payload())
            elif path == "/api/images":
                self._json(200, store.images_payload())
            elif path.startswith("/api/image/") and path.endswith("/meta"):
                idx = _route_idx(path, r"^/api/image/(\d+)/meta$")
                self._send_idx(idx, store.meta_payload)
            elif path.startswith("/api/ann/"):
                idx = _route_idx(path, r"^/api/ann/(\d+)$")
                self._send_idx(idx, store.ann_payload)
            elif path.startswith("/img/"):
                self._image(path[len("/img/") :])
            else:
                self._json(404, {"error": "not found", "path": path})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except IndexError:
            self._json(404, {"error": "no such image index"})
        except ValidationError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_PUT(self) -> None:  # noqa: N802
        parts = urllib.parse.urlsplit(self.path)
        store = self.store
        try:
            idx = _route_idx(parts.path, r"^/api/ann/(\d+)$")
        except (IndexError, ValidationError):
            return self._json(404, {"error": "not found"})
        length = _safe_int(self.headers.get("Content-Length")) or 0
        if length <= 0:
            return self._json(400, {"error": "empty body"})
        if length > MAX_BODY_BYTES:
            return self._json(413, {"error": f"body too large (>{MAX_BODY_BYTES} bytes)"})
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": f"invalid JSON: {exc}"})
        try:
            self._send_idx(idx, lambda i: store.apply_ann_update(i, payload))
        except ValidationError as exc:
            self._json(400, {"error": str(exc)})

    def _send_idx(self, idx: int | None, fn: Any) -> None:
        if idx is None:
            return self._json(404, {"error": "not found"})
        if not 0 <= idx < len(self.store.images):
            return self._json(404, {"error": "no such image index"})
        self._json(200, fn(idx))

    def _static(self, name: str) -> None:
        path = safe_join(STATIC_DIR, name)
        if path is None:
            return self._json(404, {"error": "not found"})
        body = path.read_bytes()
        self._send(200, body, CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    def _image(self, rest: str) -> None:
        path = self._image_path(rest)
        if path is None:
            return self._json(404, {"error": "image not available"})
        self._send(
            200,
            path.read_bytes(),
            CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            {"Cache-Control": "private, max-age=86400"},
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("REVIEW_APP_QUIET"):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def _route_idx(path: str, pattern: str) -> int | None:
    m = re.match(pattern, urllib.parse.unquote(path))
    return int(m.group(1)) if m else None


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str, port: int, store: ReviewStore) -> None:
        handler = type("BoundReviewHandler", (ReviewHandler,), {"store": store})
        super().__init__((host, port), handler)


def build_store(
    images_spec: str | None,
    proposals: Path | None,
    out_path: Path,
    images_root: Path = DEFAULT_IMAGES_ROOT,
    catalog: Path = DEFAULT_CATALOG,
    splits: Path = DEFAULT_SPLITS,
) -> ReviewStore:
    image_ids = resolve_images(images_spec, images_root, splits)
    store = ReviewStore(image_ids, load_classes(catalog), out_path, images_root)
    if out_path.exists():
        n = store.load_coco(out_path, as_proposals=False, human_verified=True)
        print(f"resumed {n} annotation(s) from {out_path}", file=sys.stderr)
    if proposals is not None and Path(proposals).is_file():
        n = store.load_coco(proposals, as_proposals=True)
        print(f"loaded {n} proposed annotation(s) from {proposals}", file=sys.stderr)
    elif proposals is not None:
        print(f"warning: proposals file {proposals} not found; starting empty", file=sys.stderr)
    if not out_path.exists():
        store.save()  # create the file so a crash still keeps the image list
    return store


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local ground-truth review/annotation app")
    p.add_argument("--proposals", type=Path, default=None, help="COCO json of model proposals (contract format)")
    p.add_argument("--out", type=Path, required=True, help="COCO json to write under data/gt/ (resumed if present)")
    p.add_argument("--images", default="all", help="image-id list file, split name, or 'all'")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (local) or 0.0.0.0 (tailnet)")
    p.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    store = build_store(
        args.images, args.proposals, args.out, args.images_root, args.catalog, args.splits
    )
    httpd = ReviewServer(args.host, args.port, store)
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(
        f"review app: http://{shown_host}:{httpd.server_address[1]}  "
        f"({len(store.images)} images, {sum(1 for i in store.images if i['verified'])} verified) "
        f"-> {store.out_path}",
        file=sys.stderr,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        store.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
