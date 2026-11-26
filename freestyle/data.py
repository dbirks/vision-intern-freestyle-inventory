"""Image inventory, group ids and split loading for the Freestyle corpus.

* ``RAW_ROOT``   — the read-only extracted package (``data/raw/freestyle-vision-data``).
* ``list_images()`` — one record per photograph, joining ``data/images.csv``
  (server frames) and ``data/local_images.csv`` (workstation copies), plus the
  per-opening record from ``data/events.json`` where one exists.
* ``load_splits()`` / ``images_in(split)`` — read ``data/splits.json``.

Ids follow the contract in ``docs/freestyle/ARCHITECTURE.md``:
``image_id`` is the path relative to ``RAW_ROOT/images`` (``/`` kept, no
``images/`` prefix); ``group_id`` is the door-opening uuid for server frames,
``local/blur`` for the 12-frame burst, and the image id itself for every other
local file (each of those is its own independent group).

Width/height are read from the image files themselves: the ``width``/``height``
columns of ``data/local_images.csv`` are unreliable (they report e.g.
769x1600 for a 1600x1200 frame — see docs/freestyle/DATA_AUDIT.md), so the CSV
values are kept as ``csv_width``/``csv_height`` for traceability only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "data/raw/freestyle-vision-data"
IMAGES_ROOT = RAW_ROOT / "images"
SPLITS_PATH = REPO_ROOT / "data/splits.json"

SERVER_CSV = RAW_ROOT / "data/images.csv"
LOCAL_CSV = RAW_ROOT / "data/local_images.csv"
EVENTS_JSON = RAW_ROOT / "data/events.json"

#: The 12-frame motion-blur burst is a single capture session, not 12 samples.
BLUR_ORIGIN = "blur"

VALID_SPLITS = ("train", "val", "test", "exclude")


def image_id_from_path(relative_path: str) -> str:
    """``images/local/downloads/frame.jpg`` -> ``local/downloads/frame.jpg``."""
    rp = str(relative_path).replace("\\", "/").strip("/")
    return rp[len("images/"):] if rp.startswith("images/") else rp


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _load_events() -> dict[str, dict[str, Any]]:
    if not EVENTS_JSON.exists():
        return {}
    events = json.loads(EVENTS_JSON.read_text())
    return {str(e["event_id"]): e for e in events if e.get("event_id")}


def _int(value: Any) -> int | None:
    try:
        s = str(value).strip()
    except Exception:
        return None
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _measure(path: Path) -> tuple[int | None, int | None]:
    """Real pixel dimensions, read from the file header (no decode)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def _group_id(image_id: str, row: dict[str, str]) -> str:
    event = (row.get("event_id") or "").strip()
    if event:
        return event
    if image_id.startswith(f"local/{BLUR_ORIGIN}/"):
        return f"local/{BLUR_ORIGIN}"
    return image_id


def list_images(
    *,
    include_missing: bool = False,
    measure: bool = True,
) -> list[dict[str, Any]]:
    """Every photograph in the package, sorted by ``image_id``.

    Each record:

    ``image_id``, ``path`` (absolute), ``width``, ``height``, ``bytes``,
    ``group_id``, ``source`` (``server`` | ``local``), plus the joined event
    fields (``event_id``, ``event_trigger``, ``frame_index``,
    ``server_received_at_utc_minute``, ``door_state_reported_by_device``,
    ``frame_size_name``, ``sha256``, ``event`` …).  ``event`` is the whole
    opening record for server frames (``None`` for local ones).

    ``include_missing=True`` also returns rows whose file is absent (they carry
    ``missing=True``); by default they are dropped.
    """
    events = _load_events()
    out: list[dict[str, Any]] = []

    for row in _read_csv(SERVER_CSV):
        out.append(_record(row, "server", events))
    for row in _read_csv(LOCAL_CSV):
        out.append(_record(row, "local", events))

    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for rec in out:
        if rec["image_id"] in seen:
            continue
        seen.add(rec["image_id"])
        if rec["missing"] and not include_missing:
            continue
        if measure and not rec["missing"]:
            w, h = _measure(Path(rec["path"]))
            if w:
                rec["width"], rec["height"] = w, h
        records.append(rec)

    records.sort(key=lambda r: r["image_id"])
    return records


def _record(row: dict[str, str], source: str, events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    image_id = image_id_from_path(row.get("relative_path", ""))
    path = IMAGES_ROOT / image_id
    missing = not path.exists()

    event_id = (row.get("event_id") or "").strip() or None
    event = events.get(event_id) if event_id else None

    rec: dict[str, Any] = {
        "image_id": image_id,
        "path": str(path),
        "source": source,
        "group_id": _group_id(image_id, row),
        "missing": missing,
        "bytes": _int(row.get("bytes")) or (path.stat().st_size if not missing else None),
        "width": _int(row.get("width")),
        "height": _int(row.get("height")),
        "csv_width": _int(row.get("width")),
        "csv_height": _int(row.get("height")),
        "sha256": (row.get("sha256") or None),
        "event_id": event_id,
        "device_id": (row.get("device_id") or (event or {}).get("device_id") or None),
        "event_trigger": row.get("event_trigger") or (event or {}).get("trigger") or None,
        "frame_index": _int(row.get("frame_index")),
        "server_received_at_utc_minute": row.get("server_received_at_utc_minute")
        or (event or {}).get("server_received_at_utc_minute")
        or None,
        "door_state_reported_by_device": (row.get("door_state_reported_by_device") or None),
        "frame_size_name": row.get("frame_size_name") or None,
        "jpeg_quality": _int(row.get("jpeg_quality")),
        "session_end_reason": row.get("session_end_reason") or None,
        "origin_on_workstation": row.get("origin_on_workstation") or None,
        "note": row.get("note") or None,
        "door_open_duration_ms": (event or {}).get("door_open_duration_ms"),
        "event": event,
    }
    return rec


def index_images(**kwargs: Any) -> dict[str, dict[str, Any]]:
    """``image_id -> record`` for :func:`list_images`."""
    return {r["image_id"]: r for r in list_images(**kwargs)}


def groups(**kwargs: Any) -> dict[str, list[str]]:
    """``group_id -> [image_id, ...]`` (sorted)."""
    out: dict[str, list[str]] = {}
    for rec in list_images(**kwargs):
        out.setdefault(rec["group_id"], []).append(rec["image_id"])
    return {k: sorted(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def load_splits(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``data/splits.json`` (raises ``FileNotFoundError`` with a hint)."""
    p = Path(path) if path else SPLITS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run `uv run python -m freestyle.make_splits` first."
        )
    data = json.loads(p.read_text())
    if "images" not in data or "groups" not in data:
        raise ValueError(f"{p} is not a splits file (needs 'groups' and 'images').")
    return data


def images_in(split: str, path: str | Path | None = None) -> list[str]:
    """Image ids assigned to ``split`` (sorted).

    ``split`` ∈ ``train|val|test|exclude``, or the pseudo-split ``negatives``
    (no-cabinet / fully occluded frames kept for rejection testing).
    """
    data = load_splits(path)
    if split == "negatives":
        return sorted(data.get("negatives", []))
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS + ('negatives',)}, got {split!r}")
    return sorted(i for i, s in data["images"].items() if s == split)


def group_of(image_id: str) -> str | None:
    """Group id of one image (from the raw CSVs, not from the split file)."""
    return index_images().get(image_id, {}).get("group_id")


__all__ = [
    "IMAGES_ROOT",
    "RAW_ROOT",
    "REPO_ROOT",
    "SPLITS_PATH",
    "VALID_SPLITS",
    "group_of",
    "groups",
    "image_id_from_path",
    "images_in",
    "index_images",
    "list_images",
    "load_splits",
]
