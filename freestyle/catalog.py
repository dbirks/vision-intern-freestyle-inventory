"""Cartridge catalog: the single source of class lists and display hints.

Reads ``data/raw/freestyle-vision-data/data/catalog_items.json`` (the catalog as
configured on the deployment).  Only ``active: true`` rows are classes; the
contract adds one extra class, ``unknown``.

Public API
----------
load_catalog(active_only=True) -> list[dict]
class_names()                  -> list[str]   active ids in sort_order, then 'unknown'
category_map()                 -> dict[str,int]  stable COCO ids starting at 1
category_id(name) / category_name(id)
describe(catalog_id)           -> str
cartridge_class()              -> 'cartridge' (generic super-class, not in category_map)
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data/raw/freestyle-vision-data/data/catalog_items.json"

UNKNOWN = "unknown"
#: Generic super-class used by class-agnostic detectors (see ARCHITECTURE.md).
#: It is a detection class, not a catalog identity, so it is deliberately not
#: part of :func:`category_map`.
CARTRIDGE = "cartridge"


@functools.lru_cache(maxsize=8)
def _raw_catalog(path: str | None = None) -> tuple[dict[str, Any], ...]:
    """Load the catalog once per path; returns the ``items`` tuple."""
    p = Path(path) if path else CATALOG_PATH
    data = json.loads(p.read_text())
    items = data["items"] if isinstance(data, dict) else data
    return tuple(items)


def load_catalog(active_only: bool = True, path: str | None = None) -> list[dict]:
    """Return catalog rows, sorted by ``sort_order`` then ``id``.

    ``active_only=True`` (the default) keeps only the cartridges that the
    deployment currently serves; those 18 rows are the identity classes.
    """
    items = [dict(i) for i in _raw_catalog(path)]
    if active_only:
        items = [i for i in items if i.get("active")]
    items.sort(key=lambda i: (i.get("sort_order", 0), str(i.get("id", ""))))
    return items


@functools.lru_cache(maxsize=8)
def _class_names_cached(path: str | None = None) -> tuple[str, ...]:
    names = [str(i["id"]) for i in load_catalog(True, path)]
    return tuple(names) + (UNKNOWN,)


def class_names(path: str | None = None) -> list[str]:
    """Identity classes: active catalog ids in ``sort_order``, then ``unknown``."""
    return list(_class_names_cached(path))


@functools.lru_cache(maxsize=8)
def category_map(path: str | None = None) -> dict[str, int]:
    """Stable ``class name -> COCO category_id`` map, ids starting at 1.

    Ids follow :func:`class_names` order, so they only change if the catalog's
    active rows change; a human then has to regenerate GT.  ``cartridge`` is
    not included (it is a detection super-class, never a GT identity).
    """
    return {name: i + 1 for i, name in enumerate(class_names(path))}


def category_id(name: str, path: str | None = None) -> int:
    """COCO category id for a class name (raises ``KeyError`` on unknown names)."""
    return category_map(path)[name]


def category_name(category_id_: int, path: str | None = None) -> str:
    """Class name for a COCO category id (raises ``KeyError`` on unknown ids)."""
    inv = {v: k for k, v in category_map(path).items()}
    return inv[category_id_]


def _fmt_list(values: Any, empty: str = "-") -> str:
    if isinstance(values, (list, tuple)):
        values = [str(v) for v in values if str(v).strip()]
    elif values is None:
        values = []
    else:
        values = [str(values)]
    return ", ".join(values) if values else empty


def describe(catalog_id: str, path: str | None = None) -> str:
    """One-line human description: name, aliases, colour, shelf.

    ``unknown`` and ``cartridge`` describe themselves; anything unknown to the
    catalog raises ``KeyError``.
    """
    if catalog_id == UNKNOWN:
        return "unknown (identity not determinable from the pixels)"
    if catalog_id == CARTRIDGE:
        return "cartridge (generic detection super-class, any identity)"
    item = next((i for i in load_catalog(False, path) if i.get("id") == catalog_id), None)
    if item is None:
        raise KeyError(f"not in catalog: {catalog_id!r}")
    kind = "active" if item.get("active") else "INACTIVE"
    colour = _fmt_list(item.get("color_hint"))
    shelf = str(item.get("shelf_hint") or "-")
    return (
        f"{item.get('name', catalog_id)} [{catalog_id}] "
        f"aliases={_fmt_list(item.get('aliases'))} colour={colour} "
        f"shelf={shelf} kind={item.get('kind', '-')} {kind}"
    )


def describe_all(path: str | None = None) -> list[str]:
    """Descriptions of every identity class, in :func:`class_names` order."""
    return [describe(name, path) for name in class_names(path)]


__all__ = [
    "CARTRIDGE",
    "CATALOG_PATH",
    "UNKNOWN",
    "category_id",
    "category_map",
    "category_name",
    "class_names",
    "describe",
    "describe_all",
    "load_catalog",
]
