"""Unit tests for the catalog + data inventory layer and the audit helpers.

Synthetic fixtures where possible; anything that touches the private raw
package is skipped when the package is not present (CI without data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# If tests/unit/freestyle/__init__.py causes pytest to bind `freestyle` to the
# test directory, extend its __path__ with the real package directory.
_REPO_FREESTYLE = str(Path(__file__).resolve().parents[3] / "freestyle")
if "freestyle" in sys.modules and hasattr(sys.modules["freestyle"], "__path__"):
    if _REPO_FREESTYLE not in sys.modules["freestyle"].__path__:
        sys.modules["freestyle"].__path__.append(_REPO_FREESTYLE)

from freestyle import catalog as catalog_mod  # noqa: E402
from freestyle import data as data_mod  # noqa: E402
from freestyle.audit import (  # noqa: E402
    duplicate_clusters,
    hamming,
    image_stats,
    laplacian_variance,
    parse_vlm_json,
    phash,
)

HAS_RAW = (data_mod.RAW_ROOT / "data/catalog_items.json").exists()
HAS_IMAGES = (data_mod.IMAGES_ROOT / "local/blur/00.jpg").exists()

pytestmark = pytest.mark.skipif(not HAS_RAW, reason="private raw package not present")


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def test_load_catalog_active_only_sorted_and_complete():
    active = catalog_mod.load_catalog(active_only=True)
    assert len(active) == 18, "contract says 18 active cartridges"
    orders = [i["sort_order"] for i in active]
    assert orders == sorted(orders)
    assert all(i["active"] for i in active)
    assert len(catalog_mod.load_catalog(active_only=False)) > len(active)


def test_class_names_append_unknown_last():
    names = catalog_mod.class_names()
    assert names[-1] == catalog_mod.UNKNOWN
    assert names[:18] == [i["id"] for i in catalog_mod.load_catalog(True)]
    assert len(set(names)) == len(names) == 19


def test_category_map_is_coco_style_and_stable():
    m1, m2 = catalog_mod.category_map(), catalog_mod.category_map()
    assert m1 == m2
    assert sorted(m1.values()) == list(range(1, len(m1) + 1))  # 1-based, dense
    assert m1["unknown"] == len(m1)
    assert catalog_mod.CARTRIDGE not in m1, "cartridge is a detection class, not an identity"
    assert catalog_mod.category_id("nns-sweetener") == 1
    assert catalog_mod.category_name(1) == "nns-sweetener"
    with pytest.raises(KeyError):
        catalog_mod.category_id("not-a-cartridge")


def test_describe_includes_name_aliases_colour_shelf():
    text = catalog_mod.describe("barqs-root-beer")  # inactive but in the catalog
    assert "Barq's Root Beer" in text and "BB" in text and "brown" in text and "top" in text
    assert "INACTIVE" in text
    assert "unknown" in catalog_mod.describe(catalog_mod.UNKNOWN)
    assert "super-class" in catalog_mod.describe(catalog_mod.CARTRIDGE)
    with pytest.raises(KeyError):
        catalog_mod.describe("definitely-not-here")
    # every identity class is describable
    assert len(catalog_mod.describe_all()) == len(catalog_mod.class_names())


def test_catalog_path_matches_contract():
    assert catalog_mod.CATALOG_PATH == (
        data_mod.RAW_ROOT / "data/catalog_items.json")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def test_list_images_covers_all_69_with_contract_fields():
    images = data_mod.list_images()
    assert len(images) == 69, f"expected 54 server + 15 local frames, got {len(images)}"
    required = {"image_id", "path", "width", "height", "bytes", "group_id", "source",
                "event_id", "frame_index", "server_received_at_utc_minute"}
    assert required <= set(images[0])
    assert all(set(required) <= set(r) for r in images)
    assert {r["source"] for r in images} == {"server", "local"}
    assert all(r["image_id"].startswith(("island-cabinet-01/", "local/")) for r in images)
    assert not Path(images[0]["image_id"]).name.startswith("images")
    for r in images:
        assert Path(r["path"]).is_absolute() and Path(r["path"]).exists()
        assert r["width"] > 0 and r["height"] > 0 and r["bytes"] > 0


def test_group_ids_follow_the_contract():
    gid = data_mod.groups()
    assert len(gid) == 56  # 52 server openings + blur burst + 3 lone local files
    assert len(gid["local/blur"]) == 12
    assert gid["local/downloads/frame.jpg"] == ["local/downloads/frame.jpg"]
    server = [g for g in gid if not g.startswith("local/")]
    assert all(len(i) in (1, 3) for i in (gid[g] for g in server))
    assert sum(len(v) for v in gid.values()) == 69
    # the three-frame opening really is one group
    assert sum(1 for g in server if len(gid[g]) == 3) == 1


def test_server_and_local_row_counts_and_event_join():
    images = data_mod.list_images()
    server = [r for r in images if r["source"] == "server"]
    local = [r for r in images if r["source"] == "local"]
    assert (len(server), len(local)) == (54, 15)
    assert all(r["event_id"] and r["event"] for r in server)
    assert all(r["event_id"] is None and r["group_id"] for r in local)
    # local_images.csv reports nonsense widths; the measured value must win
    frame = next(r for r in images if r["image_id"] == "local/downloads/frame.jpg")
    assert (frame["width"], frame["height"]) == (2048, 1536)
    assert (frame["csv_width"], frame["csv_height"]) == (769, 2048)


def test_images_in_and_load_splits(tmp_path):
    payload = {
        "version": 1,
        "groups": {"g1": "train", "g2": "test", "g3": "exclude"},
        "images": {"a/1.jpg": "train", "b/1.jpg": "test", "c/1.jpg": "exclude"},
        "negatives": ["c/1.jpg"],
    }
    p = tmp_path / "splits.json"
    p.write_text(json.dumps(payload))
    assert data_mod.images_in("train", p) == ["a/1.jpg"]
    assert data_mod.images_in("test", p) == ["b/1.jpg"]
    assert data_mod.images_in("negatives", p) == ["c/1.jpg"]
    with pytest.raises(ValueError):
        data_mod.images_in("bogus", p)
    with pytest.raises(FileNotFoundError):
        data_mod.images_in("train", tmp_path / "nope.json")
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text('{"hi": 1}')
        data_mod.images_in("train", tmp_path / "bad.json")

    if (data_mod.SPLITS_PATH).exists():  # the real, frozen file
        real = data_mod.load_splits()
        assert real["version"] == 1 and real["unit"] == "group"
        assert set(real["images"].values()) <= set(data_mod.VALID_SPLITS)
        assert set(real["negatives"]) <= set(data_mod.images_in("exclude"))


# ---------------------------------------------------------------------------
# audit helpers (pure, no network)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_IMAGES, reason="images not present")
def test_image_stats_are_sane_and_reproducible():
    path = data_mod.IMAGES_ROOT / "local/downloads/frame.jpg"
    s1, s2 = image_stats(path), image_stats(path)
    assert s1 == s2
    assert 0 <= s1["luma_mean"] <= 255 and 0 <= s1["saturation"] <= 255
    assert s1["sharp_full"] > 0 and s1["sharp_512"] > 0
    assert len(s1["phash"]) == 16
    assert int(s1["phash"], 16) >= 0


def test_sharpness_separates_blur_from_edges():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    blurred = np.full((128, 128), 128, dtype=np.uint8)
    assert laplacian_variance(blurred.astype("float64")) < laplacian_variance(
        noisy.astype("float64"))


@pytest.mark.skipif(not HAS_IMAGES, reason="images not present")
def test_phash_near_duplicates_and_clusters():
    img = lambda name: data_mod.IMAGES_ROOT / name  # noqa: E731
    a = phash(Image.open(img("local/blur/00.jpg")).convert("RGB"))
    b = phash(Image.open(img("local/blur/01.jpg")).convert("RGB"))
    c = phash(Image.open(img("local/bench/vga.jpg")).convert("RGB"))
    assert hamming(a, a) == 0
    assert hamming(a, b) <= 8, "consecutive burst frames must be near-duplicates"
    rows = [{"image_id": "blur/00", "phash": a}, {"image_id": "blur/01", "phash": b},
            {"image_id": "bench", "phash": c}]
    clusters = duplicate_clusters(rows)
    assert clusters == [["blur/00", "blur/01"]] or c == a
    assert duplicate_clusters([{"image_id": "x", "phash": a}]) == []


def test_parse_vlm_json_normalises_and_rejects_garbage():
    out = parse_vlm_json('Sure! {"scene_label":"CABINET-FULL","person_visible":"yes",'
                         '"motion_blur":"MILD","upside_down":0,"description":" two shelves "'
                         '} done')
    assert out["scene_label"] == "cabinet_full"
    assert out["person_visible"] is True and out["upside_down"] is False
    assert out["motion_blur"] == "mild" and out["description"] == "two shelves"

    weird = parse_vlm_json('{"scene_label":"ceiling","motion_blur":"totally"}')
    assert weird["scene_label"] == "unknown" and weird["motion_blur"] == "none"

    with pytest.raises(ValueError):
        parse_vlm_json("no json here")


# ---------------------------------------------------------------------------
# make_splits (synthetic audit; deterministic, no network)
# ---------------------------------------------------------------------------

def _synthetic_env(tmp_path, monkeypatch, rows, image_dims, frozen=None):
    """Point make_splits at a fake audit + fake inventory (and no frozen table)."""
    from freestyle import make_splits as ms

    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"summary": {}, "images": rows}))
    frozen_path = None
    if frozen is not None:
        frozen_path = tmp_path / "splits.frozen.json"
        frozen_path.write_text(json.dumps(frozen))

    def fake_list_images(**kw):
        return [
            {"image_id": i, "width": w, "height": h} for i, (w, h) in image_dims.items()
        ]

    def fake_groups(**kw):
        out: dict[str, list[str]] = {}
        for r in rows:
            gid = r["image_id"].rsplit("/", 1)[0] if r["source"] == "server" else \
                ("local/blur" if r["image_id"].startswith("local/blur/") else r["image_id"])
            out.setdefault(gid, []).append(r["image_id"])
        return {k: sorted(v) for k, v in sorted(out.items())}

    monkeypatch.setattr(ms, "list_images", fake_list_images)
    monkeypatch.setattr(ms, "groups", fake_groups)
    return ms, audit, frozen_path


def _row(image_id, source, label, sharp=15.0, blur="mild", person=False, upside=False):
    return {"image_id": image_id, "source": source, "scene_label": label,
            "sharp_full": sharp, "motion_blur": blur, "person_visible": person,
            "upside_down": upside}


def test_make_splits_is_deterministic_and_leak_free(tmp_path, monkeypatch):
    rows = [_row("ev1/000.jpg", "server", "cabinet_full", sharp=25),
            _row("ev2/000.jpg", "server", "cabinet_partial"),
            _row("ev3/000.jpg", "server", "no_cabinet"),
            _row("ev4/000.jpg", "server", "occluded", sharp=2),
            _row("local/blur/00.jpg", "local", "cabinet_full"),
            _row("local/blur/01.jpg", "local", "cabinet_full"),
            _row("local/downloads/frame.jpg", "local", "cabinet_full", blur="none", sharp=30)]
    dims = {r["image_id"]: (1600, 1200) for r in rows}
    ms, audit, _ = _synthetic_env(tmp_path, monkeypatch, rows, dims)

    a = ms.build_splits(audit_path=audit, frozen_path=None)
    b = ms.build_splits(audit_path=audit, frozen_path=None)
    assert a == b, "seed 0 must reproduce the same file"

    assert a["images"]["local/downloads/frame.jpg"] == "val"
    assert a["images"]["local/blur/00.jpg"] == "train" == a["images"]["local/blur/01.jpg"]
    assert a["images"]["ev3/000.jpg"] == "exclude" == a["images"]["ev4/000.jpg"]
    assert set(a["negatives"]) == {"ev3/000.jpg", "ev4/000.jpg"}
    # no group straddles two splits -- checked with the real membership, not with an
    # id-prefix guess (the two local groups share the "loc" prefix on purpose).
    members = ms.groups()
    assert members["local/blur"] == ["local/blur/00.jpg", "local/blur/01.jpg"]
    assert members["local/downloads/frame.jpg"] == ["local/downloads/frame.jpg"]
    assert a["groups"]["local/blur"] != a["groups"]["local/downloads/frame.jpg"]
    for gid, split in a["groups"].items():
        for iid in members[gid]:
            got = a["images"][iid]
            assert got == split or got == "exclude", (gid, iid, split, got)
    assert all(r for r in a["reasons"].values())
    assert set(a["images"].values()) <= set(data_mod.VALID_SPLITS)


def test_make_splits_test_strata_are_named(tmp_path, monkeypatch):
    """One group per required stratum, so every rule must show up in `reasons`.

    The fixture used to be self-contradictory (the frame commented as "the sharp
    full 2048x1536" one was labelled cabinet_partial at 1600x1200, and every
    cabinet_full frame was below the sharpness cut), so the sharp-full rule could
    never fire and the file named one stratum too few.
    """
    rows = [
        _row("ev0/000.jpg", "server", "cabinet_full", sharp=30, blur="none"),
        _row("ev1/000.jpg", "server", "cabinet_partial", sharp=25, blur="none"),
        _row("ev2/000.jpg", "server", "cabinet_full", sharp=2, blur="severe"),
        _row("ev3/000.jpg", "server", "cabinet_full", sharp=18, blur="none", upside=True),
        _row("ev4/000.jpg", "server", "cabinet_full", sharp=15, blur="none"),
        _row("ev5/000.jpg", "server", "cabinet_full", sharp=15, blur="none", person=True),
        _row("ev6/000.jpg", "server", "cabinet_full", sharp=15, blur="none", person=True),
    ] + [_row(f"ev{i}/000.jpg", "server", "cabinet_full", sharp=15, blur="none")
         for i in (7, 8, 9)]
    dims = {"ev0/000.jpg": (2048, 1536), "ev1/000.jpg": (2048, 1536),
            "ev2/000.jpg": (2048, 1536), "ev3/000.jpg": (2048, 1536),
            "ev4/000.jpg": (1600, 1200)}
    dims |= {f"ev{i}/000.jpg": (2048, 1536) for i in (5, 6, 7, 8, 9)}
    ms, audit, _ = _synthetic_env(tmp_path, monkeypatch, rows, dims)
    out = ms.build_splits(audit_path=audit, frozen_path=None)
    test_groups = [g for g, s in out["groups"].items() if s == "test"]
    reasons = [out["reasons"][g] for g in test_groups]
    text = " ".join(reasons)
    for want in ("sharp full", "partial", "blurry", "upside-down", "1600x1200",
                 "2048x1536", "person"):
        assert want in text, (want, reasons)
    # one distinct group per stratum: no group was counted twice to satisfy two rules
    strata = [r for r in reasons if r.startswith("stratum:")]
    assert len(strata) == len(set(strata)) == 7, reasons
    assert out["images"]["ev0/000.jpg"] == "test"      # the sharp full 2048x1536 frame
    assert out["images"]["ev2/000.jpg"] == "test"      # the severe-blur frame
    assert out["images"]["ev3/000.jpg"] == "test"      # the upside-down frame
    assert out["images"]["ev4/000.jpg"] == "test"      # the only 1600x1200 frame


def test_make_splits_frozen_table_pins_the_assignment(tmp_path, monkeypatch):
    """A frozen table overrides the draw; a stale one is an error, not a silent drop."""
    rows = [_row(f"ev{i}/000.jpg", "server", "cabinet_full", sharp=25, blur="none")
            for i in range(6)]
    dims = {r["image_id"]: (1600, 1200) for r in rows}
    ms, audit, _ = _synthetic_env(tmp_path, monkeypatch, rows, dims)
    drawn = ms.build_splits(audit_path=audit, frozen_path=None)

    frozen = json.loads(json.dumps(drawn))
    frozen["groups"]["ev0"] = "exclude"
    frozen["images"]["ev0/000.jpg"] = "exclude"
    fpath = tmp_path / "splits.frozen.json"
    fpath.write_text(json.dumps(frozen))

    out = ms.build_splits(audit_path=audit, frozen_path=fpath)
    assert out["groups"]["ev0"] == "exclude", "frozen assignment must win over the draw"
    assert out["images"]["ev0/000.jpg"] == "exclude"
    assert out["groups"] == frozen["groups"] and out["images"] == frozen["images"]
    assert "frozen_drift" not in out, "the report stays out of splits.json"
    drift: dict = {}
    ms.build_splits(audit_path=audit, frozen_path=fpath, drift=drift)
    assert drift["groups"]["ev0"] == {"generated": drawn["groups"]["ev0"], "frozen": "exclude"}
    assert drift["images"]["ev0/000.jpg"] == {"generated": drawn["images"]["ev0/000.jpg"],
                                             "frozen": "exclude"}

    stale = json.loads(json.dumps(frozen))
    del stale["images"]["ev5/000.jpg"]
    stale["images"]["ev9/000.jpg"] = "train"
    spath = tmp_path / "stale.frozen.json"
    spath.write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="does not cover"):
        ms.build_splits(audit_path=audit, frozen_path=spath)


# ---------------------------------------------------------------------------
# the real, frozen artefacts on disk (skipped without the private package)
# ---------------------------------------------------------------------------

AUDIT_JSON = data_mod.REPO_ROOT / "data/derived/audit/audit.json"
FROZEN_PATH = data_mod.REPO_ROOT / "data/splits.frozen.json"
UPSIDE_DOWN_FRAMES = (
    "island-cabinet-01/ab8c7fc6-943c-4c3d-b95f-3a71f0966ec6/043.jpg",
    "island-cabinet-01/de19c989-a538-47d7-9173-a9fce2a49e7e/032.jpg",
)


@pytest.mark.skipif(not FROZEN_PATH.exists(), reason="frozen split not present")
def test_make_splits_reproduces_the_frozen_split(tmp_path, monkeypatch):
    from freestyle import make_splits as ms

    frozen_text = FROZEN_PATH.read_text()
    frozen = json.loads(frozen_text)
    out = ms.build_splits()
    assert out["groups"] == frozen["groups"], "the frozen group assignment is official"
    assert out["images"] == frozen["images"], "the frozen image assignment is official"
    assert out["negatives"] == frozen["negatives"], "the frozen negatives list is official"
    assert out["reasons"] == frozen["reasons"]
    assert json.dumps(out, indent=1, sort_keys=True) + "\n" == frozen_text, \
        "make_splits must reproduce the frozen file byte for byte"
    ms.main(["--check"])

    # when the draw moves (the audit labels changed after the review) it is the
    # frozen table, not the generator, that pins the file
    drawn = ms.build_splits(frozen_path=None)
    drift: dict = {}
    ms.build_splits(drift=drift)
    same = all(drawn[k] == out[k] for k in ("groups", "images", "negatives"))
    assert same or drift, "frozen override must be observable as drift"


@pytest.mark.skipif(not (FROZEN_PATH.exists() and AUDIT_JSON.exists()),
                    reason="frozen split / audit output not present")
def test_frozen_split_negatives_are_the_non_cabinet_frames():
    frozen = data_mod.load_splits(FROZEN_PATH)
    audit = {r["image_id"]: r for r in json.loads(AUDIT_JSON.read_text())["images"]}
    neg = frozen["negatives"]
    assert neg == sorted(set(neg)), "negatives are a set, listed once, sorted"
    assert set(neg) <= set(data_mod.images_in("exclude")), "negatives are never trained on"
    assert {audit[i]["scene_label"] for i in neg} == {"no_cabinet", "occluded"}
    # every non-cabinet frame in the corpus is reachable as rejection-test material
    non_cabinet = {i for i, r in audit.items() if r["scene_label"] in ("no_cabinet", "occluded")}
    assert set(neg) == non_cabinet
    assert len(neg) == 29


def test_audit_overrides_record_the_upside_down_frames():
    from freestyle.audit import OVERRIDES_PATH, load_overrides

    ov = load_overrides(OVERRIDES_PATH)
    for iid in UPSIDE_DOWN_FRAMES:
        assert ov[iid]["upside_down"] is True, iid
    # the sideways-90 note on the second frame was a misread (it is a 180 flip)
    assert ov[UPSIDE_DOWN_FRAMES[1]]["rotated_90"] is False
    assert ov["island-cabinet-01/c40dc230-fc17-4edc-969a-1fea7043807a/019.jpg"]["rotated_90"] is True


@pytest.mark.skipif(not AUDIT_JSON.exists(), reason="audit output not present")
def test_audit_output_agrees_with_the_override_table():
    from freestyle.audit import load_overrides

    rows = {r["image_id"]: r for r in json.loads(AUDIT_JSON.read_text())["images"]}
    for iid, fields in load_overrides().items():
        for field, value in fields.items():
            assert rows[iid][field] == value, (iid, field)


def test_audit_merge_reuses_the_vlm_cache_without_calling_the_model(tmp_path, monkeypatch):
    """The audit merge must re-derive every row from the cache alone, no model call."""
    from freestyle import audit as audit_mod

    img = tmp_path / "000.jpg"
    Image.new("RGB", (96, 64), (40, 90, 140)).save(img)
    rec = {"image_id": "evA/000.jpg", "path": str(img), "source": "server",
           "group_id": "evA", "event_id": "evA"}
    monkeypatch.setattr(audit_mod, "list_images", lambda **kw: [rec])

    def boom(*a, **kw):
        raise AssertionError("the VLM must not be called when the cache has the row")

    monkeypatch.setattr(audit_mod, "ask_vlm", boom)
    cache = {"evA/000.jpg": {"scene_label": "cabinet_full", "person_visible": False,
                             "motion_blur": "mild", "upside_down": False,
                             "description": "two shelves", "vlm_model": "qwen3-8-flash-next"}}
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "vlm_cache.json").write_text(json.dumps(cache))
    ov = tmp_path / "overrides.json"
    ov.write_text(json.dumps({"evA/000.jpg": {"upside_down": True, "_why": "human view"}}))

    res = audit_mod.run_audit(out_dir, overrides_path=ov)
    row = res["images"][0]
    assert row["scene_label"] == "cabinet_full" and row["motion_blur"] == "mild"
    assert row["upside_down"] is True and row["overrides"] == ["upside_down"]
    assert row["label_source"] == "vlm+override"
    assert res["summary"]["upside_down"] == 1
    assert json.loads((out_dir / "vlm_cache.json").read_text()) == cache, "cache untouched"
