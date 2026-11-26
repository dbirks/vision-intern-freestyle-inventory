"""Unit tests for the LLM Collective client and the probe helpers (no network)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

# See tests/unit/freestyle/test_catalog_data.py for why this path shim exists.
_REPO_FREESTYLE = str(Path(__file__).resolve().parents[3] / "freestyle")
if "freestyle" in sys.modules and hasattr(sys.modules["freestyle"], "__path__"):
    if _REPO_FREESTYLE not in sys.modules["freestyle"].__path__:
        sys.modules["freestyle"].__path__.append(_REPO_FREESTYLE)

import freestyle.collective as collective  # noqa: E402
import freestyle.probe_collective as probe  # noqa: E402


# ----------------------------------------------------------------------
# collective helpers
# ----------------------------------------------------------------------

def test_resized_dims_passthrough():
    assert collective.resized_dims((100, 50), None) == (100, 50)
    assert collective.resized_dims((100, 50), 0) == (100, 50)
    assert collective.resized_dims((100, 50), 100) == (100, 50)


def test_resized_dims_downscale_only():
    assert collective.resized_dims((2048, 1536), 1024) == (1024, 768)
    assert collective.resized_dims((512, 1536), 1024) == (341, 1024)
    # never upscales
    assert collective.resized_dims((100, 50), 1000) == (100, 50)


def test_image_to_data_url_jpeg_and_alpha_png():
    rgb = Image.new("RGB", (40, 20), "red")
    url = collective.image_to_data_url(rgb)
    assert url.startswith("data:image/jpeg;base64,")

    rgba = Image.new("RGBA", (40, 20), (0, 0, 255, 128))
    url = collective.image_to_data_url(rgba)
    assert url.startswith("data:image/png;base64,")

    # resized only when max_side is given and image exceeds it
    big = Image.new("RGB", (100, 50), "black")
    url = collective.image_to_data_url(big, max_side=50)
    assert url.startswith("data:image/jpeg;base64,")


def test_resolve_secret_env_interpolation(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_X", "abc123")
    assert collective.resolve_secret("$TEST_SECRET_X") == "abc123"
    assert collective.resolve_secret("${TEST_SECRET_X}") == "abc123"
    assert collective.resolve_secret("prefix-$TEST_SECRET_X-suffix") == "prefix-abc123-suffix"


def test_load_image_source_accepts_pil_and_path():
    im = Image.new("RGB", (10, 10), "green")
    out = collective.load_image_source(im)
    assert out.size == (10, 10) and out.mode == "RGB"


def test_chat_image_retries_429_then_raises_within_budget():
    import openai

    def make_err():
        return openai.RateLimitError(
            "429 backend at capacity",
            response=Mock(status_code=429, headers={}),
            body=None,
        )

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise make_err()

    start = time.monotonic()
    with pytest.raises(openai.RateLimitError):
        collective.chat_image(
            "fake-model", Image.new("RGB", (16, 16)), "prompt",
            retry_budget_s=1.5, client=FakeClient(),
        )
    assert time.monotonic() - start < 10  # budget honoured, not ignored


# ----------------------------------------------------------------------
# probe helpers
# ----------------------------------------------------------------------

def test_parse_response_json_bare_text():
    assert probe.parse_response_json("no json here") is None
    assert probe.parse_response_json("[{\"bbox_2d\": [1, 2, 3, 4]}]") == [
        {"bbox_2d": [1, 2, 3, 4]}
    ]
    assert probe.parse_response_json('prefix {"top": 3, "bottom": 2} suffix') == {"top": 3, "bottom": 2}


def test_detect_coordinate_convention():
    # beyond 1000 -> absolute
    assert probe.detect_coordinate_convention([[10, 20, 3000, 1500]], 2048, 1536) == "absolute"
    # must be normalized when it cannot exceed the image under either convention
    assert probe.detect_coordinate_convention([[100, 100, 200, 200]], 2048, 1536) == "normalized"
    assert probe.detect_coordinate_convention([], 2048, 1536) == "unknown"


def test_to_pixel_boxes_normalizes_and_clamps():
    boxes, used = probe.to_pixel_boxes([[0, 0, 1000, 1000]], "normalized", 2048, 1536)
    assert used == "normalized"
    assert boxes == [[0, 0, 2048, 1536]]

    boxes, _ = probe.to_pixel_boxes([[2048, 1536, 4096, 3072]], "absolute", 2048, 1536)
    assert boxes == []  # clamped to x2==x1, dropped by the validity filter
    boxes, _ = probe.to_pixel_boxes([[0, 0, 2048, 1536]], "absolute", 2048, 1536)
    assert boxes == [[0, 0, 2048, 1536]]


def test_parse_count_json():
    assert probe.parse_count_json('{"top": 9, "bottom": 9}') == {"top": 9, "bottom": 9}
    assert probe.parse_count_json('top: 8, bottom: 9') == {"top": 8, "bottom": 9}
    assert probe.parse_count_json('{"top": "8 boxes"}') == {"top": 8}
    assert probe.parse_count_json("totally not json") is None


def _synthetic_products():
    return [
        {"id": "nns-sweetener", "name": "NNS", "aliases": ["NNS"], "color_hint": "", "shelf": "top"},
        {"id": "hi-c", "name": "Hi-C", "aliases": [], "color_hint": "red", "shelf": "top"},
        {"id": "flavor-cherry", "name": "Cherry", "aliases": [], "color_hint": "dark red", "shelf": "bottom"},
        {"id": "coca-cola-part1", "name": "Coca-Cola Part 1", "aliases": ["Coke"], "color_hint": "red", "shelf": "bottom"},
    ]


def test_match_product_id_name_alias():
    products = _synthetic_products()
    assert probe.match_product("coca-cola-part1", products) == "coca-cola-part1"
    assert probe.match_product("This is the Hi-C cartridge", products) == "hi-c"
    assert probe.match_product("Coke", products) == "coca-cola-part1"
    assert probe.match_product("unrelated words", products) is None


def test_match_product_word_boundaries():
    products = _synthetic_products()
    # "hi-c" must not match "chicken"; "coke" must not match "cokesicle"
    assert probe.match_product("chicken", products) is None
    assert probe.match_product("cokesicle", products) is None
    assert probe.match_product("coke", products) == "coca-cola-part1"


def test_build_boxes_prompt_variants():
    abs_p = probe.build_boxes_prompt("absolute", 2048, 1536)
    norm_p = probe.build_boxes_prompt("normalized", 2048, 1536)
    assert "2048" in abs_p and "1536" in abs_p
    assert "1000" in norm_p and "2048" not in norm_p
    with pytest.raises(ValueError):
        probe.build_boxes_prompt("bogus", 2048, 1536)


def test_upsert_call_replaces_by_key():
    results: dict = {}
    key_rec = {"model": "m", "image_id": "img", "max_side": None, "stage": "count", "v": 1}
    probe._upsert_call(results, "detect", key_rec)
    probe._upsert_call(results, "detect", {**key_rec, "v": 2})
    assert len(results["detect"]) == 1
    assert results["detect"][0]["v"] == 2


def test_draw_overlay_writes_png(tmp_path):
    img = Image.new("RGB", (100, 60), "blue")
    out = tmp_path / "ov.png"
    probe.draw_overlay(img, [[10, 10, 50, 40]], out, labels=["hi-c"])
    assert out.exists()
    saved = Image.open(out)
    assert saved.size == (100, 60)


def test_extract_boxes_from_body_text():
    text = '[{"bbox_2d": [0, 0, 100, 100], "label": "cartridge"}]'
    objects = probe.extract_boxes(text)
    assert len(objects) == 1
    assert objects[0]["bbox_2d"] == [0, 0, 100, 100]

    # garbage tolerated
    assert probe.extract_boxes("no boxes at all") == []
