"""Unit tests for the local ground-truth review server.

Spins up a real server on an ephemeral port against a tiny synthetic images root
(two 64x48 JPEGs) and exercises the HTTP contract end to end.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from pathlib import Path

import pytest

from freestyle.review import server as review_server


def _make_image(path: Path, color=(200, 30, 30), size=(64, 48)) -> None:
    from PIL import Image

    Image.new("RGB", size, color).save(path, format="JPEG")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def review(tmp_path):
    images_root = tmp_path / "images"
    (images_root / "grp").mkdir(parents=True)
    ids = ["grp/a.jpg", "grp/b.jpg"]
    for i, image_id in enumerate(ids):
        _make_image(images_root / image_id, color=(200, 30 * (i + 1), 30))
    out = tmp_path / "gt" / "out.json"

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "hi-c", "name": "Hi-C", "color_hint": "red", "active": True, "sort_order": 1},
                    {"id": "sprite", "name": "Sprite", "color_hint": "green", "active": True, "sort_order": 2},
                    {"id": "inactive", "name": "Nope", "active": False},
                ]
            }
        )
    )

    proposals = tmp_path / "proposals.json"
    proposals.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "grp/a.jpg", "width": 64, "height": 48},
                    {"id": 2, "file_name": "grp/b.jpg", "width": 64, "height": 48},
                ],
                "categories": [
                    {"id": 1, "name": "hi-c"},
                    {"id": 2, "name": "sprite"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 20, 15], "attributes": {"source": "vlm:test"}},
                    {"id": 2, "image_id": 2, "category_id": 2, "bbox": [10, 10, 30, 20], "attributes": {"source": "vlm:test"}},
                    {"id": 3, "image_id": 1, "category_id": 2, "bbox": [30, 20, 15, 12], "attributes": {"source": "vlm:test"}},
                ],
            }
        )
    )

    store = review_server.build_store(
        None,  # 'all' -> both images under images_root
        proposals,
        out,
        images_root,
        catalog,
        tmp_path / "splits.json",
    )
    # images_root has 2 images; proposals reference both by file_name
    assert len(store.images) == 2
    port = _free_port()
    httpd = review_server.ReviewServer("127.0.0.1", port, store)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            method,
            path,
            body=json.dumps(body) if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        ctype = resp.getheader("Content-Type") or ""
        payload = None
        if "application/json" in ctype:
            try:
                payload = json.loads(raw or b"null")
            except json.JSONDecodeError:
                payload = None
        return resp.status, payload, raw

    yield {"store": store, "out": out, "images_root": images_root, "port": port, "request": request}

    httpd.shutdown()
    httpd.server_close()


def test_classes_endpoint(review):
    status, payload, _ = review["request"]("GET", "/api/classes")
    assert status == 200
    names = [c["name"] for c in payload["classes"]]
    assert names[:2] == ["hi-c", "sprite"]
    assert "unknown" in names
    assert "inactive" not in names
    for c in payload["classes"]:
        assert c["color"].startswith("#")
        assert c["hotkey"]


def test_images_and_meta(review):
    status, payload, _ = review["request"]("GET", "/api/images")
    assert status == 200
    assert [im["image_id"] for im in payload["images"]] == ["grp/a.jpg", "grp/b.jpg"]
    assert payload["images"][0]["n_boxes"] == 2
    assert payload["images"][0]["status"] in ("in_progress", "todo")

    status, payload, _ = review["request"]("GET", "/api/image/0/meta")
    assert status == 200
    assert payload["image_id"] == "grp/a.jpg"
    assert payload["width"] == 64 and payload["height"] == 48


def test_img_serves_raw_bytes(review):
    status, _, raw = review["request"]("GET", "/img/0")
    assert status == 200
    expected = (review["images_root"] / "grp/a.jpg").read_bytes()
    assert raw == expected


def test_ann_get_and_round_trip(review):
    status, payload, _ = review["request"]("GET", "/api/ann/0")
    assert status == 200
    assert payload["image_id"] == "grp/a.jpg"
    assert len(payload["annotations"]) == 2
    assert payload["annotations"][0]["attributes"]["source"] == "vlm:test"
    assert payload["verified"] is False

    # create a new box (explicit client id) -> source human, id preserved
    status, payload, _ = review["request"](
        "PUT",
        "/api/ann/0",
        {
            "annotations": [
                {"id": 9, "category": "sprite", "bbox": [32, 24, 10, 10], "attributes": {"source": "vlm:test"}},
            ],
            "verified": True,
            "scene": "cabinet",
            "reviewer_note": "checked",
        },
    )
    assert status == 200
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["id"] == 9
    assert payload["annotations"][0]["attributes"]["source"] == "human"
    assert payload["verified"] is True

    # accepted unchanged -> human-verified:<orig source>
    status, payload, _ = review["request"](
        "PUT",
        "/api/ann/0",
        {
            "annotations": [
                {
                    "id": a["id"],
                    "category": a["category"],
                    "bbox": list(a["bbox"]),
                    "attributes": dict(a["attributes"]),
                }
                for a in payload["annotations"]
            ],
            "verified": True,
            "scene": "cabinet",
            "reviewer_note": "checked",
        },
    )
    assert status == 200
    assert payload["annotations"][0]["attributes"]["source"] == "human"  # not unchanged (attributes were already human)

    status, payload, _ = review["request"]("GET", "/api/ann/0")
    # now accept a proposal unchanged via the original ids from the proposals file
    status, payload_read, _ = review["request"]("GET", "/api/ann/1")
    status, payload2, _ = review["request"](
        "PUT",
        "/api/ann/1",
        {
            "annotations": [
                {
                    "id": a["id"],
                    "category": a["category"],
                    "bbox": list(a["bbox"]),
                    "attributes": {
                        "truncated": False,
                        "occluded": False,
                        "identity_visible": True,
                        "source": "vlm:test",
                    },
                }
                for a in payload_read["annotations"]
            ],
            "verified": True,
        },
    )
    assert status == 200
    assert payload2["annotations"][0]["attributes"]["source"] == "human-verified:vlm:test"


def test_autosave_and_resume(review, tmp_path):
    review["request"](
        "PUT",
        "/api/ann/1",
        {
            "annotations": [
                {"category": "hi-c", "bbox": [1, 2, 8, 9], "attributes": {"source": "vlm:test"}},
            ],
            "verified": True,
            "scene": "cabinet",
            "reviewer_note": "ok",
        },
    )
    out = review["out"]
    assert out.exists()
    data = json.loads(out.read_text())
    images = {img["file_name"]: img for img in data["images"]}
    assert images["grp/b.jpg"]["attributes"]["verified"] is True
    assert images["grp/b.jpg"]["attributes"]["scene"] == "cabinet"
    assert images["grp/b.jpg"]["attributes"]["reviewer_note"] == "ok"
    assert len(data["annotations"]) == 3  # all three boxes across both images survive
    human_anns = [a for a in data["annotations"] if a["attributes"]["source"].startswith("human")]
    assert human_anns

    # resume from the out file: boxes and image attributes survive a reload
    store2 = review_server.ReviewStore(
        ["grp/a.jpg", "grp/b.jpg"],
        review["store"].classes,
        out,
        review["images_root"],
    )
    store2.load_coco(out, as_proposals=False, human_verified=True)
    assert store2.images[1]["verified"] is True
    assert store2.images[1]["scene"] == "cabinet"
    assert store2.images[1]["reviewer_note"] == "ok"
    assert len(store2.images[1]["annotations"]) == 1


def test_traversal_rejected(review):
    for path in (
        "/img/..%2F..%2Fpyproject.toml",
        "/img/..%2F..%2F..%2Fetc%2Fpasswd",
        "/img/grp/../../pyproject.toml",
        "/img/..%2F..%2F..%2F..%2Fetc%2Fpasswd",
    ):
        status, payload, _ = review["request"]("GET", path)
        assert status == 404, (path, status, payload)

    # also: an image outside the current list is not served
    status, _, _ = review["request"]("GET", "/img/999")
    assert status == 404


def test_bad_put_rejected(review):
    status, payload, _ = review["request"](
        "PUT",
        "/api/ann/0",
        {"annotations": [{"category": "not-a-real-class", "bbox": [1, 2, 3, 4]}]},
    )
    assert status == 400
    assert "unknown category" in payload["error"]

    status, payload, _ = review["request"](
        "PUT", "/api/ann/0", {"annotations": [{"category": "hi-c", "bbox": [1, 2, "three", 4]}]}
    )
    assert status == 400


def test_status_transitions(review):
    status, payload, _ = review["request"]("GET", "/api/ann/0")
    assert payload["verified"] is False
    # unverified + annotations -> in progress
    status, payload, _ = review["request"](
        "PUT", "/api/ann/0", {"annotations": [{"category": "hi-c", "bbox": [0, 0, 5, 5]}], "verified": True}
    )
    assert payload["verified"] is True
    status, payload, _ = review["request"]("GET", "/api/images")
    statuses = {im["image_id"]: im["status"] for im in payload["images"]}
    assert statuses["grp/a.jpg"] == "verified"
