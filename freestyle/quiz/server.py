"""Phone-first ground-truth quiz server (stdlib + PIL, no build step).

One question at a time: a whole-frame count check, then one zoomed crop per
proposed box. Every answer is appended to a JSONL log the instant it is made,
so a phone session survives a dropped connection or a locked screen.

Run:
    uv run python -m freestyle.quiz.server \
        --queue data/gt/quiz_queue.json --answers data/gt/quiz_answers.jsonl \
        --host 127.0.0.1 --port 8766

Endpoints (JSON unless noted):
    GET  /                       UI (static/index.html)
    GET  /static/<file>          UI assets
    GET  /api/state              progress, next unanswered index, undo availability
    GET  /api/queue              full queue + answers (resume payload)
    GET  /img/frame?i=<img idx>  frame downscaled to <=1280 px, model boxes numbered
    GET  /img/crop?i=<q idx>     one padded crop from ORIGINAL pixels, upscaled
    GET  /img/context?i=<q idx>  small frame thumbnail with that box highlighted
    POST /api/answer             append one answer line
    POST /api/undo               drop the last answer line

Images are only ever read from ``--images-root`` and only for image ids present
in the queue; anything that traverses out (``..``, absolute paths) is refused.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from freestyle.quiz.build_queue import COUNT, CROP, validate_queue
from freestyle.review.server import CONTENT_TYPES, safe_join

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_ROOT = REPO_ROOT / "data" / "raw" / "freestyle-vision-data" / "images"
STATIC_DIR = Path(__file__).resolve().parent / "static"

FRAME_MAX_SIDE = 1280
FRAME_FULL_SIDE = 1600
CONTEXT_WIDTH = 320
CROP_MIN_SIDE = 720
CROP_MAX_SIDE = 1500
CROP_MAX_SCALE = 6.0
CROP_PAD = 0.15
MAX_BODY_BYTES = 256 * 1024
JPEG_QUALITY = 88
CACHE_ITEMS = 512
OPEN_IMAGES = 4

# Answers accepted from the UI, keyed by question kind.
COUNT_RESULTS = {"correct", "fix", "not_cabinet", "skip"}
CROP_RESULTS = {"correct", "class", "part", "not_cartridge", "two_plus", "partial", "skip"}


class QuizError(ValueError):
    """Bad request the UI should surface (-> HTTP 400)."""


# --------------------------------------------------------------------------- images


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit(image: Image.Image, max_side: int) -> Image.Image:
    """Return the image unchanged, or LANCZOS-downscaled so its long side is max_side."""
    side = max(image.size)
    if side <= max_side:
        return image
    scale = max_side / float(side)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS
    )


def _encode(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def draw_frame(image: Image.Image, boxes: list[dict[str, Any]], max_side: int = FRAME_MAX_SIDE) -> bytes:
    """Downscaled copy of a frame with thin, numbered model boxes drawn on it."""
    canvas = _fit(image, max_side).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    # sizes as a fraction of the rendered width: numbers stay legible on a phone
    stroke = max(2, round(canvas.width * 0.0022))
    label_size = max(18, round(canvas.width * 0.045))
    font = _font(label_size)
    for box in boxes:
        factor = canvas.width / float(max(1, image.width))
        x0, y0, x1, y1 = (v * factor for v in box["bbox"])
        color = box.get("color") or "#c8102e"
        rect = [x0, y0, x1 - 1, y1 - 1]
        # dark halo first so a pale box stays visible on any background
        draw.rectangle(rect, outline="#1b1720", width=stroke * 2 + 1)
        draw.rectangle(rect, outline=color, width=stroke)
        text = str(box.get("number", ""))
        if not text:
            continue
        pad = max(2, round(label_size * 0.12))
        tw = draw.textlength(text, font=font)
        tx, ty = x0, max(0.0, y0 - label_size - 2 * pad - 2)
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + label_size + 2 * pad], fill="#1b1720", outline=color)
        draw.text((tx + pad, ty + pad - 2), text, fill="#ffffff", font=font)
    return _encode(canvas)


def make_crop(image: Image.Image, bbox: list[float], pad: float = CROP_PAD) -> bytes:
    """Padded crop of ORIGINAL pixels, upscaled so the short side fills a phone."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if x1 <= x0 or y1 <= y0:
        raise QuizError("empty box")
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    rect = (
        max(0, math.floor(x0 - dx)),
        max(0, math.floor(y0 - dy)),
        min(image.width, math.ceil(x1 + dx)),
        min(image.height, math.ceil(y1 + dy)),
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        raise QuizError("box does not intersect the image")
    crop = image.crop(rect)
    short, long = min(crop.size), max(crop.size)
    # a 40 px box only holds so much detail; beyond CROP_MAX_SCALE the browser is
    # doing the same blur for free
    scale = max(1.0, min(CROP_MIN_SIDE / float(short), CROP_MAX_SIDE / float(long), CROP_MAX_SCALE))
    if scale > 1.0:
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS
        )
    return _encode(crop)


def make_context(image: Image.Image, bbox: list[float], highlight: str = "#c8102e") -> bytes:
    """Small frame thumbnail with one box highlighted."""
    canvas = _fit(image, CONTEXT_WIDTH * 2).convert("RGB")
    scale = canvas.width / float(image.width)
    box = [v * scale for v in bbox]
    shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle(box, fill=(255, 255, 255, 150))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shade).convert("RGB")
    ImageDraw.Draw(canvas).rectangle(
        [box[0], box[1], box[2] - 1, box[3] - 1], outline=highlight, width=max(2, round(canvas.width / 100))
    )
    return _encode(_fit(canvas, CONTEXT_WIDTH * 2))


# --------------------------------------------------------------------------- store


class QuizStore:
    """Queue + append-only answer log, with an in-memory image cache."""

    def __init__(self, queue: dict[str, Any], answers_path: Path, images_root: Path = DEFAULT_IMAGES_ROOT):
        validate_queue(queue)
        self.queue = queue
        self.questions: list[dict[str, Any]] = queue["questions"]
        self.by_qid = {q["qid"]: q for q in self.questions}
        self.classes: list[dict[str, Any]] = queue.get("classes") or []
        self.class_names = {c["name"] for c in self.classes}
        self.answers_path = Path(answers_path)
        self.images_root = Path(images_root).resolve()
        self._lock = threading.RLock()
        self._answers: list[dict[str, Any]] = []
        self._answered: dict[str, dict[str, Any]] = {}
        self._images: OrderedDict[str, Image.Image] = OrderedDict()
        self._render: OrderedDict[str, bytes] = OrderedDict()
        self.by_image: dict[str, list[dict[str, Any]]] = {}
        for question in self.questions:
            self.by_image.setdefault(question["image_id"], []).append(question)
        #: frames in queue order; ``/img/frame?i=N`` indexes this list
        self.image_ids: list[str] = list(dict.fromkeys(q["image_id"] for q in self.questions))
        for image_id in self.by_image:
            path = self.images_root / PurePosixPath(image_id)
            try:
                path.resolve().relative_to(self.images_root)
            except (ValueError, OSError):
                raise QuizError(f"queue image id escapes the images root: {image_id!r}")
        self.load_answers()

    # -- answers ------------------------------------------------------------

    def load_answers(self) -> int:
        """Replay the JSONL log; the last answer for a question wins."""
        with self._lock:
            self._answers, self._answered = [], {}
            if self.answers_path.is_file():
                for line in self.answers_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and record.get("qid") in self.by_qid:
                        self._answers.append(record)
                        self._answered[str(record["qid"])] = record
            return len(self._answers)

    def _write_log(self, records: list[dict[str, Any]]) -> None:
        """Atomic rewrite of the whole log (used by undo)."""
        path = self.answers_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def answer(self, qid: str, answer: Any) -> dict[str, Any]:
        """Validate and atomically append one answer line; returns the record."""
        question = self.by_qid.get(str(qid))
        if question is None:
            raise QuizError(f"unknown question id {qid!r}")
        record = self._normalize(question, answer)
        line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        with self._lock:
            self.answers_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.answers_path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._answers.append(record)
            self._answered[record["qid"]] = record
        return record

    def _normalize(self, question: dict[str, Any], answer: Any) -> dict[str, Any]:
        if not isinstance(answer, dict):
            raise QuizError("answer must be an object")
        result = str(answer.get("result") or "")
        kind = question["kind"]
        allowed = COUNT_RESULTS if kind == COUNT else CROP_RESULTS
        if result not in allowed:
            raise QuizError(f"answer {result!r} not valid for a {kind} question")
        payload: dict[str, Any] = {"result": result, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if result == "fix":
            counts = {}
            for key in ("top", "bottom"):
                value = answer.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise QuizError(f"{key} must be a number")
                value = int(value)
                if not 0 <= value <= 500:
                    raise QuizError(f"{key} out of range")
                counts[key] = value
            payload.update(counts)
            payload["total"] = counts["top"] + counts["bottom"]
        if result == "class":
            name = str(answer.get("class") or "")
            if name not in self.class_names:
                raise QuizError(f"unknown class {name!r}")
            payload["class"] = name
        if result == "part":
            options = {int(o["part"]): o["class"] for o in question.get("part_options") or []}
            number = answer.get("part")
            if int(number or 0) not in options:
                raise QuizError("answer must name a part this question offers")
            payload["part"] = int(number)
            payload["class"] = options[int(number)]
        return {
            "qid": question["qid"],
            "kind": kind,
            "image_id": question["image_id"],
            "split": question.get("split"),
            "ann_id": question.get("ann_id"),
            "guess": question.get("guess"),
            "guess_confidence": question.get("confidence"),
            "answer": payload,
        }

    def undo(self) -> dict[str, Any] | None:
        """Drop the last log line and re-replay; returns what was removed."""
        with self._lock:
            if not self._answers:
                return None
            removed = self._answers[-1]
            self._write_log(self._answers[:-1])
            self._answers = self._answers[:-1]
            self._answered = {}
            for record in self._answers:
                self._answered[record["qid"]] = record
            return removed

    # -- progress -----------------------------------------------------------

    def next_index(self) -> int:
        with self._lock:
            for i, question in enumerate(self.questions):
                if question["qid"] not in self._answered:
                    return i
            return len(self.questions)

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            answered = len(self._answered)
            index = self.next_index()
            return {
                "total": len(self.questions),
                "answered": answered,
                "log_lines": len(self._answers),
                "next_index": index,
                "undo_available": bool(self._answers),
                "image_total": len(self.image_ids),
                "image_number": (self.image_ids.index(self.questions[index]["image_id"]) + 1)
                if index < len(self.questions)
                else len(self.image_ids),
                "answers": {qid: rec["answer"] for qid, rec in self._answered.items()},
            }

    def queue_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.queue.get("version"),
                "splits_order": self.queue.get("splits_order"),
                "classes": self.classes,
                "images": self.queue.get("images") or [],
                "questions": self.questions,
                "answers": {qid: rec["answer"] for qid, rec in self._answered.items()},
                "next_index": self.next_index(),
                "undo_available": bool(self._answers),
            }

    # -- pixels -------------------------------------------------------------

    def image(self, image_id: str) -> Image.Image:
        """Open a queue image from the images root (refusing traversal)."""
        if image_id not in self.by_image:
            raise QuizError(f"image not in queue: {image_id!r}")
        path = safe_join(self.images_root, image_id)
        if path is None:
            raise QuizError(f"image file not readable: {image_id!r}")
        with self._lock:
            cached = self._images.get(image_id)
            if cached is not None:
                self._images.move_to_end(image_id)
                return cached
            image = Image.open(path)
            image.load()
            self._images[image_id] = image
            while len(self._images) > OPEN_IMAGES:
                self._images.popitem(last=False)
            return image

    def _cached(self, key: str, build: Any) -> bytes:
        with self._lock:
            if key in self._render:
                self._render.move_to_end(key)
                return self._render[key]
        data = build()
        with self._lock:
            self._render[key] = data
            while len(self._render) > CACHE_ITEMS:
                self._render.popitem(last=False)
        return data

    def frame_png(self, index: int, full: bool = False) -> bytes:
        if not 0 <= index < len(self.image_ids):
            raise QuizError("frame index out of range")
        image_id = self.image_ids[index]
        side = FRAME_FULL_SIDE if full else FRAME_MAX_SIDE
        return self._cached(f"frame:{image_id}:{side}", lambda: self._build_frame(image_id, side))

    def _build_frame(self, image_id: str, max_side: int = FRAME_MAX_SIDE) -> bytes:
        image = self.image(image_id)
        boxes = [
            {"bbox": q["bbox"], "number": q.get("number"), "color": self._box_color(q)}
            for q in self.by_image[image_id]
            if q["kind"] == CROP
        ]
        return draw_frame(image, boxes, max_side)

    def crop_jpeg(self, index: int) -> bytes:
        question = self.questions[index]
        if question["kind"] != CROP:
            raise QuizError("that question has no crop")
        return self._cached(
            f"crop:{question['qid']}", lambda: make_crop(self.image(question["image_id"]), question["bbox"])
        )

    def context_jpeg(self, index: int) -> bytes:
        question = self.questions[index]
        if question["kind"] != CROP:
            raise QuizError("that question has no context image")
        color = self._box_color(question)
        return self._cached(
            f"context:{question['qid']}", lambda: make_context(self.image(question["image_id"]), question["bbox"], color)
        )

    def _box_color(self, question: dict[str, Any]) -> str:
        options = question.get("part_options") or []
        if options:
            return str(options[0].get("color") or "#c8102e")
        for entry in self.classes:
            if entry.get("name") == question.get("guess"):
                return str(entry.get("color") or "#c8102e")
        return "#868e96"


# --------------------------------------------------------------------------- http


class QuizHandler(BaseHTTPRequestHandler):
    store: QuizStore
    protocol_version = "HTTP/1.1"
    server_version = "FreestyleQuiz/1.0"

    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _query(self) -> dict[str, str]:
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).items() if v}

    def _index(self, params: dict[str, str]) -> int:
        raw = params.get("i", "")
        if not str(raw).isdigit():
            raise QuizError("query parameter 'i' must be a question index")
        index = int(raw)
        if not 0 <= index < len(self.store.questions):
            raise QuizError("question index out of range")
        return index

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        store = self.store
        try:
            if path in ("/", "/index.html"):
                self._static("index.html")
            elif path.startswith("/static/"):
                self._static(path[len("/static/") :])
            elif path == "/api/state":
                self._json(200, store.state_payload())
            elif path == "/api/queue":
                self._json(200, store.queue_payload())
            elif path == "/img/frame":
                params = self._query()
                images = sorted(store.by_image)
                raw = params.get("i", "")
                if str(raw).isdigit() and 0 <= int(raw) < len(images):
                    self._send(200, store.frame_png(int(raw), params.get("full") in ("1", "true")), "image/jpeg")
                else:
                    self._json(404, {"error": "no such frame"})
            elif path in ("/img/crop", "/img/context"):
                params = self._query()
                builder = store.crop_jpeg if path.endswith("crop") else store.context_jpeg
                self._send(200, builder(self._index(params)), "image/jpeg")
            else:
                self._json(404, {"error": "not found", "path": path})
        except QuizError as exc:
            self._json(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # pragma: no cover - defensive
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _body(self) -> dict[str, Any]:
        length = self.headers.get("Content-Length")
        length = int(length) if length and str(length).isdigit() else 0
        if length <= 0:
            raise QuizError("empty body")
        if length > MAX_BODY_BYTES:
            raise QuizError("body too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise QuizError("body must be a JSON object")
        return payload

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        store = self.store
        try:
            if path == "/api/answer":
                body = self._body()
                record = store.answer(body.get("qid", ""), body.get("answer"))
                self._json(200, {"ok": True, "record": record, "state": store.state_payload()})
            elif path == "/api/undo":
                removed = store.undo()
                if removed is None:
                    self._json(409, {"error": "nothing to undo"})
                else:
                    self._json(200, {"ok": True, "removed": removed, "state": store.state_payload()})
            else:
                self._json(404, {"error": "not found", "path": path})
        except QuizError as exc:
            self._json(400, {"error": str(exc)})
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"invalid JSON: {exc}"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _static(self, name: str) -> None:
        path = safe_join(STATIC_DIR, name)
        if path is None:
            self._json(404, {"error": "not found"})
            return
        self._send(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("QUIZ_APP_QUIET"):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class QuizServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str, port: int, store: QuizStore) -> None:
        super().__init__((host, port), type("BoundQuizHandler", (QuizHandler,), {"store": store}))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phone-first ground-truth quiz app")
    p.add_argument("--queue", type=Path, required=True, help="queue json from freestyle.quiz.build_queue")
    p.add_argument("--answers", type=Path, required=True, help="append-only answers JSONL under data/gt/")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (local) or a tailnet address")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    store = QuizStore(queue, args.answers, args.images_root)
    httpd = QuizServer(args.host, args.port, store)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    state = store.state_payload()
    print(
        f"quiz app: http://{shown}:{httpd.server_address[1]}  "
        f"({state['answered']}/{state['total']} answered, resumes at {state['next_index']}) "
        f"-> {store.answers_path}",
        file=sys.stderr,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
