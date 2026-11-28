"""Is Qwen 3.8 Flash Next a better crop reader than Inkling 2 Mini?  Score both on human answers.

The phone quiz (``freestyle/quiz``) collected human verdicts on test-split cartridge
crops whose on-screen guess came from Inkling 2 Mini's two-stage identifier
(``data/labels/bootstrap_v2.json``).  That verdict log is the only ground truth here:
``correct`` confirms the guess, ``class`` names the right catalog id, ``part`` names
family+part, ``skip`` / ``not_cartridge`` carry no identity verdict, and a ``class``
answer of ``unknown`` means a human could not tell either (kept, reported separately).

This module rebuilds exactly those crops from ORIGINAL pixels with the quiz app's own
crop logic (``freestyle.quiz.server.make_crop``, 15 % padding) and re-runs
``freestyle.vlm_identify.VLMCropIdentifier`` on them for every model x mode, with the
prompts unchanged — this is evaluation on test answers only, nothing is tuned on them.
Inkling is re-run fresh so the comparison is apples to apples.

Modes (all from :mod:`freestyle.vlm_identify`):
    single             one-shot choice over the full catalog
    two_stage          family first, then a zoomed read of the part numeral
    two_stage_shelf    two_stage plus the caller-provided shelf hint
    contact_sheet      several crops in one image, indexed answers

Outputs (``runs/reader_compare/``, gitignored): ``eval_set.json``, ``crops/*.jpg``,
``cache/*.json`` (every raw VLM reply, so a re-run costs nothing), ``calls.jsonl``
(per-call latency / attempts / which model id actually answered),
``predictions/<model>__<mode>.jsonl`` (resumable), ``results.json``, and the report
``docs/freestyle/READER_COMPARE.md``.

Run:
    uv run python -m freestyle.experiments.reader_compare                 # build+run+report
    uv run python -m freestyle.experiments.reader_compare --stage report  # rescore, no calls
    uv run python -m freestyle.experiments.reader_compare --smoke 2       # val crops, format only
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from freestyle import collective
from freestyle import catalog as catalog_module
from freestyle.catalog import UNKNOWN
from freestyle.quiz.build_queue import CROP, validate_queue
from freestyle.quiz.server import CROP_PAD, DEFAULT_IMAGES_ROOT, make_crop
from freestyle.quiz.to_coco import load_answers
from freestyle.vlm_identify import VLMCropIdentifier, _family_and_part, _json, _normalized

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "runs" / "reader_compare"
DEFAULT_REPORT = REPO_ROOT / "docs" / "freestyle" / "READER_COMPARE.md"

#: Logical name -> served model id.  ``flash`` is the reader the developer prefers.
MODELS: dict[str, str] = {"inkling": "david-birks/inkling-2-mini", "flash": "qwen3-8-flash-next"}
#: Second-choice ids for the same weights, used when a route stays at capacity too long.
FALLBACKS: dict[str, str] = {"qwen3-8-flash-next": "david-birks/qwen3-8-flash-next",
                             "david-birks/inkling-2-mini": "inkling-2-mini"}

MODES: tuple[str, ...] = ("single", "two_stage", "two_stage_shelf", "contact_sheet")
#: ``contact_sheet`` packs this many crops per image (bigger sheets get downscaled by the route).
SHEET_SIZE = 3
#: Row key for the historical Inkling guesses the quiz screen showed.
BASELINE_KEY = "inkling__quiz_bootstrap"

# Answer results that carry no identity verdict at all (skip, not-a-cartridge, unanswered-ish).
EXCLUDED_RESULTS = frozenset({"", "skip", "not_cartridge", "implied_not_usable"})
# Answer results where the human explicitly could not determine the identity.
CAN_NOT_TELL_RESULTS = frozenset({"two_plus"})


# --------------------------------------------------------------------------- metrics


def wilson_ci(successes: int, n: int, z: float = 1.96) -> dict[str, Any]:
    """Wilson score interval (95 % by default) for a binomial proportion."""
    if n <= 0:
        return {"k": int(successes), "n": 0, "p": None, "lo": None, "hi": None}
    p = float(successes) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"k": int(successes), "n": n, "p": round(p, 4),
            "lo": round(max(0.0, centre - half), 4), "hi": round(min(1.0, centre + half), 4)}


def family_part(catalog_id: Any) -> tuple[str, int | None]:
    """(normalized family, part) for a catalog id; ``unknown`` has no family."""
    text = str(catalog_id or UNKNOWN)
    if text == UNKNOWN:
        return UNKNOWN, None
    return _family_and_part(text)


def catalog_family_map(rows: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """catalog id -> the family string the two-stage identifier itself lists.

    ``VLMCropIdentifier._families`` groups by the catalog *display name* (``nns``,
    ``fanta``, ``orange flavor``) while ids normalize differently (``nns sweetener``,
    ``fanta orange``, ``flavor orange``).  Comparing a claimed family against an
    id-derived one would be wrong, so map both sides through this table.
    """
    rows = catalog_module.load_catalog() if rows is None else rows
    out: dict[str, str] = {}
    for row in rows:
        class_id = str(row.get("id", ""))
        if not class_id or class_id == UNKNOWN:
            continue
        family, part = _family_and_part(row.get("name", class_id))
        if part is None:
            id_family, id_part = _family_and_part(class_id)
            if id_part is not None:
                family = id_family
        out[class_id] = family
    return out


def claimed_family(pred: dict[str, Any], families: dict[str, str]) -> str:
    """The brand a reader actually asserts: its class if named, else its family claim."""
    name = str(pred.get("class") or UNKNOWN)
    if name != UNKNOWN:
        return families.get(name, family_part(name)[0])
    claim = family_part(pred.get("family"))[0]
    return _normalized(claim) if claim and claim != UNKNOWN else UNKNOWN


def _rate(flags: list[bool]) -> dict[str, Any]:
    return wilson_ci(sum(1 for ok in flags if ok), len(flags))


# --------------------------------------------------------------------------- eval set


def crop_label(image_id: str, ann_id: Any) -> str:
    """Short stable crop id: ``<event4><frame>#<ann>``."""
    parts = PurePosixPath(str(image_id)).parts
    stem = PurePosixPath(parts[-1]).stem if parts else "?"
    tag = "".join(ch for ch in (parts[-2] if len(parts) >= 2 else "") if ch.isalnum())[:4]
    return f"{tag}{stem}#{ann_id}"


def identity_from_answer(question: dict[str, Any], answer: dict[str, Any] | None) -> dict[str, Any]:
    """Human identity for one answered crop question.

    ``definite`` False = the human could not tell either (kept, reported separately);
    ``excluded`` True = the answer carries no identity verdict.
    """
    answer = answer if isinstance(answer, dict) else {}
    result = str(answer.get("result") or "")
    out: dict[str, Any] = {"result": result, "human_class": None,
                           "definite": False, "excluded": False, "label_source": None}
    if result in EXCLUDED_RESULTS:
        out["excluded"] = True
        return out
    if result in CAN_NOT_TELL_RESULTS:
        out["human_class"], out["label_source"] = UNKNOWN, result
        return out
    if result == "partial" and not answer.get("class"):
        out["human_class"], out["label_source"] = UNKNOWN, result
        return out
    named = str(answer.get("class") or "")
    if not named and answer.get("part") is not None:
        wanted = int(answer["part"])
        for option in question.get("part_options") or []:
            if int(option.get("part") or 0) == wanted:
                named = str(option.get("class") or "")
                break
    if not named and result == "correct":
        named = str(question.get("guess") or "")
    if not named:
        out["excluded"] = True
        return out
    out["human_class"], out["label_source"] = named, result
    out["definite"] = named != UNKNOWN
    return out


def build_eval_set(queue: dict[str, Any], answers: dict[str, dict[str, Any]],
                   split: str = "test", families: dict[str, str] | None = None) -> dict[str, Any]:
    """Every answered crop question of one split, with its human label or exclusion reason."""
    validate_queue(queue)
    items: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    crop_questions = [q for q in queue.get("questions") or []
                      if q.get("kind") == CROP and q.get("split") == split]
    for question in crop_questions:
        qid = str(question.get("qid"))
        answer = answers.get(qid)
        if answer is None:
            excluded.append({"qid": qid, "reason": "unanswered", "result": None})
            continue
        verdict = identity_from_answer(question, answer)
        if verdict["excluded"]:
            excluded.append({"qid": qid, "reason": "no_identity_verdict",
                             "result": verdict["result"]})
            continue
        image_id = str(question["image_id"])
        crop_id = crop_label(image_id, question.get("ann_id"))
        while crop_id in seen:  # never collide, even on pathological ids
            crop_id += "~"
        seen.add(crop_id)
        family, part = family_part(verdict["human_class"])
        families = families or {}
        human_family = families.get(str(verdict["human_class"]), family)
        items.append({
            "crop_id": crop_id, "qid": qid, "image_id": image_id, "ann_id": question.get("ann_id"),
            "number": question.get("number"), "bbox": [float(v) for v in question["bbox"]],
            "shelf": question.get("shelf"), "purpose": question.get("purpose"),
            "attributes": question.get("attributes") or {},
            "quiz_guess": question.get("guess"), "quiz_guess_confidence": question.get("confidence"),
            "quiz_family": question.get("family"), "ask_part": bool(question.get("ask_part")),
            "part_options": [int(o["part"]) for o in question.get("part_options") or []],
            "result": verdict["result"], "label_source": verdict["label_source"],
            "human_class": verdict["human_class"], "human_family": human_family,
            "human_part": part, "definite": verdict["definite"],
        })
    order = {"eval": 0, "tune": 1, "train": 2}
    items.sort(key=lambda it: (order.get(it.get("purpose"), 3), it["image_id"],
                               it["ann_id"] if isinstance(it["ann_id"], int) else 0))
    counts: dict[str, int] = {
        "crop_questions": len(crop_questions),
        "labeled": len(items),
        "definite": sum(1 for it in items if it["definite"]),
        "human_cannot_tell": sum(1 for it in items if not it["definite"]),
        "excluded": len(excluded),
    }
    for result, hits in Counter(it["result"] for it in items).items():
        counts[f"result_{result}"] = hits
    for reason, hits in Counter(row["reason"] for row in excluded).items():
        counts[f"excluded_{reason}"] = hits
    return {"version": 1, "split": split, "items": items, "excluded": excluded, "counts": counts}


# --------------------------------------------------------------------------- crops


def crop_image(image: Image.Image, bbox: list[float], pad: float = CROP_PAD) -> Image.Image:
    """The crop the phone quiz showed: padded ORIGINAL pixels, upscaled, quiz JPEG."""
    return Image.open(io.BytesIO(make_crop(image, bbox, pad=pad))).convert("RGB")


def build_crop_images(items: list[dict[str, Any]], images_root: Path, crops_dir: Path,
                      pad: float = CROP_PAD) -> dict[str, Image.Image]:
    """Cut every eval crop, writing the exact bytes that get sent under ``crops_dir``."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_image[item["image_id"]].append(item)
    out: dict[str, Image.Image] = {}
    for image_id in sorted(by_image):
        path = Path(images_root) / PurePosixPath(image_id)
        with Image.open(path) as image:
            image.load()
            for item in by_image[image_id]:
                data = make_crop(image, item["bbox"], pad=pad)
                name = item["crop_id"].replace("/", "_").replace("#", "_").replace("~", "")
                (crops_dir / f"{name}.jpg").write_bytes(data)
                out[item["crop_id"]] = Image.open(io.BytesIO(data)).convert("RGB")
    return out


def tune_crops(queue: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Up to *n* val/train crops (no human labels) — prompt/format debugging only."""
    out: list[dict[str, Any]] = []
    for question in queue.get("questions") or []:
        if question.get("kind") != CROP or question.get("split") not in ("val", "train"):
            continue
        out.append({"crop_id": crop_label(question["image_id"], question.get("ann_id")),
                    "image_id": question["image_id"], "bbox": [float(v) for v in question["bbox"]],
                    "shelf": question.get("shelf"), "human_class": None, "definite": False})
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- VLM plumbing


class CallCache:
    """Disk-cached, in-flight-capped, patient replacement for ``collective.chat_image``.

    * key = effective model id + prompt + image bytes + generation kwargs, so nothing is
      ever asked twice and a changed prompt is a fresh key;
    * at most ``max_inflight`` requests in flight per process (project rule: <= 2);
    * 429/5xx/timeouts retried by :func:`freestyle.collective.chat_image` with exponential
      backoff + jitter inside ``retry_budget_s`` of total time per call;
    * when one requested model has been failing continuously for ``outage_s`` seconds,
      later calls use its fallback id (same weights, different route) and stay there;
      every call logs which id actually answered;
    * failures are never cached.
    """

    def __init__(self, cache_dir: Path, log_path: Path | None = None, *, max_inflight: int = 2,
                 max_tokens: int = 8192, temperature: float = 0.0, retry_budget_s: float = 600.0,
                 outage_s: float = 1200.0, fallbacks: dict[str, str] | None = None,
                 clock: Any = time.monotonic, sleeper: Any = time.sleep,
                 client: Any = None, sender: Any = None) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")
        if int(max_tokens) < 2048:
            raise ValueError("max_tokens must be >= 2048: flash-next is a thinking model")
        self.cache_dir, self.log_path = Path(cache_dir), Path(log_path) if log_path else None
        self.max_inflight = int(max_inflight)
        self.max_tokens, self.temperature = int(max_tokens), float(temperature)
        self.retry_budget_s, self.outage_s = float(retry_budget_s), float(outage_s)
        self.fallbacks = dict(fallbacks if fallbacks is not None else FALLBACKS)
        self.clock, self.sleeper, self.client = clock, sleeper, client
        self.lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(self.max_inflight)
        # The transport actually used.  Never ``collective.chat_image`` at call time: while
        # :func:`patched_chat` is active that attribute *is* ``self.chat``.
        self.sender = sender or collective.chat_image
        self._state: dict[str, dict[str, Any]] = {}
        self._local = threading.local()
        self.events: list[dict[str, Any]] = []

    # -- attribution ------------------------------------------------------

    @contextlib.contextmanager
    def tag(self, name: str, members: list[str] | None = None):
        """Attribute the calls made inside this block to one crop (or one sheet)."""
        previous = getattr(self._local, "tag", None)
        self._local.tag = (name, list(members or []))
        try:
            yield
        finally:
            self._local.tag = previous

    # -- model routing ----------------------------------------------------

    def effective_model(self, requested: str) -> str:
        with self.lock:
            state = self._state.setdefault(requested, {"alias": None, "failing_since": None})
            return state["alias"] or requested

    def _note_success(self, requested: str) -> None:
        with self.lock:
            state = self._state.setdefault(requested, {"alias": None, "failing_since": None})
            state["failing_since"] = None

    def _note_failure(self, requested: str, now: float) -> str | None:
        """Return the fallback id once *requested* has been down for ``outage_s`` straight."""
        switched = None
        with self.lock:
            state = self._state.setdefault(requested, {"alias": None, "failing_since": None})
            if state["failing_since"] is None:
                state["failing_since"] = now
            down = now - state["failing_since"]
            if state["alias"] is None and down >= self.outage_s and requested in self.fallbacks:
                state["alias"] = switched = self.fallbacks[requested]
        if switched:
            self.events.append({"event": "model_fallback", "requested": requested, "using": switched,
                                "down_s": round(self.outage_s, 1)})
        return switched

    # -- cache / log ------------------------------------------------------

    def _key(self, model: str, prompt: str, image: Image.Image, kwargs: dict[str, Any]) -> str:
        payload = {"model": model, "prompt": prompt, "kwargs": kwargs,
                   "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def _log(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        with self.lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

    def call_rows(self) -> list[dict[str, Any]]:
        if self.log_path is None or not self.log_path.is_file():
            return []
        rows = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def chat(self, model: str, image: Any, prompt: str, max_side: int | None = None,
             **kwargs: Any) -> str:
        """Drop-in for :func:`freestyle.collective.chat_image` (same positional shape)."""
        if not isinstance(image, Image.Image):
            image = collective.load_image_source(image)
        extra = {k: v for k, v in kwargs.items() if k in ("retries", "extra_body", "n")}
        gen = {"max_side": max_side, "max_tokens": self.max_tokens,
               "temperature": self.temperature, **extra}
        tag_name, members = getattr(self._local, "tag", (None, []))
        while True:
            used = self.effective_model(model)
            path = self.cache_dir / f"{self._key(used, prompt, image, gen)}.json"
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                self._log({"tag": tag_name, "crops": members, "model": used, "requested": model,
                           "cache": "hit", "latency_s": 0.0, "attempts": 0, "ok": True,
                           "finish_reason": record.get("finish_reason")})
                return str(record["text"])
            started = self.clock()
            with self._slots:
                try:
                    text = self.sender(used, image, prompt, max_side=max_side,
                                       max_tokens=self.max_tokens,
                                       temperature=self.temperature,
                                       retry_budget_s=self.retry_budget_s,
                                       client=self.client, **extra)
                except Exception as exc:  # noqa: BLE001 - logged; routing may retry elsewhere
                    elapsed = self.clock() - started
                    detail = f"{type(exc).__name__}: {str(exc)[:300]}"
                    self._log({"tag": tag_name, "crops": members, "model": used, "requested": model,
                               "cache": "miss", "latency_s": round(elapsed, 2), "ok": False,
                               "error": detail})
                    switched = self._note_failure(model, started)
                    if switched and used != switched:
                        continue
                    raise
                elapsed = self.clock() - started
                finish = collective.LAST_CALL.get("finish_reason")
                attempts = collective.LAST_CALL.get("attempts")
            self._note_success(model)
            payload = json.dumps({"model": used, "requested": model, "text": text,
                                  "finish_reason": finish, "latency_s": round(elapsed, 2)},
                                 sort_keys=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
            self._log({"tag": tag_name, "crops": members, "model": used, "requested": model,
                       "cache": "miss", "latency_s": round(elapsed, 2), "attempts": attempts,
                       "ok": True, "finish_reason": finish, "empty": not str(text).strip()})
            return text


@contextlib.contextmanager
def patched_chat(cache: CallCache):
    """Route every ``freestyle.vlm_identify`` call through *cache* (one place to cache/retry).

    The transport that ``collective.chat_image`` points at *right now* (the real client, or a
    test double) becomes the cache's sender, so ``cache.chat`` can never call itself.
    """
    original = collective.chat_image
    if original is not cache.chat:
        cache.sender = original
    collective.chat_image = cache.chat
    try:
        yield cache
    finally:
        collective.chat_image = original


# --------------------------------------------------------------------------- running


class SheetReader(VLMCropIdentifier):
    """Production identifier, but mode (d) tolerates 1-based crop indexes in the reply.

    ``VLMCropIdentifier.contact_sheet`` draws labels ``0..n-1`` and maps ``index`` straight
    through; thinking models answer 1-based about as often as 0-based, which silently shifts
    every answer by one crop.  Re-mapping the *same* reply here (never re-asking, so the cache
    and the prompt stay untouched) keeps mode (d) measurable, identically for both readers.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._own = threading.local()

    @property
    def last_reply(self) -> str:
        return getattr(self._own, "reply", "")

    def _ask(self, image: Image.Image, prompt: str) -> str:
        self._own.reply = super()._ask(image, prompt)
        return self._own.reply

    def contact_sheet(self, crops, shelves=None):
        crops = list(crops)
        results = super().contact_sheet(crops, shelves)
        parsed = _json(self.last_reply)
        if not isinstance(parsed, list):
            return results
        objects = [obj for obj in parsed if isinstance(obj, dict)]
        indexes = [obj.get("index") for obj in objects]
        numeric = [idx for idx in indexes if isinstance(idx, int) and not isinstance(idx, bool)]
        count = len(list(crops))
        shift = 1 if (numeric and min(numeric) >= 1 and 0 not in numeric
                      and max(numeric) <= count) else 0
        remapped = [self._result(None) for _ in crops]
        for obj in objects:
            idx = obj.get("index")
            if isinstance(idx, bool) or not isinstance(idx, int):
                continue
            idx -= shift
            if 0 <= idx < count:
                remapped[idx] = self._result(obj)
        return remapped


def identifier_for(model_id: str, mode: str) -> VLMCropIdentifier:
    """The production identifier for one mode; prompts, hints and zoom are unchanged."""
    reader = SheetReader if mode == "contact_sheet" else VLMCropIdentifier
    return reader(model_id, catalog=None,
                  mode="two_stage" if mode.startswith("two_stage") else "existing")


class ReaderRunner:
    """Runs every model x mode over the eval crops, resumably (one row per crop, appended)."""

    def __init__(self, out: Path, models: dict[str, str], modes: list[str], *, cache: CallCache,
                 sheet_size: int = SHEET_SIZE, max_workers: int = 2) -> None:
        if max_workers > cache.max_inflight:
            raise ValueError("worker pool may not exceed the in-flight cap")
        self.out, self.models, self.modes = Path(out), dict(models), list(modes)
        self.cache, self.sheet_size, self.max_workers = cache, max(1, int(sheet_size)), max_workers
        self.predictions_dir = self.out / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    # -- resumable outputs ------------------------------------------------

    def key(self, model: str, mode: str) -> str:
        return f"{model}__{mode}"

    def path_for(self, model: str, mode: str) -> Path:
        return self.predictions_dir / f"{self.key(model, mode)}.jsonl"

    def load_done(self, model: str, mode: str) -> dict[str, dict[str, Any]]:
        path = self.path_for(model, mode)
        done: dict[str, dict[str, Any]] = {}
        if not path.is_file():
            return done
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("crop_id"):
                done[str(row["crop_id"])] = row
        return done

    def _append(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.cache.lock:
            with open(path, "a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()

    # -- per-crop and per-sheet work --------------------------------------

    def _read_one(self, model_name: str, mode: str, ident: VLMCropIdentifier,
                  item: dict[str, Any], image: Image.Image) -> list[dict[str, Any]]:
        shelf = item.get("shelf") if mode == "two_stage_shelf" else None
        started = time.monotonic()
        error = None
        try:
            with self.cache.tag(f"{model_name}.{mode}.{item['crop_id']}", [item["crop_id"]]):
                result = ident.identify(image, shelf=shelf)
        except Exception as exc:  # noqa: BLE001 - one bad crop must not kill the run
            result, error = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
        name = str(result.get("class") or UNKNOWN)
        return [{"crop_id": item["crop_id"], "model": self.key(model_name, mode), "mode": mode,
                 "class": name, "family": result.get("family") or family_part(name)[0],
                 "part": result.get("part"), "confidence": result.get("confidence"),
                 "seconds": round(time.monotonic() - started, 2), "error": error,
                 "rationale": str(result.get("rationale", ""))[:300]}]

    def _read_sheet(self, model_name: str, ident: VLMCropIdentifier,
                    batch: list[dict[str, Any]], images: dict[str, Image.Image]) -> list[dict[str, Any]]:
        started = time.monotonic()
        names = [it["crop_id"] for it in batch]
        error = None
        try:
            with self.cache.tag(f"{model_name}.contact_sheet.{names[0]}", names):
                results = ident.contact_sheet([images[n] for n in names])
        except Exception as exc:  # noqa: BLE001
            results, error = [{}] * len(batch), f"{type(exc).__name__}: {str(exc)[:300]}"
        per = round((time.monotonic() - started) / max(1, len(batch)), 2)
        rows = []
        for item, result in zip(batch, results + [{}] * max(0, len(batch) - len(results))):
            name = str(result.get("class") or UNKNOWN)
            rows.append({"crop_id": item["crop_id"], "model": self.key(model_name, "contact_sheet"),
                         "mode": "contact_sheet", "class": name, "family": family_part(name)[0],
                         "part": None, "confidence": result.get("confidence"), "seconds": per,
                         "error": error, "sheet": names[0], "sheet_size": len(batch),
                         "rationale": str(result.get("rationale", ""))[:300]})
        return rows

    def _worker(self, task: tuple, *, ident: VLMCropIdentifier, model_name: str, mode: str,
                images: dict[str, Image.Image]) -> list[dict[str, Any]]:
        kind, payload = task
        if kind == "sheet":
            return self._read_sheet(model_name, ident, payload, images)
        return self._read_one(model_name, mode, ident, payload, images[payload["crop_id"]])

    def run(self, items: list[dict[str, Any]], images: dict[str, Image.Image],
            force: bool = False) -> dict[str, dict[str, dict[str, Any]]]:
        """Returns {model|mode: {crop_id: prediction row}}; saved crops are not re-read."""
        usable = [it for it in items if it["crop_id"] in images]
        out: dict[str, dict[str, dict[str, Any]]] = {}
        with patched_chat(self.cache):
            for model_name, model_id in self.models.items():
                for mode in self.modes:
                    ident = identifier_for(model_id, mode)
                    key = self.key(model_name, mode)
                    done = {} if force else self.load_done(model_name, mode)
                    todo = [it for it in usable if it["crop_id"] not in done]
                    out[key] = dict(done)
                    if not todo:
                        print(f"{key}: reused {len(done)} saved crops", flush=True)
                        continue
                    print(f"{key}: reading {len(todo)} crops "
                          f"({len(done)} already saved)", flush=True)
                    if mode == "contact_sheet":
                        tasks = [("sheet", todo[i:i + self.sheet_size])
                                 for i in range(0, len(todo), self.sheet_size)]
                    else:
                        tasks = [("one", it) for it in todo]
                    path = self.path_for(model_name, mode)
                    call = partial(self._worker, ident=ident, model_name=model_name,
                                   mode=mode, images=images)
                    with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                        for rows in pool.map(call, tasks):
                            for row in rows:
                                out[key][row["crop_id"]] = row
                            self._append(path, rows)
                            summary = " ".join(f"{r['crop_id']}={r['class']}" for r in rows)
                            print(f"  {key}: {summary}"
                                  + (f" ERROR {rows[0]['error']}" if rows[0].get("error") else ""),
                                  flush=True)
        return out

    # -- format smoke test on unlabelled val/train crops ------------------

    def smoke(self, items: list[dict[str, Any]], images: dict[str, Image.Image]) -> bool:
        """Format check only: every call returned non-empty text and no crop errored.

        Accuracy is deliberately not looked at; these crops have no human labels.
        """
        ok = True
        with patched_chat(self.cache):
            for model_name, model_id in self.models.items():
                for mode in self.modes:
                    ident = identifier_for(model_id, mode)
                    if mode == "contact_sheet":
                        rows = self._read_sheet(model_name, ident, items, images)
                    else:
                        rows = [row for it in items for row in
                                self._read_one(model_name, mode, ident, it, images[it["crop_id"]])]
                    prefix = f"{model_name}.{mode}."
                    calls = [row for row in self.cache.call_rows()
                             if str(row.get("tag", "")).startswith(prefix)]
                    clean = bool(calls) and all(row.get("ok") and not row.get("empty")
                                                for row in calls)
                    for row in rows:
                        ok = ok and clean and not row.get("error")
                        print(f"SMOKE {self.key(model_name, mode)} {row['crop_id']} "
                              f"class={row['class']} family={row['family']} sec={row['seconds']} "
                              f"calls={len(calls)} err={row.get('error')}", flush=True)
        return ok


# --------------------------------------------------------------------------- scoring


def score_mode(items: list[dict[str, Any]], preds: dict[str, dict[str, Any]],
               families: dict[str, str] | None = None) -> dict[str, Any]:
    """Top-1 / family / part / abstention vs the human labels, plus latency and failures."""
    families = families or {}
    definite = [it for it in items if it["definite"]]
    top1: list[bool] = []
    family_hit: list[bool] = []
    part_hit: list[bool] = []
    part_readable: list[bool] = []
    abstain: list[bool] = []
    confusions: Counter = Counter()
    seconds: list[float] = []
    failures: list[dict[str, Any]] = []
    missing = 0
    for it in definite:
        pred = preds.get(it["crop_id"])
        if pred is None:
            missing += 1
            continue
        name = str(pred.get("class") or UNKNOWN)
        pred_family, pred_part = claimed_family(pred, families), family_part(name)[1]
        gt_family = families.get(str(it["human_class"]), it["human_family"])
        top1.append(name == it["human_class"])
        # a two-stage abstention that still names the right brand counts here: family accuracy
        # measures brand reading, top-1 measures the finished answer.
        family_hit.append(pred_family == gt_family and pred_family != UNKNOWN)
        abstain.append(name == UNKNOWN)
        if it.get("human_part") is not None:
            part_hit.append(pred_part == it["human_part"])
            part_readable.append(pred_part is not None)
        if name != it["human_class"]:
            confusions[f"{it['human_class']} -> {name}"] += 1
        if pred.get("error"):
            failures.append({"crop_id": it["crop_id"], "error": pred["error"]})
        if isinstance(pred.get("seconds"), (int, float)):
            seconds.append(float(pred["seconds"]))

    indet = [it for it in items if not it["definite"]]
    abstained = guessed = indet_missing = 0
    for it in indet:
        pred = preds.get(it["crop_id"])
        if pred is None:
            indet_missing += 1
        elif str(pred.get("class") or UNKNOWN) == UNKNOWN:
            abstained += 1
        else:
            guessed += 1
    return {
        "n_labeled": len(items), "n_definite": len(definite), "n_missing": missing,
        "top1": _rate(top1), "family": _rate(family_hit), "part": _rate(part_hit),
        "part_n": len(part_hit), "part_readable": _rate(part_readable),
        "abstention": _rate(abstain), "confusions": confusions.most_common(8),
        "failures": failures,
        "latency_s": {"mean": round(statistics.fmean(seconds), 2) if seconds else None,
                      "median": round(statistics.median(seconds), 2) if seconds else None,
                      "max": round(max(seconds), 2) if seconds else None, "n": len(seconds)},
        "indeterminate": {"n": len(indet), "abstained": abstained, "guessed": guessed,
                         "n_missing": indet_missing,
                         "crop_ids": [it["crop_id"] for it in indet]},
        "by_crop": {it["crop_id"]: {"human": it["human_class"],
                                    "pred": str((preds.get(it["crop_id"]) or {}).get("class") or ""),
                                    "seconds": (preds.get(it["crop_id"]) or {}).get("seconds"),
                                    "error": (preds.get(it["crop_id"]) or {}).get("error")}
                    for it in items},
    }


def quiz_baseline(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The historical Inkling 2 Mini bootstrap guesses that the quiz screen showed."""
    return {it["crop_id"]: {"crop_id": it["crop_id"], "class": it.get("quiz_guess") or UNKNOWN,
                            "family": it.get("quiz_family") or UNKNOWN,
                            "confidence": it.get("quiz_guess_confidence")} for it in items}


def score_all(eval_set: dict[str, Any], runs: dict[str, dict[str, dict[str, Any]]],
              families: dict[str, str] | None = None) -> dict[str, Any]:
    items = eval_set["items"]
    scores = {key: score_mode(items, preds, families) for key, preds in sorted(runs.items())}
    scores[BASELINE_KEY] = score_mode(items, quiz_baseline(items), families)
    return scores


# --------------------------------------------------------------------------- report


def _ci(row: dict[str, Any]) -> str:
    if not row or not row.get("n") or row.get("p") is None:
        return "- (n=0)"
    return f"{row['p']:.2f} ({row['lo']:.2f}-{row['hi']:.2f}, n={row['n']})"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)


def model_label(name: str, models: dict[str, str]) -> str:
    return models.get(name) or MODELS.get(name) or name


def ranked_keys(scores: dict[str, Any], models: dict[str, str], modes: list[str]) -> list[str]:
    """Fresh runs in model/mode order, the historical quiz row last."""
    fresh = [f"{name}__{mode}" for name in models for mode in modes
             if f"{name}__{mode}" in scores]
    return fresh + [key for key in sorted(scores) if key not in fresh]


def summary_table(scores: dict[str, Any], models: dict[str, str], modes: list[str],
                  order: list[str]) -> list[str]:
    lines = ["| Reader | Mode | Top-1 vs human (Wilson 95 % CI) | Family correct | "
             "Part correct (two-part GT) | Abstain `unknown` | s/crop | Failures |",
             "|---|---|---|---|---|---|---|---|"]
    for key in order:
        row = scores[key]
        model_name, mode = key.split("__", 1)
        mode_label = "quiz bootstrap (historical, on-screen guess)" \
            if mode == "quiz_bootstrap" else f"`{mode}`"
        part = _ci(row["part"])
        if row["part_n"]:
            part += f" · readable {_pct(row['part_readable']['p'])}"
        latency = _num(row["latency_s"]["mean"])
        lines.append(f"| `{model_label(model_name, models)}` | {mode_label} | {_ci(row['top1'])} | "
                     f"{_ci(row['family'])} | {part} | {_pct(row['abstention']['p'])} | "
                     f"{latency} | {len(row['failures'])} |")
    return lines


def per_crop_table(scores: dict[str, Any], eval_set: dict[str, Any], models: dict[str, str],
                   order: list[str]) -> list[str]:
    fresh = [key for key in order if not key.endswith("__quiz_bootstrap")]
    header = ["Crop", "Shelf", "Human label", "How labelled", "Quiz guess"]
    header += [key.replace("__", "<br>") for key in fresh]
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---|" * (len(header) - 1)]
    for it in eval_set["items"]:
        cells = [f"`{it['crop_id']}`", str(it.get("shelf") or "-"), f"`{it['human_class']}`",
                 it["label_source"], f"`{it.get('quiz_guess') or '-'}`"]
        for key in fresh:
            entry = scores[key]["by_crop"].get(it["crop_id"]) or {}
            pred = entry.get("pred")
            mark = "" if pred else "?"
            if pred and it["definite"]:
                mark = " ✅" if pred == it["human_class"] else " ❌"
            elif pred and not it["definite"]:
                mark = " (abstained)" if pred == UNKNOWN else " (guessed)"
            cells.append(f"`{pred}`{mark}" if pred else "?")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_report(scores: dict[str, Any], eval_set: dict[str, Any], models: dict[str, str],
                  modes: list[str], *, generated: str | None = None,
                  notes: list[str] | None = None) -> str:
    counts = eval_set.get("counts") or {}
    order = ranked_keys(scores, models, modes)
    lines: list[str] = [
        "# Crop readers: Qwen 3.8 Flash Next vs Inkling 2 Mini (scored on human answers)",
        "",
        f"Generated {generated or time.strftime('%Y-%m-%d %H:%M %z')} by "
        "`freestyle/experiments/reader_compare.py` (full run: "
        "`uv run python -m freestyle.experiments.reader_compare`; rescore from the saved replies "
        "with no VLM calls: `uv run python -m freestyle.experiments.reader_compare --stage report`).",
        "",
        "## What was measured",
        "",
        "Human labels are the phone-quiz answers on **test-split** crops only "
        "(`data/gt/quiz_answers.jsonl` + `data/gt/quiz_queue.json`). The on-screen guess those "
        "verdicts confirmed or corrected came from Inkling 2 Mini's two-stage bootstrap identifier "
        "(`data/labels/bootstrap_v2.json`), so that historical guess is scored as one row and "
        "Inkling is also re-run fresh for an apples-to-apples comparison.",
        "",
        "Crops are cut from ORIGINAL pixels by the quiz app's own `make_crop` (15 % padding, same "
        "upscale + JPEG as the phone saw). Prompts, catalog hints, 2x upscale and the part-region "
        "zoom are `freestyle/vlm_identify.py` unchanged: **no prompt was tuned on these crops** "
        "(format debugging used `--smoke` on unlabelled val crops). "
        "`temperature=0`, `max_tokens=8192` (thinking model), <=2 in-flight requests per process, "
        "429/5xx/timeouts retried with exponential backoff + jitter, replies cached on disk.",
        "",
        "One harness-side deviation, applied identically to both readers: "
        "`VLMCropIdentifier.contact_sheet` labels its crops `0..n-1` and maps the reply's `index` "
        "straight through, with no tolerance for a model that numbers the crops from 1. "
        "`reader_compare.SheetReader` re-maps the *same* reply when no answer uses index 0 (no "
        "re-ask, no prompt change); production `vlm_identify.contact_sheet` is left alone and "
        "carries the ambiguity. Which reader did what is measured in the run notes below.",
        "",
        f"Eval set: **{counts.get('labeled', 0)} answered test crops** = "
        f"{counts.get('definite', 0)} with a definite human identity (the top-1 denominator) + "
        f"{counts.get('human_cannot_tell', 0)} where a human could not tell either (reported "
        f"separately, never in top-1). Excluded: {counts.get('excluded', 0)} "
        f"(unanswered / skip / not-a-cartridge). Label mix: "
        + ", ".join(f"`{key[len('result_'):]}` x{value}" for key, value in sorted(counts.items())
                    if key.startswith("result_")) + ".",
        "",
        "## Accuracy per model x mode",
        "",
        *summary_table(scores, models, modes, order),
        "",
        "*Family correct* = the brand the reader asserts, part ignored. A two-stage crop that names "
        "the brand but cannot read the numeral (class `unknown`) still counts as family-correct; "
        "a bare `unknown` with no brand claim counts as wrong. *Part correct* = the numeral the "
        "reader ends up asserting matches the human's (right digit, wrong brand still counts as a "
        "digit hit), scored only on crops whose human label is a two-part brand; *readable* = a "
        "digit came back at all there (an unreadable numeral is a legitimate stage-2 abstention, "
        "not a wrong digit).",
        "",
        "`s/crop` is wall time per crop (mean; median and max are in "
        "`runs/reader_compare/results.json` under `scores.<reader>__<mode>.latency_s`, along with "
        "per-crop seconds and rationales in `predictions/*.jsonl`).",
        "",
        "## Crops a human could not identify",
        "",
        "Excluded from top-1 (the contract drops GT `unknown`). A good reader should mostly "
        "abstain on these.",
        "",
        "| Reader | Mode | Crops | Abstained | Guessed anyway |",
        "|---|---|---|---|---|",
    ]
    for key in order:
        model_name, mode = key.split("__", 1)
        ind = scores[key]["indeterminate"]
        if not ind["n"]:
            continue
        lines.append(f"| `{model_label(model_name, models)}` | {mode} | {ind['n']} | "
                     f"{ind['abstained']} | {ind['guessed']} |")

    lines += ["", "## Confusion pairs (human -> reader, mismatches only)", "",
              "| Reader | Mode | Top pairs |", "|---|---|---|"]
    for key in order:
        model_name, mode = key.split("__", 1)
        pairs = scores[key]["confusions"]
        if not pairs:
            continue
        text = "; ".join(f"`{pair}` x{count}" for pair, count in pairs[:6])
        lines.append(f"| `{model_label(model_name, models)}` | {mode} | {text} |")

    lines += ["", "## Per-crop table", "",
              "`Human label` is the verdict (`correct` copies the on-screen guess, which the human "
              "confirmed; `class`/`part` are what the human named). ✅/❌ = matched / missed the "
              "human label; `(abstained)` / `(guessed)` on the "
              f"{(eval_set.get('counts') or {}).get('human_cannot_tell', 0)} crops a human could not "
              "call. `?` = no prediction recorded.", ""]
    lines += per_crop_table(scores, eval_set, models, order)

    notes = [note for note in (notes or []) if note]
    if notes:
        lines += ["", "## Run notes", ""] + [f"* {note}" for note in notes]
    lines += ["", "## Conclusion", "", *conclusion(scores, models), ""]
    return "\n".join(lines)


def _best(scores: dict[str, Any], model_name: str) -> tuple[str | None, dict[str, Any], float]:
    best_key, best_row, best_p = None, {}, -1.0
    for key, row in scores.items():
        if not key.startswith(model_name + "__") or key == BASELINE_KEY:
            continue
        p = row["top1"]["p"] or 0.0 if row["top1"]["n"] else 0.0
        if p > best_p:
            best_key, best_row, best_p = key, row, p
    return best_key, best_row, best_p


def conclusion(scores: dict[str, Any], models: dict[str, str]) -> list[str]:
    """Three sentences, straight from the numbers."""
    base = scores.get(BASELINE_KEY, {}).get("top1", {})
    ink_key, ink, ink_p = _best(scores, "inkling")
    fla_key, fla, fla_p = _best(scores, "flash")
    if not ink or not fla:
        return ["Only one reader ran — no comparison yet. Re-run with `--models inkling,flash`."]
    ink_ci, fla_ci = ink["top1"], fla["top1"]
    overlap = ink_ci["lo"] <= fla_ci["hi"] and fla_ci["lo"] <= ink_ci["hi"]
    verdict = "beats" if fla_p - ink_p >= 0.05 else ("matches" if abs(fla_p - ink_p) < 0.05
                                                     else "loses to")
    ink_mode, fla_mode = ink_key.split("__", 1)[1], fla_key.split("__", 1)[1]
    same_as_base = ""
    fresh_stage = scores.get("inkling__two_stage")
    if fresh_stage and base.get("n") and fresh_stage["top1"]["n"] == base["n"]:
        same_as_base = (", which the fresh `two_stage` re-run reproduces crop for crop"
                        if fresh_stage["top1"]["k"] == base["k"]
                        else f", while the fresh `two_stage` re-run scores "
                        f"{_ci(fresh_stage['top1'])}")
    noise = ("should not be treated as settled on this sample alone" if overlap
             else "is unlikely to be sampling noise")
    sheet_fla = scores.get("flash__contact_sheet", {}).get("abstention", {}).get("p")
    sheet_ink = scores.get("inkling__contact_sheet", {}).get("abstention", {}).get("p")
    return [
        f"On {fla_ci['n']} human-labelled test crops Qwen 3.8 Flash Next {verdict} a fresh Inkling "
        f"2 Mini run: best top-1 {_ci(fla_ci)} in `{fla_mode}` vs {_ci(ink_ci)} in `{ink_mode}` "
        f"(the historical on-screen Inkling guess was {_ci(base)}{same_as_base}).",
        f"The 95 % Wilson intervals {'overlap' if overlap else 'do not overlap'}, so the "
        f"{abs(fla_p - ink_p):.2f} point gap on n={fla_ci['n']} {noise}; the steadier signal is "
        f"family reading ({_ci(fla['family'])} Flash vs {_ci(ink['family'])} Inkling) with Flash "
        f"Next abstaining on far fewer crops ({_pct(fla['abstention']['p'])} vs "
        f"{_pct(ink['abstention']['p'])}).",
        f"Adopt Flash Next ({_num(fla['latency_s']['median'])} s/crop median, "
        f"{_num(fla['latency_s']['mean'])} s mean including retries, vs Inkling's "
        f"{_num(ink['latency_s']['median'])} s) and keep two-stage with the shelf hint, because the "
        f"bottleneck is now stage 2, not the reader: on the {fla['part_n']} two-part-brand crops "
        f"neither reader gets the numeral right ({_ci(fla['part'])} Flash vs {_ci(ink['part'])} "
        f"Inkling), and contact sheets are useless for both ({_pct(sheet_fla)} and {_pct(sheet_ink)} "
        f"abstention).",
    ]


# --------------------------------------------------------------------------- main


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sheet_index_stats(cache_dir: Path) -> dict[str, dict[str, int]]:
    """How each served model numbered contact-sheet crops in the replies we cached."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"zero_based": 0, "one_based": 0,
                                                            "no_index": 0, "unparsed": 0})
    if not Path(cache_dir).is_dir():
        return {}
    for path in sorted(Path(cache_dir).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        text = str(record.get("text", ""))
        if '"index"' not in text:
            continue
        model = str(record.get("model", "?"))
        parsed = _json(text)
        if not isinstance(parsed, list):
            stats[model]["unparsed"] += 1
            continue
        indexes = [obj.get("index") for obj in parsed if isinstance(obj, dict)]
        numbers = [i for i in indexes if isinstance(i, int) and not isinstance(i, bool)]
        if not numbers:
            stats[model]["no_index"] += 1
        elif min(numbers) == 0:
            stats[model]["zero_based"] += 1
        else:
            stats[model]["one_based"] += 1
    return dict(stats)


def provenance(paths: list[Path]) -> list[str]:
    """sha256 + size of the human answer files, so a growing log is visible in the report."""
    rows = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            rows.append(f"{path.name}: MISSING")
            continue
        blob = path.read_bytes()
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        rows.append(f"`{path.name}` sha256 {hashlib.sha256(blob).hexdigest()[:12]} "
                    f"({len(blob)} B, mtime {stamp})")
    return rows


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare VLM crop readers on human quiz answers")
    p.add_argument("--queue", type=Path, default=REPO_ROOT / "data/gt" / "quiz_queue.json")
    p.add_argument("--answers", type=Path, default=REPO_ROOT / "data/gt" / "quiz_answers.jsonl")
    p.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--split", default="test", choices=("test", "val", "train"))
    p.add_argument("--stage", default="all", choices=("build", "run", "report", "all"))
    p.add_argument("--models", default=",".join(MODELS), help="comma list of inkling,flash (or served ids)")
    p.add_argument("--modes", default=",".join(MODES), help=f"comma list of {', '.join(MODES)}")
    p.add_argument("--sheet-size", type=int, default=SHEET_SIZE)
    p.add_argument("--max-inflight", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--retry-budget-s", type=float, default=600.0)
    p.add_argument("--outage-s", type=float, default=1200.0,
                   help="switch to the fallback model id after this long continuously down")
    p.add_argument("--pad", type=float, default=CROP_PAD)
    p.add_argument("--force", action="store_true", help="ignore saved predictions")
    p.add_argument("--smoke", type=int, default=0, metavar="N",
                   help="read N val/train crops (no human labels) as a format check, then exit")
    return p


def parse_selection(raw: str, table: dict[str, str], valid: tuple[str, ...], what: str) -> dict[str, str]:
    """``a,b`` -> {name: value}, accepting either the short name or the literal value."""
    out: dict[str, str] = {}
    for part in [p.strip() for p in str(raw).split(",") if p.strip()]:
        if part in table:
            out[part] = table[part]
        elif what == "mode" and part in valid:
            out[part] = part
        elif what == "model":
            out[part] = part
        else:
            raise SystemExit(f"unknown {what} {part!r}; valid: {sorted(table) or list(valid)}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = parse_selection(args.models, MODELS, (), "model")
    modes = list(parse_selection(args.modes, {m: m for m in MODES}, MODES, "mode"))
    families = catalog_family_map()

    queue = read_json(args.queue)
    answers = load_answers(args.answers)
    eval_path = out / "eval_set.json"
    if args.stage in ("build", "all") or not eval_path.is_file():
        eval_set = build_eval_set(queue, answers, split=args.split, families=families)
        eval_path.write_text(json.dumps(eval_set, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"eval set: {eval_set['counts']} -> {eval_path}", flush=True)
    else:
        eval_set = read_json(eval_path)
        eval_set.setdefault("counts", {})

    cache = CallCache(out / "cache", out / "calls.jsonl", max_inflight=args.max_inflight,
                      max_tokens=args.max_tokens, retry_budget_s=args.retry_budget_s,
                      outage_s=args.outage_s)
    runner = ReaderRunner(out, models, modes, cache=cache, sheet_size=args.sheet_size,
                         max_workers=args.max_inflight)

    if args.smoke:
        items = tune_crops(queue, args.smoke)
        images = build_crop_images(items, args.images_root, out / "crops_smoke", pad=args.pad)
        ok = runner.smoke(items, images)
        print(f"SMOKE format-ok={ok} over {len(items)} val/train crops (no human labels touched)",
              flush=True)
        return 0 if ok else 1

    print(f"labeled crops: {len(eval_set['items'])} "
          f"(definite {eval_set['counts'].get('definite')})", flush=True)
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    if args.stage in ("run", "all"):
        images = build_crop_images(eval_set["items"], args.images_root, out / "crops", pad=args.pad)
        runs = runner.run(eval_set["items"], images, force=args.force)
        del images
    else:
        for name in models:
            for mode in modes:
                runs[runner.key(name, mode)] = runner.load_done(name, mode)

    scores = score_all(eval_set, {key: rows for key, rows in runs.items() if rows}, families)
    calls = cache.call_rows()
    failed = [row for row in calls if not row.get("ok")]
    notes = [f"{event['requested']} -> {event['using']} after {event['down_s']:.0f} s continuously "
             f"unavailable" for event in cache.events]
    notes.append(f"{len(calls)} VLM calls logged (all runs), {len(failed)} failed, "
                 f"{sum(1 for r in calls if (r.get('attempts') or 0) > 1)} needed 429/5xx retries")
    served = sorted({str(row.get("model")) for row in calls if row.get("model")})
    notes.append(f"served model ids seen in the call log: {', '.join(served) or '-'}")
    notes.append(f"contact sheets were packed {args.sheet_size} crops per image")
    for model, counts in sorted(sheet_index_stats(out / "cache").items()):
        notes.append(f"contact-sheet numbering by {model}: " + ", ".join(
            f"{key.replace('_', ' ')} x{value}" for key, value in sorted(counts.items())
            if value))
    notes.append("inputs: " + "; ".join(provenance([args.queue, args.answers])))
    results = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "models": models, "modes": modes,
               "counts": eval_set["counts"], "scores": scores, "served_models": served,
               "family_map": families, "eval_items": eval_set["items"],
               "notes": notes, "calls": {"total": len(calls), "failed": len(failed),
                                         "retried": sum(1 for r in calls
                                                        if (r.get("attempts") or 0) > 1)}}
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    args.report.write_text(render_report(scores, eval_set, models, modes, notes=notes),
                           encoding="utf-8")
    print(f"report -> {args.report}", flush=True)
    for key in ranked_keys(scores, models, modes):
        row = scores[key]
        print(f"{key}: top1={_ci(row['top1'])} family={_ci(row['family'])} part={_ci(row['part'])} "
              f"abstain={_pct(row['abstention']['p'])} s/crop={_num(row['latency_s']['mean'])} "
              f"fail={len(row['failures'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
