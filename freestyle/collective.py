"""Tiny client for the LLM Collective (OpenAI-compatible) vision endpoint.

Reads the API key from the ``collective`` provider in
``~/.pi/agent/models.json`` at runtime.  The key is never printed, logged
or written anywhere.

Public API
----------
get_client()                 -> openai.OpenAI
chat_image(model, image, prompt, max_side=None, **kw) -> str
list_models()                -> list[str]

``chat_image`` records latency/usage for the most recent call in the
module-level :data:`LAST_CALL` dict.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_MODELS_JSON = Path("~/.pi/agent/models.json")
FALLBACK_BASE_URL = "https://llm-collective.e-gineering.com/v1"
PROVIDER_NAME = "collective"

# Metrics for the most recent chat_image() call (never contains the key).
LAST_CALL: dict[str, Any] = {}

_clients: dict[tuple[str, str], Any] = {}


# ------------------------------------------------------------------
# Config / client
# ------------------------------------------------------------------

_ENV_PAT = re.compile(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)")


def resolve_secret(value: str) -> str:
    """Resolve a pi ``models.json`` credential to its literal value.

    Supports the same forms pi does: ``$NAME`` / ``${NAME}`` env
    interpolation, a leading ``!command`` (run, stdout = value), and
    ``$$`` / ``$!`` escapes for literal ``$`` / ``!``.
    """
    if value.startswith("!"):
        import subprocess

        out = subprocess.run(
            value[1:], shell=True, check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()

    def repl(m: re.Match[str]) -> str:
        name = m.group("braced") or m.group("bare")
        return os.environ.get(name, "")

    # Protect $$ / $! escapes, interpolate, then restore literal $ / !.
    value = value.replace("$$", "\x00").replace("$!", "\x01")
    value = _ENV_PAT.sub(repl, value)
    return value.replace("\x00", "$").replace("\x01", "!")


def read_provider_config(
    models_json: str | Path = DEFAULT_MODELS_JSON,
) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for the LLM Collective provider.

    The key is read at runtime (``${ENV}``-interpolated like pi does),
    never printed or persisted.
    """
    path = Path(models_json).expanduser()
    data = json.loads(path.read_text())
    provider = data["providers"][PROVIDER_NAME]
    base_url = resolve_secret(provider.get("baseUrl") or FALLBACK_BASE_URL)
    api_key = resolve_secret(provider["apiKey"])
    if not api_key:
        raise ValueError(f"provider {PROVIDER_NAME!r} has no usable apiKey in {path}")
    return base_url, api_key


def get_client(models_json: str | Path = DEFAULT_MODELS_JSON) -> Any:
    """Return a cached ``openai.OpenAI`` client for the LLM Collective."""
    from openai import OpenAI

    base_url, api_key = read_provider_config(models_json)
    key = (base_url, api_key)
    if key not in _clients:
        _clients[key] = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
    return _clients[key]


def list_models(client: Any | None = None) -> list[str]:
    """Return the model ids served by the endpoint (GET /v1/models)."""
    client = client or get_client()
    return sorted(m.id for m in client.models.list().data)


# ------------------------------------------------------------------
# Image encoding (pure helpers, unit-tested)
# ------------------------------------------------------------------

def resized_dims(size: tuple[int, int], max_side: int | None) -> tuple[int, int]:
    """Target (w, h) when the longest side is capped at *max_side*.

    Returns the original size unchanged when *max_side* is falsy or the
    image already fits.  Never upscales.
    """
    w, h = size
    if not max_side or max(w, h) <= max_side:
        return w, h
    scale = max_side / max(w, h)
    return max(1, round(w * scale)), max(1, round(h * scale))


def image_to_data_url(
    image: Image.Image, max_side: int | None = None, quality: int = 90,
) -> str:
    """Encode a PIL image as a base64 data URL (JPEG; PNG if it has alpha).

    The image is resized (LANCZOS) only when *max_side* is given and the
    longest side exceeds it; original detail is otherwise preserved.
    """
    tw, th = resized_dims(image.size, max_side)
    if (tw, th) != image.size:
        image = image.resize((tw, th), Image.LANCZOS)
    buf = io.BytesIO()
    if image.mode in ("RGBA", "LA", "P"):
        image.save(buf, format="PNG")
        mime = "image/png"
    else:
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{b64}"


def load_image_source(source: str | Path | Image.Image) -> Image.Image:
    """Accept a path or PIL image and return an RGB PIL image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    return Image.open(str(source)).convert("RGB")


# ------------------------------------------------------------------
# Chat with an image
# ------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    import openai

    return isinstance(
        exc,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError),
    ) or (isinstance(exc, openai.APIStatusError) and exc.status_code >= 500)


def chat_image(
    model: str,
    image_path_or_PIL: str | Path | Image.Image,
    prompt: str,
    max_side: int | None = None,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    retries: int | None = None,
    retry_budget_s: float = 600.0,
    client: Any | None = None,
    **kwargs: Any,
) -> str:
    """Send *image* + *prompt* to *model*; return the response text.

    Records ``latency_s``, token ``usage``, request byte size and attempt
    count for this call in :data:`LAST_CALL`.  Retries 429/5xx/connection
    errors with exponential backoff + jitter, staying within *retry_budget_s*
    seconds of total time per call (default ~10 min).  ``retries`` caps the
    number of additional attempts (default: unbounded, budget-limited).  Extra
    *kwargs* are passed through to ``chat.completions.create`` (e.g.
    ``extra_body``).
    """
    client = client or get_client()
    image = load_image_source(image_path_or_PIL)
    data_url = image_to_data_url(image, max_side=max_side)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }]

    last_err: Exception | None = None
    attempts = 0
    budget_start = time.monotonic()
    max_attempts = (retries + 1) if retries is not None else None
    attempt = 0
    while True:
        if time.monotonic() - budget_start >= retry_budget_s:
            err = f"retry budget exhausted: {str(last_err)[:300]}" if last_err else "retry budget exhausted"
            LAST_CALL.update({"attempts": attempts, "error": err})
            raise last_err

        attempts = attempt + 1
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            latency = time.monotonic() - t0
            usage = getattr(resp, "usage", None)
            LAST_CALL.clear()
            LAST_CALL.update({
                "model": model,
                "latency_s": round(latency, 2),
                "attempts": attempts,
                "request_data_url_bytes": len(data_url),
                "sent_image_size": resized_dims(image.size, max_side),
                "original_image_size": image.size,
                "usage": usage.model_dump() if usage is not None else None,
                "finish_reason": getattr(resp.choices[0], "finish_reason", None),
            })
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last_err = exc
            elapsed = time.monotonic() - budget_start
            if not _is_retryable(exc) or (max_attempts is not None and attempts >= max_attempts):
                LAST_CALL.update({"latency_s": round(time.monotonic() - t0, 2),
                                  "attempts": attempts, "error": str(exc)[:400]})
                raise
            if elapsed >= retry_budget_s:
                err = f"retry budget exhausted: {str(exc)[:300]}"
                LAST_CALL.update({"attempts": attempts, "error": err})
                raise
            backoff = min(30.0, 2.0 ** (attempt + 1)) * (1 + random.random() * 0.3)
            time.sleep(min(backoff, max(0.0, retry_budget_s - elapsed)))
            attempt += 1
