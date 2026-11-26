"""Shared VLM inference layer with two backends.

Backend ``"openai"``
    Uses the OpenAI Python client.  A single code-path that works with:

    - **HF Inference Providers** — ``base_url="https://router.huggingface.co/v1"``
      (default when no *base_url* is given), ``api_key`` = your HF token.
    - **vLLM server** — ``base_url="http://localhost:8000/v1"``
    - **llama-server** — ``base_url="http://localhost:8080/v1"``
    - Any other OpenAI-compatible endpoint.

Backend ``"transformers"``
    Loads the model locally with ``transformers`` + ``torch``.
    Requires a CUDA GPU.  Models are cached after the first call.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from .model_options import structured_output_template_kwargs

class VLMResponse(str):
    """String response carrying the provider finish reason when available."""

    def __new__(cls, text: str, finish_reason: str | None = None, usage: Any = None):
        obj = super().__new__(cls, text)
        obj.finish_reason = finish_reason
        obj.usage = usage
        return obj


# Cached OpenAI clients keyed by (base_url, api_key).
_openai_clients: dict[tuple[str, str], Any] = {}
_client_lock = threading.Lock()
_log_lock = threading.Lock()

HF_INFERENCE_URL = "https://router.huggingface.co/v1"


# ------------------------------------------------------------------
# Image encoding
# ------------------------------------------------------------------

_MAX_DIMENSION = 1280


def _pil_to_data_url(image: Image.Image, max_dimension: int | None = 1280,
                     image_format: str | None = None, jpeg_quality: int = 85) -> str:
    """Encode a PIL Image as a base64 data URL, resizing if needed.

    Large images are downscaled to fit within _MAX_DIMENSION and saved as
    JPEG to avoid hitting API request-size limits.
    """
    w, h = image.size
    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("max_dimension must be positive or None")
    if image_format not in (None, "jpeg", "png"):
        raise ValueError("image_format must be jpeg or png")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if max_dimension is not None and max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        image = image.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS,
        )

    buf = io.BytesIO()
    if image_format == "png" or (image_format is None and image.mode == "RGBA"):
        image.save(buf, format="PNG")
        mime = "image/png"
    else:
        image.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
        mime = "image/jpeg"

    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ------------------------------------------------------------------
# OpenAI-compatible backend
# ------------------------------------------------------------------

def _get_client(base_url: str, api_key: str) -> Any:
    key = (base_url, api_key)
    if key not in _openai_clients:
        with _client_lock:
            if key not in _openai_clients:
                from openai import OpenAI
                _openai_clients[key] = OpenAI(
                    base_url=base_url, api_key=api_key,
                )
    return _openai_clients[key]


def _resolve_api_key(api_key: str | None) -> str:
    return (
        api_key
        or os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
        or "no-key"
    )


def _run_openai(
    image: Image.Image,
    prompt: str,
    model_id: str,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    max_dimension: int | None = 1280,
    image_format: str | None = None,
    jpeg_quality: int = 85,
    temperature: float = 0, top_p: float = 1, seed: int | None = None,
    extra_body: dict | None = None,
) -> str:
    import time

    base_url = base_url or HF_INFERENCE_URL
    api_key = _resolve_api_key(api_key)
    client = _get_client(base_url, api_key)

    data_url = _pil_to_data_url(image, max_dimension, image_format, jpeg_quality)
    template_kwargs = structured_output_template_kwargs(model_id)
    body = dict(extra_body or {})
    if template_kwargs:
        body["chat_template_kwargs"] = {**template_kwargs, **body.get("chat_template_kwargs", {})}
    request_kwargs = {"extra_body": body} if body else {}
    if seed is not None:
        request_kwargs["seed"] = seed

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **request_kwargs,
            )
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            return VLMResponse(choice.message.content or "", getattr(choice, "finish_reason", None), usage)
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise last_err


# ------------------------------------------------------------------
# Transformers backend
# ------------------------------------------------------------------

def _load_vlm(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def _extract_eos_ids(model: Any, processor: Any) -> set[int]:
    """Collect EOS token IDs from generation config or tokenizer when available."""
    raw = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if raw is None:
        raw = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    if raw is None:
        raw = getattr(processor, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {int(x) for x in raw if x is not None}
    return {int(raw)}


def _extract_pad_id(model: Any, processor: Any) -> int | None:
    """Return pad token ID from generation config or tokenizer when available."""
    pad = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    if pad is None:
        pad = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if pad is None:
        pad = getattr(processor, "pad_token_id", None)
    return int(pad) if pad is not None else None


def _derive_finish_reason(
    seq: Any, max_tokens: int, eos_ids: set[int], pad_id: int | None,
) -> str | None:
    """Derive OpenAI-style finish_reason ('length' or 'stop') for a generated sequence."""
    try:
        tokens = seq.tolist() if hasattr(seq, "tolist") else list(seq)
    except TypeError:
        return None
    if pad_id is not None and pad_id not in eos_ids:
        while tokens and tokens[-1] == pad_id:
            tokens.pop()
    if any(t in eos_ids for t in tokens):
        return "stop"
    if len(tokens) >= max_tokens:
        return "length"
    return "stop"


def _run_transformers_batch(
    images: list[Image.Image],
    prompts: list[str],
    model_id: str,
    max_tokens: int,
    temperature: float = 0, top_p: float = 1, seed: int | None = None,
) -> list[VLMResponse]:
    """Generate for a *batch* of (image, prompt) pairs in one forward pass.

    This is the throughput path for the GPU judge jobs: the small judge models
    (e.g. ``LiquidAI/LFM3-VL-2B``, ``google/gemma-4-E4B-it``) are not on HF
    Inference Providers, so they run locally with ``transformers`` — feeding one
    image at a time wastes the GPU.

    Uses the standard multimodal batching path: a list of conversations (image
    embedded per turn) through ``apply_chat_template`` with ``padding=True`` and
    ``padding_side="left"`` (left padding is what decoder-only ``generate`` wants,
    so every row's new tokens start at the same offset and one slice serves the
    whole batch).
    """
    import torch

    from .utils import get_or_load

    model, processor = get_or_load(f"vlm:{model_id}", lambda: _load_vlm(model_id))

    conversations = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        for img, prompt in zip(images, prompts)
    ]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        **structured_output_template_kwargs(model_id),
    ).to(model.device)

    generation_kwargs = {"do_sample": temperature > 0}
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, **generation_kwargs)

    sequences = getattr(out, "sequences", out)
    gen = sequences[:, inputs["input_ids"].shape[1]:]
    texts = processor.batch_decode(gen, skip_special_tokens=True)
    eos_ids = _extract_eos_ids(model, processor)
    pad_id = _extract_pad_id(model, processor)
    return [
        VLMResponse(text, _derive_finish_reason(seq, max_tokens, eos_ids, pad_id))
        for text, seq in zip(texts, gen)
    ]


def _run_transformers(
    image: Image.Image,
    prompt: str,
    model_id: str,
    max_tokens: int,
    temperature: float = 0, top_p: float = 1, seed: int | None = None,
) -> VLMResponse:
    return _run_transformers_batch(
        [image], [prompt], model_id, max_tokens, temperature, top_p, seed,
    )[0]


def _append_raw_log(raw_response_log: str | os.PathLike, response: Any) -> None:
    record = {
        "response": str(response),
        "finish_reason": getattr(response, "finish_reason", None),
        "usage": getattr(response, "usage", None),
    }
    line = json.dumps(record) + "\n"
    path = Path(raw_response_log)
    with _log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as log:
            log.write(line)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def run_vlm(
    image: Image.Image,
    prompt: str,
    model_id: str,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    max_dimension: int | None = 1280,
    image_format: str | None = None,
    jpeg_quality: int = 85,
    temperature: float = 0, top_p: float = 1, seed: int | None = None,
    extra_body: dict | None = None, raw_response_log: str | os.PathLike | None = None,
) -> str:
    """Send *image* + *prompt* to a VLM and return the response text.

    Parameters
    ----------
    image : PIL.Image
        RGB image (already loaded).
    prompt : str
        User message text.
    model_id : str
        Model identifier — a HF repo id for both backends, or whatever
        your OpenAI-compatible server expects.
    backend : ``"openai"`` | ``"transformers"``
        ``"openai"`` routes through the OpenAI Python client and works
        with HF Inference Providers, vLLM, llama-server, or any
        compatible endpoint.  ``"transformers"`` loads the model locally.
    base_url : str, optional
        API base URL for the ``openai`` backend.  Defaults to
        ``https://router.huggingface.co/v1`` (HF Inference Providers).
        Set to ``http://localhost:8000/v1`` for a local vLLM server,
        ``http://localhost:8080/v1`` for llama-server, etc.
    api_key : str, optional
        API key / token.  Falls back to ``HF_TOKEN`` then
        ``OPENAI_API_KEY`` environment variables.
    max_tokens : int
        Maximum tokens to generate.

    temperature, top_p, seed
        Generation sampling controls. Seed is omitted from OpenAI requests when unset.
    extra_body : dict, optional
        Additional OpenAI-compatible request fields. Chat template kwargs are merged
        with model defaults, with the caller taking precedence.
    raw_response_log : str or Path, optional
        Append response text, finish reason, and usage as one JSON object per call.

    Returns
    -------
    str
        The model's response text.
    """
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be nonnegative and top_p in (0, 1]")
    if backend == "openai":
        response = _run_openai(image, prompt, model_id, base_url, api_key, max_tokens,
                               max_dimension, image_format, jpeg_quality, temperature,
                               top_p, seed, extra_body)
    elif backend == "transformers":
        if extra_body:
            raise ValueError("extra_body is only supported by the openai backend")
        response = _run_transformers(image, prompt, model_id, max_tokens,
                                     temperature, top_p, seed)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Use 'openai' or 'transformers'.")
    if raw_response_log is not None:
        _append_raw_log(raw_response_log, response)
    return response


def run_vlm_batch(
    images: list[Image.Image],
    prompts: list[str],
    model_id: str,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    max_dimension: int | None = 1280,
    image_format: str | None = None,
    jpeg_quality: int = 85,
    temperature: float = 0,
    top_p: float = 1,
    seed: int | None = None,
    extra_body: dict | None = None,
    raw_response_log: str | os.PathLike | None = None,
) -> list[str]:
    """Run a VLM over a *batch* of (image, prompt) pairs; one response each.

    The ``transformers`` backend runs the whole batch through a single
    ``generate`` call (the throughput path for the GPU judge jobs). The
    ``openai`` backend has no batched chat endpoint, so it falls back to
    sequential per-item calls — same results, just not faster.

    *images* and *prompts* must be the same length; the returned list is aligned
    with them.
    """
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be nonnegative and top_p in (0, 1]")
    if len(images) != len(prompts):
        raise ValueError(
            f"images ({len(images)}) and prompts ({len(prompts)}) must match",
        )
    if not images:
        return []
    if backend == "transformers":
        if extra_body:
            raise ValueError("extra_body is only supported by the openai backend")
        responses = _run_transformers_batch(
            images, prompts, model_id, max_tokens, temperature, top_p, seed,
        )
    elif backend == "openai":
        responses = [
            _run_openai(
                img, prompt, model_id, base_url, api_key, max_tokens,
                max_dimension, image_format, jpeg_quality,
                temperature, top_p, seed, extra_body,
            )
            for img, prompt in zip(images, prompts)
        ]
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Use 'openai' or 'transformers'."
        )
    if raw_response_log is not None:
        for resp in responses:
            _append_raw_log(raw_response_log, resp)
    return responses
