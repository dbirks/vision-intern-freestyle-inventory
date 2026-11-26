"""Crop embedding extraction with Hugging Face vision backbones and a test stub."""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "stub": {
        "family": "stub",
        "default_size": (64, 64),
        "gated": False,
        "dim": 168,
    },
    "facebook/dinov2-base": {
        "family": "dinov2",
        "default_size": (224, 224),
        "gated": False,
    },
    "facebook/dinov2-large": {
        "family": "dinov2",
        "default_size": (224, 224),
        "gated": False,
    },
    "facebook/dinov3-vitb16-pretrain-lvd1689m": {
        "family": "dinov3",
        "default_size": (224, 224),
        "gated": True,
    },
    "facebook/dinov3-vitl16-pretrain-lvd1689m": {
        "family": "dinov3",
        "default_size": (224, 224),
        "gated": True,
    },
    "google/siglip2-base-patch16-512": {
        "family": "siglip2",
        "default_size": (512, 512),
        "gated": False,
    },
    "google/siglip2-so400m-patch16-512": {
        "family": "siglip2",
        "default_size": (512, 512),
        "gated": False,
    },
}

MODEL_ALIASES: dict[str, str] = {
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "siglip2-base-512": "google/siglip2-base-patch16-512",
    "siglip2-so400m-512": "google/siglip2-so400m-patch16-512",
}


def resolve_model_name(name: str) -> str:
    """Resolve short convenience aliases to full model IDs."""
    return MODEL_ALIASES.get(name, name)


def _parse_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"input_size must be positive, got {size}")
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"input_size must be int or (width, height), got {size!r}")
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"input_size dimensions must be positive, got {(w, h)}")
    return (w, h)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1D vector or rows of a 2D array as float32."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr.copy()
        return (arr / max(norm, eps)).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        return arr.copy()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, eps)
    return np.where(norms > 0.0, arr / safe_norms, 0.0).astype(np.float32, copy=False)


def letterbox(
    image: Image.Image,
    size: int | tuple[int, int],
    fill: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Resize preserving aspect ratio and pad with neutral grey to ``size`` (W, H)."""
    target_w, target_h = _parse_size(size)
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Image must have positive dimensions, got {image.size}")
    rgb = image.convert("RGB") if image.mode != "RGB" else image

    scale = min(target_w / src_w, target_h / src_h)
    new_w = min(target_w, max(1, round(src_w * scale)))
    new_h = min(target_h, max(1, round(src_h * scale)))

    if (new_w, new_h) != (src_w, src_h):
        resampling = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
        resized = rgb.resize((new_w, new_h), resampling)
    else:
        resized = rgb

    canvas = Image.new("RGB", (target_w, target_h), fill)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _stub_features(image: Image.Image) -> np.ndarray:
    """Compute deterministic global + 2x2 spatial RGB colour-histogram features."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    parts: list[np.ndarray] = []

    n_pix = float(max(1, h * w))
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0.0, 256.0))
        parts.append(hist.astype(np.float32) / n_pix)

    mid_y = max(1, h // 2)
    mid_x = max(1, w // 2)
    quadrants = (
        arr[:mid_y, :mid_x],
        arr[:mid_y, mid_x:],
        arr[mid_y:, :mid_x],
        arr[mid_y:, mid_x:],
    )
    for quad in quadrants:
        q_pix = float(max(1, quad.shape[0] * quad.shape[1]))
        for ch in range(3):
            q_hist, _ = np.histogram(quad[:, :, ch], bins=8, range=(0.0, 256.0))
            parts.append(q_hist.astype(np.float32) / q_pix)
        parts.append((quad.mean(axis=(0, 1)) / 255.0).astype(np.float32))
        parts.append((quad.std(axis=(0, 1)) / 255.0).astype(np.float32))

    return np.concatenate(parts, axis=0)


def _infer_processor_size(processor: Any, model: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    crop_size = getattr(processor, "crop_size", None)
    if isinstance(crop_size, dict) and "width" in crop_size and "height" in crop_size:
        return (int(crop_size["width"]), int(crop_size["height"]))
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        if "width" in size and "height" in size:
            return (int(size["width"]), int(size["height"]))
        if "shortest_edge" in size and not crop_size:
            s = int(size["shortest_edge"])
            return (s, s)
    elif isinstance(size, int) and size > 0:
        return (size, size)

    cfg = getattr(model, "config", None)
    vision_cfg = getattr(cfg, "vision_config", None)
    for obj in (vision_cfg, cfg):
        img_size = getattr(obj, "image_size", None)
        if isinstance(img_size, int) and img_size > 0:
            return (img_size, img_size)
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            return (int(img_size[1]), int(img_size[0]))
    return fallback


def _infer_embedding_dim(model: Any) -> int | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for attr in ("projection_dim", "hidden_size"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is not None:
        val = getattr(vision_cfg, "hidden_size", None)
        if isinstance(val, int) and val > 0:
            return val
    return None


def _is_gated_error(exc: Exception) -> bool:
    cls_name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "gated" in cls_name
        or "gated repo" in msg
        or "401" in msg
        or "403" in msg
        or "restricted" in msg
    )


def _processor_options(processor: Any, input_size: tuple[int, int]) -> dict[str, Any]:
    proc_fn = getattr(processor, "preprocess", processor)
    try:
        params = inspect.signature(proc_fn).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        params = {}
        accepts_kwargs = False
    options: dict[str, Any] = {}
    for name in ("do_resize", "do_center_crop"):
        if accepts_kwargs or name in params:
            options[name] = False
    if "Siglip2ImageProcessor" in type(processor).__name__:
        patch = int(getattr(processor, "patch_size", 16))
        if any(side % patch for side in input_size):
            raise ValueError("SigLIP2 input dimensions must be divisible by patch size")
        options["max_num_patches"] = (input_size[0] // patch) * (input_size[1] // patch)
    return options


class Embedder:
    """Aspect-preserving crop embedder backed by HF transformers or a deterministic stub.

    Supported HF checkpoints include ``facebook/dinov2-base``,
    ``facebook/dinov2-large``, ``facebook/dinov3-vitb16-pretrain-lvd1689m`` (gated),
    ``google/siglip2-base-patch16-512``, and ``google/siglip2-so400m-patch16-512``,
    as well as ``"stub"`` for download-free tests.
    """

    def __init__(
        self,
        name: str = "facebook/dinov2-base",
        device: str = "auto",
        dtype: str | Any = "auto",
        *,
        input_size: int | tuple[int, int] | None = None,
        batch_size: int = 32,
        tta_hflip: bool = False,
        pad_color: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.raw_name = name
        self.name = resolve_model_name(name)
        self.spec = dict(MODEL_SPECS.get(self.name, {}))
        self.gated = bool(self.spec.get("gated", "dinov3" in self.name.lower()))
        self.batch_size = int(batch_size)
        self.tta_hflip = bool(tta_hflip)
        self.pad_color = pad_color

        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._proc_kwargs: dict[str, Any] = {}
        self._device_str: str = "cpu"
        self._dtype: Any = None

        if self.name == "stub":
            self.device = "cpu"
            self.dtype = "float32"
            default_size = self.spec.get("default_size", (64, 64))
            self.input_size = _parse_size(input_size if input_size is not None else default_size)
            self.dim: int | None = int(self.spec.get("dim", 168))
        else:
            self._init_hf(device=device, dtype=dtype, input_size=input_size)

    def _init_hf(
        self,
        device: str,
        dtype: str | Any,
        input_size: int | tuple[int, int] | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Hugging Face Embedder requires the 'train' extra (torch, transformers). "
                "Install with `uv sync --extra train` or use Embedder('stub') for tests."
            ) from exc

        self._torch = torch
        if device == "auto":
            if torch.cuda.is_available():
                self._device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device_str = "mps"
            else:
                self._device_str = "cpu"
        else:
            self._device_str = str(device)
        self.device = self._device_str

        if dtype == "auto":
            if self._device_str.startswith("cuda"):
                self._dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                self._dtype = torch.float32
        elif isinstance(dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            if dtype.lower() not in dtype_map:
                raise ValueError(f"Unsupported dtype string: {dtype!r}")
            self._dtype = dtype_map[dtype.lower()]
        else:
            self._dtype = dtype
        self.dtype = str(self._dtype)

        try:
            try:
                self._processor = AutoImageProcessor.from_pretrained(self.name)
            except Exception:
                self._processor = AutoProcessor.from_pretrained(self.name)
            self._model = AutoModel.from_pretrained(self.name, dtype=self._dtype)
        except Exception as exc:
            if _is_gated_error(exc):
                raise RuntimeError(
                    f"Model {self.name!r} is gated or requires authentication on Hugging Face. "
                    "Accept the model license on huggingface.co and set HF_TOKEN."
                ) from exc
            raise

        self._model.to(self._device_str)
        self._model.eval()

        fallback_size = self.spec.get("default_size", (224, 224))
        if input_size is not None:
            self.input_size = _parse_size(input_size)
        else:
            self.input_size = _infer_processor_size(self._processor, self._model, fallback_size)
        self.dim = _infer_embedding_dim(self._model)

        self._proc_kwargs = _processor_options(self._processor, self.input_size)

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Apply aspect-preserving neutral-grey letterbox to ``self.input_size``."""
        return letterbox(image, self.input_size, fill=self.pad_color)

    def _embed_stub_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        feats = np.stack([_stub_features(img) for img in images], axis=0)
        return l2_normalize(feats)

    def _embed_hf_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        try:
            inputs = self._processor(images=list(images), return_tensors="pt", **self._proc_kwargs)
        except TypeError:
            inputs = self._processor(images=list(images), return_tensors="pt")

        model_inputs = {
            k: v.to(self._device_str) if hasattr(v, "to") else v for k, v in inputs.items()
        }
        # Cast floating pixel tensors to the model dtype while keeping integer masks intact.
        for key, val in list(model_inputs.items()):
            if torch.is_tensor(val) and torch.is_floating_point(val):
                model_inputs[key] = val.to(dtype=self._dtype)

        with torch.inference_mode():
            if hasattr(self._model, "get_image_features"):
                out = self._model.get_image_features(**model_inputs)
            else:
                out = self._model(**model_inputs)

        if torch.is_tensor(out):
            feats = out
        elif getattr(out, "pooler_output", None) is not None:
            feats = out.pooler_output
        elif getattr(out, "last_hidden_state", None) is not None:
            feats = out.last_hidden_state[:, 0]
        else:
            raise RuntimeError(f"Unsupported model output type from {self.name}: {type(out)}")

        arr = feats.detach().to(dtype=torch.float32, device="cpu").numpy()
        if self.dim is None:
            self.dim = int(arr.shape[1])
        return l2_normalize(arr)

    def _embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        if self.name == "stub":
            return self._embed_stub_batch(images)
        return self._embed_hf_batch(images)

    def embed(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int | None = None,
        tta_hflip: bool | None = None,
    ) -> np.ndarray:
        """Embed a sequence of PIL images into L2-normalized float32 vectors ``(N, D)``."""
        if len(images) == 0:
            dim = self.dim if self.dim is not None else 0
            return np.empty((0, dim), dtype=np.float32)

        bs = self.batch_size if batch_size is None else int(batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be positive, got {bs}")
        use_hflip = self.tta_hflip if tta_hflip is None else bool(tta_hflip)

        chunks: list[np.ndarray] = []
        for start in range(0, len(images), bs):
            batch_raw = images[start : start + bs]
            letterboxed = [self.preprocess(img) for img in batch_raw]
            emb = self._embed_batch(letterboxed)
            if use_hflip:
                flipped = [ImageOps.mirror(img) for img in letterboxed]
                emb_flip = self._embed_batch(flipped)
                emb = l2_normalize(0.5 * (emb + emb_flip))
            chunks.append(emb)

        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
