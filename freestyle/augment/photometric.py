"""Photometric and lens augmentations matching Freestyle cabinet failure modes.

All transforms are pure functions ``(img, rng, ...) -> PIL.Image.Image`` that
preserve the input image's size and mode without mutating the input.
"""

from __future__ import annotations

import io
import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

RngLike = np.random.Generator | random.Random | int | None


def _coerce_rng(rng: RngLike) -> np.random.Generator:
    """Normalize ``np.random.Generator``, ``random.Random``, int seed, or None."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, random.Random):
        return np.random.default_rng(rng.getrandbits(64))
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError(f"Unsupported rng type: {type(rng)!r}")


def _split_alpha(img: Image.Image) -> tuple[Image.Image, Image.Image | None, str]:
    """Return an RGB/L working image, optional alpha channel, and original mode."""
    mode = img.mode
    if mode == "RGBA":
        rgb = img.convert("RGB")
        alpha = img.getchannel("A")
        return rgb, alpha, mode
    if mode == "L":
        return img.copy(), None, mode
    if mode == "RGB":
        return img.copy(), None, mode
    return img.convert("RGB"), None, mode


def _restore_mode(work: Image.Image, alpha: Image.Image | None, mode: str) -> Image.Image:
    """Restore the caller's original PIL image mode (and alpha if RGBA)."""
    if mode == "RGBA":
        rgb = work.convert("RGB")
        if alpha is not None:
            rgb.putalpha(alpha)
        return rgb
    if work.mode == mode:
        return work
    return work.convert(mode)


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Build a normalized 2D directional motion-blur kernel."""
    ksize = max(3, int(math.ceil(length)) + 2)
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = (ksize - 1) / 2.0
    theta = math.radians(angle_deg)
    dx = math.cos(theta) * (length / 2.0)
    dy = math.sin(theta) * (length / 2.0)

    steps = max(32, int(math.ceil(length * 8)))
    ts = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    xs = center + dx * ts
    ys = center + dy * ts

    # Bilinear splat along the continuous segment for smooth subpixel angles.
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    for ox, wx in ((0, 1.0 - fx), (1, fx)):
        for oy, wy in ((0, 1.0 - fy), (1, fy)):
            ix = np.clip(x0 + ox, 0, ksize - 1)
            iy = np.clip(y0 + oy, 0, ksize - 1)
            np.add.at(kernel, (iy, ix), wx * wy)

    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    else:
        kernel[ksize // 2, ksize // 2] = 1.0
    return kernel


def _convolve_sparse(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 2D or 3D float32 array with a sparse 2D kernel using reflect padding."""
    h, w = arr.shape[:2]
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    if arr.ndim == 2:
        padded = np.pad(arr, ((py, py), (px, px)), mode="edge")
    else:
        padded = np.pad(arr, ((py, py), (px, px), (0, 0)), mode="edge")

    out = np.zeros_like(arr, dtype=np.float32)
    nz_y, nz_x = np.nonzero(kernel > 1e-6)
    weights = kernel[nz_y, nz_x]
    weights = weights / weights.sum()
    for iy, ix, wt in zip(nz_y.tolist(), nz_x.tolist(), weights.tolist(), strict=True):
        out += padded[iy : iy + h, ix : ix + w] * wt
    return out


def motion_blur(
    img: Image.Image,
    rng: RngLike = None,
    *,
    length_range: tuple[float, float] = (3.0, 25.0),
    angle_range: tuple[float, float] = (0.0, 180.0),
) -> Image.Image:
    """Apply directional motion blur matching a swinging cabinet door camera."""
    gen = _coerce_rng(rng)
    lo_len, hi_len = length_range
    if lo_len < 0 or hi_len < lo_len:
        raise ValueError(f"Invalid length_range: {length_range}")
    length = float(gen.uniform(lo_len, hi_len))
    angle = float(gen.uniform(angle_range[0], angle_range[1]))

    work, alpha, orig_mode = _split_alpha(img)
    if length <= 1.0 or min(work.size) < 2:
        return _restore_mode(work, alpha, orig_mode)

    kernel = _motion_kernel(length, angle)
    arr = np.asarray(work, dtype=np.float32)
    blurred = np.clip(_convolve_sparse(arr, kernel), 0.0, 255.0).round().astype(np.uint8)
    out = Image.fromarray(blurred, mode=work.mode)
    return _restore_mode(out, alpha, orig_mode)


def low_light(
    img: Image.Image,
    rng: RngLike = None,
    *,
    target_luma: float = 70.0,
    luma_jitter: float = 12.0,
    read_noise: float = 4.0,
    shot_noise: float = 0.08,
) -> Image.Image:
    """Simulate dim cabinet illumination (luma mean ~70) plus sensor gain/noise."""
    if target_luma <= 0:
        raise ValueError(f"target_luma must be positive, got {target_luma}")
    gen = _coerce_rng(rng)
    work, alpha, orig_mode = _split_alpha(img)
    arr = np.asarray(work, dtype=np.float32)

    if arr.ndim == 3:
        luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        mean_luma = float(luma.mean())
    else:
        mean_luma = float(arr.mean())

    desired = float(
        np.clip(
            gen.uniform(target_luma - luma_jitter, target_luma + luma_jitter),
            15.0,
            220.0,
        )
    )
    if mean_luma > 1.0:
        gamma = float(gen.uniform(0.95, 1.15))
        norm = np.clip(arr / 255.0, 0.0, 1.0) ** gamma
        scaled = norm * 255.0
        if scaled.ndim == 3:
            new_luma = float(
                (0.299 * scaled[..., 0] + 0.587 * scaled[..., 1] + 0.114 * scaled[..., 2]).mean()
            )
        else:
            new_luma = float(scaled.mean())
        scale = desired / max(new_luma, 1.0)
        arr = scaled * scale

    if arr.ndim == 3:
        ch_gain = gen.uniform(0.96, 1.04, size=(1, 1, arr.shape[2])).astype(np.float32)
        arr = arr * ch_gain

    shot_std = shot_noise * np.sqrt(np.maximum(arr, 1.0))
    noise = gen.normal(0.0, 1.0, size=arr.shape).astype(np.float32) * shot_std
    if read_noise > 0:
        noise += gen.normal(0.0, read_noise, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 255.0).round().astype(np.uint8)
    out = Image.fromarray(noisy, mode=work.mode)
    return _restore_mode(out, alpha, orig_mode)


def jpeg_compress(
    img: Image.Image,
    rng: RngLike = None,
    *,
    quality_range: tuple[int, int] = (60, 90),
) -> Image.Image:
    """Re-encode through JPEG at random quality to match camera compression artefacts."""
    lo_q, hi_q = int(quality_range[0]), int(quality_range[1])
    if not (1 <= lo_q <= hi_q <= 100):
        raise ValueError(f"Invalid quality_range: {quality_range}")
    gen = _coerce_rng(rng)
    quality = int(gen.integers(lo_q, hi_q + 1))

    work, alpha, orig_mode = _split_alpha(img)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as decoded:
        decoded.load()
        out = decoded.copy()
    return _restore_mode(out, alpha, orig_mode)


def color_cast(
    img: Image.Image,
    rng: RngLike = None,
    *,
    gain_range: tuple[float, float] = (0.85, 1.15),
    bias_range: tuple[float, float] = (-6.0, 6.0),
) -> Image.Image:
    """Apply a random white-balance / cabinet LED colour cast."""
    lo_g, hi_g = gain_range
    if lo_g <= 0 or hi_g < lo_g:
        raise ValueError(f"Invalid gain_range: {gain_range}")
    gen = _coerce_rng(rng)
    work, alpha, orig_mode = _split_alpha(img)
    arr = np.asarray(work, dtype=np.float32)
    if arr.ndim == 3:
        gains = gen.uniform(lo_g, hi_g, size=(1, 1, arr.shape[2])).astype(np.float32)
        biases = gen.uniform(bias_range[0], bias_range[1], size=(1, 1, arr.shape[2])).astype(
            np.float32
        )
    else:
        gains = np.float32(gen.uniform(lo_g, hi_g))
        biases = np.float32(gen.uniform(bias_range[0], bias_range[1]))
    cast = np.clip(arr * gains + biases, 0.0, 255.0).round().astype(np.uint8)
    out = Image.fromarray(cast, mode=work.mode)
    return _restore_mode(out, alpha, orig_mode)


def vignette(
    img: Image.Image,
    rng: RngLike = None,
    *,
    strength_range: tuple[float, float] = (0.25, 0.65),
    power_range: tuple[float, float] = (1.5, 3.0),
    center_jitter: float = 0.05,
) -> Image.Image:
    """Apply radial illumination falloff toward frame corners."""
    lo_s, hi_s = strength_range
    if not (0.0 <= lo_s <= hi_s <= 1.0):
        raise ValueError(f"Invalid strength_range: {strength_range}")
    gen = _coerce_rng(rng)
    work, alpha, orig_mode = _split_alpha(img)
    arr = np.asarray(work, dtype=np.float32)
    h, w = arr.shape[:2]

    strength = float(gen.uniform(lo_s, hi_s))
    power = float(gen.uniform(power_range[0], power_range[1]))
    cx = w * (0.5 + float(gen.uniform(-center_jitter, center_jitter)))
    cy = h * (0.5 + float(gen.uniform(-center_jitter, center_jitter)))

    xs = (np.arange(w, dtype=np.float32) - cx) / max(w * 0.5, 1.0)
    ys = (np.arange(h, dtype=np.float32) - cy) / max(h * 0.5, 1.0)
    r = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2) / math.sqrt(2.0)
    mask = np.clip(1.0 - strength * (np.clip(r, 0.0, 1.5) ** power), 0.05, 1.0).astype(np.float32)
    if arr.ndim == 3:
        mask = mask[..., None]
    out_arr = np.clip(arr * mask, 0.0, 255.0).round().astype(np.uint8)
    out = Image.fromarray(out_arr, mode=work.mode)
    return _restore_mode(out, alpha, orig_mode)


def _bilinear_sample(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, border_mode: str) -> np.ndarray:
    """Bilinearly sample ``arr`` (H, W[, C]) at float coordinates ``(xs, ys)``."""
    h, w = arr.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = (xs - x0).astype(np.float32)
    fy = (ys - y0).astype(np.float32)

    x1 = x0 + 1
    y1 = y0 + 1

    if border_mode == "edge":
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)
        valid = None
    elif border_mode == "constant":
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)
        valid = (xs >= 0.0) & (xs <= w - 1.0) & (ys >= 0.0) & (ys <= h - 1.0)
    else:
        raise ValueError(f"border_mode must be 'edge' or 'constant', got {border_mode!r}")

    v00 = arr[y0c, x0c]
    v01 = arr[y0c, x1c]
    v10 = arr[y1c, x0c]
    v11 = arr[y1c, x1c]

    if arr.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]

    out = (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v01 * fx * (1.0 - fy)
        + v10 * (1.0 - fx) * fy
        + v11 * fx * fy
    )
    if valid is not None:
        mask = valid[..., None] if arr.ndim == 3 else valid
        out = np.where(mask, out, 0.0)
    return out


def fisheye_warp(
    img: Image.Image,
    rng: RngLike = None,
    *,
    k1_range: tuple[float, float] = (-0.35, -0.05),
    k2_range: tuple[float, float] = (0.0, 0.0),
    border_mode: str = "edge",
) -> Image.Image:
    """Apply synthetic radial barrel (k1 < 0) or pincushion (k1 > 0) warp to an image/crop."""
    if k1_range[1] < k1_range[0] or k2_range[1] < k2_range[0]:
        raise ValueError("Invalid k1_range or k2_range")
    gen = _coerce_rng(rng)
    k1 = float(gen.uniform(k1_range[0], k1_range[1]))
    k2 = float(gen.uniform(k2_range[0], k2_range[1]))

    work, alpha, orig_mode = _split_alpha(img)
    w, h = work.size
    if w < 2 or h < 2 or (k1 == 0.0 and k2 == 0.0):
        return _restore_mode(work, alpha, orig_mode)

    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    norm = max(0.5 * math.hypot(w, h), 1.0)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xn = (gx - cx) / norm
    yn = (gy - cy) / norm
    r2 = xn * xn + yn * yn

    # Negative k1 corresponds to barrel distortion (source radius > dest radius).
    if k1 <= 0.0:
        scale = 1.0 - k1 * r2 - k2 * (r2 * r2)
        scale = np.maximum(scale, 0.1)
    else:
        scale = 1.0 / np.maximum(1.0 + k1 * r2 + k2 * (r2 * r2), 0.1)

    xs = cx + xn * scale * norm
    ys = cy + yn * scale * norm

    arr = np.asarray(work, dtype=np.float32)
    warped = np.clip(_bilinear_sample(arr, xs, ys, border_mode), 0.0, 255.0).round().astype(np.uint8)
    out = Image.fromarray(warped, mode=work.mode)
    if alpha is not None:
        a_arr = np.asarray(alpha, dtype=np.float32)
        a_warped = (
            np.clip(_bilinear_sample(a_arr, xs, ys, border_mode), 0.0, 255.0).round().astype(np.uint8)
        )
        alpha = Image.fromarray(a_warped, mode="L")
    return _restore_mode(out, alpha, orig_mode)


OPS: dict[str, Callable[..., Image.Image]] = {
    "fisheye_warp": fisheye_warp,
    "motion_blur": motion_blur,
    "color_cast": color_cast,
    "low_light": low_light,
    "vignette": vignette,
    "jpeg_compress": jpeg_compress,
}

DEFAULT_PIPELINE: dict[str, dict[str, Any]] = {
    "fisheye_warp": {"p": 0.3, "k1_range": (-0.35, -0.05)},
    "motion_blur": {"p": 0.5, "length_range": (3.0, 25.0)},
    "color_cast": {"p": 0.5},
    "low_light": {"p": 0.4, "target_luma": 70.0},
    "vignette": {"p": 0.5},
    "jpeg_compress": {"p": 0.7, "quality_range": (60, 90)},
}


def compose(cfg: dict[str, Any] | None = None) -> Callable[[Image.Image, RngLike], Image.Image]:
    """Build a deterministic-when-seeded photometric pipeline from ``cfg``.

    ``cfg`` may be ``None`` (uses ``DEFAULT_PIPELINE``), a mapping of
    ``{op_name: bool | {"p": prob, ...}}``, or ``{"ops": [{"name": ..., "p": ..., ...}]}``.
    """
    spec = DEFAULT_PIPELINE if cfg is None else cfg
    steps: list[tuple[Callable[..., Image.Image], float, dict[str, Any]]] = []

    if "ops" in spec:
        for entry in spec["ops"]:
            if isinstance(entry, str):
                name, params = entry, {}
            else:
                params = dict(entry)
                name = params.pop("name")
            if name not in OPS:
                raise KeyError(f"Unknown photometric op: {name!r}")
            prob = float(params.pop("p", 1.0))
            steps.append((OPS[name], prob, params))
    else:
        for name, val in spec.items():
            if name not in OPS:
                raise KeyError(f"Unknown photometric op: {name!r}")
            if val is False or val is None:
                continue
            if val is True:
                prob, params = 1.0, {}
            elif isinstance(val, dict):
                params = dict(val)
                prob = float(params.pop("p", 1.0))
            else:
                raise TypeError(f"Invalid config for {name!r}: {val!r}")
            if prob > 0:
                steps.append((OPS[name], prob, params))

    def _apply(img: Image.Image, rng: RngLike = None) -> Image.Image:
        gen = _coerce_rng(rng)
        out = img.copy()
        for fn, prob, params in steps:
            if prob >= 1.0 or float(gen.random()) < prob:
                out = fn(out, gen, **params)
        return out

    return _apply


__all__ = [
    "DEFAULT_PIPELINE",
    "OPS",
    "color_cast",
    "compose",
    "fisheye_warp",
    "jpeg_compress",
    "low_light",
    "motion_blur",
    "vignette",
]
