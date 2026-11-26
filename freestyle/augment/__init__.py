"""Synthetic data augmentation for Freestyle cartridge recognition."""

from freestyle.augment.copy_paste import (
    blend_patch,
    build_bank,
    poisson_blend,
    slot_shuffle,
    synthesize,
)
from freestyle.augment.photometric import (
    color_cast,
    compose,
    fisheye_warp,
    jpeg_compress,
    low_light,
    motion_blur,
    vignette,
)

__all__ = [
    "blend_patch",
    "build_bank",
    "color_cast",
    "compose",
    "fisheye_warp",
    "jpeg_compress",
    "low_light",
    "motion_blur",
    "poisson_blend",
    "slot_shuffle",
    "synthesize",
    "vignette",
]
