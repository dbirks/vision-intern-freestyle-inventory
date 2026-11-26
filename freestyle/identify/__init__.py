"""Crop identification module for Freestyle cartridge recognition."""

from freestyle.identify.classify import (
    KNNClassifier,
    LinearProbe,
    PrototypeClassifier,
    calibrate_unknown_threshold,
    leave_one_group_out_eval,
    select_unknown_threshold,
)
from freestyle.identify.embed import MODEL_ALIASES, MODEL_SPECS, Embedder, l2_normalize, letterbox

__all__ = [
    "Embedder",
    "KNNClassifier",
    "LinearProbe",
    "MODEL_ALIASES",
    "MODEL_SPECS",
    "PrototypeClassifier",
    "calibrate_unknown_threshold",
    "l2_normalize",
    "leave_one_group_out_eval",
    "letterbox",
    "select_unknown_threshold",
]
