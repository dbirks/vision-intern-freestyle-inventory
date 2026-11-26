"""Zero-shot and promptable class-agnostic cartridge proposal generators."""

from __future__ import annotations

from freestyle.propose.backends import (
    BACKENDS,
    TRANSFORMERS_VERSION,
    Florence2Backend,
    GroundingDinoBackend,
    Owlv2Backend,
    ProposalBackend,
    Sam3Backend,
    build_backend,
)

__all__ = [
    "BACKENDS",
    "TRANSFORMERS_VERSION",
    "Florence2Backend",
    "GroundingDinoBackend",
    "Owlv2Backend",
    "ProposalBackend",
    "Sam3Backend",
    "build_backend",
]
