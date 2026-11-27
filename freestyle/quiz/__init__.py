"""Phone-first ground-truth quiz: one zoomed crop at a time.

Modules
-------
``build_queue``  turn model proposals + splits into an ordered question queue
``server``       stdlib HTTP app that renders crops and appends answers
``to_coco``      queue + answers -> COCO ground truth + per-image count files

Nothing here is ground truth on its own: only answers a human typed into the
quiz, and the COCO files ``to_coco`` writes under ``data/gt/``, are human GT.
"""

from __future__ import annotations

# Submodules are imported explicitly (``from freestyle.quiz import build_queue``)
# so that the HTTP server and PIL never get pulled in by ``import freestyle.quiz``.
