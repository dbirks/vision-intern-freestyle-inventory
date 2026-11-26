"""Promptable zero-shot and exemplar proposal backends for Freestyle cartridges.

Verified against ``transformers==5.24.1`` (2026-11-26) and Hugging Face Hub:
- ``owlv2``: ``google/owlv2-large-patch14-ensemble`` and ``google/owlv2-base-patch16-ensemble``
  (``Owlv2ForObjectDetection``, ``Owlv2Processor``; text and image-guided detection).
- ``grounding_dino``: ``IDEA-Research/grounding-dino-base``
  (``GroundingDinoForObjectDetection``, ``AutoProcessor``).
- ``florence2``: ``florence-community/Florence-2-large`` native transformers checkpoint
  (``Florence2ForConditionalGeneration``, ``AutoProcessor``) or ``microsoft/Florence-2-large``
  with ``trust_remote_code=True`` (note: Microsoft's remote modeling code targets
  ``transformers<5.0`` and breaks on ``5.x``; ``florence-community/Florence-2-large`` is the
  official native port of the MIT weights).
- ``sam3``: ``facebook/sam3`` (gated; ``Sam3Model``, ``Sam3Processor``; requires ``HF_TOKEN``).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from tools.utils import get_or_load

logger = logging.getLogger(__name__)

# Pin checked against PyPI and isolated import verification on 2026-11-26.
TRANSFORMERS_VERSION = "5.24.1"

OWLV2_MODELS: dict[str, str] = {
    "large": "google/owlv2-large-patch14-ensemble",
    "base": "google/owlv2-base-patch16-ensemble",
    "large-raw": "google/owlv2-large-patch14",
    "base-raw": "google/owlv2-base-patch16",
}

GROUNDING_DINO_MODELS: dict[str, str] = {
    "base": "IDEA-Research/grounding-dino-base",
    "tiny": "IDEA-Research/grounding-dino-tiny",
}

FLORENCE2_MODELS: dict[str, str] = {
    "large": "florence-community/Florence-2-large",
    "base": "florence-community/Florence-2-base",
    "ms-large": "microsoft/Florence-2-large",
    "ms-base": "microsoft/Florence-2-base",
}

SAM3_MODELS: dict[str, str] = {
    "default": "facebook/sam3",
}

Detection = tuple[list[float], float, str]


def _resolve_device(device: str | None) -> str:
    import torch

    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _move_inputs(inputs: Any, device: Any, pixel_dtype: Any = None) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, val in dict(inputs).items():
        if hasattr(val, "to"):
            if key == "pixel_values" and pixel_dtype is not None:
                val = val.to(device=device, dtype=pixel_dtype)
            else:
                val = val.to(device=device)
        moved[key] = val
    return moved


def _clip_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    x0, y0, x1, y1 = (float(c) for c in box[:4])
    x0 = max(0.0, min(x0, float(width)))
    y0 = max(0.0, min(y0, float(height)))
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


class ProposalBackend(ABC):
    """Common interface for zero-shot / promptable proposal backends."""

    name: str = "base"
    default_model_id: str = ""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        revision: str | None = None,
        score_thr: float = 0.05,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.revision = revision
        self.score_thr = float(score_thr)
        self.device = device
        self.resolved_revision: str | None = revision
        self.last_processed_dims: tuple[int, int] | None = None
        self.processed_dims_seen: set[tuple[int, int]] = set()
        self._warned_unsupported_exemplars = False

    def _cache_key(self, extra: str = "") -> str:
        rev = self.revision or "default"
        dev = self.device or "auto"
        return f"propose:{self.name}:{self.model_id}:{rev}:{dev}:{extra}"

    def _ensure_loaded(self) -> tuple[Any, Any]:
        model, processor = get_or_load(self._cache_key(), self._load)
        if self.resolved_revision is None:
            config = getattr(model, "config", None)
            self.resolved_revision = getattr(config, "_commit_hash", None)
        return model, processor

    @abstractmethod
    def _load(self) -> tuple[Any, Any]:
        """Load and return ``(model, processor)``."""

    def _record_processed_dims(self, inputs: dict[str, Any]) -> None:
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None and hasattr(pixel_values, "shape") and len(pixel_values.shape) >= 2:
            dims = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
            self.last_processed_dims = dims
            self.processed_dims_seen.add(dims)

    def _warn_exemplars_ignored(self, exemplars: list[Image.Image] | None) -> None:
        if exemplars and not self._warned_unsupported_exemplars:
            logger.warning(
                "Backend %s does not support cross-image visual exemplar crops; ignoring %d exemplars.",
                self.name,
                len(exemplars),
            )
            self._warned_unsupported_exemplars = True

    @abstractmethod
    def detect(
        self,
        pil_tile: Image.Image,
        text_prompts: list[str],
        exemplars: list[Image.Image] | None = None,
    ) -> list[Detection]:
        """Run detection on *pil_tile* and return ``[(xyxy, score, label), ...]``."""


class Owlv2Backend(ProposalBackend):
    """OWLv2 backend supporting text prompts and image-guided exemplar crops."""

    name = "owlv2"
    default_model_id = OWLV2_MODELS["large"]

    def __init__(
        self,
        model_id: str | None = None,
        *,
        variant: str | None = None,
        revision: str | None = None,
        score_thr: float = 0.05,
        exemplar_nms_thr: float = 0.3,
        device: str | None = None,
    ) -> None:
        if model_id is None and variant is not None:
            if variant not in OWLV2_MODELS:
                raise ValueError(f"Unknown OWLv2 variant {variant!r}; expected {sorted(OWLV2_MODELS)}")
            model_id = OWLV2_MODELS[variant]
        elif model_id in OWLV2_MODELS:
            model_id = OWLV2_MODELS[model_id]
        super().__init__(model_id=model_id, revision=revision, score_thr=score_thr, device=device)
        self.exemplar_nms_thr = float(exemplar_nms_thr)

    def _load(self) -> tuple[Any, Any]:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        dev = _resolve_device(self.device)
        kw = {"revision": self.revision} if self.revision else {}
        processor = Owlv2Processor.from_pretrained(self.model_id, **kw)
        model = Owlv2ForObjectDetection.from_pretrained(
            self.model_id, torch_dtype=torch.float32, **kw
        ).to(dev)
        model.eval()
        return model, processor

    def detect(
        self,
        pil_tile: Image.Image,
        text_prompts: list[str],
        exemplars: list[Image.Image] | None = None,
    ) -> list[Detection]:
        import torch

        prompts = [p.strip() for p in text_prompts if p and p.strip()]
        exemplar_imgs = [ex.convert("RGB") for ex in (exemplars or [])]
        if not prompts and not exemplar_imgs:
            raise ValueError("Owlv2Backend requires at least one text prompt or exemplar crop.")

        image = pil_tile.convert("RGB")
        model, processor = self._ensure_loaded()
        target_sizes = [(image.height, image.width)]
        detections: list[Detection] = []

        if prompts:
            raw_inputs = processor(text=[prompts], images=image, return_tensors="pt")
            inputs = _move_inputs(raw_inputs, model.device)
            self._record_processed_dims(inputs)
            with torch.inference_mode():
                outputs = model(**inputs)
            res = processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=self.score_thr,
                target_sizes=target_sizes,
                text_labels=[prompts],
            )[0]
            labels = res.get("text_labels")
            if labels is None:
                labels = [prompts[int(idx)] for idx in res["labels"]]
            for box, score, label in zip(res["boxes"], res["scores"], labels, strict=True):
                clipped = _clip_xyxy(box.tolist(), image.width, image.height)
                if clipped is not None:
                    detections.append((clipped, float(score.item()), str(label)))

        for idx, query_img in enumerate(exemplar_imgs):
            raw_inputs = processor(images=image, query_images=query_img, return_tensors="pt")
            inputs = _move_inputs(raw_inputs, model.device)
            self._record_processed_dims(inputs)
            with torch.inference_mode():
                outputs = model.image_guided_detection(**inputs)
            res = processor.post_process_image_guided_detection(
                outputs=outputs,
                threshold=self.score_thr,
                nms_threshold=self.exemplar_nms_thr,
                target_sizes=target_sizes,
            )[0]
            for box, score in zip(res["boxes"], res["scores"], strict=True):
                clipped = _clip_xyxy(box.tolist(), image.width, image.height)
                if clipped is not None:
                    detections.append((clipped, float(score.item()), f"exemplar:{idx}"))

        return detections


class GroundingDinoBackend(ProposalBackend):
    """Grounding DINO backend via ``transformers``."""

    name = "grounding_dino"
    default_model_id = GROUNDING_DINO_MODELS["base"]

    def __init__(
        self,
        model_id: str | None = None,
        *,
        revision: str | None = None,
        score_thr: float = 0.05,
        text_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        if model_id in GROUNDING_DINO_MODELS:
            model_id = GROUNDING_DINO_MODELS[model_id]
        super().__init__(model_id=model_id, revision=revision, score_thr=score_thr, device=device)
        self.text_threshold = float(text_threshold)

    def _load(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        dev = _resolve_device(self.device)
        kw = {"revision": self.revision} if self.revision else {}
        processor = AutoProcessor.from_pretrained(self.model_id, **kw)
        # float32 matches tools/grounded_detect.py (deformable-attention grid_sample).
        model = GroundingDinoForObjectDetection.from_pretrained(
            self.model_id, torch_dtype=torch.float32, **kw
        ).to(dev)
        model.eval()
        return model, processor

    @staticmethod
    def _format_caption(text_prompts: list[str]) -> str:
        cleaned = [p.strip().rstrip(".").lower() for p in text_prompts if p and p.strip()]
        if not cleaned:
            raise ValueError("GroundingDinoBackend requires at least one non-empty text prompt.")
        return " . ".join(cleaned) + "."

    def detect(
        self,
        pil_tile: Image.Image,
        text_prompts: list[str],
        exemplars: list[Image.Image] | None = None,
    ) -> list[Detection]:
        import torch

        self._warn_exemplars_ignored(exemplars)
        caption = self._format_caption(text_prompts)
        image = pil_tile.convert("RGB")
        model, processor = self._ensure_loaded()

        raw_inputs = processor(images=image, text=caption, return_tensors="pt")
        inputs = _move_inputs(raw_inputs, model.device)
        self._record_processed_dims(inputs)

        with torch.inference_mode():
            outputs = model(**inputs)

        res = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=self.score_thr,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = res.get("text_labels") if "text_labels" in res else res["labels"]
        detections: list[Detection] = []
        for box, score, label in zip(res["boxes"], res["scores"], labels, strict=True):
            clipped = _clip_xyxy(box.tolist(), image.width, image.height)
            if clipped is not None:
                detections.append((clipped, float(score.item()), str(label)))
        return detections


class Florence2Backend(ProposalBackend):
    """Florence-2 backend supporting ``<OD>`` and ``<CAPTION_TO_PHRASE_GROUNDING>``."""

    name = "florence2"
    default_model_id = FLORENCE2_MODELS["large"]

    def __init__(
        self,
        model_id: str | None = None,
        *,
        revision: str | None = None,
        score_thr: float = 0.05,
        task: str = "auto",
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        device: str | None = None,
    ) -> None:
        if model_id in FLORENCE2_MODELS:
            model_id = FLORENCE2_MODELS[model_id]
        valid_tasks = ("auto", "<OD>", "<CAPTION_TO_PHRASE_GROUNDING>", "both")
        if task not in valid_tasks:
            raise ValueError(f"Unknown Florence-2 task {task!r}; expected {valid_tasks}")
        super().__init__(model_id=model_id, revision=revision, score_thr=score_thr, device=device)
        self.task = task
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)

    def _load(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, Florence2ForConditionalGeneration

        dev = _resolve_device(self.device)
        dtype = (
            torch.bfloat16
            if dev.startswith("cuda") and torch.cuda.is_bf16_supported()
            else (torch.float16 if dev.startswith("cuda") else torch.float32)
        )
        kw: dict[str, Any] = {"revision": self.revision} if self.revision else {}
        if self.model_id.startswith("microsoft/"):
            logger.warning(
                "Loading %s with trust_remote_code=True; Microsoft's remote code targets "
                "transformers<5.0. Prefer florence-community/Florence-2-large on transformers>=5.x.",
                self.model_id,
            )
            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, **kw)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype, trust_remote_code=True, **kw
            ).to(dev)
        else:
            processor = AutoProcessor.from_pretrained(self.model_id, **kw)
            model = Florence2ForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype, **kw
            ).to(dev)
        model.eval()
        return model, processor

    def _tasks_to_run(self, text_prompts: list[str]) -> list[tuple[str, str]]:
        prompts = [p.strip() for p in text_prompts if p and p.strip()]
        caption = ", ".join(prompts)
        if self.task == "<OD>":
            return [("<OD>", "<OD>")]
        if self.task == "<CAPTION_TO_PHRASE_GROUNDING>":
            if not caption:
                raise ValueError("<CAPTION_TO_PHRASE_GROUNDING> requires non-empty text_prompts.")
            return [("<CAPTION_TO_PHRASE_GROUNDING>", f"<CAPTION_TO_PHRASE_GROUNDING>{caption}")]
        if self.task == "both":
            tasks = [("<OD>", "<OD>")]
            if caption:
                tasks.append(
                    ("<CAPTION_TO_PHRASE_GROUNDING>", f"<CAPTION_TO_PHRASE_GROUNDING>{caption}")
                )
            return tasks
        # "auto"
        if caption:
            return [("<CAPTION_TO_PHRASE_GROUNDING>", f"<CAPTION_TO_PHRASE_GROUNDING>{caption}")]
        return [("<OD>", "<OD>")]

    def detect(
        self,
        pil_tile: Image.Image,
        text_prompts: list[str],
        exemplars: list[Image.Image] | None = None,
    ) -> list[Detection]:
        import torch

        self._warn_exemplars_ignored(exemplars)
        image = pil_tile.convert("RGB")
        model, processor = self._ensure_loaded()
        detections: list[Detection] = []

        for task_token, prompt_text in self._tasks_to_run(text_prompts):
            raw_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = _move_inputs(raw_inputs, model.device, pixel_dtype=model.dtype)
            self._record_processed_dims(inputs)
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=False,
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                generated_text, task=task_token, image_size=(image.width, image.height)
            ).get(task_token, {})
            bboxes = parsed.get("bboxes", [])
            labels = parsed.get("labels", [""] * len(bboxes))
            scores = parsed.get("scores")
            if scores is None:
                scores = [1.0] * len(bboxes)
            for box, score, label in zip(bboxes, scores, labels, strict=True):
                score_val = float(score)
                if score_val < self.score_thr:
                    continue
                clipped = _clip_xyxy(list(box), image.width, image.height)
                if clipped is not None:
                    detections.append((clipped, score_val, str(label) or task_token))

        return detections


class Sam3Backend(ProposalBackend):
    """SAM 3 backend (``facebook/sam3``, gated) for text and box-exemplar concept segmentation."""

    name = "sam3"
    default_model_id = SAM3_MODELS["default"]

    def __init__(
        self,
        model_id: str | None = None,
        *,
        revision: str | None = None,
        score_thr: float = 0.05,
        mask_threshold: float = 0.5,
        hf_token: str | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id, revision=revision, score_thr=score_thr, device=device)
        self.mask_threshold = float(mask_threshold)
        self.hf_token = hf_token

    def _resolve_hf_token(self) -> str:
        token = self.hf_token if self.hf_token is not None else os.environ.get("HF_TOKEN", "")
        token = token.strip()
        if not token:
            raise RuntimeError(
                "SAM 3 (facebook/sam3) is a gated Hugging Face model. "
                "Request access at https://huggingface.co/facebook/sam3 and set the "
                "HF_TOKEN environment variable before running."
            )
        return token

    def _load(self) -> tuple[Any, Any]:
        import torch
        from transformers import Sam3Model, Sam3Processor

        token = self._resolve_hf_token()
        dev = _resolve_device(self.device)
        dtype = (
            torch.bfloat16
            if dev.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        kw: dict[str, Any] = {"token": token}
        if self.revision:
            kw["revision"] = self.revision
        processor = Sam3Processor.from_pretrained(self.model_id, **kw)
        model = Sam3Model.from_pretrained(self.model_id, torch_dtype=dtype, **kw).to(dev)
        model.eval()
        return model, processor

    def detect(
        self,
        pil_tile: Image.Image,
        text_prompts: list[str],
        exemplars: list[Image.Image] | None = None,
        *,
        box_exemplars: list[list[float]] | None = None,
    ) -> list[Detection]:
        import torch

        prompts = [p.strip() for p in text_prompts if p and p.strip()]
        boxes = [list(map(float, b)) for b in (box_exemplars or [])]
        if exemplars and not prompts and not boxes:
            raise ValueError(
                "Sam3Backend supports text concept prompts and in-image box_exemplars, "
                "not cross-image exemplar crops (use backend='owlv2' for crop exemplars)."
            )
        self._warn_exemplars_ignored(exemplars)
        if not prompts and not boxes:
            raise ValueError("Sam3Backend requires text_prompts and/or box_exemplars.")

        image = pil_tile.convert("RGB")
        model, processor = self._ensure_loaded()
        target_sizes = [(image.height, image.width)]
        detections: list[Detection] = []

        for prompt in prompts:
            raw_inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = _move_inputs(raw_inputs, model.device, pixel_dtype=model.dtype)
            self._record_processed_dims(inputs)
            with torch.inference_mode():
                outputs = model(**inputs)
            res = processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_thr,
                mask_threshold=self.mask_threshold,
                target_sizes=target_sizes,
            )[0]
            for box, score in zip(res["boxes"], res["scores"], strict=True):
                clipped = _clip_xyxy(box.tolist(), image.width, image.height)
                if clipped is not None:
                    detections.append((clipped, float(score.item()), prompt))

        if boxes:
            raw_inputs = processor(
                images=image,
                text=None,
                input_boxes=[boxes],
                input_boxes_labels=[[1] * len(boxes)],
                return_tensors="pt",
            )
            inputs = _move_inputs(raw_inputs, model.device, pixel_dtype=model.dtype)
            self._record_processed_dims(inputs)
            with torch.inference_mode():
                outputs = model(**inputs)
            res = processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_thr,
                mask_threshold=self.mask_threshold,
                target_sizes=target_sizes,
            )[0]
            for box, score in zip(res["boxes"], res["scores"], strict=True):
                clipped = _clip_xyxy(box.tolist(), image.width, image.height)
                if clipped is not None:
                    detections.append((clipped, float(score.item()), "box_exemplar"))

        return detections


BACKENDS: dict[str, type[ProposalBackend]] = {
    "owlv2": Owlv2Backend,
    "grounding_dino": GroundingDinoBackend,
    "florence2": Florence2Backend,
    "sam3": Sam3Backend,
}


def build_backend(name: str, **kwargs: Any) -> ProposalBackend:
    """Instantiate a registered proposal backend by name."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown proposal backend {name!r}; expected {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
