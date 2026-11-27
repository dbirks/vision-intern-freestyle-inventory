"""Recover RF-DETR label mappings from a run's actual training annotations.

Usage: python -m freestyle.train.recover_mapping RUN [--checkpoint CHECKPOINT]
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from freestyle.train.export_tiles import export_tiles


def category_mapping(annotations):
    """Mirror RF-DETR 1.12.1's detection category ordering and parent filtering."""
    categories = sorted(annotations["categories"], key=lambda c: int(c["id"]))
    parents = {c.get("supercategory") for c in categories
               if c.get("supercategory") not in (None, "", "none", "null")
               and c.get("supercategory") != c["name"]}
    annotated = {int(a["category_id"]) for a in annotations["annotations"]}
    kept = [c for c in categories if not (c["name"] in parents and int(c["id"]) not in annotated)]
    kept = kept or categories
    names = {int(c["id"]): c["name"] for c in kept}
    if not names or len(names) != len(kept):
        raise ValueError("Training categories must have unique IDs and cannot be empty")
    return names, {i: int(c["id"]) for i, c in enumerate(kept)}


def training_annotations(run, data_root):
    run = Path(run)
    annotation = run / "dataset" / "train" / "_annotations.coco.json"
    if annotation.is_file():
        return json.loads(annotation.read_text()), str(annotation)
    config = yaml.safe_load((run / "resolved.yaml").read_text())
    augmented = bool(config["localizer"].get("training", {}).get("augment"))
    suffix = "_augmented" if augmented else ""
    source = run / f"training_input{suffix}.json"
    splits = run / f"training_splits{suffix}.json"
    if not splits.is_file():
        raise FileNotFoundError(f"Missing actual training split manifest: {splits}")
    if augmented and not source.is_file():
        raise FileNotFoundError(f"Missing actual augmented training annotations: {source}")
    with tempfile.TemporaryDirectory(prefix="freestyle-recover-") as temp:
        if not source.is_file():
            from freestyle.experiments.entry import prepare_training

            source = prepare_training(config, Path(data_root), Path(temp))
        loc = config["localizer"]
        datasets = export_tiles(source, None, None, splits, tile=loc.get("tile", 1024),
                                overlap=loc.get("overlap", 0.25),
                                class_agnostic=config["family"] != "D_det_multiclass",
                                annotation_only=True)
        return datasets["train"], f"reconstructed from {source} and {splits}"


def recover(run, checkpoint=None, data_root="data"):
    import torch

    run = Path(run)
    annotations, source = training_annotations(run, data_root)
    names, labels = category_mapping(annotations)
    checkpoints = [Path(checkpoint)] if checkpoint else sorted((run / "training").rglob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {run / 'training'}")
    for path in checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("rfdetr_version") != "1.12.1":
            raise ValueError(f"{path}: unsupported or missing RF-DETR version")
        args = state.get("args", {})
        recorded = args.get("class_names") if isinstance(args, dict) else getattr(args, "class_names", None)
        if recorded is None or list(recorded) != [names[labels[i]] for i in labels]:
            raise ValueError(f"{path}: checkpoint class names disagree with training annotations")
        if ("freestyle_category_names" in state and state["freestyle_category_names"] != names or
                "freestyle_label_to_category" in state and state["freestyle_label_to_category"] != labels):
            raise ValueError(f"{path}: existing mapping conflicts with recovered mapping")
        state["freestyle_category_names"] = names
        state["freestyle_label_to_category"] = labels
        # Replace atomically so interruption cannot leave a partially written checkpoint.
        fd, temporary = tempfile.mkstemp(prefix=".mapping-", suffix=".pth", dir=path.parent)
        os.close(fd)
        try:
            torch.save(state, temporary)
            os.chmod(temporary, path.stat().st_mode & 0o7777)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"source": source, "category_names": names, "label_to_category": labels,
            "checkpoints": [str(p) for p in checkpoints]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(recover(args.run, args.checkpoint, args.data_root), indent=2))


if __name__ == "__main__":
    main()
