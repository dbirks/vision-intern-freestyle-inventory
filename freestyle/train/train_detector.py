"""Train a detector from a YAML experiment configuration."""

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml


def metadata():
    try:
        import torch
    except ImportError:
        torch = None

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                      text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    cuda = torch is not None and torch.cuda.is_available()
    return {"git_sha": sha, "hardware": {"torch": torch.__version__ if torch else None,
            "cuda": torch.version.cuda if torch else None, "gpu": torch.cuda.get_device_name() if cuda else None},
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if cuda else 0.0}


class RFDetrBackend:
    def train(self, config, out):
        import rfdetr

        name = config["model"]
        if name not in {"RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge", "RFDETRBase"}:
            raise ValueError(f"Unsupported RF-DETR model: {name}")
        model = getattr(rfdetr, name)(resolution=config["resolution"])
        keys = ("epochs", "batch_size", "grad_accum_steps", "lr", "multi_scale",
                "aug_config", "early_stopping", "early_stopping_patience", "seed")
        from rfdetr.datasets.coco import annotated_category_ids, filter_parent_categories

        annotations = json.loads((Path(config["dataset_dir"]) / "train" /
                                  "_annotations.coco.json").read_text())
        categories = filter_parent_categories(annotations["categories"],
                                              annotated_category_ids(annotations))
        category_names = {int(c["id"]): c["name"] for c in categories}
        if len(category_names) != len(categories) or not category_names:
            raise ValueError("Training categories must have unique IDs and cannot be empty")
        label_to_category = {i: int(c["id"]) for i, c in enumerate(categories)}
        model.train(dataset_dir=config["dataset_dir"], output_dir=str(out),
                    **{k: config[k] for k in keys if k in config})

        # Keep the training label space with the weights, including when a bundle copies them.
        import torch

        for checkpoint in Path(out).rglob("*.pth"):
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state["freestyle_category_names"] = category_names
            state["freestyle_label_to_category"] = label_to_category
            torch.save(state, checkpoint)


class YoloBackend:
    def train(self, config, out):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise NotImplementedError("Install ultralytics to use the YOLO backend") from exc
        if config.get("grad_accum_steps", 1) != 1:
            raise ValueError("YOLO backend does not support explicit grad_accum_steps")
        YOLO(config["model"]).train(data=config["dataset_dir"], imgsz=config["resolution"],
            epochs=config.get("epochs", 100), batch=config.get("batch_size", 4),
            lr0=config.get("lr", 0.01), seed=config.get("seed", 0),
            patience=config.get("early_stopping_patience", 10) if config.get("early_stopping") else 0,
            project=str(out), name="yolo", exist_ok=True, **config.get("yolo_augment", {}))


def train(config, out):
    import torch

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    seed = config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    backends = {"rfdetr": RFDetrBackend, "yolo": YoloBackend}
    backend = config.get("backend", "rfdetr")
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}")
    start = time.perf_counter()
    try:
        backends[backend]().train(config, out)
    finally:
        meta = metadata()
        meta.update(experiment=config.get("experiment", out.name), config=config,
                    wall_time_seconds=time.perf_counter() - start)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    train(yaml.safe_load(Path(args.config).read_text()), args.out)


if __name__ == "__main__":
    main()
