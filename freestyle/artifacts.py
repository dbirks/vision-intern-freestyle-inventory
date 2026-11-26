"""Tiny local model registry for frozen Freestyle pipeline bundles.

Each bundle lives under ``artifacts/<name>/`` (gitignored) and contains:

- ``pipeline.yaml`` — frozen ``Pipeline`` configuration with bundle-relative paths
- ``classes.json`` — ordered detector class list and category/display metadata
- detector checkpoint (when the localizer uses a trained checkpoint)
- ``gallery.npz`` — train-only embedding gallery (when using ``embedding_knn``)
- ``manifest.json`` — provenance (run id, git sha, metrics summary), embedder id,
  thresholds, and SHA-256 checksums for every bundled file (verified on load)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from freestyle.pipeline import EmbeddingIdentifier, Pipeline, ProposalLocalizer

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MANIFEST_VERSION = 1


class BundleIntegrityError(ValueError):
    """Raised when a bundle fails manifest or SHA-256 verification."""


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _artifacts_root(artifacts_dir: str | Path | None = None) -> Path:
    return Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR


def _safe_bundle_file(bundle_dir: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    if not rel_path or rel.is_absolute() or ".." in rel.parts:
        raise BundleIntegrityError(f"Unsafe manifest path: {rel_path!r}")
    root = bundle_dir.resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise BundleIntegrityError(f"Manifest path escapes bundle: {rel_path!r}") from exc
    return target


def _resolve_asset(raw_path: str | Path | None, run_dir: Path, fallbacks: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    if raw_path:
        raw_str = str(raw_path)
        p = Path(raw_str)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(run_dir / p)
        if raw_str.startswith("/out/"):
            candidates.append(run_dir / raw_str[len("/out/"):])
        candidates.append(run_dir / p.name)
    for rel in fallbacks:
        candidates.append(run_dir / rel)
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_optional_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_base_configs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved: dict[str, Any] = {}
    if (run_dir / "resolved.yaml").is_file():
        resolved = yaml.safe_load((run_dir / "resolved.yaml").read_text(encoding="utf-8")) or {}
    meta = _load_optional_json(run_dir / "meta.json") or {}
    if not resolved and isinstance(meta.get("config"), dict):
        resolved = meta["config"]

    pipeline_cfg: dict[str, Any] = {}
    for name in ("pipeline_test.yaml", "pipeline_val.yaml", "pipeline.yaml"):
        candidate = run_dir / name
        if candidate.is_file():
            pipeline_cfg = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            break

    if not pipeline_cfg and not resolved:
        raise FileNotFoundError(
            f"Run directory {run_dir} has neither pipeline_*.yaml nor resolved.yaml"
        )
    if not pipeline_cfg:
        pipeline_cfg = {
            "localizer": dict(resolved.get("localizer", {})),
            "identifier": dict(resolved.get("identifier", {"type": "none"})),
            "counting": dict(resolved.get("count", {})),
        }
    return pipeline_cfg, resolved


def _extract_detector_classes(
    run_dir: Path,
    pipeline_cfg: dict[str, Any],
    resolved: dict[str, Any],
) -> tuple[list[str], dict[str, int]]:
    categories = dict(pipeline_cfg.get("categories") or {})
    if not categories:
        for coco_rel in ("training_input.json", "dataset/train/_annotations.coco.json"):
            coco = _load_optional_json(run_dir / coco_rel)
            if isinstance(coco, dict) and coco.get("categories"):
                categories = {str(c["name"]): int(c["id"]) for c in coco["categories"]}
                break

    classes_file = _load_optional_json(run_dir / "classes.json")
    if isinstance(classes_file, list):
        return [str(x) for x in classes_file], categories
    if isinstance(classes_file, dict) and isinstance(classes_file.get("detector_classes"), list):
        cats = dict(classes_file.get("categories") or categories)
        return [str(x) for x in classes_file["detector_classes"]], cats

    loc_type = (pipeline_cfg.get("localizer") or {}).get("type")
    family = resolved.get("family", "")
    if loc_type == "tiled_detector" and family != "D_det_multiclass":
        return ["cartridge"], categories
    if categories:
        ordered = [name for name, _ in sorted(categories.items(), key=lambda kv: kv[1])]
        if loc_type == "tiled_detector" and family == "D_det_multiclass":
            ordered = [name for name in ordered if name != "cartridge"]
        return ordered, categories
    return [], categories


def _catalog_display_names() -> dict[str, str]:
    try:
        from freestyle import catalog

        if not catalog.CATALOG_PATH.is_file():
            return {}
        items = catalog.load_catalog(active_only=False)
        return {str(it["id"]): str(it.get("name") or it["id"]) for it in items if it.get("id")}
    except Exception:
        return {}


def _metrics_summary(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    loc = metrics.get("localization") or {}
    ident = metrics.get("identification") or {}
    cnt = metrics.get("counting") or {}
    return {
        "n_images": metrics.get("n_images"),
        "n_groups": metrics.get("n_groups"),
        "score_thr": metrics.get("score_thr"),
        "score_thr_source": metrics.get("score_thr_source"),
        "localization": {
            k: loc.get(k) for k in ("ap50", "ap50_95", "precision", "recall", "f1") if k in loc
        },
        "identification": {
            k: ident.get(k) for k in ("top1_acc", "macro_f1") if k in ident
        },
        "counting": {
            k: cnt.get(k)
            for k in ("total_count_mae", "per_class_count_mae", "exact_inventory_rate")
            if k in cnt
        },
    }


def export_bundle(
    run_dir: str | Path,
    name: str,
    *,
    artifacts_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Freeze a completed experiment run directory into ``artifacts/<name>/``."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"Invalid bundle name: {name!r}")
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    root = _artifacts_root(artifacts_dir)
    bundle_dir = root / name
    if bundle_dir.exists() and not overwrite:
        raise FileExistsError(f"Bundle already exists: {bundle_dir}")

    pipeline_cfg, resolved = _load_base_configs(run_path)
    meta = _load_optional_json(run_path / "meta.json") or {}
    metrics = _load_optional_json(run_path / "metrics.json")

    loc = dict(pipeline_cfg.get("localizer") or {})
    loc.pop("training", None)
    if not loc.get("type"):
        raise ValueError("localizer.type is required to export a bundle")

    ident = dict(pipeline_cfg.get("identifier") or {"type": "none"})
    counting = dict(pipeline_cfg.get("counting") or resolved.get("count") or {})
    counting.setdefault("identity_thr", 0.5)
    counting.setdefault("neighbour_vote", "off")
    counting.setdefault("max_rows", 4)
    if isinstance(metrics, dict) and metrics.get("score_thr") is not None:
        counting["score_thr"] = float(metrics["score_thr"])
    else:
        counting.setdefault("score_thr", 0.0)

    detector_classes, categories = _extract_detector_classes(run_path, pipeline_cfg, resolved)

    root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=str(root)))
    try:
        # 1. Detector checkpoint or proposal predictions if applicable.
        if loc["type"] == "tiled_detector":
            ckpt = _resolve_asset(
                loc.get("checkpoint"),
                run_path,
                (
                    "training/checkpoint_best_total.pth",
                    "training/checkpoint_best_ema.pth",
                    "training/checkpoint.pth",
                    "training/yolo/weights/best.pt",
                ),
            )
            if ckpt is None:
                raise FileNotFoundError(
                    f"Detector checkpoint not found in {run_path} (configured: {loc.get('checkpoint')!r})"
                )
            det_dir = stage_dir / "detector"
            det_dir.mkdir(parents=True, exist_ok=True)
            dest_ckpt = det_dir / ckpt.name
            shutil.copy2(ckpt, dest_ckpt)
            loc["checkpoint"] = f"detector/{ckpt.name}"
        elif loc["type"] == "proposals":
            prop = _resolve_asset(loc.get("path"), run_path, ("test/predictions.json", "val/predictions.json"))
            if prop is None:
                raise FileNotFoundError(f"Proposals file not found: {loc.get('path')!r}")
            shutil.copy2(prop, stage_dir / "proposals.json")
            loc["path"] = "proposals.json"

        # 2. Identifier gallery npz if embedding_knn.
        embedder_id: str | None = None
        if ident.get("type") == "embedding_knn":
            embedder_id = str(ident.get("model") or "facebook/dinov2-base")
            ident["model"] = embedder_id
            gallery = _resolve_asset(ident.get("gallery"), run_path, ("gallery.npz",))
            if gallery is None:
                raise FileNotFoundError(
                    f"Identifier gallery not found in {run_path} (configured: {ident.get('gallery')!r})"
                )
            shutil.copy2(gallery, stage_dir / "gallery.npz")
            ident["gallery"] = "gallery.npz"
        elif ident.get("type") == "vlm_crop":
            embedder_id = str(ident.get("model") or "") or None

        # 3. Frozen pipeline.yaml (no run-time manifest/split_map paths baked in).
        frozen_pipeline: dict[str, Any] = {
            "localizer": loc,
            "identifier": ident,
            "counting": counting,
        }
        if categories:
            frozen_pipeline["categories"] = categories
        (stage_dir / "pipeline.yaml").write_text(
            yaml.safe_dump(frozen_pipeline, sort_keys=False), encoding="utf-8"
        )

        # 4. Detector class list + catalog display snapshot.
        classes_payload = {
            "detector_classes": detector_classes,
            "categories": categories,
            "display_names": _catalog_display_names(),
        }
        (stage_dir / "classes.json").write_text(
            json.dumps(classes_payload, indent=2) + "\n", encoding="utf-8"
        )

        # 5. Provenance & file checksums -> manifest.json.
        git_sha = None
        if (run_path / "git_sha").is_file():
            git_sha = (run_path / "git_sha").read_text(encoding="utf-8").strip() or None
        if not git_sha:
            git_sha = meta.get("git_sha")

        files_manifest: dict[str, dict[str, Any]] = {}
        for file_path in sorted(p for p in stage_dir.rglob("*") if p.is_file()):
            rel = file_path.relative_to(stage_dir).as_posix()
            digest, size = _sha256_file(file_path)
            files_manifest[rel] = {"sha256": digest, "bytes": size}

        manifest = {
            "schema_version": MANIFEST_VERSION,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedder_id": embedder_id,
            "detector_classes": detector_classes,
            "thresholds": {
                "score_thr": float(counting["score_thr"]),
                "identity_thr": float(counting["identity_thr"]),
            },
            "provenance": {
                "run_id": run_path.name,
                "experiment": meta.get("experiment") or resolved.get("id") or run_path.name,
                "family": resolved.get("family"),
                "git_sha": git_sha,
                "git_dirty": meta.get("git_dirty"),
                "git_diff_hash": meta.get("git_diff_hash"),
                "metrics_summary": _metrics_summary(metrics),
            },
            "files": files_manifest,
        }
        (stage_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        stage_dir.rename(bundle_dir)
        return bundle_dir
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Validate ``manifest.json`` and verify SHA-256 digests of every bundled file."""
    bdir = Path(bundle_dir)
    manifest_path = bdir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleIntegrityError(f"Invalid manifest JSON in {manifest_path}: {exc}") from exc

    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise BundleIntegrityError("Bundle manifest missing 'files' checksums")
    if "pipeline.yaml" not in files:
        raise BundleIntegrityError("Bundle manifest must include 'pipeline.yaml'")

    for rel_path, meta in files.items():
        if not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str):
            raise BundleIntegrityError(f"Malformed checksum entry for {rel_path!r}")
        target = _safe_bundle_file(bdir, rel_path)
        if not target.is_file():
            raise BundleIntegrityError(f"Missing bundled file: {rel_path}")
        actual_hash, actual_bytes = _sha256_file(target)
        if actual_hash != meta["sha256"]:
            raise BundleIntegrityError(
                f"SHA-256 mismatch for {rel_path}: expected {meta['sha256']}, got {actual_hash}"
            )
        if "bytes" in meta and actual_bytes != meta["bytes"]:
            raise BundleIntegrityError(
                f"Size mismatch for {rel_path}: expected {meta['bytes']}, got {actual_bytes}"
            )
    return manifest


def _apply_localizer_device(localizer: Any, device: str) -> None:
    if device == "auto":
        return
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device!r} (expected auto, cpu, or cuda)")
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested (--device cuda) but CUDA is not available")
        except ImportError as exc:
            raise RuntimeError("CUDA device requested but PyTorch is not installed") from exc

    backend = getattr(localizer, "backend", localizer)
    model = getattr(backend, "model", None)
    for target in (model, getattr(model, "model", None), backend):
        if target is not None and callable(getattr(target, "to", None)):
            try:
                target.to(device)
                return
            except Exception:
                continue


def _resolve_bundle_path(bundle_dir: Path, raw: str | Path) -> str:
    p = Path(raw)
    return str(p if p.is_absolute() else (bundle_dir / p))


def load_bundle(
    name: str,
    *,
    device: str = "auto",
    artifacts_dir: str | Path | None = None,
) -> Pipeline:
    """Verify bundle integrity and instantiate a ready-to-run ``Pipeline``."""
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device!r} (expected auto, cpu, or cuda)")
    bundle_dir = _artifacts_root(artifacts_dir) / name
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle not found: {bundle_dir}")

    manifest = verify_bundle(bundle_dir)
    config = yaml.safe_load((bundle_dir / "pipeline.yaml").read_text(encoding="utf-8"))

    loc = dict(config["localizer"])
    loc_type = loc.pop("type")
    if loc_type == "tiled_detector":
        from freestyle.infer.detector import TiledDetector

        loc["checkpoint"] = _resolve_bundle_path(bundle_dir, loc["checkpoint"])
        localizer = TiledDetector(**loc)
        _apply_localizer_device(localizer, device)
    elif loc_type == "proposals":
        loc["path"] = _resolve_bundle_path(bundle_dir, loc["path"])
        localizer = ProposalLocalizer(categories=config.get("categories"), **loc)
    elif loc_type == "vlm":
        from freestyle.vlm_ground import Localizer

        localizer = Localizer(**loc)
    else:
        raise ValueError(f"unknown localizer: {loc_type}")

    ident = dict(config.get("identifier") or {"type": "none"})
    ident_type = ident.pop("type")
    if ident_type == "embedding_knn":
        ident["gallery"] = _resolve_bundle_path(bundle_dir, ident["gallery"])
        if device != "auto":
            ident.setdefault("device", device)
        identifier = EmbeddingIdentifier(**ident)
    elif ident_type == "none":
        identifier = None
    elif ident_type == "vlm_crop":
        from freestyle.vlm_ground import CropIdentifier

        identifier = CropIdentifier(**ident)
    else:
        raise ValueError(f"unknown identifier: {ident_type}")

    pipeline = Pipeline(localizer, identifier, **config.get("counting", {}))
    pipeline.bundle_manifest = manifest
    if (bundle_dir / "classes.json").is_file():
        pipeline.bundle_classes = json.loads((bundle_dir / "classes.json").read_text(encoding="utf-8"))
    return pipeline


def list_bundles(artifacts_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List frozen bundles under ``artifacts/`` with provenance and integrity status."""
    root = _artifacts_root(artifacts_dir)
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            entries.append({"name": path.name, "path": str(path), "valid": False, "error": str(exc)})
            continue
        prov = manifest.get("provenance") or {}
        entries.append(
            {
                "name": manifest.get("name", path.name),
                "path": str(path),
                "created_at": manifest.get("created_at"),
                "experiment": prov.get("experiment"),
                "run_id": prov.get("run_id"),
                "git_sha": prov.get("git_sha"),
                "embedder_id": manifest.get("embedder_id"),
                "detector_classes": manifest.get("detector_classes", []),
                "thresholds": manifest.get("thresholds", {}),
                "metrics_summary": prov.get("metrics_summary"),
                "valid": True,
            }
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage frozen Freestyle pipeline bundles.")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export an experiment run directory as a frozen bundle.")
    exp.add_argument("--run", required=True, help="Experiment run directory under runs/.")
    exp.add_argument("--name", required=True, help="Bundle name under artifacts/<name>/.")
    exp.add_argument("--artifacts-dir", default=None, help="Override artifacts directory.")
    exp.add_argument("--overwrite", action="store_true", help="Replace an existing bundle.")

    ver = sub.add_parser("verify", help="Verify a bundle's manifest and SHA-256 checksums.")
    ver.add_argument("name", help="Bundle name under artifacts/.")
    ver.add_argument("--artifacts-dir", default=None, help="Override artifacts directory.")

    args = parser.parse_args(argv)
    if args.command == "export":
        out = export_bundle(
            args.run, args.name, artifacts_dir=args.artifacts_dir, overwrite=args.overwrite
        )
        print(out)
        return 0
    if args.command == "verify":
        manifest = verify_bundle(_artifacts_root(args.artifacts_dir) / args.name)
        print(f"OK: {manifest['name']} ({len(manifest['files'])} files verified)")
        return 0
    return 1


__all__ = [
    "ARTIFACTS_DIR",
    "BundleIntegrityError",
    "export_bundle",
    "list_bundles",
    "load_bundle",
    "verify_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
