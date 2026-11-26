"""User-facing CLI for counting cartridges from frozen Freestyle pipeline bundles.

Examples:
    python -m freestyle.cli bundles
    python -m freestyle.cli count frame.jpg --bundle prod-v1 --out out/ --json
    python -m freestyle.cli fuse frame1.jpg frame2.jpg --bundle prod-v1 --json

Remote inference note:
    A bundle's localizer (such as a tiled RF-DETR detector) may need a GPU for
    fast inference. Pass ``--device cuda`` when a CUDA GPU is available, or
    ``--device cpu`` on CPU-only machines (CPU works, but is slow).
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from freestyle import artifacts, catalog
from freestyle.count.aggregate import fuse_counts

load_bundle = artifacts.load_bundle
list_bundles = artifacts.list_bundles
_ORIG_LOAD_BUNDLE = artifacts.load_bundle
_ORIG_LIST_BUNDLES = artifacts.list_bundles


def _call_load_bundle(name: str, device: str, artifacts_dir: str | None) -> Any:
    fn = artifacts.load_bundle if load_bundle is _ORIG_LOAD_BUNDLE else load_bundle
    params = inspect.signature(fn).parameters
    accepts_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {}
    if accepts_kw or "device" in params:
        kwargs["device"] = device
    if artifacts_dir is not None and (accepts_kw or "artifacts_dir" in params):
        kwargs["artifacts_dir"] = artifacts_dir
    return fn(name, **kwargs)


def _call_list_bundles(artifacts_dir: str | None) -> list[dict[str, Any]]:
    fn = artifacts.list_bundles if list_bundles is _ORIG_LIST_BUNDLES else list_bundles
    params = inspect.signature(fn).parameters
    accepts_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if artifacts_dir is not None and (accepts_kw or "artifacts_dir" in params):
        return fn(artifacts_dir=artifacts_dir)
    return fn()


def _display_name_map(pipeline: Any, catalog_path: str | None = None) -> dict[str, str]:
    names: dict[str, str] = {}
    bundle_classes = getattr(pipeline, "bundle_classes", None)
    if isinstance(bundle_classes, dict) and isinstance(bundle_classes.get("display_names"), dict):
        names.update({str(k): str(v) for k, v in bundle_classes["display_names"].items()})
    try:
        target = Path(catalog_path) if catalog_path else catalog.CATALOG_PATH
        if target.is_file():
            for item in catalog.load_catalog(active_only=False, path=catalog_path):
                slug = str(item.get("id") or "")
                if slug:
                    names[slug] = str(item.get("name") or slug)
    except Exception:
        pass
    return names


def _map_counts_to_display(counts: dict[str, int], display_map: dict[str, str]) -> dict[str, int]:
    mapped: dict[str, int] = {}
    for slug, count in counts.items():
        display = display_map.get(slug, slug)
        mapped[display] = mapped.get(display, 0) + int(count)
    return mapped


def _run_one_image(
    pipeline: Any,
    image_path: Path,
    image_id: str,
    group: str | None,
) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    has_gallery = getattr(getattr(pipeline, "identifier", None), "gallery_groups", None) is not None
    effective_group = group if group is not None else (f"cli:{image_path}" if has_gallery else None)
    if effective_group is not None:
        result = pipeline.run(image, image_id, group=effective_group)
    else:
        result = pipeline.run(image, image_id)
    return image, result


def _write_overlay(
    image: Image.Image,
    detections: Sequence[dict[str, Any]],
    display_map: dict[str, str],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for det in detections:
        box = det.get("box")
        if not box or len(box) != 4:
            continue
        cls_slug = str(det.get("class", "unknown"))
        label = display_map.get(cls_slug, cls_slug)
        shelf = det.get("shelf")
        text = f"{label[:22]} S{shelf}" if shelf is not None else label[:22]
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], box[1]), text, fill="yellow")
    canvas.save(out_path)
    return out_path


def _overlay_filename(image_path: Path, index: int, total: int, used: set[str]) -> str:
    candidate = f"{image_path.stem}.png"
    if candidate not in used and total == 1:
        used.add(candidate)
        return candidate
    if candidate not in used:
        used.add(candidate)
        return candidate
    numbered = f"{index:03d}_{image_path.stem}.png"
    used.add(numbered)
    return numbered


def _run_images_for_cli(
    pipeline: Any,
    image_paths: Sequence[str],
    display_map: dict[str, str],
    *,
    out_dir: Path | None = None,
    group: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for idx, raw_path in enumerate(image_paths):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        image, result = _run_one_image(pipeline, path, raw_path, group)
        raw_counts = {str(k): int(v) for k, v in (result.get("counts") or {}).items()}
        display_counts = _map_counts_to_display(raw_counts, display_map)
        timings = dict(result.get("timings_ms") or result.get("timings") or {})
        entry: dict[str, Any] = {
            "image": str(path),
            "counts": display_counts,
            "unknown": int(result.get("unknown", 0)),
            "flags": list(result.get("flags") or []),
            "timings_ms": timings,
        }
        if out_dir is not None:
            overlay_name = _overlay_filename(path, idx, len(image_paths), used_names)
            overlay_path = _write_overlay(
                image, result.get("detections") or [], display_map, out_dir / overlay_name
            )
            entry["overlay"] = str(overlay_path)
        records.append(entry)
    return records


def _format_count_text(bundle: str, device: str, records: Sequence[dict[str, Any]]) -> str:
    lines = [f"Bundle: {bundle} (device={device})"]
    for rec in records:
        lines.append(f"\nImage: {rec['image']}")
        if rec["counts"]:
            for name, count in sorted(rec["counts"].items()):
                lines.append(f"  {name}: {count}")
        else:
            lines.append("  (no identified cartridges)")
        lines.append(f"  unknown: {rec['unknown']}")
        lines.append(f"  flags: {len(rec['flags'])}")
        if rec["timings_ms"]:
            t_str = ", ".join(f"{k}={float(v):.1f}ms" for k, v in rec["timings_ms"].items())
            lines.append(f"  timings: {t_str}")
        if "overlay" in rec:
            lines.append(f"  overlay: {rec['overlay']}")
    return "\n".join(lines)


def _format_fuse_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Bundle: {payload['bundle']} (device={payload['device']}, method={payload['method']}, "
        f"frames={len(payload['frames'])})"
    ]
    if payload["fused_counts"]:
        for name, count in sorted(payload["fused_counts"].items()):
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  (no identified cartridges)")
    lines.append(f"  unknown: {payload['fused_unknown']}")
    return "\n".join(lines)


def _format_bundles_text(bundles: Sequence[dict[str, Any]]) -> str:
    if not bundles:
        return "No bundles found."
    lines = [f"{'Bundle':<24} {'Experiment':<30} {'Embedder':<26} {'Thresholds'}"]
    lines.append(f"{'-' * 24} {'-' * 30} {'-' * 26} {'-' * 20}")
    for b in bundles:
        if not b.get("valid", True):
            lines.append(f"{b['name']:<24} INVALID ({b.get('error', 'error')})")
            continue
        thr = b.get("thresholds") or {}
        thr_str = f"score={thr.get('score_thr', '-')}, id={thr.get('identity_thr', '-')}"
        lines.append(
            f"{str(b['name']):<24} {str(b.get('experiment') or '-'):<30} "
            f"{str(b.get('embedder_id') or '-'):<26} {thr_str}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    device_help = (
        "Inference device (auto, cpu, or cuda). "
        "Note: the bundle's localizer may need a GPU; CPU works but is slow."
    )
    parser = argparse.ArgumentParser(
        prog="freestyle.cli",
        description="Run frozen Freestyle cartridge counting bundles.",
        epilog=(
            "Remote inference note: the bundle's localizer may need a GPU for fast inference; "
            "--device cpu works on CPU-only machines but is slow."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_count = sub.add_parser("count", help="Count cartridges in one or more images.")
    p_count.add_argument("images", nargs="+", help="Image paths to inspect.")
    p_count.add_argument("--bundle", required=True, help="Bundle name under artifacts/<name>/.")
    p_count.add_argument("--out", default=None, help="Directory to write overlay PNGs and JSON.")
    p_count.add_argument("--json", action="store_true", help="Emit JSON output to stdout.")
    p_count.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help=device_help)
    p_count.add_argument("--catalog", default=None, help="Optional catalog_items.json path.")
    p_count.add_argument("--group", default=None, help="Optional query group id for gallery check.")
    p_count.add_argument("--artifacts-dir", default=None, help="Override artifacts directory.")

    p_fuse = sub.add_parser("fuse", help="Fuse per-frame counts from one door opening.")
    p_fuse.add_argument("images", nargs="+", help="Frame paths from one door opening.")
    p_fuse.add_argument("--bundle", required=True, help="Bundle name under artifacts/<name>/.")
    p_fuse.add_argument(
        "--method",
        choices=("max", "median", "weighted_mode"),
        default="max",
        help="Cross-frame fusion rule (default: max).",
    )
    p_fuse.add_argument("--weights", nargs="+", type=float, default=None, help="Optional frame weights.")
    p_fuse.add_argument("--out", default=None, help="Directory to write overlay PNGs and JSON.")
    p_fuse.add_argument("--json", action="store_true", help="Emit JSON output to stdout.")
    p_fuse.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help=device_help)
    p_fuse.add_argument("--catalog", default=None, help="Optional catalog_items.json path.")
    p_fuse.add_argument("--group", default=None, help="Optional query group id for gallery check.")
    p_fuse.add_argument("--artifacts-dir", default=None, help="Override artifacts directory.")

    p_bundles = sub.add_parser("bundles", help="List available frozen bundles.")
    p_bundles.add_argument("--json", action="store_true", help="Emit JSON output to stdout.")
    p_bundles.add_argument("--artifacts-dir", default=None, help="Override artifacts directory.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "bundles":
        bundles = _call_list_bundles(args.artifacts_dir)
        if args.json:
            print(json.dumps({"bundles": bundles}, indent=2))
        else:
            print(_format_bundles_text(bundles))
        return 0

    if args.command == "fuse" and args.weights is not None and len(args.weights) != len(args.images):
        parser.error("--weights length must match the number of input images")

    pipeline = _call_load_bundle(args.bundle, args.device, args.artifacts_dir)
    display_map = _display_name_map(pipeline, args.catalog)
    out_dir = Path(args.out) if args.out else None

    records = _run_images_for_cli(
        pipeline,
        args.images,
        display_map,
        out_dir=out_dir,
        group=args.group,
    )

    if args.command == "count":
        payload = {"bundle": args.bundle, "device": args.device, "images": records}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "counts.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(_format_count_text(args.bundle, args.device, records))
        return 0

    if args.command == "fuse":
        fused_counts = fuse_counts(
            [r["counts"] for r in records], weights=args.weights, method=args.method
        )
        fused_unknown = fuse_counts(
            [{"unknown": r["unknown"]} for r in records], weights=args.weights, method=args.method
        ).get("unknown", 0)
        payload = {
            "bundle": args.bundle,
            "device": args.device,
            "method": args.method,
            "fused_counts": fused_counts,
            "fused_unknown": fused_unknown,
            "frames": records,
        }
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "fused.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(_format_fuse_text(payload))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
