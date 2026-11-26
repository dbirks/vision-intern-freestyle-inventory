"""Build a frozen, leakage-safe ``data/splits.json`` for the Freestyle corpus.

    uv run python -m freestyle.make_splits            # writes data/splits.json
    uv run python -m freestyle.make_splits --check     # verify the file on disk
    uv run python -m freestyle.make_splits --test-frac 0.4 --seed 0

**The split is frozen.** ``data/splits.frozen.json`` is the reviewed, official
assignment; when it exists its ``groups`` / ``images`` / ``negatives`` are copied
into the output verbatim and the generator is *not* allowed to change them. The
random draw stays in the code because it is what produced the frozen table, and
the two are compared on every run: any group whose generated split differs from
the frozen one is reported on stdout (drift is normal — the audit labels moved
after the review — but the frozen assignment wins and the drift is visible
instead of silent; ``splits.json`` keeps its schema either way). Regenerating the
split from scratch is a deliberate act: ``--frozen ''`` (or moving the file away)
and a new human review.

Unit of splitting is the **group** (door opening / burst / single local file),
never the frame: frames inside one opening are views of the same scene and must
not straddle train and test.

Rules implemented (see docs/freestyle/DATA_AUDIT.md for the rationale):

* candidate image  = scene label ``cabinet_full`` or ``cabinet_partial``;
* ``no_cabinet`` / ``occluded`` images are ``exclude``d and additionally listed
  under ``negatives`` (rejection-test material, never trained on);
* ``local/downloads/frame.jpg`` (the one clean full view) is forced to ``val``;
* the ``local/blur`` burst is forced to ``train`` (its 12 frames are one group);
* ``test`` takes ~``--test-frac`` of the candidate *groups* and is stratified so
  it contains at least one sharp full view, one partial view, one blurry view,
  one upside-down view (if any exist) and images of both native resolutions;
* everything else is drawn with ``random.Random(seed)`` — deterministic.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freestyle.data import REPO_ROOT, SPLITS_PATH, groups, list_images  # noqa: E402

AUDIT_JSON = REPO_ROOT / "data/derived/audit/audit.json"
#: The reviewed assignment; wins over whatever the generator would draw today.
FROZEN_SPLITS_PATH = REPO_ROOT / "data/splits.frozen.json"
ASSIGNMENT_KEYS = ("groups", "images", "negatives")
CANDIDATE_LABELS = ("cabinet_full", "cabinet_partial")
NEGATIVE_LABELS = ("no_cabinet", "occluded")
BEST_VIEW = "local/downloads/frame.jpg"
BLUR_GROUP = "local/blur"
NEAR_DUP_MAX_DISTANCE = 8


def _hamming(h1: str, h2: str) -> int:
    if not h1 or not h2:
        return 64
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def _load_frozen(path: str | Path | None) -> dict[str, Any] | None:
    """The frozen assignment table, or ``None`` when there is none."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if "groups" not in data or "images" not in data:
        raise SystemExit(f"{p} is not a splits table (needs 'groups' and 'images').")
    return data


def _cross_split_pairs(audit: dict[str, dict[str, Any]],
                       images_map: dict[str, str]) -> list[dict[str, Any]]:
    """Cross-split near-duplicate report (pHash): test must not be reachable from
    train/val by a copy of the same view. Reported, not auto-fixed."""
    rows = sorted(audit.values(), key=lambda r: r["image_id"])
    cross = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            sa, sb = images_map.get(a["image_id"]), images_map.get(b["image_id"])
            if sa is None or sb is None or "exclude" in (sa, sb) or sa == sb:
                continue
            d = _hamming(a.get("phash") or "", b.get("phash") or "")
            if d <= NEAR_DUP_MAX_DISTANCE:
                cross.append({"distance": d, "a": a["image_id"], "split_a": sa,
                              "b": b["image_id"], "split_b": sb})
    cross.sort(key=lambda p: (p["distance"], p["a"]))
    return cross


def _diff_assignments(generated: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """What the generator would have done differently from the frozen table."""
    drift: dict[str, Any] = {}
    for key in ASSIGNMENT_KEYS:
        gen, frz = generated[key], frozen[key]
        if isinstance(gen, dict) and isinstance(frz, dict):
            diff = {k: {"generated": gen.get(k), "frozen": frz.get(k)}
                    for k in set(gen) | set(frz) if gen.get(k) != frz.get(k)}
        else:
            only_gen = sorted(set(gen) - set(frz))
            only_frz = sorted(set(frz) - set(gen))
            diff = ({"only_generated": only_gen} if only_gen else {}) | \
                   ({"only_frozen": only_frz} if only_frz else {})
        if diff:
            drift[key] = diff
    return drift

# A full-res Laplacian-variance sharpness below this is "blurry" for our purpose;
# the corpus is soft overall (medians ~15), so the cut is set against the corpus,
# not against a generic camera.
BLUR_SHARP_MAX = 8.0
SHARP_SHARP_MIN = 20.0


def _load_audit(path: Path = AUDIT_JSON) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} missing — run `uv run python -m freestyle.audit` first.")
    data = json.loads(path.read_text())
    return {r["image_id"]: r for r in data["images"]}


def _group_features(member_ids: list[str], audit: dict[str, dict[str, Any]],
                    recs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-group facts used by the stratification rules."""
    feats: dict[str, Any] = {
        "labels": set(), "resolutions": set(), "sharp_max": -1.0, "blurry": False,
        "upside_down": False, "rotated": False, "persons": False,
        "candidates": [], "negatives": [],
    }
    for iid in member_ids:
        a = audit.get(iid, {})
        rec = recs.get(iid, {})
        label = a.get("scene_label", "unknown")
        feats["labels"].add(label)
        feats["resolutions"].add(f"{rec.get('width')}x{rec.get('height')}")
        sharp = float(a.get("sharp_full") or 0.0)
        feats["sharp_max"] = max(feats["sharp_max"], sharp)
        if sharp < BLUR_SHARP_MAX or a.get("motion_blur") in ("mild", "severe"):
            feats["blurry"] = True
        feats["upside_down"] |= bool(a.get("upside_down"))
        feats["rotated"] |= bool(a.get("upside_down")) or bool(a.get("rotated_90"))
        feats["persons"] |= bool(a.get("person_visible"))
        if label in CANDIDATE_LABELS:
            feats["candidates"].append(iid)
        elif label in NEGATIVE_LABELS:
            feats["negatives"].append(iid)
    return feats


def _pick(available: dict[str, dict[str, Any]], pred, pool: list[str], rng: random.Random) -> str | None:
    """Randomly pick one group matching ``pred`` from ``pool`` (deterministic rng)."""
    matches = [g for g in pool if pred(available[g])]
    if not matches:
        return None
    return rng.choice(sorted(matches))


def build_splits(test_frac: float = 0.4, val_frac: float = 0.2, seed: int = 0,
                 audit_path: Path = AUDIT_JSON,
                 frozen_path: str | Path | None = FROZEN_SPLITS_PATH,
                 drift: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the split dict.

    ``drift``: optional dict that receives ``generated`` vs ``frozen`` assignment
    diffs (reporting only — the schema of ``splits.json`` is unchanged, so the
    file stays byte-for-byte identical to the frozen table while nothing drifted).
    """
    audit = _load_audit(audit_path)
    recs = {r["image_id"]: r for r in list_images()}
    gid_to_members = groups()

    available: dict[str, dict[str, Any]] = {}
    excluded_groups: dict[str, str] = {}
    for gid, members in gid_to_members.items():
        feats = _group_features(members, audit, recs)
        if feats["candidates"]:
            available[gid] = feats
        else:
            reason = ",".join(sorted(feats["labels"])) or "unlabelled"
            excluded_groups[gid] = f"no cabinet_full/cabinet_partial frame in group (labels: {reason})"

    # Forced placements first.
    forced: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for gid, feats in available.items():
        if BEST_VIEW in feats["candidates"]:
            forced[gid] = "val"
            reasons[gid] = "forced val: best single view of the cabinet, kept for development"
        elif gid == BLUR_GROUP:
            forced[gid] = "train"
            reasons[gid] = "forced train: 12-frame motion-blur burst kept as a training group"

    free = sorted(set(available) - set(forced))
    rng = random.Random(seed)
    rng.shuffle(free)

    n_test_target = max(1, round(test_frac * len(available)))
    n_val_target = max(1, round(val_frac * len(available)))
    already_test = 0

    # ---- stratified test picks (each pick is a rule the task demands) ----
    test_picks: dict[str, str] = {}

    def take(pred, why, already_in_test: bool = False):  # noqa: FBT001
        nonlocal already_test
        pool = [g for g in free if g not in test_picks]
        gid = _pick(available, pred, pool, rng)
        if gid is None:
            return None
        test_picks[gid] = why
        if not already_in_test:
            already_test += 1
        return gid

    take(lambda f: "cabinet_full" in f["labels"] and f["sharp_max"] >= SHARP_SHARP_MIN,
         "stratum: sharp full view of both shelves")
    take(lambda f: "cabinet_partial" in f["labels"],
         "stratum: partial cabinet view")
    take(lambda f: f["blurry"], "stratum: blurry view")
    take(lambda f: f["rotated"], "stratum: upside-down / rotated view")
    take(lambda f: "1600x1200" in f["resolutions"], "stratum: native 1600x1200 (UXGA)")
    take(lambda f: "2048x1536" in f["resolutions"], "stratum: native 2048x1536 (QXGA)")
    take(lambda f: f["persons"], "stratum: frame containing a person")

    remaining_test = max(0, n_test_target - already_test)
    for gid in [g for g in free if g not in test_picks][:remaining_test]:
        test_picks[gid] = f"random draw to reach {test_frac:.0%} of candidate groups (seed={seed})"

    test_set = set(test_picks)
    val_pool = [g for g in free if g not in test_set]
    val_target_extra = max(0, n_val_target - sum(1 for g, s in forced.items() if s == "val"))
    val_set = set(val_pool[:val_target_extra])

    # ---- assemble the file ----
    groups_map: dict[str, str] = {}
    images_map: dict[str, str] = {}
    negatives: list[str] = []

    for gid in sorted(set(available) | set(excluded_groups)):
        if gid in excluded_groups:
            split = "exclude"
        else:
            split = forced.get(gid) or ("test" if gid in test_set else
                                        "val" if gid in val_set else "train")
            if gid not in reasons and gid not in test_picks:
                reasons[gid] = ("random draw to the val share" if gid in val_set
                                else "random draw to the train remainder")
        groups_map[gid] = split
        for iid in sorted(gid_to_members[gid]):
            label = (audit.get(iid, {}) or {}).get("scene_label", "unknown")
            if label in NEGATIVE_LABELS:
                # Excluded from training even inside a group we otherwise use;
                # kept as rejection-test material under "negatives".
                images_map[iid] = "exclude"
                negatives.append(iid)
            elif gid in excluded_groups:
                images_map[iid] = "exclude"
            else:
                images_map[iid] = split

    negatives.sort()
    generated: dict[str, Any] = {"groups": groups_map, "images": images_map,
                                 "negatives": list(negatives)}
    out = {
        "version": 1,
        "seed": seed,
        "unit": "group",
        "source_audit": str(AUDIT_JSON.relative_to(REPO_ROOT)),
        "rules": {
            "candidate_labels": list(CANDIDATE_LABELS),
            "negative_labels": list(NEGATIVE_LABELS),
            "forced_val": BEST_VIEW,
            "forced_train_group": BLUR_GROUP,
            "test_frac": test_frac,
        },
        "reasons": {gid: (reasons.get(gid) or test_picks.get(gid) or
                          excluded_groups.get(gid) or
                          ("random draw to the test share" if gid in test_set else
                           "random draw to the val share" if gid in val_set else
                           "random draw to the train remainder"))
                    for gid in groups_map},
        "groups": groups_map,
        "images": images_map,
        "negatives": negatives,
        "near_duplicates_across_splits": _cross_split_pairs(audit, images_map),
    }
    return _apply_frozen(out, generated, audit, frozen_path, drift)


def _apply_frozen(out: dict[str, Any], generated: dict[str, Any],
                  audit: dict[str, dict[str, Any]],
                  frozen_path: str | Path | None,
                  drift: dict[str, Any] | None = None) -> dict[str, Any]:
    """Copy the reviewed assignment over the generated one and record the drift.

    The frozen table must cover the current corpus exactly: if the inventory grew
    or shrank, the freeze is stale and a human has to re-freeze — silently
    dropping the new frames (or keeping ids that no longer exist) would be worse.
    """
    frozen = _load_frozen(frozen_path)
    if frozen is None:
        return out

    frz = {"groups": frozen["groups"], "images": frozen["images"],
           "negatives": sorted(frozen.get("negatives", []))}
    missing = sorted(set(generated["images"]) - set(frz["images"]))
    stale = sorted(set(frz["images"]) - set(generated["images"]))
    if missing or stale:
        raise SystemExit(
            f"frozen split {frozen_path} does not cover the corpus: "
            f"{len(missing)} image(s) not in it (e.g. {missing[:3]}), "
            f"{len(stale)} image(s) in it but not on disk (e.g. {stale[:3]}). "
            "A human must review and re-freeze before regenerating.")

    drift_out = _diff_assignments(generated, frz)
    if drift is not None:
        drift.update(drift_out)
    out["groups"], out["images"], out["negatives"] = frz["groups"], frz["images"], frz["negatives"]
    out["reasons"] = {**out["reasons"], **(frozen.get("reasons") or {})}
    out["near_duplicates_across_splits"] = _cross_split_pairs(audit, frz["images"])
    return out


def _check(path: Path, frozen_path: str | Path | None = FROZEN_SPLITS_PATH) -> None:
    data = json.loads(path.read_text())
    gid_to_members = groups()
    bad = [g for g in data["groups"] if data["groups"][g] not in ("train", "val", "test", "exclude")]
    if bad:
        raise SystemExit(f"bad split value for groups {bad[:5]}")
    for gid, split in data["groups"].items():
        for iid in gid_to_members.get(gid, []):
            got = data["images"].get(iid)
            if got is None:
                raise SystemExit(f"image {iid} missing from splits")
            if got != "exclude" and got != split:
                raise SystemExit(f"image {iid} in {split} group is labelled {got}")
    overlap = {g for g in data["groups"] if data["groups"][g] == "test"} & \
              {g for g in data["groups"] if data["groups"][g] in ("train", "val")}
    if overlap:
        raise SystemExit(f"group in two splits: {overlap}")
    counts = {s: sum(1 for v in data["images"].values() if v == s)
              for s in ("train", "val", "test", "exclude")}
    cross = [p for p in data.get("near_duplicates_across_splits", [])
             if "test" in (p["split_a"], p["split_b"])]
    note = ""
    frozen = _load_frozen(frozen_path)
    if frozen is not None:
        for key in ASSIGNMENT_KEYS:
            want = frozen["negatives"] if key == "negatives" else frozen[key]
            got = sorted(data.get(key, [])) if key == "negatives" else data.get(key)
            if (sorted(want) if key == "negatives" else want) != got:
                raise SystemExit(f"{path} does not match the frozen split on '{key}'")
        note = ", matches frozen"
    print(f"splits ok: {len(data['groups'])} groups, {len(data['images'])} images {counts}, "
          f"{len(data['negatives'])} negatives, "
          f"{len(cross)} near-duplicates touching test (<=8 bits){note}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(SPLITS_PATH))
    p.add_argument("--test-frac", type=float, default=0.4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--audit", default=str(AUDIT_JSON))
    p.add_argument("--frozen", default=str(FROZEN_SPLITS_PATH),
                   help="reviewed assignment to copy verbatim ('' to regenerate from scratch)")
    p.add_argument("--check", action="store_true", help="validate the existing file and exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    frozen = args.frozen or None
    if args.check:
        _check(Path(args.out), frozen)
        return

    data = build_splits(args.test_frac, args.val_frac, args.seed, Path(args.audit), frozen,
                        drift := {})
    counts = {s: sum(1 for v in data["images"].values() if v == s)
              for s in ("train", "val", "test", "exclude")}
    gcounts = {s: sum(1 for v in data["groups"].values() if v == s)
               for s in ("train", "val", "test", "exclude")}
    print(f"groups {gcounts} images {counts} negatives={len(data['negatives'])}")
    if frozen:
        ngroups = len(drift.get("groups") or {})
        nimages = len(drift.get("images") or {})
        nneg = len(drift.get("negatives") or {})
        verdict = (f"{ngroups} group / {nimages} image / {nneg} negative assignment(s) "
                   "would have differed (frozen wins)" if drift else
                   "the draw still reproduces it exactly")
        print(f"frozen split {frozen}: {verdict}")
        if drift:
            for key, diff in drift.items():
                sample = list(diff.items())[:5] if isinstance(diff, dict) else diff
                print(f"  frozen_drift[{key}]: {len(diff)} entry(ies) {sample}")
    else:
        print("no frozen split found — writing the generated assignment")
    if args.dry_run:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    _check(out, frozen)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
