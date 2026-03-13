from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

FAMILY_WEIGHTS: Dict[str, List[str]] = {
    "yolo11": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
    "yolo12": ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"],
    "yolo26": ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
    # Optional legacy groups.
    "yolov10": ["yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt"],
    "yolov9": ["yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt"],
    "yolov8": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    "yolov5": ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"],
}

DEFAULT_FAMILIES = ["yolo11", "yolo12", "yolo26"]
LEGACY_FAMILIES = ["yolov10", "yolov9", "yolov8", "yolov5"]


def _parse_csv(text: str | None) -> List[str]:
    if text is None:
        return []
    out: List[str] = []
    for token in text.split(","):
        t = token.strip().lower()
        if t:
            out.append(t)
    return out


def _resolve_required_weights(families: List[str], include_legacy: bool) -> List[str]:
    chosen = list(families) if families else list(DEFAULT_FAMILIES)
    if include_legacy:
        for fam in LEGACY_FAMILIES:
            if fam not in chosen:
                chosen.append(fam)

    unknown = [f for f in chosen if f not in FAMILY_WEIGHTS]
    if unknown:
        raise ValueError(f"Unknown families: {unknown}. Valid: {sorted(FAMILY_WEIGHTS.keys())}")

    required: List[str] = []
    for fam in chosen:
        required.extend(FAMILY_WEIGHTS[fam])
    # Keep stable order, remove duplicates.
    dedup: List[str] = []
    seen = set()
    for w in required:
        if w in seen:
            continue
        seen.add(w)
        dedup.append(w)
    return dedup


def _scan_existing_weights(weights_dir: Path) -> List[str]:
    if not weights_dir.exists():
        return []
    return sorted([p.name for p in weights_dir.glob("*.pt") if p.is_file()])


def _download_missing(missing: List[str], weights_dir: Path) -> List[str]:
    if not missing:
        return []
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] ultralytics import failed: {e}")
        return list(missing)
    weights_dir.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    old_cwd = Path.cwd()
    try:
        os.chdir(str(weights_dir))
        for weight in missing:
            try:
                print(f"[DOWNLOAD] {weight}")
                YOLO(weight, task="detect")
                print(f"[OK] {weight}")
            except Exception as e:
                failed.append(weight)
                print(f"[FAILED] {weight}: {e}")
    finally:
        os.chdir(str(old_cwd))
    return failed


def _remove_extra(extra: List[str], weights_dir: Path) -> List[str]:
    removed: List[str] = []
    for name in extra:
        p = weights_dir / name
        try:
            if p.exists():
                p.unlink()
                removed.append(name)
        except Exception as e:
            print(f"[WARN] failed to remove {name}: {e}")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync YOLO weight files (missing/extra) for TEGB experiments")
    parser.add_argument(
        "--weights-dir",
        default=".",
        help="Directory to check/store weight .pt files",
    )
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated model families, e.g. yolo11,yolo12,yolo26",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include yolov10/yolov9/yolov8/yolov5 families",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report missing/extra, do not download or delete",
    )
    parser.add_argument(
        "--prune-extra",
        action="store_true",
        help="Delete extra local .pt files that are not in required list",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if there are unresolved missing weights or any extras",
    )
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    families = _parse_csv(args.families)
    required = _resolve_required_weights(families=families, include_legacy=bool(args.include_legacy))
    required_set = set(required)

    existing = _scan_existing_weights(weights_dir)
    existing_set = set(existing)

    missing = [w for w in required if w not in existing_set]
    extra = sorted([w for w in existing if w not in required_set])

    print("=== TEGB Weight Sync Report ===")
    print(f"weights_dir: {weights_dir}")
    print(f"required_count: {len(required)}")
    print(f"existing_count: {len(existing)}")
    print(f"missing_count: {len(missing)}")
    print(f"extra_count: {len(extra)}")

    if missing:
        print("[MISSING]")
        for w in missing:
            print(f"  - {w}")
    if extra:
        print("[EXTRA]")
        for w in extra:
            print(f"  - {w}")

    failed_downloads: List[str] = []
    if not args.check_only and missing:
        failed_downloads = _download_missing(missing=missing, weights_dir=weights_dir)

    removed: List[str] = []
    if not args.check_only and args.prune_extra and extra:
        removed = _remove_extra(extra=extra, weights_dir=weights_dir)
        print(f"[PRUNE] removed {len(removed)} extra files")

    # Refresh status after actions.
    existing_after = _scan_existing_weights(weights_dir)
    existing_after_set = set(existing_after)
    missing_after = [w for w in required if w not in existing_after_set]
    extra_after = sorted([w for w in existing_after if w not in required_set])

    print("=== Final Status ===")
    print(f"missing_after: {len(missing_after)}")
    print(f"extra_after: {len(extra_after)}")
    if missing_after:
        for w in missing_after:
            print(f"  missing: {w}")
    if extra_after:
        for w in extra_after:
            print(f"  extra: {w}")
    if failed_downloads:
        print("[FAILED_DOWNLOADS]")
        for w in failed_downloads:
            print(f"  - {w}")

    if args.strict and (missing_after or extra_after):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
