"""Phase 16 — CV training dataset audit.

Pure-Python sweep of `data/cv_training/` that flags:
    • corrupt / unreadable image files
    • labels with out-of-bounds coords (>1.0 or <0.0)
    • malformed label lines (wrong token count, parse errors)
    • invalid class indices
    • unlabeled frames (empty / missing .txt)
    • near-duplicate images (perceptual hash — imagehash phash)
    • per-class frequency imbalance

Writes findings to `data/cv_training/audit_report.json`.

Optional: pass --launch to open a FiftyOne browser UI (requires mongod).

Usage::

    uv run python scripts/gretzky.py audit-cv
    uv run python scripts/gretzky.py audit-cv -- --launch
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

from models.cv.player_detector import CLASS_NAMES, NUM_CLASSES

_CV         = _REPO / "data" / "cv_training"
DATASET_YAML = _CV / "dataset.yaml"
AUDIT_JSON  = _CV / "audit_report.json"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from PIL import Image
    import imagehash

    if not DATASET_YAML.exists():
        print(f"[audit] dataset.yaml not found at {DATASET_YAML}")
        sys.exit(1)

    splits = []
    for split in ("train", "val"):
        img_dir = _CV / "images" / split
        lbl_dir = _CV / "labels" / split
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits.append((split, img_dir, lbl_dir))
    if not splits:
        print(f"[audit] No images/<split> dirs found under {_CV}")
        sys.exit(1)

    total_imgs   = 0
    total_lbls   = 0
    corrupt_imgs: list[str] = []
    oob_labels:   list[dict] = []
    bad_class:    list[dict] = []
    empty_lbls:   list[str] = []
    class_counter: Counter[str] = Counter()

    # perceptual hash → list of image paths with that hash
    phash_bucket: dict[str, list[str]] = defaultdict(list)
    img_records: list[tuple[str, str]] = []

    for split, img_dir, lbl_dir in splits:
        for img_path in sorted(img_dir.glob("*.jpg")):
            total_imgs += 1

            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    # Reopen because verify() closes the file
                    ph = str(imagehash.phash(im.convert("RGB")))
            except Exception as exc:
                corrupt_imgs.append(f"{img_path.name}: {exc}")
                continue

            phash_bucket[ph].append(str(img_path))
            img_records.append((split, str(img_path)))

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                empty_lbls.append(img_path.name)
                continue

            for lineno, raw in enumerate(lbl_path.read_text().splitlines(), 1):
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                if len(parts) != 5:
                    oob_labels.append({
                        "file": str(lbl_path),
                        "line": lineno,
                        "reason": f"expected 5 tokens, got {len(parts)}",
                    })
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:])
                except ValueError as exc:
                    oob_labels.append({
                        "file": str(lbl_path),
                        "line": lineno,
                        "reason": f"parse error: {exc}",
                    })
                    continue

                if cls < 0 or cls >= NUM_CLASSES:
                    bad_class.append({
                        "file": str(lbl_path),
                        "line": lineno,
                        "class_idx": cls,
                    })
                    continue

                if any(v < 0 or v > 1 for v in (cx, cy, bw, bh)):
                    oob_labels.append({
                        "file": str(lbl_path),
                        "line": lineno,
                        "box": [cx, cy, bw, bh],
                        "reason": "coord outside [0,1]",
                    })
                    continue

                total_lbls += 1
                class_counter[CLASS_NAMES[cls]] += 1

    # Duplicate detection — same phash = visually near-identical
    near_duplicates = [paths for paths in phash_bucket.values() if len(paths) > 1]

    # Optional dedupe: keep one canonical image per phash group, delete the
    # rest (and their label sidecar). The keeper is the lexicographically
    # first path so the choice is deterministic across runs. Train-set bias
    # would skew toward keepers from earlier sequences alphabetically; the
    # detector doesn't care about source-ordering at this scale.
    deleted_imgs   = 0
    deleted_lbls   = 0
    if args.delete_dupes and near_duplicates:
        for group in near_duplicates:
            keep = sorted(group)[0]
            for p in group:
                if p == keep:
                    continue
                img_path = Path(p)
                lbl_path = img_path.parents[2] / "labels" / img_path.parent.name / (img_path.stem + ".txt")
                try:
                    img_path.unlink(missing_ok=True)
                    deleted_imgs += 1
                except OSError as e:
                    print(f"[audit] could not delete {img_path}: {e}")
                if lbl_path.exists():
                    try:
                        lbl_path.unlink()
                        deleted_lbls += 1
                    except OSError as e:
                        print(f"[audit] could not delete {lbl_path}: {e}")
        print(f"[audit] Deleted {deleted_imgs:,} duplicate images "
              f"({deleted_lbls:,} label files) — kept one canonical per group.")

    report = {
        "summary": {
            "images":            total_imgs,
            "labels":            total_lbls,
            "corrupt_images":    len(corrupt_imgs),
            "oob_labels":        len(oob_labels),
            "bad_class_idx":     len(bad_class),
            "empty_label_files": len(empty_lbls),
            "dupe_groups":       len(near_duplicates),
            "dupe_images":       sum(len(g) for g in near_duplicates),
            "deleted_images":    deleted_imgs,
            "deleted_labels":    deleted_lbls,
        },
        "class_counts":     dict(class_counter.most_common()),
        "corrupt_images":   corrupt_imgs,
        "oob_labels":       oob_labels[:200],
        "bad_class_idx":    bad_class,
        "empty_label_files": empty_lbls[:50],
        "dupe_groups":      near_duplicates[:50],
    }
    AUDIT_JSON.write_text(json.dumps(report, indent=2))

    print("\n" + "─" * 55)
    print(f"  Images            : {total_imgs:,}")
    print(f"  Labels            : {total_lbls:,}")
    print(f"  Corrupt images    : {len(corrupt_imgs)}")
    print(f"  OOB / malformed   : {len(oob_labels)}")
    print(f"  Bad class indices : {len(bad_class)}")
    print(f"  Unlabeled frames  : {len(empty_lbls)}")
    print(f"  Dupe groups / imgs: {len(near_duplicates)} / "
          f"{sum(len(g) for g in near_duplicates)}")
    print("  Per-class counts  :")
    for cls, n in class_counter.most_common():
        print(f"     {cls:<14} {n:,}")
    print(f"\n  Report → {AUDIT_JSON}")
    print("─" * 55)

    if args.launch:
        _launch_fiftyone(img_records, _CV, port=args.port)


def _launch_fiftyone(img_records, cv_root: Path, port: int) -> None:
    """Optional: open FiftyOne web UI. Requires mongod on PATH."""
    try:
        import fiftyone as fo
    except Exception as exc:
        print(f"[audit] FiftyOne not importable: {exc}")
        return

    try:
        name = "grtzky_cv_training"
        if name in fo.list_datasets():
            fo.delete_dataset(name)
        dataset = fo.Dataset(name)
    except Exception as exc:
        print(f"[audit] Could not start FiftyOne (mongod missing?): {exc}")
        print("[audit] Install mongod or set FIFTYONE_DATABASE_URI to use --launch")
        return

    samples = []
    for split, img_path in img_records:
        lbl_path = cv_root / "labels" / split / (Path(img_path).stem + ".txt")
        detections = []
        if lbl_path.exists():
            for raw in lbl_path.read_text().splitlines():
                parts = raw.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:])
                except ValueError:
                    continue
                if not (0 <= cls < NUM_CLASSES):
                    continue
                x0 = max(0.0, cx - bw / 2)
                y0 = max(0.0, cy - bh / 2)
                detections.append(fo.Detection(
                    label=CLASS_NAMES[cls],
                    bounding_box=[x0, y0, bw, bh],
                ))
        samples.append(fo.Sample(
            filepath=img_path,
            tags=[split],
            ground_truth=fo.Detections(detections=detections),
        ))
    dataset.add_samples(samples)
    print(f"[audit] Launching FiftyOne on port {port}. Ctrl-C to stop.")
    session = fo.launch_app(dataset, port=port)
    session.wait()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit data/cv_training for data quality issues.")
    p.add_argument("--launch", action="store_true",
                   help="Open FiftyOne web UI after audit (needs mongod)")
    p.add_argument("--port",   type=int, default=5151)
    p.add_argument("--delete-dupes", action="store_true",
                   help="Remove near-duplicate images (one canonical kept per "
                        "phash group). Destructive — run audit first without "
                        "this flag and inspect dupe_groups in the report.")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
