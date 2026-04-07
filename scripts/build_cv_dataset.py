"""Phase 16.1 — Build CV training dataset pipeline.

Orchestrates the full pipeline:
    1. Download (or verify) the Hockey Tracking Dataset
    2. Extract frames from video clips (2 FPS default)
    3. Convert MOT annotations → YOLO format labels
    4. Augment (10× flip/blur/jitter/warp)
    5. Write a dataset.yaml for YOLO training (used by 16.2)

Output layout::

    data/cv_training/
        raw/             ← downloaded dataset (MOT format)
        frames/          ← extracted JPEGs, one dir per sequence
        labels/          ← YOLO .txt files, one per frame
        aug_images/      ← 10× augmented images
        aug_labels/      ← 10× augmented labels
        dataset.yaml     ← YOLO training config

Usage::

    uv run python scripts/build_cv_dataset.py
    uv run python scripts/build_cv_dataset.py --skip-download
    uv run python scripts/build_cv_dataset.py --dataset-url https://example.com/hockey.zip
    uv run python scripts/build_cv_dataset.py --local-archive /path/to/hockey.zip
    uv run python scripts/build_cv_dataset.py --fps 5 --no-augment --force
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

from data.cv_training.dataset_downloader import (
    RAW_DIR,
    dataset_present,
    download,
    list_sequences,
    list_videos,
    unpack_local,
)
from data.cv_training.frame_extractor import FRAMES_DIR, extract_all
from data.cv_training.mot_to_yolo import LABELS_DIR, DEFAULT_CLASS_MAP, convert_all
from data.cv_training.augmentor import AUG_IMG, AUG_LBL, augment_all

# ---------------------------------------------------------------------------
# YOLO class names — must match 16.2 training config
# ---------------------------------------------------------------------------

YOLO_CLASSES = [
    "player_home",
    "player_away",
    "goalie_home",
    "goalie_away",
    "referee",
    "puck",
]

# ---------------------------------------------------------------------------
# Default dataset source
# ---------------------------------------------------------------------------

# MOT17 — Multi-Object Tracking benchmark dataset (pedestrian tracking).
# Freely available from motchallenge.net; ~5.5 GB; MOT16/17 format compatible.
# Contains dense multi-person tracking from diverse camera angles — a good
# proxy for hockey broadcast footage when hockey-specific data is unavailable.
# SportsMOT (https://sportsmot.is.tue.mpg.de) is the preferred source when
# accessible (requires CodaLab registration).
_DEFAULT_DATASET_URL = (
    "https://motchallenge.net/data/MOT17.zip"
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    raw_dir    = Path(args.raw_dir)
    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)
    aug_img    = Path(args.aug_img_dir)
    aug_lbl    = Path(args.aug_lbl_dir)

    print(_banner())

    # ── Step 1: Acquire dataset ───────────────────────────────────────────────
    if args.skip_download:
        if not dataset_present(raw_dir):
            print(
                f"\n[pipeline] ERROR: --skip-download was set but no data found in {raw_dir}.\n"
                "  Place your Hockey Tracking Dataset (MOT format) there and re-run.\n"
                "  Each sequence must contain: gt/gt.txt and either img1/ or a .mp4 file.\n"
            )
            sys.exit(1)
        print(f"[pipeline] ✓ Dataset present at {raw_dir}")
    elif args.local_archive:
        unpack_local(Path(args.local_archive), raw_dir)
    else:
        url = args.dataset_url or _DEFAULT_DATASET_URL
        try:
            download(url, raw_dir, force=args.force)
        except Exception as e:
            print(
                f"\n[pipeline] Download failed: {e}\n\n"
                "  Manually place the Hockey Tracking Dataset in:\n"
                f"    {raw_dir}\n\n"
                "  Then re-run with --skip-download.\n"
                "  Expected layout:\n"
                "    raw/<sequence>/gt/gt.txt\n"
                "    raw/<sequence>/<sequence>.mp4  (or img1/*.jpg)\n"
                "    raw/<sequence>/seqinfo.ini     (optional but recommended)\n"
            )
            sys.exit(1)

    # Summarise what we have
    seqs   = list_sequences(raw_dir)
    videos = list_videos(raw_dir)
    print(f"[pipeline] Found {len(seqs)} MOT sequences, {len(videos)} video files.")

    # ── Step 2: Extract frames ────────────────────────────────────────────────
    if not args.skip_extract:
        print(f"\n[pipeline] Extracting frames @ {args.fps} FPS …")
        frame_counts = extract_all(
            raw_dir=raw_dir,
            frames_dir=frames_dir,
            target_fps=args.fps,
            force=args.force,
        )
        total_frames = sum(frame_counts.values())
        print(f"[pipeline] ✓ {total_frames:,} frames across {len(frame_counts)} sequences.")
    else:
        total_frames = sum(len(list((frames_dir / s.name).glob("*.jpg"))) for s in seqs)
        print(f"[pipeline] ✓ Skipping extraction. {total_frames:,} frames already present.")

    # ── Step 3: Convert MOT → YOLO labels ────────────────────────────────────
    if not args.skip_convert:
        print(f"\n[pipeline] Converting MOT annotations → YOLO labels …")
        label_counts = convert_all(
            raw_dir=raw_dir,
            frames_dir=frames_dir,
            labels_dir=labels_dir,
            mot_class_map=DEFAULT_CLASS_MAP,
            force=args.force,
        )
        total_labels = sum(label_counts.values())
        print(f"[pipeline] ✓ {total_labels:,} YOLO label files.")
    else:
        total_labels = sum(len(list((labels_dir / s.name).glob("*.txt"))) for s in seqs)
        print(f"[pipeline] ✓ Skipping conversion. {total_labels:,} labels already present.")

    # ── Step 4: Augmentation ──────────────────────────────────────────────────
    if not args.no_augment:
        print(f"\n[pipeline] Augmenting (10× per frame pair) …")
        aug_count = augment_all(
            frames_dir=frames_dir,
            labels_dir=labels_dir,
            aug_img_dir=aug_img,
            aug_lbl_dir=aug_lbl,
            force=args.force,
        )
        print(f"[pipeline] ✓ {aug_count:,} augmented pairs.")
    else:
        print(f"\n[pipeline] ✓ Augmentation skipped (--no-augment).")
        aug_count = 0

    # ── Step 5: Write dataset.yaml ───────────────────────────────────────────
    yaml_path = raw_dir.parent / "dataset.yaml"
    _write_dataset_yaml(
        yaml_path,
        images_dir=aug_img if aug_count > 0 else frames_dir,
        labels_dir=aug_lbl if aug_count > 0 else labels_dir,
        classes=YOLO_CLASSES,
    )
    print(f"\n[pipeline] ✓ dataset.yaml written → {yaml_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_pairs = aug_count if aug_count > 0 else total_labels
    print(
        f"\n{'─'*55}\n"
        f"  Phase 16.1 complete.\n"
        f"  Sequences   : {len(seqs)}\n"
        f"  Raw frames  : {total_frames:,}\n"
        f"  Label files : {total_labels:,}\n"
        f"  Aug pairs   : {aug_count:,}\n"
        f"  Training set: {final_pairs:,} total (image, label) pairs\n"
        f"  Next step   : uv run python scripts/gretzky.py train-cv-detector\n"
        f"{'─'*55}\n"
    )


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------

def _write_dataset_yaml(
    path: Path,
    images_dir: Path,
    labels_dir: Path,
    classes: list[str],
) -> None:
    nc = len(classes)
    names_str = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    content = textwrap.dedent(f"""\
        # GRTZKY Phase 16 — YOLO Training Dataset Config
        # Auto-generated by build_cv_dataset.py (Feature 16.1)
        # Used by train_cv_detector.py (Feature 16.2)

        path: {path.parent.resolve()}
        train: {images_dir.resolve()}
        val:   {images_dir.resolve()}   # same split for now; 16.2 splits train/val

        # Number of classes
        nc: {nc}

        # Class names (index → label)
        names:
        {names_str}

        # Note: home/away split (classes 0/1 and 2/3) populated by Feature 16.4
        # (team colour classifier). Until then, all players map to player_home (0)
        # and all goalies map to goalie_home (2).
    """)
    path.write_text(content)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 16.1 — Build CV training data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              uv run python scripts/build_cv_dataset.py
              uv run python scripts/build_cv_dataset.py --skip-download
              uv run python scripts/build_cv_dataset.py --dataset-url https://example.com/hockey.zip
              uv run python scripts/build_cv_dataset.py --local-archive /tmp/hockey.zip
              uv run python scripts/build_cv_dataset.py --fps 5 --no-augment
        """),
    )
    p.add_argument("--dataset-url",    default=None,      help="HTTP URL of the hockey tracking dataset archive")
    p.add_argument("--local-archive",  default=None,      help="Path to a local .zip/.tar.gz to unpack")
    p.add_argument("--skip-download",  action="store_true", help="Skip download; assume raw/ is already populated")
    p.add_argument("--skip-extract",   action="store_true", help="Skip frame extraction")
    p.add_argument("--skip-convert",   action="store_true", help="Skip MOT → YOLO conversion")
    p.add_argument("--no-augment",     action="store_true", help="Skip augmentation step")
    p.add_argument("--fps",            type=float, default=2.0, help="Frames per second to extract (default 2.0)")
    p.add_argument("--force",          action="store_true", help="Re-run all steps even if outputs exist")
    p.add_argument("--raw-dir",        default=str(RAW_DIR),    help="Path to raw dataset directory")
    p.add_argument("--frames-dir",     default=str(FRAMES_DIR), help="Path to extracted frames directory")
    p.add_argument("--aug-img-dir",    default=str(AUG_IMG),    help="Path to augmented images output")
    p.add_argument("--aug-lbl-dir",    default=str(AUG_LBL),    help="Path to augmented labels output")
    p.add_argument("--labels-dir",     default=str(LABELS_DIR), help="Path to YOLO labels directory")
    return p.parse_args(argv)


def _banner() -> str:
    return (
        "\n"
        "╔══════════════════════════════════════════════════╗\n"
        "║  GRTZKY  ·  Phase 16.1 — CV Dataset Pipeline   ║\n"
        "║  Hockey Tracking Dataset  →  YOLO Training Data  ║\n"
        "╚══════════════════════════════════════════════════╝\n"
    )


if __name__ == "__main__":
    main()
