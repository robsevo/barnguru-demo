"""Fetch HockeyAI (SimulaMet-HOST) from HuggingFace → GRTZKY YOLO layout.

The HockeyAI dataset ships 2,100 SHL broadcast frames annotated in YOLO format.
This script downloads it via the ``datasets`` library, remaps their 7-class
schema onto GRTZKY's 6-class detector schema (dropping rink-landmark classes,
which we already handle via ``HockeyRink.pt``), and writes the standard

    data/cv_training/images/train/*.jpg
    data/cv_training/images/val/*.jpg
    data/cv_training/labels/train/*.txt
    data/cv_training/labels/val/*.txt
    data/cv_training/dataset.yaml

layout expected by ``scripts/train_cv_detector.py`` (Feature 16.2).

Class remap
-----------
    HockeyAI index  HockeyAI name   →  GRTZKY index   GRTZKY name
    0               centerIce           (drop)         —
    1               faceoff             (drop)         —
    2               goal                (drop)         —
    3               goaltender          1              goalie
    4               player              0              player
    5               puck                3              puck
    6               referee             2              referee

Home/away is decided at runtime by the torso-saturation classifier in
`useCvLiveReplay.ts`; the detector only learns the 4 "role" classes.
Previous 6-class schema left 2 classes (player_away, goalie_away) with zero
training instances, which inflated mAP and confused downstream rendering.

Usage
-----
    uv run python scripts/fetch_hockey_ai.py
    uv run python scripts/fetch_hockey_ai.py --val-split 0.15 --force
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import zipfile

from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
CV_ROOT = _REPO / "data" / "cv_training"
IMAGES_DIR = CV_ROOT / "images"
LABELS_DIR = CV_ROOT / "labels"
YAML_PATH  = CV_ROOT / "dataset.yaml"

GRTZKY_CLASSES: list[str] = [
    "player",        # 0  — home/away decided at inference from torso sat
    "goalie",        # 1  — home/away decided at inference from torso sat
    "referee",       # 2
    "puck",          # 3
]

# HockeyAI → GRTZKY remap. None = drop annotation.
HOCKEYAI_TO_GRTZKY: dict[int, int | None] = {
    0: None,  # centerIce  → drop
    1: None,  # faceoff    → drop
    2: None,  # goal       → drop
    3: 1,     # goaltender → goalie
    4: 0,     # player     → player
    5: 3,     # puck       → puck
    6: 2,     # referee    → referee
}


def remap_label_file(raw_txt: str) -> str:
    """Return HockeyAI-formatted YOLO labels remapped onto GRTZKY indices."""
    out: list[str] = []
    for line in raw_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        target = HOCKEYAI_TO_GRTZKY.get(cls)
        if target is None:
            continue
        out.append(f"{target} {' '.join(parts[1:5])}")
    return "\n".join(out) + ("\n" if out else "")


def write_dataset_yaml(path: Path) -> None:
    """Write the dataset.yaml consumed by YOLO training."""
    lines = [
        "# GRTZKY CV detector training config (Feature 16.1 / 16.2)",
        "# Source: SimulaMet-HOST/HockeyAI (CC-BY / MIT), 2,100 SHL frames.",
        "# Classes remapped via scripts/fetch_hockey_ai.py",
        "",
        f"path: {CV_ROOT.resolve()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(GRTZKY_CLASSES)}",
        "names:",
    ]
    lines += [f"  {i}: {c}" for i, c in enumerate(GRTZKY_CLASSES)]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-split", type=float, default=0.15, help="fraction held out for validation")
    ap.add_argument("--seed",      type=int,   default=1337, help="shuffle seed for deterministic split")
    ap.add_argument("--force",     action="store_true",      help="wipe any existing images/labels output")
    ap.add_argument("--cache-dir", default=None,             help="huggingface dataset cache dir")
    args = ap.parse_args()

    if args.force:
        shutil.rmtree(IMAGES_DIR, ignore_errors=True)
        shutil.rmtree(LABELS_DIR, ignore_errors=True)

    for split in ("train", "val"):
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    print("[hockey-ai] downloading SimulaMet-HOST/HockeyAI zip …")
    zip_path = Path(hf_hub_download(
        repo_id="SimulaMet-HOST/HockeyAI",
        filename="HockeyAI_Dataset.zip",
        repo_type="dataset",
        cache_dir=args.cache_dir,
    ))
    print(f"[hockey-ai] zip at {zip_path}")

    # Unpack into a sibling staging dir (cached across runs)
    stage = CV_ROOT / "raw" / "hockeyai_unpacked"
    if args.force:
        shutil.rmtree(stage, ignore_errors=True)
    if not stage.exists():
        stage.mkdir(parents=True, exist_ok=True)
        print(f"[hockey-ai] unpacking → {stage}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(stage)
    else:
        print(f"[hockey-ai] reusing existing unpacked tree {stage}")

    # Find the frames/ + annotations/ dirs anywhere under stage
    frames_src = next(stage.rglob("frames"), None)
    annot_src  = next(stage.rglob("annotations"), None)
    if frames_src is None or annot_src is None:
        tree = sorted(p.relative_to(stage) for p in stage.rglob("*") if p.is_dir())[:15]
        raise RuntimeError(
            f"expected frames/ and annotations/ under {stage}; found dirs: {tree}"
        )
    print(f"[hockey-ai] frames/={frames_src}  annotations/={annot_src}")

    frame_files = sorted(
        p for p in frames_src.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"[hockey-ai] {len(frame_files):,} frames on disk")

    # Deterministic shuffle + split
    rng = random.Random(args.seed)
    indices = list(range(len(frame_files)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_split))
    val_set = set(indices[:n_val])
    print(f"[hockey-ai] split: train={len(indices) - n_val:,}  val={n_val:,}")

    dropped_empty = 0
    missing_labels = 0
    written = {"train": 0, "val": 0}

    for idx in tqdm(indices, desc="export"):
        src_img = frame_files[idx]
        split   = "val" if idx in val_set else "train"
        stem    = src_img.stem

        label_src = annot_src / f"{stem}.txt"
        if not label_src.exists():
            missing_labels += 1
            continue
        raw = label_src.read_text()
        remapped = remap_label_file(raw)
        if not remapped.strip():
            dropped_empty += 1

        # Copy image (JPEG re-encode only if not already jpg)
        dst_img = IMAGES_DIR / split / f"{stem}.jpg"
        if src_img.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copyfile(src_img, dst_img)
        else:
            Image.open(src_img).convert("RGB").save(dst_img, "JPEG", quality=92)

        (LABELS_DIR / split / f"{stem}.txt").write_text(remapped)
        written[split] += 1

    write_dataset_yaml(YAML_PATH)

    print(
        f"\n──────────────────────────────────────────\n"
        f"  HockeyAI → GRTZKY YOLO layout complete.\n"
        f"  train: {written['train']:,} | val: {written['val']:,}\n"
        f"  empty-label frames: {dropped_empty:,}\n"
        f"  missing-label frames (skipped): {missing_labels:,}\n"
        f"  images/ → {IMAGES_DIR}\n"
        f"  labels/ → {LABELS_DIR}\n"
        f"  dataset.yaml → {YAML_PATH}\n"
        f"──────────────────────────────────────────\n"
    )


if __name__ == "__main__":
    main()
