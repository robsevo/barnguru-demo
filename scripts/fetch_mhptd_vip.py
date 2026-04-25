"""Merge MHPTD + VIP-HTD (MOT-format NHL broadcast) into our YOLO training set.

Both repos ship broadcast-domain NHL game clips annotated in MOT Challenge
format (``gt/gt.txt``). MHPTD ships MP4 clips — frames must be extracted.
VIP-HTD ships pre-extracted ``img1/`` folders.

MHPTD gt.txt column layout::

    frame, track_id, x, y, w, h, conf, class, visibility

The ``class`` column is populated:

    1: home player   (→ GRTZKY 0 player)
    2: away player   (→ GRTZKY 0 player)
    3: home goalie   (→ GRTZKY 1 goalie)
    4: away goalie   (→ GRTZKY 1 goalie)
    5: referee       (→ GRTZKY 2 referee)

Puck is not labelled in either dataset — HockeyAI already covers puck.
Home/away is decided at inference by torso saturation, so we collapse
home+away into the same class here.

MHPTD layout (videos + annotations separate)::

    clips/<game_name>/NNN.mp4   or   clips/<game_name>/<game_name>_NNN.mp4
    MOT_Challenge_Sytle_Label/{train,test}/<seq_name>/gt.txt

VIP-HTD layout (pre-extracted frames)::

    mot-challenge-format/{train,validation,test}/<seq>/img1/NNNNNN.jpg
    mot-challenge-format/{train,validation,test}/<seq>/gt/gt.txt

Output appended to the GRTZKY YOLO layout::

    data/cv_training/images/train/mhptd__<seq>__NNNNNN.jpg
    data/cv_training/labels/train/mhptd__<seq>__NNNNNN.txt
    data/cv_training/images/val/vip_htd__PIT_VS_WAS_001__NNNNNN.jpg
    ...

Frame subsampling (``--stride N``) is on by default — adjacent frames are
near-duplicates to a detector.

Usage::

    uv run python scripts/fetch_mhptd_vip.py --stride 6
    uv run python scripts/fetch_mhptd_vip.py --stride 3 --max-frames-per-seq 400
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
CV_ROOT = _REPO / "data" / "cv_training"
EXTERNAL_ROOT = CV_ROOT / "external"
IMAGES_DIR = CV_ROOT / "images"
LABELS_DIR = CV_ROOT / "labels"

VIP_ROOT    = EXTERNAL_ROOT / "vip_htd" / "mot-challenge-format"
MHPTD_ROOT  = EXTERNAL_ROOT / "mhptd"
MHPTD_GT_ROOT  = MHPTD_ROOT / "MOT_Challenge_Sytle_Label"
MHPTD_CLIP_ROOT = MHPTD_ROOT / "clips"

MIN_VISIBILITY = 0.1

# MHPTD class (column 8) → GRTZKY class. None = skip.
MHPTD_TO_GRTZKY: dict[int, int | None] = {
    1: 0,  # home player → player
    2: 0,  # away player → player
    3: 1,  # home goalie → goalie
    4: 1,  # away goalie → goalie
    5: 2,  # referee
}


# ---------------------------------------------------------------------------
# Clip resolution
# ---------------------------------------------------------------------------

# Seq name → mp4 path mapping is not 1-to-1; MHPTD stores clips under
#   clips/<game>/<NNN or GAME_NNN>.mp4
# and seq dirs are named with year embedded (eg allstar_2019_001 vs 001.mp4).
# We build a lookup by scanning every mp4 and matching against the seq tail.
def _find_mp4_for_seq(seq_name: str) -> Path | None:
    seq_lower = seq_name.lower()
    # e.g. "allstar_2019_001" → expect mp4 ending with "_001.mp4" or "/001.mp4"
    m = re.search(r"(\d{3})$", seq_name)
    tail = m.group(1) if m else None
    candidates: list[Path] = list(MHPTD_CLIP_ROOT.rglob("*.mp4"))
    # Exact-name match first
    for mp4 in candidates:
        if mp4.stem.lower() == seq_lower:
            return mp4
    # Match by trailing 3-digit index within a parent dir whose name is
    # a prefix of the seq
    if tail:
        for mp4 in candidates:
            parent = mp4.parent.name.lower()
            if (mp4.stem.endswith(tail)
                and seq_lower.startswith(parent.rstrip("_0123456789"))):
                return mp4
    # Loose contains
    for mp4 in candidates:
        if mp4.stem.lower() in seq_lower or seq_lower in mp4.stem.lower():
            return mp4
    return None


def _extract_frames(mp4: Path, dest: Path) -> int:
    """ffmpeg: extract every frame as img1/%06d.jpg (1-indexed)."""
    dest.mkdir(parents=True, exist_ok=True)
    # Use -start_number 1 so ffmpeg writes 000001.jpg, matching gt frame 1.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(mp4),
        "-start_number", "1",
        "-q:v", "3",
        str(dest / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(dest.glob("*.jpg")))


# ---------------------------------------------------------------------------
# GT parsing + YOLO conversion
# ---------------------------------------------------------------------------

def parse_gt_with_class(gt_path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Parse MOT gt.txt → {frame: [(grtzky_cls, x, y, w, h), ...]}."""
    frames: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue
        try:
            frame_id = int(parts[0])
            x   = float(parts[2])
            y   = float(parts[3])
            w   = float(parts[4])
            h   = float(parts[5])
            mc  = int(float(parts[7]))
            vis = float(parts[8])
        except ValueError:
            continue
        if vis < MIN_VISIBILITY or w <= 0 or h <= 0:
            continue
        target = MHPTD_TO_GRTZKY.get(mc)
        if target is None:
            continue
        frames[frame_id].append((target, x, y, w, h))
    return frames


def _boxes_to_yolo(
    boxes: list[tuple[int, float, float, float, float]],
    img_w: int,
    img_h: int,
) -> str:
    lines: list[str] = []
    for cls, x, y, w, h in boxes:
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        if nw <= 0 or nh <= 0:
            continue
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Sequence collection
# ---------------------------------------------------------------------------

def _find_vip_sequences() -> list[tuple[str, Path, Path]]:
    """(tag, img_dir, gt_file) for every VIP-HTD MOT seq."""
    out: list[tuple[str, Path, Path]] = []
    if not VIP_ROOT.exists():
        return out
    for gt in VIP_ROOT.rglob("gt/gt.txt"):
        seq_dir = gt.parent.parent
        img_dir = seq_dir / "img1"
        if img_dir.is_dir():
            out.append((f"vip_htd__{seq_dir.name}", img_dir, gt))
    return out


def _find_mhptd_sequences(frames_cache: Path) -> list[tuple[str, Path, Path]]:
    """(tag, img_dir, gt_file) — extracts frames into `frames_cache` on demand."""
    out: list[tuple[str, Path, Path]] = []
    if not MHPTD_GT_ROOT.exists():
        return out
    for gt in MHPTD_GT_ROOT.rglob("gt.txt"):
        seq_name = gt.parent.name
        mp4 = _find_mp4_for_seq(seq_name)
        if mp4 is None:
            print(f"[fetch-mhptd] no mp4 found for seq {seq_name}")
            continue
        img_dir = frames_cache / seq_name
        if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
            try:
                n = _extract_frames(mp4, img_dir)
                print(f"[fetch-mhptd] extracted {n} frames: {mp4.name} → {img_dir}")
            except Exception as e:  # noqa: BLE001
                print(f"[fetch-mhptd] ffmpeg failed on {mp4}: {e}")
                continue
        out.append((f"mhptd__{seq_name}", img_dir, gt))
    return out


def _strip_duplicate_seqs(seqs: list[tuple[str, Path, Path]]) -> list[tuple[str, Path, Path]]:
    """If MHPTD and VIP-HTD share a seq, prefer MHPTD.

    VIP-HTD is an error-cleaned re-curation but collapses everything to
    class=1 (player). MHPTD keeps the full class labels (home/away player,
    home/away goalie, referee). We need the class labels, so when the same
    raw seq name appears in both, take MHPTD.
    """
    kept: list[tuple[str, Path, Path]] = []
    seen_raw: set[str] = set()
    ordered: list[tuple[str, Path, Path]] = [s for s in seqs if s[0].startswith("mhptd__")]
    ordered += [s for s in seqs if s[0].startswith("vip_htd__")]
    ordered += [s for s in seqs if not (s[0].startswith("vip_htd__") or s[0].startswith("mhptd__"))]
    for tag, img_dir, gt in ordered:
        raw = tag.split("__", 1)[1].lower()
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        kept.append((tag, img_dir, gt))
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=6,
                    help="Take 1 in every N frames per sequence (default 6)")
    ap.add_argument("--max-frames-per-seq", type=int, default=400,
                    help="Cap per sequence (default 400). 0 = no cap.")
    ap.add_argument("--val-split", type=float, default=0.10,
                    help="Fraction held out for validation (by sequence)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--skip-mhptd", action="store_true", help="Only use VIP-HTD")
    ap.add_argument("--skip-vip", action="store_true", help="Only use MHPTD")
    args = ap.parse_args()

    for split in ("train", "val"):
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    frames_cache = EXTERNAL_ROOT / "mhptd_frames_cache"
    frames_cache.mkdir(parents=True, exist_ok=True)

    seqs: list[tuple[str, Path, Path]] = []
    if not args.skip_vip:
        seqs += _find_vip_sequences()
    if not args.skip_mhptd:
        seqs += _find_mhptd_sequences(frames_cache)
    seqs = _strip_duplicate_seqs(seqs)

    if not seqs:
        raise SystemExit(
            f"No usable sequences. Did clones finish?\n"
            f"  VIP_ROOT={VIP_ROOT}\n  MHPTD_GT_ROOT={MHPTD_GT_ROOT}"
        )
    print(f"[fetch-mhptd] {len(seqs)} sequences")
    for tag, img_dir, _ in seqs:
        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.is_dir() else 0
        print(f"  {tag}   ({n_img} frames in img1)")

    rng = random.Random(args.seed)
    rng.shuffle(seqs)
    n_val = max(1, int(len(seqs) * args.val_split))
    val_seqs = {s[0] for s in seqs[:n_val]}

    counts = {"train": 0, "val": 0}
    empty = 0
    for tag, img_dir, gt in tqdm(seqs, desc="sequences"):
        split = "val" if tag in val_seqs else "train"
        per_frame = parse_gt_with_class(gt)
        if not per_frame:
            continue
        frame_ids = sorted(per_frame.keys())
        if args.stride > 1:
            frame_ids = frame_ids[::args.stride]
        if args.max_frames_per_seq > 0 and len(frame_ids) > args.max_frames_per_seq:
            step = len(frame_ids) / args.max_frames_per_seq
            frame_ids = [frame_ids[int(i * step)] for i in range(args.max_frames_per_seq)]

        for fid in frame_ids:
            cands = [
                img_dir / f"{fid:06d}.jpg",
                img_dir / f"{fid:05d}.jpg",
                img_dir / f"{fid:04d}.jpg",
                img_dir / f"{fid}.jpg",
            ]
            src = next((c for c in cands if c.exists()), None)
            if src is None:
                continue
            try:
                with Image.open(src) as im:
                    w, h = im.size
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    dst_img = IMAGES_DIR / split / f"{tag}__{fid:06d}.jpg"
                    im.save(dst_img, "JPEG", quality=args.jpeg_quality)
            except Exception as e:  # noqa: BLE001
                print(f"[fetch-mhptd] skip {src}: {e}")
                continue

            yolo = _boxes_to_yolo(per_frame[fid], w, h)
            if not yolo.strip():
                empty += 1
                yolo = ""
            (LABELS_DIR / split / f"{tag}__{fid:06d}.txt").write_text(yolo)
            counts[split] += 1

    print(
        f"\n──────────────────────────────────────────\n"
        f"  MHPTD + VIP-HTD → GRTZKY YOLO layout complete.\n"
        f"  train: {counts['train']:,} | val: {counts['val']:,}\n"
        f"  empty-label frames: {empty:,}\n"
        f"  images/ → {IMAGES_DIR}\n"
        f"  labels/ → {LABELS_DIR}\n"
        f"──────────────────────────────────────────\n"
    )


if __name__ == "__main__":
    main()
