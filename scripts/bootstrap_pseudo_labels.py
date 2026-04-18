"""Phase 16 — bootstrap pseudo-labels from recorded broadcast video.

Problem
-------
``retrain_cv_detector.py`` expects YOLO pseudo-labels under
``data/cv_training/pseudo_labels/<YYYY>-W<WW>/images+labels/``, produced at
runtime by ``PuckPseudoCache`` whenever the live CV worker sees a
high-confidence puck. That works long-term but can't bootstrap from zero —
no live tracking has produced enough labels yet (3 files as of writing).

This script fills that gap. It takes a recorded broadcast video, samples
frames at ``--fps``, runs the current detector, and writes confident
detections as YOLO training labels in the exact layout
``retrain_cv_detector.py`` consumes.

Self-distillation caveat
------------------------
Self-distillation works if the teacher model's confident predictions are
usually correct. Our detector's confident-but-wrong modes (ads on boards,
crowd) are filtered here the same way they are at runtime:
    - Y-band reject (top 12% + bottom 5% of frame)
    - Min confidence 0.85 per bbox
    - At least 3 kept bboxes per frame (otherwise the frame is near-empty
      or wrong scene — not useful training signal)

Usage
-----
    uv run python scripts/gretzky.py bootstrap-pseudo -- \\
        --video data/cv_training/raw/stl_uta.mp4 --fps 2 --max 500

    # All videos in data/cv_training/raw/
    uv run python scripts/gretzky.py bootstrap-pseudo -- --all

After running, check ``data/cv_training/pseudo_labels/<YYYY>-W<WW>/`` and
proceed with ``gretzky retrain-cv`` to train on the combined base+pseudo
dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _iso_week_dir(root: Path, ts: float | None = None) -> tuple[Path, Path]:
    if ts is None:
        ts = datetime.now(timezone.utc).timestamp()
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isocalendar()
    wk  = f"{iso[0]:04d}-W{iso[1]:02d}"
    img = root / wk / "images"
    lbl = root / wk / "labels"
    img.mkdir(parents=True, exist_ok=True)
    lbl.mkdir(parents=True, exist_ok=True)
    return img, lbl


def _iter_frames(video_path: Path, target_fps: float):
    """Yield (frame_idx_in_source, bgr_frame) from the video at ``target_fps``."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride  = max(1, int(round(src_fps / max(0.1, target_fps))))
        frame_n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_n % stride == 0:
                yield frame_n, frame
            frame_n += 1
    finally:
        cap.release()


def _filter_and_write(
    detector,
    frame,
    frame_n: int,
    source_tag: str,
    img_dir: Path,
    lbl_dir: Path,
    *,
    min_conf:  float,
    min_boxes: int,
    y_top:     float,
    y_bottom:  float,
) -> bool:
    """Run detector, filter, write frame + label file. Return True if written."""
    import cv2

    H, W = frame.shape[:2]
    try:
        detections = detector.detect(frame)
    except Exception:
        return False

    lines: list[str] = []
    for d in detections:
        if d.confidence < min_conf:
            continue
        cy = (d.y1 + d.y2) / 2 / H
        # Class 5 (puck) is small and can appear anywhere; don't Y-band it.
        if d.class_id != 5 and (cy < y_top or cy > y_bottom):
            continue
        cx = (d.x1 + d.x2) / 2 / W
        w  = (d.x2 - d.x1) / W
        h  = (d.y2 - d.y1) / H
        if w <= 0 or h <= 0:
            continue
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))
        lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if len(lines) < min_boxes:
        return False

    stem = f"{source_tag}_f{frame_n:06d}"
    img_path = img_dir / f"{stem}.jpg"
    lbl_path = lbl_dir / f"{stem}.txt"

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return False
    tmp_img = img_path.with_name(img_path.name + ".tmp")
    tmp_img.write_bytes(buf.tobytes())
    os.replace(tmp_img, img_path)

    tmp_lbl = lbl_path.with_suffix(".txt.tmp")
    tmp_lbl.write_text("\n".join(lines) + "\n")
    os.replace(tmp_lbl, lbl_path)
    return True


def _process_one(
    video_path: Path,
    detector,
    *,
    root:       Path,
    fps:        float,
    max_frames: int,
    min_conf:   float,
    min_boxes:  int,
    y_top:      float,
    y_bottom:   float,
) -> int:
    img_dir, lbl_dir = _iso_week_dir(root)
    tag = video_path.stem
    written = 0
    scanned = 0
    for frame_n, frame in _iter_frames(video_path, fps):
        scanned += 1
        if _filter_and_write(
            detector, frame, frame_n, tag, img_dir, lbl_dir,
            min_conf=min_conf, min_boxes=min_boxes,
            y_top=y_top, y_bottom=y_bottom,
        ):
            written += 1
            if written >= max_frames:
                break
    print(f"  [{tag}] scanned {scanned} sampled frames → {written} written")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None,
                        help="Single video file (default: use --all)")
    parser.add_argument("--all", action="store_true",
                        help="Process every *.mp4 in data/cv_training/raw/")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Sampling rate in frames per second (default 2.0)")
    parser.add_argument("--max", dest="max_frames", type=int, default=500,
                        help="Max pseudo-label frames per video (default 500)")
    parser.add_argument("--min-conf", type=float, default=0.85,
                        help="Minimum detection confidence to keep (default 0.85)")
    parser.add_argument("--min-boxes", type=int, default=3,
                        help="Min confident bboxes per frame to keep (default 3)")
    parser.add_argument("--y-top", type=float, default=0.12,
                        help="Reject bboxes with cy above this (default 0.12)")
    parser.add_argument("--y-bottom", type=float, default=0.95,
                        help="Reject bboxes with cy below this (default 0.95)")
    args = parser.parse_args(argv)

    # Resolve sources
    sources: list[Path]
    if args.video:
        sources = [args.video]
    elif args.all:
        raw = _REPO / "data" / "cv_training" / "raw"
        sources = sorted(raw.glob("*.mp4"))
    else:
        parser.error("Pass --video PATH or --all")
        return 2

    if not sources:
        print("No video sources found.")
        return 1

    # Load detector once; re-use across videos.
    from models.cv.player_detector import PlayerDetector
    print("[bootstrap-pseudo] loading detector…")
    detector = PlayerDetector()

    root = _REPO / "data" / "cv_training" / "pseudo_labels"
    print(f"[bootstrap-pseudo] writing into {root}")
    total = 0
    for v in sources:
        if not v.exists():
            print(f"  [skip] {v} not found")
            continue
        total += _process_one(
            v, detector,
            root=root, fps=args.fps, max_frames=args.max_frames,
            min_conf=args.min_conf, min_boxes=args.min_boxes,
            y_top=args.y_top, y_bottom=args.y_bottom,
        )

    print(f"\n[bootstrap-pseudo] done — {total} pseudo-labels written total.")
    if total == 0:
        print(" No frames passed filters. Try lowering --min-conf or --min-boxes.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
