"""Phase 16.2 — Kaggle-friendly entrypoint for train_cv_detector.

You don't have a CUDA GPU locally (RX 580 / Polaris is unsupported by modern
PyTorch), so the detector retrain runs on Kaggle's free P100. This script is
the single thing a Kaggle notebook needs to invoke. It detects the Kaggle
input mount, points train_cv_detector at it, and copies outputs to
``/kaggle/working/`` so they survive the session.

──────────────────────────────────────────────────────────────────────
Workflow (one-time)
──────────────────────────────────────────────────────────────────────

1. **Package the dataset locally** (run from repo root):

   .. code-block:: bash

      cd data/cv_training
      tar --exclude='external/*/.git' \\
          --exclude='external/mhptd/clips' \\
          --exclude='external/vip_htd/clips' \\
          --exclude='external/mhptd_frames_cache' \\
          --exclude='__pycache__' \\
          --exclude='raw' \\
          --exclude='aug_*' \\
          --exclude='frames' \\
          --exclude='yolo_split' \\
          --exclude='.bak_full' \\
          --exclude='*.html' \\
          --exclude='audit_report.json' \\
          --exclude='hockeyrink_validation.*' \\
          -czf /tmp/grtzky_cv.tgz \\
          dataset.yaml images/train images/val labels/train labels/val
      ls -lh /tmp/grtzky_cv.tgz   # expect ~1–3 GB after dedupe

2. **Create a private Kaggle dataset** at https://kaggle.com/datasets/new
   - Title: ``grtzky-cv``
   - Visibility: Private
   - Upload ``/tmp/grtzky_cv.tgz`` (Kaggle auto-untars on attach)

3. **Create a notebook** at https://kaggle.com/code/new with:
   - Accelerator: GPU P100 (or T4×2)
   - Dataset: attach ``grtzky-cv``

4. **Paste these two cells** into the notebook:

   .. code-block:: python

      # Cell 1 — install
      !pip install -q ultralytics onnxruntime onnx
      !git clone --depth 1 https://github.com/robsevo/grtzky.git /kaggle/working/repo
      %cd /kaggle/working/repo

   .. code-block:: python

      # Cell 2 — train (P100: ~3–4 hr at default settings)
      !python scripts/train_cv_kaggle.py

5. **Download the artifacts** from ``/kaggle/working/out/``:
   - ``player_detector.pt``           — source weights for retrain warm-start
   - ``player_detector.onnx``         — FP32 ONNX (for export to v5.fp16)
   - ``player_detector.int8.onnx``    — browser WASM artifact
   - ``detector_metrics.json``        — gate report (paste this in the PR)

   Drop them into ``models/cv/`` locally and commit.

──────────────────────────────────────────────────────────────────────
Reproducibility note
──────────────────────────────────────────────────────────────────────

Kaggle sessions are ephemeral. Make sure the dataset version on
kaggle.com/datasets matches the dedupe state you trained on locally —
re-uploading after dedupe is the source of truth, not the repo
checkout.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# Kaggle-specific layout. ``KAGGLE_DATASET_NAME`` mirrors step 2 above —
# rename if you call your dataset something else.
KAGGLE_INPUT_ROOT  = Path("/kaggle/input")
KAGGLE_DATASET_DIR = KAGGLE_INPUT_ROOT / os.environ.get("KAGGLE_DATASET_NAME", "grtzky-cv")
KAGGLE_OUT         = Path("/kaggle/working/out")


def _on_kaggle() -> bool:
    return KAGGLE_INPUT_ROOT.is_dir() and Path("/kaggle/working").is_dir()


def main() -> None:
    if not _on_kaggle():
        print("[kaggle] not running on Kaggle — refusing to proceed. Use this "
              "script inside a Kaggle notebook (see module docstring).",
              file=sys.stderr)
        sys.exit(2)

    if not KAGGLE_DATASET_DIR.is_dir():
        print(f"[kaggle] dataset not found at {KAGGLE_DATASET_DIR}.\n"
              f"        Set KAGGLE_DATASET_NAME env var or attach the dataset.",
              file=sys.stderr)
        sys.exit(2)

    KAGGLE_OUT.mkdir(parents=True, exist_ok=True)

    # train_cv_detector saves the .pt/.onnx into the repo's models/cv/.
    # We copy them out to /kaggle/working/out at the end so they survive
    # the session.
    from scripts.train_cv_detector import main as train_main
    from models.cv.player_detector import (
        DEFAULT_PT, DEFAULT_ONNX, DEFAULT_ONNX_INT8, METRICS_JSON,
    )

    train_main([
        "--data-root", str(KAGGLE_DATASET_DIR),
        "--device",    "0",        # P100 / T4 — ultralytics picks CUDA:0
        "--backbone",  "yolov8m.pt",
        "--epochs",    "100",
        "--imgsz",     "480",
        "--batch",     "16",
        "--patience",  "20",
        "--mixup",     "0.1",
        "--mosaic",    "1.0",
    ])

    for src in (DEFAULT_PT, DEFAULT_ONNX, DEFAULT_ONNX_INT8, METRICS_JSON):
        if Path(src).exists():
            dst = KAGGLE_OUT / Path(src).name
            shutil.copy2(src, dst)
            print(f"[kaggle] copied {src} → {dst}")

    print(f"\n[kaggle] artifacts at {KAGGLE_OUT}")


if __name__ == "__main__":
    main()
