"""Phase 16.2 — Train in-house player detection model.

Fine-tunes a YOLO backbone on the 16.1 hockey tracking dataset.

Pipeline:
    1. Verify 16.1 data exists (dataset.yaml + aug_images/ + aug_labels/)
    2. Split sequences into train/val (80/20 by sequence, no frame leakage)
    3. Write Ultralytics-compatible YOLO directory layout (symlinks)
    4. Fine-tune YOLOv8n pretrained on COCO → our 6 hockey classes
    5. Evaluate on validation set → check mAP@0.5 ≥ 0.82
    6. Copy best weights → models/cv/player_detector.pt
    7. Save metrics → models/cv/detector_metrics.json

Quality gate (PLAN.md §16.2):
    mAP@0.5 ≥ 0.82 — hard warning, not a hard failure.
    If gate fails, model is still saved with a WARNING in the metrics JSON
    so the Whiz knows the model needs more data / tuning before shipping.

Usage::

    uv run python scripts/gretzky.py train-cv-detector
    uv run python scripts/gretzky.py train-cv-detector -- --epochs 100
    uv run python scripts/gretzky.py train-cv-detector -- --backbone yolov8s.pt
    uv run python scripts/gretzky.py train-cv-detector -- --no-aug  (use raw frames, not augmented)
    uv run python scripts/gretzky.py train-cv-detector -- --dry-run (validate data layout only)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

from models.cv.player_detector import (
    CLASS_NAMES, MAP50_GATE, MAP50_DOWNSTREAM_GATE, NUM_CLASSES, METRICS_JSON,
    DEFAULT_PT, DEFAULT_ONNX, DEFAULT_ONNX_INT8, RUNTIME_IMGSZ,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CV         = _REPO / "data" / "cv_training"
RAW_DIR     = _CV / "raw"
AUG_IMG     = _CV / "aug_images"
AUG_LBL     = _CV / "aug_labels"
FRAMES_DIR  = _CV / "frames"
LABELS_DIR  = _CV / "labels"
DATASET_YAML = _CV / "dataset.yaml"
SPLIT_DIR   = _CV / "yolo_split"
RUNS_DIR    = _REPO / "runs" / "cv_detector"
MODEL_DIR   = _REPO / "models" / "cv"

# Default YOLO backbone. yolov8m is the smallest model that has cleared the
# 0.92 downstream gate on broadcast hockey footage in our internal runs;
# yolov8s topped out near 0.81 on the current 8k-image dataset. The browser
# artifact is the INT8-quantized v4 ONNX (~3 MB), exported separately, so
# bumping the source backbone doesn't bloat the mobile payload.
DEFAULT_BACKBONE = "yolov8m.pt"

# Train/val split ratio (fraction of sequences for training)
TRAIN_RATIO = 0.80


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    print(_banner())

    # ── Optional data-root override (Kaggle, CI, alt layouts) ────────────
    # Keep _CV / DATASET_YAML / paths above as the in-repo defaults; only
    # override the local handles used in this function so module-level
    # imports elsewhere keep working.
    cv_root = Path(args.data_root).resolve() if args.data_root else _CV
    dataset_yaml = cv_root / "dataset.yaml"
    flat_train_imgs = cv_root / "images" / "train"
    flat_val_imgs   = cv_root / "images" / "val"
    if dataset_yaml.exists() and flat_train_imgs.is_dir() and flat_val_imgs.is_dir():
        n_train = len(list(flat_train_imgs.glob("*.jpg")))
        n_val   = len(list(flat_val_imgs.glob("*.jpg")))
        print(f"[detector] Using dataset.yaml at {dataset_yaml} "
              f"({n_train:,} train / {n_val:,} val images)")
        if args.data_root:
            # Rewrite the absolute `path:` field so Ultralytics resolves
            # train/val under the override root. Lives in the run dir so it
            # doesn't pollute the repo dataset.yaml.
            split_yaml = _rewrite_dataset_yaml(dataset_yaml, cv_root, RUNS_DIR)
        else:
            split_yaml = dataset_yaml
    else:
        # ── Legacy path: MOT-sequence layout from build_cv_dataset.py ────
        img_root = AUG_IMG if not args.no_aug else FRAMES_DIR
        lbl_root = AUG_LBL if not args.no_aug else LABELS_DIR

        try:
            _verify_data(img_root, lbl_root)
        except FileNotFoundError as exc:
            print(f"\n[detector] ✗ Pre-flight check failed:\n  {exc}\n")
            sys.exit(0)
        print(f"[detector] Data source: {'augmented' if not args.no_aug else 'raw frames'}")

        split_yaml = _prepare_split(img_root, lbl_root, SPLIT_DIR, force=args.force)
        print(f"[detector] Split YAML: {split_yaml}")

    if args.dry_run:
        print("[detector] --dry-run: data layout OK. Exiting without training.")
        return

    # ── Step 3: Fine-tune YOLO ────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[detector] Fine-tuning {args.backbone} → {NUM_CLASSES} hockey classes")
    print(f"[detector] Epochs={args.epochs}  Batch={args.batch}  ImgSz={args.imgsz}")
    print(f"[detector] Device={args.device or 'auto'}")
    print()

    results = _train(
        backbone=args.backbone,
        data_yaml=str(split_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        runs_dir=RUNS_DIR,
        project_name=args.run_name,
        lr0=args.lr,
        patience=args.patience,
        freeze=args.freeze,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        mosaic=args.mosaic,
    )

    # ── Step 4: Copy best weights → models/cv/player_detector.pt ─────────
    best_pt = _find_best_weights(results)
    shutil.copy2(best_pt, DEFAULT_PT)
    print(f"\n[detector] ✓ Model saved → {DEFAULT_PT}")

    # ── Step 4b: Export ONNX at RUNTIME_IMGSZ (matches browser cvWorker) ─
    _export_onnx(best_pt, DEFAULT_ONNX, imgsz=RUNTIME_IMGSZ, half=args.onnx_half)

    # ── Step 4c: Dynamic INT8 quantization — the deployed browser artifact.
    # Cuts model size ~4× (12 MB → ~3 MB) so every cold device load is cheap.
    _quantize_onnx_int8(DEFAULT_ONNX, DEFAULT_ONNX_INT8)

    # ── Step 5: Extract metrics + quality gate ────────────────────────────
    metrics = _extract_metrics(results)
    map50   = metrics.get("map50", 0.0)
    passed  = map50 >= MAP50_GATE

    downstream_passed = map50 >= MAP50_DOWNSTREAM_GATE
    metrics["quality_gate"] = {
        "threshold": MAP50_GATE,
        "passed":    passed,
        "map50":     map50,
    }
    metrics["downstream_gate"] = {
        "threshold": MAP50_DOWNSTREAM_GATE,
        "passed":    downstream_passed,
        "map50":     map50,
    }
    metrics["model_path"] = str(DEFAULT_PT)
    metrics["backbone"]   = args.backbone
    metrics["epochs"]     = args.epochs

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[detector] Metrics saved → {METRICS_JSON}")

    # ── Summary ───────────────────────────────────────────────────────────
    gate_sym       = "✓" if passed            else "✗"
    gate_msg       = "PASS" if passed         else f"FAIL (need ≥{MAP50_GATE})"
    downstream_sym = "✓" if downstream_passed else "✗"
    downstream_msg = ("PASS — cv_gate safe to flip ON" if downstream_passed
                      else f"FAIL (need ≥{MAP50_DOWNSTREAM_GATE} for CV→player-NN feed)")

    print(
        f"\n{'─'*55}\n"
        f"  Phase 16.2 complete.\n"
        f"  mAP@0.5    : {map50:.4f}  {gate_sym} {gate_msg}\n"
        f"  Downstream : {map50:.4f}  {downstream_sym} {downstream_msg}\n"
        f"  mAP@0.5:0.95: {metrics.get('map50_95', 0.0):.4f}\n"
        f"  Precision  : {metrics.get('precision', 0.0):.4f}\n"
        f"  Recall     : {metrics.get('recall', 0.0):.4f}\n"
        f"  Model      : {DEFAULT_PT}\n"
    )

    if not passed:
        print(
            f"  ⚠  mAP@0.5 {map50:.4f} < {MAP50_GATE} gate.\n"
            f"     Options to improve:\n"
            f"     • Run audit-cv with --delete-dupes to remove near-duplicates\n"
            f"     • Re-run fetch_mhptd_vip.py with smaller --stride for more frames\n"
            f"     • Try --backbone yolov8m.pt or --backbone yolov8l.pt\n"
            f"     • Increase --epochs (100+ at imgsz 480)\n"
            f"     • Bootstrap pseudo-labels: gretzky bootstrap-pseudo\n"
        )
    elif not downstream_passed:
        print(
            f"  ⚠  Detector passes display gate but not the 0.92 downstream gate.\n"
            f"     Boxes will render in /game/[id], but cv_gate.json should stay\n"
            f"     OFF until mAP@0.5 ≥ {MAP50_DOWNSTREAM_GATE}. Options:\n"
            f"     • Bigger backbone (yolov8l.pt) + more epochs\n"
            f"     • Add more labelled sequences via fetch-* scripts\n"
            f"     • Bootstrap pseudo-labels and retrain\n"
        )
    else:
        print(f"  ✓ Both gates passed. Ready for 16.7 (CV worker) and CV→NN feed.\n")

    print(f"{'─'*55}\n")

    if not passed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# data-root override helper
# ---------------------------------------------------------------------------

def _rewrite_dataset_yaml(src: Path, cv_root: Path, runs_dir: Path) -> Path:
    """Copy ``src`` and rewrite the absolute ``path:`` field to ``cv_root``.

    Ultralytics resolves train/val relative to the YAML's path field, so a
    repo-checked-in dataset.yaml doesn't work when the data is mounted at
    ``/kaggle/input/grtzky-cv``. We write a sibling YAML next to the run
    output rather than mutating the source.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    dst = runs_dir / "dataset_kaggle.yaml"
    out_lines: list[str] = []
    saw_path = False
    for raw in src.read_text().splitlines():
        if raw.lstrip().startswith("path:"):
            out_lines.append(f"path: {cv_root.resolve()}")
            saw_path = True
        else:
            out_lines.append(raw)
    if not saw_path:
        out_lines.insert(0, f"path: {cv_root.resolve()}")
    dst.write_text("\n".join(out_lines) + "\n")
    print(f"[detector] Rewrote dataset.yaml → {dst}  (path: {cv_root})")
    return dst


# ---------------------------------------------------------------------------
# Data verification
# ---------------------------------------------------------------------------

def _verify_data(img_root: Path, lbl_root: Path) -> None:
    if not img_root.exists():
        raise FileNotFoundError(
            f"Image directory not found: {img_root}\n"
            "Run `uv run python scripts/gretzky.py build-cv-dataset` first (Feature 16.1)."
        )
    if not lbl_root.exists():
        raise FileNotFoundError(
            f"Label directory not found: {lbl_root}\n"
            "Run `uv run python scripts/gretzky.py build-cv-dataset` first (Feature 16.1)."
        )

    seqs = [d for d in img_root.iterdir() if d.is_dir()]
    if not seqs:
        raise FileNotFoundError(
            f"No sequence directories found under {img_root}.\n"
            "Run `uv run python scripts/gretzky.py build-cv-dataset` first."
        )

    total_imgs = sum(len(list(s.glob("*.jpg"))) for s in seqs)
    total_lbls = sum(len(list((lbl_root / s.name).glob("*.txt"))) for s in seqs if (lbl_root / s.name).exists())

    print(f"[detector] Found {len(seqs)} sequences, {total_imgs:,} images, {total_lbls:,} labels.")

    if total_imgs == 0:
        raise ValueError("No images found. Re-run build-cv-dataset.")
    if total_lbls == 0:
        raise ValueError("No label files found. Re-run build-cv-dataset.")


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def _prepare_split(
    img_root: Path,
    lbl_root: Path,
    split_dir: Path,
    force: bool = False,
) -> Path:
    """Create Ultralytics-compatible directory layout via symlinks.

    Layout:
        yolo_split/
            images/
                train/   ← symlinks to aug_images/<seq>/*.jpg  (80% seqs)
                val/     ← symlinks to aug_images/<seq>/*.jpg  (20% seqs)
            labels/
                train/   ← symlinks to aug_labels/<seq>/*.txt
                val/     ← symlinks to aug_labels/<seq>/*.txt
            dataset_split.yaml

    Sequences are split by sequence name (no frame-level leakage).
    """
    yaml_path = split_dir / "dataset_split.yaml"
    if yaml_path.exists() and not force:
        print(f"[detector] Reusing existing split at {split_dir}")
        return yaml_path

    # Collect sequences present in both img and lbl roots
    seqs = sorted(
        s.name for s in img_root.iterdir()
        if s.is_dir() and (lbl_root / s.name).is_dir()
    )
    if not seqs:
        raise FileNotFoundError(f"No matching sequence dirs in {img_root} and {lbl_root}.")

    n_train = max(1, int(len(seqs) * TRAIN_RATIO))
    train_seqs = seqs[:n_train]
    val_seqs   = seqs[n_train:] or seqs[-1:]   # always at least 1 val seq

    print(f"[detector] Split: {len(train_seqs)} train seqs, {len(val_seqs)} val seqs.")

    for split, seq_list in (("train", train_seqs), ("val", val_seqs)):
        img_dir = split_dir / "images" / split
        lbl_dir = split_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for seq in seq_list:
            for img_path in sorted((img_root / seq).glob("*.jpg")):
                link = img_dir / f"{seq}__{img_path.name}"
                if link.exists() or link.is_symlink():
                    link.unlink()
                try:
                    link.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, link)

            for lbl_path in sorted((lbl_root / seq).glob("*.txt")):
                link = lbl_dir / f"{seq}__{lbl_path.name}"
                if link.exists() or link.is_symlink():
                    link.unlink()
                try:
                    link.symlink_to(lbl_path.resolve())
                except OSError:
                    shutil.copy2(lbl_path, link)

    # Write Ultralytics YAML
    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASS_NAMES))
    content = (
        "# GRTZKY Phase 16.2 — YOLO Training Split\n"
        "# Auto-generated by train_cv_detector.py\n"
        "\n"
        f"path: {split_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "\n"
        f"nc: {NUM_CLASSES}\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(content)
    return yaml_path


# ---------------------------------------------------------------------------
# YOLO training
# ---------------------------------------------------------------------------

def _train(
    backbone: str,
    data_yaml: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str | None,
    runs_dir: Path,
    project_name: str,
    lr0: float,
    patience: int,
    freeze: int = 10,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    mosaic: float = 1.0,
):
    """Fine-tune YOLO and return the Ultralytics Results object."""
    from ultralytics import YOLO
    import torch

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[detector] Using device: {selected_device}")

    model = YOLO(backbone)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=selected_device,
        project=str(runs_dir),
        name=project_name,
        exist_ok=True,
        # Backbone freeze (0 = fully fine-tune). First N layers frozen.
        freeze=freeze,
        # Learning rate
        lr0=lr0,
        lrf=0.01,
        # Early stopping
        patience=patience,
        # Training-time augmentation
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=10,
        # Dataloader workers — CPU training benefits from parallel image decode
        workers=8,
        # Verbosity
        verbose=True,
    )
    return results


# ---------------------------------------------------------------------------
# Post-training helpers
# ---------------------------------------------------------------------------

def _find_best_weights(results) -> Path:
    """Locate best.pt from the training run."""
    # Ultralytics stores weights at: <project>/<name>/weights/best.pt
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        return best
    # Fallback: last.pt
    last = save_dir / "weights" / "last.pt"
    if last.exists():
        print("[detector] WARNING: best.pt not found, using last.pt")
        return last
    raise FileNotFoundError(f"No weights found in {save_dir / 'weights'}")


def _export_onnx(pt_path: Path, dest: Path, imgsz: int, half: bool) -> None:
    """Export .pt → .onnx at the same size the browser runtime uses.

    Never call this with half=True when the output feeds _quantize_onnx_int8.
    FP16 casts in the source graph survive dynamic quantization and blow up
    in onnxruntime-web with "Type 'tensor(float16)' ... is invalid" on
    DynamicQuantizeLinear. FP32 source → INT8 quant is the only safe path
    for the browser artifact.
    """
    from ultralytics import YOLO

    if half:
        raise ValueError(
            "Refusing to export FP16 ONNX — the INT8 quant step downstream "
            "cannot consume FP16 inputs. Re-run without --onnx-half."
        )
    print(f"[detector] Exporting ONNX (imgsz={imgsz}, half=False)")
    model = YOLO(str(pt_path))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=True,
        opset=12,
        dynamic=False,
    )
    src = Path(exported) if isinstance(exported, (str, Path)) else pt_path.with_suffix(".onnx")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    print(f"[detector] ✓ ONNX saved → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def _quantize_onnx_int8(src: Path, dest: Path) -> None:
    """Dynamic INT8 quantize ``src`` → ``dest``.

    Dynamic quant is chosen over static because it needs no calibration set
    and the accuracy drop on detector-style nets is typically <1 mAP50 in
    our size range — well inside the quality gate. If that ever stops being
    true, switch to static quant with a calibration reader over val frames.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    dest.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dest),
        weight_type=QuantType.QInt8,
    )
    print(f"[detector] ✓ INT8 ONNX → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def _extract_metrics(results) -> dict:
    """Extract mAP and other metrics from Ultralytics results."""
    metrics: dict = {}
    try:
        # Ultralytics stores validation metrics in results.results_dict
        rd = results.results_dict
        metrics["map50"]     = float(rd.get("metrics/mAP50(B)",    rd.get("mAP50", 0.0)))
        metrics["map50_95"]  = float(rd.get("metrics/mAP50-95(B)", rd.get("mAP50-95", 0.0)))
        metrics["precision"] = float(rd.get("metrics/precision(B)", rd.get("precision", 0.0)))
        metrics["recall"]    = float(rd.get("metrics/recall(B)",    rd.get("recall", 0.0)))
    except Exception as e:
        print(f"[detector] Warning: could not extract metrics from results: {e}")
        # Try to load from CSV if available
        metrics = _metrics_from_csv(Path(results.save_dir))
    return metrics


def _metrics_from_csv(save_dir: Path) -> dict:
    """Fallback: parse results.csv written by Ultralytics."""
    csv = save_dir / "results.csv"
    if not csv.exists():
        return {}
    import csv as csv_mod
    rows = []
    with open(csv) as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    # Last row = final epoch metrics; strip whitespace from keys
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}
    return {
        "map50":     float(last.get("metrics/mAP50(B)",    0.0)),
        "map50_95":  float(last.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(last.get("metrics/precision(B)", 0.0)),
        "recall":    float(last.get("metrics/recall(B)",   0.0)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 16.2 — Train in-house player detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backbone",  default=DEFAULT_BACKBONE,
                   help=f"YOLO backbone .pt file (default: {DEFAULT_BACKBONE})")
    p.add_argument("--epochs",    type=int,   default=100,
                   help="Training epochs (default 100; 0.92 gate typically lands 80–120)")
    p.add_argument("--batch",     type=int,   default=16,
                   help="Batch size (default 16; reduce if OOM)")
    p.add_argument("--imgsz",     type=int,   default=480,
                   help="Training image size in pixels (default 480 — close to "
                        "broadcast aspect after letterbox; bumping to 640 helps "
                        "puck mAP at 2× compute)")
    p.add_argument("--lr",        type=float, default=0.01,
                   help="Initial learning rate (default 0.01)")
    p.add_argument("--patience",  type=int,   default=15,
                   help="Early stopping patience epochs (default 15)")
    p.add_argument("--device",    default=None,
                   help="Device: 'cuda', 'cpu', '0', etc. (default: auto)")
    p.add_argument("--run-name",  default="train",
                   help="Subdirectory name under runs/cv_detector/")
    p.add_argument("--no-aug",    action="store_true",
                   help="Use raw frames instead of augmented images")
    p.add_argument("--force",     action="store_true",
                   help="Re-create train/val split even if it exists")
    p.add_argument("--dry-run",   action="store_true",
                   help="Validate data layout only; skip training")
    p.add_argument("--onnx-half", action="store_true",
                   help="Export ONNX in FP16 (smaller + faster CPU WASM).")
    p.add_argument("--freeze",    type=int,   default=10,
                   help="Freeze first N backbone layers (0 = fully fine-tune).")
    p.add_argument("--mixup",     type=float, default=0.0,
                   help="Mixup augmentation probability (0–1).")
    p.add_argument("--copy-paste", type=float, default=0.0,
                   help="Copy-paste augmentation probability (0–1).")
    p.add_argument("--mosaic",    type=float, default=1.0,
                   help="Mosaic augmentation probability (default 1.0).")
    p.add_argument("--data-root", default=None,
                   help="Override the cv_training root (e.g. /kaggle/input/grtzky-cv "
                        "for Kaggle). When set, dataset.yaml + images/ + labels/ are "
                        "read from this directory. Outputs (.pt, .onnx, metrics) "
                        "still go under the repo's models/cv/.")
    return p.parse_args(argv)


def _banner() -> str:
    return (
        "\n"
        "╔══════════════════════════════════════════════════╗\n"
        "║  GRTZKY  ·  Phase 16.2 — Player Detector       ║\n"
        "║  Fine-tuning YOLO on Hockey Tracking Data        ║\n"
        "╚══════════════════════════════════════════════════╝\n"
    )


if __name__ == "__main__":
    main()
