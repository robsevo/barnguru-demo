"""Phase 16.3 — Jersey OCR model training entry point.

Trains a small CNN to read jersey numbers (1–99) from synthetic crops generated
by ``scripts/generate_jersey_data.py``.

Pipeline::

    1. Verify training data exists (auto-generate if missing and --no-gen not set)
    2. Build DataLoaders from labels.csv
    3. Train JerseyOCRNet with Adam + OneCycleLR scheduler
    4. Evaluate on val split → compute top-1 accuracy
    5. Check quality gate (top-1 ≥ 85%)
    6. Copy best weights to models/cv/jersey_ocr.pt
    7. Save metrics to models/cv/jersey_ocr_metrics.json

Usage::

    uv run python scripts/gretzky.py train-jersey-ocr
    uv run python scripts/gretzky.py train-jersey-ocr -- --epochs 30 --dry-run
    uv run python scripts/gretzky.py train-jersey-ocr -- --no-gen  # skip data generation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.cv.jersey_ocr import (
    INPUT_SIZE,
    JerseyOCRNet,
    METRICS_JSON,
    NUM_CLASSES,
    TOP1_GATE,
    check_quality_gate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR  = _REPO / "data" / "cv_training" / "jersey_crops"
DEFAULT_MODEL_DIR = _REPO / "models" / "cv"
DEFAULT_EPOCHS    = 20
DEFAULT_BATCH     = 64
DEFAULT_LR        = 1e-3

# Browser copy — kept in sync with the .pt so the in-browser overlay can run
# jersey OCR in the main thread via onnxruntime-web.
PUBLIC_ONNX_DIR   = _REPO / "dashboard" / "frontend" / "public" / "cv"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JerseyDataset(Dataset):
    """Loads jersey crop images from labels.csv for a given split."""

    def __init__(self, data_dir: Path, split: str = "train") -> None:
        from PIL import Image as _Image
        self._Image = _Image
        self._data_dir = data_dir
        self._split    = split

        csv_path = data_dir / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

        self._samples: list[tuple[Path, int]] = []
        split_dir = data_dir / split

        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["split"] != split:
                    continue
                img_path = split_dir / row["filename"]
                if img_path.exists():
                    self._samples.append((img_path, int(row["number"])))

        if not self._samples:
            raise ValueError(f"No samples found for split '{split}' in {data_dir}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, number = self._samples[idx]
        img = self._Image.open(img_path).convert("RGB").resize(
            (INPUT_SIZE, INPUT_SIZE), self._Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr  = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        return tensor, number   # number is the class label (1–99)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _verify_data(data_dir: Path) -> None:
    """Raise FileNotFoundError if jersey crop data is missing or empty."""
    csv_path = data_dir / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Jersey training data not found: {csv_path}\n"
            "Run `uv run python scripts/gretzky.py train-jersey-ocr` (without --no-gen) "
            "to generate synthetic data first."
        )
    with open(csv_path) as fh:
        rows = sum(1 for _ in csv.reader(fh)) - 1  # exclude header
    if rows < 100:
        raise ValueError(
            f"Jersey data too small ({rows} samples in {csv_path}). "
            "Re-generate with generate_jersey_data.py."
        )


def _build_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_ds = JerseyDataset(data_dir, split="train")
    val_ds   = JerseyDataset(data_dir, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _train(
    model: JerseyOCRNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    verbose: bool = True,
) -> dict:
    """Train model; return metrics dict with best val top-1 accuracy."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    best_top1 = 0.0
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                preds  = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        top1 = correct / max(total, 1)
        if top1 > best_top1:
            best_top1  = top1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            gate_sym = "✓" if top1 >= TOP1_GATE else "✗"
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"val_top1={top1:.4f} [{gate_sym}]")

    return {"top1_acc": best_top1, "best_state": best_state, "epochs": epochs}


def _evaluate_top1(model: JerseyOCRNet, val_loader: DataLoader, device: str) -> float:
    """Return top-1 accuracy on val_loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            preds  = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Jersey OCR model (Phase 16.3)")
    p.add_argument("--data",    default=str(DEFAULT_DATA_DIR),
                   help="Jersey crop data directory")
    p.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr",      type=float, default=DEFAULT_LR)
    p.add_argument("--device",  default=None,
                   help="'cuda' or 'cpu' (default: auto)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-gen",  action="store_true",
                   help="Skip data generation (fail if data missing)")
    p.add_argument("--count",   type=int, default=50_000,
                   help="Target sample count for generate step")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate data + validate structure; skip training")
    p.add_argument("--force",   action="store_true",
                   help="Overwrite existing model and data")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args   = _parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)

    # ── Step 1: Generate data if missing ──────────────────────────────
    needs_gen = not (data_dir / "labels.csv").exists() or args.force
    if needs_gen:
        if args.no_gen:
            _verify_data(data_dir)   # will raise FileNotFoundError
        else:
            print("[train-jersey-ocr] Generating synthetic jersey crops…")
            from scripts.generate_jersey_data import generate
            generate(data_dir, count_target=args.count, verbose=True)
    else:
        print(f"[train-jersey-ocr] Data found: {data_dir}")

    # ── Step 2: Verify data ────────────────────────────────────────────
    _verify_data(data_dir)

    # ── Dry-run exit ──────────────────────────────────────────────────
    if args.dry_run:
        train_ds = JerseyDataset(data_dir, split="train")
        val_ds   = JerseyDataset(data_dir, split="val")
        print(f"[train-jersey-ocr] DRY-RUN — train={len(train_ds)} val={len(val_ds)} samples")
        print("[train-jersey-ocr] Structure OK — skipping training (--dry-run)")
        return

    # ── Step 3: Build loaders ─────────────────────────────────────────
    train_loader, val_loader = _build_loaders(
        data_dir, args.batch, args.workers,
        pin_memory=(device != "cpu"),
    )
    print(f"[train-jersey-ocr] train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  device={device}")

    # ── Step 4: Train ─────────────────────────────────────────────────
    model = JerseyOCRNet()
    print(f"[train-jersey-ocr] Training for {args.epochs} epochs…")
    result = _train(model, train_loader, val_loader, args.epochs, args.lr, device)

    best_top1  = result["top1_acc"]
    best_state = result["best_state"]

    # ── Step 5: Load best weights ─────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Step 6: Save model ────────────────────────────────────────────
    model_path = DEFAULT_MODEL_DIR / "jersey_ocr.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(model_path))
    print(f"[train-jersey-ocr] Saved model → {model_path}")

    # ── Step 6b: Export to ONNX for onnxruntime-web (main-thread browser OCR).
    # Must match the preprocessing pipeline mirrored in useCvLiveReplay.ts:
    # 48×48 RGB float32, normalized with ImageNet mean/std.
    onnx_path = DEFAULT_MODEL_DIR / "jersey_ocr.onnx"
    model.eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    # dynamo=False keeps the exporter in legacy single-file mode. The new
    # torch.export path splits weights into a .onnx.data sidecar that
    # onnxruntime-web can't resolve from a relative URL.
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["crop"],
        output_names=["logits"],
        dynamic_axes={"crop": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"[train-jersey-ocr] Exported ONNX → {onnx_path}")

    # ── Step 6c: Dynamic INT8 quantization — deployed browser artifact.
    # Same reasoning as the detector: ~4× smaller, minimal accuracy hit.
    from onnxruntime.quantization import quantize_dynamic, QuantType
    onnx_int8_path = DEFAULT_MODEL_DIR / "jersey_ocr.int8.onnx"
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"[train-jersey-ocr] INT8 ONNX → {onnx_int8_path} "
          f"({onnx_int8_path.stat().st_size / 1e6:.1f} MB)")

    # Copy the INT8 variant to the frontend. The FP32 .onnx stays in models/
    # as the training artifact but doesn't ship to browsers anymore.
    PUBLIC_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    public_path = PUBLIC_ONNX_DIR / "jersey_ocr.int8.onnx"
    public_path.write_bytes(onnx_int8_path.read_bytes())
    print(f"[train-jersey-ocr] Published INT8 ONNX → {public_path}")

    # ── Step 7: Save metrics ──────────────────────────────────────────
    metrics = {
        "top1_acc": round(best_top1, 6),
        "epochs":   args.epochs,
        "batch":    args.batch,
        "lr":       args.lr,
        "device":   device,
        "gate":     TOP1_GATE,
    }
    with open(METRICS_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[train-jersey-ocr] Metrics → {METRICS_JSON}")

    # ── Step 8: Quality gate ──────────────────────────────────────────
    passed, top1 = check_quality_gate(metrics)
    if passed:
        print(f"[train-jersey-ocr] ✓ Quality gate PASSED — top-1={top1:.4f} ≥ {TOP1_GATE}")
    else:
        print(
            f"[train-jersey-ocr] ✗ Quality gate FAILED — top-1={top1:.4f} < {TOP1_GATE}\n"
            "  Consider: more epochs (--epochs 40), more data (--count 80000), "
            "or different backbone.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
