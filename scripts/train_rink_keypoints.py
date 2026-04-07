"""Phase 16.5 — Rink keypoint detector training entry point.

Trains a heatmap regression CNN to detect 7 rink landmarks in broadcast frames.

Pipeline::

    1. Check for annotation data at data/cv_training/keypoints/annotations.json
       (or pass --synth to generate synthetic frames for smoke-testing)
    2. Build DataLoaders from annotated frames + Gaussian heatmaps
    3. Train RinkKeypointNet with Adam + MSE loss on heatmaps
    4. Evaluate: compute PCK@10 (percentage of keypoints within 10px)
    5. Check quality gate (PCK@10 ≥ 80%)
    6. Save best weights to models/cv/rink_keypoint_detector.pt
    7. Save metrics to models/cv/rink_keypoint_detector_metrics.json

Annotation data format::

    data/cv_training/keypoints/
    ├── annotations.json
    └── frames/
        ├── clip001_frame0042.jpg
        └── ...

    annotations.json schema:
    {
      "version": "1.0",
      "annotations": [
        {
          "frame_path": "data/cv_training/keypoints/frames/clip001_frame0042.jpg",
          "arena": "Bell Centre",
          "image_w": 1280,
          "image_h": 720,
          "keypoints": {
            "center_ice_dot":     {"x": 640.0, "y": 360.0, "visible": true},
            "blue_line_left":     {"x": 480.0, "y": 350.0, "visible": true},
            "blue_line_right":    {"x": 800.0, "y": 350.0, "visible": true},
            "faceoff_home_left":  {"x": 320.0, "y": 280.0, "visible": true},
            "faceoff_home_right": {"x": 320.0, "y": 440.0, "visible": true},
            "faceoff_away_left":  {"x": 960.0, "y": 280.0, "visible": false},
            "faceoff_away_right": {"x": 960.0, "y": 440.0, "visible": false}
          }
        }
      ]
    }

    Recommend ~200 manually annotated frames extracted from 16.1 clips.
    Use any annotation tool that exports pixel coordinates (CVAT, labelme, etc.)
    and convert output to the schema above.

Usage::

    uv run python scripts/gretzky.py train-rink-kp
    uv run python scripts/gretzky.py train-rink-kp -- --epochs 30
    uv run python scripts/gretzky.py train-rink-kp -- --synth   # synthetic smoke-test
    uv run python scripts/gretzky.py train-rink-kp -- --dry-run
"""

from __future__ import annotations

import argparse
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

from models.cv.rink_keypoint_detector import (
    CONF_THR,
    HMAP_H,
    HMAP_W,
    INPUT_H,
    INPUT_W,
    KEYPOINT_NAMES,
    METRICS_JSON,
    NUM_KEYPOINTS,
    PCK_GATE,
    PCK_RADIUS,
    RinkKeypointNet,
    check_quality_gate,
    _MEAN,
    _STD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNOTATIONS_PATH = _REPO / "data" / "cv_training" / "keypoints" / "annotations.json"
DEFAULT_MODEL_OUT = _REPO / "models" / "cv" / "rink_keypoint_detector.pt"
DEFAULT_EPOCHS    = 30
DEFAULT_BATCH     = 16
DEFAULT_LR        = 3e-4
HMAP_SIGMA        = 2.5     # Gaussian sigma in heatmap pixels
VAL_SPLIT         = 0.15    # fraction of annotations reserved for validation
SYNTH_COUNT       = 100     # synthetic frames for --synth smoke-test


# ---------------------------------------------------------------------------
# Gaussian heatmap rendering
# ---------------------------------------------------------------------------

def _render_gaussian(
    center_xy: tuple[float, float],
    hmap_h: int,
    hmap_w: int,
    sigma: float = HMAP_SIGMA,
) -> np.ndarray:
    """Return a (hmap_h, hmap_w) float32 heatmap with a Gaussian at center_xy.

    center_xy is expressed in heatmap-space pixels.
    """
    cx, cy = center_xy
    xs = np.arange(hmap_w, dtype=np.float32)
    ys = np.arange(hmap_h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    hm = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


def _build_heatmaps(
    annotation: dict,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """Build (NUM_KP, HMAP_H, HMAP_W) target heatmaps from a single annotation.

    Keypoints marked visible=False produce all-zero heatmaps.
    """
    scale_x = HMAP_W / orig_w
    scale_y = HMAP_H / orig_h

    heatmaps = np.zeros((NUM_KEYPOINTS, HMAP_H, HMAP_W), dtype=np.float32)
    kps = annotation.get("keypoints", {})
    for i, name in enumerate(KEYPOINT_NAMES):
        kp = kps.get(name)
        if kp is None or not kp.get("visible", True):
            continue
        hx = float(kp["x"]) * scale_x
        hy = float(kp["y"]) * scale_y
        heatmaps[i] = _render_gaussian((hx, hy), HMAP_H, HMAP_W)
    return heatmaps


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RinkKeypointDataset(Dataset):
    """Loads annotated broadcast frames and builds Gaussian heatmap targets."""

    def __init__(self, annotations: list[dict], augment: bool = False) -> None:
        self._annotations = annotations
        self._augment = augment

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2
        ann = self._annotations[idx]
        frame_path = _REPO / ann["frame_path"]
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

        orig_h, orig_w = bgr.shape[:2]
        heatmaps = _build_heatmaps(ann, orig_h, orig_w)

        # Resize to model input
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

        if self._augment:
            resized = _augment_frame(resized)

        arr = resized.astype(np.float32) / 255.0
        for c, (m, s) in enumerate(zip(_MEAN, _STD)):
            arr[:, :, c] = (arr[:, :, c] - m) / s

        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor, torch.from_numpy(heatmaps)


def _augment_frame(rgb: np.ndarray) -> np.ndarray:
    """Apply light augmentation to an RGB frame (no geometry — keeps heatmaps valid)."""
    rng = np.random.default_rng()
    # Random brightness / contrast jitter
    alpha = float(rng.uniform(0.85, 1.15))
    beta  = float(rng.uniform(-10, 10))
    aug = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # Random horizontal noise
    noise = rng.integers(-5, 6, size=aug.shape, dtype=np.int16)
    aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return aug


# ---------------------------------------------------------------------------
# Synthetic data generator (for --synth smoke-test only)
# ---------------------------------------------------------------------------

def _generate_synthetic_annotations(
    n: int,
    frames_dir: Path,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate n synthetic rink-like frames and return annotation dicts.

    Produces simple coloured images with dots at known keypoint positions.
    NOT suitable for real training — smoke-test only.
    """
    from PIL import Image, ImageDraw

    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations: list[dict] = []

    # Approximate keypoint pixel positions in a 1280×720 rink frame (centred view)
    _BASE_KPS: dict[str, tuple[float, float]] = {
        "center_ice_dot":     (640, 360),
        "blue_line_left":     (440, 360),
        "blue_line_right":    (840, 360),
        "faceoff_home_left":  (220, 240),
        "faceoff_home_right": (220, 480),
        "faceoff_away_left":  (1060, 240),
        "faceoff_away_right": (1060, 480),
    }

    for i in range(n):
        # Jitter keypoint positions slightly
        jitter = rng.integers(-15, 16, size=(7, 2))
        img = Image.new("RGB", (1280, 720), color=(180, 210, 230))
        draw = ImageDraw.Draw(img)
        # Draw rink outline
        draw.rectangle([80, 80, 1200, 640], outline=(255, 255, 255), width=4)
        # Draw blue lines
        draw.line([(440, 80), (440, 640)], fill=(0, 100, 255), width=4)
        draw.line([(840, 80), (840, 640)], fill=(0, 100, 255), width=4)
        # Draw red centre line
        draw.line([(640, 80), (640, 640)], fill=(220, 40, 40), width=3)

        kps_ann: dict[str, dict] = {}
        for j, (name, (bx, by)) in enumerate(_BASE_KPS.items()):
            x = float(bx + jitter[j, 0])
            y = float(by + jitter[j, 1])
            r = 6
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(255, 80, 80))
            kps_ann[name] = {"x": x, "y": y, "visible": True}

        fname = f"synth_{i:04d}.jpg"
        fpath = frames_dir / fname
        img.save(str(fpath), quality=85)

        annotations.append({
            "frame_path": str(fpath.relative_to(_REPO)),
            "arena": "Synthetic",
            "image_w": 1280,
            "image_h": 720,
            "keypoints": kps_ann,
        })

    return annotations


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_pck(
    model: RinkKeypointNet,
    loader: DataLoader,
    device: torch.device,
    radius_hmap: float,
) -> float:
    """Compute PCK@radius in heatmap-space pixels."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)          # (B, 7, HMAP_H, HMAP_W)
            B = imgs.size(0)
            for b in range(B):
                for k in range(NUM_KEYPOINTS):
                    gt_hm = targets[b, k].cpu().numpy()
                    if gt_hm.max() < 0.1:   # invisible keypoint
                        continue
                    total += 1
                    # GT position
                    gt_flat = int(gt_hm.argmax())
                    gt_y, gt_x = gt_flat // HMAP_W, gt_flat % HMAP_W
                    # Pred position
                    pr_hm = preds[b, k].cpu().numpy()
                    pr_flat = int(pr_hm.argmax())
                    pr_y, pr_x = pr_flat // HMAP_W, pr_flat % HMAP_W
                    dist = ((pr_x - gt_x) ** 2 + (pr_y - gt_y) ** 2) ** 0.5
                    if dist <= radius_hmap:
                        correct += 1
    return correct / total if total > 0 else 0.0


def _train(
    model: RinkKeypointNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train the model; return metrics dict with best_state and pck_at_10."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # PCK radius in heatmap pixels: 10px at INPUT_W → 10 * HMAP_W / INPUT_W
    pck_radius_hmap = PCK_RADIUS * HMAP_W / INPUT_W

    best_pck   = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            pck = _compute_pck(model, val_loader, device, pck_radius_hmap)
            avg_loss = running_loss / len(train_loader.dataset)   # type: ignore[arg-type]
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  PCK@10={pck:.3f}")
            if pck > best_pck:
                best_pck = pck
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()
        best_pck = _compute_pck(model, val_loader, device, pck_radius_hmap)

    return {"pck_at_10": best_pck, "epochs": epochs, "best_state": best_state}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train rink keypoint detector (Phase 16.5)")
    p.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",       type=float, default=DEFAULT_LR)
    p.add_argument("--out",      default=str(DEFAULT_MODEL_OUT))
    p.add_argument("--synth",    action="store_true",
                   help="Use synthetic frames (smoke-test; won't generalise)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Build data + model but skip training and save")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args  = _parse_args(argv)
    out   = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train-rink-kp] device={device}")

    # ------------------------------------------------------------------
    # Load or generate annotation data
    # ------------------------------------------------------------------
    if args.synth:
        print(f"[train-rink-kp] --synth: generating {SYNTH_COUNT} synthetic frames…")
        rng = np.random.default_rng(42)
        frames_dir = _REPO / "data" / "cv_training" / "keypoints" / "synth_frames"
        all_annotations = _generate_synthetic_annotations(SYNTH_COUNT, frames_dir, rng)
    else:
        if not ANNOTATIONS_PATH.exists():
            print(
                f"[train-rink-kp] ERROR: No annotation data found.\n"
                f"  Expected: {ANNOTATIONS_PATH}\n\n"
                "  To create annotation data:\n"
                "    1. Extract ~200 frames from 16.1 clips.\n"
                "    2. Annotate 7 keypoints per frame using CVAT, labelme, or any\n"
                "       tool that exports pixel (x, y) coordinates.\n"
                "    3. Convert output to annotations.json (schema in this file's docstring).\n"
                "    4. Place frames under data/cv_training/keypoints/frames/\n\n"
                "  For a quick smoke-test with synthetic data, run with --synth."
            )
            sys.exit(1)

        with open(ANNOTATIONS_PATH) as fh:
            raw = json.load(fh)
        all_annotations = raw["annotations"]
        if len(all_annotations) < 20:
            print(
                f"[train-rink-kp] WARNING: Only {len(all_annotations)} annotations found "
                f"(target: ~200). Model quality will be limited."
            )

    print(f"[train-rink-kp] {len(all_annotations)} frames loaded")

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    rng_split = np.random.default_rng(0)
    indices   = rng_split.permutation(len(all_annotations))
    n_val     = max(1, int(len(all_annotations) * VAL_SPLIT))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]
    train_ann = [all_annotations[i] for i in train_idx]
    val_ann   = [all_annotations[i] for i in val_idx]
    print(f"[train-rink-kp] train={len(train_ann)}  val={len(val_ann)}")

    train_ds = RinkKeypointDataset(train_ann, augment=True)
    val_ds   = RinkKeypointDataset(val_ann,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = RinkKeypointNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train-rink-kp] model params={n_params:,}")

    if args.dry_run:
        print("[train-rink-kp] --dry-run: skipping training")
        return

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"[train-rink-kp] Training {args.epochs} epochs…")
    metrics = _train(model, train_loader, val_loader, args.epochs, args.lr, device)
    pck = metrics["pck_at_10"]
    passed, _ = check_quality_gate(metrics)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model.load_state_dict(metrics["best_state"])
    torch.save(model.state_dict(), str(out))
    metrics_out = {k: v for k, v in metrics.items() if k != "best_state"}
    metrics_out["passed_gate"] = passed
    metrics_out["gate"] = f"PCK@10 >= {PCK_GATE}"

    with open(METRICS_JSON, "w") as fh:
        json.dump(metrics_out, fh, indent=2)

    status = "PASSED" if passed else "FAILED"
    print(f"[train-rink-kp] PCK@10={pck:.3f}  gate={PCK_GATE}  [{status}]")
    print(f"[train-rink-kp] Saved → {out}")

    if not passed:
        print(
            "[train-rink-kp] Gate not met — annotate more frames or increase epochs.\n"
            "  Hint: target ~200 manually annotated frames; current performance "
            "may reflect limited synthetic data."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
