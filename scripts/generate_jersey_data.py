"""Phase 16.3 — Synthetic jersey-number training data generator.

Produces ~50K labelled 48×48 RGB crops for training the jersey OCR model
(``models/cv/jersey_ocr.py``).

Pipeline per (jersey_number, team):
    1. Render number on team-coloured background using a bold system font.
    2. Apply N augmentation variants (rotation, blur, brightness, JPEG
       compression, perspective warp, noise).

Output::

    data/cv_training/jersey_crops/
        train/
            00001_num42_ANA_v3.jpg
            ...
        val/
            ...
        labels.csv    ← frame,split,number,team,variant

Usage::

    uv run python scripts/generate_jersey_data.py
    uv run python scripts/generate_jersey_data.py --out data/cv_training/jersey_crops
    uv run python scripts/generate_jersey_data.py --count 50000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE  = 48   # pixels — must match jersey_ocr.py INPUT_SIZE
VAL_RATIO   = 0.10
MIN_NUMBER  = 1
MAX_NUMBER  = 99


# ---------------------------------------------------------------------------
# NHL team colour registry (home jersey background + number colour)
# ---------------------------------------------------------------------------
# Format: team_code → {"bg": (R,G,B), "fg": (R,G,B)}
# bg = jersey primary colour, fg = number colour
# Colours are approximate; real training will augment heavily anyway.

TEAM_COLORS: dict[str, dict[str, tuple[int, int, int]]] = {
    "ANA": {"bg": (0, 0, 0),        "fg": (252, 76, 2)},
    "BOS": {"bg": (252, 181, 20),   "fg": (0, 0, 0)},
    "BUF": {"bg": (0, 38, 84),      "fg": (252, 181, 20)},
    "CAR": {"bg": (206, 17, 38),    "fg": (255, 255, 255)},
    "CBJ": {"bg": (0, 38, 84),      "fg": (206, 17, 38)},
    "CGY": {"bg": (206, 17, 38),    "fg": (255, 215, 0)},
    "CHI": {"bg": (206, 17, 38),    "fg": (255, 255, 255)},
    "COL": {"bg": (111, 38, 61),    "fg": (35, 97, 146)},
    "DAL": {"bg": (0, 104, 71),     "fg": (143, 116, 53)},
    "DET": {"bg": (206, 17, 38),    "fg": (255, 255, 255)},
    "EDM": {"bg": (252, 76, 2),     "fg": (0, 32, 91)},
    "FLA": {"bg": (0, 32, 91),      "fg": (206, 17, 38)},
    "LAK": {"bg": (0, 0, 0),        "fg": (162, 170, 173)},
    "MIN": {"bg": (2, 73, 48),      "fg": (163, 144, 106)},
    "MTL": {"bg": (175, 30, 45),    "fg": (0, 31, 91)},
    "NJD": {"bg": (206, 17, 38),    "fg": (0, 0, 0)},
    "NSH": {"bg": (255, 184, 28),   "fg": (0, 32, 91)},
    "NYI": {"bg": (0, 83, 155),     "fg": (252, 76, 2)},
    "NYR": {"bg": (0, 56, 168),     "fg": (206, 17, 38)},
    "OTT": {"bg": (198, 146, 20),   "fg": (0, 0, 0)},
    "PHI": {"bg": (247, 73, 2),     "fg": (255, 255, 255)},
    "PIT": {"bg": (252, 181, 20),   "fg": (0, 0, 0)},
    "SEA": {"bg": (0, 22, 40),      "fg": (153, 217, 217)},
    "SJS": {"bg": (0, 109, 117),    "fg": (0, 0, 0)},
    "STL": {"bg": (0, 47, 135),     "fg": (252, 181, 20)},
    "TBL": {"bg": (0, 40, 104),     "fg": (255, 255, 255)},
    "TOR": {"bg": (0, 32, 91),      "fg": (255, 255, 255)},
    "UTA": {"bg": (110, 198, 237),  "fg": (255, 255, 255)},
    "VAN": {"bg": (0, 32, 91),      "fg": (0, 129, 64)},
    "VGK": {"bg": (185, 151, 91),   "fg": (51, 63, 72)},
    "WPG": {"bg": (4, 30, 66),      "fg": (172, 148, 105)},
    "WSH": {"bg": (200, 16, 46),    "fg": (255, 255, 255)},
}

TEAM_CODES = sorted(TEAM_COLORS.keys())   # 32 teams

# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

_FALLBACK_FONTS: list[str] = [p for p in _FONT_CANDIDATES if Path(p).exists()]


def _load_font(size: int, prefer: list[str] = _FALLBACK_FONTS) -> ImageFont.ImageFont:
    for path in prefer:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow ≥ 10.1
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_base(number: int, team: str, font_size: int = 28) -> Image.Image:
    """Render a single jersey-number crop at INPUT_SIZE×INPUT_SIZE."""
    colors = TEAM_COLORS[team]
    img    = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), colors["bg"])
    draw   = ImageDraw.Draw(img)

    text = str(number)
    font = _load_font(font_size)

    # Centre the text
    bbox  = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (INPUT_SIZE - tw) // 2 - bbox[0]
    y = (INPUT_SIZE - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=colors["fg"], font=font)
    return img


def _add_noise(img: np.ndarray, rng: np.random.Generator, sigma: float = 8.0) -> np.ndarray:
    noise = rng.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _jpeg_artifact(pil_img: Image.Image, rng: np.random.Generator) -> Image.Image:
    quality = int(rng.integers(35, 92))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _perspective_warp(pil_img: Image.Image, rng: np.random.Generator, strength: float = 0.12) -> Image.Image:
    import cv2
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    dx = max(1, int(w * strength))
    dy = max(1, int(h * strength))
    src = np.float32([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
    offsets = rng.integers(-1, 2, size=(4, 1, 2)) * np.array([[[dx, dy]]])
    dst = np.clip(src + offsets.astype(np.float32), [[[0, 0]]], [[[w - 1, h - 1]]]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


class AugResult(NamedTuple):
    image: Image.Image
    variant: int   # 0 = base, 1-N = augmented


def _augment(base: Image.Image, rng: np.random.Generator) -> list[AugResult]:
    """Return list of (image, variant_index) including the base image."""
    results: list[AugResult] = [AugResult(base, 0)]
    arr = np.array(base)

    # v1: JPEG only
    results.append(AugResult(_jpeg_artifact(base, rng), 1))

    # v2: slight rotation
    angle = float(rng.uniform(-12, 12))
    results.append(AugResult(base.rotate(angle, fillcolor=tuple(arr[0, 0])), 2))

    # v3: Gaussian blur
    sigma = float(rng.uniform(0.5, 1.8))
    results.append(AugResult(base.filter(ImageFilter.GaussianBlur(radius=sigma)), 3))

    # v4: brightness boost
    bright_arr = np.clip(arr.astype(np.int16) + int(rng.integers(20, 50)), 0, 255).astype(np.uint8)
    results.append(AugResult(Image.fromarray(bright_arr), 4))

    # v5: brightness dim
    dim_arr = np.clip(arr.astype(np.int16) - int(rng.integers(20, 50)), 0, 255).astype(np.uint8)
    results.append(AugResult(Image.fromarray(dim_arr), 5))

    # v6: Gaussian noise
    noisy = Image.fromarray(_add_noise(arr, rng, sigma=12))
    results.append(AugResult(noisy, 6))

    # v7: perspective warp
    results.append(AugResult(_perspective_warp(base, rng), 7))

    # v8: rotation + JPEG
    angle2 = float(rng.uniform(-10, 10))
    rot = base.rotate(angle2, fillcolor=tuple(arr[0, 0]))
    results.append(AugResult(_jpeg_artifact(rot, rng), 8))

    # v9: blur + noise
    blurred = base.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2))))
    noisy2  = Image.fromarray(_add_noise(np.array(blurred), rng, sigma=8))
    results.append(AugResult(noisy2, 9))

    # v10: warp + JPEG
    warped = _perspective_warp(base, rng)
    results.append(AugResult(_jpeg_artifact(warped, rng), 10))

    # v11: random crop + resize (simulate detector crop misalignment)
    pad = int(rng.integers(2, 7))
    padded = Image.new("RGB", (INPUT_SIZE + pad * 2, INPUT_SIZE + pad * 2), tuple(arr[0, 0]))
    padded.paste(base, (pad, pad))
    x0 = int(rng.integers(0, pad + 1))
    y0 = int(rng.integers(0, pad + 1))
    cropped = padded.crop((x0, y0, x0 + INPUT_SIZE, y0 + INPUT_SIZE)).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR
    )
    results.append(AugResult(cropped, 11))

    # v12: large font variant (number fills more of the crop)
    large_base = _render_base(0, "BOS", font_size=36)  # placeholder; replaced below
    results.append(AugResult(base, 12))    # slot reserved; overwritten per-number

    # v13: small font variant
    results.append(AugResult(base, 13))   # slot reserved; overwritten per-number

    # v14: random colour jitter
    h_arr = arr.copy().astype(np.int16)
    for c in range(3):
        h_arr[:, :, c] = np.clip(h_arr[:, :, c] + int(rng.integers(-30, 30)), 0, 255)
    results.append(AugResult(Image.fromarray(h_arr.astype(np.uint8)), 14))

    # v15: mirror (flip L-R — tests if model handles mirrored numbers; label remains same)
    results.append(AugResult(base.transpose(Image.FLIP_LEFT_RIGHT), 15))

    return results


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(
    out_dir: Path,
    count_target: int = 50_000,
    seed: int = 42,
    font_size: int = 28,
    verbose: bool = True,
) -> int:
    """Generate synthetic jersey crops.

    Args:
        out_dir:      Root output directory.
        count_target: Approximate total sample count; determines aug variants used.
        seed:         RNG seed for reproducibility.
        font_size:    Base font size for rendering.
        verbose:      Print progress.

    Returns:
        Total number of images written.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    train_dir = out_dir / "train"
    val_dir   = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    label_rows: list[dict] = []
    written = 0
    img_idx = 0

    numbers = list(range(MIN_NUMBER, MAX_NUMBER + 1))   # 1–99

    # How many augmentation variants to use per (number, team) pair
    n_teams   = len(TEAM_CODES)
    combos    = len(numbers) * n_teams        # 99 × 32 = 3168
    n_aug     = max(1, math.ceil(count_target / combos))
    # Cap at number of augmentation variants available (0–15 = 16 total)
    n_aug     = min(n_aug, 16)

    if verbose:
        total_est = combos * n_aug
        print(f"[jersey-gen] {len(numbers)} numbers × {n_teams} teams × {n_aug} variants "
              f"= ~{total_est} samples → {out_dir}")

    for number in numbers:
        for team in TEAM_CODES:
            base = _render_base(number, team, font_size=font_size)

            # Font-size variants replace slots 12 and 13
            large = _render_base(number, team, font_size=min(font_size + 10, 40))
            small = _render_base(number, team, font_size=max(font_size - 8, 12))

            augmented = _augment(base, rng)
            # Overwrite reserved slots
            augmented[12] = AugResult(large, 12)
            augmented[13] = AugResult(small, 13)

            for aug in augmented[:n_aug]:
                img_idx += 1
                split = "val" if rng.random() < VAL_RATIO else "train"
                dest  = val_dir if split == "val" else train_dir
                fname = f"{img_idx:06d}_num{number:02d}_{team}_v{aug.variant}.jpg"
                aug.image.save(dest / fname, format="JPEG", quality=90)
                label_rows.append({
                    "filename": fname,
                    "split":    split,
                    "number":   number,
                    "team":     team,
                    "variant":  aug.variant,
                })
                written += 1

    # Write labels CSV
    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "split", "number", "team", "variant"])
        w.writeheader()
        w.writerows(label_rows)

    if verbose:
        print(f"[jersey-gen] Done — {written} images, labels → {csv_path}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic jersey OCR training data")
    p.add_argument("--out",   default="data/cv_training/jersey_crops",
                   help="Output directory (default: data/cv_training/jersey_crops)")
    p.add_argument("--count", type=int, default=50_000,
                   help="Target sample count (default: 50000)")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--font-size", type=int, default=28)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out  = _REPO / args.out

    if out.exists() and not args.force:
        label_csv = out / "labels.csv"
        if label_csv.exists():
            import csv as _csv
            with open(label_csv) as fh:
                existing = sum(1 for _ in _csv.reader(fh)) - 1
            print(f"[jersey-gen] {out} already exists ({existing} samples). "
                  "Pass --force to regenerate.")
            return

    generate(out, count_target=args.count, seed=args.seed, font_size=args.font_size)


if __name__ == "__main__":
    main()
