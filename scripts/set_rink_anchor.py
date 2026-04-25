"""Phase 16.5 — One-time anchor calibration (canonical or broadcast pixel).

Two modes:

1. **Canonical mode** (default): writes ``models/cv/rink_anchor.json`` ↔
   the canonical-LUT → NHL feet homography. Use this when the LUT is clean
   (rink keypoints converged sensibly in the canonical 640 × 360 frame).

2. **Fixed mode** (``--fixed``): writes ``models/cv/rink_fixed_anchor.json``
   ↔ a direct broadcast-pixel → NHL feet homography. Use this when the LUT
   is contaminated (e.g. the SHL pose model fires on broadcast logos rather
   than rink lines, giving a degenerate canonical space). The runtime
   short-circuits the LUT entirely and applies the fixed anchor directly to
   detected player pixels. Tied to a single camera angle — most NHL
   broadcasts use a stable mid-ice game-cam, so this is a usable workaround
   until the rink keypoint detector is retrained.

Without either anchor file, ``cv_tracker`` produces players in canonical
pixel coordinates only — the rink dots never appear in NHL feet, so the
RinkAnimation overlay shows nothing.

Workflow
--------
1. **Render the canonical LUT viz** to identify which kp index is which
   landmark::

      uv run python -c "from models.cv.adaptive_rink_lut import AdaptiveRinkLUT; \\
                        lut = AdaptiveRinkLUT.load(); print(lut.summary())"

   Or open ``data/cv_training/lut_convergence.html`` in a browser and read
   the per-kp positions in the rendered canonical frame (640 × 360).

2. **Pair four trusted LUT keypoints with their NHL feet coordinates.**
   The four canonical landmarks recommended (any four work, but these are
   the easiest to read off the canonical render):

   ============================  ===============  ==================
   Landmark                      NHL feet (x, y)  Notes
   ============================  ===============  ==================
   Center ice dot                (   0,   0)      Always visible
   Defensive blue line midpoint  ( -25,   0)      Home end / camera-left
   Offensive blue line midpoint  ( +25,   0)      Away end / camera-right
   Offensive faceoff dot, top    ( +69, +22)      One of the 4 dots
   ============================  ===============  ==================

3. **Run this script** with the canonical pixel coords you read off the LUT::

      uv run python scripts/set_rink_anchor.py \\
          --pair  640,180   0,0       \\
          --pair  440,180  -25,0      \\
          --pair  840,180   25,0      \\
          --pair 1060,240   69,22

   ``--pair`` takes ``canonical_x,canonical_y  nhl_x_ft,nhl_y_ft``; pass it
   four times. The script computes the 3×3 homography and writes it to
   ``models/cv/rink_anchor.json``.

4. **Verify by re-running cv_tracker on a known clip** — player positions
   should now land within the rink rectangle in the
   ``RinkAnimation`` SVG overlay.

Fixed-mode workflow
-------------------
1. Open one of ``data/cv_training/labelled/<NN>_f<XXXXX>_kps.jpg`` —
   broadcast frame with kp indices overlaid.
2. Identify 4 NHL landmarks visible in the frame and read their pixel
   coordinates by eye (or use any image viewer's pixel readout).
3. Run with ``--fixed``::

      uv run python scripts/set_rink_anchor.py --fixed \\
          --pair  640,360   0,0       \\
          --pair  490,355 -25,0       \\
          --pair  790,355  25,0       \\
          --pair  990,415  69,22

   Pixel coords are broadcast-frame coords (typically 1280 × 720), NHL
   feet are the rink coords. The script writes
   ``models/cv/rink_fixed_anchor.json`` and the runtime picks it up on
   the next ``/api/cv/start``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from models.cv.adaptive_homography import ANCHOR_PATH, set_anchor
from models.cv.adaptive_runtime import FIXED_ANCHOR_PATH


def _parse_pair(text: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Parse 'x_px,y_px x_ft,y_ft' into ((px, py), (nx, ny))."""
    parts = text.replace("\t", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--pair expects 'SRC_X,SRC_Y  NHL_X,NHL_Y' but got {text!r}"
        )
    try:
        sx, sy = (float(x) for x in parts[0].split(","))
        nx, ny = (float(x) for x in parts[1].split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--pair {text!r}: numeric parse failed ({e})"
        ) from e
    return (sx, sy), (nx, ny)


def _write_fixed_anchor(
    src: list[tuple[float, float]],
    dst: list[tuple[float, float]],
    out: Path,
):
    """Solve broadcast-pixel → NHL feet directly, writing the fixed-anchor
    schema the runtime expects (``H_pixel_to_nhl``).

    With exactly 4 points we use cv2.getPerspectiveTransform (direct DLT) —
    no RANSAC, since RANSAC needs more inliers than minimum-config points
    and would otherwise reject one of our 4 as an "outlier", giving a
    degenerate H that collapses to a 3-point fit. With 5+ points we fall
    back to least-squares findHomography(method=0).
    """
    import json
    import cv2
    import numpy as np

    src_np = np.array(src, dtype=np.float32)
    dst_np = np.array(dst, dtype=np.float32)
    if len(src_np) == 4:
        H = cv2.getPerspectiveTransform(src_np, dst_np)
    else:
        H, _ = cv2.findHomography(src_np, dst_np, method=0)
    if H is None:
        raise ValueError("Homography solve returned None — the 4 src points "
                         "may be collinear or 3 of them might be coincident.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "H_pixel_to_nhl":  H.tolist(),
        "pixel_points":    list(src),
        "nhl_feet_points": list(dst),
    }, indent=2))
    return H


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pair", action="append", required=True, type=_parse_pair,
        help="Source→NHL point pair, repeat ≥4 times. Format: "
             "'sx,sy  nx,ny'. In default mode source is canonical-LUT "
             "pixels; in --fixed mode source is broadcast pixels.",
    )
    p.add_argument(
        "--fixed", action="store_true",
        help="Skip the canonical LUT and write a direct broadcast-pixel → "
             "NHL feet anchor to models/cv/rink_fixed_anchor.json. Use this "
             "when the LUT is contaminated.",
    )
    p.add_argument(
        "--out", default=None,
        help=f"Output path. Default {ANCHOR_PATH} in canonical mode, "
             f"{FIXED_ANCHOR_PATH} with --fixed.",
    )
    args = p.parse_args(argv)

    if len(args.pair) < 4:
        p.error(f"need ≥4 --pair entries (got {len(args.pair)}); homography is "
                f"under-determined with fewer points.")

    src       = [pp[0] for pp in args.pair]
    nhl_feet  = [pp[1] for pp in args.pair]

    out = Path(args.out) if args.out else (FIXED_ANCHOR_PATH if args.fixed else ANCHOR_PATH)

    if args.fixed:
        H = _write_fixed_anchor(src, nhl_feet, out)
        label, name = "broadcast pixel", "H_pixel_to_nhl"
    else:
        H = set_anchor(src, nhl_feet, path=out)
        label, name = "canonical pixel", "H_canonical_to_nhl"

    print(f"\n[set-rink-anchor] {len(args.pair)} pairs accepted.")
    print(f"[set-rink-anchor] Anchor written to {out}")
    print(f"[set-rink-anchor] {name} =")
    for row in H:
        print(f"    [{row[0]:+.5f}  {row[1]:+.5f}  {row[2]:+.5f}]")

    # Sanity check: project the input source points back through H and
    # report the residual against the supplied NHL feet coords. Anything
    # over ~1 ft means a mis-clicked point.
    import numpy as np
    import cv2
    src_np = np.array(src, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(src_np, H).reshape(-1, 2)
    print(f"\n[set-rink-anchor] Reprojection residuals (NHL feet):")
    for (sx, sy), (nx, ny), (ex, ey) in zip(src, nhl_feet, proj):
        dx, dy = ex - nx, ey - ny
        print(f"   {label}=({sx:+7.1f},{sy:+7.1f})  "
              f"nhl=({nx:+6.2f},{ny:+6.2f})  "
              f"projected=({ex:+6.2f},{ey:+6.2f})  "
              f"Δ=({dx:+5.2f},{dy:+5.2f}) ft")


if __name__ == "__main__":
    main()
