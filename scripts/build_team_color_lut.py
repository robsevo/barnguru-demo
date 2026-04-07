"""Phase 16.4 — Build team-colour lookup table (LUT).

Converts the ``TEAM_COLORS`` registry (RGB home jersey colours) into HSV
descriptors suitable for real-time histogram matching in the CV worker.

For every team the script generates two entries:
  - ``home``  — the actual home jersey colour from TEAM_COLORS
  - ``away``  — synthetic white/light away jersey (most NHL away jerseys are
                white with a coloured trim; a few dark alternates are noted)

The LUT is a JSON file at ``models/cv/team_color_lut.json``:

    {
      "teams": {
        "MTL": {
          "home": {"hue_hist": [...], "sat_mean": 0.72, ...},
          "away": {"hue_hist": [...], "sat_mean": 0.08, ...}
        },
        ...
      },
      "built_at": "2026-04-01T12:00:00"
    }

Usage::

    uv run python scripts/gretzky.py build-color-lut
    uv run python scripts/build_team_color_lut.py --show
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.generate_jersey_data import TEAM_COLORS
from models.cv.team_color_classifier import (
    HUE_BINS,
    SAT_WHITE_THR,
    _descriptor,
    _bgr_to_hsv_arr,
)

OUTPUT_PATH = _REPO / "models" / "cv" / "team_color_lut.json"

# ---------------------------------------------------------------------------
# Away jersey overrides
# Teams that do NOT wear white away jerseys (dark alternates common enough
# to warrant a separate descriptor).
# ---------------------------------------------------------------------------
# Format: team_code → (R, G, B)  of the away/alternate jersey background
_DARK_AWAY: dict[str, tuple[int, int, int]] = {
    "ANA": (211, 174, 104),   # gold alternate
    "LAK": (162, 170, 173),   # silver alternate
    "SEA": (153, 217, 217),   # ice blue alternate
    "VGK": (51, 63, 72),      # steel grey alternate
}

# Standard white away colour
_WHITE_AWAY = (248, 248, 248)


def _solid_bgr_image(rgb: tuple[int, int, int], size: int = 64) -> np.ndarray:
    """Return a (size, size, 3) BGR image filled with a single RGB colour."""
    r, g, b = rgb
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = b  # BGR
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def _blended_bgr_image(
    primary_rgb: tuple[int, int, int],
    secondary_rgb: tuple[int, int, int],
    size: int = 64,
    blend: float = 0.75,
) -> np.ndarray:
    """Return an image that blends primary (75%) with secondary (25%).

    Simulates a jersey with the number/stripe colour mixed in.
    """
    pr, pg, pb = primary_rgb
    sr, sg, sb = secondary_rgb
    r = int(pr * blend + sr * (1 - blend))
    g = int(pg * blend + sg * (1 - blend))
    b = int(pb * blend + sb * (1 - blend))
    return _solid_bgr_image((r, g, b), size)


def build_lut(show: bool = False) -> dict:
    """Build the full LUT dict from TEAM_COLORS."""
    teams_lut: dict[str, dict] = {}

    for code, colors in sorted(TEAM_COLORS.items()):
        home_rgb  = colors["bg"]
        num_rgb   = colors["fg"]
        away_rgb  = _DARK_AWAY.get(code, _WHITE_AWAY)

        # Build blended images (primary 75% + number colour 25%)
        home_img = _blended_bgr_image(home_rgb, num_rgb)
        away_img = _blended_bgr_image(away_rgb, num_rgb)

        home_hsv  = _bgr_to_hsv_arr(home_img)
        away_hsv  = _bgr_to_hsv_arr(away_img)

        home_desc = _descriptor(home_hsv)
        away_desc = _descriptor(away_hsv)

        # Serialise — numpy arrays → lists
        teams_lut[code] = {
            "home": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in home_desc.items()},
            "away": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in away_desc.items()},
        }

        if show:
            h_sat = home_desc["sat_mean"]
            a_sat = away_desc["sat_mean"]
            home_label = "DARK" if h_sat > SAT_WHITE_THR else "LIGHT"
            away_label = "DARK" if a_sat > SAT_WHITE_THR else "LIGHT"
            print(f"  {code}  home={home_label}(sat={h_sat:.2f})  "
                  f"away={away_label}(sat={a_sat:.2f})")

    return {
        "teams":    teams_lut,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "n_teams":  len(teams_lut),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build team colour LUT (Phase 16.4)")
    p.add_argument("--out",  default=str(OUTPUT_PATH), help="Output JSON path")
    p.add_argument("--show", action="store_true",      help="Print descriptor summary")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out  = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build-color-lut] Building HSV LUT for {len(TEAM_COLORS)} teams…")
    lut = build_lut(show=args.show)

    with open(out, "w") as fh:
        json.dump(lut, fh, indent=2)

    print(f"[build-color-lut] Done → {out}  ({lut['n_teams']} teams)")


if __name__ == "__main__":
    main()
