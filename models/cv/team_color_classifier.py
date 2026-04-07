"""Phase 16.4 — Team Color Classifier.

Rule-based: extract HSV histogram from a player's torso region, compare against
a per-team lookup table (LUT) built from the TEAM_COLORS registry in
``scripts/generate_jersey_data.py``.

No neural net — pure HSV color distance with optional jersey-number
disambiguation for white-on-white matchups (TOR vs TBL, etc.).

Algorithm
---------
1. Crop the *torso* of the player detection bounding box (middle 50% height,
   middle 60% width — avoids helmet, pants, skates, and background).
2. Convert BGR → HSV.
3. Compute a weighted colour descriptor:
       - Hue histogram  (36 bins, 0–360°)
       - Saturation mean + std
       - Value mean
4. Compare descriptor against every entry in the LUT using a distance metric
   that penalises hue mismatch more than brightness/saturation mismatch.
5. Return (team_code, confidence) of the best match.

White-on-white disambiguation
------------------------------
When the top-2 candidates are both low-saturation (white/light jerseys) and
their scores are within ``WOW_MARGIN``, the jersey number from Feature 16.3 is
used to look up which team a given number belongs to for that game's roster.
The ``game_roster`` dict maps ``number → team_code`` and is supplied by the
CV worker (16.7) at the start of each game.

Usage::

    from models.cv.team_color_classifier import TeamColorClassifier

    clf = TeamColorClassifier()
    team, conf = clf.classify(crop_bgr)                    # crop: np.ndarray
    team, conf = clf.classify(crop_bgr, jersey_number=88,  # OCR hint
                              game_roster={88: "CHI", 91: "STL"})

Building / refreshing the LUT::

    python scripts/build_team_color_lut.py          # writes models/cv/team_color_lut.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO    = Path(__file__).parents[2]
LUT_PATH = _REPO / "models" / "cv" / "team_color_lut.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUE_BINS      = 36          # 10° per bin
WOW_MARGIN    = 0.12        # white-on-white: score gap below which OCR hint used
SAT_WHITE_THR = 0.18        # saturation below → "white/very-light" jersey
TORSO_Y_LO    = 0.25        # torso crop: start at 25% of bbox height
TORSO_Y_HI    = 0.70        # torso crop: end at 70%
TORSO_X_LO    = 0.20        # torso crop: start at 20% of bbox width
TORSO_X_HI    = 0.80        # torso crop: end at 80%
MIN_TORSO_PX  = 16          # below this the crop is too small to classify


# ---------------------------------------------------------------------------
# Descriptor helpers
# ---------------------------------------------------------------------------

def _bgr_to_hsv_arr(bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)


def _torso_crop(bbox_bgr: np.ndarray) -> np.ndarray:
    """Return the torso sub-region of a player detection crop."""
    h, w = bbox_bgr.shape[:2]
    y0 = int(h * TORSO_Y_LO)
    y1 = int(h * TORSO_Y_HI)
    x0 = int(w * TORSO_X_LO)
    x1 = int(w * TORSO_X_HI)
    return bbox_bgr[y0:y1, x0:x1]


def _descriptor(hsv: np.ndarray) -> dict:
    """Compute colour descriptor from an HSV image (float32, OpenCV scale).

    OpenCV HSV: H ∈ [0,180), S ∈ [0,255], V ∈ [0,255].
    """
    h_chan = hsv[:, :, 0].ravel()   # [0, 180)
    s_chan = hsv[:, :, 1].ravel()   # [0, 255]
    v_chan = hsv[:, :, 2].ravel()   # [0, 255]

    # Hue histogram (only pixels with S > threshold to avoid grey noise)
    sat_norm = s_chan / 255.0
    mask = sat_norm > 0.12
    if mask.sum() > 4:
        hist, _ = np.histogram(h_chan[mask] * 2.0, bins=HUE_BINS, range=(0, 360))
    else:
        hist = np.zeros(HUE_BINS, dtype=np.float32)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

    return {
        "hue_hist":  hist,
        "sat_mean":  float(s_chan.mean() / 255.0),
        "sat_std":   float(s_chan.std()  / 255.0),
        "val_mean":  float(v_chan.mean() / 255.0),
        "is_white":  float(s_chan.mean() / 255.0) < SAT_WHITE_THR,
    }


def _hue_hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Circular chi-squared distance between two normalised hue histograms."""
    denom = a + b
    safe  = denom > 1e-9
    diff  = (a - b) ** 2
    chi2  = np.where(safe, diff / denom, 0.0).sum()
    return float(chi2)


def _descriptor_distance(desc: dict, lut_entry: dict) -> float:
    """Score ∈ [0, ∞); lower = better match."""
    hue_dist = _hue_hist_distance(
        np.array(desc["hue_hist"]),
        np.array(lut_entry["hue_hist"]),
    )
    sat_dist = abs(desc["sat_mean"] - lut_entry["sat_mean"])
    val_dist = abs(desc["val_mean"] - lut_entry["val_mean"])
    # Hue is most discriminative; saturation second; brightness last
    return 2.5 * hue_dist + 1.0 * sat_dist + 0.4 * val_dist


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TeamColorClassifier:
    """Rule-based team-colour classifier using HSV histogram matching.

    Args:
        lut_path:    Path to the pre-built JSON LUT.
                     Defaults to ``models/cv/team_color_lut.json``.
        wow_margin:  Score gap below which jersey-number hint is used for
                     white-on-white disambiguation.
    """

    def __init__(
        self,
        lut_path: Path | str = LUT_PATH,
        wow_margin: float = WOW_MARGIN,
    ) -> None:
        lut_path = Path(lut_path)
        if not lut_path.exists():
            raise FileNotFoundError(
                f"Team color LUT not found: {lut_path}\n"
                "Run `uv run python scripts/gretzky.py build-color-lut` to build it."
            )
        with open(lut_path) as fh:
            data = json.load(fh)
        # data: {"teams": {team_code: {"home": desc, "away": desc}}}
        self._lut: dict[str, dict[str, dict]] = data["teams"]
        self._wow_margin = wow_margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        crop_bgr: np.ndarray,
        jersey_number: int | None = None,
        game_roster: dict[int, str] | None = None,
    ) -> tuple[str, float]:
        """Classify a player crop to a team code.

        Args:
            crop_bgr:     Player detection crop (H, W, 3) BGR uint8.
            jersey_number: OCR-read jersey number for white-on-white hint.
            game_roster:  ``{number: team_code}`` map for the current game.

        Returns:
            ``(team_code, confidence)`` where confidence ∈ [0, 1].
            Returns ``("UNK", 0.0)`` if crop is too small or LUT is empty.
        """
        torso = _torso_crop(crop_bgr)
        if torso.size < MIN_TORSO_PX * MIN_TORSO_PX * 3:
            return "UNK", 0.0

        hsv  = _bgr_to_hsv_arr(torso)
        desc = _descriptor(hsv)

        # Score every (team, jersey_type) entry
        scores: list[tuple[float, str, str]] = []   # (dist, team, jersey_type)
        for team, jerseys in self._lut.items():
            for jtype, lut_entry in jerseys.items():
                dist = _descriptor_distance(desc, lut_entry)
                scores.append((dist, team, jtype))

        if not scores:
            return "UNK", 0.0

        scores.sort(key=lambda x: x[0])
        best_dist, best_team, _ = scores[0]

        # White-on-white disambiguation via jersey number
        if (
            jersey_number is not None
            and game_roster is not None
            and len(scores) >= 2
        ):
            second_dist = scores[1][0]
            gap = second_dist - best_dist
            if desc["is_white"] and gap < self._wow_margin:
                roster_team = game_roster.get(jersey_number)
                if roster_team is not None:
                    best_team = roster_team
                    best_dist = 0.0   # definitive from roster

        # Convert distance to [0, 1] confidence (empirically tuned)
        confidence = float(np.clip(1.0 - best_dist / 3.0, 0.0, 1.0))
        return best_team, confidence

    def classify_batch(
        self,
        crops: list[np.ndarray],
        jersey_numbers: list[int | None] | None = None,
        game_roster: dict[int, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Classify a list of crops. Returns one (team, conf) per crop."""
        numbers = jersey_numbers or [None] * len(crops)
        return [
            self.classify(c, n, game_roster)
            for c, n in zip(crops, numbers)
        ]

    @property
    def teams(self) -> list[str]:
        return sorted(self._lut.keys())

    def __repr__(self) -> str:
        return f"TeamColorClassifier(teams={len(self._lut)}, lut={LUT_PATH.name!r})"
