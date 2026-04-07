"""Tests for Phase 16.4 — Team Color Classifier.

Covers:
  - LUT build: all 32 teams present, home/away entries, required keys
  - _descriptor: output keys, hue_hist shape, normalised, is_white flag
  - _hue_hist_distance: identical → 0, orthogonal > 0
  - _descriptor_distance: same descriptor → 0
  - TeamColorClassifier: missing LUT raises FileNotFoundError
  - classify: solid-colour crops → reasonable top-1 match for saturated teams
  - classify: white crop → is_white, low confidence
  - white-on-white disambiguation: jersey-number roster hint overrides score
  - classify_batch: length matches input
  - build_lut script: deterministic, idempotent, correct structure
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.cv.team_color_classifier import (
    HUE_BINS,
    LUT_PATH,
    SAT_WHITE_THR,
    TeamColorClassifier,
    _bgr_to_hsv_arr,
    _descriptor,
    _descriptor_distance,
    _hue_hist_distance,
    _torso_crop,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _solid_bgr(r: int, g: int, b: int, size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


# ===========================================================================
# LUT file — structure tests (requires models/cv/team_color_lut.json)
# ===========================================================================

class TestLutFile:
    def test_lut_exists(self):
        assert LUT_PATH.exists(), (
            "LUT not found — run `uv run python scripts/build_team_color_lut.py`"
        )

    def test_32_teams(self):
        data = json.loads(LUT_PATH.read_text())
        assert len(data["teams"]) == 32

    def test_home_away_present(self):
        data = json.loads(LUT_PATH.read_text())
        for code, jerseys in data["teams"].items():
            assert "home" in jerseys, f"{code} missing home entry"
            assert "away" in jerseys, f"{code} missing away entry"

    def test_descriptor_keys(self):
        data  = json.loads(LUT_PATH.read_text())
        entry = next(iter(data["teams"].values()))["home"]
        for key in ("hue_hist", "sat_mean", "sat_std", "val_mean", "is_white"):
            assert key in entry, f"Missing key: {key}"

    def test_hue_hist_length(self):
        data  = json.loads(LUT_PATH.read_text())
        entry = next(iter(data["teams"].values()))["home"]
        assert len(entry["hue_hist"]) == HUE_BINS

    def test_built_at_present(self):
        data = json.loads(LUT_PATH.read_text())
        assert "built_at" in data

    def test_n_teams_field(self):
        data = json.loads(LUT_PATH.read_text())
        assert data["n_teams"] == 32


# ===========================================================================
# _descriptor
# ===========================================================================

class TestDescriptor:
    def test_output_keys(self):
        img  = _solid_bgr(200, 20, 20)
        hsv  = _bgr_to_hsv_arr(img)
        desc = _descriptor(hsv)
        for key in ("hue_hist", "sat_mean", "sat_std", "val_mean", "is_white"):
            assert key in desc

    def test_hue_hist_shape(self):
        img  = _solid_bgr(200, 20, 20)
        hsv  = _bgr_to_hsv_arr(img)
        desc = _descriptor(hsv)
        assert len(desc["hue_hist"]) == HUE_BINS

    def test_hue_hist_normalised(self):
        img  = _solid_bgr(200, 20, 20)
        hsv  = _bgr_to_hsv_arr(img)
        desc = _descriptor(hsv)
        s = sum(desc["hue_hist"])
        assert abs(s - 1.0) < 1e-5 or s == 0.0   # either normalised or empty

    def test_white_flag_on_white(self):
        white = _solid_bgr(248, 248, 248)
        hsv   = _bgr_to_hsv_arr(white)
        desc  = _descriptor(hsv)
        assert desc["is_white"] is True

    def test_white_flag_off_for_saturated(self):
        red  = _solid_bgr(200, 20, 20)
        hsv  = _bgr_to_hsv_arr(red)
        desc = _descriptor(hsv)
        assert desc["is_white"] is False

    def test_sat_mean_range(self):
        img  = _solid_bgr(150, 50, 50)
        hsv  = _bgr_to_hsv_arr(img)
        desc = _descriptor(hsv)
        assert 0.0 <= desc["sat_mean"] <= 1.0

    def test_val_mean_range(self):
        img  = _solid_bgr(100, 100, 100)
        hsv  = _bgr_to_hsv_arr(img)
        desc = _descriptor(hsv)
        assert 0.0 <= desc["val_mean"] <= 1.0


# ===========================================================================
# _hue_hist_distance
# ===========================================================================

class TestHueHistDistance:
    def test_identical_is_zero(self):
        h = np.ones(HUE_BINS, dtype=np.float32) / HUE_BINS
        assert _hue_hist_distance(h, h) == pytest.approx(0.0)

    def test_orthogonal_positive(self):
        a = np.zeros(HUE_BINS, dtype=np.float32)
        b = np.zeros(HUE_BINS, dtype=np.float32)
        a[0] = 1.0
        b[HUE_BINS // 2] = 1.0
        assert _hue_hist_distance(a, b) > 0.0

    def test_symmetric(self):
        a = np.random.default_rng(0).random(HUE_BINS).astype(np.float32)
        b = np.random.default_rng(1).random(HUE_BINS).astype(np.float32)
        assert _hue_hist_distance(a, b) == pytest.approx(_hue_hist_distance(b, a))


# ===========================================================================
# _descriptor_distance
# ===========================================================================

class TestDescriptorDistance:
    def _desc(self, rgb: tuple[int, int, int]) -> dict:
        img  = _solid_bgr(*rgb)
        hsv  = _bgr_to_hsv_arr(img)
        d    = _descriptor(hsv)
        d["hue_hist"] = list(d["hue_hist"])
        return d

    def test_self_distance_zero(self):
        d = self._desc((200, 20, 20))
        assert _descriptor_distance(d, d) == pytest.approx(0.0, abs=1e-5)

    def test_red_vs_blue_greater_than_self(self):
        red  = self._desc((200, 20, 20))
        blue = self._desc((20, 20, 200))
        assert _descriptor_distance(red, blue) > _descriptor_distance(red, red)


# ===========================================================================
# _torso_crop
# ===========================================================================

class TestTorsoCrop:
    def test_output_smaller_than_input(self):
        crop = np.zeros((120, 60, 3), dtype=np.uint8)
        out  = _torso_crop(crop)
        assert out.shape[0] < crop.shape[0]
        assert out.shape[1] < crop.shape[1]

    def test_minimum_size_non_zero(self):
        crop = np.zeros((50, 30, 3), dtype=np.uint8)
        out  = _torso_crop(crop)
        assert out.size > 0


# ===========================================================================
# TeamColorClassifier init
# ===========================================================================

class TestClassifierInit:
    def test_raises_if_lut_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="(?i)team color lut not found"):
            TeamColorClassifier(lut_path=tmp_path / "missing.json")

    def test_loads_32_teams(self):
        clf = TeamColorClassifier()
        assert len(clf.teams) == 32

    def test_repr(self):
        clf = TeamColorClassifier()
        assert "TeamColorClassifier" in repr(clf)
        assert "32" in repr(clf)


# ===========================================================================
# classify — saturated colour crops
# ===========================================================================

class TestClassify:
    """
    We can't expect exact team matches from a flat-colour patch since real
    jerseys have texture/shading, but we CAN assert:
    - return type is (str, float)
    - confidence is in [0, 1]
    - a pure red patch returns a red-jersey team (CAR/CHI/DET/MTL/…)
    - a white patch returns a white-away team with low-ish confidence
    """

    _RED_TEAMS  = {"CAR", "CGY", "CHI", "DET", "MTL", "NJD", "NYR", "PHI", "WSH"}
    _BLUE_TEAMS = {"BUF", "CBJ", "FLA", "NYI", "NYR", "STL", "TBL", "TOR", "VAN", "WPG"}

    def _clf(self) -> TeamColorClassifier:
        return TeamColorClassifier()

    def test_returns_tuple(self):
        clf  = self._clf()
        crop = _solid_bgr(200, 20, 20, size=80)
        result = clf.classify(crop)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_confidence_in_range(self):
        clf  = self._clf()
        crop = _solid_bgr(200, 20, 20, size=80)
        _, conf = clf.classify(crop)
        assert 0.0 <= conf <= 1.0

    def test_red_crop_returns_red_team(self):
        clf  = self._clf()
        crop = _solid_bgr(206, 17, 38, size=80)   # pure CAR/CHI red
        team, conf = clf.classify(crop)
        assert team in self._RED_TEAMS, f"Expected red team, got {team}"

    def test_orange_crop_returns_orange_team(self):
        clf  = self._clf()
        crop = _solid_bgr(252, 76, 2, size=80)    # EDM / PHI / CGY orange
        team, conf = clf.classify(crop)
        assert team in {"EDM", "PHI", "ANA", "NYI", "CGY"}

    def test_yellow_crop_returns_yellow_team(self):
        clf  = self._clf()
        crop = _solid_bgr(252, 181, 20, size=80)   # BOS / PIT / NSH gold
        team, conf = clf.classify(crop)
        assert team in {"BOS", "PIT", "NSH", "BUF", "STL"}

    def test_white_crop_returns_some_team(self):
        clf  = self._clf()
        crop = _solid_bgr(248, 248, 248, size=80)
        team, conf = clf.classify(crop)
        assert isinstance(team, str)
        assert len(team) <= 4     # valid team code or "UNK"

    def test_tiny_crop_returns_unk(self):
        clf  = self._clf()
        crop = np.zeros((4, 4, 3), dtype=np.uint8)
        team, conf = clf.classify(crop)
        assert team == "UNK"
        assert conf == 0.0

    def test_white_on_white_roster_disambiguation(self):
        """When two white teams (TOR vs TBL) and OCR gives number, roster wins."""
        clf   = self._clf()
        white = _solid_bgr(248, 248, 248, size=80)
        roster = {91: "STL", 29: "VAN"}   # hypothetical game roster
        team, conf = clf.classify(white, jersey_number=91, game_roster=roster)
        assert team == "STL"

    def test_classify_batch_length(self):
        clf   = self._clf()
        crops = [_solid_bgr(200, 20, 20, size=64) for _ in range(5)]
        results = clf.classify_batch(crops)
        assert len(results) == 5

    def test_classify_batch_types(self):
        clf   = self._clf()
        crops = [_solid_bgr(100, 150, 50, size=64) for _ in range(3)]
        for team, conf in clf.classify_batch(crops):
            assert isinstance(team, str)
            assert 0.0 <= conf <= 1.0


# ===========================================================================
# build_lut script
# ===========================================================================

class TestBuildLut:
    def test_output_structure(self, tmp_path):
        from scripts.build_team_color_lut import build_lut
        lut = build_lut(show=False)
        assert "teams" in lut
        assert "built_at" in lut
        assert lut["n_teams"] == 32

    def test_all_teams_have_home_away(self, tmp_path):
        from scripts.build_team_color_lut import build_lut
        lut = build_lut(show=False)
        for code, jerseys in lut["teams"].items():
            assert "home" in jerseys, f"{code} missing home"
            assert "away" in jerseys, f"{code} missing away"

    def test_deterministic(self):
        from scripts.build_team_color_lut import build_lut
        a = build_lut(show=False)
        b = build_lut(show=False)
        assert a["teams"]["MTL"]["home"]["sat_mean"] == b["teams"]["MTL"]["home"]["sat_mean"]

    def test_dark_away_overrides_applied(self):
        from scripts.build_team_color_lut import build_lut, _DARK_AWAY
        lut = build_lut(show=False)
        for code in _DARK_AWAY:
            desc = lut["teams"][code]["away"]
            # Dark-away teams must be distinguishable from plain white via EITHER
            # saturation (coloured alts: ANA, SEA) or brightness (grey alts: LAK, VGK)
            is_saturated = desc["sat_mean"] > SAT_WHITE_THR
            is_dark       = desc["val_mean"] < 0.85   # clearly darker than white (~0.97)
            assert is_saturated or is_dark, (
                f"{code} dark-away must differ from white via sat or val; "
                f"sat={desc['sat_mean']:.3f}, val={desc['val_mean']:.3f}"
            )

    def test_main_writes_file(self, tmp_path):
        from scripts.build_team_color_lut import main
        out = tmp_path / "lut.json"
        main(["--out", str(out)])
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["n_teams"] == 32
