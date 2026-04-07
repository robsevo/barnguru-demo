"""Tests for models/nhle_model.py — Feature 2.24."""
from __future__ import annotations

import math
import warnings

import polars as pl
import pytest

from models.nhle_model import (
    AGE_ADJ_CAP,
    AGE_ADJ_PIVOT,
    LEAGUE_FACTORS,
    MODEL_VERSION,
    NHLE_PROJECTION_SCHEMA,
    NHLeModel,
    _draft_bonus,
    age_adjustment,
    compute_nhle_ppg,
    league_factor,
    write_nhle_projections,
)
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prospect(pid=1, league="AHL", goals=20, assists=30, gp=60, age=20,
              player_name="Test Guy", team="MIL", position="C", draft_round=1):
    return {
        "player_id": pid, "player_name": player_name, "team": team,
        "position": position, "league": league,
        "goals": goals, "assists": assists, "gp": gp,
        "age": age, "draft_round": draft_round,
    }


def _make_prospects(rows: list[dict]) -> pl.DataFrame:
    base = {
        "player_id": pl.Int64, "player_name": pl.Utf8, "team": pl.Utf8,
        "position": pl.Utf8, "league": pl.Utf8,
        "goals": pl.Float64, "assists": pl.Float64, "gp": pl.Int64,
        "age": pl.Float64, "draft_round": pl.Int64,
    }
    if not rows:
        return pl.DataFrame(schema=base)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# league_factor
# ---------------------------------------------------------------------------

class TestLeagueFactor:
    def test_nhl_is_one(self):
        assert league_factor("NHL") == 1.0

    def test_ahl(self):
        assert league_factor("AHL") == pytest.approx(0.44)

    def test_khl(self):
        assert league_factor("KHL") == pytest.approx(0.82)

    def test_ohl(self):
        assert league_factor("OHL") == pytest.approx(0.29)

    def test_case_insensitive(self):
        assert league_factor("ahl") == league_factor("AHL")
        assert league_factor("ohl") == league_factor("OHL")

    def test_unknown_league_warns_returns_one(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = league_factor("SUPERLEAGUE2099")
        assert val == 1.0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_all_factors_in_range(self):
        for name, val in LEAGUE_FACTORS.items():
            assert 0 < val <= 1.0, f"{name} factor {val} out of range"


# ---------------------------------------------------------------------------
# age_adjustment
# ---------------------------------------------------------------------------

class TestAgeAdjustment:
    def test_age_21_returns_one(self):
        assert age_adjustment(21) == pytest.approx(1.0)

    def test_age_above_21_returns_one(self):
        assert age_adjustment(25) == pytest.approx(1.0)

    def test_age_20_gives_bonus(self):
        assert age_adjustment(20) == pytest.approx(1.05)

    def test_age_18_gives_larger_bonus(self):
        assert age_adjustment(18) == pytest.approx(1.15)

    def test_cap_enforced(self):
        # age 0 would give 1 + 0.05×21 = 2.05 but cap = 1.25
        assert age_adjustment(0) == pytest.approx(AGE_ADJ_CAP)

    def test_monotone_younger_is_higher(self):
        assert age_adjustment(17) > age_adjustment(19)
        assert age_adjustment(19) > age_adjustment(21)


# ---------------------------------------------------------------------------
# compute_nhle_ppg
# ---------------------------------------------------------------------------

class TestComputeNhlePpg:
    def test_nhl_player_raw_equals_nhle(self):
        # In NHL, factor=1.0, age=21 → adj=1.0; nhle = raw_ppg
        val = compute_nhle_ppg(20, 30, 60, "NHL", 21)
        assert val == pytest.approx(50 / 60)

    def test_ahl_applied_correctly(self):
        # 50 pts / 60 GP = 0.833 PPG × 0.44 = 0.367 NHLe (at age 21)
        val = compute_nhle_ppg(20, 30, 60, "AHL", 21)
        assert val == pytest.approx(0.44 * 50 / 60)

    def test_age_bonus_applied(self):
        val_18 = compute_nhle_ppg(20, 30, 60, "OHL", 18)
        val_21 = compute_nhle_ppg(20, 30, 60, "OHL", 21)
        assert val_18 > val_21

    def test_zero_gp_returns_none(self):
        assert compute_nhle_ppg(10, 20, 0, "AHL", 20) is None

    def test_negative_gp_returns_none(self):
        assert compute_nhle_ppg(10, 20, -1, "AHL", 20) is None


# ---------------------------------------------------------------------------
# _draft_bonus
# ---------------------------------------------------------------------------

class TestDraftBonus:
    def test_round1_positive(self):
        assert _draft_bonus(1) > 0

    def test_round2_positive(self):
        assert _draft_bonus(2) > 0

    def test_round1_greater_than_round2(self):
        assert _draft_bonus(1) > _draft_bonus(2)

    def test_high_round_small_or_zero(self):
        assert _draft_bonus(7) == pytest.approx(0.0)

    def test_none_returns_zero(self):
        assert _draft_bonus(None) == pytest.approx(0.0)

    def test_zero_draft_round_is_undrafted(self):
        assert _draft_bonus(0) < 0


# ---------------------------------------------------------------------------
# NHLeModel.fit — basic
# ---------------------------------------------------------------------------

class TestNHLeModelFit:
    def test_empty_returns_empty_schema(self):
        model = NHLeModel()
        df = model.fit(_make_prospects([]))
        assert df.is_empty()
        for col in NHLE_PROJECTION_SCHEMA:
            assert col in df.columns

    def test_single_prospect(self):
        pros = _make_prospects([_prospect(pid=1, league="AHL", goals=20, assists=30, gp=60, age=20)])
        model = NHLeModel()
        df = model.fit(pros)
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["player_id"] == 1
        assert row["league"] == "AHL"
        assert row["league_factor"] == pytest.approx(0.44)
        assert row["model_version"] == MODEL_VERSION

    def test_nhle_ppg_computed(self):
        pros = _make_prospects([_prospect(league="OHL", goals=30, assists=50, gp=60, age=18)])
        model = NHLeModel()
        df = model.fit(pros)
        row = df.row(0, named=True)
        expected = compute_nhle_ppg(30, 50, 60, "OHL", 18)
        assert row["nhle_ppg"] == pytest.approx(expected)

    def test_output_columns_match_schema(self):
        pros = _make_prospects([_prospect()])
        model = NHLeModel()
        df = model.fit(pros)
        assert set(df.columns) == set(NHLE_PROJECTION_SCHEMA.keys())

    def test_p_nhler_in_range(self):
        pros = _make_prospects([_prospect(league="AHL", goals=25, assists=35, gp=60, age=21)])
        model = NHLeModel()
        df = model.fit(pros)
        row = df.row(0, named=True)
        assert 0.0 <= row["p_nhler"] <= 1.0

    def test_p_star_in_range(self):
        pros = _make_prospects([_prospect()])
        model = NHLeModel()
        df = model.fit(pros)
        row = df.row(0, named=True)
        assert 0.0 <= row["p_star"] <= 1.0

    def test_p_star_leq_p_nhler(self):
        # P(star) ≤ P(NHLer) for any prospect
        pros = _make_prospects([
            _prospect(pid=1, league="AHL", goals=20, assists=30, gp=60, age=20),
            _prospect(pid=2, league="OHL", goals=40, assists=60, gp=60, age=18),
        ])
        model = NHLeModel()
        df = model.fit(pros)
        for row in df.to_dicts():
            assert row["p_star"] <= row["p_nhler"] + 1e-9

    def test_high_scorer_higher_probability(self):
        pros = _make_prospects([
            _prospect(pid=1, league="AHL", goals=10, assists=10, gp=60, age=21),  # low scorer
            _prospect(pid=2, league="AHL", goals=40, assists=40, gp=60, age=21),  # elite scorer
        ])
        model = NHLeModel()
        df = model.fit(pros)
        rows = {r["player_id"]: r for r in df.to_dicts()}
        assert rows[2]["p_nhler"] > rows[1]["p_nhler"]
        assert rows[2]["p_star"] >= rows[1]["p_star"]

    def test_missing_required_col_raises(self):
        bad = pl.DataFrame({"player_id": [1], "league": ["AHL"], "goals": [10.0]})
        model = NHLeModel()
        with pytest.raises(ValueError, match="missing columns"):
            model.fit(bad)

    def test_multiple_players(self):
        rows = [_prospect(pid=i, league="AHL") for i in range(5)]
        pros = _make_prospects(rows)
        model = NHLeModel()
        df = model.fit(pros)
        assert len(df) == 5

    def test_nhl_prospect_high_p_nhler(self):
        # A player already in NHL at high production should have very high P(NHLer)
        pros = _make_prospects([_prospect(league="NHL", goals=30, assists=50, gp=80, age=24)])
        model = NHLeModel()
        df = model.fit(pros)
        row = df.row(0, named=True)
        assert row["p_nhler"] > 0.8


# ---------------------------------------------------------------------------
# get_projection / get_nhle_ppg / get_p_nhler
# ---------------------------------------------------------------------------

class TestLookup:
    def _trained(self):
        pros = _make_prospects([_prospect(pid=1, league="AHL", goals=20, assists=30, gp=60, age=20)])
        model = NHLeModel()
        model.fit(pros)
        return model

    def test_get_projection_returns_dict(self):
        model = self._trained()
        proj = model.get_projection(1)
        assert isinstance(proj, dict)
        assert proj["player_id"] == 1

    def test_get_projection_unknown_returns_none(self):
        model = NHLeModel()
        assert model.get_projection(9999) is None

    def test_get_nhle_ppg(self):
        model = self._trained()
        val = model.get_nhle_ppg(1)
        assert isinstance(val, float)
        assert val > 0

    def test_get_nhle_ppg_unknown_returns_none(self):
        model = NHLeModel()
        assert model.get_nhle_ppg(9999) is None

    def test_get_p_nhler_unknown_returns_none(self):
        model = NHLeModel()
        assert model.get_p_nhler(9999) is None


# ---------------------------------------------------------------------------
# Classifier training path (≥30 records)
# ---------------------------------------------------------------------------

class TestClassifierFit:
    def test_classifier_fit_with_many_records(self):
        rows = [_prospect(pid=i, league="AHL", goals=i % 40, assists=(i + 5) % 35,
                          gp=60, age=19 + (i % 4)) for i in range(50)]
        pros = _make_prospects(rows)
        model = NHLeModel()
        df = model.fit(pros)
        # All probabilities should still be in valid range
        for row in df.to_dicts():
            assert 0 <= row["p_nhler"] <= 1.0
            assert 0 <= row["p_star"] <= 1.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        pros = _make_prospects([_prospect(pid=1, league="AHL", goals=20, assists=30, gp=60, age=20)])
        model = NHLeModel()
        model.fit(pros)
        path = tmp_path / "nhle.pkl"
        model.save(path)
        loaded = NHLeModel.load(path)
        assert loaded.get_nhle_ppg(1) == pytest.approx(model.get_nhle_ppg(1))

    def test_save_creates_parent_dirs(self, tmp_path):
        model = NHLeModel()
        model.save(tmp_path / "nested" / "dir" / "nhle.pkl")
        assert (tmp_path / "nested" / "dir" / "nhle.pkl").exists()


# ---------------------------------------------------------------------------
# write_nhle_projections
# ---------------------------------------------------------------------------

class TestWriteNhleProjections:
    def test_writes_parquet(self, tmp_path):
        pros = _make_prospects([_prospect()])
        model = NHLeModel()
        df = model.fit(pros)
        path = write_nhle_projections(df, tmp_path / "out", season=2025)
        assert path.exists()
        assert path.name == "nhle_projections_2025.parquet"

    def test_readable_back(self, tmp_path):
        pros = _make_prospects([_prospect()])
        model = NHLeModel()
        df = model.fit(pros)
        path = write_nhle_projections(df, tmp_path / "out", season=2025)
        back = pl.read_parquet(path)
        assert len(back) == 1

    def test_creates_output_dir(self, tmp_path):
        df = pl.DataFrame(schema={c: d for c, d in NHLE_PROJECTION_SCHEMA.items()})
        write_nhle_projections(df, tmp_path / "deep" / "dir", season=2024)
        assert (tmp_path / "deep" / "dir").exists()
