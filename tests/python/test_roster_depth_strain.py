"""Tests for Feature 3.16 — RosterDepthStrain."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.roster_depth_strain import (
    FULL_STRAIN_SECS,
    ROSTER_DEPTH_STRAIN_SCHEMA,
    RosterDepthStrain,
    strain_score,
    write_roster_depth_strain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _roster(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "player_id":         pl.Int64,
        "team":              pl.Utf8,
        "position":          pl.Utf8,
        "is_on_ir":          pl.Boolean,
        "baseline_toi_secs": pl.Int64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def _full_team(team: str, ir_indices: tuple[int, ...] = (),
               ir_baseline_secs: int = 1200,
               healthy_baseline_secs: int = 900,
               n_skaters: int = 18) -> list[dict]:
    """Generate a 18-skater team with selected indices on IR."""
    rows = []
    for i in range(n_skaters):
        on_ir = i in ir_indices
        rows.append({
            "player_id":         1000 * (ord(team[0]) - ord("A") + 1) + i,
            "team":              team,
            "position":          "F" if i < 12 else "D",
            "is_on_ir":          on_ir,
            "baseline_toi_secs": ir_baseline_secs if on_ir else healthy_baseline_secs,
        })
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_columns_warns(self):
        bad = pl.DataFrame({"foo": [1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = RosterDepthStrain().compute(bad, "2026-05-17")
        assert len(out) == 0
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_empty_input_returns_empty_schema(self):
        out = RosterDepthStrain().compute(_roster([]), "2026-05-17")
        for col in ROSTER_DEPTH_STRAIN_SCHEMA:
            assert col in out.columns
        assert len(out) == 0

    def test_output_schema(self):
        df = _roster(_full_team("MTL"))
        out = RosterDepthStrain().compute(df, "2026-05-17")
        assert set(out.columns) == set(ROSTER_DEPTH_STRAIN_SCHEMA.keys())

    def test_bad_as_of_date_raises(self):
        df = _roster(_full_team("MTL"))
        with pytest.raises(ValueError):
            RosterDepthStrain().compute(df, "not-a-date")

    def test_bad_params_rejected(self):
        with pytest.raises(ValueError):
            RosterDepthStrain(full_strain_secs=0)
        with pytest.raises(ValueError):
            RosterDepthStrain(full_strain_secs=-10)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_full_strain_positive(self):
        assert FULL_STRAIN_SECS > 0


# ---------------------------------------------------------------------------
# Strain-score formula
# ---------------------------------------------------------------------------

class TestStrainScore:
    def test_zero_extra_is_zero_score(self):
        assert strain_score(0.0) == pytest.approx(0.0)

    def test_negative_extra_clamps_to_zero(self):
        assert strain_score(-50.0) == pytest.approx(0.0)

    def test_full_strain_secs_is_one(self):
        assert strain_score(FULL_STRAIN_SECS) == pytest.approx(1.0)

    def test_beyond_full_strain_clamps_to_one(self):
        assert strain_score(FULL_STRAIN_SECS * 5) == pytest.approx(1.0)

    def test_linear_below_one(self):
        assert strain_score(FULL_STRAIN_SECS / 2) == pytest.approx(0.5)

    def test_unit_interval(self):
        for v in (0.0, 30.0, 60.0, 120.0, 180.0, 300.0, 1000.0):
            s = strain_score(v)
            assert 0.0 <= s <= 1.0

    def test_invalid_full_strain_raises(self):
        with pytest.raises(ValueError):
            strain_score(60.0, full_strain_secs=0.0)
        with pytest.raises(ValueError):
            strain_score(60.0, full_strain_secs=-1.0)


# ---------------------------------------------------------------------------
# Compute path
# ---------------------------------------------------------------------------

class TestCompute:
    def test_no_ir_zero_strain(self):
        df = _roster(_full_team("MTL", ir_indices=()))
        out = RosterDepthStrain().compute(df, "2026-05-17").to_dicts()
        assert len(out) == 18
        for r in out:
            assert r["is_on_ir"] is False
            assert r["ir_skater_count"] == 0
            assert r["healthy_skater_count"] == 18
            assert r["ir_minutes_secs"] == pytest.approx(0.0)
            assert r["extra_per_healthy_secs"] == pytest.approx(0.0)
            assert r["team_strain_score"] == pytest.approx(0.0)
            assert r["player_strain_score"] == pytest.approx(0.0)

    def test_one_ir_distributes_minutes(self):
        # 1 IR player with 1200s baseline → 1200s spread across 17 healthy = ≈70.6s each.
        df = _roster(_full_team("MTL", ir_indices=(0,)))
        out = RosterDepthStrain().compute(df, "2026-05-17").to_dicts()
        ir_row = next(r for r in out if r["is_on_ir"])
        healthy = [r for r in out if not r["is_on_ir"]]

        assert ir_row["ir_skater_count"] == 1
        assert ir_row["healthy_skater_count"] == 17
        assert ir_row["ir_minutes_secs"] == pytest.approx(1200.0)
        assert ir_row["player_strain_score"] == pytest.approx(0.0)   # IR carries none

        for r in healthy:
            assert r["extra_per_healthy_secs"] == pytest.approx(1200.0 / 17)
            assert r["team_strain_score"] == pytest.approx(
                strain_score(1200.0 / 17)
            )
            assert r["player_strain_score"] == r["team_strain_score"]

    def test_many_ir_caps_at_one(self):
        # 8 IR players × 1200s baseline = 9600s, spread across 10 healthy =
        # 960s per healthy → well above FULL_STRAIN_SECS → clamped to 1.0.
        df = _roster(_full_team("MTL", ir_indices=tuple(range(8))))
        out = RosterDepthStrain().compute(df, "2026-05-17").to_dicts()
        healthy = [r for r in out if not r["is_on_ir"]]
        for r in healthy:
            assert r["player_strain_score"] == pytest.approx(1.0)

    def test_team_strain_is_uniform(self):
        # Every healthy player on the same team gets the same strain score.
        df = _roster(_full_team("MTL", ir_indices=(0, 1, 12)))
        out = RosterDepthStrain().compute(df, "2026-05-17").to_dicts()
        healthy_scores = {r["player_strain_score"] for r in out if not r["is_on_ir"]}
        assert len(healthy_scores) == 1

    def test_two_teams_independent(self):
        # MTL has 3 IR, TBL has 0. Healthy MTL skaters strained; TBL not.
        rows = _full_team("MTL", ir_indices=(0, 1, 2)) + _full_team("TBL")
        out = RosterDepthStrain().compute(_roster(rows), "2026-05-17").to_dicts()
        mtl_healthy = [r for r in out if r["team"] == "MTL" and not r["is_on_ir"]]
        tbl_healthy = [r for r in out if r["team"] == "TBL" and not r["is_on_ir"]]
        assert all(r["player_strain_score"] > 0 for r in mtl_healthy)
        assert all(r["player_strain_score"] == pytest.approx(0.0) for r in tbl_healthy)

    def test_goalies_excluded(self):
        rows = _full_team("MTL")
        rows.append({
            "player_id": 8999, "team": "MTL", "position": "G",
            "is_on_ir": True,  "baseline_toi_secs": 3600,
        })
        out = RosterDepthStrain().compute(_roster(rows), "2026-05-17").to_dicts()
        # Goalie not in output, and IR count for MTL stays at 0.
        assert not any(r["player_id"] == 8999 for r in out)
        assert all(r["ir_skater_count"] == 0 for r in out if r["team"] == "MTL")

    def test_duplicate_player_in_team_deduped(self):
        rows = [
            {"player_id": 1, "team": "MTL", "position": "F",
             "is_on_ir": True, "baseline_toi_secs": 1200},
            {"player_id": 1, "team": "MTL", "position": "F",
             "is_on_ir": False, "baseline_toi_secs": 900},
        ]
        out = RosterDepthStrain().compute(_roster(rows), "2026-05-17").to_dicts()
        assert len(out) == 1

    def test_negative_baseline_treated_as_zero(self):
        rows = [
            {"player_id": 1, "team": "MTL", "position": "F",
             "is_on_ir": True, "baseline_toi_secs": -500},
            {"player_id": 2, "team": "MTL", "position": "F",
             "is_on_ir": False, "baseline_toi_secs": 900},
        ]
        out = RosterDepthStrain().compute(_roster(rows), "2026-05-17").to_dicts()
        # IR baseline clipped to 0 → no minutes to redistribute → no strain.
        ir_row = next(r for r in out if r["is_on_ir"])
        assert ir_row["ir_minutes_secs"] == pytest.approx(0.0)
        h_row = next(r for r in out if not r["is_on_ir"])
        assert h_row["player_strain_score"] == pytest.approx(0.0)

    def test_player_strain_in_unit_interval(self):
        df = _roster(_full_team("MTL", ir_indices=(0, 1, 2, 5, 8)))
        out = RosterDepthStrain().compute(df, "2026-05-17")
        for r in out.to_dicts():
            assert 0.0 <= r["player_strain_score"] <= 1.0
            assert 0.0 <= r["team_strain_score"] <= 1.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_parquet(self, tmp_path):
        df = _roster(_full_team("MTL", ir_indices=(0,)))
        out = RosterDepthStrain().compute(df, "2026-05-17")
        path = write_roster_depth_strain(out, tmp_path, "2026-05-17")
        assert path.exists()
        loaded = pl.read_parquet(path)
        for col in ROSTER_DEPTH_STRAIN_SCHEMA:
            assert col in loaded.columns

    def test_filename_contains_date(self, tmp_path):
        out = RosterDepthStrain().compute(_roster([]), "2026-05-17")
        path = write_roster_depth_strain(out, tmp_path, "2026-05-17")
        assert "2026-05-17" in path.name
