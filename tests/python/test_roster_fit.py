"""Unit tests for models/roster_fit.py — Feature 4.12."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.roster_fit import (
    DataMissingWarning,
    MODEL_VERSION,
    ROSTER_FIT_SCHEMA,
    _archetype_shares,
    _fit_for_team,
    compute_roster_fit,
    read_roster_fit,
    write_roster_fit,
)


def _style_df() -> pl.DataFrame:
    """One-team coaching style row with known ranks."""
    return pl.DataFrame([{
        "team": "TOR", "season": 2025,
        "forecheck_aggression_raw": 1.0, "forecheck_aggression_rank": 0.8,
        "dz_structure_raw": 0.5,         "dz_structure_rank": 0.6,
        "pace_raw": 1.5,                 "pace_rank": 0.7,
        "physicality_raw": 0.3,          "physicality_rank": 0.4,
        "oz_structure_raw": 0.3,         "oz_structure_rank": 0.5,
        "nz_tendency_raw": float("nan"), "nz_tendency_rank": float("nan"),
        "line_match_raw": 0.2,           "line_match_rank": 0.3,
        "st_aggression_raw": 1.2,        "st_aggression_rank": 0.9,
        "model_version": "coaching_style_v1",
    }])


def _archetype_df() -> pl.DataFrame:
    return pl.DataFrame([
        {"player_id": 1, "season": 2025, "archetype": "Elite Two-Way",    "distance": 0.5},
        {"player_id": 2, "season": 2025, "archetype": "Defensive Anchor", "distance": 0.4},
        {"player_id": 3, "season": 2025, "archetype": "Elite Scorer",     "distance": 0.3},
    ])


def _rapm_df() -> pl.DataFrame:
    return pl.DataFrame([
        {"player_id": 1, "team": "10", "toi_ev": 500.0},
        {"player_id": 2, "team": "10", "toi_ev": 400.0},
        {"player_id": 3, "team": "10", "toi_ev": 300.0},
    ])


def test_basic_schema_and_version() -> None:
    df = compute_roster_fit(
        style_df     = _style_df(),
        archetype_df = _archetype_df(),
        rapm_df      = _rapm_df(),
        team_lookup  = {10: "TOR"},
        season       = 2025,
    )
    assert not df.is_empty()
    assert set(df.columns) == set(ROSTER_FIT_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()


def test_fit_score_bounded_01() -> None:
    df = compute_roster_fit(
        style_df     = _style_df(),
        archetype_df = _archetype_df(),
        rapm_df      = _rapm_df(),
        team_lookup  = {10: "TOR"},
        season       = 2025,
    )
    for r in df.iter_rows(named=True):
        assert 0.0 <= r["fit_score"] <= 1.0


def test_archetype_shares_sum_to_one() -> None:
    shares = _archetype_shares(
        archetype_df = _archetype_df(),
        rapm_df      = _rapm_df(),
        team_lookup  = {10: "TOR"},
    )
    for team, archmap in shares.items():
        total = sum(archmap.values())
        assert total == pytest.approx(1.0, abs=0.001)


def test_fit_for_team_with_all_supporting_archetypes() -> None:
    """If every style dim is maximally supported → fit ≈ high."""
    # 100% Elite Two-Way supports most dims
    fit, _, _ = _fit_for_team(
        style_row       = {d: 1.0 for d in ["forecheck_aggression", "dz_structure", "pace",
                                              "physicality", "oz_structure", "nz_tendency",
                                              "line_match", "st_aggression"]},
        archetype_share = {"Elite Two-Way": 1.0},
    )
    assert fit > 0.3  # it supports 6/8 dims → weighted avg > 0


def test_mismatch_dim_found() -> None:
    df = compute_roster_fit(
        style_df     = _style_df(),
        archetype_df = _archetype_df(),
        rapm_df      = _rapm_df(),
        team_lookup  = {10: "TOR"},
        season       = 2025,
    )
    row = df.row(0, named=True)
    assert row["mismatch_dim"] != ""
    assert 0.0 <= row["mismatch_support"] <= 1.0


def test_empty_style_warns_and_empty() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_roster_fit(
            style_df     = pl.DataFrame(schema={c: t for c, t in ROSTER_FIT_SCHEMA.items()}).clear(),
            archetype_df = _archetype_df(),
            rapm_df      = _rapm_df(),
            team_lookup  = {10: "TOR"},
            season       = 2025,
        )
    assert df.is_empty()


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    df = compute_roster_fit(
        style_df     = _style_df(),
        archetype_df = _archetype_df(),
        rapm_df      = _rapm_df(),
        team_lookup  = {10: "TOR"},
        season       = 2025,
    )
    write_roster_fit(df, tmp_path / "roster_fit", season=2025)
    rt = read_roster_fit(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
    assert set(rt.columns) == set(ROSTER_FIT_SCHEMA.keys())
