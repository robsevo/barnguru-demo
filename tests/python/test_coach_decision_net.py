"""Unit tests for models/coach_decision_net.py — Feature 4.17."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import pytest
from models.coach_decision_net import (
    COACH_DECISION_SCHEMA, DECISION_DIMENSIONS, MODEL_VERSION,
    DataMissingWarning, compute_coach_decision_net,
    write_coach_decision_net, read_coach_decision_net,
)

def _coaches() -> list[dict]:
    return [
        {"team": "TOR", "name": "Craig Berube", "first_named_head_coach": "2024-05-17", "notes": ""},
        {"team": "MTL", "name": "Martin St. Louis", "first_named_head_coach": "2022-02-09", "notes": ""},
    ]

def _empty_df() -> pl.DataFrame:
    return pl.DataFrame()

def test_schema_and_version() -> None:
    df = compute_coach_decision_net(
        coaches=_coaches(), timeout_df=_empty_df(), goalie_pull_df=_empty_df(),
        line_deploy_df=_empty_df(), st_deploy_df=_empty_df(),
        penalty_df=_empty_df(), line_match_df=_empty_df(), season=2025,
    )
    assert set(df.columns) == set(COACH_DECISION_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()
    assert len(df) == 2

def test_all_dimensions_bounded_01() -> None:
    df = compute_coach_decision_net(
        coaches=_coaches(), timeout_df=_empty_df(), goalie_pull_df=_empty_df(),
        line_deploy_df=_empty_df(), st_deploy_df=_empty_df(),
        penalty_df=_empty_df(), line_match_df=_empty_df(), season=2025,
    )
    for dim in DECISION_DIMENSIONS + ["overall_aggression"]:
        for v in df[dim].to_list():
            assert 0.0 <= v <= 1.0, f"{dim}={v} out of [0,1]"

def test_overall_is_mean_of_dims() -> None:
    df = compute_coach_decision_net(
        coaches=_coaches(), timeout_df=_empty_df(), goalie_pull_df=_empty_df(),
        line_deploy_df=_empty_df(), st_deploy_df=_empty_df(),
        penalty_df=_empty_df(), line_match_df=_empty_df(), season=2025,
    )
    for r in df.iter_rows(named=True):
        expected = sum(r[d] for d in DECISION_DIMENSIONS) / len(DECISION_DIMENSIONS)
        assert r["overall_aggression"] == pytest.approx(expected, abs=0.01)

def test_empty_coaches_warns() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_coach_decision_net(
            coaches=[], timeout_df=_empty_df(), goalie_pull_df=_empty_df(),
            line_deploy_df=_empty_df(), st_deploy_df=_empty_df(),
            penalty_df=_empty_df(), line_match_df=_empty_df(), season=2025,
        )
    assert df.is_empty()

def test_penalty_data_affects_discipline() -> None:
    penalty_df = pl.DataFrame([
        {"team": "TOR", "penalties_taken_per_game": 4.5, "season": 2025, "model_version": "v1",
         "n_games": 82, "n_penalties_taken": 369, "n_pp_opportunities": 300,
         "pim_total": 800, "pp_opps_per_game": 3.66, "pim_per_game": 9.76, "ref_dim": "team-only"},
        {"team": "MTL", "penalties_taken_per_game": 2.5, "season": 2025, "model_version": "v1",
         "n_games": 82, "n_penalties_taken": 205, "n_pp_opportunities": 300,
         "pim_total": 500, "pp_opps_per_game": 3.66, "pim_per_game": 6.10, "ref_dim": "team-only"},
    ])
    df = compute_coach_decision_net(
        coaches=_coaches(), timeout_df=_empty_df(), goalie_pull_df=_empty_df(),
        line_deploy_df=_empty_df(), st_deploy_df=_empty_df(),
        penalty_df=penalty_df, line_match_df=_empty_df(), season=2025,
    )
    tor = df.filter(pl.col("team") == "TOR").row(0, named=True)
    mtl = df.filter(pl.col("team") == "MTL").row(0, named=True)
    # MTL takes fewer penalties → higher discipline rank
    assert mtl["penalty_discipline"] > tor["penalty_discipline"]

def test_write_read_roundtrip(tmp_path: Path) -> None:
    df = compute_coach_decision_net(
        coaches=_coaches(), timeout_df=_empty_df(), goalie_pull_df=_empty_df(),
        line_deploy_df=_empty_df(), st_deploy_df=_empty_df(),
        penalty_df=_empty_df(), line_match_df=_empty_df(), season=2025,
    )
    write_coach_decision_net(df, tmp_path / "coach_decision_net", 2025)
    rt = read_coach_decision_net(tmp_path, 2025)
    assert rt is not None and len(rt) == len(df)
