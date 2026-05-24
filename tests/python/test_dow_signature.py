"""Unit tests for models/dow_signature.py — Feature 4.21."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import pytest
from models.dow_signature import (
    DOW_SIGNATURE_SCHEMA, BROADCAST_CONTEXT_SCHEMA, MODEL_VERSION,
    DataMissingWarning, RIVALRY_PAIRS, DOW_NAMES,
    compute_dow_signature, compute_broadcast_context,
    write_dow_signature, read_dow_signature,
)

def _game_dates() -> pl.DataFrame:
    return pl.DataFrame([
        {"game_id": 1, "game_date": "2025-10-07", "day_of_week": 2, "day_name": "Tue"},
        {"game_id": 2, "game_date": "2025-10-11", "day_of_week": 6, "day_name": "Sat"},
        {"game_id": 3, "game_date": "2025-10-14", "day_of_week": 2, "day_name": "Tue"},
    ] * 10)  # Repeat to give enough games

def _pbp_goals(n_games: int = 30) -> pl.DataFrame:
    rows = []
    for gid in range(1, n_games + 1):
        rows.append({"game_id": gid, "event_type": "goal", "scorer_id": 100,
                      "event_owner_team_id": 10, "home_team_id": 10, "away_team_id": 8})
        if gid % 3 == 0:
            rows.append({"game_id": gid, "event_type": "goal", "scorer_id": 100,
                          "event_owner_team_id": 10, "home_team_id": 10, "away_team_id": 8})
        rows.append({"game_id": gid, "event_type": "shot", "scorer_id": None,
                      "event_owner_team_id": 10, "home_team_id": 10, "away_team_id": 8})
    return pl.DataFrame(rows)

def test_dow_schema() -> None:
    gd = pl.DataFrame([
        {"game_id": i, "game_date": f"2025-10-{7+i:02d}", "day_of_week": (i % 7) + 1, "day_name": DOW_NAMES[i % 7]}
        for i in range(1, 31)
    ])
    pbp = _pbp_goals(30)
    df = compute_dow_signature(pbp, gd, 2025, min_gp=5)
    assert set(df.columns) == set(DOW_SIGNATURE_SCHEMA.keys())
    if not df.is_empty():
        assert (df["model_version"] == MODEL_VERSION).all()

def test_z_scores_bounded() -> None:
    gd = pl.DataFrame([
        {"game_id": i, "game_date": f"2025-10-{7+i:02d}", "day_of_week": (i % 7) + 1, "day_name": DOW_NAMES[i % 7]}
        for i in range(1, 31)
    ])
    pbp = _pbp_goals(30)
    df = compute_dow_signature(pbp, gd, 2025, min_gp=5)
    for col in [f"{d.lower()}_z" for d in DOW_NAMES]:
        if col in df.columns:
            for v in df[col].to_list():
                assert -2.0 <= v <= 2.0, f"{col}={v} out of [-2, 2]"

def test_broadcast_context_rivalry_detection() -> None:
    gd = pl.DataFrame([{"game_id": 1, "game_date": "2025-10-07", "day_of_week": 2, "day_name": "Tue"}])
    pbp = pl.DataFrame([{"game_id": 1, "home_team_id": 10, "away_team_id": 8,
                          "event_type": "shot", "event_owner_team_id": 10}])
    bc = compute_broadcast_context(gd, {10: "TOR", 8: "MTL"}, pbp, 2025)
    assert not bc.is_empty()
    r = bc.row(0, named=True)
    assert r["is_rivalry"] is True
    assert "TOR" in r["rivalry_name"] and "MTL" in r["rivalry_name"]

def test_broadcast_context_non_rivalry() -> None:
    gd = pl.DataFrame([{"game_id": 1, "game_date": "2025-10-07", "day_of_week": 2, "day_name": "Tue"}])
    pbp = pl.DataFrame([{"game_id": 1, "home_team_id": 21, "away_team_id": 24,
                          "event_type": "shot", "event_owner_team_id": 21}])
    bc = compute_broadcast_context(gd, {21: "COL", 24: "ANA"}, pbp, 2025)
    assert bc.row(0, named=True)["is_rivalry"] is False

def test_empty_warns() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_dow_signature(pl.DataFrame(), pl.DataFrame(), 2025)
    assert df.is_empty()

def test_rivalry_pairs_symmetric() -> None:
    for a, b in RIVALRY_PAIRS:
        assert len(a) == 3 and len(b) == 3

def test_write_read_roundtrip(tmp_path: Path) -> None:
    gd = pl.DataFrame([
        {"game_id": i, "game_date": f"2025-10-{7+i:02d}", "day_of_week": (i % 7) + 1, "day_name": DOW_NAMES[i % 7]}
        for i in range(1, 31)
    ])
    pbp = _pbp_goals(30)
    df = compute_dow_signature(pbp, gd, 2025, min_gp=5)
    write_dow_signature(df, tmp_path / "dow_signature", 2025)
    rt = read_dow_signature(tmp_path, 2025)
    assert rt is not None and len(rt) == len(df)
