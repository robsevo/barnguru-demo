"""Unit tests for models/playoff_elimination.py — Feature 4.20."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import pytest
from models.playoff_elimination import (
    PLAYOFF_ELIMINATION_SCHEMA, MODEL_VERSION, DataMissingWarning,
    _playoff_prob, _elimination_drag,
    compute_playoff_elimination, write_playoff_elimination, read_playoff_elimination,
)

def _ts(rows: list[dict]) -> pl.DataFrame:
    base = {"team_id": 10, "season": 2025, "regulation_wins": 40,
            "regulation_losses": 30, "ot_games": 12}
    return pl.DataFrame([{**base, **r} for r in rows])

def test_schema_and_version() -> None:
    df = compute_playoff_elimination(_ts([{}]), {10: "TOR"}, 2025)
    assert set(df.columns) == set(PLAYOFF_ELIMINATION_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()

def test_good_team_no_drag() -> None:
    df = compute_playoff_elimination(_ts([{"regulation_wins": 50, "regulation_losses": 20, "ot_games": 10}]), {10: "TOR"}, 2025)
    r = df.row(0, named=True)
    assert r["elimination_drag"] == 0.0
    assert r["efficiency_multiplier"] == 1.0
    assert r["playoff_prob"] > 0.5

def test_bad_team_has_drag() -> None:
    df = compute_playoff_elimination(_ts([{"regulation_wins": 20, "regulation_losses": 55, "ot_games": 5}]), {10: "TOR"}, 2025)
    r = df.row(0, named=True)
    assert r["elimination_drag"] > 0
    assert r["efficiency_multiplier"] < 1.0

def test_playoff_prob_bounded() -> None:
    for w in (10, 30, 50):
        prob = _playoff_prob(2 * w / (2 * 82), 82 - w - 30)
        assert 0.0 <= prob <= 1.0

def test_elimination_drag_zero_when_not_triggered() -> None:
    assert _elimination_drag(0.50, 40) == 0.0
    assert _elimination_drag(0.30, 40) == 0.0
    assert _elimination_drag(0.20, 31) == 0.0

def test_elimination_drag_positive_when_triggered() -> None:
    assert _elimination_drag(0.10, 15) > 0
    assert _elimination_drag(0.05, 10) > 0

def test_drag_bounded_01() -> None:
    assert 0.0 <= _elimination_drag(0.01, 5) <= 1.0
    assert 0.0 <= _elimination_drag(0.24, 29) <= 1.0

def test_empty_warns() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_playoff_elimination(pl.DataFrame(), {10: "TOR"}, 2025)
    assert df.is_empty()

def test_write_read_roundtrip(tmp_path: Path) -> None:
    df = compute_playoff_elimination(_ts([{}]), {10: "TOR"}, 2025)
    write_playoff_elimination(df, tmp_path / "playoff_elimination", 2025)
    rt = read_playoff_elimination(tmp_path, 2025)
    assert rt is not None and len(rt) == len(df)
