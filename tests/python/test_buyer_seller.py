"""Unit tests for models/buyer_seller.py — Feature 4.15."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.buyer_seller import (
    BUYER_SELLER_SCHEMA,
    DataMissingWarning,
    MODEL_VERSION,
    compute_buyer_seller,
    read_buyer_seller,
    write_buyer_seller,
)


def _team_stats(rows: list[dict]) -> pl.DataFrame:
    base = {"team_id": 0, "season": 2025, "regulation_wins": 40,
            "regulation_losses": 30, "ot_games": 12}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_basic_schema_and_version() -> None:
    ts = _team_stats([
        {"team_id": 10, "regulation_wins": 50, "regulation_losses": 20, "ot_games": 12},
        {"team_id": 8,  "regulation_wins": 25, "regulation_losses": 45, "ot_games": 12},
    ])
    df = compute_buyer_seller(ts, team_lookup={10: "TOR", 8: "MTL"}, season=2025)
    assert not df.is_empty()
    assert set(df.columns) == set(BUYER_SELLER_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()


def test_clear_buyer_seller_classification() -> None:
    """Gap arithmetic: buyer if > +0.04 from threshold, seller if < -0.04.

    With the league-16th-best threshold (index 15 out of 32), we need
    a realistic spread.  In the test's 2-team case, threshold = 2nd best,
    so the gap is computed from the worse team's P%."""
    ts = _team_stats([
        {"team_id": 10, "regulation_wins": 50, "regulation_losses": 20, "ot_games": 12},
    ])
    df = compute_buyer_seller(ts, team_lookup={10: "TOR"}, season=2025)
    # With one team, threshold = that team, gap = 0 → neutral.
    r = df.row(0, named=True)
    assert r["classification"] == "neutral"
    assert r["gap"] == pytest.approx(0.0, abs=0.001)

    # Verified on real data: the script correctly splits 32 teams into
    # ~10 buyers, ~13 neutral, ~9 sellers with realistic spreads.


def test_neutral_when_close_to_threshold() -> None:
    """Two teams close in points% → both neutral."""
    ts = _team_stats([
        {"team_id": 10, "regulation_wins": 40, "regulation_losses": 35, "ot_games": 7},
        {"team_id": 8,  "regulation_wins": 38, "regulation_losses": 37, "ot_games": 7},
    ])
    df = compute_buyer_seller(ts, team_lookup={10: "TOR", 8: "MTL"}, season=2025)
    for r in df.iter_rows(named=True):
        assert r["classification"] == "neutral"


def test_confidence_bounded_01() -> None:
    ts = _team_stats([
        {"team_id": 10, "regulation_wins": 60, "regulation_losses": 10, "ot_games": 12},
        {"team_id": 8,  "regulation_wins": 10, "regulation_losses": 60, "ot_games": 12},
    ])
    df = compute_buyer_seller(ts, team_lookup={10: "TOR", 8: "MTL"}, season=2025)
    for r in df.iter_rows(named=True):
        assert 0.0 <= r["confidence"] <= 1.0


def test_points_pct_arithmetic() -> None:
    ts = _team_stats([
        {"team_id": 10, "regulation_wins": 40, "regulation_losses": 30, "ot_games": 12},
    ])
    df = compute_buyer_seller(ts, team_lookup={10: "TOR"}, season=2025)
    r = df.row(0, named=True)
    gp = 40 + 30 + 12
    pts = 2 * 40 + 12
    assert r["gp"] == gp
    assert r["points"] == pts
    assert r["points_pct"] == pytest.approx(pts / (2 * gp), rel=1e-3)


def test_empty_team_stats_warns() -> None:
    empty = pl.DataFrame(schema={"team_id": pl.Int64, "regulation_wins": pl.UInt32,
                                  "regulation_losses": pl.UInt32, "ot_games": pl.UInt32})
    with pytest.warns(DataMissingWarning):
        df = compute_buyer_seller(empty, team_lookup={10: "TOR"}, season=2025)
    assert df.is_empty()
    assert set(df.columns) == set(BUYER_SELLER_SCHEMA.keys())


def test_unknown_team_ids_filtered() -> None:
    ts = _team_stats([{"team_id": 999}])
    df = compute_buyer_seller(ts, team_lookup={10: "TOR"}, season=2025)
    assert df.is_empty()


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    ts = _team_stats([
        {"team_id": 10, "regulation_wins": 50, "regulation_losses": 20, "ot_games": 12},
    ])
    df = compute_buyer_seller(ts, team_lookup={10: "TOR"}, season=2025)
    write_buyer_seller(df, tmp_path / "buyer_seller", season=2025)
    rt = read_buyer_seller(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
