"""Unit tests for models/seller_motivation.py — Feature 4.16."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import pytest
from models.seller_motivation import (
    SELLER_MOTIVATION_SCHEMA, MODEL_VERSION, DataMissingWarning,
    compute_seller_motivation, write_seller_motivation, read_seller_motivation,
)

def _bs(rows: list[dict]) -> pl.DataFrame:
    base = {"team": "TOR", "season": 2025, "gp": 75, "wins": 35, "ot_games": 10,
            "losses": 30, "points": 80, "points_pct": 0.533,
            "threshold": 0.500, "gap": 0.033, "classification": "neutral",
            "confidence": 0.33, "model_version": "buyer_seller_v1"}
    return pl.DataFrame([{**base, **r} for r in rows])

def test_schema_and_version() -> None:
    df = compute_seller_motivation(_bs([{"team": "TOR", "classification": "seller", "confidence": 0.8, "gp": 78}]), season=2025)
    assert set(df.columns) == set(SELLER_MOTIVATION_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()

def test_seller_post_deadline_has_drag() -> None:
    df = compute_seller_motivation(_bs([{"team": "CHI", "classification": "seller", "confidence": 0.9, "gp": 75}]), season=2025)
    r = df.filter(pl.col("team") == "CHI").row(0, named=True)
    assert r["seller_drag"] > 0
    assert r["efficiency_multiplier"] < 1.0
    assert r["games_since_deadline"] == 10

def test_buyer_has_no_drag() -> None:
    df = compute_seller_motivation(_bs([{"team": "COL", "classification": "buyer", "confidence": 0.9, "gp": 75}]), season=2025)
    r = df.filter(pl.col("team") == "COL").row(0, named=True)
    assert r["seller_drag"] == 0.0
    assert r["efficiency_multiplier"] == 1.0

def test_drag_decays_with_games() -> None:
    early = compute_seller_motivation(_bs([{"team": "X", "classification": "seller", "confidence": 1.0, "gp": 67}]), season=2025)
    late  = compute_seller_motivation(_bs([{"team": "X", "classification": "seller", "confidence": 1.0, "gp": 80}]), season=2025)
    assert early.row(0, named=True)["seller_drag"] > late.row(0, named=True)["seller_drag"]

def test_drag_bounded_01() -> None:
    df = compute_seller_motivation(_bs([{"team": "X", "classification": "seller", "confidence": 1.0, "gp": 66}]), season=2025)
    assert 0.0 <= df.row(0, named=True)["seller_drag"] <= 1.0
    assert 0.0 < df.row(0, named=True)["efficiency_multiplier"] <= 1.0

def test_contextual_flag_nonempty_for_active_seller() -> None:
    df = compute_seller_motivation(_bs([{"team": "X", "classification": "seller", "confidence": 0.9, "gp": 70}]), season=2025)
    assert df.row(0, named=True)["contextual_flag"] != ""

def test_empty_warns() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_seller_motivation(pl.DataFrame(), season=2025)
    assert df.is_empty()

def test_write_read_roundtrip(tmp_path: Path) -> None:
    df = compute_seller_motivation(_bs([{"team": "X", "classification": "seller", "confidence": 0.5, "gp": 70}]), season=2025)
    write_seller_motivation(df, tmp_path / "seller_motivation", 2025)
    rt = read_seller_motivation(tmp_path, 2025)
    assert rt is not None and len(rt) == len(df)
