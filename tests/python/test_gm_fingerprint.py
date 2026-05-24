"""Unit tests for models/gm_fingerprint.py — Feature 4.18."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import pytest
from models.gm_fingerprint import (
    GM_FINGERPRINT_SCHEMA, MODEL_VERSION, DataMissingWarning, ARCHETYPES,
    compute_gm_fingerprint, write_gm_fingerprint, read_gm_fingerprint,
    _archetype_probs,
)

def _bs(rows: list[dict]) -> pl.DataFrame:
    base = {"team": "TOR", "season": 2025, "gp": 75, "wins": 35, "ot_games": 10,
            "losses": 30, "points": 80, "points_pct": 0.533, "threshold": 0.500,
            "gap": 0.033, "classification": "neutral", "confidence": 0.33,
            "model_version": "buyer_seller_v1"}
    return pl.DataFrame([{**base, **r} for r in rows])

def _tx(rows: list[dict]) -> pl.DataFrame:
    base = {"date": "2026-03-01", "event_type": "call_up", "team": "TOR",
            "description": "Recalled F Player", "secondary_team": None,
            "player_or_executive": "", "player_id_espn": None,
            "notes": None, "fetched_at": "", "source": "espn"}
    return pl.DataFrame([{**base, **r} for r in rows])

def test_schema_and_version() -> None:
    df = compute_gm_fingerprint(_bs([{}]), _tx([{}]), season=2025)
    assert set(df.columns) == set(GM_FINGERPRINT_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()

def test_archetype_probs_sum_to_1() -> None:
    for cls in ["buyer", "seller", "neutral"]:
        for activity in [0.0, 0.5, 1.0]:
            probs = _archetype_probs(cls, activity)
            assert sum(probs.values()) == pytest.approx(1.0, abs=0.001)
            assert all(k in ARCHETYPES for k in probs)

def test_buyer_high_activity_favors_add_rental() -> None:
    probs = _archetype_probs("buyer", 0.8)
    assert probs["add_rental"] >= probs["stand_pat"]

def test_seller_favors_sell_or_rebuild() -> None:
    probs = _archetype_probs("seller", 0.7)
    assert probs["sell_veteran"] + probs["rebuild"] > probs["stand_pat"]

def test_deadline_aggression_bounded() -> None:
    df = compute_gm_fingerprint(
        _bs([{"team": "COL", "classification": "buyer", "confidence": 1.0}]),
        _tx([{"team": "COL"}, {"team": "COL"}, {"team": "COL"}]),
        season=2025,
    )
    r = df.row(0, named=True)
    assert 0.0 <= r["deadline_aggression"] <= 1.0

def test_gm_name_extraction() -> None:
    tx = _tx([
        {"event_type": "front_office", "team": "TOR",
         "description": "Named Brad Treliving as general manager."},
    ])
    df = compute_gm_fingerprint(_bs([{}]), tx, season=2025)
    r = df.row(0, named=True)
    assert r["gm_name"] == "Brad Treliving"

def test_empty_bs_warns() -> None:
    with pytest.warns(DataMissingWarning):
        df = compute_gm_fingerprint(pl.DataFrame(), _tx([{}]), season=2025)
    assert df.is_empty()

def test_write_read_roundtrip(tmp_path: Path) -> None:
    df = compute_gm_fingerprint(_bs([{}]), _tx([{}]), season=2025)
    write_gm_fingerprint(df, tmp_path / "gm_fingerprint", 2025)
    rt = read_gm_fingerprint(tmp_path, 2025)
    assert rt is not None and len(rt) == len(df)
