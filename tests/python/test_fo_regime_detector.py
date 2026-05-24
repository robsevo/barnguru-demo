"""Unit tests for models/fo_regime_detector.py — Feature 4.14."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.fo_regime_detector import (
    DataMissingWarning,
    FO_REGIME_SCHEMA,
    MODEL_VERSION,
    _classify_fo_role,
    detect_fo_regime_changes,
    read_fo_regime_changes,
    write_fo_regime_changes,
)


def _tx(rows: list[dict]) -> pl.DataFrame:
    base = {"date": "2025-12-15", "event_type": "front_office",
            "team": "BUF", "description": "", "secondary_team": None,
            "player_or_executive": "", "player_id_espn": None,
            "notes": None, "fetched_at": "", "source": "espn"}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_detects_gm_firing() -> None:
    tx = _tx([
        {"description": "Fired general manager Kevyn Adams."},
    ])
    df = detect_fo_regime_changes(tx, season=2025)
    assert len(df) == 1
    r = df.row(0, named=True)
    assert r["fo_role"] == "gm"
    assert r["person_out"] == "Kevyn Adams"
    assert r["decay_games"] == 50


def test_detects_president_hockey_ops() -> None:
    tx = _tx([
        {"description": "Named Jason Kekalainen as president of hockey operations."},
    ])
    df = detect_fo_regime_changes(tx, season=2025)
    assert len(df) == 1
    assert df.row(0, named=True)["fo_role"] == "president_hockey_ops"


def test_detects_agm() -> None:
    tx = _tx([
        {"description": "Hired Marc Bergevin as assistant general manager."},
    ])
    df = detect_fo_regime_changes(tx, season=2025)
    assert len(df) == 1
    assert df.row(0, named=True)["fo_role"] == "agm"
    assert df.row(0, named=True)["decay_games"] == 30


def test_ignores_coaching_changes() -> None:
    tx = _tx([
        {"description": "Fired head coach Lindy Ruff."},
    ])
    df = detect_fo_regime_changes(tx, season=2025)
    assert df.is_empty()


def test_deduplicates() -> None:
    tx = _tx([
        {"description": "Fired general manager Tom Fitzgerald."},
        {"description": "Fired general manager Tom Fitzgerald."},
    ])
    df = detect_fo_regime_changes(tx, season=2025)
    assert len(df) == 1


def test_empty_transactions_warns() -> None:
    with pytest.warns(DataMissingWarning):
        df = detect_fo_regime_changes(pl.DataFrame(), season=2025)
    assert df.is_empty()
    assert set(df.columns) == set(FO_REGIME_SCHEMA.keys())


def test_classify_fo_role_helper() -> None:
    assert _classify_fo_role("Fired general manager X") == "gm"
    assert _classify_fo_role("Named Y president of hockey operations") == "president_hockey_ops"
    assert _classify_fo_role("Hired Z as assistant general manager") == "agm"
    assert _classify_fo_role("Named W as VP of player personnel") == "other_exec"
    assert _classify_fo_role("Fired head coach Roy") is None


def test_schema_and_version() -> None:
    tx = _tx([{"description": "Fired GM Adams."}])
    df = detect_fo_regime_changes(tx, season=2025)
    assert set(df.columns) == set(FO_REGIME_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    tx = _tx([{"description": "Fired general manager Adams."}])
    df = detect_fo_regime_changes(tx, season=2025)
    write_fo_regime_changes(df, tmp_path / "fo_regime_changes", season=2025)
    rt = read_fo_regime_changes(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
