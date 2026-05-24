"""Unit tests for models/staff_change_detector.py — Feature 4.13."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from models.staff_change_detector import (
    DataMissingWarning,
    MODEL_VERSION,
    STAFF_CHANGE_SCHEMA,
    _classify_change,
    _extract_person_out,
    detect_staff_changes,
    read_staff_changes,
    write_staff_changes,
)


def _tx(rows: list[dict]) -> pl.DataFrame:
    base = {"date": "2025-12-01", "event_type": "front_office",
            "team": "MTL", "description": "", "secondary_team": None,
            "player_or_executive": "", "player_id_espn": None,
            "notes": None, "fetched_at": "", "source": "espn"}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_detects_head_coach_firing() -> None:
    tx = _tx([
        {"description": "Fired head coach Patrick Roy."},
    ])
    df = detect_staff_changes(tx, season=2025)
    assert len(df) == 1
    r = df.row(0, named=True)
    assert r["change_type"] == "head_coach"
    assert r["person_out"] == "Patrick Roy"
    assert r["regime_change_trigger"] is True
    assert r["decay_games"] == 20


def test_detects_coordinator_mention() -> None:
    tx = _tx([
        {"description": "Named John Smith as PP coordinator."},
    ])
    df = detect_staff_changes(tx, season=2025)
    assert len(df) == 1
    assert df.row(0, named=True)["change_type"] == "coordinator"


def test_detects_goalie_coach() -> None:
    tx = _tx([
        {"description": "Fired goaltending coach Eric Raymond."},
    ])
    df = detect_staff_changes(tx, season=2025)
    assert len(df) == 1
    assert df.row(0, named=True)["change_type"] == "goalie_coach"


def test_ignores_non_coaching_front_office() -> None:
    tx = _tx([
        {"description": "Fired general manager Tom Fitzgerald."},
    ])
    df = detect_staff_changes(tx, season=2025)
    assert df.is_empty()


def test_deduplicates_identical_rows() -> None:
    tx = _tx([
        {"description": "Fired head coach Patrick Roy."},
        {"description": "Fired head coach Patrick Roy."},
    ])
    df = detect_staff_changes(tx, season=2025)
    assert len(df) == 1


def test_empty_transactions_warns() -> None:
    with pytest.warns(DataMissingWarning):
        df = detect_staff_changes(pl.DataFrame(), season=2025)
    assert df.is_empty()
    assert set(df.columns) == set(STAFF_CHANGE_SCHEMA.keys())


def test_schema_and_version() -> None:
    tx = _tx([{"description": "Named Jane Doe as head coach."}])
    df = detect_staff_changes(tx, season=2025)
    assert set(df.columns) == set(STAFF_CHANGE_SCHEMA.keys())
    assert (df["model_version"] == MODEL_VERSION).all()


def test_classify_change_helper() -> None:
    assert _classify_change("Fired head coach X") == "head_coach"
    assert _classify_change("Named Y as PP coordinator") == "coordinator"
    assert _classify_change("Fired goaltending coach Z") == "goalie_coach"
    assert _classify_change("Promoted intern to water boy") == "unknown"


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    tx = _tx([{"description": "Fired head coach Roy."}])
    df = detect_staff_changes(tx, season=2025)
    write_staff_changes(df, tmp_path / "staff_changes", season=2025)
    rt = read_staff_changes(tmp_path, season=2025)
    assert rt is not None and len(rt) == len(df)
