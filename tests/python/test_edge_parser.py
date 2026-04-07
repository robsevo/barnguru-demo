"""Tests for EdgeParser — NHL EDGE skating and shot-speed parser."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from data.pbp_parser import DataMissingWarning
from data.edge_parser import (
    EDGE_SHOT_SCHEMA,
    EDGE_SKATING_SCHEMA,
    EdgeParser,
    EdgeSkatingRecord,
    EdgeShotRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEASON = 2023
_GAME_TYPE = 2


def _skating_row(
    player_id: int = 8480801,
    player_name: str = "Connor McDavid",
    team: str = "EDM",
    games_played: int = 82,
    avg_speed: float = 21.5,
    top_speed: float = 36.2,
    total_distance: float = 3800.0,
    zone_oz: float = 42.1,
    zone_dz: float = 28.3,
    zone_nz: float = 29.6,
) -> dict:
    return {
        "playerId": player_id,
        "skaterFullName": player_name,
        "teamAbbrevs": team,
        "gamesPlayed": games_played,
        "avgSpeed": avg_speed,
        "topSpeed": top_speed,
        "totalDistance": total_distance,
        "zoneTimeOffOz": zone_oz,
        "zoneTimeOffDz": zone_dz,
        "zoneTimeOffNz": zone_nz,
    }


def _shot_row(
    player_id: int = 8480801,
    player_name: str = "Connor McDavid",
    team: str = "EDM",
    games_played: int = 82,
    avg_shot_speed: float = 78.4,
    max_shot_speed: float = 96.2,
    hard_shot_count: int = 45,
) -> dict:
    return {
        "playerId": player_id,
        "skaterFullName": player_name,
        "teamAbbrevs": team,
        "gamesPlayed": games_played,
        "avgShotSpeed": avg_shot_speed,
        "maxShotSpeed": max_shot_speed,
        "hardShotCount": hard_shot_count,
    }


def _raw_skating(rows: list[dict] | None = None) -> dict:
    return {"data": rows or [], "total": len(rows or [])}


def _raw_shots(rows: list[dict] | None = None) -> dict:
    return {"data": rows or [], "total": len(rows or [])}


# ---------------------------------------------------------------------------
# 1 — parse_skating: empty data key
# ---------------------------------------------------------------------------


def test_parse_skating_empty_data() -> None:
    result = EdgeParser.parse_skating({"data": []}, _SEASON)
    assert result == []


def test_parse_skating_absent_data_key() -> None:
    result = EdgeParser.parse_skating({}, _SEASON)
    assert result == []


# ---------------------------------------------------------------------------
# 2 — parse_skating: basic record
# ---------------------------------------------------------------------------


def test_parse_skating_basic() -> None:
    raw = _raw_skating([_skating_row()])
    records = EdgeParser.parse_skating(raw, _SEASON)
    assert len(records) == 1
    r = records[0]
    assert r.player_id == 8480801
    assert r.player_name == "Connor McDavid"
    assert r.team == "EDM"
    assert r.season == _SEASON
    assert r.game_type == _GAME_TYPE
    assert r.games_played == 82
    assert r.avg_speed_kmh == pytest.approx(21.5)
    assert r.max_speed_kmh == pytest.approx(36.2)
    assert r.total_distance_km == pytest.approx(3800.0)


# ---------------------------------------------------------------------------
# 3 — distance_per_game computed
# ---------------------------------------------------------------------------


def test_parse_skating_distance_per_game_computed() -> None:
    row = _skating_row(games_played=80, total_distance=4000.0)
    records = EdgeParser.parse_skating(_raw_skating([row]), _SEASON)
    assert records[0].distance_per_game_km == pytest.approx(50.0)


def test_parse_skating_distance_per_game_zero_gp() -> None:
    row = _skating_row(games_played=0, total_distance=0.0)
    records = EdgeParser.parse_skating(_raw_skating([row]), _SEASON)
    assert records[0].distance_per_game_km is None


# ---------------------------------------------------------------------------
# 4 — missing player_id skipped + warning
# ---------------------------------------------------------------------------


def test_parse_skating_missing_player_id_skipped() -> None:
    row = _skating_row()
    del row["playerId"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        records = EdgeParser.parse_skating(_raw_skating([row]), _SEASON)
    assert records == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 5 — missing player_name skipped + warning
# ---------------------------------------------------------------------------


def test_parse_skating_missing_player_name_skipped() -> None:
    row = _skating_row()
    del row["skaterFullName"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        records = EdgeParser.parse_skating(_raw_skating([row]), _SEASON)
    assert records == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 6 — missing optional speed → None + warning
# ---------------------------------------------------------------------------


def test_parse_skating_optional_speed_none() -> None:
    row = _skating_row()
    del row["avgSpeed"]
    del row["topSpeed"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        records = EdgeParser.parse_skating(_raw_skating([row]), _SEASON)
    assert records[0].avg_speed_kmh is None
    assert records[0].max_speed_kmh is None
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 7 — zone time fields populated
# ---------------------------------------------------------------------------


def test_parse_skating_zone_time_populated() -> None:
    row = _skating_row(zone_oz=42.1, zone_dz=28.3, zone_nz=29.6)
    records = EdgeParser.parse_skating(_raw_skating([row]), _SEASON)
    r = records[0]
    assert r.zone_time_pct_oz == pytest.approx(42.1)
    assert r.zone_time_pct_dz == pytest.approx(28.3)
    assert r.zone_time_pct_nz == pytest.approx(29.6)


# ---------------------------------------------------------------------------
# 8 — multiple players
# ---------------------------------------------------------------------------


def test_parse_skating_multiple_players() -> None:
    rows = [
        _skating_row(player_id=1, player_name="P1"),
        _skating_row(player_id=2, player_name="P2"),
        _skating_row(player_id=3, player_name="P3"),
    ]
    records = EdgeParser.parse_skating(_raw_skating(rows), _SEASON)
    assert len(records) == 3


# ---------------------------------------------------------------------------
# 9 — parse_shot_speed: empty data
# ---------------------------------------------------------------------------


def test_parse_shot_speed_empty_data() -> None:
    assert EdgeParser.parse_shot_speed({"data": []}, _SEASON) == []


# ---------------------------------------------------------------------------
# 10 — parse_shot_speed: basic record
# ---------------------------------------------------------------------------


def test_parse_shot_speed_basic() -> None:
    raw = _raw_shots([_shot_row()])
    records = EdgeParser.parse_shot_speed(raw, _SEASON)
    assert len(records) == 1
    r = records[0]
    assert r.player_id == 8480801
    assert r.avg_shot_speed_mph == pytest.approx(78.4)
    assert r.max_shot_speed_mph == pytest.approx(96.2)


# ---------------------------------------------------------------------------
# 11 — hard_shot_count extracted as int
# ---------------------------------------------------------------------------


def test_parse_shot_speed_hard_shot_count() -> None:
    records = EdgeParser.parse_shot_speed(_raw_shots([_shot_row(hard_shot_count=37)]), _SEASON)
    assert records[0].hard_shot_count == 37
    assert isinstance(records[0].hard_shot_count, int)


# ---------------------------------------------------------------------------
# 12 — missing avgShotSpeed → None + warning
# ---------------------------------------------------------------------------


def test_parse_shot_speed_missing_speed_none() -> None:
    row = _shot_row()
    del row["avgShotSpeed"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        records = EdgeParser.parse_shot_speed(_raw_shots([row]), _SEASON)
    assert records[0].avg_shot_speed_mph is None
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 13 — missing player_id → skipped + warning
# ---------------------------------------------------------------------------


def test_parse_shot_speed_missing_player_skipped() -> None:
    row = _shot_row()
    del row["playerId"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        records = EdgeParser.parse_shot_speed(_raw_shots([row]), _SEASON)
    assert records == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 14 — to_polars_skating schema
# ---------------------------------------------------------------------------


def test_to_polars_skating_schema() -> None:
    records = EdgeParser.parse_skating(_raw_skating([_skating_row()]), _SEASON)
    df = EdgeParser.to_polars_skating(records)
    assert len(df) == 1
    for col in EDGE_SKATING_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 15 — to_polars_skating empty
# ---------------------------------------------------------------------------


def test_to_polars_skating_empty() -> None:
    df = EdgeParser.to_polars_skating([])
    assert len(df) == 0
    for col in EDGE_SKATING_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 16 — to_polars_shot_speed schema
# ---------------------------------------------------------------------------


def test_to_polars_shot_speed_schema() -> None:
    records = EdgeParser.parse_shot_speed(_raw_shots([_shot_row()]), _SEASON)
    df = EdgeParser.to_polars_shot_speed(records)
    assert len(df) == 1
    for col in EDGE_SHOT_SCHEMA:
        assert col in df.columns
