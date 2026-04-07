"""Tests for EdgeSync — seasonal NHL EDGE Parquet persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from data.edge_parser import EDGE_SHOT_SCHEMA, EDGE_SKATING_SCHEMA, EdgeShotRecord, EdgeSkatingRecord
from data.edge_sync import EdgeSync, EdgeSyncResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEASON = 2023


def _skating_record() -> EdgeSkatingRecord:
    return EdgeSkatingRecord(
        player_id=8480801,
        player_name="Connor McDavid",
        team="EDM",
        season=_SEASON,
        game_type=2,
        games_played=82,
        avg_speed_kmh=21.5,
        max_speed_kmh=36.2,
        total_distance_km=3800.0,
        distance_per_game_km=46.3,
        zone_time_pct_oz=42.1,
        zone_time_pct_dz=28.3,
        zone_time_pct_nz=29.6,
    )


def _shot_record() -> EdgeShotRecord:
    return EdgeShotRecord(
        player_id=8480801,
        player_name="Connor McDavid",
        team="EDM",
        season=_SEASON,
        game_type=2,
        games_played=82,
        avg_shot_speed_mph=78.4,
        max_shot_speed_mph=96.2,
        hard_shot_count=45,
    )


_PLAYER_ID = 8480801

_EDGE_DETAIL_RAW = {
    "player": {
        "id": _PLAYER_ID,
        "firstName": {"default": "Connor"},
        "lastName": {"default": "McDavid"},
        "gamesPlayed": 82,
        "team": {"abbrev": "EDM"},
    },
    "skatingSpeed": {"speedMax": {"metric": 36.2}, "burstsOver20": {"value": 300}},
    "totalDistanceSkated": {"metric": 3800.0},
    "topShotSpeed": {"imperial": 96.2},
    "zoneTimeDetails": {
        "offensiveZonePctg": 42.1,
        "defensiveZonePctg": 28.3,
        "neutralZonePctg": 29.6,
    },
}

_ROSTER_RAW = {
    "forwards": [{"id": _PLAYER_ID}],
    "defensemen": [],
    "goalies": [],
}

_TEAMS_RAW = [{"triCode": "EDM"}]


def _make_client(
    detail_raw: dict | None = None,
    teams_exc: Exception | None = None,
    roster_exc: Exception | None = None,
    detail_exc: Exception | None = None,
) -> MagicMock:
    """Build a mock NHLClient that returns controlled EDGE per-player responses."""
    client = MagicMock()
    if teams_exc is not None:
        client.get_teams = AsyncMock(side_effect=teams_exc)
    else:
        client.get_teams = AsyncMock(return_value=_TEAMS_RAW)
    if roster_exc is not None:
        client.get_roster_season = AsyncMock(side_effect=roster_exc)
    else:
        client.get_roster_season = AsyncMock(return_value=_ROSTER_RAW)
    if detail_exc is not None:
        client.get_edge_skater_detail = AsyncMock(side_effect=detail_exc)
    else:
        client.get_edge_skater_detail = AsyncMock(return_value=detail_raw or _EDGE_DETAIL_RAW)
    return client


# ---------------------------------------------------------------------------
# 1 — sync_season ok
# ---------------------------------------------------------------------------


async def test_sync_season_ok(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    client = _make_client()

    result = await sync.sync_season(client, _SEASON)

    assert result.status == "ok"
    assert result.skating_count == 1
    assert result.shot_count == 1
    assert (tmp_path / f"edge_skating_{_SEASON}.parquet").exists()
    assert (tmp_path / f"edge_shot_speed_{_SEASON}.parquet").exists()


# ---------------------------------------------------------------------------
# 2 — sync_season skipped (already ok in manifest)
# ---------------------------------------------------------------------------


async def test_sync_season_skipped(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    client = _make_client()

    await sync.sync_season(client, _SEASON)
    first_call_count = client.get_edge_skater_detail.call_count

    result = await sync.sync_season(client, _SEASON)
    assert result.status == "skipped"
    assert client.get_edge_skater_detail.call_count == first_call_count  # not called again


# ---------------------------------------------------------------------------
# 3 — force bypasses skip
# ---------------------------------------------------------------------------


async def test_sync_season_force(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    client = _make_client()

    await sync.sync_season(client, _SEASON)
    first_call_count = client.get_edge_skater_detail.call_count
    result = await sync.sync_season(client, _SEASON, force=True)

    assert result.status == "ok"
    assert client.get_edge_skater_detail.call_count == first_call_count * 2  # called again


# ---------------------------------------------------------------------------
# 4 — error_api on skating endpoint
# ---------------------------------------------------------------------------


async def test_sync_season_error_api_skating(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    client = _make_client(teams_exc=RuntimeError("timeout"))

    result = await sync.sync_season(client, _SEASON)

    assert result.status == "error_api"
    assert result.skating_count == 0
    assert "timeout" in (result.error_message or "")
    manifest = sync.get_manifest()
    assert manifest["seasons"][str(_SEASON)]["status"] == "error_api"


# ---------------------------------------------------------------------------
# 5 — empty skating records
# ---------------------------------------------------------------------------


async def test_sync_season_empty_skating(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    # Empty roster → no player IDs → all detail fetches return None → empty result
    client = _make_client()
    client.get_roster_season = AsyncMock(return_value={"forwards": [], "defensemen": [], "goalies": []})

    result = await sync.sync_season(client, _SEASON)

    assert result.status == "empty"
    assert result.skating_count == 0
    # Empty Parquet files should still be written
    assert (tmp_path / f"edge_skating_{_SEASON}.parquet").exists()


# ---------------------------------------------------------------------------
# 6 — get_skating returns DataFrame after sync
# ---------------------------------------------------------------------------


async def test_get_skating_returns_df(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    await sync.sync_season(_make_client(), _SEASON)

    df = sync.get_skating(_SEASON)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert "avg_speed_kmh" in df.columns


# ---------------------------------------------------------------------------
# 7 — get_skating: no file → empty schema
# ---------------------------------------------------------------------------


def test_get_skating_no_file(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    df = sync.get_skating(_SEASON)
    assert len(df) == 0
    for col in EDGE_SKATING_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 8 — get_shot_speed: no file → empty schema
# ---------------------------------------------------------------------------


def test_get_shot_speed_no_file(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    df = sync.get_shot_speed(_SEASON)
    assert len(df) == 0
    for col in EDGE_SHOT_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 9 — get_manifest empty
# ---------------------------------------------------------------------------


def test_get_manifest_empty(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    manifest = sync.get_manifest()
    assert manifest == {"format_version": 1, "seasons": {}}


# ---------------------------------------------------------------------------
# 10 — clear_errors removes error entries
# ---------------------------------------------------------------------------


def test_clear_errors_removes_errors(tmp_path: Path) -> None:
    sync = EdgeSync(tmp_path)
    manifest = {
        "format_version": 1,
        "seasons": {
            "2023": {"status": "error_api"},
            "2022": {"status": "ok"},
            "2021": {"status": "empty"},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    removed = sync.clear_errors()
    assert removed == 2
    updated = sync.get_manifest()
    assert "2022" in updated["seasons"]
    assert "2023" not in updated["seasons"]
    assert "2021" not in updated["seasons"]


# ---------------------------------------------------------------------------
# 11 — storage_dir created automatically
# ---------------------------------------------------------------------------


def test_storage_dir_created_automatically(tmp_path: Path) -> None:
    new_dir = tmp_path / "nested" / "edge"
    assert not new_dir.exists()
    EdgeSync(new_dir)
    assert new_dir.exists()
