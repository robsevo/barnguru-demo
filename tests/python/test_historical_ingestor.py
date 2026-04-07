"""Tests for data.historical_ingestor — batch NHL season PBP + shift ingestor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import polars as pl
import pytest
import respx

from data.nhl_client import NHLApiError, NHLClient
from data.historical_ingestor import HistoricalIngestor, IngestResult, SeasonSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_pbp(game_id: int, n_plays: int = 2) -> dict:
    """Minimal valid PBP response with n_plays events."""
    plays = []
    for i in range(1, n_plays + 1):
        plays.append({
            "eventId": i,
            "sortOrder": i,
            "typeDescKey": "faceoff" if i % 2 == 1 else "shot-on-goal",
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": f"0{i}:00",
            "timeRemaining": f"1{9 - i}:00",
            "situationCode": "1551",
            "details": {
                "xCoord": 50.0,
                "yCoord": 0.0,
                "zoneCode": "N",
            },
        })
    return {"id": game_id, "plays": plays}


def make_shifts(game_id: int, n_shifts: int = 2) -> dict:
    """Minimal valid shifts response with n_shifts records."""
    data = []
    for i in range(1, n_shifts + 1):
        data.append({
            "gameId": game_id,
            "playerId": 8480800 + i,
            "teamId": 10,
            "period": 1,
            "shiftNumber": i,
            "startTime": "00:00",
            "endTime": "01:30",
            "duration": "01:30",
        })
    return {"data": data}


def make_season_games_response(game_ids: list[int]) -> dict:
    return {"data": [{"id": gid} for gid in game_ids]}


# ---------------------------------------------------------------------------
# Input validation — get_season_game_ids
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_season_game_ids_season_too_old_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError, match="2000"):
            await c.get_season_game_ids(1999)


@respx.mock
async def test_get_season_game_ids_invalid_game_type_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError, match="game_type"):
            await c.get_season_game_ids(2021, game_type=4)


@respx.mock
async def test_get_season_game_ids_game_type_1_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError, match="game_type"):
            await c.get_season_game_ids(2021, game_type=1)


# ---------------------------------------------------------------------------
# Input validation — ingest_game
# ---------------------------------------------------------------------------


async def test_ingest_game_zero_id_raises(tmp_path):
    ingestor = HistoricalIngestor(tmp_path)
    client = MagicMock()
    with pytest.raises(ValueError, match="positive integer"):
        await ingestor.ingest_game(client, 0, 2021)


async def test_ingest_game_negative_id_raises(tmp_path):
    ingestor = HistoricalIngestor(tmp_path)
    client = MagicMock()
    with pytest.raises(ValueError, match="positive integer"):
        await ingestor.ingest_game(client, -5, 2021)


# ---------------------------------------------------------------------------
# get_season_game_ids — known-answer
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_season_game_ids_returns_sorted_list():
    respx.get("https://api.nhle.com/stats/rest/en/game").mock(
        return_value=httpx.Response(
            200,
            json=make_season_games_response([2021020003, 2021020001, 2021020002]),
        )
    )
    async with NHLClient() as c:
        ids = await c.get_season_game_ids(2021)
    # Returned in order from API (we don't sort, the API does via sort=id param)
    assert ids == [2021020003, 2021020001, 2021020002]


@respx.mock
async def test_get_season_game_ids_empty_season():
    respx.get("https://api.nhle.com/stats/rest/en/game").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    async with NHLClient() as c:
        ids = await c.get_season_game_ids(2021)
    assert ids == []


@respx.mock
async def test_get_season_game_ids_playoffs_game_type():
    respx.get("https://api.nhle.com/stats/rest/en/game").mock(
        return_value=httpx.Response(
            200, json=make_season_games_response([2021030001, 2021030002])
        )
    )
    async with NHLClient() as c:
        ids = await c.get_season_game_ids(2021, game_type=3)
    assert ids == [2021030001, 2021030002]


# ---------------------------------------------------------------------------
# Known-answer regression — ingest_season (mocked)
# ---------------------------------------------------------------------------


async def test_ingest_season_known_answer(tmp_path):
    """3 games mocked → IngestResult counts match fixture data."""
    game_ids = [2021020001, 2021020002, 2021020003]
    n_plays = 3
    n_shifts = 2

    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(
        side_effect=lambda gid: make_pbp(gid, n_plays)
    )
    client.get_shifts = AsyncMock(
        side_effect=lambda gid: make_shifts(gid, n_shifts)
    )

    ingestor = HistoricalIngestor(tmp_path, concurrency=3)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.total_games == 3
    assert summary.successful == 3
    assert summary.skipped == 0
    assert summary.failed == 0
    # total_plays counts only "ok" games; faceoff events don't produce zone-entry
    # derivation here, so plays_count == n_plays per game
    assert summary.total_plays == n_plays * 3
    assert summary.total_shifts == n_shifts * 3
    assert summary.total_warnings >= 0


async def test_ingest_season_parquet_files_created(tmp_path):
    """ingest_season creates pbp, shifts, toi Parquet files."""
    game_ids = [2021020001, 2021020002]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid))

    ingestor = HistoricalIngestor(tmp_path)
    await ingestor.ingest_season(client, 2021)

    assert (tmp_path / "pbp_2021.parquet").exists()
    assert (tmp_path / "shifts_2021.parquet").exists()
    assert (tmp_path / "toi_2021.parquet").exists()


async def test_ingest_season_parquet_schema_correct(tmp_path):
    """Parquet files contain expected columns (game_id, event_type, period)."""
    game_ids = [2021020001]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid))

    ingestor = HistoricalIngestor(tmp_path)
    await ingestor.ingest_season(client, 2021)

    df = pl.read_parquet(tmp_path / "pbp_2021.parquet")
    assert "game_id" in df.columns
    assert "event_type" in df.columns
    assert "period" in df.columns


async def test_ingest_season_manifest_updated(tmp_path):
    """Manifest is written with correct status after ingest_season."""
    game_ids = [2021020001, 2021020002]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid))

    ingestor = HistoricalIngestor(tmp_path)
    await ingestor.ingest_season(client, 2021)

    manifest = ingestor.get_manifest()
    for gid in game_ids:
        assert str(gid) in manifest["games"]
        assert manifest["games"][str(gid)]["status"] == "ok"
        assert manifest["games"][str(gid)]["plays_count"] >= 0


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------


async def test_summary_counts_sum_to_total(tmp_path):
    """successful + skipped + failed always equals total_games."""
    game_ids = [2021020001, 2021020002, 2021020003]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid))

    ingestor = HistoricalIngestor(tmp_path)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.successful + summary.skipped + summary.failed == summary.total_games


async def test_result_counts_non_negative(tmp_path):
    """All counts on IngestResult are ≥ 0."""
    client = MagicMock(spec=NHLClient)
    client.get_play_by_play = AsyncMock(return_value=make_pbp(2021020001, 5))
    client.get_shifts = AsyncMock(return_value=make_shifts(2021020001, 3))

    ingestor = HistoricalIngestor(tmp_path)
    result = await ingestor.ingest_game(client, 2021020001, 2021)

    assert result.plays_count >= 0
    assert result.shifts_count >= 0
    assert result.warning_count >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_empty_season_all_zeros(tmp_path):
    """0 game_ids returned → SeasonSummary with all-zero counts."""
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=[])

    ingestor = HistoricalIngestor(tmp_path)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.total_games == 0
    assert summary.successful == 0
    assert summary.skipped == 0
    assert summary.failed == 0
    assert summary.total_plays == 0
    assert summary.total_shifts == 0
    # Parquet files should still be created (empty schema)
    assert (tmp_path / "pbp_2021.parquet").exists()


async def test_pbp_404_status_skipped_404(tmp_path):
    """PBP 404 → status="skipped_404", other games still ingested."""
    game_ids = [2021020001, 2021020002]

    async def pbp_side_effect(gid):
        if gid == 2021020001:
            raise NHLApiError("HTTP 404", status_code=404, url="...")
        return make_pbp(gid)

    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=pbp_side_effect)
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid))

    ingestor = HistoricalIngestor(tmp_path)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.failed == 1
    assert summary.successful == 1
    manifest = ingestor.get_manifest()
    assert manifest["games"]["2021020001"]["status"] == "skipped_404"
    assert manifest["games"]["2021020002"]["status"] == "ok"


async def test_pbp_500_status_error_pbp(tmp_path):
    """PBP 500 (exhausted retries) → status="error_pbp", other games continue."""
    game_ids = [2021020001, 2021020002]

    async def pbp_side_effect(gid):
        if gid == 2021020001:
            raise NHLApiError("HTTP 500", status_code=500, url="...")
        return make_pbp(gid)

    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=pbp_side_effect)
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid))

    ingestor = HistoricalIngestor(tmp_path)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.failed == 1
    assert summary.successful == 1
    manifest = ingestor.get_manifest()
    assert manifest["games"]["2021020001"]["status"] == "error_pbp"


async def test_shifts_error_status_error_shifts(tmp_path):
    """Shifts API error → status="error_shifts"."""
    game_ids = [2021020001]

    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(return_value=make_pbp(2021020001))
    client.get_shifts = AsyncMock(
        side_effect=NHLApiError("HTTP 500", status_code=500, url="...")
    )

    ingestor = HistoricalIngestor(tmp_path)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.failed == 1
    manifest = ingestor.get_manifest()
    assert manifest["games"]["2021020001"]["status"] == "error_shifts"


async def test_empty_plays_status_no_plays(tmp_path):
    """Empty plays list → status="no_plays", warning_count still computed."""
    game_ids = [2021020001]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(
        return_value={"id": 2021020001, "plays": []}
    )
    client.get_shifts = AsyncMock(return_value=make_shifts(2021020001))

    ingestor = HistoricalIngestor(tmp_path)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.failed == 1  # no_plays counts as failed
    assert summary.successful == 0
    manifest = ingestor.get_manifest()
    assert manifest["games"]["2021020001"]["status"] == "no_plays"
    assert manifest["games"]["2021020001"]["warning_count"] >= 0


async def test_already_ingested_game_skipped_no_api_call(tmp_path):
    """Game already in manifest as "ok" → status="skipped", API not called."""
    # Pre-populate manifest
    ingestor = HistoricalIngestor(tmp_path)
    manifest = {"format_version": 1, "games": {
        "2021020001": {
            "status": "ok", "season": 2021,
            "plays_count": 99, "shifts_count": 300,
            "warning_count": 0, "ingested_at": "2024-01-01T00:00:00Z",
        }
    }}
    ingestor._write_manifest(manifest)

    client = MagicMock(spec=NHLClient)
    client.get_play_by_play = AsyncMock()
    client.get_shifts = AsyncMock()

    result = await ingestor.ingest_game(client, 2021020001, 2021)

    assert result.status == "skipped"
    assert result.plays_count == 99
    assert result.shifts_count == 300
    client.get_play_by_play.assert_not_awaited()
    client.get_shifts.assert_not_awaited()


async def test_already_ingested_game_skipped_in_season(tmp_path):
    """Games in manifest as "ok" are skipped during ingest_season."""
    # Pre-populate one game as done
    ingestor = HistoricalIngestor(tmp_path)
    manifest = {"format_version": 1, "games": {
        "2021020001": {
            "status": "ok", "season": 2021,
            "plays_count": 50, "shifts_count": 200,
            "warning_count": 0, "ingested_at": "2024-01-01T00:00:00Z",
        }
    }}
    ingestor._write_manifest(manifest)

    game_ids = [2021020001, 2021020002]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid))

    summary = await ingestor.ingest_season(client, 2021)

    assert summary.skipped == 1
    assert summary.successful == 1
    assert summary.total_games == 2
    # API was only called for the non-skipped game
    assert client.get_play_by_play.await_count == 1


async def test_clear_errors_removes_only_error_entries(tmp_path):
    """clear_errors() removes error-status entries; "ok" and "skipped_404" entries differ."""
    ingestor = HistoricalIngestor(tmp_path)
    manifest = {"format_version": 1, "games": {
        "2021020001": {"status": "ok", "season": 2021, "plays_count": 50,
                       "shifts_count": 200, "warning_count": 0, "ingested_at": "2024-01-01T00:00:00Z"},
        "2021020002": {"status": "error_pbp", "season": 2021, "plays_count": 0,
                       "shifts_count": 0, "warning_count": 0, "ingested_at": "2024-01-01T00:00:00Z"},
        "2021020003": {"status": "error_shifts", "season": 2021, "plays_count": 0,
                       "shifts_count": 0, "warning_count": 0, "ingested_at": "2024-01-01T00:00:00Z"},
        "2021020004": {"status": "no_plays", "season": 2021, "plays_count": 0,
                       "shifts_count": 0, "warning_count": 0, "ingested_at": "2024-01-01T00:00:00Z"},
        "2021020005": {"status": "skipped_404", "season": 2021, "plays_count": 0,
                       "shifts_count": 0, "warning_count": 0, "ingested_at": "2024-01-01T00:00:00Z"},
    }}
    ingestor._write_manifest(manifest)

    removed = ingestor.clear_errors()

    assert removed == 4  # error_pbp, error_shifts, no_plays, skipped_404
    updated = ingestor.get_manifest()
    assert "2021020001" in updated["games"]  # "ok" remains
    assert "2021020002" not in updated["games"]
    assert "2021020003" not in updated["games"]
    assert "2021020004" not in updated["games"]
    assert "2021020005" not in updated["games"]


async def test_clear_errors_empty_manifest(tmp_path):
    """clear_errors() on missing manifest returns 0."""
    ingestor = HistoricalIngestor(tmp_path)
    removed = ingestor.clear_errors()
    assert removed == 0


# ---------------------------------------------------------------------------
# Integration smoke tests
# ---------------------------------------------------------------------------


async def test_full_mock_pipeline_parquet_readable(tmp_path):
    """Full pipeline: get_season_game_ids → ingest_season → Parquet readable."""
    game_ids = [2021020001, 2021020002, 2021020003]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid, 4))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid, 3))

    ingestor = HistoricalIngestor(tmp_path)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.successful == 3
    df = pl.read_parquet(tmp_path / "pbp_2021.parquet")
    assert "game_id" in df.columns
    assert "event_type" in df.columns
    assert "period" in df.columns
    # Each game produces 4 plays; faceoff plays may generate derived zone-entry events
    assert len(df) >= 3 * 4

    shifts_df = pl.read_parquet(tmp_path / "shifts_2021.parquet")
    assert "game_id" in shifts_df.columns
    assert "player_id" in shifts_df.columns
    assert len(shifts_df) == 3 * 3

    toi_df = pl.read_parquet(tmp_path / "toi_2021.parquet")
    assert "game_id" in toi_df.columns
    assert "toi_total_secs" in toi_df.columns


async def test_ingest_seasons_two_seasons(tmp_path):
    """ingest_seasons([2021, 2022]) → two Parquet files per type, two SeasonSummary."""
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=[2021020001, 2021020002])
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid))

    ingestor = HistoricalIngestor(tmp_path)
    summaries = await ingestor.ingest_seasons(client, [2021, 2022])

    assert len(summaries) == 2
    assert summaries[0].season == 2021
    assert summaries[1].season == 2022
    assert (tmp_path / "pbp_2021.parquet").exists()
    assert (tmp_path / "pbp_2022.parquet").exists()
    assert (tmp_path / "shifts_2021.parquet").exists()
    assert (tmp_path / "shifts_2022.parquet").exists()
    assert (tmp_path / "toi_2021.parquet").exists()
    assert (tmp_path / "toi_2022.parquet").exists()


async def test_ingest_seasons_summary_counts(tmp_path):
    """Each SeasonSummary from ingest_seasons has correct counts."""
    # Use season-specific game_ids (as real NHL API would return)
    def season_game_ids(season, game_type=2):
        return [int(f"{season}020001")]

    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(side_effect=season_game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid, 5))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid, 4))

    ingestor = HistoricalIngestor(tmp_path)
    summaries = await ingestor.ingest_seasons(client, [2021, 2022])

    for s in summaries:
        assert s.total_games == 1
        assert s.successful == 1
        assert s.successful + s.skipped + s.failed == s.total_games


async def test_concurrency_caps_inflight_requests(tmp_path):
    """Semaphore caps concurrent in-flight game requests at concurrency level."""
    concurrency = 2
    game_ids = [2021020001, 2021020002, 2021020003, 2021020004]

    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def mock_pbp(game_id: int) -> dict:
        nonlocal active, max_active
        async with lock:
            active += 1
            if active > max_active:
                max_active = active
        await asyncio.sleep(0.02)  # simulate slow network
        async with lock:
            active -= 1
        return make_pbp(game_id)

    async def mock_shifts(game_id: int) -> dict:
        return make_shifts(game_id)

    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=mock_pbp)
    client.get_shifts = AsyncMock(side_effect=mock_shifts)

    ingestor = HistoricalIngestor(tmp_path, concurrency=concurrency)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.successful == 4
    assert max_active <= concurrency


async def test_empty_parquet_written_when_all_fail(tmp_path):
    """When all games fail, empty Parquet files with correct schema are written."""
    game_ids = [2021020001]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(
        side_effect=NHLApiError("HTTP 500", status_code=500, url="...")
    )
    client.get_shifts = AsyncMock(
        side_effect=NHLApiError("HTTP 500", status_code=500, url="...")
    )

    ingestor = HistoricalIngestor(tmp_path)
    summary = await ingestor.ingest_season(client, 2021)

    assert summary.successful == 0
    df = pl.read_parquet(tmp_path / "pbp_2021.parquet")
    assert len(df) == 0
    assert "game_id" in df.columns
    assert "event_type" in df.columns


async def test_manifest_format_version_preserved(tmp_path):
    """Manifest always contains format_version: 1."""
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=[2021020001])
    client.get_play_by_play = AsyncMock(return_value=make_pbp(2021020001))
    client.get_shifts = AsyncMock(return_value=make_shifts(2021020001))

    ingestor = HistoricalIngestor(tmp_path)
    await ingestor.ingest_season(client, 2021)

    manifest = ingestor.get_manifest()
    assert manifest["format_version"] == 1
    assert "games" in manifest


async def test_get_manifest_missing_returns_empty(tmp_path):
    """get_manifest() with no file returns empty structure."""
    ingestor = HistoricalIngestor(tmp_path)
    manifest = ingestor.get_manifest()
    assert manifest == {"format_version": 1, "games": {}}


async def test_error_message_stored_in_manifest(tmp_path):
    """error_message is written to manifest for failed games."""
    game_ids = [2021020001]
    error_msg = "HTTP 500 from /v1/gamecenter/2021020001/play-by-play"

    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(
        side_effect=NHLApiError(error_msg, status_code=500, url="...")
    )
    client.get_shifts = AsyncMock(return_value=make_shifts(2021020001))

    ingestor = HistoricalIngestor(tmp_path)
    await ingestor.ingest_season(client, 2021)

    manifest = ingestor.get_manifest()
    entry = manifest["games"]["2021020001"]
    assert "error_message" in entry
    assert error_msg in entry["error_message"]


# ---------------------------------------------------------------------------
# Feature 1.5 — derived stats Parquet + schema coverage
# ---------------------------------------------------------------------------


async def test_player_stats_parquet_created(tmp_path):
    """After ingest_season, player_stats_{season}.parquet exists."""
    game_ids = [2021020001]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid, 4))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid, 2))

    ingestor = HistoricalIngestor(tmp_path)
    await ingestor.ingest_season(client, 2021)

    assert (tmp_path / "player_stats_2021.parquet").exists()
    assert (tmp_path / "team_stats_2021.parquet").exists()
    assert (tmp_path / "vs_team_2021.parquet").exists()
    assert (tmp_path / "vs_goalie_2021.parquet").exists()


async def test_turnover_player_id_in_pbp_schema(tmp_path):
    """PBP Parquet contains turnover_player_id, home_team_id, away_team_id columns."""
    game_ids = [2021020001]
    client = MagicMock(spec=NHLClient)
    client.get_season_game_ids = AsyncMock(return_value=game_ids)
    client.get_play_by_play = AsyncMock(side_effect=lambda gid: make_pbp(gid, 4))
    client.get_shifts = AsyncMock(side_effect=lambda gid: make_shifts(gid, 2))

    ingestor = HistoricalIngestor(tmp_path)
    await ingestor.ingest_season(client, 2021)

    df = pl.read_parquet(tmp_path / "pbp_2021.parquet")
    assert "turnover_player_id" in df.columns
    assert "home_team_id" in df.columns
    assert "away_team_id" in df.columns
