"""Tests for InjurySync (feature 1.8).

Covers:
- TTL skip logic (within TTL → skip, past TTL → re-fetch)
- Error handling (error_fetch, error_parse)
- Empty parse result → empty-schema Parquet written
- get_injuries() loads correct Parquet
- get_player_status() returns correct InjuryStatus or None
- clear_errors() removes only error/empty entries
- Integration smoke: sync → get_injuries → correct columns
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from data.injury_client import InjuryClient, InjuryFeedError
from data.injury_parser import EMPTY_SCHEMA, InjuryStatus
from data.injury_sync import InjurySync, InjurySyncResult


# ---------------------------------------------------------------------------
# Minimal ESPN response used across tests
# ---------------------------------------------------------------------------

_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

_MINIMAL_ESPN_RESPONSE = {
    "injuries": [
        {
            "team": {"abbreviation": "MTL"},
            "injuries": [
                {
                    "athlete": {"id": "3151", "displayName": "Cole Caufield"},
                    "status": "Day-To-Day",
                    "details": {"type": "Upper Body", "detail": "Upper Body", "returnDate": None},
                    "date": "2024-03-21T10:00:00Z",
                },
                {
                    "athlete": {"id": "3152", "displayName": "Nick Suzuki"},
                    "status": "Out",
                    "details": {"type": "Lower Body", "detail": "Knee", "returnDate": "2024-04-01"},
                    "date": "2024-03-20T08:00:00Z",
                },
            ],
        }
    ]
}


# ===========================================================================
# Successful sync
# ===========================================================================


class TestInjurySyncSuccess:
    def test_sync_ok_status(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                result = sync.sync(client)
        assert result.status == "ok"

    def test_sync_record_count(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                result = sync.sync(client)
        assert result.record_count == 2

    def test_sync_writes_parquet(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        parquet_path = tmp_path / f"injuries_{_TODAY}.parquet"
        assert parquet_path.exists()

    def test_sync_updates_manifest(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        manifest = sync.get_manifest()
        assert manifest["dates"][_TODAY]["status"] == "ok"
        assert "synced_at" in manifest["dates"][_TODAY]

    def test_sync_fetch_date_is_today(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                result = sync.sync(client)
        assert result.fetch_date == _TODAY


# ===========================================================================
# TTL / skip logic
# ===========================================================================


class TestInjurySyncTTL:
    def test_second_call_within_ttl_returns_skipped(self, tmp_path):
        sync = InjurySync(tmp_path, ttl_hours=2.0)
        call_count = 0

        def mock_fetch():
            nonlocal call_count
            call_count += 1
            return _MINIMAL_ESPN_RESPONSE

        with patch.object(InjuryClient, "fetch_injuries", side_effect=mock_fetch):
            with InjuryClient() as client:
                r1 = sync.sync(client)
                r2 = sync.sync(client)

        assert r1.status == "ok"
        assert r2.status == "skipped"
        assert call_count == 1  # HTTP called only once

    def test_skipped_result_preserves_record_count(self, tmp_path):
        sync = InjurySync(tmp_path, ttl_hours=2.0)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                r1 = sync.sync(client)
                r2 = sync.sync(client)
        assert r2.record_count == r1.record_count

    def test_expired_ttl_refetches(self, tmp_path):
        """After TTL expires, the sync should re-fetch even if status was ok."""
        sync = InjurySync(tmp_path, ttl_hours=1.0)
        call_count = 0

        def mock_fetch():
            nonlocal call_count
            call_count += 1
            return _MINIMAL_ESPN_RESPONSE

        # Plant a stale "ok" entry in the manifest with synced_at > 1h ago
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        stale_manifest = {
            "format_version": 1,
            "dates": {
                _TODAY: {
                    "status": "ok",
                    "record_count": 2,
                    "warning_count": 0,
                    "synced_at": stale_time,
                }
            },
        }
        manifest_path = tmp_path / "manifest.json"
        import json
        manifest_path.write_text(json.dumps(stale_manifest))

        with patch.object(InjuryClient, "fetch_injuries", side_effect=mock_fetch):
            with InjuryClient() as client:
                result = sync.sync(client)

        assert result.status == "ok"
        assert call_count == 1  # re-fetched

    def test_corrupt_synced_at_triggers_refetch(self, tmp_path):
        """Corrupt synced_at timestamp should not crash — triggers re-fetch."""
        sync = InjurySync(tmp_path, ttl_hours=1.0)
        import json
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "format_version": 1,
            "dates": {
                _TODAY: {
                    "status": "ok",
                    "record_count": 2,
                    "warning_count": 0,
                    "synced_at": "NOT_A_TIMESTAMP",
                }
            },
        }))

        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                result = sync.sync(client)

        assert result.status == "ok"


# ===========================================================================
# Error handling
# ===========================================================================


class TestInjurySyncErrorHandling:
    def test_fetch_error_returns_error_fetch_status(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(
            InjuryClient,
            "fetch_injuries",
            side_effect=InjuryFeedError("HTTP 503", status_code=503),
        ):
            with InjuryClient() as client:
                result = sync.sync(client)
        assert result.status == "error_fetch"

    def test_fetch_error_updates_manifest(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(
            InjuryClient,
            "fetch_injuries",
            side_effect=InjuryFeedError("HTTP 503", status_code=503),
        ):
            with InjuryClient() as client:
                sync.sync(client)
        manifest = sync.get_manifest()
        assert manifest["dates"][_TODAY]["status"] == "error_fetch"

    def test_fetch_error_error_message_stored(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(
            InjuryClient,
            "fetch_injuries",
            side_effect=InjuryFeedError("HTTP 503", status_code=503),
        ):
            with InjuryClient() as client:
                result = sync.sync(client)
        assert result.error_message is not None
        assert "503" in result.error_message


# ===========================================================================
# Empty records
# ===========================================================================


class TestInjurySyncEmpty:
    def test_empty_response_writes_empty_schema_parquet(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value={"injuries": []}):
            with InjuryClient() as client:
                result = sync.sync(client)

        assert result.status == "empty"
        parquet_path = tmp_path / f"injuries_{_TODAY}.parquet"
        assert parquet_path.exists()
        df = pl.read_parquet(parquet_path)
        assert len(df) == 0
        # Schema columns must be present even on empty file
        for col in EMPTY_SCHEMA:
            assert col in df.columns

    def test_empty_result_record_count_zero(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value={"injuries": []}):
            with InjuryClient() as client:
                result = sync.sync(client)
        assert result.record_count == 0


# ===========================================================================
# get_injuries
# ===========================================================================


class TestGetInjuries:
    def test_get_injuries_today_returns_dataframe(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        df = sync.get_injuries()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_get_injuries_missing_date_returns_empty_df(self, tmp_path):
        sync = InjurySync(tmp_path)
        df = sync.get_injuries("1990-01-01")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_get_injuries_columns_match_schema(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        df = sync.get_injuries()
        for col in EMPTY_SCHEMA:
            assert col in df.columns

    def test_get_injuries_specific_date(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        df = sync.get_injuries(_TODAY)
        assert len(df) == 2


# ===========================================================================
# get_player_status
# ===========================================================================


class TestGetPlayerStatus:
    def test_known_player_returns_correct_status(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        status = sync.get_player_status("3151")
        assert status == InjuryStatus.DTD

    def test_out_player_returns_out(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        status = sync.get_player_status("3152")
        assert status == InjuryStatus.OUT

    def test_absent_player_returns_none(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        status = sync.get_player_status("9999999")
        assert status is None

    def test_no_parquet_for_date_returns_none(self, tmp_path):
        sync = InjurySync(tmp_path)
        status = sync.get_player_status("3151", date="1990-01-01")
        assert status is None


# ===========================================================================
# clear_errors
# ===========================================================================


class TestClearErrors:
    def test_clear_errors_removes_error_fetch(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(
            InjuryClient,
            "fetch_injuries",
            side_effect=InjuryFeedError("503", status_code=503),
        ):
            with InjuryClient() as client:
                sync.sync(client)

        removed = sync.clear_errors()
        assert removed == 1
        manifest = sync.get_manifest()
        assert _TODAY not in manifest["dates"]

    def test_clear_errors_keeps_ok(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)

        removed = sync.clear_errors()
        assert removed == 0
        manifest = sync.get_manifest()
        assert manifest["dates"][_TODAY]["status"] == "ok"

    def test_clear_errors_removes_empty_status(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value={"injuries": []}):
            with InjuryClient() as client:
                sync.sync(client)

        removed = sync.clear_errors()
        assert removed == 1

    def test_clear_errors_returns_count(self, tmp_path):
        sync = InjurySync(tmp_path)
        # Force two different days with errors by manually writing manifest
        import json
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "format_version": 1,
            "dates": {
                "2024-01-01": {"status": "error_fetch", "record_count": 0, "warning_count": 0, "synced_at": ""},
                "2024-01-02": {"status": "error_fetch", "record_count": 0, "warning_count": 0, "synced_at": ""},
                "2024-01-03": {"status": "ok", "record_count": 5, "warning_count": 0, "synced_at": ""},
            },
        }))
        removed = sync.clear_errors()
        assert removed == 2
        manifest = sync.get_manifest()
        assert "2024-01-03" in manifest["dates"]
        assert "2024-01-01" not in manifest["dates"]


# ===========================================================================
# get_manifest
# ===========================================================================


class TestGetManifest:
    def test_missing_manifest_returns_empty_structure(self, tmp_path):
        sync = InjurySync(tmp_path)
        manifest = sync.get_manifest()
        assert manifest == {"format_version": 1, "dates": {}}

    def test_manifest_synced_at_present_after_sync(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        manifest = sync.get_manifest()
        assert "synced_at" in manifest["dates"][_TODAY]


# ===========================================================================
# Integration smoke
# ===========================================================================


class TestInjurySyncIntegration:
    def test_sync_to_get_injuries_full_pipeline(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                result = sync.sync(client)

        assert result.status == "ok"
        assert result.record_count == 2

        df = sync.get_injuries()
        assert "player_id_espn" in df.columns
        assert "team_code" in df.columns
        assert "status" in df.columns
        assert len(df) == 2

    def test_status_values_in_dataframe_are_strings(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        df = sync.get_injuries()
        statuses = df["status"].to_list()
        assert all(isinstance(s, str) for s in statuses)

    def test_filter_by_team_code(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        df = sync.get_injuries()
        mtl = df.filter(pl.col("team_code") == "MTL")
        assert len(mtl) == 2

    def test_filter_by_status_out(self, tmp_path):
        sync = InjurySync(tmp_path)
        with patch.object(InjuryClient, "fetch_injuries", return_value=_MINIMAL_ESPN_RESPONSE):
            with InjuryClient() as client:
                sync.sync(client)
        df = sync.get_injuries()
        out_players = df.filter(pl.col("status") == "Out")
        assert len(out_players) == 1
        assert out_players["player_name"][0] == "Nick Suzuki"
