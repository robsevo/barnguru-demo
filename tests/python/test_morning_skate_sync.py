"""Tests for MorningSkateSyncer (feature 1.10).

Covers:
- Successful sync: ok status, signal_count, Parquet written
- Empty report: status "empty", empty-schema Parquet written
- TTL skip logic: within TTL → skipped; past TTL → re-fetch
- Corrupt synced_at → triggers re-fetch
- API error: status "error_api", error_message stored
- get_signals: returns DataFrame with correct signals
- get_signals: missing date → empty-schema DataFrame
- clear_errors: removes error_api entries, keeps ok entries
- force=True bypasses TTL
- Manifest structure and content
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import polars as pl
import pytest

from data.morning_skate_parser import (
    SIGNAL_SCHEMA,
    LineupSignal,
    MorningSkateReport,
    MorningSkateParsedParser,
    SignalType,
)
from data.morning_skate_sync import MorningSkateSyncer, ReportSyncResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
_MODEL = "claude-haiku-4-5-20251001"

_SIGNALS_TWO = [
    LineupSignal("Connor McDavid", SignalType.PRACTICING_FULL, 0.95, "McDavid full."),
    LineupSignal("Leon Draisaitl", SignalType.PRACTICING_LIMITED, 0.7, "Draisaitl limited."),
]


def _make_parser_mock(signals: list[LineupSignal] | None = None, raise_exc: Exception | None = None) -> MorningSkateParsedParser:
    """Build a MorningSkateParsedParser that returns a canned report or raises."""
    client = MagicMock()
    parser = MagicMock(spec=MorningSkateParsedParser)

    if raise_exc is not None:
        parser.parse.side_effect = raise_exc
    else:
        effective_signals = signals if signals is not None else _SIGNALS_TWO
        def _parse(raw_text, team_abbrev=None, date=None, fetched_at=None):
            return MorningSkateReport(
                date=date or _TODAY,
                team_abbrev=team_abbrev,
                raw_text=raw_text,
                signals=effective_signals,
                fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(),
                llm_model=_MODEL,
                warning_count=0,
                input_tokens=100,
                output_tokens=40,
            )
        parser.parse.side_effect = _parse

    return parser


def _make_syncer(tmp_path, ttl_hours: float = 1.0) -> MorningSkateSyncer:
    return MorningSkateSyncer(tmp_path, ttl_hours=ttl_hours)


# ===========================================================================
# Successful sync
# ===========================================================================


class TestSyncSuccess:
    def test_sync_ok_status(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        result = syncer.sync(parser, "EDM morning skate...", date=_TODAY, team_abbrev="EDM")
        assert result.status == "ok"

    def test_sync_signal_count(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert result.signal_count == 2

    def test_sync_writes_parquet(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert (tmp_path / f"signals_{_TODAY}_EDM.parquet").exists()

    def test_sync_updates_manifest(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        manifest = syncer.get_manifest()
        key = f"{_TODAY}:EDM"
        assert manifest["dates"][key]["status"] == "ok"
        assert "synced_at" in manifest["dates"][key]

    def test_sync_date_defaults_to_today(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        result = syncer.sync(parser, "text", team_abbrev="EDM")
        assert result.date == _TODAY

    def test_sync_no_team_writes_all_parquet(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY)
        assert (tmp_path / f"signals_{_TODAY}_all.parquet").exists()

    def test_sync_tokens_recorded(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert result.input_tokens == 100
        assert result.output_tokens == 40


# ===========================================================================
# Empty report
# ===========================================================================


class TestSyncEmpty:
    def test_empty_report_status(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock(signals=[])
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="TOR")
        assert result.status == "empty"

    def test_empty_report_signal_count_zero(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock(signals=[])
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="TOR")
        assert result.signal_count == 0

    def test_empty_report_writes_empty_schema_parquet(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock(signals=[])
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="TOR")
        path = tmp_path / f"signals_{_TODAY}_TOR.parquet"
        assert path.exists()
        df = pl.read_parquet(path)
        assert len(df) == 0
        for col in SIGNAL_SCHEMA:
            assert col in df.columns


# ===========================================================================
# TTL / skip logic
# ===========================================================================


class TestTTL:
    def test_second_call_within_ttl_returns_skipped(self, tmp_path):
        syncer = _make_syncer(tmp_path, ttl_hours=2.0)
        parser = _make_parser_mock()
        r1 = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        r2 = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert r1.status == "ok"
        assert r2.status == "skipped"
        # Parser only called once
        assert parser.parse.call_count == 1

    def test_skipped_preserves_signal_count(self, tmp_path):
        syncer = _make_syncer(tmp_path, ttl_hours=2.0)
        parser = _make_parser_mock()
        r1 = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        r2 = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert r2.signal_count == r1.signal_count

    def test_expired_ttl_refetches(self, tmp_path):
        syncer = _make_syncer(tmp_path, ttl_hours=1.0)
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        key = f"{_TODAY}:EDM"
        (tmp_path / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "dates": {
                key: {
                    "status": "ok",
                    "signal_count": 2,
                    "warning_count": 0,
                    "synced_at": stale_time,
                }
            },
        }))
        parser = _make_parser_mock()
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert result.status == "ok"
        assert parser.parse.call_count == 1

    def test_corrupt_synced_at_triggers_refetch(self, tmp_path):
        syncer = _make_syncer(tmp_path, ttl_hours=1.0)
        key = f"{_TODAY}:EDM"
        (tmp_path / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "dates": {
                key: {
                    "status": "ok",
                    "signal_count": 2,
                    "warning_count": 0,
                    "synced_at": "NOT_A_TIMESTAMP",
                }
            },
        }))
        parser = _make_parser_mock()
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert result.status == "ok"

    def test_force_bypasses_ttl(self, tmp_path):
        syncer = _make_syncer(tmp_path, ttl_hours=99.0)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        # Within TTL, but force=True
        r2 = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM", force=True)
        assert r2.status == "ok"
        assert parser.parse.call_count == 2

    def test_error_entry_not_skipped_by_ttl(self, tmp_path):
        syncer = _make_syncer(tmp_path, ttl_hours=1.0)
        key = f"{_TODAY}:EDM"
        (tmp_path / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "dates": {
                key: {
                    "status": "error_api",
                    "signal_count": 0,
                    "warning_count": 0,
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                    "error_message": "HTTP 503",
                }
            },
        }))
        parser = _make_parser_mock()
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        # Error entries are not skipped even within TTL
        assert result.status == "ok"


# ===========================================================================
# API error handling
# ===========================================================================


class TestSyncErrors:
    def test_api_exception_returns_error_status(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock(raise_exc=Exception("HTTP 503"))
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert result.status == "error_api"

    def test_api_exception_stores_error_message(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock(raise_exc=Exception("timeout error"))
        result = syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        assert result.error_message is not None
        assert "timeout" in result.error_message

    def test_api_exception_updates_manifest(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock(raise_exc=Exception("fail"))
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        manifest = syncer.get_manifest()
        key = f"{_TODAY}:EDM"
        assert manifest["dates"][key]["status"] == "error_api"


# ===========================================================================
# get_signals
# ===========================================================================


class TestGetSignals:
    def test_get_signals_returns_dataframe(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        df = syncer.get_signals(_TODAY, team_abbrev="EDM")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_get_signals_missing_date_returns_empty(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        df = syncer.get_signals("1990-01-01", team_abbrev="EDM")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        for col in SIGNAL_SCHEMA:
            assert col in df.columns

    def test_get_signals_specific_team(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser_edm = _make_parser_mock(signals=[
            LineupSignal("Player EDM", SignalType.PRACTICING_FULL, 0.9, "q"),
        ])
        syncer.sync(parser_edm, "text", date=_TODAY, team_abbrev="EDM")
        df = syncer.get_signals(_TODAY, team_abbrev="EDM")
        assert len(df) == 1
        assert df["player_name"][0] == "Player EDM"

    def test_get_signals_columns_match_schema(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        df = syncer.get_signals(_TODAY, team_abbrev="EDM")
        for col in SIGNAL_SCHEMA:
            assert col in df.columns

    def test_get_signals_default_date_is_today(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        # Call without date arg — should default to today
        df = syncer.get_signals(team_abbrev="EDM")
        assert len(df) == 2


# ===========================================================================
# clear_errors
# ===========================================================================


class TestClearErrors:
    def test_clear_errors_removes_error_api(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock(raise_exc=Exception("fail"))
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        removed = syncer.clear_errors()
        assert removed == 1
        key = f"{_TODAY}:EDM"
        assert key not in syncer.get_manifest().get("dates", {})

    def test_clear_errors_keeps_ok(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        removed = syncer.clear_errors()
        assert removed == 0
        key = f"{_TODAY}:EDM"
        assert syncer.get_manifest()["dates"][key]["status"] == "ok"

    def test_clear_errors_count_multiple(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        (tmp_path / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "dates": {
                "2026-01-01:EDM": {"status": "error_api", "signal_count": 0,
                                    "warning_count": 0, "synced_at": ""},
                "2026-01-02:TOR": {"status": "error_parse", "signal_count": 0,
                                    "warning_count": 0, "synced_at": ""},
                "2026-01-03:MTL": {"status": "ok", "signal_count": 3,
                                    "warning_count": 0, "synced_at": ""},
            },
        }))
        removed = syncer.clear_errors()
        assert removed == 2
        assert "2026-01-03:MTL" in syncer.get_manifest()["dates"]


# ===========================================================================
# Manifest structure
# ===========================================================================


class TestManifest:
    def test_get_manifest_empty_returns_base_structure(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        assert syncer.get_manifest() == {"format_version": 1, "dates": {}}

    def test_manifest_contains_synced_at(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        syncer.sync(parser, "text", date=_TODAY, team_abbrev="EDM")
        key = f"{_TODAY}:EDM"
        assert "synced_at" in syncer.get_manifest()["dates"][key]

    def test_manifest_corrupt_json_returns_empty(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        (tmp_path / "manifest.json").write_text("NOT JSON {{{")
        assert syncer.get_manifest() == {"format_version": 1, "dates": {}}


# ===========================================================================
# Integration smoke
# ===========================================================================


class TestIntegrationSmoke:
    def test_full_pipeline(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser = _make_parser_mock()
        result = syncer.sync(parser, "EDM morning skate...", date=_TODAY, team_abbrev="EDM")
        assert result.status == "ok"
        assert result.signal_count == 2

        df = syncer.get_signals(_TODAY, team_abbrev="EDM")
        assert "player_name" in df.columns
        assert "signal_type" in df.columns
        assert "confidence" in df.columns
        assert len(df) == 2

    def test_two_teams_separate_parquet(self, tmp_path):
        syncer = _make_syncer(tmp_path)
        parser_edm = _make_parser_mock(signals=[
            LineupSignal("EDM Player", SignalType.EXPECTED_IN, 0.9, "q"),
        ])
        parser_tor = _make_parser_mock(signals=[
            LineupSignal("TOR Player", SignalType.EXPECTED_OUT, 0.85, "q"),
        ])
        syncer.sync(parser_edm, "text", date=_TODAY, team_abbrev="EDM")
        syncer.sync(parser_tor, "text", date=_TODAY, team_abbrev="TOR")

        assert (tmp_path / f"signals_{_TODAY}_EDM.parquet").exists()
        assert (tmp_path / f"signals_{_TODAY}_TOR.parquet").exists()

        df_edm = syncer.get_signals(_TODAY, team_abbrev="EDM")
        df_tor = syncer.get_signals(_TODAY, team_abbrev="TOR")
        assert df_edm["player_name"][0] == "EDM Player"
        assert df_tor["player_name"][0] == "TOR Player"
