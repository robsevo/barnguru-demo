"""Tests for MoneyPuck sync pipeline (client, parser, orchestrator).

Mirrors test contract from PLAN.md §1.6:
- Input validation
- Known-answer regression
- Output range
- Edge cases
- Integration smoke
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from data.moneypuck_client import MoneyPuckClient, MoneyPuckError
from data.moneypuck_parser import DataMissingWarning, MoneyPuckParser, ShotRecord
from data.moneypuck_sync import MoneyPuckSync, SyncResult


# ---------------------------------------------------------------------------
# Frozen fixture CSV
# ---------------------------------------------------------------------------

_HEADER = (
    "shotID,season,isPlayoffGame,game_id,homeTeamCode,awayTeamCode,teamCode,"
    "shooterName,shooterId,goalieIdForShot,xGoal,xOnGoal,"
    "xCord,yCord,arenaAdjustedXCord,arenaAdjustedYCord,arenaAdjustedShotDistance,"
    "shotAngle,shotDistance,shotType,shotResult,period,time,goal,"
    "homeSkatersOnIce,awaySkatersOnIce,shooterLeftRight,event,"
    "playerPositionThatDidEvent,lastEventTeam"
)
_ROW_1 = (
    "101,2021,0,2021020001,TOR,MTL,TOR,Matthews,8480801,8480353,0.142,0.831,"
    "55.0,12.0,54.2,11.8,38.5,14.3,38.5,WRIST,SHOT,1,450,0,5,5,L,SHOT,C,MTL"
)
_ROW_2 = (
    "102,2021,0,2021020001,TOR,MTL,MTL,Price,8471679,8480801,0.031,0.512,"
    "-61.0,5.0,-60.5,4.8,62.1,4.7,62.1,SNAP,GOAL,2,900,1,5,5,R,GOAL,C,TOR"
)
_VALID_CSV = f"{_HEADER}\n{_ROW_1}\n{_ROW_2}\n"


# ===========================================================================
# MoneyPuckClient — input validation
# ===========================================================================


class TestMoneyPuckClientValidation:
    def test_download_shots_season_too_low_raises(self):
        client = MoneyPuckClient()
        with pytest.raises(ValueError, match="season must be >= 2000"):
            client.download_shots(1999)

    def test_download_shots_season_zero_raises(self):
        client = MoneyPuckClient()
        with pytest.raises(ValueError):
            client.download_shots(0)


# ===========================================================================
# MoneyPuckParser — input validation
# ===========================================================================


class TestMoneyPuckParserValidation:
    def setup_method(self):
        self.parser = MoneyPuckParser()

    def test_parse_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.parser.parse(b"", season=2021)

    def test_parse_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.parser.parse("", season=2021)

    def test_parse_missing_critical_column_raises(self):
        # Remove "xGoal" from header
        bad_header = _HEADER.replace(",xGoal", "")
        bad_csv = f"{bad_header}\n{_ROW_1}\n"
        with pytest.raises(ValueError, match="missing critical column"):
            self.parser.parse(bad_csv, season=2021)

    def test_parse_missing_shot_id_column_raises(self):
        bad_header = _HEADER.replace("shotID,", "")
        with pytest.raises(ValueError, match="missing critical column"):
            self.parser.parse(bad_header, season=2021)


# ===========================================================================
# MoneyPuckParser — known-answer regression
# ===========================================================================


class TestMoneyPuckParserKnownAnswer:
    def setup_method(self):
        self.parser = MoneyPuckParser()
        self.records = self.parser.parse(_VALID_CSV, season=2021)

    def test_two_rows_parsed(self):
        assert len(self.records) == 2

    def test_first_record_shot_id(self):
        assert self.records[0].shot_id == "101"

    def test_first_record_x_goal(self):
        assert self.records[0].x_goal == pytest.approx(0.142)

    def test_first_record_is_goal_false(self):
        assert self.records[0].is_goal is False

    def test_second_record_is_goal_true(self):
        assert self.records[1].is_goal is True

    def test_second_record_shot_id(self):
        assert self.records[1].shot_id == "102"

    def test_second_record_x_goal(self):
        assert self.records[1].x_goal == pytest.approx(0.031)

    def test_shooter_id_is_int(self):
        assert isinstance(self.records[0].shooter_id, int)
        assert self.records[0].shooter_id == 8480801

    def test_is_playoff_is_bool(self):
        assert isinstance(self.records[0].is_playoff, bool)
        assert self.records[0].is_playoff is False

    def test_is_goal_is_bool(self):
        assert isinstance(self.records[0].is_goal, bool)

    def test_season_matches(self):
        assert self.records[0].season == 2021
        assert self.records[1].season == 2021

    def test_home_team(self):
        assert self.records[0].home_team == "TOR"

    def test_shooter_name(self):
        assert self.records[0].shooter_name == "Matthews"


# ===========================================================================
# MoneyPuckParser — output range
# ===========================================================================


class TestMoneyPuckParserOutputRange:
    def setup_method(self):
        self.parser = MoneyPuckParser()
        self.records = self.parser.parse(_VALID_CSV, season=2021)

    def test_x_goal_in_unit_interval(self):
        for r in self.records:
            assert r.x_goal is not None
            assert 0.0 <= r.x_goal <= 1.0

    def test_all_shot_ids_non_empty(self):
        for r in self.records:
            assert r.shot_id != ""

    def test_season_correct(self):
        for r in self.records:
            assert r.season == 2021


# ===========================================================================
# MoneyPuckParser — edge cases
# ===========================================================================


class TestMoneyPuckParserEdgeCases:
    def setup_method(self):
        self.parser = MoneyPuckParser()

    def test_row_missing_shot_id_skipped_with_warning(self):
        row = _ROW_1.replace("101,", ",", 1)  # blank shotID
        csv_content = f"{_HEADER}\n{row}\n{_ROW_2}\n"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse(csv_content, season=2021)
        assert len(records) == 1
        assert any("shotID" in str(w.message) for w in caught)

    def test_row_missing_x_goal_skipped_with_warning(self):
        # Replace xGoal value with empty
        # _ROW_1 has ",0.142," — replace with ",,"
        row = _ROW_1.replace(",0.142,", ",,", 1)
        csv_content = f"{_HEADER}\n{row}\n{_ROW_2}\n"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse(csv_content, season=2021)
        assert len(records) == 1
        assert any("xGoal" in str(w.message) for w in caught)

    def test_x_goal_out_of_range_warns_keeps_row(self):
        row = _ROW_1.replace(",0.142,", ",1.5,", 1)
        csv_content = f"{_HEADER}\n{row}\n"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse(csv_content, season=2021)
        assert len(records) == 1
        assert records[0].x_goal == pytest.approx(1.5)
        assert any("outside [0,1]" in str(w.message) for w in caught)

    def test_missing_optional_shooter_name_is_none(self):
        # Replace "Matthews" with empty
        row = _ROW_1.replace(",Matthews,", ",,", 1)
        csv_content = f"{_HEADER}\n{row}\n"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse(csv_content, season=2021)
        assert len(records) == 1
        assert records[0].shooter_name is None
        # No DataMissingWarning for optional field
        assert not any("shooterName" in str(w.message) for w in caught)

    def test_extra_unknown_columns_ignored(self):
        extended_header = _HEADER + ",extraColumn1,extraColumn2"
        extended_row = _ROW_1 + ",foo,bar"
        csv_content = f"{extended_header}\n{extended_row}\n"
        records = self.parser.parse(csv_content, season=2021)
        assert len(records) == 1

    def test_is_playoff_one_becomes_true(self):
        # Change isPlayoffGame from 0 to 1
        row = _ROW_1.replace(",2021,0,", ",2021,1,", 1)
        csv_content = f"{_HEADER}\n{row}\n"
        records = self.parser.parse(csv_content, season=2021)
        assert len(records) == 1
        assert records[0].is_playoff is True

    def test_header_only_csv_returns_empty_list(self):
        csv_content = f"{_HEADER}\n"
        records = self.parser.parse(csv_content, season=2021)
        assert records == []

    def test_bytes_input_decoded(self):
        records = self.parser.parse(_VALID_CSV.encode("utf-8"), season=2021)
        assert len(records) == 2

    def test_latin1_bytes_decoded(self):
        # Simulate accented name via latin-1
        row_accent = _ROW_1.replace("Matthews", "Dési\xe9r")
        csv_content = f"{_HEADER}\n{row_accent}\n"
        records = self.parser.parse(csv_content.encode("latin-1"), season=2021)
        assert len(records) == 1


# ===========================================================================
# MoneyPuckParser — to_polars
# ===========================================================================


class TestMoneyPuckParserToPolars:
    def setup_method(self):
        self.parser = MoneyPuckParser()

    def test_empty_list_returns_empty_dataframe(self):
        df = MoneyPuckParser.to_polars([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_two_records_gives_two_rows(self):
        records = self.parser.parse(_VALID_CSV, season=2021)
        df = MoneyPuckParser.to_polars(records)
        assert len(df) == 2

    def test_expected_columns_present(self):
        records = self.parser.parse(_VALID_CSV, season=2021)
        df = MoneyPuckParser.to_polars(records)
        for col in ("shot_id", "x_goal", "is_goal", "game_id", "season"):
            assert col in df.columns


# ===========================================================================
# MoneyPuckSync — input validation
# ===========================================================================


class TestMoneyPuckSyncValidation:
    def test_sync_season_zero_raises(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        client = MagicMock(spec=MoneyPuckClient)
        with pytest.raises(ValueError, match="season must be >= 2000"):
            sync.sync_season(client, 0)

    def test_sync_season_1999_raises(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        client = MagicMock(spec=MoneyPuckClient)
        with pytest.raises(ValueError):
            sync.sync_season(client, 1999)


# ===========================================================================
# MoneyPuckSync — HTTP error handling
# ===========================================================================


class TestMoneyPuckSyncErrorHandling:
    def test_http_404_status_error_download(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        with patch.object(
            MoneyPuckClient,
            "download_shots",
            side_effect=MoneyPuckError("HTTP 404", status_code=404),
        ):
            with MoneyPuckClient() as client:
                result = sync.sync_season(client, 2021)

        assert result.status == "error_download"
        manifest = sync.get_manifest()
        assert manifest["seasons"]["2021"]["status"] == "error_download"

    def test_error_parse_on_bad_csv(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        # CSV missing the critical xGoal column entirely
        bad_csv = "shotID,season\n101,2021\n"
        with patch.object(
            MoneyPuckClient,
            "download_shots",
            return_value=bad_csv.encode(),
        ):
            with MoneyPuckClient() as client:
                result = sync.sync_season(client, 2021)

        assert result.status == "error_parse"


# ===========================================================================
# MoneyPuckSync — resume / skipped logic
# ===========================================================================


class TestMoneyPuckSyncResume:
    def test_second_call_skipped_http_not_called(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)

        call_count = 0

        def mock_download(season: int) -> bytes:
            nonlocal call_count
            call_count += 1
            return _VALID_CSV.encode()

        with patch.object(MoneyPuckClient, "download_shots", side_effect=mock_download):
            with MoneyPuckClient() as client:
                r1 = sync.sync_season(client, 2021)
                r2 = sync.sync_season(client, 2021)

        assert r1.status == "ok"
        assert r2.status == "skipped"
        assert call_count == 1  # HTTP called only once

    def test_skipped_result_preserves_shots_count(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)

        with patch.object(
            MoneyPuckClient, "download_shots", return_value=_VALID_CSV.encode()
        ):
            with MoneyPuckClient() as client:
                r1 = sync.sync_season(client, 2021)
                r2 = sync.sync_season(client, 2021)

        assert r2.shots_count == r1.shots_count


# ===========================================================================
# MoneyPuckSync — clear_errors
# ===========================================================================


class TestMoneyPuckSyncClearErrors:
    def test_clear_errors_removes_error_statuses(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)

        with patch.object(
            MoneyPuckClient,
            "download_shots",
            side_effect=MoneyPuckError("HTTP 404", status_code=404),
        ):
            with MoneyPuckClient() as client:
                sync.sync_season(client, 2021)
                sync.sync_season(client, 2022)

        removed = sync.clear_errors()
        assert removed == 2
        manifest = sync.get_manifest()
        assert "2021" not in manifest["seasons"]
        assert "2022" not in manifest["seasons"]

    def test_clear_errors_keeps_ok(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)

        with patch.object(
            MoneyPuckClient, "download_shots", return_value=_VALID_CSV.encode()
        ):
            with MoneyPuckClient() as client:
                sync.sync_season(client, 2021)

        removed = sync.clear_errors()
        assert removed == 0
        manifest = sync.get_manifest()
        assert manifest["seasons"]["2021"]["status"] == "ok"

    def test_clear_errors_removes_empty_status(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        # Header-only CSV → empty records → "empty" status
        header_only = f"{_HEADER}\n"
        with patch.object(
            MoneyPuckClient, "download_shots", return_value=header_only.encode()
        ):
            with MoneyPuckClient() as client:
                sync.sync_season(client, 2021)

        removed = sync.clear_errors()
        assert removed == 1


# ===========================================================================
# Integration smoke tests
# ===========================================================================


class TestMoneyPuckIntegration:
    def test_sync_season_parquet_written_readable(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        with patch.object(
            MoneyPuckClient, "download_shots", return_value=_VALID_CSV.encode()
        ):
            with MoneyPuckClient() as client:
                result = sync.sync_season(client, 2021)

        assert result.status == "ok"
        assert result.shots_count == 2

        parquet_path = tmp_path / "shots_2021.parquet"
        assert parquet_path.exists()

        df = pl.read_parquet(parquet_path)
        assert "game_id" in df.columns
        assert "x_goal" in df.columns
        assert "is_goal" in df.columns

    def test_sync_season_filter_goals(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        with patch.object(
            MoneyPuckClient, "download_shots", return_value=_VALID_CSV.encode()
        ):
            with MoneyPuckClient() as client:
                sync.sync_season(client, 2021)

        df = pl.read_parquet(tmp_path / "shots_2021.parquet")
        goals = df.filter(pl.col("is_goal") == True)  # noqa: E712
        assert len(goals) == 1

    def test_sync_seasons_multiple_parquets(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)

        # Build a 2022 CSV (same structure, different season)
        csv_2022 = _VALID_CSV.replace(",2021,", ",2022,")

        def mock_download(season: int) -> bytes:
            if season == 2021:
                return _VALID_CSV.encode()
            return csv_2022.encode()

        with patch.object(MoneyPuckClient, "download_shots", side_effect=mock_download):
            with MoneyPuckClient() as client:
                summary = sync.sync_seasons(client, [2021, 2022])

        assert (tmp_path / "shots_2021.parquet").exists()
        assert (tmp_path / "shots_2022.parquet").exists()
        assert summary.successful == 2
        assert summary.failed == 0

    def test_xg_feature_columns_selectable(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        with patch.object(
            MoneyPuckClient, "download_shots", return_value=_VALID_CSV.encode()
        ):
            with MoneyPuckClient() as client:
                sync.sync_season(client, 2021)

        df = pl.read_parquet(tmp_path / "shots_2021.parquet")
        xg_features = ["x_goal", "arena_adj_x", "arena_adj_y", "shot_angle", "shot_type"]
        for col in xg_features:
            assert col in df.columns

    def test_empty_season_writes_empty_schema_parquet(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        header_only = f"{_HEADER}\n"
        with patch.object(
            MoneyPuckClient, "download_shots", return_value=header_only.encode()
        ):
            with MoneyPuckClient() as client:
                result = sync.sync_season(client, 2021)

        assert result.status == "empty"
        parquet_path = tmp_path / "shots_2021.parquet"
        assert parquet_path.exists()
        df = pl.read_parquet(parquet_path)
        assert len(df) == 0

    def test_manifest_synced_at_present(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)
        with patch.object(
            MoneyPuckClient, "download_shots", return_value=_VALID_CSV.encode()
        ):
            with MoneyPuckClient() as client:
                sync.sync_season(client, 2021)

        manifest = sync.get_manifest()
        assert "synced_at" in manifest["seasons"]["2021"]

    def test_sync_summary_totals(self, tmp_path):
        sync = MoneyPuckSync(tmp_path)

        def mock_download(season: int) -> bytes:
            return _VALID_CSV.encode()

        with patch.object(MoneyPuckClient, "download_shots", side_effect=mock_download):
            with MoneyPuckClient() as client:
                summary = sync.sync_seasons(client, [2021, 2022])

        assert summary.successful == 2
        assert summary.skipped == 0
        assert summary.failed == 0
        assert summary.total_shots == 4  # 2 rows per season
        assert summary.seasons_requested == [2021, 2022]
