"""Tests for Evolving Hockey client, parser, and sync orchestrator (Step 1.7)."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from data.evolving_hockey_client import EvolvingHockeyClient, EvolvingHockeyError
from data.evolving_hockey_parser import EvolvingHockeyParser, RAPMRecord, ZoneEntryRecord
from data.evolving_hockey_sync import EvolvingHockeySync, SyncResult, SyncSummary
from data.pbp_parser import DataMissingWarning


# ---------------------------------------------------------------------------
# Frozen fixtures
# ---------------------------------------------------------------------------

_RAPM_HEADER = (
    "Player,PlayerID,Season,Position,Team,GP,TOI,"
    "xGF_60,xGA_60,GF_60,GA_60,CF_60,CA_60,FF_60,FA_60,"
    "RAPM_xGF,RAPM_xGA,RAPM_xG,RAPM_GF,RAPM_GA,RAPM_G"
)
_RAPM_ROW_1 = "Matthews,8480801,2021,C,TOR,56,1182.3,3.21,2.14,2.98,2.01,58.3,49.7,52.1,44.2,0.82,-0.31,0.51,0.74,-0.28,0.46"
_RAPM_ROW_2 = "Marner,8481522,2021,R,TOR,55,1104.7,3.08,2.19,2.77,1.98,56.1,48.3,50.4,43.1,0.71,-0.24,0.47,0.61,-0.21,0.40"
_VALID_RAPM_CSV = f"{_RAPM_HEADER}\n{_RAPM_ROW_1}\n{_RAPM_ROW_2}\n"

_ZE_HEADER = (
    "Game_Id,season,game_date,Home_Team,Away_Team,Team,"
    "ZE_total,ZE_carries,ZE_dumps,ZE_carry_pct,"
    "ZX_total,ZX_controlled,ZX_dump_outs,ZX_controlled_pct"
)
_ZE_ROW_1 = "2021020001,2021,2021-10-13,TOR,MTL,TOR,28,18,10,0.643,25,14,11,0.560"
_ZE_ROW_2 = "2021020001,2021,2021-10-13,TOR,MTL,MTL,26,15,11,0.577,28,16,12,0.571"
_VALID_ZE_CSV = f"{_ZE_HEADER}\n{_ZE_ROW_1}\n{_ZE_ROW_2}\n"


# ---------------------------------------------------------------------------
# Client validation
# ---------------------------------------------------------------------------


class TestEvolvingHockeyClientValidation:
    def test_download_rapm_season_too_old(self):
        with EvolvingHockeyClient() as client:
            with pytest.raises(ValueError, match="season must be >= 2000"):
                client.download_rapm(1999)

    def test_download_zone_entries_season_too_old(self):
        with EvolvingHockeyClient() as client:
            with pytest.raises(ValueError, match="season must be >= 2000"):
                client.download_zone_entries(1999)


# ---------------------------------------------------------------------------
# RAPM parser — validation
# ---------------------------------------------------------------------------


class TestEvolvingHockeyParserRAPMValidation:
    def setup_method(self):
        self.parser = EvolvingHockeyParser()

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.parser.parse_rapm(b"", 2021)

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.parser.parse_rapm(b"   \n  ", 2021)

    def test_missing_critical_header_raises(self):
        # Remove RAPM_xG column from header
        bad_header = "Player,PlayerID,Season,Position,Team,GP,TOI"
        csv = f"{bad_header}\nMatthews,123,2021,C,TOR,56,1182.3\n"
        with pytest.raises(ValueError, match="RAPM_xG"):
            self.parser.parse_rapm(csv.encode(), 2021)


# ---------------------------------------------------------------------------
# RAPM parser — known-answer tests
# ---------------------------------------------------------------------------


class TestEvolvingHockeyParserRAPMKnownAnswer:
    def setup_method(self):
        self.parser = EvolvingHockeyParser()

    def test_two_rows_yield_two_records(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        assert len(records) == 2

    def test_first_record_player_name(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        assert records[0].player_name == "Matthews"

    def test_first_record_rapm_xg(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        assert records[0].rapm_xg == pytest.approx(0.51)

    def test_first_record_gp_is_int(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        assert isinstance(records[0].gp, int)
        assert records[0].gp == 56

    def test_first_record_toi_is_float(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        assert isinstance(records[0].toi, float)
        assert records[0].toi == pytest.approx(1182.3)

    def test_season_from_csv(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        assert records[0].season == 2021
        assert records[1].season == 2021

    def test_second_record_player_name(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        assert records[1].player_name == "Marner"


# ---------------------------------------------------------------------------
# RAPM parser — output range checks
# ---------------------------------------------------------------------------


class TestEvolvingHockeyParserRAPMOutputRange:
    def setup_method(self):
        self.parser = EvolvingHockeyParser()

    def test_gp_non_negative(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        for r in records:
            assert r.gp >= 0

    def test_toi_non_negative(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        for r in records:
            assert r.toi >= 0.0

    def test_rapm_xg_in_expected_range(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        for r in records:
            assert -5.0 < r.rapm_xg < 5.0


# ---------------------------------------------------------------------------
# RAPM parser — edge cases
# ---------------------------------------------------------------------------


class TestEvolvingHockeyParserRAPMEdgeCases:
    def setup_method(self):
        self.parser = EvolvingHockeyParser()

    def test_missing_player_warns_and_skips(self):
        row = ",8480801,2021,C,TOR,56,1182.3,3.21,2.14,2.98,2.01,58.3,49.7,52.1,44.2,0.82,-0.31,0.51,0.74,-0.28,0.46"
        csv = f"{_RAPM_HEADER}\n{row}\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse_rapm(csv.encode(), 2021)
        assert len(records) == 0
        assert any("Player" in str(warning.message) for warning in w)

    def test_missing_rapm_xg_warns_and_skips(self):
        # Replace RAPM_xG value with empty
        row = "Matthews,8480801,2021,C,TOR,56,1182.3,3.21,2.14,2.98,2.01,58.3,49.7,52.1,44.2,0.82,-0.31,,0.74,-0.28,0.46"
        csv = f"{_RAPM_HEADER}\n{row}\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse_rapm(csv.encode(), 2021)
        assert len(records) == 0
        assert any("RAPM_xG" in str(warning.message) for warning in w)

    def test_rapm_xg_out_of_range_warns_but_keeps(self):
        # RAPM_xG = 6.0 is outside (-5, 5)
        row = "Matthews,8480801,2021,C,TOR,56,1182.3,3.21,2.14,2.98,2.01,58.3,49.7,52.1,44.2,0.82,-0.31,6.0,0.74,-0.28,0.46"
        csv = f"{_RAPM_HEADER}\n{row}\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse_rapm(csv.encode(), 2021)
        assert len(records) == 1
        assert records[0].rapm_xg == pytest.approx(6.0)
        assert any("outside expected range" in str(warning.message) for warning in w)

    def test_missing_player_id_is_none_no_warning(self):
        # No PlayerID column at all, or empty
        header = "Player,Season,Position,Team,GP,TOI,xGF_60,xGA_60,GF_60,GA_60,CF_60,CA_60,FF_60,FA_60,RAPM_xGF,RAPM_xGA,RAPM_xG,RAPM_GF,RAPM_GA,RAPM_G"
        row = "Matthews,2021,C,TOR,56,1182.3,3.21,2.14,2.98,2.01,58.3,49.7,52.1,44.2,0.82,-0.31,0.51,0.74,-0.28,0.46"
        csv = f"{header}\n{row}\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse_rapm(csv.encode(), 2021)
        assert len(records) == 1
        assert records[0].player_id is None
        assert len(w) == 0

    def test_extra_columns_silently_ignored(self):
        header = _RAPM_HEADER + ",ExtraColumn"
        row = _RAPM_ROW_1 + ",extra_value"
        csv = f"{header}\n{row}\n"
        records = self.parser.parse_rapm(csv.encode(), 2021)
        assert len(records) == 1

    def test_header_only_yields_empty_list(self):
        csv = f"{_RAPM_HEADER}\n"
        records = self.parser.parse_rapm(csv.encode(), 2021)
        assert records == []

    def test_bytes_decoded(self):
        records = self.parser.parse_rapm(_VALID_RAPM_CSV.encode("utf-8"), 2021)
        assert len(records) == 2

    def test_latin1_fallback(self):
        # Encode with latin-1 (e.g., accented characters in player name)
        csv = _RAPM_HEADER + "\n" + "M\xe4tthews,8480801,2021,C,TOR,56,1182.3,3.21,2.14,2.98,2.01,58.3,49.7,52.1,44.2,0.82,-0.31,0.51,0.74,-0.28,0.46\n"
        raw = csv.encode("latin-1")
        records = self.parser.parse_rapm(raw, 2021)
        assert len(records) == 1
        assert "tthews" in records[0].player_name


# ---------------------------------------------------------------------------
# Zone entry parser — validation
# ---------------------------------------------------------------------------


class TestEvolvingHockeyParserZoneEntryValidation:
    def setup_method(self):
        self.parser = EvolvingHockeyParser()

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.parser.parse_zone_entries(b"", 2021)

    def test_missing_critical_header_raises(self):
        bad_header = "Game_Id,season,game_date,Home_Team,Away_Team,Team,ZE_total,ZE_carries"
        csv = f"{bad_header}\n2021020001,2021,2021-10-13,TOR,MTL,TOR,28,18\n"
        with pytest.raises(ValueError, match="ZE_dumps"):
            self.parser.parse_zone_entries(csv.encode(), 2021)


# ---------------------------------------------------------------------------
# Zone entry parser — known-answer tests
# ---------------------------------------------------------------------------


class TestEvolvingHockeyParserZoneEntryKnownAnswer:
    def setup_method(self):
        self.parser = EvolvingHockeyParser()

    def test_two_rows_yield_two_records(self):
        records = self.parser.parse_zone_entries(_VALID_ZE_CSV.encode(), 2021)
        assert len(records) == 2

    def test_first_record_game_id(self):
        records = self.parser.parse_zone_entries(_VALID_ZE_CSV.encode(), 2021)
        assert records[0].game_id == "2021020001"

    def test_first_record_ze_carries(self):
        records = self.parser.parse_zone_entries(_VALID_ZE_CSV.encode(), 2021)
        assert records[0].ze_carries == 18

    def test_first_record_ze_dumps(self):
        records = self.parser.parse_zone_entries(_VALID_ZE_CSV.encode(), 2021)
        assert records[0].ze_dumps == 10

    def test_first_record_ze_total(self):
        records = self.parser.parse_zone_entries(_VALID_ZE_CSV.encode(), 2021)
        assert records[0].ze_total == 28

    def test_ze_total_equals_carries_plus_dumps(self):
        records = self.parser.parse_zone_entries(_VALID_ZE_CSV.encode(), 2021)
        for r in records:
            assert r.ze_carries + r.ze_dumps == r.ze_total


# ---------------------------------------------------------------------------
# Zone entry parser — edge cases
# ---------------------------------------------------------------------------


class TestEvolvingHockeyParserZoneEntryEdgeCases:
    def setup_method(self):
        self.parser = EvolvingHockeyParser()

    def test_missing_game_id_warns_and_skips(self):
        row = ",2021,2021-10-13,TOR,MTL,TOR,28,18,10,0.643,25,14,11,0.560"
        csv = f"{_ZE_HEADER}\n{row}\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse_zone_entries(csv.encode(), 2021)
        assert len(records) == 0
        assert any("Game_Id" in str(warning.message) for warning in w)

    def test_inconsistent_counts_warns_but_keeps(self):
        # ze_carries + ze_dumps (18 + 11 = 29) != ze_total (28)
        row = "2021020001,2021,2021-10-13,TOR,MTL,TOR,28,18,11,0.643,25,14,11,0.560"
        csv = f"{_ZE_HEADER}\n{row}\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataMissingWarning)
            records = self.parser.parse_zone_entries(csv.encode(), 2021)
        assert len(records) == 1
        assert records[0].ze_total == 28
        assert any("data inconsistency" in str(warning.message) for warning in w)

    def test_missing_optional_zx_total_is_none(self):
        header = "Game_Id,season,game_date,Home_Team,Away_Team,Team,ZE_total,ZE_carries,ZE_dumps,ZE_carry_pct"
        row = "2021020001,2021,2021-10-13,TOR,MTL,TOR,28,18,10,0.643"
        csv = f"{header}\n{row}\n"
        records = self.parser.parse_zone_entries(csv.encode(), 2021)
        assert len(records) == 1
        assert records[0].zx_total is None

    def test_header_only_yields_empty_list(self):
        csv = f"{_ZE_HEADER}\n"
        records = self.parser.parse_zone_entries(csv.encode(), 2021)
        assert records == []


# ---------------------------------------------------------------------------
# Parser to_polars methods
# ---------------------------------------------------------------------------


class TestEvolvingHockeyParserToPolars:
    def test_rapm_empty_list_returns_empty_dataframe(self):
        df = EvolvingHockeyParser.to_polars_rapm([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_zone_entries_empty_list_returns_empty_dataframe(self):
        df = EvolvingHockeyParser.to_polars_zone_entries([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_rapm_two_records_yields_two_rows(self):
        parser = EvolvingHockeyParser()
        records = parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        df = EvolvingHockeyParser.to_polars_rapm(records)
        assert len(df) == 2

    def test_zone_entries_two_records_yields_two_rows(self):
        parser = EvolvingHockeyParser()
        records = parser.parse_zone_entries(_VALID_ZE_CSV.encode(), 2021)
        df = EvolvingHockeyParser.to_polars_zone_entries(records)
        assert len(df) == 2

    def test_rapm_expected_columns_present(self):
        parser = EvolvingHockeyParser()
        records = parser.parse_rapm(_VALID_RAPM_CSV.encode(), 2021)
        df = EvolvingHockeyParser.to_polars_rapm(records)
        assert "player_name" in df.columns
        assert "rapm_xg" in df.columns
        assert "team" in df.columns

    def test_zone_entries_expected_columns_present(self):
        parser = EvolvingHockeyParser()
        records = parser.parse_zone_entries(_VALID_ZE_CSV.encode(), 2021)
        df = EvolvingHockeyParser.to_polars_zone_entries(records)
        assert "ze_carries" in df.columns
        assert "ze_carry_pct" in df.columns


# ---------------------------------------------------------------------------
# Sync — validation
# ---------------------------------------------------------------------------


class TestEvolvingHockeySyncValidation:
    def test_season_too_old_raises(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with EvolvingHockeyClient() as client:
            with pytest.raises(ValueError, match="season must be >= 2000"):
                sync.sync_season(client, 1999)


# ---------------------------------------------------------------------------
# Sync — error handling
# ---------------------------------------------------------------------------


class TestEvolvingHockeySyncErrorHandling:
    def test_rapm_404_returns_error_download(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          side_effect=EvolvingHockeyError("HTTP 404", 404)):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    results = sync.sync_season(client, 2021)
        rapm_r = next(r for r in results if r.dataset == "rapm")
        assert rapm_r.status == "error_download"
        assert rapm_r.record_count == 0

    def test_zone_entries_404_returns_error_download(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              side_effect=EvolvingHockeyError("HTTP 404", 404)):
                with EvolvingHockeyClient() as client:
                    results = sync.sync_season(client, 2021)
        ze_r = next(r for r in results if r.dataset == "zone_entries")
        assert ze_r.status == "error_download"

    def test_parse_error_returns_error_parse(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=b"bad,csv,data\n"):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    results = sync.sync_season(client, 2021)
        rapm_r = next(r for r in results if r.dataset == "rapm")
        assert rapm_r.status == "error_parse"

    def test_rapm_error_does_not_prevent_zone_entries_sync(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          side_effect=EvolvingHockeyError("HTTP 404", 404)):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    results = sync.sync_season(client, 2021)
        rapm_r = next(r for r in results if r.dataset == "rapm")
        ze_r = next(r for r in results if r.dataset == "zone_entries")
        assert rapm_r.status == "error_download"
        assert ze_r.status == "ok"
        assert (tmp_path / "zone_entries_2021.parquet").exists()
        assert not (tmp_path / "rapm_2021.parquet").exists()


# ---------------------------------------------------------------------------
# Sync — resume
# ---------------------------------------------------------------------------


class TestEvolvingHockeySyncResume:
    def test_second_call_skipped_http_not_called(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()) as mock_rapm:
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()) as mock_ze:
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)
                    # Second call
                    results = sync.sync_season(client, 2021)

        # Both datasets should be skipped on second call
        assert all(r.status == "skipped" for r in results)
        # HTTP should only have been called once each
        assert mock_rapm.call_count == 1
        assert mock_ze.call_count == 1

    def test_skipped_record_count_preserved(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)
                    results = sync.sync_season(client, 2021)

        rapm_r = next(r for r in results if r.dataset == "rapm")
        assert rapm_r.record_count == 2

    def test_partial_resume_rapm_ok_zone_entries_reruns(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        # First run: rapm ok, zone_entries fails
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              side_effect=EvolvingHockeyError("HTTP 503", 503)):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        # Second run: zone_entries should re-run (was error), rapm should skip
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()) as mock_rapm:
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()) as mock_ze:
                with EvolvingHockeyClient() as client:
                    results = sync.sync_season(client, 2021)

        rapm_r = next(r for r in results if r.dataset == "rapm")
        ze_r = next(r for r in results if r.dataset == "zone_entries")
        assert rapm_r.status == "skipped"
        assert ze_r.status == "ok"
        assert mock_rapm.call_count == 0  # not called on second run
        assert mock_ze.call_count == 1


# ---------------------------------------------------------------------------
# Sync — clear_errors
# ---------------------------------------------------------------------------


class TestEvolvingHockeySyncClearErrors:
    def test_removes_error_statuses(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          side_effect=EvolvingHockeyError("HTTP 404", 404)):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              side_effect=EvolvingHockeyError("HTTP 404", 404)):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        removed = sync.clear_errors()
        assert removed == 2
        manifest = sync.get_manifest()
        # Season key removed because both datasets cleared
        assert "2021" not in manifest["seasons"]

    def test_keeps_ok_entries(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        removed = sync.clear_errors()
        assert removed == 0
        manifest = sync.get_manifest()
        assert manifest["seasons"]["2021"]["rapm"]["status"] == "ok"
        assert manifest["seasons"]["2021"]["zone_entries"]["status"] == "ok"

    def test_season_key_removed_when_both_datasets_cleared(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          side_effect=EvolvingHockeyError("HTTP 404", 404)):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              side_effect=EvolvingHockeyError("HTTP 404", 404)):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        sync.clear_errors()
        manifest = sync.get_manifest()
        assert "2021" not in manifest["seasons"]

    def test_returns_correct_count(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          side_effect=EvolvingHockeyError("HTTP 404", 404)):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        removed = sync.clear_errors()
        assert removed == 1

    def test_empty_status_removed(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        # empty CSV for rapm triggers "empty" status
        rapm_header_only = f"{_RAPM_HEADER}\n"
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=rapm_header_only.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        manifest = sync.get_manifest()
        assert manifest["seasons"]["2021"]["rapm"]["status"] == "empty"
        removed = sync.clear_errors()
        assert removed == 1
        manifest = sync.get_manifest()
        assert "rapm" not in manifest["seasons"]["2021"]


# ---------------------------------------------------------------------------
# Sync — integration
# ---------------------------------------------------------------------------


class TestEvolvingHockeySyncIntegration:
    def test_both_parquets_written_and_readable(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        rapm_path = tmp_path / "rapm_2021.parquet"
        ze_path = tmp_path / "zone_entries_2021.parquet"
        assert rapm_path.exists()
        assert ze_path.exists()
        rapm_df = pl.read_parquet(rapm_path)
        ze_df = pl.read_parquet(ze_path)
        assert len(rapm_df) == 2
        assert len(ze_df) == 2

    def test_rapm_parquet_has_expected_columns(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        df = pl.read_parquet(tmp_path / "rapm_2021.parquet")
        assert "player_name" in df.columns
        assert "rapm_xg" in df.columns
        assert "team" in df.columns

    def test_zone_entries_parquet_has_expected_columns(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        df = pl.read_parquet(tmp_path / "zone_entries_2021.parquet")
        assert "ze_carries" in df.columns
        assert "ze_carry_pct" in df.columns

    def test_manifest_has_synced_at(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_season(client, 2021)

        manifest = sync.get_manifest()
        assert "synced_at" in manifest["seasons"]["2021"]["rapm"]
        assert "synced_at" in manifest["seasons"]["2021"]["zone_entries"]

    def test_sync_seasons_writes_all_four_files(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    sync.sync_seasons(client, [2021, 2022])

        assert (tmp_path / "rapm_2021.parquet").exists()
        assert (tmp_path / "zone_entries_2021.parquet").exists()
        assert (tmp_path / "rapm_2022.parquet").exists()
        assert (tmp_path / "zone_entries_2022.parquet").exists()

    def test_sync_summary_totals_correct(self, tmp_path):
        sync = EvolvingHockeySync(tmp_path)
        with patch.object(EvolvingHockeyClient, "download_rapm",
                          return_value=_VALID_RAPM_CSV.encode()):
            with patch.object(EvolvingHockeyClient, "download_zone_entries",
                              return_value=_VALID_ZE_CSV.encode()):
                with EvolvingHockeyClient() as client:
                    summary = sync.sync_seasons(client, [2021, 2022])

        # 2 seasons × 2 datasets = 4 "ok" results
        assert summary.successful == 4
        assert summary.skipped == 0
        assert summary.failed == 0
        # 2 rapm records per season × 2 seasons + 2 ze records per season × 2 seasons = 8
        assert summary.total_records == 8
        assert summary.seasons_requested == [2021, 2022]
