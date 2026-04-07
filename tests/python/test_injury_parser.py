"""Tests for InjuryParser (feature 1.8).

Covers:
- Known-answer regression (frozen ESPN JSON → expected InjuryRecord list)
- Status normalization (all known ESPN status strings)
- Missing critical fields (athlete.id, displayName, status) → warning + skip
- Unknown status → warning + Out (conservative default)
- Edge cases: empty response, team with no injuries, to_polars
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from data.injury_parser import InjuryParser, InjuryRecord, InjuryStatus
from data.pbp_parser import DataMissingWarning


# ---------------------------------------------------------------------------
# Frozen test fixture — never change without updating expected values below
# ---------------------------------------------------------------------------

_FETCHED_AT = "2024-03-21T10:00:00+00:00"

_FROZEN_RESPONSE: dict = {
    "injuries": [
        {
            "team": {
                "id": "1",
                "abbreviation": "BOS",
                "displayName": "Boston Bruins",
            },
            "injuries": [
                {
                    "athlete": {"id": "3150", "displayName": "Brad Marchand"},
                    "status": "Day-To-Day",
                    "details": {
                        "type": "Upper Body",
                        "detail": "Upper Body Injury",
                        "returnDate": None,
                    },
                    "date": "2024-03-20T14:00:00Z",
                },
                {
                    "athlete": {"id": "3151", "displayName": "David Pastrnak"},
                    "status": "Out",
                    "details": {
                        "type": "Lower Body",
                        "detail": "Lower Body",
                        "returnDate": "2024-04-01",
                    },
                    "date": "2024-03-19T10:00:00Z",
                },
            ],
        },
        {
            "team": {
                "id": "6",
                "abbreviation": "TOR",
                "displayName": "Toronto Maple Leafs",
            },
            "injuries": [
                {
                    "athlete": {"id": "8480801", "displayName": "Auston Matthews"},
                    "status": "Probable",
                    "details": {
                        "type": "Wrist",
                        "detail": "Wrist Soreness",
                        "returnDate": None,
                    },
                    "date": "2024-03-21T08:00:00Z",
                },
            ],
        },
    ]
}


# ===========================================================================
# Known-answer regression
# ===========================================================================


class TestInjuryParserKnownAnswer:
    def setup_method(self):
        self.records = InjuryParser.parse(_FROZEN_RESPONSE, _FETCHED_AT)

    def test_three_records_parsed(self):
        assert len(self.records) == 3

    def test_first_record_player_name(self):
        assert self.records[0].player_name == "Brad Marchand"

    def test_first_record_team_code(self):
        assert self.records[0].team_code == "BOS"

    def test_first_record_status_dtd(self):
        assert self.records[0].status == InjuryStatus.DTD

    def test_first_record_status_raw(self):
        assert self.records[0].status_raw == "Day-To-Day"

    def test_first_record_injury_type(self):
        assert self.records[0].injury_type == "Upper Body"

    def test_first_record_injury_detail(self):
        assert self.records[0].injury_detail == "Upper Body Injury"

    def test_first_record_return_date_none(self):
        assert self.records[0].return_date_estimate is None

    def test_first_record_player_id(self):
        assert self.records[0].player_id_espn == "3150"

    def test_second_record_status_out(self):
        assert self.records[1].status == InjuryStatus.OUT

    def test_second_record_return_date(self):
        assert self.records[1].return_date_estimate == "2024-04-01"

    def test_third_record_team_tor(self):
        assert self.records[2].team_code == "TOR"

    def test_third_record_status_dtd_from_probable(self):
        assert self.records[2].status == InjuryStatus.DTD
        assert self.records[2].status_raw == "Probable"

    def test_fetched_at_propagated(self):
        for record in self.records:
            assert record.fetched_at == _FETCHED_AT

    def test_reported_at_non_empty(self):
        assert self.records[0].reported_at == "2024-03-20T14:00:00Z"


# ===========================================================================
# Status normalization — parametrize all known ESPN strings
# ===========================================================================


@pytest.mark.parametrize(
    "raw_status,expected_bucket",
    [
        ("Active", InjuryStatus.HEALTHY),
        ("active", InjuryStatus.HEALTHY),
        ("Active - Practice Participant", InjuryStatus.HEALTHY),
        ("Day-To-Day", InjuryStatus.DTD),
        ("day-to-day", InjuryStatus.DTD),
        ("Probable", InjuryStatus.DTD),
        ("probable", InjuryStatus.DTD),
        ("Questionable", InjuryStatus.DTD),
        ("questionable", InjuryStatus.DTD),
        ("Out", InjuryStatus.OUT),
        ("out", InjuryStatus.OUT),
        ("Injured Reserve", InjuryStatus.OUT),
        ("IR", InjuryStatus.OUT),
        ("ir", InjuryStatus.OUT),
        ("LTIR", InjuryStatus.OUT),
        ("ltir", InjuryStatus.OUT),
        ("Long-Term Injured Reserve", InjuryStatus.OUT),
        ("Suspended", InjuryStatus.OUT),
        ("suspended", InjuryStatus.OUT),
        ("Non-Roster", InjuryStatus.OUT),
    ],
)
def test_status_normalization(raw_status: str, expected_bucket: InjuryStatus):
    response = {
        "injuries": [
            {
                "team": {"abbreviation": "MTL"},
                "injuries": [
                    {
                        "athlete": {"id": "1", "displayName": "Test Player"},
                        "status": raw_status,
                        "details": {},
                        "date": "2024-01-01T00:00:00Z",
                    }
                ],
            }
        ]
    }
    records = InjuryParser.parse(response, _FETCHED_AT)
    assert len(records) == 1
    assert records[0].status == expected_bucket


# ===========================================================================
# Unknown status → conservative Out + warning
# ===========================================================================


class TestUnknownStatus:
    def test_unknown_status_emits_warning(self):
        response = {
            "injuries": [
                {
                    "team": {"abbreviation": "MTL"},
                    "injuries": [
                        {
                            "athlete": {"id": "1", "displayName": "Test Player"},
                            "status": "SomeFutureESPNStatus",
                            "details": {},
                            "date": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = InjuryParser.parse(response, _FETCHED_AT)

        assert len(records) == 1
        assert records[0].status == InjuryStatus.OUT
        assert any("SomeFutureESPNStatus" in str(w.message) for w in caught)

    def test_unknown_status_record_is_kept(self):
        response = {
            "injuries": [
                {
                    "team": {"abbreviation": "EDM"},
                    "injuries": [
                        {
                            "athlete": {"id": "99", "displayName": "Unknown Status Player"},
                            "status": "MYSTERY",
                            "details": {},
                            "date": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", DataMissingWarning)
            records = InjuryParser.parse(response, _FETCHED_AT)

        assert len(records) == 1
        assert records[0].player_name == "Unknown Status Player"


# ===========================================================================
# Missing critical fields → skip + warning
# ===========================================================================


class TestMissingCriticalFields:
    def _make_response(self, athlete: dict, status: str = "Out") -> dict:
        return {
            "injuries": [
                {
                    "team": {"abbreviation": "VAN"},
                    "injuries": [
                        {
                            "athlete": athlete,
                            "status": status,
                            "details": {},
                            "date": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        }

    def test_missing_athlete_id_skips_record(self):
        response = self._make_response({"displayName": "Some Player"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = InjuryParser.parse(response, _FETCHED_AT)
        assert records == []
        assert any("athlete.id" in str(w.message) for w in caught)

    def test_missing_display_name_skips_record(self):
        response = self._make_response({"id": "42"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = InjuryParser.parse(response, _FETCHED_AT)
        assert records == []
        assert any("displayName" in str(w.message) for w in caught)

    def test_missing_status_skips_record(self):
        response = {
            "injuries": [
                {
                    "team": {"abbreviation": "VAN"},
                    "injuries": [
                        {
                            "athlete": {"id": "1", "displayName": "Player"},
                            # no "status" key
                            "details": {},
                            "date": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = InjuryParser.parse(response, _FETCHED_AT)
        assert records == []
        assert any("status" in str(w.message) for w in caught)

    def test_one_bad_record_does_not_prevent_good_records(self):
        response = {
            "injuries": [
                {
                    "team": {"abbreviation": "CGY"},
                    "injuries": [
                        # Bad record — missing id
                        {
                            "athlete": {"displayName": "Bad Player"},
                            "status": "Out",
                            "details": {},
                            "date": "2024-01-01T00:00:00Z",
                        },
                        # Good record
                        {
                            "athlete": {"id": "77", "displayName": "Good Player"},
                            "status": "Day-To-Day",
                            "details": {},
                            "date": "2024-01-01T00:00:00Z",
                        },
                    ],
                }
            ]
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", DataMissingWarning)
            records = InjuryParser.parse(response, _FETCHED_AT)
        assert len(records) == 1
        assert records[0].player_name == "Good Player"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestInjuryParserEdgeCases:
    def test_empty_injuries_list_returns_empty(self):
        records = InjuryParser.parse({"injuries": []}, _FETCHED_AT)
        assert records == []

    def test_missing_injuries_key_returns_empty(self):
        records = InjuryParser.parse({}, _FETCHED_AT)
        assert records == []

    def test_team_with_empty_injuries_array_skipped_cleanly(self):
        response = {
            "injuries": [
                {
                    "team": {"abbreviation": "NSH"},
                    "injuries": [],
                }
            ]
        }
        records = InjuryParser.parse(response, _FETCHED_AT)
        assert records == []

    def test_team_missing_abbreviation_warns_and_skips_group(self):
        response = {
            "injuries": [
                {
                    "team": {"id": "99"},  # no abbreviation
                    "injuries": [
                        {
                            "athlete": {"id": "1", "displayName": "Player"},
                            "status": "Out",
                            "details": {},
                            "date": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = InjuryParser.parse(response, _FETCHED_AT)
        assert records == []
        assert any("abbreviation" in str(w.message) for w in caught)

    def test_none_details_does_not_crash(self):
        response = {
            "injuries": [
                {
                    "team": {"abbreviation": "STL"},
                    "injuries": [
                        {
                            "athlete": {"id": "5", "displayName": "Blues Player"},
                            "status": "Out",
                            "details": None,
                            "date": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        }
        records = InjuryParser.parse(response, _FETCHED_AT)
        assert len(records) == 1
        assert records[0].injury_type is None
        assert records[0].injury_detail is None
        assert records[0].return_date_estimate is None

    def test_missing_details_key_gives_none_fields(self):
        response = {
            "injuries": [
                {
                    "team": {"abbreviation": "CBJ"},
                    "injuries": [
                        {
                            "athlete": {"id": "6", "displayName": "CBJ Player"},
                            "status": "Day-To-Day",
                            # no "details" key at all
                            "date": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        }
        records = InjuryParser.parse(response, _FETCHED_AT)
        assert len(records) == 1
        assert records[0].injury_type is None

    def test_return_date_none_value_stored_as_none(self):
        """Explicit None in returnDate is stored as None, not the string "None"."""
        response = {
            "injuries": [
                {
                    "team": {"abbreviation": "MIN"},
                    "injuries": [
                        {
                            "athlete": {"id": "7", "displayName": "Wild Player"},
                            "status": "Out",
                            "details": {"type": "Knee", "detail": "ACL", "returnDate": None},
                            "date": "2024-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        }
        records = InjuryParser.parse(response, _FETCHED_AT)
        assert records[0].return_date_estimate is None


# ===========================================================================
# to_polars
# ===========================================================================


class TestInjuryParserToPolars:
    def test_empty_list_returns_empty_dataframe(self):
        df = InjuryParser.to_polars([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_records_give_correct_row_count(self):
        records = InjuryParser.parse(_FROZEN_RESPONSE, _FETCHED_AT)
        df = InjuryParser.to_polars(records)
        assert len(df) == 3

    def test_all_schema_columns_present(self):
        records = InjuryParser.parse(_FROZEN_RESPONSE, _FETCHED_AT)
        df = InjuryParser.to_polars(records)
        for col in (
            "player_id_espn",
            "player_name",
            "team_code",
            "status",
            "status_raw",
            "injury_type",
            "injury_detail",
            "return_date_estimate",
            "reported_at",
            "fetched_at",
        ):
            assert col in df.columns, f"Missing column: {col}"

    def test_status_stored_as_string_value(self):
        records = InjuryParser.parse(_FROZEN_RESPONSE, _FETCHED_AT)
        df = InjuryParser.to_polars(records)
        statuses = df["status"].to_list()
        # Should be string values, not enum repr
        assert all(isinstance(s, str) for s in statuses)
        assert "DTD" in statuses
        assert "Out" in statuses

    def test_team_codes_correct(self):
        records = InjuryParser.parse(_FROZEN_RESPONSE, _FETCHED_AT)
        df = InjuryParser.to_polars(records)
        teams = set(df["team_code"].to_list())
        assert "BOS" in teams
        assert "TOR" in teams
