"""Tests for TransactionParser — ESPN NHL transactions feed parser."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from data.pbp_parser import DataMissingWarning
from data.transaction_parser import (
    TRANSACTION_SCHEMA,
    EventType,
    TransactionEvent,
    TransactionParser,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FETCHED_AT = "2026-03-21T12:00:00+00:00"


def _entry(
    date: str = "20260321T000000",
    type_text: str = "Trade",
    team: str = "TOR",
    to_team: str | None = "EDM",
    description: str = "Toronto traded Player A to Edmonton.",
    athlete_id: str | None = "3900169",
    athlete_name: str = "Player A",
) -> dict:
    """Build a minimal ESPN transaction entry dict."""
    e: dict = {
        "id": "abc123",
        "date": date,
        "type": {"text": type_text},
        "team": {"abbreviation": team},
        "description": description,
    }
    if to_team is not None:
        e["toTeam"] = {"abbreviation": to_team}
    if athlete_id is not None:
        e["athlete"] = {"id": athlete_id, "displayName": athlete_name}
    return e


def _raw(entries: list[dict] | None = None) -> dict:
    return {"transactions": entries or []}


# ---------------------------------------------------------------------------
# 1 — empty transactions key
# ---------------------------------------------------------------------------


def test_parse_empty_transactions_key() -> None:
    result = TransactionParser.parse({}, _FETCHED_AT)
    assert result == []


def test_parse_empty_transactions_list() -> None:
    result = TransactionParser.parse(_raw([]), _FETCHED_AT)
    assert result == []


# ---------------------------------------------------------------------------
# 2 — trade
# ---------------------------------------------------------------------------


def test_parse_trade() -> None:
    events = TransactionParser.parse(_raw([_entry(type_text="Trade")]), _FETCHED_AT)
    assert len(events) == 1
    e = events[0]
    assert e.event_type == EventType.TRADE.value
    assert e.team == "TOR"
    assert e.secondary_team == "EDM"


# ---------------------------------------------------------------------------
# 3 — signing
# ---------------------------------------------------------------------------


def test_parse_signing() -> None:
    events = TransactionParser.parse(
        _raw([_entry(type_text="Signed", to_team=None, description="Player A signed.")]),
        _FETCHED_AT,
    )
    assert len(events) == 1
    assert events[0].event_type == EventType.SIGNING.value
    assert events[0].secondary_team is None


# ---------------------------------------------------------------------------
# 4 — waiver_claim
# ---------------------------------------------------------------------------


def test_parse_waiver_claim() -> None:
    events = TransactionParser.parse(
        _raw([_entry(type_text="Claimed", to_team=None, description="Player claimed.")]),
        _FETCHED_AT,
    )
    assert events[0].event_type == EventType.WAIVER_CLAIM.value


# ---------------------------------------------------------------------------
# 5 — waiver_release
# ---------------------------------------------------------------------------


def test_parse_waiver_release() -> None:
    events = TransactionParser.parse(
        _raw([_entry(type_text="Placed on waivers", to_team=None, description="Player placed.")]),
        _FETCHED_AT,
    )
    assert events[0].event_type == EventType.WAIVER_RELEASE.value


# ---------------------------------------------------------------------------
# 6 — call_up
# ---------------------------------------------------------------------------


def test_parse_call_up() -> None:
    events = TransactionParser.parse(
        _raw([_entry(type_text="Recalled", to_team=None, description="Player recalled.")]),
        _FETCHED_AT,
    )
    assert events[0].event_type == EventType.CALL_UP.value


# ---------------------------------------------------------------------------
# 7 — send_down
# ---------------------------------------------------------------------------


def test_parse_send_down() -> None:
    events = TransactionParser.parse(
        _raw([_entry(type_text="Assigned", to_team=None, description="Player assigned.")]),
        _FETCHED_AT,
    )
    assert events[0].event_type == EventType.SEND_DOWN.value


# ---------------------------------------------------------------------------
# 8 — front_office
# ---------------------------------------------------------------------------


def test_parse_front_office() -> None:
    events = TransactionParser.parse(
        _raw([_entry(type_text="Hired", to_team=None, athlete_id=None,
                     description="Coach Named as head coach.")]),
        _FETCHED_AT,
    )
    assert events[0].event_type == EventType.FRONT_OFFICE.value


# ---------------------------------------------------------------------------
# 9 — unknown type → OTHER + warning
# ---------------------------------------------------------------------------


def test_parse_unknown_type() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        events = TransactionParser.parse(
            _raw([_entry(type_text="MysteryAction", to_team=None, description="Something happened.")]),
            _FETCHED_AT,
        )
    assert events[0].event_type == EventType.OTHER.value
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 10 — missing description → skipped + warning
# ---------------------------------------------------------------------------


def test_parse_missing_description_skipped() -> None:
    entry = _entry()
    entry["description"] = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        events = TransactionParser.parse(_raw([entry]), _FETCHED_AT)
    assert events == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 11 — missing team.abbreviation → skipped + warning
# ---------------------------------------------------------------------------


def test_parse_missing_team_skipped() -> None:
    entry = _entry()
    entry["team"] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        events = TransactionParser.parse(_raw([entry]), _FETCHED_AT)
    assert events == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 12 — date normalization
# ---------------------------------------------------------------------------


def test_parse_date_normalized() -> None:
    events = TransactionParser.parse(
        _raw([_entry(date="20260321T120000")]), _FETCHED_AT
    )
    assert events[0].date == "2026-03-21"


def test_parse_date_short_format() -> None:
    events = TransactionParser.parse(
        _raw([_entry(date="20260321")]), _FETCHED_AT
    )
    assert events[0].date == "2026-03-21"


# ---------------------------------------------------------------------------
# 13 — multiple events
# ---------------------------------------------------------------------------


def test_parse_multiple_events() -> None:
    entries = [
        _entry(description="Trade 1.", athlete_name="P1"),
        _entry(description="Trade 2.", athlete_name="P2"),
        _entry(description="Trade 3.", athlete_name="P3"),
    ]
    events = TransactionParser.parse(_raw(entries), _FETCHED_AT)
    assert len(events) == 3


# ---------------------------------------------------------------------------
# 14 — athlete id extracted
# ---------------------------------------------------------------------------


def test_parse_athlete_id_extracted() -> None:
    events = TransactionParser.parse(
        _raw([_entry(athlete_id="3900169")]), _FETCHED_AT
    )
    assert events[0].player_id_espn == "3900169"


# ---------------------------------------------------------------------------
# 15 — no athlete field → player_id_espn=None
# ---------------------------------------------------------------------------


def test_parse_no_athlete() -> None:
    events = TransactionParser.parse(
        _raw([_entry(athlete_id=None)]), _FETCHED_AT
    )
    assert events[0].player_id_espn is None


# ---------------------------------------------------------------------------
# 16 — source always "espn"
# ---------------------------------------------------------------------------


def test_parse_source_always_espn() -> None:
    events = TransactionParser.parse(_raw([_entry()]), _FETCHED_AT)
    assert events[0].source == "espn"


# ---------------------------------------------------------------------------
# 17 — to_polars schema
# ---------------------------------------------------------------------------


def test_to_polars_schema() -> None:
    events = TransactionParser.parse(_raw([_entry()]), _FETCHED_AT)
    df = TransactionParser.to_polars(events)
    assert len(df) == 1
    for col in TRANSACTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 18 — to_polars empty
# ---------------------------------------------------------------------------


def test_to_polars_empty() -> None:
    df = TransactionParser.to_polars([])
    assert len(df) == 0
    for col in TRANSACTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# bonus — non-dict raw raises ValueError
# ---------------------------------------------------------------------------


def test_parse_non_dict_raw_raises() -> None:
    with pytest.raises(ValueError, match="dict"):
        TransactionParser.parse("not a dict", _FETCHED_AT)  # type: ignore[arg-type]
