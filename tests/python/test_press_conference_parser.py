"""Tests for PressConferenceParser — LLM-based injury severity extraction."""

from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from data.pbp_parser import DataMissingWarning
from data.press_conference_parser import (
    MENTION_SCHEMA,
    InjuryMention,
    InjurySeverity,
    PressConferenceParser,
    PressConferenceReport,
    _extract_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(response_json: dict | None = None, raise_exc: Exception | None = None) -> MagicMock:
    """Return a mock Anthropic client."""
    client = MagicMock()
    if raise_exc is not None:
        client.messages.create.side_effect = raise_exc
    else:
        payload = response_json or {"mentions": []}
        msg = MagicMock()
        msg.content = [MagicMock(text=json.dumps(payload))]
        msg.usage = MagicMock(input_tokens=100, output_tokens=50)
        client.messages.create.return_value = msg
    return client


def _single_mention_payload(
    player_name: str = "Connor McDavid",
    severity: str = "day_to_day",
    confidence: float = 0.85,
    speaker: str | None = "coach",
    raw_quote: str | None = "He's a game-time decision.",
    injury_location: str | None = "lower body",
    notes: str | None = None,
) -> dict:
    return {
        "mentions": [
            {
                "player_name": player_name,
                "severity": severity,
                "confidence": confidence,
                "speaker": speaker,
                "raw_quote": raw_quote,
                "injury_location": injury_location,
                "notes": notes,
            }
        ]
    }


# ---------------------------------------------------------------------------
# 1 — empty text
# ---------------------------------------------------------------------------


def test_parse_empty_text() -> None:
    client = _make_client()
    parser = PressConferenceParser(client)
    report = parser.parse("   ")
    assert isinstance(report, PressConferenceReport)
    assert report.mentions == []
    client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# 2 — non-string raises
# ---------------------------------------------------------------------------


def test_parse_non_string_raises() -> None:
    parser = PressConferenceParser(_make_client())
    with pytest.raises(ValueError, match="raw_text must be str"):
        parser.parse(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3 — full signal
# ---------------------------------------------------------------------------


def test_parse_full_signal() -> None:
    client = _make_client(_single_mention_payload())
    parser = PressConferenceParser(client)
    report = parser.parse("He's a game-time decision.", team_abbrev="EDM", date="2026-03-21")

    assert len(report.mentions) == 1
    m = report.mentions[0]
    assert m.player_name == "Connor McDavid"
    assert m.severity == InjurySeverity.DAY_TO_DAY
    assert m.confidence == 0.85
    assert m.speaker == "coach"
    assert report.input_tokens == 100
    assert report.output_tokens == 50


# ---------------------------------------------------------------------------
# 4 — day_to_day
# ---------------------------------------------------------------------------


def test_parse_day_to_day() -> None:
    client = _make_client(_single_mention_payload(severity="day_to_day"))
    report = PressConferenceParser(client).parse("...", date="2026-03-21")
    assert report.mentions[0].severity == InjurySeverity.DAY_TO_DAY


# ---------------------------------------------------------------------------
# 5 — season_ending
# ---------------------------------------------------------------------------


def test_parse_season_ending() -> None:
    client = _make_client(_single_mention_payload(severity="season_ending"))
    report = PressConferenceParser(client).parse("...", date="2026-03-21")
    assert report.mentions[0].severity == InjurySeverity.SEASON_ENDING


# ---------------------------------------------------------------------------
# 6 — multi-player
# ---------------------------------------------------------------------------


def test_parse_multi_player() -> None:
    payload = {
        "mentions": [
            {"player_name": "Player One", "severity": "no_concern", "confidence": 0.9},
            {"player_name": "Player Two", "severity": "out_next_game", "confidence": 0.95},
            {"player_name": "Player Three", "severity": "week_to_week", "confidence": 0.7},
        ]
    }
    client = _make_client(payload)
    report = PressConferenceParser(client).parse("...", date="2026-03-21")
    assert len(report.mentions) == 3
    df = PressConferenceParser.to_polars(report)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# 7 — confidence clamped
# ---------------------------------------------------------------------------


def test_parse_confidence_clamped() -> None:
    client = _make_client(_single_mention_payload(confidence=1.5))
    report = PressConferenceParser(client).parse("...", date="2026-03-21")
    assert report.mentions[0].confidence == 1.0


def test_parse_confidence_clamped_negative() -> None:
    client = _make_client(_single_mention_payload(confidence=-0.3))
    report = PressConferenceParser(client).parse("...", date="2026-03-21")
    assert report.mentions[0].confidence == 0.0


# ---------------------------------------------------------------------------
# 8 — missing player_name skipped + warning
# ---------------------------------------------------------------------------


def test_parse_missing_player_name() -> None:
    payload = {"mentions": [{"severity": "day_to_day", "confidence": 0.8}]}
    client = _make_client(payload)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        report = PressConferenceParser(client).parse("...", date="2026-03-21")
    assert report.mentions == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 9 — unknown severity skipped + warning
# ---------------------------------------------------------------------------


def test_parse_unknown_severity() -> None:
    payload = {"mentions": [{"player_name": "Player A", "severity": "nonexistent", "confidence": 0.8}]}
    client = _make_client(payload)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        report = PressConferenceParser(client).parse("...", date="2026-03-21")
    assert report.mentions == []
    assert any("nonexistent" in str(w.message) for w in caught if issubclass(w.category, DataMissingWarning))


# ---------------------------------------------------------------------------
# 10 — API error
# ---------------------------------------------------------------------------


def test_parse_api_error() -> None:
    client = _make_client(raise_exc=RuntimeError("network failure"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        report = PressConferenceParser(client).parse("some text", date="2026-03-21")
    assert report.mentions == []
    assert report.warning_count == 1
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 11 — no JSON in response
# ---------------------------------------------------------------------------


def test_parse_no_json_in_response() -> None:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text="Sure, I found some injuries but can't format them.")]
    msg.usage = MagicMock(input_tokens=50, output_tokens=20)
    client.messages.create.return_value = msg

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        report = PressConferenceParser(client).parse("some text", date="2026-03-21")
    assert report.mentions == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 12 — malformed JSON
# ---------------------------------------------------------------------------


def test_parse_malformed_json() -> None:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text='{"mentions": [{"player_name": "incomplete"')]
    msg.usage = MagicMock(input_tokens=30, output_tokens=10)
    client.messages.create.return_value = msg

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        report = PressConferenceParser(client).parse("some text", date="2026-03-21")
    assert report.mentions == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 13 — missing 'mentions' key
# ---------------------------------------------------------------------------


def test_parse_missing_mentions_key() -> None:
    client = _make_client({"signals": []})  # wrong key
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DataMissingWarning)
        report = PressConferenceParser(client).parse("some text", date="2026-03-21")
    assert report.mentions == []
    assert any(issubclass(w.category, DataMissingWarning) for w in caught)


# ---------------------------------------------------------------------------
# 14 — to_polars: empty report
# ---------------------------------------------------------------------------


def test_to_polars_empty() -> None:
    report = PressConferenceReport(
        date="2026-03-21",
        team_abbrev="EDM",
        raw_text="",
        mentions=[],
        fetched_at="2026-03-21T10:00:00+00:00",
        llm_model="claude-haiku-4-5-20251001",
        warning_count=0,
    )
    df = PressConferenceParser.to_polars(report)
    assert len(df) == 0
    for col in MENTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 15 — to_polars: schema completeness
# ---------------------------------------------------------------------------


def test_to_polars_schema() -> None:
    client = _make_client(
        {
            "mentions": [
                {"player_name": "A", "severity": "day_to_day", "confidence": 0.8},
                {"player_name": "B", "severity": "no_concern", "confidence": 0.9},
            ]
        }
    )
    report = PressConferenceParser(client).parse("text", team_abbrev="TOR", date="2026-03-21")
    df = PressConferenceParser.to_polars(report)
    assert len(df) == 2
    for col in MENTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 16 — context header in prompt
# ---------------------------------------------------------------------------


def test_parse_context_header() -> None:
    client = _make_client({"mentions": []})
    parser = PressConferenceParser(client)
    parser.parse("text", team_abbrev="EDM", date="2026-03-21")

    call_args = client.messages.create.call_args
    user_msg = call_args.kwargs["messages"][0]["content"]
    assert "Team: EDM" in user_msg
    assert "Date: 2026-03-21" in user_msg


# ---------------------------------------------------------------------------
# 17 — injury_location propagated
# ---------------------------------------------------------------------------


def test_parse_injury_location_extracted() -> None:
    client = _make_client(_single_mention_payload(injury_location="lower body"))
    report = PressConferenceParser(client).parse("text", date="2026-03-21")
    df = PressConferenceParser.to_polars(report)
    assert df["injury_location"][0] == "lower body"


# ---------------------------------------------------------------------------
# 18 — speaker propagated
# ---------------------------------------------------------------------------


def test_parse_speaker_propagated() -> None:
    client = _make_client(_single_mention_payload(speaker="coach"))
    report = PressConferenceParser(client).parse("text", date="2026-03-21")
    df = PressConferenceParser.to_polars(report)
    assert df["speaker"][0] == "coach"


# ---------------------------------------------------------------------------
# 19 — _extract_json: markdown block
# ---------------------------------------------------------------------------


def test_extract_json_markdown_block() -> None:
    text = '```json\n{"mentions": []}\n```'
    result = _extract_json(text)
    assert result is not None
    assert json.loads(result) == {"mentions": []}


# ---------------------------------------------------------------------------
# 20 — _extract_json: bare JSON
# ---------------------------------------------------------------------------


def test_extract_json_bare() -> None:
    text = 'Some preamble. {"mentions": [{"player_name": "X"}]} trailing text.'
    result = _extract_json(text)
    assert result is not None
    data = json.loads(result)
    assert "mentions" in data


# ---------------------------------------------------------------------------
# 21 — _extract_json: no JSON
# ---------------------------------------------------------------------------


def test_extract_json_none() -> None:
    assert _extract_json("no braces here") is None
    assert _extract_json("") is None
