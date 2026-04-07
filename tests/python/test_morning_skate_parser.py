"""Tests for MorningSkateParsedParser (feature 1.10).

Covers:
- Known-answer regression: frozen JSON LLM response → expected signals
- Signal type parsing and enum mapping
- Confidence clamping (< 0.0 → 0.0, > 1.0 → 1.0)
- Missing player_name → signal skipped + warning
- Unknown signal_type → signal skipped + warning
- Malformed JSON → DataMissingWarning, empty signals
- No JSON in response → DataMissingWarning
- API error → DataMissingWarning, empty signals returned (no raise)
- Empty text → empty signals, no API call made
- to_polars: correct schema, row count, dtypes
- JSON extraction from markdown-wrapped and prose-wrapped responses
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from data.morning_skate_parser import (
    SIGNAL_SCHEMA,
    LineupSignal,
    MorningSkateReport,
    MorningSkateParsedParser,
    SignalType,
    _extract_json,
)
from data.pbp_parser import DataMissingWarning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE = "2026-03-21"
_TEAM = "EDM"
_MODEL = "claude-haiku-4-5-20251001"


def _make_client(response_text: str = "", raise_exc: Exception | None = None) -> MagicMock:
    """Build a mock Anthropic client that returns response_text or raises raise_exc."""
    client = MagicMock()
    if raise_exc is not None:
        client.messages.create.side_effect = raise_exc
    else:
        content_block = MagicMock()
        content_block.text = response_text
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        message = MagicMock()
        message.content = [content_block]
        message.usage = usage
        client.messages.create.return_value = message
    return client


def _ok_json(**overrides) -> str:
    """Build a minimal valid JSON signals response."""
    signal = {
        "player_name": "Connor McDavid",
        "signal_type": "practicing_full",
        "confidence": 0.95,
        "raw_quote": "McDavid was a full participant at morning skate.",
        "notes": None,
    }
    signal.update(overrides)
    return json.dumps({"signals": [signal]})


def _make_parser(response_text: str = "", raise_exc: Exception | None = None) -> tuple[MorningSkateParsedParser, MagicMock]:
    client = _make_client(response_text, raise_exc)
    return MorningSkateParsedParser(client, model=_MODEL), client


# ===========================================================================
# Known-answer regression
# ===========================================================================


class TestKnownAnswer:
    """Frozen LLM response → deterministic signal extraction."""

    _FROZEN_RESPONSE = json.dumps({
        "signals": [
            {
                "player_name": "Connor McDavid",
                "signal_type": "practicing_full",
                "confidence": 0.95,
                "raw_quote": "McDavid was a full participant at morning skate.",
                "notes": None,
            },
            {
                "player_name": "Leon Draisaitl",
                "signal_type": "practicing_limited",
                "confidence": 0.75,
                "raw_quote": "Draisaitl was on a maintenance day.",
                "notes": "Upper-body issue from last game.",
            },
            {
                "player_name": "Evander Kane",
                "signal_type": "not_skating",
                "confidence": 0.90,
                "raw_quote": "Kane did not take part in morning skate.",
                "notes": None,
            },
        ]
    })

    def setup_method(self):
        self.parser, self.client = _make_parser(self._FROZEN_RESPONSE)
        self.report = self.parser.parse(
            "Edmonton morning skate report...",
            team_abbrev=_TEAM,
            date=_DATE,
        )

    def test_three_signals_extracted(self):
        assert len(self.report.signals) == 3

    def test_first_player_name(self):
        assert self.report.signals[0].player_name == "Connor McDavid"

    def test_first_signal_type(self):
        assert self.report.signals[0].signal_type == SignalType.PRACTICING_FULL

    def test_first_confidence(self):
        assert self.report.signals[0].confidence == pytest.approx(0.95)

    def test_first_raw_quote(self):
        assert "full participant" in self.report.signals[0].raw_quote

    def test_second_signal_limited(self):
        assert self.report.signals[1].signal_type == SignalType.PRACTICING_LIMITED

    def test_second_notes_present(self):
        assert self.report.signals[1].notes is not None

    def test_third_signal_not_skating(self):
        assert self.report.signals[2].signal_type == SignalType.NOT_SKATING

    def test_notes_none_preserved(self):
        assert self.report.signals[0].notes is None
        assert self.report.signals[2].notes is None

    def test_date_propagated(self):
        assert self.report.date == _DATE

    def test_team_abbrev_propagated(self):
        assert self.report.team_abbrev == _TEAM

    def test_llm_model_recorded(self):
        assert self.report.llm_model == _MODEL

    def test_input_tokens_recorded(self):
        assert self.report.input_tokens == 100

    def test_output_tokens_recorded(self):
        assert self.report.output_tokens == 50

    def test_warning_count_zero(self):
        assert self.report.warning_count == 0


# ===========================================================================
# Signal type mapping
# ===========================================================================


class TestSignalTypeMapping:
    @pytest.mark.parametrize("type_str,expected", [
        ("practicing_full", SignalType.PRACTICING_FULL),
        ("practicing_limited", SignalType.PRACTICING_LIMITED),
        ("not_skating", SignalType.NOT_SKATING),
        ("game_time_decision", SignalType.GAME_TIME_DECISION),
        ("expected_in", SignalType.EXPECTED_IN),
        ("expected_out", SignalType.EXPECTED_OUT),
        ("line_change", SignalType.LINE_CHANGE),
    ])
    def test_all_signal_types_parsed(self, type_str, expected):
        response = json.dumps({
            "signals": [{
                "player_name": "Test Player",
                "signal_type": type_str,
                "confidence": 0.8,
                "raw_quote": "test",
                "notes": None,
            }]
        })
        parser, _ = _make_parser(response)
        report = parser.parse("text", date=_DATE)
        assert report.signals[0].signal_type == expected

    def test_unknown_signal_type_skipped_with_warning(self):
        response = json.dumps({
            "signals": [{
                "player_name": "Test Player",
                "signal_type": "invisible",
                "confidence": 0.8,
                "raw_quote": "test",
            }]
        })
        parser, _ = _make_parser(response)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert any("invisible" in str(w.message) for w in caught)


# ===========================================================================
# Confidence parsing
# ===========================================================================


class TestConfidenceParsing:
    def test_confidence_clamped_above_1(self):
        response = json.dumps({
            "signals": [{"player_name": "Test Player", "signal_type": "expected_in",
                          "confidence": 1.5, "raw_quote": "test"}]
        })
        parser, _ = _make_parser(response)
        report = parser.parse("text", date=_DATE)
        assert report.signals[0].confidence == pytest.approx(1.0)

    def test_confidence_clamped_below_0(self):
        response = json.dumps({
            "signals": [{"player_name": "Test Player", "signal_type": "expected_in",
                          "confidence": -0.3, "raw_quote": "test"}]
        })
        parser, _ = _make_parser(response)
        report = parser.parse("text", date=_DATE)
        assert report.signals[0].confidence == pytest.approx(0.0)

    def test_invalid_confidence_defaults_to_0_5_with_warning(self):
        response = json.dumps({
            "signals": [{"player_name": "Test Player", "signal_type": "expected_in",
                          "confidence": "high", "raw_quote": "test"}]
        })
        parser, _ = _make_parser(response)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert report.signals[0].confidence == pytest.approx(0.5)
        assert any("confidence" in str(w.message).lower() for w in caught)

    def test_confidence_at_zero_valid(self):
        response = json.dumps({
            "signals": [{"player_name": "Test Player", "signal_type": "expected_in",
                          "confidence": 0.0, "raw_quote": "test"}]
        })
        parser, _ = _make_parser(response)
        report = parser.parse("text", date=_DATE)
        assert report.signals[0].confidence == pytest.approx(0.0)


# ===========================================================================
# Missing / invalid fields
# ===========================================================================


class TestMissingFields:
    def test_missing_player_name_skips_with_warning(self):
        response = json.dumps({
            "signals": [{"signal_type": "expected_in", "confidence": 0.8, "raw_quote": "test"}]
        })
        parser, _ = _make_parser(response)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert any("player_name" in str(w.message) for w in caught)

    def test_empty_player_name_skips_with_warning(self):
        response = json.dumps({
            "signals": [{"player_name": "", "signal_type": "expected_in",
                          "confidence": 0.8, "raw_quote": "test"}]
        })
        parser, _ = _make_parser(response)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0

    def test_non_dict_signal_entry_skipped_with_warning(self):
        response = json.dumps({"signals": ["not_a_dict"]})
        parser, _ = _make_parser(response)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert len(caught) > 0

    def test_one_bad_one_good_bad_skipped(self):
        response = json.dumps({
            "signals": [
                {"player_name": "", "signal_type": "expected_in", "confidence": 0.8},
                {"player_name": "Good Player", "signal_type": "expected_in",
                 "confidence": 0.9, "raw_quote": "test"},
            ]
        })
        parser, _ = _make_parser(response)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 1
        assert report.signals[0].player_name == "Good Player"


# ===========================================================================
# JSON parsing edge cases
# ===========================================================================


class TestJsonParsing:
    def test_no_json_emits_warning_returns_empty(self):
        parser, _ = _make_parser("Sorry, I cannot help with that.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert len(caught) > 0

    def test_malformed_json_emits_warning(self):
        parser, _ = _make_parser("{signals: [bad json}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert len(caught) > 0

    def test_missing_signals_key_emits_warning(self):
        parser, _ = _make_parser('{"data": []}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert any("signals" in str(w.message) for w in caught)

    def test_signals_not_list_emits_warning(self):
        parser, _ = _make_parser('{"signals": "not a list"}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert len(caught) > 0

    def test_json_wrapped_in_markdown_extracted(self):
        response = "Sure!\n```json\n" + _ok_json() + "\n```\nHope that helps!"
        parser, _ = _make_parser(response)
        report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 1

    def test_json_wrapped_in_prose_extracted(self):
        response = "Here is the output: " + _ok_json() + " Thank you."
        parser, _ = _make_parser(response)
        report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 1

    def test_empty_signals_list_valid(self):
        parser, _ = _make_parser('{"signals": []}')
        report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert report.warning_count == 0


# ===========================================================================
# API error handling
# ===========================================================================


class TestApiErrors:
    def test_api_exception_emits_warning_no_raise(self):
        parser, _ = _make_parser(raise_exc=Exception("HTTP 503"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            report = parser.parse("text", date=_DATE)
        assert len(report.signals) == 0
        assert report.warning_count == 1
        assert any("503" in str(w.message) for w in caught)

    def test_api_error_sets_warning_count_1(self):
        parser, _ = _make_parser(raise_exc=RuntimeError("timeout"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            report = parser.parse("text", date=_DATE)
        assert report.warning_count == 1

    def test_api_error_returns_empty_signals(self):
        parser, _ = _make_parser(raise_exc=Exception("error"))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            report = parser.parse("text", date=_DATE)
        assert report.signals == []


# ===========================================================================
# Empty text
# ===========================================================================


class TestEmptyText:
    def test_empty_string_returns_empty_report(self):
        parser, client = _make_parser()
        report = parser.parse("", date=_DATE)
        assert len(report.signals) == 0

    def test_whitespace_only_no_api_call(self):
        parser, client = _make_parser()
        parser.parse("   \n\t  ", date=_DATE)
        client.messages.create.assert_not_called()

    def test_empty_string_no_api_call(self):
        parser, client = _make_parser()
        parser.parse("", date=_DATE)
        client.messages.create.assert_not_called()


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_non_string_raw_text_raises(self):
        parser, _ = _make_parser()
        with pytest.raises(ValueError, match="raw_text must be str"):
            parser.parse(123, date=_DATE)  # type: ignore[arg-type]

    def test_none_raw_text_raises(self):
        parser, _ = _make_parser()
        with pytest.raises(ValueError, match="raw_text must be str"):
            parser.parse(None, date=_DATE)  # type: ignore[arg-type]


# ===========================================================================
# to_polars
# ===========================================================================


class TestToPolars:
    def test_empty_signals_returns_empty_schema_df(self):
        report = MorningSkateReport(
            date=_DATE, team_abbrev=_TEAM, raw_text="",
            signals=[], fetched_at="2026-03-21T10:00:00+00:00",
            llm_model=_MODEL, warning_count=0,
        )
        df = MorningSkateParsedParser.to_polars(report)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        for col in SIGNAL_SCHEMA:
            assert col in df.columns

    def test_two_signals_two_rows(self):
        report = MorningSkateReport(
            date=_DATE, team_abbrev=_TEAM, raw_text="...",
            signals=[
                LineupSignal("Player One", SignalType.PRACTICING_FULL, 0.9, "quote1"),
                LineupSignal("Player Two", SignalType.NOT_SKATING, 0.85, "quote2"),
            ],
            fetched_at="2026-03-21T10:00:00+00:00",
            llm_model=_MODEL, warning_count=0,
        )
        df = MorningSkateParsedParser.to_polars(report)
        assert len(df) == 2

    def test_all_schema_columns_present(self):
        report = MorningSkateReport(
            date=_DATE, team_abbrev=_TEAM, raw_text="...",
            signals=[LineupSignal("Player", SignalType.EXPECTED_IN, 0.8, "text")],
            fetched_at="2026-03-21T10:00:00+00:00",
            llm_model=_MODEL, warning_count=0,
        )
        df = MorningSkateParsedParser.to_polars(report)
        for col in SIGNAL_SCHEMA:
            assert col in df.columns, f"Missing column: {col}"

    def test_signal_type_is_string_value(self):
        report = MorningSkateReport(
            date=_DATE, team_abbrev=None, raw_text="...",
            signals=[LineupSignal("Player", SignalType.PRACTICING_LIMITED, 0.7, "t")],
            fetched_at="2026-03-21T10:00:00+00:00",
            llm_model=_MODEL, warning_count=0,
        )
        df = MorningSkateParsedParser.to_polars(report)
        assert df["signal_type"][0] == "practicing_limited"

    def test_confidence_is_float(self):
        report = MorningSkateReport(
            date=_DATE, team_abbrev=None, raw_text="...",
            signals=[LineupSignal("Player", SignalType.EXPECTED_IN, 0.88, "t")],
            fetched_at="2026-03-21T10:00:00+00:00",
            llm_model=_MODEL, warning_count=0,
        )
        df = MorningSkateParsedParser.to_polars(report)
        assert df["confidence"].dtype in (pl.Float64, pl.Float32)

    def test_date_propagated_to_rows(self):
        report = MorningSkateReport(
            date="2026-01-15", team_abbrev="TOR", raw_text="...",
            signals=[LineupSignal("Player", SignalType.EXPECTED_OUT, 0.9, "t")],
            fetched_at="2026-01-15T10:00:00+00:00",
            llm_model=_MODEL, warning_count=0,
        )
        df = MorningSkateParsedParser.to_polars(report)
        assert df["date"][0] == "2026-01-15"


# ===========================================================================
# _extract_json helper
# ===========================================================================


class TestExtractJson:
    def test_bare_json_object(self):
        text = '{"signals": []}'
        result = _extract_json(text)
        assert result == '{"signals": []}'

    def test_json_in_markdown_block(self):
        text = "```json\n{\"signals\": []}\n```"
        result = _extract_json(text)
        assert result == '{"signals": []}'

    def test_json_in_prose(self):
        text = 'Here is the data: {"signals": []} End.'
        result = _extract_json(text)
        assert result == '{"signals": []}'

    def test_no_json_returns_none(self):
        assert _extract_json("no json here at all") is None

    def test_nested_json_extracted_correctly(self):
        text = '{"signals": [{"player_name": "Test", "signal_type": "expected_in"}]}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["signals"][0]["player_name"] == "Test"

    def test_empty_string_returns_none(self):
        assert _extract_json("") is None
