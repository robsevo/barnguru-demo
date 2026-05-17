"""Tests for the pure helpers inside data/schedule_sync.py.

The live ``fetch_schedule`` function is exercised by the Phase 3 driver
scripts against the real NHL API — we don't mock httpx here.
"""

from __future__ import annotations

from data.schedule_sync import (
    GAME_TYPE_PLAYOFFS,
    GAME_TYPE_PRESEASON,
    GAME_TYPE_REGULAR,
    _extract_game_type,
    _normalize_game,
)


# ---------------------------------------------------------------------------
# _extract_game_type
# ---------------------------------------------------------------------------

class TestExtractGameType:
    def test_uses_raw_gametype_field(self):
        raw = {"gameType": 2}
        assert _extract_game_type(raw, game_id=2024020001) == 2

    def test_falls_back_to_game_id_digits(self):
        # No gameType in raw — parse digits 5-6 of game_id.
        assert _extract_game_type({}, game_id=2024010008) == GAME_TYPE_PRESEASON
        assert _extract_game_type({}, game_id=2024020001) == GAME_TYPE_REGULAR
        assert _extract_game_type({}, game_id=2024030001) == GAME_TYPE_PLAYOFFS

    def test_returns_none_for_short_id(self):
        assert _extract_game_type({}, game_id=12345) is None


# ---------------------------------------------------------------------------
# _normalize_game
# ---------------------------------------------------------------------------

def _raw_game(**overrides) -> dict:
    base = {
        "id": 2024020001,
        "gameDate": "2024-10-08",
        "homeTeam": {"abbrev": "BOS"},
        "awayTeam": {"abbrev": "FLA"},
        "gameType": 2,
    }
    base.update(overrides)
    return base


class TestNormalizeGame:
    def test_happy_path(self):
        norm = _normalize_game(_raw_game())
        assert norm == {
            "game_id":   2024020001,
            "game_date": "2024-10-08",
            "home_team": "BOS",
            "away_team": "FLA",
            "game_type": 2,
        }

    def test_missing_id_returns_none(self):
        assert _normalize_game(_raw_game(id=None)) is None

    def test_missing_home_returns_none(self):
        assert _normalize_game(_raw_game(homeTeam={})) is None

    def test_missing_date_returns_none(self):
        assert _normalize_game(_raw_game(gameDate=None, startTimeUTC="")) is None

    def test_uses_start_time_utc_when_game_date_missing(self):
        norm = _normalize_game(
            _raw_game(gameDate=None, startTimeUTC="2024-10-08T23:00:00Z")
        )
        assert norm is not None
        assert norm["game_date"] == "2024-10-08"

    def test_preseason_game_type_preserved(self):
        norm = _normalize_game(_raw_game(id=2024010008, gameType=1))
        assert norm is not None
        assert norm["game_type"] == GAME_TYPE_PRESEASON

    def test_playoff_game_type_preserved(self):
        norm = _normalize_game(_raw_game(id=2024030001, gameType=3))
        assert norm is not None
        assert norm["game_type"] == GAME_TYPE_PLAYOFFS

    def test_missing_game_type_returns_none(self):
        # Cannot recover game_type → drop the row rather than guess.
        raw = _raw_game(gameType=None, id=99)  # short id can't be parsed
        assert _normalize_game(raw) is None
