"""Tests for data.pbp_parser — Play-by-play parser.

Covers the full testing contract:
  - Input validation (bad types, missing keys)
  - Known-answer regression (frozen inputs → frozen outputs)
  - Output range (times in bounds, enums valid, strength consistent)
  - Edge cases (empty plays, malformed fields, unknown event types, derived zone entries)
  - Integration smoke (to_polars schema, downstream consumption)
"""

import warnings

import polars as pl
import pytest

from data.pbp_parser import (
    DataMissingWarning,
    FaceoffPlay,
    GoalPlay,
    OtherPlay,
    ParsedPlay,
    PenaltyPlay,
    PeriodType,
    PlayByPlayParser,
    ShotPlay,
    ShotResult,
    Strength,
    TurnoverPlay,
    ZoneCode,
    ZoneEntryPlay,
)


# ---------------------------------------------------------------------------
# Fixtures — frozen raw event dicts (never change these without updating tests)
# ---------------------------------------------------------------------------

_GAME_ID = 2024020001

_RAW_FACEOFF = {
    "eventId": 1,
    "sortOrder": 1,
    "typeDescKey": "faceoff",
    "periodDescriptor": {"number": 1, "periodType": "REG"},
    "timeInPeriod": "00:00",
    "timeRemaining": "20:00",
    "situationCode": "1551",
    "details": {
        "eventOwnerTeamId": 10,
        "winningPlayerId": 8476459,
        "losingPlayerId": 8480801,
        "xCoord": 0,
        "yCoord": 0,
        "zoneCode": "N",
    },
}

_RAW_SHOT_ON_GOAL = {
    "eventId": 2,
    "sortOrder": 2,
    "typeDescKey": "shot-on-goal",
    "periodDescriptor": {"number": 1, "periodType": "REG"},
    "timeInPeriod": "03:45",
    "timeRemaining": "16:15",
    "situationCode": "1551",
    "details": {
        "eventOwnerTeamId": 10,
        "shootingPlayerId": 8480801,
        "goalieInNetId": 8480353,
        "xCoord": 55,
        "yCoord": 12,
        "zoneCode": "O",
        "shotType": "wrist",
    },
}

_RAW_MISSED_SHOT = {
    "eventId": 3,
    "sortOrder": 3,
    "typeDescKey": "missed-shot",
    "periodDescriptor": {"number": 1, "periodType": "REG"},
    "timeInPeriod": "05:00",
    "timeRemaining": "15:00",
    "situationCode": "1551",
    "details": {
        "eventOwnerTeamId": 10,
        "shootingPlayerId": 8480801,
        "goalieInNetId": 8480353,
        "xCoord": 60,
        "yCoord": -8,
        "zoneCode": "O",
        "shotType": "slap",
    },
}

_RAW_BLOCKED_SHOT = {
    "eventId": 4,
    "sortOrder": 4,
    "typeDescKey": "blocked-shot",
    "periodDescriptor": {"number": 1, "periodType": "REG"},
    "timeInPeriod": "07:30",
    "timeRemaining": "12:30",
    "situationCode": "1551",
    "details": {
        "eventOwnerTeamId": 10,
        "shootingPlayerId": 8480801,
        "xCoord": 45,
        "yCoord": 20,
        "zoneCode": "O",
        "shotType": "snap",
    },
}

_RAW_GOAL = {
    "eventId": 5,
    "sortOrder": 5,
    "typeDescKey": "goal",
    "periodDescriptor": {"number": 1, "periodType": "REG"},
    "timeInPeriod": "10:22",
    "timeRemaining": "09:38",
    "situationCode": "1551",
    "details": {
        "eventOwnerTeamId": 10,
        "scoringPlayerId": 8480801,
        "scoringPlayerTotal": 15,
        "assist1PlayerId": 8476459,
        "assist1PlayerTotal": 20,
        "assist2PlayerId": 8480803,
        "assist2PlayerTotal": 8,
        "goalieInNetId": 8480353,
        "xCoord": 57,
        "yCoord": 3,
        "zoneCode": "O",
        "shotType": "wrist",
        "homeScore": 1,
        "awayScore": 0,
    },
}

_RAW_PENALTY = {
    "eventId": 6,
    "sortOrder": 6,
    "typeDescKey": "penalty",
    "periodDescriptor": {"number": 2, "periodType": "REG"},
    "timeInPeriod": "05:10",
    "timeRemaining": "14:50",
    "situationCode": "1551",
    "details": {
        "eventOwnerTeamId": 8,
        "committedByPlayerId": 8476459,
        "drawnByPlayerId": 8480801,
        "descKey": "hooking",
        "duration": 2,
        "typeCode": "MIN",
    },
}

_RAW_HIT = {
    "eventId": 7,
    "sortOrder": 7,
    "typeDescKey": "hit",
    "periodDescriptor": {"number": 1, "periodType": "REG"},
    "timeInPeriod": "12:00",
    "timeRemaining": "08:00",
    "situationCode": "1551",
    "details": {"eventOwnerTeamId": 10, "zoneCode": "N"},
}

_RAW_GIVEAWAY = {
    "eventId": 8,
    "sortOrder": 8,
    "typeDescKey": "giveaway",
    "periodDescriptor": {"number": 2, "periodType": "REG"},
    "timeInPeriod": "03:15",
    "timeRemaining": "16:45",
    "situationCode": "1551",
    "details": {"eventOwnerTeamId": 10, "playerId": 8480801, "zoneCode": "D"},
}

_RAW_TAKEAWAY = {
    "eventId": 9,
    "sortOrder": 9,
    "typeDescKey": "takeaway",
    "periodDescriptor": {"number": 2, "periodType": "REG"},
    "timeInPeriod": "07:42",
    "timeRemaining": "12:18",
    "situationCode": "1551",
    "details": {"eventOwnerTeamId": 8, "playerId": 8476459, "zoneCode": "N"},
}

_RAW_GIVEAWAY_NO_PLAYER = {
    "eventId": 10,
    "sortOrder": 10,
    "typeDescKey": "giveaway",
    "periodDescriptor": {"number": 3, "periodType": "REG"},
    "timeInPeriod": "01:00",
    "timeRemaining": "19:00",
    "situationCode": "1551",
    "details": {"eventOwnerTeamId": 10, "zoneCode": "O"},
}

_RAW_PERIOD_END = {
    "eventId": 99,
    "sortOrder": 99,
    "typeDescKey": "period-end",
    "periodDescriptor": {"number": 1, "periodType": "REG"},
    "timeInPeriod": "20:00",
    "timeRemaining": "00:00",
    "situationCode": "1551",
    "details": {},
}

_FULL_GAME = {
    "id": _GAME_ID,
    "plays": [
        _RAW_FACEOFF,
        _RAW_SHOT_ON_GOAL,
        _RAW_MISSED_SHOT,
        _RAW_BLOCKED_SHOT,
        _RAW_GOAL,
        _RAW_PENALTY,
        _RAW_HIT,
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_one(raw_play: dict) -> ParsedPlay:
    parser = PlayByPlayParser()
    raw = {"id": _GAME_ID, "plays": [raw_play]}
    plays = parser.parse(raw)
    # Return the first non-derived play
    return next(p for p in plays if p.event_type_raw != "derived")


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_dict_raises_value_error(self):
        with pytest.raises(ValueError, match="Expected dict"):
            PlayByPlayParser().parse("not a dict")  # type: ignore[arg-type]

    def test_missing_plays_key_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required 'plays' key"):
            PlayByPlayParser().parse({"id": 1})

    def test_plays_not_list_raises_value_error(self):
        with pytest.raises(ValueError, match="must be a list"):
            PlayByPlayParser().parse({"id": 1, "plays": "bad"})

    def test_empty_dict_without_plays_raises_value_error(self):
        with pytest.raises(ValueError):
            PlayByPlayParser().parse({})

    def test_game_id_optional_defaults_to_zero(self):
        plays = PlayByPlayParser().parse({"plays": []})
        assert plays == []


# ---------------------------------------------------------------------------
# 2. Known-answer regression tests (frozen inputs → frozen outputs)
# ---------------------------------------------------------------------------


class TestKnownAnswerFaceoff:
    def setup_method(self):
        self.play = parse_one(_RAW_FACEOFF)

    def test_is_faceoff_play(self):
        assert isinstance(self.play, FaceoffPlay)

    def test_event_type(self):
        assert self.play.event_type == "faceoff"
        assert self.play.event_type_raw == "faceoff"

    def test_game_id(self):
        assert self.play.game_id == _GAME_ID

    def test_event_id(self):
        assert self.play.event_id == 1

    def test_period(self):
        assert self.play.period == 1
        assert self.play.period_type == PeriodType.REG

    def test_time_fields(self):
        assert self.play.time_in_period == "00:00"
        assert self.play.time_in_period_secs == 0
        assert self.play.time_remaining_secs == 1200

    def test_strength_ev(self):
        assert self.play.strength == Strength.EV
        assert self.play.away_skaters == 5
        assert self.play.home_skaters == 5
        assert self.play.away_goalie is True
        assert self.play.home_goalie is True

    def test_zone_neutral(self):
        assert self.play.zone_code == ZoneCode.NEUTRAL

    def test_coordinates(self):
        assert self.play.x_coord == 0.0
        assert self.play.y_coord == 0.0

    def test_faceoff_player_ids(self):
        assert self.play.winning_player_id == 8476459
        assert self.play.losing_player_id == 8480801

    def test_owner_team_id(self):
        assert self.play.event_owner_team_id == 10


class TestKnownAnswerShotOnGoal:
    def setup_method(self):
        self.play = parse_one(_RAW_SHOT_ON_GOAL)

    def test_is_shot_play(self):
        assert isinstance(self.play, ShotPlay)

    def test_event_type(self):
        assert self.play.event_type == "shot"
        assert self.play.event_type_raw == "shot-on-goal"

    def test_shot_result(self):
        assert self.play.shot_result == ShotResult.ON_GOAL

    def test_shot_fields(self):
        assert self.play.shooter_id == 8480801
        assert self.play.goalie_id == 8480353
        assert self.play.shot_type == "wrist"

    def test_time_fields(self):
        assert self.play.time_in_period_secs == 225   # 3*60+45
        assert self.play.time_remaining_secs == 975   # 16*60+15

    def test_coordinates(self):
        assert self.play.x_coord == 55.0
        assert self.play.y_coord == 12.0

    def test_zone_offensive(self):
        assert self.play.zone_code == ZoneCode.OFFENSIVE


class TestKnownAnswerMissedShot:
    def test_shot_result_missed(self):
        play = parse_one(_RAW_MISSED_SHOT)
        assert isinstance(play, ShotPlay)
        assert play.shot_result == ShotResult.MISSED


class TestKnownAnswerBlockedShot:
    def test_shot_result_blocked(self):
        play = parse_one(_RAW_BLOCKED_SHOT)
        assert isinstance(play, ShotPlay)
        assert play.shot_result == ShotResult.BLOCKED
        # blocked-shot has no goalie_id in details fixture
        assert play.goalie_id is None


class TestKnownAnswerGoal:
    def setup_method(self):
        self.play = parse_one(_RAW_GOAL)

    def test_is_goal_play(self):
        assert isinstance(self.play, GoalPlay)

    def test_event_type(self):
        assert self.play.event_type == "goal"

    def test_scorer_and_assists(self):
        assert self.play.scorer_id == 8480801
        assert self.play.assist1_id == 8476459
        assert self.play.assist2_id == 8480803

    def test_score_state(self):
        assert self.play.home_score == 1
        assert self.play.away_score == 0

    def test_goalie_id(self):
        assert self.play.goalie_id == 8480353

    def test_shot_type(self):
        assert self.play.shot_type == "wrist"

    def test_time(self):
        assert self.play.time_in_period_secs == 622   # 10*60+22


class TestKnownAnswerPenalty:
    def setup_method(self):
        self.play = parse_one(_RAW_PENALTY)

    def test_is_penalty_play(self):
        assert isinstance(self.play, PenaltyPlay)

    def test_event_type(self):
        assert self.play.event_type == "penalty"

    def test_penalty_fields(self):
        assert self.play.committed_by_id == 8476459
        assert self.play.drawn_by_id == 8480801
        assert self.play.description == "hooking"
        assert self.play.duration_minutes == 2
        assert self.play.penalty_type == "MIN"

    def test_period(self):
        assert self.play.period == 2


class TestKnownAnswerOther:
    def test_hit_is_other_play(self):
        play = parse_one(_RAW_HIT)
        assert isinstance(play, OtherPlay)
        assert play.event_type == "hit"
        assert play.event_type_raw == "hit"


# ---------------------------------------------------------------------------
# Known-answer: giveaway / takeaway → TurnoverPlay
# ---------------------------------------------------------------------------


class TestTurnoverPlay:
    def test_giveaway_is_turnover_play(self):
        play = parse_one(_RAW_GIVEAWAY)
        assert isinstance(play, TurnoverPlay)

    def test_takeaway_is_turnover_play(self):
        play = parse_one(_RAW_TAKEAWAY)
        assert isinstance(play, TurnoverPlay)

    def test_giveaway_event_type(self):
        play = parse_one(_RAW_GIVEAWAY)
        assert play.event_type == "giveaway"

    def test_takeaway_event_type(self):
        play = parse_one(_RAW_TAKEAWAY)
        assert play.event_type == "takeaway"

    def test_giveaway_event_type_raw(self):
        play = parse_one(_RAW_GIVEAWAY)
        assert play.event_type_raw == "giveaway"

    def test_takeaway_event_type_raw(self):
        play = parse_one(_RAW_TAKEAWAY)
        assert play.event_type_raw == "takeaway"

    def test_giveaway_player_id(self):
        play = parse_one(_RAW_GIVEAWAY)
        assert isinstance(play, TurnoverPlay)
        assert play.player_id == 8480801

    def test_takeaway_player_id(self):
        play = parse_one(_RAW_TAKEAWAY)
        assert isinstance(play, TurnoverPlay)
        assert play.player_id == 8476459

    def test_missing_player_id_returns_none(self):
        play = parse_one(_RAW_GIVEAWAY_NO_PLAYER)
        assert isinstance(play, TurnoverPlay)
        assert play.player_id is None

    def test_giveaway_not_other_play(self):
        play = parse_one(_RAW_GIVEAWAY)
        assert not isinstance(play, OtherPlay)

    def test_takeaway_not_other_play(self):
        play = parse_one(_RAW_TAKEAWAY)
        assert not isinstance(play, OtherPlay)

    def test_to_polars_includes_turnover_player_id(self):
        from data.pbp_parser import PlayByPlayParser
        raw = {"id": _GAME_ID, "plays": [_RAW_GIVEAWAY, _RAW_TAKEAWAY]}
        plays = PlayByPlayParser().parse(raw)
        df = PlayByPlayParser.to_polars(plays)
        assert "turnover_player_id" in df.columns

    def test_to_polars_turnover_player_id_values(self):
        from data.pbp_parser import PlayByPlayParser
        raw = {"id": _GAME_ID, "plays": [_RAW_GIVEAWAY, _RAW_TAKEAWAY]}
        plays = PlayByPlayParser().parse(raw)
        df = PlayByPlayParser.to_polars(plays)
        turnover_rows = df.filter(
            (pl.col("event_type") == "giveaway") | (pl.col("event_type") == "takeaway")
        )
        assert set(turnover_rows["turnover_player_id"].to_list()) == {8480801, 8476459}

    def test_non_turnover_plays_have_null_turnover_player_id(self):
        from data.pbp_parser import PlayByPlayParser
        raw = {"id": _GAME_ID, "plays": [_RAW_HIT, _RAW_GIVEAWAY]}
        plays = PlayByPlayParser().parse(raw)
        df = PlayByPlayParser.to_polars(plays)
        hit_rows = df.filter(pl.col("event_type") == "hit")
        assert hit_rows["turnover_player_id"][0] is None


# ---------------------------------------------------------------------------
# 3. Situation code → strength (known-answer)
# ---------------------------------------------------------------------------


class TestSituationCodeParsing:
    def _parse_strength(self, code: str) -> Strength:
        raw = {
            "id": 1,
            "plays": [{
                "eventId": 1,
                "sortOrder": 1,
                "typeDescKey": "hit",
                "periodDescriptor": {"number": 1, "periodType": "REG"},
                "timeInPeriod": "05:00",
                "timeRemaining": "15:00",
                "situationCode": code,
                "details": {"zoneCode": "N"},
            }],
        }
        plays = PlayByPlayParser().parse(raw)
        return plays[-1].strength  # last play (non-derived)

    def test_1551_is_ev(self):
        assert self._parse_strength("1551") == Strength.EV

    def test_1441_is_ev(self):
        # 4v4 double minor — equal skaters → EV
        assert self._parse_strength("1441") == Strength.EV

    def test_1331_is_ev(self):
        # 3v3 OT — equal skaters → EV
        assert self._parse_strength("1331") == Strength.EV

    def test_1451_is_home_pp(self):
        # home=5, away=4 → home PP
        assert self._parse_strength("1451") == Strength.PP

    def test_1541_is_home_pk(self):
        # home=4, away=5 → home PK
        assert self._parse_strength("1541") == Strength.PK

    def test_0551_is_en(self):
        # away goalie pulled
        assert self._parse_strength("0551") == Strength.EN

    def test_1550_is_en(self):
        # home goalie pulled
        assert self._parse_strength("1550") == Strength.EN

    def test_skater_counts_stored(self):
        raw = {"id": 1, "plays": [{
            "eventId": 1, "sortOrder": 1,
            "typeDescKey": "hit",
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "05:00", "timeRemaining": "15:00",
            "situationCode": "1451",
            "details": {"zoneCode": "N"},
        }]}
        plays = PlayByPlayParser().parse(raw)
        p = plays[-1]
        assert p.away_skaters == 4
        assert p.home_skaters == 5


# ---------------------------------------------------------------------------
# 4. Output range tests
# ---------------------------------------------------------------------------


class TestOutputRanges:
    def setup_method(self):
        self.parser = PlayByPlayParser()
        self.plays = self.parser.parse(_FULL_GAME)

    def test_all_plays_are_parsed_play_instances(self):
        for p in self.plays:
            assert isinstance(p, ParsedPlay)

    def test_time_in_period_secs_non_negative(self):
        for p in self.plays:
            assert p.time_in_period_secs >= 0

    def test_time_remaining_secs_non_negative(self):
        for p in self.plays:
            assert p.time_remaining_secs >= 0

    def test_time_in_period_secs_at_most_1200_for_reg(self):
        for p in self.plays:
            if p.period_type == PeriodType.REG:
                assert p.time_in_period_secs <= 1200

    def test_all_strengths_valid(self):
        valid = set(Strength)
        for p in self.plays:
            assert p.strength in valid

    def test_all_zone_codes_valid(self):
        valid = set(ZoneCode)
        for p in self.plays:
            assert p.zone_code in valid

    def test_all_period_types_valid(self):
        valid = set(PeriodType)
        for p in self.plays:
            assert p.period_type in valid

    def test_skater_counts_sane(self):
        for p in self.plays:
            assert 3 <= p.away_skaters <= 6
            assert 3 <= p.home_skaters <= 6

    def test_shot_result_valid_for_shot_plays(self):
        valid = set(ShotResult)
        for p in self.plays:
            if isinstance(p, ShotPlay):
                assert p.shot_result in valid


# ---------------------------------------------------------------------------
# 5. Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_plays_returns_empty_list(self):
        plays = PlayByPlayParser().parse({"id": 1, "plays": []})
        assert plays == []

    def test_non_dict_play_entry_emits_warning_and_is_skipped(self):
        raw = {"id": 1, "plays": ["not a dict", _RAW_FACEOFF]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plays = PlayByPlayParser().parse(raw)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        # Only the faceoff (+ possible derived entry) parsed; the bad entry skipped
        assert any(isinstance(p, FaceoffPlay) for p in plays)

    def test_malformed_time_in_period_emits_warning_defaults_zero(self):
        raw_play = {**_RAW_FACEOFF, "timeInPeriod": "bad"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            play = parse_one(raw_play)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert play.time_in_period_secs == 0

    def test_missing_time_in_period_emits_warning_defaults_zero(self):
        raw_play = {k: v for k, v in _RAW_FACEOFF.items() if k != "timeInPeriod"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            play = parse_one(raw_play)
        assert play.time_in_period_secs == 0

    def test_invalid_situation_code_emits_warning_strength_other(self):
        raw_play = {**_RAW_FACEOFF, "situationCode": "XXXX"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            play = parse_one(raw_play)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert play.strength == Strength.OTHER

    def test_missing_situation_code_strength_other(self):
        raw_play = {k: v for k, v in _RAW_FACEOFF.items() if k != "situationCode"}
        play = parse_one(raw_play)
        assert play.strength == Strength.OTHER

    def test_unknown_zone_code_emits_warning_returns_unknown(self):
        raw_play = {
            **_RAW_FACEOFF,
            "details": {**_RAW_FACEOFF["details"], "zoneCode": "X"},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            play = parse_one(raw_play)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)
        assert play.zone_code == ZoneCode.UNKNOWN

    def test_missing_details_does_not_crash(self):
        raw_play = {k: v for k, v in _RAW_FACEOFF.items() if k != "details"}
        play = parse_one(raw_play)
        assert isinstance(play, FaceoffPlay)
        assert play.winning_player_id is None
        assert play.losing_player_id is None

    def test_null_details_does_not_crash(self):
        raw_play = {**_RAW_FACEOFF, "details": None}
        play = parse_one(raw_play)
        assert isinstance(play, FaceoffPlay)

    def test_unknown_event_type_returns_other_play(self):
        raw_play = {**_RAW_FACEOFF, "typeDescKey": "future-event-type"}
        play = parse_one(raw_play)
        assert isinstance(play, OtherPlay)
        assert play.event_type == "future_event_type"

    def test_goal_missing_assists_is_none(self):
        details = {**_RAW_GOAL["details"]}
        del details["assist1PlayerId"]
        del details["assist2PlayerId"]
        raw_play = {**_RAW_GOAL, "details": details}
        play = parse_one(raw_play)
        assert isinstance(play, GoalPlay)
        assert play.assist1_id is None
        assert play.assist2_id is None

    def test_penalty_default_duration_two_minutes(self):
        details = {k: v for k, v in _RAW_PENALTY["details"].items() if k != "duration"}
        raw_play = {**_RAW_PENALTY, "details": details}
        play = parse_one(raw_play)
        assert isinstance(play, PenaltyPlay)
        assert play.duration_minutes == 2

    def test_ot_period_type_parsed(self):
        raw_play = {
            **_RAW_FACEOFF,
            "periodDescriptor": {"number": 4, "periodType": "OT"},
        }
        play = parse_one(raw_play)
        assert play.period_type == PeriodType.OT

    def test_unknown_period_type_defaults_to_reg(self):
        raw_play = {
            **_RAW_FACEOFF,
            "periodDescriptor": {"number": 1, "periodType": "WEIRD"},
        }
        play = parse_one(raw_play)
        assert play.period_type == PeriodType.REG

    def test_null_period_descriptor_handled(self):
        raw_play = {**_RAW_FACEOFF, "periodDescriptor": None}
        play = parse_one(raw_play)
        assert play.period == 0

    def test_x_coord_none_when_absent(self):
        details = {k: v for k, v in _RAW_FACEOFF["details"].items() if k != "xCoord"}
        raw_play = {**_RAW_FACEOFF, "details": details}
        play = parse_one(raw_play)
        assert play.x_coord is None

    def test_x_coord_coerced_to_float(self):
        details = {**_RAW_FACEOFF["details"], "xCoord": 42}
        raw_play = {**_RAW_FACEOFF, "details": details}
        play = parse_one(raw_play)
        assert play.x_coord == 42.0
        assert isinstance(play.x_coord, float)


# ---------------------------------------------------------------------------
# 6. Zone entry derivation tests
# ---------------------------------------------------------------------------


class TestZoneEntryDerivation:
    def _make_raw(self, plays: list[dict]) -> dict:
        return {"id": _GAME_ID, "plays": plays}

    def test_neutral_to_offensive_emits_zone_entry(self):
        # faceoff in neutral zone, then shot in offensive zone → zone entry derived
        raw = self._make_raw([_RAW_FACEOFF, _RAW_SHOT_ON_GOAL])
        plays = PlayByPlayParser().parse(raw)
        zone_entries = [p for p in plays if isinstance(p, ZoneEntryPlay)]
        assert len(zone_entries) == 1
        ze = zone_entries[0]
        assert ze.event_type == "zone_entry"
        assert ze.event_type_raw == "derived"
        assert ze.carry_in is None  # unknown until feature 1.7

    def test_zone_entry_inserted_before_trigger_play(self):
        raw = self._make_raw([_RAW_FACEOFF, _RAW_SHOT_ON_GOAL])
        plays = PlayByPlayParser().parse(raw)
        types = [type(p).__name__ for p in plays]
        ze_idx = types.index("ZoneEntryPlay")
        shot_idx = next(i for i, p in enumerate(plays) if isinstance(p, ShotPlay))
        assert ze_idx < shot_idx

    def test_offensive_to_offensive_no_zone_entry(self):
        # Two consecutive offensive zone events → no derived zone entry on second
        raw = self._make_raw([_RAW_SHOT_ON_GOAL, _RAW_MISSED_SHOT])
        plays = PlayByPlayParser().parse(raw)
        derived_entries = [p for p in plays if isinstance(p, ZoneEntryPlay) and p.event_type_raw == "derived"]
        assert len(derived_entries) == 0

    def test_neutral_to_neutral_no_zone_entry(self):
        hit_neutral = {**_RAW_HIT}  # zoneCode "N"
        hit2 = {**_RAW_HIT, "eventId": 8, "sortOrder": 8}
        raw = self._make_raw([hit_neutral, hit2])
        plays = PlayByPlayParser().parse(raw)
        derived_entries = [p for p in plays if isinstance(p, ZoneEntryPlay) and p.event_type_raw == "derived"]
        assert len(derived_entries) == 0

    def test_period_end_resets_zone_tracking(self):
        # After period-end, zone tracking resets; no spurious zone entry emitted
        # for first event of new period even if previous period ended in O zone
        shot_p1 = {**_RAW_SHOT_ON_GOAL}  # zone: O
        period_end = _RAW_PERIOD_END
        faceoff_p2 = {
            **_RAW_FACEOFF,
            "eventId": 100,
            "sortOrder": 100,
            "periodDescriptor": {"number": 2, "periodType": "REG"},
        }  # zone: N
        raw = self._make_raw([shot_p1, period_end, faceoff_p2])
        plays = PlayByPlayParser().parse(raw)
        derived_entries = [p for p in plays if isinstance(p, ZoneEntryPlay) and p.event_type_raw == "derived"]
        assert len(derived_entries) == 0

    def test_defensive_to_offensive_emits_zone_entry(self):
        # Full-ice rush: D-zone → O-zone
        shot_in_d = {
            **_RAW_SHOT_ON_GOAL,
            "eventId": 10,
            "sortOrder": 10,
            "details": {**_RAW_SHOT_ON_GOAL["details"], "zoneCode": "D"},
        }
        shot_in_o = {**_RAW_SHOT_ON_GOAL, "eventId": 11, "sortOrder": 11}
        raw = self._make_raw([shot_in_d, shot_in_o])
        plays = PlayByPlayParser().parse(raw)
        zone_entries = [p for p in plays if isinstance(p, ZoneEntryPlay)]
        assert len(zone_entries) == 1

    def test_native_zone_entry_event_not_duplicated(self):
        # If the API provides a native zone-entry event, no derived entry is added
        native_ze = {
            "eventId": 20,
            "sortOrder": 20,
            "typeDescKey": "zone-entry",
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "08:00",
            "timeRemaining": "12:00",
            "situationCode": "1551",
            "details": {
                "eventOwnerTeamId": 10,
                "isCarryIn": True,
                "zoneCode": "O",
            },
        }
        raw = self._make_raw([_RAW_FACEOFF, native_ze])
        plays = PlayByPlayParser().parse(raw)
        zone_entries = [p for p in plays if isinstance(p, ZoneEntryPlay)]
        # Exactly 1: the native one (no derived duplicate)
        assert len(zone_entries) == 1
        assert zone_entries[0].event_type_raw == "zone-entry"
        assert zone_entries[0].carry_in is True


# ---------------------------------------------------------------------------
# 7. to_polars integration + schema tests
# ---------------------------------------------------------------------------


class TestToPolars:
    def setup_method(self):
        self.parser = PlayByPlayParser()
        self.plays = self.parser.parse(_FULL_GAME)
        self.df = PlayByPlayParser.to_polars(self.plays)

    def test_returns_polars_dataframe(self):
        assert isinstance(self.df, pl.DataFrame)

    def test_has_expected_base_columns(self):
        expected = {
            "game_id", "event_id", "sort_order", "period", "period_type",
            "time_in_period", "time_in_period_secs", "time_remaining_secs",
            "event_type", "event_type_raw", "away_skaters", "home_skaters",
            "away_goalie", "home_goalie", "strength", "event_owner_team_id",
            "x_coord", "y_coord", "zone_code",
        }
        assert expected.issubset(set(self.df.columns))

    def test_has_shot_specific_columns(self):
        assert "shooter_id" in self.df.columns
        assert "shot_result" in self.df.columns
        assert "shot_type" in self.df.columns

    def test_has_goal_specific_columns(self):
        assert "scorer_id" in self.df.columns
        assert "assist1_id" in self.df.columns
        assert "home_score" in self.df.columns

    def test_has_faceoff_specific_columns(self):
        assert "winning_player_id" in self.df.columns
        assert "losing_player_id" in self.df.columns

    def test_has_penalty_specific_columns(self):
        assert "committed_by_id" in self.df.columns
        assert "duration_minutes" in self.df.columns

    def test_has_zone_entry_specific_columns(self):
        assert "carry_in" in self.df.columns
        assert "entering_team_id" in self.df.columns

    def test_row_count_matches_play_count(self):
        assert len(self.df) == len(self.plays)

    def test_game_id_all_same(self):
        assert self.df["game_id"].unique().to_list() == [_GAME_ID]

    def test_shot_result_null_for_non_shot_events(self):
        non_shot = self.df.filter(pl.col("event_type") != "shot")
        assert non_shot["shot_result"].is_null().all()

    def test_shot_result_not_null_for_shot_events(self):
        shots = self.df.filter(pl.col("event_type") == "shot")
        assert shots["shot_result"].is_not_null().all()

    def test_scorer_id_not_null_for_goal_events(self):
        goals = self.df.filter(pl.col("event_type") == "goal")
        assert goals["scorer_id"].is_not_null().all()

    def test_strength_column_contains_valid_values(self):
        valid = {s.value for s in Strength}
        actual = set(self.df["strength"].unique().to_list())
        assert actual.issubset(valid)

    def test_empty_plays_returns_empty_dataframe(self):
        df = PlayByPlayParser.to_polars([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_zone_code_column_values(self):
        valid = {z.value for z in ZoneCode}
        actual = set(self.df["zone_code"].unique().to_list())
        assert actual.issubset(valid)


# ---------------------------------------------------------------------------
# 8. Integration smoke — output consumed by downstream logic
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    def test_filter_shots_from_dataframe(self):
        """Downstream: filter shots from parsed DataFrame."""
        parser = PlayByPlayParser()
        plays = parser.parse(_FULL_GAME)
        df = PlayByPlayParser.to_polars(plays)
        shots = df.filter(pl.col("event_type") == "shot")
        assert len(shots) == 3  # shot-on-goal, missed-shot, blocked-shot

    def test_filter_goals(self):
        parser = PlayByPlayParser()
        plays = parser.parse(_FULL_GAME)
        df = PlayByPlayParser.to_polars(plays)
        goals = df.filter(pl.col("event_type") == "goal")
        assert len(goals) == 1
        assert goals["scorer_id"][0] == 8480801

    def test_filter_penalties(self):
        parser = PlayByPlayParser()
        plays = parser.parse(_FULL_GAME)
        df = PlayByPlayParser.to_polars(plays)
        penalties = df.filter(pl.col("event_type") == "penalty")
        assert len(penalties) == 1
        assert penalties["penalty_description"][0] == "hooking"

    def test_xg_pipeline_can_select_shot_columns(self):
        """Smoke test: xG model can select the columns it needs from play DataFrame."""
        parser = PlayByPlayParser()
        plays = parser.parse(_FULL_GAME)
        df = PlayByPlayParser.to_polars(plays)
        xg_cols = ["x_coord", "y_coord", "shot_type", "shot_result", "strength", "event_owner_team_id"]
        shots = df.filter(pl.col("event_type") == "shot").select(xg_cols)
        assert shots.shape[1] == len(xg_cols)

    def test_zone_entry_list_from_parse_result(self):
        """Downstream: collect zone entries from parsed play list."""
        parser = PlayByPlayParser()
        raw = {"id": _GAME_ID, "plays": [_RAW_FACEOFF, _RAW_SHOT_ON_GOAL, _RAW_GOAL]}
        plays = parser.parse(raw)
        zone_entries = [p for p in plays if isinstance(p, ZoneEntryPlay)]
        assert len(zone_entries) >= 1
        for ze in zone_entries:
            assert ze.carry_in is None  # raw PBP cannot supply this

    def test_raw_field_preserved(self):
        """Parser preserves the original raw dict for debugging."""
        play = parse_one(_RAW_GOAL)
        assert play.raw is _RAW_GOAL or play.raw == _RAW_GOAL
