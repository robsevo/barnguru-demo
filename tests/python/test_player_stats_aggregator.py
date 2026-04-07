"""Tests for PlayerStatsAggregator (feature 1.5 derived stats).

Covers:
- GV/TK counts: known-answer from frozen TurnoverPlay rows
- Faceoff stats: win/loss/zone breakdown + derived percentages
- Zone starts: shift 3s after OZ faceoff → OZ zone start
- Zone start out-of-window: shift 10s after faceoff → no zone start
- PPTOI/SHTOI: shift during PP → pptoi_secs > 0
- Regulation wins: game ending 3-1 in period 3 → one team gets regulation_win
- OT game: period 4 goal → ot_games flag
- vsGoalie: 3 shots on goalie 8480353, 1 goal → shots=3, goals=1
- vsTeam: shots against opponent derived from home/away team IDs
- Empty PBP → all-zero schema output (no crash)
"""

from __future__ import annotations

import polars as pl
import pytest

from data.player_stats_aggregator import (
    PLAYER_STATS_SCHEMA,
    TEAM_STATS_SCHEMA,
    VS_GOALIE_SCHEMA,
    VS_TEAM_SCHEMA,
    PlayerStatsAggregator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agg() -> PlayerStatsAggregator:
    return PlayerStatsAggregator()


def _empty_pbp() -> pl.DataFrame:
    return pl.DataFrame()


def _empty_shifts() -> pl.DataFrame:
    return pl.DataFrame()


def _base_play(
    game_id: int = 1,
    event_id: int = 1,
    sort_order: int = 1,
    period: int = 1,
    time_in_period_secs: int = 0,
    event_type: str = "other",
    strength: str = "ev",
    home_team_id: int | None = 10,
    away_team_id: int | None = 20,
    event_owner_team_id: int | None = None,
    zone_code: str = "N",
    **kwargs,
) -> dict:
    row = {
        "game_id": game_id,
        "event_id": event_id,
        "sort_order": sort_order,
        "period": period,
        "period_type": "REG",
        "time_in_period": "00:00",
        "time_in_period_secs": time_in_period_secs,
        "time_remaining_secs": 1200 - time_in_period_secs,
        "event_type": event_type,
        "event_type_raw": event_type,
        "away_skaters": 5,
        "home_skaters": 5,
        "away_goalie": True,
        "home_goalie": True,
        "strength": strength,
        "event_owner_team_id": event_owner_team_id,
        "x_coord": None,
        "y_coord": None,
        "zone_code": zone_code,
        "shooter_id": None,
        "shot_goalie_id": None,
        "shot_type": None,
        "shot_result": None,
        "scorer_id": None,
        "assist1_id": None,
        "assist2_id": None,
        "home_score": 0,
        "away_score": 0,
        "winning_player_id": None,
        "losing_player_id": None,
        "committed_by_id": None,
        "drawn_by_id": None,
        "penalty_description": None,
        "duration_minutes": None,
        "penalty_type": None,
        "carry_in": None,
        "entering_team_id": None,
        "turnover_player_id": None,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
    }
    row.update(kwargs)
    return row


def _make_pbp(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _make_shifts(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _base_shift(
    game_id: int = 1,
    player_id: int = 100,
    team_id: int = 10,
    period: int = 1,
    shift_number: int = 1,
    start_time_secs: int = 0,
    end_time_secs: int = 90,
) -> dict:
    gsec_start = (period - 1) * 1200 + start_time_secs
    gsec_end = (period - 1) * 1200 + end_time_secs
    return {
        "game_id": game_id,
        "player_id": player_id,
        "team_id": team_id,
        "period": period,
        "shift_number": shift_number,
        "start_time": "00:00",
        "end_time": "01:30",
        "duration_secs": end_time_secs - start_time_secs,
        "start_time_secs": start_time_secs,
        "end_time_secs": end_time_secs,
        "game_seconds_start": gsec_start,
        "game_seconds_end": gsec_end,
    }


# ===========================================================================
# GV / TK
# ===========================================================================


class TestGvTkCounts:
    def test_giveaway_count(self):
        pbp = _make_pbp([
            _base_play(event_id=1, event_type="giveaway", turnover_player_id=100),
            _base_play(event_id=2, event_type="giveaway", turnover_player_id=100),
            _base_play(event_id=3, event_type="takeaway", turnover_player_id=200),
        ])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row = result.filter(pl.col("player_id") == 100)
        assert row["giveaway_count"][0] == 2
        assert row["takeaway_count"][0] == 0

    def test_takeaway_count(self):
        pbp = _make_pbp([
            _base_play(event_id=1, event_type="takeaway", turnover_player_id=200),
            _base_play(event_id=2, event_type="takeaway", turnover_player_id=200),
            _base_play(event_id=3, event_type="takeaway", turnover_player_id=200),
        ])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row = result.filter(pl.col("player_id") == 200)
        assert row["takeaway_count"][0] == 3

    def test_mixed_gv_tk(self):
        pbp = _make_pbp([
            _base_play(event_id=1, event_type="giveaway", turnover_player_id=100),
            _base_play(event_id=2, event_type="takeaway", turnover_player_id=100),
            _base_play(event_id=3, event_type="takeaway", turnover_player_id=100),
        ])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row = result.filter(pl.col("player_id") == 100)
        assert row["giveaway_count"][0] == 1
        assert row["takeaway_count"][0] == 2

    def test_no_turnover_events_returns_empty(self):
        pbp = _make_pbp([_base_play(event_id=1, event_type="faceoff")])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        # faceoff events only; no turnover rows expected
        assert "player_id" in result.columns


# ===========================================================================
# Faceoff stats
# ===========================================================================


class TestFaceoffStats:
    def _fo_play(
        self,
        event_id: int,
        winner: int,
        loser: int,
        zone: str = "O",
        time_in_period_secs: int = 0,
    ) -> dict:
        return _base_play(
            event_id=event_id,
            event_type="faceoff",
            zone_code=zone,
            time_in_period_secs=time_in_period_secs,
            winning_player_id=winner,
            losing_player_id=loser,
        )

    def test_win_count(self):
        pbp = _make_pbp([
            self._fo_play(1, winner=7, loser=8, zone="O"),
            self._fo_play(2, winner=7, loser=8, zone="D"),
        ])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row = result.filter(pl.col("player_id") == 7)
        assert row["faceoff_wins"][0] == 2
        assert row["faceoff_losses"][0] == 0

    def test_loss_count(self):
        pbp = _make_pbp([self._fo_play(1, winner=7, loser=8, zone="O")])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row = result.filter(pl.col("player_id") == 8)
        assert row["faceoff_losses"][0] == 1

    def test_zone_oz_wins(self):
        pbp = _make_pbp([
            self._fo_play(1, winner=7, loser=8, zone="O"),
            self._fo_play(2, winner=7, loser=8, zone="D"),
        ])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row = result.filter(pl.col("player_id") == 7)
        assert row["faceoff_oz_wins"][0] == 1
        assert row["faceoff_dz_wins"][0] == 1
        assert row["faceoff_nz_wins"][0] == 0

    def test_win_pct(self):
        pbp = _make_pbp([
            self._fo_play(1, winner=7, loser=8, zone="N"),
            self._fo_play(2, winner=7, loser=8, zone="N"),
            self._fo_play(3, winner=8, loser=7, zone="N"),
        ])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row7 = result.filter(pl.col("player_id") == 7)
        assert row7["faceoff_win_pct"][0] == pytest.approx(2 / 3)

    def test_oz_pct_and_dz_pct(self):
        pbp = _make_pbp([
            self._fo_play(1, winner=7, loser=8, zone="O"),
            self._fo_play(2, winner=7, loser=8, zone="D"),
            self._fo_play(3, winner=7, loser=8, zone="N"),
            self._fo_play(4, winner=7, loser=8, zone="N"),
        ])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row = result.filter(pl.col("player_id") == 7)
        assert row["faceoff_oz_pct"][0] == pytest.approx(1 / 4)
        assert row["faceoff_dz_pct"][0] == pytest.approx(1 / 4)
        assert row["faceoff_nz_pct"][0] == pytest.approx(2 / 4)

    def test_zero_faceoffs_win_pct_none(self):
        # A player with only giveaways (no faceoffs) should have None faceoff_win_pct
        pbp = _make_pbp([
            _base_play(event_id=1, event_type="giveaway", turnover_player_id=99),
        ])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        row = result.filter(pl.col("player_id") == 99)
        assert row["faceoff_win_pct"][0] is None


# ===========================================================================
# Zone starts
# ===========================================================================


class TestZoneStarts:
    def _fo_at(self, gsecs: int, zone: str, event_id: int = 1) -> dict:
        period = gsecs // 1200 + 1
        t = gsecs % 1200
        return _base_play(
            event_id=event_id,
            event_type="faceoff",
            period=period,
            time_in_period_secs=t,
            zone_code=zone,
            winning_player_id=7,
            losing_player_id=8,
        )

    def _shift_at(self, gsecs_start: int, player_id: int = 100, shift_number: int = 1) -> dict:
        period = gsecs_start // 1200 + 1
        t = gsecs_start % 1200
        return _base_shift(
            player_id=player_id,
            period=period,
            start_time_secs=t,
            end_time_secs=t + 45,
            shift_number=shift_number,
        )

    def test_oz_zone_start_within_window(self):
        # Faceoff 3s before shift start → OZ zone start
        pbp = _make_pbp([self._fo_at(gsecs=57, zone="O")])
        shifts = _make_shifts([self._shift_at(gsecs_start=60, player_id=100)])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 100)
        assert row["zone_start_oz"][0] == 1
        assert row["zone_start_dz"][0] == 0

    def test_dz_zone_start(self):
        pbp = _make_pbp([self._fo_at(gsecs=58, zone="D")])
        shifts = _make_shifts([self._shift_at(gsecs_start=60, player_id=100)])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 100)
        assert row["zone_start_dz"][0] == 1

    def test_out_of_window_no_zone_start(self):
        # Faceoff 10s before shift start → outside the [-5, +2] window
        pbp = _make_pbp([self._fo_at(gsecs=50, zone="O")])
        shifts = _make_shifts([self._shift_at(gsecs_start=60, player_id=100)])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 100)
        # player_id=100 should have 0 zone starts (or not appear in result)
        if len(row) > 0:
            assert row["zone_start_oz"][0] == 0

    def test_faceoff_after_shift_start_within_window(self):
        # Faceoff 2s after shift start → still within the window
        pbp = _make_pbp([self._fo_at(gsecs=62, zone="N")])
        shifts = _make_shifts([self._shift_at(gsecs_start=60, player_id=100)])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 100)
        assert row["zone_start_nz"][0] == 1

    def test_oz_pct_computed(self):
        pbp = _make_pbp([
            self._fo_at(gsecs=57, zone="O", event_id=1),
            self._fo_at(gsecs=157, zone="D", event_id=2),
        ])
        shifts = _make_shifts([
            self._shift_at(gsecs_start=60, player_id=100, shift_number=1),
            self._shift_at(gsecs_start=160, player_id=100, shift_number=2),
        ])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 100)
        assert row["zone_start_oz_pct"][0] == pytest.approx(0.5)
        assert row["zone_start_dz_pct"][0] == pytest.approx(0.5)


# ===========================================================================
# PPTOI / SHTOI
# ===========================================================================


class TestPptoiShtoi:
    def _pp_pbp(
        self,
        game_id: int = 1,
        home_team_id: int = 10,
        away_team_id: int = 20,
        period: int = 1,
        t: int = 30,
        strength: str = "pp",
    ) -> dict:
        return _base_play(
            game_id=game_id,
            event_type="penalty",
            period=period,
            time_in_period_secs=t,
            strength=strength,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )

    def test_home_player_pp_shift(self):
        # PP event at t=30 in period 1 (gsecs=30)
        # Home player shift from t=0 to t=90 → overlaps → PPTOI = 90s
        pbp = _make_pbp([self._pp_pbp(strength="pp")])
        shifts = _make_shifts([_base_shift(player_id=100, team_id=10, start_time_secs=0, end_time_secs=90)])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 100)
        assert row["pptoi_secs"][0] == 90
        assert row["shtoi_secs"][0] == 0

    def test_away_player_pk_shift(self):
        # Same PP event, away player is on PK
        pbp = _make_pbp([self._pp_pbp(strength="pp")])
        shifts = _make_shifts([_base_shift(player_id=200, team_id=20, start_time_secs=0, end_time_secs=60)])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 200)
        assert row["shtoi_secs"][0] == 60
        assert row["pptoi_secs"][0] == 0

    def test_home_player_pk_shift(self):
        # PK strength (home team on penalty kill)
        pbp = _make_pbp([self._pp_pbp(strength="pk")])
        shifts = _make_shifts([_base_shift(player_id=100, team_id=10, start_time_secs=0, end_time_secs=60)])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 100)
        assert row["shtoi_secs"][0] == 60
        assert row["pptoi_secs"][0] == 0

    def test_ev_shift_no_pptoi(self):
        # EV event → no PPTOI/SHTOI
        pbp = _make_pbp([self._pp_pbp(strength="ev")])
        shifts = _make_shifts([_base_shift(player_id=100, team_id=10)])
        result = _agg().aggregate_player_stats(pbp, shifts, season=2023)
        row = result.filter(pl.col("player_id") == 100)
        if len(row) > 0:
            assert row.get_column("pptoi_secs").fill_null(0)[0] == 0


# ===========================================================================
# Regulation wins / OT games
# ===========================================================================


class TestTeamStats:
    def _goal(
        self,
        event_id: int,
        sort_order: int,
        period: int,
        home_score: int,
        away_score: int,
        home_team_id: int = 10,
        away_team_id: int = 20,
        game_id: int = 1,
    ) -> dict:
        return _base_play(
            game_id=game_id,
            event_id=event_id,
            sort_order=sort_order,
            event_type="goal",
            period=period,
            home_score=home_score,
            away_score=away_score,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )

    def test_home_regulation_win(self):
        # Game ends 3-1 in period 3 — home team wins
        pbp = _make_pbp([
            self._goal(1, 1, period=1, home_score=1, away_score=0),
            self._goal(2, 2, period=2, home_score=2, away_score=0),
            self._goal(3, 3, period=2, home_score=2, away_score=1),
            self._goal(4, 4, period=3, home_score=3, away_score=1),
        ])
        result = _agg().aggregate_team_stats(pbp, season=2023)
        home_row = result.filter(pl.col("team_id") == 10)
        away_row = result.filter(pl.col("team_id") == 20)
        assert home_row["regulation_wins"][0] == 1
        assert home_row["regulation_losses"][0] == 0
        assert away_row["regulation_wins"][0] == 0
        assert away_row["regulation_losses"][0] == 1

    def test_away_regulation_win(self):
        pbp = _make_pbp([
            self._goal(1, 1, period=3, home_score=0, away_score=1),
        ])
        result = _agg().aggregate_team_stats(pbp, season=2023)
        home_row = result.filter(pl.col("team_id") == 10)
        away_row = result.filter(pl.col("team_id") == 20)
        assert away_row["regulation_wins"][0] == 1
        assert home_row["regulation_losses"][0] == 1

    def test_ot_game_flag(self):
        # Period 4 goal → OT game
        pbp = _make_pbp([
            self._goal(1, 1, period=3, home_score=1, away_score=1),
            self._goal(2, 2, period=4, home_score=2, away_score=1),
        ])
        result = _agg().aggregate_team_stats(pbp, season=2023)
        home_row = result.filter(pl.col("team_id") == 10)
        away_row = result.filter(pl.col("team_id") == 20)
        assert home_row["ot_games"][0] == 1
        assert away_row["ot_games"][0] == 1

    def test_ot_not_counted_as_regulation_win(self):
        pbp = _make_pbp([
            self._goal(1, 1, period=4, home_score=2, away_score=1),
        ])
        result = _agg().aggregate_team_stats(pbp, season=2023)
        home_row = result.filter(pl.col("team_id") == 10)
        assert home_row["regulation_wins"][0] == 0
        assert home_row["ot_games"][0] == 1

    def test_empty_pbp_returns_empty_schema(self):
        result = _agg().aggregate_team_stats(_empty_pbp(), season=2023)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        for col in TEAM_STATS_SCHEMA:
            assert col in result.columns

    def test_season_column_set(self):
        pbp = _make_pbp([self._goal(1, 1, period=3, home_score=2, away_score=1)])
        result = _agg().aggregate_team_stats(pbp, season=2023)
        assert all(result["season"] == 2023)


# ===========================================================================
# vsGoalie
# ===========================================================================


class TestVsGoalie:
    GOALIE_ID = 8480353

    def _shot(self, event_id: int, shooter: int, goalie: int = GOALIE_ID) -> dict:
        return _base_play(
            event_id=event_id,
            event_type="shot",
            shooter_id=shooter,
            shot_goalie_id=goalie,
        )

    def _goal(self, event_id: int, scorer: int, goalie: int = GOALIE_ID) -> dict:
        return _base_play(
            event_id=event_id,
            event_type="goal",
            scorer_id=scorer,
            shot_goalie_id=goalie,
        )

    def test_shots_count(self):
        pbp = _make_pbp([
            self._shot(1, shooter=100),
            self._shot(2, shooter=100),
            self._shot(3, shooter=100),
        ])
        result = _agg().aggregate_vs_goalie(pbp, season=2023)
        row = result.filter(
            (pl.col("player_id") == 100) & (pl.col("goalie_id") == self.GOALIE_ID)
        )
        assert row["shots"][0] == 3
        assert row["goals"][0] == 0

    def test_goals_count(self):
        pbp = _make_pbp([
            self._shot(1, shooter=100),
            self._shot(2, shooter=100),
            self._goal(3, scorer=100),
        ])
        result = _agg().aggregate_vs_goalie(pbp, season=2023)
        row = result.filter(
            (pl.col("player_id") == 100) & (pl.col("goalie_id") == self.GOALIE_ID)
        )
        assert row["shots"][0] == 2
        assert row["goals"][0] == 1

    def test_different_goalies_separate_rows(self):
        pbp = _make_pbp([
            self._shot(1, shooter=100, goalie=1000),
            self._shot(2, shooter=100, goalie=2000),
        ])
        result = _agg().aggregate_vs_goalie(pbp, season=2023)
        rows = result.filter(pl.col("player_id") == 100)
        assert len(rows) == 2

    def test_unknown_goalie_excluded(self):
        # shot_goalie_id is None → excluded
        pbp = _make_pbp([
            _base_play(event_id=1, event_type="shot", shooter_id=100, shot_goalie_id=None)
        ])
        result = _agg().aggregate_vs_goalie(pbp, season=2023)
        assert len(result.filter(pl.col("player_id") == 100)) == 0

    def test_schema_columns_present(self):
        result = _agg().aggregate_vs_goalie(_empty_pbp(), season=2023)
        for col in VS_GOALIE_SCHEMA:
            assert col in result.columns

    def test_season_set(self):
        pbp = _make_pbp([self._shot(1, shooter=100)])
        result = _agg().aggregate_vs_goalie(pbp, season=2022)
        assert result["season"][0] == 2022


# ===========================================================================
# vsTeam
# ===========================================================================


class TestVsTeam:
    def test_shots_against_opponent(self):
        # Home team (10) player shoots at away team (20) goalie
        pbp = _make_pbp([
            _base_play(
                event_id=1, event_type="shot",
                shooter_id=100, shot_goalie_id=999,
                event_owner_team_id=10,
                home_team_id=10, away_team_id=20,
            ),
            _base_play(
                event_id=2, event_type="shot",
                shooter_id=100, shot_goalie_id=999,
                event_owner_team_id=10,
                home_team_id=10, away_team_id=20,
            ),
        ])
        result = _agg().aggregate_vs_team(pbp, season=2023)
        row = result.filter(
            (pl.col("player_id") == 100) & (pl.col("opponent_team_id") == 20)
        )
        assert row["shots_for"][0] == 2
        assert row["goals_for"][0] == 0

    def test_goals_against_opponent(self):
        pbp = _make_pbp([
            _base_play(
                event_id=1, event_type="goal",
                scorer_id=100, shot_goalie_id=999,
                event_owner_team_id=10,
                home_score=1, away_score=0,
                home_team_id=10, away_team_id=20,
            ),
        ])
        result = _agg().aggregate_vs_team(pbp, season=2023)
        row = result.filter(
            (pl.col("player_id") == 100) & (pl.col("opponent_team_id") == 20)
        )
        assert row["goals_for"][0] == 1
        assert row["shots_for"][0] == 1  # goal also counts as shot

    def test_away_player_opponent_is_home(self):
        # Away player (team 20) shoots on home team (10)
        pbp = _make_pbp([
            _base_play(
                event_id=1, event_type="shot",
                shooter_id=200, shot_goalie_id=888,
                event_owner_team_id=20,
                home_team_id=10, away_team_id=20,
            ),
        ])
        result = _agg().aggregate_vs_team(pbp, season=2023)
        row = result.filter(
            (pl.col("player_id") == 200) & (pl.col("opponent_team_id") == 10)
        )
        assert len(row) == 1
        assert row["shots_for"][0] == 1

    def test_schema_columns_present(self):
        result = _agg().aggregate_vs_team(_empty_pbp(), season=2023)
        for col in VS_TEAM_SCHEMA:
            assert col in result.columns

    def test_season_set(self):
        pbp = _make_pbp([
            _base_play(
                event_id=1, event_type="shot",
                shooter_id=100,
                event_owner_team_id=10,
                home_team_id=10, away_team_id=20,
            ),
        ])
        result = _agg().aggregate_vs_team(pbp, season=2021)
        assert result["season"][0] == 2021


# ===========================================================================
# Empty input / schema contract
# ===========================================================================


class TestEmptyInput:
    def test_player_stats_empty_pbp_schema(self):
        result = _agg().aggregate_player_stats(_empty_pbp(), _empty_shifts(), season=2023)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        for col in PLAYER_STATS_SCHEMA:
            assert col in result.columns

    def test_team_stats_empty_pbp_schema(self):
        result = _agg().aggregate_team_stats(_empty_pbp(), season=2023)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        for col in TEAM_STATS_SCHEMA:
            assert col in result.columns

    def test_vs_team_empty_schema(self):
        result = _agg().aggregate_vs_team(_empty_pbp(), season=2023)
        assert isinstance(result, pl.DataFrame)
        for col in VS_TEAM_SCHEMA:
            assert col in result.columns

    def test_vs_goalie_empty_schema(self):
        result = _agg().aggregate_vs_goalie(_empty_pbp(), season=2023)
        assert isinstance(result, pl.DataFrame)
        for col in VS_GOALIE_SCHEMA:
            assert col in result.columns

    def test_player_stats_no_crash_only_ev_events(self):
        pbp = _make_pbp([_base_play(event_id=1, event_type="stoppage")])
        result = _agg().aggregate_player_stats(pbp, _empty_shifts(), season=2023)
        assert isinstance(result, pl.DataFrame)
