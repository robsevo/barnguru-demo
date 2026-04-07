"""Tests for Feature 16.8 — TrackingAccumulator (data/tracking_accumulator.py).

All tests use an in-memory DuckDB database so no files are written.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import polars as pl
import pytest

from data.tracking_accumulator import (
    TrackingAccumulator,
    ShotEvent,
    _now_ms,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _acc() -> TrackingAccumulator:
    """In-memory accumulator for tests."""
    return TrackingAccumulator(db_path=":memory:")


def _track(
    track_id: int = 1,
    cx: float = 640.0,
    cy: float = 360.0,
    team: str = "EDM",
    class_name: str = "player_home",
    rink_x: float | None = 10.0,
    rink_y: float | None = -5.0,
    jersey_number: int | None = 97,
    jersey_locked: bool = True,
) -> MagicMock:
    """Fake TrackPoint-like object."""
    tp = MagicMock()
    tp.track_id = track_id
    tp.class_id = 0
    tp.class_name = class_name
    tp.cx = cx
    tp.cy = cy
    tp.vx = 1.0
    tp.vy = -0.5
    tp.w = 40.0
    tp.h = 80.0
    tp.rink_x = rink_x
    tp.rink_y = rink_y
    tp.jersey_number = jersey_number
    tp.jersey_locked = jersey_locked
    tp.team = team
    tp.frame_ts = time.time()
    return tp


def _shot_event(
    game_id: int = 2024020001,
    event_id: int = 42,
    is_goal: bool = False,
) -> ShotEvent:
    return ShotEvent(
        game_id=game_id,
        event_id=event_id,
        period=2,
        time_in_period="10:33",
        time_in_period_secs=633,
        shooter_id=8478402,
        shooter_team="EDM",
        shot_result="on_goal",
        is_goal=is_goal,
        shot_x=-15.0,
        shot_y=4.5,
        home_team="EDM",
        away_team="CGY",
    )


# ── _now_ms ───────────────────────────────────────────────────────────────────

def test_now_ms_is_recent():
    ms = _now_ms()
    now = int(time.time() * 1_000)
    assert abs(ms - now) < 500


# ── Schema initialisation ─────────────────────────────────────────────────────

class TestSchema:
    def test_tables_created(self):
        acc = _acc()
        tables = {
            r[0] for r in acc._con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        assert "tracking_frames" in tables
        assert "shot_snapshots" in tables
        assert "snapshot_players" in tables
        acc.close()

    def test_shot_tracking_labels_view_exists(self):
        acc = _acc()
        views = {
            r[0] for r in acc._con.execute(
                "SELECT table_name FROM information_schema.views WHERE table_schema='main'"
            ).fetchall()
        }
        assert "shot_tracking_labels" in views
        acc.close()

    def test_empty_tables_return_empty_polars_df(self):
        with _acc() as acc:
            df = acc.tracking_frames(game_id=1)
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 0

    def test_empty_shot_labels_returns_empty_df(self):
        with _acc() as acc:
            df = acc.shot_labels()
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 0


# ── record_frame ──────────────────────────────────────────────────────────────

class TestRecordFrame:
    def test_single_frame_single_track(self):
        with _acc() as acc:
            acc.record_frame(
                game_id=1, frame_seq=0,
                positions=[_track(track_id=1)],
            )
            assert acc.frame_count(game_id=1) == 1

    def test_multiple_tracks_one_frame(self):
        with _acc() as acc:
            tracks = [_track(track_id=i) for i in range(1, 7)]
            acc.record_frame(game_id=1, frame_seq=0, positions=tracks)
            df = acc.tracking_frames(game_id=1)
            assert len(df) == 6

    def test_multiple_frames_accumulated(self):
        with _acc() as acc:
            for frame_seq in range(5):
                acc.record_frame(
                    game_id=1, frame_seq=frame_seq,
                    positions=[_track(track_id=1)],
                )
            assert acc.frame_count(game_id=1) == 5

    def test_empty_positions_does_nothing(self):
        with _acc() as acc:
            acc.record_frame(game_id=1, frame_seq=0, positions=[])
            assert acc.frame_count(game_id=1) == 0

    def test_duplicate_primary_key_ignored(self):
        """Same (game_id, frame_seq, track_id) twice — INSERT OR IGNORE."""
        with _acc() as acc:
            tp = _track(track_id=1)
            acc.record_frame(game_id=1, frame_seq=0, positions=[tp])
            acc.record_frame(game_id=1, frame_seq=0, positions=[tp])
            df = acc.tracking_frames(game_id=1)
            assert len(df) == 1  # not 2

    def test_team_and_class_stored_correctly(self):
        with _acc() as acc:
            acc.record_frame(
                game_id=7, frame_seq=0,
                positions=[_track(track_id=3, team="CGY", class_name="goalie_away")],
            )
            df = acc.tracking_frames(game_id=7)
            row = df.to_dicts()[0]
            assert row["team"] == "CGY"
            assert row["class_name"] == "goalie_away"

    def test_rink_coords_stored(self):
        with _acc() as acc:
            acc.record_frame(
                game_id=2, frame_seq=0,
                positions=[_track(rink_x=20.5, rink_y=-12.0)],
            )
            df = acc.tracking_frames(game_id=2)
            row = df.to_dicts()[0]
            assert abs(row["rink_x"] - 20.5) < 1e-6
            assert abs(row["rink_y"] - (-12.0)) < 1e-6

    def test_null_rink_coords_stored_as_null(self):
        with _acc() as acc:
            acc.record_frame(
                game_id=3, frame_seq=0,
                positions=[_track(rink_x=None, rink_y=None)],
            )
            df = acc.tracking_frames(game_id=3)
            row = df.to_dicts()[0]
            assert row["rink_x"] is None
            assert row["rink_y"] is None

    def test_jersey_number_stored(self):
        with _acc() as acc:
            acc.record_frame(
                game_id=4, frame_seq=0,
                positions=[_track(jersey_number=97, jersey_locked=True)],
            )
            df = acc.tracking_frames(game_id=4)
            row = df.to_dicts()[0]
            assert row["jersey_number"] == 97
            assert row["jersey_locked"] is True

    def test_different_games_isolated(self):
        with _acc() as acc:
            acc.record_frame(game_id=1, frame_seq=0, positions=[_track()])
            acc.record_frame(game_id=2, frame_seq=0, positions=[_track()])
            assert acc.frame_count(1) == 1
            assert acc.frame_count(2) == 1

    def test_timestamp_ms_stored(self):
        ts_ms = _now_ms()
        with _acc() as acc:
            acc.record_frame(game_id=5, frame_seq=0, positions=[_track()],
                             timestamp_ms=ts_ms)
            df = acc.tracking_frames(game_id=5)
            assert df["timestamp_ms"][0] == ts_ms


# ── notify_shot_event ─────────────────────────────────────────────────────────

class TestNotifyShotEvent:
    def test_returns_positive_snapshot_id(self):
        with _acc() as acc:
            sid = acc.notify_shot_event(_shot_event(), positions=[_track()])
            assert isinstance(sid, int)
            assert sid >= 1

    def test_snapshot_header_stored(self):
        with _acc() as acc:
            event = _shot_event(game_id=99, event_id=7, is_goal=True)
            acc.notify_shot_event(event, positions=[_track()])
            df = acc.shot_snapshot_summary(game_id=99)
            assert len(df) == 1
            row = df.to_dicts()[0]
            assert row["game_id"] == 99
            assert row["event_id"] == 7
            assert row["is_goal"] is True
            assert row["shooter_team"] == "EDM"

    def test_shot_result_and_coords_stored(self):
        with _acc() as acc:
            event = _shot_event()
            event.shot_x = -15.0
            event.shot_y = 4.5
            event.shot_result = "on_goal"
            acc.notify_shot_event(event, positions=[])
            df = acc.shot_snapshot_summary()
            row = df.to_dicts()[0]
            assert row["shot_result"] == "on_goal"
            assert abs(row["shot_x"] - (-15.0)) < 1e-6

    def test_players_stored_in_snapshot_players(self):
        with _acc() as acc:
            tracks = [_track(track_id=i) for i in range(1, 8)]
            sid = acc.notify_shot_event(_shot_event(), positions=tracks)
            df = acc.query(
                "SELECT * FROM snapshot_players WHERE snapshot_id = ?", [sid]
            )
            assert len(df) == 7

    def test_tracks_captured_count(self):
        with _acc() as acc:
            tracks = [_track(track_id=i) for i in range(1, 6)]
            acc.notify_shot_event(_shot_event(), positions=tracks)
            df = acc.shot_snapshot_summary()
            assert df["tracks_captured"][0] == 5

    def test_empty_positions_snapshot_still_written(self):
        with _acc() as acc:
            sid = acc.notify_shot_event(_shot_event(), positions=[])
            assert sid >= 1
            df = acc.shot_snapshot_summary()
            assert len(df) == 1
            assert df["tracks_captured"][0] == 0

    def test_snapshot_ids_autoincrement(self):
        with _acc() as acc:
            s1 = acc.notify_shot_event(_shot_event(event_id=1), positions=[])
            s2 = acc.notify_shot_event(_shot_event(event_id=2), positions=[])
            assert s2 > s1

    def test_shot_count(self):
        with _acc() as acc:
            for i in range(5):
                acc.notify_shot_event(_shot_event(event_id=i), positions=[])
            assert acc.shot_count() == 5

    def test_shot_count_filtered_by_game(self):
        with _acc() as acc:
            acc.notify_shot_event(_shot_event(game_id=1, event_id=1), positions=[])
            acc.notify_shot_event(_shot_event(game_id=1, event_id=2), positions=[])
            acc.notify_shot_event(_shot_event(game_id=2, event_id=3), positions=[])
            assert acc.shot_count(game_id=1) == 2
            assert acc.shot_count(game_id=2) == 1


# ── shot_tracking_labels view ─────────────────────────────────────────────────

class TestShotTrackingLabels:
    def _populate(self, acc: TrackingAccumulator, n_players: int = 4) -> None:
        tracks = [_track(track_id=i) for i in range(1, n_players + 1)]
        acc.notify_shot_event(_shot_event(is_goal=False), positions=tracks)
        acc.notify_shot_event(_shot_event(event_id=99, is_goal=True), positions=tracks)

    def test_view_returns_rows(self):
        with _acc() as acc:
            self._populate(acc, 4)
            df = acc.shot_labels()
            # 2 shots × 4 players = 8 rows
            assert len(df) == 8

    def test_view_contains_is_goal_label(self):
        with _acc() as acc:
            self._populate(acc, 2)
            df = acc.shot_labels()
            # Must have both True and False
            goal_values = set(df["is_goal"].to_list())
            assert True in goal_values
            assert False in goal_values

    def test_goals_only_filter(self):
        with _acc() as acc:
            self._populate(acc, 3)
            df = acc.shot_labels(goals_only=True)
            assert all(df["is_goal"].to_list())

    def test_game_filter(self):
        with _acc() as acc:
            tracks = [_track()]
            acc.notify_shot_event(_shot_event(game_id=1, event_id=1), positions=tracks)
            acc.notify_shot_event(_shot_event(game_id=2, event_id=2), positions=tracks)
            df = acc.shot_labels(game_id=1)
            assert all(r["game_id"] == 1 for r in df.to_dicts())
            assert len(df) == 1

    def test_player_rink_coords_in_view(self):
        with _acc() as acc:
            tp = _track(rink_x=30.0, rink_y=-8.0)
            acc.notify_shot_event(_shot_event(), positions=[tp])
            df = acc.shot_labels()
            row = df.to_dicts()[0]
            assert abs(row["player_x"] - 30.0) < 1e-6
            assert abs(row["player_y"] - (-8.0)) < 1e-6

    def test_view_columns_present(self):
        with _acc() as acc:
            acc.notify_shot_event(_shot_event(), positions=[_track()])
            df = acc.shot_labels()
            required = {
                "snapshot_id", "game_id", "event_id", "is_goal",
                "shooter_id", "shooter_team", "shot_result",
                "shot_x", "shot_y", "home_team", "away_team",
                "track_id", "player_team", "class_name",
                "player_x", "player_y", "vx", "vy",
            }
            assert required.issubset(set(df.columns))


# ── games_with_data ───────────────────────────────────────────────────────────

class TestGamesWithData:
    def test_empty_returns_empty_list(self):
        with _acc() as acc:
            assert acc.games_with_data() == []

    def test_returns_game_ids_with_frames(self):
        with _acc() as acc:
            acc.record_frame(game_id=10, frame_seq=0, positions=[_track()])
            acc.record_frame(game_id=20, frame_seq=0, positions=[_track()])
            games = acc.games_with_data()
            assert set(games) == {10, 20}

    def test_sorted_ascending(self):
        with _acc() as acc:
            for gid in [30, 10, 20]:
                acc.record_frame(game_id=gid, frame_seq=0, positions=[_track()])
            assert acc.games_with_data() == [10, 20, 30]


# ── query() passthrough ───────────────────────────────────────────────────────

def test_query_passthrough():
    with _acc() as acc:
        df = acc.query("SELECT 42 AS answer")
        assert df["answer"][0] == 42


# ── context manager / close ───────────────────────────────────────────────────

def test_context_manager_closes_cleanly():
    with TrackingAccumulator(db_path=":memory:") as acc:
        acc.record_frame(game_id=1, frame_seq=0, positions=[_track()])
    # After __exit__, further calls should raise (connection closed)
    with pytest.raises(Exception):
        acc.frame_count(1)


# ── ShotEvent dataclass ───────────────────────────────────────────────────────

def test_shot_event_fields():
    ev = _shot_event(game_id=5, event_id=99, is_goal=True)
    assert ev.game_id == 5
    assert ev.event_id == 99
    assert ev.is_goal is True
    assert ev.shot_result == "on_goal"
    assert ev.shooter_team == "EDM"
