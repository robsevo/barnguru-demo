"""Tests for Feature 16.13 — AI Game Observer (data/cv_ai_observer.py).

All tests use a temporary in-memory WhizFeed (backed by a temp file) and
mocked CvWorkerManager so no real CV worker or NHL API is needed.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from data.cv_ai_observer import (
    AiObserver,
    AiObserverManager,
    _TrackState,
    _PairState,
    _BURST_SUSTAINED_S,
    _BURST_MULT,
    _BURST_MIN_BASELINE,
    _STAT_OZ_S,
    _STAT_SPD,
    _OZ_X_THRESHOLD,
    _CHEM_OZ_THRESH,
    _CHEM_MIN_FRAMES,
)
from models.whiz_feed import WhizFeed


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_feed(tmp_path: Path) -> WhizFeed:
    """WhizFeed backed by a temp file — isolated per test."""
    return WhizFeed(path=tmp_path / "obs.json")


def _mock_cv_mgr(positions=None):
    mgr = MagicMock()
    mgr.get_positions.return_value = positions or []
    return mgr


def _make_tp(
    track_id: int = 1,
    class_name: str = "player_home",
    team: str = "EDM",
    rink_x: Optional[float] = 30.0,
    rink_y: Optional[float] = 5.0,
    vx: float = 1.0,
    vy: float = 0.0,
    jersey_number: Optional[int] = 97,
    jersey_locked: bool = True,
):
    tp = MagicMock()
    tp.track_id      = track_id
    tp.class_name    = class_name
    tp.team          = team
    tp.rink_x        = rink_x
    tp.rink_y        = rink_y
    tp.vx            = vx
    tp.vy            = vy
    tp.jersey_number = jersey_number
    tp.jersey_locked = jersey_locked
    return tp


def _make_observer(tmp_feed: WhizFeed, positions=None, game_id: int = 1) -> AiObserver:
    mgr = _mock_cv_mgr(positions)
    obs = AiObserver(
        game_id=game_id,
        cv_manager=mgr,
        feed=tmp_feed,
        game_date="2026-04-02",
        away_team="CGY",
        home_team="EDM",
    )
    return obs


# ── _TrackState ───────────────────────────────────────────────────────────────

class TestTrackState:
    def test_baseline_speed_empty(self):
        ts = _TrackState(track_id=1)
        assert ts.baseline_speed() == 0.0

    def test_current_speed_uses_last(self):
        ts = _TrackState(track_id=1, speed_hist=[1.0, 2.0, 3.0])
        assert ts.current_speed() == 3.0

    def test_is_stationary_below_threshold(self):
        ts = _TrackState(track_id=1, speed_hist=[_STAT_SPD - 0.1])
        assert ts.is_stationary() is True

    def test_is_stationary_above_threshold(self):
        ts = _TrackState(track_id=1, speed_hist=[_STAT_SPD + 0.1])
        assert ts.is_stationary() is False

    def test_is_burst_false_below_min_baseline(self):
        ts = _TrackState(track_id=1, speed_hist=[_BURST_MIN_BASELINE - 0.1])
        assert ts.is_burst() is False

    def test_is_burst_false_on_first_frame_equal_speed(self):
        ts = _TrackState(track_id=1, speed_hist=[5.0])
        # baseline == current → ratio = 1.0 < 1.8
        assert ts.is_burst() is False

    def test_is_burst_true_above_mult(self):
        # 14 slow frames (1.0) + 1 fast frame (5.0)
        # baseline = (14×1.0 + 5.0)/15 = 1.267; current = 5.0; 5.0 >= 1.8×1.267 = 2.28 ✓
        ts = _TrackState(track_id=1, speed_hist=[1.0] * 14 + [5.0])
        assert ts.is_burst() is True

    def test_in_oz_home_positive_x(self):
        ts = _TrackState(track_id=1, class_name="player_home",
                         rink_x=_OZ_X_THRESHOLD + 1)
        assert ts.in_oz() is True

    def test_in_oz_home_neutral_zone(self):
        ts = _TrackState(track_id=1, class_name="player_home",
                         rink_x=_OZ_X_THRESHOLD - 1)
        assert ts.in_oz() is False

    def test_in_oz_away_negative_x(self):
        ts = _TrackState(track_id=1, class_name="player_away",
                         rink_x=-(_OZ_X_THRESHOLD + 1))
        assert ts.in_oz() is True

    def test_in_oz_none_x(self):
        ts = _TrackState(track_id=1, class_name="player_home", rink_x=None)
        assert ts.in_oz() is False


# ── AiObserver._update_track ──────────────────────────────────────────────────

class TestUpdateTrack:
    def test_creates_new_track_state(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        tp  = _make_tp(track_id=7, class_name="player_home", team="EDM")
        obs._update_track(tp)
        assert 7 in obs._tracks
        assert obs._tracks[7].team == "EDM"

    def test_speed_appended(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        tp  = _make_tp(vx=3.0, vy=4.0)   # speed = 5.0
        obs._update_track(tp)
        assert abs(obs._tracks[1].speed_hist[-1] - 5.0) < 1e-6

    def test_jersey_locked_stores_number(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        tp  = _make_tp(jersey_number=97, jersey_locked=True)
        obs._update_track(tp)
        assert obs._tracks[1].jersey_number == 97

    def test_jersey_unlocked_ignored(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        tp  = _make_tp(jersey_number=99, jersey_locked=False)
        obs._update_track(tp)
        assert obs._tracks[1].jersey_number is None

    def test_speed_hist_capped_at_15(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        tp  = _make_tp(track_id=1)
        for _ in range(20):
            obs._update_track(tp)
        assert len(obs._tracks[1].speed_hist) == 15


# ── Trigger (a): Burst ────────────────────────────────────────────────────────

class TestBurstTrigger:
    def _seed_slow_then_fast(self, obs: AiObserver, track_id: int = 1) -> _TrackState:
        """Prime the history with 14 slow frames, then add a fast frame."""
        slow_tp = _make_tp(track_id=track_id, vx=1.0, vy=0.0)  # speed 1.0
        for _ in range(14):
            obs._update_track(slow_tp)
        fast_tp = _make_tp(track_id=track_id, vx=10.0, vy=0.0)  # speed 10 > 1.8×1
        obs._update_track(fast_tp)
        return obs._tracks[track_id]

    def test_no_obs_before_sustained(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        state = self._seed_slow_then_fast(obs)
        now = time.time()
        obs._check_burst(state, now)
        assert state.burst_since is not None   # timer started
        assert len(obs._feed.all_observations()) == 0

    def test_obs_written_after_sustained(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        state = self._seed_slow_then_fast(obs)
        now = time.time()
        obs._check_burst(state, now)                         # sets burst_since
        state.burst_since = now - _BURST_SUSTAINED_S - 1    # simulate 4s elapsed
        obs._check_burst(state, now)
        all_obs = obs._feed.all_observations()
        assert len(all_obs) == 1
        assert all_obs[0]["category"] == "form"
        assert all_obs[0]["source"] == "ai_observer"

    def test_burst_cooldown_prevents_repeat(self, tmp_feed):
        obs   = _make_observer(tmp_feed)
        state = self._seed_slow_then_fast(obs)
        now   = time.time()
        obs._check_burst(state, now)
        state.burst_since     = now - _BURST_SUSTAINED_S - 1
        obs._check_burst(state, now)   # logs first obs
        # Re-seed burst immediately
        state.burst_since     = now - _BURST_SUSTAINED_S - 1
        obs._check_burst(state, now)   # still in cooldown
        assert len(obs._feed.all_observations()) == 1

    def test_goalie_skipped(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        tp  = _make_tp(track_id=30, class_name="goalie_home", vx=20.0, vy=0.0)
        for _ in range(15):
            obs._update_track(tp)
        state = obs._tracks[30]
        state.burst_since = time.time() - _BURST_SUSTAINED_S - 1
        obs._check_burst(state, time.time())
        assert len(obs._feed.all_observations()) == 0

    def test_referee_skipped(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        tp  = _make_tp(track_id=50, class_name="referee", vx=20.0, vy=0.0)
        for _ in range(15):
            obs._update_track(tp)
        state = obs._tracks[50]
        state.burst_since = time.time() - _BURST_SUSTAINED_S - 1
        obs._check_burst(state, time.time())
        assert len(obs._feed.all_observations()) == 0

    def test_hot_hand_score_in_sim_delta(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        state = self._seed_slow_then_fast(obs)
        now   = time.time()
        obs._check_burst(state, now)
        state.burst_since = now - _BURST_SUSTAINED_S - 5  # 8s elapsed
        obs._check_burst(state, now)
        o = obs._feed.all_observations()[0]
        assert "hot_hand_score" in o["sim_delta"]
        assert o["sim_delta"]["hot_hand_score"] > 0


# ── Trigger (c): Stationary in OZ ────────────────────────────────────────────

class TestStationaryOzTrigger:
    def _make_stationary_state(self, class_name="player_home", rink_x=30.0) -> _TrackState:
        state = _TrackState(
            track_id=1,
            class_name=class_name,
            team="EDM",
            rink_x=rink_x,
            speed_hist=[_STAT_SPD - 0.5] * 5,  # below stationary threshold
        )
        return state

    def test_no_obs_before_threshold(self, tmp_feed):
        obs   = _make_observer(tmp_feed)
        state = self._make_stationary_state()
        now   = time.time()
        obs._check_stationary_oz(state, now)           # sets stationary_since
        state.stationary_since = now - (_STAT_OZ_S - 5)
        obs._check_stationary_oz(state, now)           # still too early
        assert len(obs._feed.all_observations()) == 0

    def test_obs_written_after_threshold(self, tmp_feed):
        obs   = _make_observer(tmp_feed)
        state = self._make_stationary_state()
        now   = time.time()
        obs._check_stationary_oz(state, now)           # sets timer
        state.stationary_since = now - (_STAT_OZ_S + 1)
        obs._check_stationary_oz(state, now)           # triggers
        all_obs = obs._feed.all_observations()
        assert len(all_obs) == 1
        assert all_obs[0]["category"] == "injury_concern"
        assert all_obs[0]["source"] == "ai_observer"

    def test_not_in_oz_ignored(self, tmp_feed):
        obs   = _make_observer(tmp_feed)
        state = self._make_stationary_state(rink_x=10.0)  # neutral zone for home
        now   = time.time()
        obs._check_stationary_oz(state, now)
        state.stationary_since = now - (_STAT_OZ_S + 1)
        obs._check_stationary_oz(state, now)
        assert len(obs._feed.all_observations()) == 0

    def test_moving_player_clears_timer(self, tmp_feed):
        obs   = _make_observer(tmp_feed)
        state = self._make_stationary_state()
        now   = time.time()
        obs._check_stationary_oz(state, now)           # starts timer
        state.speed_hist = [_STAT_SPD + 2.0]          # now moving
        obs._check_stationary_oz(state, now)           # clears timer
        assert state.stationary_since is None


# ── Trigger (d): Chemistry ────────────────────────────────────────────────────

class TestChemistryTrigger:
    def _make_pair_positions(self, team="EDM", x=30.0):
        a = _make_tp(track_id=1, class_name="player_home", team=team, rink_x=x, rink_y=5.0)
        b = _make_tp(track_id=2, class_name="player_home", team=team, rink_x=x+5, rink_y=5.0)
        return [a, b]

    def test_no_obs_when_below_min_frames(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        positions = self._make_pair_positions()
        now = time.time()
        # Feed only a few frames — below _CHEM_MIN_FRAMES
        for _ in range(5):
            obs._update_track(positions[0])
            obs._update_track(positions[1])
        obs._check_chemistry(positions, now)
        assert len(obs._feed.all_observations()) == 0

    def test_pair_state_created(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        positions = self._make_pair_positions()
        obs._check_chemistry(positions, time.time())
        assert (1, 2) in obs._pairs

    def test_chemistry_fires_after_full_window(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        positions = self._make_pair_positions()
        now = time.time()
        obs._check_chemistry(positions, now)

        pair_key = (1, 2)
        ps = obs._pairs[pair_key]
        # Simulate a full window of co-location in OZ
        ps.together_frames = _CHEM_MIN_FRAMES
        ps.total_frames    = _CHEM_MIN_FRAMES
        ps.window_start    = now - 50  # expired window

        obs._check_chemistry(positions, now + 50)  # triggers window reset
        all_obs = obs._feed.all_observations()
        assert len(all_obs) == 1
        assert all_obs[0]["category"] == "chemistry"
        assert all_obs[0]["source"] == "ai_observer"

    def test_chemistry_cooldown_prevents_repeat(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        positions = self._make_pair_positions()
        now = time.time()
        obs._check_chemistry(positions, now)

        pair_key = (1, 2)
        ps = obs._pairs[pair_key]
        ps.together_frames = _CHEM_MIN_FRAMES
        ps.total_frames    = _CHEM_MIN_FRAMES
        ps.window_start    = now - 50

        obs._check_chemistry(positions, now + 50)   # first fire
        # Immediately try again
        ps.together_frames = _CHEM_MIN_FRAMES
        ps.total_frames    = _CHEM_MIN_FRAMES
        ps.window_start    = now + 50 - 50
        obs._fire_chemistry(positions[0], positions[1], "EDM", 1.0, ps, now + 60)
        assert len(obs._feed.all_observations()) == 1  # still 1

    def test_no_obs_below_threshold_ratio(self, tmp_feed):
        obs = _make_observer(tmp_feed)
        positions = self._make_pair_positions()
        now = time.time()
        obs._check_chemistry(positions, now)

        pair_key = (1, 2)
        ps = obs._pairs[pair_key]
        # Only 50% together — below 70% threshold
        ps.together_frames = 25
        ps.total_frames    = 50
        ps.window_start    = now - 50

        obs._check_chemistry(positions, now + 50)
        assert len(obs._feed.all_observations()) == 0


# ── AiObserverManager ─────────────────────────────────────────────────────────

class TestAiObserverManager:
    def test_start_creates_observer(self, tmp_feed):
        mgr = AiObserverManager(cv_manager=_mock_cv_mgr(), feed=tmp_feed)
        status = mgr.start(game_id=1, game_date="2026-04-02")
        assert status["game_id"] == 1
        assert status["alive"] is True
        mgr.shutdown()

    def test_start_returns_existing_alive(self, tmp_feed):
        mgr = AiObserverManager(cv_manager=_mock_cv_mgr(), feed=tmp_feed)
        s1 = mgr.start(game_id=1)
        s2 = mgr.start(game_id=1)
        assert s1["started_at"] == s2["started_at"]  # same observer
        mgr.shutdown()

    def test_stop_returns_true_if_running(self, tmp_feed):
        mgr = AiObserverManager(cv_manager=_mock_cv_mgr(), feed=tmp_feed)
        mgr.start(game_id=5)
        assert mgr.stop(5) is True

    def test_stop_returns_false_if_not_running(self, tmp_feed):
        mgr = AiObserverManager(cv_manager=_mock_cv_mgr(), feed=tmp_feed)
        assert mgr.stop(999) is False

    def test_all_statuses_lists_all(self, tmp_feed):
        mgr = AiObserverManager(cv_manager=_mock_cv_mgr(), feed=tmp_feed)
        mgr.start(game_id=10)
        mgr.start(game_id=20)
        statuses = mgr.all_statuses()
        game_ids = {s["game_id"] for s in statuses}
        assert {10, 20}.issubset(game_ids)
        mgr.shutdown()

    def test_shutdown_stops_all(self, tmp_feed):
        mgr = AiObserverManager(cv_manager=_mock_cv_mgr(), feed=tmp_feed)
        mgr.start(game_id=1)
        mgr.start(game_id=2)
        mgr.shutdown()
        assert mgr.all_statuses() == []

    def test_observer_to_dict_fields(self, tmp_feed):
        mgr = AiObserverManager(cv_manager=_mock_cv_mgr(), feed=tmp_feed)
        status = mgr.start(game_id=7, away_team="CGY", home_team="EDM")
        required = {"game_id", "alive", "obs_count", "started_at",
                    "away_team", "home_team", "game_date"}
        assert required.issubset(status.keys())
        mgr.shutdown()


# ── WhizFeed.add_observation source field ─────────────────────────────────────

class TestWhizFeedSource:
    def test_manual_source_stored(self, tmp_feed):
        tmp_feed.add_observation(
            subject_name="Test Player", team="EDM", observation="Manual obs",
            source="manual",
        )
        obs = tmp_feed.all_observations()
        assert obs[0]["source"] == "manual"

    def test_ai_observer_source_stored(self, tmp_feed):
        tmp_feed.add_observation(
            subject_name="Track 1 (EDM)", team="EDM", observation="[AUTO] Burst",
            source="ai_observer",
        )
        obs = tmp_feed.all_observations()
        assert obs[0]["source"] == "ai_observer"

    def test_default_source_is_manual(self, tmp_feed):
        tmp_feed.add_observation(
            subject_name="Test", team="EDM", observation="No source param",
        )
        obs = tmp_feed.all_observations()
        assert obs[0]["source"] == "manual"


# ── API endpoint smoke tests ──────────────────────────────────────────────────

class TestObserverApiEndpoints:
    def test_status_endpoint_returns_empty_when_no_observers(self):
        from starlette.testclient import TestClient
        from dashboard.api.main import app
        if hasattr(app.state, "observer"):
            del app.state.observer

        client = TestClient(app)
        res = client.get("/api/observer/status")
        assert res.status_code == 200
        data = res.json()
        assert "observers" in data
        assert isinstance(data["observers"], list)

    def test_events_endpoint_empty_game(self, tmp_path):
        from starlette.testclient import TestClient
        from dashboard.api.main import app
        client = TestClient(app)
        res = client.get("/api/observer/events/99999")
        assert res.status_code == 200
        data = res.json()
        assert data["game_id"] == 99999
        assert isinstance(data["events"], list)

    def test_stop_unknown_game_returns_not_stopped(self):
        from starlette.testclient import TestClient
        from dashboard.api.main import app
        # Seed app.state.cv with a mock so _cv_manager() never touches any
        # DuckDB state that may have been left by other test files.
        app.state.cv = _mock_cv_mgr()
        if hasattr(app.state, "observer"):
            del app.state.observer

        client = TestClient(app)
        res = client.post("/api/observer/stop/88888")
        assert res.status_code == 200
        assert res.json()["stopped"] is False
