"""Tests for Feature 16.9 — WebSocket position feed (/ws/game/{id}/positions).

Uses Starlette's synchronous WebSocket test client so no real asyncio is
needed.  The CV worker is never started — the handler's inner try/except
ensures an empty payload is sent instead of crashing.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

# Import the FastAPI app (path bootstrap happens inside main.py)
from dashboard.api.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trackpoint(
    track_id: int = 1,
    class_name: str = "player_home",
    team: str = "EDM",
    rink_x: float = 20.0,
    rink_y: float = -5.0,
    vx: float = 3.0,
    vy: float = -1.5,
    jersey_number: int | None = 97,
    jersey_locked: bool = True,
) -> MagicMock:
    tp = MagicMock()
    tp.track_id      = track_id
    tp.class_name    = class_name
    tp.class_id      = 0
    tp.team          = team
    tp.rink_x        = rink_x
    tp.rink_y        = rink_y
    tp.vx            = vx
    tp.vy            = vy
    tp.cx            = 640.0
    tp.cy            = 360.0
    tp.w             = 40.0
    tp.h             = 80.0
    tp.jersey_number = jersey_number
    tp.jersey_locked = jersey_locked
    tp.frame_ts      = 0.0
    return tp


# ── Empty state (no CV worker) ────────────────────────────────────────────────

class TestWsEmpty:
    """Endpoint sends well-formed empty payload when no CV worker is active."""

    def test_message_structure_no_worker(self):
        # Force cv manager initialisation to fail / return no positions
        if hasattr(app.state, "cv"):
            del app.state.cv

        client = TestClient(app)
        with client.websocket_connect("/ws/game/9999/positions") as ws:
            data = ws.receive_json()

        assert "game_id" in data
        assert data["game_id"] == 9999
        assert "players" in data
        assert isinstance(data["players"], list)
        assert "puck" in data

    def test_players_list_is_empty_without_worker(self):
        if hasattr(app.state, "cv"):
            del app.state.cv

        client = TestClient(app)
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        assert data["players"] == []

    def test_puck_is_null_without_worker(self):
        if hasattr(app.state, "cv"):
            del app.state.cv

        client = TestClient(app)
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        assert data["puck"] is None

    def test_different_game_ids_in_payload(self):
        if hasattr(app.state, "cv"):
            del app.state.cv

        client = TestClient(app)
        for gid in [1, 42, 2024020001]:
            with client.websocket_connect(f"/ws/game/{gid}/positions") as ws:
                data = ws.receive_json()
            assert data["game_id"] == gid


# ── With mocked CV manager ────────────────────────────────────────────────────

class TestWsWithMockedPositions:
    """Endpoint correctly serialises TrackPoint objects from a mocked manager."""

    def _get_client_with_positions(self, positions: list) -> TestClient:
        mgr = MagicMock()
        mgr.get_positions.return_value = positions
        app.state.cv = mgr
        return TestClient(app)

    def teardown_method(self, _):
        if hasattr(app.state, "cv"):
            del app.state.cv

    def test_player_fields_present(self):
        tp = _make_trackpoint(track_id=7, class_name="player_home", team="EDM",
                              rink_x=30.0, rink_y=-10.0, vx=2.0, vy=1.0,
                              jersey_number=97)
        client = self._get_client_with_positions([tp])
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        assert len(data["players"]) == 1
        p = data["players"][0]
        required = {"track_id", "team", "class_name", "x", "y",
                    "vx", "vy", "speed_ms", "is_burst",
                    "jersey_number", "jersey_locked"}
        assert required.issubset(p.keys())

    def test_player_values_correct(self):
        tp = _make_trackpoint(track_id=3, class_name="player_away", team="CGY",
                              rink_x=50.0, rink_y=20.0, vx=1.0, vy=0.0,
                              jersey_number=13)
        client = self._get_client_with_positions([tp])
        with client.websocket_connect("/ws/game/5/positions") as ws:
            data = ws.receive_json()

        p = data["players"][0]
        assert p["track_id"]      == 3
        assert p["team"]          == "CGY"
        assert p["class_name"]    == "player_away"
        assert abs(p["x"] - 50.0) < 1e-6
        assert abs(p["y"] - 20.0) < 1e-6
        assert p["jersey_number"] == 13

    def test_speed_ms_computed(self):
        # vx=3, vy=4 → speed = 5.0
        tp = _make_trackpoint(vx=3.0, vy=4.0)
        client = self._get_client_with_positions([tp])
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        p = data["players"][0]
        assert abs(p["speed_ms"] - 5.0) < 0.01

    def test_is_burst_false_on_first_frame(self):
        """First frame: history len=1, baseline=speed, burst impossible."""
        tp = _make_trackpoint(vx=100.0, vy=0.0)   # very fast but baseline=100 too
        client = self._get_client_with_positions([tp])
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        assert data["players"][0]["is_burst"] is False

    def test_multiple_players_returned(self):
        tracks = [_make_trackpoint(track_id=i) for i in range(1, 8)]
        client = self._get_client_with_positions(tracks)
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        assert len(data["players"]) == 7

    def test_puck_separated_from_players(self):
        player = _make_trackpoint(track_id=1, class_name="player_home")
        puck   = _make_trackpoint(track_id=99, class_name="puck",
                                  rink_x=5.0, rink_y=2.0, vx=10.0, vy=0.0)
        client = self._get_client_with_positions([player, puck])
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        assert len(data["players"]) == 1
        assert data["puck"] is not None

    def test_puck_fields(self):
        puck = _make_trackpoint(track_id=0, class_name="puck",
                                rink_x=-15.0, rink_y=3.0, vx=30.0, vy=40.0)
        client = self._get_client_with_positions([puck])
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        pk = data["puck"]
        assert {"x", "y", "speed_ms", "occluded"}.issubset(pk.keys())
        assert abs(pk["x"] - (-15.0)) < 1e-6
        assert abs(pk["y"] - 3.0)    < 1e-6
        # speed: sqrt(30²+40²) = 50
        assert abs(pk["speed_ms"] - 50.0) < 0.1
        assert pk["occluded"] is False

    def test_null_rink_coords_passed_through(self):
        tp = _make_trackpoint(rink_x=None, rink_y=None)
        client = self._get_client_with_positions([tp])
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        p = data["players"][0]
        assert p["x"] is None
        assert p["y"] is None

    def test_goalie_appears_in_players(self):
        goalie = _make_trackpoint(track_id=30, class_name="goalie_home")
        client = self._get_client_with_positions([goalie])
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        assert len(data["players"]) == 1
        assert data["players"][0]["class_name"] == "goalie_home"

    def test_referee_appears_in_players(self):
        ref = _make_trackpoint(track_id=50, class_name="referee", team=None)
        ref.team = None
        client = self._get_client_with_positions([ref])
        with client.websocket_connect("/ws/game/1/positions") as ws:
            data = ws.receive_json()

        p = data["players"][0]
        assert p["class_name"] == "referee"
        assert p["team"] == ""   # None → ""


# ── Burst detection (multi-frame) ─────────────────────────────────────────────

class TestBurstDetection:
    """Burst flag requires speed > 1.8× rolling baseline across multiple frames."""

    def teardown_method(self, _):
        if hasattr(app.state, "cv"):
            del app.state.cv

    def test_burst_triggers_after_sustained_slow_then_fast(self):
        """15 slow frames then 1 fast → is_burst should be True on frame 16."""
        slow_tp  = _make_trackpoint(track_id=1, vx=1.0, vy=0.0)    # speed 1.0
        fast_tp  = _make_trackpoint(track_id=1, vx=10.0, vy=0.0)   # speed 10.0 > 1.8×1 = 1.8

        call_count = [0]

        def _get_positions(game_id: int):
            call_count[0] += 1
            if call_count[0] <= 15:
                return [slow_tp]
            return [fast_tp]

        mgr = MagicMock()
        mgr.get_positions.side_effect = _get_positions
        app.state.cv = mgr

        client = TestClient(app)
        burst_seen = False
        with client.websocket_connect("/ws/game/1/positions") as ws:
            for _ in range(16):
                data = ws.receive_json()
            burst_seen = data["players"][0]["is_burst"]

        assert burst_seen is True
