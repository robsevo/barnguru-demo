"""Tests for Feature 16.12 — Puck Kalman tracker.

Covers:
  - PuckTracker Kalman update and prediction
  - Occlusion handling (frames_occluded, track reset)
  - Re-association distance guard (teleport prevention)
  - TrackPoint.occluded field propagation
  - TrackingAccumulator.record_puck / puck_positions round-trip
  - WebSocket occluded field reads from TrackPoint
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# PuckTracker unit tests
# ---------------------------------------------------------------------------

class TestPuckTrackerBasic:
    def test_no_detection_no_active_track_returns_none(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        assert pt.update(None) is None

    def test_first_detection_initialises_track(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        state = pt.update((640.0, 360.0, 0.9))
        assert state is not None
        assert state.cx == pytest.approx(640.0, abs=1.0)
        assert state.cy == pytest.approx(360.0, abs=1.0)
        assert state.occluded is False
        assert state.lost is False
        assert state.frames_occluded == 0

    def test_velocity_estimated_after_two_frames(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((100.0, 100.0, 0.9))
        state = pt.update((110.0, 105.0, 0.9))
        assert state is not None
        # After two observations moving right+down, vx should be positive
        assert state.vx > 0
        assert state.vy > 0

    def test_speed_px_is_nonnegative(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((200.0, 200.0, 0.9))
        state = pt.update((220.0, 200.0, 0.9))
        assert state is not None
        assert state.speed_px >= 0.0

    def test_speed_px_approx_magnitude(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((0.0, 0.0, 0.9))
        # Move 30 px purely in x — Kalman will damp but vx should be positive
        state = pt.update((30.0, 0.0, 0.9))
        assert state is not None
        assert state.speed_px == pytest.approx(
            math.sqrt(state.vx ** 2 + state.vy ** 2), abs=1e-6
        )

    def test_confidence_stored_on_detection(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        state = pt.update((500.0, 300.0, 0.75))
        assert state is not None
        assert state.confidence == pytest.approx(0.75, abs=1e-6)

    def test_confidence_zero_on_occlusion(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((500.0, 300.0, 0.9))
        state = pt.update(None)
        assert state is not None
        assert state.confidence == pytest.approx(0.0, abs=1e-9)

    def test_is_active_flag(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        assert pt.is_active is False
        pt.update((100.0, 100.0, 0.9))
        assert pt.is_active is True


class TestPuckTrackerOcclusion:
    def test_occluded_flag_set_on_missed_frame(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((400.0, 300.0, 0.9))
        state = pt.update(None)
        assert state is not None
        assert state.occluded is True
        assert state.frames_occluded == 1

    def test_frames_occluded_increments(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((400.0, 300.0, 0.9))
        for i in range(5):
            state = pt.update(None)
        assert state is not None
        assert state.frames_occluded == 5

    def test_frames_occluded_resets_on_detection(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((400.0, 300.0, 0.9))
        pt.update(None)
        pt.update(None)
        state = pt.update((405.0, 305.0, 0.9))
        assert state is not None
        assert state.occluded is False
        assert state.frames_occluded == 0

    def test_track_resets_after_max_occlusion(self):
        from data.puck_tracker import PuckTracker
        max_occ = 5
        pt = PuckTracker(max_occlusion_frames=max_occ)
        pt.update((400.0, 300.0, 0.9))
        for _ in range(max_occ):
            state = pt.update(None)
        # One more frame with no detection should reset
        state = pt.update(None)
        assert state is None
        assert pt.is_active is False

    def test_occluded_prediction_moves_with_velocity(self):
        """After two frames establishing velocity, occluded prediction extrapolates."""
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((100.0, 100.0, 0.9))
        pt.update((110.0, 100.0, 0.9))   # vx ≈ 10 px/frame
        state = pt.update(None)
        # Predicted position should be > 110 in x
        assert state is not None
        assert state.cx > 110.0

    def test_manual_reset_clears_track(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((300.0, 300.0, 0.9))
        pt.reset()
        assert pt.is_active is False
        assert pt.update(None) is None


class TestPuckTrackerReassociation:
    def test_close_detection_accepted(self):
        """Detection within reassoc_dist is accepted without reset."""
        from data.puck_tracker import PuckTracker
        pt = PuckTracker(reassoc_dist=80.0)
        pt.update((400.0, 300.0, 0.9))
        state = pt.update((420.0, 300.0, 0.9))   # 20 px away — well within 80 px
        assert state is not None
        assert state.lost is False

    def test_far_detection_triggers_reset_and_reinit(self):
        """Detection > reassoc_dist causes tracker reset; new track starts."""
        from data.puck_tracker import PuckTracker
        pt = PuckTracker(reassoc_dist=80.0)
        pt.update((400.0, 300.0, 0.9))
        # Teleport 200 px away
        state = pt.update((600.0, 300.0, 0.9))
        assert state is not None
        assert state.lost is True
        # New track centred near the detection
        assert state.cx == pytest.approx(600.0, abs=5.0)

    def test_reassoc_tracker_still_active_after_teleport(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker(reassoc_dist=80.0)
        pt.update((400.0, 300.0, 0.9))
        pt.update((600.0, 300.0, 0.9))
        assert pt.is_active is True

    def test_frames_occluded_property(self):
        from data.puck_tracker import PuckTracker
        pt = PuckTracker()
        pt.update((300.0, 300.0, 0.9))
        pt.update(None)
        pt.update(None)
        assert pt.frames_occluded == 2


# ---------------------------------------------------------------------------
# TrackPoint.occluded field
# ---------------------------------------------------------------------------

class TestTrackPointOccluded:
    def _make_tp(self, occluded: bool = False):
        from data.cv_tracker import TrackPoint
        return TrackPoint(
            track_id=-1, class_id=5, class_name="puck",
            cx=640.0, cy=360.0, vx=0.0, vy=0.0, w=4.0, h=4.0,
            rink_x=None, rink_y=None,
            jersey_number=None, jersey_locked=False, team=None,
            frame_ts=0.0, occluded=occluded,
        )

    def test_default_occluded_false(self):
        tp = self._make_tp()
        assert tp.occluded is False

    def test_occluded_true(self):
        tp = self._make_tp(occluded=True)
        assert tp.occluded is True

    def test_to_dict_includes_occluded(self):
        tp = self._make_tp(occluded=True)
        d = tp.to_dict()
        assert "occluded" in d
        assert d["occluded"] is True


# ---------------------------------------------------------------------------
# TrackingAccumulator puck_frames round-trip
# ---------------------------------------------------------------------------

class TestAccumulatorPuckFrames:
    def _puck_state(self, cx=640.0, cy=360.0, vx=5.0, vy=2.0, occluded=False, conf=0.9):
        from data.puck_tracker import PuckState
        return PuckState(
            cx=cx, cy=cy, vx=vx, vy=vy,
            speed_px=math.sqrt(vx ** 2 + vy ** 2),
            confidence=conf,
            occluded=occluded,
            frames_occluded=0 if not occluded else 2,
            lost=False,
        )

    def test_record_puck_then_query(self):
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            ps = self._puck_state()
            acc.record_puck(game_id=1, frame_seq=10, puck_state=ps,
                            rink_x=25.0, rink_y=-5.0)
            df = acc.puck_positions(1)
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["game_id"] == 1
        assert row["frame_seq"] == 10
        assert row["cx"] == pytest.approx(640.0)
        assert row["rink_x"] == pytest.approx(25.0)
        assert row["rink_y"] == pytest.approx(-5.0)
        assert row["occluded"] is False

    def test_occluded_stored_correctly(self):
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            ps = self._puck_state(occluded=True, conf=0.0)
            acc.record_puck(game_id=2, frame_seq=5, puck_state=ps)
            df = acc.puck_positions(2)
        assert df["occluded"][0] is True
        assert df["confidence"][0] == pytest.approx(0.0)

    def test_multiple_frames_ordered_descending(self):
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            for seq in range(5):
                ps = self._puck_state(cx=float(seq * 10))
                acc.record_puck(game_id=3, frame_seq=seq, puck_state=ps)
            df = acc.puck_positions(3)
        # Latest frame first
        assert df["frame_seq"][0] == 4
        assert df["frame_seq"][-1] == 0

    def test_null_rink_coords_accepted(self):
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            ps = self._puck_state()
            acc.record_puck(game_id=4, frame_seq=0, puck_state=ps,
                            rink_x=None, rink_y=None)
            df = acc.puck_positions(4)
        assert df["rink_x"][0] is None
        assert df["rink_y"][0] is None

    def test_primary_key_upsert(self):
        """INSERT OR REPLACE: second write for same (game_id, frame_seq) replaces."""
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            ps1 = self._puck_state(cx=100.0)
            ps2 = self._puck_state(cx=200.0)
            acc.record_puck(game_id=5, frame_seq=0, puck_state=ps1)
            acc.record_puck(game_id=5, frame_seq=0, puck_state=ps2)
            df = acc.puck_positions(5)
        assert len(df) == 1
        assert df["cx"][0] == pytest.approx(200.0)

    def test_speed_px_stored(self):
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            ps = self._puck_state(vx=3.0, vy=4.0)
            acc.record_puck(game_id=6, frame_seq=0, puck_state=ps)
            df = acc.puck_positions(6)
        assert df["speed_px"][0] == pytest.approx(5.0, abs=0.01)

    def test_empty_result_for_unknown_game(self):
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            df = acc.puck_positions(999)
        assert len(df) == 0

    def test_limit_parameter(self):
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            for seq in range(20):
                ps = self._puck_state()
                acc.record_puck(game_id=7, frame_seq=seq, puck_state=ps)
            df = acc.puck_positions(7, limit=5)
        assert len(df) == 5


# ---------------------------------------------------------------------------
# Integration: PuckTracker → record_puck end-to-end
# ---------------------------------------------------------------------------

class TestPuckTrackerIntegration:
    def test_tracker_output_storable(self):
        """PuckState fields are compatible with record_puck without conversion."""
        from data.puck_tracker import PuckTracker
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            pt = PuckTracker()
            pt.update((400.0, 300.0, 0.9))
            state = pt.update((420.0, 305.0, 0.85))
            assert state is not None
            acc.record_puck(game_id=1, frame_seq=1, puck_state=state,
                            rink_x=10.0, rink_y=5.0)
            df = acc.puck_positions(1)
        assert len(df) == 1

    def test_occluded_state_propagates_to_db(self):
        from data.puck_tracker import PuckTracker
        from data.tracking_accumulator import TrackingAccumulator
        with TrackingAccumulator(":memory:") as acc:
            pt = PuckTracker()
            pt.update((400.0, 300.0, 0.9))
            state = pt.update(None)   # occluded frame
            assert state is not None
            acc.record_puck(game_id=1, frame_seq=2, puck_state=state)
            df = acc.puck_positions(1)
        assert df["occluded"][0] is True
        assert df["confidence"][0] == pytest.approx(0.0)
